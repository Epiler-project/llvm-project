//===-- RISCVBOSCAME.cpp
//---------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVBOSCAME.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include <array>
#include <optional>
#include <tuple>

using namespace llvm;

namespace {
constexpr uint64_t TileA = 0xffff;
constexpr uint64_t TileB = TileA << 16;
constexpr uint64_t Acc = uint64_t(0xff) << 32;
constexpr uint64_t Config = uint64_t(7) << 40;
constexpr uint64_t FP8Mode = uint64_t(1) << 43;
constexpr uint64_t Top = TileA | TileB | Acc | Config | FP8Mode;

[[noreturn]] void fail(const Function &F, const Twine &Message) {
  reportFatalUsageError("BOSC bound AME function '" + F.getName() +
                        "': " + Message);
}

unsigned configIndex(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_bosc_ame_setm:
    return 0;
  case Intrinsic::riscv_bosc_ame_setn:
    return 1;
  case Intrinsic::riscv_bosc_ame_setk:
    return 2;
  default:
    return 3;
  }
}

uint64_t configure(uint64_t State, unsigned Index) {
  State &= ~(Index == 2 ? TileA | TileB : TileA | TileB | Acc);
  return State | (uint64_t(1) << (40 + Index));
}

unsigned slot(const IntrinsicInst &II, unsigned Index, bool IsAcc) {
  const auto *Value = dyn_cast<ConstantInt>(II.getArgOperand(Index));
  if (!Value || Value->getValue().getActiveBits() > (IsAcc ? 3u : 4u))
    fail(*II.getFunction(), IsAcc ? "accumulator binding must be Acc0..Acc7"
                                  : "tile binding must be Tr0..Tr15");
  return Value->getZExtValue();
}

ConstantRange scalarRange(const Value *V, unsigned Depth = 0) {
  unsigned Width = V->getType()->getIntegerBitWidth();
  if (auto *C = dyn_cast<ConstantInt>(V))
    return ConstantRange(C->getValue());
  if (Depth >= 16)
    return ConstantRange::getFull(Width);
  // Receive constructive scalar expressions only. An argument range attribute
  // or nearby assume is not an independent proof of hardware resource safety.
  if (auto *B = dyn_cast<BinaryOperator>(V))
    return scalarRange(B->getOperand(0), Depth + 1)
        .binaryOp(B->getOpcode(), scalarRange(B->getOperand(1), Depth + 1));
  if (auto *S = dyn_cast<SelectInst>(V))
    return scalarRange(S->getTrueValue(), Depth + 1)
        .unionWith(scalarRange(S->getFalseValue(), Depth + 1));
  if (auto *C = dyn_cast<CastInst>(V)) {
    if (C->getSrcTy()->isIntegerTy())
      return scalarRange(C->getOperand(0), Depth + 1)
          .castOp(C->getOpcode(), Width);
  }
  if (auto *I = dyn_cast<IntrinsicInst>(V)) {
    auto ID = I->getIntrinsicID();
    if (configIndex(ID) != 3) {
      auto Requested = scalarRange(I->getArgOperand(0), Depth + 1);
      if (!Requested.getUnsignedMin().isZero() &&
          !Requested.getUnsignedMax().isNegative()) {
        APInt Limit(Width, 64);
        return ConstantRange(APIntOps::umin(Requested.getUnsignedMin(), Limit),
                             APIntOps::umin(Requested.getUnsignedMax(), Limit) +
                                 1);
      }
    }
    if (ID == Intrinsic::umin || ID == Intrinsic::umax ||
        ID == Intrinsic::smin || ID == Intrinsic::smax) {
      auto L = scalarRange(I->getArgOperand(0), Depth + 1);
      auto R = scalarRange(I->getArgOperand(1), Depth + 1);
      switch (ID) {
      case Intrinsic::umin:
        return L.umin(R);
      case Intrinsic::umax:
        return L.umax(R);
      case Intrinsic::smin:
        return L.smin(R);
      case Intrinsic::smax:
        return L.smax(R);
      default:
        llvm_unreachable("checked min/max intrinsic");
      }
    }
  }
  return ConstantRange::getFull(Width);
}

unsigned request(const IntrinsicInst &II) {
  auto Range = scalarRange(II.getArgOperand(0));
  if (Range.getUnsignedMin().isZero() || Range.getUnsignedMax().isNegative())
    fail(*II.getFunction(),
         "dimension request must be provably positive and nonwrapping");
  return std::min(uint64_t(64), Range.getUnsignedMax().getZExtValue());
}

bool isDimension(const Value *V, const Value *Requested, uint64_t Scale) {
  bool Same = V == Requested;
  if (auto *I = dyn_cast<IntrinsicInst>(V))
    Same |= configIndex(I->getIntrinsicID()) != 3 &&
            I->getArgOperand(0) == Requested;
  return Same && scalarRange(V).getUnsignedMax().ule(UINT64_MAX / Scale);
}

bool coversActualRow(const Value *Stride, const Value *Requested,
                     uint64_t ElementBytes) {
  if (ElementBytes == 1)
    return isDimension(Stride, Requested, 1);
  auto *B = dyn_cast<BinaryOperator>(Stride);
  if (!B)
    return false;
  auto Matches = [&](const Value *Dimension, const Value *Factor,
                     uint64_t Expected) {
    auto *C = dyn_cast<ConstantInt>(Factor);
    return C && C->equalsInt(Expected) &&
           isDimension(Dimension, Requested, ElementBytes);
  };
  if (B->getOpcode() == Instruction::Mul)
    return Matches(B->getOperand(0), B->getOperand(1), ElementBytes) ||
           Matches(B->getOperand(1), B->getOperand(0), ElementBytes);
  return B->getOpcode() == Instruction::Shl && ElementBytes == 4 &&
         Matches(B->getOperand(0), B->getOperand(1), 2);
}

bool validStride(const Value *Stride, const Value *Requested, uint64_t RowBytes,
                 bool Accumulator, const IntrinsicInst &II) {
  auto Range = scalarRange(Stride);
  if ((Range.getUnsignedMin().ult(RowBytes) &&
       !coversActualRow(Stride, Requested, Accumulator ? 4 : 1)) ||
      Range.getUnsignedMax().isNegative())
    return false;
  if (!Accumulator)
    return true;
  auto Known = computeKnownBits(Stride, II.getModule()->getDataLayout());
  return Known.Zero.countTrailingOnes() >= 2;
}

// Three exact dimension facts accompany A/B/Acc must-definitions. A tile load
// carries a role: a later load of B into the same Tr kills its prior A value.
// LLVM never substitutes an arbitrary vector type for these physical facts.
bool samePredicate(const Value *A, const Value *B) {
  if (A == B)
    return true;
  // CodeGenPrepare sinks a shared comparison into its guarded users. Those
  // comparisons denote the same boolean when their SSA operands are identical;
  // pointer identity of the comparison itself is not a value identity.
  auto *CompareA = dyn_cast<ICmpInst>(A);
  auto *CompareB = dyn_cast<ICmpInst>(B);
  return CompareA && CompareB && CompareA->isIdenticalTo(CompareB);
}

bool predicateDependsOn(const Value *Predicate, const Instruction &I) {
  if (Predicate == &I)
    return true;
  auto *Compare = dyn_cast<ICmpInst>(Predicate);
  return Compare && llvm::is_contained(Compare->operands(), &I);
}

struct State {
  uint64_t Bits = Top;
  std::array<unsigned, 3> Dimensions{~0u, ~0u, ~0u};
  std::array<const Value *, 3> Requests{};
  struct Selection {
    const Value *Predicate;
    unsigned FalseReg, TrueReg;
    bool operator==(const Selection &Other) const {
      return samePredicate(Predicate, Other.Predicate) &&
             FalseReg == Other.FalseReg && TrueReg == Other.TrueReg;
    }
  };
  SmallVector<Selection> SelectedB;
  static State entry() { return {0, {0, 0, 0}, {}, {}}; }
  bool operator==(const State &Other) const {
    return Bits == Other.Bits && Dimensions == Other.Dimensions &&
           Requests == Other.Requests && SelectedB == Other.SelectedB;
  }
  bool operator!=(const State &Other) const { return !(*this == Other); }
  void meet(const State &Other) {
    Bits &= Other.Bits;
    llvm::erase_if(SelectedB, [&](const Selection &S) {
      return !llvm::is_contained(Other.SelectedB, S);
    });
    for (unsigned I = 0; I < 3; ++I) {
      if (Dimensions[I] == ~0u) {
        Dimensions[I] = Other.Dimensions[I];
        Requests[I] = Other.Requests[I];
      } else if (Other.Dimensions[I] != ~0u &&
                 (Dimensions[I] != Other.Dimensions[I] ||
                  Requests[I] != Other.Requests[I])) {
        Dimensions[I] = 0;
        Requests[I] = nullptr;
      }
    }
  }
};

State transferIR(const BasicBlock &BB, State S, bool Verify) {
  for (const Instruction &I : BB) {
    llvm::erase_if(S.SelectedB, [&](const State::Selection &Selection) {
      return predicateDependsOn(Selection.Predicate, I);
    });
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !isBOSCAMEIntrinsic(II->getIntrinsicID()))
      continue;
    auto ID = II->getIntrinsicID();
    if (ID == Intrinsic::riscv_bosc_ame_setfp8_e4m3) {
      S.Bits |= FP8Mode;
      continue;
    }
    unsigned Index = configIndex(ID);
    if (Index != 3) {
      S.Bits = configure(S.Bits, Index);
      S.Dimensions[Index] = request(*II);
      S.Requests[Index] = II->getArgOperand(0);
      S.SelectedB.clear();
      continue;
    }
    if (Verify &&
        ((S.Bits & Config) != Config || llvm::is_contained(S.Dimensions, 0u) ||
         llvm::is_contained(S.Dimensions, ~0u)))
      fail(*BB.getParent(), "operation requires M/N/K configuration with "
                            "matching dimensions on every incoming path");
    bool SelectLoad = ID == Intrinsic::riscv_bosc_ame_load_b_select;
    bool SelectMMA = ID == Intrinsic::riscv_bosc_ame_mma_b_select;
    if (SelectLoad || SelectMMA) {
      unsigned SelectorIndex = SelectLoad ? 0 : 2;
      unsigned BIndex = SelectLoad ? 1 : 3;
      State::Selection Selected{II->getArgOperand(SelectorIndex),
                                slot(*II, BIndex, false),
                                slot(*II, BIndex + 1, false)};
      if (Selected.FalseReg == Selected.TrueReg)
        fail(*BB.getParent(), "selected B stages must bind distinct registers");
      if (SelectLoad) {
        if (!validStride(II->getArgOperand(4), S.Requests[2], S.Dimensions[2],
                         false, *II))
          fail(*BB.getParent(), "selected B load stride must cover the row");
        S.Bits &= ~((uint64_t(1) << Selected.FalseReg) |
                    (uint64_t(1) << Selected.TrueReg));
        if (!llvm::is_contained(S.SelectedB, Selected))
          S.SelectedB.push_back(Selected);
      } else if (Verify &&
                 (!(S.Bits & (uint64_t(1) << (32 + slot(*II, 0, true)))) ||
                  !(S.Bits & (uint64_t(1) << slot(*II, 1, false))) ||
                  !llvm::is_contained(S.SelectedB, Selected))) {
        fail(*BB.getParent(), "selected MMA requires initialized C/A and the "
                              "same B selector on every incoming path");
      }
      continue;
    }
    bool IsA = ID == Intrinsic::riscv_bosc_ame_load_a;
    bool IsB = ID == Intrinsic::riscv_bosc_ame_load_b;
    bool IsFPMMA = ID == Intrinsic::riscv_bosc_ame_fp_mma;
    bool IsMMA = ID == Intrinsic::riscv_bosc_ame_mma || IsFPMMA;
    if (Verify && IsFPMMA && !(S.Bits & FP8Mode))
      fail(*BB.getParent(), "floating MMA requires E4M3 configuration on "
                            "every incoming path");
    bool IsStore = ID == Intrinsic::riscv_bosc_ame_store_c;
    unsigned Dst = slot(*II, 0, !IsA && !IsB);
    if (Verify && !IsMMA) {
      uint64_t RowBytes = IsA || IsB ? S.Dimensions[2] : 4 * S.Dimensions[1];
      if (!validStride(II->getArgOperand(2), S.Requests[IsA || IsB ? 2 : 1],
                       RowBytes, !IsA && !IsB, *II))
        fail(*BB.getParent(),
             "byte stride must be provably bounded covering the "
             "row; accumulator stride must be 4-byte aligned");
    }
    auto Read = [&](unsigned Offset, unsigned Register, StringRef Role) {
      if (Verify && !(S.Bits & (uint64_t(1) << (Offset + Register))))
        fail(*BB.getParent(),
             "read of undefined " + Role + " register " + Twine(Register));
    };
    if (IsMMA) {
      Read(0, slot(*II, 1, false), "A tile");
      Read(16, slot(*II, 2, false), "B tile");
      Read(32, Dst, "accumulator");
    } else if (IsStore) {
      Read(32, Dst, "accumulator");
    } else if (IsA || IsB) {
      if (IsA)
        llvm::erase_if(S.SelectedB, [&](const State::Selection &Selection) {
          return Selection.FalseReg == Dst || Selection.TrueReg == Dst;
        });
      S.Bits &= ~((uint64_t(1) | (uint64_t(1) << 16)) << Dst);
      S.Bits |= uint64_t(1) << (Dst + (IsB ? 16 : 0));
    } else {
      S.Bits |= uint64_t(1) << (Dst + 32);
    }
  }
  return S;
}

unsigned machineConfig(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::BOSC_AME_SETM:
    return 0;
  case RISCV::BOSC_AME_SETN:
    return 1;
  case RISCV::BOSC_AME_SETK:
    return 2;
  default:
    return 3;
  }
}

bool isAction(unsigned Opcode) {
  return Opcode == RISCV::BOSC_AME_LOADA || Opcode == RISCV::BOSC_AME_LOADB ||
         Opcode == RISCV::BOSC_AME_LOADC || Opcode == RISCV::BOSC_AME_STOREC ||
         Opcode == RISCV::BOSC_AME_MMA || Opcode == RISCV::BOSC_AME_FPMMA;
}

bool isSelectedLoad(unsigned Opcode) {
  return Opcode == RISCV::PseudoBOSCAMELoadBSelectBound;
}
bool isSelectedMMA(unsigned Opcode) {
  return Opcode == RISCV::PseudoBOSCAMEMMABSelectBound;
}
// Scalar selector identity must survive legal copies, boolean rematerialization
// and register-allocation spills. Track only these exact forms; all other
// definitions start a new value identity. Re-executing a definition in a loop
// invalidates its earlier selected-stage facts.
struct SelectorValue {
  enum class Projection { Whole, LowBit, Nonzero };
  const MachineInstr *Definition = nullptr;
  Register Entry;
  Projection Form = Projection::Whole;
  bool Negated = false;
  explicit operator bool() const { return Definition || Entry; }
  bool operator==(const SelectorValue &Other) const {
    return Definition == Other.Definition && Entry == Other.Entry &&
           Form == Other.Form && Negated == Other.Negated;
  }
  bool operator!=(const SelectorValue &Other) const {
    return !(*this == Other);
  }
};

struct SelectorValues {
  bool Top = true;
  DenseMap<Register, SelectorValue> Registers;
  DenseMap<int, SelectorValue> Spills;
  bool operator==(const SelectorValues &Other) const {
    return Top == Other.Top && Registers == Other.Registers &&
           Spills == Other.Spills;
  }
  void meet(const SelectorValues &Other) {
    if (Other.Top)
      return;
    if (Top) {
      *this = Other;
      return;
    }
    auto Intersect = [](auto &Values, const auto &Incoming) {
      Values.remove_if([&](const auto &Entry) {
        auto It = Incoming.find(Entry.first);
        return It == Incoming.end() || !(It->second == Entry.second);
      });
    };
    Intersect(Registers, Other.Registers);
    Intersect(Spills, Other.Spills);
  }
};

std::optional<int> selectorSpill(const MachineInstr &MI) {
  if (!MI.hasOneMemOperand())
    return std::nullopt;
  const MachineMemOperand &MMO = **MI.memoperands_begin();
  auto *Stack =
      dyn_cast_or_null<FixedStackPseudoSourceValue>(MMO.getPseudoValue());
  if (!Stack || MMO.getOffset() || MMO.getSize() != LocationSize::precise(8))
    return std::nullopt;
  return Stack->getFrameIndex();
}

void transferSelectorValues(const MachineInstr &MI, SelectorValues &S) {
  if (S.Top)
    return;
  auto Read = [&](unsigned Operand) {
    return S.Registers.lookup(MI.getOperand(Operand).getReg());
  };
  SelectorValue Result;
  unsigned Op = MI.getOpcode();
  auto ImmediateIs = [&](unsigned Operand, int64_t Value) {
    return MI.getOperand(Operand).isImm() &&
           MI.getOperand(Operand).getImm() == Value;
  };
  bool Propagates = false;
  if (MI.isCopy() ||
      (Op == RISCV::ADDI && MI.getOperand(1).isReg() && ImmediateIs(2, 0))) {
    Propagates = true;
    Result = Read(1);
  } else if (Op == RISCV::ANDI && ImmediateIs(2, 1)) {
    Propagates = true;
    Result = Read(1);
    if (Result.Form == SelectorValue::Projection::Whole)
      Result.Form = SelectorValue::Projection::LowBit;
  } else if (Op == RISCV::SLTU && MI.getOperand(1).getReg() == RISCV::X0) {
    Propagates = true;
    Result = Read(2);
    if (Result.Form == SelectorValue::Projection::Whole)
      Result.Form = SelectorValue::Projection::Nonzero;
  } else if (Op == RISCV::XORI && ImmediateIs(2, 1)) {
    Result = Read(1);
    if (Result.Form != SelectorValue::Projection::Whole) {
      Propagates = true;
      Result.Negated = !Result.Negated;
    }
  } else if (Op == RISCV::LD) {
    if (auto Frame = selectorSpill(MI)) {
      Propagates = true;
      Result = S.Spills.lookup(*Frame);
    }
  }
  if (MI.mayStore()) {
    // Keep only exact compiler-private spills. Any partial write to a frame
    // object invalidates that object's previous scalar provenance.
    for (const MachineMemOperand *MMO : MI.memoperands())
      if (auto *Stack = dyn_cast_or_null<FixedStackPseudoSourceValue>(
              MMO->getPseudoValue()))
        S.Spills.erase(Stack->getFrameIndex());
    if (Op == RISCV::SD)
      if (auto Frame = selectorSpill(MI)) {
        auto Value = Read(0);
        if (Value)
          S.Spills[*Frame] = Value;
      }
  }
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef() || !MO.getReg() || MO.getReg() == RISCV::X0)
      continue;
    // An unproved incoming value still has an exact identity after this
    // definition. It cannot be equated with a distinct rematerialization.
    auto Value = Propagates && Result ? Result : SelectorValue{&MI, {}};
    if (Value)
      S.Registers[MO.getReg()] = Value;
    else
      S.Registers.erase(MO.getReg());
  }
}

DenseMap<const MachineInstr *, SelectorValue>
resolveMachineSelectors(const MachineFunction &MF) {
  if (!llvm::any_of(MF, [](const MachineBasicBlock &MBB) {
        return llvm::any_of(MBB, [](const MachineInstr &MI) {
          return isSelectedLoad(MI.getOpcode()) ||
                 isSelectedMMA(MI.getOpcode());
        });
      }))
    return {};
  DenseMap<const MachineBasicBlock *, SelectorValues> Out;
  for (const MachineBasicBlock &MBB : MF)
    Out[&MBB] = SelectorValues();
  auto In = [&](const MachineBasicBlock &MBB) {
    SelectorValues S;
    if (&MBB == &MF.front() || MBB.pred_empty()) {
      S.Top = false;
      for (unsigned Reg = RISCV::X0; Reg <= RISCV::X31; ++Reg)
        S.Registers[Reg] = {nullptr, Register(Reg)};
    }
    for (const MachineBasicBlock *Pred : MBB.predecessors())
      S.meet(Out[Pred]);
    return S;
  };
  bool Changed;
  do {
    Changed = false;
    for (const MachineBasicBlock &MBB : MF) {
      auto Next = In(MBB);
      for (const MachineInstr &MI : MBB)
        transferSelectorValues(MI, Next);
      if (!(Next == Out[&MBB])) {
        Out[&MBB] = std::move(Next);
        Changed = true;
      }
    }
  } while (Changed);
  DenseMap<const MachineInstr *, SelectorValue> Selectors;
  for (const MachineBasicBlock &MBB : MF) {
    auto S = In(MBB);
    for (const MachineInstr &MI : MBB) {
      bool Load = isSelectedLoad(MI.getOpcode());
      if (Load || isSelectedMMA(MI.getOpcode()))
        Selectors[&MI] =
            S.Registers.lookup(MI.getOperand(Load ? 0 : 2).getReg());
      transferSelectorValues(MI, S);
    }
  }
  return Selectors;
}

struct MachineState {
  struct Selection {
    SelectorValue Predicate;
    unsigned FalseReg, TrueReg;
    bool operator==(const Selection &Other) const {
      return Predicate == Other.Predicate && FalseReg == Other.FalseReg &&
             TrueReg == Other.TrueReg;
    }
  };
  uint64_t Bits = Top;
  SmallVector<Selection> SelectedB;
  bool operator==(const MachineState &Other) const {
    return Bits == Other.Bits && SelectedB == Other.SelectedB;
  }
  void meet(const MachineState &Other) {
    Bits &= Other.Bits;
    llvm::erase_if(SelectedB, [&](const Selection &S) {
      return !llvm::is_contained(Other.SelectedB, S);
    });
  }
};
MachineState::Selection
selection(const MachineInstr &MI,
          const DenseMap<const MachineInstr *, SelectorValue> &Selectors) {
  bool Load = isSelectedLoad(MI.getOpcode());
  return {Selectors.lookup(&MI), unsigned(MI.getOperand(Load ? 1 : 3).getImm()),
          unsigned(MI.getOperand(Load ? 2 : 4).getImm())};
}
MachineState transferMachine(
    const MachineBasicBlock &MBB, MachineState S, bool Verify,
    const DenseMap<const MachineInstr *, SelectorValue> &Selectors) {
  for (const MachineInstr &MI : MBB) {
    llvm::erase_if(S.SelectedB, [&](const MachineState::Selection &Selected) {
      return Selected.Predicate.Definition == &MI;
    });
    unsigned Op = MI.getOpcode();
    if (Op == RISCV::BOSC_AME_SETFP8_E4M3) {
      S.Bits |= FP8Mode;
      continue;
    }
    unsigned Index = machineConfig(Op);
    if (Index != 3) {
      S.Bits = configure(S.Bits, Index);
      S.SelectedB.clear();
      continue;
    }
    bool SelectLoad = isSelectedLoad(Op), SelectMMA = isSelectedMMA(Op);
    if (!isAction(Op) && !SelectLoad && !SelectMMA)
      continue;
    const Function &F = MBB.getParent()->getFunction();
    if (Verify && (S.Bits & Config) != Config)
      fail(F, "machine operation requires M/N/K on every incoming path");
    if (SelectLoad || SelectMMA) {
      auto Selected = selection(MI, Selectors);
      if (Verify && !Selected.Predicate)
        fail(F, "selected machine B predicate has no proven scalar value");
      if (Selected.FalseReg >= 16 || Selected.TrueReg >= 16 ||
          Selected.FalseReg == Selected.TrueReg)
        fail(F, "selected machine B alternatives require distinct Tr0..Tr15");
      if (SelectLoad) {
        S.Bits &= ~((uint64_t(1) << Selected.FalseReg) |
                    (uint64_t(1) << Selected.TrueReg));
        if (!llvm::is_contained(S.SelectedB, Selected))
          S.SelectedB.push_back(Selected);
      } else if (Verify) {
        unsigned Acc = MI.getOperand(0).getImm();
        unsigned A = MI.getOperand(1).getImm();
        if (Acc >= 8 || A >= 16 || !(S.Bits & (uint64_t(1) << (32 + Acc))) ||
            !(S.Bits & (uint64_t(1) << A)) ||
            !llvm::is_contained(S.SelectedB, Selected))
          fail(F, "machine selected MMA requires initialized C/A and the same "
                  "live B selector");
      }
      continue;
    }
    bool IsA = Op == RISCV::BOSC_AME_LOADA;
    bool IsB = Op == RISCV::BOSC_AME_LOADB;
    unsigned Dst = MI.getOperand(0).getReg() -
                   (IsA || IsB ? RISCV::BOSC_TR0 : RISCV::BOSC_ACC0);
    auto Read = [&](unsigned Offset, unsigned Reg) {
      if (Verify && !(S.Bits & (uint64_t(1) << (Offset + Reg))))
        fail(F, "machine read of undefined physical AME register");
    };
    if (Op == RISCV::BOSC_AME_MMA || Op == RISCV::BOSC_AME_FPMMA) {
      if (Verify && Op == RISCV::BOSC_AME_FPMMA && !(S.Bits & FP8Mode))
        fail(F, "machine floating MMA requires E4M3 on every incoming path");
      Read(32, Dst);
      Read(0, MI.getOperand(2).getReg() - RISCV::BOSC_TR0);
      Read(16, MI.getOperand(3).getReg() - RISCV::BOSC_TR0);
    } else if (Op == RISCV::BOSC_AME_STOREC) {
      Read(32, Dst);
    } else if (IsA || IsB) {
      if (IsA)
        llvm::erase_if(
            S.SelectedB, [&](const MachineState::Selection &Selected) {
              return Selected.FalseReg == Dst || Selected.TrueReg == Dst;
            });
      S.Bits &= ~((uint64_t(1) | (uint64_t(1) << 16)) << Dst);
      S.Bits |= uint64_t(1) << (Dst + (IsB ? 16 : 0));
    } else {
      S.Bits |= uint64_t(1) << (Dst + 32);
    }
  }
  llvm::sort(S.SelectedB, [](const auto &A, const auto &B) {
    return std::tie(A.Predicate.Definition, A.Predicate.Entry, A.Predicate.Form,
                    A.Predicate.Negated, A.FalseReg, A.TrueReg) <
           std::tie(B.Predicate.Definition, B.Predicate.Entry, B.Predicate.Form,
                    B.Predicate.Negated, B.FalseReg, B.TrueReg);
  });
  return S;
}

bool isAMERegister(Register Reg, const MachineRegisterInfo &MRI) {
  if (Reg.isPhysical())
    return RISCV::BOSCTrRegRegClass.contains(Reg) ||
           RISCV::BOSCAccRegRegClass.contains(Reg) ||
           RISCV::BOSCConfigRegRegClass.contains(Reg);
  const auto *RC = MRI.getRegClassOrNull(Reg);
  return RC &&
         (RC == &RISCV::BOSCTrRegRegClass || RC == &RISCV::BOSCAccRegRegClass ||
          RC == &RISCV::BOSCConfigRegRegClass);
}
} // namespace

bool llvm::isBOSCAMEIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_bosc_ame_setm:
  case Intrinsic::riscv_bosc_ame_setn:
  case Intrinsic::riscv_bosc_ame_setk:
  case Intrinsic::riscv_bosc_ame_setfp8_e4m3:
  case Intrinsic::riscv_bosc_ame_load_a:
  case Intrinsic::riscv_bosc_ame_load_b:
  case Intrinsic::riscv_bosc_ame_load_c:
  case Intrinsic::riscv_bosc_ame_store_c:
  case Intrinsic::riscv_bosc_ame_mma:
  case Intrinsic::riscv_bosc_ame_fp_mma:
    return true;
  case Intrinsic::riscv_bosc_ame_load_b_select:
  case Intrinsic::riscv_bosc_ame_mma_b_select:
    return true;
  default:
    return false;
  }
}

bool llvm::isBOSCAMEOpcode(unsigned Opcode) {
  return machineConfig(Opcode) != 3 || isAction(Opcode) ||
         Opcode == RISCV::BOSC_AME_SETFP8_E4M3 ||
         Opcode == RISCV::PseudoBOSCAMESetM ||
         Opcode == RISCV::PseudoBOSCAMESetN ||
         Opcode == RISCV::PseudoBOSCAMESetK ||
         Opcode == RISCV::PseudoBOSCAMESetFP8E4M3 ||
         Opcode == RISCV::PseudoBOSCAMELoadA ||
         Opcode == RISCV::PseudoBOSCAMELoadB ||
         Opcode == RISCV::PseudoBOSCAMELoadC ||
         Opcode == RISCV::PseudoBOSCAMEStoreC ||
         Opcode == RISCV::PseudoBOSCAMEMMA ||
         Opcode == RISCV::PseudoBOSCAMEFPMMA ||
         Opcode == RISCV::PseudoBOSCAMELoadBSelect ||
         Opcode == RISCV::PseudoBOSCAMEMMABSelect ||
         Opcode == RISCV::PseudoBOSCAMELoadBSelectBound ||
         Opcode == RISCV::PseudoBOSCAMEMMABSelectBound;
}

void llvm::verifyBOSCAMEIR(Function &F) {
  bool Bound = F.hasFnAttribute("trex-bosc-bound-ame");
  for (const Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (isBOSCAMEIntrinsic(II->getIntrinsicID()) && !Bound)
        fail(F, "intrinsic requires trex-bosc-bound-ame=1");
  if (!Bound)
    return;
  if (F.getFnAttribute("trex-bosc-bound-ame").getValueAsString() != "1")
    fail(F, "unsupported bound ABI version");
  DenseMap<const BasicBlock *, State> Out;
  State TopState;
  for (const Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
      bool Load =
          II->getIntrinsicID() == Intrinsic::riscv_bosc_ame_load_b_select;
      if (!Load &&
          II->getIntrinsicID() != Intrinsic::riscv_bosc_ame_mma_b_select)
        continue;
      State::Selection Selection{II->getArgOperand(Load ? 0 : 2),
                                 slot(*II, Load ? 1 : 3, false),
                                 slot(*II, Load ? 2 : 4, false)};
      if (!llvm::is_contained(TopState.SelectedB, Selection))
        TopState.SelectedB.push_back(Selection);
    }
  for (const BasicBlock &BB : F)
    Out[&BB] = TopState;
  auto In = [&](const BasicBlock &BB) {
    State S = BB.isEntryBlock() || pred_empty(&BB) ? State::entry() : TopState;
    for (const BasicBlock *Pred : predecessors(&BB))
      S.meet(Out[Pred]);
    return S;
  };
  bool Changed;
  do {
    Changed = false;
    for (const BasicBlock &BB : F) {
      State Next = transferIR(BB, In(BB), false);
      if (Next != Out[&BB]) {
        Out[&BB] = Next;
        Changed = true;
      }
    }
  } while (Changed);
  for (const BasicBlock &BB : F)
    transferIR(BB, In(BB), true);
}

void llvm::verifyBOSCAMEMachine(const MachineFunction &MF) {
  auto Selectors = resolveMachineSelectors(MF);
  const Function &F = MF.getFunction();
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  bool Bound = F.hasFnAttribute("trex-bosc-bound-ame");
  if (Bound && (!ST.is64Bit() || !ST.hasVendorXBOSCAME()))
    fail(F, "requires bound AME scope and RV64 +xboscame");
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB) {
      bool Action = isAction(MI.getOpcode()) ||
                    isSelectedLoad(MI.getOpcode()) ||
                    isSelectedMMA(MI.getOpcode());
      unsigned Configuration = machineConfig(MI.getOpcode());
      bool Format = MI.getOpcode() == RISCV::BOSC_AME_SETFP8_E4M3;
      bool Native = Action || Configuration != 3 || Format;
      if (Native && !Bound)
        fail(F, "machine operation requires bound AME scope");
      if (Action)
        for (MCPhysReg Reg : {RISCV::BOSC_M, RISCV::BOSC_N, RISCV::BOSC_K})
          if (!MI.readsRegister(Reg, ST.getRegisterInfo()))
            fail(F, "missing AME dimension dependency");
      if (MI.getOpcode() == RISCV::BOSC_AME_FPMMA &&
          !MI.readsRegister(RISCV::BOSC_MTYPE, ST.getRegisterInfo()))
        fail(F, "missing AME floating format dependency");
      if (Format &&
          !MI.modifiesRegister(RISCV::BOSC_MTYPE, ST.getRegisterInfo()))
        fail(F, "missing AME floating format definition");
      if (isSelectedLoad(MI.getOpcode()) || isSelectedMMA(MI.getOpcode())) {
        bool Load = isSelectedLoad(MI.getOpcode());
        auto Selected = selection(MI, Selectors);
        for (unsigned Reg : {Selected.FalseReg, Selected.TrueReg}) {
          if (Reg >= 16)
            fail(F, "selected machine B alternative exceeds Tr0..Tr15");
          bool Present = Load ? MI.modifiesRegister(RISCV::BOSC_TR0 + Reg,
                                                    ST.getRegisterInfo())
                              : MI.readsRegister(RISCV::BOSC_TR0 + Reg,
                                                 ST.getRegisterInfo());
          if (!Present)
            fail(F, "missing selected B alternative dependency");
        }
        if (!Load) {
          unsigned Acc = MI.getOperand(0).getImm();
          unsigned A = MI.getOperand(1).getImm();
          if (Acc >= 8 || A >= 16 ||
              !MI.readsRegister(RISCV::BOSC_ACC0 + Acc, ST.getRegisterInfo()) ||
              !MI.modifiesRegister(RISCV::BOSC_ACC0 + Acc,
                                   ST.getRegisterInfo()) ||
              !MI.readsRegister(RISCV::BOSC_TR0 + A, ST.getRegisterInfo()))
            fail(F, "missing selected MMA A/Acc dependency");
        }
      }
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg() ||
            !isAMERegister(MO.getReg(), MF.getRegInfo()))
          continue;
        if (MO.getReg().isVirtual())
          fail(F, "virtual AME register is forbidden");
        if (MO.isRenamable() || MO.getSubReg())
          fail(F, "renamable or subregister AME operand is forbidden");
        if (!Native)
          fail(F, "unbound AME operation, copy or spill is forbidden");
      }
    }
  if (!Bound)
    return;
  MachineState TopState;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB)
      if (isSelectedLoad(MI.getOpcode()) || isSelectedMMA(MI.getOpcode())) {
        auto Selected = selection(MI, Selectors);
        if (!llvm::is_contained(TopState.SelectedB, Selected))
          TopState.SelectedB.push_back(Selected);
      }
  DenseMap<const MachineBasicBlock *, MachineState> Out;
  for (const MachineBasicBlock &MBB : MF)
    Out[&MBB] = TopState;
  auto In = [&](const MachineBasicBlock &MBB) {
    MachineState S = TopState;
    if (&MBB == &MF.front() || MBB.pred_empty()) {
      S.Bits = 0;
      S.SelectedB.clear();
    }
    for (const MachineBasicBlock *Pred : MBB.predecessors())
      S.meet(Out[Pred]);
    return S;
  };
  bool Changed;
  do {
    Changed = false;
    for (const MachineBasicBlock &MBB : MF) {
      auto Next = transferMachine(MBB, In(MBB), false, Selectors);
      if (!(Next == Out[&MBB])) {
        Out[&MBB] = std::move(Next);
        Changed = true;
      }
    }
  } while (Changed);
  for (const MachineBasicBlock &MBB : MF)
    transferMachine(MBB, In(MBB), true, Selectors);
}

SDValue llvm::lowerBOSCAMEIntrinsic(SDValue Op, SelectionDAG &DAG,
                                    const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isBOSCAMEIntrinsic(ID))
    return SDValue();
  const Function &F = DAG.getMachineFunction().getFunction();
  if (!F.hasFnAttribute("trex-bosc-bound-ame") || !ST.is64Bit() ||
      !ST.hasVendorXBOSCAME())
    fail(F, "requires bound AME scope and RV64 +xboscame");
  unsigned Opcode = 0;
  switch (ID) {
  case Intrinsic::riscv_bosc_ame_setm:
    Opcode = RISCV::PseudoBOSCAMESetM;
    break;
  case Intrinsic::riscv_bosc_ame_setn:
    Opcode = RISCV::PseudoBOSCAMESetN;
    break;
  case Intrinsic::riscv_bosc_ame_setk:
    Opcode = RISCV::PseudoBOSCAMESetK;
    break;
  case Intrinsic::riscv_bosc_ame_setfp8_e4m3:
    Opcode = RISCV::PseudoBOSCAMESetFP8E4M3;
    break;
  case Intrinsic::riscv_bosc_ame_load_a:
    Opcode = RISCV::PseudoBOSCAMELoadA;
    break;
  case Intrinsic::riscv_bosc_ame_load_b:
    Opcode = RISCV::PseudoBOSCAMELoadB;
    break;
  case Intrinsic::riscv_bosc_ame_load_c:
    Opcode = RISCV::PseudoBOSCAMELoadC;
    break;
  case Intrinsic::riscv_bosc_ame_store_c:
    Opcode = RISCV::PseudoBOSCAMEStoreC;
    break;
  case Intrinsic::riscv_bosc_ame_mma:
    Opcode = RISCV::PseudoBOSCAMEMMA;
    break;
  case Intrinsic::riscv_bosc_ame_fp_mma:
    Opcode = RISCV::PseudoBOSCAMEFPMMA;
    break;
  case Intrinsic::riscv_bosc_ame_load_b_select:
    Opcode = RISCV::PseudoBOSCAMELoadBSelect;
    break;
  case Intrinsic::riscv_bosc_ame_mma_b_select:
    Opcode = RISCV::PseudoBOSCAMEMMABSelect;
    break;
  default:
    llvm_unreachable("checked AME intrinsic");
  }
  SDLoc DL(Op);
  bool Configuration = configIndex(ID) != 3;
  bool Format = ID == Intrinsic::riscv_bosc_ame_setfp8_e4m3;
  bool SelectLoad = ID == Intrinsic::riscv_bosc_ame_load_b_select;
  bool SelectMMA = ID == Intrinsic::riscv_bosc_ame_mma_b_select;
  bool MMA = ID == Intrinsic::riscv_bosc_ame_mma ||
             ID == Intrinsic::riscv_bosc_ame_fp_mma || SelectMMA;
  SmallVector<SDValue, 4> Args;
  for (unsigned I = 2; I != Op.getNumOperands(); ++I) {
    SDValue Value = Op.getOperand(I);
    bool Selection = SelectLoad || SelectMMA;
    unsigned Argument = I - 2;
    bool Immediate = Selection ? (SelectLoad ? Argument == 1 || Argument == 2
                                             : Argument != 2)
                               : !Configuration && (I == 2 || MMA);
    if (Immediate) {
      auto *CI = dyn_cast<ConstantSDNode>(Value);
      unsigned Limit = Selection ? (SelectMMA && Argument == 0 ? 8 : 16)
                       : I == 2 && ID != Intrinsic::riscv_bosc_ame_load_a &&
                               ID != Intrinsic::riscv_bosc_ame_load_b
                           ? 8
                           : 16;
      if (!CI || CI->getZExtValue() >= Limit)
        fail(F, "physical operand lost its binding");
      Value = DAG.getTargetConstant(CI->getZExtValue(), DL, MVT::i32);
    } else if (Selection && Argument == (SelectLoad ? 0u : 2u)) {
      Value = DAG.getZExtOrTrunc(Value, DL, MVT::i64);
      Value = DAG.getZeroExtendInReg(Value, DL, MVT::i1);
    }
    Args.push_back(Value);
  }
  Args.push_back(Op.getOperand(0));
  auto VTs = Configuration ? DAG.getVTList(MVT::i64, MVT::Other)
                           : DAG.getVTList(MVT::Other);
  auto *Node = DAG.getMachineNode(Opcode, DL, VTs, Args);
  if (!Configuration && !Format && !MMA) {
    auto Flags = ID == Intrinsic::riscv_bosc_ame_store_c
                     ? MachineMemOperand::MOStore
                     : MachineMemOperand::MOLoad;
    auto *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(), Flags | MachineMemOperand::MOVolatile,
        LocationSize::beforeOrAfterPointer(), Align(1));
    DAG.setNodeMemRefs(Node, {MMO});
  }
  return SDValue(Node, 0);
}

MachineBasicBlock *llvm::emitBOSCAMEInstruction(MachineInstr &MI,
                                                MachineBasicBlock *MBB,
                                                const RISCVSubtarget &ST) {
  if (MI.getOpcode() == RISCV::PseudoBOSCAMELoadBSelect ||
      MI.getOpcode() == RISCV::PseudoBOSCAMEMMABSelect) {
    bool Load = MI.getOpcode() == RISCV::PseudoBOSCAMELoadBSelect;
    auto MIB = BuildMI(
        *MBB, MI, MI.getDebugLoc(),
        ST.getInstrInfo()->get(Load ? RISCV::PseudoBOSCAMELoadBSelectBound
                                    : RISCV::PseudoBOSCAMEMMABSelectBound));
    for (unsigned I = 0; I != 5; ++I)
      MIB.add(MI.getOperand(I));
    for (unsigned I : {Load ? 1u : 3u, Load ? 2u : 4u})
      MIB.addReg(RISCV::BOSC_TR0 + MI.getOperand(I).getImm(),
                 Load ? RegState::ImplicitDefine : RegState::Implicit);
    if (!Load) {
      unsigned Acc = RISCV::BOSC_ACC0 + MI.getOperand(0).getImm();
      MIB.addReg(Acc, RegState::Implicit)
          .addReg(Acc, RegState::ImplicitDefine)
          .addReg(RISCV::BOSC_TR0 + MI.getOperand(1).getImm(),
                  RegState::Implicit);
    }
    MIB.cloneMemRefs(MI);
    MI.eraseFromParent();
    return MBB;
  }
  unsigned Native;
  switch (MI.getOpcode()) {
  case RISCV::PseudoBOSCAMESetM:
    Native = RISCV::BOSC_AME_SETM;
    break;
  case RISCV::PseudoBOSCAMESetN:
    Native = RISCV::BOSC_AME_SETN;
    break;
  case RISCV::PseudoBOSCAMESetK:
    Native = RISCV::BOSC_AME_SETK;
    break;
  case RISCV::PseudoBOSCAMESetFP8E4M3:
    Native = RISCV::BOSC_AME_SETFP8_E4M3;
    break;
  case RISCV::PseudoBOSCAMELoadA:
    Native = RISCV::BOSC_AME_LOADA;
    break;
  case RISCV::PseudoBOSCAMELoadB:
    Native = RISCV::BOSC_AME_LOADB;
    break;
  case RISCV::PseudoBOSCAMELoadC:
    Native = RISCV::BOSC_AME_LOADC;
    break;
  case RISCV::PseudoBOSCAMEStoreC:
    Native = RISCV::BOSC_AME_STOREC;
    break;
  case RISCV::PseudoBOSCAMEMMA:
    Native = RISCV::BOSC_AME_MMA;
    break;
  case RISCV::PseudoBOSCAMEFPMMA:
    Native = RISCV::BOSC_AME_FPMMA;
    break;
  default:
    return nullptr;
  }
  auto MIB =
      BuildMI(*MBB, MI, MI.getDebugLoc(), ST.getInstrInfo()->get(Native));
  if (Native == RISCV::BOSC_AME_SETFP8_E4M3) {
    MIB.addReg(RISCV::X0, RegState::Define);
  } else if (machineConfig(Native) != 3) {
    MIB.add(MI.getOperand(0)).add(MI.getOperand(1));
  } else if (Native == RISCV::BOSC_AME_MMA || Native == RISCV::BOSC_AME_FPMMA) {
    unsigned Dst = RISCV::BOSC_ACC0 + MI.getOperand(0).getImm();
    MIB.addReg(Dst, RegState::Define)
        .addReg(Dst)
        .addReg(RISCV::BOSC_TR0 + MI.getOperand(1).getImm())
        .addReg(RISCV::BOSC_TR0 + MI.getOperand(2).getImm());
  } else {
    bool Tile =
        Native == RISCV::BOSC_AME_LOADA || Native == RISCV::BOSC_AME_LOADB;
    unsigned Reg =
        (Tile ? RISCV::BOSC_TR0 : RISCV::BOSC_ACC0) + MI.getOperand(0).getImm();
    MIB.addReg(Reg, Native == RISCV::BOSC_AME_STOREC ? RegState::NoFlags
                                                     : RegState::Define)
        .add(MI.getOperand(1))
        .add(MI.getOperand(2));
    MIB.cloneMemRefs(MI);
  }
  MI.eraseFromParent();
  return MBB;
}
