//===-- RISCVBOSCAME.cpp ---------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVBOSCAME.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>

using namespace llvm;

namespace {
constexpr uint64_t TileA = 0xffff;
constexpr uint64_t TileB = TileA << 16;
constexpr uint64_t Acc = uint64_t(0xff) << 32;
constexpr uint64_t Config = uint64_t(7) << 40;
constexpr uint64_t Top = TileA | TileB | Acc | Config;

[[noreturn]] void fail(const Function &F, const Twine &Message) {
  reportFatalUsageError("BOSC bound AME function '" + F.getName() +
                       "': " + Message);
}

unsigned configIndex(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_bosc_ame_setm: return 0;
  case Intrinsic::riscv_bosc_ame_setn: return 1;
  case Intrinsic::riscv_bosc_ame_setk: return 2;
  default: return 3;
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

unsigned request(const IntrinsicInst &II) {
  const auto *Value = dyn_cast<ConstantInt>(II.getArgOperand(0));
  if (!Value || Value->getValue().getActiveBits() > 7 ||
      !Value->getZExtValue() || Value->getZExtValue() > 64)
    fail(*II.getFunction(), "dimension request must be a constant in 1..64");
  return Value->getZExtValue();
}

// Three exact dimension facts accompany A/B/Acc must-definitions. A tile load
// carries a role: a later load of B into the same Tr kills its prior A value.
// LLVM never substitutes an arbitrary vector type for these physical facts.
struct State {
  uint64_t Bits = Top;
  std::array<unsigned, 3> Dimensions{~0u, ~0u, ~0u};
  static State entry() { return {0, {0, 0, 0}}; }
  bool operator==(const State &Other) const {
    return Bits == Other.Bits && Dimensions == Other.Dimensions;
  }
  bool operator!=(const State &Other) const { return !(*this == Other); }
  void meet(const State &Other) {
    Bits &= Other.Bits;
    for (unsigned I = 0; I < 3; ++I) {
      if (Dimensions[I] == ~0u)
        Dimensions[I] = Other.Dimensions[I];
      else if (Other.Dimensions[I] != ~0u &&
               Dimensions[I] != Other.Dimensions[I])
        Dimensions[I] = 0;
    }
  }
};

State transferIR(const BasicBlock &BB, State S, bool Verify) {
  for (const Instruction &I : BB) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !isBOSCAMEIntrinsic(II->getIntrinsicID()))
      continue;
    auto ID = II->getIntrinsicID();
    unsigned Index = configIndex(ID);
    if (Index != 3) {
      S.Bits = configure(S.Bits, Index);
      S.Dimensions[Index] = request(*II);
      continue;
    }
    if (Verify && ((S.Bits & Config) != Config ||
                   llvm::is_contained(S.Dimensions, 0u) ||
                   llvm::is_contained(S.Dimensions, ~0u)))
      fail(*BB.getParent(), "operation requires M/N/K configuration with "
                            "matching dimensions on every incoming path");
    bool IsA = ID == Intrinsic::riscv_bosc_ame_load_a;
    bool IsB = ID == Intrinsic::riscv_bosc_ame_load_b;
    bool IsMMA = ID == Intrinsic::riscv_bosc_ame_mma;
    bool IsStore = ID == Intrinsic::riscv_bosc_ame_store_c;
    unsigned Dst = slot(*II, 0, !IsA && !IsB);
    if (Verify && !IsMMA) {
      auto *Stride = dyn_cast<ConstantInt>(II->getArgOperand(2));
      uint64_t RowBytes = IsA || IsB ? S.Dimensions[2] : 4 * S.Dimensions[1];
      if (!Stride || Stride->getValue().isNegative() ||
          Stride->getZExtValue() < RowBytes ||
          ((!IsA && !IsB) && Stride->getZExtValue() % 4))
        fail(*BB.getParent(), "byte stride must be a constant covering the "
                              "row; accumulator stride must be 4-byte aligned");
    }
    auto Read = [&](unsigned Offset, unsigned Register, StringRef Role) {
      if (Verify && !(S.Bits & (uint64_t(1) << (Offset + Register))))
        fail(*BB.getParent(), "read of undefined " + Role + " register " +
                                  Twine(Register));
    };
    if (IsMMA) {
      Read(0, slot(*II, 1, false), "A tile");
      Read(16, slot(*II, 2, false), "B tile");
      Read(32, Dst, "accumulator");
    } else if (IsStore) {
      Read(32, Dst, "accumulator");
    } else if (IsA || IsB) {
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
  case RISCV::BOSC_AME_SETM: return 0;
  case RISCV::BOSC_AME_SETN: return 1;
  case RISCV::BOSC_AME_SETK: return 2;
  default: return 3;
  }
}

bool isAction(unsigned Opcode) {
  return Opcode == RISCV::BOSC_AME_LOADA || Opcode == RISCV::BOSC_AME_LOADB ||
         Opcode == RISCV::BOSC_AME_LOADC || Opcode == RISCV::BOSC_AME_STOREC ||
         Opcode == RISCV::BOSC_AME_MMA;
}

uint64_t transferMachine(const MachineBasicBlock &MBB, uint64_t Bits,
                         bool Verify) {
  for (const MachineInstr &MI : MBB) {
    unsigned Op = MI.getOpcode();
    unsigned Index = machineConfig(Op);
    if (Index != 3) {
      Bits = configure(Bits, Index);
      continue;
    }
    if (!isAction(Op))
      continue;
    const Function &F = MBB.getParent()->getFunction();
    if (Verify && (Bits & Config) != Config)
      fail(F, "machine operation requires M/N/K on every incoming path");
    bool IsA = Op == RISCV::BOSC_AME_LOADA;
    bool IsB = Op == RISCV::BOSC_AME_LOADB;
    unsigned Dst = MI.getOperand(0).getReg() -
                   (IsA || IsB ? RISCV::BOSC_TR0 : RISCV::BOSC_ACC0);
    auto Read = [&](unsigned Offset, unsigned Reg) {
      if (Verify && !(Bits & (uint64_t(1) << (Offset + Reg))))
        fail(F, "machine read of undefined physical AME register");
    };
    if (Op == RISCV::BOSC_AME_MMA) {
      Read(32, Dst);
      Read(0, MI.getOperand(2).getReg() - RISCV::BOSC_TR0);
      Read(16, MI.getOperand(3).getReg() - RISCV::BOSC_TR0);
    } else if (Op == RISCV::BOSC_AME_STOREC) {
      Read(32, Dst);
    } else if (IsA || IsB) {
      Bits &= ~((uint64_t(1) | (uint64_t(1) << 16)) << Dst);
      Bits |= uint64_t(1) << (Dst + (IsB ? 16 : 0));
    } else {
      Bits |= uint64_t(1) << (Dst + 32);
    }
  }
  return Bits;
}

bool isAMERegister(Register Reg, const MachineRegisterInfo &MRI) {
  if (Reg.isPhysical())
    return RISCV::BOSCTrRegRegClass.contains(Reg) ||
           RISCV::BOSCAccRegRegClass.contains(Reg) ||
           RISCV::BOSCConfigRegRegClass.contains(Reg);
  const auto *RC = MRI.getRegClassOrNull(Reg);
  return RC && (RC == &RISCV::BOSCTrRegRegClass ||
                RC == &RISCV::BOSCAccRegRegClass ||
                RC == &RISCV::BOSCConfigRegRegClass);
}
} // namespace

bool llvm::isBOSCAMEIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_bosc_ame_setm:
  case Intrinsic::riscv_bosc_ame_setn:
  case Intrinsic::riscv_bosc_ame_setk:
  case Intrinsic::riscv_bosc_ame_load_a:
  case Intrinsic::riscv_bosc_ame_load_b:
  case Intrinsic::riscv_bosc_ame_load_c:
  case Intrinsic::riscv_bosc_ame_store_c:
  case Intrinsic::riscv_bosc_ame_mma: return true;
  default: return false;
  }
}

bool llvm::isBOSCAMEOpcode(unsigned Opcode) {
  return machineConfig(Opcode) != 3 || isAction(Opcode) ||
         Opcode == RISCV::PseudoBOSCAMESetM ||
         Opcode == RISCV::PseudoBOSCAMESetN ||
         Opcode == RISCV::PseudoBOSCAMESetK ||
         Opcode == RISCV::PseudoBOSCAMELoadA ||
         Opcode == RISCV::PseudoBOSCAMELoadB ||
         Opcode == RISCV::PseudoBOSCAMELoadC ||
         Opcode == RISCV::PseudoBOSCAMEStoreC ||
         Opcode == RISCV::PseudoBOSCAMEMMA;
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
  for (const BasicBlock &BB : F)
    Out[&BB] = State{};
  auto In = [&](const BasicBlock &BB) {
    State S = BB.isEntryBlock() || pred_empty(&BB) ? State::entry() : State{};
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
  const Function &F = MF.getFunction();
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  bool Bound = F.hasFnAttribute("trex-bosc-bound-ame");
  if (Bound && (!ST.is64Bit() || !ST.hasVendorXBOSCAME()))
    fail(F, "requires bound AME scope and RV64 +xboscame");
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB) {
      bool Action = isAction(MI.getOpcode());
      unsigned Configuration = machineConfig(MI.getOpcode());
      bool Native = Action || Configuration != 3;
      if (Native && !Bound)
        fail(F, "machine operation requires bound AME scope");
      if (Action)
        for (MCPhysReg Reg : RISCV::BOSCConfigRegRegClass)
          if (!MI.readsRegister(Reg, ST.getRegisterInfo()))
            fail(F, "missing AME dimension dependency");
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
  DenseMap<const MachineBasicBlock *, uint64_t> Out;
  for (const MachineBasicBlock &MBB : MF)
    Out[&MBB] = Top;
  auto In = [&](const MachineBasicBlock &MBB) {
    uint64_t Bits = &MBB == &MF.front() || MBB.pred_empty() ? 0 : Top;
    for (const MachineBasicBlock *Pred : MBB.predecessors())
      Bits &= Out[Pred];
    return Bits;
  };
  bool Changed;
  do {
    Changed = false;
    for (const MachineBasicBlock &MBB : MF) {
      uint64_t Next = transferMachine(MBB, In(MBB), false);
      if (Next != Out[&MBB]) {
        Out[&MBB] = Next;
        Changed = true;
      }
    }
  } while (Changed);
  for (const MachineBasicBlock &MBB : MF)
    transferMachine(MBB, In(MBB), true);
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
  case Intrinsic::riscv_bosc_ame_setm: Opcode = RISCV::PseudoBOSCAMESetM; break;
  case Intrinsic::riscv_bosc_ame_setn: Opcode = RISCV::PseudoBOSCAMESetN; break;
  case Intrinsic::riscv_bosc_ame_setk: Opcode = RISCV::PseudoBOSCAMESetK; break;
  case Intrinsic::riscv_bosc_ame_load_a: Opcode = RISCV::PseudoBOSCAMELoadA; break;
  case Intrinsic::riscv_bosc_ame_load_b: Opcode = RISCV::PseudoBOSCAMELoadB; break;
  case Intrinsic::riscv_bosc_ame_load_c: Opcode = RISCV::PseudoBOSCAMELoadC; break;
  case Intrinsic::riscv_bosc_ame_store_c: Opcode = RISCV::PseudoBOSCAMEStoreC; break;
  case Intrinsic::riscv_bosc_ame_mma: Opcode = RISCV::PseudoBOSCAMEMMA; break;
  default: llvm_unreachable("checked AME intrinsic");
  }
  SDLoc DL(Op);
  bool Configuration = configIndex(ID) != 3;
  bool MMA = ID == Intrinsic::riscv_bosc_ame_mma;
  SmallVector<SDValue, 4> Args;
  for (unsigned I = 2; I != Op.getNumOperands(); ++I) {
    SDValue Value = Op.getOperand(I);
    if (!Configuration && (I == 2 || MMA)) {
      auto *CI = dyn_cast<ConstantSDNode>(Value);
      unsigned Limit = I == 2 && ID != Intrinsic::riscv_bosc_ame_load_a &&
                                   ID != Intrinsic::riscv_bosc_ame_load_b ? 8 : 16;
      if (!CI || CI->getZExtValue() >= Limit)
        fail(F, "physical operand lost its binding");
      Value = DAG.getTargetConstant(CI->getZExtValue(), DL, MVT::i32);
    }
    Args.push_back(Value);
  }
  Args.push_back(Op.getOperand(0));
  auto VTs = Configuration ? DAG.getVTList(MVT::i64, MVT::Other)
                           : DAG.getVTList(MVT::Other);
  auto *Node = DAG.getMachineNode(Opcode, DL, VTs, Args);
  if (!Configuration && !MMA) {
    auto Flags = ID == Intrinsic::riscv_bosc_ame_store_c
                     ? MachineMemOperand::MOStore : MachineMemOperand::MOLoad;
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
  unsigned Native;
  switch (MI.getOpcode()) {
  case RISCV::PseudoBOSCAMESetM: Native = RISCV::BOSC_AME_SETM; break;
  case RISCV::PseudoBOSCAMESetN: Native = RISCV::BOSC_AME_SETN; break;
  case RISCV::PseudoBOSCAMESetK: Native = RISCV::BOSC_AME_SETK; break;
  case RISCV::PseudoBOSCAMELoadA: Native = RISCV::BOSC_AME_LOADA; break;
  case RISCV::PseudoBOSCAMELoadB: Native = RISCV::BOSC_AME_LOADB; break;
  case RISCV::PseudoBOSCAMELoadC: Native = RISCV::BOSC_AME_LOADC; break;
  case RISCV::PseudoBOSCAMEStoreC: Native = RISCV::BOSC_AME_STOREC; break;
  case RISCV::PseudoBOSCAMEMMA: Native = RISCV::BOSC_AME_MMA; break;
  default: return nullptr;
  }
  auto MIB = BuildMI(*MBB, MI, MI.getDebugLoc(), ST.getInstrInfo()->get(Native));
  if (machineConfig(Native) != 3) {
    MIB.add(MI.getOperand(0)).add(MI.getOperand(1));
  } else if (Native == RISCV::BOSC_AME_MMA) {
    unsigned Dst = RISCV::BOSC_ACC0 + MI.getOperand(0).getImm();
    MIB.addReg(Dst, RegState::Define).addReg(Dst)
        .addReg(RISCV::BOSC_TR0 + MI.getOperand(1).getImm())
        .addReg(RISCV::BOSC_TR0 + MI.getOperand(2).getImm());
  } else {
    bool Tile = Native == RISCV::BOSC_AME_LOADA || Native == RISCV::BOSC_AME_LOADB;
    unsigned Reg = (Tile ? RISCV::BOSC_TR0 : RISCV::BOSC_ACC0) +
                   MI.getOperand(0).getImm();
    MIB.addReg(Reg, Native == RISCV::BOSC_AME_STOREC ? RegState::NoFlags
                                                    : RegState::Define)
        .add(MI.getOperand(1)).add(MI.getOperand(2));
    MIB.cloneMemRefs(MI);
  }
  MI.eraseFromParent();
  return MBB;
}
