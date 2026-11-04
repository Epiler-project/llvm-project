//===-- RISCVBOSCBound.cpp ------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVBOSCBound.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCVBOSCAME.h"
#include "RISCVISelLowering.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/RISCV/RISCVBOSC.h"
#include <array>
#include <limits>
#include <optional>
#include <tuple>

using namespace llvm;

namespace {
constexpr unsigned E32M1TaMa = 0xd0;

// A known configuration writes a prefix of a physical group. An unknown VL
// may be consumed only under the same configuration occurrence. Configurations
// are tracked separately: their union is a least fixed point, whereas lane
// definitions use a greatest fixed point and meet by minimum prefix.
struct DefinedLanes {
  unsigned Prefix = std::numeric_limits<unsigned>::max();
  const void *DynamicConfig = nullptr;

  bool isTop() const {
    return !DynamicConfig && Prefix == std::numeric_limits<unsigned>::max();
  }
  bool operator==(const DefinedLanes &Other) const {
    return Prefix == Other.Prefix && DynamicConfig == Other.DynamicConfig;
  }
};

std::optional<DefinedLanes> meetLanes(std::optional<DefinedLanes> LHS,
                                      std::optional<DefinedLanes> RHS) {
  if (!LHS || !RHS)
    return std::nullopt;
  if (LHS->isTop())
    return RHS;
  if (RHS->isTop())
    return LHS;
  DefinedLanes Result{
      std::min(LHS->Prefix, RHS->Prefix),
      LHS->DynamicConfig == RHS->DynamicConfig ? LHS->DynamicConfig : nullptr};
  if (!Result.Prefix && !Result.DynamicConfig)
    return std::nullopt;
  return Result;
}

struct LaneState {
  std::array<std::optional<DefinedLanes>, 32> Initialized;
  DenseMap<const void *, std::optional<unsigned>> Configurations;

  static LaneState top() {
    LaneState State;
    State.Initialized.fill(DefinedLanes{});
    return State;
  }
  static LaneState entry() {
    LaneState State;
    State.Configurations[nullptr] = std::nullopt;
    return State;
  }
  bool operator==(const LaneState &Other) const {
    return Initialized == Other.Initialized &&
           Configurations == Other.Configurations;
  }
  bool operator!=(const LaneState &Other) const { return !(*this == Other); }
  void meet(const LaneState &Other) {
    for (unsigned Reg = 0; Reg != 32; ++Reg)
      Initialized[Reg] = meetLanes(Initialized[Reg], Other.Initialized[Reg]);
    Configurations.insert(Other.Configurations.begin(),
                          Other.Configurations.end());
  }
  void configure(const void *Token, std::optional<unsigned> Lanes) {
    // Scalar seeds survive imprecise transitions, but full-prefix proofs and
    // dynamic-occurrence tokens cannot be promoted across unknown VL values.
    bool Precise = Lanes && llvm::all_of(Configurations, [](const auto &Entry) {
                     return Entry.first && Entry.second.has_value();
                   });
    for (auto &Definition : Initialized)
      if (Definition) {
        if (!Precise)
          Definition->Prefix = std::min(Definition->Prefix, 1u);
        Definition->DynamicConfig = nullptr;
        if (!Definition->Prefix)
          Definition = std::nullopt;
      }
    Configurations.clear();
    Configurations[Token] = Lanes;
  }
  bool configured() const {
    return !Configurations.empty() && !Configurations.count(nullptr);
  }
  bool readable(unsigned Reg) const {
    if (!Initialized[Reg])
      return false;
    const DefinedLanes &Definition = *Initialized[Reg];
    for (const auto &[Token, Lanes] : Configurations) {
      if (Lanes) {
        if (Definition.Prefix < *Lanes)
          return false;
      } else if (Definition.DynamicConfig != Token)
        return false;
    }
    return true;
  }
  bool readableZero(unsigned Reg) const {
    return Initialized[Reg] && Initialized[Reg]->Prefix > 0;
  }
  bool noActiveLanes() const {
    return !Configurations.empty() &&
           llvm::all_of(Configurations,
                        [](const auto &Entry) { return Entry.second == 0; });
  }
  bool readableLane(unsigned Reg, unsigned Lane) const {
    return Initialized[Reg] && Initialized[Reg]->Prefix > Lane;
  }
  void reduce(unsigned Reg) {
    // The receiver always uses the destination itself as seed. With VL=0 it
    // is preserved; otherwise reduction defines lane zero and poisons tails.
    if (llvm::all_of(Configurations,
                     [](const auto &Entry) { return Entry.second == 0; }))
      return;
    Initialized[Reg] = DefinedLanes{1, nullptr};
  }
  void define(unsigned Reg) {
    std::optional<DefinedLanes> Written = DefinedLanes{};
    for (const auto &[Token, Lanes] : Configurations) {
      // VL=0 writes no elements and does not apply tail-agnostic changes.
      // Nonzero ta writes invalidate this destination's old tail only.
      std::optional<DefinedLanes> OnPath = Initialized[Reg];
      if (!Token)
        OnPath = std::nullopt;
      else if (Lanes != 0)
        OnPath = Lanes ? DefinedLanes{*Lanes, nullptr}
                       : DefinedLanes{readableZero(Reg) ? 1u : 0u, Token};
      Written = meetLanes(Written, OnPath);
    }
    Initialized[Reg] = Written;
  }
};

std::optional<unsigned> exactLanes(uint64_t AVL, bool Exact512) {
  // +zvl512b is a lower bound, not an exact VLEN. Values through 16 are exact
  // on every supported target; clamping larger values needs exact VLEN=512.
  if (AVL <= 16)
    return AVL;
  if (Exact512 && AVL >= 32)
    return 16;
  // VLMAX < AVL < 2*VLMAX permits an implementation-selected VL.
  return std::nullopt;
}

Error invalid(const Function &F, const Twine &Message) {
  return createStringError("BOSC bound RVV function '" + F.getName() +
                           "': " + Message);
}

[[noreturn]] void fail(const Function &F, const Twine &Message) {
  reportFatalUsageError(Twine(toString(invalid(F, Message))));
}

bool isBOSCFPRVVIntrinsic(Intrinsic::ID ID);
bool isBOSCSFUIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_sfu_exp2 ||
         ID == Intrinsic::riscv_bosc_sfu_rcp;
}

bool isScalarObservationIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::readcyclecounter ||
         ID == Intrinsic::riscv_bosc_read_mcycle;
}

bool isBoundIntrinsic(Intrinsic::ID ID) {
  // Scalar counter observations neither create VRF values nor change
  // AME/DMA ownership. The BOSC intrinsic selects its exact target timebase;
  // the standard LLVM intrinsic retains its ordinary target semantics.
  if (isScalarObservationIntrinsic(ID))
    return true;
  return ID == Intrinsic::riscv_bosc_setvl ||
         ID == Intrinsic::riscv_bosc_read_bits ||
         ID == Intrinsic::riscv_bosc_load || ID == Intrinsic::riscv_bosc_add ||
         ID == Intrinsic::riscv_bosc_maximum ||
         ID == Intrinsic::riscv_bosc_isplat ||
         ID == Intrinsic::riscv_bosc_store || isBOSCFPRVVIntrinsic(ID) ||
         ID == Intrinsic::riscv_bosc_merge ||
         ID == Intrinsic::riscv_bosc_iadd_scalar ||
         ID == Intrinsic::riscv_bosc_gather_scalar ||
         ID == Intrinsic::riscv_bosc_shl || isBOSCSFUIntrinsic(ID);
}

bool isBOSCFPRVVIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_fadd || ID == Intrinsic::riscv_bosc_fsub ||
         ID == Intrinsic::riscv_bosc_fmul || ID == Intrinsic::riscv_bosc_fdiv ||
         ID == Intrinsic::riscv_bosc_fsplat ||
         ID == Intrinsic::riscv_bosc_fextract ||
         ID == Intrinsic::riscv_bosc_freduce_sum ||
         ID == Intrinsic::riscv_bosc_freduce_max ||
         ID == Intrinsic::riscv_bosc_fmax ||
         ID == Intrinsic::riscv_bosc_fmacc ||
         ID == Intrinsic::riscv_bosc_fcvt_to_i32 ||
         ID == Intrinsic::riscv_bosc_fcvt_from_i32 ||
         ID == Intrinsic::riscv_bosc_fcmp_lt ||
         ID == Intrinsic::riscv_bosc_fcmp_eq;
}

bool isOrderedIntegerContentsIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_rvv_contents_add ||
         ID == Intrinsic::riscv_bosc_rvv_contents_maximum ||
         ID == Intrinsic::riscv_bosc_rvv_contents_isplat ||
         ID == Intrinsic::riscv_bosc_rvv_contents_iadd_scalar ||
         ID == Intrinsic::riscv_bosc_rvv_contents_shl ||
         ID == Intrinsic::riscv_bosc_rvv_contents_gather_scalar;
}

bool requiresRVVF32(Intrinsic::ID ID) {
  return isBOSCFPRVVIntrinsic(ID) ||
         (isBOSCOrderedRVVContentsIntrinsic(ID) &&
          !isOrderedIntegerContentsIntrinsic(ID) &&
          ID != Intrinsic::riscv_bosc_rvv_contents_load &&
          ID != Intrinsic::riscv_bosc_rvv_contents_store &&
          ID != Intrinsic::riscv_bosc_read_bits_contents);
}
bool isBoundUnaryIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_fcvt_to_i32 ||
         ID == Intrinsic::riscv_bosc_fcvt_from_i32 ||
         ID == Intrinsic::riscv_bosc_iadd_scalar ||
         ID == Intrinsic::riscv_bosc_shl || isBOSCSFUIntrinsic(ID);
}
bool isMaskOperand(Intrinsic::ID ID, unsigned Index) {
  return ((ID == Intrinsic::riscv_bosc_fcmp_lt ||
           ID == Intrinsic::riscv_bosc_fcmp_eq) &&
          Index == 0) ||
         (ID == Intrinsic::riscv_bosc_merge && Index == 3);
}

bool isBoundBinaryIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_add || ID == Intrinsic::riscv_bosc_fadd ||
         ID == Intrinsic::riscv_bosc_maximum ||
         ID == Intrinsic::riscv_bosc_fsub || ID == Intrinsic::riscv_bosc_fmul ||
         ID == Intrinsic::riscv_bosc_fdiv || ID == Intrinsic::riscv_bosc_fmax ||
         ID == Intrinsic::riscv_bosc_fmacc ||
         ID == Intrinsic::riscv_bosc_fcmp_lt ||
         ID == Intrinsic::riscv_bosc_fcmp_eq;
}
bool isBoundReduceIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_freduce_sum ||
         ID == Intrinsic::riscv_bosc_freduce_max;
}
unsigned boundPhysicalOperands(Intrinsic::ID ID) {
  if (ID == Intrinsic::riscv_bosc_merge)
    return 4;
  if (isBoundBinaryIntrinsic(ID))
    return 3;
  return isBoundReduceIntrinsic(ID) || isBoundUnaryIntrinsic(ID) ||
                 ID == Intrinsic::riscv_bosc_gather_scalar
             ? 2
             : 1;
}

Expected<unsigned> gatherIndex(const IntrinsicInst &II, bool Exact512) {
  const auto *Index = dyn_cast<ConstantInt>(II.getArgOperand(2));
  if (!Index || Index->getValue().ugt(16))
    return invalid(*II.getFunction(),
                   "gather index must be a constant in 0..16");
  unsigned Lane = Index->getZExtValue();
  if (Lane == 16 && !Exact512)
    return invalid(*II.getFunction(),
                   "gather index 16 requires exact VLEN=512");
  return Lane;
}

bool isDMAIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_dma_writes ||
         ID == Intrinsic::riscv_bosc_dma_reads ||
         ID == Intrinsic::riscv_bosc_dma_fence;
}

bool hasDMAIntrinsic(const Function &F) {
  return llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && isDMAIntrinsic(II->getIntrinsicID());
  });
}

bool requiresBOSCExplicitVectorState(const Function &F) {
  // The contents ABI owns real vector SSA and lets LLVM allocate its registers.
  // Fixed-register and standalone DMA functions instead admit scalar state and
  // explicit bound operations. Scalar memory combines must not invent VRF
  // contents merely because the target also supports the V extension.
  return !isBOSCContentsRVVFunction(F) &&
         (isBOSCBoundFunction(F) || hasDMAIntrinsic(F));
}

bool hasBOSCObservationIntrinsic(const Function &F) {
  return llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && II->getIntrinsicID() == Intrinsic::riscv_bosc_read_mcycle;
  });
}

Expected<unsigned> physicalArgument(const IntrinsicInst &II, unsigned Index) {
  auto *CI = dyn_cast<ConstantInt>(II.getArgOperand(Index));
  if (isMaskOperand(II.getIntrinsicID(), Index)) {
    if (!CI || !CI->isZero())
      return invalid(*II.getFunction(), "physical mask operand must be v0");
  } else if (!CI || CI->getValue().getActiveBits() > 5 || !CI->getZExtValue())
    return invalid(
        *II.getFunction(),
        "physical vector register must be in v1..v31; v0 is reserved");
  return CI->getZExtValue();
}

bool containsVector(Type *Ty) {
  if (Ty->isVectorTy())
    return true;
  if (auto *ST = dyn_cast<StructType>(Ty))
    return llvm::any_of(ST->elements(), containsVector);
  if (auto *AT = dyn_cast<ArrayType>(Ty))
    return containsVector(AT->getElementType());
  return false;
}

bool containsFloatingVector(Type *Ty) {
  if (auto *VT = dyn_cast<VectorType>(Ty))
    return VT->getElementType()->isFloatingPointTy();
  if (auto *ST = dyn_cast<StructType>(Ty))
    return llvm::any_of(ST->elements(), containsFloatingVector);
  return false;
}

bool isContentsRVVIntrinsic(Intrinsic::ID ID) {
  return RISCVVIntrinsicsTable::getRISCVVIntrinsicInfo(ID) ||
         ID == Intrinsic::riscv_vsetvli || ID == Intrinsic::riscv_vsetvlimax ||
         isBOSCOrderedRVVContentsIntrinsic(ID) ||
         isBOSCContentsSFUIntrinsic(ID);
}

bool isActualE32M1VL(const Value *VL) {
  SmallVector<const Value *, 8> Worklist{VL};
  SmallPtrSet<const Value *, 16> Visited;
  SmallPtrSet<const Value *, 16> Seeded;
  SmallVector<const PHINode *, 8> PHIs;
  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
      if (!CI->isZero())
        return false;
      Seeded.insert(V);
      continue;
    }
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      PHIs.push_back(Phi);
      Worklist.append(Phi->incoming_values().begin(),
                      Phi->incoming_values().end());
      continue;
    }
    auto *II = dyn_cast<IntrinsicInst>(V);
    if (!II)
      return false;
    unsigned SEWIndex;
    if (II->getIntrinsicID() == Intrinsic::riscv_vsetvli)
      SEWIndex = 1;
    else if (II->getIntrinsicID() == Intrinsic::riscv_vsetvlimax)
      SEWIndex = 0;
    else
      return false;
    auto *SEW = dyn_cast<ConstantInt>(II->getArgOperand(SEWIndex));
    auto *LMUL = dyn_cast<ConstantInt>(II->getArgOperand(SEWIndex + 1));
    if (!SEW || SEW->getZExtValue() != 2 || !LMUL || !LMUL->isZero())
      return false;
    Seeded.insert(V);
  }
  // Every leaf must be an actual result or zero, and every PHI cycle must
  // reach such a seed. Merely revisiting a PHI cannot validate a seedless SCC.
  bool Changed;
  do {
    Changed = false;
    for (const PHINode *Phi : PHIs)
      if (!Seeded.contains(Phi) &&
          llvm::any_of(Phi->incoming_values(), [&](const Value *Incoming) {
            return Seeded.contains(Incoming);
          })) {
        Seeded.insert(Phi);
        Changed = true;
      }
  } while (Changed);
  return llvm::all_of(PHIs,
                      [&](const PHINode *Phi) { return Seeded.contains(Phi); });
}

struct ContentsVLRange {
  unsigned Minimum = 16;
  bool MayBePositive = false;
};

// Only a prefix through lane 15 is needed by the immediate gather receiver.
// The architectural middle AVL interval guarantees ceil(AVL/2), not VLMAX.
ContentsVLRange contentsVLRange(const Value *VL,
                                const Value *PositiveVL = nullptr) {
  ContentsVLRange Result;
  SmallVector<const Value *, 8> Worklist{VL};
  SmallPtrSet<const Value *, 16> Visited;
  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      Worklist.append(Phi->incoming_values().begin(),
                      Phi->incoming_values().end());
      continue;
    }
    if (auto *CI = dyn_cast<ConstantInt>(V); CI && CI->isZero()) {
      Result.Minimum = 0;
      continue;
    }
    auto *II = dyn_cast<IntrinsicInst>(V);
    if (II && II->getIntrinsicID() == Intrinsic::riscv_vsetvlimax) {
      Result.MayBePositive = true;
      continue;
    }
    if (II && II->getIntrinsicID() == Intrinsic::riscv_vsetvli)
      if (auto *AVL = dyn_cast<ConstantInt>(II->getArgOperand(0))) {
        unsigned Prefix = AVL->getValue().uge(32) ? 16
                          : AVL->getValue().ule(16)
                              ? AVL->getZExtValue()
                              : (AVL->getZExtValue() + 1) / 2;
        Result.Minimum = std::min(Result.Minimum, Prefix);
        Result.MayBePositive |= !AVL->isZero();
        continue;
      }
    Result.Minimum = 0;
    Result.MayBePositive = true;
  }
  if (VL == PositiveVL) {
    Result.Minimum = std::max(Result.Minimum, 1u);
    Result.MayBePositive = true;
  }
  return Result;
}

// A prefix produced under the same actual VL, or under VLMAX, covers every
// lane read by the reduction. A guaranteed full prefix also suffices when the
// function carries the exact VLEN=512 contract. Unrelated dynamic VL results
// cannot establish coverage, even if they happen to agree on one execution.
bool contentsVLCovers(const Value *ProducerVL, const Value *RequiredVL,
                      bool Exact512) {
  if (ProducerVL == RequiredVL)
    return true;
  if (const auto *II = dyn_cast<IntrinsicInst>(ProducerVL);
      II && II->getIntrinsicID() == Intrinsic::riscv_vsetvlimax)
    return true;
  return Exact512 && contentsVLRange(ProducerVL).Minimum >= 16;
}

// Gather may read a lane outside active VL, but never an undefined tail.
// Reduction lane zero additionally depends on its scalar seed and the entire
// actual active source prefix. A prefix requirement is vacuous when its VL is
// zero; this does not initialize the scalar seed or any result lane.
// Receive both requirements from actual contents producers, preserving CFG
// PHIs and seeded cycles without physical selectors or repair copies.
Expected<bool> isContentsLaneDefined(const Value *Source, unsigned Lane,
                                     const Value *PositiveVL, bool Exact512) {
  enum class Region { Lane, ActivePrefix };
  // Context is the positive gather VL for Lane, or the demanded actual VL for
  // ActivePrefix. Keeping it in the key preserves distinct path assumptions.
  using Node = std::tuple<const Value *, Region, unsigned, const Value *>;
  DenseMap<Node, SmallVector<Node, 3>> Dependencies;
  DenseMap<Node, bool> Seeded;
  SmallVector<Node, 16> Worklist{{Source, Region::Lane, Lane, PositiveVL}};
  while (!Worklist.empty()) {
    Node Key = Worklist.pop_back_val();
    auto [It, Inserted] = Dependencies.try_emplace(Key);
    if (!Inserted)
      continue;
    auto &[V, Requirement, Index, Context] = Key;
    auto Add = [&](const Value *Input, Region InputRegion, unsigned InputLane,
                   const Value *InputContext) {
      Node Dependency{Input, InputRegion, InputLane, InputContext};
      It->second.push_back(Dependency);
      Worklist.push_back(Dependency);
    };
    auto AddSame = [&](const Value *Input) {
      Add(Input, Requirement, Index, Context);
    };
    if (Requirement == Region::ActivePrefix &&
        !contentsVLRange(Context).MayBePositive) {
      Seeded[Key] = true;
      continue;
    }
    if (isa<UndefValue>(V))
      return false;
    if (auto *C = dyn_cast<Constant>(V)) {
      const Constant *Element = C->getSplatValue();
      if (!Element || isa<UndefValue>(Element))
        return false;
      Seeded[Key] = true;
      continue;
    }
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      // A PHI may carry the previous iteration's contents. A positive current
      // VL cannot initialize that old value, and equal SSA VL operands do not
      // identify the same dynamic occurrence across a backedge. Only transport
      // a prefix demand when its VL cannot be reevaluated after this PHI.
      if (Requirement == Region::ActivePrefix)
        if (const auto *VLInst = dyn_cast<Instruction>(Context);
            VLInst &&
            isPotentiallyReachable(Phi->getParent(), VLInst->getParent()))
          return false;
      for (const Value *Incoming : Phi->incoming_values())
        Add(Incoming, Requirement, Index,
            Requirement == Region::Lane ? nullptr : Context);
      continue;
    }
    if (auto *Cast = dyn_cast<BitCastInst>(V)) {
      auto *SrcTy = dyn_cast<VectorType>(Cast->getSrcTy());
      auto *DstTy = dyn_cast<VectorType>(Cast->getDestTy());
      if (!SrcTy || !DstTy ||
          SrcTy->getElementCount() != DstTy->getElementCount())
        return false;
      AddSame(Cast->getOperand(0));
      continue;
    }
    auto *II = dyn_cast<IntrinsicInst>(V);
    if (!II)
      return false;
    auto ID = II->getIntrinsicID();
    bool Merge = ID == Intrinsic::riscv_vmerge;
    if (!Merge && !isBOSCOrderedRVVContentsIntrinsic(ID) &&
        !isBOSCContentsSFUIntrinsic(ID))
      return false;
    const Value *VL = II->getArgOperand(II->arg_size() - 1);
    if (Merge) {
      auto *VT = dyn_cast<ScalableVectorType>(II->getType());
      if (!VT || VT->getMinNumElements() != 2 ||
          !VT->getElementType()->isIntegerTy(32) ||
          II->getArgOperand(1)->getType() != VT ||
          II->getArgOperand(2)->getType() != VT || !isActualE32M1VL(VL))
        return false;
    }
    auto Range =
        contentsVLRange(VL, Requirement == Region::Lane ? Context : nullptr);
    if (!Range.MayBePositive) {
      AddSame(II->getArgOperand(0));
      continue;
    }
    if (Merge) {
      // Official vmerge receives a real passthru: unlike the ordered
      // tail-agnostic operations below, its inactive tails can retain defined
      // old contents. If this demand extends beyond the known active region,
      // receive both that passthru and the actually-read active prefix.
      bool Active = Requirement == Region::Lane
                        ? Range.Minimum > Index
                        : contentsVLCovers(VL, Context, Exact512);
      if (!Active)
        AddSame(II->getArgOperand(0));
      auto AddActive = [&](const Value *Input) {
        if (Active)
          AddSame(Input);
        else
          Add(Input, Region::ActivePrefix, 0, VL);
      };
      const auto *Mask = dyn_cast<Constant>(II->getArgOperand(3));
      const auto *MaskBit =
          Mask ? dyn_cast_or_null<ConstantInt>(Mask->getSplatValue()) : nullptr;
      if (MaskBit)
        AddActive(II->getArgOperand(MaskBit->isZero() ? 1 : 2));
      else {
        AddActive(II->getArgOperand(1));
        AddActive(II->getArgOperand(2));
        AddActive(II->getArgOperand(3));
      }
      continue;
    }
    if (ID == Intrinsic::riscv_bosc_rvv_contents_freduce_sum ||
        ID == Intrinsic::riscv_bosc_rvv_contents_freduce_max) {
      // Positive-VL reductions define only lane zero; the remaining lanes are
      // tail-agnostic even if the old destination was fully initialized.
      if (Requirement != Region::Lane || Index != 0)
        return false;
      Add(II->getArgOperand(0), Region::Lane, 0, Context);
      Add(II->getArgOperand(1), Region::ActivePrefix, 0, VL);
      continue;
    }
    if (Requirement == Region::Lane ? Range.Minimum <= Index
                                    : !contentsVLCovers(VL, Context, Exact512))
      return false;
    switch (ID) {
    case Intrinsic::riscv_bosc_rvv_contents_fcmp_lt:
    case Intrinsic::riscv_bosc_rvv_contents_fcmp_eq:
      // Only the active mask prefix is initialized. Positive-VL mask tails
      // remain agnostic even when the compare received a defined old mask.
      AddSame(II->getArgOperand(1));
      AddSame(II->getArgOperand(2));
      break;
    case Intrinsic::riscv_bosc_rvv_contents_load:
      Seeded[Key] = true;
      break;
    case Intrinsic::riscv_bosc_rvv_contents_isplat:
    case Intrinsic::riscv_bosc_rvv_contents_fsplat:
      if (isa<UndefValue>(II->getArgOperand(1)))
        return false;
      Seeded[Key] = true;
      break;
    case Intrinsic::riscv_bosc_rvv_contents_gather_scalar: {
      auto InputLane = gatherIndex(*II, Exact512);
      if (!InputLane)
        return InputLane.takeError();
      if (*InputLane == 16)
        Seeded[Key] = true;
      else
        Add(II->getArgOperand(1), Region::Lane, *InputLane, VL);
      break;
    }
    case Intrinsic::riscv_bosc_rvv_contents_fmacc:
      AddSame(II->getArgOperand(0));
      [[fallthrough]];
    case Intrinsic::riscv_bosc_rvv_contents_add:
    case Intrinsic::riscv_bosc_rvv_contents_maximum:
    case Intrinsic::riscv_bosc_rvv_contents_fadd:
    case Intrinsic::riscv_bosc_rvv_contents_fsub:
    case Intrinsic::riscv_bosc_rvv_contents_fmul:
    case Intrinsic::riscv_bosc_rvv_contents_fdiv:
    case Intrinsic::riscv_bosc_rvv_contents_fmax:
      AddSame(II->getArgOperand(2));
      [[fallthrough]];
    case Intrinsic::riscv_bosc_rvv_contents_iadd_scalar:
    case Intrinsic::riscv_bosc_rvv_contents_shl:
      if (isa<UndefValue>(II->getArgOperand(2)))
        return false;
      [[fallthrough]];
    case Intrinsic::riscv_bosc_rvv_contents_fcvt_to_i32:
    case Intrinsic::riscv_bosc_rvv_contents_fcvt_from_i32:
    case Intrinsic::riscv_bosc_sfu_exp2_contents:
    case Intrinsic::riscv_bosc_sfu_rcp_contents:
      AddSame(II->getArgOperand(1));
      break;
    default:
      return false;
    }
  }
  bool Changed;
  do {
    Changed = false;
    for (const auto &[Key, Inputs] : Dependencies)
      if (!Seeded.lookup(Key) && llvm::any_of(Inputs, [&](Node Input) {
            return Seeded.lookup(Input);
          })) {
        Seeded[Key] = true;
        Changed = true;
      }
  } while (Changed);
  return llvm::all_of(Dependencies, [&](const auto &Entry) {
    return Seeded.lookup(Entry.first);
  });
}

Error verifyContentsRVVIR(Function &F, bool Exact512) {
  if (F.getFnAttribute("trex-bosc-contents-rvv").getValueAsString() != "1")
    return invalid(F, "unsupported RVV contents ABI version");
  if (F.hasFnAttribute("trex-bosc-bound-rvv"))
    return invalid(F, "bound and contents RVV ABIs cannot share a function");
  if (containsVector(F.getReturnType()) ||
      llvm::any_of(F.args(), [](const Argument &A) {
        return containsVector(A.getType());
      }))
    return invalid(F, "RVV contents cannot cross the scalar function ABI");
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (II && (isBOSCOrderedRVVContentsIntrinsic(II->getIntrinsicID()) ||
               isBOSCContentsSFUIntrinsic(II->getIntrinsicID()))) {
      unsigned VLIndex =
          II->getIntrinsicID() == Intrinsic::riscv_bosc_read_bits_contents
              ? 1
              : II->arg_size() - 1;
      if (!isActualE32M1VL(II->getArgOperand(VLIndex)))
        return invalid(
            F, "RVV contents require an actual e32,m1 VL result or literal "
               "zero");
      if (II->getIntrinsicID() == Intrinsic::riscv_bosc_read_bits_contents) {
        const auto *Width = dyn_cast<ConstantInt>(II->getArgOperand(2));
        if (!Width ||
            (Width->getZExtValue() != 8 && Width->getZExtValue() != 32 &&
             Width->getZExtValue() != 64))
          return invalid(F, "bit observation width must be 8, 32, or 64");
      }
      if (II->getIntrinsicID() == Intrinsic::riscv_bosc_rvv_contents_shl) {
        const auto *Amount = dyn_cast<ConstantInt>(II->getArgOperand(2));
        if (!Amount || Amount->getValue().ugt(31))
          return invalid(F, "e32 shift amount must be in 0..31");
      }
      if (II->getIntrinsicID() ==
          Intrinsic::riscv_bosc_rvv_contents_gather_scalar) {
        auto Lane = gatherIndex(*II, Exact512);
        if (!Lane)
          return Lane.takeError();
        const Value *VL = II->getArgOperand(VLIndex);
        if (*Lane != 16 && contentsVLRange(VL).MayBePositive) {
          auto Defined =
              isContentsLaneDefined(II->getArgOperand(1), *Lane, VL, Exact512);
          if (!Defined)
            return Defined.takeError();
          if (!*Defined)
            return invalid(
                F, "gather source lane is not defined on every incoming path");
        }
      }
    }
    bool Vector = containsVector(I.getType()) ||
                  llvm::any_of(I.operands(), [](const Value *V) {
                    return containsVector(V->getType());
                  });
    if (!Vector)
      continue;
    if (isa<PHINode, BitCastInst>(I))
      continue;
    if (!II || !isContentsRVVIntrinsic(II->getIntrinsicID()))
      return invalid(
          F, "RVV contents require formal RVV/SFU intrinsics or PHI/bitcast");
    if (RISCVVIntrinsicsTable::getRISCVVIntrinsicInfo(II->getIntrinsicID()) &&
        (II->mayReadOrWriteMemory() || containsFloatingVector(II->getType()) ||
         llvm::any_of(II->args(), [](const Use &U) {
           return containsFloatingVector(U->getType());
         })))
      return invalid(
          F,
          "RVV contents memory and FP operations require ordered intrinsics");
  }
  return Error::success();
}

Expected<LaneState> transfer(const BasicBlock &BB, LaneState State, bool Verify,
                             bool Exact512) {
  for (const Instruction &I : BB) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !isBoundIntrinsic(II->getIntrinsicID()))
      continue;
    auto ID = II->getIntrinsicID();
    // Counter observations have no vector-register operands and do not
    // participate in the RVV lane/configuration dataflow below.
    if (isScalarObservationIntrinsic(ID))
      continue;
    SmallVector<unsigned, 4> Registers;
    if (ID != Intrinsic::riscv_bosc_setvl)
      for (unsigned Index = 0; Index != boundPhysicalOperands(ID); ++Index) {
        auto Register = physicalArgument(*II, Index);
        if (!Register)
          return Register.takeError();
        Registers.push_back(*Register);
      }
    if (ID == Intrinsic::riscv_bosc_read_bits) {
      if (Verify) {
        auto *Width = dyn_cast<ConstantInt>(II->getArgOperand(3));
        if (!Width ||
            (Width->getZExtValue() != 8 && Width->getZExtValue() != 32 &&
             Width->getZExtValue() != 64))
          return invalid(*BB.getParent(),
                         "bit observation width must be 8, 32, or 64");
        if (State.Configurations.size() != 1)
          return invalid(
              *BB.getParent(),
              "bit observation requires one reaching vector configuration");
        const void *Config = State.Configurations.begin()->first;
        Value *Restore = II->getArgOperand(2);
        if (Config && Config != Restore)
          return invalid(
              *BB.getParent(),
              "bit observation must restore the active setvl result");
        if (!Config) {
          auto *Zero = dyn_cast<ConstantInt>(Restore);
          if (!Zero || !Zero->isZero())
            return invalid(*BB.getParent(),
                           "bit observation without a source "
                           "configuration must restore literal zero");
        }
      }
      // The temporary configuration is private to this operation. Do not
      // expose the scratch as an initialized business e32 vector: only one
      // element was read, under a potentially different element width.
      State.Initialized[Registers[0]] = std::nullopt;
      continue;
    }
    if (ID == Intrinsic::riscv_bosc_setvl) {
      auto *AVL = dyn_cast<ConstantInt>(II->getArgOperand(0));
      State.configure(II, AVL ? exactLanes(AVL->getZExtValue(), Exact512)
                              : std::nullopt);
      continue;
    }
    if (Verify && !State.configured())
      return invalid(*BB.getParent(),
                     "vector operation requires an explicit setvl on "
                     "every incoming path");
    auto Read = [&](unsigned Index) -> Error {
      unsigned Reg = Registers[Index];
      if (Verify && !State.readable(Reg))
        return invalid(*BB.getParent(),
                       "read of undefined physical vector register v" +
                           Twine(Reg) + " covering the active setvl lanes");
      return Error::success();
    };
    if (isBoundBinaryIntrinsic(ID) || ID == Intrinsic::riscv_bosc_merge) {
      if (Error E = Read(1))
        return std::move(E);
      if (Error E = Read(2))
        return std::move(E);
    }
    if (ID == Intrinsic::riscv_bosc_fmacc)
      if (Error E = Read(0))
        return std::move(E);
    if (ID == Intrinsic::riscv_bosc_merge)
      if (Error E = Read(3))
        return std::move(E);
    if (isBoundUnaryIntrinsic(ID))
      if (Error E = Read(1))
        return std::move(E);
    if (ID == Intrinsic::riscv_bosc_gather_scalar) {
      auto Index = gatherIndex(*II, Exact512);
      if (!Index)
        return Index.takeError();
      if (Verify && *Index < 16 && !State.noActiveLanes() &&
          !State.readableLane(Registers[1], *Index))
        return invalid(*BB.getParent(),
                       "gather index reads an uninitialized source lane");
    }
    if (isBoundReduceIntrinsic(ID))
      if (Error E = Read(1))
        return std::move(E);
    if (isBoundReduceIntrinsic(ID) || ID == Intrinsic::riscv_bosc_fextract) {
      unsigned Reg = Registers[0];
      if (Verify && !State.readableZero(Reg))
        return invalid(*BB.getParent(),
                       "scalar lane read requires initialized lane zero");
      if (isBoundReduceIntrinsic(ID))
        State.reduce(Reg);
      continue;
    }
    if (ID == Intrinsic::riscv_bosc_store) {
      if (Error E = Read(0))
        return std::move(E);
    } else
      State.define(Registers[0]);
  }
  return State;
}

Error verifyIR(Function &F, bool Supported, bool Exact512 = false) {
  bool Bound = isBOSCBoundFunction(F);
  bool RVV = F.hasFnAttribute("trex-bosc-bound-rvv");
  bool DMA = hasDMAIntrinsic(F);
  bool Observation = hasBOSCObservationIntrinsic(F);
  bool ContentsRVV = isBOSCContentsRVVFunction(F);
  if (ContentsRVV)
    if (Error E = verifyContentsRVVIR(F, Exact512))
      return E;
  for (Instruction &I : instructions(F)) {
    auto *Call = dyn_cast<CallBase>(&I);
    if (!Call)
      continue;
    auto *Callee = Call->getCalledFunction();
    auto *II = dyn_cast<IntrinsicInst>(Call);
    if (Callee && Callee->getName().starts_with("llvm.riscv.bosc.") &&
        (!II || (!isBoundIntrinsic(II->getIntrinsicID()) &&
                 !isBOSCAMEIntrinsic(II->getIntrinsicID()) &&
                 !isBOSCOrderedRVVContentsIntrinsic(II->getIntrinsicID()) &&
                 !isBOSCContentsSFUIntrinsic(II->getIntrinsicID()) &&
                 !isDMAIntrinsic(II->getIntrinsicID()))))
      return invalid(F, "unknown BOSC intrinsic");
    if (II && II->getIntrinsicID() == Intrinsic::riscv_bosc_read_bits && !RVV)
      return invalid(F, "bit observation requires trex-bosc-bound-rvv=1");
    if (II && isBOSCContentsSFUIntrinsic(II->getIntrinsicID()) && !ContentsRVV)
      return invalid(F, "SFU contents require RVV contents scope");
    if (II && isBOSCOrderedRVVContentsIntrinsic(II->getIntrinsicID()) &&
        !ContentsRVV)
      return invalid(F, "ordered RVV contents require RVV contents scope");
    if (II && isBoundIntrinsic(II->getIntrinsicID()) &&
        !isScalarObservationIntrinsic(II->getIntrinsicID()) && !RVV &&
        !F.hasFnAttribute("trex-bosc-bound-ame"))
      return invalid(F, "intrinsic requires trex-bosc-bound-rvv=1");
  }
  if (Error E = verifyBOSCAMEIR(F))
    return E;
  if (!Bound && !DMA && !Observation && !isBOSCRegisterContentsFunction(F))
    return Error::success();
  if (!Supported)
    return invalid(F, "requires the SelectionDAG legacy codegen pipeline");
  if (RVV && F.getFnAttribute("trex-bosc-bound-rvv").getValueAsString() != "1")
    return invalid(F, "unsupported bound ABI version");
  if (!ContentsRVV && containsVector(F.getReturnType()))
    return invalid(F, "LLVM vector values are forbidden in bound scope");
  for (const Argument &A : F.args())
    if (!ContentsRVV && containsVector(A.getType()))
      return invalid(F, "LLVM vector values are forbidden in bound scope");
  for (Instruction &I : instructions(F)) {
    if (!ContentsRVV && (containsVector(I.getType()) ||
                         llvm::any_of(I.operands(), [](const Value *V) {
                           return containsVector(V->getType());
                         })))
      return invalid(F, "LLVM vector values are forbidden in bound scope");
    auto *Call = dyn_cast<CallBase>(&I);
    if (!Call)
      continue;
    auto *II = dyn_cast<IntrinsicInst>(Call);
    if (II && isa<DbgInfoIntrinsic>(II))
      continue;
    if (II && ContentsRVV && isContentsRVVIntrinsic(II->getIntrinsicID()))
      continue;
    // Compiler-authored math and address guards terminate bound execution.
    // Their validity does not depend on whether the function also issues DMA.
    if (II && II->getIntrinsicID() == Intrinsic::trap) {
      if (!isa_and_nonnull<UnreachableInst>(I.getNextNode()))
        return invalid(F, "trap must be immediately followed by unreachable");
      continue;
    }
    // Explicit scalar memory max-number uses the official f32 intrinsic;
    // RISCV selects fmax.s directly. No other unbound math or vector form is
    // admitted, and this cannot create VRF ownership or a host/libm call.
    if (II && II->getIntrinsicID() == Intrinsic::maximumnum &&
        II->getType()->isFloatTy() && II->arg_size() == 2 &&
        II->getArgOperand(0)->getType()->isFloatTy() &&
        II->getArgOperand(1)->getType()->isFloatTy())
      continue;
    // Codegen canonicalizes scalar address-guard comparisons/selects to
    // integer min/max, including loop strength reduction's remaining extent.
    // These native scalar widths lower to comparisons/branches or min/max
    // instructions; they require neither a helper call nor VRF ownership.
    if (II &&
        (II->getIntrinsicID() == Intrinsic::umin ||
         II->getIntrinsicID() == Intrinsic::umax ||
         II->getIntrinsicID() == Intrinsic::smin ||
         II->getIntrinsicID() == Intrinsic::smax) &&
        (II->getType()->isIntegerTy(32) || II->getType()->isIntegerTy(64)) &&
        II->arg_size() == 2 &&
        II->getArgOperand(0)->getType() == II->getType() &&
        II->getArgOperand(1)->getType() == II->getType())
      continue;
    if (!II || (!isBoundIntrinsic(II->getIntrinsicID()) &&
                !isBOSCAMEIntrinsic(II->getIntrinsicID()) &&
                !isDMAIntrinsic(II->getIntrinsicID())))
      return invalid(
          F, "calls, inline assembly and unbound intrinsics are forbidden "
             "in bound scope");
    if (isBOSCAMEIntrinsic(II->getIntrinsicID()) ||
        isDMAIntrinsic(II->getIntrinsicID()))
      continue;
    // The cycle counter is a scalar observation boundary.  It is admitted in
    // bound scope, but has no physical vector operands or lane effects.
    if (isScalarObservationIntrinsic(II->getIntrinsicID()))
      continue;
    if (II->getIntrinsicID() != Intrinsic::riscv_bosc_setvl)
      for (unsigned I = 0; I != boundPhysicalOperands(II->getIntrinsicID());
           ++I) {
        auto Register = physicalArgument(*II, I);
        if (!Register)
          return Register.takeError();
      }
    if (II->getIntrinsicID() == Intrinsic::riscv_bosc_shl) {
      const auto *Amount = dyn_cast<ConstantInt>(II->getArgOperand(2));
      if (!Amount || Amount->getValue().getActiveBits() > 5)
        return invalid(F, "e32 shift amount must be in 0..31");
    }
    if (II->getIntrinsicID() == Intrinsic::riscv_bosc_gather_scalar) {
      auto Index = gatherIndex(*II, Exact512);
      if (!Index)
        return Index.takeError();
      auto Destination = physicalArgument(*II, 0);
      if (!Destination)
        return Destination.takeError();
      auto Source = physicalArgument(*II, 1);
      if (!Source)
        return Source.takeError();
      if (*Destination == *Source)
        return invalid(F, "gather destination must not overlap its source");
    }
  }
  // Must-definition analysis, including CFG joins and loop backedges. Top for
  // non-entry blocks gives the greatest fixed point for intersecting facts.
  DenseMap<const BasicBlock *, LaneState> Out;
  for (const BasicBlock &BB : F)
    Out[&BB] = LaneState::top();
  bool Changed;
  do {
    Changed = false;
    for (const BasicBlock &BB : F) {
      LaneState In = BB.isEntryBlock() || pred_empty(&BB) ? LaneState::entry()
                                                          : LaneState::top();
      for (const BasicBlock *Pred : predecessors(&BB))
        In.meet(Out[Pred]);
      auto Next = transfer(BB, In, false, Exact512);
      if (!Next)
        return Next.takeError();
      if (Out[&BB] != *Next) {
        Out[&BB] = *Next;
        Changed = true;
      }
    }
  } while (Changed);
  for (const BasicBlock &BB : F) {
    LaneState In = BB.isEntryBlock() || pred_empty(&BB) ? LaneState::entry()
                                                        : LaneState::top();
    for (const BasicBlock *Pred : predecessors(&BB))
      In.meet(Out[Pred]);
    auto Next = transfer(BB, In, true, Exact512);
    if (!Next)
      return Next.takeError();
  }
  return Error::success();
}

Error verifyBOSCFunction(Function &F, const RISCVSubtarget &ST,
                         bool Supported) {
  if (Error E = verifyIR(F, Supported,
                         ST.hasVInstructions() && ST.getRealVLen() == 512))
    return E;
  if (isBOSCBoundFunction(F) || isBOSCRegisterContentsFunction(F) ||
      hasDMAIntrinsic(F) || hasBOSCObservationIntrinsic(F)) {
    if (hasBOSCObservationIntrinsic(F) && !ST.is64Bit())
      return invalid(F, "BOSC observation counter requires RV64");
    if (hasDMAIntrinsic(F) && !ST.is64Bit())
      return invalid(F, "patch7 DMA requires RV64");
    if (isBOSCContentsRVVFunction(F) &&
        (!ST.is64Bit() || !ST.hasVInstructions() || ST.getRealMinVLen() < 512))
      return invalid(F, "RVV contents require RV64 +v +zvl512b");
    if (F.hasFnAttribute("trex-bosc-bound-rvv") &&
        (!ST.is64Bit() || !ST.hasStdExtV() || ST.getRealMinVLen() < 512))
      return invalid(F, "requires bound scope and RV64 +v +zvl512b");
    if (llvm::any_of(instructions(F),
                     [](const Instruction &I) {
                       const auto *II = dyn_cast<IntrinsicInst>(&I);
                       return II && requiresRVVF32(II->getIntrinsicID());
                     }) &&
        !ST.hasVInstructionsF32())
      return invalid(
          F, "FP32 vector arithmetic requires vector floating-point F32");
    if ((F.hasFnAttribute("trex-bosc-bound-ame") ||
         isBOSCContentsAMEFunction(F)) &&
        (!ST.is64Bit() || !ST.hasVendorXBOSCAME()))
      return invalid(F, "requires bound AME scope and RV64 +xboscame");
    if (!ST.hasVendorXBOSCSFU() &&
        llvm::any_of(instructions(F), [](const Instruction &I) {
          const auto *II = dyn_cast<IntrinsicInst>(&I);
          return II && (isBOSCSFUIntrinsic(II->getIntrinsicID()) ||
                        isBOSCContentsSFUIntrinsic(II->getIntrinsicID()));
        }))
      return invalid(F, "SFU offline code generation requires +xboscsfu");
  }
  return Error::success();
}

class RISCVBOSCIRVerification final : public FunctionPass {
  bool Supported;

public:
  static char ID;
  explicit RISCVBOSCIRVerification(bool Supported = true)
      : FunctionPass(ID), Supported(Supported) {
    initializeRISCVBOSCIRVerificationPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override {
    auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
    const auto &ST = TM.getSubtarget<RISCVSubtarget>(F);
    if (Error E = verifyBOSCFunction(F, ST, Supported))
      reportFatalUsageError(Twine(toString(std::move(E))));
    // Establish the standard LLVM codegen policy at machine ingress, after the
    // actual function has passed the BOSC ABI verifier. This covers SelectionDAG
    // store merging, memory-intrinsic expansion and other implicit vector uses;
    // it preserves explicit scalar FP and bound vector operations. Library IR
    // verification remains read-only, and standalone llc receives the same
    // policy without relying on a TReX producer or a function-name convention.
    if (requiresBOSCExplicitVectorState(F) &&
        !F.hasFnAttribute(Attribute::NoImplicitFloat)) {
      F.addFnAttr(Attribute::NoImplicitFloat);
      return true;
    }
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    // Only a codegen policy attribute changes. Keep the already-run stack
    // protector: rebuilding it here invalidates AA required by SelectionDAG
    // and creates an unschedulable legacy analysis dependency.
    AU.addPreserved<StackProtector>();
    AU.addRequired<TargetPassConfig>();
  }
};

bool isVRF(Register Reg, const MachineRegisterInfo &MRI,
           const TargetRegisterInfo &TRI) {
  if (!Reg)
    return false;
  if (Reg.isVirtual()) {
    const auto *RC = MRI.getRegClassOrNull(Reg);
    return RC && RISCVRegisterInfo::isRVVRegClass(RC);
  }
  for (MCRegAliasIterator Alias(Reg, &TRI, true); Alias.isValid(); ++Alias)
    if (RISCV::VRRegClass.contains(*Alias))
      return true;
  return false;
}

bool isCompareOpcode(unsigned Opcode) {
  return Opcode == RISCV::VMFLT_VV || Opcode == RISCV::VMFEQ_VV;
}
bool isRoundedBinaryOpcode(unsigned Opcode) {
  return Opcode == RISCV::VFADD_VV || Opcode == RISCV::VFSUB_VV ||
         Opcode == RISCV::VFMUL_VV || Opcode == RISCV::VFDIV_VV;
}
bool isConversionOpcode(unsigned Opcode) {
  return Opcode == RISCV::VFCVT_X_F_V || Opcode == RISCV::VFCVT_F_X_V;
}
bool isSFUOpcode(unsigned Opcode) {
  return Opcode == RISCV::BOSC_SFU_EXP2 || Opcode == RISCV::BOSC_SFU_RCP;
}
bool isUnaryOpcode(unsigned Opcode) {
  return isConversionOpcode(Opcode) || Opcode == RISCV::VADD_VX ||
         Opcode == RISCV::VSLL_VI || isSFUOpcode(Opcode);
}
bool isBinaryOpcode(unsigned Opcode) {
  return Opcode == RISCV::VADD_VV || Opcode == RISCV::VMAX_VV ||
         isRoundedBinaryOpcode(Opcode) || Opcode == RISCV::VFMAX_VV ||
         isCompareOpcode(Opcode) || Opcode == RISCV::VMERGE_VVM;
}

// Keep the scalar index independently checkable after ordinary GPR allocation.
// Only the bounded gather constants are facts: -2 is the initial lattice top,
// -1 is unknown, and 0..16 are known values. Physical aliases and register
// masks kill facts, so no LLVM intrinsic annotation is trusted by the MIR
// verifier.
struct MachineState : LaneState {
  SmallVector<int8_t, 64> Constants;
  unsigned PhysicalCount;

  MachineState(const MachineFunction &MF, bool Entry)
      : LaneState(Entry ? LaneState::entry() : LaneState::top()),
        PhysicalCount(MF.getSubtarget().getRegisterInfo()->getNumRegs()) {
    Constants.assign(PhysicalCount + MF.getRegInfo().getNumVirtRegs(),
                     Entry ? -1 : -2);
  }
  unsigned index(Register Reg) const {
    return Reg.isVirtual() ? PhysicalCount + Reg.virtRegIndex() : Reg.id();
  }
  std::optional<unsigned> constant(Register Reg) const {
    if (Reg == RISCV::X0)
      return 0;
    int8_t Value = Constants[index(Reg)];
    return Value >= 0 ? std::optional<unsigned>(Value) : std::nullopt;
  }
  void meet(const MachineState &Other) {
    LaneState::meet(Other);
    for (unsigned I = 0; I != Constants.size(); ++I) {
      if (Constants[I] == -2)
        Constants[I] = Other.Constants[I];
      else if (Other.Constants[I] != -2 && Constants[I] != Other.Constants[I])
        Constants[I] = -1;
    }
  }
  bool operator==(const MachineState &Other) const {
    return LaneState::operator==(Other) && Constants == Other.Constants;
  }
  bool operator!=(const MachineState &Other) const { return !(*this == Other); }

  void updateConstants(const MachineInstr &MI) {
    std::optional<unsigned> Value;
    Register Destination;
    if ((MI.getOpcode() == RISCV::ADDI || MI.getOpcode() == RISCV::ADDIW) &&
        MI.getOperand(0).isReg() && MI.getOperand(1).isReg() &&
        MI.getOperand(2).isImm()) {
      Destination = MI.getOperand(0).getReg();
      if (auto Source = constant(MI.getOperand(1).getReg())) {
        int64_t Sum = *Source + MI.getOperand(2).getImm();
        if (Sum >= 0 && Sum <= 16)
          Value = Sum;
      }
    } else if (MI.isCopy() && MI.getOperand(0).isReg() &&
               MI.getOperand(1).isReg() && !MI.getOperand(1).getSubReg()) {
      Destination = MI.getOperand(0).getReg();
      Value = constant(MI.getOperand(1).getReg());
    }
    const auto &TRI = *MI.getMF()->getSubtarget().getRegisterInfo();
    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask()) {
        for (unsigned Reg = 1; Reg != PhysicalCount; ++Reg)
          if (MO.clobbersPhysReg(Reg))
            Constants[Reg] = -1;
      } else if (MO.isReg() && MO.isDef() && MO.getReg()) {
        Register Reg = MO.getReg();
        if (Reg.isVirtual())
          Constants[index(Reg)] = -1;
        else
          for (MCRegAliasIterator Alias(Reg, &TRI, true); Alias.isValid();
               ++Alias)
            Constants[*Alias] = -1;
      }
    }
    if (Destination && Value)
      Constants[index(Destination)] = *Value;
  }
};

MachineState transferMachine(const MachineBasicBlock &MBB, MachineState State,
                             bool Verify, bool Exact512) {
  for (const MachineInstr &MI : MBB) {
    unsigned Opcode = MI.getOpcode();
    auto GatherIndex = Opcode == RISCV::VRGATHER_VX
                           ? State.constant(MI.getOperand(2).getReg())
                           : std::nullopt;
    State.updateConstants(MI);
    if (Opcode == RISCV::PseudoBOSCReadBitsBound) {
      State.Initialized[MI.getOperand(1).getReg() - RISCV::V0] = std::nullopt;
      continue;
    }
    if (Opcode == RISCV::VSETVLI || Opcode == RISCV::VSETIVLI) {
      State.configure(&MI, Opcode == RISCV::VSETIVLI
                               ? exactLanes(MI.getOperand(1).getImm(), Exact512)
                               : std::nullopt);
      continue;
    }
    bool Load = Opcode == RISCV::VLE32_V;
    bool Binary = isBinaryOpcode(Opcode);
    bool Unary = isUnaryOpcode(Opcode);
    bool Gather = Opcode == RISCV::VRGATHER_VX;
    bool Accumulate = Opcode == RISCV::VFMACC_VV;
    bool Store = Opcode == RISCV::VSE32_V;
    bool Splat = Opcode == RISCV::VFMV_V_F || Opcode == RISCV::VMV_V_X;
    bool Extract = Opcode == RISCV::VFMV_F_S;
    bool Reduce = Opcode == RISCV::VFREDOSUM_VS || Opcode == RISCV::VFREDMAX_VS;
    if (!Load && !Binary && !Unary && !Gather && !Accumulate && !Store &&
        !Splat && !Extract && !Reduce)
      continue;
    if (Verify && !State.configured())
      fail(MBB.getParent()->getFunction(),
           "machine vector operation requires an explicit setvl on every "
           "incoming path");
    auto Read = [&](unsigned Index) {
      unsigned Reg = MI.getOperand(Index).getReg() - RISCV::V0;
      if (Verify && !State.readable(Reg))
        fail(MBB.getParent()->getFunction(),
             "machine read of undefined physical vector register v" +
                 Twine(Reg) + " covering the active setvl lanes");
    };
    if (Binary) {
      Read(1);
      Read(2);
    }
    if (Opcode == RISCV::VMERGE_VVM)
      Read(3);
    if (Unary)
      Read(1);
    if (Gather && Verify) {
      if (!GatherIndex)
        fail(MBB.getParent()->getFunction(),
             "machine gather index must be a constant in 0..16");
      if (*GatherIndex == 16 && !Exact512)
        fail(MBB.getParent()->getFunction(),
             "gather index 16 requires exact VLEN=512");
      if (*GatherIndex < 16 && !State.noActiveLanes() &&
          !State.readableLane(MI.getOperand(1).getReg() - RISCV::V0,
                              *GatherIndex))
        fail(MBB.getParent()->getFunction(),
             "machine gather index reads an uninitialized source lane");
    }
    if (Accumulate) {
      Read(1);
      Read(2);
      Read(3);
    }
    if (Reduce)
      Read(1);
    if (Reduce || Extract) {
      unsigned Index = Extract ? 1 : 2;
      unsigned Reg = MI.getOperand(Index).getReg() - RISCV::V0;
      if (Verify && !State.readableZero(Reg))
        fail(MBB.getParent()->getFunction(),
             "machine scalar lane read requires initialized lane zero");
      if (Reduce)
        State.reduce(MI.getOperand(0).getReg() - RISCV::V0);
      continue;
    }
    if (Store)
      Read(0);
    else
      State.define(MI.getOperand(0).getReg() - RISCV::V0);
  }
  return State;
}

void verifyMachineFlow(const MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  bool Exact512 = ST.hasVInstructions() && ST.getRealVLen() == 512;
  DenseMap<const MachineBasicBlock *, MachineState> Out;
  for (const MachineBasicBlock &MBB : MF)
    Out.try_emplace(&MBB, MF, false);
  bool Changed;
  do {
    Changed = false;
    for (const MachineBasicBlock &MBB : MF) {
      MachineState In(MF, &MBB == &MF.front() || MBB.pred_empty());
      for (const MachineBasicBlock *Pred : MBB.predecessors())
        In.meet(Out.find(Pred)->second);
      MachineState Next = transferMachine(MBB, In, false, Exact512);
      if (Out.find(&MBB)->second != Next) {
        Out.find(&MBB)->second = Next;
        Changed = true;
      }
    }
  } while (Changed);
  for (const MachineBasicBlock &MBB : MF) {
    MachineState In(MF, &MBB == &MF.front() || MBB.pred_empty());
    for (const MachineBasicBlock *Pred : MBB.predecessors())
      In.meet(Out.find(Pred)->second);
    transferMachine(MBB, In, true, Exact512);
  }
}

// The selected accelerator sequence includes configuration, vector/matrix
// actions and exact DMA/volatile boundaries. Ordinary scalar arithmetic and
// nonvolatile scalar memory are deliberately absent: LLVM still owns their
// scheduling and register allocation.
bool requiresIssueOrder(const MachineInstr &MI) {
  unsigned Opcode = RISCV::getRVVMCOpcode(MI.getOpcode());
  if (!Opcode)
    Opcode = MI.getOpcode();
  if (isBOSCAMEOpcode(Opcode) || isBinaryOpcode(Opcode) ||
      isUnaryOpcode(Opcode) || Opcode == RISCV::VSETVLI ||
      Opcode == RISCV::VSETIVLI || Opcode == RISCV::VLE32_V ||
      Opcode == RISCV::VSE32_V || Opcode == RISCV::VFMACC_VV ||
      Opcode == RISCV::VFMV_V_F || Opcode == RISCV::VMV_V_X ||
      Opcode == RISCV::VFMV_F_S || Opcode == RISCV::VFREDOSUM_VS ||
      Opcode == RISCV::VFREDMAX_VS || Opcode == RISCV::VRGATHER_VX ||
      Opcode == RISCV::PseudoBOSCReadBitsBound)
    return true;
  switch (Opcode) {
  case RISCV::FENCE:
  case RISCV::FENCE_I:
  case RISCV::CSRRW:
  case RISCV::CSRRS:
  case RISCV::CSRRC:
  case RISCV::CSRRWI:
  case RISCV::CSRRSI:
  case RISCV::CSRRCI:
    return true;
  default:
    return llvm::any_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
      return MMO->isVolatile();
    });
  }
}

void verifyIssueOrder(const MachineInstr &MI) {
  const auto &MF = *MI.getMF();
  unsigned Uses = 0, Defs = 0;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    Register Reg = MO.getReg();
    if (Reg.isVirtual()) {
      if (MF.getRegInfo().getRegClassOrNull(Reg) ==
          &RISCV::BOSCIssueStateRegClass)
        fail(MF.getFunction(), "virtual BOSC issue state is forbidden");
      continue;
    }
    if (Reg != RISCV::BOSC_ISSUE)
      continue;
    if (!MO.isImplicit() || MO.getSubReg() || MO.isRenamable() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead())
      fail(MF.getFunction(), "invalid BOSC issue-order operand");
    (MO.isDef() ? Defs : Uses)++;
  }
  if (requiresIssueOrder(MI)) {
    if (Uses != 1 || Defs != 1)
      fail(MF.getFunction(), "missing BOSC issue-order use/def");
  } else if (Uses || Defs) {
    fail(MF.getFunction(),
         "BOSC issue state on ordinary scalar instruction is forbidden");
  }
}

void verifyReadBits(const MachineInstr &MI) {
  const auto &MF = *MI.getMF();
  const auto &F = MF.getFunction();
  const auto &MRI = MF.getRegInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  bool Contents = isBOSCContentsRVVFunction(F);
  if (!F.hasFnAttribute("trex-bosc-bound-rvv") && !Contents)
    fail(F, "bit observation requires bound or contents RVV scope");
  if (MI.getNumExplicitOperands() != 5)
    fail(F, "bit observation has invalid operand count");
  const MachineOperand &WidthOperand = MI.getOperand(4);
  if (!WidthOperand.isImm() ||
      (WidthOperand.getImm() != 8 && WidthOperand.getImm() != 32 &&
       WidthOperand.getImm() != 64))
    fail(F, "bit observation width must be 8, 32, or 64");
  unsigned WidthBytes = WidthOperand.getImm() / 8;
  for (unsigned I = 0; I != 4; ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.getReg() || MO.getSubReg() || MO.isImplicit() ||
        MO.isUndef() || MO.isDef() != (I < 2))
      fail(F, "bit observation has invalid register operand");
    Register Reg = MO.getReg();
    if (I == 1) {
      if (Contents) {
        if (Reg.isVirtual()
                ? !RISCV::VRRegClass.hasSubClassEq(MRI.getRegClass(Reg))
                : !RISCV::VRRegClass.contains(Reg))
          fail(F, "bit observation scratch requires an allocator-owned VR");
      } else if (Reg.isVirtual() || !RISCV::VRRegClass.contains(Reg) ||
                 Reg == RISCV::V0 || MO.isRenamable())
        fail(F, "bit observation requires a bound physical scratch v1..v31");
      continue;
    }
    if (Reg.isVirtual()
            ? !RISCV::GPRRegClass.hasSubClassEq(MRI.getRegClass(Reg))
            : !RISCV::GPRRegClass.contains(Reg))
      fail(F, "bit observation scalar operand requires a GPR");
    if (I == 0 && !MO.isEarlyClobber())
      fail(F, "bit observation requires nonzero early-clobber scalar outputs");
    if ((I == 0 || I == 3) && Reg.isPhysical() && Reg == RISCV::X0)
      fail(F, "bit observation result and restore VL require nonzero GPRs");
    if ((I == 0 || I == 3) && Reg.isVirtual() &&
        !RISCV::GPRNoX0RegClass.hasSubClassEq(MRI.getRegClass(Reg)))
      fail(F, "bit observation result and restore VL require nonzero GPRs");
    if (I != 0)
      continue;
    for (unsigned J = 2; J != 4; ++J) {
      if (!MI.getOperand(J).isReg())
        continue;
      Register Other = MI.getOperand(J).getReg();
      if (Reg == Other || (Reg.isPhysical() && Other.isPhysical() &&
                           TRI.regsOverlap(Reg, Other)))
        fail(F, "bit observation scalar temporaries overlap");
    }
  }
  unsigned UsesVL = 0, DefsVL = 0, UsesVType = 0, DefsVType = 0;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg() || MO.getSubReg() || MO.isRenamable() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead())
      fail(F, "bit observation has invalid implicit operand");
    if (MO.getReg() == RISCV::VL)
      (MO.isDef() ? DefsVL : UsesVL)++;
    else if (MO.getReg() == RISCV::VTYPE)
      (MO.isDef() ? DefsVType : UsesVType)++;
    else if (MO.getReg() != RISCV::BOSC_ISSUE)
      fail(F, "bit observation has unexpected implicit register");
  }
  if (UsesVL != 1 || DefsVL != 1 || UsesVType != 1 || DefsVType != 1)
    fail(F, "bit observation requires VL/VTYPE use/def");
  if (MI.memoperands().size() != 1)
    fail(F, "bit observation requires one exact naturally aligned volatile "
            "load matching its width");
  const MachineMemOperand &MMO = **MI.memoperands_begin();
  if (!MMO.isLoad() || MMO.isStore() || !MMO.isVolatile() || MMO.isAtomic() ||
      MMO.getSize() != LocationSize::precise(WidthBytes) ||
      MMO.getAlign() != Align(WidthBytes))
    fail(F, "bit observation requires one exact naturally aligned volatile "
            "load matching its width");
}

class RISCVBOSCIssueIngress final : public MachineFunctionPass {
public:
  static char ID;
  RISCVBOSCIssueIngress() : MachineFunctionPass(ID) {
    initializeRISCVBOSCIssueIngressPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!isBOSCBoundFunction(MF.getFunction()) &&
        !isBOSCRegisterContentsFunction(MF.getFunction()) &&
        !hasDMAIntrinsic(MF.getFunction()))
      return false;
    bool Changed = false;
    for (MachineBasicBlock &MBB : MF)
      for (MachineInstr &MI : MBB) {
        if (!requiresIssueOrder(MI))
          continue;
        if (llvm::any_of(MI.operands(), [](const MachineOperand &MO) {
              return MO.isReg() && MO.getReg() == RISCV::BOSC_ISSUE;
            })) {
          verifyIssueOrder(MI);
          continue;
        }
        MachineInstrBuilder(MF, &MI)
            .addReg(RISCV::BOSC_ISSUE, RegState::Implicit)
            .addReg(RISCV::BOSC_ISSUE, RegState::ImplicitDefine);
        Changed = true;
      }
    return Changed;
  }
};

class RISCVBOSCBoundVerification final : public MachineFunctionPass {
public:
  static char ID;
  RISCVBOSCBoundVerification() : MachineFunctionPass(ID) {
    initializeRISCVBOSCBoundVerificationPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!isBOSCBoundFunction(MF.getFunction()) &&
        !isBOSCRegisterContentsFunction(MF.getFunction()) &&
        !hasDMAIntrinsic(MF.getFunction()))
      return false;
    const auto &MRI = MF.getRegInfo();
    const auto &ST = MF.getSubtarget<RISCVSubtarget>();
    const auto &TRI = *ST.getRegisterInfo();
    if (isBOSCRegisterContentsFunction(MF.getFunction())) {
      auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
      if (Info->hasTensixCodegenFailed() ||
          MF.getProperties().hasFailedRegAlloc())
        return false;
      for (const MachineBasicBlock &MBB : MF)
        for (const MachineInstr &MI : MBB) {
          if (MI.isCall() || MI.isInlineAsm()) {
            Info->setTensixCodegenFailed();
            MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
                MF.getFunction(),
                "machine call or inline assembly escapes contents scope"));
            return false;
          }
        }
    }
    verifyBOSCAMEMachine(MF);
    if (isBOSCRegisterContentsFunction(MF.getFunction()) &&
        !MF.getFunction().hasFnAttribute("trex-bosc-bound-rvv")) {
      for (const MachineBasicBlock &MBB : MF)
        for (const MachineInstr &MI : MBB) {
          if (MI.getOpcode() == RISCV::PseudoBOSCReadBitsBound)
            verifyReadBits(MI);
          verifyIssueOrder(MI);
          unsigned Opcode = RISCV::getRVVMCOpcode(MI.getOpcode());
          if (!Opcode)
            Opcode = MI.getOpcode();
          bool FP = isRoundedBinaryOpcode(Opcode) ||
                    isConversionOpcode(Opcode) || isCompareOpcode(Opcode) ||
                    Opcode == RISCV::VFMAX_VV || Opcode == RISCV::VFMACC_VV ||
                    Opcode == RISCV::VFREDOSUM_VS ||
                    Opcode == RISCV::VFREDMAX_VS;
          if (FP && !MI.mayRaiseFPException())
            fail(MF.getFunction(),
                 "RVV contents FP operation lost accrued exception effects");
        }
      return false;
    }
    if (MF.getFunction().hasFnAttribute("trex-bosc-bound-rvv") &&
        (!ST.is64Bit() || !ST.hasStdExtV() || ST.getRealMinVLen() < 512))
      fail(MF.getFunction(), "requires RV64 +v +zvl512b");
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB) {
        if (isBOSCAMEOpcode(MI.getOpcode()))
          continue;
        if (MI.isCall() || MI.isInlineAsm())
          fail(MF.getFunction(), "machine call or inline assembly escapes "
                                 "bound scope");
        unsigned Opcode = MI.getOpcode();
        if (Opcode == RISCV::PseudoBOSCReadBitsBound) {
          verifyReadBits(MI);
          continue;
        }
        bool Accumulate = Opcode == RISCV::VFMACC_VV;
        bool Compare = isCompareOpcode(Opcode);
        bool Merge = Opcode == RISCV::VMERGE_VVM;
        bool Gather = Opcode == RISCV::VRGATHER_VX;
        bool NativeFloat = isRoundedBinaryOpcode(Opcode) || Accumulate ||
                           isConversionOpcode(Opcode);
        bool Splat = Opcode == RISCV::VFMV_V_F || Opcode == RISCV::VMV_V_X;
        bool SFU = isSFUOpcode(Opcode);
        bool Extract = MI.getOpcode() == RISCV::VFMV_F_S;
        bool Reduce = MI.getOpcode() == RISCV::VFREDOSUM_VS ||
                      MI.getOpcode() == RISCV::VFREDMAX_VS;
        bool FPVector = NativeFloat || Compare || Opcode == RISCV::VFMAX_VV ||
                        Opcode == RISCV::VFMV_V_F || Extract || Reduce;
        bool Native = MI.getOpcode() == RISCV::VLE32_V ||
                      MI.getOpcode() == RISCV::VSE32_V ||
                      isBinaryOpcode(Opcode) || isUnaryOpcode(Opcode) ||
                      Accumulate || Splat || Extract || Reduce || Gather;
        bool Config = MI.getOpcode() == RISCV::VSETVLI ||
                      MI.getOpcode() == RISCV::VSETIVLI;
        if ((Native || Config) &&
            !MF.getFunction().hasFnAttribute("trex-bosc-bound-rvv"))
          fail(MF.getFunction(), "vector operation requires bound RVV scope");
        if (Config && (!MI.getOperand(2).isImm() ||
                       MI.getOperand(2).getImm() != E32M1TaMa))
          fail(MF.getFunction(), "configuration differs from e32,m1,ta,ma");
        unsigned Expected = Accumulate ? 5
                            : isBinaryOpcode(Opcode) || Reduce ||
                                    Opcode == RISCV::VADD_VX ||
                                    Opcode == RISCV::VSLL_VI || Gather
                                ? 4
                            : Splat || Extract || SFU ? 2
                                                      : 3;
        if (Native && MI.getNumExplicitOperands() != Expected)
          fail(MF.getFunction(),
               "bound native operation has invalid operand count");
        if (Native && !Splat && !Extract && !Merge && !SFU &&
            (!MI.getOperand(Expected - 1).isReg() ||
             MI.getOperand(Expected - 1).getReg()))
          fail(MF.getFunction(), "bound native operation must be unmasked");
        if (SFU && !ST.hasVendorXBOSCSFU())
          fail(MF.getFunction(),
               "SFU offline code generation requires +xboscsfu");
        if ((Compare && MI.getOperand(0).getReg() != RISCV::V0) ||
            (Merge && MI.getOperand(3).getReg() != RISCV::V0))
          fail(MF.getFunction(), "physical mask operand must be v0");
        if (Accumulate &&
            MI.getOperand(0).getReg() != MI.getOperand(1).getReg())
          fail(MF.getFunction(),
               "FP32 accumulate must read its old destination");
        if (Gather && MI.getOperand(0).getReg() == MI.getOperand(1).getReg())
          fail(MF.getFunction(),
               "gather destination must not overlap its source");
        if (Opcode == RISCV::VSLL_VI && (!MI.getOperand(2).isImm() ||
                                         !isUInt<5>(MI.getOperand(2).getImm())))
          fail(MF.getFunction(), "e32 shift amount must be in 0..31");
        if (Native && ((!Extract && !MI.readsRegister(RISCV::VL, &TRI)) ||
                       !MI.readsRegister(RISCV::VTYPE, &TRI)))
          fail(MF.getFunction(), "missing VL/VTYPE dependency");
        if ((isRoundedBinaryOpcode(Opcode) || Accumulate ||
             isConversionOpcode(Opcode) || Opcode == RISCV::VFREDOSUM_VS) &&
            !MI.readsRegister(RISCV::FRM, &TRI))
          fail(MF.getFunction(),
               "missing FRM dependency on FP32 vector operation");
        if (FPVector && !ST.hasVInstructionsF32())
          fail(MF.getFunction(),
               "FP32 vector arithmetic requires vector floating-point F32");
        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.getReg())
            continue;
          Register Reg = MO.getReg();
          if ((Reg == RISCV::VL || Reg == RISCV::VTYPE) && !Native && !Config)
            fail(MF.getFunction(),
                 "uncontrolled vector configuration operation");
          if (!isVRF(Reg, MRI, TRI))
            continue;
          if (Reg.isVirtual())
            fail(MF.getFunction(), "virtual vector register is forbidden");
          if (MO.isRenamable() || MO.getSubReg())
            fail(MF.getFunction(), "renamable or subregister VRF is forbidden");
          bool MaskOperand = (Compare && &MO == &MI.getOperand(0)) ||
                             (Merge && &MO == &MI.getOperand(3));
          if (!Native || !RISCV::VRRegClass.contains(Reg) ||
              (Reg == RISCV::V0 && !MaskOperand))
            fail(MF.getFunction(), "unbound vector operation, copy, spill or "
                                   "reserved vector register is forbidden");
        }
      }
    verifyMachineFlow(MF);
    // Independently receive the physical issue dependency after ingress and
    // at every existing pre/post-RA and final-emission verification gate.
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB)
        verifyIssueOrder(MI);
    return false;
  }
};
} // namespace

Error llvm::RISCV::verifyBOSCModule(Module &M, TargetMachine &TM) {
  std::string Details;
  raw_string_ostream OS(Details);
  if (verifyModule(M, &OS))
    return createStringError("invalid LLVM module: " + OS.str());
  if (!TM.getTargetTriple().isRISCV() ||
      M.getTargetTriple() != TM.getTargetTriple())
    return createStringError(
        "BOSC module requires the actual RISC-V target machine triple");
  auto &RTM = static_cast<RISCVTargetMachine &>(TM);
  // Obtain target facts without importing untrusted per-function vscale
  // bounds. A separate module leaves the candidate and its users untouched.
  Module Baseline("bosc.target", M.getContext());
  Baseline.setTargetTriple(TM.getTargetTriple());
  Baseline.setDataLayout(TM.createDataLayout());
  Function *BaselineFunction = Function::Create(
      FunctionType::get(Type::getVoidTy(M.getContext()), false),
      GlobalValue::ExternalLinkage, "bosc.target", Baseline);
  const auto &BaselineST = RTM.getSubtarget<RISCVSubtarget>(*BaselineFunction);
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    // Library ingress uses the actual machine's already-accepted target
    // configuration. Do not let per-function overrides reach subtarget
    // construction, where incompatible CPU/features may report a fatal error.
    for (auto [Name, Value] :
         {std::pair<StringRef, StringRef>{"target-cpu", TM.getTargetCPU()},
          {"target-features", TM.getTargetFeatureString()}}) {
      Attribute A = F.getFnAttribute(Name);
      if (A.isValid() && A.getValueAsString() != Value)
        return invalid(F,
                       Twine(Name) + " must match the actual target machine");
    }
    // The generic verifier checks power-of-two vscale bounds. RISC-V
    // subtarget construction additionally requires 64 <= VLEN <= 65536.
    Attribute Scale = F.getFnAttribute(Attribute::VScaleRange);
    if (Scale.isValid() && (Scale.getVScaleRangeMin() > 1024 ||
                            Scale.getVScaleRangeMax().value_or(0) > 1024))
      return invalid(F, "vscale_range exceeds the RISC-V VLEN limit");
    if (Scale.isValid())
      if (Error E = BaselineST.verifyRVVVectorBounds(
              Scale.getVScaleRangeMin() * RISCV::RVVBitsPerBlock,
              Scale.getVScaleRangeMax().value_or(0) * RISCV::RVVBitsPerBlock))
        return E;
    const auto &ST = RTM.getSubtarget<RISCVSubtarget>(F);
    if (Error E = verifyBOSCFunction(F, ST, true))
      return E;
  }
  return Error::success();
}

bool llvm::isBOSCContentsSFUIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_sfu_exp2_contents ||
         ID == Intrinsic::riscv_bosc_sfu_rcp_contents;
}

bool llvm::isBOSCOrderedRVVContentsIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_bosc_rvv_contents_add:
  case Intrinsic::riscv_bosc_rvv_contents_maximum:
  case Intrinsic::riscv_bosc_rvv_contents_isplat:
  case Intrinsic::riscv_bosc_rvv_contents_iadd_scalar:
  case Intrinsic::riscv_bosc_rvv_contents_shl:
  case Intrinsic::riscv_bosc_rvv_contents_gather_scalar:
  case Intrinsic::riscv_bosc_rvv_contents_load:
  case Intrinsic::riscv_bosc_rvv_contents_store:
  case Intrinsic::riscv_bosc_rvv_contents_fadd:
  case Intrinsic::riscv_bosc_rvv_contents_fsub:
  case Intrinsic::riscv_bosc_rvv_contents_fmul:
  case Intrinsic::riscv_bosc_rvv_contents_fdiv:
  case Intrinsic::riscv_bosc_rvv_contents_fmax:
  case Intrinsic::riscv_bosc_rvv_contents_fmacc:
  case Intrinsic::riscv_bosc_rvv_contents_fcvt_to_i32:
  case Intrinsic::riscv_bosc_rvv_contents_fcvt_from_i32:
  case Intrinsic::riscv_bosc_rvv_contents_fcmp_lt:
  case Intrinsic::riscv_bosc_rvv_contents_fcmp_eq:
  case Intrinsic::riscv_bosc_rvv_contents_freduce_sum:
  case Intrinsic::riscv_bosc_rvv_contents_freduce_max:
  case Intrinsic::riscv_bosc_rvv_contents_fsplat:
  case Intrinsic::riscv_bosc_rvv_contents_fextract:
  case Intrinsic::riscv_bosc_read_bits_contents:
    return true;
  default:
    return false;
  }
}

namespace {
SDValue lowerOrderedRVVContentsIntrinsic(SDValue Op, SelectionDAG &DAG,
                                         const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isBOSCOrderedRVVContentsIntrinsic(ID))
    return SDValue();
  const Function &F = DAG.getMachineFunction().getFunction();
  if (!isBOSCContentsRVVFunction(F) || !ST.is64Bit() ||
      !ST.hasVInstructions() || ST.getRealMinVLen() < 512)
    fail(F, "ordered RVV contents require RV64 +v +zvl512b contents scope");
  bool Load = ID == Intrinsic::riscv_bosc_rvv_contents_load;
  bool Store = ID == Intrinsic::riscv_bosc_rvv_contents_store;
  bool Observe = ID == Intrinsic::riscv_bosc_read_bits_contents;
  if (requiresRVVF32(ID) && !ST.hasVInstructionsF32())
    fail(F, "FP32 vector arithmetic requires vector floating-point F32");

  SDLoc DL(Op);
  auto Target = [&](uint64_t Value) {
    return DAG.getTargetConstant(Value, DL, MVT::i64);
  };
  SDValue Chain = Op.getOperand(0);
  if (Observe) {
    const auto *Width = dyn_cast<ConstantSDNode>(Op.getOperand(4));
    if (!Width || (Width->getZExtValue() != 8 && Width->getZExtValue() != 32 &&
                   Width->getZExtValue() != 64))
      fail(F, "bit observation width must be 8, 32, or 64");
    SDValue Operands[] = {
        Op.getOperand(2), Op.getOperand(3),
        DAG.getTargetConstant(Width->getZExtValue(), DL, MVT::i32), Chain};
    // The unused vector output is an ordinary allocator-owned scratch, not a
    // physical selector or spill slot. Keep the entire observation closed.
    auto *Node = DAG.getMachineNode(
        RISCV::PseudoBOSCReadBitsBound, DL,
        DAG.getVTList(MVT::i64, MVT::nxv2i32, MVT::Other), Operands);
    auto *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(),
        MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile,
        LocationSize::precise(Width->getZExtValue() / 8),
        Align(Width->getZExtValue() / 8));
    DAG.setNodeMemRefs(Node, {MMO});
    return DAG.getMergeValues({SDValue(Node, 0), SDValue(Node, 2)}, DL);
  }

  SmallVector<SDValue, 9> Operands;
  unsigned Opcode = 0;
  bool Rounded = false;
  bool Compare = ID == Intrinsic::riscv_bosc_rvv_contents_fcmp_lt ||
                 ID == Intrinsic::riscv_bosc_rvv_contents_fcmp_eq;
  bool Extract = ID == Intrinsic::riscv_bosc_rvv_contents_fextract;
  bool Reduce = ID == Intrinsic::riscv_bosc_rvv_contents_freduce_sum ||
                ID == Intrinsic::riscv_bosc_rvv_contents_freduce_max;
  switch (ID) {
  case Intrinsic::riscv_bosc_rvv_contents_add:
    Opcode = RISCV::PseudoVADD_VV_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_maximum:
    Opcode = RISCV::PseudoVMAX_VV_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_isplat:
    Opcode = RISCV::PseudoVMV_V_X_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_iadd_scalar:
    Opcode = RISCV::PseudoVADD_VX_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_shl:
    Opcode = RISCV::PseudoVSLL_VI_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_gather_scalar:
    Opcode = RISCV::PseudoVRGATHER_VI_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_load:
    Opcode = RISCV::PseudoVLE32_V_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_store:
    Opcode = RISCV::PseudoVSE32_V_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fadd:
    Opcode = RISCV::PseudoVFADD_VV_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fsub:
    Opcode = RISCV::PseudoVFSUB_VV_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fmul:
    Opcode = RISCV::PseudoVFMUL_VV_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fdiv:
    Opcode = RISCV::PseudoVFDIV_VV_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fmax:
    Opcode = RISCV::PseudoVFMAX_VV_M1_E32;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fmacc:
    Opcode = RISCV::PseudoVFMACC_VV_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fcvt_to_i32:
    Opcode = RISCV::PseudoVFCVT_X_F_V_M1;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fcvt_from_i32:
    Opcode = RISCV::PseudoVFCVT_F_X_V_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fcmp_lt:
    Opcode = RISCV::PseudoBOSC_VMFLT_VV_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fcmp_eq:
    Opcode = RISCV::PseudoBOSC_VMFEQ_VV_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_freduce_sum:
    Opcode = RISCV::PseudoVFREDOSUM_VS_M1_E32;
    Rounded = true;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_freduce_max:
    Opcode = RISCV::PseudoVFREDMAX_VS_M1_E32;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fsplat:
    Opcode = RISCV::PseudoVFMV_V_FPR32_M1;
    break;
  case Intrinsic::riscv_bosc_rvv_contents_fextract:
    Opcode = RISCV::PseudoVFMV_FPR32_S;
    break;
  default:
    llvm_unreachable("classified ordered RVV contents intrinsic");
  }
  unsigned LastArgument = Op.getNumOperands() - 1;
  if (Compare) {
    // VL=0 preserves every old mask bit. With positive VL, inactive mask tails
    // are still architecturally agnostic; the tie makes no stronger promise.
    if (Op.getOperand(2).isUndef()) {
      // An undefined prior mask has no bits to preserve, including at VL=0.
      // Avoid manufacturing a passthru register copy for initialization.
      Opcode = ID == Intrinsic::riscv_bosc_rvv_contents_fcmp_lt
                   ? RISCV::PseudoVMFLT_VV_M1
                   : RISCV::PseudoVMFEQ_VV_M1;
      Operands.append({Op.getOperand(3), Op.getOperand(4)});
    } else {
      Operands.append({Op.getOperand(2), Op.getOperand(3), Op.getOperand(4)});
    }
  } else if (Extract) {
    Operands.push_back(Op.getOperand(2));
  } else {
    for (unsigned I = 2; I != LastArgument; ++I) {
      bool Immediate =
          I == 4 && (ID == Intrinsic::riscv_bosc_rvv_contents_shl ||
                     ID == Intrinsic::riscv_bosc_rvv_contents_gather_scalar);
      Operands.push_back(Immediate ? Target(Op.getConstantOperandVal(I))
                                   : Op.getOperand(I));
    }
    if (Reduce)
      Operands.push_back(Op.getOperand(2)); // old lane zero is the seed.
  }
  if (Rounded)
    Operands.push_back(Target(RISCVFPRndMode::DYN));
  if (!Extract)
    Operands.push_back(Op.getOperand(LastArgument));
  Operands.push_back(Target(5)); // log2(e32)
  if (!Store && !Compare && !Extract)
    Operands.push_back(
        Target(RISCVVType::TAIL_AGNOSTIC | RISCVVType::MASK_AGNOSTIC));
  Operands.push_back(Chain);
  auto VTs = Store ? DAG.getVTList(MVT::Other)
                   : DAG.getVTList(Op.getValueType(), MVT::Other);
  // The official pseudos own encoding, overlap/tied constraints, vsetvli
  // insertion and dynamic FRM dependencies. An empty flag set intentionally
  // retains possible FP exceptions; never attach NoFPExcept here.
  auto *Node = DAG.getMachineNode(Opcode, DL, VTs, Operands);
  if (Load || Store) {
    auto *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(),
        (Load ? MachineMemOperand::MOLoad : MachineMemOperand::MOStore) |
            MachineMemOperand::MOVolatile,
        LocationSize::beforeOrAfterPointer(), Align(4));
    DAG.setNodeMemRefs(Node, {MMO});
  }
  return SDValue(Node, 0);
}
} // namespace

char RISCVBOSCIRVerification::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVBOSCIRVerification, "riscv-bosc-ir-verify",
                      "Verify BOSC bound RVV IR", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVBOSCIRVerification, "riscv-bosc-ir-verify",
                    "Verify BOSC bound RVV IR", false, false)
char RISCVBOSCIssueIngress::ID = 0;
INITIALIZE_PASS(RISCVBOSCIssueIngress, "riscv-bosc-issue-ingress",
                "Materialize BOSC accelerator issue-order dependencies", false,
                false)
char RISCVBOSCBoundVerification::ID = 0;
INITIALIZE_PASS(RISCVBOSCBoundVerification, "riscv-bosc-bound-verify",
                "Verify BOSC physical vector machine operands", false, false)

FunctionPass *llvm::createRISCVBOSCIRVerificationPass(bool Supported) {
  return new RISCVBOSCIRVerification(Supported);
}
FunctionPass *llvm::createRISCVBOSCIssueIngressPass() {
  return new RISCVBOSCIssueIngress();
}
FunctionPass *llvm::createRISCVBOSCBoundVerificationPass() {
  return new RISCVBOSCBoundVerification();
}
PreservedAnalyses llvm::RISCVBOSCNewPMGate::run(Function &F,
                                                FunctionAnalysisManager &) {
  if (Error E = verifyIR(F, false))
    reportFatalUsageError(Twine(toString(std::move(E))));
  return PreservedAnalyses::all();
}

SDValue llvm::lowerBOSCBoundIntrinsic(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &ST) {
  // Ordinary scalar intrinsics may reach this shared lowering hook without
  // the intrinsic-id operand used by RISCV intrinsic nodes.  They must fall
  // through to the target's normal lowering path.
  if (Op.getNumOperands() <= 1)
    return SDValue();
  if (SDValue Contents = lowerOrderedRVVContentsIntrinsic(Op, DAG, ST))
    return Contents;
  if (SDValue Contents = lowerBOSCContentsSFUIntrinsic(Op, DAG, ST))
    return Contents;
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (ID == Intrinsic::riscv_bosc_read_mcycle) {
    if (!ST.is64Bit())
      fail(DAG.getMachineFunction().getFunction(),
           "BOSC observation counter requires RV64");
    // Preserve the intrinsic's chain through the side-effecting CSR read.
    // Bound machine ingress already orders CSRRS with accelerator issues;
    // no helper, physical vector operand, or hardware fence is introduced.
    SDLoc DL(Op);
    SDValue Operands[] = {
        DAG.getTargetConstant(RISCVSysReg::mcycle, DL, MVT::i64),
        DAG.getRegister(RISCV::X0, MVT::i64), Op.getOperand(0)};
    return SDValue(DAG.getMachineNode(RISCV::CSRRS, DL,
                                      DAG.getVTList(MVT::i64, MVT::Other),
                                      Operands),
                   0);
  }
  // The generic intrinsic keeps the standard RISC-V CYCLE lowering.
  if (ID == Intrinsic::readcyclecounter)
    return SDValue();
  if (SDValue AME = lowerBOSCAMEIntrinsic(Op, DAG, ST))
    return AME;
  if (isDMAIntrinsic(ID)) {
    if (!ST.is64Bit())
      fail(DAG.getMachineFunction().getFunction(), "patch7 DMA requires RV64");
    unsigned Opcode =
        ID == Intrinsic::riscv_bosc_dma_writes  ? RISCV::PseudoBOSCDMAWrites
        : ID == Intrinsic::riscv_bosc_dma_reads ? RISCV::PseudoBOSCDMAReads
                                                : RISCV::PseudoBOSCDMAFence;
    return SDValue(
        DAG.getMachineNode(Opcode, SDLoc(Op), MVT::Other, Op.getOperand(0)), 0);
  }
  if (!isBoundIntrinsic(ID))
    return SDValue();
  Function &F = DAG.getMachineFunction().getFunction();
  if (!isBOSCBoundFunction(F) || !ST.is64Bit() || !ST.hasStdExtV() ||
      ST.getRealMinVLen() < 512)
    fail(F, "requires bound scope and RV64 +v +zvl512b");
  if (isBOSCFPRVVIntrinsic(ID) && !ST.hasVInstructionsF32())
    fail(F, "FP32 vector arithmetic requires vector floating-point F32");
  if (isBOSCSFUIntrinsic(ID) && !ST.hasVendorXBOSCSFU())
    fail(F, "SFU offline code generation requires +xboscsfu");
  SDLoc DL(Op);
  SmallVector<SDValue, 4> Args;
  unsigned Opcode;
  if (ID == Intrinsic::riscv_bosc_setvl) {
    Opcode = RISCV::PseudoBOSCSetVL;
    SDValue AVL = Op.getOperand(2);
    if (auto *CI = dyn_cast<ConstantSDNode>(AVL)) {
      uint64_t Requested = CI->getZExtValue();
      if (ST.getRealVLen() == 512 && Requested >= 32)
        Requested = 16;
      if (isUInt<5>(Requested)) {
        Opcode = RISCV::PseudoBOSCSetVLI;
        AVL = DAG.getTargetConstant(Requested, DL, MVT::i64);
      }
    }
    Args.push_back(AVL);
  } else {
    Opcode =
        ID == Intrinsic::riscv_bosc_read_bits     ? RISCV::PseudoBOSCReadBits
        : ID == Intrinsic::riscv_bosc_load        ? RISCV::PseudoBOSCLoad
        : ID == Intrinsic::riscv_bosc_add         ? RISCV::PseudoBOSCAdd
        : ID == Intrinsic::riscv_bosc_maximum     ? RISCV::PseudoBOSCMaximum
        : ID == Intrinsic::riscv_bosc_fadd        ? RISCV::PseudoBOSCFAdd
        : ID == Intrinsic::riscv_bosc_fsub        ? RISCV::PseudoBOSCFSub
        : ID == Intrinsic::riscv_bosc_fmul        ? RISCV::PseudoBOSCFMul
        : ID == Intrinsic::riscv_bosc_fdiv        ? RISCV::PseudoBOSCFDiv
        : ID == Intrinsic::riscv_bosc_fsplat      ? RISCV::PseudoBOSCFSplat
        : ID == Intrinsic::riscv_bosc_isplat      ? RISCV::PseudoBOSCISplat
        : ID == Intrinsic::riscv_bosc_fextract    ? RISCV::PseudoBOSCFExtract
        : ID == Intrinsic::riscv_bosc_freduce_sum ? RISCV::PseudoBOSCFReduceSum
        : ID == Intrinsic::riscv_bosc_freduce_max ? RISCV::PseudoBOSCFReduceMax
        : ID == Intrinsic::riscv_bosc_fmax        ? RISCV::PseudoBOSCFMax
        : ID == Intrinsic::riscv_bosc_fmacc       ? RISCV::PseudoBOSCFMacc
        : ID == Intrinsic::riscv_bosc_fcvt_to_i32 ? RISCV::PseudoBOSCFCvtToI32
        : ID == Intrinsic::riscv_bosc_fcvt_from_i32
            ? RISCV::PseudoBOSCFCvtFromI32
        : ID == Intrinsic::riscv_bosc_fcmp_lt     ? RISCV::PseudoBOSCFCmpLt
        : ID == Intrinsic::riscv_bosc_fcmp_eq     ? RISCV::PseudoBOSCFCmpEq
        : ID == Intrinsic::riscv_bosc_merge       ? RISCV::PseudoBOSCMerge
        : ID == Intrinsic::riscv_bosc_iadd_scalar ? RISCV::PseudoBOSCIAddScalar
        : ID == Intrinsic::riscv_bosc_gather_scalar
            ? RISCV::PseudoBOSCGatherScalar
        : ID == Intrinsic::riscv_bosc_shl      ? RISCV::PseudoBOSCShl
        : ID == Intrinsic::riscv_bosc_sfu_exp2 ? RISCV::PseudoBOSCSFUExp2
        : ID == Intrinsic::riscv_bosc_sfu_rcp  ? RISCV::PseudoBOSCSFURcp
                                               : RISCV::PseudoBOSCStore;
    unsigned Count = boundPhysicalOperands(ID);
    for (unsigned I = 0; I != Count; ++I) {
      auto *CI = dyn_cast<ConstantSDNode>(Op.getOperand(I + 2));
      if (!CI || CI->getZExtValue() > 31 ||
          (isMaskOperand(ID, I) ? CI->getZExtValue() != 0
                                : CI->getZExtValue() == 0))
        fail(F, "physical operand lost its binding");
      Args.push_back(DAG.getTargetConstant(CI->getZExtValue(), DL, MVT::i32));
    }
    if (ID == Intrinsic::riscv_bosc_read_bits ||
        ID == Intrinsic::riscv_bosc_load || ID == Intrinsic::riscv_bosc_store ||
        ID == Intrinsic::riscv_bosc_fsplat ||
        ID == Intrinsic::riscv_bosc_isplat)
      Args.push_back(Op.getOperand(3));
    else if (ID == Intrinsic::riscv_bosc_iadd_scalar)
      Args.push_back(Op.getOperand(4));
    else if (ID == Intrinsic::riscv_bosc_gather_scalar)
      // The intrinsic index is immediate, while VRGATHER_VX needs a GPR.
      // A machine pseudo's constant operand is not selected recursively.
      Args.push_back(SDValue(
          DAG.getMachineNode(
              RISCV::ADDI, DL, MVT::i64, DAG.getRegister(RISCV::X0, MVT::i64),
              DAG.getTargetConstant(Op.getConstantOperandVal(4), DL, MVT::i64)),
          0));
    else if (ID == Intrinsic::riscv_bosc_shl)
      Args.push_back(
          DAG.getTargetConstant(Op.getConstantOperandVal(4), DL, MVT::i32));
    if (ID == Intrinsic::riscv_bosc_read_bits) {
      SDValue Restore = Op.getOperand(4);
      // Even a zero VL must occupy a nonzero register: rs1=x0 would request
      // VSETVLI's special AVL semantics instead of restoring the value zero.
      if (auto *CI = dyn_cast<ConstantSDNode>(Restore))
        Restore = SDValue(
            DAG.getMachineNode(
                RISCV::ADDI, DL, MVT::i64, DAG.getRegister(RISCV::X0, MVT::i64),
                DAG.getTargetConstant(CI->getZExtValue(), DL, MVT::i64)),
            0);
      Args.push_back(Restore);
      auto *Width = dyn_cast<ConstantSDNode>(Op.getOperand(5));
      if (!Width ||
          (Width->getZExtValue() != 8 && Width->getZExtValue() != 32 &&
           Width->getZExtValue() != 64))
        fail(F, "bit observation width must be 8, 32, or 64");
      Args.push_back(
          DAG.getTargetConstant(Width->getZExtValue(), DL, MVT::i32));
    }
  }
  Args.push_back(Op.getOperand(0));
  auto VTs =
      ID == Intrinsic::riscv_bosc_setvl || ID == Intrinsic::riscv_bosc_read_bits
          ? DAG.getVTList(MVT::i64, MVT::Other)
      : ID == Intrinsic::riscv_bosc_fextract
          ? DAG.getVTList(MVT::f32, MVT::Other)
          : DAG.getVTList(MVT::Other);
  auto *Node = DAG.getMachineNode(Opcode, DL, VTs, Args);
  if (ID == Intrinsic::riscv_bosc_load || ID == Intrinsic::riscv_bosc_store ||
      ID == Intrinsic::riscv_bosc_read_bits) {
    bool Observation = ID == Intrinsic::riscv_bosc_read_bits;
    unsigned WidthBytes = Observation ? Op.getConstantOperandVal(5) / 8 : 4;
    auto Flags = ID == Intrinsic::riscv_bosc_store ? MachineMemOperand::MOStore
                                                   : MachineMemOperand::MOLoad;
    auto *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(), Flags | MachineMemOperand::MOVolatile,
        Observation ? LocationSize::precise(WidthBytes)
                    : LocationSize::beforeOrAfterPointer(),
        Align(WidthBytes));
    DAG.setNodeMemRefs(Node, {MMO});
  }
  return SDValue(Node, 0);
}

namespace {
MachineBasicBlock *emitDMAInstruction(MachineInstr &MI, MachineBasicBlock *MBB,
                                      const RISCVSubtarget &ST) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode != RISCV::PseudoBOSCDMAWrites &&
      Opcode != RISCV::PseudoBOSCDMAReads &&
      Opcode != RISCV::PseudoBOSCDMAFence)
    return nullptr;
  const auto &TII = *ST.getInstrInfo();
  auto &MRI = MBB->getParent()->getRegInfo();
  auto GPR = [&]() { return MRI.createVirtualRegister(&RISCV::GPRRegClass); };
  auto Build = [&](unsigned Op) {
    return BuildMI(*MBB, MI, MI.getDebugLoc(), TII.get(Op));
  };
  auto Fence = [&]() { Build(RISCV::FENCE).addImm(15).addImm(15); };
  auto WriteCSR = [&](unsigned CSR, Register Value) {
    Build(RISCV::CSRRW)
        .addReg(RISCV::X0, RegState::Define)
        .addImm(CSR)
        .addReg(Value);
  };
  auto Immediate = [&](uint64_t Value) {
    // Materialize with distinct virtual scalar definitions so machine SSA
    // remains valid. The ordinary allocator may spill these GPR temporaries.
    Register Source = RISCV::X0;
    for (const RISCVMatInt::Inst &Inst :
         RISCVMatInt::generateInstSeq(Value, ST)) {
      Register Destination = GPR();
      auto MIB = Build(Inst.getOpcode()).addReg(Destination, RegState::Define);
      switch (Inst.getOpndKind()) {
      case RISCVMatInt::Imm:
        MIB.addImm(Inst.getImm());
        break;
      case RISCVMatInt::RegImm:
        MIB.addReg(Source).addImm(Inst.getImm());
        break;
      case RISCVMatInt::RegReg:
        MIB.addReg(Source).addReg(Source);
        break;
      case RISCVMatInt::RegX0:
        MIB.addReg(Source).addReg(RISCV::X0);
        break;
      }
      Source = Destination;
    }
    return Source;
  };
  Fence();
  if (Opcode == RISCV::PseudoBOSCDMAWrites) {
    // Support owns TOR bounds and permissions for both execution profiles.
    // Inline kernel phases change only the received PMA attributes; rebuilding
    // geometry here would discard the resident stack and caller-arena regions.
    WriteCSR(0xbc0, Immediate(0x0f0f0f0f0f0f0f0f));
    Fence();
  } else if (Opcode == RISCV::PseudoBOSCDMAReads) {
    // Entry 6 keeps stacks/fallback cacheable; entry 7 acquires the resident
    // arena without cache hits. Preload leaves entry 7 disabled in pmpcfg0.
    WriteCSR(0xbc0, Immediate(0x000f000f000f000f));
    Fence();
  }
  MI.eraseFromParent();
  return MBB;
}
} // namespace

MachineBasicBlock *llvm::emitBOSCBoundInstruction(MachineInstr &MI,
                                                  MachineBasicBlock *MBB,
                                                  const RISCVSubtarget &ST) {
  if (auto *AME = emitBOSCAMEInstruction(MI, MBB, ST))
    return AME;
  if (auto *DMA = emitDMAInstruction(MI, MBB, ST))
    return DMA;
  if (MI.getOpcode() == RISCV::PseudoBOSCReadBits) {
    BuildMI(*MBB, MI, MI.getDebugLoc(),
            ST.getInstrInfo()->get(RISCV::PseudoBOSCReadBitsBound))
        .addReg(MI.getOperand(0).getReg(),
                RegState::Define | RegState::EarlyClobber)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm(), RegState::Define)
        .add(MI.getOperand(2))
        .add(MI.getOperand(3))
        .add(MI.getOperand(4))
        .cloneMemRefs(MI);
    MI.eraseFromParent();
    return MBB;
  }
  unsigned Native;
  switch (MI.getOpcode()) {
  default:
    return nullptr;
  case RISCV::PseudoBOSCSetVL:
    Native = RISCV::VSETVLI;
    break;
  case RISCV::PseudoBOSCSetVLI:
    Native = RISCV::VSETIVLI;
    break;
  case RISCV::PseudoBOSCLoad:
    Native = RISCV::VLE32_V;
    break;
  case RISCV::PseudoBOSCAdd:
    Native = RISCV::VADD_VV;
    break;
  case RISCV::PseudoBOSCMaximum:
    Native = RISCV::VMAX_VV;
    break;
  case RISCV::PseudoBOSCFAdd:
    Native = RISCV::VFADD_VV;
    break;
  case RISCV::PseudoBOSCFSub:
    Native = RISCV::VFSUB_VV;
    break;
  case RISCV::PseudoBOSCFMul:
    Native = RISCV::VFMUL_VV;
    break;
  case RISCV::PseudoBOSCFDiv:
    Native = RISCV::VFDIV_VV;
    break;
  case RISCV::PseudoBOSCStore:
    Native = RISCV::VSE32_V;
    break;
  case RISCV::PseudoBOSCFSplat:
    Native = RISCV::VFMV_V_F;
    break;
  case RISCV::PseudoBOSCISplat:
    Native = RISCV::VMV_V_X;
    break;
  case RISCV::PseudoBOSCFExtract:
    Native = RISCV::VFMV_F_S;
    break;
  case RISCV::PseudoBOSCFReduceSum:
    Native = RISCV::VFREDOSUM_VS;
    break;
  case RISCV::PseudoBOSCFReduceMax:
    Native = RISCV::VFREDMAX_VS;
    break;
  case RISCV::PseudoBOSCFMax:
    Native = RISCV::VFMAX_VV;
    break;
  case RISCV::PseudoBOSCFMacc:
    Native = RISCV::VFMACC_VV;
    break;
  case RISCV::PseudoBOSCFCvtToI32:
    Native = RISCV::VFCVT_X_F_V;
    break;
  case RISCV::PseudoBOSCFCvtFromI32:
    Native = RISCV::VFCVT_F_X_V;
    break;
  case RISCV::PseudoBOSCFCmpLt:
    Native = RISCV::VMFLT_VV;
    break;
  case RISCV::PseudoBOSCFCmpEq:
    Native = RISCV::VMFEQ_VV;
    break;
  case RISCV::PseudoBOSCMerge:
    Native = RISCV::VMERGE_VVM;
    break;
  case RISCV::PseudoBOSCIAddScalar:
    Native = RISCV::VADD_VX;
    break;
  case RISCV::PseudoBOSCGatherScalar:
    Native = RISCV::VRGATHER_VX;
    break;
  case RISCV::PseudoBOSCShl:
    Native = RISCV::VSLL_VI;
    break;
  case RISCV::PseudoBOSCSFUExp2:
    Native = RISCV::BOSC_SFU_EXP2;
    break;
  case RISCV::PseudoBOSCSFURcp:
    Native = RISCV::BOSC_SFU_RCP;
    break;
  }
  auto Builder =
      BuildMI(*MBB, MI, MI.getDebugLoc(), ST.getInstrInfo()->get(Native));
  if (Native == RISCV::VSETVLI || Native == RISCV::VSETIVLI) {
    Builder.add(MI.getOperand(0)).add(MI.getOperand(1)).addImm(E32M1TaMa);
  } else if (Native == RISCV::VFMV_V_F || Native == RISCV::VMV_V_X) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .add(MI.getOperand(1));
  } else if (Native == RISCV::VFMV_F_S) {
    Builder.add(MI.getOperand(0)).addReg(RISCV::V0 + MI.getOperand(1).getImm());
  } else if (Native == RISCV::VFREDOSUM_VS || Native == RISCV::VFREDMAX_VS) {
    Register Dst = RISCV::V0 + MI.getOperand(0).getImm();
    Builder.addReg(Dst, RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(Dst)
        .addReg(RISCV::NoRegister);
  } else if (Native == RISCV::VFMACC_VV) {
    Register Dst = RISCV::V0 + MI.getOperand(0).getImm();
    Builder.addReg(Dst, RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(0).getImm())
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(RISCV::V0 + MI.getOperand(2).getImm())
        .addReg(RISCV::NoRegister);
  } else if (Native == RISCV::VMFLT_VV || Native == RISCV::VMFEQ_VV) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(RISCV::V0 + MI.getOperand(2).getImm())
        .addReg(RISCV::NoRegister);
  } else if (Native == RISCV::VMERGE_VVM) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(RISCV::V0 + MI.getOperand(2).getImm())
        .addReg(RISCV::V0 + MI.getOperand(3).getImm());
  } else if (Native == RISCV::VADD_VX || Native == RISCV::VRGATHER_VX) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .add(MI.getOperand(2))
        .addReg(RISCV::NoRegister);
  } else if (Native == RISCV::VSLL_VI) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .add(MI.getOperand(2))
        .addReg(RISCV::NoRegister);
  } else if (isSFUOpcode(Native)) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm());
  } else if (Native == RISCV::VFCVT_X_F_V || Native == RISCV::VFCVT_F_X_V) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(RISCV::NoRegister);
  } else if (Native == RISCV::VFMAX_VV || Native == RISCV::VMAX_VV) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(RISCV::V0 + MI.getOperand(2).getImm())
        .addReg(RISCV::NoRegister);
  } else {
    unsigned Count = Native == RISCV::VADD_VV || Native == RISCV::VFADD_VV ||
                             Native == RISCV::VFSUB_VV ||
                             Native == RISCV::VFMUL_VV ||
                             Native == RISCV::VFDIV_VV
                         ? 3
                         : 1;
    for (unsigned I = 0; I != Count; ++I)
      Builder.addReg(RISCV::V0 + MI.getOperand(I).getImm(),
                     I == 0 && Native != RISCV::VSE32_V ? RegState::Define
                                                        : RegState::NoFlags);
    if (Count == 1)
      Builder.add(MI.getOperand(1));
    Builder.addReg(RISCV::NoRegister); // Unmasked standard RVV form.
    Builder.cloneMemRefs(MI);
  }
  MI.eraseFromParent();
  return MBB;
}
