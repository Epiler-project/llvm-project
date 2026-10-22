//===-- RISCVBOSCBound.cpp ------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVBOSCBound.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMatInt.h"
#include "RISCVBOSCAME.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <limits>
#include <optional>

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

[[noreturn]] void fail(const Function &F, const Twine &Message) {
  reportFatalUsageError("BOSC bound RVV function '" + F.getName() +
                        "': " + Message);
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

unsigned gatherIndex(const IntrinsicInst &II, bool Exact512) {
  const auto *Index = dyn_cast<ConstantInt>(II.getArgOperand(2));
  if (!Index || Index->getValue().ugt(16))
    fail(*II.getFunction(), "gather index must be a constant in 0..16");
  unsigned Lane = Index->getZExtValue();
  if (Lane == 16 && !Exact512)
    fail(*II.getFunction(), "gather index 16 requires exact VLEN=512");
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

bool hasBOSCObservationIntrinsic(const Function &F) {
  return llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && II->getIntrinsicID() == Intrinsic::riscv_bosc_read_mcycle;
  });
}

unsigned physicalArgument(const IntrinsicInst &II, unsigned Index) {
  auto *CI = dyn_cast<ConstantInt>(II.getArgOperand(Index));
  if (isMaskOperand(II.getIntrinsicID(), Index)) {
    if (!CI || !CI->isZero())
      fail(*II.getFunction(), "physical mask operand must be v0");
  } else if (!CI || CI->getValue().getActiveBits() > 5 || !CI->getZExtValue())
    fail(*II.getFunction(),
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

LaneState transfer(const BasicBlock &BB, LaneState State, bool Verify,
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
    if (ID == Intrinsic::riscv_bosc_read_bits) {
      if (Verify) {
        auto *Width = dyn_cast<ConstantInt>(II->getArgOperand(3));
        if (!Width ||
            (Width->getZExtValue() != 8 && Width->getZExtValue() != 32 &&
             Width->getZExtValue() != 64))
          fail(*BB.getParent(), "bit observation width must be 8, 32, or 64");
        if (State.Configurations.size() != 1)
          fail(*BB.getParent(),
               "bit observation requires one reaching vector configuration");
        const void *Config = State.Configurations.begin()->first;
        Value *Restore = II->getArgOperand(2);
        if (Config && Config != Restore)
          fail(*BB.getParent(),
               "bit observation must restore the active setvl result");
        if (!Config) {
          auto *Zero = dyn_cast<ConstantInt>(Restore);
          if (!Zero || !Zero->isZero())
            fail(*BB.getParent(), "bit observation without a source "
                                  "configuration must restore literal zero");
        }
      }
      // The temporary configuration is private to this operation. Do not
      // expose the scratch as an initialized business e32 vector: only one
      // element was read, under a potentially different element width.
      State.Initialized[physicalArgument(*II, 0)] = std::nullopt;
      continue;
    }
    if (ID == Intrinsic::riscv_bosc_setvl) {
      auto *AVL = dyn_cast<ConstantInt>(II->getArgOperand(0));
      State.configure(II, AVL ? exactLanes(AVL->getZExtValue(), Exact512)
                              : std::nullopt);
      continue;
    }
    if (Verify && !State.configured())
      fail(*BB.getParent(), "vector operation requires an explicit setvl on "
                            "every incoming path");
    auto Read = [&](unsigned Index) {
      unsigned Reg = physicalArgument(*II, Index);
      if (Verify && !State.readable(Reg))
        fail(*BB.getParent(), "read of undefined physical vector register v" +
                                  Twine(Reg) +
                                  " covering the active setvl lanes");
    };
    if (isBoundBinaryIntrinsic(ID) || ID == Intrinsic::riscv_bosc_merge) {
      Read(1);
      Read(2);
    }
    if (ID == Intrinsic::riscv_bosc_fmacc)
      Read(0);
    if (ID == Intrinsic::riscv_bosc_merge)
      Read(3);
    if (isBoundUnaryIntrinsic(ID))
      Read(1);
    if (ID == Intrinsic::riscv_bosc_gather_scalar) {
      unsigned Index = gatherIndex(*II, Exact512);
      if (Verify && Index < 16 && !State.noActiveLanes() &&
          !State.readableLane(physicalArgument(*II, 1), Index))
        fail(*BB.getParent(),
             "gather index reads an uninitialized source lane");
    }
    if (isBoundReduceIntrinsic(ID))
      Read(1);
    if (isBoundReduceIntrinsic(ID) || ID == Intrinsic::riscv_bosc_fextract) {
      unsigned Reg = physicalArgument(*II, 0);
      if (Verify && !State.readableZero(Reg))
        fail(*BB.getParent(),
             "scalar lane read requires initialized lane zero");
      if (isBoundReduceIntrinsic(ID))
        State.reduce(Reg);
      continue;
    }
    if (ID == Intrinsic::riscv_bosc_store)
      Read(0);
    else
      State.define(physicalArgument(*II, 0));
  }
  return State;
}

void verifyIR(Function &F, bool Supported, bool Exact512 = false) {
  bool Bound = isBOSCBoundFunction(F);
  bool RVV = F.hasFnAttribute("trex-bosc-bound-rvv");
  bool DMA = hasDMAIntrinsic(F);
  bool Observation = hasBOSCObservationIntrinsic(F);
  for (Instruction &I : instructions(F)) {
    auto *Call = dyn_cast<CallBase>(&I);
    if (!Call)
      continue;
    auto *Callee = Call->getCalledFunction();
    auto *II = dyn_cast<IntrinsicInst>(Call);
    if (Callee && Callee->getName().starts_with("llvm.riscv.bosc.") &&
        (!II || (!isBoundIntrinsic(II->getIntrinsicID()) &&
                 !isBOSCAMEIntrinsic(II->getIntrinsicID()) &&
                 !isDMAIntrinsic(II->getIntrinsicID()))))
      fail(F, "unknown BOSC intrinsic");
    if (II && II->getIntrinsicID() == Intrinsic::riscv_bosc_read_bits && !RVV)
      fail(F, "bit observation requires trex-bosc-bound-rvv=1");
    if (II && isBoundIntrinsic(II->getIntrinsicID()) &&
        !isScalarObservationIntrinsic(II->getIntrinsicID()) && !RVV &&
        !F.hasFnAttribute("trex-bosc-bound-ame"))
      fail(F, "intrinsic requires trex-bosc-bound-rvv=1");
  }
  verifyBOSCAMEIR(F);
  if (!Bound && !DMA && !Observation)
    return;
  if (!Supported)
    fail(F, "requires the SelectionDAG legacy codegen pipeline");
  if (RVV && F.getFnAttribute("trex-bosc-bound-rvv").getValueAsString() != "1")
    fail(F, "unsupported bound ABI version");
  if (containsVector(F.getReturnType()))
    fail(F, "LLVM vector values are forbidden in bound scope");
  for (const Argument &A : F.args())
    if (containsVector(A.getType()))
      fail(F, "LLVM vector values are forbidden in bound scope");
  for (Instruction &I : instructions(F)) {
    if (containsVector(I.getType()) ||
        llvm::any_of(I.operands(), [](const Value *V) {
          return containsVector(V->getType());
        }))
      fail(F, "LLVM vector values are forbidden in bound scope");
    auto *Call = dyn_cast<CallBase>(&I);
    if (!Call)
      continue;
    auto *II = dyn_cast<IntrinsicInst>(Call);
    if (II && isa<DbgInfoIntrinsic>(II))
      continue;
    // Compiler-authored math and address guards terminate bound execution.
    // Their validity does not depend on whether the function also issues DMA.
    if (II && II->getIntrinsicID() == Intrinsic::trap) {
      if (!isa_and_nonnull<UnreachableInst>(I.getNextNode()))
        fail(F, "trap must be immediately followed by unreachable");
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
      fail(F, "calls, inline assembly and unbound intrinsics are forbidden "
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
           ++I)
        physicalArgument(*II, I);
    if (II->getIntrinsicID() == Intrinsic::riscv_bosc_shl) {
      const auto *Amount = dyn_cast<ConstantInt>(II->getArgOperand(2));
      if (!Amount || Amount->getValue().getActiveBits() > 5)
        fail(F, "e32 shift amount must be in 0..31");
    }
    if (II->getIntrinsicID() == Intrinsic::riscv_bosc_gather_scalar) {
      gatherIndex(*II, Exact512);
      if (physicalArgument(*II, 0) == physicalArgument(*II, 1))
        fail(F, "gather destination must not overlap its source");
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
      LaneState Next = transfer(BB, In, false, Exact512);
      if (Out[&BB] != Next) {
        Out[&BB] = Next;
        Changed = true;
      }
    }
  } while (Changed);
  for (const BasicBlock &BB : F) {
    LaneState In = BB.isEntryBlock() || pred_empty(&BB) ? LaneState::entry()
                                                        : LaneState::top();
    for (const BasicBlock *Pred : predecessors(&BB))
      In.meet(Out[Pred]);
    transfer(BB, In, true, Exact512);
  }
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
    verifyIR(F, Supported, ST.hasVInstructions() && ST.getRealVLen() == 512);
    if (isBOSCBoundFunction(F) || hasDMAIntrinsic(F) ||
        hasBOSCObservationIntrinsic(F)) {
      if (hasBOSCObservationIntrinsic(F) && !ST.is64Bit())
        fail(F, "BOSC observation counter requires RV64");
      if (hasDMAIntrinsic(F) && !ST.is64Bit())
        fail(F, "patch7 DMA requires RV64");
      if (F.hasFnAttribute("trex-bosc-bound-rvv") &&
          (!ST.is64Bit() || !ST.hasStdExtV() || ST.getRealMinVLen() < 512))
        fail(F, "requires bound scope and RV64 +v +zvl512b");
      if (llvm::any_of(instructions(F),
                       [](const Instruction &I) {
                         const auto *II = dyn_cast<IntrinsicInst>(&I);
                         return II &&
                                isBOSCFPRVVIntrinsic(II->getIntrinsicID());
                       }) &&
          !ST.hasVInstructionsF32())
        fail(F, "FP32 vector arithmetic requires vector floating-point F32");
      if (F.hasFnAttribute("trex-bosc-bound-ame") &&
          (!ST.is64Bit() || !ST.hasVendorXBOSCAME()))
        fail(F, "requires bound AME scope and RV64 +xboscame");
      if (!ST.hasVendorXBOSCSFU() &&
          llvm::any_of(instructions(F), [](const Instruction &I) {
            const auto *II = dyn_cast<IntrinsicInst>(&I);
            return II && isBOSCSFUIntrinsic(II->getIntrinsicID());
          }))
        fail(F, "SFU offline code generation requires +xboscsfu");
    }
    return false;
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
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
  unsigned Opcode = MI.getOpcode();
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
  if (!F.hasFnAttribute("trex-bosc-bound-rvv"))
    fail(F, "bit observation requires bound RVV scope");
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
      if (Reg.isVirtual() || !RISCV::VRRegClass.contains(Reg) ||
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
        !hasDMAIntrinsic(MF.getFunction()))
      return false;
    const auto &MRI = MF.getRegInfo();
    const auto &ST = MF.getSubtarget<RISCVSubtarget>();
    const auto &TRI = *ST.getRegisterInfo();
    verifyBOSCAMEMachine(MF);
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
  verifyIR(F, false);
  return PreservedAnalyses::all();
}

SDValue llvm::lowerBOSCBoundIntrinsic(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &ST) {
  // Ordinary scalar intrinsics may reach this shared lowering hook without
  // the intrinsic-id operand used by RISCV intrinsic nodes.  They must fall
  // through to the target's normal lowering path.
  if (Op.getNumOperands() <= 1)
    return SDValue();
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
