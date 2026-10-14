//===-- RISCVBOSCBound.cpp ------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVBOSCBound.h"
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

bool isBoundIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_setvl ||
         ID == Intrinsic::riscv_bosc_load || ID == Intrinsic::riscv_bosc_add ||
         ID == Intrinsic::riscv_bosc_store || isBOSCFPRVVIntrinsic(ID) ||
         ID == Intrinsic::riscv_bosc_merge ||
         ID == Intrinsic::riscv_bosc_iadd_scalar ||
         ID == Intrinsic::riscv_bosc_gather_scalar ||
         ID == Intrinsic::riscv_bosc_shl;
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
         ID == Intrinsic::riscv_bosc_shl;
}
bool isMaskOperand(Intrinsic::ID ID, unsigned Index) {
  return ((ID == Intrinsic::riscv_bosc_fcmp_lt ||
           ID == Intrinsic::riscv_bosc_fcmp_eq) &&
          Index == 0) ||
         (ID == Intrinsic::riscv_bosc_merge && Index == 3);
}

bool isBoundBinaryIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_bosc_add || ID == Intrinsic::riscv_bosc_fadd ||
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
        fail(*BB.getParent(), "gather index reads an uninitialized source lane");
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
    if (II && isBoundIntrinsic(II->getIntrinsicID()) && !RVV)
      fail(F, "intrinsic requires trex-bosc-bound-rvv=1");
  }
  verifyBOSCAMEIR(F);
  if (!Bound && !DMA)
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
    if (DMA && II && II->getIntrinsicID() == Intrinsic::trap) {
      if (!isa_and_nonnull<UnreachableInst>(I.getNextNode()))
        fail(F, "trap must be immediately followed by unreachable");
      continue;
    }
    if (!II || (!isBoundIntrinsic(II->getIntrinsicID()) &&
                !isBOSCAMEIntrinsic(II->getIntrinsicID()) &&
                !isDMAIntrinsic(II->getIntrinsicID())))
      fail(F, "calls, inline assembly and unbound intrinsics are forbidden "
              "in bound scope");
    if (isBOSCAMEIntrinsic(II->getIntrinsicID()) ||
        isDMAIntrinsic(II->getIntrinsicID()))
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
    if (isBOSCBoundFunction(F) || hasDMAIntrinsic(F)) {
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
bool isUnaryOpcode(unsigned Opcode) {
  return isConversionOpcode(Opcode) || Opcode == RISCV::VADD_VX ||
         Opcode == RISCV::VSLL_VI;
}
bool isBinaryOpcode(unsigned Opcode) {
  return Opcode == RISCV::VADD_VV || isRoundedBinaryOpcode(Opcode) ||
         Opcode == RISCV::VFMAX_VV || isCompareOpcode(Opcode) ||
         Opcode == RISCV::VMERGE_VVM;
}

// Keep the scalar index independently checkable after ordinary GPR allocation.
// Only the bounded gather constants are facts: -2 is the initial lattice top,
// -1 is unknown, and 0..16 are known values. Physical aliases and register masks
// kill facts, so no LLVM intrinsic annotation is trusted by the MIR verifier.
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
    return Reg.isVirtual() ? PhysicalCount + Reg.virtRegIndex()
                           : Reg.id();
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
          for (MCRegAliasIterator Alias(Reg, &TRI, true); Alias.isValid(); ++Alias)
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
    bool Splat = Opcode == RISCV::VFMV_V_F;
    bool Extract = Opcode == RISCV::VFMV_F_S;
    bool Reduce = Opcode == RISCV::VFREDOSUM_VS || Opcode == RISCV::VFREDMAX_VS;
    if (!Load && !Binary && !Unary && !Gather && !Accumulate && !Store && !Splat &&
        !Extract && !Reduce)
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
      Opcode == RISCV::VFMV_V_F || Opcode == RISCV::VFMV_F_S ||
      Opcode == RISCV::VFREDOSUM_VS || Opcode == RISCV::VFREDMAX_VS ||
      Opcode == RISCV::VRGATHER_VX)
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
        bool Accumulate = Opcode == RISCV::VFMACC_VV;
        bool Compare = isCompareOpcode(Opcode);
        bool Merge = Opcode == RISCV::VMERGE_VVM;
        bool Gather = Opcode == RISCV::VRGATHER_VX;
        bool NativeFloat = isRoundedBinaryOpcode(Opcode) || Accumulate ||
                           isConversionOpcode(Opcode);
        bool Splat = MI.getOpcode() == RISCV::VFMV_V_F;
        bool Extract = MI.getOpcode() == RISCV::VFMV_F_S;
        bool Reduce = MI.getOpcode() == RISCV::VFREDOSUM_VS ||
                      MI.getOpcode() == RISCV::VFREDMAX_VS;
        bool FPVector = NativeFloat || Compare || Opcode == RISCV::VFMAX_VV ||
                        Splat || Extract || Reduce;
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
                            : Splat || Extract ? 2
                                               : 3;
        if (Native && MI.getNumExplicitOperands() != Expected)
          fail(MF.getFunction(),
               "bound native operation has invalid operand count");
        if (Native && !Splat && !Extract && !Merge &&
            (!MI.getOperand(Expected - 1).isReg() ||
             MI.getOperand(Expected - 1).getReg()))
          fail(MF.getFunction(), "bound native operation must be unmasked");
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
  if (SDValue AME = lowerBOSCAMEIntrinsic(Op, DAG, ST))
    return AME;
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
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
        ID == Intrinsic::riscv_bosc_load          ? RISCV::PseudoBOSCLoad
        : ID == Intrinsic::riscv_bosc_add         ? RISCV::PseudoBOSCAdd
        : ID == Intrinsic::riscv_bosc_fadd        ? RISCV::PseudoBOSCFAdd
        : ID == Intrinsic::riscv_bosc_fsub        ? RISCV::PseudoBOSCFSub
        : ID == Intrinsic::riscv_bosc_fmul        ? RISCV::PseudoBOSCFMul
        : ID == Intrinsic::riscv_bosc_fdiv        ? RISCV::PseudoBOSCFDiv
        : ID == Intrinsic::riscv_bosc_fsplat      ? RISCV::PseudoBOSCFSplat
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
        : ID == Intrinsic::riscv_bosc_shl         ? RISCV::PseudoBOSCShl
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
    if (ID == Intrinsic::riscv_bosc_load || ID == Intrinsic::riscv_bosc_store ||
        ID == Intrinsic::riscv_bosc_fsplat)
      Args.push_back(Op.getOperand(3));
    else if (ID == Intrinsic::riscv_bosc_iadd_scalar)
      Args.push_back(Op.getOperand(4));
    else if (ID == Intrinsic::riscv_bosc_gather_scalar)
      // The intrinsic index is immediate, while VRGATHER_VX needs a GPR.
      // A machine pseudo's constant operand is not selected recursively.
      Args.push_back(SDValue(
          DAG.getMachineNode(
              RISCV::ADDI, DL, MVT::i64,
              DAG.getRegister(RISCV::X0, MVT::i64),
              DAG.getTargetConstant(Op.getConstantOperandVal(4), DL, MVT::i64)),
          0));
    else if (ID == Intrinsic::riscv_bosc_shl)
      Args.push_back(
          DAG.getTargetConstant(Op.getConstantOperandVal(4), DL, MVT::i32));
  }
  Args.push_back(Op.getOperand(0));
  auto VTs = ID == Intrinsic::riscv_bosc_setvl
                 ? DAG.getVTList(MVT::i64, MVT::Other)
             : ID == Intrinsic::riscv_bosc_fextract
                 ? DAG.getVTList(MVT::f32, MVT::Other)
                 : DAG.getVTList(MVT::Other);
  auto *Node = DAG.getMachineNode(Opcode, DL, VTs, Args);
  if (ID == Intrinsic::riscv_bosc_load || ID == Intrinsic::riscv_bosc_store) {
    auto Flags = ID == Intrinsic::riscv_bosc_load ? MachineMemOperand::MOLoad
                                                  : MachineMemOperand::MOStore;
    auto *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(), Flags | MachineMemOperand::MOVolatile,
        LocationSize::beforeOrAfterPointer(), Align(4));
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
  auto Address = [&](const char *Symbol) {
    Register Address = GPR();
    Build(RISCV::PseudoLLA)
        .addReg(Address, RegState::Define)
        .addExternalSymbol(Symbol);
    Register PMPAddress = GPR();
    Build(RISCV::SRLI)
        .addReg(PMPAddress, RegState::Define)
        .addReg(Address)
        .addImm(2);
    return PMPAddress;
  };
  Fence();
  if (Opcode == RISCV::PseudoBOSCDMAWrites) {
    // Match runtime/bosc/runtime/patch7_dma_support.S exactly: TOR regions
    // cover SPM, DMA MMIO and linker-owned mutable DDR; other ranges retain
    // the received platform attributes. This does not provide cache cleanup.
    WriteCSR(0x3b0, Immediate(0x20000000 >> 2));
    WriteCSR(0x3b1, Immediate(0x20420000 >> 2));
    WriteCSR(0x3b2, Immediate(0x60000000 >> 2));
    WriteCSR(0x3b3, Immediate(0x60020000 >> 2));
    WriteCSR(0x3b4, Address("__bosc_data_start"));
    WriteCSR(0x3b5, Address("__bosc_data_end"));
    WriteCSR(0x3b6, Immediate(0x003fffffffffffff));
    WriteCSR(0x3a0, Immediate(0x000f0f0f0f0f0f0f));
    WriteCSR(0xbc0, Immediate(0x0f0f0f0f0f0f0f0f));
    Fence();
  } else if (Opcode == RISCV::PseudoBOSCDMAReads) {
    WriteCSR(0xbc0, Immediate(0x0f0f000f000f000f));
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
  }
  auto Builder =
      BuildMI(*MBB, MI, MI.getDebugLoc(), ST.getInstrInfo()->get(Native));
  if (Native == RISCV::VSETVLI || Native == RISCV::VSETIVLI) {
    Builder.add(MI.getOperand(0)).add(MI.getOperand(1)).addImm(E32M1TaMa);
  } else if (Native == RISCV::VFMV_V_F) {
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
  } else if (Native == RISCV::VFCVT_X_F_V || Native == RISCV::VFCVT_F_X_V) {
    Builder.addReg(RISCV::V0 + MI.getOperand(0).getImm(), RegState::Define)
        .addReg(RISCV::V0 + MI.getOperand(1).getImm())
        .addReg(RISCV::NoRegister);
  } else if (Native == RISCV::VFMAX_VV) {
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
