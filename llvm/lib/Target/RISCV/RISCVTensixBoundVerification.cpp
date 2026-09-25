//===-- RISCVTensixBoundVerification.cpp ----------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixBoundLowering.h"
#include "RISCVTensixReplay.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

namespace {

bool isSFPURegister(Register Reg, const MachineRegisterInfo &MRI) {
  if (!Reg)
    return false;
  if (Reg.isPhysical())
    return RISCV::SFPRReadRegClass.contains(Reg) ||
           RISCV::SFPStageRegClass.contains(Reg) ||
           RISCV::SFPStateRegClass.contains(Reg);
  const TargetRegisterClass *RC = MRI.getRegClassOrNull(Reg);
  return RC && (RISCV::SFPRReadRegClass.hasSubClassEq(RC) ||
                RISCV::SFPStageRegClass.hasSubClassEq(RC) ||
                RISCV::SFPStateRegClass.hasSubClassEq(RC));
}

bool isDstPort(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
         MI.getOpcode() == RISCV::PseudoTTSFPSTORE;
}

bool isNativeIssue(const MachineInstr &MI) {
  return RISCV::getTensixEncoding(MI.getOpcode()) || isDstPort(MI) ||
         RISCV::getTensixMachineInfoByPort(MI.getOpcode()) ||
         RISCV::getTensixMachineInfoByMop(MI.getOpcode()) ||
         MI.getOpcode() == RISCV::PseudoTTMOPClear ||
         MI.getOpcode() == RISCV::PseudoTTReplayRecordEnd;
}

} // namespace

// Fixed groups and CC/config/Dst are dependencies, including the inactive
// destination and the non-result clobbers. A subset or an extra undeclared
// effect is not an equivalent description. Compare multiplicities as well as
// membership; a duplicate use cannot stand in for a different required use.
Error llvm::verifyTensixMachineEffects(const MachineInstr &MI) {
  const MCInstrDesc &Desc = MI.getDesc();
  auto Fail = [] {
    return createStringError(
        "bound Tensix SFPU physical effects differ from the native descriptor");
  };
  if (MI.getNumExplicitOperands() != Desc.getNumOperands() ||
      MI.getNumOperands() != Desc.getNumOperands() +
                                 Desc.implicit_uses().size() +
                                 Desc.implicit_defs().size())
    return Fail();
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isInternalRead() || MO.isEarlyClobber() ||
        MO.isRenamable())
      return Fail();
    if (MO.isDef() ? !is_contained(Desc.implicit_defs(), MO.getReg().id())
                   : !is_contained(Desc.implicit_uses(), MO.getReg().id()))
      return Fail();
  }
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isUse() && MO.getReg() == Reg;
        }) != 1)
      return Fail();
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isDef() && MO.getReg() == Reg;
        }) != 1)
      return Fail();
  return Error::success();
}

Error llvm::verifyTensixMachineOperands(const MachineInstr &MI,
                                        const MachineRegisterInfo &MRI,
                                        const TargetRegisterInfo &TRI) {
  const MCInstrDesc &Desc = MI.getDesc();
  if (MI.getNumExplicitOperands() != Desc.getNumOperands())
    return createStringError("bound Tensix SFPU explicit operands differ from "
                             "the native descriptor");
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    int ClassID = Desc.operands()[I].RegClass;
    if (ClassID < 0) {
      if (!MO.isImm())
        return createStringError(
            "bound Tensix SFPU requires logical immediate fields");
      continue;
    }
    if (!MO.isReg() || !MO.getReg())
      return createStringError(
          "bound Tensix SFPU operand does not match its native register class");
    Register Reg = MO.getReg();
    const TargetRegisterClass &Expected = *TRI.getRegClass(ClassID);
    const TargetRegisterClass *Actual =
        Reg.isVirtual() ? MRI.getRegClassOrNull(Reg) : nullptr;
    if (Reg.isVirtual() ? !Actual || !Expected.hasSubClassEq(Actual)
                        : !Expected.contains(Reg))
      return createStringError(
          "bound Tensix SFPU operand does not match its native register class");
    bool EarlyClobber = Desc.getOperandConstraint(I, MCOI::EARLY_CLOBBER) >= 0;
    if (MO.getSubReg() || MO.isUndef() || MO.isInternalRead() ||
        MO.isDef() != (I < Desc.getNumDefs()) ||
        MO.isEarlyClobber() != EarlyClobber)
      return createStringError(
          "bound Tensix SFPU operand does not match its native register role");
    int TiedTo = Desc.getOperandConstraint(I, MCOI::TIED_TO);
    if (TiedTo >= 0 && (!MI.getOperand(TiedTo).isReg() ||
                        MI.getOperand(TiedTo).getReg() != Reg))
      return createStringError("bound Tensix SFPU operands do not satisfy "
                               "their native register tie");
    if (!EarlyClobber)
      continue;
    if (Reg.isPhysical() && TRI.isConstantPhysReg(Reg))
      return createStringError(
          "bound Tensix SFPU scalar scratch must be writable");
    for (unsigned J = 0; J != Desc.getNumOperands(); ++J) {
      const MachineOperand &Other = MI.getOperand(J);
      if (J == I || !Other.isReg() || !Other.getReg())
        continue;
      // A descriptor-tied input is intentionally the same physical object.
      // Every other operand must survive the early scratch definition.
      if (Desc.getOperandConstraint(J, MCOI::TIED_TO) == int(I) ||
          TiedTo == int(J))
        continue;
      Register OtherReg = Other.getReg();
      bool Overlap = Reg.isPhysical() && OtherReg.isPhysical()
                         ? TRI.regsOverlap(Reg, OtherReg)
                         : Reg == OtherReg;
      if (Overlap)
        return createStringError(
            "bound Tensix SFPU early-clobber operand overlaps another operand");
    }
  }
  return Error::success();
}

namespace {

// Dynamic Dst issue has only ordinary scalar scratch. Verify it without
// demanding that GPR allocation already ran, then project the actual physical
// SFPU operands to the same MC contract as static issue. No second encoding
// or SFPU register/tie/mode table is maintained here.
Error verifyDstPort(const MachineInstr &MI, const MachineRegisterInfo &MRI,
                    const RISCVInstrInfo &TII, const RISCVRegisterInfo &TRI) {
  bool Load = MI.getOpcode() == RISCV::PseudoTTSFPLOAD;
  unsigned WordIndex = Load ? 1 : 0;
  unsigned AddressIndex = WordIndex + 1;
  unsigned OffsetIndex = Load ? 4 : 3;
  for (unsigned I : {WordIndex, AddressIndex, OffsetIndex}) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.getReg() || MO.getSubReg() || MO.isUndef() ||
        MO.isInternalRead() || MO.isDef() != (I != OffsetIndex) ||
        MO.isEarlyClobber() != (I != OffsetIndex))
      return createStringError("bound Tensix SFPU requires explicit scalar "
                               "scratch and offset operands");
    Register Reg = MO.getReg();
    const TargetRegisterClass *RC =
        Reg.isVirtual() ? MRI.getRegClassOrNull(Reg) : nullptr;
    if (Reg.isVirtual() ? !RC || !RISCV::GPRRegClass.hasSubClassEq(RC)
                        : !RISCV::GPRRegClass.contains(Reg))
      return createStringError("bound Tensix SFPU port scratch must use GPRs");
  }
  Register Word = MI.getOperand(WordIndex).getReg();
  Register Address = MI.getOperand(AddressIndex).getReg();
  Register Offset = MI.getOperand(OffsetIndex).getReg();
  if (Word == Address || Word == Offset || Address == Offset ||
      Word == RISCV::X0 || Address == RISCV::X0)
    return createStringError(
        "bound Tensix SFPU scalar scratch must preserve all inputs");

  MCInst Native;
  Native.setOpcode(Load ? RISCV::TTSFPLOAD : RISCV::TTSFPSTORE);
  const unsigned DataIndices[] = {Load ? 0u : 2u, 3};
  for (unsigned I : ArrayRef(DataIndices).take_front(Load ? 2 : 1)) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.getReg().isPhysical() ||
        MO.isDef() != (Load && I == 0))
      return createStringError(
          "bound Tensix SFPU requires physical Dst data operands");
    Native.addOperand(MCOperand::createReg(MO.getReg()));
  }
  Native.addOperand(MCOperand::createImm(0));
  for (unsigned I : {OffsetIndex + 1, OffsetIndex + 2}) {
    if (!MI.getOperand(I).isImm())
      return createStringError(
          "bound Tensix SFPU Dst format and addrmod must be immediate");
    Native.addOperand(MCOperand::createImm(MI.getOperand(I).getImm()));
  }
  return RISCV::verifyTensixMCInstruction(Native, TII, TRI);
}

Error verifyNativeOperands(const MachineInstr &MI, const RISCVInstrInfo &TII,
                           const RISCVRegisterInfo &TRI) {
  MCInst Native;
  Native.setOpcode(MI.getOpcode());
  const MCInstrDesc &Desc = MI.getDesc();
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg()) {
      if (!MO.getReg().isPhysical() || MO.isDef() != (I < Desc.getNumDefs()) ||
          MO.isEarlyClobber() !=
              (Desc.getOperandConstraint(I, MCOI::EARLY_CLOBBER) >= 0))
        return createStringError(
            "bound Tensix SFPU requires physical native operand roles");
      Native.addOperand(MCOperand::createReg(MO.getReg()));
    } else if (MO.isImm()) {
      Native.addOperand(MCOperand::createImm(MO.getImm()));
    } else {
      return createStringError("bound Tensix SFPU requires physical registers "
                               "and logical immediate fields");
    }
  }
  return RISCV::verifyTensixMCInstruction(Native, TII, TRI);
}

class RISCVTensixBoundVerification final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixBoundVerification() : MachineFunctionPass(ID) {
    initializeRISCVTensixBoundVerificationPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
    if (!Info->usesBoundTensixSFPU() || Info->hasTensixCodegenFailed())
      return false;
    auto Fail = [&](const Twine &Message) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(
          DiagnosticInfoUnsupported(MF.getFunction(), Message));
      return false;
    };
    const auto &ST = MF.getSubtarget<RISCVSubtarget>();
    if (!ST.hasVendorXTTTensixBH())
      return Fail("bound Tensix SFPU requires +xtttensixbh");
    Attribute Executor = MF.getFunction().getFnAttribute("tensix-executor");
    if (!Executor.isStringAttribute() ||
        (Executor.getValueAsString() != "trisc0" &&
         Executor.getValueAsString() != "trisc1" &&
         Executor.getValueAsString() != "trisc2"))
      return Fail("bound Tensix SFPU requires a TRISC executor");
    const auto &MRI = MF.getRegInfo();
    const auto &TII = *ST.getInstrInfo();
    const auto &TRI = *ST.getRegisterInfo();
    for (const MachineBasicBlock &MBB : MF)
      for (const MachineInstr &MI : MBB.instrs()) {
        if (MI.isDebugInstr())
          continue;
        if (MI.isBundled())
          return Fail("bound Tensix SFPU bundle has no verified issue order");
        if (MI.isCall())
          return Fail("bound Tensix SFPU call has no preservation ABI");
        if (MI.isInlineAsm())
          return Fail(
              "bound Tensix SFPU inline assembly has no preservation ABI");
        bool TouchesSFPU = false;
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isRegMask())
            return Fail(
                "bound Tensix SFPU register mask has no preservation ABI");
          if (!MO.isReg() || !isSFPURegister(MO.getReg(), MRI))
            continue;
          TouchesSFPU = true;
          if (MO.getReg().isVirtual())
            return Fail(
                "bound Tensix SFPU cannot contain virtual SFPU registers");
          if (MO.getSubReg() || MO.isUndef() || MO.isInternalRead() ||
              MO.isRenamable())
            return Fail(
                "bound Tensix SFPU requires defined whole-register operands");
        }
        if (TouchesSFPU && MI.isCopy())
          return Fail(
              "bound Tensix SFPU cannot contain generic register copies");
        if (TouchesSFPU && MI.isPHI())
          return Fail("bound Tensix SFPU cannot contain register PHIs");
        // Automatic replay's variable effects come from its exact recording,
        // independently reconstructed below; they are not the fixed descriptor.
        if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay)
          continue;
        if (!isNativeIssue(MI)) {
          if (TouchesSFPU)
            return Fail("bound Tensix SFPU register is used by an unsupported "
                        "machine operation");
          continue;
        }
        if (Error E = verifyTensixMachineEffects(MI))
          return Fail(toString(std::move(E)));
        if (isDstPort(MI)) {
          if (Error E = verifyDstPort(MI, MRI, TII, TRI))
            return Fail(toString(std::move(E)));
        } else if (RISCV::getTensixEncoding(MI.getOpcode())) {
          if (Error E = verifyNativeOperands(MI, TII, TRI))
            return Fail(toString(std::move(E)));
        }
        if (Error E = verifyTensixMachineOperands(MI, MRI, TRI))
          return Fail(toString(std::move(E)));
        if (RISCV::getTensixMachineInfoByPort(MI.getOpcode()) &&
            MI.getOperand(3).getImm() !=
                static_cast<int64_t>(RISCV::TensixInstructionPort::Local))
          return Fail("bound Tensix SFPU requires the local instruction port");
      }
    if (Error E = verifyTensixReplay(MF))
      return Fail(toString(std::move(E)));
    return false;
  }
};
} // namespace

char RISCVTensixBoundVerification::ID = 0;
INITIALIZE_PASS(RISCVTensixBoundVerification, "riscv-tensix-bound-verify",
                "Verify bound Tensix SFPU physical machine operands", false,
                false)
FunctionPass *llvm::createRISCVTensixBoundVerificationPass() {
  return new RISCVTensixBoundVerification();
}
