//===-- RISCVTensixSFPUPhysicalIngress.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixIRVerification.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"

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

class RISCVTensixSFPUPhysicalIngress final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixSFPUPhysicalIngress() : MachineFunctionPass(ID) {
    initializeRISCVTensixSFPUPhysicalIngressPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
    if (!Info->usesTensixSFPU() || Info->hasTensixCodegenFailed())
      return false;

    // Admit virtual numerical contents only from the actual verified ingress.
    // Serialized machine flags alone cannot turn an arbitrary MIR value into a
    // received contents program; fixed/staging state never becomes allocatable.
    bool Contents = !Info->usesBoundTensixSFPU() &&
                    llvm::any_of(instructions(MF.getFunction()),
                                 [](const Instruction &I) {
                                   const auto *II = dyn_cast<IntrinsicInst>(&I);
                                   return II && isTensixContentsIntrinsic(
                                                    II->getIntrinsicID());
                                 });
    const auto &MRI = MF.getRegInfo();
    for (unsigned I = 0; I != MRI.getNumVirtRegs(); ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (MRI.reg_nodbg_empty(Reg) || !isSFPURegister(Reg, MRI))
        continue;
      const auto *RC = MRI.getRegClassOrNull(Reg);
      if (Contents && RC && RISCV::SFPRRegClass.hasSubClassEq(RC))
        continue;

      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
          MF.getFunction(),
          "Tensix SFPU virtual registers require verified contents SSA ingress"));
      return false;
    }
    return false;
  }
};

} // namespace

char RISCVTensixSFPUPhysicalIngress::ID = 0;
INITIALIZE_PASS(
    RISCVTensixSFPUPhysicalIngress, "riscv-tensix-sfpu-physical-ingress",
    "Verify Tensix SFPU physical ingress before register allocation", false,
    false)

FunctionPass *llvm::createRISCVTensixSFPUPhysicalIngressPass() {
  return new RISCVTensixSFPUPhysicalIngress();
}
