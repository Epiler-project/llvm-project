//===-- RISCVTensixRegisterAllocation.cpp ---------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;
namespace {
class RISCVTensixNoSpill : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixNoSpill() : MachineFunctionPass(ID) {
    initializeRISCVTensixNoSpillPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MRI = MF.getRegInfo();
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    for (unsigned I = 0; I != MRI.getNumVirtRegs(); ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (!MRI.reg_nodbg_empty(Reg) &&
          RISCV::SFPRRegClass.hasSubClassEq(MRI.getRegClass(Reg)))
        LIS.getInterval(Reg).markNotSpillable();
    }
    return false;
  }
};
class RISCVTensixAllocated : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixAllocated() : MachineFunctionPass(ID) {
    initializeRISCVTensixAllocatedPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (MF.getProperties().hasFailedRegAlloc()) {
      // RegAllocBase has already emitted DiagnosticInfoRegAllocFailure. Library
      // clients capture it and discard the artifact; do not terminate them.
      MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
      return false;
    }
    const auto &MRI = MF.getRegInfo();
    for (unsigned I = 0; I != MRI.getNumVirtRegs(); ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (!MRI.reg_nodbg_empty(Reg) &&
          RISCV::SFPRRegClass.hasSubClassEq(MRI.getRegClass(Reg))) {
        MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
        MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
            MF.getFunction(), "unallocated Tensix SFPU virtual register"));
        return false;
      }
    }
    return false;
  }
};
} // namespace
char RISCVTensixNoSpill::ID = 0;
char RISCVTensixAllocated::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVTensixNoSpill, "riscv-tensix-no-spill",
                      "Mark Tensix SFPU intervals nonspillable", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(RISCVTensixNoSpill, "riscv-tensix-no-spill",
                    "Mark Tensix SFPU intervals nonspillable", false, false)
INITIALIZE_PASS(RISCVTensixAllocated, "riscv-tensix-allocated",
                "Verify Tensix SFPU register allocation", false, false)
FunctionPass *llvm::createRISCVTensixNoSpillPass() {
  return new RISCVTensixNoSpill();
}
FunctionPass *llvm::createRISCVTensixAllocatedPass() {
  return new RISCVTensixAllocated();
}
