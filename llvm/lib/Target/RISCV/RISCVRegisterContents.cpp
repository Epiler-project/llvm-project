//===-- RISCVRegisterContents.cpp - Register contents allocation
//------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVRegisterContents.h"
#include "RISCVBOSCBound.h"
#include "RISCVRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

bool llvm::isRISCVRegisterContentsClass(const MachineFunction &MF,
                                        const TargetRegisterClass &RC) {
  const Function &F = MF.getFunction();
  return RISCV::SFPRReadRegClass.hasSubClassEq(&RC) ||
         RISCV::SFPRIndexPairRegClass.hasSubClassEq(&RC) ||
         (isBOSCContentsRVVFunction(F) &&
          RISCVRegisterInfo::isRVVRegClass(&RC)) ||
         (isBOSCContentsAMEFunction(F) &&
          (RISCV::BOSCTrRegRegClass.hasSubClassEq(&RC) ||
           RISCV::BOSCAccRegRegClass.hasSubClassEq(&RC)));
}

namespace {
class RISCVRegisterContentsAllocation : public MachineFunctionPass {
public:
  static char ID;
  RISCVRegisterContentsAllocation() : MachineFunctionPass(ID) {
    initializeRISCVRegisterContentsAllocationPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "RISC-V register contents allocation policy";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto &MRI = MF.getRegInfo();
    auto &LIS = getAnalysis<LiveIntervalsWrapperPass>().getLIS();
    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (MRI.reg_nodbg_empty(Reg) ||
          !isRISCVRegisterContentsClass(MF, *MRI.getRegClass(Reg)))
        continue;
      // Use the allocator's existing unspillable interval policy. A rejected
      // allocation must go through its regular failure diagnostic, before an
      // InlineSpiller can request nonexistent target spill/reload instructions.
      LIS.getInterval(Reg).markNotSpillable();
    }
    return false;
  }
};
} // namespace

char RISCVRegisterContentsAllocation::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVRegisterContentsAllocation,
                      "riscv-register-contents-allocation",
                      "RISC-V register contents allocation policy", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(RISCVRegisterContentsAllocation,
                    "riscv-register-contents-allocation",
                    "RISC-V register contents allocation policy", false, false)

FunctionPass *llvm::createRISCVRegisterContentsAllocationPass() {
  return new RISCVRegisterContentsAllocation();
}
