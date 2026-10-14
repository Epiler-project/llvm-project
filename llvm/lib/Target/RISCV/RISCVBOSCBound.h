//===-- RISCVBOSCBound.h ----------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVBOSCBOUND_H
#define LLVM_LIB_TARGET_RISCV_RISCVBOSCBOUND_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class FunctionPass;
class MachineBasicBlock;
class MachineInstr;
class PassRegistry;
class RISCVSubtarget;
class SelectionDAG;

inline bool isBOSCBoundFunction(const Function &F) {
  return F.hasFnAttribute("trex-bosc-bound-rvv") ||
         F.hasFnAttribute("trex-bosc-bound-ame");
}
FunctionPass *createRISCVBOSCIRVerificationPass(bool Supported = true);
void initializeRISCVBOSCIRVerificationPass(PassRegistry &);
FunctionPass *createRISCVBOSCIssueIngressPass();
void initializeRISCVBOSCIssueIngressPass(PassRegistry &);
FunctionPass *createRISCVBOSCBoundVerificationPass();
void initializeRISCVBOSCBoundVerificationPass(PassRegistry &);
class RISCVBOSCNewPMGate : public RequiredPassInfoMixin<RISCVBOSCNewPMGate> {
public:
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};
SDValue lowerBOSCBoundIntrinsic(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &ST);
MachineBasicBlock *emitBOSCBoundInstruction(MachineInstr &, MachineBasicBlock *,
                                            const RISCVSubtarget &);
} // namespace llvm
#endif
