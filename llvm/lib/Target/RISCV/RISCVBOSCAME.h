//===-- RISCVBOSCAME.h -------------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVBOSCAME_H
#define LLVM_LIB_TARGET_RISCV_RISCVBOSCAME_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Intrinsics.h"

namespace llvm {
class Function;
class MachineFunction;
class MachineBasicBlock;
class MachineInstr;
class RISCVSubtarget;
class SelectionDAG;
bool isBOSCAMEIntrinsic(Intrinsic::ID);
bool isBOSCAMEOpcode(unsigned);
void verifyBOSCAMEIR(Function &);
void verifyBOSCAMEMachine(const MachineFunction &);
SDValue lowerBOSCAMEIntrinsic(SDValue, SelectionDAG &, const RISCVSubtarget &);
MachineBasicBlock *emitBOSCAMEInstruction(MachineInstr &, MachineBasicBlock *,
                                         const RISCVSubtarget &);
} // namespace llvm
#endif
