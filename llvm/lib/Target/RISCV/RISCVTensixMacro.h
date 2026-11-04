//===-- RISCVTensixMacro.h ------------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXMACRO_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXMACRO_H
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"
namespace llvm {
class IntrinsicInst;
class MachineFunction;
class MachineInstr;
class RISCVSubtarget;
class SelectionDAG;
bool isTensixMacroIntrinsic(Intrinsic::ID ID);
Error verifyTensixMacroIntrinsicFields(const IntrinsicInst &II);
SDValue lowerTensixMacroIntrinsic(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &ST);
bool expandTensixMacroPseudo(MachineInstr &MI, const RISCVSubtarget &ST);
bool isTensixMacroInstruction(unsigned Opcode);
Error verifyTensixMacroMachineFunction(const MachineFunction &MF);
} // namespace llvm
#endif
