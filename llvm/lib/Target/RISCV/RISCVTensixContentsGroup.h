//===-- RISCVTensixContentsGroup.h -----------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXCONTENTSGROUP_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXCONTENTSGROUP_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"

namespace llvm {
class IntrinsicInst;
class MachineInstr;
class RISCVSubtarget;
class SelectionDAG;

bool isTensixContentsGroupIntrinsic(Intrinsic::ID ID);
unsigned getTensixContentsGroupResultCount(Intrinsic::ID ID);
Error verifyTensixContentsGroupIntrinsicFields(const IntrinsicInst &II);
SDValue lowerTensixContentsGroupIntrinsic(SDValue Op, SelectionDAG &DAG,
                                          const RISCVSubtarget &ST);
bool expandTensixContentsGroupPseudo(MachineInstr &MI,
                                     const RISCVSubtarget &ST);
} // namespace llvm
#endif
