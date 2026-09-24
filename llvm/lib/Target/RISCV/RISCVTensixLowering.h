//===-- RISCVTensixLowering.h -----------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXLOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXLOWERING_H
#include "llvm/CodeGen/SelectionDAGNodes.h"
namespace llvm {
class SelectionDAG;
class RISCVSubtarget;
SDValue lowerTensixSFPUIntrinsic(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &ST);
SDValue lowerTensixOrdinaryIntrinsic(SDValue Op, SelectionDAG &DAG,
                                    const RISCVSubtarget &ST);
} // namespace llvm
#endif
