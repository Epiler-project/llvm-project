//===-- RISCVTensixNativeLowering.h ---------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXNATIVELOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXNATIVELOWERING_H

#include "llvm/CodeGen/SelectionDAGNodes.h"

namespace llvm {
class RISCVSubtarget;
class SelectionDAG;

SDValue lowerTensixNativeIntrinsic(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &ST);
} // namespace llvm

#endif
