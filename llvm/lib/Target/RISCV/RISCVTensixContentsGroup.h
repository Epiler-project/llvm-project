//===-- RISCVTensixContentsGroup.h -----------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXCONTENTSGROUP_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXCONTENTSGROUP_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"
#include <optional>

namespace llvm {
class IntrinsicInst;
class MachineInstr;
class RISCVSubtarget;
class SelectionDAG;

bool isTensixContentsGroupIntrinsic(Intrinsic::ID ID);
bool isTensixWormholeContentsGroupIntrinsic(Intrinsic::ID ID);
unsigned getTensixContentsGroupResultCount(Intrinsic::ID ID);
// Private projections consume the exact generated intrinsic ABI. Constants
// retain their actual i32 bits; absent entries are numerical/scalar SSA.
Expected<unsigned>
getTensixContentsGroupOpcode(Intrinsic::ID ID,
                             ArrayRef<std::optional<int64_t>> Arguments);
Error verifyTensixContentsGroupFields(
    Intrinsic::ID ID, ArrayRef<std::optional<int64_t>> Arguments,
    bool Wormhole);
Error verifyTensixContentsGroupIntrinsicFields(const IntrinsicInst &II,
                                               bool Wormhole = false);
SDValue lowerTensixContentsGroupIntrinsic(SDValue Op, SelectionDAG &DAG,
                                          const RISCVSubtarget &ST);
/// Check the final paired/index native or scalar-port receiver against its
/// descriptor, memory effects and architecture fields without accepting
/// undefined or repaired old contents.
Error verifyTensixContentsIndexNative(const MachineInstr &MI,
                                      const RISCVSubtarget &ST);
bool expandTensixContentsGroupPseudo(MachineInstr &MI,
                                     const RISCVSubtarget &ST);
} // namespace llvm
#endif
