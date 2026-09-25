//===-- RISCVTensixBoundLowering.h -----------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXBOUNDLOWERING_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXBOUNDLOWERING_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"
#include <optional>

namespace llvm {
class IntrinsicInst;
class MachineBasicBlock;
class MachineInstr;
class MachineRegisterInfo;
class TargetRegisterInfo;
class RISCVSubtarget;
class SelectionDAG;

bool isTensixBoundSFPUIntrinsic(Intrinsic::ID ID);

/// Descriptor-owned machine tuple checks, shared by private ingress and final
/// native verification. They do not reconstruct source objects or bindings.
Error verifyTensixMachineEffects(const MachineInstr &MI);
Error verifyTensixMachineOperands(const MachineInstr &MI,
                                  const MachineRegisterInfo &MRI,
                                  const TargetRegisterInfo &TRI);

/// Check each physical operand and its native form's ties, fixed group and
/// logical fields. The caller separately proves a dynamic Dst offset is
/// defined, non-poison and in range, and checks the entire execution context.
Error verifyTensixBoundSFPUIntrinsic(const IntrinsicInst &II);
std::optional<unsigned> getTensixBoundDstOffsetOperand(Intrinsic::ID ID);

/// Produce only a chain, plus ordinary GPR scratch for a dynamic Dst issue.
/// No SFPR DAG result, CopyToReg binding, or numerical temporary is created.
SDValue lowerTensixBoundSFPUIntrinsic(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &ST);

/// FinalizeISel expands the opcode-specific ingress pseudo to actual physical
/// defs/uses before machine optimization and allocation. Returns nullptr for
/// another owner's opcode. It never stages operands or repairs a bad binding.
MachineBasicBlock *emitTensixBoundSFPUInstruction(MachineInstr &MI,
                                                  MachineBasicBlock *MBB,
                                                  const RISCVSubtarget &ST);
} // namespace llvm

#endif
