//===-- RISCVTensixBoundContract.h ------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVTENSIXBOUNDCONTRACT_H
#define LLVM_LIB_TARGET_RISCV_MCTARGETDESC_RISCVTENSIXBOUNDCONTRACT_H

#include "llvm/MC/MCRegister.h"
#include "llvm/Target/RISCV/RISCVTensix.h"
#include <memory>

namespace llvm {
class MCInstrInfo;
class MCRegisterInfo;
namespace RISCV {
// Reuse the target's generated descriptors without global target registration.
std::unique_ptr<MCInstrInfo> createTensixBoundMCInstrInfo();
std::unique_ptr<MCRegisterInfo> createTensixBoundMCRegisterInfo();

// Intrinsic/ingress identity and native projection are shared with lowering.
// These private IDs and operand positions never cross the public query ABI.
struct TensixBoundInstruction {
  Intrinsic::ID ID;
  unsigned Pseudo;
  unsigned ArgumentCount;
  std::optional<unsigned> DstOffset = std::nullopt;
};
const TensixBoundInstruction *getTensixBoundInstruction(Intrinsic::ID ID);
const TensixBoundInstruction *
getTensixBoundInstructionForPseudo(unsigned Opcode);

struct TensixBoundNativeOperand {
  std::optional<unsigned> Argument;
  int64_t Constant = 0;
};
struct TensixBoundNativeInstruction {
  unsigned Opcode;
  SmallVector<TensixBoundNativeOperand, 6> Operands;
};
// Complete validation precedes this projection. A dynamic offset remains an
// argument and requires only ordinary GPR scratch in the machine receiver.
Expected<TensixBoundNativeInstruction>
getTensixBoundNativeInstruction(Intrinsic::ID ID,
                                ArrayRef<TensixBoundValue> Arguments);
MCRegister getTensixSFPURegister(uint32_t Number);
} // namespace RISCV
} // namespace llvm

#endif
