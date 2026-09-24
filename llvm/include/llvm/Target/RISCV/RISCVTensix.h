//===-- RISCVTensix.h - Public Tensix target contracts -----------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_TARGET_RISCV_RISCVTENSIX_H
#define LLVM_TARGET_RISCV_RISCVTENSIX_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
class Module;
class Triple;
namespace RISCV {

// Logical instruction contracts deliberately expose no bit positions or masks.
struct TensixInstructionInfo {
  StringTable::Offset Name;
  unsigned IntrinsicID;
  unsigned PortIntrinsicID;
  unsigned MopIntrinsicID;
  uint8_t NumFields;
  StringRef getName() const;
};
struct TensixInstructionField {
  unsigned IntrinsicID;
  uint8_t Index;
  StringTable::Offset Name;
  uint32_t MaxValue;
  StringRef getName() const;
};

const TensixInstructionInfo *getTensixInstructionByName(StringRef Name);
const TensixInstructionInfo *getTensixInstructionByIntrinsic(Intrinsic::ID ID);
const TensixInstructionField *
getTensixInstructionField(const TensixInstructionInfo &Info, unsigned Index);
bool isTensixIntrinsic(Intrinsic::ID ID);

// Check width and Tensix feature compatibility without constructing a target
// subtarget whose CLI diagnostics terminate the process.
Expected<bool> verifyTensixTargetFeatures(const Triple &TT, StringRef CPU,
                                        StringRef Features);

// Recoverable preflight for compiler library clients. Function target-features
// must describe the same feature set used to construct the TargetMachine.
// Checks LLVM IR validity and the target's logical field, issue, feature,
// definedness, carrier ABI and condition-stack contracts before code generation.
Error verifyTensixModule(Module &M);

} // namespace RISCV
} // namespace llvm
#endif
