//===-- RISCVRegisterContents.h - Register contents allocation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_RISCV_RISCVREGISTERCONTENTS_H
#define LLVM_LIB_TARGET_RISCV_RISCVREGISTERCONTENTS_H

#include "llvm/CodeGen/TargetRegisterInfo.h"

namespace llvm {
class FunctionPass;
class MachineFunction;
class PassRegistry;

bool isRISCVRegisterContentsClass(const MachineFunction &MF,
                                  const TargetRegisterClass &RC);
FunctionPass *createRISCVRegisterContentsAllocationPass();
void initializeRISCVRegisterContentsAllocationPass(PassRegistry &);
} // namespace llvm

#endif
