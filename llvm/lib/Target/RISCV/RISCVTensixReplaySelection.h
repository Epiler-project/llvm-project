//===-- RISCVTensixReplaySelection.h ----------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAYSELECTION_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAYSELECTION_H
namespace llvm {
class MachineFunction;
bool selectTensixReplay(MachineFunction &MF);
}
#endif
