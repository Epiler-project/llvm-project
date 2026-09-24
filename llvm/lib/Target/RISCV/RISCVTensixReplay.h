//===-- RISCVTensixReplay.h -------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAY_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAY_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Error.h"
namespace llvm {
class MachineFunction;
class MachineInstr;
struct TensixSFPUReplayEffects {
  SmallVector<MCRegister, 16> Uses;
  SmallVector<MCRegister, 8> Defs;
};
bool isTensixSFPUReplayCandidate(const MachineInstr &MI);
Expected<TensixSFPUReplayEffects>
getTensixSFPUReplayEffects(ArrayRef<const MachineInstr *> Body);
Error verifyTensixReplay(const MachineFunction &MF);
}
#endif
