//===-- RISCVTensixReplay.h -------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAY_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAY_H
#include "RISCVTensixExplicitReplay.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <optional>
namespace llvm {
class FunctionPass;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;
class RISCVInstrInfo;
class PassRegistry;
class RISCVSubtarget;
/// A finite native pipeline transfer. Entry 0 is the empty state, entries
/// 1..32 are singleton SFPU dependencies, and entries 33..35 are Dst delays.
/// This represents a whole hardware loop without copying its instruction body.
struct TensixMOPHazardTransfer {
  struct Result {
    TensixReplayHazardState State;
    bool RequiresGap = false;
    bool operator==(const Result &Other) const {
      return State == Other.State && RequiresGap == Other.RequiresGap;
    }
  };
  std::array<Result, 36> Inputs;
  Result apply(TensixReplayHazardState State) const;
  bool operator==(const TensixMOPHazardTransfer &Other) const {
    return Inputs == Other.Inputs;
  }
};
struct TensixMOPExecutionEffects {
  TensixSFPUReplayEffects Physical;
  TensixMOPHazardTransfer Hazards;
};
using TensixMOPExecutionAnalysis =
    DenseMap<const MachineInstr *, TensixMOPExecutionEffects>;
bool isTensixSFPUReplayCandidate(const MachineInstr &MI);
Expected<TensixSFPUReplayEffects>
getTensixSFPUReplayEffects(ArrayRef<const MachineInstr *> Body);

/// Physical raw accesses, reconstructed by the same receiver which validates
/// their contents and effects. A recorded range is not a function-long lease.
struct TensixReplayStorageAnalysis {
  using Range = TensixReplayRange;
  struct MOPWrite {
    unsigned Slot;
    uint32_t Mask;
  };
  DenseMap<const MachineInstr *, Range> RecordWrites;
  DenseMap<const MachineInstr *, Range> Executions;
  DenseMap<const MachineInstr *, MOPWrite> MOPWrites;
  // Reaching raw references at each actual MOP execution.
  // Symbolic MOP bindings kill the former raw reference; their identities are
  // transported separately by the template receiver.
  DenseMap<const MachineInstr *, std::array<uint32_t, 7>> MOPExecutions;
};
/// Query-local products reconstructed from actual CFG and instructions. Neither
/// these instruction pointers nor word identities survive machine IR mutation.
struct TensixExplicitReplayAnalysis {
  SmallVector<TensixExplicitReplayRecord, 4> Records;
  DenseMap<const MachineInstr *, SmallVector<TensixRecordedWord, 32>>
      ExecutionWords;
  TensixSFPUReplayExecutionEffects ExecutionEffects;
  TensixMOPExecutionAnalysis MOPExecutions;
  TensixReplayStorageAnalysis Storage;
};
Expected<TensixExplicitReplayAnalysis>
analyzeTensixExplicitReplay(const MachineFunction &MF);
Error verifyTensixReplay(
    const MachineFunction &MF,
    TensixSFPUReplayExecutionEffects *ExecutionEffects = nullptr,
    TensixMOPExecutionAnalysis *MOPExecutions = nullptr);
/// Exact variable physical effects, independently reconstructed from the
/// current MOP controls and replay words. Native operand legality is separate.
Error verifyTensixMOPExecutionEffects(
    const MachineInstr &MI, const TensixMOPExecutionEffects &Effects);
} // namespace llvm
#endif
