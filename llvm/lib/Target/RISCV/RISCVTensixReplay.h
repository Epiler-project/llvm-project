//===-- RISCVTensixReplay.h -------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAY_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAY_H
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
class TargetRegisterInfo;
inline constexpr unsigned TensixReplaySlotCount = 32;
// Shared native latency transfer for verified, non-REPLAY/MOP body words.
// Neither templates nor their MC emitter maintain another opcode table.
struct TensixReplayHazardState {
  unsigned SFPU = 0;
  unsigned DstCycles = 0;
  bool operator==(const TensixReplayHazardState &Other) const {
    return SFPU == Other.SFPU && DstCycles == Other.DstCycles;
  }
};
bool needsTensixReplayHazardGap(const MachineInstr &MI,
                               const TargetRegisterInfo &TRI,
                               TensixReplayHazardState State);
TensixReplayHazardState advanceTensixReplayHazard(
    const MachineInstr &MI, const TargetRegisterInfo &TRI,
    TensixReplayHazardState State);
struct TensixSFPUReplayEffects {
  SmallVector<MCRegister, 16> Uses;
  SmallVector<MCRegister, 8> Defs;
  // Exact exit dependency/cooldown from the shared native hazard transfer.
  // A trailing NOP clears the state; an explicit grouped rotate retains it.
  unsigned PendingSFPU = 0;
};
using TensixSFPUReplayExecutionEffects =
    DenseMap<const MachineInstr *, TensixSFPUReplayEffects>;
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

/// Compiler-private native word identity. Registers are physical MC register
/// numbers, not value snapshots, and never become record-only numerical defs.
struct TensixRecordedOperand {
  enum class Kind { Register, Immediate, CapturedScalar };
  Kind OperandKind;
  int64_t Value;
  bool operator==(const TensixRecordedOperand &Other) const {
    return OperandKind == Other.OperandKind && Value == Other.Value;
  }
};
struct TensixRecordedWord {
  unsigned Opcode;
  SmallVector<TensixRecordedOperand, 8> Operands;
  // Query-local identity of the actual instruction which captures a dynamic
  // field. Never dereferenced or retained across machine IR mutation. Equal
  // physical GPR numbers at different recording sites do not mean equal bits.
  const MachineInstr *Capture = nullptr;
  // Symbolic encoding/effect-shape equality, not runtime word-bit equality.
  // One capture site can run repeatedly with different scalar values. Its
  // numerical register/hazard shape stays fixed; clients must not constant-fold
  // captured bits, reuse a former dynamic generation, or reissue its GPR here.
  bool operator==(const TensixRecordedWord &Other) const {
    return Opcode == Other.Opcode && Operands == Other.Operands &&
           Capture == Other.Capture;
  }
};
/// Record-only payloads keep scalar capture/scratch/MMIO, not numerical effects.
MachineInstrBuilder insertTensixRecordedPayload(
    MachineInstr &Before, const TensixRecordedWord &Word,
    const RISCVInstrInfo &TII, std::optional<unsigned> Identity);

struct TensixExplicitReplayRecord {
  const MachineInstr *Header;
  const MachineInstr *End;
  unsigned Stream;
  unsigned Start;
  bool ExecuteWhileLoading;
  SmallVector<const MachineInstr *, 32> Body;
  SmallVector<TensixRecordedWord, 32> Words;
};
/// Physical raw accesses, reconstructed by the same receiver which validates
/// their contents and effects. A recorded range is not a function-long lease.
struct TensixReplayStorageAnalysis {
  struct Range {
    unsigned Stream;
    uint32_t Mask;
  };
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
Expected<TensixRecordedWord> getTensixRecordedWord(const MachineInstr &MI,
                                                   const MachineFunction &MF);
Error verifyTensixRecordedWord(const TensixRecordedWord &Word,
                               const MachineFunction &MF);
/// Return a verified, uninserted native instruction; the caller must release
/// it with MF.deleteMachineInstr. No encoding or latency table is duplicated.
Expected<MachineInstr *>
createTensixRecordedInstruction(const TensixRecordedWord &Word,
                                MachineFunction &MF);
Expected<TensixSFPUReplayEffects>
getTensixExplicitReplayEffects(ArrayRef<TensixRecordedWord> Words,
                               const MachineFunction &MF);
FunctionPass *createRISCVTensixExplicitReplayPass();
void initializeRISCVTensixExplicitReplayPass(PassRegistry &);
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
