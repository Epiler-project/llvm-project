//===-- RISCVTensixExplicitReplay.h ----------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXEXPLICITREPLAY_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXEXPLICITREPLAY_H
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>
#include <optional>
namespace llvm {
class FunctionPass;
class MachineBasicBlock;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;
class RISCVInstrInfo;
class PassRegistry;
class RISCVSubtarget;

/// One complete native issue window with verified aggregate dependencies.
struct TensixIssueBundle {
  const MachineInstr *Bundle;
  const MachineInstr *Header;
  const MachineInstr *End;
  SmallVector<const MachineInstr *, 32> Members;
  bool IsMacro = false;
};
Expected<TensixIssueBundle> getTensixIssueBundle(const MachineInstr &Bundle);
bool isTensixIssueBundleMember(const MachineInstr &MI);
/// True only for an actual member of the currently verified live window.
bool isTensixReplayBundleMember(const MachineInstr &MI);
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
// Reconstructed words are detached MIs. Their owner must supply the actual
// subtarget; neither a parent MBB nor an implicit BH model is available here.
bool needsTensixReplayHazardGap(const MachineInstr &MI,
                                const RISCVSubtarget &ST,
                                TensixReplayHazardState State);
TensixReplayHazardState
advanceTensixReplayHazard(const MachineInstr &MI, const RISCVSubtarget &ST,
                          TensixReplayHazardState State);
struct TensixSFPUReplayEffects {
  SmallVector<MCRegister, 16> Uses;
  SmallVector<MCRegister, 8> Defs;
  // Exact exit dependency/cooldown from the shared native hazard transfer.
  // A trailing NOP clears the state; an explicit grouped rotate retains it.
  unsigned PendingSFPU = 0;
};
// Shared physical predicates and transfer used by the transitional receiver.
// Candidate selection and automatic profitability remain in the legacy owner.
bool isDirectSFPUArithmetic(unsigned Opcode);
bool isDirectSFPURecordedCandidate(const MachineInstr &MI);
bool isOrdinaryRecordedOpcode(unsigned Opcode);
bool isOrdinaryRecordedCandidate(const MachineInstr &MI);
void composeSFPUReplayWord(const MachineInstr &MI, const RISCVSubtarget &ST,
                           TensixSFPUReplayEffects &Effects);

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
/// Record-only payloads keep scalar capture/scratch/MMIO, not numerical
/// effects.
MachineInstrBuilder insertTensixRecordedPayload(
    MachineInstr &Before, const TensixRecordedWord &Word,
    const RISCVInstrInfo &TII, std::optional<unsigned> Identity);

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
struct TensixReplayRange {
  unsigned Stream;
  uint32_t Mask;
};

enum class TensixReplayWriteKind {
  InitializationOnly,
  NativeContents,
};

// Produced by the physical scanner or the legacy fixed-record adapter.
// End markers are unique and belong to the queried MF. NativeContents words
// occupy the contiguous Range beginning at Start; nullopt is ordinary unknown.
struct TensixReplayRecordWrite {
  const MachineInstr *End;
  TensixReplayRange Range;
  unsigned Start;
  TensixReplayWriteKind Kind;
  SmallVector<std::optional<TensixRecordedWord>, 32> Words;
};

enum class TensixTargetStateCallEffect {
  Preserve,
  Invalidate,
};

class TensixReplayBankAnalysis {
  struct SlotContents {
    // A symbolic encoding/effect shape, not captured runtime scalar bits.
    enum class Knowledge { Top, Known, Unknown };
    Knowledge Content = Knowledge::Unknown;
    unsigned Word = 0;
    bool MayContainSFPU = false;

    bool operator==(const SlotContents &Other) const;
    void meet(const SlotContents &Other);
  };

  struct State {
    std::array<uint32_t, 3> Recorded{};
    std::array<std::array<SlotContents, TensixReplaySlotCount>, 3> Contents{};

    bool operator==(const State &Other) const;
    bool operator!=(const State &Other) const;
    static State top();
    void meet(const State &Other);
  };

  // All numeric identities are private to this query. The value equality of
  // TensixRecordedWord retains the actual capture-site MachineInstr identity.
  SmallVector<TensixRecordedWord, 32> Words;
  SmallPtrSet<const MachineBasicBlock *, 16> Reachable;
  DenseMap<const MachineInstr *, State> Observations;
  DenseMap<const MachineInstr *, TensixTargetStateCallEffect> CallEffects;

  unsigned intern(TensixRecordedWord Word);
  const State &before(const MachineInstr &MI) const;

  friend Expected<TensixReplayBankAnalysis>
  analyzeTensixReplayBank(const MachineFunction &MF,
                          ArrayRef<TensixReplayRecordWrite> RecordWrites,
                          ArrayRef<const MachineInstr *> ObservationSites);

public:
  bool isReachable(const MachineBasicBlock &BB) const;
  bool hasObservationBefore(const MachineInstr &MI) const;
  // Observation queries require a published snapshot and Stream < 3.
  // Missing snapshots are a caller contract violation, not Unknown contents.
  bool isRecordedBefore(const MachineInstr &MI, TensixReplayRange Range) const;
  bool mayContainSFPUBefore(const MachineInstr &MI,
                            TensixReplayRange Range) const;
  std::optional<SmallVector<TensixRecordedWord, 32>>
  readKnownWordsBefore(const MachineInstr &MI, TensixReplayRange Range) const;
  // Requires an actual call or inline asm from the analyzed MF. Classification
  // is shared with the legacy MOP transfer and is independent of snapshots.
  TensixTargetStateCallEffect getCallEffect(const MachineInstr &MI) const;
};

// RecordWrites and ObservationSites contain instructions from the same live MF.
// The scanner/adapter requests every execution it validates: ordinary/native,
// automatic and remote REPLAY forms as well as actual MOP executions. It may
// also request record.end to inspect the old bank before its commit.
// Move preserves snapshots; mutation/destruction of the MF invalidates queries.
Expected<TensixReplayBankAnalysis>
analyzeTensixReplayBank(const MachineFunction &MF,
                        ArrayRef<TensixReplayRecordWrite> RecordWrites,
                        ArrayRef<const MachineInstr *> ObservationSites);

bool hasTensixReplayEffects(const MachineInstr &MI, ArrayRef<MCRegister> Uses,
                           ArrayRef<MCRegister> Defs,
                           bool AllowMemory = false);

struct TensixReplayIssue {
  unsigned Opcode;
  unsigned Stream;
  unsigned First;
};
std::optional<TensixReplayIssue> getTensixReplayIssue(const MachineInstr &MI);
Expected<uint32_t> getTensixReplayMask(const MachineInstr &MI, unsigned First);

struct TensixReplayRecordRegion {
  uint32_t Mask;
  unsigned Stream;
  unsigned Length;
  unsigned Count = 0;
  bool AutomaticSFPU = false;
  const MachineInstr *Header = nullptr;
  const MachineInstr *End = nullptr;
  unsigned Start = 0;
  bool ExecuteWhileLoading = false;
  bool HasSFPU = false;
  SmallVector<const MachineInstr *, 32> Body;
  SmallVector<std::optional<TensixRecordedWord>, 32> Words;
};
struct TensixExplicitReplayRecord {
  const MachineInstr *Header;
  const MachineInstr *End;
  unsigned Stream;
  unsigned Start;
  bool ExecuteWhileLoading;
  SmallVector<const MachineInstr *, 32> Body;
  SmallVector<TensixRecordedWord, 32> Words;
};
using TensixSFPUReplayExecutionEffects =
    DenseMap<const MachineInstr *, TensixSFPUReplayEffects>;
struct TensixReplayScan {
  SmallVector<TensixReplayRecordRegion, 4> Regions;
  SmallVector<TensixExplicitReplayRecord, 4> ExplicitRecords;
};
struct TensixNativeReplayAnalysis {
  SmallVector<TensixExplicitReplayRecord, 4> Records;
  DenseMap<const MachineInstr *, SmallVector<TensixRecordedWord, 32>>
      ExecutionWords;
  TensixSFPUReplayExecutionEffects ExecutionEffects;
  DenseMap<const MachineInstr *, TensixReplayRange> RecordWrites;
  DenseMap<const MachineInstr *, TensixReplayRange> Executions;
};
Expected<TensixReplayScan> scanTensixReplay(const MachineFunction &MF,
                                            bool RequireNormalized);
SmallVector<TensixReplayRecordWrite, 4>
getTensixReplayRecordWrites(const TensixReplayScan &Scan);
Expected<TensixNativeReplayAnalysis> analyzeTensixNativeReplay(
    const MachineFunction &MF, const TensixReplayScan &Scan,
    const TensixReplayBankAnalysis &Bank, bool RequireNormalized);
// The standalone physical route does not query templates, selection or MOP.
Expected<TensixNativeReplayAnalysis>
analyzeTensixNativeReplay(const MachineFunction &MF, bool RequireNormalized);

FunctionPass *createRISCVTensixExplicitReplayPass();
void initializeRISCVTensixExplicitReplayPass(PassRegistry &);
FunctionPass *createRISCVTensixReplayBundleUnpackPass();
void initializeRISCVTensixReplayBundleUnpackPass(PassRegistry &);
} // namespace llvm
#endif
