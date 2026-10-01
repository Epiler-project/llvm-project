//===-- RISCVTensixReplayTemplate.h --------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAYTEMPLATE_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXREPLAYTEMPLATE_H

#include "RISCVTensixReplay.h"
#include <cstdint>

namespace llvm {
class FunctionPass;
class MachineFunction;
class PassRegistry;

enum class TensixReplayTemplateMode : uint8_t {
  RecordOnly = 0,
  RecordAndExecute = 1,
};

enum class TensixReplayTemplatePlacement : uint8_t {
  Automatic = 0,
  Fixed = 1,
};

/// Query-local physical view of authored fixed recordings. Words include the
/// canonical internal hazard repair, so raw slice effects agree before and
/// after finalization. Automatic templates never implicitly supply raw reads.
struct TensixReplayTemplateStorage {
  struct FixedRecord {
    const MachineInstr *Begin;
    const MachineInstr *End;
    unsigned Start;
    SmallVector<TensixRecordedWord, 16> Words;
  };
  SmallVector<FixedRecord, 4> FixedRecords;
  // Symbolic consumers do not require a physical placement. Both automatic
  // and fixed templates expose the same finalized native words to the unique
  // MOP execution receiver; they never supply an automatic raw range read.
  DenseMap<unsigned, SmallVector<TensixRecordedWord, 16>> SymbolicWords;
};
Expected<TensixReplayTemplateStorage>
getTensixReplayTemplateStorage(const MachineFunction &MF, bool RequirePrepared);

/// Independently verify prepared templates and execution effects without
/// normalizing instructions, repairing hazards, or assigning physical slots.
Error verifyTensixReplayTemplates(MachineFunction &MF);

/// Normalize record-time versus execute-time physical effects immediately
/// after instruction selection, without assigning a length or physical slots.
FunctionPass *createRISCVTensixReplayTemplatePreparePass();
void initializeRISCVTensixReplayTemplatePreparePass(PassRegistry &);

/// Repair internal native hazards, count words, and place live templates before
/// the ordinary final hazard repair. Never duplicate a source iteration.
FunctionPass *createRISCVTensixReplayTemplateFinalizePass();
void initializeRISCVTensixReplayTemplateFinalizePass(PassRegistry &);
} // namespace llvm

#endif
