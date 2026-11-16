//===-- RISCVTensixReplay.cpp -----------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixReplay.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixBoundLowering.h"
#include "RISCVTensixReplayTemplate.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/TensixMOPSequence.h"
#include <array>
#include <limits>
#include <optional>

using namespace llvm;

bool llvm::isTensixSFPUReplayCandidate(const MachineInstr &MI) {
  return isDirectSFPUArithmetic(MI.getOpcode()) &&
         isDirectSFPURecordedCandidate(MI);
}

Expected<TensixSFPUReplayEffects>
llvm::getTensixSFPUReplayEffects(ArrayRef<const MachineInstr *> Body) {
  if (Body.empty() || Body.size() > TensixReplaySlotCount)
    return createStringError(
        "automatic SFPU replay body must fit 32 instruction slots");
  TensixSFPUReplayEffects Effects;
  bool HasValue = false;
  for (const MachineInstr *MI : Body) {
    if (!isTensixSFPUReplayCandidate(*MI))
      return createStringError(
          "automatic SFPU replay body has unsupported physical effects");
    composeSFPUReplayWord(
        *MI, MI->getMF()->getSubtarget<RISCVSubtarget>(), Effects);
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isDef())
        HasValue |= RISCV::SFPRRegClass.contains(MO.getReg());
  }
  if (!HasValue)
    return createStringError(
        "automatic SFPU replay requires an arithmetic value sequence");
  llvm::sort(Effects.Uses);
  llvm::sort(Effects.Defs);
  return Effects;
}

namespace {
struct Inventory {
  struct MOPProgram {
    uint32_t Range = 0;
    SmallVector<unsigned, 16> Words;
    // Only the slot instruction opcode is relevant to MOP's NOP special case;
    // a REPLAY of a NOP is still a REPLAY, and must not be skipped.
    bool IsNOP = false;
    bool operator==(const MOPProgram &Other) const {
      return Range == Other.Range && Words == Other.Words &&
             IsNOP == Other.IsNOP;
    }
  };
  SmallVector<TensixRecordedWord, 32> Words;
  SmallVector<MOPProgram, 8> MOPPrograms;
  DenseMap<unsigned, SmallVector<unsigned, 16>> SymbolicWords;
  TensixExplicitReplayAnalysis Analysis;

  unsigned intern(TensixRecordedWord Word) {
    // Intern a query-local symbolic shape for effects/hazards, never a scalar
    // value snapshot. Distinct capture sites cannot share this identity.
    auto Found = llvm::find(Words, Word);
    if (Found != Words.end())
      return unsigned(Found - Words.begin());
    Words.push_back(std::move(Word));
    return Words.size() - 1;
  }
  unsigned internMOP(MOPProgram Program) {
    auto Found = llvm::find(MOPPrograms, Program);
    if (Found != MOPPrograms.end())
      return unsigned(Found - MOPPrograms.begin());
    MOPPrograms.push_back(std::move(Program));
    return MOPPrograms.size() - 1;
  }
};

Error verifyAutomaticOwner(const MachineFunction &MF) {
  bool HasAutomatic = false;
  bool HasExternalOwner = false;
  bool HasVirtual = false;
  for (const MachineBasicBlock &MBB : MF)
    for (const MachineInstr &MI : MBB) {
      HasAutomatic |= MI.getOpcode() == RISCV::PseudoTTSFPUReplay;
      HasVirtual |= any_of(MI.operands(), [](const MachineOperand &MO) {
        return MO.isReg() && MO.getReg().isVirtual();
      });
      HasExternalOwner |= MI.isCall() || MI.isInlineAsm() ||
                          RISCV::getTensixMachineInfoByPort(MI.getOpcode()) ||
                          RISCV::getTensixMachineInfoByMop(MI.getOpcode()) ||
                          MI.getOpcode() == RISCV::TTREPLAY ||
                          MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay ||
                          MI.getOpcode() == RISCV::PseudoTTContentsSFPUReplay ||
                          MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
                          MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord ||
                          MI.getOpcode() == RISCV::TTMOP ||
                          MI.getOpcode() == RISCV::TTMOP_CFG ||
                          MI.getOpcode() == RISCV::PseudoTTMOPControlWrite ||
                          MI.getOpcode() == RISCV::PseudoTTMOPControlWriteImm ||
                          MI.getOpcode() == RISCV::PseudoTTMOPClear ||
                          MI.getOpcode() == RISCV::PseudoTTReplayTemplateBegin;
    }
  if (HasAutomatic && HasExternalOwner)
    return createStringError(
        "automatic SFPU replay conflicts with another replay or call owner");
  if (HasAutomatic && HasVirtual)
    return createStringError(
        "automatic SFPU replay requires final physical register allocation");
  return Error::success();
}

// Recording is a finite straight-line issue region. Scalar address arithmetic
// may be interspersed, but no scalar control transfer may interrupt it. End
// markers survive instruction selection and scheduling until this final check.
Error receiveAutomaticReplay(const MachineFunction &MF,
                             const TensixReplayScan &Scan,
                             TensixSFPUReplayExecutionEffects &Executions) {
  DenseMap<const MachineInstr *, const TensixReplayRecordRegion *> Headers,
      Ends;
  for (const auto &Region : Scan.Regions) {
    Headers.try_emplace(Region.Header, &Region);
    Ends.try_emplace(Region.End, &Region);
  }
  bool SeenAutomatic = false;
  for (const auto &BB : MF) {
    const TensixReplayRecordRegion *Active = nullptr;
    const TensixReplayRecordRegion *Automatic = nullptr;
    std::optional<TensixSFPUReplayEffects> AutomaticEffects;
    for (const auto &MI : BB) {
      if (auto End = Ends.find(&MI); End != Ends.end()) {
        if (End->second->AutomaticSFPU) {
          auto Effects = getTensixSFPUReplayEffects(End->second->Body);
          if (!Effects)
            return Effects.takeError();
          Automatic = End->second;
          AutomaticEffects = std::move(*Effects);
          Active = nullptr;
        }
        continue;
      }
      auto Issued = getTensixReplayIssue(MI);
      if (!Issued) {
        if (Active)
          return createStringError("automatic SFPU replay recording must be "
                                   "uninterrupted native issue");
        // A completed automatic template cannot be reused after an observable
        // scalar/control boundary. Its bit contents may persist, but that is
        // outside the single-block candidate contract proved by selection.
        Automatic = nullptr;
        AutomaticEffects.reset();
        continue;
      }
      if (Issued->Opcode == RISCV::TTREPLAY) {
        if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay) {
          auto Mask = getTensixReplayMask(MI, Issued->First);
          if (!Mask)
            return Mask.takeError();
          bool Load = MI.getOperand(Issued->First).getImm() != 0;
          if (!MI.getOperand(1).isImm() ||
              MI.getOperand(0).getImm() != (Load ? 1 : 0) ||
              MI.getOperand(1).getImm() != (Load ? 1 : 0) ||
              MI.getOperand(3).getImm() != 0)
            return createStringError(
                "automatic SFPU replay requires static record-and-execute or "
                "execute in bank zero");
          if (Load) {
            if (SeenAutomatic)
              return createStringError(
                  "automatic SFPU replay permits one record per function");
            SeenAutomatic = true;
            SmallVector<MCRegister, 2> Uses{RISCV::TT_CONFIG, RISCV::TT_ISSUE};
            SmallVector<MCRegister, 1> Defs{RISCV::TT_ISSUE};
            llvm::sort(Uses);
            if (!hasTensixReplayEffects(MI, Uses, Defs))
              return createStringError("automatic SFPU replay record header "
                                       "has unexpected physical effects");
            Active = Headers.lookup(&MI);
            assert(Active && "automatic record missing from physical scan");
          } else {
            if (!Automatic || Automatic->Mask != *Mask || !AutomaticEffects)
              return createStringError(
                  "automatic SFPU replay execution requires its exact "
                  "preceding local record");
            if (!hasTensixReplayEffects(MI, AutomaticEffects->Uses,
                                        AutomaticEffects->Defs))
              return createStringError(
                  "automatic SFPU replay execution physical effects differ "
                  "from its recording");
            Executions.try_emplace(&MI, *AutomaticEffects);
          }
        }
        continue;
      }
      if (Automatic && !isTensixSFPUReplayCandidate(MI)) {
        Automatic = nullptr;
        AutomaticEffects.reset();
      }
      if (Active && !isTensixSFPUReplayCandidate(MI))
        return createStringError(
            "automatic SFPU replay body has unsupported physical effects");
    }
  }
  return Error::success();
}

// A dynamic control is captured by the write, not read again from the GPR at
// MOP execution. Its domain is the architectural field width. Missing differs
// from a received dynamic value and is never replaced by a firmware default.
struct MOPControlValue {
  enum class Knowledge { Top, Missing, Constant, Dynamic };
  Knowledge Content = Knowledge::Missing;
  uint32_t Value = 0;
  bool operator==(const MOPControlValue &Other) const {
    return Content == Other.Content &&
           (Content != Knowledge::Constant || Value == Other.Value);
  }
  void meet(const MOPControlValue &Other) {
    if (Other.Content == Knowledge::Top)
      return;
    if (Content == Knowledge::Top) {
      *this = Other;
      return;
    }
    if (Content == Knowledge::Missing || Other.Content == Knowledge::Missing)
      Content = Knowledge::Missing;
    else if (!(*this == Other))
      Content = Knowledge::Dynamic;
  }

};

struct MOPState {
  uint8_t MOPKnown = 0;
  std::array<uint32_t, 7> MOPRequires{};
  std::array<SmallVector<unsigned, 2>, 7> MOPPrograms;
  std::array<MOPControlValue, 2> MOPControls;
  MOPControlValue MOPMaskHi;
  bool operator==(const MOPState &Other) const {
    return MOPKnown == Other.MOPKnown && MOPRequires == Other.MOPRequires &&
           MOPPrograms == Other.MOPPrograms && MOPControls == Other.MOPControls &&
           MOPMaskHi == Other.MOPMaskHi;
  }
  bool operator!=(const MOPState &Other) const { return !(*this == Other); }

  static MOPState top() {
    MOPState S;
    S.MOPKnown = 0x7f;
    for (auto &Control : S.MOPControls)
      Control.Content = MOPControlValue::Knowledge::Top;
    S.MOPMaskHi.Content = MOPControlValue::Knowledge::Top;
    return S;
  }
  void meet(const MOPState &Other) {
    MOPKnown &= Other.MOPKnown;
    // Either path's reference can execute. All possible referenced entries
    // must have been recorded on every incoming path.
    for (unsigned I = 0; I != 7; ++I) {
      MOPRequires[I] |= Other.MOPRequires[I];
      for (unsigned Program : Other.MOPPrograms[I])
        if (!is_contained(MOPPrograms[I], Program))
          MOPPrograms[I].push_back(Program);
      llvm::sort(MOPPrograms[I]);
    }
    for (unsigned I = 0; I != 2; ++I)
      MOPControls[I].meet(Other.MOPControls[I]);
    MOPMaskHi.meet(Other.MOPMaskHi);
  }
};

TensixReplayHazardState hazardBasis(unsigned Index) {
  if (!Index)
    return {};
  if (Index <= 32)
    return {1u << (Index - 1), 0};
  return {0, Index - 32};
}

void joinHazard(TensixMOPHazardTransfer::Result &Into,
                const TensixMOPHazardTransfer::Result &Other) {
  Into.State.SFPU |= Other.State.SFPU;
  Into.State.DstCycles = std::max(Into.State.DstCycles, Other.State.DstCycles);
  Into.RequiresGap |= Other.RequiresGap;
}

using PhysicalRegisters = SmallVector<MCRegister, 16>;
void addRegisters(PhysicalRegisters &Into, ArrayRef<MCRegister> Other) {
  for (MCRegister Reg : Other)
    if (!is_contained(Into, Reg))
      Into.push_back(Reg);
}

// Transfer algebra for the finite MOP control graph. Sequence, choice and a
// hardware-loop edge share slot summaries; none copies an instruction body or
// enumerates a source/hardware iteration. SFPU lane predicates remain in the
// canonical native operands, independently of MOP control-flow predicates.
struct MOPSummary {
  PhysicalRegisters Reads, MayDef, MustDef;
  TensixMOPHazardTransfer Hazards;
  MOPSummary() {
    for (unsigned I = 0; I != Hazards.Inputs.size(); ++I)
      Hazards.Inputs[I].State = hazardBasis(I);
  }
};

MOPSummary sequence(const MOPSummary &A, const MOPSummary &B) {
  MOPSummary Result = A;
  for (MCRegister Reg : B.Reads)
    if (!is_contained(A.MustDef, Reg) && !is_contained(Result.Reads, Reg))
      Result.Reads.push_back(Reg);
  addRegisters(Result.MayDef, B.MayDef);
  addRegisters(Result.MustDef, B.MustDef);
  for (unsigned I = 0; I != Result.Hazards.Inputs.size(); ++I) {
    auto Next = B.Hazards.apply(A.Hazards.Inputs[I].State);
    Next.RequiresGap |= A.Hazards.Inputs[I].RequiresGap;
    Result.Hazards.Inputs[I] = Next;
  }
  return Result;
}

MOPSummary choice(const MOPSummary &A, const MOPSummary &B) {
  MOPSummary Result = A;
  addRegisters(Result.Reads, B.Reads);
  addRegisters(Result.MayDef, B.MayDef);
  llvm::erase_if(Result.MustDef,
                 [&](MCRegister Reg) { return !is_contained(B.MustDef, Reg); });
  for (unsigned I = 0; I != Result.Hazards.Inputs.size(); ++I)
    joinHazard(Result.Hazards.Inputs[I], B.Hazards.Inputs[I]);
  return Result;
}

MOPSummary repeat(MOPSummary Body, unsigned Count) {
  MOPSummary Result;
  // Exact bounded-loop transfer by squaring the finite relation, not by
  // expansion into Count instruction sequences or per-iteration CFG nodes.
  while (Count) {
    if (Count & 1)
      Result = sequence(Result, Body);
    Count >>= 1;
    if (Count)
      Body = sequence(Body, Body);
  }
  return Result;
}

MOPSummary repeatRange(const MOPSummary &Body, unsigned Minimum,
                       unsigned Maximum) {
  assert(Minimum <= Maximum);
  if (Minimum == Maximum)
    return repeat(Body, Minimum);
  // Summarize I | Body | ... | Body^N by squaring finite transfer relations.
  // This folds a CountedRepeat edge; it does not enumerate its iterations or
  // interpret any hardware count/flag field a second time.
  uint64_t Length = uint64_t(Maximum) - Minimum + 1;
  MOPSummary Power = Body, Prefix, AccumulatedPower;
  std::optional<MOPSummary> AccumulatedPrefix;
  while (Length) {
    if (Length & 1) {
      MOPSummary Chunk = sequence(AccumulatedPower, Prefix);
      AccumulatedPrefix = AccumulatedPrefix
                              ? choice(*AccumulatedPrefix, Chunk)
                              : std::move(Chunk);
      AccumulatedPower = sequence(AccumulatedPower, Power);
    }
    Length >>= 1;
    if (Length) {
      Prefix = choice(Prefix, sequence(Power, Prefix));
      Power = sequence(Power, Power);
    }
  }
  return sequence(repeat(Body, Minimum), *AccumulatedPrefix);
}

Expected<MOPSummary> summarizeMOPWord(const TensixRecordedWord &Word,
                                      MachineFunction &MF) {
  auto Native = createTensixRecordedInstruction(Word, MF);
  if (!Native)
    return Native.takeError();
  MOPSummary Result;
  for (const MachineOperand &MO : (*Native)->operands())
    if (MO.isReg()) {
      MCRegister Reg = MO.getReg().asMCReg();
      auto &Set = MO.isDef() ? Result.MayDef : Result.Reads;
      if (!is_contained(Set, Reg))
        Set.push_back(Reg);
    }
  Result.MustDef = Result.MayDef;
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  for (unsigned I = 0; I != Result.Hazards.Inputs.size(); ++I) {
    TensixReplayHazardState Input = hazardBasis(I);
    Result.Hazards.Inputs[I] = {
        advanceTensixReplayHazard(**Native, ST, Input),
        needsTensixReplayHazardGap(**Native, ST, Input)};
  }
  MF.deleteMachineInstr(*Native);
  return Result;
}

// Machine-specific interpretation of the shared value-only sequencer graph.
// Recorded-range admission, physical uses/defs and hazard transfer remain here;
// count/flag/mask/slot selection belongs exclusively to TensixMOPSequence.
class MOPSequenceEffects {
  Inventory &Records;
  MachineFunction &MF;
  const TensixReplayBankAnalysis &Bank;
  const MachineInstr &Execution;

  Expected<MOPSummary> program(const Inventory::MOPProgram &Program) {
    SmallVector<TensixRecordedWord, 32> Words;
    for (unsigned Index : Program.Words)
      Words.push_back(Records.Words[Index]);
    if (Program.Range) {
      TensixReplayRange Range{0, Program.Range};
      if (!Bank.isRecordedBefore(Execution, Range))
        return createStringError("Tensix MOP replay reference is not "
                                 "recorded on every incoming path");
      auto Reaching = Bank.readKnownWordsBefore(Execution, Range);
      if (!Reaching)
        return createStringError("SFPU MOP execution has differing or unknown "
                                 "replay words on incoming paths");
      llvm::append_range(Words, *Reaching);
    }
    MOPSummary Result;
    for (const auto &NativeWord : Words) {
      auto Word = summarizeMOPWord(NativeWord, MF);
      if (!Word)
        return Word.takeError();
      Result = sequence(Result, *Word);
    }
    return Result;
  }

public:
  MOPSequenceEffects(Inventory &Records, MachineFunction &MF,
                     const TensixReplayBankAnalysis &Bank,
                     const MachineInstr &Execution)
      : Records(Records), MF(MF), Bank(Bank), Execution(Execution) {}

  Expected<MOPSummary> summarize(const TensixMOPSequence &Graph) {
    // Only executed graph leaves require actual recorded words. For example,
    // Inner=1 may use slot6 to select a mode without executing its instruction.
    BitVector Reachable(Graph.Nodes.size());
    SmallVector<unsigned> Pending{Graph.Root};
    while (!Pending.empty()) {
      unsigned Index = Pending.pop_back_val();
      if (Reachable.test(Index))
        continue;
      Reachable.set(Index);
      append_range(Pending, Graph.Nodes[Index].Children);
    }
    SmallVector<std::optional<MOPSummary>, 0> Summaries(Graph.Nodes.size());
    for (auto [Index, Node] : enumerate(Graph.Nodes)) {
      if (!Reachable.test(Index))
        continue;
      MOPSummary Result;
      switch (Node.Form) {
      case TensixMOPSequenceNode::Kind::Exit:
        break;
      case TensixMOPSequenceNode::Kind::SlotRef: {
        auto Summary = program(Records.MOPPrograms[Node.Declaration.Identity]);
        if (!Summary)
          return Summary.takeError();
        Result = std::move(*Summary);
        break;
      }
      case TensixMOPSequenceNode::Kind::Sequence:
        for (unsigned Child : Node.Children)
          Result = sequence(Result, *Summaries[Child]);
        break;
      case TensixMOPSequenceNode::Kind::Choice:
        Result = *Summaries[Node.Children.front()];
        for (unsigned Child : drop_begin(Node.Children))
          Result = choice(Result, *Summaries[Child]);
        break;
      case TensixMOPSequenceNode::Kind::CountedRepeat:
        Result = repeatRange(*Summaries[Node.Children.front()],
                             Node.Count.Minimum, Node.Count.Maximum);
        break;
      }
      Summaries[Index] = std::move(Result);
    }
    return std::move(*Summaries[Graph.Root]);
  }
};

TensixMOPControlValue projectControl(const MOPControlValue &Control) {
  switch (Control.Content) {
  case MOPControlValue::Knowledge::Constant:
    return TensixMOPControlValue::constant(Control.Value);
  case MOPControlValue::Knowledge::Dynamic:
    return TensixMOPControlValue::dynamic();
  case MOPControlValue::Knowledge::Top:
  case MOPControlValue::Knowledge::Missing:
    return {};
  }
  llvm_unreachable("invalid MOP control knowledge");
}

Error describeSequenceError(Error E, unsigned Kind, const MOPState &S) {
  // Preserve established receiver diagnostics while letting the shared query
  // own all sequencer legality decisions.
  return handleErrors(std::move(E), [&](const TensixMOPSequenceError &Error) {
    switch (Error.getReason()) {
    case TensixMOPSequenceFailure::MissingControl:
      if (Kind == 1)
        return createStringError("SFPU MOP execution requires typed control-cell "
                                 "0 and 1 writes on every incoming path");
      if (projectControl(S.MOPControls[1]).Knowledge ==
          TensixMOPControlValue::Kind::Missing)
        return createStringError("SFPU MOP execution requires a typed control-cell "
                                 "1 write on every incoming path");
      return createStringError("SFPU MOP execution requires a received MaskHi "
                               "for selected high mask bits");
    case TensixMOPSequenceFailure::UnverifiedZeroInner:
      return createStringError("SFPU MOP zero-inner NOP-start with non-NOP end "
                               "uses an unverified hardware corner");
    default:
      return createStringError(Error.message());
    }
  });
}

bool mopMayContainSFPU(const MachineInstr &MI, const MOPState &S,
                       const Inventory &Records,
                       const TensixReplayBankAnalysis &Bank) {
  for (const auto &Programs : S.MOPPrograms)
    for (unsigned ID : Programs) {
      const auto &P = Records.MOPPrograms[ID];
      if (P.Range && Bank.mayContainSFPUBefore(MI, {0, P.Range}))
        return true;
      for (unsigned Word : P.Words)
        if (!isOrdinaryRecordedOpcode(Records.Words[Word].Opcode))
          return true;
    }
  return false;
}

Expected<TensixMOPExecutionEffects>
summarizeMOPExecution(const MachineInstr &MI, const MOPState &S,
                       Inventory &Records,
                       const TensixReplayBankAnalysis &Bank) {
  if (MI.getOpcode() != RISCV::TTMOP ||
      !MI.getMF()->getInfo<RISCVMachineFunctionInfo>()->usesTensixSFPU())
    return createStringError(
        "SFPU MOP execution requires the local contents/bound ABI "
        "and known native MOP control fields");
  for (unsigned I = 0; I != 3; ++I)
    if (!MI.getOperand(I).isImm())
      return createStringError("SFPU MOP execution requires native control fields");
  unsigned MaskLo = MI.getOperand(0).getImm();
  unsigned Count = MI.getOperand(1).getImm();
  unsigned Kind = MI.getOperand(2).getImm();
  if (MaskLo > 65535 || Count > 127 || Kind > 1)
    return createStringError("SFPU MOP execution control field is out of range");
  if (Kind == 1 && (MaskLo || Count))
    return createStringError("SFPU MOP count overrides are unverified on Blackhole");
  TensixMOPSlots Slots;
  for (unsigned Slot = 0; Slot != Slots.size(); ++Slot)
    for (unsigned ID : S.MOPPrograms[Slot]) {
      const auto &Program = Records.MOPPrograms[ID];
      using Kind = TensixMOPSlotDeclaration::Kind;
      Slots[Slot].push_back(
          {Program.IsNOP ? Kind::Nop
                        : Program.Range ? Kind::Template : Kind::Opaque,
           ID});
    }
  auto Graph =
      Kind == 0
          ? buildTensixMOPSequence(
                TensixMOPTemplate0{MaskLo, Count, projectControl(S.MOPControls[1]),
                                  projectControl(S.MOPMaskHi)},
                Slots)
          : buildTensixMOPSequence(
                TensixMOPTemplate1{projectControl(S.MOPControls[0]),
                                  projectControl(S.MOPControls[1])},
                Slots);
  if (!Graph)
    return describeSequenceError(Graph.takeError(), Kind, S);
  MOPSequenceEffects Effects(Records,
                            *const_cast<MachineFunction *>(MI.getMF()), Bank, MI);
  auto Summary = Effects.summarize(*Graph);
  if (!Summary)
    return Summary.takeError();
  TensixMOPExecutionEffects Result;
  Result.Physical.Uses.assign(Summary->Reads.begin(), Summary->Reads.end());
  Result.Physical.Defs.assign(Summary->MayDef.begin(), Summary->MayDef.end());
  // A conditionally written register retains its incoming value on the other
  // paths. It is an output with passthrough, never an unconditional kill.
  for (MCRegister Reg : Summary->MayDef)
    if (!is_contained(Summary->MustDef, Reg) &&
        !is_contained(Result.Physical.Uses, Reg))
      Result.Physical.Uses.push_back(Reg);
  for (MCPhysReg Reg : MI.getDesc().implicit_uses())
    if (!is_contained(Result.Physical.Uses, MCRegister(Reg)))
      Result.Physical.Uses.push_back(Reg);
  for (MCPhysReg Reg : MI.getDesc().implicit_defs())
    if (!is_contained(Result.Physical.Defs, MCRegister(Reg)))
      Result.Physical.Defs.push_back(Reg);
  llvm::sort(Result.Physical.Uses);
  llvm::sort(Result.Physical.Defs);
  Result.Hazards = std::move(Summary->Hazards);
  return Result;
}

Error transfer(const MachineBasicBlock &BB, MOPState &S, Inventory &Records,
               const TensixReplayBankAnalysis &Bank, bool Check,
               bool RequireNormalized) {
  for (const auto &MI : BB) {
    if (MI.isCall() || MI.isInlineAsm()) {
      if (Bank.getCallEffect(MI) == TensixTargetStateCallEffect::Invalidate)
        S = MOPState();
      continue;
    }
    if (MI.getOpcode() == RISCV::PseudoTTMOPControlWrite ||
        MI.getOpcode() == RISCV::PseudoTTMOPControlWriteImm) {
      bool Constant = MI.getOpcode() == RISCV::PseudoTTMOPControlWriteImm;
      unsigned First = Constant ? 2 : 1;
      if (MI.getNumExplicitOperands() != First + 2 ||
          !MI.getOperand(First).isImm() || MI.getOperand(First).getImm() < 0 ||
          MI.getOperand(First).getImm() > 1 ||
          (Constant ? !MI.getOperand(First + 1).isImm()
                    : !MI.getOperand(First + 1).isReg()))
        return createStringError("Tensix MOP control write requires cell 0 or 1 "
                                 "and its captured scalar value");
      if (Error E = verifyTensixMachineEffects(MI))
        return E;
      auto &Control = S.MOPControls[MI.getOperand(First).getImm()];
      Control.Content = Constant ? MOPControlValue::Knowledge::Constant
                                 : MOPControlValue::Knowledge::Dynamic;
      Control.Value = Constant ? uint32_t(MI.getOperand(First + 1).getImm()) : 0;
      continue;
    }
    const auto *MOP = RISCV::getTensixMachineInfoByMop(MI.getOpcode());
    bool Clear = MI.getOpcode() == RISCV::PseudoTTMOPClear;
    if (MI.getOpcode() == RISCV::PseudoTTReplayTemplateMop) {
      // Its symbolic identity is received by the template owner. It replaces
      // this slot's old raw reference without inventing a physical placement.
      unsigned Slot = MI.getOperand(3).getImm();
      if (Slot < 2 || Slot > 8)
        return createStringError(
            "Tensix MOP instruction slot must be in [2, 8]");
      S.MOPKnown |= 1u << (Slot - 2);
      S.MOPRequires[Slot - 2] = 0;
      auto Words = Records.SymbolicWords.find(MI.getOperand(2).getImm());
      if (Words == Records.SymbolicWords.end())
        return createStringError("Tensix MOP symbolic binding has no native words");
      Inventory::MOPProgram Program;
      Program.Words = Words->second;
      S.MOPPrograms[Slot - 2] = {Records.internMOP(std::move(Program))};
      continue;
    }
    if (MOP || Clear) {
      unsigned Slot = MI.getOperand(2).getImm();
      if (Slot < 2 || Slot > 8)
        return createStringError(
            "Tensix MOP instruction slot must be in [2, 8]");
      uint32_t Required = 0;
      if (MOP && MOP->Opcode == RISCV::TTREPLAY) {
        auto Mask = getTensixReplayMask(MI, 3);
        if (!Mask)
          return Mask.takeError();
        if (MI.getOperand(3).getImm())
          return createStringError(
              "Tensix MOP replay slot must execute, not record");
        Required = *Mask;
      }
      S.MOPKnown |= 1u << (Slot - 2);
      S.MOPRequires[Slot - 2] = Required;
      Inventory::MOPProgram Program;
      Program.Range = Required;
      if (!Required) {
        unsigned Opcode = Clear ? unsigned(RISCV::TTNOP) : MOP->Opcode;
        if (!isOrdinaryRecordedOpcode(Opcode))
          return createStringError("Tensix MOP slot has unsupported nested control");
        TensixRecordedWord Word{Opcode, {}};
        if (!Clear)
          for (unsigned I = 3; I != MI.getNumExplicitOperands(); ++I) {
            if (!MI.getOperand(I).isImm())
              return createStringError("Tensix MOP slot fields must be immediate");
            Word.Operands.push_back({TensixRecordedOperand::Kind::Immediate,
                                     MI.getOperand(I).getImm()});
          }
        Program.IsNOP = Opcode == RISCV::TTNOP;
        Program.Words.push_back(Records.intern(std::move(Word)));
      }
      S.MOPPrograms[Slot - 2] = {Records.internMOP(std::move(Program))};
      if (Check)
        Records.Analysis.Storage.MOPWrites.try_emplace(
            &MI, TensixReplayStorageAnalysis::MOPWrite{Slot, Required});
      continue;
    }
    auto Issued = getTensixReplayIssue(MI);
    if (!Issued)
      continue;
    if (Issued->Opcode == RISCV::TTMOP_CFG) {
      if (Issued->Stream != 0)
        continue;
      const auto &Value = MI.getOperand(Issued->First);
      S.MOPMaskHi.Content = Value.isImm()
                               ? MOPControlValue::Knowledge::Constant
                               : MOPControlValue::Knowledge::Dynamic;
      S.MOPMaskHi.Value = Value.isImm() ? uint32_t(Value.getImm()) : 0;
      continue;
    }
    if (Issued->Opcode == RISCV::TTMOP && Check) {
      Records.Analysis.Storage.MOPExecutions.try_emplace(&MI, S.MOPRequires);
      if (Issued->Stream != 0)
        return createStringError(
            "remote Tensix MOP requires a verified template preservation ABI");
      if (S.MOPKnown != 0x7f)
        return createStringError("Tensix MOP execution requires all seven "
                                 "instruction slots to be configured");
      assert(Bank.hasObservationBefore(MI) && "missing MOP execution observation");
      if (mopMayContainSFPU(MI, S, Records, Bank)) {
        auto Effects = summarizeMOPExecution(MI, S, Records, Bank);
        if (!Effects)
          return Effects.takeError();
        if (RequireNormalized) {
          if (Error E = verifyTensixMOPExecutionEffects(MI, *Effects))
            return E;
        } else if (Error E = verifyTensixMOPExecutionEffects(MI, *Effects)) {
          consumeError(std::move(E));
          // Accept either the original descriptor or the exact normalized
          // effects, never an arbitrary manually-added implicit operand.
          if (Error Original = verifyTensixMachineEffects(MI))
            return Original;
        }
        Records.Analysis.MOPExecutions.try_emplace(&MI, std::move(*Effects));
      } else {
        if (Error E = verifyTensixMachineEffects(MI))
          return E;
        for (uint32_t Required : S.MOPRequires)
          if (!Bank.isRecordedBefore(MI, {0, Required}))
            return createStringError("Tensix MOP replay reference is not "
                                     "recorded on every incoming path");
      }
    }
  }
  return Error::success();
}
Expected<TensixExplicitReplayAnalysis> analyzeReplay(const MachineFunction &MF,
                                                     bool RequireNormalized) {
  if (MF.empty())
    return TensixExplicitReplayAnalysis();
  if (Error E = verifyAutomaticOwner(MF))
    return std::move(E);
  auto Scan = scanTensixReplay(MF, RequireNormalized);
  if (!Scan)
    return Scan.takeError();
  Inventory Records;
  if (Error E =
          receiveAutomaticReplay(MF, *Scan, Records.Analysis.ExecutionEffects))
    return std::move(E);
  auto Templates = getTensixReplayTemplateStorage(MF, RequireNormalized);
  if (!Templates)
    return Templates.takeError();
  for (const auto &[Identity, Words] : Templates->SymbolicWords) {
    auto &Symbolic = Records.SymbolicWords[Identity];
    for (const auto &Word : Words)
      Symbolic.push_back(Records.intern(Word));
  }
  auto Writes = getTensixReplayRecordWrites(*Scan);
  SmallPtrSet<const MachineInstr *, 8> WrittenEnds;
  for (const auto &Write : Writes)
    WrittenEnds.insert(Write.End);
  for (const auto &Fixed : Templates->FixedRecords) {
    if (!WrittenEnds.insert(Fixed.End).second)
      continue;
    uint32_t Mask =
        uint32_t(((uint64_t(1) << Fixed.Words.size()) - 1) << Fixed.Start);
    TensixReplayRecordWrite Overlay{Fixed.End,
                                    {0, Mask},
                                    Fixed.Start,
                                    TensixReplayWriteKind::NativeContents,
                                    {}};
    for (const auto &Word : Fixed.Words)
      Overlay.Words.push_back(Word);
    // This is a physical view, not another normalization owner. In particular
    // a fixed symbolic begin is never rewritten as an ordinary raw header here.
    Writes.push_back(std::move(Overlay));
  }
  SmallVector<const MachineInstr *, 8> Sites;
  for (const auto &BB : MF)
    for (const auto &MI : BB)
      if (auto Issue = getTensixReplayIssue(MI))
        if ((Issue->Opcode == RISCV::TTREPLAY &&
             !MI.getOperand(Issue->First).getImm()) ||
            Issue->Opcode == RISCV::TTMOP)
          Sites.push_back(&MI);
  auto Bank = analyzeTensixReplayBank(MF, Writes, Sites);
  if (!Bank)
    return Bank.takeError();
  auto Native = analyzeTensixNativeReplay(MF, *Scan, *Bank, RequireNormalized);
  if (!Native)
    return Native.takeError();
  Records.Analysis.Records = std::move(Native->Records);
  Records.Analysis.ExecutionWords = std::move(Native->ExecutionWords);
  for (auto &[MI, Effects] : Native->ExecutionEffects)
    Records.Analysis.ExecutionEffects.try_emplace(MI, std::move(Effects));
  Records.Analysis.Storage.RecordWrites = std::move(Native->RecordWrites);
  Records.Analysis.Storage.Executions = std::move(Native->Executions);

  DenseMap<const MachineBasicBlock *, MOPState> Entries, Exits;
  for (const auto &BB : MF)
    if (Bank->isReachable(BB))
      Exits[&BB] = MOPState::top();
  bool Changed;
  do {
    Changed = false;
    for (const auto &BB : MF) {
      if (!Bank->isReachable(BB))
        continue;
      MOPState Incoming = &BB == &MF.front() ? MOPState() : MOPState::top();
      for (const auto *Pred : BB.predecessors())
        if (Bank->isReachable(*Pred))
          Incoming.meet(Exits.lookup(Pred));
      Entries[&BB] = Incoming;
      if (Error E =
              transfer(BB, Incoming, Records, *Bank, false, RequireNormalized))
        return std::move(E);
      if (Incoming != Exits.lookup(&BB)) {
        Exits[&BB] = Incoming;
        Changed = true;
      }
    }
  } while (Changed);
  for (const auto &BB : MF) {
    if (!Bank->isReachable(BB))
      continue;
    MOPState Incoming = Entries.lookup(&BB);
    if (Error E =
            transfer(BB, Incoming, Records, *Bank, true, RequireNormalized))
      return std::move(E);
  }
  return std::move(Records.Analysis);
}

} // namespace

Expected<TensixExplicitReplayAnalysis>
llvm::analyzeTensixExplicitReplay(const MachineFunction &MF) {
  return analyzeReplay(MF, false);
}

Error llvm::verifyTensixReplay(
    const MachineFunction &MF,
    TensixSFPUReplayExecutionEffects *ExecutionEffects,
    TensixMOPExecutionAnalysis *MOPExecutions) {
  if (ExecutionEffects)
    ExecutionEffects->clear();
  auto Analysis = analyzeReplay(MF, true);
  if (!Analysis)
    return Analysis.takeError();
  if (ExecutionEffects)
    *ExecutionEffects = std::move(Analysis->ExecutionEffects);
  if (MOPExecutions)
    *MOPExecutions = std::move(Analysis->MOPExecutions);
  return Error::success();
}

TensixMOPHazardTransfer::Result
TensixMOPHazardTransfer::apply(TensixReplayHazardState State) const {
  Result Output = Inputs[0];
  for (unsigned Bit = 0; Bit != 32; ++Bit)
    if (State.SFPU & (1u << Bit))
      joinHazard(Output, Inputs[Bit + 1]);
  if (State.DstCycles)
    joinHazard(Output, Inputs[32 + std::min(State.DstCycles, 3u)]);
  return Output;
}

Error llvm::verifyTensixMOPExecutionEffects(
    const MachineInstr &MI, const TensixMOPExecutionEffects &Effects) {
  const auto &Expected = Effects.Physical;
  auto Fail = [] {
    return createStringError("Tensix MOP execution physical effects differ from "
                             "its selected native words");
  };
  if (MI.isBundled() ||
      MI.getNumExplicitOperands() != MI.getDesc().getNumOperands() ||
      MI.getNumOperands() != MI.getNumExplicitOperands() + Expected.Uses.size() +
                                 Expected.Defs.size())
    return Fail();
  SmallVector<MCRegister, 16> Uses, Defs;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead() ||
        MO.isRenamable())
      return Fail();
    (MO.isDef() ? Defs : Uses).push_back(MO.getReg().asMCReg());
  }
  llvm::sort(Uses);
  llvm::sort(Defs);
  if (ArrayRef(Uses) != ArrayRef(Expected.Uses) ||
      ArrayRef(Defs) != ArrayRef(Expected.Defs))
    return Fail();
  return Error::success();
}
