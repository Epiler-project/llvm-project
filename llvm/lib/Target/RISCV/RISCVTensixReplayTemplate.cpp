//===-- RISCVTensixReplayTemplate.cpp ----------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixReplayTemplate.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixBoundLowering.h"
#include "RISCVTensixReplay.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include <limits>
#include <optional>
#include <array>

using namespace llvm;

namespace {

struct ReplayTemplate {
  unsigned Identity;
  TensixReplayTemplateMode Mode;
  TensixReplayTemplatePlacement Placement;
  unsigned FixedStart;
  MachineInstr *Begin;
  MachineInstr *End = nullptr;
  SmallVector<MachineInstr *, 16> Body;
  SmallVector<TensixRecordedWord, 16> Words;
  SmallVector<MachineInstr *, 4> Executions;
  TensixSFPUReplayEffects Effects;
};

// These are query-local products, rebuilt after every mutation. No instruction
// pointer, reaching contents, or placement survives as a second authority.
struct TemplateInventory {
  SmallVector<ReplayTemplate, 4> Templates;
  DenseMap<unsigned, unsigned> Identities;
  DenseMap<const MachineInstr *, unsigned> Begins;
  DenseMap<const MachineInstr *, unsigned> Executions;
  struct MOPBinding {
    MachineInstr *Instruction;
    unsigned Template;
    unsigned Slot;
  };
  SmallVector<MOPBinding, 4> MOPBindings;
  using MOPState = std::array<BitVector, 7>;
  DenseMap<const MachineInstr *, MOPState> MOPStates;
  DenseMap<const MachineInstr *, BitVector> MOPExecutions;
};

bool isTemplateControl(unsigned Opcode) {
  return Opcode == RISCV::PseudoTTReplayTemplateBegin ||
         Opcode == RISCV::PseudoTTReplayTemplateEnd ||
         Opcode == RISCV::PseudoTTReplayTemplateExecute ||
         Opcode == RISCV::PseudoTTReplayTemplateMop;
}

bool hasTemplates(const MachineFunction &MF) {
  return any_of(MF, [](const MachineBasicBlock &MBB) {
    return any_of(MBB, [](const MachineInstr &MI) {
      return isTemplateControl(MI.getOpcode()) ||
             MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord ||
             MI.getOpcode() == RISCV::PseudoTTReplayTemplateDstWord;
    });
  });
}

Expected<unsigned> unsignedOperand(const MachineInstr &MI, unsigned Index) {
  if (MI.getNumExplicitOperands() <= Index || !MI.getOperand(Index).isImm() ||
      MI.getOperand(Index).getImm() < 0 ||
      uint64_t(MI.getOperand(Index).getImm()) >
          std::numeric_limits<unsigned>::max())
    return createStringError(
        "structured replay requires unsigned compiler identity and controls");
  return unsigned(MI.getOperand(Index).getImm());
}

bool hasEffects(const MachineInstr &MI,
                const TensixSFPUReplayEffects &Expected) {
  SmallVector<MCRegister, 16> Uses, Defs;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isInternalRead() || MO.isEarlyClobber() ||
        MO.isRenamable())
      return false;
    (MO.isDef() ? Defs : Uses).push_back(MO.getReg().asMCReg());
  }
  llvm::sort(Uses);
  llvm::sort(Defs);
  return Uses == Expected.Uses && Defs == Expected.Defs &&
         MI.getNumOperands() ==
             MI.getNumExplicitOperands() + Uses.size() + Defs.size();
}

// MOP programming is an independent machine owner when it is outside a
// structured recording body.  It still has to be rejected from a body: a MOP
// executes/configures a slot rather than contributing one linear recordable
// word, and admitting it there would make the template's words/effects
// incomplete.  Keep this distinction local to the receiver so a function
// containing a structured template and an independent (possibly dynamic
// port-form) MOP can share the same machine stream.
bool isMOPOwner(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode == RISCV::TTMOP || Opcode == RISCV::TTMOP_CFG ||
      Opcode == RISCV::PseudoTTMOPControlWrite ||
      Opcode == RISCV::PseudoTTMOPControlWriteImm ||
      Opcode == RISCV::PseudoTTReplayTemplateMop ||
      Opcode == RISCV::PseudoTTMOPClear ||
      RISCV::getTensixMachineInfoByMop(Opcode))
    return true;
  if (const auto *Port = RISCV::getTensixMachineInfoByPort(Opcode))
    return Port->Opcode == RISCV::TTMOP || Port->Opcode == RISCV::TTMOP_CFG;
  return false;
}

bool isMOPExecution(const MachineInstr &MI) {
  if (MI.getOpcode() == RISCV::TTMOP)
    return true;
  if (const auto *Port = RISCV::getTensixMachineInfoByPort(MI.getOpcode()))
    return Port->Opcode == RISCV::TTMOP;
  return false;
}

// Reconstruct persistent MOP identities at actual execution points. This is
// a finite CFG dataflow over seven slots, not source iteration enumeration.
// A clear or any ordinary MOP write kills that slot's former template; joins
// retain every incoming possibility and backedges keep its future consumers.
void collectMOPExecutions(MachineFunction &MF, TemplateInventory &Inventory) {
  using Slots = TemplateInventory::MOPState;
  unsigned Count = Inventory.Templates.size();
  auto empty = [&] {
    Slots Result;
    for (auto &Slot : Result)
      Slot.resize(Count);
    return Result;
  };
  DenseMap<const MachineBasicBlock *, Slots> In, Out;
  DenseMap<const MachineInstr *, const TemplateInventory::MOPBinding *> Bindings;
  for (const auto &Binding : Inventory.MOPBindings)
    Bindings.try_emplace(Binding.Instruction, &Binding);
  for (const MachineBasicBlock &MBB : MF) {
    In.try_emplace(&MBB, empty());
    Out.try_emplace(&MBB, empty());
  }
  auto transfer = [&](const MachineBasicBlock &MBB, Slots State, bool Collect) {
    for (const MachineInstr &MI : MBB) {
      if (Collect && (Inventory.Begins.contains(&MI) ||
                      MI.getOpcode() == RISCV::PseudoTTReplayRecordEnd))
        Inventory.MOPStates.try_emplace(&MI, State);
      if (auto Found = Bindings.find(&MI); Found != Bindings.end()) {
        const auto &Binding = *Found->second;
        State[Binding.Slot - 2].reset();
        State[Binding.Slot - 2].set(Binding.Template);
      } else if (RISCV::getTensixMachineInfoByMop(MI.getOpcode()) ||
                 MI.getOpcode() == RISCV::PseudoTTMOPClear) {
        unsigned Slot = unsigned(MI.getOperand(2).getImm());
        if (Slot >= 2 && Slot <= 8)
          State[Slot - 2].reset();
      } else if (Collect && isMOPExecution(MI)) {
        BitVector Used(Count);
        for (const auto &Slot : State)
          Used |= Slot;
        if (Used.none())
          continue;
        Inventory.MOPExecutions.try_emplace(&MI, Used);

      }
    }
    return State;
  };
  bool Changed;
  do {
    Changed = false;
    for (const MachineBasicBlock &MBB : MF) {
      Slots Incoming = empty();
      for (const MachineBasicBlock *Pred : MBB.predecessors())
        for (unsigned Slot = 0; Slot != 7; ++Slot)
          Incoming[Slot] |= Out.find(Pred)->second[Slot];
      Slots Next = transfer(MBB, Incoming, false);
      In[&MBB] = std::move(Incoming);
      if (Next != Out.find(&MBB)->second) {
        Out[&MBB] = std::move(Next);
        Changed = true;
      }
    }
  } while (Changed);
  for (const MachineBasicBlock &MBB : MF)
    transfer(MBB, In.find(&MBB)->second, true);
}

bool hasRawReplayOwner(const MachineInstr &MI) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode == RISCV::TTREPLAY ||
      Opcode == RISCV::PseudoTTExplicitSFPUReplay ||
      Opcode == RISCV::PseudoTTSFPUReplay ||
      Opcode == RISCV::PseudoTTSFPURecordWord ||
      Opcode == RISCV::PseudoTTSFPUDstRecordWord ||
      Opcode == RISCV::PseudoTTReplayRecordEnd)
    return true;
  if (const auto *Port = RISCV::getTensixMachineInfoByPort(Opcode))
    return Port->Opcode == RISCV::TTREPLAY;
  return false;
}

bool isRecordedIssue(const MachineInstr &MI) {
  return RISCV::getTensixEncoding(MI.getOpcode()) ||
         RISCV::getTensixMachineInfoByPort(MI.getOpcode()) ||
         MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
         MI.getOpcode() == RISCV::PseudoTTSFPSTORE ||
         MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord ||
         MI.getOpcode() == RISCV::PseudoTTReplayTemplateDstWord;
}

Expected<TemplateInventory> collectTemplates(MachineFunction &MF,
                                            bool RequirePrepared) {
  TemplateInventory Result;
  SmallVector<MachineInstr *, 8> Executions;
  SmallVector<MachineInstr *, 8> MOPBindings;

  for (MachineBasicBlock &MBB : MF) {
    std::optional<unsigned> Active;
    for (MachineInstr &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      if (MI.isBundled())
        return createStringError(
            "structured replay requires an unbundled native issue stream");
      const bool MOP = isMOPOwner(MI);
      if (Active && !MOP && hasRawReplayOwner(MI))
        return createStringError(
            "raw and structured replay recordings cannot nest");
      if (MI.isCall() || MI.isInlineAsm())
        return createStringError(
            "structured replay has no call or inline-asm preservation ABI");
      if (Opcode == RISCV::PseudoTTReplayTemplateBegin) {
        if (Active)
          return createStringError("structured replay recordings cannot nest");
        if (MI.getNumExplicitOperands() != 4)
          return createStringError("structured replay begin needs four controls");
        if (Error E = verifyTensixMachineEffects(MI))
          return std::move(E);
        auto Identity = unsignedOperand(MI, 0);
        if (!Identity)
          return Identity.takeError();
        auto Mode = unsignedOperand(MI, 1);
        if (!Mode)
          return Mode.takeError();
        auto Placement = unsignedOperand(MI, 2);
        if (!Placement)
          return Placement.takeError();
        auto Start = unsignedOperand(MI, 3);
        if (!Start)
          return Start.takeError();
        if (*Mode > unsigned(TensixReplayTemplateMode::RecordAndExecute) ||
            *Placement > unsigned(TensixReplayTemplatePlacement::Fixed) ||
            *Start >= TensixReplaySlotCount ||
            (*Placement == unsigned(TensixReplayTemplatePlacement::Automatic) &&
             *Start != 0))
          return createStringError("invalid structured replay mode or placement");
        unsigned Index = Result.Templates.size();
        if (!Result.Identities.try_emplace(*Identity, Index).second)
          return createStringError(
              "structured replay identity has multiple record producers");
        Result.Templates.push_back(
            {*Identity, TensixReplayTemplateMode(*Mode),
             TensixReplayTemplatePlacement(*Placement), *Start, &MI});
        Result.Begins.try_emplace(&MI, Index);
        Active = Index;
        continue;
      }
      if (Opcode == RISCV::PseudoTTReplayTemplateEnd) {
        if (!Active || MI.getNumExplicitOperands() != 1)
          return createStringError("structured replay end has no active record");
        if (Error E = verifyTensixMachineEffects(MI))
          return std::move(E);
        auto Identity = unsignedOperand(MI, 0);
        if (!Identity)
          return Identity.takeError();
        ReplayTemplate &T = Result.Templates[*Active];
        if (*Identity != T.Identity)
          return createStringError("structured replay end has a different identity");
        T.End = &MI;
        auto Effects = getTensixExplicitReplayEffects(T.Words, MF);
        if (!Effects)
          return Effects.takeError();
        T.Effects = std::move(*Effects);
        Active.reset();
        continue;
      }
      if (Opcode == RISCV::PseudoTTReplayTemplateExecute) {
        if (Active || MI.getNumExplicitOperands() != 1)
          return createStringError(
              "structured replay execution must be outside recording bodies");
        Executions.push_back(&MI);
        continue;
      }
      if (Opcode == RISCV::PseudoTTReplayTemplateMop) {
        if (Active || MI.getNumExplicitOperands() != 4)
          return createStringError(
              "structured replay MOP binding must be outside recording bodies");
        if (Error E = verifyTensixMachineEffects(MI))
          return std::move(E);
        MOPBindings.push_back(&MI);
        continue;
      }
      if (Active && MOP)
        return createStringError(
            "structured replay recording cannot contain MOP");
      if ((Opcode == RISCV::PseudoTTReplayTemplateWord ||
           Opcode == RISCV::PseudoTTReplayTemplateDstWord) && !Active)
        return createStringError("symbolic replay word has no recording owner");
      if (!Active)
        continue;
      if (MI.isTerminator())
        return createStringError(
            "structured replay recording cannot cross scalar control flow");
      if (!isRecordedIssue(MI))
        continue;
      ReplayTemplate &T = Result.Templates[*Active];
      bool Payload = Opcode == RISCV::PseudoTTReplayTemplateWord ||
                     Opcode == RISCV::PseudoTTReplayTemplateDstWord;
      bool RecordOnly = T.Mode == TensixReplayTemplateMode::RecordOnly;
      if (Payload) {
        auto Identity = unsignedOperand(
            MI, Opcode == RISCV::PseudoTTReplayTemplateDstWord ? 2 : 0);
        if (!Identity)
          return Identity.takeError();
        if (!RecordOnly || *Identity != T.Identity)
          return createStringError(
              "symbolic replay word differs from its recording owner or mode");
      } else if (RequirePrepared && RecordOnly) {
        return createStringError(
            "record-only template was not normalized before machine optimization");
      }
      auto Word = getTensixRecordedWord(MI, MF);
      if (!Word)
        return Word.takeError();
      if (!RISCV::getTensixMachineInfo(Word->Opcode) &&
          !MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU())
        return createStringError(
            "structured SFPU replay requires the local bound physical ABI");
      T.Body.push_back(&MI);
      T.Words.push_back(std::move(*Word));
    }
    if (Active)
      return createStringError(
          "structured replay recording requires an end in its basic block");
  }

  MachineDominatorTree Dominators(MF);
  for (MachineInstr *MI : MOPBindings) {
    auto Identity = unsignedOperand(*MI, 2);
    if (!Identity)
      return Identity.takeError();
    auto Slot = unsignedOperand(*MI, 3);
    if (!Slot)
      return Slot.takeError();
    if (*Slot < 2 || *Slot > 8)
      return createStringError("structured replay MOP slot must be in [2, 8]");
    auto Found = Result.Identities.find(*Identity);
    if (Found == Result.Identities.end())
      return createStringError("structured replay MOP binding has no record producer");
    ReplayTemplate &T = Result.Templates[Found->second];
    if (!T.End || !Dominators.dominates(T.End, MI))
      return createStringError(
          "structured replay recording must dominate its MOP binding");
    Result.MOPBindings.push_back({MI, Found->second, *Slot});
  }
  for (MachineInstr *MI : Executions) {
    auto Identity = unsignedOperand(*MI, 0);
    if (!Identity)
      return Identity.takeError();
    auto Found = Result.Identities.find(*Identity);
    if (Found == Result.Identities.end())
      return createStringError("structured replay execution has no record producer");
    ReplayTemplate &T = Result.Templates[Found->second];
    if (!T.End || !Dominators.dominates(T.End, MI))
      return createStringError(
          "structured replay recording must dominate its execution");
    if (!hasEffects(*MI, T.Effects)) {
      if (RequirePrepared)
        return createStringError(
            "structured replay execution effects differ from its body");
      if (Error E = verifyTensixMachineEffects(*MI))
        return std::move(E);
    }
    T.Executions.push_back(MI);
    Result.Executions.try_emplace(MI, Found->second);
  }
  collectMOPExecutions(MF, Result);
  return Result;
}

void retainRegister(MachineFunction &MF, MCRegister Reg) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MRI.clearKillFlags(Reg);
  for (MachineOperand &MO : MRI.def_operands(Reg))
    MO.setIsDead(false);
}

void retainEffects(MachineFunction &MF, const TensixSFPUReplayEffects &Effects) {
  for (MCRegister Reg : Effects.Uses)
    retainRegister(MF, Reg);
  for (MCRegister Reg : Effects.Defs)
    retainRegister(MF, Reg);
}

void addExecutionEffects(MachineInstrBuilder &Builder,
                         const TensixSFPUReplayEffects &Effects) {
  const MCInstrDesc &Desc = Builder->getDesc();
  for (MCRegister Reg : Effects.Uses)
    if (!is_contained(Desc.implicit_uses(), Reg.id()))
      Builder.addReg(Reg, RegState::Implicit);
  for (MCRegister Reg : Effects.Defs)
    if (!is_contained(Desc.implicit_defs(), Reg.id()))
      Builder.addReg(Reg, RegState::Define | RegState::Implicit);
}

bool prepareTemplates(MachineFunction &MF, TemplateInventory &Inventory) {
  const RISCVInstrInfo &TII = *MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
  bool Changed = false;
  for (ReplayTemplate &T : Inventory.Templates) {
    if (T.Mode == TensixReplayTemplateMode::RecordOnly)
      for (auto [Original, Word] : zip(T.Body, T.Words)) {
        if (Original->getOpcode() == RISCV::PseudoTTReplayTemplateWord ||
            Original->getOpcode() == RISCV::PseudoTTReplayTemplateDstWord)
          continue;
        // Removing false numerical effects can expose an earlier physical
        // value even when no later execute exists. Preserve it before erasure.
        // Architectural state such as Dst is implicit in the native descriptor,
        // so retaining only the encoded register operands is insufficient.
        retainEffects(MF, T.Effects);
        for (const TensixRecordedOperand &Operand : Word.Operands)
          if (Operand.OperandKind == TensixRecordedOperand::Kind::Register)
            retainRegister(MF, MCRegister(unsigned(Operand.Value)));
        insertTensixRecordedPayload(*Original, Word, TII, T.Identity)
            .setMIFlags(Original->getFlags());
        Original->eraseFromParent();
        Changed = true;
      }
    for (MachineInstr *Original : T.Executions) {
      if (hasEffects(*Original, T.Effects))
        continue;
      retainEffects(MF, T.Effects);
      auto Builder = BuildMI(*Original->getParent(), *Original,
                             Original->getDebugLoc(),
                             TII.get(RISCV::PseudoTTReplayTemplateExecute));
      Builder.addImm(T.Identity)
          .cloneMemRefs(*Original)
          .setMIFlags(Original->getFlags());
      addExecutionEffects(Builder, T.Effects);
      Original->eraseFromParent();
      Changed = true;
    }
  }
  return Changed;
}

struct FinalTemplate {
  SmallVector<unsigned, 16> Gaps;
  SmallVector<TensixRecordedWord, 16> Words;
  TensixSFPUReplayEffects Effects;
  std::optional<unsigned> Start;
};

Expected<FinalTemplate> finalizeBody(const ReplayTemplate &T,
                                    MachineFunction &MF) {
  FinalTemplate Result;
  const auto &TRI = *MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();
  TensixReplayHazardState State;
  const TensixRecordedWord Nop{RISCV::TTSFPNOP, {}};
  auto NativeNop = createTensixRecordedInstruction(Nop, MF);
  if (!NativeNop)
    return NativeNop.takeError();
  auto DeleteNop = scope_exit([&] { MF.deleteMachineInstr(*NativeNop); });
  for (const TensixRecordedWord &Word : T.Words) {
    auto Native = createTensixRecordedInstruction(Word, MF);
    if (!Native)
      return Native.takeError();
    auto DeleteNative = scope_exit([&] { MF.deleteMachineInstr(*Native); });
    unsigned Gaps = 0;
    while (needsTensixReplayHazardGap(**Native, TRI, State)) {
      if (Result.Words.size() >= TensixReplaySlotCount)
        return createStringError(
            "structured replay hazard repair exceeds replay capacity");
      Result.Words.push_back(Nop);
      ++Gaps;
      State = advanceTensixReplayHazard(**NativeNop, TRI, State);
    }
    Result.Gaps.push_back(Gaps);
    Result.Words.push_back(Word);
    if (Result.Words.size() > TensixReplaySlotCount)
      return createStringError(
          "structured replay final words exceed replay capacity");
    State = advanceTensixReplayHazard(**Native, TRI, State);
  }
  auto Effects = getTensixExplicitReplayEffects(Result.Words, MF);
  if (!Effects)
    return Effects.takeError();
  Result.Effects = std::move(*Effects);
  if (Result.Effects.Uses != T.Effects.Uses ||
      Result.Effects.Defs != T.Effects.Defs)
    return createStringError(
        "structured replay finalization changed numerical effects");
  return Result;
}

struct TemplateInterference {
  SmallVector<BitVector, 4> Templates;
  SmallVector<uint32_t, 4> Raw;
};

uint32_t templateMask(const ReplayTemplate &T, const FinalTemplate &Final) {
  return uint32_t(((uint64_t(1) << Final.Words.size()) - 1) << T.FixedStart);
}

TemplateInterference
computeInterference(const MachineFunction &MF,
                    const TemplateInventory &Inventory,
                    ArrayRef<FinalTemplate> Final,
                    const TensixReplayStorageAnalysis &Raw) {
  unsigned Count = Inventory.Templates.size();
  struct LiveState {
    BitVector Templates;
    uint8_t Slots = 0;
    std::array<uint32_t, 3> RawDemand{};
    std::array<uint32_t, 7> RawMOPDemand{};
    bool operator==(const LiveState &Other) const {
      return Templates == Other.Templates && Slots == Other.Slots &&
             RawDemand == Other.RawDemand &&
             RawMOPDemand == Other.RawMOPDemand;
    }
  };
  DenseMap<const MachineBasicBlock *, LiveState> LiveIn, LiveOut;
  DenseMap<const MachineInstr *, const TemplateInventory::MOPBinding *> Bindings;
  for (const auto &Binding : Inventory.MOPBindings)
    Bindings.try_emplace(Binding.Instruction, &Binding);
  for (const MachineBasicBlock &MBB : MF) {
    LiveIn.try_emplace(&MBB, LiveState{BitVector(Count)});
    LiveOut.try_emplace(&MBB, LiveState{BitVector(Count)});
  }
  // Slot demand is transported independently of template identity. Turning
  // every reaching MOP identity into a globally live value loses overwrite
  // correlation at a loop backedge and incorrectly pins dead generations.
  // Raw MOP contents also have a lifetime distinct from their binding. A later
  // rerecord can satisfy a future execution without rewriting the MOP slot;
  // carry its pending physical demand per slot until a binding or recording
  // resolves it, instead of restoring forward ranges at every template begin.
  auto consume = [&](const MachineInstr &MI, LiveState &Live) {
    if (isMOPExecution(MI))
      Live.Slots = 0x7f;
    if (auto It = Raw.MOPExecutions.find(&MI);
        It != Raw.MOPExecutions.end())
      for (unsigned Slot = 0; Slot != 7; ++Slot)
        Live.RawMOPDemand[Slot] |= It->second[Slot];
    if (auto It = Inventory.Executions.find(&MI);
        It != Inventory.Executions.end())
      Live.Templates.set(It->second);
    if (auto It = Raw.Executions.find(&MI); It != Raw.Executions.end())
      Live.RawDemand[It->second.Stream] |= It->second.Mask;
    if (auto It = Bindings.find(&MI); It != Bindings.end()) {
      Live.Templates.set(It->second->Template);
      Live.Slots &= ~(1u << (It->second->Slot - 2));
      Live.RawMOPDemand[It->second->Slot - 2] = 0;
    } else if (auto It = Raw.MOPWrites.find(&MI);
               It != Raw.MOPWrites.end()) {
      uint8_t Slot = 1u << (It->second.Slot - 2);
      Live.RawDemand[0] |=
          Live.RawMOPDemand[It->second.Slot - 2] & It->second.Mask;
      Live.RawMOPDemand[It->second.Slot - 2] = 0;
      Live.Slots &= ~Slot;
    }
  };
  auto killRecord = [&](const MachineInstr &MI, LiveState &Live) {
    if (auto It = Raw.RecordWrites.find(&MI); It != Raw.RecordWrites.end()) {
      Live.RawDemand[It->second.Stream] &= ~It->second.Mask;
      if (It->second.Stream == 0)
        for (uint32_t &Demand : Live.RawMOPDemand)
          Demand &= ~It->second.Mask;
    }
    if (auto It = Inventory.Begins.find(&MI); It != Inventory.Begins.end()) {
      unsigned Index = It->second;
      const ReplayTemplate &T = Inventory.Templates[Index];
      if (T.Placement == TensixReplayTemplatePlacement::Fixed) {
        uint32_t Mask = templateMask(T, Final[Index]);
        Live.RawDemand[0] &= ~Mask;
        for (uint32_t &Demand : Live.RawMOPDemand)
          Demand &= ~Mask;
      }
      Live.Templates.reset(Index);
    }
  };
  auto transfer = [&](const MachineBasicBlock &MBB, LiveState Live) {
    for (const MachineInstr &MI : reverse(MBB)) {
      consume(MI, Live);
      killRecord(MI, Live);
    }
    return Live;
  };
  bool Changed;
  do {
    Changed = false;
    for (const MachineBasicBlock &MBB : reverse(MF)) {
      LiveState Out{BitVector(Count)};
      for (const MachineBasicBlock *Successor : MBB.successors()) {
        const LiveState &Incoming = LiveIn.find(Successor)->second;
        Out.Templates |= Incoming.Templates;
        Out.Slots |= Incoming.Slots;
        for (unsigned Stream = 0; Stream != 3; ++Stream)
          Out.RawDemand[Stream] |= Incoming.RawDemand[Stream];
        for (unsigned Slot = 0; Slot != 7; ++Slot)
          Out.RawMOPDemand[Slot] |= Incoming.RawMOPDemand[Slot];
      }
      LiveState In = transfer(MBB, Out);
      if (!(LiveIn.find(&MBB)->second == In) ||
          !(LiveOut.find(&MBB)->second == Out)) {
        LiveIn[&MBB] = std::move(In);
        LiveOut[&MBB] = std::move(Out);
        Changed = true;
      }
    }
  } while (Changed);

  TemplateInterference Conflicts{SmallVector<BitVector, 4>(Count,
                                                          BitVector(Count)),
                                 SmallVector<uint32_t, 4>(Count, 0)};
  for (const MachineBasicBlock &MBB : MF) {
    LiveState Live = LiveOut.find(&MBB)->second;
    for (const MachineInstr &MI : reverse(MBB)) {
      consume(MI, Live);
      auto Begin = Inventory.Begins.find(&MI);
      BitVector Occupants = Live.Templates;
      if (auto It = Inventory.MOPStates.find(&MI);
          It != Inventory.MOPStates.end())
        for (unsigned Slot = 0; Slot != 7; ++Slot)
          if (Live.Slots & (1u << Slot))
            Occupants |= It->second[Slot];
      // A later raw writer is as important as raw contents already present at
      // a begin. A still-live symbolic use cannot silently become that writer's
      // new version, even when it happens to encode the same opcode shape.
      if (auto It = Raw.RecordWrites.find(&MI);
          It != Raw.RecordWrites.end() && It->second.Stream == 0)
        for (int Other = Occupants.find_first(); Other >= 0;
             Other = Occupants.find_next(Other))
          Conflicts.Raw[Other] |= It->second.Mask;
      if (Begin == Inventory.Begins.end()) {
        killRecord(MI, Live);
        continue;
      }
      unsigned Index = Begin->second;
      if (Inventory.Templates[Index].Placement ==
          TensixReplayTemplatePlacement::Automatic) {
        uint32_t Demand = Live.RawDemand[0];
        for (uint32_t MOPDemand : Live.RawMOPDemand)
          Demand |= MOPDemand;
        Conflicts.Raw[Index] |= Demand;
      }
      // Even an unused recording overwrites hardware slots. Future direct
      // executes and persistent MOP consumers constrain the same placement.
      for (int Other = Occupants.find_first(); Other >= 0;
           Other = Occupants.find_next(Other))
        if (unsigned(Other) != Index) {
          Conflicts.Templates[Index].set(Other);
          Conflicts.Templates[Other].set(Index);
        }
      killRecord(MI, Live);
    }
  }
  return Conflicts;
}

Error placeTemplates(const MachineFunction &MF,
                      const TemplateInventory &Inventory,
                      SmallVectorImpl<FinalTemplate> &Final) {
  auto Raw = analyzeTensixExplicitReplay(MF);
  if (!Raw)
    return Raw.takeError();
  auto Conflicts = computeInterference(MF, Inventory, Final, Raw->Storage);
  auto Fits = [&](unsigned Index, unsigned Start) {
    unsigned Length = Final[Index].Words.size();
    if (Start >= TensixReplaySlotCount || Length > TensixReplaySlotCount - Start)
      return false;
    uint32_t Mask = uint32_t(((uint64_t(1) << Length) - 1) << Start);
    if (Mask & Conflicts.Raw[Index])
      return false;
    for (int Other = Conflicts.Templates[Index].find_first(); Other >= 0;
         Other = Conflicts.Templates[Index].find_next(Other)) {
      const FinalTemplate &Placed = Final[Other];
      if (Placed.Start && Start < *Placed.Start + Placed.Words.size() &&
          *Placed.Start < Start + Length)
        return false;
    }
    return true;
  };
  for (auto [Index, T] : enumerate(Inventory.Templates)) {
    if (T.Placement != TensixReplayTemplatePlacement::Fixed)
      continue;
    if (!Fits(Index, T.FixedStart))
      return createStringError(
          "structured replay fixed placement conflicts or exceeds capacity");
    Final[Index].Start = T.FixedStart;
  }
  SmallVector<unsigned, 4> Automatic;
  for (unsigned Index = 0; Index < Final.size(); ++Index)
    if (!Final[Index].Start)
      Automatic.push_back(Index);
  llvm::stable_sort(Automatic, [&](unsigned Left, unsigned Right) {
    return Final[Left].Words.size() > Final[Right].Words.size();
  });
  for (unsigned Index : Automatic) {
    for (unsigned Start = 0; Start < TensixReplaySlotCount; ++Start)
      if (Fits(Index, Start)) {
        Final[Index].Start = Start;
        break;
      }
    if (!Final[Index].Start)
      return createStringError(
          "structured replay automatic placement exceeds available capacity");
  }
  return Error::success();
}

void replaceControl(MachineInstr &Original, unsigned Load, unsigned Execute,
                     const FinalTemplate &Final, const RISCVInstrInfo &TII) {
  auto Builder = BuildMI(*Original.getParent(), Original, Original.getDebugLoc(),
                         TII.get(RISCV::PseudoTTExplicitSFPUReplay));
  Builder.addImm(Load)
      .addImm(Execute)
      .addImm(Final.Words.size())
      .addImm(*Final.Start)
      .cloneMemRefs(Original)
      .setMIFlags(Original.getFlags());
  if (!Load)
    addExecutionEffects(Builder, Final.Effects);
  Original.eraseFromParent();
}

void commitTemplates(MachineFunction &MF, TemplateInventory &Inventory,
                     ArrayRef<FinalTemplate> Final) {
  const auto &TII = *MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
  const TensixRecordedWord Nop{RISCV::TTSFPNOP, {}};
  for (auto [T, Plan] : zip(Inventory.Templates, Final)) {
    bool RecordOnly = T.Mode == TensixReplayTemplateMode::RecordOnly;
    for (auto [Original, Word, Gaps] : zip(T.Body, T.Words, Plan.Gaps)) {
      for (unsigned Gap = 0; Gap < Gaps; ++Gap) {
        if (RecordOnly)
          insertTensixRecordedPayload(*Original, Nop, TII, std::nullopt);
        else
          BuildMI(*Original->getParent(), *Original, Original->getDebugLoc(),
                  TII.get(RISCV::TTSFPNOP));
      }
      if (RecordOnly) {
        insertTensixRecordedPayload(*Original, Word, TII, std::nullopt)
            .setMIFlags(Original->getFlags());
        Original->eraseFromParent();
      }
    }
    replaceControl(*T.Begin, 1, !RecordOnly, Plan, TII);
    BuildMI(*T.End->getParent(), *T.End, T.End->getDebugLoc(),
            TII.get(RISCV::PseudoTTReplayRecordEnd))
        .setMIFlags(T.End->getFlags());
    T.End->eraseFromParent();
    for (MachineInstr *Execution : T.Executions)
      replaceControl(*Execution, 0, 0, Plan, TII);
  }
  for (const auto &Binding : Inventory.MOPBindings) {
    MachineInstr &Original = *Binding.Instruction;
    const FinalTemplate &Plan = Final[Binding.Template];
    auto Builder = BuildMI(*Original.getParent(), Original,
                           Original.getDebugLoc(),
                           TII.get(RISCV::PseudoTTREPLAYMop));
    Builder.add(Original.getOperand(0)).add(Original.getOperand(1));
    Builder.addImm(Binding.Slot).addImm(0).addImm(0)
        .addImm(Plan.Words.size()).addImm(*Plan.Start)
        .cloneMemRefs(Original).setMIFlags(Original.getFlags());
    Original.eraseFromParent();
  }
}

bool fail(MachineFunction &MF, Error E) {
  MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
  MF.getFunction().getContext().diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), toString(std::move(E))));
  return false;
}

class RISCVTensixReplayTemplatePrepare final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixReplayTemplatePrepare() : MachineFunctionPass(ID) {
    initializeRISCVTensixReplayTemplatePreparePass(
        *PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (MF.getInfo<RISCVMachineFunctionInfo>()->hasTensixCodegenFailed() ||
        !hasTemplates(MF))
      return false;
    auto Inventory = collectTemplates(MF, false);
    if (!Inventory)
      return fail(MF, Inventory.takeError());
    return prepareTemplates(MF, *Inventory);
  }
};

class RISCVTensixReplayTemplateFinalize final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixReplayTemplateFinalize() : MachineFunctionPass(ID) {
    initializeRISCVTensixReplayTemplateFinalizePass(
        *PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (MF.getInfo<RISCVMachineFunctionInfo>()->hasTensixCodegenFailed() ||
        !hasTemplates(MF))
      return false;
    auto Inventory = collectTemplates(MF, true);
    if (!Inventory)
      return fail(MF, Inventory.takeError());
    SmallVector<FinalTemplate, 4> Final;
    for (const ReplayTemplate &T : Inventory->Templates) {
      auto Plan = finalizeBody(T, MF);
      if (!Plan)
        return fail(MF, Plan.takeError());
      Final.push_back(std::move(*Plan));
    }
    if (Error E = placeTemplates(MF, *Inventory, Final))
      return fail(MF, std::move(E));
    // Input, effects, all bodies, and the whole placement succeed before any
    // mutation. The commit does not discover new numerical work or control.
    commitTemplates(MF, *Inventory, Final);
    return true;
  }
};

} // namespace

Expected<TensixReplayTemplateStorage>
llvm::getTensixReplayTemplateStorage(const MachineFunction &Function,
                                     bool RequirePrepared) {
  TensixReplayTemplateStorage Result;
  if (!hasTemplates(Function))
    return Result;
  // Body finalization only creates and immediately releases uninserted native
  // instructions for the canonical hazard/effect queries. It does not mutate
  // the function, nor retain its temporary word/placement view across a pass.
  auto &MF = const_cast<MachineFunction &>(Function);
  auto Inventory = collectTemplates(MF, RequirePrepared);
  if (!Inventory)
    return Inventory.takeError();
  for (const ReplayTemplate &T : Inventory->Templates) {
    auto Final = finalizeBody(T, MF);
    if (!Final)
      return Final.takeError();
    Result.SymbolicWords.try_emplace(T.Identity, Final->Words);
    if (T.Placement != TensixReplayTemplatePlacement::Fixed)
      continue;
    if (Final->Words.size() > TensixReplaySlotCount - T.FixedStart)
      return createStringError(
          "structured replay fixed placement conflicts or exceeds capacity");
    Result.FixedRecords.push_back(
        {T.Begin, T.End, T.FixedStart, std::move(Final->Words)});
  }
  return Result;
}

Error llvm::verifyTensixReplayTemplates(MachineFunction &MF) {
  if (!hasTemplates(MF))
    return Error::success();
  auto Inventory = collectTemplates(MF, /*RequirePrepared=*/true);
  if (!Inventory)
    return Inventory.takeError();
  return Error::success();
}

char RISCVTensixReplayTemplatePrepare::ID = 0;
INITIALIZE_PASS(RISCVTensixReplayTemplatePrepare,
                "riscv-tensix-replay-template-prepare",
                "Prepare structured Tensix replay effects", false, false)
FunctionPass *llvm::createRISCVTensixReplayTemplatePreparePass() {
  return new RISCVTensixReplayTemplatePrepare();
}

char RISCVTensixReplayTemplateFinalize::ID = 0;
INITIALIZE_PASS(RISCVTensixReplayTemplateFinalize,
                "riscv-tensix-replay-template-finalize",
                "Finalize structured Tensix replay words and placement", false,
                false)
FunctionPass *llvm::createRISCVTensixReplayTemplateFinalizePass() {
  return new RISCVTensixReplayTemplateFinalize();
}
