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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <limits>
#include <optional>

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
  bool Contents = false;
  bool MOPRepeated = false;
  SmallVector<Register, 8> InitialMutables;
  SmallVector<Register, 8> Mutables;
  SmallVector<Register, 8> Captures;
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

// A closed recipe set keeps non-executing recording distinct from numerical
// machine instructions. The native descriptor remains the encoding/effect owner.
unsigned contentsReplayRecipeOpcode(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::PseudoTTContentsReplayTemplateRecordIAdd:
    return RISCV::TTSFPIADD;
  case RISCV::PseudoTTContentsReplayTemplateRecordAnd:
    return RISCV::TTSFPAND;
  case RISCV::PseudoTTContentsReplayTemplateRecordOr:
    return RISCV::TTSFPOR;
  case RISCV::PseudoTTContentsReplayTemplateRecordXor:
    return RISCV::TTSFPXOR;
  case RISCV::PseudoTTContentsReplayTemplateRecordNot:
    return RISCV::TTSFPNOT;
  default:
    return 0;
  }
}

bool isContentsMOPExecute(unsigned Opcode) {
  return RISCV::getTensixContentsMOPExecuteInfo(Opcode) != nullptr;
}

bool isTemplateControl(unsigned Opcode) {
  return Opcode == RISCV::PseudoTTReplayTemplateBegin ||
         Opcode == RISCV::PseudoTTReplayTemplateEnd ||
         Opcode == RISCV::PseudoTTReplayTemplateExecute ||
         Opcode == RISCV::PseudoTTReplayTemplateMop ||
         Opcode == RISCV::PseudoTTContentsReplayTemplateBegin ||
         Opcode == RISCV::PseudoTTContentsReplayTemplateEnd ||
         RISCV::getTensixContentsReplayExecuteInfo(Opcode) ||
         isContentsMOPExecute(Opcode);
}

bool hasTemplates(const MachineFunction &MF) {
  return any_of(MF, [](const MachineBasicBlock &MBB) {
    return any_of(MBB, [](const MachineInstr &MI) {
      return isTemplateControl(MI.getOpcode()) ||
             contentsReplayRecipeOpcode(MI.getOpcode()) ||
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
  if (isContentsMOPExecute(Opcode))
    return true;
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
  if (MI.getOpcode() == RISCV::TTMOP ||
      isContentsMOPExecute(MI.getOpcode()))
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
      Opcode == RISCV::PseudoTTContentsSFPUReplay ||
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
  return contentsReplayRecipeOpcode(MI.getOpcode()) ||
         RISCV::getTensixEncoding(MI.getOpcode()) ||
         RISCV::getTensixMachineInfoByPort(MI.getOpcode()) ||
         MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
         MI.getOpcode() == RISCV::PseudoTTSFPLOADFull ||
         MI.getOpcode() == RISCV::PseudoTTSFPLOADIndex ||
         MI.getOpcode() == RISCV::PseudoTTSFPSTORE ||
         MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord ||
         MI.getOpcode() == RISCV::PseudoTTReplayTemplateDstWord;
}

bool isContentsRegister(const MachineOperand &MO, const MachineFunction &MF,
                        bool BeforeAllocation, bool AllowEarlyClobber = false) {
  if (!MO.isReg() || !MO.getReg() || MO.getSubReg() || MO.isUndef() ||
      MO.isInternalRead() || MO.isDebug() ||
      (!AllowEarlyClobber && MO.isEarlyClobber()))
    return false;
  Register Reg = MO.getReg();
  if (!BeforeAllocation)
    return Reg.isPhysical() && RISCV::SFPRRegClass.contains(Reg);
  const auto *RC = Reg.isVirtual()
                       ? MF.getRegInfo().getRegClassOrNull(Reg)
                       : nullptr;
  return RC && RISCV::SFPRRegClass.hasSubClassEq(RC);
}

// Contents lowering attaches exactly one unknown-pointer issue-port MMO to
// begin and both execute forms. Preserve it verbatim through finalization.
// Direct MIR may omit this annotation: opcode side effects and exact implicit
// state operands still model issue ordering. A present annotation may not claim
// a different address, access, alias relation or ordering than the lowering.
Error verifyContentsReplayIssueMemory(const MachineInstr &MI) {
  if (MI.memoperands_empty())
    return Error::success();
  auto Fail = [] {
    return createStringError(
        "contents replay issue memory differs from formal lowering");
  };
  if (!MI.hasOneMemOperand())
    return Fail();
  const MachineMemOperand &MMO = **MI.memoperands_begin();
  constexpr auto Flags = MachineMemOperand::MOLoad |
                         MachineMemOperand::MOStore |
                         MachineMemOperand::MOVolatile;
  if (MMO.getFlags() != Flags || MMO.getOpaqueValue() || MMO.getOffset() ||
      MMO.getAddrSpace() || MMO.getPointerInfo().StackID ||
      MMO.getSize() != LocationSize::precise(4) ||
      MMO.getBaseAlign() != Align(4) || MMO.getAlign() != Align(4) ||
      MMO.getSuccessOrdering() != AtomicOrdering::NotAtomic ||
      MMO.getFailureOrdering() != AtomicOrdering::NotAtomic ||
      MMO.getSyncScopeID() != SyncScope::System ||
      MMO.getAAInfo() != AAMDNodes() || MMO.getRanges() ||
      MMO.getMemCacheHint())
    return Fail();
  return Error::success();
}

// Mutable defs and their old-value ties come from the static descriptor.
// Check the actual variadic capture layout before supplying its exact count
// to the implicit-effect receiver; captures do not waive effects checks.
struct ContentsExecutionLayout {
  unsigned Mutables;
  unsigned Identity;
  unsigned FirstOld;
  unsigned Roles;
};

Expected<ContentsExecutionLayout>
contentsExecutionLayout(const MachineInstr &MI, const MachineFunction &MF,
                        bool BeforeAllocation) {
  unsigned M = MI.getNumExplicitDefs();
  unsigned N = MI.getNumExplicitOperands();
  if (Error E = verifyContentsReplayIssueMemory(MI))
    return std::move(E);
  if (MI.isBundled() || MI.peekDebugInstrNum() || M < 1 || M > 8 ||
      N < 2 * M + 2)
    return createStringError("contents replay execute has invalid flat role layout");
  unsigned First = M + 2;
  unsigned Roles = N - First;
  if (Roles < M || Roles > 8 || !MI.getOperand(M).isImm() ||
      MI.getOperand(M).getImm() < 0 ||
      uint64_t(MI.getOperand(M).getImm()) >
          std::numeric_limits<uint32_t>::max() ||
      !MI.getOperand(M + 1).isImm() ||
      MI.getOperand(M + 1).getImm() != M)
    return createStringError("contents replay execute mutable count differs from its defs");
  SmallSet<Register, 8> Uses, Defs;
  for (unsigned R = 0; R < Roles; ++R) {
    const MachineOperand &Use = MI.getOperand(First + R);
    if (!isContentsRegister(Use, MF, BeforeAllocation) || !Use.isUse() ||
        Use.isImplicit() || !Uses.insert(Use.getReg()).second ||
        (R >= M && Use.isTied()))
      return createStringError("contents replay execute requires distinct contents roles");
  }
  for (unsigned R = 0; R < M; ++R) {
    const MachineOperand &Def = MI.getOperand(R);
    unsigned TiedUse = 0, TiedDef = 0;
    if (!isContentsRegister(Def, MF, BeforeAllocation,
                            /*AllowEarlyClobber=*/true) ||
        !Def.isDef() ||
        Def.isImplicit() || !Def.isEarlyClobber() ||
        !Defs.insert(Def.getReg()).second ||
        !MI.isRegTiedToUseOperand(R, &TiedUse) || TiedUse != First + R ||
        !MI.isRegTiedToDefOperand(First + R, &TiedDef) || TiedDef != R)
      return createStringError("contents replay execute requires exact early-clobber old ties");
    for (unsigned S = 0; S < Roles; ++S)
      if (S != R && Def.getReg() == MI.getOperand(First + S).getReg())
        return createStringError("contents replay execute aliases independent roles");
  }
  if (Error E = verifyTensixMachineEffects(MI, First + Roles))
    return std::move(E);
  return ContentsExecutionLayout{M, M, First, Roles};
}

enum class ContentsRole { Mutable, Capture };

// Machine scheduling may interleave scalar calculations with a recording.
// These do not issue Tensix words or touch its recorded contents/state.
bool isScalarReplayInterleave(const MachineInstr &MI,
                              const MachineFunction &MF) {
  if (MI.isTerminator() || MI.mayLoadOrStore() || !MI.memoperands_empty() ||
      MI.hasUnmodeledSideEffects())
    return false;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask())
      return false;
    if (!MO.isReg() || !MO.getReg())
      continue;
    Register Reg = MO.getReg();
    if (Reg.isPhysical()) {
      if (!RISCV::GPRRegClass.contains(Reg))
        return false;
      continue;
    }
    const auto *RC = MF.getRegInfo().getRegClassOrNull(Reg);
    if (!RC || !RISCV::GPRRegClass.hasSubClassEq(RC))
      return false;
  }
  return true;
}

// This is a pre-allocation SSA query, independently checked for MIR ingress.
// A transport copy is not permission for a final physical move: the ordinary
// no-copy receiver and the exact post-allocation template roles still apply.
bool isRecordedContents(Register Root, Register Seed, const ReplayTemplate &T,
                        const MachineFunction &MF, ContentsRole Role,
                        unsigned MutableIndex = 0) {
  const auto &MRI = MF.getRegInfo();
  SmallSet<Register, 32> Visited, Seeded;
  DenseMap<Register, SmallVector<Register, 4>> Users;
  SmallVector<Register, 16> Pending{Root};
  while (!Pending.empty()) {
    Register Reg = Pending.pop_back_val();
    if (!Visited.insert(Reg).second || Reg == Seed)
      continue;
    if (!Reg.isVirtual())
      return false;
    const MachineInstr *Def = MRI.getUniqueVRegDef(Reg);
    if (!Def)
      return false;
    SmallVector<Register, 4> Inputs;
    auto addInput = [&](unsigned Index) {
      if (!isContentsRegister(Def->getOperand(Index), MF,
                              /*BeforeAllocation=*/true))
        return false;
      Inputs.push_back(Def->getOperand(Index).getReg());
      return true;
    };
    if (Def->isPHI()) {
      if (Def->getNumExplicitOperands() < 3 ||
          Def->getNumExplicitOperands() % 2 != 1)
        return false;
      for (unsigned I = 1, E = Def->getNumExplicitOperands(); I < E; I += 2)
        if (!Def->getOperand(I + 1).isMBB() || !addInput(I))
          return false;
    } else if (Def->isCopy()) {
      if (Def->getNumExplicitOperands() != 2 || !addInput(1))
        return false;
    } else if (Role == ContentsRole::Mutable &&
               (RISCV::getTensixContentsReplayExecuteInfo(Def->getOpcode()) ||
                isContentsMOPExecute(Def->getOpcode()))) {
      auto Layout = contentsExecutionLayout(*Def, MF, true);
      if (!Layout) {
        consumeError(Layout.takeError());
        return false;
      }
      if (MutableIndex >= Layout->Mutables ||
          Def->getOperand(MutableIndex).getReg() != Reg ||
          Def->getOperand(Layout->Identity).getImm() != T.Identity ||
          !addInput(Layout->FirstOld + MutableIndex))
        return false;
    } else {
      return false;
    }
    for (Register Input : Inputs) {
      Users[Input].push_back(Reg);
      Pending.push_back(Input);
    }
  }
  if (!Visited.contains(Seed))
    return false;
  Pending.push_back(Seed);
  while (!Pending.empty()) {
    Register Reg = Pending.pop_back_val();
    if (!Seeded.insert(Reg).second)
      continue;
    auto Found = Users.find(Reg);
    if (Found != Users.end())
      append_range(Pending, Found->second);
  }
  return Seeded.size() == Visited.size();
}

Error collectContentsRecipe(ReplayTemplate &T, MachineInstr &MI,
                            MachineFunction &MF, bool BeforeAllocation) {
  unsigned NativeOpcode = contentsReplayRecipeOpcode(MI.getOpcode());
  // Ordinary numerical instructions do not carry recipe identity operands.
  if (!NativeOpcode)
    return createStringError(
        "contents replay recipe differs from its record-only owner");
  auto Identity = unsignedOperand(MI, 0);
  if (!Identity)
    return Identity.takeError();
  if (T.Mode != TensixReplayTemplateMode::RecordOnly ||
      *Identity != T.Identity || MI.isBundled() || MI.peekDebugInstrNum() ||
      !MI.memoperands_empty() ||
      MI.getNumExplicitOperands() != (NativeOpcode == RISCV::TTSFPNOT ? 2 : 3))
    return createStringError(
        "contents replay recipe differs from its record-only owner");
  if (Error E = verifyTensixMachineEffects(MI))
    return E;
  auto Mutable = unsignedOperand(MI, 1);
  if (!Mutable)
    return Mutable.takeError();
  if (*Mutable >= T.InitialMutables.size())
    return createStringError(
        "contents replay recipe destination requires a declared mutable role");
  unsigned SourceRole = *Mutable;
  if (NativeOpcode != RISCV::TTSFPNOT) {
    auto Source = unsignedOperand(MI, 2);
    if (!Source)
      return Source.takeError();
    SourceRole = *Source;
    if (SourceRole >= T.InitialMutables.size() + T.Captures.size())
      return createStringError(
          "contents replay recipe source requires a declared contents role");
  }
  T.Body.push_back(&MI);
  if (BeforeAllocation)
    return Error::success();

  // Only allocated role identities enter the encoding. A source mutable
  // reads its current runtime contents, including earlier recipe writes.
  // Recording has no numerical def: each SSA remains original until execute.
  using Operand = TensixRecordedOperand;
  Register Destination = T.InitialMutables[*Mutable];
  Register Source = SourceRole < T.InitialMutables.size()
                        ? T.InitialMutables[SourceRole]
                        : T.Captures[SourceRole - T.InitialMutables.size()];
  TensixRecordedWord Word{
      NativeOpcode,
      {{Operand::Kind::Register, Destination.id()},
       {Operand::Kind::Register, Destination.id()},
       {Operand::Kind::Register, Source.id()}}};
  if (Error E = verifyTensixRecordedWord(Word, MF))
    return E;
  T.Words.push_back(std::move(Word));
  return Error::success();
}

Error collectContentsBody(ReplayTemplate &T, MachineInstr &MI,
                          MachineFunction &MF, bool BeforeAllocation) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode == RISCV::TTSFPLOADFull || Opcode == RISCV::PseudoTTSFPLOADFull)
    return createStringError(
        "contents full LOAD is not supported in replay recording");
  bool SelfSource = Opcode == RISCV::TTSFPNOT;
  bool Supported = SelfSource || Opcode == RISCV::TTSFPIADD ||
                   Opcode == RISCV::TTSFPISUB ||
                   Opcode == RISCV::TTSFPIADDCCLT ||
                   Opcode == RISCV::TTSFPISUBCCLT ||
                   Opcode == RISCV::TTSFPIADDCCGE ||
                   Opcode == RISCV::TTSFPISUBCCGE ||
                   Opcode == RISCV::TTSFPAND || Opcode == RISCV::TTSFPOR ||
                   Opcode == RISCV::TTSFPXOR ||
                   Opcode == RISCV::TTSFPDIVP2 ||
                   Opcode == RISCV::TTSFPADDI ||
                   Opcode == RISCV::TTSFPMULI ||
                   Opcode == RISCV::TTSFPADD || Opcode == RISCV::TTSFPMUL ||
                   Opcode == RISCV::TTSFPMAD ||
                   Opcode == RISCV::TTSFPMOV ||
                   Opcode == RISCV::TTSFPMOVNeg ||
                   Opcode == RISCV::TTSFPLOADI;
  const MCInstrDesc &Desc = MI.getDesc();
  unsigned Operands = Desc.getNumOperands();
  unsigned Registers = 0;
  while (Registers < Operands && Desc.operands()[Registers].RegClass >= 0)
    ++Registers;
  auto Invalid = [] {
    return createStringError(
        "contents replay body lost its mutable chain or read-only capture");
  };
  if (T.Mode != TensixReplayTemplateMode::RecordAndExecute ||
      !Supported || MI.isBundled() ||
      !MI.memoperands_empty() || MI.getNumExplicitOperands() != Operands)
    return Invalid();
  // Native descriptors determine the exact register/immediate boundary. The
  // final recorded-word receiver checks every immediate's legal range.
  for (unsigned I = 0; I != Registers; ++I)
    if (!isContentsRegister(MI.getOperand(I), MF, BeforeAllocation) ||
        (I == 0 ? !MI.getOperand(I).isDef() : !MI.getOperand(I).isUse()))
      return Invalid();
  for (unsigned I = Registers; I != Operands; ++I)
    if (!MI.getOperand(I).isImm())
      return Invalid();
  auto Mutable = find(T.Mutables, MI.getOperand(1).getReg());
  if (Mutable == T.Mutables.end() ||
      (SelfSource && MI.getOperand(2).getReg() != *Mutable) ||
      is_contained(T.Captures, MI.getOperand(0).getReg()) ||
      (!BeforeAllocation &&
       MI.getOperand(0).getReg() !=
           T.InitialMutables[Mutable - T.Mutables.begin()]))
    return Invalid();
  for (unsigned I = 2; I < Registers; ++I)
    if (!is_contained(T.Mutables, MI.getOperand(I).getReg()) &&
        !is_contained(T.Captures, MI.getOperand(I).getReg()))
      return Invalid();
  for (auto It = T.Mutables.begin(); It != T.Mutables.end(); ++It)
    if (It != Mutable && *It == MI.getOperand(0).getReg())
      return createStringError("contents replay body aliases independent mutable roles");
  if (Error E = verifyTensixMachineEffects(MI))
    return E;
  *Mutable = MI.getOperand(0).getReg();
  T.Body.push_back(&MI);
  if (BeforeAllocation)
    return Error::success();
  // Allocation has finished. Canonicalize only an analysis-local clone; the
  // complete candidate is checked before any instruction is changed.
  MachineInstr *Canonical = MF.CloneMachineInstr(&MI);
  auto Delete = scope_exit([&] { MF.deleteMachineInstr(Canonical); });
  for (MachineOperand &MO : Canonical->operands())
    if (MO.isReg() && MO.getReg().isPhysical())
      MO.setIsRenamable(false);
  auto Word = getTensixRecordedWord(*Canonical, MF);
  if (!Word)
    return Word.takeError();
  T.Words.push_back(std::move(*Word));
  return Error::success();
}

// Data-memory writes cannot change the nine MOP cells. Admit only precise,
// nonwrapping accesses within the actual subtarget's local worker L1, or the
// compiler-owned stack. Merely being disjoint from MOP is insufficient: other
// MMIO windows can issue instructions or change control state.
bool preservesContentsMOPSetup(const MachineMemOperand &MMO,
                               const MachineFunction &MF) {
  if (!MMO.isStore())
    return false;
  if (const auto *PSV = MMO.getPseudoValue())
    return PSV->isStack() || PSV->kind() == PseudoSourceValue::FixedStack;
  const Value *Pointer = MMO.getValue();
  const DataLayout &DL = MF.getDataLayout();
  LocationSize Size = MMO.getSize();
  if (!Pointer || MMO.getAddrSpace() != 0 || DL.getPointerSizeInBits() != 32 ||
      !Size.hasValue() || !Size.isPrecise() || Size.isScalable() ||
      Size.getValue().isZero())
    return false;
  int64_t PointerOffset;
  const Value *Base =
      GetPointerBaseWithConstantOffset(Pointer, PointerOffset, DL);
  uint64_t BaseAddress = 0;
  if (!isa<ConstantPointerNull>(Base)) {
    const auto *Cast = dyn_cast<Operator>(Base);
    if (!Cast || Cast->getOpcode() != Instruction::IntToPtr)
      return false;
    KnownBits Integer = computeKnownBits(Cast->getOperand(0), DL);
    if (!Integer.isConstant() || Integer.getConstant().getActiveBits() > 32)
      return false;
    BaseAddress = Integer.getConstant().getZExtValue();
  }
  int64_t Start;
  if (AddOverflow(int64_t(BaseAddress), PointerOffset, Start) ||
      AddOverflow(Start, MMO.getOffset(), Start) || Start < 0)
    return false;
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  uint64_t End;
  if (ST.hasVendorXTTTensixWH())
    End = 1464 * 1024;
  else if (ST.hasVendorXTTTensixBH())
    End = 1536 * 1024;
  else
    return false;
  return uint64_t(Start) < End &&
         Size.getValue().getFixedValue() <= End - Start;
}

// MOP1 repeats the same contents body outer*inner times in hardware. LoopOp
// and Loop0Last name it; nonfinal outer iterations additionally need Loop1Last
// to name it. Plain NOP LoopOp1 disables alternation. The common BH/WH profile
// has static outer/inner=1..127 without instruction count overrides.
// This repeats recorded words; it does not duplicate any source loop body.
// Keep setup unique and outside every CFG cycle: hardware samples its cells
// during expansion, so reprogramming while an earlier MOP runs is not legal.
Error verifyContentsMOPProfile(MachineFunction &MF,
                               TemplateInventory &Inventory,
                               MachineDominatorTree &Dominators) {
  ReplayTemplate *Receiver = nullptr;
  SmallVector<const MachineInstr *, 4> Executes;
  for (ReplayTemplate &T : Inventory.Templates)
    for (const MachineInstr *MI : T.Executions)
      if (isContentsMOPExecute(MI->getOpcode())) {
        if (Receiver && Receiver != &T)
          return createStringError(
              "contents MOP profile requires one template");
        Receiver = &T;
        Executes.push_back(MI);
      }
  for (const auto &Binding : Inventory.MOPBindings)
    if (Inventory.Templates[Binding.Template].Contents) {
      auto *T = &Inventory.Templates[Binding.Template];
      if (Receiver && Receiver != T)
        return createStringError("contents MOP profile requires one template");
      Receiver = T;
    }
  if (!Receiver)
    return Error::success();
  Attribute Executor = MF.getFunction().getFnAttribute("tensix-executor");
  if (!Executor.isStringAttribute() || Executor.getValueAsString() != "trisc1")
    return createStringError(
        "contents MOP execution requires tensix-executor=trisc1");
  std::array<const MachineInstr *, 9> Setup{};
  unsigned Outer = 0, Inner = 0;
  auto add = [&](unsigned Cell, const MachineInstr &MI) -> Error {
    if (Cell >= Setup.size() || Setup[Cell])
      return createStringError("contents MOP profile requires unique setup");
    if (Error E = verifyTensixMachineEffects(MI))
      return E;
    Setup[Cell] = &MI;
    return Error::success();
  };
  for (const auto &MBB : MF)
    for (const auto &MI : MBB) {
      unsigned Opcode = MI.getOpcode();
      if (isContentsMOPExecute(Opcode))
        continue;
      if (Opcode == RISCV::PseudoTTMOPControlWriteImm) {
        if (MI.getNumExplicitOperands() != 4 || !MI.getOperand(2).isImm() ||
            !MI.getOperand(3).isImm() || MI.getOperand(2).getImm() < 0 ||
            MI.getOperand(2).getImm() > 1 ||
            MI.getOperand(3).getImm() < 1 ||
            MI.getOperand(3).getImm() > 127)
          return createStringError(
              "contents MOP profile requires outer and inner in [1, 127]");
        if (MI.getOperand(2).getImm() == 0)
          Outer = MI.getOperand(3).getImm();
        else
          Inner = MI.getOperand(3).getImm();
        if (Error E = add(MI.getOperand(2).getImm(), MI))
          return E;
      } else if (Opcode == RISCV::PseudoTTMOPClear) {
        if (MI.getNumExplicitOperands() != 3 || !MI.getOperand(2).isImm() ||
            MI.getOperand(2).getImm() < 2 || MI.getOperand(2).getImm() > 8 ||
            MI.getOperand(2).getImm() == 5 || MI.getOperand(2).getImm() == 7)
          return createStringError(
              "contents MOP profile requires slots5/7 binding");
        if (Error E = add(MI.getOperand(2).getImm(), MI))
          return E;
      } else if (Opcode == RISCV::PseudoTTReplayTemplateMop) {
        if (MI.getNumExplicitOperands() != 4 || !MI.getOperand(2).isImm() ||
            !MI.getOperand(3).isImm() ||
            MI.getOperand(2).getImm() != Receiver->Identity ||
            (MI.getOperand(3).getImm() != 5 &&
             MI.getOperand(3).getImm() != 7 && MI.getOperand(3).getImm() != 8))
          return createStringError(
              "contents MOP profile requires slots5/7/8 binding");
        if (Error E = add(MI.getOperand(3).getImm(), MI))
          return E;
      } else if (isMOPOwner(MI)) {
        return createStringError(
            "contents MOP requires explicit contents execution and setup");
      } else if (Opcode == RISCV::PseudoTTCoprocessorWait) {
        if (Error E = verifyTensixCoprocessorWait(
                MI, MF.getSubtarget<RISCVSubtarget>()))
          return E;
      } else if (Opcode == RISCV::PseudoTTSFPMacroAddressReset) {
        // This fixed expansion resets thread configuration, address modifiers
        // and Dst counters. It never writes the nine MOP cells. Its conservative
        // memory effects must not turn it into an unknown scalar MMIO write.
        if (MI.getNumExplicitOperands() != 0 || MI.isBundled())
          return createStringError(
              "macro address reset requires one fixed machine owner");
        if (Error E = verifyTensixMachineEffects(MI))
          return E;
        if (Error E = verifyTensixMachineOperands(
                MI, MF.getRegInfo(),
                *MF.getSubtarget<RISCVSubtarget>().getRegisterInfo()))
          return E;
      } else if (Opcode == RISCV::PseudoTTReadResponse) {
        // Full memory effects order this dependency consumer, whose only
        // emitted instruction is ADDI x0, response, 0. It performs no write.
        if (!MI.memoperands_empty())
          return createStringError(
              "read response requires a nonwriting dependency consumer");
        if (Error E = verifyTensixMachineEffects(MI))
          return E;
        if (Error E = verifyTensixMachineOperands(
                MI, MF.getRegInfo(),
                *MF.getSubtarget<RISCVSubtarget>().getRegisterInfo()))
          return E;
      } else if (MI.mayStore() && !isTemplateControl(Opcode) &&
                 !RISCV::getTensixEncoding(Opcode) &&
                 !RISCV::getTensixMachineInfoByPort(Opcode)) {
        if (MI.memoperands_empty() ||
            any_of(MI.memoperands(), [&](const auto *MMO) {
              return !preservesContentsMOPSetup(*MMO, MF);
            }))
          return createStringError(
              "contents MOP profile cannot admit unknown scalar writes");
      }
    }
  if (any_of(Setup, [](const auto *MI) { return !MI; }))
    return createStringError(
        "contents MOP profile requires all nine setup cells");
  // Every nonfinal outer iteration replaces its last LoopOp with Loop1Last.
  // A plain NOP there would lose one numerical update per nonfinal outer.
  if (Outer > 1 && Setup[8]->getOpcode() != RISCV::PseudoTTReplayTemplateMop)
    return createStringError(
        "contents MOP outer>1 requires slot8 bound to its template");
  for (const MachineInstr *Site : Setup) {
    SmallPtrSet<const MachineBasicBlock *, 16> Seen;
    SmallVector<const MachineBasicBlock *, 8> Pending;
    append_range(Pending, Site->getParent()->successors());
    while (!Pending.empty()) {
      const auto *Block = Pending.pop_back_val();
      if (Block == Site->getParent())
        return createStringError(
            "contents MOP setup cannot be inside a CFG cycle");
      if (Seen.insert(Block).second)
        append_range(Pending, Block->successors());
    }
    for (const MachineInstr *Execute : Executes)
      if (!Dominators.dominates(Site, Execute))
        return createStringError(
            "contents MOP setup must dominate every execution");
  }
  Receiver->MOPRepeated = Outer > 1 || Inner > 1;
  return Error::success();
}

Expected<TemplateInventory> collectTemplates(MachineFunction &MF,
                                            bool RequirePrepared,
                                            bool BeforeAllocation = false) {
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
      if (Opcode == RISCV::PseudoTTReplayTemplateBegin ||
          Opcode == RISCV::PseudoTTContentsReplayTemplateBegin) {
        bool Contents = Opcode == RISCV::PseudoTTContentsReplayTemplateBegin;
        if (Active)
          return createStringError("structured replay recordings cannot nest");
        unsigned Explicit = MI.getNumExplicitOperands();
        if (MI.getNumExplicitDefs() ||
            (Contents ? (Explicit < 6 || Explicit > 13) : Explicit != 4))
          return createStringError("structured replay begin has invalid controls or roles");
        unsigned MutableCount = 0;
        if (Contents) {
          if (Error E = verifyContentsReplayIssueMemory(MI))
            return std::move(E);
          auto Count = unsignedOperand(MI, 4);
          if (!Count)
            return Count.takeError();
          MutableCount = *Count;
          if (!MutableCount || MutableCount > Explicit - 5)
            return createStringError("contents replay mutable count differs from its flat roles");
          SmallSet<Register, 8> Roles;
          for (unsigned R = 5; R < Explicit; ++R)
            if (!isContentsRegister(MI.getOperand(R), MF, BeforeAllocation) ||
                !MI.getOperand(R).isUse() || MI.getOperand(R).isTied() ||
                !Roles.insert(MI.getOperand(R).getReg()).second)
              return createStringError(
                  "contents replay requires distinct allocated contents roles");
        }
        if (Error E = Contents ? verifyTensixMachineEffects(MI, Explicit)
                               : verifyTensixMachineEffects(MI))
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
        ReplayTemplate &T = Result.Templates.back();
        T.Contents = Contents;
        if (Contents) {
          for (unsigned R = 5; R < Explicit; ++R)
            (R < 5 + MutableCount ? T.Mutables : T.Captures)
                .push_back(MI.getOperand(R).getReg());
          T.InitialMutables = T.Mutables;
        }
        Result.Begins.try_emplace(&MI, Index);
        Active = Index;
        continue;
      }
      if (Opcode == RISCV::PseudoTTReplayTemplateEnd ||
          Opcode == RISCV::PseudoTTContentsReplayTemplateEnd) {
        bool Contents = Opcode == RISCV::PseudoTTContentsReplayTemplateEnd;
        if (!Active || MI.getNumExplicitOperands() < 1)
          return createStringError("structured replay end has no active record");
        ReplayTemplate &T = Result.Templates[*Active];
        unsigned Expected = Contents ? 1 + T.Mutables.size() : 1;
        if (MI.getNumExplicitDefs() || MI.getNumExplicitOperands() != Expected)
          return createStringError("contents replay end lost its actual final contents");
        auto Identity = unsignedOperand(MI, 0);
        if (!Identity)
          return Identity.takeError();
        if (*Identity != T.Identity || Contents != T.Contents)
          return createStringError("structured replay end has a different identity");
        if (Contents) {
          if (T.Body.empty())
            return createStringError("contents replay end lost its actual final contents");
          for (auto [R, Final] : enumerate(T.Mutables))
            if (!isContentsRegister(MI.getOperand(1 + R), MF, BeforeAllocation) ||
                !MI.getOperand(1 + R).isUse() ||
                MI.getOperand(1 + R).isTied() ||
                MI.getOperand(1 + R).getReg() != Final)
              return createStringError("contents replay end lost its actual final contents");
        }
        if (Error E = Contents ? verifyTensixMachineEffects(MI, Expected)
                               : verifyTensixMachineEffects(MI))
          return std::move(E);
        T.End = &MI;
        if (!Contents || !BeforeAllocation) {
          auto Effects = getTensixExplicitReplayEffects(T.Words, MF);
          if (!Effects)
            return Effects.takeError();
          T.Effects = std::move(*Effects);
        }
        Active.reset();
        continue;
      }
      if (Opcode == RISCV::PseudoTTReplayTemplateExecute ||
          RISCV::getTensixContentsReplayExecuteInfo(Opcode) ||
          isContentsMOPExecute(Opcode)) {
        bool Contents = Opcode != RISCV::PseudoTTReplayTemplateExecute;
        if (Active || (!Contents && MI.getNumExplicitOperands() != 1))
          return createStringError(
              "structured replay execution must be outside recording bodies");
        if (Contents) {
          auto Layout = contentsExecutionLayout(MI, MF, BeforeAllocation);
          if (!Layout)
            return Layout.takeError();
        }
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
      if ((contentsReplayRecipeOpcode(Opcode) ||
           Opcode == RISCV::PseudoTTReplayTemplateWord ||
           Opcode == RISCV::PseudoTTReplayTemplateDstWord) && !Active)
        return createStringError("symbolic replay word has no recording owner");
      if (!Active)
        continue;
      ReplayTemplate &T = Result.Templates[*Active];
      if (T.Contents) {
        if (MI.isDebugInstr() ||
            (!isRecordedIssue(MI) && isScalarReplayInterleave(MI, MF)))
          continue;
        Error E = T.Mode == TensixReplayTemplateMode::RecordOnly
                      ? collectContentsRecipe(T, MI, MF, BeforeAllocation)
                      : collectContentsBody(T, MI, MF, BeforeAllocation);
        if (E)
          return std::move(E);
        continue;
      }
      if (MI.isTerminator())
        return createStringError(
            "structured replay recording cannot cross scalar control flow");
      if (!isRecordedIssue(MI))
        continue;
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
    if (T.Contents && *Slot != 5 && *Slot != 7 && *Slot != 8)
      return createStringError("contents MOP binding requires slot5, slot7 or slot8");
    if (!T.End || !Dominators.dominates(T.End, MI))
      return createStringError(
          "structured replay recording must dominate its MOP binding");
    Result.MOPBindings.push_back({MI, Found->second, *Slot});
  }
  for (MachineInstr *MI : Executions) {
    bool Contents =
        RISCV::getTensixContentsReplayExecuteInfo(MI->getOpcode()) ||
        isContentsMOPExecute(MI->getOpcode());
    auto Identity = unsignedOperand(*MI, Contents ? MI->getNumExplicitDefs() : 0);
    if (!Identity)
      return Identity.takeError();
    auto Found = Result.Identities.find(*Identity);
    if (Found == Result.Identities.end())
      return createStringError("structured replay execution has no record producer");
    ReplayTemplate &T = Result.Templates[Found->second];
    if (Contents != T.Contents)
      return createStringError("structured replay execution has a different receiver");
    if (!T.End || !Dominators.dominates(T.End, MI))
      return createStringError(
          "structured replay recording must dominate its execution");
    if (Contents) {
      auto Layout = contentsExecutionLayout(*MI, MF, BeforeAllocation);
      if (!Layout)
        return Layout.takeError();
      if (Layout->Mutables != T.Mutables.size() ||
          Layout->Roles != T.Mutables.size() + T.Captures.size())
        return createStringError("contents replay execute roles differ from its record");
      for (auto [R, Mutable] : enumerate(T.Mutables)) {
        Register Old = MI->getOperand(Layout->FirstOld + R).getReg();
        if (BeforeAllocation &&
            !isRecordedContents(Old, Mutable, T, MF, ContentsRole::Mutable, R))
          return createStringError(
              "contents replay execution lost its recorded SSA contents");
        if (!BeforeAllocation &&
            (Old != T.InitialMutables[R] ||
             MI->getOperand(R).getReg() != T.InitialMutables[R]))
          return createStringError(
              "contents replay allocation differs from its recorded register roles");
      }
      for (auto [R, Capture] : enumerate(T.Captures)) {
        Register Actual =
            MI->getOperand(Layout->FirstOld + T.Mutables.size() + R).getReg();
        if (BeforeAllocation &&
            !isRecordedContents(Actual, Capture, T, MF, ContentsRole::Capture))
          return createStringError(
              "contents replay execution lost its recorded SSA contents");
        if (!BeforeAllocation && Actual != Capture)
          return createStringError(
              "contents replay allocation differs from its recorded register roles");
      }
    } else if (!hasEffects(*MI, T.Effects)) {
      if (RequirePrepared)
        return createStringError(
            "structured replay execution effects differ from its body");
      if (Error E = verifyTensixMachineEffects(*MI))
        return std::move(E);
    }
    T.Executions.push_back(MI);
    Result.Executions.try_emplace(MI, Found->second);
  }
  if (Error E = verifyContentsMOPProfile(MF, Result, Dominators))
    return std::move(E);
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
    if (T.Contents)
      continue;
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
  unsigned TailGaps = 0;
  SmallVector<TensixRecordedWord, 16> Words;
  TensixSFPUReplayEffects Effects;
  std::optional<unsigned> Start;
};

Expected<FinalTemplate> finalizeBody(const ReplayTemplate &T,
                                    MachineFunction &MF) {
  FinalTemplate Result;
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
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
    while (needsTensixReplayHazardGap(**Native, ST, State)) {
      if (Result.Words.size() >= TensixReplaySlotCount)
        return createStringError(
            "structured replay hazard repair exceeds replay capacity");
      Result.Words.push_back(Nop);
      ++Gaps;
      State = advanceTensixReplayHazard(**NativeNop, ST, State);
    }
    Result.Gaps.push_back(Gaps);
    Result.Words.push_back(Word);
    if (Result.Words.size() > TensixReplaySlotCount)
      return createStringError(
          "structured replay final words exceed replay capacity");
    State = advanceTensixReplayHazard(**Native, ST, State);
  }
  if (T.MOPRepeated) {
    // A MOP repeats the complete body with no RISCV issue between copies.
    // Drain the finite native hazard state using canonical payload NOPs. This
    // closes every re-entry, including a dependency spanning the first word;
    // the final MOP receiver independently checks the resulting finite graph.
    while (State.SFPU || State.DstCycles) {
      if (Result.Words.size() >= TensixReplaySlotCount)
        return createStringError(
            "structured replay MOP boundary repair exceeds replay capacity");
      Result.Words.push_back(Nop);
      ++Result.TailGaps;
      State = advanceTensixReplayHazard(**NativeNop, ST, State);
    }
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
                     const FinalTemplate &Final, const RISCVInstrInfo &TII,
                     bool Contents = false) {
  auto Builder = BuildMI(*Original.getParent(), Original, Original.getDebugLoc(),
                         TII.get(Contents ? RISCV::PseudoTTContentsSFPUReplay
                                          : RISCV::PseudoTTExplicitSFPUReplay));
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
    if (T.Contents) {
      retainEffects(MF, T.Effects);
      // The recorded physical identities are now fixed. Later passes may not
      // rename a body register independently of the encoded template.
      for (MachineInstr *MI : T.Body)
        for (MachineOperand &MO : MI->operands())
          if (MO.isReg() && MO.getReg().isPhysical())
            MO.setIsRenamable(false);
    }
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
    for (unsigned Gap = 0; Gap < Plan.TailGaps; ++Gap) {
      if (RecordOnly)
        insertTensixRecordedPayload(*T.End, Nop, TII, std::nullopt);
      else
        BuildMI(*T.End->getParent(), *T.End, T.End->getDebugLoc(),
                TII.get(RISCV::TTSFPNOP));
    }
    replaceControl(*T.Begin, 1, !RecordOnly, Plan, TII, T.Contents);
    BuildMI(*T.End->getParent(), *T.End, T.End->getDebugLoc(),
            TII.get(RISCV::PseudoTTReplayRecordEnd))
        .setMIFlags(T.End->getFlags());
    T.End->eraseFromParent();
    for (MachineInstr *Execution : T.Executions) {
      if (isContentsMOPExecute(Execution->getOpcode())) {
        // Its checked MOP1 profile repeats one template outer*inner times.
        // The existing MOP receiver later reconstructs the final physical
        // effects and hazard transfer from actual native configuration.
        auto Builder = BuildMI(*Execution->getParent(), *Execution,
                               Execution->getDebugLoc(), TII.get(RISCV::TTMOP));
        Builder.addImm(0)
            .addImm(0)
            .addImm(1)
            .cloneMemRefs(*Execution)
            .setMIFlags(Execution->getFlags());
        addExecutionEffects(Builder, Plan.Effects);
        Execution->eraseFromParent();
      } else {
        replaceControl(*Execution, 0, 0, Plan, TII, T.Contents);
      }
    }
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
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (MF.getInfo<RISCVMachineFunctionInfo>()->hasTensixCodegenFailed() ||
        !hasTemplates(MF))
      return false;
    auto Inventory = collectTemplates(MF, false, true);
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
