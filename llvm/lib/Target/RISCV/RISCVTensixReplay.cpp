//===-- RISCVTensixReplay.cpp -----------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixReplay.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include <array>
#include <optional>

using namespace llvm;

bool llvm::isTensixSFPUReplayCandidate(const MachineInstr &MI) {
  // These exact single-cycle forms only update their explicit LReg result.
  // In particular IADDI's shared CC-def descriptor, CC-setting IADD variants,
  // Dst operations, CONFIG, LUT and cross-lane pipelines are not admitted.
  switch (MI.getOpcode()) {
  case RISCV::TTSFPIADD:
  case RISCV::TTSFPISUB:
  case RISCV::TTSFPAND:
  case RISCV::TTSFPOR:
  case RISCV::TTSFPXOR:
  case RISCV::TTSFPNOT:
  case RISCV::TTSFPLOADI:
  case RISCV::TTSFPMOV:
  case RISCV::TTSFPMOVNeg:
  case RISCV::TTSFPMOVAll:
  case RISCV::TTSFPNOP:
    break;
  default:
    return false;
  }
  const MCInstrDesc &Desc = MI.getDesc();
  if (MI.isBundled() || !MI.memoperands_empty() || MI.peekDebugInstrNum() ||
      MI.getNumExplicitOperands() != Desc.getNumOperands() ||
      MI.getNumOperands() != Desc.getNumOperands() +
                                 Desc.implicit_uses().size() +
                                 Desc.implicit_defs().size())
    return false;
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isImm()) {
      if (I < Desc.getNumDefs())
        return false;
      continue;
    }
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead() ||
        MO.isDef() != (I < Desc.getNumDefs()) ||
        !(MO.isDef() ? RISCV::SFPRRegClass.contains(MO.getReg())
                     : RISCV::SFPRReadRegClass.contains(MO.getReg())))
      return false;
  }
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead())
      return false;
    if (MO.isDef() ? !is_contained(Desc.implicit_defs(), MO.getReg().id())
                   : !is_contained(Desc.implicit_uses(), MO.getReg().id()))
      return false;
  }
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isUse() && MO.getReg() == Reg;
        }) != 1)
      return false;
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isDef() && MO.getReg() == Reg;
        }) != 1)
      return false;
  // All result-producing admitted forms except the all-lane copy are tied to
  // their old destination. Preserve the allocated inactive-lane passthrough.
  if (Desc.getNumDefs() && MI.getOpcode() != RISCV::TTSFPMOVAll &&
      MI.getOperand(0).getReg() != MI.getOperand(1).getReg())
    return false;
  return true;
}

Expected<TensixSFPUReplayEffects> llvm::getTensixSFPUReplayEffects(
    ArrayRef<const MachineInstr *> Body) {
  if (Body.empty() || Body.size() > 32)
    return createStringError("automatic SFPU replay body must fit 32 instruction slots");
  TensixSFPUReplayEffects Effects;
  bool HasValue = false;
  for (const MachineInstr *MI : Body) {
    if (!isTensixSFPUReplayCandidate(*MI))
      return createStringError("automatic SFPU replay body has unsupported physical effects");
    // Read all operands before processing defs: destructive ties read the old
    // physical value even though the output operand is written first in MIR.
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isUse() &&
          !is_contained(Effects.Defs, MO.getReg().asMCReg()) &&
          !is_contained(Effects.Uses, MO.getReg().asMCReg()))
        Effects.Uses.push_back(MO.getReg().asMCReg());
    for (const MachineOperand &MO : MI->operands())
      if (MO.isReg() && MO.isDef()) {
        HasValue |= RISCV::SFPRRegClass.contains(MO.getReg());
        if (!is_contained(Effects.Defs, MO.getReg().asMCReg()))
          Effects.Defs.push_back(MO.getReg().asMCReg());
      }
  }
  if (!HasValue)
    return createStringError("automatic SFPU replay requires an arithmetic value sequence");
  llvm::sort(Effects.Uses);
  llvm::sort(Effects.Defs);
  return Effects;
}

namespace {
struct Issue {
  unsigned Opcode;
  unsigned Stream;
  unsigned First;
};

std::optional<Issue> getIssue(const MachineInstr &MI) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay)
    return Issue{RISCV::TTREPLAY, 0, 0};
  if (RISCV::getTensixEncoding(MI.getOpcode()))
    return Issue{MI.getOpcode(), 0, 0};
  if (const auto *Info = RISCV::getTensixMachineInfoByPort(MI.getOpcode()))
    return Issue{Info->Opcode, unsigned(MI.getOperand(3).getImm()), 4};
  if (MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
      MI.getOpcode() == RISCV::PseudoTTSFPSTORE)
    return Issue{MI.getOpcode(), 0, 0};
  return std::nullopt;
}

Expected<uint32_t> replayMask(const MachineInstr &MI, unsigned First) {
  // Logical fields: load_mode, execute_while_loading, len, start_idx.
  if (!MI.getOperand(First + 2).isImm() ||
      !MI.getOperand(First + 3).isImm())
    return createStringError("Tensix replay control must remain static");
  int64_t Length = MI.getOperand(First + 2).getImm();
  int64_t Start = MI.getOperand(First + 3).getImm();
  if (Length <= 0 || Start < 0 || Start >= 32 || Length > 32 - Start)
    return createStringError("Tensix replay range must fit 32 instruction slots");
  return uint32_t(((uint64_t(1) << Length) - 1) << Start);
}

struct Record {
  uint32_t Mask;
  unsigned Stream;
  unsigned Length;
  unsigned Count = 0;
  bool AutomaticSFPU = false;
  SmallVector<const MachineInstr *, 32> Body;
};

bool hasReplayEffects(const MachineInstr &MI, ArrayRef<MCRegister> Uses,
                      ArrayRef<MCRegister> Defs) {
  if (MI.isBundled() || !MI.memoperands_empty() ||
      MI.getNumExplicitOperands() != 4 ||
      MI.getNumOperands() != 4 + Uses.size() + Defs.size())
    return false;
  SmallVector<MCRegister, 16> ActualUses, ActualDefs;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead())
      return false;
    (MO.isDef() ? ActualDefs : ActualUses).push_back(MO.getReg().asMCReg());
  }
  llvm::sort(ActualUses);
  llvm::sort(ActualDefs);
  return ArrayRef(ActualUses) == Uses && ArrayRef(ActualDefs) == Defs;
}

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
                          MI.getOpcode() == RISCV::TTMOP ||
                          MI.getOpcode() == RISCV::TTMOP_CFG ||
                          MI.getOpcode() == RISCV::PseudoTTMOPClear;
    }
  if (HasAutomatic && HasExternalOwner)
    return createStringError("automatic SFPU replay conflicts with another replay or call owner");
  if (HasAutomatic && HasVirtual)
    return createStringError("automatic SFPU replay requires final physical register allocation");
  return Error::success();
}

// Recording is a finite straight-line issue region. Scalar address arithmetic
// may be interspersed, but no scalar control transfer may interrupt it. End
// markers survive instruction selection and scheduling until this final check.
Expected<DenseMap<const MachineInstr *, Record>>
findRecords(const MachineFunction &MF) {
  DenseMap<const MachineInstr *, Record> Ends;
  bool SeenAutomatic = false;
  for (const auto &BB : MF) {
    std::optional<Record> Active;
    std::optional<Record> Automatic;
    std::optional<TensixSFPUReplayEffects> AutomaticEffects;
    for (const auto &MI : BB) {
      if (MI.getOpcode() == RISCV::PseudoTTReplayRecordEnd) {
        if (!Active)
          return createStringError("Tensix replay end has no active record");
        if (Active->Count != Active->Length)
          return createStringError("Tensix replay final issue count " +
                                   Twine(Active->Count) +
                                   " differs from declared length " +
                                   Twine(Active->Length));
        Ends.try_emplace(&MI, *Active);
        if (Active->AutomaticSFPU) {
          auto Effects = getTensixSFPUReplayEffects(Active->Body);
          if (!Effects)
            return Effects.takeError();
          Automatic = *Active;
          AutomaticEffects = std::move(*Effects);
        }
        Active.reset();
        continue;
      }
      if (Active && (MI.isTerminator() || MI.isCall() || MI.isInlineAsm()))
        return createStringError("Tensix replay recording cannot cross scalar control flow");
      auto Issued = getIssue(MI);
      if (!Issued) {
        if (Active && Active->AutomaticSFPU)
          return createStringError("automatic SFPU replay recording must be uninterrupted native issue");
        // A completed automatic template cannot be reused after an observable
        // scalar/control boundary. Its bit contents may persist, but that is
        // outside the single-block candidate contract proved by selection.
        Automatic.reset();
        AutomaticEffects.reset();
        continue;
      }
      if (Issued->Stream > 2)
        return createStringError("unknown Tensix replay instruction stream");
      if (Issued->Opcode == RISCV::TTREPLAY) {
        auto Mask = replayMask(MI, Issued->First);
        if (!Mask)
          return Mask.takeError();
        if (!MI.getOperand(Issued->First).isImm())
          return createStringError("Tensix replay control must remain static");
        if (Active && Active->Stream == Issued->Stream)
          return createStringError("Tensix replay recording cannot contain REPLAY");
        bool IsAutomatic = MI.getOpcode() == RISCV::PseudoTTSFPUReplay;
        bool Load = MI.getOperand(Issued->First).getImm() != 0;
        if (IsAutomatic) {
          if (!MI.getOperand(1).isImm() ||
              MI.getOperand(0).getImm() != (Load ? 1 : 0) ||
              MI.getOperand(1).getImm() != (Load ? 1 : 0) ||
              MI.getOperand(3).getImm() != 0)
            return createStringError("automatic SFPU replay requires static record-and-execute or execute in bank zero");
          if (Load) {
            if (SeenAutomatic)
              return createStringError("automatic SFPU replay permits one record per function");
            SeenAutomatic = true;
            SmallVector<MCRegister, 2> Uses{RISCV::TT_CONFIG, RISCV::TT_ISSUE};
            SmallVector<MCRegister, 1> Defs{RISCV::TT_ISSUE};
            llvm::sort(Uses);
            if (!hasReplayEffects(MI, Uses, Defs))
              return createStringError("automatic SFPU replay record header has unexpected physical effects");
          } else {
            if (!Automatic || Automatic->Mask != *Mask || !AutomaticEffects)
              return createStringError("automatic SFPU replay execution requires its exact preceding local record");
            if (!hasReplayEffects(MI, AutomaticEffects->Uses,
                                  AutomaticEffects->Defs))
              return createStringError("automatic SFPU replay execution physical effects differ from its recording");
          }
        }
        if (Load) {
          if (Active)
            return createStringError("Tensix replay records cannot overlap");
          Active = Record{*Mask, Issued->Stream,
              unsigned(MI.getOperand(Issued->First + 2).getImm())};
          Active->AutomaticSFPU = IsAutomatic;
        }
        continue;
      }
      if (Automatic && !isTensixSFPUReplayCandidate(MI)) {
        Automatic.reset();
        AutomaticEffects.reset();
      }
      if (!Active || Issued->Stream != Active->Stream)
        continue;
      if (Active->AutomaticSFPU) {
        if (!isTensixSFPUReplayCandidate(MI))
          return createStringError("automatic SFPU replay body has unsupported physical effects");
        Active->Body.push_back(&MI);
        ++Active->Count;
        continue;
      }
      if ((RISCV::getTensixEncoding(Issued->Opcode) &&
           !RISCV::getTensixMachineInfo(Issued->Opcode)) ||
          MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
          MI.getOpcode() == RISCV::PseudoTTSFPSTORE)
        return createStringError("SFPU replay recording requires a typed SSA preservation ABI");
      if (Issued->Opcode == RISCV::TTMOP)
        return createStringError("Tensix replay recording cannot contain MOP");
      ++Active->Count;
    }
    if (Active)
      return createStringError("Tensix replay recording requires an end marker in its basic block");
  }
  return Ends;
}

struct State {
  std::array<uint32_t, 3> Recorded{};
  uint8_t MOPKnown = 0;
  std::array<uint32_t, 7> MOPRequires{};
  bool operator==(const State &Other) const {
    return Recorded == Other.Recorded && MOPKnown == Other.MOPKnown &&
           MOPRequires == Other.MOPRequires;
  }
  bool operator!=(const State &Other) const { return !(*this == Other); }

  static State top() {
    State S;
    S.Recorded.fill(~uint32_t(0));
    S.MOPKnown = 0x7f;
    return S;
  }
  void meet(const State &Other) {
    for (unsigned I = 0; I != 3; ++I)
      Recorded[I] &= Other.Recorded[I];
    MOPKnown &= Other.MOPKnown;
    // Either path's reference can execute. All possible referenced entries
    // must have been recorded on every incoming path.
    for (unsigned I = 0; I != 7; ++I)
      MOPRequires[I] |= Other.MOPRequires[I];
  }
};

// A scalar polling helper may retain a configured MOP and completed replay
// records. Prove that property from its exact definition instead of trusting a
// name or memory attribute: even a readnone call may modify target state.
// Keep the admitted arithmetic scalar and free of possible runtime libcalls.
// This is deliberately narrower than a general interprocedural effect summary.
class CallPreservation {
  DenseMap<const Function *, bool> Proven;
  SmallPtrSet<const Function *, 8> Visiting;

  static bool isScalar(Type *Ty) {
    return Ty->isPointerTy() ||
           (Ty->isIntegerTy() && Ty->getIntegerBitWidth() <= 64);
  }

  static bool isPureIntrinsic(const IntrinsicInst &II) {
    // Inspect the intrinsic identity, not overridable call-site attributes.
    switch (II.getIntrinsicID()) {
    case Intrinsic::assume:
      return true;
    case Intrinsic::expect:
    case Intrinsic::expect_with_probability:
    case Intrinsic::bswap:
    case Intrinsic::bitreverse:
      return II.getType()->isIntegerTy() &&
             II.getType()->getIntegerBitWidth() <= 64;
    default:
      return false;
    }
  }

  bool preserves(const Instruction &I) {
    if (const auto *Call = dyn_cast<CallBase>(&I)) {
      if (!isa<CallInst>(I) || Call->isInlineAsm() ||
          Call->hasOperandBundles() ||
          Call->getCallingConv() != CallingConv::C)
        return false;
      if (const auto *II = dyn_cast<IntrinsicInst>(Call))
        return isPureIntrinsic(*II);
      const Function *Callee = Call->getCalledFunction();
      return Callee && preserves(*Callee);
    }
    if (const auto *Load = dyn_cast<LoadInst>(&I))
      return !Load->isAtomic() && isScalar(Load->getType());
    if (isa<FenceInst, CondBrInst, UncondBrInst, SwitchInst, ReturnInst>(I))
      return true;
    switch (I.getOpcode()) {
    case Instruction::PHI:
    case Instruction::Select:
    case Instruction::Freeze:
    case Instruction::ICmp:
    case Instruction::GetElementPtr:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::BitCast:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
      return isScalar(I.getType()) &&
             llvm::all_of(I.operands(), [](const Use &Operand) {
               return isScalar(Operand->getType());
             });
    default:
      // This includes stores, atomics, inline assembly, target operations and
      // arithmetic that could introduce a call with an unproved machine ABI.
      return false;
    }
  }

  bool preserves(const Function &F) {
    if (auto It = Proven.find(&F); It != Proven.end())
      return It->second;
    if (!F.hasExactDefinition() || F.isInterposable() ||
        F.hasFnAttribute(Attribute::Naked) || F.hasFnAttribute("interrupt") ||
        F.getCallingConv() != CallingConv::C || F.isVarArg() ||
        !(F.getReturnType()->isVoidTy() || isScalar(F.getReturnType())) ||
        !llvm::all_of(F.args(), [](const Argument &Arg) {
          return isScalar(Arg.getType());
        }) ||
        !Visiting.insert(&F).second)
      return false;
    bool Result = llvm::all_of(instructions(F), [this](const Instruction &I) {
      return preserves(I);
    });
    Visiting.erase(&F);
    Proven[&F] = Result;
    return Result;
  }

public:
  bool preserves(const MachineInstr &MI) {
    if (!MI.isCall() || MI.isInlineAsm())
      return false;
    const Function *Callee = nullptr;
    for (const MachineOperand &Operand : MI.operands()) {
      if (!Operand.isGlobal())
        continue;
      const auto *Candidate = dyn_cast<Function>(Operand.getGlobal());
      if (!Candidate || Callee || Operand.getOffset() != 0)
        return false;
      Callee = Candidate;
    }
    return Callee && preserves(*Callee);
  }
};

Error transfer(const MachineBasicBlock &BB, State &S,
               const DenseMap<const MachineInstr *, Record> &Ends,
               CallPreservation &Calls, bool Check) {
  for (const auto &MI : BB) {
    if (auto It = Ends.find(&MI); It != Ends.end()) {
      S.Recorded[It->second.Stream] |= It->second.Mask;
      continue;
    }
    if (MI.isCall() || MI.isInlineAsm()) {
      if (!Calls.preserves(MI))
        S = State();
      continue;
    }
    const auto *MOP = RISCV::getTensixMachineInfoByMop(MI.getOpcode());
    bool Clear = MI.getOpcode() == RISCV::PseudoTTMOPClear;
    if (MOP || Clear) {
      unsigned Slot = MI.getOperand(2).getImm();
      if (Slot < 2 || Slot > 8)
        return createStringError("Tensix MOP instruction slot must be in [2, 8]");
      uint32_t Required = 0;
      if (MOP && MOP->Opcode == RISCV::TTREPLAY) {
        auto Mask = replayMask(MI, 3);
        if (!Mask)
          return Mask.takeError();
        if (MI.getOperand(3).getImm())
          return createStringError("Tensix MOP replay slot must execute, not record");
        Required = *Mask;
      }
      S.MOPKnown |= 1u << (Slot - 2);
      S.MOPRequires[Slot - 2] = Required;
      continue;
    }
    auto Issued = getIssue(MI);
    if (!Issued)
      continue;
    if (Issued->Opcode == RISCV::TTREPLAY &&
        MI.getOperand(Issued->First).getImm() == 0) {
      auto Mask = replayMask(MI, Issued->First);
      if (!Mask)
        return Mask.takeError();
      if (Check && (S.Recorded[Issued->Stream] & *Mask) != *Mask)
        return createStringError("Tensix replay execution references slots not recorded on every incoming path");
    }
    if (Issued->Opcode == RISCV::TTMOP && Check) {
      if (Issued->Stream != 0)
        return createStringError("remote Tensix MOP requires a verified template preservation ABI");
      if (S.MOPKnown != 0x7f)
        return createStringError("Tensix MOP execution requires all seven instruction slots to be configured");
      for (uint32_t Required : S.MOPRequires)
        if ((S.Recorded[0] & Required) != Required)
          return createStringError("Tensix MOP replay reference is not recorded on every incoming path");
    }
  }
  return Error::success();
}
} // namespace

Error llvm::verifyTensixReplay(const MachineFunction &MF) {
  if (MF.empty())
    return Error::success();
  if (Error E = verifyAutomaticOwner(MF))
    return E;
  auto Records = findRecords(MF);
  if (!Records)
    return Records.takeError();
  SmallPtrSet<const MachineBasicBlock *, 16> Reachable;
  SmallVector<const MachineBasicBlock *> Work{&MF.front()};
  while (!Work.empty()) {
    const auto *BB = Work.pop_back_val();
    if (!Reachable.insert(BB).second)
      continue;
    llvm::append_range(Work, BB->successors());
  }
  DenseMap<const MachineBasicBlock *, State> Entries, Exits;
  CallPreservation Calls;
  for (const auto *BB : Reachable)
    Exits[BB] = State::top();
  bool Changed;
  do {
    Changed = false;
    for (const auto &BB : MF) {
      if (!Reachable.contains(&BB))
        continue;
      State Incoming = &BB == &MF.front() ? State() : State::top();
      for (const auto *Pred : BB.predecessors())
        if (Reachable.contains(Pred))
          Incoming.meet(Exits.lookup(Pred));
      Entries[&BB] = Incoming;
      if (Error E = transfer(BB, Incoming, *Records, Calls, false))
        return E;
      if (Incoming != Exits.lookup(&BB)) {
        Exits[&BB] = Incoming;
        Changed = true;
      }
    }
  } while (Changed);
  for (const auto &BB : MF) {
    if (!Reachable.contains(&BB))
      continue;
    State Incoming = Entries.lookup(&BB);
    if (Error E = transfer(BB, Incoming, *Records, Calls, true))
      return E;
  }
  return Error::success();
}
