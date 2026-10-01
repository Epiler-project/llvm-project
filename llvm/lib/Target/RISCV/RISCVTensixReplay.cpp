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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/TensixMOPSequence.h"
#include <array>
#include <limits>
#include <optional>

using namespace llvm;

namespace {
bool isDirectSFPUArithmetic(unsigned Opcode) {
  // These exact forms only update their explicit LReg result. ADD/MUL/MAD
  // retain their two-cycle result dependency in the replay effects below.
  // In particular IADDI's shared CC-def descriptor, CC-setting IADD variants,
  // Dst operations, CONFIG, LUT and cross-lane pipelines are not admitted.
  switch (Opcode) {
  case RISCV::TTSFPADD:
  case RISCV::TTSFPMUL:
  case RISCV::TTSFPMAD:
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
    return true;
  default:
    return false;
  }
}
bool isDirectSFPURecordedOpcode(unsigned Opcode) {
  // Explicit replay receives static Dst fields and bounded native state
  // operations. Their descriptors own all physical effects, reconstructed at
  // execution against current contents. Automatic arithmetic mining remains
  // separate; MOP and CC-stack words are not admitted. Dynamic Dst captures
  // use their scalar-aware receiver and project to these same native effects.
  switch (Opcode) {
  case RISCV::TTSFPLOAD:
  case RISCV::TTSFPSTORE:
  case RISCV::TTSFPENCC:
  case RISCV::TTSFPCONFIGC11:
  case RISCV::TTSFPCONFIGC12:
  case RISCV::TTSFPCONFIGC13:
  case RISCV::TTSFPCONFIGC14:
  case RISCV::TTSFPCONFIGLane:
  case RISCV::TTSFPCONFIGReset:
    return true;
  default:
    return isDirectSFPUArithmetic(Opcode);
  }
}

// Ordinary engine words use the same native descriptor and MC field contract
// as direct issue. Replay/MOP controls are not linear recording payloads.
bool isOrdinaryRecordedOpcode(unsigned Opcode) {
  return RISCV::getTensixMachineInfo(Opcode) && Opcode != RISCV::TTREPLAY &&
         Opcode != RISCV::TTMOP && Opcode != RISCV::TTMOP_CFG;
}

bool isRecordedOpcode(unsigned Opcode) {
  return isDirectSFPURecordedOpcode(Opcode) ||
         isOrdinaryRecordedOpcode(Opcode);
}

bool hasOrdinaryIssueMemory(const MachineInstr &MI) {
  return all_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
    return MMO->isVolatile() && MMO->isLoad() && MMO->isStore() &&
           !MMO->isAtomic() && MMO->getSize() == LocationSize::precise(4);
  });
}

bool isOrdinaryRecordedCandidate(const MachineInstr &MI) {
  const MCInstrDesc &Desc = MI.getDesc();
  return isOrdinaryRecordedOpcode(MI.getOpcode()) && !MI.isBundled() &&
         !MI.peekDebugInstrNum() && hasOrdinaryIssueMemory(MI) &&
         MI.getNumExplicitOperands() == Desc.getNumOperands() &&
         MI.getNumOperands() == Desc.getNumOperands() +
                                   Desc.implicit_uses().size() +
                                   Desc.implicit_defs().size();
}

bool isDirectSFPURecordedCandidate(const MachineInstr &MI) {
  if (!isDirectSFPURecordedOpcode(MI.getOpcode()))
    return false;
  const MCInstrDesc &Desc = MI.getDesc();
  if (MI.isBundled() || !MI.memoperands_empty() || MI.peekDebugInstrNum() ||
      MI.getNumExplicitOperands() != Desc.getNumOperands() ||
      MI.getNumOperands() != Desc.getNumOperands() +
                                 Desc.implicit_uses().size() +
                                 Desc.implicit_defs().size())
    return false;
  if (MI.getOpcode() == RISCV::TTSFPADD || MI.getOpcode() == RISCV::TTSFPMUL ||
      MI.getOpcode() == RISCV::TTSFPMAD) {
    // Only direct sources/destinations are covered by these physical effects.
    // Bits 0/1 negate operands; the indirect-address modes need a wider proof.
    const MachineOperand &Mode = MI.getOperand(Desc.getNumOperands() - 1);
    if (!Mode.isImm() || Mode.getImm() < 0 || Mode.getImm() > 3)
      return false;
  }
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
} // namespace

bool llvm::isTensixSFPUReplayCandidate(const MachineInstr &MI) {
  return isDirectSFPUArithmetic(MI.getOpcode()) &&
         isDirectSFPURecordedCandidate(MI);
}

namespace {
void composeSFPUReplayWord(const MachineInstr &MI,
                           TensixSFPUReplayEffects &Effects) {
  Effects.PendingMADResult = MI.getOpcode() == RISCV::TTSFPADD ||
                                     MI.getOpcode() == RISCV::TTSFPMUL ||
                                     MI.getOpcode() == RISCV::TTSFPMAD
                                 ? MI.getOperand(0).getReg().asMCReg()
                                 : MCRegister();
  // Reads precede writes, including destructive old-value ties and implicit
  // native state. Word identity never captures the runtime register value.
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isUse() &&
        !is_contained(Effects.Defs, MO.getReg().asMCReg()) &&
        !is_contained(Effects.Uses, MO.getReg().asMCReg()))
      Effects.Uses.push_back(MO.getReg().asMCReg());
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() &&
        !is_contained(Effects.Defs, MO.getReg().asMCReg()))
      Effects.Defs.push_back(MO.getReg().asMCReg());
}
} // namespace

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
    composeSFPUReplayWord(*MI, Effects);
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

Expected<MachineInstr *>
llvm::createTensixRecordedInstruction(const TensixRecordedWord &Word,
                                      MachineFunction &MF) {
  if (!isRecordedOpcode(Word.Opcode))
    return createStringError(
        "explicit SFPU replay word is not a supported direct native issue");
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  const auto &TII = *ST.getInstrInfo();
  const auto &TRI = *ST.getRegisterInfo();
  const MCInstrDesc &Desc = TII.get(Word.Opcode);
  if (Word.Operands.size() != Desc.getNumOperands())
    return createStringError(
        "explicit SFPU replay word has the wrong native operand count");
  MCInst Native;
  Native.setOpcode(Word.Opcode);
  unsigned Captures = 0;
  for (auto [I, Operand] : enumerate(Word.Operands)) {
    if (Operand.OperandKind == TensixRecordedOperand::Kind::CapturedScalar) {
      bool Load = Word.Opcode == RISCV::TTSFPLOAD;
      bool Store = Word.Opcode == RISCV::TTSFPSTORE;
      if (Operand.Value <= 0 || uint64_t(Operand.Value) > UINT32_MAX)
        return createStringError("replay Dst capture has an invalid scalar register");
      Register Reg{unsigned(Operand.Value)};
      const auto &MRI = MF.getRegInfo();
      if (Reg.isVirtual() && Reg.virtRegIndex() >= MRI.getNumVirtRegs())
        return createStringError("replay Dst capture has a foreign virtual GPR");
      const auto *RC = Reg.isVirtual() ? MRI.getRegClassOrNull(Reg) : nullptr;
      if ((!Load && !Store) || I != (Load ? 2u : 1u) || !Word.Capture ||
          (Reg.isVirtual() ? !RC || !RISCV::GPRRegClass.hasSubClassEq(RC)
                           : !RISCV::GPRRegClass.contains(Reg)))
        return createStringError("replay Dst capture requires its exact scalar field");
      ++Captures;
      // Only a descriptor/effect projection, never an emitted or recorded word.
      // The captured address cannot affect register or pipeline effect shape.
      Native.addOperand(MCOperand::createImm(0));
      continue;
    }
    if (Operand.OperandKind != TensixRecordedOperand::Kind::Register &&
        Operand.OperandKind != TensixRecordedOperand::Kind::Immediate)
      return createStringError(
          "explicit SFPU replay word has an unknown operand kind");
    bool IsRegister = Desc.operands()[I].RegClass >= 0;
    if (IsRegister !=
        (Operand.OperandKind == TensixRecordedOperand::Kind::Register))
      return createStringError(
          "explicit SFPU replay word differs from its native operand kind");
    if (IsRegister) {
      if (Operand.Value <= 0 || uint64_t(Operand.Value) >= TRI.getNumRegs())
        return createStringError(
            "explicit SFPU replay word requires a physical register");
      Native.addOperand(
          MCOperand::createReg(MCRegister(unsigned(Operand.Value))));
    } else {
      Native.addOperand(MCOperand::createImm(Operand.Value));
    }
  }
  if (Captures != unsigned(Word.Capture != nullptr))
    return createStringError("replay captured fields lost their recording identity");
  if (Error E = RISCV::verifyTensixMCInstruction(Native, TII, TRI))
    return std::move(E);
  MachineInstr *MI = MF.CreateMachineInstr(Desc, DebugLoc());
  MachineInstrBuilder Builder(MF, MI);
  for (auto [I, Operand] : enumerate(Word.Operands)) {
    if (Operand.OperandKind == TensixRecordedOperand::Kind::Register)
      Builder.addReg(unsigned(Operand.Value), I < Desc.getNumDefs()
                                                  ? RegState::Define
                                                  : RegState::NoFlags);
    else
      Builder.addImm(Operand.OperandKind ==
                             TensixRecordedOperand::Kind::CapturedScalar
                         ? 0
                         : Operand.Value);
  }
  if (Error E = verifyTensixMachineOperands(*MI, MF.getRegInfo(), TRI)) {
    MF.deleteMachineInstr(MI);
    return std::move(E);
  }
  if (!isDirectSFPURecordedCandidate(*MI) &&
      !isOrdinaryRecordedCandidate(*MI)) {
    MF.deleteMachineInstr(MI);
    return createStringError(
        "explicit SFPU replay word has unsupported native effects");
  }
  return MI;
}

Error llvm::verifyTensixRecordedWord(const TensixRecordedWord &Word,
                                     const MachineFunction &MF) {
  // The temporary instruction is never inserted into the machine graph.
  auto &Allocator = const_cast<MachineFunction &>(MF);
  auto MI = createTensixRecordedInstruction(Word, Allocator);
  if (!MI)
    return MI.takeError();
  Allocator.deleteMachineInstr(*MI);
  return Error::success();
}

namespace {
bool isDstPayload(unsigned Opcode) {
  return Opcode == RISCV::PseudoTTReplayTemplateDstWord ||
         Opcode == RISCV::PseudoTTSFPUDstRecordWord;
}
Expected<TensixRecordedWord> readDstCapture(const MachineInstr &MI,
                                           const MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if (MI.isBundled() || MI.peekDebugInstrNum())
    return createStringError("replay Dst capture has unmodeled instruction effects");
  if (Error E = verifyTensixMachineEffects(MI))
    return std::move(E);
  if (Error E = verifyTensixMachineOperands(
          MI, MF.getRegInfo(), *ST.getRegisterInfo()))
    return std::move(E);
  bool Payload = isDstPayload(MI.getOpcode());
  unsigned First = MI.getOpcode() == RISCV::PseudoTTReplayTemplateDstWord ? 3 : 2;
  bool Load = Payload ? MI.getOperand(First).getImm() == RISCV::TTSFPLOAD
                      : MI.getOpcode() == RISCV::PseudoTTSFPLOAD;
  if (Payload && MI.getOperand(First).getImm() != RISCV::TTSFPLOAD &&
      MI.getOperand(First).getImm() != RISCV::TTSFPSTORE)
    return createStringError("replay Dst payload must encode LOAD or STORE");
  if (Payload && !MI.hasOneMemOperand())
    return createStringError("replay Dst capture lost its instruction-port store");
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (!MMO->isVolatile() || !MMO->isStore() || MMO->isLoad() ||
        MMO->isAtomic() || MMO->getSize() != LocationSize::precise(4))
      return createStringError("replay Dst capture has a foreign memory effect");
  unsigned Offset = Payload ? First + 3 : Load ? 4 : 3;
  unsigned Data = Payload ? First + 1 : Load ? 0 : 2;
  auto Physical = [&](unsigned Index) -> int64_t {
    return Payload ? MI.getOperand(Index).getImm()
                   : MI.getOperand(Index).getReg().id();
  };
  TensixRecordedWord Word{Load ? RISCV::TTSFPLOAD : RISCV::TTSFPSTORE, {}, &MI};
  Word.Operands.push_back({TensixRecordedOperand::Kind::Register, Physical(Data)});
  if (Load)
    Word.Operands.push_back({TensixRecordedOperand::Kind::Register,
                             Physical(Payload ? First + 2 : 3)});
  else if (Payload && Physical(First + 2) != Physical(Data))
    return createStringError("replay STORE payload has a noncanonical old field");
  Word.Operands.push_back({TensixRecordedOperand::Kind::CapturedScalar,
                           MI.getOperand(Offset).getReg().id()});
  Word.Operands.push_back({TensixRecordedOperand::Kind::Immediate,
                           MI.getOperand(Offset + 1).getImm()});
  Word.Operands.push_back({TensixRecordedOperand::Kind::Immediate,
                           MI.getOperand(Offset + 2).getImm()});
  if (Error E = verifyTensixRecordedWord(Word, MF))
    return std::move(E);
  return Word;
}
} // namespace

MachineInstrBuilder llvm::insertTensixRecordedPayload(
    MachineInstr &Before, const TensixRecordedWord &Word,
    const RISCVInstrInfo &TII, std::optional<unsigned> Identity) {
  bool Dynamic = Word.Capture != nullptr;
  unsigned Opcode = Dynamic
      ? (Identity ? RISCV::PseudoTTReplayTemplateDstWord
                  : RISCV::PseudoTTSFPUDstRecordWord)
      : (Identity ? RISCV::PseudoTTReplayTemplateWord
                  : RISCV::PseudoTTSFPURecordWord);
  auto Builder = BuildMI(*Before.getParent(), Before, Before.getDebugLoc(),
                         TII.get(Opcode));
  if (Dynamic) {
    bool Load = Word.Opcode == RISCV::TTSFPLOAD;
    unsigned Scratch = Before.getOpcode() == RISCV::PseudoTTSFPLOAD ? 1 : 0;
    for (unsigned I : {Scratch, Scratch + 1})
      Builder.addReg(Before.getOperand(I).getReg(),
                     RegState::Define | RegState::EarlyClobber);
    if (Identity)
      Builder.addImm(*Identity);
    Builder.addImm(Word.Opcode).addImm(Word.Operands[0].Value);
    Builder.addImm(Word.Operands[Load ? 1 : 0].Value);
    Builder.addReg(unsigned(Word.Operands[Load ? 2 : 1].Value));
    Builder.addImm(Word.Operands[Load ? 3 : 2].Value);
    Builder.addImm(Word.Operands[Load ? 4 : 3].Value);
    if (Before.memoperands_empty())
      Builder.addMemOperand(Before.getMF()->getMachineMemOperand(
          MachinePointerInfo(),
          MachineMemOperand::MOStore | MachineMemOperand::MOVolatile, 4,
          Align(4)));
    else
      Builder.cloneMemRefs(Before);
  } else {
    if (Identity)
      Builder.addImm(*Identity);
    Builder.addImm(Word.Opcode);
    for (const auto &Operand : Word.Operands)
      Builder.addImm(Operand.Value);
  }
  return Builder;
}

Expected<TensixRecordedWord>
llvm::getTensixRecordedWord(const MachineInstr &MI, const MachineFunction &MF) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
      MI.getOpcode() == RISCV::PseudoTTSFPSTORE || isDstPayload(MI.getOpcode()))
    return readDstCapture(MI, MF);
  bool IsTemplate = MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord;
  bool IsPayload = MI.getOpcode() == RISCV::PseudoTTSFPURecordWord || IsTemplate;
  unsigned OpcodeField = IsTemplate ? 1 : 0;
  if (MI.isBundled() || MI.peekDebugInstrNum() ||
      (!MI.memoperands_empty() &&
       (IsPayload || !isOrdinaryRecordedCandidate(MI))))
    return createStringError(
        "explicit SFPU recorded word has unmodeled instruction effects");
  unsigned Opcode = MI.getOpcode();
  if (IsPayload) {
    if (MI.getNumExplicitOperands() <= OpcodeField ||
        !MI.getOperand(OpcodeField).isImm() ||
        MI.getOperand(OpcodeField).getImm() < 0 ||
        uint64_t(MI.getOperand(OpcodeField).getImm()) >
            std::numeric_limits<unsigned>::max())
      return createStringError(
          "explicit SFPU record payload requires a native opcode");
    Opcode = unsigned(MI.getOperand(OpcodeField).getImm());
  }
  if (!isRecordedOpcode(Opcode))
    return createStringError(
        "explicit SFPU replay word is not a supported direct native issue");
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  const MCInstrDesc &Desc = ST.getInstrInfo()->get(Opcode);
  unsigned First = IsPayload ? OpcodeField + 1 : 0;
  if (MI.getNumExplicitOperands() != First + Desc.getNumOperands())
    return createStringError(
        "explicit SFPU record payload has the wrong native operand count");
  if (IsPayload) {
    unsigned Uses = 0, Defs = 0;
    for (const MachineOperand &MO : MI.implicit_operands()) {
      if (!MO.isReg() || MO.getReg() != RISCV::TT_ISSUE || MO.getSubReg() ||
          MO.isUndef() || MO.isInternalRead() || MO.isEarlyClobber() ||
          MO.isRenamable())
        return createStringError(
            "record-only word must have only issue ordering effects");
      (MO.isDef() ? Defs : Uses)++;
    }
    if (Uses != 1 || Defs != 1 ||
        MI.getNumOperands() != MI.getNumExplicitOperands() + 2)
      return createStringError(
          "record-only word must have exact issue ordering effects");
  } else {
    if (!isDirectSFPURecordedCandidate(MI) &&
        !isOrdinaryRecordedCandidate(MI))
      return createStringError(
          "explicit SFPU replay word has unsupported native effects");
    if (Error E = verifyTensixMachineEffects(MI))
      return std::move(E);
    if (Error E = verifyTensixMachineOperands(MI, MF.getRegInfo(),
                                              *ST.getRegisterInfo()))
      return std::move(E);
  }
  TensixRecordedWord Word{Opcode, {}};
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(First + I);
    bool IsRegister = Desc.operands()[I].RegClass >= 0;
    if (IsPayload) {
      if (!MO.isImm())
        return createStringError(
            "record-only native operands must be immediate payload fields");
      Word.Operands.push_back({IsRegister
                                   ? TensixRecordedOperand::Kind::Register
                                   : TensixRecordedOperand::Kind::Immediate,
                               MO.getImm()});
    } else if (IsRegister) {
      if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
          MO.isUndef() || MO.isInternalRead() || MO.isRenamable())
        return createStringError(
            "explicit SFPU replay requires exact physical operands");
      Word.Operands.push_back(
          {TensixRecordedOperand::Kind::Register, MO.getReg().id()});
    } else {
      if (!MO.isImm())
        return createStringError(
            "explicit SFPU replay requires native immediate fields");
      Word.Operands.push_back(
          {TensixRecordedOperand::Kind::Immediate, MO.getImm()});
    }
  }
  if (Error E = verifyTensixRecordedWord(Word, MF))
    return std::move(E);
  return Word;
}

Expected<TensixSFPUReplayEffects>
llvm::getTensixExplicitReplayEffects(ArrayRef<TensixRecordedWord> Words,
                                     const MachineFunction &MF) {
  if (Words.empty() || Words.size() > TensixReplaySlotCount)
    return createStringError(
        "explicit SFPU replay body must fit 32 instruction slots");
  TensixSFPUReplayEffects Effects;
  auto &Allocator = const_cast<MachineFunction &>(MF);
  for (const auto &Word : Words) {
    auto MI = createTensixRecordedInstruction(Word, Allocator);
    if (!MI)
      return MI.takeError();
    composeSFPUReplayWord(**MI, Effects);
    Allocator.deleteMachineInstr(*MI);
  }
  // The control itself reads configuration/issue state, even a one-word NOP
  // recording. No arithmetic-output profitability condition applies here.
  for (MCRegister Reg :
       {MCRegister(RISCV::TT_CONFIG), MCRegister(RISCV::TT_ISSUE)})
    if (!is_contained(Effects.Uses, Reg))
      Effects.Uses.push_back(Reg);
  if (!is_contained(Effects.Defs, MCRegister(RISCV::TT_ISSUE)))
    Effects.Defs.push_back(RISCV::TT_ISSUE);
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
  if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
      MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay)
    return Issue{RISCV::TTREPLAY, 0, 0};
  if (MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
      MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord)
    return Issue{MI.getOpcode(), 0, 0};
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
  if (MI.getNumExplicitOperands() < First + 4 ||
      !MI.getOperand(First).isImm() || !MI.getOperand(First + 1).isImm() ||
      !MI.getOperand(First + 2).isImm() || !MI.getOperand(First + 3).isImm())
    return createStringError("Tensix replay control must remain static");
  if (MI.getOperand(First).getImm() < 0 || MI.getOperand(First).getImm() > 1)
    return createStringError("Tensix replay load field must be a bit");
  int64_t Length = MI.getOperand(First + 2).getImm();
  int64_t Start = MI.getOperand(First + 3).getImm();
  if (Length <= 0 || Start < 0 || Start >= TensixReplaySlotCount ||
      Length > TensixReplaySlotCount - Start)
    return createStringError(
        "Tensix replay range must fit 32 instruction slots");
  return uint32_t(((uint64_t(1) << Length) - 1) << Start);
}

struct Record {
  uint32_t Mask;
  unsigned Stream;
  unsigned Length;
  unsigned Count = 0;
  bool AutomaticSFPU = false;
  const MachineInstr *Header = nullptr;
  unsigned Start = 0;
  bool ExecuteWhileLoading = false;
  bool HasSFPU = false;
  SmallVector<const MachineInstr *, 32> Body;
  // Unknown ordinary words remain ordinary. SFPU identities must always be
  // known, and are interned within this analysis only.
  SmallVector<std::optional<unsigned>, 32> Words;
};

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
  DenseMap<const MachineInstr *, Record> Ends;
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

bool hasReplayEffects(const MachineInstr &MI, ArrayRef<MCRegister> Uses,
                      ArrayRef<MCRegister> Defs, bool AllowMemory = false) {
  if (MI.isBundled() || (!AllowMemory && !MI.memoperands_empty()) ||
      MI.getNumExplicitOperands() != 4 ||
      MI.getNumOperands() != 4 + Uses.size() + Defs.size())
    return false;
  SmallVector<MCRegister, 16> ActualUses, ActualDefs;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead() ||
        MO.isRenamable())
      return false;
    (MO.isDef() ? ActualDefs : ActualUses).push_back(MO.getReg().asMCReg());
  }
  llvm::sort(ActualUses);
  llvm::sort(ActualDefs);
  return ArrayRef(ActualUses) == Uses && ArrayRef(ActualDefs) == Defs;
}

Error verifyOriginalReplayControl(const MachineInstr &MI,
                                  const MachineFunction &MF) {
  // Normalization specializes the broad ordinary descriptor, but cannot
  // launder a malformed native instruction into a valid private pseudo.
  if (Error E = verifyTensixMachineEffects(MI))
    return E;
  return verifyTensixMachineOperands(
      MI, MF.getRegInfo(),
      *MF.getSubtarget<RISCVSubtarget>().getRegisterInfo());
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
                          MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay ||
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
Expected<Inventory> findRecords(const MachineFunction &MF,
                                bool RequireNormalized) {
  Inventory Result;
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
          return createStringError(
              "Tensix replay final issue count " + Twine(Active->Count) +
              " differs from declared length " + Twine(Active->Length));
        Result.Ends.try_emplace(&MI, *Active);
        Result.Analysis.Storage.RecordWrites.try_emplace(
            &MI, TensixReplayStorageAnalysis::Range{Active->Stream,
                                                   Active->Mask});
        if (Active->AutomaticSFPU) {
          auto Effects = getTensixSFPUReplayEffects(Active->Body);
          if (!Effects)
            return Effects.takeError();
          Automatic = *Active;
          AutomaticEffects = std::move(*Effects);
        } else if (Active->HasSFPU ||
                   Active->Header->getOpcode() ==
                       RISCV::PseudoTTExplicitSFPUReplay) {
          if (Active->Stream != 0 ||
              (Active->Header->getOpcode() != RISCV::TTREPLAY &&
               Active->Header->getOpcode() !=
                   RISCV::PseudoTTExplicitSFPUReplay) ||
              (Active->HasSFPU &&
               !MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU()))
            return createStringError(
                "SFPU replay recording requires the local bound physical ABI");
          if (RequireNormalized &&
              Active->Header->getOpcode() != RISCV::PseudoTTExplicitSFPUReplay)
            return createStringError("explicit SFPU replay recording was not "
                                     "normalized before machine optimization");
          if (Active->Header->getOperand(1).getImm() < 0 ||
              Active->Header->getOperand(1).getImm() > 1)
            return createStringError(
                "explicit SFPU record execute-while-loading must be a bit");
          if (Active->Header->getOpcode() == RISCV::TTREPLAY)
            if (Error E = verifyOriginalReplayControl(*Active->Header, MF))
              return std::move(E);
          TensixExplicitReplayRecord Public{Active->Header,
                                            &MI,
                                            Active->Stream,
                                            Active->Start,
                                            Active->ExecuteWhileLoading,
                                            Active->Body,
                                            {}};
          for (std::optional<unsigned> Word : Active->Words) {
            if (!Word)
              return createStringError("mixed ordinary and SFPU replay words "
                                       "need an executed-effect receiver");
            Public.Words.push_back(Result.Words[*Word]);
          }
          if (Active->Header->getOpcode() ==
              RISCV::PseudoTTExplicitSFPUReplay) {
            SmallVector<MCRegister, 2> Uses{RISCV::TT_CONFIG, RISCV::TT_ISSUE};
            SmallVector<MCRegister, 1> Defs{RISCV::TT_ISSUE};
            llvm::sort(Uses);
            if (!hasReplayEffects(*Active->Header, Uses, Defs, true))
              return createStringError("explicit SFPU replay record header has "
                                       "unexpected physical effects");
          }
          Result.Analysis.Records.push_back(std::move(Public));
        }
        Active.reset();
        continue;
      }
      if (Active && (MI.isTerminator() || MI.isCall() || MI.isInlineAsm()))
        return createStringError(
            "Tensix replay recording cannot cross scalar control flow");
      if (Active && MI.getOpcode() == RISCV::PseudoTTReplayTemplateBegin)
        return createStringError("raw and structured replay recordings cannot nest");
      auto Issued = getIssue(MI);
      if (!Issued) {
        if (Active && Active->AutomaticSFPU)
          return createStringError("automatic SFPU replay recording must be "
                                   "uninterrupted native issue");
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
          return createStringError(
              "Tensix replay recording cannot contain REPLAY");
        bool IsAutomatic = MI.getOpcode() == RISCV::PseudoTTSFPUReplay;
        bool Load = MI.getOperand(Issued->First).getImm() != 0;
        if (IsAutomatic) {
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
            if (!hasReplayEffects(MI, Uses, Defs))
              return createStringError("automatic SFPU replay record header "
                                       "has unexpected physical effects");
          } else {
            if (!Automatic || Automatic->Mask != *Mask || !AutomaticEffects)
              return createStringError(
                  "automatic SFPU replay execution requires its exact "
                  "preceding local record");
            if (!hasReplayEffects(MI, AutomaticEffects->Uses,
                                  AutomaticEffects->Defs))
              return createStringError(
                  "automatic SFPU replay execution physical effects differ "
                  "from its recording");
            Result.Analysis.ExecutionEffects.try_emplace(&MI,
                                                         *AutomaticEffects);
          }
        }
        if (Load) {
          if (Active)
            return createStringError("Tensix replay records cannot overlap");
          Active = Record{*Mask, Issued->Stream,
                          unsigned(MI.getOperand(Issued->First + 2).getImm())};
          Active->AutomaticSFPU = IsAutomatic;
          Active->Header = &MI;
          Active->Start = unsigned(MI.getOperand(Issued->First + 3).getImm());
          Active->ExecuteWhileLoading =
              MI.getOperand(Issued->First + 1).getImm() != 0;
        }
        continue;
      }
      if (Automatic && !isTensixSFPUReplayCandidate(MI)) {
        Automatic.reset();
        AutomaticEffects.reset();
      }
      bool IsPayload = MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
                       MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord;
      if (IsPayload && (!Active || Active->ExecuteWhileLoading ||
                        Active->AutomaticSFPU || Active->Stream != 0))
        return createStringError(
            "record-only payload must belong to a local record-only region");
      if (!Active || Issued->Stream != Active->Stream)
        continue;
      if (Active->AutomaticSFPU) {
        if (!isTensixSFPUReplayCandidate(MI))
          return createStringError(
              "automatic SFPU replay body has unsupported physical effects");
        Active->Body.push_back(&MI);
        ++Active->Count;
        continue;
      }
      if (Issued->Opcode == RISCV::TTMOP)
        return createStringError("Tensix replay recording cannot contain MOP");
      bool IsSFPU =
                    (RISCV::getTensixEncoding(Issued->Opcode) &&
                     !RISCV::getTensixMachineInfo(Issued->Opcode)) ||
                    MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
                    MI.getOpcode() == RISCV::PseudoTTSFPSTORE;
      Active->Body.push_back(&MI);
      bool Received = IsPayload || IsSFPU || isOrdinaryRecordedCandidate(MI) ||
                      Active->Header->getOpcode() ==
                          RISCV::PseudoTTExplicitSFPUReplay;
      if (Received) {
        auto Word = getTensixRecordedWord(MI, MF);
        if (!Word)
          return Word.takeError();
        IsSFPU = !isOrdinaryRecordedOpcode(Word->Opcode);
        if (IsSFPU &&
            !MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU())
          return createStringError(
              "SFPU replay recording requires a typed SSA preservation ABI");
        if (RequireNormalized && !Active->ExecuteWhileLoading && !IsPayload &&
            (IsSFPU || Active->Header->getOpcode() ==
                           RISCV::PseudoTTExplicitSFPUReplay))
          return createStringError("record-only SFPU body requires "
                                   "nonexecuting native payload words");
        Active->HasSFPU |= IsSFPU;
        Active->Words.push_back(Result.intern(std::move(*Word)));
      } else {
        Active->Words.push_back(std::nullopt);
      }
      ++Active->Count;
    }
    if (Active)
      return createStringError(
          "Tensix replay recording requires an end marker in its basic block");
  }
  return Result;
}

struct SlotContents {
  // Known means a known symbolic encoding/effect shape. Captured scalar bits
  // can differ across dynamic visits to the same record site. The hardware
  // holds the latest actual generation; this analysis does not know those bits.
  enum class Knowledge { Top, Known, Unknown };
  Knowledge Content = Knowledge::Unknown;
  unsigned Word = 0;
  bool MayContainSFPU = false;
  bool operator==(const SlotContents &Other) const {
    return Content == Other.Content &&
           (Content != Knowledge::Known || Word == Other.Word) &&
           MayContainSFPU == Other.MayContainSFPU;
  }
  void meet(const SlotContents &Other) {
    MayContainSFPU |= Other.MayContainSFPU;
    if (Other.Content == Knowledge::Top)
      return;
    if (Content == Knowledge::Top) {
      Content = Other.Content;
      Word = Other.Word;
    } else if (Content != Knowledge::Known ||
               Other.Content != Knowledge::Known || Word != Other.Word) {
      Content = Knowledge::Unknown;
    }
  }
};

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

struct State {
  std::array<uint32_t, 3> Recorded{};
  std::array<std::array<SlotContents, TensixReplaySlotCount>, 3> Contents{};
  uint8_t MOPKnown = 0;
  std::array<uint32_t, 7> MOPRequires{};
  std::array<SmallVector<unsigned, 2>, 7> MOPPrograms;
  std::array<MOPControlValue, 2> MOPControls;
  MOPControlValue MOPMaskHi;
  bool operator==(const State &Other) const {
    return Recorded == Other.Recorded && Contents == Other.Contents &&
           MOPKnown == Other.MOPKnown && MOPRequires == Other.MOPRequires &&
           MOPPrograms == Other.MOPPrograms && MOPControls == Other.MOPControls &&
           MOPMaskHi == Other.MOPMaskHi;
  }
  bool operator!=(const State &Other) const { return !(*this == Other); }

  static State top() {
    State S;
    S.Recorded.fill(~uint32_t(0));
    for (auto &Stream : S.Contents)
      for (auto &Slot : Stream)
        Slot.Content = SlotContents::Knowledge::Top;
    S.MOPKnown = 0x7f;
    for (auto &Control : S.MOPControls)
      Control.Content = MOPControlValue::Knowledge::Top;
    S.MOPMaskHi.Content = MOPControlValue::Knowledge::Top;
    return S;
  }
  void meet(const State &Other) {
    for (unsigned I = 0; I != 3; ++I) {
      Recorded[I] &= Other.Recorded[I];
      for (unsigned J = 0; J != TensixReplaySlotCount; ++J)
        Contents[I][J].meet(Other.Contents[I][J]);
    }
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
          Call->hasOperandBundles() || Call->getCallingConv() != CallingConv::C)
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
        !llvm::all_of(
            F.args(),
            [](const Argument &Arg) { return isScalar(Arg.getType()); }) ||
        !Visiting.insert(&F).second)
      return false;
    bool Result = llvm::all_of(
        instructions(F), [this](const Instruction &I) { return preserves(I); });
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

bool mayContainSFPU(const State &S, unsigned Stream, uint32_t Mask) {
  for (unsigned I = 0; I != TensixReplaySlotCount; ++I)
    if ((Mask & (uint32_t(1) << I)) && S.Contents[Stream][I].MayContainSFPU)
      return true;
  return false;
}

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
  const auto &TRI = *MF.getSubtarget<RISCVSubtarget>().getRegisterInfo();
  for (unsigned I = 0; I != Result.Hazards.Inputs.size(); ++I) {
    TensixReplayHazardState Input = hazardBasis(I);
    Result.Hazards.Inputs[I] = {
        advanceTensixReplayHazard(**Native, TRI, Input),
        needsTensixReplayHazardGap(**Native, TRI, Input)};
  }
  MF.deleteMachineInstr(*Native);
  return Result;
}

// Machine-specific interpretation of the shared value-only sequencer graph.
// Recorded-range admission, physical uses/defs and hazard transfer remain here;
// count/flag/mask/slot selection belongs exclusively to TensixMOPSequence.
class MOPSequenceEffects {
  const State &S;
  Inventory &Records;
  MachineFunction &MF;

  Expected<MOPSummary> program(const Inventory::MOPProgram &Program) {
    SmallVector<unsigned, 32> Words(Program.Words.begin(), Program.Words.end());
    if (Program.Range) {
      if ((S.Recorded[0] & Program.Range) != Program.Range)
        return createStringError("Tensix MOP replay reference is not "
                                 "recorded on every incoming path");
      for (unsigned I = 0; I != TensixReplaySlotCount; ++I) {
        if (!(Program.Range & (1u << I)))
          continue;
        const auto &Word = S.Contents[0][I];
        if (Word.Content != SlotContents::Knowledge::Known)
          return createStringError("SFPU MOP execution has differing or unknown "
                                   "replay words on incoming paths");
        Words.push_back(Word.Word);
      }
    }
    MOPSummary Result;
    for (unsigned Index : Words) {
      auto Word = summarizeMOPWord(Records.Words[Index], MF);
      if (!Word)
        return Word.takeError();
      Result = sequence(Result, *Word);
    }
    return Result;
  }

public:
  MOPSequenceEffects(const State &S, Inventory &Records, MachineFunction &MF)
      : S(S), Records(Records), MF(MF) {}

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

Error describeSequenceError(Error E, unsigned Kind, const State &S) {
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

bool mopMayContainSFPU(const State &S, const Inventory &Records) {
  for (const auto &Programs : S.MOPPrograms)
    for (unsigned ID : Programs) {
      const auto &P = Records.MOPPrograms[ID];
      if (P.Range && mayContainSFPU(S, 0, P.Range))
        return true;
      for (unsigned Word : P.Words)
        if (!isOrdinaryRecordedOpcode(Records.Words[Word].Opcode))
          return true;
    }
  return false;
}

Expected<TensixMOPExecutionEffects>
summarizeMOPExecution(const MachineInstr &MI, const State &S,
                       Inventory &Records) {
  if (MI.getOpcode() != RISCV::TTMOP ||
      !MI.getMF()->getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU())
    return createStringError("SFPU MOP execution requires the local bound ABI "
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
  MOPSequenceEffects Effects(S, Records,
                            *const_cast<MachineFunction *>(MI.getMF()));
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

Error transfer(const MachineBasicBlock &BB, State &S, Inventory &Records,
               CallPreservation &Calls, bool Check, bool RequireNormalized) {
  for (const auto &MI : BB) {
    if (auto It = Records.Ends.find(&MI); It != Records.Ends.end()) {
      const Record &Record = It->second;
      S.Recorded[Record.Stream] |= Record.Mask;
      if (!Record.AutomaticSFPU)
        for (auto [Offset, Word] : enumerate(Record.Words)) {
          auto &Slot = S.Contents[Record.Stream][Record.Start + Offset];
          Slot.Content = Word ? SlotContents::Knowledge::Known
                              : SlotContents::Knowledge::Unknown;
          Slot.Word = Word.value_or(0);
          Slot.MayContainSFPU =
              Word && !isOrdinaryRecordedOpcode(Records.Words[*Word].Opcode);
        }
      continue;
    }
    if (MI.isCall() || MI.isInlineAsm()) {
      if (!Calls.preserves(MI))
        S = State();
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
        auto Mask = replayMask(MI, 3);
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
    auto Issued = getIssue(MI);
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
    if (Issued->Opcode == RISCV::TTREPLAY &&
        MI.getOperand(Issued->First).getImm() == 0) {
      auto Mask = replayMask(MI, Issued->First);
      if (!Mask)
        return Mask.takeError();
      if (Check)
        Records.Analysis.Storage.Executions.try_emplace(
            &MI, TensixReplayStorageAnalysis::Range{Issued->Stream, *Mask});
      if (Check && (S.Recorded[Issued->Stream] & *Mask) != *Mask)
        return createStringError("Tensix replay execution references slots not "
                                 "recorded on every incoming path");
      if (!Check || MI.getOpcode() == RISCV::PseudoTTSFPUReplay)
        continue;
      bool IsExplicit = MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay;
      bool HasSFPU = mayContainSFPU(S, Issued->Stream, *Mask);
      if (!HasSFPU && !IsExplicit) {
        continue;
      }
      if (Issued->Stream != 0 ||
          (MI.getOpcode() != RISCV::TTREPLAY && !IsExplicit) ||
          (HasSFPU && !MI.getMF()
               ->getInfo<RISCVMachineFunctionInfo>()
               ->usesBoundTensixSFPU()))
        return createStringError(
            "SFPU replay execution requires the local bound physical ABI");
      if (MI.getOperand(1).getImm() != 0)
        return createStringError("explicit SFPU replay execution must not set "
                                 "execute-while-loading");
      if (RequireNormalized && !IsExplicit)
        return createStringError("explicit SFPU replay execution was not "
                                 "normalized before machine optimization");
      if (!IsExplicit)
        if (Error E = verifyOriginalReplayControl(MI, *MI.getMF()))
          return E;
      SmallVector<TensixRecordedWord, 32> Words;
      for (unsigned I = 0; I != TensixReplaySlotCount; ++I) {
        if (!(*Mask & (uint32_t(1) << I)))
          continue;
        const SlotContents &Slot = S.Contents[Issued->Stream][I];
        if (Slot.Content != SlotContents::Knowledge::Known)
          return createStringError("SFPU replay execution has differing or "
                                   "unknown templates on incoming paths");
        Words.push_back(Records.Words[Slot.Word]);
      }
      auto Effects = getTensixExplicitReplayEffects(Words, *MI.getMF());
      if (!Effects)
        return Effects.takeError();
      if (IsExplicit &&
          !hasReplayEffects(MI, Effects->Uses, Effects->Defs, true))
        return createStringError("explicit SFPU replay execution physical "
                                 "effects differ from its reaching words");
      Records.Analysis.ExecutionEffects.try_emplace(&MI, std::move(*Effects));
      Records.Analysis.ExecutionWords.try_emplace(&MI, std::move(Words));
    }
    if (Issued->Opcode == RISCV::TTMOP && Check) {
      Records.Analysis.Storage.MOPExecutions.try_emplace(&MI, S.MOPRequires);
      if (Issued->Stream != 0)
        return createStringError(
            "remote Tensix MOP requires a verified template preservation ABI");
      if (S.MOPKnown != 0x7f)
        return createStringError("Tensix MOP execution requires all seven "
                                 "instruction slots to be configured");
      if (mopMayContainSFPU(S, Records)) {
        auto Effects = summarizeMOPExecution(MI, S, Records);
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
          if ((S.Recorded[0] & Required) != Required)
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
  auto Records = findRecords(MF, RequireNormalized);
  if (!Records)
    return Records.takeError();
  auto Templates = getTensixReplayTemplateStorage(MF, RequireNormalized);
  if (!Templates)
    return Templates.takeError();
  for (const auto &[Identity, Words] : Templates->SymbolicWords) {
    auto &Symbolic = Records->SymbolicWords[Identity];
    for (const auto &Word : Words)
      Symbolic.push_back(Records->intern(Word));
  }
  for (const auto &Fixed : Templates->FixedRecords) {
    uint32_t Mask = uint32_t(((uint64_t(1) << Fixed.Words.size()) - 1)
                             << Fixed.Start);
    Record Overlay{Mask, 0, unsigned(Fixed.Words.size())};
    Overlay.Header = Fixed.Begin;
    Overlay.Start = Fixed.Start;
    for (const auto &Word : Fixed.Words)
      Overlay.Words.push_back(Records->intern(Word));
    // This is a physical view, not another normalization owner. In particular
    // a fixed symbolic begin is never rewritten as an ordinary raw header here.
    Records->Ends.try_emplace(Fixed.End, std::move(Overlay));
  }
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
      if (Error E =
              transfer(BB, Incoming, *Records, Calls, false, RequireNormalized))
        return std::move(E);
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
    if (Error E =
            transfer(BB, Incoming, *Records, Calls, true, RequireNormalized))
      return std::move(E);
  }
  return std::move(Records->Analysis);
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
