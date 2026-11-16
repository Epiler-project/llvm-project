//===-- RISCVTensixExplicitReplay.cpp ------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixExplicitReplay.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixBoundLowering.h"
#include "RISCVTensixContentsGroup.h"
#include "RISCVTensixReplay.h"
#include "RISCVTensixReplayTemplate.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCInst.h"
#include <limits>

using namespace llvm;

namespace {
bool isDirectSFPURecordedOpcode(unsigned Opcode) {
  // Explicit replay receives static Dst fields and bounded native state
  // operations. Their descriptors own all physical effects, reconstructed at
  // execution against current contents. Automatic arithmetic mining remains
  // separate; MOP and CC-stack words are not admitted. Dynamic Dst captures
  // use their scalar-aware receiver and project to these same native effects.
  switch (Opcode) {
  case RISCV::TTSFPIADDCCLT:
  case RISCV::TTSFPISUBCCLT:
  case RISCV::TTSFPIADDCCGE:
  case RISCV::TTSFPISUBCCGE:
  case RISCV::TTSFPSHFT2Copy4:
  case RISCV::TTSFPSHFT2Rotate4:
  case RISCV::TTSFPSHFT2:
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

bool isRecordedOpcode(unsigned Opcode) {
  return isDirectSFPURecordedOpcode(Opcode) || isOrdinaryRecordedOpcode(Opcode);
}

bool hasOrdinaryIssueMemory(const MachineInstr &MI) {
  return all_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
    return MMO->isVolatile() && MMO->isLoad() && MMO->isStore() &&
           !MMO->isAtomic() && MMO->getSize() == LocationSize::precise(4);
  });
}

bool isDstPayload(unsigned Opcode) {
  return Opcode == RISCV::PseudoTTReplayTemplateDstWord ||
         Opcode == RISCV::PseudoTTSFPUDstRecordWord;
}

Expected<unsigned> receiveRecordOnlyOpcode(const MachineInstr &MI,
                                           const MachineFunction &MF) {
  bool Template = MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord;
  unsigned OpcodeField = Template ? 1 : 0;
  if (MI.getNumOperands() <= OpcodeField ||
      MI.getNumExplicitOperands() <= OpcodeField ||
      !MI.getOperand(OpcodeField).isImm() ||
      MI.getOperand(OpcodeField).getImm() < 0 ||
      uint64_t(MI.getOperand(OpcodeField).getImm()) >
          std::numeric_limits<unsigned>::max())
    return createStringError(
        "explicit SFPU record payload requires a native opcode");
  unsigned Opcode = unsigned(MI.getOperand(OpcodeField).getImm());
  if (!isRecordedOpcode(Opcode))
    return createStringError(
        "explicit SFPU replay word is not a supported direct native issue");
  const auto &Desc =
      MF.getSubtarget<RISCVSubtarget>().getInstrInfo()->get(Opcode);
  unsigned Explicit = OpcodeField + 1 + Desc.getNumOperands();
  if (MI.getNumOperands() < Explicit || MI.getNumExplicitOperands() != Explicit)
    return createStringError(
        "explicit SFPU record payload has the wrong native operand count");
  unsigned Uses = 0, Defs = 0;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || MO.getReg() != RISCV::TT_ISSUE || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isRenamable())
      return createStringError(
          "record-only word must have only issue ordering effects");
    (MO.isDef() ? Defs : Uses)++;
  }
  if (Uses != 1 || Defs != 1 || MI.getNumOperands() != Explicit + 2)
    return createStringError(
        "record-only word must have exact issue ordering effects");
  return Opcode;
}

Expected<TensixRecordedWord> readDstCapture(const MachineInstr &MI,
                                            const MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if ((MI.isBundled() && !isTensixReplayBundleMember(MI)) ||
      MI.peekDebugInstrNum())
    return createStringError(
        "replay Dst capture has unmodeled instruction effects");
  if (Error E = verifyTensixMachineEffects(MI))
    return std::move(E);
  if (Error E = verifyTensixMachineOperands(MI, MF.getRegInfo(),
                                            *ST.getRegisterInfo()))
    return std::move(E);
  bool Payload = isDstPayload(MI.getOpcode());
  unsigned First =
      MI.getOpcode() == RISCV::PseudoTTReplayTemplateDstWord ? 3 : 2;
  bool Load = Payload ? MI.getOperand(First).getImm() == RISCV::TTSFPLOAD
                      : MI.getOpcode() == RISCV::PseudoTTSFPLOAD;
  if (Payload && MI.getOperand(First).getImm() != RISCV::TTSFPLOAD &&
      MI.getOperand(First).getImm() != RISCV::TTSFPSTORE)
    return createStringError("replay Dst payload must encode LOAD or STORE");
  if (Payload && !MI.hasOneMemOperand())
    return createStringError(
        "replay Dst capture lost its instruction-port store");
  for (const MachineMemOperand *MMO : MI.memoperands())
    if (!MMO->isVolatile() || !MMO->isStore() || MMO->isLoad() ||
        MMO->isAtomic() || MMO->getSize() != LocationSize::precise(4))
      return createStringError(
          "replay Dst capture has a foreign memory effect");
  unsigned Offset = Payload ? First + 3 : Load ? 4 : 3;
  unsigned Data = Payload ? First + 1 : Load ? 0 : 2;
  auto Physical = [&](unsigned Index) -> int64_t {
    return Payload ? MI.getOperand(Index).getImm()
                   : MI.getOperand(Index).getReg().id();
  };
  TensixRecordedWord Word{Load ? RISCV::TTSFPLOAD : RISCV::TTSFPSTORE, {}, &MI};
  Word.Operands.push_back(
      {TensixRecordedOperand::Kind::Register, Physical(Data)});
  if (Load)
    Word.Operands.push_back({TensixRecordedOperand::Kind::Register,
                             Physical(Payload ? First + 2 : 3)});
  else if (Payload && Physical(First + 2) != Physical(Data))
    return createStringError(
        "replay STORE payload has a noncanonical old field");
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

bool llvm::isDirectSFPUArithmetic(unsigned Opcode) {
  // These exact forms only update their explicit LReg result. ADD/MUL/MAD
  // retain their two-cycle result dependency in the replay effects below.
  // In particular IADDI's shared CC-def descriptor, CC-setting IADD variants,
  // Dst operations, CONFIG, LUT and cross-lane pipelines are not admitted.
  switch (Opcode) {
  case RISCV::TTSFPADD:
  case RISCV::TTSFPMUL:
  case RISCV::TTSFPMAD:
  case RISCV::TTSFPDIVP2:
  case RISCV::TTSFPADDI:
  case RISCV::TTSFPMULI:
  case RISCV::TTSFPMUL24:
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

// Ordinary engine words use the same native descriptor and MC field contract
// as direct issue. Replay/MOP controls are not linear recording payloads.
bool llvm::isOrdinaryRecordedOpcode(unsigned Opcode) {
  return RISCV::getTensixMachineInfo(Opcode) && Opcode != RISCV::TTREPLAY &&
         Opcode != RISCV::TTMOP && Opcode != RISCV::TTMOP_CFG;
}

bool llvm::isOrdinaryRecordedCandidate(const MachineInstr &MI) {
  const MCInstrDesc &Desc = MI.getDesc();
  return isOrdinaryRecordedOpcode(MI.getOpcode()) &&
         (!MI.isBundled() || isTensixReplayBundleMember(MI)) &&
         !MI.peekDebugInstrNum() && hasOrdinaryIssueMemory(MI) &&
         MI.getNumExplicitOperands() == Desc.getNumOperands() &&
         MI.getNumOperands() == Desc.getNumOperands() +
                                    Desc.implicit_uses().size() +
                                    Desc.implicit_defs().size();
}

bool llvm::isDirectSFPURecordedCandidate(const MachineInstr &MI) {
  if (!isDirectSFPURecordedOpcode(MI.getOpcode()))
    return false;
  const MCInstrDesc &Desc = MI.getDesc();
  if ((MI.isBundled() && !isTensixReplayBundleMember(MI)) ||
      !MI.memoperands_empty() || MI.peekDebugInstrNum() ||
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
        MO.isUndef() || MO.isEarlyClobber() ||
        (MO.isInternalRead() && !isTensixReplayBundleMember(MI)) ||
        MO.isDef() != (I < Desc.getNumDefs()) ||
        !(MO.isDef() ? RISCV::SFPRRegClass.contains(MO.getReg())
                     : RISCV::SFPRReadRegClass.contains(MO.getReg())))
      return false;
  }
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() ||
        (MO.isInternalRead() && !isTensixReplayBundleMember(MI)))
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

void llvm::composeSFPUReplayWord(const MachineInstr &MI,
                                 const RISCVSubtarget &ST,
                                 TensixSFPUReplayEffects &Effects) {
  Effects.PendingSFPU =
      advanceTensixReplayHazard(MI, ST, {Effects.PendingSFPU, 0}).SFPU;
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
        return createStringError(
            "replay Dst capture has an invalid scalar register");
      Register Reg{unsigned(Operand.Value)};
      const auto &MRI = MF.getRegInfo();
      if (Reg.isVirtual() && Reg.virtRegIndex() >= MRI.getNumVirtRegs())
        return createStringError(
            "replay Dst capture has a foreign virtual GPR");
      const auto *RC = Reg.isVirtual() ? MRI.getRegClassOrNull(Reg) : nullptr;
      if ((!Load && !Store) || I != (Load ? 2u : 1u) || !Word.Capture ||
          (Reg.isVirtual() ? !RC || !RISCV::GPRRegClass.hasSubClassEq(RC)
                           : !RISCV::GPRRegClass.contains(Reg)))
        return createStringError(
            "replay Dst capture requires its exact scalar field");
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
    return createStringError(
        "replay captured fields lost their recording identity");
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

MachineInstrBuilder llvm::insertTensixRecordedPayload(
    MachineInstr &Before, const TensixRecordedWord &Word,
    const RISCVInstrInfo &TII, std::optional<unsigned> Identity) {
  bool Dynamic = Word.Capture != nullptr;
  unsigned Opcode = Dynamic ? (Identity ? RISCV::PseudoTTReplayTemplateDstWord
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
  if (getTensixContentsSelectedImmediateOpcodeCount(MI.getOpcode()))
    return createStringError(
        "contents selected immediate has no replay capture receiver");
  if (MI.getOpcode() == RISCV::TTSFPLOADFull ||
      MI.getOpcode() == RISCV::PseudoTTSFPLOADFull)
    return createStringError(
        "contents full LOAD is not supported in replay recording");
  if (MI.getOpcode() == RISCV::PseudoTTSFPLOADIndex)
    return createStringError(
        "contents paired LOAD scalar issue has no replay capture receiver");
  if (MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
      MI.getOpcode() == RISCV::PseudoTTSFPSTORE || isDstPayload(MI.getOpcode()))
    return readDstCapture(MI, MF);
  bool IsTemplate = MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord;
  bool IsPayload =
      MI.getOpcode() == RISCV::PseudoTTSFPURecordWord || IsTemplate;
  unsigned OpcodeField = IsTemplate ? 1 : 0;
  if ((MI.isBundled() && !isTensixReplayBundleMember(MI)) ||
      MI.peekDebugInstrNum() ||
      (!MI.memoperands_empty() &&
       (IsPayload || !isOrdinaryRecordedCandidate(MI))))
    return createStringError(
        "explicit SFPU recorded word has unmodeled instruction effects");
  unsigned Opcode = MI.getOpcode();
  if (IsPayload) {
    auto Received = receiveRecordOnlyOpcode(MI, MF);
    if (!Received)
      return Received.takeError();
    Opcode = *Received;
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
    for (const MachineOperand &MO : MI.implicit_operands())
      if (MO.isReg() && MO.isInternalRead() &&
          !isTensixReplayBundleMember(MI))
        return createStringError(
            "record-only word must have only issue ordering effects");
  } else {
    if (!isDirectSFPURecordedCandidate(MI) && !isOrdinaryRecordedCandidate(MI))
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
          MO.isUndef() ||
          (MO.isInternalRead() && !isTensixReplayBundleMember(MI)) ||
          MO.isRenamable())
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
    composeSFPUReplayWord(**MI, MF.getSubtarget<RISCVSubtarget>(), Effects);
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
// Copied byte-for-byte from the frozen A1 CallPreservation class. Delete the
// old class when integration transfers ownership; do not retain two
// classifiers.
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

} // namespace

bool llvm::TensixReplayBankAnalysis::SlotContents::operator==(
    const SlotContents &Other) const {
  return Content == Other.Content &&
         (Content != Knowledge::Known || Word == Other.Word) &&
         MayContainSFPU == Other.MayContainSFPU;
}

void llvm::TensixReplayBankAnalysis::SlotContents::meet(
    const SlotContents &Other) {
  MayContainSFPU |= Other.MayContainSFPU;
  if (Other.Content == Knowledge::Top)
    return;
  if (Content == Knowledge::Top) {
    Content = Other.Content;
    Word = Other.Word;
  } else if (Content != Knowledge::Known || Other.Content != Knowledge::Known ||
             Word != Other.Word) {
    Content = Knowledge::Unknown;
  }
}

bool llvm::TensixReplayBankAnalysis::State::operator==(
    const State &Other) const {
  return Recorded == Other.Recorded && Contents == Other.Contents;
}

bool llvm::TensixReplayBankAnalysis::State::operator!=(
    const State &Other) const {
  return !(*this == Other);
}

llvm::TensixReplayBankAnalysis::State
llvm::TensixReplayBankAnalysis::State::top() {
  State S;
  S.Recorded.fill(~uint32_t(0));
  for (auto &Stream : S.Contents)
    for (auto &Slot : Stream)
      Slot.Content = SlotContents::Knowledge::Top;
  return S;
}

void llvm::TensixReplayBankAnalysis::State::meet(const State &Other) {
  for (unsigned I = 0; I != 3; ++I) {
    Recorded[I] &= Other.Recorded[I];
    for (unsigned J = 0; J != TensixReplaySlotCount; ++J)
      Contents[I][J].meet(Other.Contents[I][J]);
  }
}

unsigned llvm::TensixReplayBankAnalysis::intern(TensixRecordedWord Word) {
  // Preserve Capture in value equality. These IDs never cross this owner.
  auto Found = llvm::find(Words, Word);
  if (Found != Words.end())
    return unsigned(Found - Words.begin());
  Words.push_back(std::move(Word));
  return Words.size() - 1;
}

const llvm::TensixReplayBankAnalysis::State &
llvm::TensixReplayBankAnalysis::before(const MachineInstr &MI) const {
  auto It = Observations.find(&MI);
  assert(It != Observations.end() && "missing Replay bank observation");
  return It->second;
}

bool llvm::TensixReplayBankAnalysis::isReachable(
    const MachineBasicBlock &BB) const {
  return Reachable.contains(&BB);
}

bool llvm::TensixReplayBankAnalysis::hasObservationBefore(
    const MachineInstr &MI) const {
  return Observations.contains(&MI);
}

bool llvm::TensixReplayBankAnalysis::isRecordedBefore(
    const MachineInstr &MI, TensixReplayRange Range) const {
  assert(Range.Stream < 3 && "invalid Replay bank stream");
  const State &S = before(MI);
  return (S.Recorded[Range.Stream] & Range.Mask) == Range.Mask;
}

bool llvm::TensixReplayBankAnalysis::mayContainSFPUBefore(
    const MachineInstr &MI, TensixReplayRange Range) const {
  assert(Range.Stream < 3 && "invalid Replay bank stream");
  const State &S = before(MI);
  for (unsigned I = 0; I != TensixReplaySlotCount; ++I)
    if ((Range.Mask & (uint32_t(1) << I)) &&
        S.Contents[Range.Stream][I].MayContainSFPU)
      return true;
  return false;
}

std::optional<llvm::SmallVector<llvm::TensixRecordedWord, 32>>
llvm::TensixReplayBankAnalysis::readKnownWordsBefore(
    const MachineInstr &MI, TensixReplayRange Range) const {
  assert(Range.Stream < 3 && "invalid Replay bank stream");
  const State &S = before(MI);
  SmallVector<TensixRecordedWord, 32> Result;
  for (unsigned I = 0; I != TensixReplaySlotCount; ++I) {
    if (!(Range.Mask & (uint32_t(1) << I)))
      continue;
    const SlotContents &Slot = S.Contents[Range.Stream][I];
    if (Slot.Content != SlotContents::Knowledge::Known)
      return std::nullopt;
    Result.push_back(Words[Slot.Word]);
  }
  return Result;
}

llvm::TensixTargetStateCallEffect
llvm::TensixReplayBankAnalysis::getCallEffect(const MachineInstr &MI) const {
  auto It = CallEffects.find(&MI);
  assert(It != CallEffects.end() && "missing target-state call classification");
  return It->second;
}

Expected<TensixReplayBankAnalysis>
llvm::analyzeTensixReplayBank(const MachineFunction &MF,
                              ArrayRef<TensixReplayRecordWrite> RecordWrites,
                              ArrayRef<const MachineInstr *> ObservationSites) {
  TensixReplayBankAnalysis Result;
  using State = TensixReplayBankAnalysis::State;
  using SlotContents = TensixReplayBankAnalysis::SlotContents;

  // Neutral scanner facts are interned once. All snapshots hold owner-local
  // integer IDs rather than pointers into the movable symbolic word vector.
  struct RecordWrite {
    const TensixReplayRecordWrite *Description;
    SmallVector<std::optional<unsigned>, 32> Words;
  };
  DenseMap<const MachineInstr *, RecordWrite> Writes;
  for (const auto &Write : RecordWrites) {
    assert(Write.End && Write.End->getMF() == &MF &&
           "record end must belong to the analyzed MF");
    assert(Write.Range.Stream < 3 && Write.Range.Mask &&
           "invalid Replay record range");
    RecordWrite Indexed{&Write, {}};
    if (Write.Kind == TensixReplayWriteKind::NativeContents) {
      assert(Write.Start < TensixReplaySlotCount && !Write.Words.empty() &&
             Write.Words.size() <= TensixReplaySlotCount - Write.Start &&
             "native contents must fit the Replay bank");
      assert(Write.Range.Mask ==
                 uint32_t(((uint64_t(1) << Write.Words.size()) - 1)
                          << Write.Start) &&
             "native contents must match the validated contiguous range");
      for (const auto &Word : Write.Words)
        Indexed.Words.push_back(
            Word ? std::optional<unsigned>(Result.intern(*Word))
                 : std::nullopt);
    }
    auto Inserted = Writes.try_emplace(Write.End, std::move(Indexed));
    assert(Inserted.second && "duplicate physical Replay record end");
    (void)Inserted;
  }

  SmallPtrSet<const MachineInstr *, 16> Sites;
  for (const MachineInstr *MI : ObservationSites) {
    assert(MI && MI->getMF() == &MF &&
           "observation must belong to the analyzed MF");
    Sites.insert(MI);
  }

  // Classify once for both bank and MOP domains, including inline assembly.
  CallPreservation Calls;
  for (const auto &BB : MF)
    for (const auto &MI : BB.instrs())
      if (MI.isCall() || MI.isInlineAsm())
        Result.CallEffects.try_emplace(
            &MI, Calls.preserves(MI) ? TensixTargetStateCallEffect::Preserve
                                     : TensixTargetStateCallEffect::Invalidate);

  if (MF.empty())
    return Result;
  SmallVector<const MachineBasicBlock *> Work{&MF.front()};
  while (!Work.empty()) {
    const auto *BB = Work.pop_back_val();
    if (!Result.Reachable.insert(BB).second)
      continue;
    llvm::append_range(Work, BB->successors());
  }

  auto Transfer = [&](const MachineBasicBlock &BB, State &S, bool Publish) {
    for (const auto &MI : BB.instrs()) {
      // A requested end-marker snapshot sees the old bank. The record becomes
      // visible only after this observation, matching the original transfer.
      if (Publish && Sites.contains(&MI))
        Result.Observations.try_emplace(&MI, S);
      if (auto It = Writes.find(&MI); It != Writes.end()) {
        const auto &Write = *It->second.Description;
        S.Recorded[Write.Range.Stream] |= Write.Range.Mask;
        // Automatic initialization deliberately leaves prior contents intact.
        if (Write.Kind != TensixReplayWriteKind::InitializationOnly)
          for (auto [Offset, Word] : enumerate(It->second.Words)) {
            auto &Slot = S.Contents[Write.Range.Stream][Write.Start + Offset];
            Slot.Content = Word ? SlotContents::Knowledge::Known
                                : SlotContents::Knowledge::Unknown;
            Slot.Word = Word.value_or(0);
            Slot.MayContainSFPU =
                Word && !isOrdinaryRecordedOpcode(Result.Words[*Word].Opcode);
          }
        continue;
      }
      if (MI.isCall() || MI.isInlineAsm()) {
        if (Result.getCallEffect(MI) == TensixTargetStateCallEffect::Invalidate)
          S = State();
        continue;
      }
    }
  };

  DenseMap<const MachineBasicBlock *, State> Entries, Exits;
  for (const auto *BB : Result.Reachable)
    Exits[BB] = State::top();
  bool Changed;
  do {
    Changed = false;
    for (const auto &BB : MF) {
      if (!Result.Reachable.contains(&BB))
        continue;
      State Incoming = &BB == &MF.front() ? State() : State::top();
      for (const auto *Pred : BB.predecessors())
        if (Result.Reachable.contains(Pred))
          Incoming.meet(Exits.lookup(Pred));
      Entries[&BB] = Incoming;
      Transfer(BB, Incoming, false);
      if (Incoming != Exits.lookup(&BB)) {
        Exits[&BB] = Incoming;
        Changed = true;
      }
    }
  } while (Changed);

  // Publish only after the unique bank fixedpoint converges. Unreachable sites
  // retain no observation; missing is distinct from a present Unknown slot.
  for (const auto &BB : MF) {
    if (!Result.Reachable.contains(&BB))
      continue;
    State Incoming = Entries.lookup(&BB);
    Transfer(BB, Incoming, true);
  }
  return Result;
}

bool llvm::hasTensixReplayEffects(const MachineInstr &MI,
                                  ArrayRef<MCRegister> Uses,
                                  ArrayRef<MCRegister> Defs, bool AllowMemory) {
  if ((MI.isBundled() && !isTensixReplayBundleMember(MI)) ||
      (!AllowMemory && !MI.memoperands_empty()) ||
      MI.getNumExplicitOperands() != 4 ||
      MI.getNumOperands() != 4 + Uses.size() + Defs.size())
    return false;
  SmallVector<MCRegister, 16> ActualUses, ActualDefs;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() ||
        (MO.isInternalRead() && !isTensixReplayBundleMember(MI)) ||
        MO.isRenamable())
      return false;
    (MO.isDef() ? ActualDefs : ActualUses).push_back(MO.getReg().asMCReg());
  }
  llvm::sort(ActualUses);
  llvm::sort(ActualDefs);
  return ArrayRef(ActualUses) == Uses && ArrayRef(ActualDefs) == Defs;
}
namespace {
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
} // namespace

std::optional<TensixReplayIssue>
llvm::getTensixReplayIssue(const MachineInstr &MI) {
  if (getTensixContentsSelectedImmediateOpcodeCount(MI.getOpcode()))
    return TensixReplayIssue{MI.getOpcode(), 0, 0};
  if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
      MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay ||
      MI.getOpcode() == RISCV::PseudoTTContentsSFPUReplay)
    return TensixReplayIssue{RISCV::TTREPLAY, 0, 0};
  if (MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
      MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord)
    return TensixReplayIssue{MI.getOpcode(), 0, 0};
  if (RISCV::getTensixEncoding(MI.getOpcode()))
    return TensixReplayIssue{MI.getOpcode(), 0, 0};
  if (const auto *Info = RISCV::getTensixMachineInfoByPort(MI.getOpcode())) {
    // Field-form REPLAY/MOP control keeps logical fields after the port.
    unsigned PortIndex = RISCV::getTensixPortOperandIndex(MI.getDesc());
    if (PortIndex >= MI.getNumOperands() ||
        PortIndex >= MI.getNumExplicitOperands() ||
        !MI.getOperand(PortIndex).isImm())
      return std::nullopt;
    return TensixReplayIssue{Info->Opcode,
                             unsigned(MI.getOperand(PortIndex).getImm()),
                             PortIndex + 1};
  }
  if (MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
      MI.getOpcode() == RISCV::PseudoTTSFPLOADFull ||
      MI.getOpcode() == RISCV::PseudoTTSFPLOADIndex ||
      MI.getOpcode() == RISCV::PseudoTTSFPSTORE)
    return TensixReplayIssue{MI.getOpcode(), 0, 0};
  return std::nullopt;
}

Expected<uint32_t> llvm::getTensixReplayMask(const MachineInstr &MI,
                                             unsigned First) {
  // Logical fields: load_mode, execute_while_loading, len, start_idx.
  if (MI.getNumOperands() < First + 4 ||
      MI.getNumExplicitOperands() < First + 4 ||
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

Expected<TensixReplayBundle>
llvm::getTensixReplayBundle(const MachineInstr &Bundle) {
  const MachineBasicBlock *BB = Bundle.getParent();
  if (!BB || !Bundle.isBundle() || Bundle.isBundledWithPred() ||
      !Bundle.isBundledWithSucc())
    return createStringError(
        "bound Tensix SFPU bundle has no verified issue order");
  TensixReplayBundle Result{&Bundle, nullptr, nullptr, {}};
  auto It = std::next(Bundle.getIterator());
  for (;;) {
    if (It == BB->instr_end() || !It->isBundledWithPred() || It->isBundle())
      return createStringError(
          "explicit replay bundle has inconsistent boundaries");
    Result.Members.push_back(&*It);
    if (!It->isBundledWithSucc())
      break;
    ++It;
  }
  Result.Header = Result.Members.front();
  Result.End = Result.Members.back();
  const MachineInstr &Header = *Result.Header;
  if (Header.getOpcode() != RISCV::PseudoTTExplicitSFPUReplay ||
      Header.getNumExplicitOperands() != 4)
    return createStringError(
        "bound Tensix SFPU bundle has no verified issue order");
  auto Mask = getTensixReplayMask(Header, 0);
  if (!Mask)
    return Mask.takeError();
  if (Header.getOperand(0).getImm() != 1 || Header.getOperand(1).getImm() < 0 ||
      Header.getOperand(1).getImm() > 1)
    return createStringError(
        "explicit replay bundle requires a recording header");
  if (Result.End->getOpcode() != RISCV::PseudoTTReplayRecordEnd)
    return createStringError(
        "explicit replay bundle requires its final record end marker");
  unsigned Count = 0;
  for (const MachineInstr *MI : drop_begin(Result.Members)) {
    if (MI == Result.End)
      break;
    if (MI->isDebugInstr())
      continue;
    if (MI->isCall() || MI->isInlineAsm() || MI->isTerminator() ||
        MI->getOpcode() == RISCV::PseudoTTReplayRecordEnd ||
        MI->getOpcode() == RISCV::PseudoTTReplayTemplateBegin ||
        MI->getOpcode() == RISCV::PseudoTTReplayTemplateEnd)
      return createStringError("explicit replay bundle cannot contain scalar "
                               "control or nested records");
    // Selected destinations need a dedicated dynamic capture contract. The
    // candidate count is not a native word count; their scalar expansion must
    // not stand in for an admitted static native replay payload.
    if (getTensixContentsSelectedImmediateOpcodeCount(MI->getOpcode()))
      return createStringError(
          "contents selected immediate has no replay capture receiver");
    if (RISCV::getTensixMachineInfoByPort(MI->getOpcode())) {
      unsigned Port = RISCV::getTensixPortOperandIndex(MI->getDesc());
      if (MI->getNumOperands() < MI->getDesc().getNumOperands() ||
          MI->getNumExplicitOperands() != MI->getDesc().getNumOperands() ||
          Port >= MI->getNumOperands() ||
          Port >= MI->getNumExplicitOperands() || !MI->getOperand(Port).isImm())
        return createStringError("explicit replay bundle requires a complete "
                                 "static instruction-port shape");
    }
    if (auto Issue = getTensixReplayIssue(*MI)) {
      if (Issue->Stream != 0 || Issue->Opcode == RISCV::TTREPLAY ||
          Issue->Opcode == RISCV::TTMOP || Issue->Opcode == RISCV::TTMOP_CFG)
        return createStringError(
            "explicit replay bundle cannot contain another instruction stream "
            "or sequencer control");
      ++Count;
    } else if (MI->mayLoadOrStore() || MI->hasUnmodeledSideEffects() ||
               MI->isBarrier() || MI->isPosition() || MI->isPHI()) {
      return createStringError(
          "explicit replay bundle has an unsupported scalar interleave");
    }
  }
  if (Count != uint64_t(Header.getOperand(2).getImm()))
    return createStringError("explicit replay bundle final issue count " +
                             Twine(Count) + " differs from declared length " +
                             Twine(Header.getOperand(2).getImm()));

  // Reconstruct finalizeBundle's dependency contract from the actual members.
  // In particular, uses precede defs, and a physical internal read requires
  // every register unit to have been defined earlier, not merely an alias.
  const auto &TRI = *Bundle.getMF()->getSubtarget().getRegisterInfo();
  SmallSetVector<Register, 32> LocalDefs;
  BitVector DefinedUnits(TRI.getNumRegUnits());
  SmallSet<Register, 8> DeadDefs, KilledUses, UndefUses;
  SmallSetVector<Register, 8> ExternUses;
  SmallVector<std::pair<Register, Register>, 4> Ties;
  SmallVector<const MachineInstr *, 4> MemoryMembers;
  bool FrameSetup = false, FrameDestroy = false;
  for (const MachineInstr *MI : Result.Members) {
    if (MI->isDebugInstr())
      continue;
    // Operand ranges use descriptor-sized prefixes, including for variadic
    // instructions. Check actual storage before querying those ranges.
    if (MI->getNumOperands() < MI->getDesc().getNumOperands())
      return createStringError(
          "explicit replay bundle requires a complete native operand shape");
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask())
        return createStringError(
            "explicit replay bundle has no register-mask preservation ABI");
      if (MO.isReg() && MO.isDef() && MO.isInternalRead())
        return createStringError(
            "explicit replay bundle definition cannot be an internal read");
    }
    for (const MachineOperand &MO : MI->all_uses()) {
      Register Reg = MO.getReg();
      if (!Reg)
        continue;
      bool Internal =
          Reg.isVirtual()
              ? LocalDefs.contains(Reg)
              : all_of(TRI.regunits(Reg.asMCReg()), [&](MCRegUnit Unit) {
                  return DefinedUnits[static_cast<unsigned>(Unit)];
                });
      if (MO.isInternalRead() && !Internal)
        return createStringError("explicit replay bundle internal read lacks "
                                 "its earlier definition");
      if (MO.isInternalRead() != Internal)
        return createStringError("explicit replay bundle earlier definition "
                                 "requires an internal read");
      if (Internal) {
        if (MO.isKill())
          DeadDefs.insert(Reg);
      } else {
        if (ExternUses.insert(Reg) && MO.isUndef())
          UndefUses.insert(Reg);
        if (MO.isKill())
          KilledUses.insert(Reg);
        if (MO.isTied() && Reg.isVirtual())
          Ties.emplace_back(
              MI->getOperand(MI->findTiedOperandIdx(MO.getOperandNo()))
                  .getReg(),
              Reg);
      }
    }
    for (const MachineOperand &MO : MI->all_defs()) {
      Register Reg = MO.getReg();
      if (!Reg)
        continue;
      if (LocalDefs.insert(Reg)) {
        if (!MO.isDead() && Reg.isPhysical())
          for (MCRegUnit Unit : TRI.regunits(Reg.asMCReg()))
            DefinedUnits.set(static_cast<unsigned>(Unit));
      } else if (!MO.isDead()) {
        DeadDefs.erase(Reg);
      }
      if (MO.isDead())
        DeadDefs.insert(Reg);
    }
    FrameSetup |= MI->getFlag(MachineInstr::FrameSetup);
    FrameDestroy |= MI->getFlag(MachineInstr::FrameDestroy);
    if (MI->mayLoadOrStore())
      MemoryMembers.push_back(MI);
  }
  for (const MachineInstr *MI : Result.Members)
    if (MI->getOpcode() == RISCV::PseudoTTSFPURecordWord) {
      auto Opcode = receiveRecordOnlyOpcode(*MI, *Bundle.getMF());
      if (!Opcode)
        return Opcode.takeError();
    }
  auto BadAggregate = [] {
    return createStringError(
        "explicit replay bundle aggregate effects differ from its members");
  };
  if (Bundle.getNumExplicitOperands() ||
      Bundle.getNumOperands() != LocalDefs.size() + ExternUses.size() ||
      Bundle.getFlag(MachineInstr::FrameSetup) != FrameSetup ||
      Bundle.getFlag(MachineInstr::FrameDestroy) != FrameDestroy)
    return BadAggregate();
  SmallSet<Register, 32> SeenDefs, SeenUses;
  for (const MachineOperand &MO : Bundle.operands()) {
    if (!MO.isReg() || !MO.getReg() || !MO.isImplicit() || MO.getSubReg() ||
        MO.isInternalRead() || MO.isEarlyClobber() || MO.isRenamable())
      return BadAggregate();
    Register Reg = MO.getReg();
    if (MO.isDef()) {
      if (!LocalDefs.contains(Reg) || !SeenDefs.insert(Reg).second ||
          MO.isDead() != DeadDefs.contains(Reg) || MO.isUndef())
        return BadAggregate();
    } else if (!ExternUses.contains(Reg) || !SeenUses.insert(Reg).second ||
               MO.isKill() != KilledUses.contains(Reg) ||
               MO.isUndef() != UndefUses.contains(Reg)) {
      return BadAggregate();
    }
    if (MO.isTied()) {
      const MachineOperand &Other =
          Bundle.getOperand(Bundle.findTiedOperandIdx(MO.getOperandNo()));
      auto Tie = MO.isDef() ? std::make_pair(Reg, Other.getReg())
                            : std::make_pair(Other.getReg(), Reg);
      if (!is_contained(Ties, Tie))
        return BadAggregate();
    }
  }
  for (auto [Def, Use] : Ties) {
    auto DefIt = find_if(Bundle.operands(), [Def](const MachineOperand &MO) {
      return MO.isDef() && MO.getReg() == Def;
    });
    auto UseIt = find_if(Bundle.operands(), [Use](const MachineOperand &MO) {
      return MO.isUse() && MO.getReg() == Use;
    });
    if (DefIt == Bundle.operands_end() || UseIt == Bundle.operands_end() ||
        !DefIt->isTied() || !UseIt->isTied() ||
        Bundle.findTiedOperandIdx(DefIt->getOperandNo()) !=
            UseIt->getOperandNo())
      return BadAggregate();
  }
  // Match cloneMergedMemRefs, including its conservative empty projection and
  // identical-first-member de-duplication. Compare semantics, not MMO pointers.
  SmallVector<const MachineMemOperand *, 4> ExpectedMemory;
  auto SameMemory = [](auto A, auto B) {
    return std::equal(A.begin(), A.end(), B.begin(), B.end(),
                      [](const auto *L, const auto *R) { return *L == *R; });
  };
  if (!MemoryMembers.empty()) {
    append_range(ExpectedMemory, MemoryMembers.front()->memoperands());
    for (const MachineInstr *MI : drop_begin(MemoryMembers)) {
      if (ExpectedMemory.empty())
        break;
      if (SameMemory(MemoryMembers.front()->memoperands(), MI->memoperands()))
        continue;
      if (MI->memoperands_empty()) {
        ExpectedMemory.clear();
        break;
      }
      append_range(ExpectedMemory, MI->memoperands());
    }
  }
  if (!SameMemory(ArrayRef(ExpectedMemory), Bundle.memoperands()))
    return createStringError(
        "explicit replay bundle aggregate memory differs from its members");
  return Result;
}

bool llvm::isTensixReplayBundleMember(const MachineInstr &MI) {
  const MachineBasicBlock *BB = MI.getParent();
  if (!BB || !MI.isBundledWithPred() || MI.isBundle())
    return false;
  auto It = MI.getIterator();
  while (It->isBundledWithPred()) {
    if (It == BB->instr_begin())
      return false;
    --It;
  }
  auto Bundle = getTensixReplayBundle(*It);
  if (!Bundle) {
    consumeError(Bundle.takeError());
    return false;
  }
  return is_contained(Bundle->Members, &MI);
}

Expected<TensixReplayScan> llvm::scanTensixReplay(const MachineFunction &MF,
                                                  bool RequireNormalized) {
  TensixReplayScan Result;
  for (const auto &BB : MF) {
    std::optional<TensixReplayRecordRegion> Active;
    for (const auto &MI : BB.instrs()) {
      if (MI.isBundled() &&
          !MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU())
        continue;
      if (MI.isBundle()) {
        auto Bundle = getTensixReplayBundle(MI);
        if (!Bundle)
          return Bundle.takeError();
        continue;
      }
      if (MI.isBundled() && !isTensixReplayBundleMember(MI))
        return createStringError(
            "bound Tensix SFPU bundle has no verified issue order");
      if (MI.getOpcode() == RISCV::PseudoTTReplayRecordEnd) {
        if (!Active)
          return createStringError("Tensix replay end has no active record");
        if (Active->Count != Active->Length)
          return createStringError(
              "Tensix replay final issue count " + Twine(Active->Count) +
              " differs from declared length " + Twine(Active->Length));
        Active->End = &MI;
        if (!Active->AutomaticSFPU &&
            (Active->HasSFPU ||
             Active->Header->getOpcode() == RISCV::PseudoTTExplicitSFPUReplay ||
             Active->Header->getOpcode() ==
                 RISCV::PseudoTTContentsSFPUReplay)) {
          bool Contents =
              Active->Header->getOpcode() == RISCV::PseudoTTContentsSFPUReplay;
          if (Contents && !Active->ExecuteWhileLoading &&
              any_of(Active->Body, [](const MachineInstr *Body) {
                return Body->getOpcode() != RISCV::PseudoTTSFPURecordWord &&
                       Body->getOpcode() != RISCV::PseudoTTSFPUDstRecordWord;
              }))
            return createStringError("contents record-only body requires "
                                     "nonexecuting native payload words");
          if (Active->Stream != 0 ||
              (Active->Header->getOpcode() != RISCV::TTREPLAY &&
               Active->Header->getOpcode() !=
                   RISCV::PseudoTTExplicitSFPUReplay &&
               !Contents) ||
              (Active->HasSFPU && !Contents &&
               !MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU()))
            return createStringError(
                "SFPU replay recording requires the local bound physical ABI");
          if (RequireNormalized &&
              Active->Header->getOpcode() !=
                  RISCV::PseudoTTExplicitSFPUReplay &&
              !Contents)
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
          for (const auto &Word : Active->Words) {
            if (!Word)
              return createStringError("mixed ordinary and SFPU replay words "
                                       "need an executed-effect receiver");
            Public.Words.push_back(*Word);
          }
          if (Active->Header->getOpcode() ==
                  RISCV::PseudoTTExplicitSFPUReplay ||
              Contents) {
            SmallVector<MCRegister, 2> Uses{RISCV::TT_CONFIG, RISCV::TT_ISSUE};
            SmallVector<MCRegister, 1> Defs{RISCV::TT_ISSUE};
            llvm::sort(Uses);
            if (!hasTensixReplayEffects(*Active->Header, Uses, Defs, true))
              return createStringError("explicit SFPU replay record header has "
                                       "unexpected physical effects");
          }
          Result.ExplicitRecords.push_back(std::move(Public));
        }
        Result.Regions.push_back(std::move(*Active));
        Active.reset();
        continue;
      }
      if (Active && (MI.isTerminator() || MI.isCall() || MI.isInlineAsm()))
        return createStringError(
            "Tensix replay recording cannot cross scalar control flow");
      if (Active && MI.getOpcode() == RISCV::PseudoTTReplayTemplateBegin)
        return createStringError(
            "raw and structured replay recordings cannot nest");
      auto Issued = getTensixReplayIssue(MI);
      if (!Issued)
        continue;
      if (Issued->Stream > 2)
        return createStringError("unknown Tensix replay instruction stream");
      if (Issued->Opcode == RISCV::TTREPLAY) {
        auto Mask = getTensixReplayMask(MI, Issued->First);
        if (!Mask)
          return Mask.takeError();
        if (Active && Active->Stream == Issued->Stream)
          return createStringError(
              "Tensix replay recording cannot contain REPLAY");
        if (MI.getOperand(Issued->First).getImm()) {
          if (Active)
            return createStringError("Tensix replay records cannot overlap");
          Active = TensixReplayRecordRegion{
              *Mask, Issued->Stream,
              unsigned(MI.getOperand(Issued->First + 2).getImm())};
          Active->AutomaticSFPU = MI.getOpcode() == RISCV::PseudoTTSFPUReplay;
          Active->Header = &MI;
          Active->Start = unsigned(MI.getOperand(Issued->First + 3).getImm());
          Active->ExecuteWhileLoading =
              MI.getOperand(Issued->First + 1).getImm() != 0;
        }
        continue;
      }
      bool IsPayload = MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
                       MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord;
      if (IsPayload && (!Active || Active->ExecuteWhileLoading ||
                        Active->AutomaticSFPU || Active->Stream != 0))
        return createStringError(
            "record-only payload must belong to a local record-only region");
      if (!Active || Issued->Stream != Active->Stream)
        continue;
      Active->Body.push_back(&MI);
      if (Active->AutomaticSFPU) {
        ++Active->Count;
        continue;
      }
      if (Issued->Opcode == RISCV::TTMOP)
        return createStringError("Tensix replay recording cannot contain MOP");
      bool IsSFPU =
          getTensixContentsSelectedImmediateOpcodeCount(MI.getOpcode()) ||
          (RISCV::getTensixEncoding(Issued->Opcode) &&
           !RISCV::getTensixMachineInfo(Issued->Opcode)) ||
          MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
          MI.getOpcode() == RISCV::PseudoTTSFPLOADFull ||
          MI.getOpcode() == RISCV::PseudoTTSFPLOADIndex ||
          MI.getOpcode() == RISCV::PseudoTTSFPSTORE;
      bool Received =
          IsPayload || IsSFPU || isOrdinaryRecordedCandidate(MI) ||
          Active->Header->getOpcode() == RISCV::PseudoTTExplicitSFPUReplay ||
          Active->Header->getOpcode() == RISCV::PseudoTTContentsSFPUReplay;
      if (Received) {
        auto Word = getTensixRecordedWord(MI, MF);
        if (!Word)
          return Word.takeError();
        IsSFPU = !isOrdinaryRecordedOpcode(Word->Opcode);
        if (IsSFPU &&
            Active->Header->getOpcode() != RISCV::PseudoTTContentsSFPUReplay &&
            !MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU())
          return createStringError(
              "SFPU replay recording requires a typed SSA preservation ABI");
        if (RequireNormalized && !Active->ExecuteWhileLoading && !IsPayload &&
            (IsSFPU ||
             Active->Header->getOpcode() == RISCV::PseudoTTExplicitSFPUReplay))
          return createStringError("record-only SFPU body requires "
                                   "nonexecuting native payload words");
        Active->HasSFPU |= IsSFPU;
        Active->Words.push_back(std::move(*Word));
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

SmallVector<TensixReplayRecordWrite, 4>
llvm::getTensixReplayRecordWrites(const TensixReplayScan &Scan) {
  SmallVector<TensixReplayRecordWrite, 4> Writes;
  for (const auto &Region : Scan.Regions)
    Writes.push_back({Region.End,
                      {Region.Stream, Region.Mask},
                      Region.Start,
                      Region.AutomaticSFPU
                          ? TensixReplayWriteKind::InitializationOnly
                          : TensixReplayWriteKind::NativeContents,
                      Region.Words});
  return Writes;
}

Expected<TensixNativeReplayAnalysis> llvm::analyzeTensixNativeReplay(
    const MachineFunction &MF, const TensixReplayScan &Scan,
    const TensixReplayBankAnalysis &Bank, bool RequireNormalized) {
  TensixNativeReplayAnalysis Result;
  Result.Records = Scan.ExplicitRecords;
  for (const auto &Region : Scan.Regions)
    Result.RecordWrites.try_emplace(
        Region.End, TensixReplayRange{Region.Stream, Region.Mask});
  for (const auto &BB : MF) {
    if (!Bank.isReachable(BB))
      continue;
    for (const auto &MI : BB.instrs()) {
      auto Issued = getTensixReplayIssue(MI);
      if (!Issued || Issued->Opcode != RISCV::TTREPLAY ||
          MI.getOperand(Issued->First).getImm())
        continue;
      auto Mask = getTensixReplayMask(MI, Issued->First);
      if (!Mask)
        return Mask.takeError();
      TensixReplayRange Range{Issued->Stream, *Mask};
      assert(Bank.hasObservationBefore(MI) &&
             "missing Replay execution observation");
      Result.Executions.try_emplace(&MI, Range);
      if (!Bank.isRecordedBefore(MI, Range))
        return createStringError("Tensix replay execution references slots not "
                                 "recorded on every incoming path");
      if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay)
        continue;
      bool Contents = MI.getOpcode() == RISCV::PseudoTTContentsSFPUReplay;
      bool IsExplicit =
          MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay || Contents;
      bool HasSFPU = Bank.mayContainSFPUBefore(MI, Range);
      if (!HasSFPU && !IsExplicit)
        continue;
      if (Issued->Stream != 0 ||
          (MI.getOpcode() != RISCV::TTREPLAY && !IsExplicit) ||
          (HasSFPU && !Contents &&
           !MI.getMF()
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
      auto Words = Bank.readKnownWordsBefore(MI, Range);
      if (!Words)
        return createStringError("SFPU replay execution has differing or "
                                 "unknown templates on incoming paths");
      auto Effects = getTensixExplicitReplayEffects(*Words, *MI.getMF());
      if (!Effects)
        return Effects.takeError();
      if (IsExplicit &&
          !hasTensixReplayEffects(MI, Effects->Uses, Effects->Defs, true))
        return createStringError("explicit SFPU replay execution physical "
                                 "effects differ from its reaching words");
      Result.ExecutionEffects.try_emplace(&MI, std::move(*Effects));
      Result.ExecutionWords.try_emplace(&MI, std::move(*Words));
    }
  }
  return Result;
}

Expected<TensixNativeReplayAnalysis>
llvm::analyzeTensixNativeReplay(const MachineFunction &MF,
                                bool RequireNormalized) {
  auto Scan = scanTensixReplay(MF, RequireNormalized);
  if (!Scan)
    return Scan.takeError();
  for (const auto &Region : Scan->Regions)
    if (Region.AutomaticSFPU)
      return createStringError(
          "automatic SFPU replay requires its selection owner");
  auto Writes = getTensixReplayRecordWrites(*Scan);
  SmallVector<const MachineInstr *, 8> Sites;
  for (const auto &BB : MF)
    for (const auto &MI : BB.instrs())
      if (auto Issue = getTensixReplayIssue(MI))
        if (Issue->Opcode == RISCV::TTREPLAY &&
            !MI.getOperand(Issue->First).getImm())
          Sites.push_back(&MI);
  auto Bank = analyzeTensixReplayBank(MF, Writes, Sites);
  if (!Bank)
    return Bank.takeError();
  return analyzeTensixNativeReplay(MF, *Scan, *Bank, RequireNormalized);
}

namespace {

void retainLivePhysicalRegister(MachineFunction &MF, MCRegister Reg) {
  auto &MRI = MF.getRegInfo();
  MRI.clearKillFlags(Reg);
  for (MachineOperand &MO : MRI.def_operands(Reg))
    MO.setIsDead(false);
}

void retainLivePhysicalEffects(MachineFunction &MF,
                               const TensixSFPUReplayEffects &Effects) {
  for (MCRegister Reg : Effects.Uses)
    retainLivePhysicalRegister(MF, Reg);
  for (MCRegister Reg : Effects.Defs)
    retainLivePhysicalRegister(MF, Reg);
}

void normalizeControl(MachineInstr &MI, const RISCVInstrInfo &TII,
                      const TensixSFPUReplayEffects *Effects) {
  if (MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay)
    return;
  assert(MI.getOpcode() == RISCV::TTREPLAY && "requires direct local control");
  const MCInstrDesc &Desc = TII.get(RISCV::PseudoTTExplicitSFPUReplay);
  auto Builder = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), Desc);
  for (unsigned I = 0; I != 4; ++I)
    Builder.addImm(MI.getOperand(I).getImm());
  Builder.cloneMemRefs(MI).setMIFlags(MI.getFlags());
  if (Effects) {
    for (MCRegister Reg : Effects->Uses)
      if (!is_contained(Desc.implicit_uses(), Reg.id()))
        Builder.addReg(Reg, RegState::Implicit);
    for (MCRegister Reg : Effects->Defs)
      if (!is_contained(Desc.implicit_defs(), Reg.id()))
        Builder.addReg(Reg, RegState::Define | RegState::Implicit);
  }
  MI.eraseFromParent();
}

void normalizeRecordWord(MachineInstr &MI, const TensixRecordedWord &Word,
                         const RISCVInstrInfo &TII) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
      MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord)
    return;
  insertTensixRecordedPayload(MI, Word, TII, std::nullopt)
      .setMIFlags(MI.getFlags());
  MI.eraseFromParent();
}

class RISCVTensixExplicitReplay final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixExplicitReplay() : MachineFunctionPass(ID) {
    initializeRISCVTensixExplicitReplayPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
    if (!Info->usesBoundTensixSFPU() || Info->hasTensixCodegenFailed())
      return false;
    auto Analysis = analyzeTensixExplicitReplay(MF);
    if (!Analysis) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
          MF.getFunction(), toString(Analysis.takeError())));
      return false;
    }
    if (Analysis->Records.empty() && Analysis->ExecutionWords.empty() &&
        Analysis->MOPExecutions.empty())
      return false;
    bool NeedsNormalization =
        any_of(Analysis->Records,
               [](const auto &Record) {
                 return Record.Header->getOpcode() !=
                            RISCV::PseudoTTExplicitSFPUReplay ||
                        (!Record.ExecuteWhileLoading &&
                         any_of(Record.Body, [](const MachineInstr *MI) {
                           return MI->getOpcode() != RISCV::PseudoTTSFPURecordWord &&
                                  MI->getOpcode() != RISCV::PseudoTTSFPUDstRecordWord;
                         }));
               }) ||
        any_of(Analysis->ExecutionWords,
               [](const auto &Execution) {
                 return Execution.first->getOpcode() !=
                        RISCV::PseudoTTExplicitSFPUReplay;
               }) ||
        any_of(Analysis->MOPExecutions, [](const auto &Execution) {
          if (Error E = verifyTensixMOPExecutionEffects(*Execution.first,
                                                       Execution.second)) {
            consumeError(std::move(E));
            return true;
          }
          return false;
        });
    auto Templates = getTensixReplayTemplateStorage(MF, false);
    if (!Templates) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
          MF.getFunction(), toString(Templates.takeError())));
      return false;
    }
    bool DeferredTemplates = !Templates->SymbolicWords.empty();
    bool NeedsBundling =
        !DeferredTemplates && any_of(Analysis->Records, [](const auto &Record) {
          return !Record.Header->isBundled();
        });
    if (!NeedsNormalization && !NeedsBundling)
      return false;
    const auto &TII = *MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
    bool Changed = false;
    // Analysis of every actual incoming path must succeed before mutation.
    // Recording stores instruction fields; execution reads current registers.
    for (const auto &Record : Analysis->Records) {
      if (!Record.ExecuteWhileLoading)
        for (auto [Original, Word] : zip(Record.Body, Record.Words)) {
          if (Original->getOpcode() == RISCV::PseudoTTSFPURecordWord ||
              Original->getOpcode() == RISCV::PseudoTTSFPUDstRecordWord)
            continue;
          // Removing a falsely executing def can expose an earlier value.
          // Old last-use/dead annotations cannot survive that correction,
          // including a recording that is never subsequently replayed.
          // Dst/config/CC dependencies are implicit in the native descriptor,
          // not encoded operand fields. Removing their false record-time
          // effects must also invalidate any stale liveness flags.
          for (const MachineOperand &Operand : Original->operands())
            if (Operand.isReg() && Operand.getReg().isPhysical())
              retainLivePhysicalRegister(MF, Operand.getReg().asMCReg());
          auto &MI = *const_cast<MachineInstr *>(Original);
          Changed = true;
          normalizeRecordWord(MI, Word, TII);
        }
      auto &Header = *const_cast<MachineInstr *>(Record.Header);
      Changed |= Header.getOpcode() != RISCV::PseudoTTExplicitSFPUReplay;
      normalizeControl(Header, TII, nullptr);
    }
    for (const auto &Execution : Analysis->ExecutionWords) {
      auto &MI = *const_cast<MachineInstr *>(Execution.first);
      const auto &Effects = Analysis->ExecutionEffects.find(&MI)->second;
      retainLivePhysicalEffects(MF, Effects);
      Changed |= MI.getOpcode() != RISCV::PseudoTTExplicitSFPUReplay;
      normalizeControl(MI, TII, &Effects);
    }
    for (const auto &[Instruction, Summary] : Analysis->MOPExecutions) {
      auto *Original = const_cast<MachineInstr *>(Instruction);
      if (Error E = verifyTensixMOPExecutionEffects(*Original, Summary))
        consumeError(std::move(E));
      else
        continue;
      retainLivePhysicalEffects(MF, Summary.Physical);
      auto Builder = BuildMI(*Original->getParent(), *Original,
                             Original->getDebugLoc(), Original->getDesc());
      for (const auto &Operand : Original->explicit_operands())
        Builder.add(Operand);
      Builder.cloneMemRefs(*Original).setMIFlags(Original->getFlags());
      const auto &Desc = Original->getDesc();
      for (MCRegister Reg : Summary.Physical.Uses)
        if (!is_contained(Desc.implicit_uses(), Reg.id()))
          Builder.addReg(Reg, RegState::Implicit);
      for (MCRegister Reg : Summary.Physical.Defs)
        if (!is_contained(Desc.implicit_defs(), Reg.id()))
          Builder.addReg(Reg, RegState::Define | RegState::Implicit);
      Original->eraseFromParent();
      Changed = true;
    }
    // Prepared legacy templates still own allocation and raw-range lifetime.
    // The normalizer runs again after their finalization, before emission.
    if (DeferredTemplates)
      return Changed;
    auto Normalized = scanTensixReplay(MF, true);
    if (!Normalized) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
          MF.getFunction(), toString(Normalized.takeError())));
      return Changed;
    }
    for (const auto &Record : Normalized->ExplicitRecords) {
      if (Record.Header->isBundled())
        continue;
      // Pure raw windows form pre-RA. Mixed raw/template functions defer
      // formation until the frozen template owner has committed placement.
      // From this point the aggregate prevents interleaving the capture.
      auto *Header = const_cast<MachineInstr *>(Record.Header);
      auto *End = const_cast<MachineInstr *>(Record.End);
      finalizeBundle(*Header->getParent(), Header->getIterator(),
                     std::next(End->getIterator()));
      Changed = true;
    }
    return Changed;
  }
};

// KCFI's post-RA bundles need the historical unpacking behavior. A native
// capture is an independent issue-order contract: preserve it per bundle,
// even when the same function also contains a KCFI check/call bundle.
class RISCVTensixReplayBundleUnpack final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixReplayBundleUnpack() : MachineFunctionPass(ID) {
    initializeRISCVTensixReplayBundleUnpackPass(
        *PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getFunction().getParent()->getModuleFlag("kcfi"))
      return false;
    auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
    if (Info->hasTensixCodegenFailed())
      return false;
    bool Changed = false;
    auto Fail = [&](const Twine &Message) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(
          DiagnosticInfoUnsupported(MF.getFunction(), Message));
      return Changed;
    };
    for (auto &BB : MF) {
      for (auto It = BB.instr_begin(); It != BB.instr_end();) {
        if (!It->isBundle()) {
          ++It;
          continue;
        }
        MachineInstr &Bundle = *It;
        auto First = std::next(It);
        // A recording header is never laundered by clearing its bundle bits.
        // Ordinary non-Tensix functions retain generic unpack behavior.
        bool Native = First != BB.instr_end() &&
                      First->getOpcode() == RISCV::PseudoTTExplicitSFPUReplay;
        if (Native) {
          auto Window = getTensixReplayBundle(Bundle);
          if (!Window)
            return Fail(toString(Window.takeError()));
          It =
              std::next(const_cast<MachineInstr *>(Window->End)->getIterator());
          continue;
        }
        if (Info->usesBoundTensixSFPU()) {
          if (First == BB.instr_end() ||
              First->getOpcode() != RISCV::KCFI_CHECK ||
              !First->isBundledWithPred() || !First->isBundledWithSucc())
            return Fail("bound Tensix SFPU bundle has no verified issue order");
          auto Call = std::next(First);
          if (Call == BB.instr_end() || !Call->isCall() ||
              !Call->isBundledWithPred() || Call->isBundledWithSucc())
            return Fail("bound Tensix SFPU bundle has no verified issue order");
        }
        // The generic algorithm applies only to this ordinary bundle. It
        // never clears internal-read flags to make a native window admissible.
        ++It;
        while (It != BB.instr_end() && It->isBundledWithPred()) {
          It->unbundleFromPred();
          for (MachineOperand &MO : It->operands())
            if (MO.isReg() && MO.isInternalRead())
              MO.setIsInternalRead(false);
          ++It;
        }
        Bundle.eraseFromParent();
        Changed = true;
      }
    }
    return Changed;
  }
};
} // namespace

char RISCVTensixReplayBundleUnpack::ID = 0;
INITIALIZE_PASS(RISCVTensixReplayBundleUnpack,
                "riscv-tensix-replay-bundle-unpack",
                "Unpack ordinary bundles while preserving native Tensix replay",
                false, false)
FunctionPass *llvm::createRISCVTensixReplayBundleUnpackPass() {
  return new RISCVTensixReplayBundleUnpack();
}

char RISCVTensixExplicitReplay::ID = 0;
INITIALIZE_PASS(RISCVTensixExplicitReplay, "riscv-tensix-explicit-replay",
                "Normalize authored bound Tensix SFPU recordings", false, false)
FunctionPass *llvm::createRISCVTensixExplicitReplayPass() {
  return new RISCVTensixExplicitReplay();
}
