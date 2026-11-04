//===-- RISCVTensixMacro.cpp --------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixMacro.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "MCTargetDesc/RISCVTensixBoundContract.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixBoundLowering.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"
#include <array>
#include <memory>
#include <optional>

using namespace llvm;

namespace {
constexpr uint32_t MacroMiscFP32 = 3;
constexpr unsigned MacroDrainIssues = 4;
constexpr unsigned BlackholeConfigStateGapIssues = 3;
uint32_t macroSequence(unsigned Index) {
  // Simple/Round disabled, MAD template[Index] at t+1, builtin STORE at t+3.
  // Misc selects elapsed cycles for all sub-units. Four actual load-unit NOP
  // issues also advance the queues if an outstanding instruction ever makes
  // their shared countdown instruction-driven. No wait is used as a drain.
  return (0x13u << 24) | ((4u + Index) << 8);
}

Expected<uint32_t> encodeADDITemplate(const MachineFunction &MF,
                                      uint16_t BF16) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  const auto &TM = MF.getTarget();
  MCContext Context(TM.getTargetTriple(), TM.getMCAsmInfo(),
                    TM.getMCRegisterInfo(), ST);
  std::unique_ptr<MCCodeEmitter> Encoder(
      createRISCVMCCodeEmitter(*TM.getMCInstrInfo(), Context));
  // Reuse the received native ADDI descriptor. In particular there is no
  // macro-local opcode/immediate encoding table. SFPLOADMACRO overrides VD
  // and VC with the actual parity-selected target at execution.
  MCInst Native;
  Native.setOpcode(RISCV::TTSFPADDI);
  Native.addOperand(MCOperand::createReg(RISCV::TT_L0));
  Native.addOperand(MCOperand::createReg(RISCV::TT_L0));
  Native.addOperand(MCOperand::createImm(BF16));
  Native.addOperand(MCOperand::createImm(0));
  Error Fields = ST.hasVendorXTTTensixWH()
                     ? RISCV::verifyTensixWormholeInstruction(
                           Native, *TM.getMCInstrInfo())
                     : RISCV::verifyTensixMCInstructionFields(
                           Native, *TM.getMCInstrInfo());
  if (Fields)
    return std::move(Fields);
  SmallVector<char, 4> Bytes;
  SmallVector<MCFixup, 0> Fixups;
  Encoder->encodeInstruction(Native, Bytes, Fixups, ST);
  if (Bytes.size() != 4 || !Fixups.empty())
    return createStringError(
        "SFPLOADMACRO ADDI template failed native encoding");
  return rotr(support::endian::read32le(Bytes.data()), 2);
}

bool immediate(const MachineInstr &MI, unsigned Index, int64_t Value) {
  return Index < MI.getNumExplicitOperands() && MI.getOperand(Index).isImm() &&
         MI.getOperand(Index).getImm() == Value;
}
bool isSFPU(const MachineInstr &MI) {
  return RISCV::getTensixEncoding(MI.getOpcode()) &&
         !RISCV::getTensixMachineInfo(MI.getOpcode());
}

// Scalar stack spills cannot alias hardware configuration. Require every
// actual memory effect to carry the compiler-owned stack provenance; neither
// the instruction opcode nor an apparent SP base establishes that fact.
bool isProvenStackStore(const MachineInstr &MI) {
  return MI.mayStore() && !MI.memoperands_empty() &&
         all_of(MI.memoperands(), [](const MachineMemOperand *MMO) {
           const auto *PSV = MMO->getPseudoValue();
           return MMO->isStore() && PSV &&
                  (PSV->isStack() ||
                   PSV->kind() == PseudoSourceValue::FixedStack);
         });
}

// Facts here are computed from the emitted instructions, not attributes or
// pseudo provenance. The CFG meet retains only hardware state agreed by every
// predecessor; loop bodies do not acquire a fresh initialization convention.
struct MacroState {
  bool StateZero = false;
  bool MathZero = false;
  bool DstZero = false;
  bool Mod0Zero = false;
  bool Mod4Zero = false;
  uint8_t BaseBytesZero = 0;
  // Blackhole ordinary FP32 accesses add (Sp + SpBase) & 3. These facts
  // require actual BH setters, independently of the common Dst reset.
  bool SpZero = false;
  uint8_t SpBaseBytesZero = 0;
  std::optional<bool> Enable;
  std::optional<bool> Flags;
  bool LaneZero = false;
  bool ConfigReady = false;
  std::optional<uint32_t> L0;
  std::array<std::optional<uint32_t>, 4> Templates;
  std::array<std::optional<uint32_t>, 4> Sequences;
  std::optional<uint32_t> Misc;

  bool operator==(const MacroState &Other) const {
    return StateZero == Other.StateZero && MathZero == Other.MathZero &&
           DstZero == Other.DstZero && Mod0Zero == Other.Mod0Zero &&
           Mod4Zero == Other.Mod4Zero && BaseBytesZero == Other.BaseBytesZero &&
           SpZero == Other.SpZero &&
           SpBaseBytesZero == Other.SpBaseBytesZero &&
           Enable == Other.Enable && Flags == Other.Flags &&
           LaneZero == Other.LaneZero && ConfigReady == Other.ConfigReady &&
           L0 == Other.L0 && Templates == Other.Templates &&
           Sequences == Other.Sequences && Misc == Other.Misc;
  }
  bool operator!=(const MacroState &Other) const { return !(*this == Other); }
  bool allActive() const {
    return LaneZero && (Enable == false || Flags == true);
  }
  bool addressZero(const RISCVSubtarget &ST) const {
    bool ArchitectureZero = ST.hasVendorXTTTensixWH()
                                ? Mod4Zero
                                : SpZero && SpBaseBytesZero == 3;
    return StateZero && MathZero && DstZero && Mod0Zero && ArchitectureZero &&
           BaseBytesZero == 3;
  }
  void meet(const MacroState &Other) {
    StateZero &= Other.StateZero;
    MathZero &= Other.MathZero;
    DstZero &= Other.DstZero;
    Mod0Zero &= Other.Mod0Zero;
    Mod4Zero &= Other.Mod4Zero;
    BaseBytesZero &= Other.BaseBytesZero;
    SpZero &= Other.SpZero;
    SpBaseBytesZero &= Other.SpBaseBytesZero;
    if (Enable != Other.Enable)
      Enable.reset();
    if (Flags != Other.Flags)
      Flags.reset();
    LaneZero &= Other.LaneZero;
    ConfigReady &= Other.ConfigReady;
    if (L0 != Other.L0)
      L0.reset();
    for (unsigned I = 0; I != Templates.size(); ++I) {
      if (Templates[I] != Other.Templates[I])
        Templates[I].reset();
      if (Sequences[I] != Other.Sequences[I])
        Sequences[I].reset();
    }
    if (Misc != Other.Misc)
      Misc.reset();
  }
};

void advance(const MachineInstr &MI, MacroState &State,
             const RISCVSubtarget &ST) {
  unsigned Opcode = MI.getOpcode();
  if (MI.isDebugInstr() || MI.isCFIInstruction() || MI.isTerminator())
    return;
  // These native issue controls do not mutate macro configuration, lane
  // state or Dst addressing. Their ordinary descriptor has conservative
  // ordering defs and memory effects, not architectural configuration writes.
  // Preserve facts only: neither control establishes ConfigReady nor replaces
  // the actual consecutive SFPU load-unit NOP drain checked below.
  if (Opcode == RISCV::TTSTALLWAIT || Opcode == RISCV::TTNOP)
    return;
  if (Opcode == RISCV::TTSFPNOP) {
    State.ConfigReady = true;
    return;
  }
  if (Opcode == RISCV::TTSETC16 && MI.getNumExplicitOperands() == 2 &&
      MI.getOperand(0).isImm() && MI.getOperand(1).isImm()) {
    bool Zero = MI.getOperand(0).getImm() == 0;
    unsigned Index = MI.getOperand(1).getImm();
    bool Wormhole = ST.hasVendorXTTTensixWH();
    if (Index == 0) {
      State.StateZero = Zero;
      // Config writes in another bank cannot initialize the selected bank.
      if (!Zero) {
        State.BaseBytesZero = 0;
        State.SpBaseBytesZero = 0;
      }
    } else if (Index == 1) {
      State.MathZero = Zero;
    } else if (Index == (Wormhole ? 23u : 28u)) {
      State.Mod0Zero = Zero;
    } else if (Wormhole && Index == 27) {
      State.Mod4Zero = Zero;
    }
    return;
  }
  if (Opcode == RISCV::TTRMWCIB0 || Opcode == RISCV::TTRMWCIB1) {
    bool SpBase = !ST.hasVendorXTTTensixWH() && immediate(MI, 0, 7);
    if (State.StateZero && (immediate(MI, 0, 6) || SpBase)) {
      unsigned Bit = Opcode == RISCV::TTRMWCIB0 ? 1 : 2;
      uint8_t &Bytes = SpBase ? State.SpBaseBytesZero : State.BaseBytesZero;
      if (immediate(MI, 1, 0) && immediate(MI, 2, 255))
        Bytes |= Bit;
      else
        Bytes &= ~Bit;
    }
    return;
  }
  if (Opcode == RISCV::TTSETRWC) {
    if (MI.getNumExplicitOperands() != 6 || !MI.getOperand(0).isImm()) {
      State.DstZero = State.SpZero = false;
      return;
    }
    unsigned Mask = MI.getOperand(0).getImm();
    bool Wormhole = ST.hasVendorXTTTensixWH();
    // BH assembly.yaml defines bit4 as SFPU SP, using A/B/D as its low,
    // middle and high immediate nibbles. Bit5 has no received semantics.
    if (!Wormhole && (Mask & ~0x1fu)) {
      State.DstZero = State.SpZero = false;
      return;
    }
    bool Zero = immediate(MI, 1, 0) && immediate(MI, 2, 0) &&
                immediate(MI, 3, 0) && immediate(MI, 4, 0) &&
                immediate(MI, 5, 0);
    if (Mask & 4)
      State.DstZero = Zero;
    if (!Wormhole && (Mask & 16))
      State.SpZero = Zero;
    return;
  }
  if (Opcode == RISCV::TTSFPENCC) {
    unsigned Value = MI.getOperand(0).getImm();
    unsigned Mode = MI.getOperand(1).getImm();
    if (Mode & 2)
      State.Enable = bool(Value & 1);
    else if ((Mode & 1) && State.Enable)
      State.Enable = !*State.Enable;
    State.Flags = (Mode & 8) ? bool(Value & 2) : true;
    return;
  }
  if (Opcode == RISCV::TTSFPCONFIGReset) {
    State.LaneZero = State.Enable == false || State.Flags == true;
    State.ConfigReady = false;
    return;
  }
  if (Opcode == RISCV::TTSFPLOADI &&
      MI.getOperand(0).getReg() == RISCV::TT_L0 && State.allActive()) {
    uint32_t Value = MI.getOperand(2).getImm();
    unsigned Mode = MI.getOperand(3).getImm();
    if (Mode == 0)
      State.L0 = Value << 16;
    else if (Mode == 10 && State.L0)
      State.L0 = (*State.L0 & 0xffff0000u) | Value;
    else
      State.L0.reset();
    return;
  }
  if (Opcode == RISCV::TTSFPCONFIGMacroTemplate ||
      Opcode == RISCV::TTSFPCONFIGMacroSequence) {
    unsigned Index = MI.getOperand(0).getImm();
    if (Index < 4) {
      auto &Array = Opcode == RISCV::TTSFPCONFIGMacroTemplate ? State.Templates
                                                              : State.Sequences;
      Array[Index] = State.allActive() ? State.L0 : std::nullopt;
    }
    State.ConfigReady = false;
    return;
  }
  if (Opcode == RISCV::TTSFPCONFIGMacroMisc) {
    State.Misc = State.allActive()
                     ? std::optional<uint32_t>(MI.getOperand(0).getImm())
                     : std::nullopt;
    State.ConfigReady = false;
    return;
  }
  if (Opcode == RISCV::TTSFPLOAD || Opcode == RISCV::TTSFPLOADFull ||
      Opcode == RISCV::TTSFPSTORE || Opcode == RISCV::TTSFPLOADMACROFP32Even) {
    // BH uses its direct AddrMod0 profile. WH can select profile0 or 4
    // through its extra modifier bit, so both must actually be zero.
    unsigned AddrMode = Opcode == RISCV::TTSFPLOAD ? 3 : 2;
    if (Opcode != RISCV::TTSFPLOADMACROFP32Even && !immediate(MI, AddrMode, 0))
      State.DstZero = false;
    if (!State.Mod0Zero || (ST.hasVendorXTTTensixWH() && !State.Mod4Zero))
      State.DstZero = false;
    if (!ST.hasVendorXTTTensixWH() &&
        Opcode != RISCV::TTSFPLOADMACROFP32Even) {
      unsigned Mode = Opcode == RISCV::TTSFPLOAD ? 4 : 3;
      // INT32_ALL has stack push/pop semantics, outside this FP32 profile.
      if (!MI.getOperand(Mode).isImm() || immediate(MI, Mode, 10))
        State.SpZero = false;
    }
    if (Opcode != RISCV::TTSFPSTORE &&
        MI.getOperand(0).getReg() == RISCV::TT_L0)
      State.L0.reset();
    return;
  }
  const auto &TRI = *ST.getRegisterInfo();
  if (MI.modifiesRegister(RISCV::TT_L0, &TRI))
    State.L0.reset();
  if (MI.modifiesRegister(RISCV::TT_CC, &TRI)) {
    State.Enable.reset();
    State.Flags.reset();
  }
  if (MI.modifiesRegister(RISCV::TT_CONFIG, &TRI) ||
      MI.modifiesRegister(RISCV::TT_DST, &TRI) || MI.isCall() ||
      MI.isInlineAsm() ||
      (!isSFPU(MI) && MI.mayStore() && !isProvenStackStore(MI)))
    State = {};
}

Error verifyCurrent(const MachineInstr &MI, const MacroState &State,
                    uint32_t TemplateBase, uint32_t TemplateMask) {
  unsigned Opcode = MI.getOpcode();
  bool Configure = Opcode == RISCV::TTSFPCONFIGMacroTemplate ||
                   Opcode == RISCV::TTSFPCONFIGMacroSequence ||
                   Opcode == RISCV::TTSFPCONFIGMacroMisc;
  if ((Configure || Opcode == RISCV::TTSFPLOADMACROFP32Even) &&
      !State.allActive())
    return createStringError("SFPLOADMACRO configuration and launch require "
                             "all-active zero LaneConfig");
  if (Opcode != RISCV::TTSFPLOADMACROFP32Even)
    return Error::success();
  if (Error E = verifyTensixMachineEffects(MI))
    return E;
  const auto &ST = MI.getMF()->getSubtarget<RISCVSubtarget>();
  if (Error E = verifyTensixMachineOperands(MI, MI.getMF()->getRegInfo(),
                                            *ST.getRegisterInfo()))
    return E;
  if (!State.addressZero(ST))
    return createStringError("SFPLOADMACRO requires actual zero Dst address "
                             "setup on every incoming path");
  unsigned Index = MI.getOperand(3).getImm();
  if (Index > 3 || !State.Templates[Index] ||
      (*State.Templates[Index] & ~TemplateMask) != TemplateBase ||
      State.Sequences[Index] != macroSequence(Index) ||
      State.Misc != MacroMiscFP32 || !State.ConfigReady)
    return createStringError("SFPLOADMACRO requires its actual ADDI template, "
                             "sequence and FP32 misc configuration");
  // Scheduled STORE occupies t+3 after the load; these actual SFPU load-unit
  // NOPs cannot collide with MAD/STORE and bound every outstanding countdown.
  // A STALLWAIT, scalar delay or CFG edge is intentionally not accepted here.
  auto It = std::next(MI.getIterator());
  for (unsigned I = 0; I != MacroDrainIssues; ++I) {
    while (It != MI.getParent()->end() && It->isDebugInstr())
      ++It;
    if (It == MI.getParent()->end() || It->getOpcode() != RISCV::TTSFPNOP)
      return createStringError("SFPLOADMACRO requires four actual consecutive "
                               "SFPNOP load-unit drain issues");
    ++It;
  }
  return Error::success();
}
} // namespace

bool llvm::isTensixMacroIntrinsic(Intrinsic::ID ID) {
  return ID == Intrinsic::riscv_tt_contents_macro_address_reset ||
         ID == Intrinsic::riscv_tt_contents_macro_configure_addi ||
         ID == Intrinsic::riscv_tt_contents_load_macro_addi;
}

Error llvm::verifyTensixMacroIntrinsicFields(const IntrinsicInst &II) {
  auto Index = [&](unsigned I, uint64_t Max) -> bool {
    auto *CI = dyn_cast<ConstantInt>(II.getArgOperand(I));
    return CI && CI->getValue().getActiveBits() <= 32 &&
           CI->getZExtValue() <= Max;
  };
  switch (II.getIntrinsicID()) {
  case Intrinsic::riscv_tt_contents_macro_address_reset:
    return Error::success();
  case Intrinsic::riscv_tt_contents_macro_configure_addi:
    if (!Index(0, 3))
      return createStringError("SFPLOADMACRO macro index must be in [0, 3]");
    if (!Index(1, 65535))
      return createStringError(
          "SFPLOADMACRO ADDI immediate must fit BF16 bits");
    return Error::success();
  case Intrinsic::riscv_tt_contents_load_macro_addi:
    if (!Index(1, 1022) ||
        (cast<ConstantInt>(II.getArgOperand(1))->getZExtValue() & 1))
      return createStringError("SFPLOADMACRO FP32 offset must be an even "
                               "constant in [0, 1022]");
    if (!Index(2, 3))
      return createStringError("SFPLOADMACRO macro index must be in [0, 3]");
    return Error::success();
  default:
    return createStringError("unknown SFPLOADMACRO intrinsic");
  }
}

SDValue llvm::lowerTensixMacroIntrinsic(SDValue Op, SelectionDAG &DAG,
                                        const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isTensixMacroIntrinsic(ID))
    return SDValue();
  if (!ST.hasVendorXTTTensixWH() && !ST.hasVendorXTTTensixBH())
    reportFatalUsageError(
        "SFPLOADMACRO requires a Wormhole or Blackhole target");
  const Function &F = DAG.getMachineFunction().getFunction();
  Attribute Executor = F.getFnAttribute("tensix-executor");
  if (!Executor.isStringAttribute() || Executor.getValueAsString() != "trisc1")
    reportFatalUsageError("SFPLOADMACRO requires the TRISC1 executor");
  SDLoc DL(Op);
  unsigned Opcode = RISCV::PseudoTTSFPMacroAddressReset;
  SmallVector<SDValue, 4> Operands;
  if (ID == Intrinsic::riscv_tt_contents_macro_configure_addi) {
    Opcode = RISCV::PseudoTTSFPMacroConfigureADDI;
    Operands.push_back(
        DAG.getTargetConstant(Op.getConstantOperandVal(2), DL, MVT::i32));
    Operands.push_back(
        DAG.getTargetConstant(Op.getConstantOperandVal(3), DL, MVT::i32));
  } else if (ID == Intrinsic::riscv_tt_contents_load_macro_addi) {
    Opcode = RISCV::PseudoTTSFPContentsLoadMacroADDI;
    Operands.push_back(Op.getOperand(2));
    Operands.push_back(
        DAG.getTargetConstant(Op.getConstantOperandVal(3) >> 1, DL, MVT::i32));
    Operands.push_back(
        DAG.getTargetConstant(Op.getConstantOperandVal(4), DL, MVT::i32));
  }
  Operands.push_back(Op.getOperand(0));
  bool Load = ID == Intrinsic::riscv_tt_contents_load_macro_addi;
  MachineSDNode *Node = DAG.getMachineNode(
      Opcode, DL,
      Load ? DAG.getVTList(MVT::v32i32, MVT::Other) : DAG.getVTList(MVT::Other),
      Operands);
  if (Load || ID == Intrinsic::riscv_tt_contents_macro_address_reset) {
    auto *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(),
        MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
            MachineMemOperand::MOVolatile,
        4, Align(4));
    DAG.setNodeMemRefs(Node, {MMO});
  }
  return SDValue(Node, 0);
}

bool llvm::expandTensixMacroPseudo(MachineInstr &MI, const RISCVSubtarget &ST) {
  unsigned Opcode = MI.getOpcode();
  if (Opcode != RISCV::PseudoTTSFPMacroAddressReset &&
      Opcode != RISCV::PseudoTTSFPMacroConfigureADDI &&
      Opcode != RISCV::PseudoTTSFPContentsLoadMacroADDI)
    return false;
  MachineFunction &MF = *MI.getMF();
  auto Reject = [&](const Twine &Message) {
    MF.getProperties().setFailedRegAlloc();
    MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
    MF.getFunction().getContext().diagnose(
        DiagnosticInfoUnsupported(MF.getFunction(), Message));
    return true;
  };
  if (!ST.hasVendorXTTTensixWH() && !ST.hasVendorXTTTensixBH())
    return Reject("SFPLOADMACRO requires a Wormhole or Blackhole target");
  if (Error E = verifyTensixMachineEffects(MI))
    return Reject(toString(std::move(E)));
  if (Error E = verifyTensixMachineOperands(MI, MF.getRegInfo(),
                                            *ST.getRegisterInfo()))
    return Reject(toString(std::move(E)));
  const auto &TII = *ST.getInstrInfo();
  auto Emit = [&](unsigned Native) {
    auto Builder =
        BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), TII.get(Native));
    Builder.setMIFlags(MI.getFlags());
    return Builder;
  };
  if (Opcode == RISCV::PseudoTTSFPMacroAddressReset) {
    Emit(RISCV::TTSETC16).addImm(0).addImm(0);
    if (!ST.hasVendorXTTTensixWH())
      // A bank-dependent write must not share the at-most-four .ttinsn
      // fusion group with CFG_STATE_ID. These are native issue NOPs.
      for (unsigned I = 0; I != BlackholeConfigStateGapIssues; ++I)
        Emit(RISCV::TTNOP);
    Emit(RISCV::TTSETC16).addImm(0).addImm(1);
    Emit(RISCV::TTRMWCIB0).addImm(6).addImm(0).addImm(255);
    Emit(RISCV::TTRMWCIB1).addImm(6).addImm(0).addImm(255);
    bool Wormhole = ST.hasVendorXTTTensixWH();
    if (Wormhole) {
      Emit(RISCV::TTSETC16).addImm(0).addImm(23);
      Emit(RISCV::TTSETC16).addImm(0).addImm(27);
    } else {
      // BH DEST_SP_BASE is config word7, not thread config. Its direct
      // AddrMod0 destination and A/B words are thread words28 and12.
      Emit(RISCV::TTRMWCIB0).addImm(7).addImm(0).addImm(255);
      Emit(RISCV::TTRMWCIB1).addImm(7).addImm(0).addImm(255);
      Emit(RISCV::TTSETC16).addImm(0).addImm(28);
      Emit(RISCV::TTSETC16).addImm(0).addImm(12);
    }
    Emit(RISCV::TTSETRWC)
        .addImm(Wormhole ? 4 : 20)
        .addImm(0)
        .addImm(0)
        .addImm(0)
        .addImm(0)
        .addImm(0);
  } else if (Opcode == RISCV::PseudoTTSFPMacroConfigureADDI) {
    unsigned Index = MI.getOperand(0).getImm();
    if (Index > 3 || !isUInt<16>(MI.getOperand(1).getImm()))
      return Reject(
          "SFPLOADMACRO configuration fields are outside the profile");
    auto Template = encodeADDITemplate(MF, MI.getOperand(1).getImm());
    if (!Template)
      return Reject(toString(Template.takeError()));
    auto LoadWord = [&](uint32_t Word) {
      // FLOATB writes upper16 and clears lower16. Under the independently
      // verified all-active LaneConfig it is a complete first L0 definition;
      // the subsequent LOWER write consumes that actual upper-half contents.
      Emit(RISCV::TTSFPLOADI)
          .addReg(RISCV::TT_L0, RegState::Define)
          .addReg(RISCV::TT_L0, RegState::Undef)
          .addImm(Word >> 16)
          .addImm(0);
      Emit(RISCV::TTSFPLOADI)
          .addReg(RISCV::TT_L0, RegState::Define)
          .addReg(RISCV::TT_L0)
          .addImm(Word & 65535)
          .addImm(10);
    };
    LoadWord(*Template);
    Emit(RISCV::TTSFPCONFIGMacroTemplate).addImm(Index);
    Emit(RISCV::TTSFPNOP);
    LoadWord(macroSequence(Index));
    Emit(RISCV::TTSFPCONFIGMacroSequence).addImm(Index);
    Emit(RISCV::TTSFPNOP);
    Emit(RISCV::TTSFPCONFIGMacroMisc).addImm(MacroMiscFP32);
    Emit(RISCV::TTSFPNOP);
  } else {
    Emit(RISCV::TTSFPLOADMACROFP32Even)
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .add(MI.getOperand(3))
        .cloneMemRefs(MI);
    for (unsigned I = 0; I != MacroDrainIssues; ++I)
      Emit(RISCV::TTSFPNOP);
  }
  MI.eraseFromParent();
  return true;
}

bool llvm::isTensixMacroInstruction(unsigned Opcode) {
  switch (Opcode) {
  case RISCV::TTSFPCONFIGMacroTemplate:
  case RISCV::TTSFPCONFIGMacroSequence:
  case RISCV::TTSFPCONFIGMacroMisc:
  case RISCV::TTSFPLOADMACROFP32Even:
  case RISCV::PseudoTTSFPMacroAddressReset:
  case RISCV::PseudoTTSFPMacroConfigureADDI:
  case RISCV::PseudoTTSFPContentsLoadMacroADDI:
    return true;
  default:
    return false;
  }
}

Error llvm::verifyTensixMacroMachineFunction(const MachineFunction &MF) {
  bool Uses = any_of(MF, [](const MachineBasicBlock &BB) {
    return any_of(BB, [](const MachineInstr &MI) {
      return isTensixMacroInstruction(MI.getOpcode());
    });
  });
  if (!Uses)
    return Error::success();
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasVendorXTTTensixWH() && !ST.hasVendorXTTTensixBH())
    return createStringError(
        "SFPLOADMACRO requires a Wormhole or Blackhole target");
  // MIR can bypass IR and post-RA expansion. Validate every native field
  // before dataflow reads it; final issue state never trusts a forged pseudo,
  // an omitted implicit effect, or an out-of-range truncated MC operand.
  for (const auto &BB : MF)
    for (const auto &MI : BB) {
      unsigned Opcode = MI.getOpcode();
      bool Native =
          isTensixMacroInstruction(Opcode) && RISCV::getTensixEncoding(Opcode);
      bool Setup = Opcode == RISCV::TTSETC16 || Opcode == RISCV::TTSETRWC ||
                   Opcode == RISCV::TTRMWCIB0 || Opcode == RISCV::TTRMWCIB1 ||
                   Opcode == RISCV::TTSTALLWAIT || Opcode == RISCV::TTNOP;
      bool Lane = Opcode == RISCV::TTSFPENCC ||
                  Opcode == RISCV::TTSFPCONFIGReset ||
                  Opcode == RISCV::TTSFPLOADI || Opcode == RISCV::TTSFPNOP;
      if (!Native && !Setup && !Lane)
        continue;
      if (Error E = verifyTensixMachineEffects(MI))
        return E;
      // Only the first complete FLOATB setup definition may have an undef
      // old L0. Its all-active precondition is checked from reaching native
      // ENCC/LaneConfig state before any subsequent configuration consumes it.
      bool FullL0 =
          Opcode == RISCV::TTSFPLOADI && MI.getNumExplicitOperands() == 4 &&
          MI.getOperand(0).isReg() && MI.getOperand(0).isDef() &&
          MI.getOperand(0).getReg() == RISCV::TT_L0 &&
          !MI.getOperand(0).getSubReg() && !MI.getOperand(0).isEarlyClobber() &&
          !MI.getOperand(0).isInternalRead() && MI.getOperand(1).isReg() &&
          MI.getOperand(1).isUse() &&
          MI.getOperand(1).getReg() == RISCV::TT_L0 &&
          !MI.getOperand(1).getSubReg() && !MI.getOperand(1).isInternalRead() &&
          MI.getOperand(1).isUndef() && immediate(MI, 3, 0);
      if (!FullL0)
        if (Error E = verifyTensixMachineOperands(MI, MF.getRegInfo(),
                                                  *ST.getRegisterInfo()))
          return E;
      MCInst MC;
      MC.setOpcode(Opcode);
      for (const auto &MO : MI.explicit_operands()) {
        if (MO.isReg() && MO.getReg().isPhysical() && !MO.getSubReg())
          MC.addOperand(MCOperand::createReg(MO.getReg()));
        else if (MO.isImm())
          MC.addOperand(MCOperand::createImm(MO.getImm()));
        else
          return createStringError("SFPLOADMACRO final issue requires physical "
                                   "registers and logical immediates");
      }
      Error Fields = ST.hasVendorXTTTensixWH()
                         ? RISCV::verifyTensixWormholeInstruction(
                               MC, *ST.getInstrInfo())
                         : RISCV::verifyTensixMCInstructionFields(
                               MC, *ST.getInstrInfo());
      if (Fields)
        return Fields;
      if (ST.hasVendorXTTTensixBH() && Opcode == RISCV::TTSETC16 &&
          immediate(MI, 1, 0)) {
        // A bank-dependent write must not be fused with CFG_STATE_ID.
        // Keep this native profile as strict as the emitted reset sequence;
        // an attribute or a CFG edge cannot establish the issue separation.
        auto It = std::next(MI.getIterator());
        for (unsigned I = 0; I != BlackholeConfigStateGapIssues; ++I) {
          while (It != BB.end() &&
                 (It->isDebugInstr() || It->isCFIInstruction()))
            ++It;
          if (It == BB.end() || It->getOpcode() != RISCV::TTNOP)
            return createStringError(
                "SFPLOADMACRO Blackhole CFG_STATE_ID requires three actual "
                "consecutive TTNOP issue separators");
          ++It;
        }
      }
    }
  auto Base = encodeADDITemplate(MF, 0);
  if (!Base)
    return Base.takeError();
  auto Maximum = encodeADDITemplate(MF, 65535);
  if (!Maximum)
    return Maximum.takeError();
  uint32_t TemplateMask = *Maximum ^ *Base;
  DenseMap<const MachineBasicBlock *, MacroState> Outputs;
  bool Changed;
  do {
    Changed = false;
    for (const auto &BB : MF) {
      std::optional<MacroState> Incoming;
      if (&BB == &MF.front())
        Incoming.emplace();
      for (const auto *Pred : BB.predecessors()) {
        auto It = Outputs.find(Pred);
        if (It == Outputs.end())
          continue;
        if (Incoming)
          Incoming->meet(It->second);
        else
          Incoming = It->second;
      }
      if (!Incoming)
        continue;
      MacroState State = *Incoming;
      for (const auto &MI : BB)
        advance(MI, State, ST);
      auto [It, Inserted] = Outputs.try_emplace(&BB, State);
      if (Inserted || It->second != State) {
        It->second = State;
        Changed = true;
      }
    }
  } while (Changed);
  for (const auto &BB : MF) {
    if (!Outputs.contains(&BB))
      continue;
    std::optional<MacroState> Incoming;
    if (&BB == &MF.front())
      Incoming.emplace();
    for (const auto *Pred : BB.predecessors()) {
      auto It = Outputs.find(Pred);
      if (It == Outputs.end())
        continue;
      if (Incoming)
        Incoming->meet(It->second);
      else
        Incoming = It->second;
    }
    MacroState State = *Incoming;
    for (const auto &MI : BB) {
      if (isTensixMacroInstruction(MI.getOpcode()) &&
          !RISCV::getTensixEncoding(MI.getOpcode()))
        return createStringError("unexpanded SFPLOADMACRO pseudo reached "
                                 "final issue validation");
      if (Error E = verifyCurrent(MI, State, *Base, TemplateMask))
        return E;
      advance(MI, State, ST);
    }
  }
  return Error::success();
}
