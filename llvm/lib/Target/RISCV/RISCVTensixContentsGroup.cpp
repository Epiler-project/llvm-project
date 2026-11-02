//===-- RISCVTensixContentsGroup.cpp -------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixContentsGroup.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVTensixBoundContract.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/ErrorHandling.h"
#include <optional>

using namespace llvm;

namespace {
Error verifyConfigurationFields(unsigned Opcode, uint64_t Mask, uint64_t Mode,
                                const MCInstrInfo &Instructions) {
  MCInst Native;
  Native.setOpcode(Opcode);
  Native.addOperand(MCOperand::createImm(Mask));
  Native.addOperand(MCOperand::createImm(Mode));
  return RISCV::verifyTensixMCInstructionFields(Native, Instructions);
}
} // namespace

unsigned llvm::getTensixContentsGroupResultCount(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_tt_contents_transpose8:
  case Intrinsic::riscv_tt_contents_lutfp32_three:
    return 8;
  case Intrinsic::riscv_tt_contents_copy4:
  case Intrinsic::riscv_tt_contents_rotate4:
    return 4;
  case Intrinsic::riscv_tt_contents_swap:
    return 2;
  case Intrinsic::riscv_tt_contents_lut:
  case Intrinsic::riscv_tt_contents_lutfp32_six:
    return 1;
  default:
    return 0;
  }
}

bool llvm::isTensixContentsGroupIntrinsic(Intrinsic::ID ID) {
  // Configuration changes architectural state and has no numerical result.
  return getTensixContentsGroupResultCount(ID) != 0 ||
         ID == Intrinsic::riscv_tt_contents_config_creg ||
         ID == Intrinsic::riscv_tt_contents_config_lane;
}

Error llvm::verifyTensixContentsGroupIntrinsicFields(const IntrinsicInst &II) {
  Intrinsic::ID ID = II.getIntrinsicID();
  if (!isTensixContentsGroupIntrinsic(ID))
    return createStringError("unknown contents group intrinsic");
  unsigned Count = getTensixContentsGroupResultCount(ID);
  Type *Carrier = FixedVectorType::get(Type::getInt32Ty(II.getContext()), 32);
  if (Count > 1) {
    auto *Results = dyn_cast<StructType>(II.getType());
    if (!Results || !Results->isLiteral() || Results->isPacked() ||
        Results->getNumElements() != Count)
      return createStringError(
          "contents group requires its literal result tuple");
    for (Type *Element : Results->elements())
      if (Element != Carrier)
        return createStringError(
            "contents group requires complete SFPU contents");
  } else if (Count ? II.getType() != Carrier : !II.getType()->isVoidTy()) {
    return createStringError("contents group has an invalid numerical result");
  }
  unsigned Inputs = Count;
  switch (ID) {
  case Intrinsic::riscv_tt_contents_rotate4:
  case Intrinsic::riscv_tt_contents_lut:
    Inputs = 5;
    break;
  case Intrinsic::riscv_tt_contents_lutfp32_six:
    Inputs = 8;
    break;
  case Intrinsic::riscv_tt_contents_config_creg:
  case Intrinsic::riscv_tt_contents_config_lane:
    Inputs = 1;
    break;
  default:
    break;
  }
  for (unsigned I = 0; I != Inputs; ++I)
    if (II.getArgOperand(I)->getType() != Carrier)
      return createStringError(
          "contents group requires complete SFPU contents");
  auto Immediate = [&](unsigned Index) -> std::optional<uint64_t> {
    const auto *Value = dyn_cast<ConstantInt>(II.getArgOperand(Index));
    if (!Value || Value->getValue().getActiveBits() > 32)
      return std::nullopt;
    return Value->getZExtValue();
  };
  switch (ID) {
  case Intrinsic::riscv_tt_contents_copy4:
    if (!Immediate(4) || *Immediate(4) > 1)
      return createStringError("contents copy4 mode must be 0 or 1");
    break;
  case Intrinsic::riscv_tt_contents_swap:
    if (!Immediate(2) || *Immediate(2) > 9)
      return createStringError("contents swap mode must be in 0..9");
    break;
  case Intrinsic::riscv_tt_contents_lut:
    if (!Immediate(5) || (*Immediate(5) != 0 && *Immediate(5) != 4))
      return createStringError("contents LUT mode must be 0 or 4");
    break;
  case Intrinsic::riscv_tt_contents_lutfp32_six:
    if (!Immediate(8) || *Immediate(8) > 7 || !(0x00dd & (1u << *Immediate(8))))
      return createStringError("contents LUT six mode must be 0,2,3,4,6 or 7");
    break;
  case Intrinsic::riscv_tt_contents_lutfp32_three:
    if (!Immediate(8) || *Immediate(8) > 1)
      return createStringError("contents LUT three sign must be 0 or 1");
    break;
  case Intrinsic::riscv_tt_contents_config_creg:
    if (!Immediate(1) || *Immediate(1) < 11 || *Immediate(1) > 14)
      return createStringError("contents configuration requires C11..C14");
    if (!Immediate(2) || *Immediate(2) > 0xffff)
      return createStringError("contents configuration mask must fit u16");
    if (!Immediate(3) || (*Immediate(3) != 0 && *Immediate(3) != 8))
      return createStringError(
          "contents CReg configuration mode must be 0 or 8");
    break;
  case Intrinsic::riscv_tt_contents_config_lane:
    if (!Immediate(1) || *Immediate(1) > 0xffff)
      return createStringError("contents configuration mask must fit u16");
    if (!Immediate(2) || *Immediate(2) != 8)
      return createStringError("contents lane configuration mode must be 8");
    break;
  default:
    break;
  }
  if (ID == Intrinsic::riscv_tt_contents_config_creg ||
      ID == Intrinsic::riscv_tt_contents_config_lane) {
    static const auto Instructions = RISCV::createTensixBoundMCInstrInfo();
    if (ID == Intrinsic::riscv_tt_contents_config_lane)
      return verifyConfigurationFields(RISCV::TTSFPCONFIGLane, *Immediate(1),
                                       *Immediate(2), *Instructions);
    static const unsigned Opcodes[] = {
        RISCV::TTSFPCONFIGC11, RISCV::TTSFPCONFIGC12, RISCV::TTSFPCONFIGC13,
        RISCV::TTSFPCONFIGC14};
    return verifyConfigurationFields(Opcodes[*Immediate(1) - 11], *Immediate(2),
                                     *Immediate(3), *Instructions);
  }
  return Error::success();
}

SDValue llvm::lowerTensixContentsGroupIntrinsic(SDValue Op, SelectionDAG &DAG,
                                                const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isTensixContentsGroupIntrinsic(ID))
    return SDValue();
  unsigned Count = getTensixContentsGroupResultCount(ID);
  const Function &F = DAG.getMachineFunction().getFunction();
  Attribute Executor = F.getFnAttribute("tensix-executor");
  if (!ST.hasVendorXTTTensixBH() || !Executor.isStringAttribute() ||
      Executor.getValueAsString() != "trisc1")
    reportFatalUsageError("contents groups require Blackhole TRISC1");

  unsigned Opcode;
  switch (ID) {
  case Intrinsic::riscv_tt_contents_transpose8:
    Opcode = RISCV::PseudoTTSFPContentsTranspose8;
    break;
  case Intrinsic::riscv_tt_contents_copy4:
    Opcode = RISCV::PseudoTTSFPContentsCopy4;
    break;
  case Intrinsic::riscv_tt_contents_rotate4:
    Opcode = RISCV::PseudoTTSFPContentsRotate4;
    break;
  case Intrinsic::riscv_tt_contents_swap:
    Opcode = RISCV::TTSFPSWAP;
    break;
  case Intrinsic::riscv_tt_contents_lut:
    Opcode = RISCV::PseudoTTSFPContentsLUT;
    break;
  case Intrinsic::riscv_tt_contents_lutfp32_six:
    Opcode = RISCV::PseudoTTSFPContentsLUTSix;
    break;
  case Intrinsic::riscv_tt_contents_lutfp32_three:
    Opcode = RISCV::PseudoTTSFPContentsLUTThree;
    break;
  case Intrinsic::riscv_tt_contents_config_creg: {
    unsigned Register = Op.getConstantOperandVal(3);
    if (Register < 11 || Register > 14)
      reportFatalUsageError("contents configuration requires C11..C14");
    static const unsigned Opcodes[] = {RISCV::PseudoTTSFPContentsConfigC11,
                                       RISCV::PseudoTTSFPContentsConfigC12,
                                       RISCV::PseudoTTSFPContentsConfigC13,
                                       RISCV::PseudoTTSFPContentsConfigC14};
    Opcode = Opcodes[Register - 11];
    break;
  }
  case Intrinsic::riscv_tt_contents_config_lane:
    Opcode = RISCV::PseudoTTSFPContentsConfigLane;
    break;
  default:
    llvm_unreachable("classified contents group has no lowering");
  }
  SDLoc DL(Op);
  SmallVector<SDValue, 10> Operands;
  for (unsigned I = 2; I != Op.getNumOperands(); ++I) {
    if (ID == Intrinsic::riscv_tt_contents_config_creg && I == 3)
      continue;
    SDValue Input = Op.getOperand(I);
    if (const auto *Immediate = dyn_cast<ConstantSDNode>(Input))
      Input = DAG.getTargetConstant(Immediate->getSExtValue(), DL, MVT::i32);
    Operands.push_back(Input);
  }
  Operands.push_back(Op.getOperand(0));
  SmallVector<EVT, 9> Results(Count, MVT::v32i32);
  Results.push_back(MVT::Other);
  return SDValue(
      DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results), Operands), 0);
}

bool llvm::expandTensixContentsGroupPseudo(MachineInstr &MI,
                                           const RISCVSubtarget &ST) {
  unsigned Count, Opcode, FixedStart, FixedCount, Extras;
  switch (MI.getOpcode()) {
  case RISCV::PseudoTTSFPContentsTranspose8:
    Count = 8;
    Opcode = RISCV::TTSFPTRANSP;
    FixedStart = 8;
    FixedCount = 8;
    Extras = 0;
    break;
  case RISCV::PseudoTTSFPContentsCopy4:
    Count = 4;
    Opcode = RISCV::TTSFPSHFT2Copy4;
    FixedStart = 4;
    FixedCount = 4;
    Extras = 1;
    break;
  case RISCV::PseudoTTSFPContentsRotate4:
    Count = 4;
    Opcode = RISCV::TTSFPSHFT2Rotate4;
    FixedStart = 4;
    FixedCount = 4;
    Extras = 1;
    break;
  case RISCV::PseudoTTSFPContentsLUTThree:
    Count = 8;
    Opcode = RISCV::TTSFPLUTFP32Three;
    FixedStart = 8;
    FixedCount = 8;
    Extras = 1;
    break;
  case RISCV::PseudoTTSFPContentsLUT:
  case RISCV::PseudoTTSFPContentsLUTSix:
    Count = 1;
    Opcode = MI.getOpcode() == RISCV::PseudoTTSFPContentsLUT
                 ? RISCV::TTSFPLUT
                 : RISCV::TTSFPLUTFP32Six;
    FixedStart = 2;
    FixedCount = Opcode == RISCV::TTSFPLUT ? 4 : 7;
    Extras = 1;
    break;
  case RISCV::PseudoTTSFPContentsConfigC11:
  case RISCV::PseudoTTSFPContentsConfigC12:
  case RISCV::PseudoTTSFPContentsConfigC13:
  case RISCV::PseudoTTSFPContentsConfigC14:
  case RISCV::PseudoTTSFPContentsConfigLane:
    Count = 0;
    FixedStart = 0;
    FixedCount = 1;
    Extras = 2;
    switch (MI.getOpcode()) {
    case RISCV::PseudoTTSFPContentsConfigC11:
      Opcode = RISCV::TTSFPCONFIGC11;
      break;
    case RISCV::PseudoTTSFPContentsConfigC12:
      Opcode = RISCV::TTSFPCONFIGC12;
      break;
    case RISCV::PseudoTTSFPContentsConfigC13:
      Opcode = RISCV::TTSFPCONFIGC13;
      break;
    case RISCV::PseudoTTSFPContentsConfigC14:
      Opcode = RISCV::TTSFPCONFIGC14;
      break;
    default:
      Opcode = RISCV::TTSFPCONFIGLane;
      break;
    }
    break;
  default:
    return false;
  }

  MachineFunction &MF = *MI.getMF();
  auto Reject = [&](const Twine &Message) {
    // Keep the failed pseudo's real definitions and uses for subsequent
    // verification. The printer suppresses the entire failed function.
    MF.getProperties().setFailedRegAlloc();
    MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
    MF.getFunction().getContext().diagnose(
        DiagnosticInfoUnsupported(MF.getFunction(), Message));
    return true;
  };
  auto IsUse = [](const MachineOperand &MO, Register Expected) {
    return MO.isReg() && MO.isUse() && MO.getReg() == Expected &&
           !MO.getSubReg() && !MO.isUndef();
  };
  auto IsDef = [](const MachineOperand &MO, Register Expected) {
    return MO.isReg() && MO.isDef() && MO.getReg() == Expected &&
           !MO.getSubReg();
  };
  unsigned ExtraStart = FixedStart + FixedCount;
  if (MI.getNumExplicitOperands() != ExtraStart + Extras)
    return Reject("contents group has a malformed machine tuple");
  for (unsigned I = 0; I != FixedCount; ++I) {
    Register Expected = RISCV::TT_L0 + I;
    if (!IsUse(MI.getOperand(FixedStart + I), Expected) ||
        (Count > 1 && !IsDef(MI.getOperand(I), Expected)))
      return Reject("contents group requires its actual fixed-register ties");
  }
  if (Count == 1) {
    const MachineOperand &Output = MI.getOperand(0);
    if (!Output.isReg() || !RISCV::SFPRRegClass.contains(Output.getReg()) ||
        !IsDef(Output, Output.getReg()) ||
        !IsUse(MI.getOperand(1), Output.getReg()))
      return Reject("contents LUT requires its allocated destination tie");
  }
  auto Immediate = [&](unsigned Index) -> std::optional<int64_t> {
    const auto &Operand = MI.getOperand(Index);
    return Operand.isImm() ? std::optional<int64_t>(Operand.getImm())
                           : std::nullopt;
  };
  if (Opcode == RISCV::TTSFPSHFT2Rotate4) {
    const MachineOperand &Source = MI.getOperand(ExtraStart);
    if (!Source.isReg() || !Source.isUse() || Source.getSubReg() ||
        Source.isUndef() || !RISCV::SFPRReadRegClass.contains(Source.getReg()))
      return Reject("contents rotate4 requires an allocated SFPU source");
  } else if (Extras) {
    auto Mode = Immediate(ExtraStart + Extras - 1);
    if (!Mode)
      return Reject("contents group requires a logical immediate mode");
    if (Opcode == RISCV::TTSFPSHFT2Copy4 && (*Mode < 0 || *Mode > 1))
      return Reject("contents copy4 mode must be 0 or 1");
    if (Opcode == RISCV::TTSFPLUTFP32Three && (*Mode < 0 || *Mode > 1))
      return Reject("contents LUT three sign must be 0 or 1");
    if (Opcode == RISCV::TTSFPLUT && *Mode != 0 && *Mode != 4)
      return Reject("contents LUT mode must be 0 or 4");
    if (Opcode == RISCV::TTSFPLUTFP32Six &&
        (*Mode < 0 || *Mode > 7 || !(0x00dd & (1u << *Mode))))
      return Reject("contents LUT six mode must be 0,2,3,4,6 or 7");
    if (!Count) {
      auto Mask = Immediate(ExtraStart);
      if (!Mask || *Mask < 0 || *Mask > 0xffff)
        return Reject("contents configuration mask must fit u16");
      if (Opcode == RISCV::TTSFPCONFIGLane ? *Mode != 8
                                           : (*Mode != 0 && *Mode != 8))
        return Reject("contents configuration has an unsupported mode");
      if (Error E = verifyConfigurationFields(Opcode, *Mask, *Mode,
                                              *ST.getInstrInfo()))
        return Reject(toString(std::move(E)));
    }
  }

  MachineInstrBuilder Native = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                                       ST.getInstrInfo()->get(Opcode));
  if (Count == 1) {
    Native.add(MI.getOperand(0));
    Native.add(MI.getOperand(1));
  }
  for (unsigned I = 0; I != Extras; ++I)
    Native.add(MI.getOperand(ExtraStart + I));
  if (Opcode == RISCV::TTSFPTRANSP) {
    // The legacy bound form discards the upper four outputs and its base
    // descriptor only needs the lower inputs. This full contents form also
    // consumes L4..L7. Preserve those real dependencies on the native MI.
    for (unsigned I = 4; I != 8; ++I)
      Native.addReg(RISCV::TT_L0 + I, RegState::Implicit);
  }
  Native.setMIFlags(MI.getFlags());
  MI.eraseFromParent();
  return true;
}
