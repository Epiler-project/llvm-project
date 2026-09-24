//===-- RISCVBaseInfo.cpp - Top level definitions for RISC-V MC -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains small standalone enum definitions for the RISC-V target
// useful for the compiler back-end and the MC libraries.
//
//===----------------------------------------------------------------------===//

#include "RISCVBaseInfo.h"
#include "RISCVMCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

namespace RISCVSysReg {
#define GET_SysRegsList_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCVSysReg

namespace RISCVInsnOpcode {
#define GET_RISCVOpcodesList_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCVInsnOpcode

namespace RISCVVInversePseudosTable {
using namespace RISCV;
#define GET_RISCVVInversePseudosTable_IMPL
#include "RISCVGenSearchableTables.inc"
} // namespace RISCVVInversePseudosTable

namespace RISCV {
#define GET_RISCVVSSEGTable_IMPL
#define GET_RISCVVLSEGTable_IMPL
#define GET_RISCVVLXSEGTable_IMPL
#define GET_RISCVVSXSEGTable_IMPL
#define GET_RISCVVLETable_IMPL
#define GET_RISCVVSETable_IMPL
#define GET_RISCVVLXTable_IMPL
#define GET_RISCVVSXTable_IMPL
#define GET_RISCVNDSVLNTable_IMPL
#define GET_RISCVTensixEncodingTable_IMPL
#define GET_RISCVTensixInstructionTable_IMPL
#define GET_RISCVTensixMachineTable_IMPL
#define GET_RISCVTensixFieldTable_IMPL
#include "RISCVGenSearchableTables.inc"

uint32_t getTensixInstructionPortAddress(TensixInstructionPort Port) {
  switch (Port) {
  case TensixInstructionPort::Local:
    return 0xffe40000;
  case TensixInstructionPort::BriscToTrisc1:
    return 0xffe50000;
  case TensixInstructionPort::BriscToTrisc2:
    return 0xffe60000;
  }
  llvm_unreachable("unknown Tensix instruction port");
}

Error verifyTensixFeatureBits(const Triple &TT, const FeatureBitset &FeatureBits) {
  if (FeatureBits[RISCV::FeatureVendorXTTTensixBH]) {
    if (TT.getArch() != Triple::riscv32 || !TT.isLittleEndian())
      return createStringError("XTTTensixBH requires RV32 little-endian");
    if (FeatureBits[RISCV::FeatureStdExtC] ||
        FeatureBits[RISCV::FeatureStdExtZca] ||
        FeatureBits[RISCV::FeatureStdExtZcb] ||
        FeatureBits[RISCV::FeatureStdExtZcd] ||
        FeatureBits[RISCV::FeatureStdExtZce] ||
        FeatureBits[RISCV::FeatureStdExtZcf] ||
        FeatureBits[RISCV::FeatureStdExtZclsd] ||
        FeatureBits[RISCV::FeatureStdExtZcmop] ||
        FeatureBits[RISCV::FeatureStdExtZcmp] ||
        FeatureBits[RISCV::FeatureStdExtZcmt])
      return createStringError("XTTTensixBH is incompatible with C/Zc extensions");
    if (FeatureBits[RISCV::FeatureStdExtV] ||
        FeatureBits[RISCV::FeatureStdExtZve32x] ||
        FeatureBits[RISCV::FeatureStdExtZve32f] ||
        FeatureBits[RISCV::FeatureStdExtZve64x] ||
        FeatureBits[RISCV::FeatureStdExtZve64f] ||
        FeatureBits[RISCV::FeatureStdExtZve64d])
      return createStringError(
          "XTTTensixBH is incompatible with V/Zve extensions");
  }
  return Error::success();
}

const TensixInstructionInfo *getTensixInstructionByName(StringRef Name) {
  return lookupTensixInstructionByName(Name.upper());
}

StringRef TensixInstructionInfo::getName() const {
  return getTensixInstructionInfoStr(Name);
}

StringRef TensixInstructionField::getName() const {
  return getTensixInstructionFieldStr(Name);
}

const TensixInstructionInfo *
getTensixInstructionByIntrinsic(Intrinsic::ID ID) {
  if (const auto *Info = lookupTensixInstruction(ID))
    return Info;
  if (const auto *Info = lookupTensixInstructionByPort(ID))
    return Info;
  return lookupTensixInstructionByMop(ID);
}

const TensixInstructionField *
getTensixInstructionField(const TensixInstructionInfo &Info, unsigned Index) {
  if (Index >= Info.NumFields)
    return nullptr;
  return lookupTensixInstructionField(Info.IntrinsicID, Index);
}

Error verifyTensixMCInstruction(const MCInst &MI, const MCInstrInfo &MCII,
                                const MCRegisterInfo &MRI) {
  const TensixEncoding *Encoding = getTensixEncoding(MI.getOpcode());
  if (!Encoding)
    return Error::success();

  const MCInstrDesc &Desc = MCII.get(MI.getOpcode());
  if (MI.getNumOperands() != Desc.getNumOperands())
    return createStringError("incorrect number of Tensix instruction operands");
  if (const auto *Machine = getTensixMachineInfo(MI.getOpcode())) {
    const auto *Info = lookupTensixInstruction(Machine->IntrinsicID);
    for (unsigned I = 0; I != Info->NumFields; ++I) {
      const auto *Field = getTensixInstructionField(*Info, I);
      const MCOperand &Op = MI.getOperand(I);
      if (!Op.isImm() || Op.getImm() < 0 ||
          uint64_t(Op.getImm()) > Field->MaxValue)
        return createStringError(Twine(Info->getName()) + " field " + Field->getName() +
                                 " must be in [0, " +
                                 Twine(Field->MaxValue) + "]");
    }
    return Error::success();
  }
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I) {
    const MCOperand &Op = MI.getOperand(I);
    const MCOperandInfo &Info = Desc.operands()[I];
    if (Info.RegClass >= 0) {
      if (!Op.isReg() || !MRI.getRegClass(Info.RegClass).contains(Op.getReg()))
        return createStringError(
            "invalid register for Tensix instruction operand");
      int Tied = Desc.getOperandConstraint(I, MCOI::TIED_TO);
      if (Tied >= 0 && (!MI.getOperand(Tied).isReg() ||
                        MI.getOperand(Tied).getReg() != Op.getReg()))
        return createStringError(
            "Tensix destination and passthrough must be tied");
      continue;
    }
    if (!Op.isImm())
      return createStringError(
          "Tensix instruction requires immediate operands");
    unsigned Bits;
    switch (Info.OperandType) {
    case RISCVOp::OPERAND_UIMM2:
      Bits = 2;
      break;
    case RISCVOp::OPERAND_UIMM3:
      Bits = 3;
      break;
    case RISCVOp::OPERAND_UIMM4:
      Bits = 4;
      break;
    case RISCVOp::OPERAND_UIMM8:
      Bits = 8;
      break;
    case RISCVOp::OPERAND_UIMM10:
      Bits = 10;
      break;
    case RISCVOp::OPERAND_UIMM12:
      Bits = 12;
      break;
    case RISCVOp::OPERAND_UIMM16:
      Bits = 16;
      break;
    case RISCVOp::OPERAND_SIMM12:
      if (!isInt<12>(Op.getImm()))
        return createStringError("Tensix immediate operand must fit in 12 signed bits");
      continue;
    default:
      return createStringError("unsupported Tensix immediate operand contract");
    }
    if (!isUIntN(Bits, Op.getImm()))
      return createStringError("Tensix immediate operand must fit in " +
                               Twine(Bits) + " unsigned bits");
  }

  if (Encoding->ConfigCount &&
      uint64_t(MI.getOperand(0).getImm()) >= Encoding->ConfigCount)
    return createStringError("SETC16 configuration index must be in [0, 67]");

  if (Encoding->AllowedModes) {
    if (Encoding->ModeOperandIndex >= MI.getNumOperands() ||
        !MI.getOperand(Encoding->ModeOperandIndex).isImm())
      return createStringError("invalid Tensix mode operand contract");
    uint64_t Mode = MI.getOperand(Encoding->ModeOperandIndex).getImm();
    if (Mode >= 16 || !(Encoding->AllowedModes & (uint16_t(1) << Mode))) {
      std::string Message;
      raw_string_ostream OS(Message);
      OS << MCII.getName(MI.getOpcode()).drop_front(2)
         << ((MI.getOpcode() == RISCV::TTSFPLOAD ||
              MI.getOpcode() == RISCV::TTSFPSTORE)
                 ? " format"
                 : " mode")
         << " must be one of {";
      bool First = true;
      for (unsigned M = 0; M != 16; ++M)
        if (Encoding->AllowedModes & (uint16_t(1) << M)) {
          if (!First)
            OS << ", ";
          OS << M;
          First = false;
        }
      OS << '}';
      return createStringError(Message);
    }
  }
  switch (MI.getOpcode()) {
  case RISCV::TTSFPCONFIGC11:
  case RISCV::TTSFPCONFIGC12:
  case RISCV::TTSFPCONFIGC13:
  case RISCV::TTSFPCONFIGC14: {
    uint64_t Mask = MI.getOperand(0).getImm();
    unsigned Mode = MI.getOperand(1).getImm();
    if (Mode == 0 && Mask != 0)
      return createStringError("SFPCONFIG mode 0 requires a zero mask");
    if (Mode == 8 && (Mask & ~uint64_t(0x5555)) != 0)
      return createStringError(
          "SFPCONFIG mode 8 mask must select even lane bits");
    break;
  }
  case RISCV::TTSFPSETEXP:
  case RISCV::TTSFPSETMAN:
  case RISCV::TTSFPSETSGN: {
    unsigned Mode = MI.getOperand(4).getImm();
    int64_t Value = MI.getOperand(3).getImm();
    unsigned Max = MI.getOpcode() == RISCV::TTSFPSETEXP ? 255
                 : MI.getOpcode() == RISCV::TTSFPSETMAN ? 4095 : 1;
    if (!(Mode & 1))
      Max = 0;
    if (uint64_t(Value) > Max)
      return createStringError("Tensix field immediate must be in [0, " +
                               Twine(Max) + "] for this mode");
    break;
  }
  case RISCV::TTSFPSHFT:
    if (!(MI.getOperand(4).getImm() & 1) && MI.getOperand(3).getImm() != 0)
      return createStringError("Tensix vector shift requires zero immediate field");
    break;
  case RISCV::TTSFPSTOCHRNDI: {
    unsigned Mode = MI.getOperand(4).getImm();
    uint64_t Descale = MI.getOperand(3).getImm();
    if (Descale > (Mode == 12 || Mode == 13 ? 31u : 0u))
      return createStringError("Tensix floating conversion requires zero descale; integer descale must fit five bits");
    [[fallthrough]];
  }
  case RISCV::TTSFPSTOCHRNDV:
    if (MI.getOperand(5).getImm() > 2)
      return createStringError("Tensix stochastic rounding mode must be in [0, 2]");
    break;
  default:
    break;
  }
  return Error::success();
}
} // namespace RISCV

namespace RISCVABI {
Expected<ABI> computeTargetABI(const MCSubtargetInfo &STI, StringRef ABIName) {
  const Triple &TT = STI.getTargetTriple();
  const FeatureBitset &FeatureBits = STI.getFeatureBits();
  auto TargetABI = getTargetABI(ABIName);
  bool IsRV64 = TT.isArch64Bit();
  bool IsRVE = FeatureBits[RISCV::FeatureStdExtE];
  bool IsXCheriot = FeatureBits[RISCV::FeatureVendorXCheriot];

  if (!ABIName.empty() && TargetABI == ABI_Unknown) {
    return createStringError(Twine("'") + ABIName +
                             "' is not a recognized ABI for this target");
  }
  if (IsRV64 &&
      (ABIName.starts_with("ilp32") || ABIName.starts_with("il32pc64"))) {
    return createStringError(
        "32-bit ABIs are not supported for 64-bit targets");
  }
  if (!IsRV64 &&
      (ABIName.starts_with("lp64") || ABIName.starts_with("l64pc128"))) {
    return createStringError(
        "64-bit ABIs are not supported for 32-bit targets");
  }
  if (ABIName.ends_with('f') && !FeatureBits[RISCV::FeatureStdExtF]) {
    return createStringError(
        "hard-float 'f' ABI can't be used for a target that doesn't "
        "support the F instruction set extension");
  }
  if (ABIName.ends_with('d') && !FeatureBits[RISCV::FeatureStdExtD]) {
    return createStringError(
        "hard-float 'd' ABI can't be used for a target that doesn't "
        "support the D instruction set extension");
  }
  if (!FeatureBits[RISCV::FeatureStdExtY] &&
      (ABIName.starts_with("il32pc64") || ABIName.starts_with("l64pc128"))) {
    return createStringError(Twine('\'') + ABIName +
                             "' ABI is only supported for RVY targets");
  }
  if (!IsRV64 && IsRVE && !IsXCheriot && TargetABI != ABI_ILP32E &&
      TargetABI != ABI_Unknown) {
    return createStringError("only the ilp32e ABI is supported for RV32E");
  }
  if (!IsRV64 && IsRVE && IsXCheriot && TargetABI != ABI_CHERIOT &&
      TargetABI != ABI_Unknown) {
    return createStringError("only the cheriot ABI is supported for XCheriot");
  }
  if (IsRV64 && IsRVE && TargetABI != ABI_LP64E && TargetABI != ABI_Unknown) {
    return createStringError("only the lp64e ABI is supported for RV64E");
  }

  // Unconditionally fatal: no sensible default ABI to fall back to here.
  if ((TargetABI == ABI_ILP32E ||
       (TargetABI == ABI_Unknown && IsRVE && !IsRV64)) &&
      FeatureBits[RISCV::FeatureStdExtD])
    reportFatalUsageError("ILP32E cannot be used with the D ISA extension");

  if (TargetABI != ABI_Unknown)
    return TargetABI;

  // If no explicit ABI is given, try to compute the default ABI.
  auto ISAInfo = RISCVFeatures::parseFeatureBits(STI);
  if (!ISAInfo)
    reportFatalUsageError(ISAInfo.takeError());
  return getTargetABI((*ISAInfo)->computeDefaultABI());
}

ABI getTargetABI(StringRef ABIName) {
  auto TargetABI = StringSwitch<ABI>(ABIName)
                       .Case("ilp32", ABI_ILP32)
                       .Case("ilp32f", ABI_ILP32F)
                       .Case("ilp32d", ABI_ILP32D)
                       .Case("ilp32e", ABI_ILP32E)
                       .Case("il32pc64", ABI_IL32PC64)
                       .Case("il32pc64f", ABI_IL32PC64F)
                       .Case("il32pc64d", ABI_IL32PC64D)
                       .Case("il32pc64e", ABI_IL32PC64E)
                       .Case("lp64", ABI_LP64)
                       .Case("lp64f", ABI_LP64F)
                       .Case("lp64d", ABI_LP64D)
                       .Case("lp64e", ABI_LP64E)
                       .Case("l64pc128", ABI_L64PC128)
                       .Case("l64pc128f", ABI_L64PC128F)
                       .Case("l64pc128d", ABI_L64PC128D)
                       .Case("cheriot", ABI_CHERIOT)
                       .Default(ABI_Unknown);
  return TargetABI;
}

// To avoid the BP value clobbered by a function call, we need to choose a
// callee saved register to save the value. RV32E only has X8 and X9 as callee
// saved registers and X8 will be used as fp. So we choose X9 as bp.
MCRegister getBPReg() { return RISCV::X9; }

// Returns the register holding shadow call stack pointer.
MCRegister getSCSPReg() { return RISCV::X3; }

} // namespace RISCVABI

namespace RISCVFeatures {

void validate(const Triple &TT, const FeatureBitset &FeatureBits) {
  if (Error E = RISCV::verifyTensixFeatureBits(TT, FeatureBits))
    reportFatalUsageError(Twine(toString(std::move(E))));
  if (TT.isArch64Bit() && !FeatureBits[RISCV::Feature64Bit])
    reportFatalUsageError("RV64 target requires an RV64 CPU");
  if (!TT.isArch64Bit() && !FeatureBits[RISCV::Feature32Bit])
    reportFatalUsageError("RV32 target requires an RV32 CPU");
  if (FeatureBits[RISCV::Feature32Bit] &&
      FeatureBits[RISCV::Feature64Bit])
    reportFatalUsageError("RV32 and RV64 can't be combined");
}

llvm::Expected<std::unique_ptr<RISCVISAInfo>>
parseFeatureBits(const MCSubtargetInfo &STI) {
  const FeatureBitset &FeatureBits = STI.getFeatureBits();
  unsigned XLen = FeatureBits[RISCV::Feature64Bit] ? 64 : 32;
  std::vector<std::string> FeatureVector;
  // Convert FeatureBitset to FeatureVector.
  for (const auto &Feature : STI.getAllProcessorFeatures()) {
    if (FeatureBits[Feature.Value] &&
        llvm::RISCVISAInfo::isSupportedExtensionFeature(Feature.key()))
      FeatureVector.push_back(std::string("+") + Feature.key());
  }
  return llvm::RISCVISAInfo::parseFeatures(XLen, FeatureVector);
}

} // namespace RISCVFeatures

// Include the auto-generated portion of the compress emitter.
#define GEN_UNCOMPRESS_INSTR
#define GEN_COMPRESS_INSTR
#include "RISCVGenCompressInstEmitter.inc"

bool RISCVRVC::compress(MCInst &OutInst, const MCInst &MI,
                        const MCSubtargetInfo &STI) {
  return compressInst(OutInst, MI, STI);
}

bool RISCVRVC::uncompress(MCInst &OutInst, const MCInst &MI,
                          const MCSubtargetInfo &STI) {
  return uncompressInst(OutInst, MI, STI);
}

// Lookup table for fli.s for entries 2-31.
static constexpr std::pair<uint8_t, uint8_t> LoadFP32ImmArr[] = {
    {0b01101111, 0b00}, {0b01110000, 0b00}, {0b01110111, 0b00},
    {0b01111000, 0b00}, {0b01111011, 0b00}, {0b01111100, 0b00},
    {0b01111101, 0b00}, {0b01111101, 0b01}, {0b01111101, 0b10},
    {0b01111101, 0b11}, {0b01111110, 0b00}, {0b01111110, 0b01},
    {0b01111110, 0b10}, {0b01111110, 0b11}, {0b01111111, 0b00},
    {0b01111111, 0b01}, {0b01111111, 0b10}, {0b01111111, 0b11},
    {0b10000000, 0b00}, {0b10000000, 0b01}, {0b10000000, 0b10},
    {0b10000001, 0b00}, {0b10000010, 0b00}, {0b10000011, 0b00},
    {0b10000110, 0b00}, {0b10000111, 0b00}, {0b10001110, 0b00},
    {0b10001111, 0b00}, {0b11111111, 0b00}, {0b11111111, 0b10},
};

int RISCVLoadFPImm::getLoadFPImm(APFloat FPImm) {
  assert((&FPImm.getSemantics() == &APFloat::IEEEsingle() ||
          &FPImm.getSemantics() == &APFloat::IEEEdouble() ||
          &FPImm.getSemantics() == &APFloat::IEEEhalf()) &&
         "Unexpected semantics");

  // Handle the minimum normalized value which is different for each type.
  if (FPImm.isSmallestNormalized() && !FPImm.isNegative())
    return 1;

  // Convert to single precision to use its lookup table.
  bool LosesInfo;
  APFloat::opStatus Status = FPImm.convert(
      APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &LosesInfo);
  if (Status != APFloat::opOK || LosesInfo)
    return -1;

  APInt Imm = FPImm.bitcastToAPInt();

  if (Imm.extractBitsAsZExtValue(21, 0) != 0)
    return -1;

  bool Sign = Imm.extractBitsAsZExtValue(1, 31);
  uint8_t Mantissa = Imm.extractBitsAsZExtValue(2, 21);
  uint8_t Exp = Imm.extractBitsAsZExtValue(8, 23);

  auto EMI = llvm::lower_bound(LoadFP32ImmArr, std::make_pair(Exp, Mantissa));
  if (EMI == std::end(LoadFP32ImmArr) || EMI->first != Exp ||
      EMI->second != Mantissa)
    return -1;

  // Table doesn't have entry 0 or 1.
  int Entry = std::distance(std::begin(LoadFP32ImmArr), EMI) + 2;

  // The only legal negative value is -1.0(entry 0). 1.0 is entry 16.
  if (Sign) {
    if (Entry == 16)
      return 0;
    return -1;
  }

  return Entry;
}

float RISCVLoadFPImm::getFPImm(unsigned Imm) {
  assert(Imm != 1 && Imm != 30 && Imm != 31 && "Unsupported immediate");

  // Entry 0 is -1.0, the only negative value. Entry 16 is 1.0.
  uint32_t Sign = 0;
  if (Imm == 0) {
    Sign = 0b1;
    Imm = 16;
  }

  uint32_t Exp = LoadFP32ImmArr[Imm - 2].first;
  uint32_t Mantissa = LoadFP32ImmArr[Imm - 2].second;

  uint32_t I = Sign << 31 | Exp << 23 | Mantissa << 21;
  return bit_cast<float>(I);
}

void RISCVZC::printRegList(unsigned RlistEncode, raw_ostream &OS) {
  assert(RlistEncode >= RLISTENCODE::RA &&
         RlistEncode <= RLISTENCODE::RA_S0_S11 && "Invalid Rlist");
  OS << "{ra";
  if (RlistEncode > RISCVZC::RA) {
    OS << ", s0";
    if (RlistEncode == RISCVZC::RA_S0_S11)
      OS << "-s11";
    else if (RlistEncode > RISCVZC::RA_S0 && RlistEncode <= RISCVZC::RA_S0_S11)
      OS << "-s" << (RlistEncode - RISCVZC::RA_S0);
  }
  OS << "}";
}

} // namespace llvm
