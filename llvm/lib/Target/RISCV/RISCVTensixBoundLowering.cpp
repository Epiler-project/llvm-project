//===-- RISCVTensixBoundLowering.cpp --------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixBoundLowering.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
struct BoundInstruction {
  Intrinsic::ID ID;
  unsigned Pseudo;
  unsigned ArgumentCount;
};

// This is the intrinsic-to-ingress mapping, not another encoding table.
// The existing native instruction descriptor owns registers and effects.
constexpr BoundInstruction Instructions[] = {
    {Intrinsic::riscv_tt_bound_sfploadi, RISCV::PseudoTTBoundSFPLOADI, 4},
    {Intrinsic::riscv_tt_bound_sfpload, RISCV::PseudoTTBoundSFPLOAD, 5},
    {Intrinsic::riscv_tt_bound_sfpstore, RISCV::PseudoTTBoundSFPSTORE, 4},
    {Intrinsic::riscv_tt_bound_sfpadd, RISCV::PseudoTTBoundSFPADD, 5},
    {Intrinsic::riscv_tt_bound_sfpmul, RISCV::PseudoTTBoundSFPMUL, 5},
    {Intrinsic::riscv_tt_bound_sfpmad, RISCV::PseudoTTBoundSFPMAD, 6},
    {Intrinsic::riscv_tt_bound_sfpiadd, RISCV::PseudoTTBoundSFPIADD, 4},
    {Intrinsic::riscv_tt_bound_sfpmov, RISCV::PseudoTTBoundSFPMOV, 4},
    {Intrinsic::riscv_tt_bound_sfpmov_all, RISCV::PseudoTTBoundSFPMOVAll, 2},
    {Intrinsic::riscv_tt_bound_sfparecip, RISCV::PseudoTTBoundSFPARECIP, 4},
    {Intrinsic::riscv_tt_bound_sfpexexp, RISCV::PseudoTTBoundSFPEXEXP, 4},
    {Intrinsic::riscv_tt_bound_sfpexman, RISCV::PseudoTTBoundSFPEXMAN, 4},
    {Intrinsic::riscv_tt_bound_sfpabs, RISCV::PseudoTTBoundSFPABS, 4},
    {Intrinsic::riscv_tt_bound_sfplz, RISCV::PseudoTTBoundSFPLZ, 4},
    {Intrinsic::riscv_tt_bound_sfpcast, RISCV::PseudoTTBoundSFPCAST, 4},
    {Intrinsic::riscv_tt_bound_sfpsetexp_i, RISCV::PseudoTTBoundSFPSETEXPI, 5},
    {Intrinsic::riscv_tt_bound_sfpsetman_i, RISCV::PseudoTTBoundSFPSETMANI, 5},
    {Intrinsic::riscv_tt_bound_sfpsetsgn_i, RISCV::PseudoTTBoundSFPSETSGNI, 5},
    {Intrinsic::riscv_tt_bound_sfpsetexp_v, RISCV::PseudoTTBoundSFPSETEXPV, 5},
    {Intrinsic::riscv_tt_bound_sfpsetman_v, RISCV::PseudoTTBoundSFPSETMANV, 5},
    {Intrinsic::riscv_tt_bound_sfpsetsgn_v, RISCV::PseudoTTBoundSFPSETSGNV, 5},
    {Intrinsic::riscv_tt_bound_sfpiadd_i, RISCV::PseudoTTBoundSFPIADDI, 4},
    {Intrinsic::riscv_tt_bound_sfpshft_i, RISCV::PseudoTTBoundSFPSHFTI, 4},
    {Intrinsic::riscv_tt_bound_sfpshft_v, RISCV::PseudoTTBoundSFPSHFTV, 4},
    {Intrinsic::riscv_tt_bound_sfpand, RISCV::PseudoTTBoundSFPAND, 3},
    {Intrinsic::riscv_tt_bound_sfpor, RISCV::PseudoTTBoundSFPOR, 3},
    {Intrinsic::riscv_tt_bound_sfpxor, RISCV::PseudoTTBoundSFPXOR, 3},
    {Intrinsic::riscv_tt_bound_sfpnot, RISCV::PseudoTTBoundSFPNOT, 2},
    {Intrinsic::riscv_tt_bound_sfpshft2, RISCV::PseudoTTBoundSFPSHFT2, 5},
    {Intrinsic::riscv_tt_bound_sfpstochrnd_i, RISCV::PseudoTTBoundSFPSTOCHRNDI,
     6},
    {Intrinsic::riscv_tt_bound_sfpstochrnd_v, RISCV::PseudoTTBoundSFPSTOCHRNDV,
     6},
    {Intrinsic::riscv_tt_bound_sfplut, RISCV::PseudoTTBoundSFPLUT, 7},
    {Intrinsic::riscv_tt_bound_sfpswap, RISCV::PseudoTTBoundSFPSWAP, 5},
    {Intrinsic::riscv_tt_bound_sfptransp, RISCV::PseudoTTBoundSFPTRANSP, 12},
    {Intrinsic::riscv_tt_bound_sfpsetcc, RISCV::PseudoTTBoundSFPSETCC, 3},
    {Intrinsic::riscv_tt_bound_sfpencc, RISCV::PseudoTTBoundSFPENCC, 2},
    {Intrinsic::riscv_tt_bound_sfppushc, RISCV::PseudoTTBoundSFPPUSHC, 2},
    {Intrinsic::riscv_tt_bound_sfppopc, RISCV::PseudoTTBoundSFPPOPC, 2},
    {Intrinsic::riscv_tt_bound_sfpconfig_creg,
     RISCV::PseudoTTBoundSFPCONFIGCReg, 4},
    {Intrinsic::riscv_tt_bound_sfpnop, RISCV::PseudoTTBoundSFPNOP, 0},
    {Intrinsic::riscv_tt_bound_sfpcompc, RISCV::PseudoTTBoundSFPCOMPC, 0},
    {Intrinsic::riscv_tt_bound_sfpconfig_reset,
     RISCV::PseudoTTBoundSFPCONFIGReset, 0},
};

const BoundInstruction *getInstruction(Intrinsic::ID ID) {
  auto It = llvm::find_if(Instructions,
                          [ID](const auto &Info) { return Info.ID == ID; });
  return It == std::end(Instructions) ? nullptr : It;
}

const BoundInstruction *getInstructionForPseudo(unsigned Opcode) {
  if (Opcode == RISCV::PseudoTTBoundSFPLOADDynamic)
    Opcode = RISCV::PseudoTTBoundSFPLOAD;
  if (Opcode == RISCV::PseudoTTBoundSFPSTOREDynamic)
    Opcode = RISCV::PseudoTTBoundSFPSTORE;
  auto It = llvm::find_if(Instructions, [Opcode](const auto &Info) {
    return Info.Pseudo == Opcode;
  });
  return It == std::end(Instructions) ? nullptr : It;
}

Error invalid(const Twine &Message) {
  return createStringError(Twine("bound SFPU ") + Message);
}

/// Null is allowed only for a dynamic Dst offset. Use this same check for
/// LLVM calls and ingress pseudos, so direct MIR cannot hide invalid fields
/// which are not encoded by the selected native instruction.
Error verifyOperands(const BoundInstruction &Info,
                     ArrayRef<std::optional<int64_t>> Args) {
  if (Args.size() != Info.ArgumentCount)
    return invalid("argument tuple has the wrong size");
  auto Offset = getTensixBoundDstOffsetOperand(Info.ID);
  for (auto [I, Value] : llvm::enumerate(Args))
    if (!Value && (!Offset || I != *Offset))
      return invalid(
          "physical operands and logical immediates must be constant");

  auto Range = [&](unsigned I, int64_t Low, int64_t High,
                   StringRef What = "logical field") -> Error {
    if (!Args[I] || *Args[I] < Low || *Args[I] > High)
      return invalid(Twine(What) + " operand " + Twine(I) + " must be in [" +
                     Twine(Low) + ", " + Twine(High) + "]");
    return Error::success();
  };
  auto Mode = [&](unsigned I, std::initializer_list<int64_t> Values) -> Error {
    if (!Args[I] || !llvm::is_contained(Values, *Args[I]))
      return invalid("unsupported mode at operand " + Twine(I));
    return Error::success();
  };
  // Hardware mode sets come from the native descriptor. Only ABI-specific
  // splits and logical-to-native opcode choices below have their own checks.
  auto NativeMode = [&](unsigned I, unsigned Opcode) -> Error {
    const auto *Encoding = RISCV::getTensixEncoding(Opcode);
    assert(Encoding && Encoding->AllowedModes && "native mode contract");
    if (!Args[I] || *Args[I] < 0 || *Args[I] >= 16 ||
        !(Encoding->AllowedModes & (uint16_t(1) << *Args[I])))
      return invalid("unsupported mode at operand " + Twine(I));
    return Error::success();
  };
  auto Tie = [&](unsigned First, unsigned Second) -> Error {
    if (Args[First] != Args[Second])
      return invalid("destructive tie requires identical physical registers");
    return Error::success();
  };
  auto Fixed = [&](unsigned I, int64_t Register) -> Error {
    if (Args[I] != Register)
      return invalid("fixed register group does not match the instruction");
    return Error::success();
  };
  auto Predicated = [&](unsigned RegisterCount) -> Error {
    if (Error E = Range(0, 0, 7, "writable LReg"))
      return E;
    if (Error E = Range(1, 0, 7, "old-destination LReg"))
      return E;
    if (Error E = Tie(0, 1))
      return E;
    for (unsigned I = 2; I != RegisterCount; ++I)
      if (Error E = Range(I, 0, 15, "read register"))
        return E;
    return Error::success();
  };

  switch (Info.Pseudo) {
  case RISCV::PseudoTTBoundSFPLOADI:
    if (Error E = Predicated(2))
      return E;
    if (Error E = Range(2, 0, 65535))
      return E;
    return NativeMode(3, RISCV::TTSFPLOADI);
  case RISCV::PseudoTTBoundSFPLOAD:
  case RISCV::PseudoTTBoundSFPSTORE: {
    bool Load = Info.Pseudo == RISCV::PseudoTTBoundSFPLOAD;
    if (Error E = Load ? Predicated(2) : Range(0, 0, 7, "read LReg"))
      return E;
    unsigned Address = Load ? 2 : 1;
    if (Args[Address])
      if (Error E = Range(Address, 0, 1023, "Dst offset"))
        return E;
    if (Error E = Range(Address + 1, 0, 7))
      return E;
    return NativeMode(Address + 2, Load ? RISCV::TTSFPLOAD : RISCV::TTSFPSTORE);
  }
  case RISCV::PseudoTTBoundSFPADD:
  case RISCV::PseudoTTBoundSFPMUL:
    if (Error E = Predicated(4))
      return E;
    return Range(4, 0, 3);
  case RISCV::PseudoTTBoundSFPMAD:
    if (Error E = Predicated(5))
      return E;
    return Range(5, 0, 3);
  case RISCV::PseudoTTBoundSFPIADD:
    if (Error E = Predicated(3))
      return E;
    return Mode(3, {0, 2, 4, 6, 8, 10});
  case RISCV::PseudoTTBoundSFPMOV:
    if (Error E = Predicated(3))
      return E;
    return Mode(3, {0, 1});
  case RISCV::PseudoTTBoundSFPMOVAll:
    if (Error E = Range(0, 0, 7, "writable LReg"))
      return E;
    return Range(1, 0, 15, "read register");
  case RISCV::PseudoTTBoundSFPARECIP:
    if (Error E = Predicated(3))
      return E;
    return NativeMode(3, RISCV::TTSFPARECIP);
  case RISCV::PseudoTTBoundSFPEXEXP:
    if (Error E = Predicated(3))
      return E;
    return NativeMode(3, RISCV::TTSFPEXEXP);
  case RISCV::PseudoTTBoundSFPEXMAN:
  case RISCV::PseudoTTBoundSFPABS:
    if (Error E = Predicated(3))
      return E;
    return NativeMode(3, Info.Pseudo == RISCV::PseudoTTBoundSFPEXMAN
                             ? RISCV::TTSFPEXMAN
                             : RISCV::TTSFPABS);
  case RISCV::PseudoTTBoundSFPLZ:
    if (Error E = Predicated(3))
      return E;
    return NativeMode(3, RISCV::TTSFPLZ);
  case RISCV::PseudoTTBoundSFPCAST:
    if (Error E = Predicated(3))
      return E;
    return NativeMode(3, RISCV::TTSFPCAST);
  case RISCV::PseudoTTBoundSFPSETEXPI:
  case RISCV::PseudoTTBoundSFPSETMANI:
  case RISCV::PseudoTTBoundSFPSETSGNI:
    if (Error E = Predicated(3))
      return E;
    if (Error E = Range(3, 0,
                        Info.Pseudo == RISCV::PseudoTTBoundSFPSETEXPI   ? 255
                        : Info.Pseudo == RISCV::PseudoTTBoundSFPSETMANI ? 4095
                                                                        : 1))
      return E;
    return Info.Pseudo == RISCV::PseudoTTBoundSFPSETEXPI ? Mode(4, {1, 3})
                                                         : Mode(4, {1});
  case RISCV::PseudoTTBoundSFPSETEXPV:
  case RISCV::PseudoTTBoundSFPSETMANV:
  case RISCV::PseudoTTBoundSFPSETSGNV:
    if (Error E = Predicated(4))
      return E;
    if (Error E = Tie(0, 3))
      return E;
    return Info.Pseudo == RISCV::PseudoTTBoundSFPSETEXPV ? Mode(4, {0, 2})
                                                         : Mode(4, {0});
  case RISCV::PseudoTTBoundSFPIADDI:
  case RISCV::PseudoTTBoundSFPSHFTI:
    if (Error E = Predicated(2))
      return E;
    if (Error E = Range(2, -2048, 2047))
      return E;
    return Info.Pseudo == RISCV::PseudoTTBoundSFPIADDI ? Mode(3, {1, 5, 9})
                                                       : Mode(3, {1, 3, 5, 7});
  case RISCV::PseudoTTBoundSFPSHFTV:
    if (Error E = Predicated(3))
      return E;
    return Mode(3, {0, 2});
  case RISCV::PseudoTTBoundSFPAND:
  case RISCV::PseudoTTBoundSFPOR:
  case RISCV::PseudoTTBoundSFPXOR:
    return Predicated(3);
  case RISCV::PseudoTTBoundSFPNOT:
    return Predicated(2);
  case RISCV::PseudoTTBoundSFPSHFT2:
    if (Error E = Predicated(3))
      return E;
    if (Error E = Range(3, 0, 0))
      return E;
    return NativeMode(4, RISCV::TTSFPSHFT2);
  case RISCV::PseudoTTBoundSFPSTOCHRNDI:
  case RISCV::PseudoTTBoundSFPSTOCHRNDV: {
    bool Immediate = Info.Pseudo == RISCV::PseudoTTBoundSFPSTOCHRNDI;
    if (Error E = Predicated(Immediate ? 3 : 4))
      return E;
    if (Error E = Immediate ? Range(4, 0, 7) : Mode(4, {4, 5}))
      return E;
    if (Error E = Range(5, 0, 2))
      return E;
    if (Immediate)
      return Range(3, 0, *Args[4] == 4 || *Args[4] == 5 ? 31 : 0);
    return Error::success();
  }
  case RISCV::PseudoTTBoundSFPLUT:
    if (Error E = Predicated(2))
      return E;
    for (unsigned I = 0; I != 4; ++I)
      if (Error E = Fixed(I + 2, I))
        return E;
    return NativeMode(6, RISCV::TTSFPLUT);
  case RISCV::PseudoTTBoundSFPSWAP:
    for (unsigned I = 0; I != 4; ++I)
      if (Error E = Range(I, 0, 7, I < 2 ? "writable LReg" : "read LReg"))
        return E;
    if (Error E = Tie(0, 2))
      return E;
    if (Error E = Tie(1, 3))
      return E;
    return NativeMode(4, RISCV::TTSFPSWAP);
  case RISCV::PseudoTTBoundSFPTRANSP:
    for (unsigned I = 0; I != 12; ++I)
      if (Error E = Fixed(I, I < 8 ? I : I - 8))
        return E;
    return Error::success();
  case RISCV::PseudoTTBoundSFPSETCC:
    if (Error E = Range(0, 0, 15, "read register"))
      return E;
    if (Error E = Range(1, 0, 0))
      return E;
    return Mode(2, {0, 2, 4, 6});
  case RISCV::PseudoTTBoundSFPENCC:
    if (Error E = Range(0, 0, 3))
      return E;
    return NativeMode(1, RISCV::TTSFPENCC);
  case RISCV::PseudoTTBoundSFPPUSHC:
  case RISCV::PseudoTTBoundSFPPOPC:
    if (Error E = Range(0, 0, 0))
      return E;
    return Range(1, 0, 0);
  case RISCV::PseudoTTBoundSFPCONFIGCReg:
    if (Error E = Fixed(0, 0))
      return E;
    if (Error E = Range(1, 11, 14, "fixed writable CReg"))
      return E;
    if (Error E = Range(2, 0, 65535))
      return E;
    if (Error E = NativeMode(3, RISCV::TTSFPCONFIGC11))
      return E;
    if ((*Args[3] == 0 && *Args[2] != 0) ||
        (*Args[3] == 8 && (*Args[2] & ~int64_t(0x5555))))
      return invalid("configuration mask does not match its mode");
    return Error::success();
  case RISCV::PseudoTTBoundSFPNOP:
  case RISCV::PseudoTTBoundSFPCOMPC:
  case RISCV::PseudoTTBoundSFPCONFIGReset:
    return Error::success();
  default:
    llvm_unreachable("closed bound intrinsic mapping");
  }
}

MCRegister physicalRegister(int64_t Number) {
  assert(Number >= 0 && Number < 16 && "verified physical register");
  return Number < 8 ? RISCV::SFPRRegClass.getRegister(Number)
                    : RISCV::SFPCRRegClass.getRegister(Number - 8);
}
} // namespace

bool llvm::isTensixBoundSFPUIntrinsic(Intrinsic::ID ID) {
  return getInstruction(ID) != nullptr;
}

std::optional<unsigned> llvm::getTensixBoundDstOffsetOperand(Intrinsic::ID ID) {
  if (ID == Intrinsic::riscv_tt_bound_sfpload)
    return 2;
  if (ID == Intrinsic::riscv_tt_bound_sfpstore)
    return 1;
  return std::nullopt;
}

Error llvm::verifyTensixBoundSFPUIntrinsic(const IntrinsicInst &II) {
  const auto *Info = getInstruction(II.getIntrinsicID());
  if (!Info)
    return invalid("unknown intrinsic");
  if (!II.getType()->isVoidTy())
    return invalid("intrinsic must not return an allocator-owned value");
  SmallVector<std::optional<int64_t>, 12> Args;
  for (const Value *Arg : II.args()) {
    if (!Arg->getType()->isIntegerTy(32))
      return invalid("all argument fields must have type i32");
    const auto *Constant = dyn_cast<ConstantInt>(Arg);
    Args.push_back(Constant ? std::optional<int64_t>(Constant->getSExtValue())
                            : std::nullopt);
  }
  return verifyOperands(*Info, Args);
}

SDValue llvm::lowerTensixBoundSFPUIntrinsic(SDValue Op, SelectionDAG &DAG,
                                            const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  const auto *Info = getInstruction(ID);
  if (!Info)
    return SDValue();
  auto Fail = [&](const Twine &Message) {
    auto &MF = DAG.getMachineFunction();
    MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
    MF.getFunction().getContext().diagnose(
        DiagnosticInfoUnsupported(MF.getFunction(), Message));
    // This void-only ingress has no numerical result to fabricate. Returning
    // an empty value would allow another lowering path to handle the failure.
    return Op.getOperand(0);
  };
  if (!ST.hasVendorXTTTensixBH())
    return Fail("bound SFPU intrinsic requires +xtttensixbh");
  Attribute Executor =
      DAG.getMachineFunction().getFunction().getFnAttribute("tensix-executor");
  if (!Executor.isStringAttribute() ||
      (Executor.getValueAsString() != "trisc0" &&
       Executor.getValueAsString() != "trisc1" &&
       Executor.getValueAsString() != "trisc2"))
    return Fail("bound SFPU execution requires a TRISC executor");

  SDLoc DL(Op);
  auto Offset = getTensixBoundDstOffsetOperand(ID);
  bool Dynamic = Offset && !isa<ConstantSDNode>(Op.getOperand(*Offset + 2));
  unsigned Opcode = Info->Pseudo;
  if (Dynamic)
    Opcode = ID == Intrinsic::riscv_tt_bound_sfpload
                 ? RISCV::PseudoTTBoundSFPLOADDynamic
                 : RISCV::PseudoTTBoundSFPSTOREDynamic;
  SmallVector<SDValue, 13> Operands;
  for (unsigned I = 0; I != Info->ArgumentCount; ++I) {
    SDValue Value = Op.getOperand(I + 2);
    if (Dynamic && I == *Offset) {
      Operands.push_back(Value);
      continue;
    }
    const auto *Constant = dyn_cast<ConstantSDNode>(Value);
    if (!Constant)
      return Fail("bound SFPU operand lost its constant binding");
    Operands.push_back(
        DAG.getTargetConstant(Constant->getAPIntValue(), DL, MVT::i32));
  }
  Operands.push_back(Op.getOperand(0));
  SmallVector<EVT, 3> Results(Dynamic ? 2 : 0, MVT::i32);
  Results.push_back(MVT::Other);
  MachineSDNode *Node =
      DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results), Operands);
  return SDValue(Node, Dynamic ? 2 : 0);
}

MachineBasicBlock *
llvm::emitTensixBoundSFPUInstruction(MachineInstr &MI, MachineBasicBlock *MBB,
                                     const RISCVSubtarget &ST) {
  const auto *Info = getInstructionForPseudo(MI.getOpcode());
  if (!Info)
    return nullptr;
  auto &MF = *MBB->getParent();
  auto Fail = [&](const Twine &Message) {
    MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
    MF.getFunction().getContext().diagnose(
        DiagnosticInfoUnsupported(MF.getFunction(), Message));
    // Diagnostic handlers may return and generic machine passes still run.
    // Retain definitions for existing scalar scratch users in this poisoned
    // function; never create a numerical SFPU value or replacement issue.
    // AsmPrinter suppresses the entire failed function.
    auto &MRI = MF.getRegInfo();
    for (const MachineOperand &Def : MI.defs()) {
      Register Reg = Def.getReg();
      if (!Reg.isVirtual())
        continue;
      const auto *RC = MRI.getRegClassOrNull(Reg);
      if (RC && RISCV::GPRRegClass.hasSubClassEq(RC))
        BuildMI(*MBB, MI, MI.getDebugLoc(),
                ST.getInstrInfo()->get(TargetOpcode::IMPLICIT_DEF), Reg);
    }
    MI.eraseFromParent();
    return MBB;
  };
  if (!MF.getInfo<RISCVMachineFunctionInfo>()->usesBoundTensixSFPU())
    return Fail("bound SFPU ingress requires bound machine mode");
  if (Error E = verifyTensixMachineEffects(MI))
    return Fail(toString(std::move(E)));
  if (Error E = verifyTensixMachineOperands(MI, MF.getRegInfo(),
                                            *ST.getRegisterInfo()))
    return Fail(toString(std::move(E)));
  bool Dynamic = MI.getOpcode() == RISCV::PseudoTTBoundSFPLOADDynamic ||
                 MI.getOpcode() == RISCV::PseudoTTBoundSFPSTOREDynamic;
  unsigned FirstArgument = Dynamic ? 2 : 0;
  auto Offset = getTensixBoundDstOffsetOperand(Info->ID);
  if (MI.getNumExplicitOperands() != Info->ArgumentCount + FirstArgument)
    return Fail("bound SFPU ingress pseudo has an invalid tuple");
  SmallVector<std::optional<int64_t>, 12> Args;
  for (unsigned I = 0; I != Info->ArgumentCount; ++I) {
    const MachineOperand &MO = MI.getOperand(FirstArgument + I);
    if (Dynamic && I == *Offset) {
      if (!MO.isReg())
        return Fail("bound SFPU dynamic Dst offset must be a GPR");
      Args.push_back(std::nullopt);
    } else {
      if (!MO.isImm())
        return Fail("bound SFPU ingress lost a physical immediate");
      Args.push_back(MO.getImm());
    }
  }
  if (Error E = verifyOperands(*Info, Args))
    return Fail(toString(std::move(E)));

  auto Imm = [&](unsigned I) { return *Args[I]; };
  auto Reg = [&](unsigned I) { return physicalRegister(Imm(I)); };
  auto Emit = [&](unsigned Opcode) {
    auto Builder =
        BuildMI(*MBB, MI, MI.getDebugLoc(), ST.getInstrInfo()->get(Opcode));
    Builder.cloneMemRefs(MI);
    return Builder;
  };
  auto Unary = [&](unsigned Opcode) {
    return Emit(Opcode)
        .addReg(Reg(0), RegState::Define)
        .addReg(Reg(1))
        .addReg(Reg(2));
  };

  switch (Info->Pseudo) {
  case RISCV::PseudoTTBoundSFPLOADI:
    Emit(RISCV::TTSFPLOADI)
        .addReg(Reg(0), RegState::Define)
        .addReg(Reg(1))
        .addImm(Imm(2))
        .addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPLOAD:
  case RISCV::PseudoTTBoundSFPSTORE: {
    bool Load = Info->Pseudo == RISCV::PseudoTTBoundSFPLOAD;
    auto Builder =
        Emit(Dynamic ? (Load ? RISCV::PseudoTTSFPLOAD : RISCV::PseudoTTSFPSTORE)
                     : (Load ? RISCV::TTSFPLOAD : RISCV::TTSFPSTORE));
    if (Load)
      Builder.addReg(Reg(0), RegState::Define);
    if (Dynamic) {
      for (unsigned I = 0; I != 2; ++I)
        Builder.addReg(MI.getOperand(I).getReg(),
                       RegState::Define | RegState::EarlyClobber);
    }
    Builder.addReg(Reg(Load ? 1 : 0));
    if (Dynamic)
      Builder.addReg(MI.getOperand(FirstArgument + *Offset).getReg());
    else
      Builder.addImm(Imm(*Offset));
    Builder.addImm(Imm(*Offset + 1)).addImm(Imm(*Offset + 2));
    break;
  }
  case RISCV::PseudoTTBoundSFPADD:
  case RISCV::PseudoTTBoundSFPMUL:
    Unary(Info->Pseudo == RISCV::PseudoTTBoundSFPADD ? RISCV::TTSFPADD
                                                     : RISCV::TTSFPMUL)
        .addReg(Reg(3))
        .addImm(Imm(4));
    break;
  case RISCV::PseudoTTBoundSFPMAD:
    Unary(RISCV::TTSFPMAD).addReg(Reg(3)).addReg(Reg(4)).addImm(Imm(5));
    break;
  case RISCV::PseudoTTBoundSFPIADD: {
    constexpr unsigned Opcodes[] = {RISCV::TTSFPIADDCCLT, RISCV::TTSFPISUBCCLT,
                                    RISCV::TTSFPIADD,     RISCV::TTSFPISUB,
                                    RISCV::TTSFPIADDCCGE, RISCV::TTSFPISUBCCGE};
    Unary(Opcodes[Imm(3) / 2]);
    break;
  }
  case RISCV::PseudoTTBoundSFPMOV:
    Unary(Imm(3) == 0 ? RISCV::TTSFPMOV : RISCV::TTSFPMOVNeg);
    break;
  case RISCV::PseudoTTBoundSFPMOVAll:
    Emit(RISCV::TTSFPMOVAll).addReg(Reg(0), RegState::Define).addReg(Reg(1));
    break;
  case RISCV::PseudoTTBoundSFPARECIP:
    Unary(RISCV::TTSFPARECIP).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPEXEXP:
    Unary(RISCV::TTSFPEXEXP).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPEXMAN:
    Unary(RISCV::TTSFPEXMAN).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPABS:
    Unary(RISCV::TTSFPABS).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPLZ:
    Unary(RISCV::TTSFPLZ).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPCAST:
    Unary(RISCV::TTSFPCAST).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPSETEXPI:
  case RISCV::PseudoTTBoundSFPSETMANI:
  case RISCV::PseudoTTBoundSFPSETSGNI:
  case RISCV::PseudoTTBoundSFPSETEXPV:
  case RISCV::PseudoTTBoundSFPSETMANV:
  case RISCV::PseudoTTBoundSFPSETSGNV: {
    unsigned Opcode;
    bool Vector;
    switch (Info->Pseudo) {
    case RISCV::PseudoTTBoundSFPSETEXPI:
    case RISCV::PseudoTTBoundSFPSETEXPV:
      Opcode = RISCV::TTSFPSETEXP;
      Vector = Info->Pseudo == RISCV::PseudoTTBoundSFPSETEXPV;
      break;
    case RISCV::PseudoTTBoundSFPSETMANI:
    case RISCV::PseudoTTBoundSFPSETMANV:
      Opcode = RISCV::TTSFPSETMAN;
      Vector = Info->Pseudo == RISCV::PseudoTTBoundSFPSETMANV;
      break;
    default:
      Opcode = RISCV::TTSFPSETSGN;
      Vector = Info->Pseudo == RISCV::PseudoTTBoundSFPSETSGNV;
      break;
    }
    Unary(Opcode).addImm(Vector ? 0 : Imm(3)).addImm(Imm(4));
    break;
  }
  case RISCV::PseudoTTBoundSFPIADDI:
  case RISCV::PseudoTTBoundSFPSHFTI:
    Emit(Info->Pseudo == RISCV::PseudoTTBoundSFPIADDI ? RISCV::TTSFPIADDI
                                                      : RISCV::TTSFPSHFT)
        .addReg(Reg(0), RegState::Define)
        .addReg(Reg(1))
        .addReg(Reg(1))
        .addImm(Imm(2))
        .addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPSHFTV:
    Unary(RISCV::TTSFPSHFT).addImm(0).addImm(Imm(3));
    break;
  case RISCV::PseudoTTBoundSFPAND:
    Unary(RISCV::TTSFPAND);
    break;
  case RISCV::PseudoTTBoundSFPOR:
    Unary(RISCV::TTSFPOR);
    break;
  case RISCV::PseudoTTBoundSFPXOR:
    Unary(RISCV::TTSFPXOR);
    break;
  case RISCV::PseudoTTBoundSFPNOT:
    Emit(RISCV::TTSFPNOT)
        .addReg(Reg(0), RegState::Define)
        .addReg(Reg(1))
        .addReg(Reg(1));
    break;
  case RISCV::PseudoTTBoundSFPSHFT2:
    Unary(RISCV::TTSFPSHFT2).addImm(Imm(4));
    break;
  case RISCV::PseudoTTBoundSFPSTOCHRNDI: {
    int64_t Mode = Imm(4);
    if (Mode == 4 || Mode == 5)
      Mode |= 8;
    Unary(RISCV::TTSFPSTOCHRNDI).addImm(Imm(3)).addImm(Mode).addImm(Imm(5));
    break;
  }
  case RISCV::PseudoTTBoundSFPSTOCHRNDV:
    Unary(RISCV::TTSFPSTOCHRNDV).addReg(Reg(3)).addImm(Imm(4)).addImm(Imm(5));
    break;
  case RISCV::PseudoTTBoundSFPLUT:
    Emit(RISCV::TTSFPLUT)
        .addReg(Reg(0), RegState::Define)
        .addReg(Reg(1))
        .addImm(Imm(6));
    break;
  case RISCV::PseudoTTBoundSFPSWAP:
    Emit(RISCV::TTSFPSWAP)
        .addReg(Reg(0), RegState::Define)
        .addReg(Reg(1), RegState::Define)
        .addReg(Reg(2))
        .addReg(Reg(3))
        .addImm(Imm(4));
    break;
  case RISCV::PseudoTTBoundSFPTRANSP:
    Emit(RISCV::TTSFPTRANSP);
    break;
  case RISCV::PseudoTTBoundSFPSETCC: {
    constexpr unsigned Opcodes[] = {RISCV::TTSFPSETCCLT, RISCV::TTSFPSETCCNE,
                                    RISCV::TTSFPSETCCGE, RISCV::TTSFPSETCCEQ};
    Emit(Opcodes[Imm(2) / 2]).addReg(Reg(0));
    break;
  }
  case RISCV::PseudoTTBoundSFPENCC:
    Emit(RISCV::TTSFPENCC).addImm(Imm(0)).addImm(Imm(1));
    break;
  case RISCV::PseudoTTBoundSFPPUSHC:
    Emit(RISCV::TTSFPPUSHC);
    break;
  case RISCV::PseudoTTBoundSFPPOPC:
    Emit(RISCV::TTSFPPOPC);
    break;
  case RISCV::PseudoTTBoundSFPCONFIGCReg: {
    constexpr unsigned Opcodes[] = {
        RISCV::TTSFPCONFIGC11, RISCV::TTSFPCONFIGC12, RISCV::TTSFPCONFIGC13,
        RISCV::TTSFPCONFIGC14};
    Emit(Opcodes[Imm(1) - 11]).addImm(Imm(2)).addImm(Imm(3));
    break;
  }
  case RISCV::PseudoTTBoundSFPNOP:
    Emit(RISCV::TTSFPNOP);
    break;
  case RISCV::PseudoTTBoundSFPCOMPC:
    Emit(RISCV::TTSFPCOMPC);
    break;
  case RISCV::PseudoTTBoundSFPCONFIGReset:
    Emit(RISCV::TTSFPCONFIGReset);
    break;
  default:
    llvm_unreachable("closed bound ingress pseudo");
  }
  MI.eraseFromParent();
  return MBB;
}
