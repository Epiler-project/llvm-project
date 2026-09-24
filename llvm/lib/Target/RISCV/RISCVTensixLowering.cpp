//===-- RISCVTensixLowering.cpp --------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCVTensixLowering.h"
#include "RISCVTensixIRVerification.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVMachineFunctionInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

SDValue llvm::lowerTensixOrdinaryIntrinsic(SDValue Op, SelectionDAG &DAG,
                                          const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  bool Clear = ID == Intrinsic::riscv_tt_mop_clear;
  bool End = ID == Intrinsic::riscv_tt_replay_record_end;
  const auto *Info = RISCV::getTensixInstructionByIntrinsic(ID);
  if (!Info && !Clear && !End)
    return SDValue();
  if (!ST.hasVendorXTTTensixBH())
    reportFatalUsageError("Tensix intrinsic requires +xtttensixbh");
  SDLoc DL(Op);
  if (End)
    return SDValue(DAG.getMachineNode(RISCV::PseudoTTReplayRecordEnd, DL,
                                     MVT::Other, Op.getOperand(0)), 0);
  bool Port = Info && ID == Info->PortIntrinsicID;
  bool Mop = Clear || (Info && ID == Info->MopIntrinsicID);
  const auto *Machine = Info ? RISCV::getTensixMachineInfoByIntrinsic(
                                  Info->IntrinsicID) : nullptr;
  SmallVector<SDValue, 16> Operands;
  unsigned First = Port || Mop ? 3 : 2;
  if (Port || Mop)
    Operands.push_back(DAG.getTargetConstant(Op.getConstantOperandVal(2), DL,
                                            MVT::i32));
  if (Info)
    for (unsigned I = 0; I != Info->NumFields; ++I) {
      SDValue Value = Op.getOperand(First + I);
      bool Dynamic = Port && Machine->Opcode != RISCV::TTREPLAY;
      Operands.push_back(Dynamic ? Value : DAG.getTargetConstant(
          cast<ConstantSDNode>(Value)->getZExtValue(), DL, MVT::i32));
    }
  Operands.push_back(Op.getOperand(0));
  unsigned Scratch = Port ? 3 : Mop ? 2 : 0;
  SmallVector<EVT> Results(Scratch, MVT::i32);
  Results.push_back(MVT::Other);
  unsigned Opcode = Clear ? unsigned(RISCV::PseudoTTMOPClear)
                          : Port ? Machine->PortOpcode
                                 : Mop ? Machine->MopOpcode : Machine->Opcode;
  MachineSDNode *Node = DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results),
                                         Operands);
  auto &MF = DAG.getMachineFunction();
  auto MemoryFlags = MachineMemOperand::MOStore |
                     MachineMemOperand::MOVolatile;
  if (!Mop)
    MemoryFlags |= MachineMemOperand::MOLoad;
  MachineMemOperand *MMO = MF.getMachineMemOperand(
      MachinePointerInfo(), MemoryFlags, 4, Align(4));
  DAG.setNodeMemRefs(Node, {MMO});
  return SDValue(Node, Scratch);
}

SDValue llvm::lowerTensixSFPUIntrinsic(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isTensixSFPUIntrinsic(ID))
    return SDValue();
  if (!ST.hasVendorXTTTensixBH())
    reportFatalUsageError("Tensix SFPU intrinsic requires +xtttensixbh");
  const Function &F = DAG.getMachineFunction().getFunction();
  Attribute Executor = F.getFnAttribute("tensix-executor");
  if (!Executor.isStringAttribute() || Executor.getValueAsString() != "trisc1")
    reportFatalUsageError("Tensix SFPU execution requires tensix-executor=trisc1");
  SDLoc DL(Op);
  SDValue InputChain = Op.getOperand(0);
  auto Arg = [&](unsigned N) { return Op.getOperand(N + 2); };
  auto Imm = [&](unsigned N) {
    return DAG.getTargetConstant(Op.getConstantOperandVal(N + 2), DL, MVT::i32);
  };
  unsigned Opcode = 0;
  SmallVector<SDValue> Operands;
  unsigned NumResults = Op.getOpcode() == ISD::INTRINSIC_W_CHAIN
                            ? Op->getNumValues() - 1 : 0;
  unsigned ScratchResults = 0;
  switch (ID) {
  case Intrinsic::riscv_tt_lreg_read:
  case Intrinsic::riscv_tt_creg_read: {
    unsigned Index = Op.getConstantOperandVal(2);
    MCRegister Reg = ID == Intrinsic::riscv_tt_lreg_read
                         ? RISCV::SFPRRegClass.getRegister(Index)
                         : RISCV::SFPCRRegClass.getRegister(Index - 8);
    Opcode = RISCV::TTSFPMOVAll;
    Operands.push_back(DAG.getRegister(Reg, MVT::v32i32));
    break;
  }
  case Intrinsic::riscv_tt_lreg_write:
    return DAG.getCopyToReg(Op.getOperand(0), DL,
                           RISCV::SFPRRegClass.getRegister(
                               Op.getConstantOperandVal(2)), Arg(1));
  case Intrinsic::riscv_tt_sfploadi:
    Opcode = RISCV::TTSFPLOADI;
    Operands = {Arg(0), Imm(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpmov:
    switch (Op.getConstantOperandVal(4)) {
    case 0: Opcode = RISCV::TTSFPMOV; break;
    case 1: Opcode = RISCV::TTSFPMOVNeg; break;
    case 2: Opcode = RISCV::TTSFPMOVAll; break;
    default: llvm_unreachable("verified SFPMOV mode");
    }
    if (Opcode != RISCV::TTSFPMOVAll)
      Operands.push_back(Arg(0));
    Operands.push_back(Arg(1));
    break;
  case Intrinsic::riscv_tt_sfpadd:
    Opcode = RISCV::TTSFPADD;
    Operands = {Arg(0), Arg(1), Arg(2), Imm(3)};
    break;
  case Intrinsic::riscv_tt_sfpmad:
    Opcode = RISCV::TTSFPMAD;
    Operands = {Arg(0), Arg(1), Arg(2), Arg(3), Imm(4)};
    break;
  case Intrinsic::riscv_tt_sfpmul:
    Opcode = RISCV::TTSFPMUL;
    Operands = {Arg(0), Arg(1), Arg(2), Imm(3)};
    break;
  case Intrinsic::riscv_tt_sfpiadd: {
    static constexpr unsigned Opcodes[] = {
        RISCV::TTSFPIADDCCLT, RISCV::TTSFPISUBCCLT, RISCV::TTSFPIADD,
        RISCV::TTSFPISUB, RISCV::TTSFPIADDCCGE, RISCV::TTSFPISUBCCGE};
    Opcode = Opcodes[Op.getConstantOperandVal(4) / 2];
    Operands = {Arg(0), Arg(1)};
    break;
  }
  case Intrinsic::riscv_tt_sfparecip:
    Opcode = RISCV::TTSFPARECIP;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpexexp:
    Opcode = RISCV::TTSFPEXEXP;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpexman:
    Opcode = RISCV::TTSFPEXMAN;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpabs:
    Opcode = RISCV::TTSFPABS;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfplz:
    Opcode = RISCV::TTSFPLZ;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpcast:
    Opcode = RISCV::TTSFPCAST;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpsetexp_i:
  case Intrinsic::riscv_tt_sfpsetman_i:
  case Intrinsic::riscv_tt_sfpsetsgn_i:
    Opcode = ID == Intrinsic::riscv_tt_sfpsetexp_i ? RISCV::TTSFPSETEXP
           : ID == Intrinsic::riscv_tt_sfpsetman_i ? RISCV::TTSFPSETMAN
                                                  : RISCV::TTSFPSETSGN;
    Operands = {Arg(0), Arg(1), Imm(2), Imm(3)};
    break;
  case Intrinsic::riscv_tt_sfpsetexp_v:
  case Intrinsic::riscv_tt_sfpsetman_v:
  case Intrinsic::riscv_tt_sfpsetsgn_v: {
    Opcode = ID == Intrinsic::riscv_tt_sfpsetexp_v ? RISCV::TTSFPSETEXP
           : ID == Intrinsic::riscv_tt_sfpsetman_v ? RISCV::TTSFPSETMAN
                                                  : RISCV::TTSFPSETSGN;
    // The field instruction reads the active field from its destination.
    // Predicated staging keeps the independent old destination in disabled
    // lanes. An all-lane copy would incorrectly replace that passthrough.
    auto *Stage = DAG.getMachineNode(RISCV::TTSFPMOV, DL,
        DAG.getVTList(MVT::v32i32, MVT::Other), {Arg(0), Arg(2), InputChain});
    InputChain = SDValue(Stage, 1);
    Operands = {SDValue(Stage, 0), Arg(1),
                DAG.getTargetConstant(0, DL, MVT::i32), Imm(3)};
    break;
  }
  case Intrinsic::riscv_tt_sfpand:
  case Intrinsic::riscv_tt_sfpor:
  case Intrinsic::riscv_tt_sfpxor:
    Opcode = ID == Intrinsic::riscv_tt_sfpand ? RISCV::TTSFPAND
           : ID == Intrinsic::riscv_tt_sfpor ? RISCV::TTSFPOR : RISCV::TTSFPXOR;
    Operands = {Arg(0), Arg(1)};
    break;
  case Intrinsic::riscv_tt_sfpnot:
    Opcode = RISCV::TTSFPNOT;
    Operands = {Arg(0), Arg(0)};
    break;
  case Intrinsic::riscv_tt_sfpiadd_i:
    Opcode = RISCV::TTSFPIADDI;
    Operands = {Arg(0), Arg(0), Imm(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpshft_i:
    Opcode = RISCV::TTSFPSHFT;
    Operands = {Arg(0), Arg(0), Imm(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpshft_v:
    Opcode = RISCV::TTSFPSHFT;
    Operands = {Arg(0), Arg(1), DAG.getTargetConstant(0, DL, MVT::i32), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfpshft2:
    Opcode = RISCV::TTSFPSHFT2;
    Operands = {Arg(0), Arg(1), Imm(3)};
    break;
  case Intrinsic::riscv_tt_sfpstochrnd_i: {
    Opcode = RISCV::TTSFPSTOCHRNDI;
    unsigned Mode = Op.getConstantOperandVal(5);
    if (Mode == 4 || Mode == 5)
      Mode |= 8;
    Operands = {Arg(0), Arg(1), Imm(2),
                DAG.getTargetConstant(Mode, DL, MVT::i32), Imm(4)};
    break;
  }
  case Intrinsic::riscv_tt_sfpstochrnd_v:
    Opcode = RISCV::TTSFPSTOCHRNDV;
    Operands = {Arg(0), Arg(1), Arg(2), Imm(3), Imm(4)};
    break;
  case Intrinsic::riscv_tt_sfpswap:
    Opcode = RISCV::TTSFPSWAP;
    Operands = {Arg(0), Arg(1), Imm(2)};
    break;
  case Intrinsic::riscv_tt_sfplut:
    Opcode = RISCV::TTSFPLUT;
    Operands = {Arg(0), Imm(5)};
    break;
  case Intrinsic::riscv_tt_sfptransp:
    Opcode = RISCV::TTSFPTRANSP;
    break;
  case Intrinsic::riscv_tt_sfpload:
  case Intrinsic::riscv_tt_sfpstore: {
    bool Load = ID == Intrinsic::riscv_tt_sfpload;
    if (auto *Offset = dyn_cast<ConstantSDNode>(Arg(1))) {
      Opcode = Load ? RISCV::TTSFPLOAD : RISCV::TTSFPSTORE;
      Operands = {Arg(0), DAG.getTargetConstant(Offset->getZExtValue(), DL,
                                               MVT::i32), Imm(2), Imm(3)};
    } else {
      Opcode = Load ? RISCV::PseudoTTSFPLOAD : RISCV::PseudoTTSFPSTORE;
      Operands = {Arg(0), Arg(1), Imm(2), Imm(3)};
      ScratchResults = 2;
    }
    break;
  }
  case Intrinsic::riscv_tt_sfpencc:
    Opcode = RISCV::TTSFPENCC;
    Operands = {Imm(0), Imm(1)};
    break;
  case Intrinsic::riscv_tt_sfpsetcc: {
    static constexpr unsigned Opcodes[] = {RISCV::TTSFPSETCCLT,
        RISCV::TTSFPSETCCNE, RISCV::TTSFPSETCCGE, RISCV::TTSFPSETCCEQ};
    Opcode = Opcodes[Op.getConstantOperandVal(4) / 2];
    Operands = {Arg(0)};
    break;
  }
  case Intrinsic::riscv_tt_sfppushc: Opcode = RISCV::TTSFPPUSHC; break;
  case Intrinsic::riscv_tt_sfppopc: Opcode = RISCV::TTSFPPOPC; break;
  case Intrinsic::riscv_tt_sfpnop: Opcode = RISCV::TTSFPNOP; break;
  case Intrinsic::riscv_tt_sfpcompc: Opcode = RISCV::TTSFPCOMPC; break;
  case Intrinsic::riscv_tt_sfpconfig_creg:
    // Operand 1 is the intrinsic ID.  The destination CReg is the second
    // intrinsic argument (SelectionDAG operand 3); the carrier in Arg(0) is
    // staged through TT_L0 below and is not an explicit machine operand.
    Opcode = RISCV::TTSFPCONFIGC11 +
             (Op.getConstantOperandVal(3) - 11);
    Operands = {Imm(2), Imm(3)};
    break;
  case Intrinsic::riscv_tt_sfpconfig_reset:
    Opcode = RISCV::TTSFPCONFIGReset;
    break;
  default: llvm_unreachable("admitted SFPU intrinsic");
  }
  // Fixed-group instructions have explicit whole-register footprints. Save
  // named fixed bindings, stage SSA operands, capture results, and restore the
  // bindings. Ordinary live SSA values are protected by physical clobbers and
  // the LLVM allocator; this code never chooses their register numbers.
  bool Transpose = ID == Intrinsic::riscv_tt_sfptransp;
  bool LUT = ID == Intrinsic::riscv_tt_sfplut;
  bool Config = ID == Intrinsic::riscv_tt_sfpconfig_creg;
  unsigned StageCount = Transpose || LUT ? 4 : Config ? 1 : 0;
  unsigned ClobberMask = Transpose ? 0xff : LUT ? 0x0f : Config ? 1 : 0;
  unsigned PreserveMask = ClobberMask &
      DAG.getMachineFunction().getInfo<RISCVMachineFunctionInfo>()
          ->getTensixFixedLRegs();
  SmallVector<std::pair<MCRegister, SDValue>> Saved;
  for (unsigned I = 0; I != 8; ++I) {
    if (!(PreserveMask & (1u << I)))
      continue;
    MCRegister Reg = RISCV::SFPRRegClass.getRegister(I);
    SDValue Value = DAG.getCopyFromReg(InputChain, DL, Reg, MVT::v32i32);
    InputChain = Value.getValue(1);
    Saved.emplace_back(Reg, Value);
  }
  for (unsigned I = 0; I != StageCount; ++I)
    InputChain = DAG.getCopyToReg(InputChain, DL,
        RISCV::SFPRRegClass.getRegister(I), Arg(I + unsigned(LUT)));
  Operands.push_back(InputChain);
  SmallVector<EVT> Results;
  if (!Transpose)
    Results.append(NumResults, MVT::v32i32);
  Results.append(ScratchResults, MVT::i32);
  unsigned ChainIndex = Results.size();
  Results.push_back(MVT::Other);
  // An implicit physical result is live only when InstrEmitter can see its
  // CopyFromReg on the glue chain. A memory chain alone orders these nodes
  // but does not associate the snapshots with the instruction's fixed defs.
  if (Transpose)
    Results.push_back(MVT::Glue);
  MachineSDNode *Node = DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results),
                                         Operands);
  SDValue Chain(Node, ChainIndex);
  SDValue Glue = Transpose ? SDValue(Node, ChainIndex + 1) : SDValue();
  SmallVector<SDValue> Values;
  for (unsigned I = 0; I != NumResults; ++I) {
    if (Transpose) {
      SDValue Value = DAG.getCopyFromReg(Chain, DL,
          RISCV::SFPRRegClass.getRegister(I), MVT::v32i32, Glue);
      Chain = Value.getValue(1);
      Glue = Value.getValue(2);
      Values.push_back(Value);
    } else {
      Values.push_back(SDValue(Node, I));
    }
  }
  for (auto [Reg, Value] : Saved)
    Chain = DAG.getCopyToReg(Chain, DL, Reg, Value);
  if (!NumResults)
    return Chain;
  Values.push_back(Chain);
  return DAG.getMergeValues(Values, DL);
}
