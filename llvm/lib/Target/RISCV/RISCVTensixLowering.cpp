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

namespace {
void attachIssueMemory(MachineSDNode *Node, SelectionDAG &DAG,
                       MachineMemOperand::Flags Access) {
  MachineMemOperand *MMO = DAG.getMachineFunction().getMachineMemOperand(
      MachinePointerInfo(), Access | MachineMemOperand::MOVolatile, 4,
      Align(4));
  DAG.setNodeMemRefs(Node, {MMO});
}
} // namespace

SDValue llvm::lowerTensixOrdinaryIntrinsic(SDValue Op, SelectionDAG &DAG,
                                          const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  bool Clear = ID == Intrinsic::riscv_tt_mop_clear;
  bool Control = ID == Intrinsic::riscv_tt_mop_control_write;
  bool End = ID == Intrinsic::riscv_tt_replay_record_end;
  unsigned TemplateOpcode = 0;
  switch (ID) {
  case Intrinsic::riscv_tt_replay_template_begin:
    TemplateOpcode = RISCV::PseudoTTReplayTemplateBegin;
    break;
  case Intrinsic::riscv_tt_replay_template_end:
    TemplateOpcode = RISCV::PseudoTTReplayTemplateEnd;
    break;
  case Intrinsic::riscv_tt_replay_template_execute:
    TemplateOpcode = RISCV::PseudoTTReplayTemplateExecute;
    break;
  case Intrinsic::riscv_tt_replay_template_mop:
    TemplateOpcode = RISCV::PseudoTTReplayTemplateMop;
    break;
  default:
    break;
  }
  const auto *Info = RISCV::getTensixInstructionByIntrinsic(ID);
  bool Word = ID == Intrinsic::riscv_tt_issue_word;
  if (!Info && !Clear && !Control && !End && !TemplateOpcode && !Word)
    return SDValue();
  if (!ST.hasVendorXTTTensixBH())
    reportFatalUsageError("Tensix intrinsic requires +xtttensixbh");
  SDLoc DL(Op);
  if (Word) {
    // The issue-word fold encoded every field; only the port store remains.
    const auto *Machine = RISCV::getTensixMachineInfoByRawOpcode(
        Op.getConstantOperandVal(3));
    unsigned Port = Op.getConstantOperandVal(2);
    if (!Machine || RISCV::isTensixFieldPortOpcode(Machine->Opcode) ||
        Port > unsigned(RISCV::TensixInstructionPort::BriscToTrisc2))
      reportFatalUsageError("invalid compiler-private Tensix issue word");
    SDValue Address = DAG.getConstant(
        RISCV::getTensixInstructionPortAddress(
            static_cast<RISCV::TensixInstructionPort>(Port)),
        DL, MVT::i32);
    MachineSDNode *Node = DAG.getMachineNode(
        Machine->PortOpcode, DL, MVT::Other,
        {DAG.getTargetConstant(Port, DL, MVT::i32), Op.getOperand(4), Address,
         Op.getOperand(0)});
    attachIssueMemory(Node, DAG,
                      MachineMemOperand::MOLoad | MachineMemOperand::MOStore);
    return SDValue(Node, 0);
  }
  if (Control) {
    auto *Constant = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    // Machine immediates use a signed 64-bit container. Keep the i32 payload
    // zero-extended so bit 31 does not turn a legal uimm32 into a negative MI.
    SDValue Value = Constant
                        ? DAG.getTargetConstant(Constant->getZExtValue(), DL,
                                                MVT::i64)
                        : Op.getOperand(3);
    unsigned Scratch = Constant ? 2 : 1;
    SmallVector<EVT, 3> Results(Scratch, MVT::i32);
    Results.push_back(MVT::Other);
    MachineSDNode *Node = DAG.getMachineNode(
        Constant ? RISCV::PseudoTTMOPControlWriteImm
                 : RISCV::PseudoTTMOPControlWrite,
        DL, DAG.getVTList(Results),
        {DAG.getTargetConstant(Op.getConstantOperandVal(2), DL, MVT::i32),
         Value, Op.getOperand(0)});
    attachIssueMemory(Node, DAG, MachineMemOperand::MOStore);
    return SDValue(Node, Scratch);
  }
  if (TemplateOpcode) {
    SmallVector<SDValue, 5> Fields;
    for (unsigned I = 2; I != Op.getNumOperands(); ++I)
      Fields.push_back(DAG.getTargetConstant(Op.getConstantOperandVal(I), DL,
                                            MVT::i32));
    Fields.push_back(Op.getOperand(0));
    bool TemplateMop = TemplateOpcode == RISCV::PseudoTTReplayTemplateMop;
    SmallVector<EVT, 3> Results(TemplateMop ? 2 : 0, MVT::i32);
    Results.push_back(MVT::Other);
    MachineSDNode *Node = DAG.getMachineNode(
        TemplateOpcode, DL, DAG.getVTList(Results), Fields);
    // Actual recording/execution boundaries retain ordinary REPLAY's memory
    // ordering through effect normalization and final control replacement.
    // End is only an issue-order marker, not an executed memory operation.
    if (TemplateMop)
      attachIssueMemory(Node, DAG, MachineMemOperand::MOStore);
    else if (TemplateOpcode != RISCV::PseudoTTReplayTemplateEnd)
      attachIssueMemory(Node, DAG, MachineMemOperand::MOLoad |
                                       MachineMemOperand::MOStore);
    return SDValue(Node, Results.size() - 1);
  }
  if (End)
    return SDValue(DAG.getMachineNode(RISCV::PseudoTTReplayRecordEnd, DL,
                                     MVT::Other, Op.getOperand(0)), 0);
  bool Port = Info && ID == Info->PortIntrinsicID;
  bool Mop = Clear || (Info && ID == Info->MopIntrinsicID);
  const auto *Machine = Info ? RISCV::getTensixMachineInfoByIntrinsic(
                                  Info->IntrinsicID) : nullptr;
  if (Port && !RISCV::isTensixFieldPortOpcode(Machine->Opcode))
    reportFatalUsageError(
        "Tensix instruction-port issue reached selection without its "
        "issue-word fold");
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
  auto MemoryFlags = MachineMemOperand::MOStore;
  if (!Mop)
    MemoryFlags |= MachineMemOperand::MOLoad;
  attachIssueMemory(Node, DAG, MemoryFlags);
  return SDValue(Node, Scratch);
}
