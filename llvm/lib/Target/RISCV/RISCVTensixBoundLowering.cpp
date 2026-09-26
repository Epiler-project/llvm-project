//===-- RISCVTensixBoundLowering.cpp --------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixBoundLowering.h"
#include "MCTargetDesc/RISCVTensixBoundContract.h"
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
Error invalid(const Twine &Message) {
  return createStringError(Twine("bound SFPU ") + Message);
}
} // namespace

bool llvm::isTensixBoundSFPUIntrinsic(Intrinsic::ID ID) {
  return RISCV::getTensixBoundInstruction(ID) != nullptr;
}

std::optional<unsigned> llvm::getTensixBoundDstOffsetOperand(Intrinsic::ID ID) {
  const auto *Info = RISCV::getTensixBoundInstruction(ID);
  return Info ? Info->DstOffset : std::nullopt;
}

Error llvm::verifyTensixBoundSFPUIntrinsic(const IntrinsicInst &II) {
  const auto *Info = RISCV::getTensixBoundInstruction(II.getIntrinsicID());
  if (!Info)
    return invalid("unknown intrinsic");
  if (!II.getType()->isVoidTy())
    return invalid("intrinsic must not return an allocator-owned value");
  SmallVector<RISCV::TensixBoundValue, 12> Args;
  for (const Value *Arg : II.args()) {
    if (!Arg->getType()->isIntegerTy(32))
      return invalid("all argument fields must have type i32");
    const auto *Constant = dyn_cast<ConstantInt>(Arg);
    Args.push_back(
        Constant
            ? RISCV::TensixBoundValue{RISCV::TensixBoundValueKind::Constant,
                                      Constant->getSExtValue()}
            : RISCV::TensixBoundValue{
                  RISCV::TensixBoundValueKind::DynamicScalar});
  }
  return RISCV::verifyTensixBoundSFPUOperands(Info->ID, Args);
}

SDValue llvm::lowerTensixBoundSFPUIntrinsic(SDValue Op, SelectionDAG &DAG,
                                            const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  const auto *Info = RISCV::getTensixBoundInstruction(ID);
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
  const auto *Info = RISCV::getTensixBoundInstructionForPseudo(MI.getOpcode());
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
  SmallVector<RISCV::TensixBoundValue, 12> Args;
  for (unsigned I = 0; I != Info->ArgumentCount; ++I) {
    const MachineOperand &MO = MI.getOperand(FirstArgument + I);
    if (Dynamic && I == *Offset) {
      if (!MO.isReg())
        return Fail("bound SFPU dynamic Dst offset must be a GPR");
      Args.push_back({RISCV::TensixBoundValueKind::DynamicScalar});
    } else {
      if (!MO.isImm())
        return Fail("bound SFPU ingress lost a physical immediate");
      Args.push_back({RISCV::TensixBoundValueKind::Constant, MO.getImm()});
    }
  }
  auto Native = RISCV::getTensixBoundNativeInstruction(Info->ID, Args);
  if (!Native)
    return Fail(toString(Native.takeError()));

  bool Load = Info->ID == Intrinsic::riscv_tt_bound_sfpload;
  unsigned Opcode =
      Dynamic ? unsigned(Load ? RISCV::PseudoTTSFPLOAD : RISCV::PseudoTTSFPSTORE)
              : Native->Opcode;
  const MCInstrDesc &NativeDesc = ST.getInstrInfo()->get(Native->Opcode);
  auto Builder =
      BuildMI(*MBB, MI, MI.getDebugLoc(), ST.getInstrInfo()->get(Opcode));
  Builder.cloneMemRefs(MI);
  for (auto [I, Operand] : llvm::enumerate(Native->Operands)) {
    // Dynamic Dst issue retains only the already-created ordinary GPR defs.
    // In a load they follow the physical destination; in a store they lead.
    if (Dynamic && I == (Load ? 1u : 0u))
      for (unsigned Scratch = 0; Scratch != 2; ++Scratch)
        Builder.addReg(MI.getOperand(Scratch).getReg(),
                       RegState::Define | RegState::EarlyClobber);
    if (NativeDesc.operands()[I].RegClass >= 0) {
      assert(Operand.Argument && "verified native register projection");
      Builder.addReg(
          RISCV::getTensixSFPURegister(Args[*Operand.Argument].Constant),
          I < NativeDesc.getNumDefs() ? RegState::Define : RegState::NoFlags);
    } else if (Dynamic && Operand.Argument == Offset) {
      Builder.addReg(MI.getOperand(FirstArgument + *Offset).getReg());
    } else {
      Builder.addImm(Operand.Argument ? Args[*Operand.Argument].Constant
                                      : Operand.Constant);
    }
  }
  MI.eraseFromParent();
  return MBB;
}
