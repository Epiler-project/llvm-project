//===-- RISCVTensixContentsGroup.cpp -------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixContentsGroup.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVTensixBoundContract.h"
#include "RISCVInstrInfo.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixBoundLowering.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetOpcodes.h"
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
                                const MCInstrInfo &Instructions, bool Wormhole) {
  MCInst Native;
  Native.setOpcode(Opcode);
  Native.addOperand(MCOperand::createImm(Mask));
  Native.addOperand(MCOperand::createImm(Mode));
  return Wormhole
             ? RISCV::verifyTensixWormholeInstruction(Native, Instructions)
             : RISCV::verifyTensixMCInstructionFields(Native, Instructions);
}
// A pair is an internal allocation aggregate. Its contents remain two
// independently observable numerical SSA values in the public intrinsic ABI.
SDValue buildIndexPair(SelectionDAG &DAG, const SDLoc &DL, SDValue Value,
                       SDValue Index) {
  SDValue Operands[] = {
      DAG.getTargetConstant(RISCV::SFPRIndexPairRegClassID, DL, MVT::i32),
      Value, DAG.getTargetConstant(RISCV::sub_sfpu_value, DL, MVT::i32), Index,
      DAG.getTargetConstant(RISCV::sub_sfpu_index, DL, MVT::i32)};
  return SDValue(DAG.getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                    MVT::Untyped, Operands),
                 0);
}

void extractIndexPair(SelectionDAG &DAG, const SDLoc &DL, SDValue Pair,
                      SmallVectorImpl<SDValue> &Values) {
  Values.push_back(
      DAG.getTargetExtractSubreg(RISCV::sub_sfpu_value, DL, MVT::v32i32, Pair));
  Values.push_back(
      DAG.getTargetExtractSubreg(RISCV::sub_sfpu_index, DL, MVT::v32i32, Pair));
}

SDValue lowerIndexGroup(SDValue Op, SelectionDAG &DAG, bool DynamicLoad,
                        unsigned Opcode) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  bool Swap = ID == Intrinsic::riscv_tt_contents_swap_index;
  SDLoc DL(Op);
  SmallVector<SDValue, 8> Operands;
  // Source SWAP order is D,C,D-index,C-index; pair order follows hardware.
  Operands.push_back(
      buildIndexPair(DAG, DL, Op.getOperand(2), Op.getOperand(Swap ? 4 : 3)));
  if (Swap)
    Operands.push_back(
        buildIndexPair(DAG, DL, Op.getOperand(3), Op.getOperand(5)));
  for (unsigned I = Swap ? 6 : 4; I != Op.getNumOperands(); ++I) {
    SDValue Input = Op.getOperand(I);
    if (const auto *Immediate = dyn_cast<ConstantSDNode>(Input))
      Input = DAG.getTargetConstant(Immediate->getSExtValue(), DL, MVT::i32);
    Operands.push_back(Input);
  }
  Operands.push_back(Op.getOperand(0));
  SmallVector<EVT, 4> Results(Swap ? 2 : 1, MVT::Untyped);
  if (DynamicLoad)
    Results.append(2, MVT::i32);
  unsigned Chain = Results.size();
  Results.push_back(MVT::Other);
  MachineSDNode *Node =
      DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results), Operands);
  if (!Swap) {
    auto Access = MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile;
    if (DynamicLoad)
      Access |= MachineMemOperand::MOStore;
    MachineMemOperand *MMO = DAG.getMachineFunction().getMachineMemOperand(
        MachinePointerInfo(), Access, 4, Align(4));
    DAG.setNodeMemRefs(Node, {MMO});
  }
  SmallVector<SDValue, 5> Values;
  extractIndexPair(DAG, DL, SDValue(Node, 0), Values);
  if (Swap) {
    extractIndexPair(DAG, DL, SDValue(Node, 1), Values);
    // Re-establish the intrinsic's numerical result order, not pair order.
    std::swap(Values[1], Values[2]);
  }
  Values.push_back(SDValue(Node, Chain));
  return DAG.getMergeValues(Values, DL);
}

bool expandIndexGroup(MachineInstr &MI, const RISCVSubtarget &ST) {
  bool Swap = MI.getOpcode() == RISCV::PseudoTTSFPContentsSwapIndex;
  bool Dynamic = MI.getOpcode() == RISCV::PseudoTTSFPContentsLoadIndexDynamic;
  if (!Swap && !Dynamic &&
      MI.getOpcode() != RISCV::PseudoTTSFPContentsLoadIndex)
    return false;
  MachineFunction &MF = *MI.getMF();
  auto Reject = [&](const Twine &Message) {
    // Keep actual defs/uses on failure; no native artifact is published.
    MF.getProperties().setFailedRegAlloc();
    MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
    MF.getFunction().getContext().diagnose(
        DiagnosticInfoUnsupported(MF.getFunction(), Message));
    return true;
  };
  const auto &TRI = *ST.getRegisterInfo();
  if (Error E = verifyTensixMachineEffects(MI))
    return Reject(toString(std::move(E)));
  if (Error E = verifyTensixMachineOperands(MI, MF.getRegInfo(), TRI))
    return Reject(toString(std::move(E)));
  for (const MachineOperand &MO : MI.explicit_operands())
    if (MO.isReg() && !MO.getReg().isPhysical())
      return Reject("contents index groups require allocated native operands");

  unsigned Opcode = Swap      ? RISCV::TTSFPSWAPIndex
                    : Dynamic ? RISCV::PseudoTTSFPLOADIndex
                              : RISCV::TTSFPLOADIndex;
  MachineInstrBuilder Native = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(),
                                       ST.getInstrInfo()->get(Opcode));
  auto AddLeaf = [&](unsigned Operand, unsigned Subreg, bool Def) {
    const MachineOperand &Parent = MI.getOperand(Operand);
    MCRegister Leaf = TRI.getSubReg(Parent.getReg(), Subreg);
    RegState Flags = getRenamableRegState(Parent.isRenamable());
    if (Def)
      Flags |= RegState::Define | getDeadRegState(Parent.isDead());
    else
      Flags |= getKillRegState(Parent.isKill());
    Native.addReg(Leaf, Flags);
  };
  AddLeaf(0, RISCV::sub_sfpu_value, true);
  if (Swap)
    AddLeaf(1, RISCV::sub_sfpu_value, true);
  AddLeaf(0, RISCV::sub_sfpu_index, true);
  if (Swap)
    AddLeaf(1, RISCV::sub_sfpu_index, true);
  else if (Dynamic) {
    Native.add(MI.getOperand(1));
    Native.add(MI.getOperand(2));
  }
  unsigned Old = Swap ? 2 : Dynamic ? 3 : 1;
  AddLeaf(Old, RISCV::sub_sfpu_value, false);
  if (Swap)
    AddLeaf(Old + 1, RISCV::sub_sfpu_value, false);
  AddLeaf(Old, RISCV::sub_sfpu_index, false);
  if (Swap)
    AddLeaf(Old + 1, RISCV::sub_sfpu_index, false);
  for (unsigned I = Old + (Swap ? 2 : 1); I != MI.getDesc().getNumOperands();
       ++I)
    Native.add(MI.getOperand(I));
  Native.cloneMemRefs(MI);
  Native.setMIFlags(MI.getFlags());
  if (Error E = verifyTensixContentsIndexNative(*Native, ST)) {
    Native->eraseFromParent();
    return Reject(toString(std::move(E)));
  }
  MI.eraseFromParent();
  return true;
}

} // namespace

unsigned llvm::getTensixContentsGroupResultCount(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_tt_contents_transpose8:
  case Intrinsic::riscv_tt_contents_lutfp32_three:
    return 8;
  case Intrinsic::riscv_tt_contents_transpose4:
  case Intrinsic::riscv_tt_contents_copy4:
  case Intrinsic::riscv_tt_contents_rotate4:
  case Intrinsic::riscv_tt_contents_swap_index:
    return 4;
  case Intrinsic::riscv_tt_contents_swap:
  case Intrinsic::riscv_tt_contents_load_index:
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

bool llvm::isTensixWormholeContentsGroupIntrinsic(Intrinsic::ID ID) {
  // These WH/BH groups share complete numerical inputs/results. LUTFP32
  // selects its architecture-specific encoding only after allocation; no
  // Blackhole scheduler mirror is emitted on Wormhole. Lane configuration
  // uses the same masked VD15 native form on both architectures.
  return ID == Intrinsic::riscv_tt_contents_load_index ||
         ID == Intrinsic::riscv_tt_contents_config_creg ||
         ID == Intrinsic::riscv_tt_contents_config_lane ||
         ID == Intrinsic::riscv_tt_contents_swap ||
         ID == Intrinsic::riscv_tt_contents_swap_index ||
         ID == Intrinsic::riscv_tt_contents_transpose4 ||
         ID == Intrinsic::riscv_tt_contents_transpose8 ||
         ID == Intrinsic::riscv_tt_contents_copy4 ||
         ID == Intrinsic::riscv_tt_contents_rotate4 ||
         ID == Intrinsic::riscv_tt_contents_lut ||
         ID == Intrinsic::riscv_tt_contents_lutfp32_six ||
         ID == Intrinsic::riscv_tt_contents_lutfp32_three;
}

Error llvm::verifyTensixContentsGroupIntrinsicFields(const IntrinsicInst &II,
                                                    bool Wormhole) {
  Intrinsic::ID ID = II.getIntrinsicID();
  if (!isTensixContentsGroupIntrinsic(ID))
    return createStringError("unknown contents group intrinsic");
  if (Wormhole && !isTensixWormholeContentsGroupIntrinsic(ID))
    return createStringError("contents SFPU group is not supported on Wormhole");
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
  SmallVector<std::optional<int64_t>, 10> Arguments;
  for (const Value *Argument : II.args()) {
    const auto *CI = dyn_cast<ConstantInt>(Argument);
    Arguments.push_back(CI && CI->getValue().getActiveBits() <= 32
                            ? std::optional<int64_t>(CI->getSExtValue())
                            : std::nullopt);
  }
  return verifyTensixContentsGroupFields(ID, Arguments, Wormhole);
}

Error llvm::verifyTensixContentsGroupFields(
    Intrinsic::ID ID, ArrayRef<std::optional<int64_t>> Arguments,
    bool Wormhole) {
  if (!isTensixContentsGroupIntrinsic(ID))
    return createStringError("unknown contents group intrinsic");
  if (Wormhole && !isTensixWormholeContentsGroupIntrinsic(ID))
    return createStringError(
        "contents SFPU group is not supported on Wormhole");
  auto Immediate = [&](unsigned Index) -> std::optional<uint64_t> {
    if (!Arguments[Index])
      return std::nullopt;
    return static_cast<uint32_t>(*Arguments[Index]);
  };
  switch (ID) {
  case Intrinsic::riscv_tt_contents_load_index: {
    if (Immediate(2) && *Immediate(2) > 1023)
      return createStringError(
          "contents paired LOAD Dst offset must be in [0, 1023]");
    if (!Immediate(3) || !Immediate(4))
      return createStringError(
          "contents paired LOAD AddrMod and format must be immediate");
    static const auto Instructions = RISCV::createTensixBoundMCInstrInfo();
    MCInst Native;
    // The paired view emits the ordinary LOAD word. Receive its scalar
    // fields through that instruction; its +4 register relation is checked
    // after allocation, when the actual result/old registers exist.
    Native.setOpcode(RISCV::TTSFPLOAD);
    for (unsigned I = 0; I != 2; ++I)
      Native.addOperand(MCOperand());
    // Scalar SSA coordinates are checked independently at their actual use.
    // Offset zero checks only the remaining architecture-specific fields.
    Native.addOperand(MCOperand::createImm(Immediate(2).value_or(0)));
    for (unsigned I : {3u, 4u})
      Native.addOperand(MCOperand::createImm(*Immediate(I)));
    return Wormhole
               ? RISCV::verifyTensixWormholeInstruction(Native, *Instructions)
               : RISCV::verifyTensixMCInstructionFields(Native, *Instructions);
  }
  case Intrinsic::riscv_tt_contents_copy4:
    if (!Immediate(4) || *Immediate(4) > 1)
      return createStringError("contents copy4 mode must be 0 or 1");
    break;
  case Intrinsic::riscv_tt_contents_swap:
    if (!Immediate(2) || *Immediate(2) > 9)
      return createStringError("contents swap mode must be in 0..9");
    break;
  case Intrinsic::riscv_tt_contents_swap_index:
    if (!Immediate(4) || *Immediate(4) > 9)
      return createStringError("contents swap index mode must be in 0..9");
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
                                       *Immediate(2), *Instructions, Wormhole);
    static const unsigned Opcodes[] = {
        RISCV::TTSFPCONFIGC11, RISCV::TTSFPCONFIGC12, RISCV::TTSFPCONFIGC13,
        RISCV::TTSFPCONFIGC14};
    return verifyConfigurationFields(Opcodes[*Immediate(1) - 11], *Immediate(2),
                                     *Immediate(3), *Instructions, Wormhole);
  }
  return Error::success();
}
Expected<unsigned>
llvm::getTensixContentsGroupOpcode(Intrinsic::ID ID,
                                   ArrayRef<std::optional<int64_t>> Arguments) {
  switch (ID) {
  case Intrinsic::riscv_tt_contents_load_index:
    return Arguments[2] ? RISCV::PseudoTTSFPContentsLoadIndex
                        : RISCV::PseudoTTSFPContentsLoadIndexDynamic;
  case Intrinsic::riscv_tt_contents_swap_index:
    return RISCV::PseudoTTSFPContentsSwapIndex;
  case Intrinsic::riscv_tt_contents_transpose4:
    return RISCV::PseudoTTSFPContentsTranspose4;
  case Intrinsic::riscv_tt_contents_transpose8:
    return RISCV::PseudoTTSFPContentsTranspose8;
  case Intrinsic::riscv_tt_contents_copy4:
    return RISCV::PseudoTTSFPContentsCopy4;
  case Intrinsic::riscv_tt_contents_rotate4:
    return RISCV::PseudoTTSFPContentsRotate4;
  case Intrinsic::riscv_tt_contents_swap:
    return RISCV::TTSFPSWAP;
  case Intrinsic::riscv_tt_contents_lut:
    return RISCV::PseudoTTSFPContentsLUT;
  case Intrinsic::riscv_tt_contents_lutfp32_six:
    return RISCV::PseudoTTSFPContentsLUTSix;
  case Intrinsic::riscv_tt_contents_lutfp32_three:
    return RISCV::PseudoTTSFPContentsLUTThree;
  case Intrinsic::riscv_tt_contents_config_creg: {
    if (!Arguments[1])
      return createStringError("contents configuration requires C11..C14");
    int64_t Register = *Arguments[1];
    if (Register < 11 || Register > 14)
      return createStringError("contents configuration requires C11..C14");
    static const unsigned Opcodes[] = {RISCV::PseudoTTSFPContentsConfigC11,
                                       RISCV::PseudoTTSFPContentsConfigC12,
                                       RISCV::PseudoTTSFPContentsConfigC13,
                                       RISCV::PseudoTTSFPContentsConfigC14};
    return Opcodes[Register - 11];
  }
  case Intrinsic::riscv_tt_contents_config_lane:
    return RISCV::PseudoTTSFPContentsConfigLane;
  default:
    return createStringError("unknown contents group intrinsic");
  }
}

SDValue llvm::lowerTensixContentsGroupIntrinsic(SDValue Op, SelectionDAG &DAG,
                                                const RISCVSubtarget &ST) {
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isTensixContentsGroupIntrinsic(ID))
    return SDValue();
  unsigned Count = getTensixContentsGroupResultCount(ID);
  const Function &F = DAG.getMachineFunction().getFunction();
  Attribute Executor = F.getFnAttribute("tensix-executor");
  bool Supported = ST.hasVendorXTTTensixBH() ||
                   (ST.hasVendorXTTTensixWH() &&
                    isTensixWormholeContentsGroupIntrinsic(ID));
  if (!Supported || !Executor.isStringAttribute() ||
      Executor.getValueAsString() != "trisc1")
    reportFatalUsageError("contents groups require a supported Tensix TRISC1");

  bool DynamicLoad = ID == Intrinsic::riscv_tt_contents_load_index &&
                     !isa<ConstantSDNode>(Op.getOperand(4));
  SmallVector<std::optional<int64_t>, 10> Arguments;
  for (unsigned I = 2; I != Op.getNumOperands(); ++I) {
    const auto *Immediate = dyn_cast<ConstantSDNode>(Op.getOperand(I));
    Arguments.push_back(Immediate
                            ? std::optional<int64_t>(Immediate->getSExtValue())
                            : std::nullopt);
  }
  auto Selected = getTensixContentsGroupOpcode(ID, Arguments);
  if (!Selected)
    reportFatalUsageError(Twine(toString(Selected.takeError())));
  unsigned Opcode = *Selected;
  if (ID == Intrinsic::riscv_tt_contents_load_index ||
      ID == Intrinsic::riscv_tt_contents_swap_index)
    return lowerIndexGroup(Op, DAG, DynamicLoad, Opcode);
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
  SmallVector<EVT, 11> Results(Count, MVT::v32i32);
  Results.push_back(MVT::Other);
  MachineSDNode *Node =
      DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results), Operands);
  return SDValue(Node, 0);
}

Error llvm::verifyTensixContentsIndexNative(const MachineInstr &MI,
                                             const RISCVSubtarget &ST) {
  switch (MI.getOpcode()) {
  case RISCV::PseudoTTSFPContentsLoadIndex:
  case RISCV::PseudoTTSFPContentsLoadIndexDynamic:
  case RISCV::PseudoTTSFPContentsSwapIndex:
    return createStringError("unexpanded coupled SFPU index group reached "
                             "the final native receiver");
  case RISCV::TTSFPLOADIndex:
  case RISCV::PseudoTTSFPLOADIndex:
  case RISCV::TTSFPSWAPIndex:
    break;
  default:
    return Error::success();
  }
  if (Error E = verifyTensixMachineEffects(MI))
    return E;
  if (Error E = verifyTensixMachineOperands(MI, MI.getMF()->getRegInfo(),
                                           *ST.getRegisterInfo()))
    return E;
  bool DynamicLoad = MI.getOpcode() == RISCV::PseudoTTSFPLOADIndex;
  if (DynamicLoad) {
    if (!MI.hasOneMemOperand())
      return createStringError(
          "contents paired LOAD port requires its volatile memory effects");
    const MachineMemOperand &MMO = **MI.memoperands_begin();
    if (!MMO.isVolatile() || !MMO.isLoad() || !MMO.isStore() ||
        MMO.isAtomic() || MMO.getSize() != LocationSize::precise(4))
      return createStringError(
          "contents paired LOAD port requires its volatile memory effects");
  }
  MCInst Native;
  Native.setOpcode(DynamicLoad ? unsigned(RISCV::TTSFPLOADIndex)
                               : MI.getOpcode());
  for (unsigned I = 0; I != MI.getDesc().getNumOperands(); ++I) {
    // Scratch and the dynamic scalar are separate machine operands, never
    // physical SFPU selectors. The native MC view checks offset zero only.
    if (DynamicLoad && (I == 2 || I == 3 || I == 6)) {
      if (!MI.getOperand(I).getReg().isPhysical())
        return createStringError(
            "contents paired LOAD port requires allocated scalar operands");
      if (I == 6)
        Native.addOperand(MCOperand::createImm(0));
      continue;
    }
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg()) {
      if (!MO.getReg().isPhysical())
        return createStringError(
            "contents index native receiver requires physical registers");
      Native.addOperand(MCOperand::createReg(MO.getReg()));
    } else {
      Native.addOperand(MCOperand::createImm(MO.getImm()));
    }
  }
  const auto &Instructions = *ST.getInstrInfo();
  return ST.hasVendorXTTTensixWH()
             ? RISCV::verifyTensixWormholeInstruction(Native, Instructions)
             : RISCV::verifyTensixMCInstructionFields(Native, Instructions);
}

bool llvm::expandTensixContentsGroupPseudo(MachineInstr &MI,
                                           const RISCVSubtarget &ST) {
  if (expandIndexGroup(MI, ST))
    return true;
  unsigned Count, Opcode, FixedStart, FixedCount, Extras;
  switch (MI.getOpcode()) {
  case RISCV::PseudoTTSFPContentsTranspose4:
    Count = 4;
    Opcode = RISCV::TTSFPTRANSP;
    FixedStart = 4;
    FixedCount = 4;
    Extras = 0;
    break;
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
  if (MI.getOpcode() == RISCV::PseudoTTSFPContentsTranspose4)
    if (Error E = verifyTensixMachineEffects(MI))
      return Reject(toString(std::move(E)));
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
                                              *ST.getInstrInfo(),
                                              ST.hasVendorXTTTensixWH()))
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
  if (Opcode == RISCV::TTSFPTRANSP && Count == 8) {
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
