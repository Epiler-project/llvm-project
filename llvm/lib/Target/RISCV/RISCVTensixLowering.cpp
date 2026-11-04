//===-- RISCVTensixLowering.cpp
//--------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCVTensixLowering.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVTensixBoundContract.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixContentsGroup.h"
#include "RISCVTensixIRVerification.h"
#include "RISCVTensixMacro.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

namespace {
struct ContentsOperand {
  std::optional<unsigned> Argument;
  int64_t Immediate = 0;
};

struct ContentsInstruction {
  unsigned Opcode = 0;
  SmallVector<ContentsOperand, 6> Operands;
};

// This projection selects native operations and expresses only ABI fields.
// Register classes, old-destination ties, lane/configuration/PRNG effects and
// immediate legality remain owned by the existing native descriptors.
Expected<ContentsInstruction>
contentsInstruction(Intrinsic::ID ID,
                    ArrayRef<std::optional<int64_t>> Arguments) {
  ContentsInstruction Result;
  auto Select = [&](unsigned Opcode, std::initializer_list<unsigned> Inputs) {
    Result.Opcode = Opcode;
    for (unsigned Index : Inputs)
      Result.Operands.push_back({Index});
  };
  auto Literal = [&](int64_t Value) {
    Result.Operands.push_back({std::nullopt, Value});
  };
  auto Mode = [&](unsigned Index) { return Arguments[Index].value_or(-1); };
  auto Invalid = []() -> Expected<ContentsInstruction> {
    return createStringError("unsupported contents instruction form or mode");
  };
  switch (ID) {
  case Intrinsic::riscv_tt_contents_add:
  case Intrinsic::riscv_tt_contents_mul:
    Select(ID == Intrinsic::riscv_tt_contents_add ? RISCV::TTSFPADD
                                                  : RISCV::TTSFPMUL,
           {0, 1, 2, 3});
    break;
  case Intrinsic::riscv_tt_contents_load:
  case Intrinsic::riscv_tt_contents_store:
    Select(ID == Intrinsic::riscv_tt_contents_load ? RISCV::TTSFPLOAD
                                                   : RISCV::TTSFPSTORE,
           {0, 1, 2, 3});
    break;
  case Intrinsic::riscv_tt_contents_load_full:
    Select(RISCV::TTSFPLOADFull, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_load_immediate:
    Select(RISCV::TTSFPLOADI, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_divp2:
    Select(RISCV::TTSFPDIVP2, {0, 1, 2, 3});
    break;
  case Intrinsic::riscv_tt_contents_add_immediate:
  case Intrinsic::riscv_tt_contents_mul_immediate:
    Select(ID == Intrinsic::riscv_tt_contents_add_immediate ? RISCV::TTSFPADDI
                                                            : RISCV::TTSFPMULI,
           {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_mul23:
    Select(RISCV::TTSFPMUL24, {0, 1, 2, 3});
    break;
  case Intrinsic::riscv_tt_contents_mad:
    Select(RISCV::TTSFPMAD, {0, 1, 2, 3, 4});
    break;
  case Intrinsic::riscv_tt_contents_iadd: {
    constexpr unsigned Opcodes[] = {RISCV::TTSFPIADDCCLT, RISCV::TTSFPISUBCCLT,
                                    RISCV::TTSFPIADD,     RISCV::TTSFPISUB,
                                    RISCV::TTSFPIADDCCGE, RISCV::TTSFPISUBCCGE};
    int64_t Value = Mode(2);
    if (Value < 0 || Value > 10 || (Value & 1))
      return Invalid();
    Select(Opcodes[Value / 2], {0, 1});
    break;
  }
  case Intrinsic::riscv_tt_contents_mov:
    if (Mode(2) != 0 && Mode(2) != 1)
      return Invalid();
    Select(Mode(2) ? RISCV::TTSFPMOVNeg : RISCV::TTSFPMOV, {0, 1});
    break;
  case Intrinsic::riscv_tt_contents_mov_all:
    Select(RISCV::TTSFPMOVAll, {0});
    break;
  case Intrinsic::riscv_tt_contents_prng_advance:
    Select(RISCV::TTSFPMOVPRNGAdvance, {0});
    break;
  case Intrinsic::riscv_tt_contents_arecip:
    Select(RISCV::TTSFPARECIP, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_exexp:
    Select(RISCV::TTSFPEXEXP, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_exman:
    Select(RISCV::TTSFPEXMAN, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_abs:
    Select(RISCV::TTSFPABS, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_lz:
    Select(RISCV::TTSFPLZ, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_cast:
    Select(RISCV::TTSFPCAST, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_setexp_immediate:
  case Intrinsic::riscv_tt_contents_setman_immediate:
  case Intrinsic::riscv_tt_contents_setsgn_immediate:
    if (Mode(3) < 0 || !(Mode(3) & 1))
      return Invalid();
    Select(ID == Intrinsic::riscv_tt_contents_setexp_immediate
               ? RISCV::TTSFPSETEXP
           : ID == Intrinsic::riscv_tt_contents_setman_immediate
               ? RISCV::TTSFPSETMAN
               : RISCV::TTSFPSETSGN,
           {0, 1, 2, 3});
    break;
  case Intrinsic::riscv_tt_contents_setexp_vector:
  case Intrinsic::riscv_tt_contents_setman_vector:
  case Intrinsic::riscv_tt_contents_setsgn_vector:
    if (Mode(2) < 0 || (Mode(2) & 1))
      return Invalid();
    // Vector field bits come from VD before the update, which is exactly the
    // tied old contents. There is no independent field value or hidden copy.
    Select(ID == Intrinsic::riscv_tt_contents_setexp_vector ? RISCV::TTSFPSETEXP
           : ID == Intrinsic::riscv_tt_contents_setman_vector
               ? RISCV::TTSFPSETMAN
               : RISCV::TTSFPSETSGN,
           {0, 1});
    Literal(0);
    Result.Operands.push_back({2});
    break;
  case Intrinsic::riscv_tt_contents_iadd_immediate:
  case Intrinsic::riscv_tt_contents_shft_immediate:
    if (ID == Intrinsic::riscv_tt_contents_shft_immediate &&
        (Mode(2) < 0 || !(Mode(2) & 1)))
      return Invalid();
    // Immediate IADD/SHFT operate on the old destination itself.
    Select(ID == Intrinsic::riscv_tt_contents_iadd_immediate ? RISCV::TTSFPIADDI
                                                             : RISCV::TTSFPSHFT,
           {0, 0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_shft_vector:
    if (Mode(2) < 0 || (Mode(2) & 1))
      return Invalid();
    Select(RISCV::TTSFPSHFT, {0, 1});
    Literal(0);
    Result.Operands.push_back({2});
    break;
  case Intrinsic::riscv_tt_contents_and:
    Select(RISCV::TTSFPAND, {0, 1});
    break;
  case Intrinsic::riscv_tt_contents_or:
    Select(RISCV::TTSFPOR, {0, 1});
    break;
  case Intrinsic::riscv_tt_contents_xor:
    Select(RISCV::TTSFPXOR, {0, 1});
    break;
  case Intrinsic::riscv_tt_contents_not:
    Select(RISCV::TTSFPNOT, {0, 0});
    break;
  case Intrinsic::riscv_tt_contents_shft2:
    Select(RISCV::TTSFPSHFT2, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_stochrnd_immediate: {
    int64_t Value = Mode(3);
    if (Value < 0 || Value > 7)
      return Invalid();
    Select(RISCV::TTSFPSTOCHRNDI, {0, 1, 2});
    Literal(Value == 4 || Value == 5 ? Value | 8 : Value);
    Result.Operands.push_back({4});
    break;
  }
  case Intrinsic::riscv_tt_contents_stochrnd_vector:
    Select(RISCV::TTSFPSTOCHRNDV, {0, 1, 2, 3, 4});
    break;
  case Intrinsic::riscv_tt_contents_encc:
    Select(RISCV::TTSFPENCC, {0, 1});
    break;
  case Intrinsic::riscv_tt_contents_pushc:
    if (Mode(0) != 0 && Mode(0) != 1)
      return Invalid();
    Select(Mode(0) ? RISCV::TTSFPPUSHCReplaceTop : RISCV::TTSFPPUSHC, {});
    break;
  case Intrinsic::riscv_tt_contents_popc: {
    unsigned Opcode;
    switch (Mode(0)) {
    case 0:
      Opcode = RISCV::TTSFPPOPC;
      break;
    case 1:
      Opcode = RISCV::TTSFPPOPCPeek;
      break;
    case 3:
      Opcode = RISCV::TTSFPPOPCAndTop;
      break;
    case 4:
      Opcode = RISCV::TTSFPPOPCOrTop;
      break;
    default:
      return Invalid();
    }
    Select(Opcode, {});
    break;
  }
  case Intrinsic::riscv_tt_contents_setcc: {
    constexpr unsigned Opcodes[] = {RISCV::TTSFPSETCCLT, RISCV::TTSFPSETCCNE,
                                    RISCV::TTSFPSETCCGE, RISCV::TTSFPSETCCEQ};
    int64_t Value = Mode(1);
    if (Value < 0 || Value > 6 || (Value & 1))
      return Invalid();
    Select(Opcodes[Value / 2], {0});
    break;
  }
  case Intrinsic::riscv_tt_contents_gt:
    Select(RISCV::TTSFPGT, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_le:
    Select(RISCV::TTSFPLE, {0, 1, 2});
    break;
  case Intrinsic::riscv_tt_contents_compc:
    Select(RISCV::TTSFPCOMPC, {});
    break;
  case Intrinsic::riscv_tt_contents_nop:
    Select(RISCV::TTSFPNOP, {});
    break;
  case Intrinsic::riscv_tt_contents_config_reset:
    Select(RISCV::TTSFPCONFIGReset, {});
    break;
  default:
    return Invalid();
  }
  return Result;
}

Expected<MCRegister> contentsSnapshotRegister(Intrinsic::ID ID,
                                              uint32_t Selector) {
  if (ID == Intrinsic::riscv_tt_contents_read_creg) {
    if (Selector < 11 || Selector > 14)
      return createStringError("contents CReg read requires mutable C11-C14");
  } else if (ID == Intrinsic::riscv_tt_contents_read_constant) {
    if (Selector != 8 && Selector != 9 && Selector != 10 && Selector != 15)
      return createStringError(
          "contents constant read requires immutable C8/C9/C10/C15");
  } else {
    return createStringError("unknown contents snapshot intrinsic");
  }
  return RISCV::getTensixSFPURegister(Selector);
}

Error verifyContentsInstructionFields(
    Intrinsic::ID ID, ArrayRef<std::optional<int64_t>> Arguments,
    bool Wormhole) {
  if (ID == Intrinsic::riscv_tt_contents_load_full && Arguments[2] &&
      (*Arguments[2] == 14 || *Arguments[2] == 15))
    return createStringError(
        "contents full LOAD requires a complete-word format");
  static const auto Instructions = RISCV::createTensixBoundMCInstrInfo();
  auto Verify = [&](const MCInst &Native) {
    return Wormhole
               ? RISCV::verifyTensixWormholeInstruction(Native, *Instructions)
               : RISCV::verifyTensixMCInstructionFields(Native, *Instructions);
  };
  if (ID == Intrinsic::riscv_tt_contents_read_constant ||
      ID == Intrinsic::riscv_tt_contents_read_creg) {
    if (!Arguments[0])
      return createStringError("contents snapshot selector must be immediate");
    auto Register =
        contentsSnapshotRegister(ID, static_cast<uint32_t>(*Arguments[0]));
    if (!Register)
      return Register.takeError();
    MCInst Native;
    Native.setOpcode(RISCV::TTSFPMOVAll);
    Native.addOperand(MCOperand());
    Native.addOperand(MCOperand());
    return Verify(Native);
  }
  SmallVector<std::optional<int64_t>, 6> Fields(Arguments);
  // This checks only the static native tuple. A dynamic coordinate remains
  // actual scalar SSA and still needs the independent IR definedness/range
  // receiver at its eventual use.
  std::optional<unsigned> DstOffset;
  if (ID == Intrinsic::riscv_tt_contents_load_full)
    DstOffset = 0;
  else if (ID == Intrinsic::riscv_tt_contents_load ||
           ID == Intrinsic::riscv_tt_contents_store)
    DstOffset = 1;
  if (DstOffset && !Fields[*DstOffset])
    Fields[*DstOffset] = 0;
  auto Projection = contentsInstruction(ID, Fields);
  if (!Projection)
    return Projection.takeError();
  const MCInstrDesc &Desc = Instructions->get(Projection->Opcode);
  MCInst Native;
  Native.setOpcode(Projection->Opcode);
  for (unsigned I = 0; I != Desc.getNumDefs(); ++I)
    Native.addOperand(MCOperand());
  for (const ContentsOperand &Operand : Projection->Operands) {
    if (Desc.operands()[Native.getNumOperands()].RegClass >= 0) {
      Native.addOperand(MCOperand());
    } else if (!Operand.Argument) {
      Native.addOperand(MCOperand::createImm(Operand.Immediate));
    } else if (auto Immediate = Fields[*Operand.Argument]) {
      Native.addOperand(MCOperand::createImm(*Immediate));
    } else {
      return createStringError("contents native field must be immediate");
    }
  }
  return Verify(Native);
}

void attachIssueMemory(MachineSDNode *Node, SelectionDAG &DAG,
                       MachineMemOperand::Flags Access) {
  MachineMemOperand *MMO = DAG.getMachineFunction().getMachineMemOperand(
      MachinePointerInfo(), Access | MachineMemOperand::MOVolatile, 4,
      Align(4));
  DAG.setNodeMemRefs(Node, {MMO});
}
} // namespace

Expected<RISCV::TensixContentsIntrinsicInfo>
RISCV::getTensixContentsIntrinsicInfo(Intrinsic::ID ID,
                                      ArrayRef<TensixContentsValue> Arguments,
                                      TensixSFPUArchitecture Architecture) {
  if (Architecture != TensixSFPUArchitecture::Wormhole &&
      Architecture != TensixSFPUArchitecture::Blackhole)
    return createStringError("unknown contents SFPU architecture");
  // Classification is exact and safe even for an arbitrary numeric ID.
  // Membership alone grants no local ABI/field/effect admission.
  if (!isTensixContentsIntrinsic(ID) || isTensixMacroIntrinsic(ID))
    return createStringError("intrinsic has no local contents contract");
  if (Intrinsic::isOverloaded(ID))
    return createStringError(
        "contents contract requires a fixed intrinsic ABI");

  // Read the generated ABI and ImmArg positions directly. No LLVM Module or
  // temporary intrinsic call is constructed, and no caller context is mutated.
  LLVMContext Context;
  FunctionType *Signature = Intrinsic::getType(Context, ID);
  AttributeList Attributes = Intrinsic::getAttributes(Context, ID, Signature);
  if (Signature->isVarArg() || Arguments.size() != Signature->getNumParams())
    return createStringError("incorrect number of contents intrinsic operands");
  auto IsContents = [](Type *Ty) {
    const auto *Vector = dyn_cast<FixedVectorType>(Ty);
    return Vector && Vector->getNumElements() == 32 &&
           Vector->getElementType()->isIntegerTy(32);
  };
  SmallVector<std::optional<int64_t>, 10> NativeArguments;
  for (unsigned I = 0; I != Arguments.size(); ++I) {
    Type *Ty = Signature->getParamType(I);
    const TensixContentsValue &Argument = Arguments[I];
    if (IsContents(Ty)) {
      if (Argument.Kind != TensixContentsValueKind::Contents ||
          Argument.Constant)
        return createStringError(
            "contents operand requires numerical v32i32 SSA");
    } else if (Ty->isIntegerTy(32)) {
      if (Argument.Kind != TensixContentsValueKind::Scalar)
        return createStringError("contents scalar operand requires actual i32");
      if (Attributes.hasParamAttr(I, Attribute::ImmArg) && !Argument.Constant)
        return createStringError("contents immediate operand must be constant");
    } else {
      return createStringError("intrinsic has no local contents operand ABI");
    }
    NativeArguments.push_back(
        Argument.Constant ? std::optional<int64_t>(SignExtend64<32>(
                                static_cast<uint64_t>(*Argument.Constant)))
                          : std::nullopt);
  }
  TensixContentsIntrinsicInfo Info;
  Type *Return = Signature->getReturnType();
  if (Return->isVoidTy()) {
    Info.NumResults = 0;
  } else if (IsContents(Return)) {
    Info.NumResults = 1;
  } else if (const auto *Tuple = dyn_cast<StructType>(Return)) {
    if (!Tuple->isLiteral() || Tuple->isPacked() ||
        Tuple->getNumElements() < 2 || !all_of(Tuple->elements(), IsContents))
      return createStringError(
          "contents result requires its exact literal tuple");
    Info.NumResults = Tuple->getNumElements();
  } else {
    return createStringError("intrinsic has no local contents result ABI");
  }

  const bool Wormhole = Architecture == TensixSFPUArchitecture::Wormhole;
  unsigned Opcode;
  std::optional<ContentsInstruction> Projection;
  const bool Snapshot = ID == Intrinsic::riscv_tt_contents_read_constant ||
                        ID == Intrinsic::riscv_tt_contents_read_creg;
  if (isTensixContentsGroupIntrinsic(ID)) {
    if (Error E =
            verifyTensixContentsGroupFields(ID, NativeArguments, Wormhole))
      return std::move(E);
    auto Selected = getTensixContentsGroupOpcode(ID, NativeArguments);
    if (!Selected)
      return Selected.takeError();
    Opcode = *Selected;
  } else {
    if (Error E =
            verifyContentsInstructionFields(ID, NativeArguments, Wormhole))
      return std::move(E);
    if (ID == Intrinsic::riscv_tt_contents_read_constant ||
        ID == Intrinsic::riscv_tt_contents_read_creg) {
      Opcode = RISCV::TTSFPMOVAll;
      auto Resource = getTensixSFPUArchitecturalResource(
          getTensixSFPURegister(*Arguments[0].Constant));
      if (!Resource)
        return Resource.takeError();
      appendTensixSFPUArchitecturalEffect(Info.Effects, *Resource,
                                          TensixSFPUAccess::Read);
    } else {
      auto Selected = contentsInstruction(ID, NativeArguments);
      if (!Selected)
        return Selected.takeError();
      Opcode = Selected->Opcode;
      Projection = std::move(*Selected);
    }
  }
  static const auto Instructions = createTensixBoundMCInstrInfo();
  const MCInstrDesc &Descriptor = Instructions->get(Opcode);
  for (MCPhysReg Register : Descriptor.implicit_uses()) {
    auto Resource = getTensixSFPUArchitecturalResource(Register);
    if (!Resource)
      return Resource.takeError();
    appendTensixSFPUArchitecturalEffect(Info.Effects, *Resource,
                                        TensixSFPUAccess::Read);
  }
  for (MCPhysReg Register : Descriptor.implicit_defs()) {
    auto Resource = getTensixSFPUArchitecturalResource(Register);
    if (!Resource)
      return Resource.takeError();
    // Numerical results are explicit descriptor operands. An implicit SFPR
    // definition is real scratch clobber; mutable CRegs/state remain writes.
    const auto *Fixed = std::get_if<TensixFixedRegisterRef>(&*Resource);
    TensixSFPUAccess Access = Fixed && Fixed->Number < 8
                                  ? TensixSFPUAccess::Clobber
                                  : TensixSFPUAccess::Write;
    appendTensixSFPUArchitecturalEffect(Info.Effects, *Resource, Access);
  }
  // The selected native evaluator is the same owner used by bound forms.
  // Actual ABI mappings only translate its value indices; they own no second
  // opcode transfer, predication, lane or configuration table.
  SmallVector<std::optional<int64_t>, 12> NativeFields(
      Descriptor.getNumOperands(), std::nullopt);
  if (Projection) {
    if (Projection->Operands.size() + Descriptor.getNumDefs() !=
        Descriptor.getNumOperands())
      return createStringError(
          "contents native mapping differs from descriptor");
    for (auto [Index, Operand] : enumerate(Projection->Operands))
      NativeFields[Descriptor.getNumDefs() + Index] =
          Operand.Argument ? NativeArguments[*Operand.Argument]
                           : std::optional<int64_t>(Operand.Immediate);
  }
  auto Semantics = getTensixSFPUInstructionSemantics(Opcode, NativeFields);
  if (!Semantics)
    return Semantics.takeError();
  auto Publish = [&](unsigned Index) -> Expected<unsigned> {
    if (Index >= Semantics->Values.size())
      return createStringError("contents semantic value is out of range");
    TensixContentsTransferValue Value;
    if (const auto *Fixed =
            std::get_if<TensixFixedRegisterRef>(&Semantics->Values[Index])) {
      Value = *Fixed;
    } else {
      unsigned Operand =
          std::get<TensixSFPUInstructionOperandRef>(Semantics->Values[Index])
              .Operand;
      if (Operand < Descriptor.getNumDefs()) {
        if (Operand >= Info.NumResults)
          return createStringError("native definition has no contents result");
        Value = TensixContentsResultRef{Operand};
      } else if (Snapshot && Operand == Descriptor.getNumDefs()) {
        Value = TensixFixedRegisterRef{*Arguments[0].Constant};
      } else {
        unsigned Use = Operand - Descriptor.getNumDefs();
        if (!Projection || Use >= Projection->Operands.size() ||
            !Projection->Operands[Use].Argument)
          return createStringError(
              "native transfer has no actual contents SSA");
        unsigned Argument = *Projection->Operands[Use].Argument;
        if (Argument >= Arguments.size() ||
            Arguments[Argument].Kind != TensixContentsValueKind::Contents)
          return createStringError("native transfer input is not contents SSA");
        Value = TensixContentsArgumentRef{Argument};
      }
    }
    unsigned Published = Info.TransferValues.size();
    Info.TransferValues.push_back(std::move(Value));
    return Published;
  };
  for (auto Transfer : Semantics->Transfers) {
    auto Destination = Publish(Transfer.DestinationArgument);
    if (!Destination)
      return Destination.takeError();
    Transfer.DestinationArgument = *Destination;
    auto Read = [&](std::optional<unsigned> &Index) -> Error {
      if (!Index)
        return Error::success();
      auto Published = Publish(*Index);
      if (!Published)
        return Published.takeError();
      Index = *Published;
      return Error::success();
    };
    if (Error E = Read(Transfer.SourceArgument))
      return std::move(E);
    if (Error E = Read(Transfer.OldArgument))
      return std::move(E);
    for (auto &Input : Transfer.Inputs) {
      auto Published = Publish(Input.Argument);
      if (!Published)
        return Published.takeError();
      Input.Argument = *Published;
    }
    for (auto &Source : Transfer.LaneSources) {
      auto Published = Publish(Source.Argument);
      if (!Published)
        return Published.takeError();
      Source.Argument = *Published;
    }
    Info.Transfers.push_back(std::move(Transfer));
  }
  Info.LaneControl = Semantics->LaneControl;
  if (Semantics->Configuration) {
    Info.Configuration = *Semantics->Configuration;
    for (auto &Requirement : Info.Configuration->Requirements) {
      auto Published = Publish(Requirement.Argument);
      if (!Published)
        return Published.takeError();
      Requirement.Argument = *Published;
    }
  }
  return Info;
}

Error llvm::verifyTensixContentsIntrinsicFields(const IntrinsicInst &II,
                                                bool Wormhole) {
  switch (II.getIntrinsicID()) {
  case Intrinsic::riscv_tt_contents_replay_template_begin:
  case Intrinsic::riscv_tt_contents_replay_template_end:
  case Intrinsic::riscv_tt_contents_replay_template_mop:
  case Intrinsic::riscv_tt_contents_replay_template_mop_execute:
  case Intrinsic::riscv_tt_contents_replay_template_execute:
  case Intrinsic::riscv_tt_contents_replay_template_record_iadd:
  case Intrinsic::riscv_tt_contents_replay_template_record_and:
  case Intrinsic::riscv_tt_contents_replay_template_record_or:
  case Intrinsic::riscv_tt_contents_replay_template_record_xor:
  case Intrinsic::riscv_tt_contents_replay_template_record_not:
    return Error::success();
  default:
    break;
  }
  if (isTensixMacroIntrinsic(II.getIntrinsicID()))
    return verifyTensixMacroIntrinsicFields(II);
  if (isTensixContentsGroupIntrinsic(II.getIntrinsicID())) {
    return verifyTensixContentsGroupIntrinsicFields(II, Wormhole);
  }
  SmallVector<std::optional<int64_t>, 6> Arguments;
  for (const Value *Argument : II.args()) {
    const auto *CI = dyn_cast<ConstantInt>(Argument);
    Arguments.push_back(CI ? std::optional<int64_t>(CI->getSExtValue())
                           : std::nullopt);
  }
  return verifyContentsInstructionFields(II.getIntrinsicID(), Arguments,
                                         Wormhole);
}

SDValue llvm::lowerTensixContentsIntrinsic(SDValue Op, SelectionDAG &DAG,
                                           const RISCVSubtarget &ST) {
  if (SDValue Macro = lowerTensixMacroIntrinsic(Op, DAG, ST))
    return Macro;
  if (SDValue Group = lowerTensixContentsGroupIntrinsic(Op, DAG, ST))
    return Group;
  auto ID = static_cast<Intrinsic::ID>(Op.getConstantOperandVal(1));
  if (!isTensixContentsIntrinsic(ID))
    return SDValue();
  if (!ST.hasVendorXTTTensixBH() && !ST.hasVendorXTTTensixWH())
    reportFatalUsageError(
        "contents SFPU intrinsic requires +xtttensixbh or +xtttensixwh");
  Attribute Executor =
      DAG.getMachineFunction().getFunction().getFnAttribute("tensix-executor");
  if (!Executor.isStringAttribute() || Executor.getValueAsString() != "trisc1")
    reportFatalUsageError(
        "contents SFPU execution requires tensix-executor=trisc1");

  SDLoc DL(Op);
  auto Immediate = [&](unsigned Index) {
    return DAG.getTargetConstant(Op.getConstantOperandVal(Index), DL, MVT::i32);
  };
  SmallVector<SDValue, 6> Operands;
  unsigned Opcode = 0;
  switch (ID) {
  case Intrinsic::riscv_tt_contents_replay_template_mop: {
    // The ordinary slot emitter owns both GPR scratch definitions and MMIO.
    auto *Node =
        DAG.getMachineNode(RISCV::PseudoTTReplayTemplateMop, DL,
                           DAG.getVTList(MVT::i32, MVT::i32, MVT::Other),
                           {Immediate(2), Immediate(3), Op.getOperand(0)});
    attachIssueMemory(Node, DAG, MachineMemOperand::MOStore);
    return SDValue(Node, 2);
  }
  case Intrinsic::riscv_tt_contents_replay_template_mop_execute:
    if (const auto *Info = RISCV::getTensixContentsMOPExecuteInfoByCount(
            Op.getConstantOperandVal(3)))
      Opcode = Info->Opcode;
    else
      reportFatalUsageError("contents MOP execute requires 1..8 mutable roles");
    Operands.append({Immediate(2), Immediate(3)});
    for (unsigned I = 4, E = Op.getNumOperands(); I != E; ++I)
      Operands.push_back(Op.getOperand(I));
    break;
  case Intrinsic::riscv_tt_contents_replay_template_begin:
    Opcode = RISCV::PseudoTTContentsReplayTemplateBegin;
    Operands.append(
        {Immediate(2), Immediate(3), Immediate(4), Immediate(5), Immediate(6)});
    for (unsigned I = 7, E = Op.getNumOperands(); I != E; ++I)
      Operands.push_back(Op.getOperand(I));
    break;
  case Intrinsic::riscv_tt_contents_replay_template_end:
    Opcode = RISCV::PseudoTTContentsReplayTemplateEnd;
    Operands.push_back(Immediate(2));
    for (unsigned I = 3, E = Op.getNumOperands(); I != E; ++I)
      Operands.push_back(Op.getOperand(I));
    break;
  case Intrinsic::riscv_tt_contents_replay_template_execute:
    if (const auto *Info = RISCV::getTensixContentsReplayExecuteInfoByCount(
            Op.getConstantOperandVal(3)))
      Opcode = Info->Opcode;
    else
      reportFatalUsageError(
          "contents replay execute requires 1..8 mutable roles");
    Operands.append({Immediate(2), Immediate(3)});
    for (unsigned I = 4, E = Op.getNumOperands(); I != E; ++I)
      Operands.push_back(Op.getOperand(I));
    break;
  case Intrinsic::riscv_tt_contents_replay_template_record_iadd:
    Opcode = RISCV::PseudoTTContentsReplayTemplateRecordIAdd;
    Operands.append({Immediate(2), Immediate(3), Immediate(4)});
    break;
  case Intrinsic::riscv_tt_contents_replay_template_record_and:
    Opcode = RISCV::PseudoTTContentsReplayTemplateRecordAnd;
    Operands.append({Immediate(2), Immediate(3), Immediate(4)});
    break;
  case Intrinsic::riscv_tt_contents_replay_template_record_or:
    Opcode = RISCV::PseudoTTContentsReplayTemplateRecordOr;
    Operands.append({Immediate(2), Immediate(3), Immediate(4)});
    break;
  case Intrinsic::riscv_tt_contents_replay_template_record_xor:
    Opcode = RISCV::PseudoTTContentsReplayTemplateRecordXor;
    Operands.append({Immediate(2), Immediate(3), Immediate(4)});
    break;
  case Intrinsic::riscv_tt_contents_replay_template_record_not:
    Opcode = RISCV::PseudoTTContentsReplayTemplateRecordNot;
    Operands.append({Immediate(2), Immediate(3)});
    break;
  default:
    break;
  }
  if (Opcode) {
    bool Execute =
        ID == Intrinsic::riscv_tt_contents_replay_template_execute ||
        ID == Intrinsic::riscv_tt_contents_replay_template_mop_execute;
    Operands.push_back(Op.getOperand(0));
    MachineSDNode *Node = DAG.getMachineNode(
        Opcode, DL,
        Execute ? Op.getNode()->getVTList() : DAG.getVTList(MVT::Other),
        Operands);
    if (ID == Intrinsic::riscv_tt_contents_replay_template_begin || Execute)
      attachIssueMemory(Node, DAG,
                        MachineMemOperand::MOLoad | MachineMemOperand::MOStore);
    return SDValue(Node, 0);
  }
  bool FullLoad = ID == Intrinsic::riscv_tt_contents_load_full;
  bool Load = FullLoad || ID == Intrinsic::riscv_tt_contents_load;
  bool Store = ID == Intrinsic::riscv_tt_contents_store;
  unsigned DstOffset = FullLoad ? 2 : 3;
  bool Dynamic =
      (Load || Store) && !isa<ConstantSDNode>(Op.getOperand(DstOffset));
  switch (ID) {
  case Intrinsic::riscv_tt_contents_read_constant:
  case Intrinsic::riscv_tt_contents_read_creg:
    Opcode = RISCV::TTSFPMOVAll;
    Operands.push_back(DAG.getRegister(
        RISCV::getTensixSFPURegister(Op.getConstantOperandVal(2)),
        MVT::v32i32));
    break;
  case Intrinsic::riscv_tt_contents_add:
  case Intrinsic::riscv_tt_contents_mul:
    Opcode = ID == Intrinsic::riscv_tt_contents_add ? RISCV::TTSFPADD
                                                    : RISCV::TTSFPMUL;
    Operands.append(
        {Op.getOperand(2), Op.getOperand(3), Op.getOperand(4), Immediate(5)});
    break;
  case Intrinsic::riscv_tt_contents_load:
  case Intrinsic::riscv_tt_contents_store:
    Opcode = Load ? (Dynamic ? RISCV::PseudoTTSFPLOAD : RISCV::TTSFPLOAD)
                  : (Dynamic ? RISCV::PseudoTTSFPSTORE : RISCV::TTSFPSTORE);
    Operands.append({Op.getOperand(2),
                     Dynamic ? Op.getOperand(3) : Immediate(3), Immediate(4),
                     Immediate(5)});
    break;
  case Intrinsic::riscv_tt_contents_load_full:
    Opcode = Dynamic ? RISCV::PseudoTTSFPLOADFull : RISCV::TTSFPLOADFull;
    Operands.append({Dynamic ? Op.getOperand(2) : Immediate(2), Immediate(3),
                     Immediate(4)});
    break;
  default:
    SmallVector<std::optional<int64_t>, 6> Arguments;
    for (unsigned I = 2; I != Op.getNumOperands(); ++I) {
      const auto *Constant = dyn_cast<ConstantSDNode>(Op.getOperand(I));
      Arguments.push_back(Constant
                              ? std::optional<int64_t>(Constant->getSExtValue())
                              : std::nullopt);
    }
    auto Projection = contentsInstruction(ID, Arguments);
    if (!Projection)
      reportFatalUsageError(Twine(toString(Projection.takeError())));
    Opcode = Projection->Opcode;
    for (const ContentsOperand &Operand : Projection->Operands) {
      if (!Operand.Argument)
        Operands.push_back(
            DAG.getTargetConstant(Operand.Immediate, DL, MVT::i32));
      else if (Arguments[*Operand.Argument])
        Operands.push_back(Immediate(*Operand.Argument + 2));
      else
        Operands.push_back(Op.getOperand(*Operand.Argument + 2));
    }
    break;
  }
  Operands.push_back(Op.getOperand(0));
  SmallVector<EVT, 4> Results;
  bool ProducesContents = Op.getValueType() == MVT::v32i32;
  if (ProducesContents)
    Results.push_back(MVT::v32i32);
  if (Dynamic)
    Results.append(2, MVT::i32);
  Results.push_back(MVT::Other);
  MachineSDNode *Node =
      DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results), Operands);
  if (Load || Store) {
    auto Access = Load ? MachineMemOperand::MOLoad : MachineMemOperand::MOStore;
    if (Dynamic && Load)
      Access |= MachineMemOperand::MOStore;
    attachIssueMemory(Node, DAG, Access);
  }
  if (Dynamic && Load)
    return DAG.getMergeValues({SDValue(Node, 0), SDValue(Node, 3)}, DL);
  return SDValue(Node, ProducesContents ? 0 : Results.size() - 1);
}

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
  if (!ST.hasVendorXTTTensixBH() && !ST.hasVendorXTTTensixWH())
    reportFatalUsageError("Tensix intrinsic requires +xtttensixbh");
  // Native REPLAY supports allocated contents receivers on WH. This does not
  // admit the legacy raw intrinsic or its bound/template/MOP ingress.
  if (ST.hasVendorXTTTensixWH() &&
      (ID == Intrinsic::riscv_tt_replay || ID == Intrinsic::riscv_tt_mop ||
       (ID != Intrinsic::riscv_tt_mop_control_write &&
        ID != Intrinsic::riscv_tt_mop_clear &&
        !RISCV::isTensixWormholeIntrinsic(ID))))
    reportFatalUsageError("Tensix intrinsic is not supported on Wormhole");
  SDLoc DL(Op);
  if (Word) {
    // The issue-word fold encoded every field; only the port store remains.
    const auto *Machine =
        RISCV::getTensixMachineInfoByRawOpcode(Op.getConstantOperandVal(3));
    unsigned Port = Op.getConstantOperandVal(2);
    if (!Machine || RISCV::isTensixFieldPortOpcode(Machine->Opcode) ||
        Port > unsigned(RISCV::TensixInstructionPort::BriscToTrisc2))
      reportFatalUsageError("invalid compiler-private Tensix issue word");
    SDValue Address =
        DAG.getConstant(RISCV::getTensixInstructionPortAddress(
                            static_cast<RISCV::TensixInstructionPort>(Port)),
                        DL, MVT::i32);
    MachineSDNode *Node =
        DAG.getMachineNode(Machine->PortOpcode, DL, MVT::Other,
                           {DAG.getTargetConstant(Port, DL, MVT::i32),
                            Op.getOperand(4), Address, Op.getOperand(0)});
    attachIssueMemory(Node, DAG,
                      MachineMemOperand::MOLoad | MachineMemOperand::MOStore);
    return SDValue(Node, 0);
  }
  if (Control) {
    auto *Constant = dyn_cast<ConstantSDNode>(Op.getOperand(3));
    // Machine immediates use a signed 64-bit container. Keep the i32 payload
    // zero-extended so bit 31 does not turn a legal uimm32 into a negative MI.
    SDValue Value =
        Constant ? DAG.getTargetConstant(Constant->getZExtValue(), DL, MVT::i64)
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
      Fields.push_back(
          DAG.getTargetConstant(Op.getConstantOperandVal(I), DL, MVT::i32));
    Fields.push_back(Op.getOperand(0));
    bool TemplateMop = TemplateOpcode == RISCV::PseudoTTReplayTemplateMop;
    SmallVector<EVT, 3> Results(TemplateMop ? 2 : 0, MVT::i32);
    Results.push_back(MVT::Other);
    MachineSDNode *Node =
        DAG.getMachineNode(TemplateOpcode, DL, DAG.getVTList(Results), Fields);
    // Actual recording/execution boundaries retain ordinary REPLAY's memory
    // ordering through effect normalization and final control replacement.
    // End is only an issue-order marker, not an executed memory operation.
    if (TemplateMop)
      attachIssueMemory(Node, DAG, MachineMemOperand::MOStore);
    else if (TemplateOpcode != RISCV::PseudoTTReplayTemplateEnd)
      attachIssueMemory(Node, DAG,
                        MachineMemOperand::MOLoad | MachineMemOperand::MOStore);
    return SDValue(Node, Results.size() - 1);
  }
  if (End)
    return SDValue(DAG.getMachineNode(RISCV::PseudoTTReplayRecordEnd, DL,
                                      MVT::Other, Op.getOperand(0)),
                   0);
  bool Port = Info && ID == Info->PortIntrinsicID;
  bool Mop = Clear || (Info && ID == Info->MopIntrinsicID);
  const auto *Machine =
      Info ? RISCV::getTensixMachineInfoByIntrinsic(Info->IntrinsicID)
           : nullptr;
  if (Port && !RISCV::isTensixFieldPortOpcode(Machine->Opcode))
    reportFatalUsageError(
        "Tensix instruction-port issue reached selection without its "
        "issue-word fold");
  SmallVector<SDValue, 16> Operands;
  unsigned First = Port || Mop ? 3 : 2;
  if (Port || Mop)
    Operands.push_back(
        DAG.getTargetConstant(Op.getConstantOperandVal(2), DL, MVT::i32));
  if (Info)
    for (unsigned I = 0; I != Info->NumFields; ++I) {
      SDValue Value = Op.getOperand(First + I);
      bool Dynamic = Port && Machine->Opcode != RISCV::TTREPLAY;
      Operands.push_back(
          Dynamic
              ? Value
              : DAG.getTargetConstant(
                    cast<ConstantSDNode>(Value)->getZExtValue(), DL, MVT::i32));
    }
  Operands.push_back(Op.getOperand(0));
  unsigned Scratch = Port ? 3 : Mop ? 2 : 0;
  SmallVector<EVT> Results(Scratch, MVT::i32);
  Results.push_back(MVT::Other);
  unsigned Opcode = Clear  ? unsigned(RISCV::PseudoTTMOPClear)
                    : Port ? Machine->PortOpcode
                    : Mop  ? Machine->MopOpcode
                           : Machine->Opcode;
  MachineSDNode *Node =
      DAG.getMachineNode(Opcode, DL, DAG.getVTList(Results), Operands);
  auto MemoryFlags = MachineMemOperand::MOStore;
  if (!Mop)
    MemoryFlags |= MachineMemOperand::MOLoad;
  attachIssueMemory(Node, DAG, MemoryFlags);
  return SDValue(Node, Scratch);
}
