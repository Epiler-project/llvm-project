//===-- RISCVTensixBoundContract.cpp
//---------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixBoundContract.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"

using namespace llvm;
using namespace llvm::RISCV;

namespace {
using Role = TensixBoundArgumentRole;
using Access = TensixSFPUAccess;
using ValueKind = TensixBoundValueKind;

// The formal ABI-to-ingress mapping owns no encoding or machine constraints.
constexpr TensixBoundInstruction Instructions[] = {
    {Intrinsic::riscv_tt_bound_sfploadi, RISCV::PseudoTTBoundSFPLOADI, 4},
    {Intrinsic::riscv_tt_bound_sfpload, RISCV::PseudoTTBoundSFPLOAD, 5, 2},
    {Intrinsic::riscv_tt_bound_sfpstore, RISCV::PseudoTTBoundSFPSTORE, 4, 1},
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

constexpr unsigned ConfigOpcodes[] = {
    RISCV::TTSFPCONFIGC11, RISCV::TTSFPCONFIGC12, RISCV::TTSFPCONFIGC13,
    RISCV::TTSFPCONFIGC14};

const MCInstrInfo &instructions() {
  static const auto Info = createTensixBoundMCInstrInfo();
  return *Info;
}
const MCRegisterInfo &registers() {
  static const auto Info = createTensixBoundMCRegisterInfo();
  return *Info;
}
Error invalid(const Twine &Message) {
  return createStringError(Twine("bound SFPU ") + Message);
}

std::optional<uint32_t> registerNumber(MCRegister Register) {
  const auto &MRI = registers();
  if (!MRI.getRegClass(RISCV::SFPRReadRegClassID).contains(Register))
    return std::nullopt;
  return MRI.getEncodingValue(Register);
}
SmallVector<MCRegister, 8> inClass(ArrayRef<MCPhysReg> Regs, unsigned ClassID) {
  SmallVector<MCRegister, 8> Result;
  for (MCPhysReg Reg : Regs)
    if (registers().getRegClass(ClassID).contains(Reg))
      Result.push_back(Reg);
  return Result;
}
Expected<MCRegister> configRegister(unsigned Opcode) {
  auto Defs = inClass(instructions().get(Opcode).implicit_defs(),
                      RISCV::SFPCRRegClassID);
  if (Defs.size() != 1 ||
      !is_contained(instructions().get(Opcode).implicit_uses(), Defs[0]))
    return invalid("configuration descriptor lacks its unique read/write CReg");
  return Defs[0];
}

struct FixedBinding {
  unsigned Argument;
  MCRegister Register;
  Role ArgumentRole;
  Access Effect;
};
struct RegisterAlias {
  unsigned Argument;
  unsigned NativeOperand;
};
struct NativeMapping {
  TensixBoundNativeInstruction Native;
  SmallVector<std::optional<Role>, 12> RoleOverrides;
  SmallVector<FixedBinding, 12> Fixed;
  SmallVector<RegisterAlias, 2> Aliases;
};

// One mapping drives native emission and the public descriptor projection.
// ABI-only selectors, omitted fields and vector-field aliases live here;
// native classes, ties, effects and logical field legality stay
// descriptor-owned.
Expected<NativeMapping> mapInstruction(const TensixBoundInstruction &Info,
                                       ArrayRef<TensixBoundValue> Args) {
  if (Args.size() != Info.ArgumentCount)
    return invalid("argument tuple has the wrong size");
  NativeMapping Map;
  Map.RoleOverrides.resize(Info.ArgumentCount);
  if (Info.DstOffset)
    Map.RoleOverrides[*Info.DstOffset] = Role::DstOffset;
  auto Select = [&](unsigned Opcode, std::initializer_list<unsigned> Indices) {
    Map.Native.Opcode = Opcode;
    for (unsigned Index : Indices)
      Map.Native.Operands.push_back({Index, 0});
  };
  auto Old = [&] { Map.RoleOverrides[1] = Role::OldDestination; };
  auto Mode = [&](unsigned Index,
                  std::initializer_list<int64_t> Values) -> Error {
    if (Args[Index].Kind != ValueKind::Constant ||
        !is_contained(Values, Args[Index].Constant))
      return invalid("unsupported mode at operand " + Twine(Index));
    return Error::success();
  };
  auto Zero = [&](unsigned Index) -> Error {
    if (Args[Index].Kind != ValueKind::Constant || Args[Index].Constant != 0)
      return invalid("logical field operand " + Twine(Index) +
                     " must be in [0, 0]");
    return Error::success();
  };
  switch (Info.Pseudo) {
  case RISCV::PseudoTTBoundSFPLOADI:
    Select(RISCV::TTSFPLOADI, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPLOAD:
    Select(RISCV::TTSFPLOAD, {0, 1, 2, 3, 4});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPSTORE:
    Select(RISCV::TTSFPSTORE, {0, 1, 2, 3});
    break;
  case RISCV::PseudoTTBoundSFPADD:
    Select(RISCV::TTSFPADD, {0, 1, 2, 3, 4});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPMUL:
    Select(RISCV::TTSFPMUL, {0, 1, 2, 3, 4});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPMAD:
    Select(RISCV::TTSFPMAD, {0, 1, 2, 3, 4, 5});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPIADD: {
    constexpr unsigned Opcodes[] = {RISCV::TTSFPIADDCCLT, RISCV::TTSFPISUBCCLT,
                                    RISCV::TTSFPIADD,     RISCV::TTSFPISUB,
                                    RISCV::TTSFPIADDCCGE, RISCV::TTSFPISUBCCGE};
    if (Error E = Mode(3, {0, 2, 4, 6, 8, 10}))
      return std::move(E);
    Select(Opcodes[Args[3].Constant / 2], {0, 1, 2});
    Old();
    break;
  }
  case RISCV::PseudoTTBoundSFPMOV:
    if (Error E = Mode(3, {0, 1}))
      return std::move(E);
    Select(Args[3].Constant == 0 ? RISCV::TTSFPMOV : RISCV::TTSFPMOVNeg,
           {0, 1, 2});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPMOVAll:
    Select(RISCV::TTSFPMOVAll, {0, 1});
    break;
  case RISCV::PseudoTTBoundSFPARECIP:
    Select(RISCV::TTSFPARECIP, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPEXEXP:
    Select(RISCV::TTSFPEXEXP, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPEXMAN:
    Select(RISCV::TTSFPEXMAN, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPABS:
    Select(RISCV::TTSFPABS, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPLZ:
    Select(RISCV::TTSFPLZ, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPCAST:
    Select(RISCV::TTSFPCAST, {0, 1, 2, 3});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPSETEXPI:
  case RISCV::PseudoTTBoundSFPSETMANI:
  case RISCV::PseudoTTBoundSFPSETSGNI: {
    bool Exp = Info.Pseudo == RISCV::PseudoTTBoundSFPSETEXPI;
    if (Error E = Exp ? Mode(4, {1, 3}) : Mode(4, {1}))
      return std::move(E);
    Select(Exp                                             ? RISCV::TTSFPSETEXP
           : Info.Pseudo == RISCV::PseudoTTBoundSFPSETMANI ? RISCV::TTSFPSETMAN
                                                           : RISCV::TTSFPSETSGN,
           {0, 1, 2, 3, 4});
    Old();
    break;
  }
  case RISCV::PseudoTTBoundSFPSETEXPV:
  case RISCV::PseudoTTBoundSFPSETMANV:
  case RISCV::PseudoTTBoundSFPSETSGNV: {
    bool Exp = Info.Pseudo == RISCV::PseudoTTBoundSFPSETEXPV;
    if (Error E = Exp ? Mode(4, {0, 2}) : Mode(4, {0}))
      return std::move(E);
    Select(Exp                                             ? RISCV::TTSFPSETEXP
           : Info.Pseudo == RISCV::PseudoTTBoundSFPSETMANV ? RISCV::TTSFPSETMAN
                                                           : RISCV::TTSFPSETSGN,
           {0, 1, 2});
    Map.Native.Operands.push_back({std::nullopt, 0});
    Map.Native.Operands.push_back({4, 0});
    // The field is the same physical input as the native old destination.
    // Projecting its descriptor tie requires dst == old == field, with no move.
    Map.Aliases.push_back({3, 1});
    Old();
    break;
  }
  case RISCV::PseudoTTBoundSFPIADDI:
  case RISCV::PseudoTTBoundSFPSHFTI:
    if (Error E = Info.Pseudo == RISCV::PseudoTTBoundSFPIADDI
                      ? Mode(3, {1, 5, 9})
                      : Mode(3, {1, 3, 5, 7}))
      return std::move(E);
    Select(Info.Pseudo == RISCV::PseudoTTBoundSFPIADDI ? RISCV::TTSFPIADDI
                                                       : RISCV::TTSFPSHFT,
           {0, 1, 1, 2, 3});
    break;
  case RISCV::PseudoTTBoundSFPSHFTV:
    if (Error E = Mode(3, {0, 2}))
      return std::move(E);
    Select(RISCV::TTSFPSHFT, {0, 1, 2});
    Map.Native.Operands.push_back({std::nullopt, 0});
    Map.Native.Operands.push_back({3, 0});
    break;
  case RISCV::PseudoTTBoundSFPAND:
    Select(RISCV::TTSFPAND, {0, 1, 2});
    break;
  case RISCV::PseudoTTBoundSFPOR:
    Select(RISCV::TTSFPOR, {0, 1, 2});
    break;
  case RISCV::PseudoTTBoundSFPXOR:
    Select(RISCV::TTSFPXOR, {0, 1, 2});
    break;
  case RISCV::PseudoTTBoundSFPNOT:
    Select(RISCV::TTSFPNOT, {0, 1, 1});
    break;
  case RISCV::PseudoTTBoundSFPSHFT2:
    if (Error E = Zero(3))
      return std::move(E);
    Select(RISCV::TTSFPSHFT2, {0, 1, 2, 4});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPSTOCHRNDI: {
    if (Error E = Mode(4, {0, 1, 2, 3, 4, 5, 6, 7}))
      return std::move(E);
    Select(RISCV::TTSFPSTOCHRNDI, {0, 1, 2, 3});
    int64_t Mode = Args[4].Constant;
    if (Mode == 4 || Mode == 5)
      Mode |= 8;
    Map.Native.Operands.push_back({std::nullopt, Mode});
    Map.Native.Operands.push_back({5, 0});
    Old();
    break;
  }
  case RISCV::PseudoTTBoundSFPSTOCHRNDV:
    Select(RISCV::TTSFPSTOCHRNDV, {0, 1, 2, 3, 4, 5});
    Old();
    break;
  case RISCV::PseudoTTBoundSFPLUT: {
    Select(RISCV::TTSFPLUT, {0, 1, 6});
    auto Inputs = inClass(instructions().get(Map.Native.Opcode).implicit_uses(),
                          RISCV::SFPRRegClassID);
    if (Inputs.size() != 4)
      return invalid("LUT descriptor differs from its fixed-group ABI");
    for (auto [I, Reg] : enumerate(Inputs))
      Map.Fixed.push_back(
          {unsigned(I + 2), Reg, Role::FixedGroupRead, Access::Read});
    Old();
    break;
  }
  case RISCV::PseudoTTBoundSFPSWAP:
    Select(RISCV::TTSFPSWAP, {0, 1, 2, 3, 4});
    break;
  case RISCV::PseudoTTBoundSFPTRANSP: {
    Select(RISCV::TTSFPTRANSP, {});
    const auto &Desc = instructions().get(Map.Native.Opcode);
    auto Inputs = inClass(Desc.implicit_uses(), RISCV::SFPRRegClassID);
    auto Defs = inClass(Desc.implicit_defs(), RISCV::SFPRRegClassID);
    SmallVector<MCRegister, 4> Outputs, Clobbers;
    for (MCRegister Reg : Defs)
      (is_contained(Inputs, Reg) ? Outputs : Clobbers).push_back(Reg);
    if (Inputs.size() != 4 || Outputs.size() != 4 || Clobbers.size() != 4)
      return invalid("transpose descriptor differs from its fixed-group ABI");
    for (unsigned I = 0; I != 4; ++I) {
      Map.Fixed.push_back(
          {I, Outputs[I], Role::FixedGroupWrite, Access::Write});
      Map.Fixed.push_back(
          {I + 4, Clobbers[I], Role::FixedGroupClobber, Access::Clobber});
      Map.Fixed.push_back(
          {I + 8, Inputs[I], Role::FixedGroupRead, Access::Read});
    }
    break;
  }
  case RISCV::PseudoTTBoundSFPSETCC: {
    constexpr unsigned Opcodes[] = {RISCV::TTSFPSETCCLT, RISCV::TTSFPSETCCNE,
                                    RISCV::TTSFPSETCCGE, RISCV::TTSFPSETCCEQ};
    if (Error E = Zero(1))
      return std::move(E);
    if (Error E = Mode(2, {0, 2, 4, 6}))
      return std::move(E);
    Select(Opcodes[Args[2].Constant / 2], {0});
    break;
  }
  case RISCV::PseudoTTBoundSFPENCC:
    Select(RISCV::TTSFPENCC, {0, 1});
    break;
  case RISCV::PseudoTTBoundSFPPUSHC:
  case RISCV::PseudoTTBoundSFPPOPC:
    if (Error E = Zero(0))
      return std::move(E);
    if (Error E = Zero(1))
      return std::move(E);
    Select(Info.Pseudo == RISCV::PseudoTTBoundSFPPUSHC ? RISCV::TTSFPPUSHC
                                                       : RISCV::TTSFPPOPC,
           {});
    break;
  case RISCV::PseudoTTBoundSFPCONFIGCReg: {
    if (Args[1].Kind != ValueKind::Constant)
      return invalid("configuration native selection requires a bound CReg");
    bool Found = false;
    for (unsigned Opcode : ConfigOpcodes) {
      auto CReg = configRegister(Opcode);
      if (!CReg)
        return CReg.takeError();
      if (Args[1].Constant != *registerNumber(*CReg))
        continue;
      Select(Opcode, {2, 3});
      auto Stage = inClass(instructions().get(Opcode).implicit_uses(),
                           RISCV::SFPRRegClassID);
      if (Stage.size() != 1)
        return invalid("configuration descriptor lacks its unique stage input");
      Map.Fixed.push_back({0, Stage[0], Role::FixedGroupRead, Access::Read});
      Map.Fixed.push_back({1, *CReg, Role::WriteCReg, Access::ReadWrite});
      Found = true;
      break;
    }
    if (!Found)
      return invalid("invalid fixed writable CReg operand");
    break;
  }
  case RISCV::PseudoTTBoundSFPNOP:
    Select(RISCV::TTSFPNOP, {});
    break;
  case RISCV::PseudoTTBoundSFPCOMPC:
    Select(RISCV::TTSFPCOMPC, {});
    break;
  case RISCV::PseudoTTBoundSFPCONFIGReset:
    Select(RISCV::TTSFPCONFIGReset, {});
    break;
  default:
    return invalid("unknown native instruction mapping");
  }
  return Map;
}

bool sameResource(const TensixSFPUResource &First,
                  const TensixSFPUResource &Second) {
  if (const auto *Argument = std::get_if<TensixBoundArgumentRef>(&First)) {
    const auto *Other = std::get_if<TensixBoundArgumentRef>(&Second);
    return Other && Argument->Argument == Other->Argument;
  }
  if (const auto *Register = std::get_if<TensixFixedRegisterRef>(&First)) {
    const auto *Other = std::get_if<TensixFixedRegisterRef>(&Second);
    return Other && Register->Number == Other->Number;
  }
  const auto *Other = std::get_if<TensixSFPUState>(&Second);
  return Other && *Other == std::get<TensixSFPUState>(First);
}
void addEffect(SmallVectorImpl<TensixSFPUArchitecturalEffect> &Effects,
               TensixSFPUResource Resource, Access Mode) {
  auto Found = find_if(Effects, [&](const auto &Effect) {
    return sameResource(Effect.Resource, Resource);
  });
  if (Found == Effects.end()) {
    Effects.push_back({Resource, Mode});
    return;
  }
  if (Found->Access != Mode)
    Found->Access = Access::ReadWrite;
}
Expected<TensixSFPUResource> architecturalResource(MCRegister Register) {
  if (auto Number = registerNumber(Register))
    return TensixFixedRegisterRef{*Number};
  switch (Register.id()) {
  case RISCV::TT_CC:
    return TensixSFPUState::CC;
  case RISCV::TT_CCSTACK:
    return TensixSFPUState::CCStack;
  case RISCV::TT_CONFIG:
    return TensixSFPUState::Configuration;
  case RISCV::TT_DST:
    return TensixSFPUState::Dst;
  case RISCV::TT_ISSUE:
    return TensixSFPUState::Issue;
  default:
    return invalid("native descriptor names an unknown architectural effect");
  }
}

bool sameConstraints(ArrayRef<TensixBoundRegisterConstraint> First,
                     ArrayRef<TensixBoundRegisterConstraint> Second) {
  return First.size() == Second.size() &&
         equal(First, Second, [](const auto &A, const auto &B) {
           return A.Kind == B.Kind && A.FirstArgument == B.FirstArgument &&
                  A.SecondArgument == B.SecondArgument;
         });
}
bool sameEffects(ArrayRef<TensixSFPUArchitecturalEffect> First,
                 ArrayRef<TensixSFPUArchitecturalEffect> Second) {
  return First.size() == Second.size() &&
         equal(First, Second, [](const auto &A, const auto &B) {
           return sameResource(A.Resource, B.Resource) && A.Access == B.Access;
         });
}
} // namespace

const TensixBoundInstruction *
RISCV::getTensixBoundInstruction(Intrinsic::ID ID) {
  auto It =
      find_if(Instructions, [ID](const auto &Info) { return Info.ID == ID; });
  return It == std::end(Instructions) ? nullptr : It;
}
const TensixBoundInstruction *
RISCV::getTensixBoundInstructionForPseudo(unsigned Opcode) {
  if (Opcode == RISCV::PseudoTTBoundSFPLOADDynamic)
    Opcode = RISCV::PseudoTTBoundSFPLOAD;
  if (Opcode == RISCV::PseudoTTBoundSFPSTOREDynamic)
    Opcode = RISCV::PseudoTTBoundSFPSTORE;
  auto It = find_if(Instructions, [Opcode](const auto &Info) {
    return Info.Pseudo == Opcode;
  });
  return It == std::end(Instructions) ? nullptr : It;
}

MCRegister RISCV::getTensixSFPURegister(uint32_t Number) {
  for (MCRegister Reg : registers().getRegClass(RISCV::SFPRReadRegClassID))
    if (registerNumber(Reg) == Number)
      return Reg;
  llvm_unreachable("verified architectural SFPU register");
}

Expected<TensixBoundSFPUContract>
RISCV::getTensixBoundSFPUContract(Intrinsic::ID ID,
                                  ArrayRef<TensixBoundValue> Args) {
  const auto *Info = getTensixBoundInstruction(ID);
  if (!Info)
    return invalid("unknown intrinsic");
  if (Args.size() != Info->ArgumentCount)
    return invalid("argument tuple has the wrong size");

  // An unbound CONFIG destination is a symbolic parameter over all actual
  // descriptors. Check their complete projected structure before returning a
  // candidate set; no representative CReg becomes a provisional assignment.
  if (ID == Intrinsic::riscv_tt_bound_sfpconfig_creg &&
      Args[1].Kind == ValueKind::UnboundRegister) {
    std::optional<TensixBoundSFPUContract> Common;
    SmallVector<uint32_t, 16> Candidates;
    for (unsigned Opcode : ConfigOpcodes) {
      auto Register = configRegister(Opcode);
      if (!Register)
        return Register.takeError();
      uint32_t Number = *registerNumber(*Register);
      SmallVector<TensixBoundValue, 4> Bound(Args);
      Bound[1] = {ValueKind::Constant, Number};
      auto Current = getTensixBoundSFPUContract(ID, Bound);
      if (!Current)
        return Current.takeError();
      auto Destination = find_if(Current->Registers, [](const auto &Reg) {
        return Reg.Argument == 1;
      });
      if (Destination == Current->Registers.end() ||
          Destination->AllowedRegisters != SmallVector<uint32_t, 16>{Number} ||
          Destination->FixedRegister != Number)
        return invalid("configuration CReg descriptor has an unknown shape");
      Destination->AllowedRegisters.clear();
      Destination->FixedRegister.reset();
      if (!Common) {
        Common = std::move(*Current);
      } else {
        bool SameRegisters =
            Common->Registers.size() == Current->Registers.size() &&
            equal(Common->Registers, Current->Registers,
                  [](const auto &A, const auto &B) {
                    return A.Argument == B.Argument && A.Role == B.Role &&
                           A.AllowedRegisters == B.AllowedRegisters &&
                           A.FixedRegister == B.FixedRegister;
                  });
        if (Common->Roles != Current->Roles || !SameRegisters ||
            !sameConstraints(Common->Constraints, Current->Constraints) ||
            !sameEffects(Common->Effects, Current->Effects))
          return invalid(
              "configuration descriptors have incompatible contracts");
      }
      Candidates.push_back(Number);
    }
    if (!Common)
      return invalid("configuration has no native descriptor");
    auto Destination = find_if(
        Common->Registers, [](const auto &Reg) { return Reg.Argument == 1; });
    Destination->AllowedRegisters = std::move(Candidates);
    return std::move(*Common);
  }

  auto Mapping = mapInstruction(*Info, Args);
  if (!Mapping)
    return Mapping.takeError();
  const auto &Native = Mapping->Native;
  const auto &Desc = instructions().get(Native.Opcode);
  if (Native.Operands.size() != Desc.getNumOperands())
    return invalid("native operand projection differs from its descriptor");
  TensixBoundSFPUContract Contract;
  Contract.ID = ID;
  Contract.Roles.assign(Args.size(), Role::LogicalImmediate);

  auto FindRegister = [&](unsigned Argument) -> TensixBoundRegisterArgument * {
    auto It = find_if(Contract.Registers, [Argument](const auto &Reg) {
      return Reg.Argument == Argument;
    });
    return It == Contract.Registers.end() ? nullptr : &*It;
  };
  auto AddRegister =
      [&](unsigned Argument, Role ArgumentRole, ArrayRef<uint32_t> Allowed,
          std::optional<uint32_t> Fixed = std::nullopt) -> Error {
    if (Argument >= Args.size() || Allowed.empty())
      return invalid("unknown native register operand contract");
    if (auto *Existing = FindRegister(Argument)) {
      erase_if(Existing->AllowedRegisters,
               [&](uint32_t Reg) { return !is_contained(Allowed, Reg); });
      if (Existing->AllowedRegisters.empty() ||
          (Fixed && Existing->FixedRegister &&
           Fixed != Existing->FixedRegister))
        return invalid("native operand has incompatible register classes");
      if (Fixed)
        Existing->FixedRegister = Fixed;
      // Multiple native uses may narrow SFPRRead to SFPR (in-place forms).
      if (ArgumentRole == Role::ReadLReg &&
          Existing->Role == Role::ReadRegister)
        Existing->Role = ArgumentRole;
      return Error::success();
    }
    Contract.Registers.push_back(
        {Argument, ArgumentRole, SmallVector<uint32_t, 16>(Allowed), Fixed});
    return Error::success();
  };
  auto ProjectOperand = [&](unsigned Argument, unsigned Operand) -> Error {
    const auto &OperandInfo = Desc.operands()[Operand];
    if (OperandInfo.RegClass < 0)
      return invalid("register projection refers to a logical field");
    SmallVector<uint32_t, 16> Allowed;
    bool HasCReg = false;
    for (MCRegister Reg : registers().getRegClass(OperandInfo.RegClass)) {
      auto Number = registerNumber(Reg);
      if (!Number)
        return invalid("native SFPU operand has a non-SFPU register class");
      Allowed.push_back(*Number);
      HasCReg |= registers().getRegClass(RISCV::SFPCRRegClassID).contains(Reg);
    }
    bool Write = Operand < Desc.getNumDefs();
    Role ArgumentRole = Write     ? Role::WriteLReg
                        : HasCReg ? Role::ReadRegister
                                  : Role::ReadLReg;
    if (Mapping->RoleOverrides[Argument])
      ArgumentRole = *Mapping->RoleOverrides[Argument];
    if (Error E = AddRegister(Argument, ArgumentRole, Allowed))
      return E;
    addEffect(Contract.Effects, TensixBoundArgumentRef{Argument},
              Write ? Access::Write : Access::Read);
    return Error::success();
  };
  for (auto [I, Operand] : enumerate(Native.Operands)) {
    if (Desc.operands()[I].RegClass < 0)
      continue;
    if (!Operand.Argument)
      return invalid("native register lacks an explicit ABI argument");
    if (Error E = ProjectOperand(*Operand.Argument, I))
      return std::move(E);
  }
  for (const RegisterAlias &Alias : Mapping->Aliases)
    if (Error E = ProjectOperand(Alias.Argument, Alias.NativeOperand))
      return std::move(E);
  for (const FixedBinding &Binding : Mapping->Fixed) {
    auto Number = registerNumber(Binding.Register);
    if (!Number)
      return invalid("fixed group has an unknown register");
    if (Error E = AddRegister(Binding.Argument, Binding.ArgumentRole, {*Number},
                              *Number))
      return std::move(E);
  }
  sort(Contract.Registers,
       [](const auto &A, const auto &B) { return A.Argument < B.Argument; });
  for (const auto &Register : Contract.Registers)
    Contract.Roles[Register.Argument] = Register.Role;
  for (auto [I, Override] : enumerate(Mapping->RoleOverrides))
    if (Override)
      Contract.Roles[I] = *Override;

  auto ArgumentsForOperand = [&](unsigned Operand) {
    SmallVector<unsigned, 2> Result;
    if (Native.Operands[Operand].Argument)
      Result.push_back(*Native.Operands[Operand].Argument);
    for (const RegisterAlias &Alias : Mapping->Aliases)
      if (Alias.NativeOperand == Operand)
        Result.push_back(Alias.Argument);
    return Result;
  };
  auto Constraint = [&](TensixBoundConstraintKind Kind, unsigned First,
                        unsigned Second) {
    if (First == Second)
      return;
    if (Kind == TensixBoundConstraintKind::SameLocation && Second < First)
      std::swap(First, Second);
    if (!any_of(Contract.Constraints, [&](const auto &Existing) {
          return Existing.Kind == Kind && Existing.FirstArgument == First &&
                 Existing.SecondArgument == Second;
        }))
      Contract.Constraints.push_back({Kind, First, Second});
  };
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I) {
    if (Desc.operands()[I].RegClass < 0)
      continue;
    auto Arguments = ArgumentsForOperand(I);
    for (unsigned Other : Arguments)
      Constraint(TensixBoundConstraintKind::SameLocation, Arguments[0], Other);
    int Tied = Desc.getOperandConstraint(I, MCOI::TIED_TO);
    if (Tied >= 0)
      for (unsigned First : Arguments)
        for (unsigned Second : ArgumentsForOperand(Tied))
          Constraint(TensixBoundConstraintKind::SameLocation, First, Second);
    if (Desc.getOperandConstraint(I, MCOI::EARLY_CLOBBER) < 0)
      continue;
    for (unsigned J = 0; J != Desc.getNumOperands(); ++J) {
      if (J == I || Desc.operands()[J].RegClass < 0 || int(J) == Tied ||
          Desc.getOperandConstraint(J, MCOI::TIED_TO) == int(I))
        continue;
      for (unsigned First : Arguments)
        for (unsigned Second : ArgumentsForOperand(J))
          Constraint(TensixBoundConstraintKind::EarlyClobber, First, Second);
    }
  }

  // A descriptor effect mapped to an explicit fixed-group ABI argument stays
  // attached to that argument. Remaining hardware effects retain their fixed
  // register/state identities; no fake source contents are manufactured.
  auto ImplicitEffect = [&](MCRegister Register, bool Write) -> Error {
    bool Mapped = false;
    for (const FixedBinding &Binding : Mapping->Fixed) {
      if (Binding.Register != Register)
        continue;
      bool Matches = Write ? Binding.Effect != Access::Read
                           : Binding.Effect == Access::Read ||
                                 Binding.Effect == Access::ReadWrite;
      if (!Matches)
        continue;
      addEffect(Contract.Effects, TensixBoundArgumentRef{Binding.Argument},
                Write ? (Binding.Effect == Access::Clobber ? Access::Clobber
                                                           : Access::Write)
                      : Access::Read);
      Mapped = true;
    }
    if (Mapped)
      return Error::success();
    if (registers().getRegClass(RISCV::SFPRRegClassID).contains(Register))
      return invalid(
          "native LReg effect lacks an explicit fixed-group argument");
    auto Resource = architecturalResource(Register);
    if (!Resource)
      return Resource.takeError();
    addEffect(Contract.Effects, *Resource,
              Write ? Access::Write : Access::Read);
    return Error::success();
  };
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (Error E = ImplicitEffect(Reg, false))
      return std::move(E);
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (Error E = ImplicitEffect(Reg, true))
      return std::move(E);

  for (auto [I, Argument] : enumerate(Args)) {
    const auto *Register = FindRegister(I);
    if (Register) {
      if (Argument.Kind == ValueKind::UnboundRegister)
        continue;
      if (Argument.Kind != ValueKind::Constant)
        return invalid(
            "physical operands and logical immediates must be constant");
      if (Argument.Constant < 0 || Argument.Constant > UINT32_MAX ||
          !is_contained(Register->AllowedRegisters,
                        uint32_t(Argument.Constant))) {
        if (Register->FixedRegister)
          return invalid("fixed register group does not match the instruction");
        return invalid(Register->Role == Role::WriteLReg
                           ? "invalid writable LReg operand"
                           : "invalid read register operand class");
      }
      continue;
    }
    if (Argument.Kind == ValueKind::Constant)
      continue;
    if (Argument.Kind != ValueKind::DynamicScalar ||
        Contract.Roles[I] != Role::DstOffset)
      return invalid(
          "physical operands and logical immediates must be constant");
  }
  for (const auto &Relation : Contract.Constraints) {
    const auto &First = Args[Relation.FirstArgument];
    const auto &Second = Args[Relation.SecondArgument];
    if (First.Kind == ValueKind::UnboundRegister ||
        Second.Kind == ValueKind::UnboundRegister)
      continue;
    bool Same = First.Constant == Second.Constant;
    if (Relation.Kind == TensixBoundConstraintKind::SameLocation) {
      if (!Same)
        return invalid("destructive tie requires identical physical registers");
    } else {
      auto Overlap =
          tensixSFPURegistersOverlap(First.Constant, Second.Constant);
      if (!Overlap)
        return Overlap.takeError();
      if (*Overlap)
        return invalid("early-clobber or distinct operands overlap");
    }
  }

  MCInst Fields;
  Fields.setOpcode(Native.Opcode);
  for (auto [I, Operand] : enumerate(Native.Operands)) {
    if (Desc.operands()[I].RegClass >= 0) {
      // Absent register slot, not a selected physical or virtual placeholder.
      Fields.addOperand(MCOperand());
      continue;
    }
    int64_t Value = Operand.Constant;
    if (Operand.Argument) {
      const auto &Argument = Args[*Operand.Argument];
      // This query checks the static tuple only. Dynamic offset definedness
      // and range remain independent ingress/program obligations.
      Value = Argument.Kind == ValueKind::DynamicScalar ? 0 : Argument.Constant;
    }
    Fields.addOperand(MCOperand::createImm(Value));
  }
  if (Error E = verifyTensixMCInstructionFields(Fields, instructions()))
    return std::move(E);
  return Contract;
}

Error RISCV::verifyTensixBoundSFPUOperands(
    Intrinsic::ID ID, ArrayRef<TensixBoundValue> Arguments) {
  for (const auto &Argument : Arguments)
    if (Argument.Kind == ValueKind::UnboundRegister)
      return invalid(
          "physical operands and logical immediates must be constant");
  auto Contract = getTensixBoundSFPUContract(ID, Arguments);
  if (!Contract)
    return Contract.takeError();
  return Error::success();
}

Expected<TensixBoundNativeInstruction>
RISCV::getTensixBoundNativeInstruction(Intrinsic::ID ID,
                                       ArrayRef<TensixBoundValue> Arguments) {
  if (Error E = verifyTensixBoundSFPUOperands(ID, Arguments))
    return std::move(E);
  auto Mapping = mapInstruction(*getTensixBoundInstruction(ID), Arguments);
  if (!Mapping)
    return Mapping.takeError();
  return std::move(Mapping->Native);
}

Expected<bool> RISCV::tensixSFPURegistersOverlap(uint32_t First,
                                                 uint32_t Second) {
  const auto &MRI = registers();
  MCRegister FirstRegister, SecondRegister;
  for (MCRegister Reg : MRI.getRegClass(RISCV::SFPRReadRegClassID)) {
    if (registerNumber(Reg) == First)
      FirstRegister = Reg;
    if (registerNumber(Reg) == Second)
      SecondRegister = Reg;
  }
  if (!FirstRegister || !SecondRegister)
    return invalid("invalid architectural register number");
  return MRI.regsOverlap(FirstRegister, SecondRegister);
}
