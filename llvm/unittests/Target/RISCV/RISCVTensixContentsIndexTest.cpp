//===-- RISCVTensixContentsIndexTest.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVInstrInfo.h"
#include "RISCVRegisterContents.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "RISCVTensixContentsGroup.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace llvm;

namespace {

// Build MachineInstrs directly to exercise the target receiver independently
// of the generic MIR verifier, retaining both checks for malformed leaf roles.
class RISCVTensixContentsIndexTest
    : public testing::TestWithParam<const char *> {
protected:
  std::unique_ptr<RISCVTargetMachine> TM;
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<RISCVSubtarget> ST;
  std::unique_ptr<MachineFunction> MF;
  MachineBasicBlock *Block = nullptr;

  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
  }

  void SetUp() override {
    Triple TT("riscv32-unknown-elf");
    std::string Details;
    const Target *Target = TargetRegistry::lookupTarget(TT, Details);
    ASSERT_NE(Target, nullptr) << Details;
    TM.reset(static_cast<RISCVTargetMachine *>(Target->createTargetMachine(
        TT, "generic-rv32", GetParam(), TargetOptions(), std::nullopt,
        std::nullopt, CodeGenOptLevel::Default)));
    ASSERT_NE(TM, nullptr);
    M = std::make_unique<Module>("index-native-receiver", Context);
    M->setTargetTriple(TM->getTargetTriple());
    M->setDataLayout(TM->createDataLayout());
    auto *FType = FunctionType::get(Type::getVoidTy(Context), false);
    auto *F =
        Function::Create(FType, GlobalValue::ExternalLinkage, "index", *M);
    F->addFnAttr("tensix-executor", "trisc1");
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    ST = std::make_unique<RISCVSubtarget>(
        TM->getTargetTriple(), TM->getTargetCPU(), TM->getTargetCPU(),
        TM->getTargetFeatureString(), "ilp32", 0, 0, *TM);
    MF = std::make_unique<MachineFunction>(*F, *TM, *ST, MMI->getContext(), 0);
    Block = MF->CreateMachineBasicBlock();
    MF->push_back(Block);
  }

  MachineInstr &native(unsigned Opcode) {
    auto Builder = BuildMI(*Block, Block->end(), DebugLoc(),
                           ST->getInstrInfo()->get(Opcode));
    SmallVector<Register, 4> Registers = {RISCV::TT_L0};
    if (Opcode == RISCV::TTSFPSWAPIndex)
      Registers.push_back(RISCV::TT_L1);
    Registers.push_back(RISCV::TT_L4);
    if (Opcode == RISCV::TTSFPSWAPIndex)
      Registers.push_back(RISCV::TT_L5);
    for (Register Reg : Registers)
      Builder.addReg(Reg, RegState::Define);
    for (Register Reg : Registers)
      Builder.addReg(Reg);
    if (Opcode == RISCV::TTSFPLOADIndex)
      Builder.addImm(0).addImm(0).addImm(3);
    else
      Builder.addImm(1);
    return *Builder.getInstr();
  }

  void accepts(const MachineInstr &MI) {
    std::string Details = toString(verifyTensixContentsIndexNative(MI, *ST));
    EXPECT_TRUE(Details.empty()) << Details;
  }

  void rejects(const MachineInstr &MI, StringRef Expected) {
    Error E = verifyTensixContentsIndexNative(MI, *ST);
    ASSERT_TRUE(bool(E));
    EXPECT_NE(toString(std::move(E)).find(Expected.str()), std::string::npos);
  }
};

TEST_P(RISCVTensixContentsIndexTest,
       AcceptsCompletePhysicalPairsAndStaticTies) {
  for (unsigned Opcode : {RISCV::TTSFPLOADIndex, RISCV::TTSFPSWAPIndex}) {
    SCOPED_TRACE(Opcode);
    MachineInstr &MI = native(Opcode);
    unsigned Defs = MI.getDesc().getNumDefs();
    ASSERT_EQ(Defs, Opcode == RISCV::TTSFPLOADIndex ? 2u : 4u);
    for (unsigned I = 0; I != Defs; ++I) {
      EXPECT_EQ(MI.getDesc().getOperandConstraint(Defs + I, MCOI::TIED_TO),
                int(I));
      EXPECT_EQ(MI.getOperand(I).getReg(), MI.getOperand(Defs + I).getReg());
      ASSERT_TRUE(MI.getOperand(Defs + I).isTied());
      EXPECT_EQ(MI.findTiedOperandIdx(Defs + I), I);
    }
    accepts(MI);
  }
}

TEST_P(RISCVTensixContentsIndexTest,
       RejectsEveryWrongOldRegisterClassDirectly) {
  for (unsigned Opcode : {RISCV::TTSFPLOADIndex, RISCV::TTSFPSWAPIndex}) {
    unsigned Defs = ST->getInstrInfo()->get(Opcode).getNumDefs();
    for (unsigned I = 0; I != Defs; ++I) {
      SCOPED_TRACE(Opcode);
      SCOPED_TRACE(I);
      MachineInstr &MI = native(Opcode);
      MI.getOperand(Defs + I).setReg(RISCV::TT_C9);
      rejects(MI, "operand does not match its native register class");
    }
  }
}

TEST_P(RISCVTensixContentsIndexTest, RejectsEveryBrokenOldTieDirectly) {
  for (unsigned Opcode : {RISCV::TTSFPLOADIndex, RISCV::TTSFPSWAPIndex}) {
    const MCInstrDesc &Desc = ST->getInstrInfo()->get(Opcode);
    for (unsigned I = 0; I != Desc.getNumDefs(); ++I) {
      SCOPED_TRACE(Opcode);
      SCOPED_TRACE(I);
      MachineInstr &MI = native(Opcode);
      // Distinct virtual values with the same native class isolate the
      // descriptor tie check. No generic machine verifier runs here.
      const TargetRegisterClass *Class =
          ST->getRegisterInfo()->getRegClass(Desc.operands()[I].RegClass);
      Register Result = MF->getRegInfo().createVirtualRegister(Class);
      Register Old = MF->getRegInfo().createVirtualRegister(Class);
      MI.getOperand(I).setReg(Result);
      MI.getOperand(Desc.getNumDefs() + I).setReg(Old);
      rejects(MI, "operands do not satisfy their native register tie");
    }
  }
}

TEST_P(RISCVTensixContentsIndexTest, RejectsEveryUndefinedOldContentsDirectly) {
  for (unsigned Opcode : {RISCV::TTSFPLOADIndex, RISCV::TTSFPSWAPIndex}) {
    unsigned Defs = ST->getInstrInfo()->get(Opcode).getNumDefs();
    for (unsigned I = 0; I != Defs; ++I) {
      SCOPED_TRACE(Opcode);
      SCOPED_TRACE(I);
      MachineInstr &MI = native(Opcode);
      MI.getOperand(Defs + I).setIsUndef(true);
      rejects(MI, "operand does not match its native register role");
    }
  }
}

TEST_P(RISCVTensixContentsIndexTest,
       RejectsEveryMissingPhysicalEffectDirectly) {
  for (unsigned Opcode : {RISCV::TTSFPLOADIndex, RISCV::TTSFPSWAPIndex}) {
    const MCInstrDesc &Desc = ST->getInstrInfo()->get(Opcode);
    unsigned Effects =
        Desc.implicit_uses().size() + Desc.implicit_defs().size();
    ASSERT_NE(Effects, 0u);
    for (unsigned I = 0; I != Effects; ++I) {
      SCOPED_TRACE(Opcode);
      SCOPED_TRACE(I);
      MachineInstr &MI = native(Opcode);
      MI.removeOperand(Desc.getNumOperands() + I);
      rejects(MI, "physical effects differ from the native descriptor");
    }
  }
}

TEST_P(RISCVTensixContentsIndexTest,
       RejectsVirtualValuesAtFinalNativeReceiver) {
  for (unsigned Opcode : {RISCV::TTSFPLOADIndex, RISCV::TTSFPSWAPIndex}) {
    SCOPED_TRACE(Opcode);
    MachineInstr &MI = native(Opcode);
    const TargetRegisterClass *Class =
        ST->getRegisterInfo()->getRegClass(MI.getDesc().operands()[0].RegClass);
    Register Value = MF->getRegInfo().createVirtualRegister(Class);
    MI.getOperand(0).setReg(Value);
    MI.getOperand(MI.getDesc().getNumDefs()).setReg(Value);
    rejects(MI, "native receiver requires physical registers");
  }
}

TEST_P(RISCVTensixContentsIndexTest, RejectsIllegalModesAndLoadOffsetDirectly) {
  MachineInstr &LoadFormat = native(RISCV::TTSFPLOADIndex);
  LoadFormat.getOperand(6).setImm(7);
  rejects(LoadFormat, "SFPLOADIndex mode must be one of");
  MachineInstr &LoadOffset = native(RISCV::TTSFPLOADIndex);
  LoadOffset.getOperand(4).setImm(1024);
  rejects(LoadOffset, "immediate operand must fit in 10 unsigned bits");
  MachineInstr &SwapMode = native(RISCV::TTSFPSWAPIndex);
  SwapMode.getOperand(8).setImm(10);
  rejects(SwapMode, "SFPSWAPIndex mode must be one of");
}

TEST_P(RISCVTensixContentsIndexTest,
       LoadAddressModifierUsesActualArchitecture) {
  MachineInstr &Load = native(RISCV::TTSFPLOADIndex);
  Load.getOperand(5).setImm(4);
  if (ST->hasVendorXTTTensixWH())
    rejects(Load, "Wormhole SFPLOADIndex operand 5 must be in [0, 3]");
  else
    accepts(Load);
}

TEST_P(RISCVTensixContentsIndexTest, AcceptsEveryHardwareLoadIndexPair) {
  static constexpr MCPhysReg Primaries[] = {RISCV::TT_L0, RISCV::TT_L1,
                                            RISCV::TT_L2, RISCV::TT_L3};
  static constexpr MCPhysReg Indices[] = {RISCV::TT_L4, RISCV::TT_L5,
                                          RISCV::TT_L6, RISCV::TT_L7};
  for (unsigned P = 0; P != 4; ++P) {
    SCOPED_TRACE(P);
    MachineInstr &MI = native(RISCV::TTSFPLOADIndex);
    for (unsigned I : {0u, 2u})
      MI.getOperand(I).setReg(Primaries[P]);
    for (unsigned I : {1u, 3u})
      MI.getOperand(I).setReg(Indices[P]);
    accepts(MI);
  }
}

TEST_P(RISCVTensixContentsIndexTest, AcceptsEveryDistinctLowSwapIndexGroup) {
  static constexpr MCPhysReg Primaries[] = {RISCV::TT_L0, RISCV::TT_L1,
                                            RISCV::TT_L2, RISCV::TT_L3};
  static constexpr MCPhysReg Indices[] = {RISCV::TT_L4, RISCV::TT_L5,
                                          RISCV::TT_L6, RISCV::TT_L7};
  for (unsigned P = 0; P != 4; ++P)
    for (unsigned Q = 0; Q != 4; ++Q) {
      if (P == Q)
        continue;
      SCOPED_TRACE(P);
      SCOPED_TRACE(Q);
      MachineInstr &MI = native(RISCV::TTSFPSWAPIndex);
      MCPhysReg Registers[] = {Primaries[P], Primaries[Q], Indices[P],
                               Indices[Q]};
      for (unsigned I = 0; I != 4; ++I) {
        MI.getOperand(I).setReg(Registers[I]);
        MI.getOperand(4 + I).setReg(Registers[I]);
      }
      accepts(MI);
    }
}

TEST_P(RISCVTensixContentsIndexTest,
       RejectsMismatchedCompanionsEvenWithCorrectClassesAndTies) {
  MachineInstr &Load = native(RISCV::TTSFPLOADIndex);
  Load.getOperand(1).setReg(RISCV::TT_L5);
  Load.getOperand(3).setReg(RISCV::TT_L5);
  rejects(Load, "hardware +4 companion");
  MachineInstr &Swap = native(RISCV::TTSFPSWAPIndex);
  Swap.getOperand(2).setReg(RISCV::TT_L6);
  Swap.getOperand(6).setReg(RISCV::TT_L6);
  rejects(Swap, "hardware +4 companion");
}

TEST_P(RISCVTensixContentsIndexTest,
       RejectsIdenticalSwapRolesEvenWhenEveryNativeTieMatches) {
  MachineInstr &Swap = native(RISCV::TTSFPSWAPIndex);
  for (unsigned I : {1u, 5u})
    Swap.getOperand(I).setReg(RISCV::TT_L0);
  for (unsigned I : {3u, 7u})
    Swap.getOperand(I).setReg(RISCV::TT_L4);
  rejects(Swap, "distinct low primaries");
}

TEST_P(RISCVTensixContentsIndexTest,
       PairAllocationAliasesOnlyItsActualArchitecturalLeaves) {
  const auto &TRI = *ST->getRegisterInfo();
  EXPECT_EQ(RISCV::SFPRIndexPairRegClass.getNumRegs(), 4u);
  EXPECT_TRUE(isRISCVRegisterContentsClass(*MF, RISCV::SFPRIndexPairRegClass));
  for (MCPhysReg Pair : RISCV::SFPRIndexPairRegClass) {
    MCRegister Value = TRI.getSubReg(Pair, RISCV::sub_sfpu_value);
    MCRegister Index = TRI.getSubReg(Pair, RISCV::sub_sfpu_index);
    ASSERT_TRUE(RISCV::SFPRLowRegClass.contains(Value));
    ASSERT_TRUE(RISCV::SFPRHighRegClass.contains(Index));
    EXPECT_EQ(TRI.getEncodingValue(Index), TRI.getEncodingValue(Value) + 4);
    for (MCPhysReg Leaf : RISCV::SFPRRegClass)
      EXPECT_EQ(TRI.regsOverlap(Pair, Leaf), Leaf == Value || Leaf == Index);
    for (MCPhysReg Other : RISCV::SFPRIndexPairRegClass)
      EXPECT_EQ(TRI.regsOverlap(Pair, Other), Pair == Other);
  }
}

TEST_P(RISCVTensixContentsIndexTest,
       SwapPairPseudoRetainsBothTiesAndHardDisjointRoles) {
  const MCInstrDesc &Desc =
      ST->getInstrInfo()->get(RISCV::PseudoTTSFPContentsSwapIndex);
  EXPECT_EQ(Desc.getNumDefs(), 2u);
  EXPECT_EQ(Desc.getOperandConstraint(2, MCOI::TIED_TO), 0);
  EXPECT_EQ(Desc.getOperandConstraint(3, MCOI::TIED_TO), 1);
  EXPECT_GE(Desc.getOperandConstraint(0, MCOI::EARLY_CLOBBER), 0);
  EXPECT_GE(Desc.getOperandConstraint(1, MCOI::EARLY_CLOBBER), 0);
  EXPECT_EQ(Desc.operands()[0].RegClass, RISCV::SFPRIndexPairRegClassID);
  EXPECT_EQ(Desc.operands()[1].RegClass, RISCV::SFPRIndexPairRegClassID);
}

TEST_P(RISCVTensixContentsIndexTest,
       ProjectsNoncanonicalStaticAndDynamicPairsToCompleteNativeOperands) {
  for (bool Dynamic : {false, true}) {
    SCOPED_TRACE(Dynamic);
    Register Pair = RISCV::TT_L1_TT_L5;
    auto Builder =
        BuildMI(*Block, Block->end(), DebugLoc(),
                ST->getInstrInfo()->get(
                    Dynamic ? RISCV::PseudoTTSFPContentsLoadIndexDynamic
                            : RISCV::PseudoTTSFPContentsLoadIndex));
    Builder.addReg(Pair, RegState::Define);
    if (Dynamic)
      Builder.addReg(RISCV::X5, RegState::Define | RegState::EarlyClobber)
          .addReg(RISCV::X6, RegState::Define | RegState::EarlyClobber);
    Builder.addReg(Pair);
    if (Dynamic)
      Builder.addReg(RISCV::X10);
    else
      Builder.addImm(4);
    Builder.addImm(0).addImm(3);
    auto Flags = MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile;
    if (Dynamic)
      Flags |= MachineMemOperand::MOStore;
    MachineMemOperand *MMO =
        MF->getMachineMemOperand(MachinePointerInfo(), Flags, 4, Align(4));
    Builder.addMemOperand(MMO);
    ASSERT_TRUE(expandTensixContentsGroupPseudo(*Builder, *ST));
    MachineInstr &Native = Block->back();
    ASSERT_EQ(Native.getOpcode(),
              Dynamic ? RISCV::PseudoTTSFPLOADIndex : RISCV::TTSFPLOADIndex);
    EXPECT_EQ(Native.getOperand(0).getReg(), RISCV::TT_L1);
    EXPECT_EQ(Native.getOperand(1).getReg(), RISCV::TT_L5);
    unsigned Old = Dynamic ? 4 : 2;
    EXPECT_EQ(Native.getOperand(Old).getReg(), RISCV::TT_L1);
    EXPECT_EQ(Native.getOperand(Old + 1).getReg(), RISCV::TT_L5);
    ASSERT_TRUE(Native.hasOneMemOperand());
    EXPECT_EQ(*Native.memoperands_begin(), MMO);
    if (Dynamic) {
      EXPECT_EQ(Native.getOperand(2).getReg(), RISCV::X5);
      EXPECT_EQ(Native.getOperand(3).getReg(), RISCV::X6);
      EXPECT_TRUE(Native.getOperand(2).isEarlyClobber());
      EXPECT_TRUE(Native.getOperand(3).isEarlyClobber());
      EXPECT_EQ(Native.getOperand(6).getReg(), RISCV::X10);
    }
    accepts(Native);
  }
}

TEST_P(RISCVTensixContentsIndexTest,
       ProjectsDistinctAlternateSwapPairsWithoutLosingNumericalRoles) {
  auto Builder =
      BuildMI(*Block, Block->end(), DebugLoc(),
              ST->getInstrInfo()->get(RISCV::PseudoTTSFPContentsSwapIndex));
  Builder.addReg(RISCV::TT_L1_TT_L5, RegState::Define | RegState::EarlyClobber)
      .addReg(RISCV::TT_L3_TT_L7, RegState::Define | RegState::EarlyClobber)
      .addReg(RISCV::TT_L1_TT_L5)
      .addReg(RISCV::TT_L3_TT_L7)
      .addImm(1);
  ASSERT_TRUE(expandTensixContentsGroupPseudo(*Builder, *ST));
  MachineInstr &Native = Block->back();
  ASSERT_EQ(Native.getOpcode(), RISCV::TTSFPSWAPIndex);
  MCPhysReg Registers[] = {RISCV::TT_L1, RISCV::TT_L3, RISCV::TT_L5,
                           RISCV::TT_L7};
  for (unsigned I = 0; I != 4; ++I) {
    EXPECT_EQ(Native.getOperand(I).getReg(), Registers[I]);
    EXPECT_EQ(Native.getOperand(4 + I).getReg(), Registers[I]);
    EXPECT_FALSE(Native.getOperand(I).isEarlyClobber());
  }
  accepts(Native);
}

INSTANTIATE_TEST_SUITE_P(BlackholeAndWormhole, RISCVTensixContentsIndexTest,
                         testing::Values("+xtttensixbh", "+xtttensixwh"));

} // namespace
