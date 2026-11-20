//===-- RISCVTensixElementwiseTest.cpp ------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsTT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(RISCVTensixElementwise, WormholeReceivesAllAuthoredFieldSelectors) {
  ASSERT_TRUE(RISCV::isTensixWormholeIntrinsic(Intrinsic::riscv_tt_elwadd));
  // WH ELWADD.md specifies DstRow:u10, AddrMod:u2, Broadcast:u2,
  // AddDst:bool, and FlipSrcA/B:u2. No configuration or completion is inferred.
  for (uint32_t Row : {0u, 1u, 1023u})
    for (uint32_t Address : {0u, 1u, 2u, 3u})
      for (uint32_t Broadcast : {0u, 1u, 2u, 3u})
        for (uint32_t AddDst : {0u, 1u})
          for (uint32_t Clear : {0u, 1u, 2u, 3u}) {
            const uint32_t Fields[] = {Row, Address, Broadcast, AddDst, Clear};
            std::string Diagnostic =
                toString(RISCV::verifyTensixWormholeInstructionFields(
                    Intrinsic::riscv_tt_elwadd, Fields));
            EXPECT_TRUE(Diagnostic.empty()) << Diagnostic;
          }
}

TEST(RISCVTensixElementwise, WormholeRejectsEveryOutOfRangeField) {
  struct InvalidField {
    unsigned Index;
    uint32_t Value;
    StringRef Name;
  };
  const InvalidField Invalid[] = {
      {0, 1024, "dst"},        {0, UINT32_MAX, "dst"}, {1, 4, "addr_mode"},
      {1, 8, "addr_mode"},     {1, 31, "addr_mode"},   {2, 4, "instr_mod19"},
      {3, 2, "dest_accum_en"}, {4, 4, "clear_dvalid"}};
  for (const InvalidField &Case : Invalid) {
    SCOPED_TRACE(Case.Index);
    SCOPED_TRACE(Case.Value);
    uint32_t Fields[] = {0, 0, 0, 0, 0};
    Fields[Case.Index] = Case.Value;
    std::string Diagnostic =
        toString(RISCV::verifyTensixWormholeInstructionFields(
            Intrinsic::riscv_tt_elwadd, Fields));
    EXPECT_NE(Diagnostic.find(("field " + Case.Name + " must be in").str()),
              std::string::npos)
        << Diagnostic;
  }
}

TEST(RISCVTensixElementwise, WormholeRequiresExactlyFiveLogicalFields) {
  for (unsigned Count : {0u, 4u, 6u}) {
    SCOPED_TRACE(Count);
    SmallVector<uint32_t, 6> Fields(Count, 0);
    EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(
                           Intrinsic::riscv_tt_elwadd, Fields))
                  .find("incorrect number of Tensix instruction operands"),
              std::string::npos);
  }
}

TEST(RISCVTensixElementwise, WormholeDoesNotAdmitPortOrMopForms) {
  const uint32_t Fields[] = {0, 0, 0, 0, 0};
  for (Intrinsic::ID ID :
       {Intrinsic::riscv_tt_elwadd_port, Intrinsic::riscv_tt_elwadd_mop}) {
    SCOPED_TRACE(unsigned(ID));
    EXPECT_FALSE(RISCV::isTensixWormholeIntrinsic(ID));
    EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(ID, Fields))
                  .find("not supported on Wormhole"),
              std::string::npos);
  }
}

class RISCVTensixElementwiseMC : public testing::Test {
protected:
  Triple TT{"riscv32-unknown-unknown"};
  const Target *TheTarget = nullptr;
  // MCAsmInfo retains this by reference; it must outlive every MCContext.
  MCTargetOptions MCOptions;
  std::unique_ptr<MCInstrInfo> MCII;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  unsigned Opcode = 0;

  void SetUp() override {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    std::string Diagnostic;
    TheTarget = TargetRegistry::lookupTarget(TT, Diagnostic);
    ASSERT_NE(TheTarget, nullptr) << Diagnostic;
    MCII.reset(TheTarget->createMCInstrInfo());
    MRI.reset(TheTarget->createMCRegInfo(TT));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCOptions));
    const auto *Machine =
        RISCV::getTensixMachineInfoByIntrinsic(Intrinsic::riscv_tt_elwadd);
    ASSERT_NE(Machine, nullptr);
    Opcode = Machine->Opcode;
  }

  MCInst instruction(ArrayRef<uint32_t> Fields) {
    MCInst MI;
    MI.setOpcode(Opcode);
    for (uint32_t Field : Fields)
      MI.addOperand(MCOperand::createImm(Field));
    return MI;
  }

  SmallVector<uint8_t, 4> encode(StringRef Features,
                                 ArrayRef<uint32_t> Fields) {
    std::unique_ptr<MCSubtargetInfo> STI(
        TheTarget->createMCSubtargetInfo(TT, "", Features));
    MCContext Context(TT, *MAI, *MRI, *STI);
    Context.setDiagnosticHandler([](const SMDiagnostic &Diagnostic, bool,
                                    const SourceMgr &,
                                    std::vector<const MDNode *> &) {
      ADD_FAILURE() << Diagnostic.getMessage().str();
    });
    std::unique_ptr<MCCodeEmitter> Emitter(
        TheTarget->createMCCodeEmitter(*MCII, Context));
    SmallVector<char, 4> Bytes;
    SmallVector<MCFixup, 0> Fixups;
    Emitter->encodeInstruction(instruction(Fields), Bytes, Fixups, *STI);
    EXPECT_TRUE(Fixups.empty());
    return {Bytes.begin(), Bytes.end()};
  }
};

TEST_F(RISCVTensixElementwiseMC,
       BlackholeRetainsAddressSelectorsRejectedOnWormhole) {
  for (uint32_t Address : {4u, 5u, 6u, 7u}) {
    SCOPED_TRACE(Address);
    const uint32_t Fields[] = {0, Address, 0, 0, 0};
    EXPECT_TRUE(toString(RISCV::verifyTensixMCInstructionFields(
                             instruction(Fields), *MCII))
                    .empty());
    EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(
                           Intrinsic::riscv_tt_elwadd, Fields))
                  .find("field addr_mode must be in [0, 3]"),
              std::string::npos);
  }
}

TEST_F(RISCVTensixElementwiseMC, ReceivesArchitectureSpecificDirectEncoding) {
  // Independently derived from Bits32_ELWADD.svg: WH AddrMod is raw16:15,
  // BH's existing native ABI starts at raw14. Direct issue rotates raw left2.
  const uint32_t AddressOnly[] = {0, 1, 0, 0, 0};
  EXPECT_EQ(encode("+xtttensixwh", AddressOnly),
            (SmallVector<uint8_t, 4>{0x00, 0x00, 0x02, 0xa0}));
  EXPECT_EQ(encode("+xtttensixbh", AddressOnly),
            (SmallVector<uint8_t, 4>{0x00, 0x00, 0x01, 0xa0}));
  const uint32_t MaxWH[] = {1023, 3, 3, 1, 3};
  EXPECT_EQ(encode("+xtttensixwh", MaxWH),
            (SmallVector<uint8_t, 4>{0xfc, 0x0f, 0xe6, 0xa3}));
  EXPECT_EQ(encode("+xtttensixbh", MaxWH),
            (SmallVector<uint8_t, 4>{0xfc, 0x0f, 0xe3, 0xa3}));
  const uint32_t MaxBH[] = {16383, 31, 3, 1, 3};
  EXPECT_EQ(encode("+xtttensixbh", MaxBH),
            (SmallVector<uint8_t, 4>{0xfc, 0xff, 0xff, 0xa3}));
}

TEST(RISCVTensixElementwise, WormholeIRReceivesStaticIssueOnly) {
  LLVMContext Context;
  SMDiagnostic Diagnostic;
  auto M = parseAssemblyString(R"(
    declare void @llvm.riscv.tt.elwadd(i32 immarg, i32 immarg, i32 immarg,
                                      i32 immarg, i32 immarg)
    define void @kernel() "target-features"="+xtttensixwh"
                          "tensix-executor"="trisc1" {
      call void @llvm.riscv.tt.elwadd(i32 1023, i32 3, i32 3, i32 1, i32 3)
      ret void
    }
  )",
                               Diagnostic, Context);
  ASSERT_TRUE(M);
  M->setTargetTriple(Triple("riscv32-unknown-unknown"));
  EXPECT_TRUE(toString(RISCV::verifyTensixModule(*M)).empty());
}

TEST(RISCVTensixElementwise, WormholeIRRejectsDynamicDirectFields) {
  LLVMContext Context;
  SMDiagnostic Diagnostic;
  auto M = parseAssemblyString(R"(
    declare void @llvm.riscv.tt.elwadd(i32 immarg, i32 immarg, i32 immarg,
                                      i32 immarg, i32 immarg)
    define void @kernel(i32 %row) "target-features"="+xtttensixwh"
                                  "tensix-executor"="trisc1" {
      call void @llvm.riscv.tt.elwadd(i32 %row, i32 0, i32 0, i32 0, i32 0)
      ret void
    }
  )",
                               Diagnostic, Context);
  ASSERT_TRUE(M);
  M->setTargetTriple(Triple("riscv32-unknown-unknown"));
  EXPECT_NE(toString(RISCV::verifyTensixModule(*M))
                .find("immarg operand has non-immediate parameter"),
            std::string::npos);
}

} // namespace
