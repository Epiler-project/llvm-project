//===-- RISCVTensixTest.cpp ----------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "MCTargetDesc/RISCVTensixBoundContract.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsTT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <future>

using namespace llvm;

namespace {
std::unique_ptr<Module> parse(LLVMContext &Ctx, StringRef Body) {
  SMDiagnostic Diagnostic;
  auto M = parseAssemblyString(Body, Diagnostic, Ctx);
  EXPECT_TRUE(M);
  if (M)
    M->setTargetTriple(Triple("riscv32-unknown-unknown"));
  return M;
}

using SFPUArchitecture = RISCV::TensixSFPUArchitecture;
using SFPUAccess = RISCV::TensixSFPUAccess;
using SFPUState = RISCV::TensixSFPUState;

TEST(RISCVTensix, OrdinaryLogicalContract) {
  const auto *Info = RISCV::getTensixInstructionByName("stallwait");
  ASSERT_NE(Info, nullptr);
  EXPECT_EQ(Info->getName(), "STALLWAIT");
  EXPECT_EQ(Info->IntrinsicID, Intrinsic::tt_stallwait);
  EXPECT_EQ(Info->PortIntrinsicID, Intrinsic::tt_stallwait_port);
  EXPECT_EQ(Info->MopIntrinsicID, Intrinsic::tt_stallwait_mop);
  EXPECT_EQ(Info->NumFields, 2);
  const auto *Field = RISCV::getTensixInstructionField(*Info, 0);
  ASSERT_NE(Field, nullptr);
  EXPECT_EQ(Field->getName(), "wait_res");
  EXPECT_EQ(Field->MaxValue, 32767u);
  EXPECT_EQ(RISCV::getTensixInstructionField(*Info, 2), nullptr);
  EXPECT_EQ(RISCV::getTensixInstructionByIntrinsic(Intrinsic::tt_stallwait_port),
            Info);
  EXPECT_NE(RISCV::getTensixInstructionByName("mop_cfg"), nullptr);
  EXPECT_EQ(RISCV::getTensixInstructionByName("sfpadd"), nullptr);
}

TEST(RISCVTensix, ZeroaccKeepsFiveFieldLogicalContract) {
  const auto *Info = RISCV::getTensixInstructionByName("zeroacc");
  ASSERT_NE(Info, nullptr);
  EXPECT_EQ(Info->IntrinsicID, Intrinsic::tt_zeroacc);
  EXPECT_EQ(Info->PortIntrinsicID, Intrinsic::tt_zeroacc_port);
  EXPECT_EQ(Info->MopIntrinsicID, Intrinsic::tt_zeroacc_mop);
  ASSERT_EQ(Info->NumFields, 5);
  const StringRef Names[] = {"where", "addr_mode", "clear_zero_flags",
                             "use_32_bit_mode", "clear_mode"};
  const uint32_t Maxima[] = {16383, 7, 1, 1, 31};
  for (unsigned I = 0; I != 5; ++I) {
    const auto *Field = RISCV::getTensixInstructionField(*Info, I);
    ASSERT_NE(Field, nullptr);
    EXPECT_EQ(Field->getName(), Names[I]);
    EXPECT_EQ(Field->MaxValue, Maxima[I]);
  }
  EXPECT_EQ(RISCV::getTensixInstructionField(*Info, 5), nullptr);
}

TEST(RISCVTensix, WormholeZeroaccReceivesCanonicalFullAndFaceForms) {
  EXPECT_TRUE(RISCV::isTensixWormholeIntrinsic(Intrinsic::tt_zeroacc));
  EXPECT_TRUE(RISCV::isTensixWormholeIntrinsic(Intrinsic::tt_zeroacc_port));
  for (uint32_t Use32 : {0u, 1u}) {
    const uint32_t Fields[] = {0, 1, 0, Use32, 3};
    EXPECT_TRUE(toString(RISCV::verifyTensixWormholeInstructionFields(
                             Intrinsic::tt_zeroacc, Fields))
                    .empty());
  }
  for (Intrinsic::ID ID : {Intrinsic::tt_zeroacc, Intrinsic::tt_zeroacc_port}) {
    for (auto [Use32, LastFace] : {std::pair{0u, 63u}, std::pair{1u, 31u}}) {
      for (uint32_t Face : {0u, LastFace}) {
        const uint32_t Fields[] = {Face, 0, 0, Use32, 1};
        EXPECT_TRUE(
            toString(RISCV::verifyTensixWormholeInstructionFields(ID, Fields))
                .empty());
      }
    }
  }
}

TEST(RISCVTensix, WormholeZeroaccRejectsInvalidFieldsAndIssueForms) {
  struct InvalidField {
    unsigned Index;
    uint32_t Value;
    StringRef Diagnostic;
  };
  const InvalidField Invalid[] = {
      {0, 1, "does not match an admitted logical form"},
      {0, 64, "field where must be in [0, 63]"},
      {1, 0, "does not match an admitted logical form"},
      {1, 2, "field addr_mode must be in [0, 1]"},
      {2, 1, "field clear_zero_flags must be in [0, 0]"},
      {3, 2, "field use_32_bit_mode must be in [0, 1]"},
      {4, 2, "does not match an admitted logical form"},
      {4, 7, "field clear_mode must be in [0, 3]"},
  };
  for (const InvalidField &Case : Invalid) {
    SCOPED_TRACE(Case.Index);
    SCOPED_TRACE(Case.Value);
    uint32_t Fields[] = {0, 1, 0, 0, 3};
    Fields[Case.Index] = Case.Value;
    std::string Diagnostic =
        toString(RISCV::verifyTensixWormholeInstructionFields(
            Intrinsic::tt_zeroacc, Fields));
    EXPECT_NE(Diagnostic.find(Case.Diagnostic.str()), std::string::npos)
        << Diagnostic;
  }
  for (unsigned Count : {4u, 6u}) {
    SmallVector<uint32_t, 6> Fields(Count, 0);
    EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(
                           Intrinsic::tt_zeroacc, Fields))
                  .find("incorrect number of Tensix instruction operands"),
              std::string::npos);
  }
  for (Intrinsic::ID ID : {Intrinsic::tt_zeroacc, Intrinsic::tt_zeroacc_port}) {
    const uint32_t Fields[] = {32, 0, 0, 1, 1};
    EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(ID, Fields))
                  .find("16-row FP32 face must be in [0, 31]"),
              std::string::npos);
  }
  const uint32_t FullFields[] = {0, 1, 0, 0, 3};
  EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(
                         Intrinsic::tt_zeroacc_port, FullFields))
                .find("instruction-port emission requires 16-row mode"),
            std::string::npos);
  EXPECT_FALSE(RISCV::isTensixWormholeIntrinsic(Intrinsic::tt_zeroacc_mop));
  EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(
                         Intrinsic::tt_zeroacc_mop, FullFields))
                .find("not supported on Wormhole"),
            std::string::npos);
}

TEST(RISCVTensix, WormholeNativePipelinesReceiveArchitecturalFieldBounds) {
  struct Case {
    Intrinsic::ID ID;
    SmallVector<uint32_t> Fields;
  };
  const Case Cases[] = {
      {Intrinsic::tt_mvmul, {0, 1, 0, 0}},
      {Intrinsic::tt_mvmul, {1023, 3, 0, 3}},
      {Intrinsic::tt_mvmul, {0, 0, 1, 0}},
      {Intrinsic::tt_unpacr, {1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0}},
      {Intrinsic::tt_unpacr, {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 17, 1}},
      {Intrinsic::tt_unpacr, {0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 255, 1}},
      {Intrinsic::tt_zerosrc, {3, 1, 0, 0}},
      {Intrinsic::tt_zerosrc, {1, 0, 1, 1}},
      {Intrinsic::tt_zerosrc, {3, 1, 1, 1}}};
  for (const Case &C : Cases) {
    SCOPED_TRACE(unsigned(C.ID));
    EXPECT_TRUE(RISCV::isTensixWormholeIntrinsic(C.ID));
    std::string Diagnostic =
        toString(RISCV::verifyTensixWormholeInstructionFields(C.ID, C.Fields));
    EXPECT_TRUE(Diagnostic.empty()) << Diagnostic;
  }
}

TEST(RISCVTensix, WormholeNativePipelinesRejectUnreceivedAndReservedFields) {
  struct Case {
    Intrinsic::ID ID;
    SmallVector<uint32_t> Fields;
    StringRef Field;
  };
  const Case Cases[] = {
      {Intrinsic::tt_mvmul, {1024, 0, 0, 0}, "dst"},
      {Intrinsic::tt_mvmul, {0, 4, 0, 0}, "addr_mode"},
      {Intrinsic::tt_mvmul, {0, 0, 2, 0}, "instr_mod19"},
      {Intrinsic::tt_mvmul, {0, 0, 0, 4}, "clear_dvalid"},
      {Intrinsic::tt_unpacr,
       {1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
       "searchcacheflush"},
      {Intrinsic::tt_unpacr,
       {1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
       "rowsearch"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0},
       "autoinccontextid"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0},
       "srcb_bcast"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
       "ovrdthreadid"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0},
       "addrcntcontextid"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0},
       "cfgcontextid"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0},
       "cfgcontextcntinc"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 256, 0},
       "addrmode"},
      {Intrinsic::tt_unpacr,
       {1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2},
       "unpack_block_selection"},
      {Intrinsic::tt_zerosrc, {4, 0, 0, 0}, "src_mask"},
      {Intrinsic::tt_zerosrc, {1, 2, 0, 0}, "bank_mask"},
      {Intrinsic::tt_zerosrc, {1, 0, 2, 0}, "write_mode"},
      {Intrinsic::tt_zerosrc, {1, 0, 0, 2}, "zero_val"}};
  for (const Case &C : Cases) {
    SCOPED_TRACE(unsigned(C.ID));
    SCOPED_TRACE(C.Field.str());
    std::string Diagnostic =
        toString(RISCV::verifyTensixWormholeInstructionFields(C.ID, C.Fields));
    EXPECT_NE(Diagnostic.find(("field " + C.Field + " must be in").str()),
              std::string::npos)
        << Diagnostic;
  }
}

TEST(RISCVTensix, WormholeNativePipelinesRejectPortAndMopForms) {
  for (Intrinsic::ID ID : std::initializer_list<Intrinsic::ID>{
           Intrinsic::tt_mvmul_port, Intrinsic::tt_mvmul_mop,
           Intrinsic::tt_unpacr_port, Intrinsic::tt_unpacr_mop,
           Intrinsic::tt_zerosrc_port, Intrinsic::tt_zerosrc_mop}) {
    SCOPED_TRACE(unsigned(ID));
    EXPECT_FALSE(RISCV::isTensixWormholeIntrinsic(ID));
    EXPECT_NE(toString(RISCV::verifyTensixWormholeInstructionFields(ID, {}))
                  .find("not supported on Wormhole"),
              std::string::npos);
  }
}

TEST(RISCVTensix, ScalarDependentUseAcceptsBothModelsAndAllExecutorRoles) {
  LLVMContext Ctx;
  auto M = parse(Ctx, R"(
    declare void @llvm.tt.dependent.use(i32)
    define void @readback(ptr %address, ptr %output) {
      store volatile i32 1, ptr %address, align 4
      %value = load volatile i32, ptr %address, align 4
      call void @llvm.tt.dependent.use(i32 %value)
      store volatile i32 2, ptr %output, align 4
      ret void
    }
  )");
  ASSERT_TRUE(M);
  Function *F = M->getFunction("readback");
  for (StringRef Features : {"+xtttensixbh", "+xtttensixwh"}) {
    SCOPED_TRACE(Features.str());
    F->addFnAttr("target-features", Features);
    for (StringRef Executor :
         {"ncrisc", "brisc", "trisc0", "trisc1", "trisc2"}) {
      SCOPED_TRACE(Executor.str());
      F->addFnAttr("tensix-executor", Executor);
      std::string Diagnostic = toString(RISCV::verifyTensixModule(*M));
      EXPECT_TRUE(Diagnostic.empty()) << Diagnostic;
    }
  }
}

TEST(RISCVTensix, ScalarDependentUseRequiresFeatureAndValidExecutor) {
  LLVMContext Ctx;
  auto M = parse(Ctx, R"(
    declare void @llvm.tt.dependent.use(i32)
    define void @readback(i32 %value) {
      call void @llvm.tt.dependent.use(i32 %value)
      ret void
    }
  )");
  ASSERT_TRUE(M);
  Function *F = M->getFunction("readback");
  F->addFnAttr("tensix-executor", "ncrisc");
  EXPECT_NE(toString(RISCV::verifyTensixModule(*M))
                .find("Tensix intrinsic requires +xtttensix"),
            std::string::npos);
  F->addFnAttr("target-features", "+xtttensixwh");
  F->removeFnAttr("tensix-executor");
  EXPECT_NE(toString(RISCV::verifyTensixModule(*M))
                .find("requires a tensix-executor function attribute"),
            std::string::npos);
  F->addFnAttr("tensix-executor", "host");
  EXPECT_NE(toString(RISCV::verifyTensixModule(*M))
                .find("invalid tensix-executor value: host"),
            std::string::npos);
}

TEST(RISCVTensix, ScalarDependentUseRejectsNonDirectCallShapes) {
  const StringRef Calls[] = {
      "call void @llvm.tt.dependent.use(i32 %value) [ \"deopt\"(i32 "
      "%value) ]",
      "call fastcc void @llvm.tt.dependent.use(i32 %value)"};
  for (StringRef Call : Calls) {
    SCOPED_TRACE(Call.str());
    LLVMContext Ctx;
    std::string Body = "declare void @llvm.tt.dependent.use(i32)\n"
                       "define void @readback(i32 %value) "
                       "\"target-features\"=\"+xtttensixwh\" "
                       "\"tensix-executor\"=\"ncrisc\" {\n" +
                       Call.str() + "\nret void\n}\n";
    auto M = parse(Ctx, Body);
    ASSERT_TRUE(M);
    EXPECT_NE(toString(RISCV::verifyTensixModule(*M))
                  .find("requires a direct C call without operand bundles"),
              std::string::npos);
  }
}

TEST(RISCVTensix, RecoverableOrdinaryRange) {
  LLVMContext Ctx;
  auto M = parse(Ctx, R"(
    declare void @llvm.tt.stallwait(i32 immarg, i32 immarg)
    define void @kernel() "target-features"="+xtttensixbh"
                          "tensix-executor"="trisc1" {
      call void @llvm.tt.stallwait(i32 32768, i32 0)
      ret void
    }
  )");
  ASSERT_TRUE(M);
  Error E = RISCV::verifyTensixModule(*M);
  ASSERT_TRUE(bool(E));
  EXPECT_NE(toString(std::move(E)).find("[0, 32767]"), std::string::npos);
}

TEST(RISCVTensix, FeatureOrderAndCPUClosure) {
  LLVMContext Ctx;
  auto M = parse(Ctx, R"(
    declare void @llvm.tt.nop()
    define void @kernel() "target-features"="+xtttensixbh,-xtttensixbh"
                          "tensix-executor"="trisc1" {
      call void @llvm.tt.nop()
      ret void
    }
  )");
  ASSERT_TRUE(M);
  Error Disabled = RISCV::verifyTensixModule(*M);
  ASSERT_TRUE(bool(Disabled));
  EXPECT_NE(toString(std::move(Disabled)).find("requires +xtttensixbh"),
            std::string::npos);
  Function *F = M->getFunction("kernel");
  F->addFnAttr("target-features", "-xtttensixbh,+xtttensixbh");
  EXPECT_FALSE(bool(RISCV::verifyTensixModule(*M)));
  F->addFnAttr("target-cpu", "generic-rv32");
  F->addFnAttr("target-features", "+xtttensixbh,+c");
  Error Compressed = RISCV::verifyTensixModule(*M);
  ASSERT_TRUE(bool(Compressed));
  EXPECT_NE(toString(std::move(Compressed)).find("C/Zc"), std::string::npos);
}

TEST(RISCVTensix, DefinednessIsSeparateFromRange) {
  LLVMContext Ctx;
  auto M = parse(Ctx, R"(
    declare void @llvm.tt.stallwait.port(i32 immarg, i32, i32)
    define void @kernel(i32 %input) "target-features"="+xtttensixbh"
                          "tensix-executor"="trisc1" {
      %bounded = and i32 %input, 32767
      call void @llvm.tt.stallwait.port(i32 0, i32 %bounded, i32 0)
      ret void
    }
  )");
  ASSERT_TRUE(M);
  Error E = RISCV::verifyTensixModule(*M);
  ASSERT_TRUE(bool(E));
  EXPECT_NE(toString(std::move(E)).find("defined and non-poison"),
            std::string::npos);
  M->getFunction("kernel")->addParamAttr(0, Attribute::NoUndef);
  EXPECT_FALSE(bool(RISCV::verifyTensixModule(*M)));
}

// Literal physical ABI cases are independent of the native descriptor table.
// A missing receiver, an accidentally shared BH/WH mode range, or a dynamic
// register selector must fail even when the intrinsic declaration still exists.
struct BoundFormCase {
  Intrinsic::ID Blackhole;
  Intrinsic::ID Wormhole;
  SmallVector<int64_t, 17> Arguments;
  int DynamicOffset = -1;
  int ModeArgument = -1;
  uint16_t BlackholeModes = 0;
  uint16_t WormholeModes = 0;
};

const BoundFormCase BoundForms[] = {
    {Intrinsic::tt_bh_sfpu_loadi,
     Intrinsic::tt_wh_sfpu_loadi,
     {0, 0, 65535, 0},
     -1,
     3,
     0x0517,
     0x0517},
    {Intrinsic::tt_bh_sfpu_load,
     Intrinsic::tt_wh_sfpu_load,
     {0, 0, 1023, 3, 4},
     2,
     4,
     0xc17f,
     0xc17f},
    {Intrinsic::tt_bh_sfpu_store,
     Intrinsic::tt_wh_sfpu_store,
     {0, 1023, 3, 4},
     1,
     3,
     0xc17f,
     0xc17f},
    {Intrinsic::tt_bh_sfpu_add,
     Intrinsic::tt_wh_sfpu_add,
     {0, 0, 9, 10, 0},
     -1,
     4,
     0x000f,
     0x0001},
    {Intrinsic::tt_bh_sfpu_mul,
     Intrinsic::tt_wh_sfpu_mul,
     {0, 0, 9, 10, 0},
     -1,
     4,
     0x000f,
     0x0001},
    {Intrinsic::tt_bh_sfpu_mul23, Intrinsic::not_intrinsic,
     {0, 0, 2, 3, 0}, -1, 4, 0x0003, 0},
    {Intrinsic::tt_bh_sfpu_mad,
     Intrinsic::tt_wh_sfpu_mad,
     {0, 0, 9, 10, 8, 0},
     -1,
     5,
     0x000f,
     0x0001},
    {Intrinsic::tt_bh_sfpu_iadd,
     Intrinsic::tt_wh_sfpu_iadd,
     {0, 0, 10, 4},
     -1,
     3,
     0x0555,
     0x0555},
    {Intrinsic::tt_bh_sfpu_mov,
     Intrinsic::tt_wh_sfpu_mov,
     {0, 0, 15, 0},
     -1,
     3,
     0x0003,
     0x0003},
    {Intrinsic::tt_bh_sfpu_mov_all, Intrinsic::tt_wh_sfpu_mov_all, {0, 15}},
    {Intrinsic::tt_bh_sfpu_mov_prng_advance,
     Intrinsic::tt_wh_sfpu_mov_prng_advance,
     {7, 7}},
    {Intrinsic::tt_bh_sfpu_arecip,
     Intrinsic::not_intrinsic,
     {0, 0, 10, 0},
     -1,
     3,
     0x0007,
     0},
    {Intrinsic::tt_bh_sfpu_exexp,
     Intrinsic::tt_wh_sfpu_exexp,
     {0, 0, 10, 0},
     -1,
     3,
     0x0f0f,
     0x0f0f},
    {Intrinsic::tt_bh_sfpu_exman,
     Intrinsic::tt_wh_sfpu_exman,
     {0, 0, 10, 0},
     -1,
     3,
     0x0003,
     0x0003},
    {Intrinsic::tt_bh_sfpu_abs,
     Intrinsic::tt_wh_sfpu_abs,
     {0, 0, 10, 0},
     -1,
     3,
     0x0003,
     0x0003},
    {Intrinsic::tt_bh_sfpu_lz,
     Intrinsic::tt_wh_sfpu_lz,
     {0, 0, 10, 0},
     -1,
     3,
     0x5555,
     0x5555},
    {Intrinsic::tt_bh_sfpu_cast,
     Intrinsic::tt_wh_sfpu_cast,
     {0, 0, 10, 0},
     -1,
     3,
     0x000f,
     0x0003},
    {Intrinsic::tt_bh_sfpu_setexp_i,
     Intrinsic::tt_wh_sfpu_setexp_i,
     {0, 0, 10, 255, 1},
     -1,
     4,
     0x000a,
     0x000a},
    {Intrinsic::tt_bh_sfpu_setexp_v,
     Intrinsic::tt_wh_sfpu_setexp_v,
     {0, 0, 10, 0, 0},
     -1,
     4,
     0x0005,
     0x0005},
    {Intrinsic::tt_bh_sfpu_setman_i,
     Intrinsic::tt_wh_sfpu_setman_i,
     {0, 0, 10, 4095, 1},
     -1,
     4,
     0x0002,
     0x0002},
    {Intrinsic::tt_bh_sfpu_setman_v,
     Intrinsic::tt_wh_sfpu_setman_v,
     {0, 0, 10, 0, 0},
     -1,
     4,
     0x0001,
     0x0001},
    {Intrinsic::tt_bh_sfpu_setsgn_i,
     Intrinsic::tt_wh_sfpu_setsgn_i,
     {0, 0, 10, 1, 1},
     -1,
     4,
     0x0002,
     0x0002},
    {Intrinsic::tt_bh_sfpu_setsgn_v,
     Intrinsic::tt_wh_sfpu_setsgn_v,
     {0, 0, 10, 0, 0},
     -1,
     4,
     0x0001,
     0x0001},
    {Intrinsic::tt_bh_sfpu_iadd_i,
     Intrinsic::tt_wh_sfpu_iadd_i,
     {0, 0, -2048, 1},
     -1,
     3,
     0x0222,
     0x0222},
    {Intrinsic::tt_bh_sfpu_shft_i,
     Intrinsic::tt_wh_sfpu_shft_i,
     {0, 0, 2047, 1},
     -1,
     3,
     0x00aa,
     0x0002},
    {Intrinsic::tt_bh_sfpu_and, Intrinsic::tt_wh_sfpu_and, {0, 0, 15}},
    {Intrinsic::tt_bh_sfpu_or, Intrinsic::tt_wh_sfpu_or, {0, 0, 15}},
    {Intrinsic::tt_bh_sfpu_xor, Intrinsic::tt_wh_sfpu_xor, {0, 0, 15}},
    {Intrinsic::tt_bh_sfpu_not, Intrinsic::tt_wh_sfpu_not, {0, 0}},
    {Intrinsic::tt_bh_sfpu_shft_v,
     Intrinsic::tt_wh_sfpu_shft_v,
     {0, 0, 15, 0},
     -1,
     3,
     0x0005,
     0x0001},
    {Intrinsic::tt_bh_sfpu_shft2,
     Intrinsic::tt_wh_sfpu_shft2,
     {0, 0, 1, 0, 3},
     -1,
     4,
     0x0018,
     0x0008},
    {Intrinsic::tt_bh_sfpu_shft2_copy4,
     Intrinsic::tt_wh_sfpu_shft2_copy4,
     {0, 1, 2, 3, 0, 1, 2, 3, 0},
     -1,
     8,
     0x0003,
     0x0003},
    {Intrinsic::tt_bh_sfpu_shft2_rotate4,
     Intrinsic::tt_wh_sfpu_shft2_rotate4,
     {0, 1, 2, 3, 0, 1, 2, 3, 15}},
    {Intrinsic::tt_bh_sfpu_stochrnd_i,
     Intrinsic::tt_wh_sfpu_stochrnd_i,
     {0, 0, 15, 0, 0, 0},
     -1,
     4,
     0x00ff,
     0x00ff},
    {Intrinsic::tt_bh_sfpu_stochrnd_v,
     Intrinsic::tt_wh_sfpu_stochrnd_v,
     {0, 0, 15, 14, 4, 0},
     -1,
     4,
     0x0030,
     0x0030},
    {Intrinsic::tt_bh_sfpu_lutfp32_6r,
     Intrinsic::tt_wh_sfpu_lutfp32_6r,
     {7, 7, 0, 1, 2, 3, 4, 5, 6, 0},
     -1,
     9,
     0x00dd,
     0x00dd},
    {Intrinsic::tt_bh_sfpu_lutfp32_3r,
     Intrinsic::tt_wh_sfpu_lutfp32_3r,
     {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 10},
     -1,
     16,
     0x4400,
     0x4400},
    {Intrinsic::tt_bh_sfpu_lut,
     Intrinsic::tt_wh_sfpu_lut,
     {4, 4, 0, 1, 2, 3, 0},
     -1,
     6,
     0x0011,
     0x0011},
    {Intrinsic::tt_bh_sfpu_swap,
     Intrinsic::tt_wh_sfpu_swap,
     {0, 1, 0, 1, 0},
     -1,
     4,
     0x03ff,
     0x03ff},
    {Intrinsic::tt_bh_sfpu_transp,
     Intrinsic::tt_wh_sfpu_transp,
     {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3}},
    {Intrinsic::tt_bh_sfpu_setcc,
     Intrinsic::tt_wh_sfpu_setcc,
     {15, 0, 0},
     -1,
     2,
     0x0055,
     0x0055},
    {Intrinsic::tt_bh_sfpu_gt,
     Intrinsic::not_intrinsic,
     {0, 15, 1},
     -1,
     2,
     0x0002,
     0},
    {Intrinsic::tt_bh_sfpu_le,
     Intrinsic::not_intrinsic,
     {0, 15, 1},
     -1,
     2,
     0x0002,
     0},
    {Intrinsic::tt_bh_sfpu_encc,
     Intrinsic::tt_wh_sfpu_encc,
     {3, 10},
     -1,
     1,
     0x0707,
     0x0707},
    {Intrinsic::tt_bh_sfpu_pushc,
     Intrinsic::tt_wh_sfpu_pushc,
     {0, 0},
     -1,
     1,
     0x0003,
     0x0001},
    {Intrinsic::tt_bh_sfpu_popc,
     Intrinsic::tt_wh_sfpu_popc,
     {0, 0},
     -1,
     1,
     0x001b,
     0x001b},
    {Intrinsic::tt_bh_sfpu_config_creg,
     Intrinsic::tt_wh_sfpu_config_creg,
     {0, 11, 0, 0},
     -1,
     3,
     0x0101,
     0x0101},
    {Intrinsic::tt_bh_sfpu_config_lane,
     Intrinsic::tt_wh_sfpu_config_lane,
     {0, 0x5555, 8},
     -1,
     2,
     0x0100,
     0x0100},
    {Intrinsic::tt_bh_sfpu_nop, Intrinsic::tt_wh_sfpu_nop, {}},
    {Intrinsic::tt_bh_sfpu_compc, Intrinsic::tt_wh_sfpu_compc, {}},
    {Intrinsic::tt_bh_sfpu_config_reset,
     Intrinsic::tt_wh_sfpu_config_reset,
     {}},
};

struct BoundABICase {
  const BoundFormCase *Form;
  Intrinsic::ID ID;
  SFPUArchitecture Architecture;
};

SmallVector<BoundABICase, 97> boundABICases() {
  SmallVector<BoundABICase, 97> Cases;
  for (const auto &Form : BoundForms) {
    Cases.push_back({&Form, Form.Blackhole, SFPUArchitecture::Blackhole});
    if (Form.Wormhole != Intrinsic::not_intrinsic)
      Cases.push_back({&Form, Form.Wormhole, SFPUArchitecture::Wormhole});
  }
  return Cases;
}

SmallVector<RISCV::TensixBoundValue, 17>
boundConstants(ArrayRef<int64_t> Values) {
  SmallVector<RISCV::TensixBoundValue, 17> Arguments;
  for (int64_t Value : Values)
    Arguments.push_back({RISCV::TensixBoundValueKind::Constant, Value});
  return Arguments;
}

class RISCVTensixPhysicalABITest : public testing::TestWithParam<BoundABICase> {
};

TEST_P(RISCVTensixPhysicalABITest, ReceivesArchitectureAndExactDeclaration) {
  const auto &Case = GetParam();
  LLVMContext Context;
  Module M("physical-abi", Context);
  Function *F = Intrinsic::getOrInsertDeclaration(&M, Case.ID);
  SCOPED_TRACE(F->getName().str());
  EXPECT_EQ(RISCV::getTensixBoundSFPUArchitecture(Case.ID), Case.Architecture);
  EXPECT_TRUE(F->getReturnType()->isVoidTy());
  ASSERT_EQ(F->arg_size(), Case.Form->Arguments.size());
  EXPECT_TRUE(F->isConvergent());
  EXPECT_TRUE(F->cannotDuplicate());
  for (unsigned I = 0; I != F->arg_size(); ++I) {
    EXPECT_TRUE(F->getArg(I)->getType()->isIntegerTy(32));
    EXPECT_EQ(F->hasParamAttribute(I, Attribute::ImmArg),
              int(I) != Case.Form->DynamicOffset);
  }
  auto Contract = RISCV::getTensixBoundSFPUContract(
      Case.ID, boundConstants(Case.Form->Arguments));
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_EQ(Contract->getIntrinsicID(), Case.ID);
  EXPECT_EQ(Contract->getArgumentCount(), Case.Form->Arguments.size());
  EXPECT_FALSE(Contract->getArchitecturalEffects().empty());
}

TEST_P(RISCVTensixPhysicalABITest, OnlyDstOffsetReceivesDynamicScalar) {
  const auto &Case = GetParam();
  for (unsigned I = 0; I != Case.Form->Arguments.size(); ++I) {
    SCOPED_TRACE(I);
    auto Arguments = boundConstants(Case.Form->Arguments);
    Arguments[I] = {RISCV::TensixBoundValueKind::DynamicScalar};
    auto Contract = RISCV::getTensixBoundSFPUContract(Case.ID, Arguments);
    const bool Expected = int(I) == Case.Form->DynamicOffset;
    EXPECT_EQ(bool(Contract), Expected);
    if (!Contract)
      consumeError(Contract.takeError());
  }
  auto Extra = boundConstants(Case.Form->Arguments);
  Extra.push_back({RISCV::TensixBoundValueKind::Constant, 0});
  auto Contract = RISCV::getTensixBoundSFPUContract(Case.ID, Extra);
  EXPECT_FALSE(bool(Contract));
  if (!Contract)
    consumeError(Contract.takeError());
}

TEST_P(RISCVTensixPhysicalABITest, ReceivesExactNativeModesAndRejectsReserved) {
  const auto &Case = GetParam();
  if (Case.Form->ModeArgument < 0)
    return;
  const uint16_t Allowed = Case.Architecture == SFPUArchitecture::Wormhole
                               ? Case.Form->WormholeModes
                               : Case.Form->BlackholeModes;
  for (int64_t Mode = -1; Mode <= 16; ++Mode) {
    SCOPED_TRACE(Mode);
    auto Arguments = boundConstants(Case.Form->Arguments);
    Arguments[Case.Form->ModeArgument].Constant = Mode;
    auto Contract = RISCV::getTensixBoundSFPUContract(Case.ID, Arguments);
    const bool Expected = Mode >= 0 && Mode < 16 && (Allowed & (1u << Mode));
    EXPECT_EQ(bool(Contract), Expected);
    if (!Contract)
      consumeError(Contract.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(BothArchitectures, RISCVTensixPhysicalABITest,
                         testing::ValuesIn(boundABICases()),
                         [](const testing::TestParamInfo<BoundABICase> &Info) {
                           std::string Name =
                               Intrinsic::getBaseName(Info.param.ID).str();
                           std::replace(Name.begin(), Name.end(), '.', '_');
                           return Name;
                         });

TEST(RISCVTensix, PhysicalArchitectureDoesNotClassifyOtherIngress) {
  for (Intrinsic::ID ID : std::initializer_list<Intrinsic::ID>{
           Intrinsic::not_intrinsic, Intrinsic::fma, Intrinsic::tt_nop})
    EXPECT_FALSE(RISCV::getTensixBoundSFPUArchitecture(ID));
  // Unsupported WH instructions have no declaration; they cannot acquire a
  // BH receiver simply because their suffix names a known native instruction.
  for (StringRef Name :
       {"llvm.tt.wh.sfpu.arecip", "llvm.tt.wh.sfpu.gt", "llvm.tt.wh.sfpu.le"})
    EXPECT_EQ(Intrinsic::lookupIntrinsicID(Name), Intrinsic::not_intrinsic);
}

TEST(RISCVTensix, PhysicalDstAddressModesRemainArchitectureSpecific) {
  for (auto [ID, Offset, AddressMode, Maximum] :
       {std::tuple{Intrinsic::tt_wh_sfpu_load, 2u, 3u, 3},
        std::tuple{Intrinsic::tt_wh_sfpu_store, 1u, 2u, 3},
        std::tuple{Intrinsic::tt_bh_sfpu_load, 2u, 3u, 7},
        std::tuple{Intrinsic::tt_bh_sfpu_store, 1u, 2u, 7}}) {
    for (int Mode = -1; Mode <= 8; ++Mode) {
      auto Arguments = Offset == 2 ? boundConstants({0, 0, 1023, 0, 4})
                                   : boundConstants({0, 1023, 0, 4});
      Arguments[AddressMode].Constant = Mode;
      auto Contract = RISCV::getTensixBoundSFPUContract(ID, Arguments);
      EXPECT_EQ(bool(Contract), Mode >= 0 && Mode <= Maximum);
      if (!Contract)
        consumeError(Contract.takeError());
    }
    for (int64_t InvalidOffset : {-1, 1024}) {
      auto Arguments = Offset == 2 ? boundConstants({0, 0, 0, 0, 4})
                                   : boundConstants({0, 0, 0, 4});
      Arguments[Offset].Constant = InvalidOffset;
      auto Contract = RISCV::getTensixBoundSFPUContract(ID, Arguments);
      EXPECT_FALSE(bool(Contract));
      if (!Contract)
        consumeError(Contract.takeError());
    }
  }
}

TEST(RISCVTensix, BoundDstLoadTransferUsesActualABIArguments) {
  using namespace RISCV;
  for (TensixBoundValue Offset :
       {TensixBoundValue{TensixBoundValueKind::Constant, 37},
        TensixBoundValue{TensixBoundValueKind::Constant, 1023},
        TensixBoundValue{TensixBoundValueKind::DynamicScalar}}) {
    auto Arguments = boundConstants({5, 5, 0, 7, 2});
    Arguments[2] = Offset;
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_load, Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_TRUE(Contract->getDstTransferFacts());
    const auto &Facts = *Contract->getDstTransferFacts();
    const auto *Load = std::get_if<TensixSFPUDstLoadTransfer>(&Facts.Packet);
    ASSERT_NE(Load, nullptr);
    EXPECT_EQ(Load->DestinationValue, 0u);
    ASSERT_TRUE(Load->OldValue);
    EXPECT_EQ(*Load->OldValue, 1u);
    EXPECT_EQ(Facts.OffsetArgument, 2u);
    EXPECT_EQ(Facts.AddressMode, 7u);
    EXPECT_EQ(Facts.Format, TensixSFPUDstFormat::BFloat16);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    EXPECT_TRUE(Contract->getTransferFacts().empty());
    EXPECT_EQ(Contract->getArgumentCount(), 5u);
  }
}

TEST(RISCVTensix, BoundDstStoreTransferHasNoNumericalResult) {
  using namespace RISCV;
  for (TensixBoundValue Offset :
       {TensixBoundValue{TensixBoundValueKind::Constant, 37},
        TensixBoundValue{TensixBoundValueKind::Constant, 1023},
        TensixBoundValue{TensixBoundValueKind::DynamicScalar}}) {
    auto Arguments = boundConstants({3, 0, 7, 2});
    Arguments[1] = Offset;
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_store, Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_TRUE(Contract->getDstTransferFacts());
    const auto &Facts = *Contract->getDstTransferFacts();
    const auto *Store = std::get_if<TensixSFPUDstStoreTransfer>(&Facts.Packet);
    ASSERT_NE(Store, nullptr);
    EXPECT_EQ(Store->SourceValue, 0u);
    EXPECT_EQ(Facts.OffsetArgument, 1u);
    EXPECT_EQ(Facts.AddressMode, 7u);
    EXPECT_EQ(Facts.Format, TensixSFPUDstFormat::BFloat16);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    EXPECT_TRUE(Contract->getTransferFacts().empty());
    EXPECT_EQ(Contract->getArgumentCount(), 4u);
  }
}

// Native format 3 is ToFP32, not BF16 or Raw32. Its side transfer must
// preserve the actual old/source and scalar-offset ABI identities.
TEST(RISCVTensix, BoundFloat32DstLoadTransferRetainsActualABIArguments) {
  using namespace RISCV;
  for (TensixBoundValue Offset :
       {TensixBoundValue{TensixBoundValueKind::Constant, 37},
        TensixBoundValue{TensixBoundValueKind::Constant, 1023},
        TensixBoundValue{TensixBoundValueKind::DynamicScalar}}) {
    SCOPED_TRACE(Offset.Kind == TensixBoundValueKind::DynamicScalar);
    auto Arguments = boundConstants({5, 5, 0, 7, 3});
    Arguments[2] = Offset;
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_load, Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_TRUE(Contract->getDstTransferFacts())
        << "Blackhole slot7 format3 has no received FP32 Dst load transfer";
    const auto &Facts = *Contract->getDstTransferFacts();
    const auto *Load = std::get_if<TensixSFPUDstLoadTransfer>(&Facts.Packet);
    ASSERT_NE(Load, nullptr);
    EXPECT_EQ(Load->DestinationValue, 0u);
    ASSERT_TRUE(Load->OldValue);
    EXPECT_EQ(*Load->OldValue, 1u);
    EXPECT_EQ(Facts.OffsetArgument, 2u);
    ASSERT_LT(Facts.OffsetArgument, Contract->getArgumentCount());
    EXPECT_EQ(Contract->getArgumentRoles()[Facts.OffsetArgument],
              TensixBoundArgumentRole::DstOffset);
    EXPECT_EQ(Facts.AddressMode, 7u);
    EXPECT_EQ(Facts.Format, TensixSFPUDstFormat::Float32);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    EXPECT_TRUE(Contract->getTransferFacts().empty());
    EXPECT_EQ(Contract->getArgumentCount(), 5u);
    EXPECT_EQ(Contract->getIntrinsicID(), Intrinsic::tt_bh_sfpu_load);
  }
}

TEST(RISCVTensix, BoundFloat32DstStoreTransferHasNoNumericalResult) {
  using namespace RISCV;
  for (TensixBoundValue Offset :
       {TensixBoundValue{TensixBoundValueKind::Constant, 37},
        TensixBoundValue{TensixBoundValueKind::Constant, 1023},
        TensixBoundValue{TensixBoundValueKind::DynamicScalar}}) {
    SCOPED_TRACE(Offset.Kind == TensixBoundValueKind::DynamicScalar);
    auto Arguments = boundConstants({3, 0, 7, 3});
    Arguments[1] = Offset;
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_store, Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_TRUE(Contract->getDstTransferFacts())
        << "Blackhole slot7 format3 has no received FP32 Dst store transfer";
    const auto &Facts = *Contract->getDstTransferFacts();
    const auto *Store = std::get_if<TensixSFPUDstStoreTransfer>(&Facts.Packet);
    ASSERT_NE(Store, nullptr);
    EXPECT_EQ(Store->SourceValue, 0u);
    EXPECT_EQ(Facts.OffsetArgument, 1u);
    ASSERT_LT(Facts.OffsetArgument, Contract->getArgumentCount());
    EXPECT_EQ(Contract->getArgumentRoles()[Facts.OffsetArgument],
              TensixBoundArgumentRole::DstOffset);
    EXPECT_EQ(Facts.AddressMode, 7u);
    EXPECT_EQ(Facts.Format, TensixSFPUDstFormat::Float32);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    EXPECT_TRUE(Contract->getTransferFacts().empty());
    EXPECT_EQ(Contract->getArgumentCount(), 4u);
    EXPECT_EQ(Contract->getIntrinsicID(), Intrinsic::tt_bh_sfpu_store);
  }
}

TEST(RISCVTensix, BoundFloat32DstTransferAdmitsAddressModeZeroOnBothTargets) {
  using namespace RISCV;
  struct Case {
    Intrinsic::ID ID;
    unsigned Mode;
  };
  constexpr Case Cases[] = {
      {Intrinsic::tt_bh_sfpu_load, 0},  {Intrinsic::tt_bh_sfpu_load, 3},
      {Intrinsic::tt_bh_sfpu_load, 6},  {Intrinsic::tt_bh_sfpu_store, 0},
      {Intrinsic::tt_bh_sfpu_store, 3}, {Intrinsic::tt_bh_sfpu_store, 6},
      {Intrinsic::tt_wh_sfpu_load, 0},  {Intrinsic::tt_wh_sfpu_load, 3},
      {Intrinsic::tt_wh_sfpu_store, 0}, {Intrinsic::tt_wh_sfpu_store, 3},
  };
  for (const Case &Item : Cases) {
    SCOPED_TRACE(Intrinsic::getName(Item.ID));
    SCOPED_TRACE(Item.Mode);
    const bool Load = Item.ID == Intrinsic::tt_bh_sfpu_load ||
                      Item.ID == Intrinsic::tt_wh_sfpu_load;
    auto Arguments = Load ? boundConstants({0, 0, 0, Item.Mode, 3})
                          : boundConstants({0, 0, Item.Mode, 3});
    Arguments[Load ? 2 : 1] = {TensixBoundValueKind::DynamicScalar};
    auto Contract = getTensixBoundSFPUContract(Item.ID, Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    EXPECT_TRUE(Contract->getTransferFacts().empty());
    const auto &Transfer = Contract->getDstTransferFacts();
    if (Item.Mode != 0) {
      EXPECT_FALSE(Transfer);
      continue;
    }
    ASSERT_TRUE(Transfer);
    EXPECT_EQ(Transfer->AddressMode, 0u);
    EXPECT_EQ(Transfer->Format, TensixSFPUDstFormat::Float32);
    EXPECT_EQ(Transfer->OffsetArgument, Load ? 2u : 1u);
    EXPECT_EQ(Contract->getArgumentRoles()[Transfer->OffsetArgument],
              TensixBoundArgumentRole::DstOffset);
    EXPECT_TRUE(Transfer->RequiresLaneEnable);
    EXPECT_TRUE(Transfer->RequiresConditionState);
    EXPECT_TRUE(Transfer->RequiresRowMaskState);
    if (Load) {
      const auto *Packet =
          std::get_if<TensixSFPUDstLoadTransfer>(&Transfer->Packet);
      ASSERT_NE(Packet, nullptr);
      EXPECT_EQ(Packet->DestinationValue, 0u);
      EXPECT_EQ(Packet->OldValue, 1u);
    } else {
      const auto *Packet =
          std::get_if<TensixSFPUDstStoreTransfer>(&Transfer->Packet);
      ASSERT_NE(Packet, nullptr);
      EXPECT_EQ(Packet->SourceValue, 0u);
    }
  }
}

// Native format4 copies the complete datum without format3's denormal flush.
// Query the actual encoded format and retain its distinct typed semantics.
TEST(RISCVTensix, BoundRaw32DstTransferRetainsActualABIArguments) {
  using namespace RISCV;
  for (Intrinsic::ID ID :
       {Intrinsic::tt_bh_sfpu_load, Intrinsic::tt_bh_sfpu_store}) {
    const bool Load = ID == Intrinsic::tt_bh_sfpu_load;
    SCOPED_TRACE(Intrinsic::getName(ID));
    for (TensixBoundValue Offset :
         {TensixBoundValue{TensixBoundValueKind::Constant, 37},
          TensixBoundValue{TensixBoundValueKind::Constant, 1023},
          TensixBoundValue{TensixBoundValueKind::DynamicScalar}}) {
      auto Arguments =
          Load ? boundConstants({5, 5, 0, 7, 4}) : boundConstants({5, 0, 7, 4});
      const unsigned OffsetIndex = Load ? 2 : 1;
      Arguments[OffsetIndex] = Offset;
      auto Contract = getTensixBoundSFPUContract(ID, Arguments);
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_TRUE(Contract->getDstTransferFacts())
          << "Blackhole slot7 format4 has no received Raw32 side transfer";
      const auto &Facts = *Contract->getDstTransferFacts();
      EXPECT_EQ(Facts.Format, TensixSFPUDstFormat::Raw32);
      EXPECT_EQ(Facts.OffsetArgument, OffsetIndex);
      EXPECT_EQ(Contract->getArgumentRoles()[Facts.OffsetArgument],
                TensixBoundArgumentRole::DstOffset);
      EXPECT_EQ(Facts.AddressMode, 7u);
      EXPECT_TRUE(Facts.RequiresLaneEnable);
      EXPECT_TRUE(Facts.RequiresConditionState);
      EXPECT_TRUE(Facts.RequiresRowMaskState);
      EXPECT_TRUE(Contract->getTransferFacts().empty());
      EXPECT_EQ(Contract->getArgumentCount(), Load ? 5u : 4u);
      if (Load) {
        const auto *Transfer =
            std::get_if<TensixSFPUDstLoadTransfer>(&Facts.Packet);
        ASSERT_NE(Transfer, nullptr);
        EXPECT_EQ(Transfer->DestinationValue, 0u);
        ASSERT_TRUE(Transfer->OldValue);
        EXPECT_EQ(*Transfer->OldValue, 1u);
      } else {
        const auto *Transfer =
            std::get_if<TensixSFPUDstStoreTransfer>(&Facts.Packet);
        ASSERT_NE(Transfer, nullptr);
        EXPECT_EQ(Transfer->SourceValue, 0u);
      }
    }
  }
}

TEST(RISCVTensix, Raw32DstTransferDoesNotWidenArchitectureOrAddressMode) {
  using namespace RISCV;
  for (bool Wormhole : {false, true}) {
    SCOPED_TRACE(Wormhole);
    for (unsigned Mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
      if (Wormhole && Mode > 3)
        continue;
      SCOPED_TRACE(Mode);
      for (bool Load : {false, true}) {
        auto ID = Load ? (Wormhole ? Intrinsic::tt_wh_sfpu_load
                                   : Intrinsic::tt_bh_sfpu_load)
                       : (Wormhole ? Intrinsic::tt_wh_sfpu_store
                                   : Intrinsic::tt_bh_sfpu_store);
        auto Arguments = Load ? boundConstants({0, 0, 0, Mode, 4})
                              : boundConstants({0, 0, Mode, 4});
        Arguments[Load ? 2 : 1] = {TensixBoundValueKind::DynamicScalar};
        auto Contract = getTensixBoundSFPUContract(ID, Arguments);
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        EXPECT_FALSE(Contract->getDstTransferFacts());
      }
    }
  }
}

TEST(RISCVTensix, BoundDstTransferRemainsAbsentForUnreceivedForms) {
  for (Intrinsic::ID ID :
       {Intrinsic::tt_bh_sfpu_load, Intrinsic::tt_bh_sfpu_store,
        Intrinsic::tt_wh_sfpu_load, Intrinsic::tt_wh_sfpu_store}) {
    bool Load =
        ID == Intrinsic::tt_bh_sfpu_load || ID == Intrinsic::tt_wh_sfpu_load;
    bool Wormhole =
        ID == Intrinsic::tt_wh_sfpu_load || ID == Intrinsic::tt_wh_sfpu_store;
    auto Query = [&](unsigned Mode, unsigned Format) {
      auto Arguments = Load ? boundConstants({0, 0, 0, Mode, Format})
                            : boundConstants({0, 0, Mode, Format});
      Arguments[Load ? 2 : 1] = {RISCV::TensixBoundValueKind::DynamicScalar};
      return RISCV::getTensixBoundSFPUContract(ID, Arguments);
    };
    for (unsigned Mode : {0u, 1u, 2u, 3u, 4u, 5u, 6u}) {
      if (Wormhole && Mode > 3)
        continue;
      auto Contract = Query(Mode, 2);
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      EXPECT_FALSE(Contract->getDstTransferFacts());
      EXPECT_TRUE(Contract->getTransferFacts().empty());
    }
    for (unsigned Format : {0u, 1u, 3u, 4u, 5u, 6u, 8u, 14u, 15u}) {
      if (!Wormhole && (Format == 3 || Format == 4))
        continue; // Received BH slot7 forms have FP32 and Raw32 positives.
      auto Contract = Query(Wormhole ? 3 : 7, Format);
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      EXPECT_FALSE(Contract->getDstTransferFacts());
    }
  }
}

TEST(RISCVTensix, PhysicalRoundingChecksDescaleAndArchitecture) {
  for (auto [ID, Immediate, MaximumRound] :
       {std::tuple{Intrinsic::tt_wh_sfpu_stochrnd_i, true, 1},
        std::tuple{Intrinsic::tt_wh_sfpu_stochrnd_v, false, 1},
        std::tuple{Intrinsic::tt_bh_sfpu_stochrnd_i, true, 2},
        std::tuple{Intrinsic::tt_bh_sfpu_stochrnd_v, false, 2}}) {
    for (int Round = -1; Round <= 3; ++Round) {
      auto Contract = RISCV::getTensixBoundSFPUContract(
          ID, boundConstants({0, 0, 9, 0, 4, Round}));
      EXPECT_EQ(bool(Contract), Round >= 0 && Round <= MaximumRound);
      if (!Contract)
        consumeError(Contract.takeError());
    }
    if (!Immediate)
      continue;
    for (int64_t Mode = 0; Mode != 8; ++Mode)
      for (int64_t Descale : {-1, 0, 1, 31, 32}) {
        auto Contract = RISCV::getTensixBoundSFPUContract(
            ID, boundConstants({0, 0, 9, Descale, Mode, 0}));
        const bool Expected = (Mode == 4 || Mode == 5)
                                  ? Descale >= 0 && Descale <= 31
                                  : Descale == 0;
        EXPECT_EQ(bool(Contract), Expected);
        if (!Contract)
          consumeError(Contract.takeError());
      }
  }
}

TEST(RISCVTensix, PhysicalWormholePRNGRetainsReadWriteState) {
  for (auto [ID, Arguments] :
       {std::pair{Intrinsic::tt_wh_sfpu_mov_prng_advance,
                  boundConstants({0, 0})},
        std::pair{Intrinsic::tt_wh_sfpu_stochrnd_i,
                  boundConstants({0, 0, 9, 0, 0, 0})},
        std::pair{Intrinsic::tt_wh_sfpu_stochrnd_v,
                  boundConstants({0, 0, 9, 10, 4, 0})}}) {
    auto Contract = RISCV::getTensixBoundSFPUContract(ID, Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    unsigned PRNGEffects = 0;
    for (const auto &Effect : Contract->getArchitecturalEffects()) {
      if (const auto *State = std::get_if<SFPUState>(&Effect.Resource)) {
        if (*State != SFPUState::PRNG)
          continue;
        ++PRNGEffects;
        EXPECT_EQ(Effect.Access, SFPUAccess::ReadWrite);
      }
      // Special selector 9 names the PRNG action, not a hidden CReg9 source.
      if (ID == Intrinsic::tt_wh_sfpu_mov_prng_advance)
        EXPECT_FALSE(std::holds_alternative<RISCV::TensixFixedRegisterRef>(
            Effect.Resource));
    }
    EXPECT_EQ(PRNGEffects, 1u);
  }
}

TEST(RISCVTensix, BoundMoveHasWholeRegisterFootprint) {
  using namespace RISCV;
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_mov_all, {{TensixBoundValueKind::UnboundRegister},
                                      {TensixBoundValueKind::UnboundRegister}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  ASSERT_EQ(Contract->getRegisterArguments().size(), 2u);
  EXPECT_EQ(Contract->getRegisterArguments()[0].Role,
            TensixBoundArgumentRole::WriteLReg);
  EXPECT_EQ(Contract->getRegisterArguments()[1].Role,
            TensixBoundArgumentRole::ReadRegister);
  for (const auto &Argument : Contract->getRegisterArguments())
    EXPECT_EQ(Argument.Footprint, TensixSFPURegisterFootprint::WholeRegister);
}

TEST(RISCVTensix, SFPURegisterGeometryUsesNativeClasses) {
  for (uint32_t Number = 0; Number != 16; ++Number) {
    SCOPED_TRACE(Number);
    auto Geometry = RISCV::getTensixSFPURegisterGeometry(Number);
    ASSERT_TRUE(bool(Geometry)) << toString(Geometry.takeError());
    EXPECT_EQ(Geometry->NumLanes, 32u);
    EXPECT_EQ(Geometry->BitsPerLane, 32u);
  }
}

TEST(RISCVTensix, SFPURegisterGeometryRejectsInvalidNumbers) {
  for (uint32_t Number : {16u, UINT32_MAX}) {
    auto Geometry = RISCV::getTensixSFPURegisterGeometry(Number);
    ASSERT_FALSE(bool(Geometry));
    consumeError(Geometry.takeError());
  }
}

TEST(RISCVTensix, BoundPRNGAdvanceHasOnlyDestinationAndOldWithStateEffects) {
  using namespace RISCV;
  const TensixBoundValue Unbound{TensixBoundValueKind::UnboundRegister};
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_mov_prng_advance, {Unbound, Unbound});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  ASSERT_EQ(Contract->getArgumentCount(), 2u);
  const auto Registers = Contract->getRegisterArguments();
  ASSERT_EQ(Registers.size(), 2u);
  EXPECT_EQ(Registers[0].Argument, 0u);
  EXPECT_EQ(Registers[0].Role, TensixBoundArgumentRole::WriteLReg);
  EXPECT_EQ(Registers[1].Argument, 1u);
  EXPECT_EQ(Registers[1].Role, TensixBoundArgumentRole::OldDestination);
  for (const auto &Register : Registers) {
    EXPECT_FALSE(Register.FixedRegister);
    EXPECT_EQ(Register.AllowedRegisters,
              (SmallVector<uint32_t, 16>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(Register.Footprint, TensixSFPURegisterFootprint::StateDependent);
  }
  const auto Constraints = Contract->getRegisterConstraints();
  ASSERT_EQ(Constraints.size(), 1u);
  EXPECT_EQ(Constraints[0].Kind, TensixBoundConstraintKind::SameLocation);
  EXPECT_EQ(Constraints[0].FirstArgument, 0u);
  EXPECT_EQ(Constraints[0].SecondArgument, 1u);

  // Special selector 9 is not CReg9. The two register arguments and four
  // architectural resources exhaust the action's accesses.
  ASSERT_EQ(Contract->getArchitecturalEffects().size(), 6u);
  unsigned RegisterEffects = 0, CC = 0, Config = 0, Issue = 0, PRNG = 0;
  for (const auto &Effect : Contract->getArchitecturalEffects()) {
    EXPECT_FALSE(
        std::holds_alternative<TensixFixedRegisterRef>(Effect.Resource));
    if (const auto *Argument =
            std::get_if<TensixBoundArgumentRef>(&Effect.Resource)) {
      ++RegisterEffects;
      ASSERT_LT(Argument->Argument, 2u);
      EXPECT_EQ(Effect.Access, Argument->Argument == 0
                                   ? TensixSFPUAccess::Write
                                   : TensixSFPUAccess::Read);
      continue;
    }
    const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
    ASSERT_NE(State, nullptr);
    switch (*State) {
    case TensixSFPUState::CC:
      ++CC;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      break;
    case TensixSFPUState::Configuration:
      ++Config;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      break;
    case TensixSFPUState::Issue:
      ++Issue;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
      break;
    case TensixSFPUState::PRNG:
      ++PRNG;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
      break;
    default:
      ADD_FAILURE() << "PRNG advance acquired an unrelated resource";
    }
  }
  EXPECT_EQ(RegisterEffects, 2u);
  EXPECT_EQ(CC, 1u);
  EXPECT_EQ(Config, 1u);
  EXPECT_EQ(Issue, 1u);
  EXPECT_EQ(PRNG, 1u);
  EXPECT_TRUE(
      std::holds_alternative<std::monostate>(Contract->getLaneControl()));
  // The bounded initializedness helper has no PRNG-content input. It must
  // not claim a copy or an immediately initialized numerical value.
  EXPECT_TRUE(Contract->getTransferFacts().empty());
}

TEST(RISCVTensix, BoundPRNGAdvanceRejectsUnauthoredSourceOrBrokenBinding) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (uint32_t Number = 0; Number != 8; ++Number) {
    SCOPED_TRACE(Number);
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_mov_prng_advance,
                                   {Constant(Number), Constant(Number)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  }
  for (const SmallVector<TensixBoundValue, 3> &Args :
       {SmallVector<TensixBoundValue, 3>{Constant(0), Constant(1)},
        {Constant(9), Constant(9)},
        {Constant(0), Constant(9)},
        {Constant(-1), Constant(-1)},
        {Constant(0)},
        {Constant(0), Constant(0), Constant(9)},
        {{TensixBoundValueKind::DynamicScalar}, Constant(0)}}) {
    auto Rejected = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_mov_prng_advance, Args);
    ASSERT_FALSE(bool(Rejected));
    consumeError(Rejected.takeError());
  }
}

TEST(RISCVTensix, BoundRoundingRetainsPRNGForDeterministicModes) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Rounding : {0, 1, 2}) {
    for (auto ID :
         {Intrinsic::tt_bh_sfpu_stochrnd_i, Intrinsic::tt_bh_sfpu_stochrnd_v}) {
      bool Immediate = ID == Intrinsic::tt_bh_sfpu_stochrnd_i;
      auto Contract = getTensixBoundSFPUContract(
          ID, {Constant(0), Constant(0), Constant(9), Constant(0),
               Constant(Immediate ? 0 : 4), Constant(Rounding)});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      EXPECT_TRUE(
          any_of(Contract->getArchitecturalEffects(), [](const auto &E) {
            auto *State = std::get_if<TensixSFPUState>(&E.Resource);
            return State && *State == TensixSFPUState::PRNG &&
                   E.Access == TensixSFPUAccess::ReadWrite;
          }));
    }
  }
}

TEST(RISCVTensix, BoundTransferFactsDistinguishImmediateHalfWrites) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (auto [Mode, Kind, Generated, Demanded] :
       {std::tuple<int64_t, TensixSFPUTransferKind, uint32_t, uint32_t>{
            0, TensixSFPUTransferKind::ImmediateFull, 0xffffffffu, 0u},
        {1, TensixSFPUTransferKind::ImmediateFull, 0xffffffffu, 0u},
        {2, TensixSFPUTransferKind::ImmediateFull, 0xffffffffu, 0u},
        {4, TensixSFPUTransferKind::ImmediateFull, 0xffffffffu, 0u},
        {8, TensixSFPUTransferKind::ImmediateUpperHalf, 0xffff0000u,
         0x0000ffffu},
        {10, TensixSFPUTransferKind::ImmediateLowerHalf, 0x0000ffffu,
         0xffff0000u}}) {
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_loadi,
        {Constant(0), Constant(0), Constant(0), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Facts = Contract->getTransferFacts().front();
    EXPECT_EQ(Facts.Kind, Kind);
    EXPECT_EQ(Facts.DestinationArgument, 0u);
    EXPECT_EQ(Facts.OldArgument, 1u);
    EXPECT_FALSE(Facts.SourceArgument);
    EXPECT_EQ(Facts.GeneratedBits, Generated);
    EXPECT_EQ(Facts.GuaranteedWriteBits, Generated);
    EXPECT_EQ(Facts.DemandedOldBits, Demanded);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
  }
  auto Move = getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_mov_all,
                                         {Constant(0), Constant(1)});
  ASSERT_TRUE(bool(Move)) << toString(Move.takeError());
  ASSERT_EQ(Move->getTransferFacts().size(), 1u);
  const auto &MoveFacts = Move->getTransferFacts().front();
  EXPECT_EQ(MoveFacts.Kind, TensixSFPUTransferKind::Identity);
  EXPECT_EQ(MoveFacts.GeneratedBits, 0xffffffffu);
  EXPECT_FALSE(MoveFacts.RequiresLaneEnable);
  EXPECT_FALSE(MoveFacts.RequiresConditionState);
  EXPECT_FALSE(MoveFacts.RequiresRowMaskState);
  EXPECT_EQ(MoveFacts.DestinationArgument, 0u);
  EXPECT_EQ(MoveFacts.SourceArgument, 1u);
  EXPECT_FALSE(MoveFacts.OldArgument);
}

TEST(RISCVTensix, BoundMaskedMoveDeclaresSourceOldAndLaneFacts) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Mode : {0, 1}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_mov,
        {Constant(1), Constant(1), Constant(2), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getRegisterArguments().size(), 3u);
    EXPECT_EQ(Contract->getRegisterArguments()[0].Argument, 0u);
    EXPECT_EQ(Contract->getRegisterArguments()[0].Role,
              TensixBoundArgumentRole::WriteLReg);
    EXPECT_EQ(Contract->getRegisterArguments()[1].Argument, 1u);
    EXPECT_EQ(Contract->getRegisterArguments()[1].Role,
              TensixBoundArgumentRole::OldDestination);
    EXPECT_EQ(Contract->getRegisterArguments()[2].Argument, 2u);
    EXPECT_EQ(Contract->getRegisterArguments()[2].Role,
              TensixBoundArgumentRole::ReadRegister);

    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Facts = Contract->getTransferFacts().front();
    EXPECT_EQ(Facts.Kind, Mode == 0 ? TensixSFPUTransferKind::Identity
                                    : TensixSFPUTransferKind::LaneWise);
    if (Mode == 0) {
      EXPECT_TRUE(Facts.Inputs.empty());
      EXPECT_EQ(Facts.SourceArgument, 2u);
    } else {
      EXPECT_FALSE(Facts.SourceArgument);
      ASSERT_EQ(Facts.Inputs.size(), 1u);
      EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
      EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
    }
    EXPECT_EQ(Facts.DestinationArgument, 0u);
    EXPECT_EQ(Facts.OldArgument, 1u);
    EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
    EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
    EXPECT_EQ(Facts.DemandedOldBits, 0u);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
  }
}

TEST(RISCVTensix, BoundIAddWithoutConditionUpdateDeclaresLaneWiseInputs) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Mode : {4, 6}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_iadd,
        {Constant(1), Constant(1), Constant(2), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Facts = Contract->getTransferFacts().front();
    EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
    EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
    EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
    // Active lanes consume the old destination numerically; they preserve
    // none of its bits. Inactive-lane preservation is a separate fact.
    EXPECT_EQ(Facts.DemandedOldBits, 0u);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    ASSERT_EQ(Facts.Inputs.size(), 2u);
    EXPECT_EQ(Facts.Inputs[0].Argument, 1u);
    EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
    EXPECT_EQ(Facts.Inputs[1].Argument, 2u);
    EXPECT_EQ(Facts.Inputs[1].DemandedBits, 0xffffffffu);
  }
}

TEST(RISCVTensix, BoundIAddConditionUpdatesRemainUnsupportedTransfers) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Mode : {0, 2, 8, 10}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_iadd,
        {Constant(1), Constant(1), Constant(2), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    EXPECT_TRUE(Contract->getTransferFacts().empty());
  }
}

TEST(RISCVTensix, BoundIAddWithoutConditionUpdateRetainsNativeEffects) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Mode : {4, 6}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_iadd,
        {Constant(1), Constant(1), Constant(2), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getRegisterArguments().size(), 3u);
    EXPECT_EQ(Contract->getRegisterArguments()[1].Role,
              TensixBoundArgumentRole::OldDestination);
    EXPECT_TRUE(any_of(Contract->getRegisterConstraints(), [](const auto &C) {
      return C.Kind == TensixBoundConstraintKind::SameLocation &&
             C.FirstArgument == 0 && C.SecondArgument == 1;
    }));
    bool ReadsCondition = false, ReadsConfiguration = false;
    bool ReadsAndWritesIssue = false;
    for (const auto &Effect : Contract->getArchitecturalEffects()) {
      EXPECT_FALSE(
          std::holds_alternative<TensixFixedRegisterRef>(Effect.Resource));
      const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
      if (!State)
        continue;
      EXPECT_NE(*State, TensixSFPUState::PRNG);
      if (*State == TensixSFPUState::CC) {
        ReadsCondition = true;
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      } else if (*State == TensixSFPUState::Configuration) {
        ReadsConfiguration = true;
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      } else if (*State == TensixSFPUState::Issue) {
        ReadsAndWritesIssue = true;
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
      }
    }
    EXPECT_TRUE(ReadsCondition);
    EXPECT_TRUE(ReadsConfiguration);
    EXPECT_TRUE(ReadsAndWritesIssue);
  }
}

TEST(RISCVTensix, BoundConditionControlKeepsEnableAndFlagsDistinct) {
  using namespace RISCV;
  using Enable = TensixSFPUConditionEnableUpdate;
  struct Case {
    int64_t Value, Mode;
    Enable Action;
    bool Flags;
  };
  for (auto C :
       {Case{0, 2, Enable::Disable, true}, Case{1, 2, Enable::Enable, true},
        Case{0, 10, Enable::Disable, false}, Case{3, 10, Enable::Enable, true},
        Case{0, 0, Enable::Preserve, true}, Case{0, 1, Enable::Invert, true},
        Case{0, 8, Enable::Preserve, false},
        Case{0, 9, Enable::Invert, false}}) {
    SCOPED_TRACE(C.Mode);
    SCOPED_TRACE(C.Value);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_encc, {{TensixBoundValueKind::Constant, C.Value},
                                     {TensixBoundValueKind::Constant, C.Mode}});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    auto *Update =
        std::get_if<TensixSFPUConditionUpdate>(&Contract->getLaneControl());
    ASSERT_NE(Update, nullptr);
    EXPECT_EQ(Update->Enable, C.Action);
    EXPECT_EQ(Update->Flags, C.Flags);
  }
}

TEST(RISCVTensix, BoundConditionControlReadsAndWritesConditionState) {
  using namespace RISCV;
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_encc, {{TensixBoundValueKind::Constant, 0},
                                   {TensixBoundValueKind::Constant, 0}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_TRUE(any_of(Contract->getArchitecturalEffects(), [](const auto &E) {
    auto *State = std::get_if<TensixSFPUState>(&E.Resource);
    return State && *State == TensixSFPUState::CC &&
           E.Access == TensixSFPUAccess::ReadWrite;
  }));
}

TEST(RISCVTensix, BoundConfigurationResetRetainsColumnPredicateAndOwner) {
  using namespace RISCV;
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_config_reset, {});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  auto *Reset =
      std::get_if<TensixSFPULaneConfigReset>(&Contract->getLaneControl());
  ASSERT_NE(Reset, nullptr);
  EXPECT_EQ(Reset->Columns, 8u);
  EXPECT_TRUE(any_of(Contract->getArchitecturalEffects(), [](const auto &E) {
    auto *State = std::get_if<TensixSFPUState>(&E.Resource);
    return State && *State == TensixSFPUState::Configuration &&
           E.Access == TensixSFPUAccess::ReadWrite;
  }));
}

TEST(RISCVTensix, BoundLaneConfigAssignmentRetainsActualSourceAndColumns) {
  using namespace RISCV;
  struct Selection {
    uint32_t Mask;
    uint32_t Columns;
  };
  for (auto ID : {Intrinsic::tt_bh_sfpu_config_lane,
                  Intrinsic::tt_wh_sfpu_config_lane})
    for (auto Case : {Selection{0, 0}, Selection{1, 1},
                      Selection{0x5554, 0xfe}, Selection{0xaaaa, 0},
                      Selection{0xffff, 0xff}}) {
      SCOPED_TRACE(unsigned(ID));
      SCOPED_TRACE(Case.Mask);
      auto Contract = getTensixBoundSFPUContract(
          ID, boundConstants({0, Case.Mask, 8}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      const auto *Assign =
          std::get_if<TensixSFPULaneConfigAssign>(&Contract->getLaneControl());
      ASSERT_NE(Assign, nullptr);
      EXPECT_EQ(Assign->SourceValue, 0u);
      EXPECT_EQ(Assign->Columns, 8u);
      EXPECT_EQ(Assign->SelectedColumns, Case.Columns);
      EXPECT_EQ(Assign->ValueBits, 18u);
      auto Registers = Contract->getRegisterArguments();
      ASSERT_EQ(Registers.size(), 1u);
      EXPECT_EQ(Registers[0].Argument, Assign->SourceValue);
      EXPECT_EQ(Registers[0].Role, TensixBoundArgumentRole::FixedGroupRead);
      EXPECT_EQ(Registers[0].FixedRegister, 0u);
      EXPECT_EQ(Registers[0].Footprint,
                TensixSFPURegisterFootprint::StateDependent);
      EXPECT_TRUE(Contract->getTransferFacts().empty());
      EXPECT_FALSE(Contract->getDstTransferFacts());
      ASSERT_TRUE(Contract->getConfigurationFacts());
      EXPECT_TRUE(Contract->getConfigurationFacts()->Requirements.empty());
    }
}

TEST(RISCVTensix, BoundLaneConfigAssignmentDoesNotWritePacketsOrCondition) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_bh_sfpu_config_lane,
                  Intrinsic::tt_wh_sfpu_config_lane}) {
    auto Contract =
        getTensixBoundSFPUContract(ID, boundConstants({0, 0x5554, 8}));
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_TRUE(std::holds_alternative<TensixSFPULaneConfigAssign>(
        Contract->getLaneControl()));
    unsigned Source = 0, Condition = 0, Configuration = 0, Issue = 0;
    for (const auto &Effect : Contract->getArchitecturalEffects()) {
      if (const auto *Argument =
              std::get_if<TensixBoundArgumentRef>(&Effect.Resource)) {
        ++Source;
        EXPECT_EQ(Argument->Argument, 0u);
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
        continue;
      }
      const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
      ASSERT_NE(State, nullptr);
      switch (*State) {
      case TensixSFPUState::CC:
        ++Condition;
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
        break;
      case TensixSFPUState::Configuration:
        ++Configuration;
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
        break;
      case TensixSFPUState::Issue:
        ++Issue;
        EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
        break;
      default:
        ADD_FAILURE() << "LaneConfig acquired an unrelated resource";
      }
    }
    EXPECT_EQ(Source, 1u);
    EXPECT_EQ(Condition, 1u);
    EXPECT_EQ(Configuration, 1u);
    EXPECT_EQ(Issue, 1u);
  }
}

TEST(RISCVTensix, BoundLaneConfigRejectsForeignStageModeAndDynamicSelector) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_bh_sfpu_config_lane,
                  Intrinsic::tt_wh_sfpu_config_lane}) {
    for (int64_t Stage : {1, 2, 7, 8}) {
      SCOPED_TRACE(Stage);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({Stage, 0x5554, 8}));
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
    }
    for (int64_t Mode : {0, 1, 9, 10, 12, 14}) {
      SCOPED_TRACE(Mode);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({0, 0x5554, Mode}));
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
    }
    for (unsigned Field : {0u, 1u, 2u}) {
      SCOPED_TRACE(Field);
      auto Arguments = boundConstants({0, 0x5554, 8});
      Arguments[Field] = {TensixBoundValueKind::DynamicScalar};
      auto Contract = getTensixBoundSFPUContract(ID, Arguments);
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
    }
  }
}

TEST(RISCVTensix, LaneConfigValueDecodeRequiresActualKnownBits) {
  using namespace RISCV;
  auto Unknown = decodeTensixSFPULaneConfig(0, 0xf000);
  EXPECT_FALSE(Unknown.EnableFP16AInf.has_value());
  EXPECT_FALSE(Unknown.DisableBackdoorLoad.has_value());
  EXPECT_FALSE(Unknown.EnableDestIndex.has_value());
  EXPECT_FALSE(Unknown.CaptureDefaultDestIndex.has_value());
  EXPECT_FALSE(Unknown.BlockDestWriteFromSFPU.has_value());
  EXPECT_FALSE(Unknown.BlockSFPUReadFromDest.has_value());
  EXPECT_FALSE(Unknown.DestReadColumnExchange.has_value());
  EXPECT_FALSE(Unknown.DestWriteColumnExchange.has_value());
  EXPECT_FALSE(Unknown.ExchangeSourceBAndC.has_value());
  EXPECT_FALSE(Unknown.BlockDestMoveEven.has_value());
  EXPECT_FALSE(Unknown.BlockDestMoveOdd.has_value());
  for (auto Row : Unknown.RowMask)
    EXPECT_FALSE(Row.has_value());
  EXPECT_EQ(Unknown.ReservedKnownBits, 0u);
  EXPECT_EQ(Unknown.ReservedValueBits, 0u);

  auto Marker = decodeTensixSFPULaneConfig(0x3ffff, 0xf000);
  EXPECT_EQ(Marker.EnableFP16AInf, false);
  EXPECT_EQ(Marker.DisableBackdoorLoad, false);
  EXPECT_EQ(Marker.EnableDestIndex, false);
  EXPECT_EQ(Marker.CaptureDefaultDestIndex, false);
  EXPECT_EQ(Marker.BlockDestWriteFromSFPU, false);
  EXPECT_EQ(Marker.BlockSFPUReadFromDest, false);
  EXPECT_EQ(Marker.DestReadColumnExchange, false);
  EXPECT_EQ(Marker.DestWriteColumnExchange, false);
  EXPECT_EQ(Marker.ExchangeSourceBAndC, false);
  EXPECT_EQ(Marker.BlockDestMoveEven, false);
  EXPECT_EQ(Marker.BlockDestMoveOdd, false);
  for (auto Row : Marker.RowMask)
    EXPECT_EQ(Row, true);
  EXPECT_EQ(Marker.ReservedKnownBits, 0x30800u);
  EXPECT_EQ(Marker.ReservedValueBits, 0u);
}

TEST(RISCVTensix, LaneConfigValueDecodeKeepsIndependentPartialFields) {
  using namespace RISCV;
  using Field = std::optional<bool> TensixSFPULaneConfigFields::*;
  const Field Fields[] = {
      &TensixSFPULaneConfigFields::EnableFP16AInf,
      &TensixSFPULaneConfigFields::DisableBackdoorLoad,
      &TensixSFPULaneConfigFields::EnableDestIndex,
      &TensixSFPULaneConfigFields::CaptureDefaultDestIndex,
      &TensixSFPULaneConfigFields::BlockDestWriteFromSFPU,
      &TensixSFPULaneConfigFields::BlockSFPUReadFromDest,
      &TensixSFPULaneConfigFields::DestReadColumnExchange,
      &TensixSFPULaneConfigFields::DestWriteColumnExchange,
      &TensixSFPULaneConfigFields::ExchangeSourceBAndC,
      &TensixSFPULaneConfigFields::BlockDestMoveEven,
      &TensixSFPULaneConfigFields::BlockDestMoveOdd};
  for (unsigned Bit = 0; Bit != 11; ++Bit) {
    SCOPED_TRACE(Bit);
    auto Set = decodeTensixSFPULaneConfig(1u << Bit, 0xffffffff);
    auto Clear = decodeTensixSFPULaneConfig(1u << Bit, ~(1u << Bit));
    for (auto [Index, Member] : enumerate(Fields)) {
      EXPECT_EQ((Set.*Member).has_value(), Index == Bit);
      EXPECT_EQ((Clear.*Member).has_value(), Index == Bit);
      if (Index == Bit) {
        EXPECT_EQ(Set.*Member, true);
        EXPECT_EQ(Clear.*Member, false);
      }
    }
    for (auto Row : Set.RowMask)
      EXPECT_FALSE(Row.has_value());
  }
  for (unsigned Row = 0; Row != 4; ++Row) {
    SCOPED_TRACE(Row);
    auto Bits = decodeTensixSFPULaneConfig(1u << (12 + Row), 0xffffffff);
    for (unsigned Other = 0; Other != 4; ++Other) {
      EXPECT_EQ(Bits.RowMask[Other].has_value(), Other == Row);
      if (Other == Row)
        EXPECT_EQ(Bits.RowMask[Other], true);
    }
    EXPECT_FALSE(Bits.EnableDestIndex.has_value());
  }
}

TEST(RISCVTensix, LaneConfigValueDecodePreservesReservedAndDropsHighPayload) {
  using namespace RISCV;
  auto Full = decodeTensixSFPULaneConfig(0xffffffff, 0xffffffff);
  auto Low = decodeTensixSFPULaneConfig(0x3ffff, 0x3ffff);
  EXPECT_EQ(Full.ReservedKnownBits, 0x30800u);
  EXPECT_EQ(Full.ReservedValueBits, 0x30800u);
  EXPECT_EQ(Full.RowMask, Low.RowMask);
  EXPECT_EQ(Full.EnableDestIndex, Low.EnableDestIndex);
  auto Partial = decodeTensixSFPULaneConfig(0x10000, 0x3ffff);
  EXPECT_EQ(Partial.ReservedKnownBits, 0x10000u);
  EXPECT_EQ(Partial.ReservedValueBits, 0x10000u);
  EXPECT_FALSE(Partial.EnableDestIndex.has_value());
  auto High = decodeTensixSFPULaneConfig(0xfffc0000, 0xffffffff);
  EXPECT_EQ(High.ReservedKnownBits, 0u);
  EXPECT_EQ(High.ReservedValueBits, 0u);
  for (auto Row : High.RowMask)
    EXPECT_FALSE(Row.has_value());
}

TEST(RISCVTensix, BoundStateDependentFootprintIsNotWholeRegister) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  struct Case {
    Intrinsic::ID ID;
    SmallVector<TensixBoundValue, 12> Arguments;
  };
  const Case Cases[] = {
      {Intrinsic::tt_bh_sfpu_mov,
       {Constant(0), Constant(0), Constant(1), Constant(0)}},
      {Intrinsic::tt_bh_sfpu_mul,
       {Constant(0), Constant(0), Constant(1), Constant(2), Constant(0)}},
      {Intrinsic::tt_bh_sfpu_loadi,
       {Constant(0), Constant(0), Constant(0x1234), Constant(8)}},
      {Intrinsic::tt_bh_sfpu_loadi,
       {Constant(0), Constant(0), Constant(0x1234), Constant(10)}},
      {Intrinsic::tt_bh_sfpu_transp,
       {Constant(0), Constant(1), Constant(2), Constant(3), Constant(4),
        Constant(5), Constant(6), Constant(7), Constant(0), Constant(1),
        Constant(2), Constant(3)}},
      {Intrinsic::tt_bh_sfpu_config_creg,
       {Constant(0),
        {TensixBoundValueKind::UnboundRegister},
        Constant(0x5555),
        Constant(8)}},
  };
  for (const auto &Test : Cases) {
    SCOPED_TRACE(unsigned(Test.ID));
    auto Contract = getTensixBoundSFPUContract(Test.ID, Test.Arguments);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_FALSE(Contract->getRegisterArguments().empty());
    for (const auto &Argument : Contract->getRegisterArguments())
      EXPECT_EQ(Argument.Footprint,
                TensixSFPURegisterFootprint::StateDependent);
  }
}

class RISCVTensixOptimizationTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmPrinter();
  }

  static std::unique_ptr<TargetMachine> createMachine() {
    Triple TT("riscv32-unknown-unknown");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      return nullptr;
    return std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TT, "generic-rv32", "+xtttensixbh", TargetOptions(), std::nullopt,
        std::nullopt, CodeGenOptLevel::Default));
  }

  static Expected<std::string> emit(StringRef IR,
                                    RISCV::TensixOptimizationOptions Options) {
    auto TM = createMachine();
    if (!TM)
      return createStringError("could not create test RISC-V target machine");
    if (Error E = RISCV::setTensixOptimizationOptions(*TM, Options))
      return std::move(E);
    LLVMContext Context;
    SMDiagnostic Diagnostic;
    auto M = parseAssemblyString(IR, Diagnostic, Context);
    if (!M)
      return createStringError(Diagnostic.getMessage());
    M->setTargetTriple(TM->getTargetTriple());
    M->setDataLayout(TM->createDataLayout());
    SmallVector<char, 0> Bytes;
    raw_svector_ostream Output(Bytes);
    legacy::PassManager Passes;
    if (TM->addPassesToEmitFile(Passes, Output, nullptr,
                                CodeGenFileType::AssemblyFile))
      return createStringError("test target cannot emit assembly");
    Passes.run(*M);
    return std::string(Bytes.begin(), Bytes.end());
  }
};

TEST_F(RISCVTensixOptimizationTest, IndependentMachinesKeepValueOwnedOptions) {
  auto First = createMachine();
  auto Second = createMachine();
  auto Untouched = createMachine();
  ASSERT_TRUE(First && Second && Untouched);
  auto Defaults = RISCV::getTensixOptimizationOptions(*Untouched);
  ASSERT_TRUE(bool(Defaults)) << toString(Defaults.takeError());
  for (bool Scheduling : {false, true}) {
    RISCV::TensixOptimizationOptions Options{Scheduling};
    EXPECT_FALSE(bool(RISCV::setTensixOptimizationOptions(*First, Options)));
    EXPECT_FALSE(bool(RISCV::setTensixOptimizationOptions(
        *Second, {!Options.LatencyScheduling})));
    auto A = RISCV::getTensixOptimizationOptions(*First);
    auto B = RISCV::getTensixOptimizationOptions(*Second);
    auto C = RISCV::getTensixOptimizationOptions(*Untouched);
    ASSERT_TRUE(bool(A)) << toString(A.takeError());
    ASSERT_TRUE(bool(B)) << toString(B.takeError());
    ASSERT_TRUE(bool(C)) << toString(C.takeError());
    EXPECT_EQ(A->LatencyScheduling, Options.LatencyScheduling);
    EXPECT_EQ(B->LatencyScheduling, !Options.LatencyScheduling);
    EXPECT_EQ(C->LatencyScheduling, Defaults->LatencyScheduling);
  }
}

TEST_F(RISCVTensixOptimizationTest, RejectsForeignTargetEvenWithRISCVTriple) {
  class ForeignMachine final : public TargetMachine {
  public:
    explicit ForeignMachine(const Target &T, StringRef TripleName)
        : TargetMachine(T, "", Triple(TripleName), "", "", TargetOptions()) {}
  };
  Target Foreign;
  for (StringRef TripleName :
       {"x86_64-unknown-unknown", "riscv32-unknown-unknown"}) {
    ForeignMachine TM(Foreign, TripleName);
    Error E = RISCV::setTensixOptimizationOptions(TM, {});
    ASSERT_TRUE(bool(E));
    EXPECT_NE(toString(std::move(E)).find("RISC-V target machine"),
              std::string::npos);
    auto Options = RISCV::getTensixOptimizationOptions(TM);
    ASSERT_FALSE(bool(Options));
    EXPECT_NE(toString(Options.takeError()).find("RISC-V target machine"),
              std::string::npos);
  }
}

constexpr StringLiteral SchedulingOptionsIR = R"(
  declare void @llvm.tt.bh.sfpu.encc(i32 immarg, i32 immarg)
  declare void @llvm.tt.bh.sfpu.mov.all(i32 immarg, i32 immarg)
  declare void @llvm.tt.bh.sfpu.add(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
  declare void @llvm.tt.bh.sfpu.iadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
  declare void @llvm.tt.bh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
  define void @kernel() "target-features"="+xtttensixbh" "tensix-executor"="trisc1" {
    call void @llvm.tt.bh.sfpu.encc(i32 3, i32 10)
    call void @llvm.tt.bh.sfpu.mov.all(i32 0, i32 9)
    call void @llvm.tt.bh.sfpu.add(i32 0, i32 0, i32 9, i32 10, i32 1)
    call void @llvm.tt.bh.sfpu.iadd(i32 0, i32 0, i32 9, i32 4)
    call void @llvm.tt.bh.sfpu.mov.all(i32 4, i32 10)
    call void @llvm.tt.bh.sfpu.store(i32 0, i32 0, i32 0, i32 4)
    call void @llvm.tt.bh.sfpu.store(i32 4, i32 1, i32 0, i32 4)
    ret void
  }
)";

TEST_F(RISCVTensixOptimizationTest, DisabledSchedulingStillRepairsHazards) {
  for (bool Enable : {true, false, true}) {
    auto Assembly = emit(SchedulingOptionsIR, {Enable});
    ASSERT_TRUE(bool(Assembly)) << toString(Assembly.takeError());
    EXPECT_EQ(StringRef(*Assembly).count(".word"), Enable ? 7u : 8u);
    // The off configuration must still include the mandatory SFPNOP.
    EXPECT_EQ(StringRef(*Assembly).count("0x3c000002"), Enable ? 0u : 1u);
  }
}

#if LLVM_ENABLE_THREADS
TEST_F(RISCVTensixOptimizationTest, ConcurrentRequestsUseDistinctOptions) {
  auto Enabled = std::async(std::launch::async, [] {
    return emit(SchedulingOptionsIR, {true});
  });
  auto Disabled = std::async(std::launch::async, [] {
    return emit(SchedulingOptionsIR, {false});
  });
  auto A = Enabled.get();
  auto B = Disabled.get();
  ASSERT_TRUE(bool(A)) << toString(A.takeError());
  ASSERT_TRUE(bool(B)) << toString(B.takeError());
  EXPECT_EQ(StringRef(*A).count(".word"), 7u);
  EXPECT_EQ(StringRef(*B).count(".word"), 8u);
  EXPECT_EQ(StringRef(*A).count("0x3c000002"), 0u);
  EXPECT_EQ(StringRef(*B).count("0x3c000002"), 1u);
}
#endif

TEST(RISCVTensix, BoundAbsRetainsUnarySourceAndOldDestinationTie) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_abs, Intrinsic::tt_bh_sfpu_abs}) {
    SCOPED_TRACE(static_cast<unsigned>(ID));
    for (int64_t Mode : {0, 1}) {
      SCOPED_TRACE(Mode);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({1, 1, 2, Mode}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getArgumentCount(), 4u);
      auto Roles = Contract->getArgumentRoles();
      ASSERT_EQ(Roles.size(), 4u);
      EXPECT_EQ(Roles[0], TensixBoundArgumentRole::WriteLReg);
      EXPECT_EQ(Roles[1], TensixBoundArgumentRole::OldDestination);
      EXPECT_EQ(Roles[2], TensixBoundArgumentRole::ReadRegister);
      EXPECT_EQ(Roles[3], TensixBoundArgumentRole::LogicalImmediate);
      auto Registers = Contract->getRegisterArguments();
      ASSERT_EQ(Registers.size(), 3u);
      for (unsigned I = 0; I != 3; ++I) {
        EXPECT_EQ(Registers[I].Argument, I);
        EXPECT_EQ(Registers[I].Role, Roles[I]);
        EXPECT_FALSE(Registers[I].FixedRegister);
      }
      auto Constraints = Contract->getRegisterConstraints();
      ASSERT_EQ(Constraints.size(), 1u);
      EXPECT_EQ(Constraints[0].Kind, TensixBoundConstraintKind::SameLocation);
      EXPECT_EQ(Constraints[0].FirstArgument, 0u);
      EXPECT_EQ(Constraints[0].SecondArgument, 1u);
      for (const auto &Effect : Contract->getArchitecturalEffects())
        EXPECT_FALSE(
            std::holds_alternative<TensixFixedRegisterRef>(Effect.Resource));
    }
  }
}

TEST(RISCVTensix, BoundAbsDeclaresLaneWiseSourceAndInactiveOldPreservation) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_abs, Intrinsic::tt_bh_sfpu_abs}) {
    SCOPED_TRACE(static_cast<unsigned>(ID));
    for (int64_t Mode : {0, 1}) {
      SCOPED_TRACE(Mode);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({1, 1, 2, Mode}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Facts = Contract->getTransferFacts().front();
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
      EXPECT_EQ(Facts.DestinationArgument, 0u);
      EXPECT_EQ(Facts.OldArgument, 1u);
      EXPECT_FALSE(Facts.SourceArgument);
      ASSERT_EQ(Facts.Inputs.size(), 1u);
      EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
      EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
      EXPECT_TRUE(Facts.SourceLanes.empty());
      EXPECT_TRUE(Facts.LaneSources.empty());
      // These are conditional initializedness facts, not an always-zero sign.
      // Float negative NaNs and integer INT_MIN retain a set sign bit.
      EXPECT_EQ(Facts.DemandedOldBits, 0u);
      EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
      EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
      EXPECT_TRUE(Facts.RequiresLaneEnable);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      EXPECT_TRUE(Facts.RequiresConditionState);
      EXPECT_TRUE(Facts.RequiresRowMaskState);
    }
  }
}

TEST(RISCVTensix, BoundAbsDestIndexRequirementNamesOnlyActualDestination) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_abs, Intrinsic::tt_bh_sfpu_abs}) {
    SCOPED_TRACE(static_cast<unsigned>(ID));
    for (int64_t Mode : {0, 1}) {
      SCOPED_TRACE(Mode);
      for (int64_t Dst : {1, 4}) {
        SCOPED_TRACE(Dst);
        int64_t Src = Dst == 1 ? 4 : 1;
        auto Contract = getTensixBoundSFPUContract(
            ID, boundConstants({Dst, Dst, Src, Mode}));
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        const auto &Configuration = Contract->getConfigurationFacts();
        ASSERT_TRUE(bool(Configuration));
        ASSERT_EQ(Configuration->Requirements.size(), 1u);
        const auto &Requirement = Configuration->Requirements.front();
        EXPECT_EQ(Requirement.Argument, 0u);
        EXPECT_EQ(Requirement.Field,
                  TensixSFPUConfigurationField::EnableDestIndex);
        EXPECT_EQ(Requirement.Registers,
                  (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
      }
    }
  }
}

TEST(RISCVTensix, BoundSignedImmediateHasExplicitConfigurationContract) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_loadi, Intrinsic::tt_bh_sfpu_loadi}) {
    SCOPED_TRACE(static_cast<unsigned>(ID));
    for (int64_t Dst : {0, 4, 7}) {
      SCOPED_TRACE(Dst);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({Dst, Dst, 0xffff, 4}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      // SFPLOADI SHORT sign-extends all 16 immediate bits and is exempt from
      // TEN-2932, including destinations L4..L7. It still needs actual lanes.
      const auto &Configuration = Contract->getConfigurationFacts();
      ASSERT_TRUE(bool(Configuration));
      EXPECT_TRUE(Configuration->Requirements.empty());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Transfer = Contract->getTransferFacts().front();
      EXPECT_EQ(Transfer.Kind, TensixSFPUTransferKind::ImmediateFull);
      EXPECT_EQ(Transfer.GeneratedBits, 0xffffffffu);
      EXPECT_EQ(Transfer.DemandedOldBits, 0u);
      EXPECT_TRUE(Transfer.RequiresLaneEnable);
      EXPECT_TRUE(Transfer.PreservesInactiveLanes);
      EXPECT_TRUE(Transfer.RequiresConditionState);
      EXPECT_TRUE(Transfer.RequiresRowMaskState);
    }
  }
}

TEST(RISCVTensix, BoundSetSignImmediateRetainsExactABIAndRejectsBrokenFields) {
  using namespace RISCV;
  for (auto ID :
       {Intrinsic::tt_wh_sfpu_setsgn_i, Intrinsic::tt_bh_sfpu_setsgn_i}) {
    for (int64_t Sign : {0, 1}) {
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({1, 1, 2, Sign, 1}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getArgumentCount(), 5u);
      const TensixBoundArgumentRole Roles[] = {
          TensixBoundArgumentRole::WriteLReg,
          TensixBoundArgumentRole::OldDestination,
          TensixBoundArgumentRole::ReadRegister,
          TensixBoundArgumentRole::LogicalImmediate,
          TensixBoundArgumentRole::LogicalImmediate};
      EXPECT_EQ(Contract->getArgumentRoles(),
                ArrayRef<TensixBoundArgumentRole>(Roles));
      ASSERT_EQ(Contract->getRegisterArguments().size(), 3u);
      for (unsigned I = 0; I != 3; ++I)
        EXPECT_EQ(Contract->getRegisterArguments()[I].Argument, I);
      auto Constraints = Contract->getRegisterConstraints();
      ASSERT_EQ(Constraints.size(), 1u);
      EXPECT_EQ(Constraints[0].Kind, TensixBoundConstraintKind::SameLocation);
      EXPECT_EQ(Constraints[0].FirstArgument, 0u);
      EXPECT_EQ(Constraints[0].SecondArgument, 1u);
    }
    const int64_t Rejected[][5] = {
        {1, 2, 3, 0, 1}, // Broken dst/old tie.
        {1, 1, 3, 2, 1}, // Sign is an actual u1, not a masked u16.
        {1, 1, 3, 0, 0}, // Vector mode is outside immediate ABI.
        {1, 1, 3, 0, 2}};
    for (const auto &Tuple : Rejected) {
      auto Contract = getTensixBoundSFPUContract(
          ID,
          boundConstants({Tuple[0], Tuple[1], Tuple[2], Tuple[3], Tuple[4]}));
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
    }
  }
}

TEST(RISCVTensix, BoundSetSignImmediateDeclaresExactMaskedIdentity) {
  using namespace RISCV;
  for (auto ID :
       {Intrinsic::tt_wh_sfpu_setsgn_i, Intrinsic::tt_bh_sfpu_setsgn_i}) {
    SCOPED_TRACE(static_cast<unsigned>(ID));
    for (int64_t Sign : {0, 1}) {
      SCOPED_TRACE(Sign);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({1, 1, 2, Sign, 1}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Facts = Contract->getTransferFacts().front();
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::MaskedIdentity);
      EXPECT_EQ(Facts.DestinationArgument, 0u);
      EXPECT_EQ(Facts.OldArgument, 1u);
      EXPECT_EQ(Facts.SourceArgument, 2u);
      EXPECT_EQ(Facts.CopiedBits, 0x7fffffffu);
      EXPECT_EQ(Facts.ConstantBits, 0x80000000u);
      EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
      EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
      EXPECT_EQ(Facts.DemandedOldBits, 0u);
      EXPECT_TRUE(Facts.Inputs.empty());
      EXPECT_TRUE(Facts.SourceLanes.empty());
      EXPECT_TRUE(Facts.LaneSources.empty());
      EXPECT_TRUE(Facts.RequiresLaneEnable);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      EXPECT_TRUE(Facts.RequiresConditionState);
      EXPECT_TRUE(Facts.RequiresRowMaskState);
    }
    auto Vector = getTensixBoundSFPUContract(
        ID == Intrinsic::tt_wh_sfpu_setsgn_i ? Intrinsic::tt_wh_sfpu_setsgn_v
                                             : Intrinsic::tt_bh_sfpu_setsgn_v,
        boundConstants({1, 1, 2, 1, 0}));
    ASSERT_TRUE(bool(Vector)) << toString(Vector.takeError());
    ASSERT_EQ(Vector->getTransferFacts().size(), 1u);
    const auto &Transfer = Vector->getTransferFacts().front();
    EXPECT_EQ(Transfer.Kind, TensixSFPUTransferKind::LaneWise);
    EXPECT_EQ(Transfer.ConstantBits, 0u);
    EXPECT_FALSE(Transfer.ImmediateValue);
    ASSERT_EQ(Transfer.Inputs.size(), 2u);
    EXPECT_EQ(Transfer.Inputs[0].DemandedBits, 0x80000000u);
    EXPECT_EQ(Transfer.Inputs[1].DemandedBits, 0x7fffffffu);
    ASSERT_TRUE(Vector->getConfigurationFacts());
    auto BrokenTie = getTensixBoundSFPUContract(
        Vector->getIntrinsicID(), boundConstants({1, 1, 2, 3, 0}));
    ASSERT_FALSE(bool(BrokenTie));
    consumeError(BrokenTie.takeError());
  }
}

TEST(RISCVTensix,
     BoundSetSignImmediateDestIndexRequirementNamesOnlyActualDestination) {
  using namespace RISCV;
  for (auto ID :
       {Intrinsic::tt_wh_sfpu_setsgn_i, Intrinsic::tt_bh_sfpu_setsgn_i})
    for (int64_t Sign : {0, 1})
      for (int64_t Dst : {1, 4}) {
        const int64_t Src = Dst == 1 ? 4 : 1;
        auto Contract = getTensixBoundSFPUContract(
            ID, boundConstants({Dst, Dst, Src, Sign, 1}));
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        const auto &Configuration = Contract->getConfigurationFacts();
        ASSERT_TRUE(bool(Configuration));
        ASSERT_EQ(Configuration->Requirements.size(), 1u);
        const auto &Requirement = Configuration->Requirements.front();
        EXPECT_EQ(Requirement.Argument, 0u);
        EXPECT_EQ(Requirement.Field,
                  TensixSFPUConfigurationField::EnableDestIndex);
        EXPECT_EQ(Requirement.Registers,
                  (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
      }
}

TEST(RISCVTensix, BoundLeadingZerosRetainsModePreciseMachineCCDependencies) {
  using namespace RISCV;
  auto Instructions = createTensixBoundMCInstrInfo();
  for (auto ID : {Intrinsic::tt_wh_sfpu_lz, Intrinsic::tt_bh_sfpu_lz})
    for (int64_t Mode : {0, 2, 4, 6, 8, 10, 12, 14}) {
      SCOPED_TRACE(static_cast<unsigned>(ID));
      SCOPED_TRACE(Mode);
      const bool PreservesCC = Mode == 0 || Mode == 4;
      auto Arguments = boundConstants({4, 4, 1, Mode});
      auto Contract = getTensixBoundSFPUContract(ID, Arguments);
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      const TensixBoundArgumentRole Roles[] = {
          TensixBoundArgumentRole::WriteLReg,
          TensixBoundArgumentRole::OldDestination,
          TensixBoundArgumentRole::ReadRegister,
          TensixBoundArgumentRole::LogicalImmediate};
      EXPECT_EQ(Contract->getArgumentRoles(),
                ArrayRef<TensixBoundArgumentRole>(Roles));
      ASSERT_EQ(Contract->getRegisterArguments().size(), 3u);
      for (unsigned I = 0; I != 3; ++I)
        EXPECT_EQ(Contract->getRegisterArguments()[I].Argument, I);
      auto Constraints = Contract->getRegisterConstraints();
      ASSERT_EQ(Constraints.size(), 1u);
      EXPECT_EQ(Constraints[0].Kind, TensixBoundConstraintKind::SameLocation);
      EXPECT_EQ(Constraints[0].FirstArgument, 0u);
      EXPECT_EQ(Constraints[0].SecondArgument, 1u);
      auto StateAccess = [&](TensixSFPUState State) {
        std::optional<TensixSFPUAccess> Access;
        for (const auto &Effect : Contract->getArchitecturalEffects())
          if (const auto *Actual =
                  std::get_if<TensixSFPUState>(&Effect.Resource))
            if (*Actual == State) {
              EXPECT_FALSE(Access);
              Access = Effect.Access;
            }
        return Access;
      };
      EXPECT_EQ(StateAccess(TensixSFPUState::CC),
                PreservesCC ? TensixSFPUAccess::Read
                            : TensixSFPUAccess::ReadWrite);
      EXPECT_EQ(StateAccess(TensixSFPUState::Configuration),
                TensixSFPUAccess::Read);
      EXPECT_EQ(StateAccess(TensixSFPUState::Issue),
                TensixSFPUAccess::ReadWrite);
      EXPECT_FALSE(StateAccess(TensixSFPUState::PRNG));
      EXPECT_TRUE(
          std::holds_alternative<std::monostate>(Contract->getLaneControl()));

      // Inspect the actual selected machine descriptor too. Correct public
      // effects must not come from filtering a descriptor's real CC def.
      auto Native = getTensixBoundNativeInstruction(ID, Arguments);
      ASSERT_TRUE(bool(Native)) << toString(Native.takeError());
      const auto &Descriptor = Instructions->get(Native->Opcode);
      EXPECT_TRUE(is_contained(Descriptor.implicit_uses(), TT_CC));
      EXPECT_TRUE(is_contained(Descriptor.implicit_uses(), TT_CONFIG));
      EXPECT_TRUE(is_contained(Descriptor.implicit_uses(), TT_ISSUE));
      EXPECT_EQ(is_contained(Descriptor.implicit_defs(), TT_CC), !PreservesCC);
      EXPECT_TRUE(is_contained(Descriptor.implicit_defs(), TT_ISSUE));
      const auto *Encoding = getTensixEncoding(Native->Opcode);
      ASSERT_NE(Encoding, nullptr);
      EXPECT_EQ(Encoding->RawOpcode, 0x81u);
      EXPECT_TRUE(Encoding->WormholeSupported);
      if (!PreservesCC) {
        EXPECT_TRUE(Contract->getTransferFacts().empty());
        EXPECT_FALSE(Contract->getConfigurationFacts());
      }
    }
}

TEST(RISCVTensix, BoundVectorDescaleUsesActualShiftAndInactiveOld) {
  using namespace RISCV;
  for (auto [ID, Rounding] :
       {std::pair{Intrinsic::tt_wh_sfpu_stochrnd_v, 0},
        std::pair{Intrinsic::tt_wh_sfpu_stochrnd_v, 1},
        std::pair{Intrinsic::tt_bh_sfpu_stochrnd_v, 0},
        std::pair{Intrinsic::tt_bh_sfpu_stochrnd_v, 1},
        std::pair{Intrinsic::tt_bh_sfpu_stochrnd_v, 2}}) {
    for (int Mode : {4, 5}) {
      SCOPED_TRACE(static_cast<unsigned>(ID));
      SCOPED_TRACE(Rounding);
      SCOPED_TRACE(Mode);
      auto Contract = getTensixBoundSFPUContract(
          ID, boundConstants({4, 4, 1, 2, Mode, Rounding}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Facts = Contract->getTransferFacts().front();
      EXPECT_EQ(Facts.DestinationArgument, 0u);
      EXPECT_EQ(Facts.OldArgument, 1u);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
      ASSERT_EQ(Facts.Inputs.size(), 2u);
      EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
      EXPECT_EQ(Facts.Inputs[0].DemandedBits,
                Mode == 5 || Rounding == 1 ? 0xffffffffu : 0x7fffffffu);
      EXPECT_EQ(Facts.Inputs[1].Argument, 3u);
      EXPECT_EQ(Facts.Inputs[1].DemandedBits, 0x1fu);
      const auto &Configuration = Contract->getConfigurationFacts();
      ASSERT_TRUE(bool(Configuration));
      ASSERT_EQ(Configuration->Requirements.size(), 1u);
      EXPECT_EQ(Configuration->Requirements.front().Field,
                TensixSFPUConfigurationField::EnableDestIndex);
    }
  }
}

TEST(RISCVTensix, BoundFloatToIntegerRoundingPreservesInactiveOld) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_stochrnd_i,
                  Intrinsic::tt_bh_sfpu_stochrnd_i}) {
    for (int Mode : {2, 3, 6, 7}) {
      for (int Round : {0, 1}) {
        SCOPED_TRACE(static_cast<unsigned>(ID));
        SCOPED_TRACE(Mode);
        SCOPED_TRACE(Round);
        auto Contract = getTensixBoundSFPUContract(
            ID, boundConstants({1, 1, 0, 0, Mode, Round}));
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
        const auto &Facts = Contract->getTransferFacts().front();
        EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
        EXPECT_EQ(Facts.DestinationArgument, 0u);
        EXPECT_EQ(Facts.OldArgument, 1u);
        EXPECT_TRUE(Facts.RequiresLaneEnable);
        EXPECT_TRUE(Facts.PreservesInactiveLanes);
        EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
        ASSERT_EQ(Facts.Inputs.size(), 1u);
        EXPECT_EQ(Facts.Inputs.front().Argument, 2u);
        EXPECT_EQ(Facts.Inputs.front().DemandedBits,
                  Mode == 2 || Mode == 6 ? 0x7fffffffu : 0xffffffffu);
        EXPECT_TRUE(any_of(Contract->getArchitecturalEffects(), [](const auto &E) {
          auto *State = std::get_if<TensixSFPUState>(&E.Resource);
          return State && *State == TensixSFPUState::PRNG &&
                 E.Access == TensixSFPUAccess::ReadWrite;
        }));
      }
    }
  }
}

TEST(RISCVTensix, BoundImmediateDescaleUsesLiteralAndInactiveOld) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_stochrnd_i,
                  Intrinsic::tt_bh_sfpu_stochrnd_i}) {
    for (int Mode : {4, 5}) {
      for (int Round : {0, 1}) {
        auto Contract = getTensixBoundSFPUContract(
            ID, boundConstants({4, 4, 1, 31, Mode, Round}));
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
        const auto &Facts = Contract->getTransferFacts().front();
        EXPECT_EQ(Facts.DestinationArgument, 0u);
        EXPECT_EQ(Facts.OldArgument, 1u);
        EXPECT_TRUE(Facts.PreservesInactiveLanes);
        ASSERT_EQ(Facts.Inputs.size(), 1u);
        EXPECT_EQ(Facts.Inputs.front().Argument, 2u);
        EXPECT_EQ(Facts.Inputs.front().DemandedBits,
                  Mode == 5 || Round == 1 ? 0xffffffffu : 0x7fffffffu);
      }
    }
  }
}

TEST(RISCVTensix, BoundLeadingZerosDeclaresExactDemandAndInactiveOnlyOld) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_lz, Intrinsic::tt_bh_sfpu_lz})
    for (int64_t Mode : {0, 4}) {
      SCOPED_TRACE(static_cast<unsigned>(ID));
      SCOPED_TRACE(Mode);
      auto Contract =
          getTensixBoundSFPUContract(ID, boundConstants({1, 1, 2, Mode}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Facts = Contract->getTransferFacts().front();
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
      EXPECT_EQ(Facts.DestinationArgument, 0u);
      EXPECT_EQ(Facts.OldArgument, 1u);
      EXPECT_FALSE(Facts.SourceArgument);
      ASSERT_EQ(Facts.Inputs.size(), 1u);
      EXPECT_EQ(Facts.Inputs.front().Argument, 2u);
      EXPECT_EQ(Facts.Inputs.front().DemandedBits,
                Mode == 0 ? 0xffffffffu : 0x7fffffffu);
      EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
      EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
      EXPECT_EQ(Facts.DemandedOldBits, 0u);
      EXPECT_EQ(Facts.CopiedBits, 0u);
      EXPECT_EQ(Facts.ConstantBits, 0u);
      EXPECT_TRUE(Facts.SourceLanes.empty());
      EXPECT_TRUE(Facts.LaneSources.empty());
      EXPECT_TRUE(Facts.RequiresLaneEnable);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      EXPECT_TRUE(Facts.RequiresConditionState);
      EXPECT_TRUE(Facts.RequiresRowMaskState);
    }
}

TEST(RISCVTensix,
     BoundLeadingZerosDestIndexRequirementNamesOnlyActualDestination) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_lz, Intrinsic::tt_bh_sfpu_lz})
    for (int64_t Mode : {0, 4})
      for (int64_t Dst : {1, 4}) {
        const int64_t Src = Dst == 1 ? 4 : 1;
        auto Contract = getTensixBoundSFPUContract(
            ID, boundConstants({Dst, Dst, Src, Mode}));
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        const auto &Configuration = Contract->getConfigurationFacts();
        ASSERT_TRUE(bool(Configuration));
        ASSERT_EQ(Configuration->Requirements.size(), 1u);
        const auto &Requirement = Configuration->Requirements.front();
        EXPECT_EQ(Requirement.Argument, 0u);
        EXPECT_EQ(Requirement.Field,
                  TensixSFPUConfigurationField::EnableDestIndex);
        EXPECT_EQ(Requirement.Registers,
                  (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
      }
}

TEST(RISCVTensix, LeadingZerosRejectsReservedModes) {
  using namespace RISCV;
  for (auto Architecture :
       {SFPUArchitecture::Wormhole, SFPUArchitecture::Blackhole}) {
    const auto ID = Architecture == SFPUArchitecture::Wormhole
                        ? Intrinsic::tt_wh_sfpu_lz
                        : Intrinsic::tt_bh_sfpu_lz;
    for (int64_t Mode : {-1, 1, 3, 5, 7, 9, 11, 13, 15, 16}) {
      SCOPED_TRACE(Mode);
      auto Bound =
          getTensixBoundSFPUContract(ID, boundConstants({1, 1, 2, Mode}));
      ASSERT_FALSE(bool(Bound));
      consumeError(Bound.takeError());
    }
    auto Arguments = boundConstants({1, 1, 2, 0});
    Arguments[3] = {TensixBoundValueKind::DynamicScalar};
    auto Bound = getTensixBoundSFPUContract(ID, Arguments);
    ASSERT_FALSE(bool(Bound));
    consumeError(Bound.takeError());
  }
}

} // namespace

TEST(RISCVTensix,
     BoundRotateMode3DestIndexRequirementNamesOnlyActualDestination) {
  using namespace RISCV;
  for (auto ID : {Intrinsic::tt_wh_sfpu_shft2, Intrinsic::tt_bh_sfpu_shft2})
    for (int64_t Dst : {1, 4}) {
      const int64_t Src = Dst == 1 ? 4 : 1;
      auto Contract = getTensixBoundSFPUContract(
          ID, boundConstants({Dst, Dst, Src, 0, 3}));
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      const auto &Configuration = Contract->getConfigurationFacts();
      ASSERT_TRUE(bool(Configuration));
      ASSERT_EQ(Configuration->Requirements.size(), 1u);
      const auto &Requirement = Configuration->Requirements.front();
      EXPECT_EQ(Requirement.Argument, 0u);
      EXPECT_EQ(Requirement.Field,
                TensixSFPUConfigurationField::EnableDestIndex);
      EXPECT_EQ(Requirement.Registers, (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
    }
  auto Shift = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_shft2, boundConstants({4, 4, 1, 0, 4}));
  ASSERT_TRUE(bool(Shift)) << toString(Shift.takeError());
  EXPECT_TRUE(Shift->getTransferFacts().empty());
  EXPECT_FALSE(Shift->getConfigurationFacts());
}
