//===-- RISCVTensixTest.cpp ----------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

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

TEST(RISCVTensix, OrdinaryLogicalContract) {
  const auto *Info = RISCV::getTensixInstructionByName("stallwait");
  ASSERT_NE(Info, nullptr);
  EXPECT_EQ(Info->getName(), "STALLWAIT");
  EXPECT_EQ(Info->IntrinsicID, Intrinsic::riscv_tt_stallwait);
  EXPECT_EQ(Info->PortIntrinsicID, Intrinsic::riscv_tt_stallwait_port);
  EXPECT_EQ(Info->MopIntrinsicID, Intrinsic::riscv_tt_stallwait_mop);
  EXPECT_EQ(Info->NumFields, 2);
  const auto *Field = RISCV::getTensixInstructionField(*Info, 0);
  ASSERT_NE(Field, nullptr);
  EXPECT_EQ(Field->getName(), "wait_res");
  EXPECT_EQ(Field->MaxValue, 32767u);
  EXPECT_EQ(RISCV::getTensixInstructionField(*Info, 2), nullptr);
  EXPECT_EQ(RISCV::getTensixInstructionByIntrinsic(
                Intrinsic::riscv_tt_stallwait_port), Info);
  EXPECT_NE(RISCV::getTensixInstructionByName("mop_cfg"), nullptr);
  EXPECT_EQ(RISCV::getTensixInstructionByName("sfpadd"), nullptr);
}

TEST(RISCVTensix, RecoverableOrdinaryRange) {
  LLVMContext Ctx;
  auto M = parse(Ctx, R"(
    declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
    define void @kernel() "target-features"="+xtttensixbh"
                          "tensix-executor"="trisc1" {
      call void @llvm.riscv.tt.stallwait(i32 32768, i32 0)
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
    declare void @llvm.riscv.tt.nop()
    define void @kernel() "target-features"="+xtttensixbh,-xtttensixbh"
                          "tensix-executor"="trisc1" {
      call void @llvm.riscv.tt.nop()
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
    declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
    define void @kernel(i32 %input) "target-features"="+xtttensixbh"
                          "tensix-executor"="trisc1" {
      %bounded = and i32 %input, 32767
      call void @llvm.riscv.tt.stallwait.port(i32 0, i32 %bounded, i32 0)
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
} // namespace
