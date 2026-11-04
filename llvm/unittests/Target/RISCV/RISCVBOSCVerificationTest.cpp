//===-- RISCVBOSCVerificationTest.cpp -------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/RISCV/RISCVBOSC.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

constexpr StringLiteral MixedContents = R"(
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @mixed(ptr %a, ptr %b, ptr %c, ptr %out, i64 %count)
    "trex-bosc-contents-ame"="1" "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %count, i64 2, i64 0)
  %x = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %y = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> %x, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %y, ptr %out, i64 %vl)
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 16, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 64, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %out, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}
)";

class RISCVBOSCVerificationTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> Machine;

  static void SetUpTestSuite() {
    LLVMInitializeRISCVTargetInfo();
    LLVMInitializeRISCVTarget();
    LLVMInitializeRISCVTargetMC();
    LLVMInitializeRISCVAsmPrinter();
  }

  void SetUp() override {
    Machine = machine("+m,+a,+f,+d,+c,+v,+zvl512b,+xboscame,+xboscsfu");
    ASSERT_NE(Machine, nullptr);
  }

  std::unique_ptr<TargetMachine> machine(StringRef Features) {
    Triple TT("riscv64-unknown-elf");
    std::string Details;
    const Target *Target = TargetRegistry::lookupTarget(TT, Details);
    if (!Target)
      return nullptr;
    return std::unique_ptr<TargetMachine>(Target->createTargetMachine(
        TT, "generic-rv64", Features, TargetOptions(), Reloc::Static,
        CodeModel::Medium, CodeGenOptLevel::Default));
  }

  std::unique_ptr<Module> parse(StringRef Source) {
    SMDiagnostic Diagnostic;
    auto M = parseAssemblyString(Source, Diagnostic, Context);
    if (!M) {
      Diagnostic.print("BOSC verifier test", errs());
      return nullptr;
    }
    M->setTargetTriple(Machine->getTargetTriple());
    M->setDataLayout(Machine->createDataLayout());
    for (Function &F : *M)
      if (!F.isDeclaration())
        F.addFnAttr(Attribute::getWithVScaleRangeArgs(Context, 8, 8));
    return M;
  }

  void expectRejectionAndRecovery(Module &Invalid, StringRef Expected) {
    Error E = RISCV::verifyBOSCModule(Invalid, *Machine);
    ASSERT_TRUE(bool(E));
    EXPECT_NE(toString(std::move(E)).find(Expected), std::string::npos);
    // Reuse the same LLVMContext and TargetMachine after every rejected input.
    // A diagnostic handler, fatal-error interceptor or subprocess cannot hide
    // a failed in-process receiver here. The next candidate must emit ELF.
    auto Valid = parse(MixedContents);
    ASSERT_NE(Valid, nullptr);
    std::string Details = toString(RISCV::verifyBOSCModule(*Valid, *Machine));
    ASSERT_TRUE(Details.empty()) << Details;
    SmallVector<char> Bytes;
    raw_svector_ostream OS(Bytes);
    legacy::PassManager PM;
    ASSERT_FALSE(Machine->addPassesToEmitFile(PM, OS, nullptr,
                                              CodeGenFileType::ObjectFile));
    PM.run(*Valid);
    ASSERT_GE(Bytes.size(), 4u);
    EXPECT_EQ(StringRef(Bytes.data(), 4), StringRef("\177ELF", 4));
  }
};

TEST_F(RISCVBOSCVerificationTest, ReturnsActualVLFailureAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  Function *F = M->getFunction("mixed");
  for (Instruction &I : instructions(F))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_bosc_rvv_contents_load) {
        II->setArgOperand(2, F->getArg(4));
        break;
      }
  expectRejectionAndRecovery(*M, "actual e32,m1 VL");
}

TEST_F(RISCVBOSCVerificationTest, ReturnsAMERoleFailureAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  for (Instruction &I : instructions(M->getFunction("mixed")))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_bosc_ame_contents_mma) {
        II->setArgOperand(2, II->getArgOperand(1));
        break;
      }
  expectRejectionAndRecovery(*M, "incompatible role or configuration");
}

TEST_F(RISCVBOSCVerificationTest, ReturnsDimensionFailureAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  for (Instruction &I : instructions(M->getFunction("mixed")))
    if (auto *II = dyn_cast<IntrinsicInst>(&I))
      if (II->getIntrinsicID() == Intrinsic::riscv_bosc_ame_setm) {
        II->setArgOperand(0, ConstantInt::get(Type::getInt64Ty(Context), 0));
        break;
      }
  expectRejectionAndRecovery(*M, "dimension request must be provably positive");
}

TEST_F(RISCVBOSCVerificationTest, ReturnsScopeFailureAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  M->getFunction("mixed")->removeFnAttr("trex-bosc-contents-rvv");
  expectRejectionAndRecovery(*M, "RVV contents scope");
}

TEST_F(RISCVBOSCVerificationTest,
       ReturnsAggregateABIFailureAndCanEmitNextModule) {
  auto M = parse(R"(
    define void @bad({ i64, [2 x target("riscv.bosc.ame.acc")] } %x)
        "trex-bosc-contents-ame"="1" {
      ret void
    }
  )");
  ASSERT_NE(M, nullptr);
  expectRejectionAndRecovery(*M, "AME contents cannot cross a function ABI");
}

TEST_F(RISCVBOSCVerificationTest, ReturnsGatherFailureAndCanEmitNextModule) {
  auto M = parse(R"(
    declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
    declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
    define void @bad() "trex-bosc-contents-rvv"="1" {
      %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
      %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> poison, i64 15, i64 %vl)
      ret void
    }
  )");
  ASSERT_NE(M, nullptr);
  expectRejectionAndRecovery(*M, "gather source lane is not defined");
}

TEST_F(RISCVBOSCVerificationTest, ReadsActualTargetMachineFeatures) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  auto Scalar = machine("+m,+a,+f,+d,+c,+v,+zvl512b");
  ASSERT_NE(Scalar, nullptr);
  Error E = RISCV::verifyBOSCModule(*M, *Scalar);
  ASSERT_TRUE(bool(E));
  EXPECT_NE(toString(std::move(E)).find("xboscame"), std::string::npos);
  // The same module remains valid for a machine that owns the extensions.
  EXPECT_TRUE(toString(RISCV::verifyBOSCModule(*M, *Machine)).empty());
}

TEST_F(RISCVBOSCVerificationTest,
       RejectsTargetCPUOverrideAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  M->getFunction("mixed")->addFnAttr("target-cpu", "generic-rv32");
  expectRejectionAndRecovery(*M,
                             "target-cpu must match the actual target machine");
}

TEST_F(RISCVBOSCVerificationTest,
       RejectsTargetFeaturesOverrideAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  M->getFunction("mixed")->addFnAttr("target-features", "+e,+d");
  expectRejectionAndRecovery(
      *M, "target-features must match the actual target machine");
}

TEST_F(RISCVBOSCVerificationTest, AcceptsMatchingTargetAttributes) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  Function *F = M->getFunction("mixed");
  F->addFnAttr("target-cpu", Machine->getTargetCPU());
  F->addFnAttr("target-features", Machine->getTargetFeatureString());
  EXPECT_TRUE(toString(RISCV::verifyBOSCModule(*M, *Machine)).empty());
}

TEST_F(RISCVBOSCVerificationTest, ReturnsVLENLimitFailureAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  M->getFunction("mixed")->addFnAttr(
      Attribute::getWithVScaleRangeArgs(Context, 2048, 2048));
  expectRejectionAndRecovery(*M, "vscale_range exceeds the RISC-V VLEN limit");
}

TEST_F(RISCVBOSCVerificationTest,
       ReturnsVLENBelowTargetFailureAndCanEmitNextModule) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  M->getFunction("mixed")->addFnAttr(
      Attribute::getWithVScaleRangeArgs(Context, 1, 1));
  expectRejectionAndRecovery(*M, "lower than the Zvl*b limitation");
}

TEST_F(RISCVBOSCVerificationTest, AcceptsVLENAboveTargetMinimum) {
  auto M = parse(MixedContents);
  ASSERT_NE(M, nullptr);
  M->getFunction("mixed")->addFnAttr(
      Attribute::getWithVScaleRangeArgs(Context, 16, 16));
  EXPECT_TRUE(toString(RISCV::verifyBOSCModule(*M, *Machine)).empty());
}

TEST_F(RISCVBOSCVerificationTest,
       ReturnsObservationWidthFailureAndCanEmitNextModule) {
  auto M = parse(R"(
    declare i64 @llvm.riscv.bosc.read.bits.contents(ptr, i64, i32 immarg)
    define void @bad(ptr %p) "trex-bosc-contents-rvv"="1" {
      %x = call i64 @llvm.riscv.bosc.read.bits.contents(ptr %p, i64 0, i32 16)
      ret void
    }
  )");
  ASSERT_NE(M, nullptr);
  expectRejectionAndRecovery(*M, "bit observation width must be 8, 32, or 64");
}

TEST_F(RISCVBOSCVerificationTest, ReturnsBoundRVVFailureAndCanEmitNextModule) {
  auto M = parse(R"(
    declare i64 @llvm.riscv.bosc.setvl(i64)
    declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
    define void @bad(ptr %p) "trex-bosc-bound-rvv"="1" {
      %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
      call void @llvm.riscv.bosc.load(i32 0, ptr %p)
      ret void
    }
  )");
  ASSERT_NE(M, nullptr);
  expectRejectionAndRecovery(*M, "physical vector register must be in v1..v31");
}

TEST_F(RISCVBOSCVerificationTest, ReturnsBoundAMEFailureAndCanEmitNextModule) {
  auto M = parse(R"(
    declare i64 @llvm.riscv.bosc.ame.setm(i64)
    declare i64 @llvm.riscv.bosc.ame.setn(i64)
    declare i64 @llvm.riscv.bosc.ame.setk(i64)
    declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
    define void @bad(ptr %p) "trex-bosc-bound-ame"="1" {
      %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
      %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
      %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
      call void @llvm.riscv.bosc.ame.load.a(i32 16, ptr %p, i64 64)
      ret void
    }
  )");
  ASSERT_NE(M, nullptr);
  expectRejectionAndRecovery(*M, "tile binding must be Tr0..Tr15");
}

} // namespace
