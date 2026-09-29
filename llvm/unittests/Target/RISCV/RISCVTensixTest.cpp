//===-- RISCVTensixTest.cpp ----------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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
                Intrinsic::riscv_tt_stallwait_port),
            Info);
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

TEST(RISCVTensix, BoundMoveHasWholeRegisterFootprint) {
  using namespace RISCV;
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpmov_all,
                                 {{TensixBoundValueKind::UnboundRegister},
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

TEST(RISCVTensix, BoundRoundingRetainsPRNGForDeterministicModes) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Rounding : {0, 1, 2}) {
    for (auto ID : {Intrinsic::riscv_tt_bound_sfpstochrnd_i,
                    Intrinsic::riscv_tt_bound_sfpstochrnd_v}) {
      bool Immediate = ID == Intrinsic::riscv_tt_bound_sfpstochrnd_i;
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
        Intrinsic::riscv_tt_bound_sfploadi,
        {Constant(0), Constant(0), Constant(0), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getTransferFacts();
    EXPECT_EQ(Facts.Kind, Kind);
    EXPECT_EQ(Facts.GeneratedBits, Generated);
    EXPECT_EQ(Facts.GuaranteedWriteBits, Generated);
    EXPECT_EQ(Facts.DemandedOldBits, Demanded);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
  }
  auto Move = getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpmov_all,
                                         {Constant(0), Constant(1)});
  ASSERT_TRUE(bool(Move)) << toString(Move.takeError());
  EXPECT_EQ(Move->getTransferFacts().Kind, TensixSFPUTransferKind::Identity);
  EXPECT_EQ(Move->getTransferFacts().GeneratedBits, 0xffffffffu);
  EXPECT_FALSE(Move->getTransferFacts().RequiresLaneEnable);
  EXPECT_FALSE(Move->getTransferFacts().RequiresConditionState);
  EXPECT_FALSE(Move->getTransferFacts().RequiresRowMaskState);
}

TEST(RISCVTensix, BoundMaskedMoveDeclaresSourceOldAndLaneFacts) {
  using namespace RISCV;
  auto Constant = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (int64_t Mode : {0, 1}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::riscv_tt_bound_sfpmov,
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

    const auto &Facts = Contract->getTransferFacts();
    EXPECT_EQ(Facts.Kind, Mode == 0 ? TensixSFPUTransferKind::Identity
                                   : TensixSFPUTransferKind::LaneWise);
    if (Mode == 0) {
      EXPECT_TRUE(Facts.Inputs.empty());
    } else {
      ASSERT_EQ(Facts.Inputs.size(), 1u);
      EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
      EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
    }
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
        Intrinsic::riscv_tt_bound_sfpiadd,
        {Constant(1), Constant(1), Constant(2), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getTransferFacts();
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
        Intrinsic::riscv_tt_bound_sfpiadd,
        {Constant(1), Constant(1), Constant(2), Constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getTransferFacts();
    EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::Unsupported);
    EXPECT_TRUE(Facts.Inputs.empty());
    EXPECT_EQ(Facts.GeneratedBits, 0u);
    EXPECT_EQ(Facts.GuaranteedWriteBits, 0u);
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
        Intrinsic::riscv_tt_bound_sfpiadd,
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
      EXPECT_FALSE(std::holds_alternative<TensixFixedRegisterRef>(
          Effect.Resource));
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
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpencc,
                                   {{TensixBoundValueKind::Constant, C.Value},
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
      Intrinsic::riscv_tt_bound_sfpencc, {{TensixBoundValueKind::Constant, 0},
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
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpconfig_reset, {});
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
      {Intrinsic::riscv_tt_bound_sfpmov,
       {Constant(0), Constant(0), Constant(1), Constant(0)}},
      {Intrinsic::riscv_tt_bound_sfpmul,
       {Constant(0), Constant(0), Constant(1), Constant(2), Constant(0)}},
      {Intrinsic::riscv_tt_bound_sfploadi,
       {Constant(0), Constant(0), Constant(0x1234), Constant(8)}},
      {Intrinsic::riscv_tt_bound_sfploadi,
       {Constant(0), Constant(0), Constant(0x1234), Constant(10)}},
      {Intrinsic::riscv_tt_bound_sfptransp,
       {Constant(0), Constant(1), Constant(2), Constant(3), Constant(4),
        Constant(5), Constant(6), Constant(7), Constant(0), Constant(1),
        Constant(2), Constant(3)}},
      {Intrinsic::riscv_tt_bound_sfpconfig_creg,
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
} // namespace
