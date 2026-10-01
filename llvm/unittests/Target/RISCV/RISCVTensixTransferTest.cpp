//===-- RISCVTensixTransferTest.cpp -------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::RISCV;

namespace {
TEST(RISCVTensixTransfer, RotateKeepsMaskedBitPreservingTransfer) {
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpshft2,
                                 {{TensixBoundValueKind::Constant, 0},
                                  {TensixBoundValueKind::Constant, 0},
                                  {TensixBoundValueKind::Constant, 1},
                                  {TensixBoundValueKind::Constant, 0},
                                  {TensixBoundValueKind::Constant, 3}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getTransferFacts();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LanePermutation);
  EXPECT_EQ(Facts.SourceLanes, (SmallVector<uint32_t, 0>{
                                   7,  0,  1,  2,  3,  4,  5,  6,  15, 8,  9,
                                   10, 11, 12, 13, 14, 23, 16, 17, 18, 19, 20,
                                   21, 22, 31, 24, 25, 26, 27, 28, 29, 30}));
  EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
  EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
  EXPECT_EQ(Facts.DemandedOldBits, 0u);
  EXPECT_TRUE(Facts.Inputs.empty());
  EXPECT_TRUE(Facts.RequiresLaneEnable);
  EXPECT_TRUE(Facts.PreservesInactiveLanes);
  EXPECT_TRUE(Facts.RequiresConditionState);
  EXPECT_TRUE(Facts.RequiresRowMaskState);
}

TEST(RISCVTensixTransfer, ShiftModeRemainsLegalWithoutPermutationSummary) {
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpshft2,
                                 {{TensixBoundValueKind::Constant, 6},
                                  {TensixBoundValueKind::Constant, 6},
                                  {TensixBoundValueKind::Constant, 5},
                                  {TensixBoundValueKind::Constant, 0},
                                  {TensixBoundValueKind::Constant, 4}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_EQ(Contract->getTransferFacts().Kind,
            TensixSFPUTransferKind::Unsupported);
  EXPECT_TRUE(Contract->getTransferFacts().SourceLanes.empty());
}

class RISCVTensixFloatBinaryTransferTest
    : public testing::TestWithParam<std::tuple<Intrinsic::ID, int64_t>> {
protected:
  Expected<TensixBoundSFPUContract> contract() const {
    // The old destination is a passthrough, distinct from both arithmetic
    // inputs. The selected forms do not permit indirect register indices.
    return getTensixBoundSFPUContract(
        std::get<0>(GetParam()),
        {{TensixBoundValueKind::Constant, 4},
         {TensixBoundValueKind::Constant, 4},
         {TensixBoundValueKind::Constant, 1},
         {TensixBoundValueKind::Constant, 2},
         {TensixBoundValueKind::Constant, std::get<1>(GetParam())}});
  }
};

TEST_P(RISCVTensixFloatBinaryTransferTest,
       ActiveLanesDemandBothArithmeticInputsAndPreserveInactiveOld) {
  auto Contract = contract();
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getTransferFacts();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
  EXPECT_EQ(Facts.DemandedOldBits, 0u);
  EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
  EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
  EXPECT_TRUE(Facts.RequiresLaneEnable);
  EXPECT_TRUE(Facts.PreservesInactiveLanes);
  EXPECT_TRUE(Facts.RequiresConditionState);
  EXPECT_TRUE(Facts.RequiresRowMaskState);
  ASSERT_EQ(Facts.Inputs.size(), 2u);
  EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
  EXPECT_EQ(Facts.Inputs[1].Argument, 3u);
  for (const auto &Input : Facts.Inputs)
    EXPECT_EQ(Input.DemandedBits, 0xffffffffu);
}

TEST_P(RISCVTensixFloatBinaryTransferTest,
       KeepsFixedConstantReadAndActualDestinationConfiguration) {
  auto Contract = contract();
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  unsigned Fixed =
      std::get<0>(GetParam()) == Intrinsic::riscv_tt_bound_sfpadd ? 10u : 9u;
  EXPECT_TRUE(any_of(Contract->getArchitecturalEffects(), [&](const auto &E) {
    const auto *Register = std::get_if<TensixFixedRegisterRef>(&E.Resource);
    return Register && Register->Number == Fixed &&
           E.Access == TensixSFPUAccess::Read;
  }));
  const auto &Config = Contract->getConfigurationFacts();
  ASSERT_TRUE(Config);
  ASSERT_EQ(Config->Requirements.size(), 1u);
  EXPECT_EQ(Config->Requirements[0].Argument, 0u);
  EXPECT_EQ(Config->Requirements[0].Field,
            TensixSFPUConfigurationField::EnableDestIndex);
  EXPECT_EQ(Config->Requirements[0].Registers,
            (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
}

INSTANTIATE_TEST_SUITE_P(
    DirectModes, RISCVTensixFloatBinaryTransferTest,
    testing::Combine(testing::Values(Intrinsic::riscv_tt_bound_sfpadd,
                                     Intrinsic::riscv_tt_bound_sfpmul),
                     testing::Values(int64_t(0), int64_t(1), int64_t(2),
                                     int64_t(3))));

TEST(RISCVTensixTransfer,
     MaskedNegationSeparatesNumericalSourceAndPassthrough) {
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfpmov, {{TensixBoundValueKind::Constant, 0},
                                         {TensixBoundValueKind::Constant, 0},
                                         {TensixBoundValueKind::Constant, 1},
                                         {TensixBoundValueKind::Constant, 1}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getTransferFacts();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
  ASSERT_EQ(Facts.Inputs.size(), 1u);
  EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
  EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
  EXPECT_EQ(Facts.DemandedOldBits, 0u);
  EXPECT_TRUE(Facts.PreservesInactiveLanes);
  auto Registers = Contract->getRegisterArguments();
  ASSERT_EQ(Registers.size(), 3u);
  EXPECT_EQ(Registers[1].Argument, 1u);
  EXPECT_EQ(Registers[1].Role, TensixBoundArgumentRole::OldDestination);
  EXPECT_TRUE(any_of(Contract->getRegisterConstraints(), [](const auto &C) {
    return C.Kind == TensixBoundConstraintKind::SameLocation &&
           C.FirstArgument == 0 && C.SecondArgument == 1;
  }));
}

class RISCVTensixBitwiseTransferTest
    : public testing::TestWithParam<Intrinsic::ID> {
protected:
  SmallVector<TensixBoundValue, 3> arguments() const {
    // Current bound forms use the destination as the first numerical input.
    // NOT aliases the native old and source fields to its sole ABI input.
    SmallVector<TensixBoundValue, 3> Args{{TensixBoundValueKind::Constant, 0},
                                          {TensixBoundValueKind::Constant, 0}};
    if (GetParam() != Intrinsic::riscv_tt_bound_sfpnot)
      Args.push_back({TensixBoundValueKind::Constant, 1});
    return Args;
  }
};

TEST_P(RISCVTensixBitwiseTransferTest, DemandsEachActualNumericalInput) {
  auto Contract = getTensixBoundSFPUContract(GetParam(), arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getTransferFacts();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
  EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
  EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
  // Active-lane old bits are consumed numerically, not copied to the result.
  EXPECT_EQ(Facts.DemandedOldBits, 0u);
  EXPECT_TRUE(Facts.RequiresLaneEnable);
  EXPECT_TRUE(Facts.PreservesInactiveLanes);
  EXPECT_TRUE(Facts.RequiresConditionState);
  EXPECT_TRUE(Facts.RequiresRowMaskState);
  unsigned Inputs = GetParam() == Intrinsic::riscv_tt_bound_sfpnot ? 1 : 2;
  ASSERT_EQ(Facts.Inputs.size(), Inputs);
  EXPECT_EQ(Facts.Inputs[0].Argument, 1u);
  EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
  if (Inputs == 2) {
    EXPECT_EQ(Facts.Inputs[1].Argument, 2u);
    EXPECT_EQ(Facts.Inputs[1].DemandedBits, 0xffffffffu);
  }
}

TEST_P(RISCVTensixBitwiseTransferTest, RetainsTiedOldForInactiveLanes) {
  auto Contract = getTensixBoundSFPUContract(GetParam(), arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  auto Registers = Contract->getRegisterArguments();
  ASSERT_EQ(Registers.size(),
            GetParam() == Intrinsic::riscv_tt_bound_sfpnot ? 2u : 3u);
  EXPECT_EQ(Registers[0].Argument, 0u);
  EXPECT_EQ(Registers[0].Role, TensixBoundArgumentRole::WriteLReg);
  EXPECT_EQ(Registers[1].Argument, 1u);
  EXPECT_EQ(Registers[1].Role, TensixBoundArgumentRole::OldDestination);
  EXPECT_TRUE(any_of(Contract->getRegisterConstraints(), [](const auto &C) {
    return C.Kind == TensixBoundConstraintKind::SameLocation &&
           C.FirstArgument == 0 && C.SecondArgument == 1;
  }));
}

TEST_P(RISCVTensixBitwiseTransferTest, PreservesNativeSharedStateEffects) {
  auto Contract = getTensixBoundSFPUContract(GetParam(), arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  bool ReadsCondition = false, ReadsConfiguration = false;
  bool ReadsAndWritesIssue = false;
  for (const auto &Effect : Contract->getArchitecturalEffects()) {
    EXPECT_FALSE(
        std::holds_alternative<TensixFixedRegisterRef>(Effect.Resource));
    const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
    if (!State)
      continue;
    EXPECT_NE(*State, TensixSFPUState::PRNG);
    EXPECT_NE(*State, TensixSFPUState::Dst);
    EXPECT_NE(*State, TensixSFPUState::CCStack);
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

TEST_P(RISCVTensixBitwiseTransferTest, RejectsRepairOfDestructiveInput) {
  auto Args = arguments();
  Args[0].Constant = 3;
  auto Error = verifyTensixBoundSFPUOperands(GetParam(), Args);
  ASSERT_TRUE(bool(Error));
  consumeError(std::move(Error));
}

INSTANTIATE_TEST_SUITE_P(CurrentForms, RISCVTensixBitwiseTransferTest,
                         testing::Values(Intrinsic::riscv_tt_bound_sfpand,
                                         Intrinsic::riscv_tt_bound_sfpor,
                                         Intrinsic::riscv_tt_bound_sfpxor,
                                         Intrinsic::riscv_tt_bound_sfpnot));

class RISCVTensixApproxRecipTransferTest
    : public testing::TestWithParam<int64_t> {
protected:
  SmallVector<TensixBoundValue, 4> arguments(bool Unbound = false) const {
    const auto RegisterKind = Unbound ? TensixBoundValueKind::UnboundRegister
                                      : TensixBoundValueKind::Constant;
    // VB is the tied old destination; VC is a separate actual input.
    return {{RegisterKind, 4},
            {RegisterKind, 4},
            {RegisterKind, 1},
            {TensixBoundValueKind::Constant, GetParam()}};
  }
};

TEST_P(RISCVTensixApproxRecipTransferTest,
       ActiveLanesDefineEveryBitAndDemandTheActualInput) {
  // Blackhole SFPARECIP writes VD in every enabled lane, even when the
  // conditional reciprocal selects the unchanged VC. Its predicate is the
  // sign of VB, not CC and not an active-lane old-value passthrough.
  for (bool Unbound : {false, true}) {
    SCOPED_TRACE(Unbound);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::riscv_tt_bound_sfparecip, arguments(Unbound));
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getTransferFacts();
    EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
    EXPECT_EQ(Facts.DemandedOldBits, 0u);
    EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
    EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    ASSERT_EQ(Facts.Inputs.size(), GetParam() == 1 ? 2u : 1u);
    const auto Input = find_if(Facts.Inputs, [](const auto &I) {
      return I.Argument == 2;
    });
    ASSERT_NE(Input, Facts.Inputs.end());
    EXPECT_EQ(Input->DemandedBits, 0xffffffffu);
    const auto Old = find_if(Facts.Inputs, [](const auto &I) {
      return I.Argument == 1;
    });
    if (GetParam() == 1) {
      ASSERT_NE(Old, Facts.Inputs.end());
      EXPECT_EQ(Old->DemandedBits, 0x80000000u);
    } else {
      EXPECT_EQ(Old, Facts.Inputs.end());
    }
  }
}

TEST_P(RISCVTensixApproxRecipTransferTest,
       PredicationReadsStateWithoutChangingConditionOrInventingResources) {
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfparecip, arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_TRUE(std::holds_alternative<std::monostate>(Contract->getLaneControl()));
  unsigned Conditions = 0, Configurations = 0, Issues = 0;
  unsigned Registers = 0;
  for (const auto &Effect : Contract->getArchitecturalEffects()) {
    EXPECT_FALSE(std::holds_alternative<TensixFixedRegisterRef>(Effect.Resource));
    if (const auto *Register =
            std::get_if<TensixBoundArgumentRef>(&Effect.Resource)) {
      ++Registers;
      EXPECT_LE(Register->Argument, 2u);
      EXPECT_EQ(Effect.Access, Register->Argument == 0
                                   ? TensixSFPUAccess::Write
                                   : TensixSFPUAccess::Read);
      continue;
    }
    const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
    ASSERT_NE(State, nullptr);
    switch (*State) {
    case TensixSFPUState::CC:
      ++Conditions;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      break;
    case TensixSFPUState::Configuration:
      ++Configurations;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      break;
    case TensixSFPUState::Issue:
      ++Issues;
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
      break;
    default:
      ADD_FAILURE() << "reciprocal has an unauthored state effect";
    }
  }
  EXPECT_EQ(Registers, 3u);
  EXPECT_EQ(Conditions, 1u);
  EXPECT_EQ(Configurations, 1u);
  EXPECT_EQ(Issues, 1u);
}

TEST_P(RISCVTensixApproxRecipTransferTest,
       NativeOldTiePreservesInactiveLanesInEveryMode) {
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfparecip, arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto Registers = Contract->getRegisterArguments();
  ASSERT_EQ(Registers.size(), 3u);
  EXPECT_EQ(Registers[0].Argument, 0u);
  EXPECT_EQ(Registers[0].Role, TensixBoundArgumentRole::WriteLReg);
  EXPECT_EQ(Registers[1].Argument, 1u);
  EXPECT_EQ(Registers[1].Role, TensixBoundArgumentRole::OldDestination);
  EXPECT_EQ(Registers[2].Argument, 2u);
  EXPECT_EQ(Registers[2].Role, TensixBoundArgumentRole::ReadRegister);
  for (const auto &Register : Registers)
    EXPECT_EQ(Register.Footprint, TensixSFPURegisterFootprint::StateDependent);
  EXPECT_TRUE(any_of(Contract->getRegisterConstraints(), [](const auto &C) {
    return C.Kind == TensixBoundConstraintKind::SameLocation &&
           C.FirstArgument == 0 && C.SecondArgument == 1;
  }));

  auto Args = arguments();
  Args[1].Constant = 3;
  auto Rejected = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfparecip, Args);
  ASSERT_FALSE(bool(Rejected));
  consumeError(Rejected.takeError());
}

INSTANTIATE_TEST_SUITE_P(ReceivedModes, RISCVTensixApproxRecipTransferTest,
                         testing::Values(int64_t(0), int64_t(1), int64_t(2)));

TEST(RISCVTensixTransfer, ReciprocalRejectsUnreceivedOrNonconstantModes) {
  // The formal intrinsic receives only the three authored modes. Do not
  // silently interpret another raw field as EXP or accept a dynamic mode.
  SmallVector<TensixBoundValue, 4> Args{
      {TensixBoundValueKind::Constant, 0},
      {TensixBoundValueKind::Constant, 0},
      {TensixBoundValueKind::Constant, 1},
      {TensixBoundValueKind::Constant, 0}};
  for (int64_t Mode : {-1, 3, 4, 15, 16}) {
    SCOPED_TRACE(Mode);
    Args[3] = {TensixBoundValueKind::Constant, Mode};
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::riscv_tt_bound_sfparecip, Args);
    ASSERT_FALSE(bool(Contract));
    consumeError(Contract.takeError());
  }
  for (auto Kind : {TensixBoundValueKind::UnboundRegister,
                    TensixBoundValueKind::DynamicScalar}) {
    Args[3] = {Kind, 0};
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::riscv_tt_bound_sfparecip, Args);
    ASSERT_FALSE(bool(Contract));
    consumeError(Contract.takeError());
  }
}
} // namespace
