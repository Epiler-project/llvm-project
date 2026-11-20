//===-- RISCVTensixConfigurationTest.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicsTT.h"
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::RISCV;

namespace {

TensixBoundValue constant(int64_t Value) {
  return {TensixBoundValueKind::Constant, Value};
}

struct ConfigurationCase {
  Intrinsic::ID ID;
  SmallVector<TensixBoundValue, 6> Arguments;
};

TEST(RISCVTensixConfiguration, TransposeHasProvedErratumExemptionAndLaneReads) {
  SmallVector<TensixBoundValue, 12> Args;
  for (unsigned I = 0; I != 12; ++I)
    Args.push_back(constant(I < 8 ? I : I - 8));
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_transp, Args);
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  // SFPTRANSP is explicitly exempt from TEN-2932. Its actual VD field is
  // zero, so the backdoor-load conditional also needs no configuration guard.
  // This does not make its CC and ROW_MASK lane predicates unconditional.
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
  for (auto State : {TensixSFPUState::CC, TensixSFPUState::Configuration})
    EXPECT_TRUE(any_of(Contract->getArchitecturalEffects(), [&](const auto &E) {
      const auto *Resource = std::get_if<TensixSFPUState>(&E.Resource);
      return Resource && *Resource == State &&
             E.Access == TensixSFPUAccess::Read;
    }));
}

class RISCVTensixDestinationConfigurationTest
    : public testing::TestWithParam<ConfigurationCase> {};

TEST(RISCVTensixConfiguration, SwapRequiresBothMinMaxControlsForBothWrites) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_swap, Intrinsic::tt_bh_sfpu_swap}) {
    auto Contract = getTensixBoundSFPUContract(
        ID, {constant(2), constant(6), constant(2), constant(6), constant(1)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getConfigurationFacts();
    ASSERT_TRUE(Facts);
    ASSERT_EQ(Facts->Requirements.size(), 4u);
    for (unsigned Index = 0; Index != 4; ++Index) {
      const auto &Requirement = Facts->Requirements[Index];
      EXPECT_EQ(Requirement.Argument, Index / 2);
      // Two independent fields must be disabled for ordinary min+max.
      EXPECT_EQ(static_cast<unsigned>(Requirement.Field), Index % 2);
      EXPECT_EQ(Requirement.Registers,
                (SmallVector<uint32_t, 8>{0, 1, 2, 3, 4, 5, 6, 7}));
    }
  }
}

TEST_P(RISCVTensixDestinationConfigurationTest,
       RequirementNamesOnlyActualWriteArgumentAndAffectedRegisters) {
  // Requirements describe the native write class, including when its current
  // binding is below L4. A high-numbered source or tied old value is not a
  // second write and must not acquire its own requirement.
  for (uint32_t Destination : {0u, 7u}) {
    SCOPED_TRACE(Destination);
    auto Args = GetParam().Arguments;
    Args[0] = constant(Destination);
    if (GetParam().ID != Intrinsic::tt_bh_sfpu_mov_all)
      Args[1] = constant(Destination);
    auto Contract = getTensixBoundSFPUContract(GetParam().ID, Args);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getConfigurationFacts();
    ASSERT_TRUE(Facts);
    ASSERT_EQ(Facts->Requirements.size(), 1u);
    const auto &Requirement = Facts->Requirements.front();
    EXPECT_EQ(Requirement.Argument, 0u);
    EXPECT_EQ(Requirement.Field, TensixSFPUConfigurationField::EnableDestIndex);
    EXPECT_EQ(Requirement.Registers, (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
    EXPECT_TRUE(any_of(Contract->getRegisterArguments(), [&](const auto &R) {
      return R.Argument == Requirement.Argument &&
             R.Role == TensixBoundArgumentRole::WriteLReg;
    }));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SelectedNativeWrites, RISCVTensixDestinationConfigurationTest,
    testing::Values(
        ConfigurationCase{Intrinsic::tt_bh_sfpu_mov_all,
                          {constant(0), constant(6)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_mov_prng_advance,
                          {constant(0), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_mov,
                          {constant(0), constant(0), constant(6), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_mov,
                          {constant(0), constant(0), constant(6), constant(1)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_iadd,
                          {constant(0), constant(0), constant(6), constant(4)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_iadd,
                          {constant(0), constant(0), constant(6), constant(6)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_and,
                          {constant(0), constant(0), constant(6)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_or,
                          {constant(0), constant(0), constant(6)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_xor,
                          {constant(0), constant(0), constant(6)}},
        ConfigurationCase{
            Intrinsic::tt_bh_sfpu_mul,
            {constant(0), constant(0), constant(1), constant(2), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_not,
                          {constant(0), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_arecip,
                          {constant(0), constant(0), constant(6), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_arecip,
                          {constant(0), constant(0), constant(6), constant(1)}},
        ConfigurationCase{
            Intrinsic::tt_bh_sfpu_arecip,
            {constant(0), constant(0), constant(6), constant(2)}}));

TEST(RISCVTensixConfiguration,
     ReciprocalUnboundWriteRetainsErratumRequirement) {
  // Blackhole SFPCONFIG shares the Wormhole definition. TEN-2932 exempts
  // SFPLOAD/LOADI/SWAP/TRANSP, not SFPARECIP. The requirement must survive
  // until the one existing binder chooses the actual destination.
  const TensixBoundValue Unbound{TensixBoundValueKind::UnboundRegister};
  for (int64_t Mode : {0, 1, 2}) {
    SCOPED_TRACE(Mode);
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip,
                                   {Unbound, Unbound, Unbound, constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getConfigurationFacts();
    ASSERT_TRUE(Facts);
    ASSERT_EQ(Facts->Requirements.size(), 1u);
    EXPECT_EQ(Facts->Requirements.front().Argument, 0u);
    EXPECT_EQ(Facts->Requirements.front().Field,
              TensixSFPUConfigurationField::EnableDestIndex);
    EXPECT_EQ(Facts->Requirements.front().Registers,
              (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
  }
}

class RISCVTensixLoadImmediateConfigurationTest
    : public testing::TestWithParam<int64_t> {};

TEST_P(RISCVTensixLoadImmediateConfigurationTest,
       ErratumExemptionHasProvedEmptyRequirements) {
  // TEN-2932 excludes SFPLOADI, including half-word updates to L4-L7.
  for (auto ID : {Intrinsic::tt_bh_sfpu_loadi, Intrinsic::tt_wh_sfpu_loadi}) {
    SCOPED_TRACE(ID);
    for (auto Immediate : {0, 0x1234, 0xffff}) {
      SCOPED_TRACE(Immediate);
      auto Contract = getTensixBoundSFPUContract(ID, {constant(7), constant(7),
                                                      constant(Immediate),
                                                      constant(GetParam())});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      const auto &Facts = Contract->getConfigurationFacts();
      ASSERT_TRUE(Facts);
      EXPECT_TRUE(Facts->Requirements.empty());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ReceivedImmediateForms,
                         RISCVTensixLoadImmediateConfigurationTest,
                         testing::Values(0, 1, 2, 4, 8, 10));

class RISCVTensixDstConfigurationTest
    : public testing::TestWithParam<ConfigurationCase> {};

TEST_P(RISCVTensixDstConfigurationTest,
       NativeDstTransferHasProvedEmptyRequirements) {
  for (uint32_t Register : {0u, 7u}) {
    SCOPED_TRACE(Register);
    auto Args = GetParam().Arguments;
    Args[0] = constant(Register);
    if (GetParam().ID == Intrinsic::tt_bh_sfpu_load)
      Args[1] = constant(Register);
    auto Contract = getTensixBoundSFPUContract(GetParam().ID, Args);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getConfigurationFacts();
    ASSERT_TRUE(Facts);
    EXPECT_TRUE(Facts->Requirements.empty());
  }
}

INSTANTIATE_TEST_SUITE_P(
    ReceivedDstTransfers, RISCVTensixDstConfigurationTest,
    testing::Values(ConfigurationCase{Intrinsic::tt_bh_sfpu_load,
                                      {constant(0), constant(0), constant(0),
                                       constant(0), constant(3)}},
                    ConfigurationCase{
                        Intrinsic::tt_bh_sfpu_store,
                        {constant(0), constant(0), constant(0), constant(3)}}));

TEST(RISCVTensixConfiguration, DstTransferStillRequiresValidRegisterBinding) {
  auto Load = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_load,
      {constant(7), constant(6), constant(0), constant(0), constant(3)});
  EXPECT_FALSE(bool(Load));
  if (!Load)
    consumeError(Load.takeError());

  auto Store = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_store,
      {constant(16), constant(0), constant(0), constant(3)});
  EXPECT_FALSE(bool(Store));
  if (!Store)
    consumeError(Store.takeError());
}

TEST(RISCVTensixConfiguration, ConditionControlHasProvedEmptyRequirements) {
  auto Contract = getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_encc,
                                             {constant(0), constant(2)});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

TEST(RISCVTensixConfiguration, SetCCProjectsActualSourceAndConditionEffects) {
  using Predicate = TensixSFPUConditionPredicate;
  using Role = TensixBoundArgumentRole;
  using State = TensixSFPUState;
  using Access = TensixSFPUAccess;
  const std::pair<int64_t, Predicate> Modes[] = {{0, Predicate::Negative},
                                                 {2, Predicate::NonZero},
                                                 {4, Predicate::NonNegative},
                                                 {6, Predicate::Zero}};
  for (Intrinsic::ID ID :
       {Intrinsic::tt_wh_sfpu_setcc, Intrinsic::tt_bh_sfpu_setcc}) {
    for (auto [Mode, Pred] : Modes) {
      SCOPED_TRACE(ID);
      SCOPED_TRACE(Mode);
      auto Contract = getTensixBoundSFPUContract(
          ID, {constant(13), constant(0), constant(Mode)});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_TRUE(Contract->getConfigurationFacts());
      EXPECT_TRUE(Contract->getConfigurationFacts()->Requirements.empty());
      EXPECT_TRUE(Contract->getTransferFacts().empty());
      EXPECT_TRUE(Contract->getRegisterConstraints().empty());
      ASSERT_EQ(Contract->getArgumentRoles().size(), 3u);
      EXPECT_EQ(Contract->getArgumentRoles()[0], Role::ReadRegister);
      EXPECT_EQ(Contract->getArgumentRoles()[1], Role::LogicalImmediate);
      EXPECT_EQ(Contract->getArgumentRoles()[2], Role::LogicalImmediate);

      const auto *Compare =
          std::get_if<TensixSFPUConditionCompare>(&Contract->getLaneControl());
      ASSERT_NE(Compare, nullptr);
      EXPECT_EQ(Compare->SourceValue, 0u);
      EXPECT_EQ(Compare->Predicate, Pred);
      ASSERT_EQ(Contract->getRegisterArguments().size(), 1u);
      const auto &Source = Contract->getRegisterArguments()[0];
      EXPECT_EQ(Source.Argument, Compare->SourceValue);
      EXPECT_EQ(Source.Role, Role::ReadRegister);
      EXPECT_FALSE(Source.FixedRegister);
      ASSERT_EQ(Source.AllowedRegisters.size(), 16u);
      for (unsigned Register = 0; Register != 16; ++Register)
        EXPECT_TRUE(is_contained(Source.AllowedRegisters, Register));

      const auto Effects = Contract->getArchitecturalEffects();
      ASSERT_EQ(Effects.size(), 4u);
      EXPECT_EQ(count_if(Effects,
                         [](const auto &Effect) {
                           const auto *Ref =
                               std::get_if<TensixBoundArgumentRef>(
                                   &Effect.Resource);
                           return Ref && Ref->Argument == 0 &&
                                  Effect.Access == Access::Read;
                         }),
                1);
      for (auto [Resource, Permission] :
           {std::pair{State::CC, Access::ReadWrite},
            std::pair{State::Configuration, Access::Read},
            std::pair{State::Issue, Access::ReadWrite}}) {
        EXPECT_EQ(count_if(Effects,
                           [&](const auto &Effect) {
                             const auto *Actual =
                                 std::get_if<State>(&Effect.Resource);
                             return Actual && *Actual == Resource &&
                                    Effect.Access == Permission;
                           }),
                  1);
      }
    }
  }
}

TEST(RISCVTensixConfiguration, SetCCRequiresOnlyTheActualComparisonBits) {
  using Predicate = TensixSFPUConditionPredicate;
  const TensixSFPUConditionCompare Neg{0, Predicate::Negative};
  const TensixSFPUConditionCompare NonNeg{0, Predicate::NonNegative};
  const TensixSFPUConditionCompare NonZero{0, Predicate::NonZero};
  const TensixSFPUConditionCompare Zero{0, Predicate::Zero};
  constexpr uint32_t SignBit = 0x80000000u;
  constexpr uint32_t AllBits = 0xffffffffu;
  const TensixSFPUValueBits Unknown{0, 0};
  EXPECT_EQ(Neg.getRequiredSourceBits(), SignBit);
  EXPECT_EQ(NonNeg.getRequiredSourceBits(), SignBit);
  EXPECT_EQ(NonZero.getRequiredSourceBits(), AllBits);
  EXPECT_EQ(Zero.getRequiredSourceBits(), AllBits);
  EXPECT_EQ(Neg.evaluateFlags(false, true, true, {SignBit, SignBit}), true);
  EXPECT_EQ(NonNeg.evaluateFlags(false, true, true, {SignBit, 0}), true);
  EXPECT_EQ(Neg.evaluateFlags(false, true, true, {1, 1}), std::nullopt);
  EXPECT_EQ(NonZero.evaluateFlags(false, true, true, {1, 1}), true);
  EXPECT_EQ(Zero.evaluateFlags(false, true, true, {1, 1}), false);
  EXPECT_EQ(Zero.evaluateFlags(false, true, true, {1, 0}), std::nullopt);
  EXPECT_EQ(Zero.evaluateFlags(false, true, true, {0, 1}), std::nullopt);
  EXPECT_EQ(Zero.evaluateFlags(false, true, true, {AllBits, 0}), true);
  EXPECT_EQ(NonZero.evaluateFlags(false, true, true, {AllBits, 0}), false);
  EXPECT_EQ(Zero.evaluateFlags(false, true, true, {AllBits, SignBit}), false);
  EXPECT_EQ(Neg.evaluateFlags(true, false, true, Unknown), true);
  EXPECT_EQ(Neg.evaluateFlags(true, true, false, Unknown), false);
  EXPECT_EQ(Neg.evaluateFlags(false, false, true, Unknown), false);
  EXPECT_EQ(Neg.evaluateFlags(false, true, false, Unknown), false);
  EXPECT_EQ(Neg.evaluateFlags(false, true, true, Unknown), std::nullopt);
}

TEST(RISCVTensixConfiguration, SetCCContentsProjectsActualSourceValue) {
  for (auto Architecture :
       {TensixSFPUArchitecture::Wormhole, TensixSFPUArchitecture::Blackhole}) {
    auto Info = getTensixContentsIntrinsicInfo(
        Intrinsic::riscv_tt_contents_setcc,
        {{TensixContentsValueKind::Contents, std::nullopt},
         {TensixContentsValueKind::Scalar, 6}},
        Architecture);
    ASSERT_TRUE(bool(Info)) << toString(Info.takeError());
    const auto *Compare =
        std::get_if<TensixSFPUConditionCompare>(&Info->LaneControl);
    ASSERT_NE(Compare, nullptr);
    EXPECT_EQ(Compare->Predicate, TensixSFPUConditionPredicate::Zero);
    ASSERT_LT(Compare->SourceValue, Info->TransferValues.size());
    const auto *Source = std::get_if<TensixContentsArgumentRef>(
        &Info->TransferValues[Compare->SourceValue]);
    ASSERT_NE(Source, nullptr);
    EXPECT_EQ(Source->Argument, 0u);
  }
}

TEST(RISCVTensixConfiguration, SetCCRejectsOtherFieldsAndWrongRegisterClass) {
  using ValueKind = TensixBoundValueKind;
  for (Intrinsic::ID ID :
       {Intrinsic::tt_wh_sfpu_setcc, Intrinsic::tt_bh_sfpu_setcc}) {
    SCOPED_TRACE(ID);
    const std::pair<unsigned, TensixBoundValue> InvalidFields[] = {
        {0, constant(-1)},
        {0, constant(16)},
        {0, {ValueKind::DynamicScalar}},
        {1, constant(1)},
        {1, {ValueKind::UnboundRegister}},
        {2, constant(1)},
        {2, constant(8)},
        {2, {ValueKind::DynamicScalar}}};
    for (const auto &[Field, Value] : InvalidFields) {
      SmallVector<TensixBoundValue, 3> Args{constant(0), constant(0),
                                            constant(0)};
      Args[Field] = Value;
      auto Rejected = getTensixBoundSFPUContract(ID, Args);
      ASSERT_FALSE(bool(Rejected));
      consumeError(Rejected.takeError());
    }
    auto Unbound = getTensixBoundSFPUContract(
        ID, {{ValueKind::UnboundRegister}, constant(0), constant(4)});
    ASSERT_TRUE(bool(Unbound)) << toString(Unbound.takeError());
    ASSERT_EQ(Unbound->getRegisterArguments().size(), 1u);
    EXPECT_EQ(Unbound->getRegisterArguments()[0].Argument, 0u);
    auto UnboundIngress = verifyTensixBoundSFPUOperands(
        ID, {{ValueKind::UnboundRegister}, constant(0), constant(4)});
    EXPECT_EQ(toString(std::move(UnboundIngress)),
              "bound SFPU physical operands and logical immediates must be "
              "constant");
  }
}

class RISCVTensixConditionStackConfigurationTest
    : public testing::TestWithParam<Intrinsic::ID> {};

TEST_P(RISCVTensixConditionStackConfigurationTest,
       PlainStackHasProvedEmptyRequirementsAndExactStateEffects) {
  SmallVector<TensixBoundValue, 2> Args;
  if (GetParam() != Intrinsic::tt_bh_sfpu_compc)
    Args = {constant(0), constant(0)};
  auto Contract = getTensixBoundSFPUContract(GetParam(), Args);
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  // All three native words encode VD=0. Their behavior therefore cannot be
  // disabled by the configurable VD>=12 backdoor-load path, and no numerical
  // destination introduces an EnableDestIndex requirement.
  const auto &Facts = Contract->getConfigurationFacts();
  EXPECT_TRUE(Facts);
  if (Facts)
    EXPECT_TRUE(Facts->Requirements.empty());
  EXPECT_TRUE(Contract->getRegisterArguments().empty());
  EXPECT_TRUE(Contract->getRegisterConstraints().empty());
  EXPECT_TRUE(Contract->getTransferFacts().empty());

  using State = TensixSFPUState;
  using Access = TensixSFPUAccess;
  SmallVector<std::pair<State, Access>, 4> Expected{
      {State::Configuration, Access::Read}, {State::Issue, Access::ReadWrite}};
  switch (GetParam()) {
  case Intrinsic::tt_bh_sfpu_pushc:
    Expected.append(
        {{State::CC, Access::Read}, {State::CCStack, Access::ReadWrite}});
    break;
  case Intrinsic::tt_bh_sfpu_popc:
    Expected.append(
        {{State::CC, Access::Write}, {State::CCStack, Access::ReadWrite}});
    break;
  case Intrinsic::tt_bh_sfpu_compc:
    Expected.append(
        {{State::CC, Access::ReadWrite}, {State::CCStack, Access::Read}});
    break;
  default:
    FAIL() << "unexpected stack fixture";
  }
  EXPECT_EQ(Contract->getArchitecturalEffects().size(), Expected.size());
  for (auto [Resource, Permission] : Expected) {
    unsigned Matches = 0;
    for (const auto &Effect : Contract->getArchitecturalEffects()) {
      const auto *Actual = std::get_if<State>(&Effect.Resource);
      if (Actual && *Actual == Resource) {
        EXPECT_EQ(Effect.Access, Permission);
        ++Matches;
      }
    }
    EXPECT_EQ(Matches, 1u);
  }
}

TEST_P(RISCVTensixConditionStackConfigurationTest,
       SelectedNativeOpcodePublishesTypedStackAction) {
  SmallVector<TensixBoundValue, 2> Args;
  if (GetParam() != Intrinsic::tt_bh_sfpu_compc)
    Args = {constant(0), constant(0)};
  auto Contract = getTensixBoundSFPUContract(GetParam(), Args);
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto *Update =
      std::get_if<TensixSFPUConditionStackUpdate>(&Contract->getLaneControl());
  ASSERT_NE(Update, nullptr);
  using Action = TensixSFPUConditionStackAction;
  switch (GetParam()) {
  case Intrinsic::tt_bh_sfpu_pushc:
    EXPECT_EQ(Update->Action, Action::Push);
    break;
  case Intrinsic::tt_bh_sfpu_popc:
    EXPECT_EQ(Update->Action, Action::Pop);
    break;
  case Intrinsic::tt_bh_sfpu_compc:
    EXPECT_EQ(Update->Action, Action::Complement);
    break;
  default:
    FAIL() << "unexpected stack fixture";
  }
}

INSTANTIATE_TEST_SUITE_P(PlainConditionStack,
                         RISCVTensixConditionStackConfigurationTest,
                         testing::Values(Intrinsic::tt_bh_sfpu_pushc,
                                         Intrinsic::tt_bh_sfpu_popc,
                                         Intrinsic::tt_bh_sfpu_compc));

TEST(RISCVTensixConfiguration,
     ModeOneStackHasDistinctTypedActionsAndExactStateEffects) {
  for (Intrinsic::ID ID :
       {Intrinsic::tt_bh_sfpu_pushc, Intrinsic::tt_bh_sfpu_popc}) {
    SCOPED_TRACE(ID);
    auto Plain = getTensixBoundSFPUContract(ID, {constant(0), constant(0)});
    ASSERT_TRUE(bool(Plain)) << toString(Plain.takeError());
    auto ModeOne = getTensixBoundSFPUContract(ID, {constant(0), constant(1)});
    ASSERT_TRUE(bool(ModeOne)) << toString(ModeOne.takeError());
    const auto *PlainUpdate =
        std::get_if<TensixSFPUConditionStackUpdate>(&Plain->getLaneControl());
    const auto *Update =
        std::get_if<TensixSFPUConditionStackUpdate>(&ModeOne->getLaneControl());
    ASSERT_NE(PlainUpdate, nullptr);
    ASSERT_NE(Update, nullptr);
    // Mode one cannot borrow the plain push/pop transition: replacement
    // preserves depth and peek does not consume the saved frame.
    using Action = TensixSFPUConditionStackAction;
    EXPECT_NE(Update->Action, PlainUpdate->Action);
    EXPECT_EQ(Update->Action, ID == Intrinsic::tt_bh_sfpu_pushc
                                  ? Action::ReplaceTop
                                  : Action::Peek);
    ASSERT_TRUE(ModeOne->getConfigurationFacts());
    EXPECT_TRUE(ModeOne->getConfigurationFacts()->Requirements.empty());
    EXPECT_TRUE(ModeOne->getRegisterArguments().empty());
    EXPECT_TRUE(ModeOne->getRegisterConstraints().empty());
    EXPECT_TRUE(ModeOne->getTransferFacts().empty());

    using State = TensixSFPUState;
    using Access = TensixSFPUAccess;
    const bool Replace = ID == Intrinsic::tt_bh_sfpu_pushc;
    const SmallVector<std::pair<State, Access>, 4> Expected{
        {State::CC, Replace ? Access::Read : Access::Write},
        {State::CCStack, Replace ? Access::ReadWrite : Access::Read},
        {State::Configuration, Access::Read},
        {State::Issue, Access::ReadWrite}};
    EXPECT_EQ(ModeOne->getArchitecturalEffects().size(), Expected.size());
    for (auto [Resource, Permission] : Expected) {
      unsigned Matches = 0;
      for (const auto &Effect : ModeOne->getArchitecturalEffects()) {
        const auto *Actual = std::get_if<State>(&Effect.Resource);
        if (Actual && *Actual == Resource) {
          EXPECT_EQ(Effect.Access, Permission);
          ++Matches;
        }
      }
      EXPECT_EQ(Matches, 1u);
    }
  }
}

TEST(RISCVTensixConfiguration,
     SFPPOPCCombineModesPeekTopAndPublishTypedActions) {
  using Action = TensixSFPUConditionStackAction;
  using State = TensixSFPUState;
  using Access = TensixSFPUAccess;
  for (auto [Mode, ExpectedAction] :
       {std::pair<int64_t, Action>{3, Action::AndTop},
        std::pair<int64_t, Action>{4, Action::OrTop}}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_popc,
                                               {constant(0), constant(Mode)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto *Update = std::get_if<TensixSFPUConditionStackUpdate>(
        &Contract->getLaneControl());
    ASSERT_NE(Update, nullptr);
    EXPECT_EQ(Update->Action, ExpectedAction);
    ASSERT_TRUE(Contract->getConfigurationFacts());
    EXPECT_TRUE(Contract->getConfigurationFacts()->Requirements.empty());

    const SmallVector<std::pair<State, Access>, 4> Expected{
        {State::Configuration, Access::Read},
        {State::Issue, Access::ReadWrite},
        // Boolean-combine modes consume the current LaneFlags as well as
        // the saved top. Only plain pop/peek replace CC without reading it.
        {State::CC, Access::ReadWrite},
        {State::CCStack, Access::Read}};
    EXPECT_EQ(Contract->getArchitecturalEffects().size(), Expected.size());
    for (auto [Resource, Permission] : Expected) {
      unsigned Matches = 0;
      for (const auto &Effect : Contract->getArchitecturalEffects()) {
        const auto *Actual = std::get_if<State>(&Effect.Resource);
        if (Actual && *Actual == Resource) {
          EXPECT_EQ(Effect.Access, Permission);
          ++Matches;
        }
      }
      EXPECT_EQ(Matches, 1u);
    }
  }
}

TEST(RISCVTensixConfiguration, SFPPOPCRejectsUnspecifiedCombineMode) {
  auto Contract = getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_popc,
                                             {constant(0), constant(2)});
  ASSERT_FALSE(bool(Contract));
  EXPECT_EQ(toString(Contract.takeError()),
            "bound SFPU unsupported mode at operand 1");
}

TEST(RISCVTensixConfiguration, ConditionStackRejectsUnsupportedFields) {
  // VD stays zero. Push receives only modes zero and one; pop additionally
  // receives Blackhole's mode-three/four Boolean-combine forms. Other
  // mutation and flag-setting modes remain unavailable.
  for (Intrinsic::ID ID :
       {Intrinsic::tt_bh_sfpu_pushc, Intrinsic::tt_bh_sfpu_popc}) {
    SCOPED_TRACE(ID);
    for (unsigned Field : {0u, 1u}) {
      SCOPED_TRACE(Field);
      for (int64_t Value = -1; Value != 17; ++Value) {
        const bool ValidMode =
            Field == 1 &&
            (Value == 0 || Value == 1 ||
             (ID == Intrinsic::tt_bh_sfpu_popc && (Value == 3 || Value == 4)));
        if (Field == 0 && Value == 0)
          continue;
        if (ValidMode)
          continue;
        SCOPED_TRACE(Value);
        SmallVector<TensixBoundValue, 2> Args{constant(0), constant(0)};
        Args[Field] = constant(Value);
        auto Rejected = getTensixBoundSFPUContract(ID, Args);
        ASSERT_FALSE(bool(Rejected));
        EXPECT_EQ(toString(Rejected.takeError()),
                  Field == 0
                      ? "bound SFPU logical field operand 0 must be in [0, 0]"
                      : "bound SFPU unsupported mode at operand 1");
      }
      for (TensixBoundValueKind Kind :
           {TensixBoundValueKind::DynamicScalar,
            TensixBoundValueKind::UnboundRegister}) {
        SCOPED_TRACE(unsigned(Kind));
        SmallVector<TensixBoundValue, 2> Args{constant(0), constant(0)};
        Args[Field] = {Kind};
        auto Rejected = getTensixBoundSFPUContract(ID, Args);
        ASSERT_FALSE(bool(Rejected));
        EXPECT_EQ(toString(Rejected.takeError()),
                  Field == 0
                      ? "bound SFPU logical field operand 0 must be in [0, 0]"
                      : "bound SFPU unsupported mode at operand 1");
      }
    }
  }
}

TEST(RISCVTensixConfiguration, PlainConditionStackRequiresExactArgumentTuple) {
  for (Intrinsic::ID ID :
       {Intrinsic::tt_bh_sfpu_pushc, Intrinsic::tt_bh_sfpu_popc,
        Intrinsic::tt_bh_sfpu_compc}) {
    SCOPED_TRACE(ID);
    for (unsigned Count : {0u, 1u, 2u, 3u}) {
      if (Count == (ID == Intrinsic::tt_bh_sfpu_compc ? 0u : 2u))
        continue;
      SCOPED_TRACE(Count);
      SmallVector<TensixBoundValue, 3> Args(Count, constant(0));
      auto Rejected = getTensixBoundSFPUContract(ID, Args);
      ASSERT_FALSE(bool(Rejected));
      EXPECT_EQ(toString(Rejected.takeError()),
                "bound SFPU argument tuple has the wrong size");
    }
  }
}

TEST(RISCVTensixConfiguration, ConfigurationResetHasProvedEmptyRequirements) {
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_config_reset, {});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

TEST(RISCVTensixConfiguration, NoOpHasProvedEmptyRequirements) {
  auto Contract = getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_nop, {});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

TEST(RISCVTensixConfiguration, ConfigurationResetClearsReceivedLaneFields) {
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_config_reset, {});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto *Reset =
      std::get_if<TensixSFPULaneConfigReset>(&Contract->getLaneControl());
  ASSERT_NE(Reset, nullptr);
  EXPECT_EQ(Reset->Columns, 8u);
  EXPECT_TRUE(Reset->ClearsRowMask);
  EXPECT_TRUE(Reset->ClearsDestIndex);
  EXPECT_TRUE(Reset->ClearsDstReadBlock);
  EXPECT_TRUE(Reset->ClearsDstWriteBlock);
  EXPECT_TRUE(Reset->ClearsDstReadColumnExchange);
  EXPECT_TRUE(Reset->ClearsDstWriteColumnExchange);
  EXPECT_TRUE(Reset->ClearsDstIndexCapture);
  EXPECT_TRUE(Reset->ClearsDstMoveBlock);
  EXPECT_TRUE(Reset->ClearsExchangeSourceBAndC);
}

TEST(RISCVTensixConfiguration, SignedImmediateHasProvedEmptyRequirements) {
  // This exact mode/zero-immediate case previously had no receiver. Its
  // reception must leave the other unreceived native forms rejected below.
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_loadi,
      {constant(7), constant(7), constant(0), constant(4)});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

class RISCVTensixUnreceivedConfigurationTest
    : public testing::TestWithParam<ConfigurationCase> {};

TEST_P(RISCVTensixUnreceivedConfigurationTest,
       MissingReceiverCannotBecomeProvedEmpty) {
  auto Contract =
      getTensixBoundSFPUContract(GetParam().ID, GetParam().Arguments);
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_FALSE(Contract->getConfigurationFacts());
}

INSTANTIATE_TEST_SUITE_P(
    OtherNativeForms, RISCVTensixUnreceivedConfigurationTest,
    testing::Values(
        // Mode 4 remains legal on Blackhole without a configuration receiver;
        // this is distinct from SFPLOADI FloatA's proved empty requirements.
        ConfigurationCase{
            Intrinsic::tt_bh_sfpu_shft2,
            {constant(4), constant(4), constant(1), constant(0), constant(4)}},
        ConfigurationCase{
            Intrinsic::tt_bh_sfpu_config_creg,
            {constant(0), constant(11), constant(0), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_iadd,
                          {constant(0), constant(0), constant(1), constant(0)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_iadd,
                          {constant(0), constant(0), constant(1), constant(2)}},
        ConfigurationCase{Intrinsic::tt_bh_sfpu_iadd,
                          {constant(0), constant(0), constant(1), constant(8)}},
        ConfigurationCase{
            Intrinsic::tt_bh_sfpu_iadd,
            {constant(0), constant(0), constant(1), constant(10)}}));

class RISCVTensixMadConfigurationTest
    : public testing::TestWithParam<Intrinsic::ID> {};

TEST_P(RISCVTensixMadConfigurationTest,
       RequirementNamesOnlyWriteArgumentAndAffectedRegisters) {
  // TEN-2932 applies to MAD on both architectures. The native write class
  // retains its requirement below L4; high source registers add no requirement.
  for (uint32_t Destination : {0u, 7u}) {
    SCOPED_TRACE(Destination);
    auto Contract = getTensixBoundSFPUContract(
        GetParam(), {constant(Destination), constant(Destination), constant(4),
                     constant(5), constant(6), constant(0)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const auto &Facts = Contract->getConfigurationFacts();
    ASSERT_TRUE(Facts);
    ASSERT_EQ(Facts->Requirements.size(), 1u);
    const auto &Requirement = Facts->Requirements.front();
    EXPECT_EQ(Requirement.Argument, 0u);
    EXPECT_EQ(Requirement.Field, TensixSFPUConfigurationField::EnableDestIndex);
    EXPECT_EQ(Requirement.Registers, (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
    EXPECT_TRUE(any_of(Contract->getRegisterArguments(), [&](const auto &R) {
      return R.Argument == Requirement.Argument &&
             R.Role == TensixBoundArgumentRole::WriteLReg;
    }));
  }
}

INSTANTIATE_TEST_SUITE_P(Direct0, RISCVTensixMadConfigurationTest,
                         testing::Values(Intrinsic::tt_wh_sfpu_mad,
                                         Intrinsic::tt_bh_sfpu_mad));

} // namespace
