//===-- RISCVTensixConfigurationTest.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IntrinsicsRISCV.h"
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
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfptransp, Args);
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

TEST_P(RISCVTensixDestinationConfigurationTest,
       RequirementNamesOnlyActualWriteArgumentAndAffectedRegisters) {
  // Requirements describe the native write class, including when its current
  // binding is below L4. A high-numbered source or tied old value is not a
  // second write and must not acquire its own requirement.
  for (uint32_t Destination : {0u, 7u}) {
    SCOPED_TRACE(Destination);
    auto Args = GetParam().Arguments;
    Args[0] = constant(Destination);
    if (GetParam().ID != Intrinsic::riscv_tt_bound_sfpmov_all)
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
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpmov_all,
                          {constant(0), constant(6)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpmov_prng_advance,
                          {constant(0), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpmov,
                          {constant(0), constant(0), constant(6), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpmov,
                          {constant(0), constant(0), constant(6), constant(1)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpiadd,
                          {constant(0), constant(0), constant(6), constant(4)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpiadd,
                          {constant(0), constant(0), constant(6), constant(6)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpand,
                          {constant(0), constant(0), constant(6)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpor,
                          {constant(0), constant(0), constant(6)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpxor,
                          {constant(0), constant(0), constant(6)}},
        ConfigurationCase{
            Intrinsic::riscv_tt_bound_sfpmul,
            {constant(0), constant(0), constant(1), constant(2), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpnot,
                          {constant(0), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfparecip,
                          {constant(0), constant(0), constant(6), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfparecip,
                          {constant(0), constant(0), constant(6), constant(1)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfparecip,
                          {constant(0), constant(0), constant(6), constant(2)}}));

TEST(RISCVTensixConfiguration, ReciprocalUnboundWriteRetainsErratumRequirement) {
  // Blackhole SFPCONFIG shares the Wormhole definition. TEN-2932 exempts
  // SFPLOAD/LOADI/SWAP/TRANSP, not SFPARECIP. The requirement must survive
  // until the one existing binder chooses the actual destination.
  const TensixBoundValue Unbound{TensixBoundValueKind::UnboundRegister};
  for (int64_t Mode : {0, 1, 2}) {
    SCOPED_TRACE(Mode);
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::riscv_tt_bound_sfparecip,
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
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfploadi,
      {constant(7), constant(7), constant(0x1234), constant(GetParam())});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

INSTANTIATE_TEST_SUITE_P(ReceivedImmediateForms,
                         RISCVTensixLoadImmediateConfigurationTest,
                         testing::Values(0, 8, 10));

class RISCVTensixDstConfigurationTest
    : public testing::TestWithParam<ConfigurationCase> {};

TEST_P(RISCVTensixDstConfigurationTest,
       NativeDstTransferHasProvedEmptyRequirements) {
  for (uint32_t Register : {0u, 7u}) {
    SCOPED_TRACE(Register);
    auto Args = GetParam().Arguments;
    Args[0] = constant(Register);
    if (GetParam().ID == Intrinsic::riscv_tt_bound_sfpload)
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
    testing::Values(ConfigurationCase{Intrinsic::riscv_tt_bound_sfpload,
                                      {constant(0), constant(0), constant(0),
                                       constant(0), constant(3)}},
                    ConfigurationCase{
                        Intrinsic::riscv_tt_bound_sfpstore,
                        {constant(0), constant(0), constant(0), constant(3)}}));

TEST(RISCVTensixConfiguration, DstTransferStillRequiresValidRegisterBinding) {
  auto Load = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfpload,
      {constant(7), constant(6), constant(0), constant(0), constant(3)});
  EXPECT_FALSE(bool(Load));
  if (!Load)
    consumeError(Load.takeError());

  auto Store = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfpstore,
      {constant(16), constant(0), constant(0), constant(3)});
  EXPECT_FALSE(bool(Store));
  if (!Store)
    consumeError(Store.takeError());
}

TEST(RISCVTensixConfiguration, ConditionControlHasProvedEmptyRequirements) {
  auto Contract = getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpencc,
                                             {constant(0), constant(2)});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

class RISCVTensixConditionStackConfigurationTest
    : public testing::TestWithParam<Intrinsic::ID> {};

TEST_P(RISCVTensixConditionStackConfigurationTest,
       PlainStackHasProvedEmptyRequirementsAndExactStateEffects) {
  SmallVector<TensixBoundValue, 2> Args;
  if (GetParam() != Intrinsic::riscv_tt_bound_sfpcompc)
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
  case Intrinsic::riscv_tt_bound_sfppushc:
    Expected.append(
        {{State::CC, Access::Read}, {State::CCStack, Access::ReadWrite}});
    break;
  case Intrinsic::riscv_tt_bound_sfppopc:
    Expected.append(
        {{State::CC, Access::Write}, {State::CCStack, Access::ReadWrite}});
    break;
  case Intrinsic::riscv_tt_bound_sfpcompc:
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
  if (GetParam() != Intrinsic::riscv_tt_bound_sfpcompc)
    Args = {constant(0), constant(0)};
  auto Contract = getTensixBoundSFPUContract(GetParam(), Args);
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto *Update =
      std::get_if<TensixSFPUConditionStackUpdate>(&Contract->getLaneControl());
  ASSERT_NE(Update, nullptr);
  using Action = TensixSFPUConditionStackAction;
  switch (GetParam()) {
  case Intrinsic::riscv_tt_bound_sfppushc:
    EXPECT_EQ(Update->Action, Action::Push);
    break;
  case Intrinsic::riscv_tt_bound_sfppopc:
    EXPECT_EQ(Update->Action, Action::Pop);
    break;
  case Intrinsic::riscv_tt_bound_sfpcompc:
    EXPECT_EQ(Update->Action, Action::Complement);
    break;
  default:
    FAIL() << "unexpected stack fixture";
  }
}

INSTANTIATE_TEST_SUITE_P(PlainConditionStack,
                         RISCVTensixConditionStackConfigurationTest,
                         testing::Values(Intrinsic::riscv_tt_bound_sfppushc,
                                         Intrinsic::riscv_tt_bound_sfppopc,
                                         Intrinsic::riscv_tt_bound_sfpcompc));

TEST(RISCVTensixConfiguration,
     ModeOneStackHasDistinctTypedActionsAndExactStateEffects) {
  for (Intrinsic::ID ID : {Intrinsic::riscv_tt_bound_sfppushc,
                           Intrinsic::riscv_tt_bound_sfppopc}) {
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
    EXPECT_EQ(Update->Action, ID == Intrinsic::riscv_tt_bound_sfppushc
                                  ? Action::ReplaceTop
                                  : Action::Peek);
    ASSERT_TRUE(ModeOne->getConfigurationFacts());
    EXPECT_TRUE(ModeOne->getConfigurationFacts()->Requirements.empty());
    EXPECT_TRUE(ModeOne->getRegisterArguments().empty());
    EXPECT_TRUE(ModeOne->getRegisterConstraints().empty());
    EXPECT_TRUE(ModeOne->getTransferFacts().empty());

    using State = TensixSFPUState;
    using Access = TensixSFPUAccess;
    const bool Replace = ID == Intrinsic::riscv_tt_bound_sfppushc;
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
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::riscv_tt_bound_sfppopc, {constant(0), constant(Mode)});
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
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::riscv_tt_bound_sfppopc,
      {constant(0), constant(2)});
  ASSERT_FALSE(bool(Contract));
  EXPECT_EQ(toString(Contract.takeError()),
            "bound SFPU unsupported mode at operand 1");
}

TEST(RISCVTensixConfiguration, ConditionStackRejectsUnsupportedFields) {
  // VD stays zero. Push receives only modes zero and one; pop additionally
  // receives Blackhole's mode-three/four Boolean-combine forms. Other
  // mutation and flag-setting modes remain unavailable.
  for (Intrinsic::ID ID : {Intrinsic::riscv_tt_bound_sfppushc,
                           Intrinsic::riscv_tt_bound_sfppopc}) {
    SCOPED_TRACE(ID);
    for (unsigned Field : {0u, 1u}) {
      SCOPED_TRACE(Field);
      for (int64_t Value = -1; Value != 17; ++Value) {
        const bool ValidMode =
            Field == 1 && (Value == 0 || Value == 1 ||
                           (ID == Intrinsic::riscv_tt_bound_sfppopc &&
                            (Value == 3 || Value == 4)));
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
       {Intrinsic::riscv_tt_bound_sfppushc, Intrinsic::riscv_tt_bound_sfppopc,
        Intrinsic::riscv_tt_bound_sfpcompc}) {
    SCOPED_TRACE(ID);
    for (unsigned Count : {0u, 1u, 2u, 3u}) {
      if (Count == (ID == Intrinsic::riscv_tt_bound_sfpcompc ? 0u : 2u))
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
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpconfig_reset, {});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

TEST(RISCVTensixConfiguration, NoOpHasProvedEmptyRequirements) {
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpnop, {});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  const auto &Facts = Contract->getConfigurationFacts();
  ASSERT_TRUE(Facts);
  EXPECT_TRUE(Facts->Requirements.empty());
}

TEST(RISCVTensixConfiguration, ConfigurationResetClearsReceivedLaneFields) {
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::riscv_tt_bound_sfpconfig_reset, {});
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
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpcast,
                          {constant(0), constant(0), constant(1), constant(1)}},
        ConfigurationCase{
            Intrinsic::riscv_tt_bound_sfpconfig_creg,
            {constant(0), constant(11), constant(0), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfploadi,
                          {constant(7), constant(7), constant(0), constant(1)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfploadi,
                          {constant(7), constant(7), constant(0), constant(2)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfploadi,
                          {constant(7), constant(7), constant(0), constant(4)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpiadd,
                          {constant(0), constant(0), constant(1), constant(0)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpiadd,
                          {constant(0), constant(0), constant(1), constant(2)}},
        ConfigurationCase{Intrinsic::riscv_tt_bound_sfpiadd,
                          {constant(0), constant(0), constant(1), constant(8)}},
        ConfigurationCase{
            Intrinsic::riscv_tt_bound_sfpiadd,
            {constant(0), constant(0), constant(1), constant(10)}}));

} // namespace
