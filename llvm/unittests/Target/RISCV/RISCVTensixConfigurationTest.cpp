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
