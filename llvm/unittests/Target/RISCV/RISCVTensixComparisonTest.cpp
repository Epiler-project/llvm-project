//===-- RISCVTensixComparisonTest.cpp ------------------------------------===//
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

class RISCVTensixComparisonTest : public testing::TestWithParam<Intrinsic::ID> {};

TEST_P(RISCVTensixComparisonTest, ReadsBothSourcesAndUpdatesOnlyConditionState) {
  const TensixBoundValue Unbound{TensixBoundValueKind::UnboundRegister};
  auto Contract = getTensixBoundSFPUContract(GetParam(),
                                           {Unbound, Unbound, constant(1)});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_EQ(Contract->getArgumentCount(), 3u);
  EXPECT_EQ(Contract->getArgumentRoles(),
            (ArrayRef<TensixBoundArgumentRole>{
                TensixBoundArgumentRole::ReadLReg,
                TensixBoundArgumentRole::ReadRegister,
                TensixBoundArgumentRole::LogicalImmediate}));
  ASSERT_EQ(Contract->getRegisterArguments().size(), 2u);
  EXPECT_EQ(Contract->getRegisterArguments()[0].AllowedRegisters,
            (SmallVector<uint32_t, 16>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(Contract->getRegisterArguments()[1].AllowedRegisters,
            (SmallVector<uint32_t, 16>{0, 1, 2, 3, 4, 5, 6, 7,
                                        8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_TRUE(Contract->getRegisterConstraints().empty());
  EXPECT_TRUE(Contract->getTransferFacts().empty());
  // A data-dependent predicate is not a constant SFPENCC lane transition.
  EXPECT_TRUE(std::holds_alternative<std::monostate>(Contract->getLaneControl()));
  ASSERT_TRUE(Contract->getConfigurationFacts());
  EXPECT_TRUE(Contract->getConfigurationFacts()->Requirements.empty());
  unsigned RegisterReads = 0, CCReadWrites = 0, ConfigurationReads = 0,
           IssueReadWrites = 0;
  for (const auto &Effect : Contract->getArchitecturalEffects()) {
    if (const auto *Argument =
            std::get_if<TensixBoundArgumentRef>(&Effect.Resource)) {
      EXPECT_LT(Argument->Argument, 2u);
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      ++RegisterReads;
      continue;
    }
    const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
    ASSERT_NE(State, nullptr);
    if (*State == TensixSFPUState::CC) {
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
      ++CCReadWrites;
    } else if (*State == TensixSFPUState::Configuration) {
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::Read);
      ++ConfigurationReads;
    } else {
      EXPECT_EQ(*State, TensixSFPUState::Issue);
      EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
      ++IssueReadWrites;
    }
  }
  EXPECT_EQ(RegisterReads, 2u);
  EXPECT_EQ(CCReadWrites, 1u);
  EXPECT_EQ(ConfigurationReads, 1u);
  EXPECT_EQ(IssueReadWrites, 1u);
}

TEST_P(RISCVTensixComparisonTest, AllowsReadAliasAndRejectsWritesOrStackModes) {
  for (int64_t Source : {0, 7}) {
    auto Contract = getTensixBoundSFPUContract(
        GetParam(), {constant(Source), constant(Source), constant(1)});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    EXPECT_TRUE(Contract->getRegisterConstraints().empty());
  }
  for (int64_t Mode = 0; Mode != 16; ++Mode) {
    if (Mode == 1)
      continue;
    auto Rejected = getTensixBoundSFPUContract(
        GetParam(), {constant(0), constant(1), constant(Mode)});
    ASSERT_FALSE(bool(Rejected)) << Mode;
    EXPECT_NE(toString(Rejected.takeError()).find("mode must be one of {1}"),
              std::string::npos);
  }
  for (int64_t Backdoor : {8, 11, 12, 15}) {
    auto Rejected = getTensixBoundSFPUContract(
        GetParam(), {constant(Backdoor), constant(0), constant(1)});
    ASSERT_FALSE(bool(Rejected));
    consumeError(Rejected.takeError());
  }
}

INSTANTIATE_TEST_SUITE_P(BlackholeConditionOnly, RISCVTensixComparisonTest,
                         testing::Values(Intrinsic::riscv_tt_bound_sfpgt,
                                         Intrinsic::riscv_tt_bound_sfple));
} // namespace
