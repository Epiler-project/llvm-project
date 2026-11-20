//===-- RISCVTensixTransferTest.cpp -------------------------------------===//
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
TEST(RISCVTensixTransfer, MaxSwapDemandsBothInputsForBothSimultaneousResults) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_swap, Intrinsic::tt_bh_sfpu_swap}) {
    auto Contract =
        getTensixBoundSFPUContract(ID, {{TensixBoundValueKind::Constant, 2},
                                        {TensixBoundValueKind::Constant, 0},
                                        {TensixBoundValueKind::Constant, 2},
                                        {TensixBoundValueKind::Constant, 0},
                                        {TensixBoundValueKind::Constant, 1}});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    const TensixBoundArgumentRole Roles[] = {
        TensixBoundArgumentRole::WriteLReg, TensixBoundArgumentRole::WriteLReg,
        TensixBoundArgumentRole::OldDestination,
        TensixBoundArgumentRole::OldDestination,
        TensixBoundArgumentRole::LogicalImmediate};
    EXPECT_EQ(Contract->getArgumentRoles(), ArrayRef(Roles));
    auto Transfers = Contract->getTransferFacts();
    ASSERT_EQ(Transfers.size(), 2u);
    for (unsigned Destination = 0; Destination != 2; ++Destination) {
      const auto &Facts = Transfers[Destination];
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
      EXPECT_EQ(Facts.DestinationArgument, Destination);
      EXPECT_EQ(Facts.OldArgument, Destination + 2);
      EXPECT_FALSE(Facts.SourceArgument);
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
      EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
      EXPECT_EQ(Facts.Inputs[1].DemandedBits, 0xffffffffu);
      EXPECT_FALSE(Facts.ImmediateValue);
    }
  }
}

TEST(RISCVTensixTransfer, TransposeHasReceivedRegisterPermutation) {
  for (auto Kind : {TensixBoundValueKind::Constant,
                    TensixBoundValueKind::UnboundRegister}) {
    SCOPED_TRACE(static_cast<unsigned>(Kind));
    SmallVector<TensixBoundValue, 12> Args;
    for (unsigned I = 0; I != 12; ++I)
      Args.push_back({Kind, I < 8 ? I : I - 8});
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_transp, Args);
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    auto Transfers = Contract->getTransferFacts();
    ASSERT_EQ(Transfers.size(), 4u);
    for (unsigned Destination = 0; Destination != Transfers.size();
         ++Destination) {
      SCOPED_TRACE(Destination);
      const auto &Facts = Transfers[Destination];
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::RegisterPermutation);
      EXPECT_EQ(Facts.DestinationArgument, Destination);
      EXPECT_FALSE(Facts.SourceArgument);
      EXPECT_EQ(Facts.OldArgument, 8 + Destination);
      EXPECT_EQ(Facts.DemandedOldBits, 0u);
      EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
      EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
      EXPECT_TRUE(Facts.RequiresLaneEnable);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      EXPECT_TRUE(Facts.RequiresConditionState);
      EXPECT_TRUE(Facts.RequiresRowMaskState);
      EXPECT_TRUE(Facts.SourceLanes.empty());
      ASSERT_EQ(Facts.Inputs.size(), 4u);
      for (unsigned Source = 0; Source != 4; ++Source) {
        EXPECT_EQ(Facts.Inputs[Source].Argument, 8 + Source);
        EXPECT_EQ(Facts.Inputs[Source].DemandedBits, 0xffffffffu);
      }
      ASSERT_EQ(Facts.LaneSources.size(), 32u);
      for (unsigned Lane = 0; Lane != 32; ++Lane) {
        SCOPED_TRACE(Lane);
        EXPECT_EQ(Facts.LaneSources[Lane].Argument, 8 + Lane / 8);
        EXPECT_EQ(Facts.LaneSources[Lane].Lane, Destination * 8 + Lane % 8);
      }
    }
  }
}

TEST(RISCVTensixTransfer, TransposeRejectsEveryWrongFixedGroupArgument) {
  SmallVector<TensixBoundValue, 12> Args;
  for (unsigned I = 0; I != 12; ++I)
    Args.push_back({TensixBoundValueKind::Constant, I < 8 ? I : I - 8});
  for (unsigned I = 0; I != Args.size(); ++I) {
    SCOPED_TRACE(I);
    auto Wrong = Args;
    Wrong[I].Constant = (Wrong[I].Constant + 1) % 8;
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_transp, Wrong);
    ASSERT_FALSE(bool(Contract));
    EXPECT_NE(toString(Contract.takeError()).find("fixed register group"),
              std::string::npos);
  }
}

TEST(RISCVTensixTransfer, RotateKeepsMaskedBitPreservingTransfer) {
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_shft2, {{TensixBoundValueKind::Constant, 0},
                                    {TensixBoundValueKind::Constant, 0},
                                    {TensixBoundValueKind::Constant, 1},
                                    {TensixBoundValueKind::Constant, 0},
                                    {TensixBoundValueKind::Constant, 3}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
  const auto &Facts = Contract->getTransferFacts().front();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LanePermutation);
  EXPECT_EQ(Facts.DestinationArgument, 0u);
  EXPECT_EQ(Facts.SourceArgument, 2u);
  EXPECT_EQ(Facts.OldArgument, 1u);
  EXPECT_TRUE(Facts.LaneSources.empty());
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
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_shft2, {{TensixBoundValueKind::Constant, 6},
                                    {TensixBoundValueKind::Constant, 6},
                                    {TensixBoundValueKind::Constant, 5},
                                    {TensixBoundValueKind::Constant, 0},
                                    {TensixBoundValueKind::Constant, 4}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_TRUE(Contract->getTransferFacts().empty());
}

TEST(RISCVTensixTransfer, ImmediateShiftDemandsTheSourceBitsSurvivingItsSignedDistance) {
  for (auto ID : {Intrinsic::tt_bh_sfpu_shft_i, Intrinsic::tt_wh_sfpu_shft_i})
    for (unsigned Mode : {1u, 3u, 5u, 7u})
      for (int Count : {-2049, -2048, -1, 0, 1, 2047, 2048}) {
        auto Contract = getTensixBoundSFPUContract(ID, {
            {TensixBoundValueKind::Constant, 4}, {TensixBoundValueKind::Constant, 4},
            {TensixBoundValueKind::Constant, Count}, {TensixBoundValueKind::Constant, Mode}});
        if (Count < -2048 || Count > 2047 || (ID == Intrinsic::tt_wh_sfpu_shft_i && Mode != 1)) {
          ASSERT_FALSE(bool(Contract));
          consumeError(Contract.takeError());
          continue;
        }
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        ASSERT_TRUE(Contract->getConfigurationFacts());
        ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
        const auto &T = Contract->getTransferFacts().front();
        EXPECT_EQ(T.OldArgument, 1u);
        ASSERT_EQ(T.Inputs.size(), 1u);
        EXPECT_EQ(T.Inputs[0].Argument, 1u);
        uint32_t Expected = 0xffffffffu;
        if (Count == -1) Expected = 0xfffffffeu;
        if (Count == 1) Expected = 0x7fffffffu;
        if (Count == 2047) Expected = 1u;
        EXPECT_EQ(T.Inputs[0].DemandedBits, Expected);
      }
}

TEST(RISCVTensixTransfer, LogicalVectorShiftReadsDirectionAndDistance) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_shft_v, Intrinsic::tt_bh_sfpu_shft_v}) {
    auto Contract = getTensixBoundSFPUContract(
        ID, {{TensixBoundValueKind::Constant, 5},
             {TensixBoundValueKind::Constant, 5},
             {TensixBoundValueKind::Constant, 2},
             {TensixBoundValueKind::Constant, 0}});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Facts = Contract->getTransferFacts().front();
    ASSERT_EQ(Facts.Inputs.size(), 2u);
    EXPECT_EQ(Facts.Inputs[0].Argument, 1u);
    EXPECT_EQ(Facts.Inputs[0].DemandedBits, 0xffffffffu);
    EXPECT_EQ(Facts.Inputs[1].Argument, 2u);
    EXPECT_EQ(Facts.Inputs[1].DemandedBits, 0x8000001fu);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_EQ(Facts.OldArgument, 1u);
    const auto &Configuration = Contract->getConfigurationFacts();
    ASSERT_TRUE(Configuration);
    ASSERT_EQ(Configuration->Requirements.size(), 1u);
    const auto &Guard = Configuration->Requirements.front();
    EXPECT_EQ(Guard.Argument, 0u);
    EXPECT_EQ(Guard.Field, TensixSFPUConfigurationField::EnableDestIndex);
    EXPECT_EQ(Guard.Registers, (SmallVector<uint32_t, 4>{4, 5, 6, 7}));
  }
}

TEST(RISCVTensixTransfer, ArithmeticVectorShiftRequiresBlackhole) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_shft_v, Intrinsic::tt_bh_sfpu_shft_v}) {
    auto Contract = getTensixBoundSFPUContract(
        ID, {{TensixBoundValueKind::Constant, 0},
             {TensixBoundValueKind::Constant, 0},
             {TensixBoundValueKind::Constant, 1},
             {TensixBoundValueKind::Constant, 2}});
    if (ID == Intrinsic::tt_wh_sfpu_shft_v) {
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
      continue;
    }
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Transfer = Contract->getTransferFacts().front();
    ASSERT_EQ(Transfer.Inputs.size(), 2u);
    EXPECT_EQ(Transfer.Inputs[0].DemandedBits, 0xffffffffu);
    EXPECT_EQ(Transfer.Inputs[1].DemandedBits, 0x8000001fu);
    EXPECT_EQ(Transfer.OldArgument, 1u);
    EXPECT_TRUE(Transfer.PreservesInactiveLanes);
  }
}

TEST(RISCVTensixTransfer, ExtractMantissaPreservesEachKnownFractionBit) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_exman, Intrinsic::tt_bh_sfpu_exman}) {
    for (int64_t Mode : {0, 1}) {
      auto Contract = getTensixBoundSFPUContract(
          ID, {{TensixBoundValueKind::Constant, 6},
               {TensixBoundValueKind::Constant, 6},
               {TensixBoundValueKind::Constant, 2},
               {TensixBoundValueKind::Constant, Mode}});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Facts = Contract->getTransferFacts().front();
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::MaskedIdentity);
      EXPECT_EQ(Facts.SourceArgument, 2u);
      EXPECT_EQ(Facts.OldArgument, 1u);
      EXPECT_EQ(Facts.CopiedBits, 0x007fffffu);
      EXPECT_EQ(Facts.ConstantBits, 0xff800000u);
      ASSERT_TRUE(Facts.ImmediateValue);
      EXPECT_EQ(Facts.ImmediateValue->KnownBitsMask, 0xff800000u);
      EXPECT_EQ(Facts.ImmediateValue->Value, Mode == 0 ? 0x00800000u : 0u);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      ASSERT_TRUE(Contract->getConfigurationFacts());
      ASSERT_EQ(Contract->getConfigurationFacts()->Requirements.size(), 1u);
      EXPECT_EQ(Contract->getConfigurationFacts()->Requirements.front().Field,
                TensixSFPUConfigurationField::EnableDestIndex);
    }
  }
}

TEST(RISCVTensixTransfer, ExponentExtractionUsesOnlyExponentBitsAndPreservesCC) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_exexp, Intrinsic::tt_bh_sfpu_exexp}) {
    for (int64_t Mode : {0, 1, 2, 3, 8, 9, 10, 11}) {
      SCOPED_TRACE(Mode);
      auto Contract = getTensixBoundSFPUContract(
          ID, {{TensixBoundValueKind::Constant, 5},
               {TensixBoundValueKind::Constant, 5},
               {TensixBoundValueKind::Constant, 2},
               {TensixBoundValueKind::Constant, Mode}});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      bool FoundCC = false;
      for (const auto &Effect : Contract->getArchitecturalEffects()) {
        const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
        if (State && *State == TensixSFPUState::CC) {
          FoundCC = true;
          EXPECT_EQ(Effect.Access, Mode <= 1 ? TensixSFPUAccess::Read
                                              : TensixSFPUAccess::ReadWrite);
        }
      }
      ASSERT_TRUE(FoundCC);
      if (Mode > 1) {
        EXPECT_TRUE(Contract->getTransferFacts().empty());
        continue;
      }
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Transfer = Contract->getTransferFacts().front();
      EXPECT_EQ(Transfer.OldArgument, 1u);
      EXPECT_TRUE(Transfer.PreservesInactiveLanes);
      ASSERT_EQ(Transfer.Inputs.size(), 1u);
      EXPECT_EQ(Transfer.Inputs[0].Argument, 2u);
      EXPECT_EQ(Transfer.Inputs[0].DemandedBits, 0x7f800000u);
      ASSERT_TRUE(Contract->getConfigurationFacts());
      ASSERT_EQ(Contract->getConfigurationFacts()->Requirements.size(), 1u);
      EXPECT_EQ(Contract->getConfigurationFacts()->Requirements.front().Field,
                TensixSFPUConfigurationField::EnableDestIndex);
    }
  }
}

TEST(RISCVTensixTransfer, ImmediateExponentAndMantissaKeepUnmodifiedBits) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_setexp_i, Intrinsic::tt_bh_sfpu_setexp_i,
                  Intrinsic::tt_wh_sfpu_setman_i, Intrinsic::tt_bh_sfpu_setman_i}) {
    const bool Exponent = ID == Intrinsic::tt_wh_sfpu_setexp_i ||
                          ID == Intrinsic::tt_bh_sfpu_setexp_i;
    const int64_t Immediate = Exponent ? 0xa5 : 0xb4c;
    const uint32_t Mask = Exponent ? 0x7f800000u : 0x007fffffu;
    auto Contract = getTensixBoundSFPUContract(
        ID, {{TensixBoundValueKind::Constant, 6},
             {TensixBoundValueKind::Constant, 6},
             {TensixBoundValueKind::Constant, 2},
             {TensixBoundValueKind::Constant, Immediate},
             {TensixBoundValueKind::Constant, 1}});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Facts = Contract->getTransferFacts().front();
    EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::MaskedIdentity);
    EXPECT_EQ(Facts.DestinationArgument, 0u);
    EXPECT_EQ(Facts.OldArgument, 1u);
    EXPECT_EQ(Facts.SourceArgument, 2u);
    EXPECT_EQ(Facts.ConstantBits, Mask);
    EXPECT_EQ(Facts.CopiedBits, ~Mask);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    ASSERT_TRUE(Facts.ImmediateValue);
    EXPECT_EQ(Facts.ImmediateValue->KnownBitsMask, Mask);
    EXPECT_EQ(Facts.ImmediateValue->Value, Exponent ? 0x52800000u : 0x005a6000u);
    ASSERT_TRUE(Contract->getConfigurationFacts());
    const auto &Guards = Contract->getConfigurationFacts()->Requirements;
    ASSERT_EQ(Guards.size(), 1u);
    EXPECT_EQ(Guards.front().Field, TensixSFPUConfigurationField::EnableDestIndex);
  }
}

TEST(RISCVTensixTransfer, DirectLookupsReadTheirFixedGroupAndPreserveInactiveOld) {
  for (auto ID : {Intrinsic::tt_bh_sfpu_lut, Intrinsic::tt_wh_sfpu_lut,
                  Intrinsic::tt_bh_sfpu_lutfp32_6r, Intrinsic::tt_wh_sfpu_lutfp32_6r}) {
    const bool Packed8 = ID == Intrinsic::tt_bh_sfpu_lut || ID == Intrinsic::tt_wh_sfpu_lut;
    const unsigned NumInputs = Packed8 ? 4 : 7;
    for (unsigned Mode : {0u, 2u, 3u, 4u, 6u, 7u}) {
      SmallVector<TensixBoundValue> Args(2, {TensixBoundValueKind::Constant, 0});
      for (unsigned Slot = 0; Slot != NumInputs; ++Slot)
        Args.push_back({TensixBoundValueKind::Constant, Slot});
      Args.push_back({TensixBoundValueKind::Constant, Mode});
      auto Contract = getTensixBoundSFPUContract(ID, Args);
      if (Packed8 && (Mode & 3)) {
        ASSERT_FALSE(bool(Contract));
        consumeError(Contract.takeError());
        continue;
      }
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_TRUE(Contract->getConfigurationFacts());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Transfer = Contract->getTransferFacts().front();
      EXPECT_EQ(Transfer.OldArgument, 1u);
      EXPECT_TRUE(Transfer.PreservesInactiveLanes);
      ASSERT_EQ(Transfer.Inputs.size(), NumInputs);
      for (unsigned Slot = 0; Slot != NumInputs; ++Slot) {
        uint32_t Bits = Packed8 ? 0xffffu : 0xffffffffu;
        if (Slot == 3)
          Bits = Mode & 4 ? 0xffffffffu : 0x7fffffffu;
        EXPECT_EQ(Transfer.Inputs[Slot].Argument, Slot + 2);
        EXPECT_EQ(Transfer.Inputs[Slot].DemandedBits, Bits);
      }
      Args[2].Constant = 1;
      auto WrongSlot = getTensixBoundSFPUContract(ID, Args);
      ASSERT_FALSE(bool(WrongSlot));
      consumeError(WrongSlot.takeError());
    }
  }
}

TEST(RISCVTensixTransfer, CastSeparatesStochasticStateAndGenerationSpecificModes) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_cast, Intrinsic::tt_bh_sfpu_cast}) {
    for (int Mode = 0; Mode != 4; ++Mode) {
      auto Contract = getTensixBoundSFPUContract(ID, {
          {TensixBoundValueKind::Constant, 4}, {TensixBoundValueKind::Constant, 4},
          {TensixBoundValueKind::Constant, 1}, {TensixBoundValueKind::Constant, Mode}});
      if (ID == Intrinsic::tt_wh_sfpu_cast && Mode > 1) {
        ASSERT_FALSE(bool(Contract));
        consumeError(Contract.takeError());
        continue;
      }
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      bool UpdatesPRNG = false;
      for (const auto &Effect : Contract->getArchitecturalEffects()) {
        auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
        if (State && *State == TensixSFPUState::PRNG) {
          UpdatesPRNG = true;
          EXPECT_EQ(Effect.Access, TensixSFPUAccess::ReadWrite);
        }
      }
      EXPECT_EQ(UpdatesPRNG, Mode == 1);
      ASSERT_TRUE(Contract->getConfigurationFacts());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Transfer = Contract->getTransferFacts().front();
      EXPECT_EQ(Transfer.Kind, TensixSFPUTransferKind::LaneWise);
      EXPECT_EQ(Transfer.DestinationArgument, 0u);
      EXPECT_EQ(Transfer.OldArgument, 1u);
      EXPECT_EQ(Transfer.GuaranteedWriteBits, 0xffffffffu);
      EXPECT_TRUE(Transfer.RequiresLaneEnable);
      EXPECT_TRUE(Transfer.PreservesInactiveLanes);
      EXPECT_TRUE(Transfer.RequiresConditionState);
      EXPECT_TRUE(Transfer.RequiresRowMaskState);
      ASSERT_EQ(Transfer.Inputs.size(), 1u);
      EXPECT_EQ(Transfer.Inputs.front().Argument, 2u);
      EXPECT_EQ(Transfer.Inputs.front().DemandedBits, 0xffffffffu);
    }
  }
}

TEST(RISCVTensixTransfer, ImmediateRoundingProjectsFPSourceAndInactiveOld) {
  using Kind = TensixBoundValueKind;
  using Role = TensixBoundArgumentRole;
  for (auto ID :
       {Intrinsic::tt_wh_sfpu_stochrnd_i, Intrinsic::tt_bh_sfpu_stochrnd_i}) {
    const int MaximumRound = ID == Intrinsic::tt_bh_sfpu_stochrnd_i ? 2 : 1;
    for (int Mode : {0, 1, 2, 3, 6, 7}) {
      for (int Round = 0; Round <= MaximumRound; ++Round) {
        SCOPED_TRACE(ID);
        SCOPED_TRACE(Mode);
        SCOPED_TRACE(Round);
        auto Contract =
            getTensixBoundSFPUContract(ID, {{Kind::Constant, 4},
                                            {Kind::Constant, 4},
                                            {Kind::Constant, 13},
                                            {Kind::Constant, 0},
                                            {Kind::Constant, Mode},
                                            {Kind::Constant, Round}});
        ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
        const Role Roles[] = {Role::WriteLReg,        Role::OldDestination,
                              Role::ReadRegister,     Role::LogicalImmediate,
                              Role::LogicalImmediate, Role::LogicalImmediate};
        EXPECT_EQ(Contract->getArgumentRoles(), ArrayRef(Roles));
        ASSERT_TRUE(Contract->getConfigurationFacts());
        const auto &Guards = Contract->getConfigurationFacts()->Requirements;
        ASSERT_EQ(Guards.size(), 1u);
        EXPECT_EQ(Guards[0].Argument, 0u);
        EXPECT_EQ(Guards[0].Field,
                  TensixSFPUConfigurationField::EnableDestIndex);
        EXPECT_EQ(ArrayRef(Guards[0].Registers),
                  (ArrayRef<uint32_t>{4, 5, 6, 7}));
        const auto Effects = Contract->getArchitecturalEffects();
        EXPECT_TRUE(any_of(Effects, [](const auto &Effect) {
          const auto *State = std::get_if<TensixSFPUState>(&Effect.Resource);
          return State && *State == TensixSFPUState::PRNG &&
                 Effect.Access == TensixSFPUAccess::ReadWrite;
        }));
        const auto Transfers = Contract->getTransferFacts();
        ASSERT_EQ(Transfers.size(), 1u);
        const auto &Facts = Transfers.front();
        EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
        EXPECT_EQ(Facts.DestinationArgument, 0u);
        EXPECT_EQ(Facts.OldArgument, 1u);
        EXPECT_FALSE(Facts.SourceArgument);
        EXPECT_EQ(Facts.DemandedOldBits, 0u);
        EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
        EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
        EXPECT_TRUE(Facts.RequiresLaneEnable);
        EXPECT_TRUE(Facts.PreservesInactiveLanes);
        EXPECT_TRUE(Facts.RequiresConditionState);
        EXPECT_TRUE(Facts.RequiresRowMaskState);
        EXPECT_FALSE(Facts.ImmediateValue);
        ASSERT_EQ(Facts.Inputs.size(), 1u);
        EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
        EXPECT_EQ(Facts.Inputs[0].DemandedBits,
                  Mode == 2 || Mode == 6 ? 0x7fffffffu : 0xffffffffu);
      }
    }
  }
}

TEST(RISCVTensixTransfer, ImmediateRoundingRetainsDescaleAndGenerationLimits) {
  using Kind = TensixBoundValueKind;
  for (auto ID :
       {Intrinsic::tt_wh_sfpu_stochrnd_i, Intrinsic::tt_bh_sfpu_stochrnd_i}) {
    auto Query = [&](int Descale, int Mode, int Round) {
      return getTensixBoundSFPUContract(ID, {{Kind::Constant, 0},
                                             {Kind::Constant, 0},
                                             {Kind::Constant, 9},
                                             {Kind::Constant, Descale},
                                             {Kind::Constant, Mode},
                                             {Kind::Constant, Round}});
    };
    for (int Mode : {4, 5}) {
      for (int Round : {0, 1}) {
        SCOPED_TRACE(Mode);
        SCOPED_TRACE(Round);
        auto Integer = Query(31, Mode, Round);
        ASSERT_TRUE(bool(Integer)) << toString(Integer.takeError());
        ASSERT_TRUE(Integer->getConfigurationFacts());
        const auto Transfers = Integer->getTransferFacts();
        ASSERT_EQ(Transfers.size(), 1u);
        const auto &Facts = Transfers.front();
        EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
        EXPECT_EQ(Facts.DestinationArgument, 0u);
        EXPECT_EQ(Facts.OldArgument, 1u);
        EXPECT_TRUE(Facts.PreservesInactiveLanes);
        EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
        ASSERT_EQ(Facts.Inputs.size(), 1u);
        EXPECT_EQ(Facts.Inputs[0].Argument, 2u);
        EXPECT_EQ(Facts.Inputs[0].DemandedBits,
                  Mode == 5 || Round == 1 ? 0xffffffffu : 0x7fffffffu);
      }
    }
    for (auto [Descale, Mode, Round] :
         {std::tuple{1, 1, 0}, std::tuple{32, 4, 0}, std::tuple{0, 8, 0},
          std::tuple{0, 6, ID == Intrinsic::tt_bh_sfpu_stochrnd_i ? 3 : 2}}) {
      auto Rejected = Query(Descale, Mode, Round);
      ASSERT_FALSE(bool(Rejected));
      consumeError(Rejected.takeError());
    }
  }
}

TEST(RISCVTensixTransfer, ImmediateRoundingContentsProjectsActualValues) {
  using Kind = TensixContentsValueKind;
  for (auto Architecture :
       {TensixSFPUArchitecture::Wormhole, TensixSFPUArchitecture::Blackhole}) {
    auto Info = getTensixContentsIntrinsicInfo(
        Intrinsic::riscv_tt_contents_stochrnd_immediate,
        {{Kind::Contents},
         {Kind::Contents},
         {Kind::Scalar, 0},
         {Kind::Scalar, 1},
         {Kind::Scalar, 0}},
        Architecture);
    ASSERT_TRUE(bool(Info)) << toString(Info.takeError());
    ASSERT_EQ(Info->Transfers.size(), 1u);
    const auto &Transfer = Info->Transfers.front();
    ASSERT_LT(Transfer.DestinationArgument, Info->TransferValues.size());
    const auto *Destination = std::get_if<TensixContentsResultRef>(
        &Info->TransferValues[Transfer.DestinationArgument]);
    ASSERT_NE(Destination, nullptr);
    EXPECT_EQ(Destination->Result, 0u);
    ASSERT_TRUE(Transfer.OldArgument);
    ASSERT_EQ(Transfer.Inputs.size(), 1u);
    for (auto [Index, Argument] :
         {std::pair{*Transfer.OldArgument, 0u},
          std::pair{Transfer.Inputs[0].Argument, 1u}}) {
      ASSERT_LT(Index, Info->TransferValues.size());
      const auto *Source =
          std::get_if<TensixContentsArgumentRef>(&Info->TransferValues[Index]);
      ASSERT_NE(Source, nullptr);
      EXPECT_EQ(Source->Argument, Argument);
    }
    ASSERT_TRUE(Info->Configuration);
    ASSERT_EQ(Info->Configuration->Requirements.size(), 1u);
    auto Guard = Info->Configuration->Requirements.front();
    ASSERT_LT(Guard.Argument, Info->TransferValues.size());
    const auto *Guarded = std::get_if<TensixContentsResultRef>(
        &Info->TransferValues[Guard.Argument]);
    ASSERT_NE(Guarded, nullptr);
    EXPECT_EQ(Guarded->Result, Destination->Result);
  }
}

TEST(RISCVTensixTransfer, RegisterFieldsDemandOnlyTheSelectedBitsFromOldAndC) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_setexp_v, Intrinsic::tt_bh_sfpu_setexp_v,
                  Intrinsic::tt_wh_sfpu_setman_v, Intrinsic::tt_bh_sfpu_setman_v,
                  Intrinsic::tt_wh_sfpu_setsgn_v, Intrinsic::tt_bh_sfpu_setsgn_v}) {
    const bool Exp = ID == Intrinsic::tt_wh_sfpu_setexp_v ||
                     ID == Intrinsic::tt_bh_sfpu_setexp_v;
    const bool Man = ID == Intrinsic::tt_wh_sfpu_setman_v ||
                     ID == Intrinsic::tt_bh_sfpu_setman_v;
    const uint32_t Field = Exp ? 0x7f800000u : Man ? 0x007fffffu : 0x80000000u;
    for (int Mode : {0, 2}) {
      if (Mode && !Exp)
        continue;
      auto Contract = getTensixBoundSFPUContract(
          ID, {{TensixBoundValueKind::Constant, 4},
               {TensixBoundValueKind::Constant, 4},
               {TensixBoundValueKind::Constant, 2},
               {TensixBoundValueKind::Constant, 4},
               {TensixBoundValueKind::Constant, Mode}});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Facts = Contract->getTransferFacts().front();
      EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
      EXPECT_TRUE(Facts.PreservesInactiveLanes);
      EXPECT_EQ(Facts.OldArgument, 1u);
      ASSERT_EQ(Facts.Inputs.size(), 2u);
      EXPECT_EQ(Facts.Inputs[0].Argument, 1u);
      EXPECT_EQ(Facts.Inputs[0].DemandedBits, Exp && !Mode ? 0xffu : Field);
      EXPECT_EQ(Facts.Inputs[1].Argument, 2u);
      EXPECT_EQ(Facts.Inputs[1].DemandedBits, ~Field);
      ASSERT_TRUE(Contract->getConfigurationFacts());
      EXPECT_EQ(Contract->getConfigurationFacts()->Requirements.size(), 1u);
    }
  }
}

TEST(RISCVTensixTransfer, BlackholeMul23ReadsLow23AndKeepsInactiveDestination) {
  for (int64_t Half : {0, 1}) {
    auto Contract = getTensixBoundSFPUContract(
        Intrinsic::tt_bh_sfpu_mul23,
        {{TensixBoundValueKind::Constant, 4},
         {TensixBoundValueKind::Constant, 4},
         {TensixBoundValueKind::Constant, 1},
         {TensixBoundValueKind::Constant, 2},
         {TensixBoundValueKind::Constant, Half}});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Transfer = Contract->getTransferFacts().front();
    EXPECT_EQ(Transfer.Kind, TensixSFPUTransferKind::LaneWise);
    EXPECT_EQ(Transfer.DestinationArgument, 0u);
    EXPECT_EQ(Transfer.OldArgument, 1u);
    EXPECT_EQ(Transfer.GuaranteedWriteBits, 0xffffffffu);
    EXPECT_TRUE(Transfer.PreservesInactiveLanes);
    ASSERT_EQ(Transfer.Inputs.size(), 2u);
    EXPECT_EQ(Transfer.Inputs[0].Argument, 2u);
    EXPECT_EQ(Transfer.Inputs[1].Argument, 3u);
    for (const auto &Input : Transfer.Inputs)
      EXPECT_EQ(Input.DemandedBits, 0x7fffffu);
    EXPECT_TRUE(any_of(Contract->getArchitecturalEffects(), [](const auto &E) {
      const auto *Reg = std::get_if<TensixFixedRegisterRef>(&E.Resource);
      return Reg && Reg->Number == 9 && E.Access == TensixSFPUAccess::Read;
    }));
    const auto &Config = Contract->getConfigurationFacts();
    ASSERT_TRUE(Config);
    ASSERT_EQ(Config->Requirements.size(), 1u);
    EXPECT_EQ(Config->Requirements.front().Field,
              TensixSFPUConfigurationField::EnableDestIndex);
  }
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
  ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
  const auto &Facts = Contract->getTransferFacts().front();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
  EXPECT_EQ(Facts.DestinationArgument, 0u);
  EXPECT_EQ(Facts.OldArgument, 1u);
  EXPECT_FALSE(Facts.SourceArgument);
  EXPECT_TRUE(Facts.LaneSources.empty());
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
      std::get<0>(GetParam()) == Intrinsic::tt_bh_sfpu_add ? 10u : 9u;
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
    testing::Combine(
        testing::Values(Intrinsic::tt_bh_sfpu_add, Intrinsic::tt_bh_sfpu_mul),
        testing::Values(int64_t(0), int64_t(1), int64_t(2), int64_t(3))));

TEST(RISCVTensixTransfer,
     MaskedNegationSeparatesNumericalSourceAndPassthrough) {
  auto Contract = getTensixBoundSFPUContract(
      Intrinsic::tt_bh_sfpu_mov, {{TensixBoundValueKind::Constant, 0},
                                  {TensixBoundValueKind::Constant, 0},
                                  {TensixBoundValueKind::Constant, 1},
                                  {TensixBoundValueKind::Constant, 1}});
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
    if (GetParam() != Intrinsic::tt_bh_sfpu_not)
      Args.push_back({TensixBoundValueKind::Constant, 1});
    return Args;
  }
};

TEST_P(RISCVTensixBitwiseTransferTest, DemandsEachActualNumericalInput) {
  auto Contract = getTensixBoundSFPUContract(GetParam(), arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
  const auto &Facts = Contract->getTransferFacts().front();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
  EXPECT_EQ(Facts.DestinationArgument, 0u);
  EXPECT_EQ(Facts.OldArgument, 1u);
  EXPECT_FALSE(Facts.SourceArgument);
  EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
  EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
  // Active-lane old bits are consumed numerically, not copied to the result.
  EXPECT_EQ(Facts.DemandedOldBits, 0u);
  EXPECT_TRUE(Facts.RequiresLaneEnable);
  EXPECT_TRUE(Facts.PreservesInactiveLanes);
  EXPECT_TRUE(Facts.RequiresConditionState);
  EXPECT_TRUE(Facts.RequiresRowMaskState);
  unsigned Inputs = GetParam() == Intrinsic::tt_bh_sfpu_not ? 1 : 2;
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
            GetParam() == Intrinsic::tt_bh_sfpu_not ? 2u : 3u);
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
                         testing::Values(Intrinsic::tt_bh_sfpu_and,
                                         Intrinsic::tt_bh_sfpu_or,
                                         Intrinsic::tt_bh_sfpu_xor,
                                         Intrinsic::tt_bh_sfpu_not));

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
    auto Contract = getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip,
                                               arguments(Unbound));
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    const auto &Facts = Contract->getTransferFacts().front();
    EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
    EXPECT_EQ(Facts.DestinationArgument, 0u);
    EXPECT_EQ(Facts.OldArgument, 1u);
    EXPECT_FALSE(Facts.SourceArgument);
    EXPECT_EQ(Facts.DemandedOldBits, 0u);
    EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
    EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
    EXPECT_TRUE(Facts.RequiresLaneEnable);
    EXPECT_TRUE(Facts.PreservesInactiveLanes);
    EXPECT_TRUE(Facts.RequiresConditionState);
    EXPECT_TRUE(Facts.RequiresRowMaskState);
    ASSERT_EQ(Facts.Inputs.size(), GetParam() == 1 ? 2u : 1u);
    const auto Input =
        find_if(Facts.Inputs, [](const auto &I) { return I.Argument == 2; });
    ASSERT_NE(Input, Facts.Inputs.end());
    EXPECT_EQ(Input->DemandedBits, 0xffffffffu);
    const auto Old =
        find_if(Facts.Inputs, [](const auto &I) { return I.Argument == 1; });
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
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip, arguments());
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  EXPECT_TRUE(
      std::holds_alternative<std::monostate>(Contract->getLaneControl()));
  unsigned Conditions = 0, Configurations = 0, Issues = 0;
  unsigned Registers = 0;
  for (const auto &Effect : Contract->getArchitecturalEffects()) {
    EXPECT_FALSE(
        std::holds_alternative<TensixFixedRegisterRef>(Effect.Resource));
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
  auto Contract =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip, arguments());
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
  auto Rejected =
      getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip, Args);
  ASSERT_FALSE(bool(Rejected));
  consumeError(Rejected.takeError());
}

INSTANTIATE_TEST_SUITE_P(ReceivedModes, RISCVTensixApproxRecipTransferTest,
                         testing::Values(int64_t(0), int64_t(1), int64_t(2)));

TEST(RISCVTensixTransfer, ReciprocalRejectsUnreceivedOrNonconstantModes) {
  // The formal intrinsic receives only the three authored modes. Do not
  // silently interpret another raw field as EXP or accept a dynamic mode.
  SmallVector<TensixBoundValue, 4> Args{{TensixBoundValueKind::Constant, 0},
                                        {TensixBoundValueKind::Constant, 0},
                                        {TensixBoundValueKind::Constant, 1},
                                        {TensixBoundValueKind::Constant, 0}};
  for (int64_t Mode : {-1, 3, 4, 15, 16}) {
    SCOPED_TRACE(Mode);
    Args[3] = {TensixBoundValueKind::Constant, Mode};
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip, Args);
    ASSERT_FALSE(bool(Contract));
    consumeError(Contract.takeError());
  }
  for (auto Kind : {TensixBoundValueKind::UnboundRegister,
                    TensixBoundValueKind::DynamicScalar}) {
    Args[3] = {Kind, 0};
    auto Contract =
        getTensixBoundSFPUContract(Intrinsic::tt_bh_sfpu_arecip, Args);
    ASSERT_FALSE(bool(Contract));
    consumeError(Contract.takeError());
  }
}
class RISCVTensixMadTransferTest
    : public testing::TestWithParam<Intrinsic::ID> {};

TEST_P(RISCVTensixMadTransferTest,
       ActiveLanesDemandThreeInputsAndPreserveInactiveOld) {
  // The tied old destination supplies inactive lanes; it is not a fourth
  // arithmetic input. Direct mode has three independent numerical inputs.
  auto Contract = getTensixBoundSFPUContract(
      GetParam(), {{TensixBoundValueKind::Constant, 4},
                   {TensixBoundValueKind::Constant, 4},
                   {TensixBoundValueKind::Constant, 1},
                   {TensixBoundValueKind::Constant, 2},
                   {TensixBoundValueKind::Constant, 3},
                   {TensixBoundValueKind::Constant, 0}});
  ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
  ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
  const auto &Facts = Contract->getTransferFacts().front();
  EXPECT_EQ(Facts.Kind, TensixSFPUTransferKind::LaneWise);
  EXPECT_EQ(Facts.DestinationArgument, 0u);
  EXPECT_EQ(Facts.OldArgument, 1u);
  EXPECT_FALSE(Facts.SourceArgument);
  EXPECT_EQ(Facts.DemandedOldBits, 0u);
  EXPECT_EQ(Facts.GeneratedBits, 0xffffffffu);
  EXPECT_EQ(Facts.GuaranteedWriteBits, 0xffffffffu);
  EXPECT_TRUE(Facts.RequiresLaneEnable);
  EXPECT_TRUE(Facts.PreservesInactiveLanes);
  EXPECT_TRUE(Facts.RequiresConditionState);
  EXPECT_TRUE(Facts.RequiresRowMaskState);
  EXPECT_TRUE(Facts.SourceLanes.empty());
  EXPECT_TRUE(Facts.LaneSources.empty());
  EXPECT_TRUE(
      std::holds_alternative<std::monostate>(Contract->getLaneControl()));
  ASSERT_EQ(Facts.Inputs.size(), 3u);
  for (unsigned Input = 0; Input != 3; ++Input) {
    EXPECT_EQ(Facts.Inputs[Input].Argument, 2u + Input);
    EXPECT_EQ(Facts.Inputs[Input].DemandedBits, 0xffffffffu);
  }
}

INSTANTIATE_TEST_SUITE_P(Direct0, RISCVTensixMadTransferTest,
                         testing::Values(Intrinsic::tt_wh_sfpu_mad,
                                         Intrinsic::tt_bh_sfpu_mad));

struct LoadImmediateValueCase {
  uint32_t Mode;
  uint32_t Immediate;
  uint32_t Mask;
  uint32_t Value;
};
// Exact ISA outputs. FloatA deliberately differs from IEEE half for exponent
// zero and 31; using APFloat/host half conversion would fail these cases.
constexpr LoadImmediateValueCase LoadImmediateValues[] = {
    {0, 0x0000, 0xffffffff, 0x00000000},  {0, 0x8000, 0xffffffff, 0x80000000},
    {0, 0x7f80, 0xffffffff, 0x7f800000},  {0, 0x7fc1, 0xffffffff, 0x7fc10000},
    {1, 0x0000, 0xffffffff, 0x38000000},  {1, 0x0001, 0xffffffff, 0x38002000},
    {1, 0x03ff, 0xffffffff, 0x387fe000},  {1, 0x0400, 0xffffffff, 0x38800000},
    {1, 0x3c00, 0xffffffff, 0x3f800000},  {1, 0x7bff, 0xffffffff, 0x477fe000},
    {1, 0x7c00, 0xffffffff, 0x47800000},  {1, 0x7e00, 0xffffffff, 0x47c00000},
    {1, 0x8000, 0xffffffff, 0xb8000000},  {1, 0xffff, 0xffffffff, 0xc7ffe000},
    {2, 0x0000, 0xffffffff, 0x00000000},  {2, 0x8000, 0xffffffff, 0x00008000},
    {2, 0xf000, 0xffffffff, 0x0000f000},  {2, 0xffff, 0xffffffff, 0x0000ffff},
    {4, 0x0000, 0xffffffff, 0x00000000},  {4, 0x7fff, 0xffffffff, 0x00007fff},
    {4, 0x8000, 0xffffffff, 0xffff8000},  {4, 0xffff, 0xffffffff, 0xffffffff},
    {8, 0x0000, 0xffff0000, 0x00000000},  {8, 0xffff, 0xffff0000, 0xffff0000},
    {10, 0x0000, 0x0000ffff, 0x00000000}, {10, 0xabcd, 0x0000ffff, 0x0000abcd},
};

TEST(RISCVTensixTransfer, LoadImmediatePublishesActualActiveLaneValueBits) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_loadi, Intrinsic::tt_bh_sfpu_loadi})
    for (const auto &Case : LoadImmediateValues) {
      SCOPED_TRACE(static_cast<unsigned>(ID));
      SCOPED_TRACE(Case.Mode);
      SCOPED_TRACE(Case.Immediate);
      auto Contract = getTensixBoundSFPUContract(
          ID, {{TensixBoundValueKind::Constant, 7},
               {TensixBoundValueKind::Constant, 7},
               {TensixBoundValueKind::Constant, Case.Immediate},
               {TensixBoundValueKind::Constant, Case.Mode}});
      ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
      ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
      const auto &Transfer = Contract->getTransferFacts().front();
      ASSERT_TRUE(Transfer.ImmediateValue);
      EXPECT_EQ(Transfer.ImmediateValue->KnownBitsMask, Case.Mask);
      EXPECT_EQ(Transfer.ImmediateValue->Value, Case.Value);
      EXPECT_EQ(Transfer.ImmediateValue->KnownBitsMask, Transfer.GeneratedBits);
      EXPECT_EQ(Transfer.ImmediateValue->Value & ~Case.Mask, 0u);
      // Values do not grant active participation or overwrite inactive lanes.
      EXPECT_TRUE(Transfer.RequiresLaneEnable);
      EXPECT_TRUE(Transfer.RequiresConditionState);
      EXPECT_TRUE(Transfer.RequiresRowMaskState);
      EXPECT_TRUE(Transfer.PreservesInactiveLanes);
      EXPECT_EQ(Transfer.DemandedOldBits, ~Case.Mask);
    }
}

TEST(RISCVTensixTransfer, ContentsLoadImmediateRetainsTheSameNativeValue) {
  for (auto Architecture :
       {TensixSFPUArchitecture::Wormhole, TensixSFPUArchitecture::Blackhole})
    for (const auto &Case : LoadImmediateValues)
      for (bool Full : {false, true}) {
        if (Full && Case.Mask != 0xffffffffu)
          continue; // Full native form has no partial-write modes.
        SCOPED_TRACE(static_cast<unsigned>(Architecture));
        SCOPED_TRACE(Case.Mode);
        SCOPED_TRACE(Case.Immediate);
        SCOPED_TRACE(Full);
        SmallVector<TensixContentsValue, 3> Arguments;
        if (!Full)
          Arguments.push_back(
              {TensixContentsValueKind::Contents, std::nullopt});
        Arguments.push_back({TensixContentsValueKind::Scalar, Case.Immediate});
        Arguments.push_back({TensixContentsValueKind::Scalar, Case.Mode});
        auto Info = getTensixContentsIntrinsicInfo(
            Full ? Intrinsic::riscv_tt_contents_load_immediate_full
                 : Intrinsic::riscv_tt_contents_load_immediate,
            Arguments, Architecture);
        ASSERT_TRUE(bool(Info)) << toString(Info.takeError());
        ASSERT_EQ(Info->Transfers.size(), 1u);
        const auto &Transfer = Info->Transfers.front();
        ASSERT_TRUE(Transfer.ImmediateValue);
        EXPECT_EQ(Transfer.ImmediateValue->KnownBitsMask, Case.Mask);
        EXPECT_EQ(Transfer.ImmediateValue->Value, Case.Value);
        EXPECT_EQ(Transfer.OldArgument.has_value(), !Full);
        EXPECT_TRUE(Transfer.RequiresLaneEnable);
      }
}

TEST(RISCVTensixTransfer, NonImmediateTransferDoesNotInventNumericalBits) {
  for (auto ID :
       {Intrinsic::tt_wh_sfpu_mov_all, Intrinsic::tt_bh_sfpu_mov_all}) {
    auto Contract =
        getTensixBoundSFPUContract(ID, {{TensixBoundValueKind::Constant, 0},
                                        {TensixBoundValueKind::Constant, 9}});
    ASSERT_TRUE(bool(Contract)) << toString(Contract.takeError());
    ASSERT_EQ(Contract->getTransferFacts().size(), 1u);
    EXPECT_FALSE(Contract->getTransferFacts().front().ImmediateValue);
  }
}

TEST(RISCVTensixTransfer, ImmediateValueRejectsNonContractualModesAndFields) {
  for (auto ID : {Intrinsic::tt_wh_sfpu_loadi, Intrinsic::tt_bh_sfpu_loadi}) {
    for (int64_t Mode : {3, 5, 6, 7, 9, 11, 12, 13, 14, 15}) {
      auto Contract = getTensixBoundSFPUContract(
          ID, {{TensixBoundValueKind::Constant, 0},
               {TensixBoundValueKind::Constant, 0},
               {TensixBoundValueKind::Constant, 0x1234},
               {TensixBoundValueKind::Constant, Mode}});
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
    }
    for (int64_t Immediate : {-1, 65536}) {
      auto Contract = getTensixBoundSFPUContract(
          ID, {{TensixBoundValueKind::Constant, 0},
               {TensixBoundValueKind::Constant, 0},
               {TensixBoundValueKind::Constant, Immediate},
               {TensixBoundValueKind::Constant, 2}});
      ASSERT_FALSE(bool(Contract));
      consumeError(Contract.takeError());
    }
  }
}

TEST(RISCVTensixTransfer, NeutralAddressModifierProjectsFloat32Transfers) {
  auto Literal = [](int64_t Value) {
    return TensixBoundValue{TensixBoundValueKind::Constant, Value};
  };
  for (auto [LoadID, StoreID] : {
           std::pair{Intrinsic::tt_wh_sfpu_load, Intrinsic::tt_wh_sfpu_store},
           std::pair{Intrinsic::tt_bh_sfpu_load, Intrinsic::tt_bh_sfpu_store}}) {
    SCOPED_TRACE(LoadID);
    auto Load = getTensixBoundSFPUContract(
        LoadID, {Literal(3), Literal(3), Literal(16), Literal(0), Literal(3)});
    ASSERT_TRUE(bool(Load)) << toString(Load.takeError());
    const auto &LoadFacts = Load->getDstTransferFacts();
    ASSERT_TRUE(LoadFacts);
    EXPECT_EQ(LoadFacts->OffsetArgument, 2u);
    EXPECT_EQ(LoadFacts->AddressMode, 0u);
    EXPECT_EQ(LoadFacts->Format, TensixSFPUDstFormat::Float32);
    auto *LoadPacket = std::get_if<TensixSFPUDstLoadTransfer>(&LoadFacts->Packet);
    ASSERT_TRUE(LoadPacket);
    EXPECT_EQ(LoadPacket->DestinationValue, 0u);
    EXPECT_EQ(LoadPacket->OldValue, 1u);

    auto Store = getTensixBoundSFPUContract(
        StoreID, {Literal(3), Literal(16), Literal(0), Literal(3)});
    ASSERT_TRUE(bool(Store)) << toString(Store.takeError());
    const auto &StoreFacts = Store->getDstTransferFacts();
    ASSERT_TRUE(StoreFacts);
    EXPECT_EQ(StoreFacts->OffsetArgument, 1u);
    EXPECT_EQ(StoreFacts->AddressMode, 0u);
    EXPECT_EQ(StoreFacts->Format, TensixSFPUDstFormat::Float32);
    auto *StorePacket =
        std::get_if<TensixSFPUDstStoreTransfer>(&StoreFacts->Packet);
    ASSERT_TRUE(StorePacket);
    EXPECT_EQ(StorePacket->SourceValue, 0u);

    for (auto [Modifier, Mode] : {std::pair{0u, 2u}, std::pair{1u, 3u}}) {
      auto Outside = getTensixBoundSFPUContract(
          StoreID, {Literal(3), Literal(16), Literal(Modifier), Literal(Mode)});
      ASSERT_TRUE(bool(Outside)) << toString(Outside.takeError());
      EXPECT_FALSE(Outside->getDstTransferFacts());
    }
  }
}

} // namespace
