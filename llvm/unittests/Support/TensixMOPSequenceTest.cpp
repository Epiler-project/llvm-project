//===- TensixMOPSequenceTest.cpp - Blackhole MOP sequencer tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TensixMOPSequence.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
using Control = TensixMOPControlValue;
using Declaration = TensixMOPSlotDeclaration;
using Node = TensixMOPSequenceNode;
using Failure = TensixMOPSequenceFailure;

TensixMOPSlots nopSlots() {
  TensixMOPSlots Slots;
  for (unsigned I = 0; I != Slots.size(); ++I)
    Slots[I].push_back({Declaration::Kind::Nop, I + 2});
  return Slots;
}

void set(TensixMOPSlots &Slots, unsigned Slot, Declaration::Kind Kind,
         unsigned Identity = 17) {
  Slots[Slot - 2].assign(1, {Kind, Identity});
}

TensixMOPTemplate1 counts(unsigned Outer, unsigned Inner) {
  return {Control::constant(Outer), Control::constant(Inner)};
}

uint16_t bits(std::initializer_list<unsigned> Slots) {
  uint16_t Result = 0;
  for (unsigned Slot : Slots)
    Result |= 1u << Slot;
  return Result;
}

// A bounded test-only interpreter. Production keeps CountedRepeat edges and
// never constructs this per-iteration trace.
void trace(const TensixMOPSequence &Graph, unsigned Index,
           SmallVectorImpl<unsigned> &Result) {
  const auto &N = Graph.Nodes[Index];
  switch (N.Form) {
  case Node::Kind::Exit:
    return;
  case Node::Kind::SlotRef:
    Result.push_back(N.Slot);
    return;
  case Node::Kind::Sequence:
    for (unsigned Child : N.Children)
      trace(Graph, Child, Result);
    return;
  case Node::Kind::Choice:
    ADD_FAILURE() << "a concrete trace has an unresolved choice";
    return;
  case Node::Kind::CountedRepeat:
    ASSERT_EQ(N.Count.Minimum, N.Count.Maximum);
    ASSERT_LE(N.Count.Maximum, 128u);
    ASSERT_EQ(N.Children.size(), 1u);
    for (unsigned I = 0; I != N.Count.Maximum; ++I)
      trace(Graph, N.Children.front(), Result);
    return;
  }
}

SmallVector<unsigned> trace(const TensixMOPSequence &Graph) {
  SmallVector<unsigned> Result;
  trace(Graph, Graph.Root, Result);
  return Result;
}

void expectFailure(Expected<TensixMOPSequence> Result, Failure Reason) {
  ASSERT_FALSE(bool(Result));
  handleAllErrors(Result.takeError(), [&](const TensixMOPSequenceError &Error) {
    EXPECT_EQ(Error.getReason(), Reason);
  });
}

TEST(TensixMOPSequenceTest, Template1LastReplacesInnerIteration) {
  auto Slots = nopSlots();
  for (unsigned Slot : {5u, 7u, 8u})
    set(Slots, Slot, Declaration::Kind::Template);
  auto Two = buildTensixMOPSequence(counts(2, 2), Slots);
  ASSERT_TRUE(bool(Two)) << toString(Two.takeError());
  EXPECT_EQ(trace(*Two), (SmallVector<unsigned>{5, 8, 5, 7}));
  EXPECT_EQ(Two->maySelect(), bits({5, 7, 8}));
  EXPECT_EQ(Two->mustSelect(), bits({5, 7, 8}));
  auto Three = buildTensixMOPSequence(counts(2, 3), Slots);
  ASSERT_TRUE(bool(Three)) << toString(Three.takeError());
  EXPECT_EQ(trace(*Three), (SmallVector<unsigned>{5, 5, 8, 5, 5, 7}));
}

TEST(TensixMOPSequenceTest, InnerOneUsesLoop1OnlyForModeSelection) {
  auto Slots = nopSlots();
  for (unsigned Slot : {5u, 6u, 7u, 8u})
    set(Slots, Slot, Declaration::Kind::Template);
  auto Graph = buildTensixMOPSequence(counts(2, 1), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(trace(*Graph), (SmallVector<unsigned>{5, 8, 5, 7}));
  EXPECT_EQ(Graph->maySelect(), bits({5, 7, 8}));
  auto Two = buildTensixMOPSequence(counts(1, 2), Slots);
  ASSERT_TRUE(bool(Two)) << toString(Two.takeError());
  EXPECT_EQ(trace(*Two), (SmallVector<unsigned>{5, 6, 5, 7}));
}

TEST(TensixMOPSequenceTest, SlotAlternativesRetainActualIdentity) {
  auto Slots = nopSlots();
  Slots[7 - 2] = {{Declaration::Kind::Template, 91},
                  {Declaration::Kind::Template, 92}};
  auto Graph = buildTensixMOPSequence(counts(1, 1), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(Graph->maySelect(), bits({7}));
  EXPECT_EQ(Graph->mustSelect(), bits({7}));
  SmallVector<unsigned> Identities;
  for (const auto &N : Graph->Nodes)
    if (N.Form == Node::Kind::SlotRef && N.Slot == 7)
      Identities.push_back(N.Declaration.Identity);
  EXPECT_EQ(Identities, (SmallVector<unsigned>{91, 92}));
}

TEST(TensixMOPSequenceTest, EqualTemplatesKeepDistinctStaticSlots) {
  auto Slots = nopSlots();
  for (unsigned Slot : {5u, 7u, 8u})
    set(Slots, Slot, Declaration::Kind::Template, 91);
  auto Graph = buildTensixMOPSequence(counts(2, 3), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  unsigned Leaves = 0;
  for (const auto &N : Graph->Nodes)
    if (N.Form == Node::Kind::SlotRef && N.Declaration.Identity == 91)
      ++Leaves;
  EXPECT_EQ(Leaves, 3u);
}

TEST(TensixMOPSequenceTest, ZeroOuterHasNoSelectedSlots) {
  auto Slots = nopSlots();
  auto Graph = buildTensixMOPSequence(counts(0, 3), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_TRUE(trace(*Graph).empty());
  EXPECT_EQ(Graph->maySelect(), 0);
  EXPECT_EQ(Graph->mustSelect(), 0);
}

TEST(TensixMOPSequenceTest, SuppressedEndDoesNotRequireEnd1) {
  auto Slots = nopSlots();
  Slots[4 - 2].clear();
  auto Graph = buildTensixMOPSequence(counts(1, 1), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(trace(*Graph), (SmallVector<unsigned>{7}));
  set(Slots, 3, Declaration::Kind::Opaque);
  expectFailure(buildTensixMOPSequence(counts(1, 1), Slots),
                Failure::MissingSlot);
}

TEST(TensixMOPSequenceTest, UnverifiedZeroInnerCornerIsRejected) {
  auto Slots = nopSlots();
  set(Slots, 3, Declaration::Kind::Opaque);
  expectFailure(buildTensixMOPSequence(counts(1, 0), Slots),
                Failure::UnverifiedZeroInner);
  auto Dynamic = counts(1, 0);
  Dynamic.Inner = Control::dynamic();
  expectFailure(buildTensixMOPSequence(Dynamic, Slots),
                Failure::UnverifiedZeroInner);
}

TEST(TensixMOPSequenceTest, NonNopStartAllowsZeroInner) {
  auto Slots = nopSlots();
  for (unsigned Slot : {2u, 3u, 4u})
    set(Slots, Slot, Declaration::Kind::Opaque);
  auto Graph = buildTensixMOPSequence(counts(2, 0), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(trace(*Graph), (SmallVector<unsigned>{2, 3, 4, 2, 3, 4}));
}

TEST(TensixMOPSequenceTest, MaximumCountsRetainBoundedDAG) {
  auto Slots = nopSlots();
  for (unsigned Slot : {5u, 6u, 7u, 8u})
    set(Slots, Slot, Declaration::Kind::Template);
  auto Graph = buildTensixMOPSequence(counts(1023, 1023), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_LT(Graph->Nodes.size(), 24u);
  unsigned Repeats = 0;
  for (unsigned I = 0; I != Graph->Nodes.size(); ++I) {
    const auto &N = Graph->Nodes[I];
    for (unsigned Child : N.Children)
      EXPECT_LT(Child, I);
    if (N.Form == Node::Kind::CountedRepeat) {
      EXPECT_EQ(N.Count.Minimum, 1022u);
      EXPECT_EQ(N.Count.Maximum, 1022u);
      ++Repeats;
    }
  }
  EXPECT_GE(Repeats, 2u);
}

TEST(TensixMOPSequenceTest, DynamicCountRetainsControlSourceAndZeroPath) {
  auto Slots = nopSlots();
  auto Input = counts(1, 2);
  Input.Outer = Control::dynamic();
  auto Graph = buildTensixMOPSequence(Input, Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(Graph->maySelect(), bits({5, 7, 8}));
  EXPECT_EQ(Graph->mustSelect(), 0);
  bool Found = false;
  for (const auto &N : Graph->Nodes)
    if (N.Form == Node::Kind::CountedRepeat &&
        N.Count.Control == TensixMOPRepeatCount::Source::Outer) {
      EXPECT_EQ(N.Count.Offset, -1);
      EXPECT_EQ(N.Count.Minimum, 0u);
      EXPECT_EQ(N.Count.Maximum, 1022u);
      Found = true;
    }
  EXPECT_TRUE(Found);
}

TEST(TensixMOPSequenceTest, Template0MaskAndFlagOrder) {
  auto Slots = nopSlots();
  TensixMOPTemplate0 Input{0b10, 2, Control::constant(3), {}};
  auto Graph = buildTensixMOPSequence(Input, Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(trace(*Graph),
            (SmallVector<unsigned>{3, 4, 5, 6, 2, 7, 8, 3, 4, 5, 6, 2}));
  EXPECT_EQ(Graph->mustSelect(), bits({2, 3, 4, 5, 6, 7, 8}));
}

TEST(TensixMOPSequenceTest, Template0FlagsSelectOnlyRequiredSlots) {
  TensixMOPSlots Slots;
  set(Slots, 3, Declaration::Kind::Template);
  auto Graph = buildTensixMOPSequence(
      TensixMOPTemplate0{0, 0, Control::constant(0), {}}, Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(trace(*Graph), (SmallVector<unsigned>{3}));
  EXPECT_EQ(Graph->maySelect(), bits({3}));
}

TEST(TensixMOPSequenceTest, Template0HighMaskAndBeyondMaskWidth) {
  auto Slots = nopSlots();
  TensixMOPTemplate0 Input{65535, 32, Control::constant(0),
                          Control::constant(65535)};
  auto Graph = buildTensixMOPSequence(Input, Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  SmallVector<unsigned> Expected(32, 7);
  Expected.push_back(3);
  EXPECT_EQ(trace(*Graph), Expected);
  EXPECT_EQ(Graph->mustSelect(), bits({3, 7}));
}

TEST(TensixMOPSequenceTest, DynamicFlagsAndHighMaskKeepMayMustSelection) {
  auto Slots = nopSlots();
  auto Graph = buildTensixMOPSequence(
      TensixMOPTemplate0{0, 16, Control::dynamic(), Control::dynamic()}, Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(Graph->maySelect(), bits({2, 3, 4, 5, 6, 7, 8}));
  EXPECT_EQ(Graph->mustSelect(), bits({3}));
}

TEST(TensixMOPSequenceTest, ControlWritesUseOnlyISASelectedLowBits) {
  auto Slots = nopSlots();
  auto Graph = buildTensixMOPSequence(counts(1025, 1025), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(trace(*Graph), (SmallVector<unsigned>{7}));
  auto Flags = buildTensixMOPSequence(
      TensixMOPTemplate0{0, 0, Control::constant(4), {}}, Slots);
  ASSERT_TRUE(bool(Flags)) << toString(Flags.takeError());
  EXPECT_EQ(trace(*Flags), (SmallVector<unsigned>{3}));
}

TEST(TensixMOPSequenceTest, MissingControlAndOutOfRangeFieldsReject) {
  auto Slots = nopSlots();
  expectFailure(buildTensixMOPSequence(TensixMOPTemplate1{}, Slots),
                Failure::MissingControl);
  expectFailure(buildTensixMOPSequence(TensixMOPTemplate0{}, Slots),
                Failure::MissingControl);
  expectFailure(buildTensixMOPSequence(
                    TensixMOPTemplate0{0, 16, Control::constant(0), {}}, Slots),
                Failure::MissingControl);
  expectFailure(buildTensixMOPSequence(
                    TensixMOPTemplate0{65536, 0, Control::constant(0), {}}, Slots),
                Failure::InvalidField);
  expectFailure(buildTensixMOPSequence(
                    TensixMOPTemplate0{0, 128, Control::constant(0), {}}, Slots),
                Failure::InvalidField);
  auto Override = counts(1, 1);
  Override.OuterOverride = 1;
  expectFailure(buildTensixMOPSequence(Override, Slots),
                Failure::UnsupportedOverride);
}

// Independent ISA pseudocode oracles. Only tests enumerate small concrete
// hardware traces; the production graph must remain counted and shared.
SmallVector<unsigned> template0Oracle(unsigned Flags, uint32_t Mask,
                                     unsigned Count) {
  SmallVector<unsigned> Result;
  for (unsigned I = 0; I != Count; ++I) {
    if (!(Mask & 1)) {
      Result.push_back(3);
      if (Flags & 2)
        Result.append({4, 5, 6});
      if (Flags & 1)
        Result.push_back(2);
    } else {
      Result.push_back(7);
      if (Flags & 1)
        Result.push_back(8);
    }
    Mask >>= 1;
  }
  return Result;
}

SmallVector<unsigned> template1Oracle(unsigned Outer, unsigned Inner,
                                     bool StartNOP, bool End0NOP, bool End1NOP,
                                     bool Loop1NOP) {
  SmallVector<unsigned> Result;
  unsigned Iterations = Inner * (Loop1NOP ? 1 : 2);
  for (unsigned J = 0; J != Outer; ++J) {
    if (!StartNOP)
      Result.push_back(2);
    for (unsigned I = 0; I != Iterations; ++I)
      Result.push_back(I + 1 == Iterations ? (J + 1 == Outer ? 7 : 8)
                       : Loop1NOP || !(I & 1) ? 5 : 6);
    if (!End0NOP) {
      Result.push_back(3);
      if (!End1NOP)
        Result.push_back(4);
    }
  }
  return Result;
}

void expectConcreteSelection(const TensixMOPSequence &Graph,
                             ArrayRef<unsigned> Expected) {
  uint16_t Selected = 0;
  for (unsigned Slot : Expected)
    Selected |= 1u << Slot;
  EXPECT_EQ(trace(Graph), Expected);
  EXPECT_EQ(Graph.maySelect(), Selected);
  EXPECT_EQ(Graph.mustSelect(), Selected);
}

TEST(TensixMOPSequenceTest, Template0MatchesIndependentMaskFlagOracle) {
  auto Slots = nopSlots();
  for (unsigned Flags = 0; Flags != 4; ++Flags)
    for (unsigned Mask = 0; Mask != 16; ++Mask)
      for (unsigned Count : {1u, 2u, 4u, 6u, 17u, 33u, 128u}) {
        SCOPED_TRACE(testing::Message()
                     << "flags=" << Flags << " mask=" << Mask
                     << " count=" << Count);
        TensixMOPTemplate0 Input{Mask, Count - 1, Control::constant(Flags),
                                Control::constant(5)};
        auto Graph = buildTensixMOPSequence(Input, Slots);
        ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
        auto Expected = template0Oracle(Flags, Mask | (5u << 16), Count);
        expectConcreteSelection(*Graph, Expected);
      }
}

TEST(TensixMOPSequenceTest, Template1MatchesIndependentNestedLoopOracle) {
  for (unsigned Outer = 0; Outer != 4; ++Outer)
    for (unsigned Inner = 0; Inner != 4; ++Inner)
      for (unsigned Nops = 0; Nops != 16; ++Nops) {
        bool StartNOP = Nops & 1, End0NOP = Nops & 2;
        bool End1NOP = Nops & 4, Loop1NOP = Nops & 8;
        SCOPED_TRACE(testing::Message()
                     << "outer=" << Outer << " inner=" << Inner
                     << " nops=" << Nops);
        auto Slots = nopSlots();
        for (unsigned Slot : {5u, 7u, 8u})
          set(Slots, Slot, Declaration::Kind::Template);
        for (auto [Slot, Nop] :
             {std::pair{2u, StartNOP}, {3u, End0NOP}, {4u, End1NOP},
              {6u, Loop1NOP}})
          if (!Nop)
            set(Slots, Slot, Declaration::Kind::Opaque);
        auto Graph = buildTensixMOPSequence(counts(Outer, Inner), Slots);
        if (!Inner && StartNOP && !End0NOP) {
          expectFailure(std::move(Graph), Failure::UnverifiedZeroInner);
          continue;
        }
        ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
        auto Expected = template1Oracle(Outer, Inner, StartNOP, End0NOP,
                                        End1NOP, Loop1NOP);
        expectConcreteSelection(*Graph, Expected);
      }
}

TEST(TensixMOPSequenceTest, ModeAndEndAlternativesDoNotInventMustEffects) {
  auto Slots = nopSlots();
  Slots[6 - 2].push_back({Declaration::Kind::Template, 99});
  Slots[3 - 2].push_back({Declaration::Kind::Opaque, 100});
  set(Slots, 4, Declaration::Kind::Opaque, 101);
  auto Graph = buildTensixMOPSequence(counts(1, 1), Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(Graph->maySelect(), bits({3, 4, 5, 7}));
  EXPECT_EQ(Graph->mustSelect(), bits({7}));
}

TEST(TensixMOPSequenceTest, DynamicInnerUsesBoundedCountWithoutSlotExpansion) {
  auto Slots = nopSlots();
  TensixMOPTemplate1 Input{Control::constant(1), Control::dynamic()};
  auto Graph = buildTensixMOPSequence(Input, Slots);
  ASSERT_TRUE(bool(Graph)) << toString(Graph.takeError());
  EXPECT_EQ(Graph->maySelect(), bits({5, 7}));
  EXPECT_EQ(Graph->mustSelect(), 0);
  bool Found = false;
  for (const auto &N : Graph->Nodes)
    if (N.Form == Node::Kind::CountedRepeat &&
        N.Count.Control == TensixMOPRepeatCount::Source::Inner) {
      EXPECT_EQ(N.Count.Minimum, 0u);
      EXPECT_EQ(N.Count.Maximum, 1022u);
      EXPECT_EQ(N.Count.Offset, -1);
      Found = true;
    }
  EXPECT_TRUE(Found);
  EXPECT_LT(Graph->Nodes.size(), 16u);
}

TEST(TensixMOPSequenceTest, MissingSelectionMetadataCannotMeanNop) {
  auto Slots = nopSlots();
  Slots[3 - 2].clear();
  expectFailure(buildTensixMOPSequence(counts(1, 1), Slots),
                Failure::MissingSlot);
  Slots = nopSlots();
  Slots[6 - 2].clear();
  expectFailure(buildTensixMOPSequence(counts(1, 1), Slots),
                Failure::MissingSlot);
  Slots = nopSlots();
  Slots[7 - 2].front().Content = static_cast<Declaration::Kind>(255);
  expectFailure(buildTensixMOPSequence(counts(1, 1), Slots),
                Failure::InvalidDeclaration);
}
} // namespace
