//===- TensixMOPSequence.cpp - Value-only Blackhole MOP graph ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TensixMOPSequence.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;

char TensixMOPSequenceError::ID = 0;
void TensixMOPSequenceError::log(raw_ostream &OS) const { OS << Message; }
std::error_code TensixMOPSequenceError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

namespace {
using Failure = TensixMOPSequenceFailure;
using Control = TensixMOPControlValue;
using Declaration = TensixMOPSlotDeclaration;
using Node = TensixMOPSequenceNode;
using Count = TensixMOPRepeatCount;

Error fail(Failure Reason, const Twine &Detail) {
  return make_error<TensixMOPSequenceError>(Reason, Detail);
}

Error validateControl(const Control &Value) {
  switch (Value.Knowledge) {
  case Control::Kind::Constant:
  case Control::Kind::Dynamic:
    return Error::success();
  case Control::Kind::Missing:
    return fail(Failure::MissingControl,
                "MOP requires its actual received control-cell contents");
  }
  return fail(Failure::InvalidField, "MOP control has an invalid knowledge kind");
}

std::optional<unsigned> constant(const Control &Value, unsigned Mask) {
  if (Value.Knowledge == Control::Kind::Constant)
    return Value.Value & Mask;
  return std::nullopt;
}

class SequenceBuilder {
  const TensixMOPSlots &Slots;
  TensixMOPSequence Graph;
  std::array<SmallVector<std::optional<unsigned>, 2>, 7> Leaves;

  unsigned append(Node N) {
    unsigned Index = Graph.Nodes.size();
    Graph.Nodes.push_back(std::move(N));
    return Index;
  }

  unsigned leaf(unsigned Slot, unsigned Alternative) {
    auto &Cached = Leaves[Slot - 2][Alternative];
    if (Cached)
      return *Cached;
    Node N;
    N.Form = Node::Kind::SlotRef;
    N.Slot = Slot;
    N.Alternative = Alternative;
    N.Declaration = Slots[Slot - 2][Alternative];
    N.MaySelect = N.MustSelect = 1u << Slot;
    Cached = append(std::move(N));
    return *Cached;
  }

  unsigned sequence(ArrayRef<unsigned> Children) {
    Node N;
    N.Form = Node::Kind::Sequence;
    for (unsigned Child : Children) {
      if (!Child)
        continue;
      N.Children.push_back(Child);
      N.MaySelect |= Graph.Nodes[Child].MaySelect;
      N.MustSelect |= Graph.Nodes[Child].MustSelect;
    }
    if (N.Children.empty())
      return 0;
    if (N.Children.size() == 1)
      return N.Children.front();
    return append(std::move(N));
  }

  unsigned choice(ArrayRef<unsigned> Children) {
    assert(!Children.empty());
    Node N;
    N.Form = Node::Kind::Choice;
    N.MustSelect = Graph.Nodes[Children.front()].MustSelect;
    for (unsigned Child : Children) {
      if (is_contained(N.Children, Child))
        continue;
      N.Children.push_back(Child);
      N.MaySelect |= Graph.Nodes[Child].MaySelect;
      N.MustSelect &= Graph.Nodes[Child].MustSelect;
    }
    if (N.Children.size() == 1)
      return N.Children.front();
    return append(std::move(N));
  }

  unsigned repeat(unsigned Child, Count Repetitions) {
    if (!Child || !Repetitions.Maximum)
      return 0;
    if (Repetitions.Minimum == 1 && Repetitions.Maximum == 1)
      return Child;
    Node N;
    N.Form = Node::Kind::CountedRepeat;
    N.Children.push_back(Child);
    N.Count = Repetitions;
    N.MaySelect = Graph.Nodes[Child].MaySelect;
    N.MustSelect = Repetitions.Minimum ? Graph.Nodes[Child].MustSelect : 0;
    return append(std::move(N));
  }

  unsigned repeat(unsigned Child, unsigned Times) {
    return repeat(Child, {Count::Source::Constant, Times, Times, 0});
  }

  unsigned prefix(unsigned Child, std::optional<unsigned> Times,
                  Count::Source Source) {
    assert(!Times || *Times > 0);
    if (Times)
      return repeat(Child, *Times - 1);
    // This is the positive-count branch; a separate Exit choice represents
    // zero. The ten-bit Blackhole control count is not enumerated.
    return repeat(Child, {Source, 0, 1022, -1});
  }

  bool hasNOP(unsigned Slot) const {
    return any_of(Slots[Slot - 2], [](const Declaration &D) {
      return D.Content == Declaration::Kind::Nop;
    });
  }

  bool hasNonNOP(unsigned Slot) const {
    return any_of(Slots[Slot - 2], [](const Declaration &D) {
      return D.Content != Declaration::Kind::Nop;
    });
  }

  Expected<unsigned> slot(unsigned Slot, bool SkipNOP = false) {
    if (Slots[Slot - 2].empty())
      return fail(Failure::MissingSlot,
                  "MOP selected slot has no reaching declaration");
    SmallVector<unsigned> Alternatives;
    for (auto [I, D] : enumerate(Slots[Slot - 2]))
      Alternatives.push_back(SkipNOP && D.Content == Declaration::Kind::Nop
                                 ? 0
                                 : leaf(Slot, I));
    return choice(Alternatives);
  }

  // Traverse only the finite ISA mask field, combining equal-bit runs. Count
  // outside that field is another counted edge, not copied loop iterations.
  unsigned maskRun(uint32_t Bits, unsigned Width, unsigned A, unsigned Skip) {
    SmallVector<unsigned> Runs;
    while (Width) {
      bool Set = Bits & 1;
      unsigned Run = 1;
      while (Run < Width && bool((Bits >> Run) & 1) == Set)
        ++Run;
      Runs.push_back(repeat(Set ? Skip : A, Run));
      Bits >>= Run;
      Width -= Run;
    }
    return sequence(Runs);
  }

public:
  explicit SequenceBuilder(const TensixMOPSlots &Slots) : Slots(Slots) {
    Graph.Nodes.emplace_back(); // Unique Exit node.
    for (unsigned I = 0; I != Slots.size(); ++I)
      Leaves[I].resize(Slots[I].size());
  }

  Error validateSlots() {
    for (const auto &Alternatives : Slots)
      for (const auto &D : Alternatives)
        switch (D.Content) {
        case Declaration::Kind::Nop:
        case Declaration::Kind::Opaque:
        case Declaration::Kind::Template:
          break;
        default:
          return fail(Failure::InvalidDeclaration,
                      "MOP slot has an invalid declaration kind");
        }
    return Error::success();
  }

  TensixMOPSequence finish(unsigned Root) {
    Graph.Root = Root;
    return std::move(Graph);
  }

  Expected<unsigned> template0(const TensixMOPTemplate0 &Input) {
    if (Input.MaskLow > 65535 || Input.CountMinusOne > 127)
      return fail(Failure::InvalidField,
                  "MOP template0 needs a 16-bit mask and a 7-bit count field");
    if (Error E = validateControl(Input.Flags))
      return std::move(E);
    unsigned Count = Input.CountMinusOne + 1;
    if (Count > 16)
      if (Error E = validateControl(Input.MaskHigh))
        return std::move(E);

    auto Flags = constant(Input.Flags, 3);
    unsigned LowWidth = std::min(Count, 16u);
    unsigned LowMask = (1u << LowWidth) - 1;
    unsigned LowBits = Input.MaskLow & LowMask;
    bool NeedsA = LowBits != LowMask || Count > 32;
    bool NeedsSkip = LowBits != 0;
    unsigned HighWidth = Count > 16 ? std::min(Count - 16, 16u) : 0;
    auto High = constant(Input.MaskHigh, 65535);
    if (HighWidth) {
      if (High) {
        unsigned HighMask = (1u << HighWidth) - 1;
        NeedsA |= (*High & HighMask) != HighMask;
        NeedsSkip |= (*High & HighMask) != 0;
      } else {
        NeedsA = NeedsSkip = true;
      }
    }
    SmallVector<unsigned, 4> Paths;
    for (unsigned F = 0; F != 4; ++F) {
      if (Flags && F != *Flags)
        continue;
      auto A = NeedsA ? slot(3) : Expected<unsigned>(0);
      if (!A)
        return A.takeError();
      auto Skip = NeedsSkip ? slot(7) : Expected<unsigned>(0);
      if (!Skip)
        return Skip.takeError();
      if ((F & 2) && NeedsA)
        for (unsigned I : {4u, 5u, 6u}) {
          auto Extra = slot(I);
          if (!Extra)
            return Extra.takeError();
          *A = sequence({*A, *Extra});
        }
      if (F & 1) {
        auto B = NeedsA ? slot(2) : Expected<unsigned>(0);
        if (!B)
          return B.takeError();
        auto SkipB = NeedsSkip ? slot(8) : Expected<unsigned>(0);
        if (!SkipB)
          return SkipB.takeError();
        *A = sequence({*A, *B});
        *Skip = sequence({*Skip, *SkipB});
      }
      unsigned LowPath = maskRun(Input.MaskLow, LowWidth, *A, *Skip);
      unsigned HighPath = 0;
      if (HighWidth)
        HighPath = High ? maskRun(*High, HighWidth, *A, *Skip)
                        : repeat(choice({*A, *Skip}), HighWidth);
      unsigned Tail = Count > 32 ? repeat(*A, Count - 32) : 0;
      Paths.push_back(sequence({LowPath, HighPath, Tail}));
    }
    return choice(Paths);
  }

  Expected<unsigned> template1(const TensixMOPTemplate1 &Input) {
    if (Input.OuterOverride > 1023 || Input.InnerOverride > 1023)
      return fail(Failure::InvalidField,
                  "MOP template1 override fields must fit ten bits");
    if (Input.OuterOverride || Input.InnerOverride)
      return fail(Failure::UnsupportedOverride,
                  "MOP template1 count overrides are unverified on Blackhole");
    if (Error E = validateControl(Input.Outer))
      return std::move(E);
    if (Error E = validateControl(Input.Inner))
      return std::move(E);
    auto Outer = constant(Input.Outer, 1023);
    auto Inner = constant(Input.Inner, 1023);
    if ((!Inner || !*Inner) && hasNOP(2) && hasNonNOP(3))
      return fail(Failure::UnverifiedZeroInner,
                  "MOP zero-inner NOP-start with non-NOP end is unverified");
    if (Outer && !*Outer)
      return 0;
    auto Start = slot(2, true);
    if (!Start)
      return Start.takeError();
    // The end opcode is itself needed to determine whether the end path is
    // suppressed. Absence is not proof that it is NOP.
    if (Slots[3 - 2].empty())
      return fail(Failure::MissingSlot,
                  "MOP end selection has no reaching declaration");
    unsigned End = 0;
    if (hasNonNOP(3)) {
      SmallVector<unsigned> Ends;
      for (auto [I, D] : enumerate(Slots[3 - 2])) {
        if (D.Content == Declaration::Kind::Nop) {
          Ends.push_back(0);
          continue;
        }
        auto End1 = slot(4, true);
        if (!End1)
          return End1.takeError();
        Ends.push_back(sequence({leaf(3, I), *End1}));
      }
      End = choice(Ends);
    }
    auto InnerPath = [&](unsigned Last) -> Expected<unsigned> {
      if (Inner && !*Inner)
        return 0;
      if (Slots[6 - 2].empty())
        return fail(Failure::MissingSlot,
                    "MOP Loop1 mode has no reaching declaration");
      auto L0 = !Inner || *Inner > 1 || hasNonNOP(6)
                    ? slot(5) : Expected<unsigned>(0);
      if (!L0)
        return L0.takeError();
      auto Final = slot(Last);
      if (!Final)
        return Final.takeError();
      SmallVector<unsigned> Paths;
      if (hasNOP(6))
        Paths.push_back(sequence(
            {prefix(*L0, Inner, Count::Source::Inner), *Final}));
      for (auto [I, D] : enumerate(Slots[6 - 2])) {
        if (D.Content == Declaration::Kind::Nop)
          continue;
        unsigned Prefix = 0;
        if (!Inner || *Inner > 1)
          Prefix = prefix(sequence({*L0, leaf(6, I)}), Inner,
                          Count::Source::Inner);
        // Inner=1 consumes slot6 only to select alternating mode. The last
        // override suppresses its instruction, including any replay range.
        Paths.push_back(sequence({Prefix, *L0, *Final}));
      }
      if (!Inner)
        Paths.push_back(0);
      return choice(Paths);
    };
    auto FinalInner = InnerPath(7);
    if (!FinalInner)
      return FinalInner.takeError();
    unsigned LastOuter = sequence({*Start, *FinalInner, End});
    if (Outer && *Outer == 1)
      return LastOuter;
    auto OtherInner = InnerPath(8);
    if (!OtherInner)
      return OtherInner.takeError();
    unsigned OtherOuter = sequence({*Start, *OtherInner, End});
    unsigned Result = sequence(
        {prefix(OtherOuter, Outer, Count::Source::Outer), LastOuter});
    if (!Outer)
      Result = choice({0, Result});
    return Result;
  }
};
} // namespace

Expected<TensixMOPSequence>
llvm::buildTensixMOPSequence(const TensixMOPTemplate0 &Input,
                           const TensixMOPSlots &Slots) {
  SequenceBuilder Builder(Slots);
  if (Error E = Builder.validateSlots())
    return std::move(E);
  auto Root = Builder.template0(Input);
  if (!Root)
    return Root.takeError();
  return Builder.finish(*Root);
}
Expected<TensixMOPSequence>
llvm::buildTensixMOPSequence(const TensixMOPTemplate1 &Input,
                           const TensixMOPSlots &Slots) {
  SequenceBuilder Builder(Slots);
  if (Error E = Builder.validateSlots())
    return std::move(E);
  auto Root = Builder.template1(Input);
  if (!Root)
    return Root.takeError();
  return Builder.finish(*Root);
}
