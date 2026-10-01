//===- TensixMOPSequence.h - Value-only Blackhole MOP graph ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TENSIXMOPSEQUENCE_H
#define LLVM_SUPPORT_TENSIXMOPSEQUENCE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <array>
#include <cstdint>

namespace llvm {

/// Reached control-cell contents. Constant values are raw 32-bit writes: the
/// sequencer selects the ISA-defined low bits. Dynamic is a received value;
/// Missing cannot stand in for an unknown but actually configured cell.
struct TensixMOPControlValue {
  enum class Kind : uint8_t { Missing, Constant, Dynamic };
  Kind Knowledge = Kind::Missing;
  uint32_t Value = 0;

  static TensixMOPControlValue constant(uint32_t Value) {
    return {Kind::Constant, Value};
  }
  static TensixMOPControlValue dynamic() { return {Kind::Dynamic, 0}; }
};

/// A client-owned declaration, with no instruction encoding or resource facts.
/// Nop means the architectural plain NOP, never DMANOP or SFPNOP. Each array
/// entry describes alternatives reaching one static slot, indexed by slot - 2.
struct TensixMOPSlotDeclaration {
  enum class Kind : uint8_t { Nop, Opaque, Template };
  Kind Content = Kind::Nop;
  uint32_t Identity = 0;
};
using TensixMOPSlots =
    std::array<SmallVector<TensixMOPSlotDeclaration, 2>, 7>;

struct TensixMOPTemplate0 {
  uint32_t MaskLow = 0;
  uint32_t CountMinusOne = 0;
  TensixMOPControlValue Flags;
  TensixMOPControlValue MaskHigh;
};

struct TensixMOPTemplate1 {
  TensixMOPControlValue Outer;
  TensixMOPControlValue Inner;
  uint32_t OuterOverride = 0;
  uint32_t InnerOverride = 0;
};

/// Counts are not expanded into source iterations. A dynamic count retains its
/// control-cell source and offset, with the bounds applicable on this path.
struct TensixMOPRepeatCount {
  enum class Source : uint8_t { Constant, Outer, Inner };
  Source Control = Source::Constant;
  uint32_t Minimum = 0;
  uint32_t Maximum = 0;
  int32_t Offset = 0;
};

struct TensixMOPSequenceNode {
  enum class Kind : uint8_t { Exit, SlotRef, Sequence, Choice, CountedRepeat };
  Kind Form = Kind::Exit;
  SmallVector<unsigned, 4> Children;
  uint32_t Slot = 0;
  uint32_t Alternative = 0;
  TensixMOPSlotDeclaration Declaration;
  TensixMOPRepeatCount Count;
  /// Bits are indexed by the physical slot ID, not slot - 2. Mode-selection
  /// metadata is excluded when that slot's instruction is not executed.
  uint16_t MaySelect = 0;
  uint16_t MustSelect = 0;
};

/// An acyclic, owned graph; children precede parents in Nodes. Unknown inputs
/// yield conservative Choice/CountedRepeat paths, suitable for effect analysis,
/// not a replacement executable schedule. Consumers retain declaration identity
/// and validate recorded words, ranges, physical effects and hazards themselves.
struct TensixMOPSequence {
  SmallVector<TensixMOPSequenceNode, 32> Nodes;
  unsigned Root = 0;

  uint16_t maySelect() const { return Nodes[Root].MaySelect; }
  uint16_t mustSelect() const { return Nodes[Root].MustSelect; }
};

enum class TensixMOPSequenceFailure : uint8_t {
  InvalidField,
  MissingControl,
  MissingSlot,
  UnsupportedOverride,
  UnverifiedZeroInner,
  InvalidDeclaration
};

class LLVM_ABI TensixMOPSequenceError
    : public ErrorInfo<TensixMOPSequenceError> {
public:
  static char ID;
  TensixMOPSequenceError(TensixMOPSequenceFailure Reason, const Twine &Message)
      : Reason(Reason), Message(Message.str()) {}
  TensixMOPSequenceFailure getReason() const { return Reason; }
  void log(raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

private:
  TensixMOPSequenceFailure Reason;
  std::string Message;
};

LLVM_ABI Expected<TensixMOPSequence>
buildTensixMOPSequence(const TensixMOPTemplate0 &Input,
                      const TensixMOPSlots &Slots);
LLVM_ABI Expected<TensixMOPSequence>
buildTensixMOPSequence(const TensixMOPTemplate1 &Input,
                      const TensixMOPSlots &Slots);

} // namespace llvm

#endif // LLVM_SUPPORT_TENSIXMOPSEQUENCE_H
