//===-- RISCVTensixMOP.h ----------------------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LLVM_TARGET_RISCV_RISCVTENSIXMOP_H
#define LLVM_TARGET_RISCV_RISCVTENSIXMOP_H

#include "llvm/Support/TensixMOPSequence.h"
#include "llvm/Target/RISCV/RISCVTensix.h"

namespace llvm {

inline Expected<TensixMOPSequence>
decodeTensixMOPTemplate1(TensixMOPTemplate1 Input, const TensixMOPSlots &Slots,
                        RISCV::TensixSFPUArchitecture Arch) {
  if (Arch == RISCV::TensixSFPUArchitecture::Blackhole)
    return buildTensixMOPSequence(Input, Slots);

  constexpr unsigned CountMask = (1u << 7) - 1;
  if (Input.OuterOverride || Input.InnerOverride)
    return make_error<TensixMOPSequenceError>(
        TensixMOPSequenceFailure::UnsupportedOverride,
        "Wormhole MOP template1 requires zero overrides");
  Input.Outer.Value &= CountMask;
  Input.Inner.Value &= CountMask;
  auto Graph = buildTensixMOPSequence(Input, Slots);
  if (!Graph)
    return Graph.takeError();

  for (auto &Node : Graph->Nodes)
    if (Node.Form == TensixMOPSequenceNode::Kind::CountedRepeat &&
        Node.Count.Control != TensixMOPRepeatCount::Source::Constant) {
      assert(Node.Count.Offset == -1 && "expected a positive-count prefix");
      Node.Count.Maximum = CountMask - 1;
    }
  return Graph;
}

} // namespace llvm

#endif
