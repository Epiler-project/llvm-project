//===-- IntrinsicsRISCVTensix.td --------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The pre-bound implicit-register ABI remains retired. Explicit physical
// bindings and numerical contents SSA have separate receivers and cannot mix
// in a function. All Tensix intrinsics share this convergent, non-duplicable
// side-effect contract.
let TargetPrefix = "riscv" in {
  class TensixIntrinsic<list<LLVMType> Results, list<LLVMType> Inputs,
                        list<IntrinsicProperty> Properties = []>
      : Intrinsic<Results, Inputs,
          !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
                      Properties)>;
}

include "llvm/IR/IntrinsicsRISCVTensixContentsGroup.td"
