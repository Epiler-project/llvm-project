//===-- IntrinsicsRISCVTensix.td --------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The pre-bound vector-carrier SFPU ABI was retired. Physical SFPU/LReg
// bindings are represented exclusively by IntrinsicsRISCVTensixBound.td.
// Keep the common class here because ordinary and bound Tensix intrinsics use
// the same convergent, non-duplicable side-effect contract.
let TargetPrefix = "riscv" in {
  class TensixIntrinsic<list<LLVMType> Results, list<LLVMType> Inputs,
                        list<IntrinsicProperty> Properties = []>
      : Intrinsic<Results, Inputs,
          !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
                      Properties)>;
}
