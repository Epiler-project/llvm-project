//===-- IntrinsicsRISCVTensixContentsGroup.td ------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Results are numerical contents in architectural group order. Each old
// contents operand preserves disabled lanes; these are not register numbers.
// Literal struct results may only be projected with extractvalue, not passed
// through an ordinary function ABI or assembled into arbitrary aggregates.
let TargetPrefix = "riscv" in {
  def int_riscv_tt_contents_transpose8 :
      TensixIntrinsic<[llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty]>;
  def int_riscv_tt_contents_copy4 :
      TensixIntrinsic<[llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<4>>]>;
  def int_riscv_tt_contents_rotate4 :
      TensixIntrinsic<[llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty]>;
  // The pair form requires ENABLE_DEST_INDEX known disabled. Its two values
  // may be allocated anywhere in L0..L7; it does not describe argmin's extra
  // physical companion writes.
  def int_riscv_tt_contents_swap :
      TensixIntrinsic<[llvm_v32i32_ty, llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<2>>]>;
  // Direct table forms return only their actual destination. Old destination
  // preserves disabled lanes; following numerical sources occupy the actual
  // fixed table/input positions without hidden staging copies.
  def int_riscv_tt_contents_lut :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<5>>]>;
  def int_riscv_tt_contents_lutfp32_six :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<8>>]>;
  // FP16-3entry has a hardware destination selected per lane by old L7's low
  // four bits. Receive the whole eight-value update, including preserved lanes
  // and selector values >=8, which write no numerical register.
  def int_riscv_tt_contents_lutfp32_three :
      TensixIntrinsic<[llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<8>>]>;
  // Configuration writes hardware state, with no fabricated numerical result.
  // The source's first eight lanes are vertically broadcast from actual L0.
  // This admits C11..C14 source modes 0/8 and LaneConfig source mode 8 only;
  // mutable configured-register reads and load-macro programming are separate.
  def int_riscv_tt_contents_config_creg :
      TensixIntrinsic<[], [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
                       ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_contents_config_lane :
      TensixIntrinsic<[], [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
}
