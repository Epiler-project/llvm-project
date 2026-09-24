//===-- IntrinsicsRISCVTensix.td --------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Exact bit carrier, not an overloaded LLVM/RVV vector interface. All calls
// carry hidden lane/configuration/issue effects and must not be duplicated.
let TargetPrefix = "riscv" in {
  class TensixIntrinsic<list<LLVMType> Results, list<LLVMType> Inputs,
                        list<IntrinsicProperty> Properties = []>
      : Intrinsic<Results, Inputs,
          !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
                      Properties)>;
  def int_riscv_tt_sfploadi : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
  def int_riscv_tt_sfpload : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_sfpstore : TensixIntrinsic<[],
      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_sfpadd : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_sfpmad : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
      llvm_i32_ty], [ImmArg<ArgIndex<4>>]>;
  // SFPMUL is the C9-specialized MAD form.  The old destination remains an
  // explicit passthrough for inactive lanes.
  def int_riscv_tt_sfpmul : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_sfpiadd : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<2>>]>;
  def int_riscv_tt_sfpmov : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<2>>]>;
  def int_riscv_tt_lreg_read : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
  def int_riscv_tt_lreg_write : TensixIntrinsic<[],
      [llvm_i32_ty, llvm_v32i32_ty], [ImmArg<ArgIndex<0>>]>;
  def int_riscv_tt_creg_read : TensixIntrinsic<[llvm_v32i32_ty],
      [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
  def int_riscv_tt_sfpencc : TensixIntrinsic<[],
      [llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
  def int_riscv_tt_sfpsetcc : TensixIntrinsic<[],
      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
  def int_riscv_tt_sfppushc : TensixIntrinsic<[],
      [llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
  def int_riscv_tt_sfppopc : TensixIntrinsic<[],
      [llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
  def int_riscv_tt_sfpnop : TensixIntrinsic<[], []>;
  // Condition complement has no data operands: ISA fields are all zero.
  def int_riscv_tt_sfpcompc : TensixIntrinsic<[], []>;
  // The admitted CONFIG forms are deliberately separate from arbitrary
  // LaneConfig/macro programming.  CReg writes consume L0 staging values.
  def int_riscv_tt_sfpconfig_creg : TensixIntrinsic<[],
      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_sfpconfig_reset : TensixIntrinsic<[], []>;
}
