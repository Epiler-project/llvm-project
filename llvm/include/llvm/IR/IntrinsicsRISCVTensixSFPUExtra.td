//===-- IntrinsicsRISCVTensixSFPUExtra.td -----------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Numerical contents SSA is distinct from the retired implicit-register ABI.
// Each value is the raw bits of 32 SFPU lanes. It is never an RVV value, a
// register identity, or a memory address. Predicated/partial definitions take
// their previous destination explicitly; the machine receiver ties it to the
// result and diagnoses any allocation that would require an implicit copy.
let TargetPrefix = "riscv" in {
  def int_riscv_tt_contents_read_constant :
      TensixIntrinsic<[llvm_v32i32_ty], [llvm_i32_ty],
                      [ImmArg<ArgIndex<0>>]>;
  def int_riscv_tt_contents_load :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  foreach Operation = ["add", "mul"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[llvm_v32i32_ty],
                        [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                         llvm_i32_ty], [ImmArg<ArgIndex<3>>]>;
  def int_riscv_tt_contents_store :
      TensixIntrinsic<[],
                      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

  def int_riscv_tt_contents_load_immediate :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
  def int_riscv_tt_contents_mad :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_v32i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<4>>]>;
  foreach Operation = ["iadd", "mov", "arecip", "exexp", "exman", "abs",
                       "lz", "cast", "setexp_vector", "setman_vector",
                       "setsgn_vector", "shft_vector", "shft2"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[llvm_v32i32_ty],
                        [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
                        [ImmArg<ArgIndex<2>>]>;
  foreach Operation = ["mov_all", "prng_advance", "not"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[llvm_v32i32_ty], [llvm_v32i32_ty]>;
  foreach Operation = ["and", "or", "xor"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[llvm_v32i32_ty], [llvm_v32i32_ty, llvm_v32i32_ty]>;
  foreach Operation = ["setexp_immediate", "setman_immediate",
                       "setsgn_immediate"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[llvm_v32i32_ty],
                        [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty,
                         llvm_i32_ty],
                        [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
  foreach Operation = ["iadd_immediate", "shft_immediate"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[llvm_v32i32_ty],
                        [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
                        [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
  def int_riscv_tt_contents_stochrnd_immediate :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty,
                       llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>,
                       ImmArg<ArgIndex<4>>]>;
  def int_riscv_tt_contents_stochrnd_vector :
      TensixIntrinsic<[llvm_v32i32_ty],
                      [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
                       llvm_i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

  // Fixed architectural state remains state, with no fabricated value result.
  // Data-dependent predicate actions read numerical SSA. Descriptors retain
  // the actual CC/CC-stack/configuration/issue/PRNG use and def footprint.
  def int_riscv_tt_contents_encc : TensixIntrinsic<[], [llvm_i32_ty, llvm_i32_ty],
      [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
  foreach Operation = ["pushc", "popc"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
  def int_riscv_tt_contents_setcc :
      TensixIntrinsic<[], [llvm_v32i32_ty, llvm_i32_ty],
                      [ImmArg<ArgIndex<1>>]>;
  foreach Operation = ["gt", "le"] in
    def int_riscv_tt_contents_#Operation :
        TensixIntrinsic<[], [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
                        [ImmArg<ArgIndex<2>>]>;
  foreach Operation = ["compc", "nop", "config_reset"] in
    def int_riscv_tt_contents_#Operation : TensixIntrinsic<[], []>;
}
