//===-- IntrinsicsRISCVBOSC.td ----------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// BOSC's initial bound ABI is RV64, e32/m1, tail/mask agnostic, unmasked.
// Physical registers are MLIR-owned immediates, never LLVM vector SSA values.
// Unspecified memory effects deliberately serialize the external register state
// with scalar memory. No pure/readonly attribute may erase that state effect.
let TargetPrefix = "riscv" in {
class BOSCBoundIntrinsic<list<LLVMType> Results, list<LLVMType> Inputs,
                         list<IntrinsicProperty> Properties = []>
    : DefaultAttrsIntrinsic<Results, Inputs,
        !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate,
                     IntrNoCallback, IntrNoFree, IntrWillReturn], Properties)>;
// The patch9 observation timebase is the RV64 machine counter, not the
// unverified user-mode CYCLE alias selected by llvm.readcyclecounter. This
// fixed scalar operation preserves memory/issue ordering without introducing
// accelerator waits, fences, vector configuration, or an arbitrary CSR API.
def int_riscv_bosc_read_mcycle : BOSCBoundIntrinsic<[llvm_i64_ty], []>;
// One naturally aligned 8/32/64-bit element through a bound scratch group.
// The width is immediate; the result contains the zero-extended source bits.
// The closed operation
// restores the explicit active setvl result under the fixed e32 configuration,
// or canonical e32/VL0 without a source configuration. No CSR reads are used.
def int_riscv_bosc_read_bits : BOSCBoundIntrinsic<[llvm_i64_ty],
    [llvm_i32_ty, llvm_ptr_ty, llvm_i64_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>]>;
def int_riscv_bosc_setvl : BOSCBoundIntrinsic<[llvm_i64_ty], [llvm_i64_ty]>;
def int_riscv_bosc_load : BOSCBoundIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                                           [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_add : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_store : BOSCBoundIntrinsic<[], [llvm_i32_ty, llvm_ptr_ty],
                                            [ImmArg<ArgIndex<0>>]>;
// FP32 bit-pattern operations on MLIR-bound e32 physical VRF groups. Their
// operands are register numbers, not LLVM floating-point/vector SSA values.
def int_riscv_bosc_fadd : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_fsub : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_fmul : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_fdiv : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_fsplat : BOSCBoundIntrinsic<[], [llvm_i32_ty, llvm_float_ty],
                                              [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_fextract : BOSCBoundIntrinsic<[llvm_float_ty], [llvm_i32_ty],
                                                [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_freduce_sum : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_riscv_bosc_freduce_max : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
class BOSCBoundBinaryIntrinsic : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
class BOSCBoundUnaryIntrinsic : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_riscv_bosc_fmax : BOSCBoundBinaryIntrinsic;
def int_riscv_bosc_fmacc : BOSCBoundBinaryIntrinsic;
def int_riscv_bosc_fcmp_lt : BOSCBoundBinaryIntrinsic;
def int_riscv_bosc_fcmp_eq : BOSCBoundBinaryIntrinsic;
def int_riscv_bosc_fcvt_to_i32 : BOSCBoundUnaryIntrinsic;
def int_riscv_bosc_fcvt_from_i32 : BOSCBoundUnaryIntrinsic;
// Offline SFU operations consume and overwrite explicitly bound e32 groups.
// Their approximate numerical mode and launch admission stay target-owned.
def int_riscv_bosc_sfu_exp2 : BOSCBoundUnaryIntrinsic;
def int_riscv_bosc_sfu_rcp : BOSCBoundUnaryIntrinsic;
def int_riscv_bosc_merge : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
     ImmArg<ArgIndex<3>>]>;
def int_riscv_bosc_iadd_scalar : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
// Initial bounded gather accepts only constants 0..16.  The two physical
// groups must differ; index 16 returns zero only for exact VLEN=512.
def int_riscv_bosc_gather_scalar : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i64_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_shl : BOSCBoundBinaryIntrinsic;

// Exact patch7 PMA transitions, selected only after TReX receives the
// write-through-store/uncached-read platform memory contract. They expand in
// the leaf kernel to scalar CSR writes and full I/O fences, never support
// calls. Unspecified memory effects preserve DMA and bound register ordering.
def int_riscv_bosc_dma_writes : BOSCBoundIntrinsic<[], []>;
def int_riscv_bosc_dma_reads : BOSCBoundIntrinsic<[], []>;
def int_riscv_bosc_dma_fence : BOSCBoundIntrinsic<[], []>;

// AME uses MLIR-owned physical tile/accumulator slots.  The dimensions are
// scalar state operations; LLVM does not model a matrix value or allocate an
// AME register.  Load/store strides remain scalar operands so the receiver
// can preserve the hardware's row layout without reconstructing source IR.
def int_riscv_bosc_ame_setm : BOSCBoundIntrinsic<[llvm_i64_ty], [llvm_i64_ty]>;
def int_riscv_bosc_ame_setn : BOSCBoundIntrinsic<[llvm_i64_ty], [llvm_i64_ty]>;
def int_riscv_bosc_ame_setk : BOSCBoundIntrinsic<[llvm_i64_ty], [llvm_i64_ty]>;
def int_riscv_bosc_ame_load_a : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_ptr_ty, llvm_i64_ty], [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_ame_load_b : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_ptr_ty, llvm_i64_ty], [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_ame_load_c : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_ptr_ty, llvm_i64_ty], [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_ame_store_c : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_ptr_ty, llvm_i64_ty], [ImmArg<ArgIndex<0>>]>;
def int_riscv_bosc_ame_mma : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
// A boolean chooses one of two explicitly bound B registers. Keep the choice
// typed through IR and machine verification; MC emission expands one selected
// operation to two legal immediate-register alternatives, never a source loop.
def int_riscv_bosc_ame_load_b_select : BOSCBoundIntrinsic<[],
    [llvm_i1_ty, llvm_i32_ty, llvm_i32_ty, llvm_ptr_ty, llvm_i64_ty],
    [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_bosc_ame_mma_b_select : BOSCBoundIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>,
     ImmArg<ArgIndex<4>>]>;
}
