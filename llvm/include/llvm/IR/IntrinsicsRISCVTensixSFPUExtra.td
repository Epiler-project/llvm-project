//===-- IntrinsicsRISCVTensixSFPUExtra.td -----------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Explicit destination passthroughs preserve inactive lanes. The v32i32 values
// are SFPU SSA carriers; they are not RVV values or numeric register indices.
let TargetPrefix = "riscv" in {
class TensixUnaryIntrinsic : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;
def int_riscv_tt_sfparecip : TensixUnaryIntrinsic;
def int_riscv_tt_sfpexexp : TensixUnaryIntrinsic;
def int_riscv_tt_sfpexman : TensixUnaryIntrinsic;
def int_riscv_tt_sfpabs : TensixUnaryIntrinsic;
def int_riscv_tt_sfplz : TensixUnaryIntrinsic;
def int_riscv_tt_sfpcast : TensixUnaryIntrinsic;

class TensixSetImmediateIntrinsic : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
class TensixSetVectorIntrinsic : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<3>>]>;
def int_riscv_tt_sfpsetexp_i : TensixSetImmediateIntrinsic;
def int_riscv_tt_sfpsetexp_v : TensixSetVectorIntrinsic;
def int_riscv_tt_sfpsetman_i : TensixSetImmediateIntrinsic;
def int_riscv_tt_sfpsetman_v : TensixSetVectorIntrinsic;
def int_riscv_tt_sfpsetsgn_i : TensixSetImmediateIntrinsic;
def int_riscv_tt_sfpsetsgn_v : TensixSetVectorIntrinsic;

class TensixBitwiseIntrinsic : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty]>;
def int_riscv_tt_sfpand : TensixBitwiseIntrinsic;
def int_riscv_tt_sfpor : TensixBitwiseIntrinsic;
def int_riscv_tt_sfpxor : TensixBitwiseIntrinsic;
def int_riscv_tt_sfpnot : TensixIntrinsic<[llvm_v32i32_ty], [llvm_v32i32_ty]>;

class TensixScalarImmediateIntrinsic : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_tt_sfpiadd_i : TensixScalarImmediateIntrinsic;
def int_riscv_tt_sfpshft_i : TensixScalarImmediateIntrinsic;
def int_riscv_tt_sfpshft_v : TensixUnaryIntrinsic;
// Group-of-eight lane rotate/shift. Immediate is reserved and must be zero.
def int_riscv_tt_sfpshft2 : TensixSetImmediateIntrinsic;

def int_riscv_tt_sfpstochrnd_i : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_riscv_tt_sfpstochrnd_v : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_tt_sfplut : TensixIntrinsic<[llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty,
     llvm_v32i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<5>>]>;
def int_riscv_tt_sfpswap : TensixIntrinsic<[llvm_v32i32_ty, llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;
def int_riscv_tt_sfptransp : TensixIntrinsic<
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty],
    [llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty, llvm_v32i32_ty]>;
}
