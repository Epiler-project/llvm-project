//===-- IntrinsicsRISCVTensixBound.td ---------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Physical SFPU ingress. Every register is an explicit i32 immarg, every call
// returns void, and logical fields remain separate. No vector carrier or
// implicit preservation/allocation is part of this ABI.
let TargetPrefix = "riscv" in {
class TensixBoundIntrinsic<int Count> : TensixIntrinsic<[],
    !listsplat(llvm_i32_ty, Count),
    !foreach(Index, !range(Count), ImmArg<ArgIndex<Index>>)>;

def int_riscv_tt_bound_sfploadi : TensixBoundIntrinsic<4>;
// The Dst offset may be a separately proven defined, in-range scalar.
def int_riscv_tt_bound_sfpload : TensixIntrinsic<[],
    !listsplat(llvm_i32_ty, 5),
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
     ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_riscv_tt_bound_sfpstore : TensixIntrinsic<[],
    !listsplat(llvm_i32_ty, 4),
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_riscv_tt_bound_sfpadd : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpmul : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpmad : TensixBoundIntrinsic<6>;
def int_riscv_tt_bound_sfpiadd : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpmov : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpmov_all : TensixBoundIntrinsic<2>;
def int_riscv_tt_bound_sfparecip : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpexexp : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpexman : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpabs : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfplz : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpcast : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpsetexp_i : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpsetexp_v : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpsetman_i : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpsetman_v : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpsetsgn_i : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpsetsgn_v : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpiadd_i : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpshft_i : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpand : TensixBoundIntrinsic<3>;
def int_riscv_tt_bound_sfpor : TensixBoundIntrinsic<3>;
def int_riscv_tt_bound_sfpxor : TensixBoundIntrinsic<3>;
def int_riscv_tt_bound_sfpnot : TensixBoundIntrinsic<2>;
def int_riscv_tt_bound_sfpshft_v : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpshft2 : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfpstochrnd_i : TensixBoundIntrinsic<6>;
def int_riscv_tt_bound_sfpstochrnd_v : TensixBoundIntrinsic<6>;
def int_riscv_tt_bound_sfplut : TensixBoundIntrinsic<7>;
def int_riscv_tt_bound_sfpswap : TensixBoundIntrinsic<5>;
def int_riscv_tt_bound_sfptransp : TensixBoundIntrinsic<12>;
def int_riscv_tt_bound_sfpsetcc : TensixBoundIntrinsic<3>;
def int_riscv_tt_bound_sfpencc : TensixBoundIntrinsic<2>;
def int_riscv_tt_bound_sfppushc : TensixBoundIntrinsic<2>;
def int_riscv_tt_bound_sfppopc : TensixBoundIntrinsic<2>;
def int_riscv_tt_bound_sfpconfig_creg : TensixBoundIntrinsic<4>;
def int_riscv_tt_bound_sfpnop : TensixBoundIntrinsic<0>;
def int_riscv_tt_bound_sfpcompc : TensixBoundIntrinsic<0>;
def int_riscv_tt_bound_sfpconfig_reset : TensixBoundIntrinsic<0>;
}
