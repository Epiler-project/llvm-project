//===-- IntrinsicsTT.td ------------------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Architecture-qualified physical SFPU ingress. Register slots and logical
// fields remain separate i32 operands; no vector carrier or allocation ABI.
// Only Dst offsets may be separately proven defined, in-range scalar values.
let TargetPrefix = "tt" in {
class TTPhysicalSFPUIntrinsic<int Count, list<IntrinsicProperty> Fields =
    !foreach(Index, !range(Count), ImmArg<ArgIndex<Index>>)>
    : Intrinsic<[], !listsplat(llvm_i32_ty, Count),
        !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
                    Fields)>;

// The recording identity is symbolic. Machine replay determines the final
// length and placement after checking the actual body's native effects.
class TTReplayTemplateControl<int Count>
    : Intrinsic<[], !listsplat(llvm_i32_ty, Count),
        !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
                    !foreach(Index, !range(Count), ImmArg<ArgIndex<Index>>))>;
def int_tt_wh_replay_template_begin : TTReplayTemplateControl<4>;
def int_tt_wh_replay_template_end : TTReplayTemplateControl<1>;
def int_tt_wh_replay_template_execute : TTReplayTemplateControl<1>;
def int_tt_wh_replay_template_mop : TTReplayTemplateControl<2>;
def int_tt_wh_mop_execute : TTReplayTemplateControl<3>;
def int_tt_bh_replay_template_begin : TTReplayTemplateControl<4>;
def int_tt_bh_replay_template_end : TTReplayTemplateControl<1>;
def int_tt_bh_replay_template_execute : TTReplayTemplateControl<1>;
def int_tt_bh_replay_template_mop : TTReplayTemplateControl<2>;

// Wormhole B0: no ARECIP/GT/LE hardware instructions.
def int_tt_wh_sfpu_loadi : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_macro_address_reset : TTPhysicalSFPUIntrinsic<0>;
def int_tt_wh_sfpu_macro_configure_addi : TTPhysicalSFPUIntrinsic<2>;
def int_tt_wh_sfpu_load_macro_addi : TTPhysicalSFPUIntrinsic<3>;
def int_tt_wh_sfpu_load : TTPhysicalSFPUIntrinsic<5,
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_tt_wh_sfpu_store : TTPhysicalSFPUIntrinsic<4,
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_tt_wh_sfpu_add : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_mul : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_mad : TTPhysicalSFPUIntrinsic<6>;
def int_tt_wh_sfpu_iadd : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_mov : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_mov_all : TTPhysicalSFPUIntrinsic<2>;
// Active lanes advance PRNG once and receive the updated state. The old
// destination preserves inactive lanes; selector9/mode8 are fixed.
def int_tt_wh_sfpu_mov_prng_advance : TTPhysicalSFPUIntrinsic<2>;
def int_tt_wh_sfpu_exexp : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_exman : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_abs : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_lz : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_cast : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_setexp_i : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_setexp_v : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_setman_i : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_setman_v : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_setsgn_i : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_setsgn_v : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_iadd_i : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_shft_i : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_and : TTPhysicalSFPUIntrinsic<3>;
def int_tt_wh_sfpu_or : TTPhysicalSFPUIntrinsic<3>;
def int_tt_wh_sfpu_xor : TTPhysicalSFPUIntrinsic<3>;
def int_tt_wh_sfpu_not : TTPhysicalSFPUIntrinsic<2>;
def int_tt_wh_sfpu_shft_v : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_shft2 : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_shft2_copy4 : TTPhysicalSFPUIntrinsic<9>;
def int_tt_wh_sfpu_shft2_rotate4 : TTPhysicalSFPUIntrinsic<9>;
// round: 0 nearest, 1 stochastic; round 2 is unsupported.
def int_tt_wh_sfpu_stochrnd_i : TTPhysicalSFPUIntrinsic<6>;
// round: 0 nearest, 1 stochastic; round 2 is unsupported.
def int_tt_wh_sfpu_stochrnd_v : TTPhysicalSFPUIntrinsic<6>;
def int_tt_wh_sfpu_lutfp32_6r : TTPhysicalSFPUIntrinsic<10>;
def int_tt_wh_sfpu_lutfp32_3r : TTPhysicalSFPUIntrinsic<17>;
def int_tt_wh_sfpu_lut : TTPhysicalSFPUIntrinsic<7>;
def int_tt_wh_sfpu_swap : TTPhysicalSFPUIntrinsic<5>;
def int_tt_wh_sfpu_transp : TTPhysicalSFPUIntrinsic<12>;
def int_tt_wh_sfpu_setcc : TTPhysicalSFPUIntrinsic<3>;
def int_tt_wh_sfpu_encc : TTPhysicalSFPUIntrinsic<2>;
def int_tt_wh_sfpu_pushc : TTPhysicalSFPUIntrinsic<2>;
def int_tt_wh_sfpu_popc : TTPhysicalSFPUIntrinsic<2>;
def int_tt_wh_sfpu_config_creg : TTPhysicalSFPUIntrinsic<4>;
def int_tt_wh_sfpu_config_lane : TTPhysicalSFPUIntrinsic<3>;
def int_tt_wh_sfpu_nop : TTPhysicalSFPUIntrinsic<0>;
def int_tt_wh_sfpu_compc : TTPhysicalSFPUIntrinsic<0>;
def int_tt_wh_sfpu_config_reset : TTPhysicalSFPUIntrinsic<0>;

// Blackhole: receives its native modes, independent of Wormhole limits.
def int_tt_bh_sfpu_loadi : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_macro_address_reset : TTPhysicalSFPUIntrinsic<0>;
def int_tt_bh_sfpu_macro_configure_addi : TTPhysicalSFPUIntrinsic<2>;
def int_tt_bh_sfpu_load_macro_addi : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_load : TTPhysicalSFPUIntrinsic<5,
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_tt_bh_sfpu_store : TTPhysicalSFPUIntrinsic<4,
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_tt_bh_sfpu_add : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_mul : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_mul23 : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_mad : TTPhysicalSFPUIntrinsic<6>;
def int_tt_bh_sfpu_iadd : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_mov : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_mov_all : TTPhysicalSFPUIntrinsic<2>;
// Active lanes advance PRNG once and receive the updated state. The old
// destination preserves inactive lanes; selector9/mode8 are fixed.
def int_tt_bh_sfpu_mov_prng_advance : TTPhysicalSFPUIntrinsic<2>;
def int_tt_bh_sfpu_arecip : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_exexp : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_exman : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_abs : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_lz : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_cast : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_setexp_i : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_setexp_v : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_setman_i : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_setman_v : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_setsgn_i : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_setsgn_v : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_iadd_i : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_shft_i : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_and : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_or : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_xor : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_not : TTPhysicalSFPUIntrinsic<2>;
def int_tt_bh_sfpu_shft_v : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_shft2 : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_shft2_copy4 : TTPhysicalSFPUIntrinsic<9>;
def int_tt_bh_sfpu_shft2_rotate4 : TTPhysicalSFPUIntrinsic<9>;
// round: 0 nearest, 1 stochastic, 2 zero.
def int_tt_bh_sfpu_stochrnd_i : TTPhysicalSFPUIntrinsic<6>;
// round: 0 nearest, 1 stochastic, 2 zero.
def int_tt_bh_sfpu_stochrnd_v : TTPhysicalSFPUIntrinsic<6>;
def int_tt_bh_sfpu_lutfp32_6r : TTPhysicalSFPUIntrinsic<10>;
def int_tt_bh_sfpu_lutfp32_3r : TTPhysicalSFPUIntrinsic<17>;
def int_tt_bh_sfpu_lut : TTPhysicalSFPUIntrinsic<7>;
def int_tt_bh_sfpu_swap : TTPhysicalSFPUIntrinsic<5>;
def int_tt_bh_sfpu_transp : TTPhysicalSFPUIntrinsic<12>;
def int_tt_bh_sfpu_setcc : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_gt : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_le : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_encc : TTPhysicalSFPUIntrinsic<2>;
def int_tt_bh_sfpu_pushc : TTPhysicalSFPUIntrinsic<2>;
def int_tt_bh_sfpu_popc : TTPhysicalSFPUIntrinsic<2>;
def int_tt_bh_sfpu_config_creg : TTPhysicalSFPUIntrinsic<4>;
def int_tt_bh_sfpu_config_lane : TTPhysicalSFPUIntrinsic<3>;
def int_tt_bh_sfpu_nop : TTPhysicalSFPUIntrinsic<0>;
def int_tt_bh_sfpu_compc : TTPhysicalSFPUIntrinsic<0>;
def int_tt_bh_sfpu_config_reset : TTPhysicalSFPUIntrinsic<0>;

}
