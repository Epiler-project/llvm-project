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

// Scalar device read-response ordering and coprocessor synchronization.
// These preserve full memory effects, convergence and nonduplication.
class TTIntrinsic<list<LLVMType> Results, list<LLVMType> Inputs,
                  list<IntrinsicProperty> Properties = []>
    : Intrinsic<Results, Inputs,
        !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
                    Properties)>;

def int_tt_dependent_use : TTIntrinsic<[], [llvm_i32_ty]>;
def int_tt_read_response : TTIntrinsic<[], [llvm_i32_ty]>;
def int_tt_coprocessor_wait : TTIntrinsic<[], []>;

// Local MOP slot clearing, scalar control writes, replay record markers,
// symbolic replay templates, and compiler-private folded instruction-port words.
def int_tt_mop_clear : TTIntrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_tt_mop_control_write : TTIntrinsic<[], [llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>]>;
def int_tt_replay_record_end : TTIntrinsic<[], []>;
def int_tt_replay_template_begin : TTIntrinsic<[],
    [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
     ImmArg<ArgIndex<3>>]>;
def int_tt_replay_template_end : TTIntrinsic<[], [llvm_i32_ty],
    [ImmArg<ArgIndex<0>>]>;
def int_tt_replay_template_execute : TTIntrinsic<[], [llvm_i32_ty],
    [ImmArg<ArgIndex<0>>]>;
def int_tt_replay_template_mop : TTIntrinsic<[], [llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_tt_issue_word : TTIntrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// Non-SFPU physical Tensix instructions (direct, instruction-port, and MOP slot).
class TTOrdinaryIntrinsic<int N, bit Port>
    : Intrinsic<[], !listsplat(llvm_i32_ty, !add(N, !if(Port, 1, 0))),
        !listconcat([IntrHasSideEffects, IntrConvergent, IntrNoDuplicate],
          !if(Port, [ImmArg<ArgIndex<0>>],
              !foreach(I, !range(N), ImmArg<ArgIndex<I>>)))>;
def int_tt_adddmareg : TTOrdinaryIntrinsic<4, 0>;
def int_tt_adddmareg_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_adddmareg_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_addrcrxy : TTOrdinaryIntrinsic<6, 0>;
def int_tt_addrcrxy_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_addrcrxy_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_addrcrzw : TTOrdinaryIntrinsic<6, 0>;
def int_tt_addrcrzw_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_addrcrzw_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_apool3s1 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_apool3s1_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_apool3s1_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_apool3s2 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_apool3s2_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_apool3s2_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_atcas : TTOrdinaryIntrinsic<6, 0>;
def int_tt_atcas_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_atcas_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_atgetm : TTOrdinaryIntrinsic<1, 0>;
def int_tt_atgetm_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_atgetm_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_atincget : TTOrdinaryIntrinsic<5, 0>;
def int_tt_atincget_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_atincget_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_atincgetptr : TTOrdinaryIntrinsic<7, 0>;
def int_tt_atincgetptr_port : TTOrdinaryIntrinsic<7, 1>;
def int_tt_atincgetptr_mop : TTOrdinaryIntrinsic<8, 0>;
def int_tt_atrelm : TTOrdinaryIntrinsic<1, 0>;
def int_tt_atrelm_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_atrelm_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_atswap : TTOrdinaryIntrinsic<4, 0>;
def int_tt_atswap_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_atswap_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_bitwopdmareg : TTOrdinaryIntrinsic<5, 0>;
def int_tt_bitwopdmareg_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_bitwopdmareg_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_cfgshiftmask : TTOrdinaryIntrinsic<6, 0>;
def int_tt_cfgshiftmask_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_cfgshiftmask_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_cleardvalid : TTOrdinaryIntrinsic<2, 0>;
def int_tt_cleardvalid_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_cleardvalid_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_clrexphist : TTOrdinaryIntrinsic<0, 0>;
def int_tt_clrexphist_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_clrexphist_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_cmpdmareg : TTOrdinaryIntrinsic<5, 0>;
def int_tt_cmpdmareg_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_cmpdmareg_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_conv3s1 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_conv3s1_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_conv3s1_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_conv3s2 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_conv3s2_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_conv3s2_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_dmanop : TTOrdinaryIntrinsic<0, 0>;
def int_tt_dmanop_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_dmanop_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_dotpv : TTOrdinaryIntrinsic<5, 0>;
def int_tt_dotpv_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_dotpv_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_elwadd : TTOrdinaryIntrinsic<5, 0>;
def int_tt_elwadd_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_elwadd_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_elwmul : TTOrdinaryIntrinsic<5, 0>;
def int_tt_elwmul_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_elwmul_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_elwsub : TTOrdinaryIntrinsic<5, 0>;
def int_tt_elwsub_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_elwsub_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_flushdma : TTOrdinaryIntrinsic<1, 0>;
def int_tt_flushdma_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_flushdma_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_gapool : TTOrdinaryIntrinsic<5, 0>;
def int_tt_gapool_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_gapool_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_gatesrcrst : TTOrdinaryIntrinsic<2, 0>;
def int_tt_gatesrcrst_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_gatesrcrst_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_gmpool : TTOrdinaryIntrinsic<5, 0>;
def int_tt_gmpool_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_gmpool_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_incadcxy : TTOrdinaryIntrinsic<5, 0>;
def int_tt_incadcxy_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_incadcxy_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_incadczw : TTOrdinaryIntrinsic<5, 0>;
def int_tt_incadczw_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_incadczw_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_incrwc : TTOrdinaryIntrinsic<4, 0>;
def int_tt_incrwc_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_incrwc_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_loadind : TTOrdinaryIntrinsic<5, 0>;
def int_tt_loadind_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_loadind_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_loadreg : TTOrdinaryIntrinsic<2, 0>;
def int_tt_loadreg_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_loadreg_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_mfconv3s1 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_mfconv3s1_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_mfconv3s1_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_mop_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_mop_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_mop_cfg : TTOrdinaryIntrinsic<1, 0>;
def int_tt_mop_cfg_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_mop_cfg_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_mova2d : TTOrdinaryIntrinsic<5, 0>;
def int_tt_mova2d_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_mova2d_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_movb2a : TTOrdinaryIntrinsic<4, 0>;
def int_tt_movb2a_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_movb2a_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_movb2d : TTOrdinaryIntrinsic<5, 0>;
def int_tt_movb2d_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_movb2d_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_movd2a : TTOrdinaryIntrinsic<5, 0>;
def int_tt_movd2a_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_movd2a_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_movd2b : TTOrdinaryIntrinsic<5, 0>;
def int_tt_movd2b_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_movd2b_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_movdbga2d : TTOrdinaryIntrinsic<5, 0>;
def int_tt_movdbga2d_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_movdbga2d_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_movdbgb2d : TTOrdinaryIntrinsic<5, 0>;
def int_tt_movdbgb2d_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_movdbgb2d_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_mpool3s1 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_mpool3s1_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_mpool3s1_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_mpool3s2 : TTOrdinaryIntrinsic<4, 0>;
def int_tt_mpool3s2_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_mpool3s2_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_muldmareg : TTOrdinaryIntrinsic<4, 0>;
def int_tt_muldmareg_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_muldmareg_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_mvmul : TTOrdinaryIntrinsic<4, 0>;
def int_tt_mvmul_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_mvmul_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_nop : TTOrdinaryIntrinsic<0, 0>;
def int_tt_nop_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_nop_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_pacr : TTOrdinaryIntrinsic<12, 0>;
def int_tt_pacr_port : TTOrdinaryIntrinsic<12, 1>;
def int_tt_pacr_mop : TTOrdinaryIntrinsic<13, 0>;
def int_tt_pacr_setreg : TTOrdinaryIntrinsic<8, 0>;
def int_tt_pacr_setreg_port : TTOrdinaryIntrinsic<8, 1>;
def int_tt_pacr_setreg_mop : TTOrdinaryIntrinsic<9, 0>;
def int_tt_rareb : TTOrdinaryIntrinsic<0, 0>;
def int_tt_rareb_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_rareb_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_rdcfg : TTOrdinaryIntrinsic<2, 0>;
def int_tt_rdcfg_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_rdcfg_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_reg2flop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_reg2flop_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_reg2flop_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_replay : TTOrdinaryIntrinsic<4, 0>;
def int_tt_replay_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_replay_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_resourcedecl : TTOrdinaryIntrinsic<3, 0>;
def int_tt_resourcedecl_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_resourcedecl_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_rmwcib0 : TTOrdinaryIntrinsic<3, 0>;
def int_tt_rmwcib0_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_rmwcib0_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_rmwcib1 : TTOrdinaryIntrinsic<3, 0>;
def int_tt_rmwcib1_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_rmwcib1_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_rmwcib2 : TTOrdinaryIntrinsic<3, 0>;
def int_tt_rmwcib2_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_rmwcib2_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_rmwcib3 : TTOrdinaryIntrinsic<3, 0>;
def int_tt_rmwcib3_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_rmwcib3_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_rstdma : TTOrdinaryIntrinsic<0, 0>;
def int_tt_rstdma_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_rstdma_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_semget : TTOrdinaryIntrinsic<1, 0>;
def int_tt_semget_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_semget_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_seminit : TTOrdinaryIntrinsic<3, 0>;
def int_tt_seminit_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_seminit_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_sempost : TTOrdinaryIntrinsic<1, 0>;
def int_tt_sempost_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_sempost_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_semwait : TTOrdinaryIntrinsic<3, 0>;
def int_tt_semwait_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_semwait_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_setadc : TTOrdinaryIntrinsic<4, 0>;
def int_tt_setadc_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_setadc_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_setadcxx : TTOrdinaryIntrinsic<3, 0>;
def int_tt_setadcxx_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_setadcxx_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_setadcxy : TTOrdinaryIntrinsic<6, 0>;
def int_tt_setadcxy_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_setadcxy_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_setadczw : TTOrdinaryIntrinsic<6, 0>;
def int_tt_setadczw_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_setadczw_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_setashrmh : TTOrdinaryIntrinsic<2, 0>;
def int_tt_setashrmh_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_setashrmh_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_setashrmh0 : TTOrdinaryIntrinsic<2, 0>;
def int_tt_setashrmh0_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_setashrmh0_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_setashrmh1 : TTOrdinaryIntrinsic<2, 0>;
def int_tt_setashrmh1_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_setashrmh1_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_setashrmv : TTOrdinaryIntrinsic<1, 0>;
def int_tt_setashrmv_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_setashrmv_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_setc16 : TTOrdinaryIntrinsic<2, 0>;
def int_tt_setc16_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_setc16_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_setdmareg : TTOrdinaryIntrinsic<4, 0>;
def int_tt_setdmareg_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_setdmareg_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_setdvalid : TTOrdinaryIntrinsic<1, 0>;
def int_tt_setdvalid_port : TTOrdinaryIntrinsic<1, 1>;
def int_tt_setdvalid_mop : TTOrdinaryIntrinsic<2, 0>;
def int_tt_setibrwc : TTOrdinaryIntrinsic<3, 0>;
def int_tt_setibrwc_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_setibrwc_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_setpkedgof : TTOrdinaryIntrinsic<4, 0>;
def int_tt_setpkedgof_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_setpkedgof_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_setrwc : TTOrdinaryIntrinsic<6, 0>;
def int_tt_setrwc_port : TTOrdinaryIntrinsic<6, 1>;
def int_tt_setrwc_mop : TTOrdinaryIntrinsic<7, 0>;
def int_tt_shiftdmareg : TTOrdinaryIntrinsic<5, 0>;
def int_tt_shiftdmareg_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_shiftdmareg_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_shiftxa : TTOrdinaryIntrinsic<2, 0>;
def int_tt_shiftxa_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_shiftxa_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_shiftxb : TTOrdinaryIntrinsic<3, 0>;
def int_tt_shiftxb_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_shiftxb_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_stallwait : TTOrdinaryIntrinsic<2, 0>;
def int_tt_stallwait_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_stallwait_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_storeind : TTOrdinaryIntrinsic<7, 0>;
def int_tt_storeind_port : TTOrdinaryIntrinsic<7, 1>;
def int_tt_storeind_mop : TTOrdinaryIntrinsic<8, 0>;
def int_tt_storereg : TTOrdinaryIntrinsic<2, 0>;
def int_tt_storereg_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_storereg_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_streamwait : TTOrdinaryIntrinsic<4, 0>;
def int_tt_streamwait_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_streamwait_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_streamwrcfg : TTOrdinaryIntrinsic<3, 0>;
def int_tt_streamwrcfg_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_streamwrcfg_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_subdmareg : TTOrdinaryIntrinsic<4, 0>;
def int_tt_subdmareg_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_subdmareg_mop : TTOrdinaryIntrinsic<5, 0>;
def int_tt_tbufcmd : TTOrdinaryIntrinsic<0, 0>;
def int_tt_tbufcmd_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_tbufcmd_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_trnspsrca : TTOrdinaryIntrinsic<0, 0>;
def int_tt_trnspsrca_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_trnspsrca_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_trnspsrcb : TTOrdinaryIntrinsic<0, 0>;
def int_tt_trnspsrcb_port : TTOrdinaryIntrinsic<0, 1>;
def int_tt_trnspsrcb_mop : TTOrdinaryIntrinsic<1, 0>;
def int_tt_unpacr : TTOrdinaryIntrinsic<13, 0>;
def int_tt_unpacr_port : TTOrdinaryIntrinsic<13, 1>;
def int_tt_unpacr_mop : TTOrdinaryIntrinsic<14, 0>;
def int_tt_unpacr_nop : TTOrdinaryIntrinsic<9, 0>;
def int_tt_unpacr_nop_port : TTOrdinaryIntrinsic<9, 1>;
def int_tt_unpacr_nop_mop : TTOrdinaryIntrinsic<10, 0>;
def int_tt_wrcfg : TTOrdinaryIntrinsic<3, 0>;
def int_tt_wrcfg_port : TTOrdinaryIntrinsic<3, 1>;
def int_tt_wrcfg_mop : TTOrdinaryIntrinsic<4, 0>;
def int_tt_xmov : TTOrdinaryIntrinsic<2, 0>;
def int_tt_xmov_port : TTOrdinaryIntrinsic<2, 1>;
def int_tt_xmov_mop : TTOrdinaryIntrinsic<3, 0>;
def int_tt_zeroacc : TTOrdinaryIntrinsic<5, 0>;
def int_tt_zeroacc_port : TTOrdinaryIntrinsic<5, 1>;
def int_tt_zeroacc_mop : TTOrdinaryIntrinsic<6, 0>;
def int_tt_zerosrc : TTOrdinaryIntrinsic<4, 0>;
def int_tt_zerosrc_port : TTOrdinaryIntrinsic<4, 1>;
def int_tt_zerosrc_mop : TTOrdinaryIntrinsic<5, 0>;

}
