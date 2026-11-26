; Wormhole selects all 47 physical forms with its own field restrictions.
; Every frozen bound form must select physical operands before allocation.
; In particular vector field writes and fixed groups must not stage values.
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefixes=CHECK,EXPLICIT-FIXED-GROUP --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefixes=CHECK,EXPLICIT-FIXED-GROUP --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.tt.wh.sfpu.loadi(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.load(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.add(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mul(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.iadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mov(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mov.all(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.exexp(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.exman(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.abs(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.lz(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.cast(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setexp.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setexp.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setman.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setman.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setsgn.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setsgn.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.iadd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.shft.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.shft.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.and(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.or(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.xor(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.not(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.shft2(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.stochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.stochrnd.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.lut(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.swap(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.transp(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.setcc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.encc(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.pushc(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.popc(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.config.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.nop()
declare void @llvm.tt.wh.sfpu.compc()
declare void @llvm.tt.wh.sfpu.config.reset()

define void @bound_arithmetic_trisc0() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: bound_arithmetic_trisc0
; CHECK: TTSFPENCC 3, 10
; CHECK-NEXT: $tt_l0 = TTSFPMOVAll $tt_c9
; CHECK-NEXT: $tt_l1 = TTSFPMOVAll $tt_c10
; CHECK-NEXT: $tt_l2 = TTSFPMOVAll $tt_c9
; CHECK-NEXT: $tt_l0 = TTSFPLOADI $tt_l0, 7, 2
; CHECK-NEXT: $tt_l2 = TTSFPADD $tt_l2, $tt_l0, $tt_l1, 0
; CHECK-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 0
; CHECK-NEXT: $tt_l2 = TTSFPMAD $tt_l2, $tt_l0, $tt_l1, $tt_c9, 0
; CHECK-NEXT: $tt_l2 = TTSFPIADD $tt_l2, $tt_l1
; CHECK-NEXT: $tt_l2 = TTSFPMOV $tt_l2, $tt_l0
; CHECK-NEXT: $tt_l2 = TTSFPMOVNeg $tt_l2, $tt_l1
; CHECK-NEXT: $tt_l2 = TTSFPEXEXP $tt_l2, $tt_l0, 8
; CHECK-NEXT: $tt_l2 = TTSFPEXMAN $tt_l2, $tt_l0, 1
; CHECK-NEXT: $tt_l2 = TTSFPABS $tt_l2, $tt_l0, 0
; CHECK-NEXT: $tt_l2 = TTSFPLZ $tt_l2, $tt_l0, 2
; CHECK-NEXT: $tt_l2 = TTSFPCAST $tt_l2, $tt_l0, 1
; CHECK-NEXT: $tt_l2 = TTSFPSETEXP $tt_l2, $tt_l0, 127, 1
; CHECK-NEXT: $tt_l2 = TTSFPSETMAN $tt_l2, $tt_l0, 123, 1
; CHECK-NEXT: $tt_l2 = TTSFPSETSGN $tt_l2, $tt_l0, 1, 1
; CHECK-NEXT: $tt_l2 = TTSFPSETEXP $tt_l2, $tt_l0, 0, 2
; CHECK-NEXT: $tt_l2 = TTSFPSETMAN $tt_l2, $tt_l0, 0, 0
; CHECK-NEXT: $tt_l2 = TTSFPSETSGN $tt_l2, $tt_l0, 0, 0
; CHECK-NEXT: $tt_l2 = TTSFPIADDI $tt_l2, $tt_l2, -1, 5
; CHECK-NEXT: $tt_l2 = TTSFPSHFT $tt_l2, $tt_l2, 3, 1
; CHECK-NEXT: $tt_l2 = TTSFPSHFT $tt_l2, $tt_l0, 0, 0
; CHECK-NEXT: $tt_l2 = TTSFPAND $tt_l2, $tt_l0
; CHECK-NEXT: $tt_l2 = TTSFPOR $tt_l2, $tt_l1
; CHECK-NEXT: $tt_l2 = TTSFPXOR $tt_l2, $tt_c9
; CHECK-NEXT: $tt_l2 = TTSFPNOT $tt_l2, $tt_l2
; CHECK-NEXT: $tt_l2 = TTSFPSHFT2 $tt_l2, $tt_l0, 3
; CHECK-NEXT: $tt_l2 = TTSFPSTOCHRNDI $tt_l2, $tt_l0, 7, 12, 1
; CHECK-NEXT: $tt_l2 = TTSFPSTOCHRNDV $tt_l2, $tt_l0, $tt_l1, 5, 0
; CHECK-NEXT: $tt_l0, $tt_l1 = TTSFPSWAP $tt_l0, $tt_l1, 9
; CHECK-NEXT: PseudoRET
  call void @llvm.tt.wh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 1, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 2, i32 9)
  call void @llvm.tt.wh.sfpu.loadi(i32 0, i32 0, i32 7, i32 2)
  call void @llvm.tt.wh.sfpu.add(i32 2, i32 2, i32 0, i32 1, i32 0)
  call void @llvm.tt.wh.sfpu.mul(i32 2, i32 2, i32 0, i32 1, i32 0)
  call void @llvm.tt.wh.sfpu.mad(i32 2, i32 2, i32 0, i32 1, i32 9, i32 0)
  call void @llvm.tt.wh.sfpu.iadd(i32 2, i32 2, i32 1, i32 4)
  call void @llvm.tt.wh.sfpu.mov(i32 2, i32 2, i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.mov(i32 2, i32 2, i32 1, i32 1)
  call void @llvm.tt.wh.sfpu.exexp(i32 2, i32 2, i32 0, i32 8)
  call void @llvm.tt.wh.sfpu.exman(i32 2, i32 2, i32 0, i32 1)
  call void @llvm.tt.wh.sfpu.abs(i32 2, i32 2, i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.lz(i32 2, i32 2, i32 0, i32 2)
  call void @llvm.tt.wh.sfpu.cast(i32 2, i32 2, i32 0, i32 1)
  call void @llvm.tt.wh.sfpu.setexp.i(i32 2, i32 2, i32 0, i32 127, i32 1)
  call void @llvm.tt.wh.sfpu.setman.i(i32 2, i32 2, i32 0, i32 123, i32 1)
  call void @llvm.tt.wh.sfpu.setsgn.i(i32 2, i32 2, i32 0, i32 1, i32 1)
  call void @llvm.tt.wh.sfpu.setexp.v(i32 2, i32 2, i32 0, i32 2, i32 2)
  call void @llvm.tt.wh.sfpu.setman.v(i32 2, i32 2, i32 0, i32 2, i32 0)
  call void @llvm.tt.wh.sfpu.setsgn.v(i32 2, i32 2, i32 0, i32 2, i32 0)
  call void @llvm.tt.wh.sfpu.iadd.i(i32 2, i32 2, i32 -1, i32 5)
  call void @llvm.tt.wh.sfpu.shft.i(i32 2, i32 2, i32 3, i32 1)
  call void @llvm.tt.wh.sfpu.shft.v(i32 2, i32 2, i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.and(i32 2, i32 2, i32 0)
  call void @llvm.tt.wh.sfpu.or(i32 2, i32 2, i32 1)
  call void @llvm.tt.wh.sfpu.xor(i32 2, i32 2, i32 9)
  call void @llvm.tt.wh.sfpu.not(i32 2, i32 2)
  call void @llvm.tt.wh.sfpu.shft2(i32 2, i32 2, i32 0, i32 0, i32 3)
  call void @llvm.tt.wh.sfpu.stochrnd.i(i32 2, i32 2, i32 0, i32 7, i32 4, i32 1)
  call void @llvm.tt.wh.sfpu.stochrnd.v(i32 2, i32 2, i32 0, i32 1, i32 5, i32 0)
  call void @llvm.tt.wh.sfpu.swap(i32 0, i32 1, i32 0, i32 1, i32 9)
  ret void
}

define void @bound_groups_trisc2() "tensix-executor"="trisc2" {
; EXPLICIT-FIXED-GROUP-LABEL: name: bound_groups_trisc2
; EXPLICIT-FIXED-GROUP: $tt_l4 = TTSFPMOVAll $tt_c9
; EXPLICIT-FIXED-GROUP-NEXT: $tt_l4 = TTSFPLUT $tt_l4, 4, {{.*}}implicit $tt_l0, implicit $tt_l1, implicit $tt_l2, implicit $tt_l3
; EXPLICIT-FIXED-GROUP-NEXT: TTSFPCONFIGC11 0, 0, {{.*}}implicit-def $tt_c11{{.*}}implicit $tt_l0
; EXPLICIT-FIXED-GROUP-NEXT: TTSFPTRANSP {{.*}}implicit-def $tt_l0, implicit-def $tt_l1, implicit-def $tt_l2, implicit-def $tt_l3, implicit-def $tt_l4, implicit-def $tt_l5, implicit-def $tt_l6, implicit-def $tt_l7
; EXPLICIT-FIXED-GROUP-NEXT: PseudoRET
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 1, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 2, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 3, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 4, i32 9)
  call void @llvm.tt.wh.sfpu.lut(i32 4, i32 4, i32 0, i32 1, i32 2, i32 3, i32 4)
  call void @llvm.tt.wh.sfpu.config.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.transp(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3)
  ret void
}

define void @bound_state() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_state
; CHECK: TTSFPENCC 3, 10, implicit-def $tt_cc
; CHECK-NEXT: TTSFPPUSHC implicit-def $tt_ccstack
; CHECK-NEXT: TTSFPSETCCEQ $tt_c9, implicit-def $tt_cc
; CHECK-NEXT: TTSFPCOMPC implicit-def $tt_cc
; CHECK-NEXT: TTSFPPOPC implicit-def $tt_cc, implicit-def $tt_ccstack
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPCONFIGReset
; CHECK-NEXT: PseudoRET
  call void @llvm.tt.wh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.wh.sfpu.pushc(i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.setcc(i32 9, i32 0, i32 6)
  call void @llvm.tt.wh.sfpu.compc()
  call void @llvm.tt.wh.sfpu.popc(i32 0, i32 0)
  call void @llvm.tt.wh.sfpu.nop()
  call void @llvm.tt.wh.sfpu.config.reset()
  ret void
}

define void @bound_dynamic_dst(i32 noundef %offset) "tensix-executor"="trisc2" {
; CHECK-LABEL: name: bound_dynamic_dst
; CHECK: $tt_l5 = TTSFPMOVAll $tt_c9
; CHECK: $tt_l5, early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSFPLOAD $tt_l5, %{{[0-9]+}}, 3, 4
; CHECK-NEXT: early-clobber %{{[0-9]+}}:gpr, early-clobber %{{[0-9]+}}:gpr = PseudoTTSFPSTORE $tt_l5, %{{[0-9]+}}, 3, 0
; CHECK-NEXT: PseudoRET
  %bounded = and i32 %offset, 1023
  call void @llvm.tt.wh.sfpu.mov.all(i32 5, i32 9)
  call void @llvm.tt.wh.sfpu.load(i32 5, i32 5, i32 %bounded, i32 3, i32 4)
  call void @llvm.tt.wh.sfpu.store(i32 5, i32 %bounded, i32 3, i32 0)
  ret void
}

; Complete the grouped/configuration forms independently of the numerical
; arithmetic sequence above. Every fixed physical def/use survives selection.
declare void @llvm.tt.wh.sfpu.mov.prng.advance(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.shft2.copy4(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.shft2.rotate4(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.lutfp32.6r(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.lutfp32.3r(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.config.lane(i32 immarg, i32 immarg, i32 immarg)

define void @bound_remaining_groups() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_remaining_groups
; CHECK: $tt_l7 = TTSFPMOVPRNGAdvance $tt_l7, {{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; CHECK-NEXT: TTSFPSHFT2Copy4 1, {{.*}}implicit-def $tt_l0, implicit-def $tt_l1, implicit-def $tt_l2, implicit-def $tt_l3
; CHECK-NEXT: TTSFPSHFT2Rotate4 $tt_l4,
; CHECK-NEXT: $tt_l5 = TTSFPLUTFP32Six $tt_l5, 6, {{.*}}implicit $tt_l0, implicit $tt_l1, implicit $tt_l2, implicit $tt_l3, implicit $tt_l4, implicit $tt_l5, implicit $tt_l6
; CHECK-NEXT: TTSFPLUTFP32Three 1, implicit-def $tt_l0, implicit-def $tt_l1, implicit-def $tt_l2, implicit-def $tt_l3, implicit-def $tt_l4, implicit-def $tt_l5, implicit-def $tt_l6, implicit-def $tt_l7
; CHECK-NEXT: TTSFPCONFIGLane 21845,
; CHECK-NEXT: PseudoRET
  call void @llvm.tt.wh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 1, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 2, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 3, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 4, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 5, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 6, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 7, i32 9)
  call void @llvm.tt.wh.sfpu.mov.prng.advance(i32 7, i32 7)
  call void @llvm.tt.wh.sfpu.shft2.copy4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 1)
  call void @llvm.tt.wh.sfpu.shft2.rotate4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 4)
  call void @llvm.tt.wh.sfpu.lutfp32.6r(i32 5, i32 5, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 6)
  call void @llvm.tt.wh.sfpu.lutfp32.3r(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 14)
  call void @llvm.tt.wh.sfpu.config.lane(i32 0, i32 21845, i32 8)
  ret void
}

; Modes 0/4 read CC for predication without defining it. Mode 2 still writes it.
define void @wh_bound_lz_cc() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: wh_bound_lz_cc
; CHECK: $tt_l0 = TTSFPLZPreserveCC $tt_l0, $tt_l1, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue{{$}}
; CHECK: $tt_l0 = TTSFPLZPreserveCC $tt_l0, $tt_l1, 4, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue{{$}}
; CHECK: $tt_l0 = TTSFPLZ $tt_l0, $tt_l1, 2, implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue{{$}}
entry:
  call void @llvm.tt.wh.sfpu.lz(i32 0, i32 0, i32 1, i32 0)
  call void @llvm.tt.wh.sfpu.lz(i32 0, i32 0, i32 1, i32 4)
  call void @llvm.tt.wh.sfpu.lz(i32 0, i32 0, i32 1, i32 2)
  ret void
}
