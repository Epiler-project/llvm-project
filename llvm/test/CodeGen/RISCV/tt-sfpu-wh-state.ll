; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=PseudoTTBound --implicit-check-not=PseudoCALL --implicit-check-not=':sfpr'
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=PseudoTTBound --implicit-check-not=PseudoCALL --implicit-check-not=':sfpr'
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.tt.wh.sfpu.encc(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mov.all(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.mov.prng.advance(i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.stochrnd.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.load(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)

; VC=L1, VB=L2, VD=L0, mode=4. WH round is bit 21, so the raw
; nearest/stochastic words 0x8e002104/0x8e202104 rotate to these native words.
; Both operations read and advance PRNG, even though nearest is deterministic.
define void @vector_rounding() "tensix-executor"="trisc1" {
; MIR-LABEL: name: vector_rounding
; MIR: $tt_l0 = TTSFPSTOCHRNDV $tt_l0, $tt_l1, $tt_l2, 4, 0, {{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; MIR-NEXT: $tt_l0 = TTSFPSTOCHRNDV $tt_l0, $tt_l1, $tt_l2, 4, 1, {{.*}}implicit-def $tt_prng{{.*}}implicit $tt_prng
; ASM-LABEL: vector_rounding:
; ASM: .word 0x38008412
; ASM: .word 0x38808412
  call void @llvm.tt.wh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 1, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 2, i32 9)
  call void @llvm.tt.wh.sfpu.stochrnd.v(i32 0, i32 0, i32 1, i32 2, i32 4, i32 0)
  call void @llvm.tt.wh.sfpu.stochrnd.v(i32 0, i32 0, i32 1, i32 2, i32 4, i32 1)
  ret void
}

; The SCEV-bounded offset and the authored backedge reach the WH receiver.
; One static load/store and one state advance remain in the loop; no copies of
; iterations, SFPR virtual registers or implicit replay records are permitted.
define void @dynamic_dst_loop() "tensix-executor"="trisc2" {
; MIR-LABEL: name: dynamic_dst_loop
; MIR: bb.1.loop:
; MIR: PseudoTTSFPLOAD $tt_l0, %{{[0-9]+}}, 3, 4
; MIR: PseudoTTSFPSTORE $tt_l0, %{{[0-9]+}}, 3, 4
; MIR: $tt_l1 = TTSFPMOVPRNGAdvance $tt_l1,
; MIR-NOT: PseudoTTSFPLOAD
; MIR-NOT: PseudoTTSFPSTORE
; MIR-NOT: TTSFPMOVPRNGAdvance
; MIR: BNE {{.*}}%bb.1
; MIR-NOT: PseudoTTSFPLOAD
; MIR-NOT: PseudoTTSFPSTORE
; MIR-NOT: TTSFPMOVPRNGAdvance
; MIR: PseudoRET
; ASM-LABEL: dynamic_dst_loop:
; ASM: .word 0xf0002461
; ASM-NOT: .word 0xf0002461
; ASM: bne{{[a-z]*}} {{.*}}.LBB
; ASM-NOT: .word 0xf0002461
; ASM: ret
entry:
  call void @llvm.tt.wh.sfpu.encc(i32 3, i32 10)
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.wh.sfpu.mov.all(i32 1, i32 9)
  br label %loop
loop:
  %count = phi i32 [ 3, %entry ], [ %next, %loop ]
  call void @llvm.tt.wh.sfpu.load(i32 0, i32 0, i32 %count, i32 3, i32 4)
  call void @llvm.tt.wh.sfpu.store(i32 0, i32 %count, i32 3, i32 4)
  call void @llvm.tt.wh.sfpu.mov.prng.advance(i32 1, i32 1)
  %next = add i32 %count, -1
  %again = icmp ne i32 %next, 0
  br i1 %again, label %loop, label %exit
exit:
  ret void
}
