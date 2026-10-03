; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=PseudoTTBound --implicit-check-not=PseudoCALL --implicit-check-not=PseudoTTSFPUReplay
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=PseudoTTBound --implicit-check-not=PseudoCALL --implicit-check-not=PseudoTTSFPUReplay
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NATIVE --implicit-check-not=call --implicit-check-not=fence
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=NATIVE --implicit-check-not=call --implicit-check-not=fence
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; SFPMOV FROM_SPECIAL (exact mode 8), special selector 9, returns each active
; lane's old PRNG word and advances that lane once. This is a stateful action,
; even when no later operation consumes the sampled LReg. The special selector
; is an encoding constant, not CReg9 or a numerical SFPR source operand.
;
; Only destination and inactive-lane passthrough old are authored operands.
; These lowering checks deliberately do not author a seed/configuration proof;
; LLVM must not require a general user-program PRNG initializedness analysis
; to emit a legal native instruction. They are not execution/numerical checks.

declare void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 immarg, i32 immarg)

; raw SFPMOV(L0,special9,mode8) = 0x7c000908; rol32(raw,2) = 0xf0002421.
; raw SFPMOV(L7,special9,mode8) = 0x7c000978; rol32(raw,2) = 0xf00025e1.
define void @sample_prng_exact_operands() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: sample_prng_exact_operands
; ISEL: registers: []
; ISEL: $tt_l0 = TTSFPMOVPRNGAdvance $tt_l0, implicit-def $tt_issue, implicit-def $tt_prng, implicit $tt_cc, implicit $tt_config, implicit $tt_issue, implicit $tt_prng{{$}}
; ISEL-NEXT: $tt_l7 = TTSFPMOVPRNGAdvance $tt_l7, implicit-def $tt_issue, implicit-def $tt_prng, implicit $tt_cc, implicit $tt_config, implicit $tt_issue, implicit $tt_prng{{$}}
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: sample_prng_exact_operands:
; NATIVE-NOT: .word
; NATIVE: .word 0xf0002421
; NATIVE-NEXT: .word 0xf00025e1
; NATIVE-NEXT: ret
; NATIVE-NOT: .word
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 7, i32 7)
  ret void
}

; Repeated samples are two advances. Neither physical-copy cleanup nor dead
; numerical-result elimination may delete or combine these authored actions.
define void @unused_prng_samples_still_advance() "tensix-executor"="trisc0" {
; ISEL-LABEL: name: unused_prng_samples_still_advance
; ISEL: registers: []
; ISEL: $tt_l0 = TTSFPMOVPRNGAdvance $tt_l0,
; ISEL-NEXT: $tt_l0 = TTSFPMOVPRNGAdvance $tt_l0,
; ISEL-NEXT: PseudoRET
; NATIVE-LABEL: unused_prng_samples_still_advance:
; NATIVE-NOT: .word
; NATIVE: .word 0xf0002421
; NATIVE-NEXT: .word 0xf0002421
; NATIVE-NEXT: ret
; NATIVE-NOT: .word
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 0)
  ret void
}

; Three dynamic executions retain one static sample and the authored backedge.
; No loop-body duplication and no implicit replay mining of a stateful sample.
define void @prng_three_iterations() "tensix-executor"="trisc2" {
; ISEL-LABEL: name: prng_three_iterations
; ISEL-NOT: TTSFPMOVPRNGAdvance
; ISEL: bb.1.loop:
; ISEL: $tt_l0 = TTSFPMOVPRNGAdvance $tt_l0,
; ISEL-NOT: TTSFPMOVPRNGAdvance
; ISEL: BNE {{.*}}%bb.1
; ISEL-NOT: TTSFPMOVPRNGAdvance
; ISEL: PseudoRET
; NATIVE-LABEL: prng_three_iterations:
; NATIVE-NOT: .word
; NATIVE: .word 0xf0002421
; NATIVE-NOT: .word
; NATIVE: bne{{[a-z]*}} {{.*}}.LBB
; NATIVE-NOT: .word
; NATIVE: ret
; NATIVE-NOT: .word
entry:
  br label %loop
loop:
  %count = phi i32 [ 3, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.bound.sfpmov.prng.advance(i32 0, i32 0)
  %next = add i32 %count, -1
  %again = icmp ne i32 %next, 0
  br i1 %again, label %loop, label %exit
exit:
  ret void
}
