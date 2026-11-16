; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-bundle-unpack %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -M no-aliases %s -o - | FileCheck %s --check-prefix=ASM
;
; The kcfi module flag applies to both functions. Unpacking the independent
; scalar KCFI call must preserve the native recording in the same module.
; The exact words also detect skipped or duplicated BUNDLE children.

declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.tt.bh.sfpu.mov.all(i32 immarg, i32 immarg)

define void @native_recording() "tensix-executor"="trisc1" {
; MIR-LABEL: name: native_recording
; MIR: BUNDLE {{.*}}{
; MIR-NEXT: PseudoTTExplicitSFPUReplay 1, 1, 2, 5,
; MIR-NEXT: $tt_l0 = TTSFPMOVAll $tt_c9,
; MIR-NEXT: $tt_l1 = TTSFPMOVAll internal $tt_l0,
; MIR-NEXT: PseudoTTReplayRecordEnd
; MIR-NEXT: }
; MIR-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 2, 5,
; ASM-LABEL: native_recording:
; ASM-NOT: .word
; ASM: .word 0x1005008c
; ASM-NEXT: .word 0xf0002409
; ASM-NEXT: .word 0xf0000049
; ASM-NEXT: .word 0x10050080
; ASM-NEXT: jalr zero, 0(ra)
; ASM-NOT: .word
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 2, i32 5)
  call void @llvm.tt.bh.sfpu.mov.all(i32 0, i32 9)
  call void @llvm.tt.bh.sfpu.mov.all(i32 1, i32 0)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 2, i32 5)
  ret void
}

define void @independent_kcfi(ptr noundef %target) {
; MIR-LABEL: name: independent_kcfi
; MIR-NOT: BUNDLE
; MIR: KCFI_CHECK $x10, 12345678,
; MIR-NEXT: PseudoTAILIndirect
; MIR-NOT: BUNDLE
; ASM-LABEL: independent_kcfi:
; ASM: lw t1, -4(a0)
; ASM-NEXT: lui t2, 3014
; ASM-NEXT: addi t2, t2, 334
; ASM-NEXT: beq t1, t2, [[PASS:.Ltmp[0-9]+]]
; ASM: ebreak
; ASM: .section .kcfi_traps,"ao",@progbits,.text
; ASM: .word .Ltmp{{[0-9]+}}-.Ltmp{{[0-9]+}}
; ASM: [[PASS]]:
; ASM-NEXT: jalr zero, 0(a0)
  tail call void %target() [ "kcfi"(i32 12345678) ]
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
