; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

declare void @llvm.riscv.tt.nop.mop(i32 immarg)
declare void @llvm.riscv.tt.mvmul.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)

; Slot programming must not execute its instruction. In particular, an MVMUL
; slot becomes a configuration SW, and the REPLAY reference may be configured
; before its record is materialized. Referencing a subrange is intentional.
define void @mop_template() "tensix-executor"="trisc1" {
; CHECK-LABEL: mop_template:
; CHECK: lui [[ADDR:[a-z0-9]+]], 1047424
; CHECK: sw {{[a-z0-9]+}}, 8([[ADDR]])
; CHECK: sw
; CHECK: .word 0x10000084
; CHECK: .word 0x08000000
; CHECK: .word 0x08000000
; CHECK: .word 0x06000000
; CHECK: ret
; ISEL-LABEL: name: mop_template
; ISEL: PseudoTTNOPMop 2
; ISEL: PseudoTTMVMULMop 3, 0, 0, 0, 0
; ISEL: PseudoTTREPLAYMop 7, 0, 0, 1, 1
; ISEL: PseudoTTReplayRecordEnd
  call void @llvm.riscv.tt.nop.mop(i32 2)
  call void @llvm.riscv.tt.mvmul.mop(i32 3, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.mop(i32 7, i32 0, i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.nop.mop(i32 8)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 2, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ret void
}
