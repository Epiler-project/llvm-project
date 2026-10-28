; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=O0
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL

declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.seminit(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.semget(i32 immarg)
declare void @llvm.riscv.tt.sempost(i32 immarg)
declare void @llvm.riscv.tt.semwait(i32 immarg, i32 immarg, i32 immarg)

define void @direct() "tensix-executor"="trisc1" {
; CHECK-LABEL: direct:
; CHECK: .word 0x08000000
; CHECK: .word 0x8bfffffe
; CHECK: .word 0x8ffc0ff2
; CHECK: .word 0x94000ff2
; CHECK: .word 0x90000ff2
; CHECK: .word 0x9bfe0ffe
; CHECK-NOT: call
; CHECK: ret
; O0-LABEL: direct:
; O0: .word 0x08000000
; O0: .word 0x9bfe0ffe
; ISEL-LABEL: name: direct
; ISEL: TTNOP
; ISEL: TTSTALLWAIT 32767, 511
; ISEL: TTSEMINIT 255, 15, 15
; ISEL: TTSEMGET 255
; ISEL: TTSEMPOST 255
; ISEL: TTSEMWAIT 3, 255, 511
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.stallwait(i32 32767, i32 511)
  call void @llvm.riscv.tt.seminit(i32 255, i32 15, i32 15)
  call void @llvm.riscv.tt.semget(i32 255)
  call void @llvm.riscv.tt.sempost(i32 255)
  call void @llvm.riscv.tt.semwait(i32 3, i32 255, i32 511)
  ret void
}

define void @retained_loop(i32 %limit) "tensix-executor"="trisc0" {
; CHECK-LABEL: retained_loop:
; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: .word 0x90000012
; CHECK: bne{{z?}} {{.*}}, .LBB{{[0-9]+}}_1
; CHECK: ret
entry:
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.sempost(i32 1)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
