; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

; The synchronization pseudo has an actual TT_ISSUE dependency. Bound SFPU
; verification must receive its dedicated effect/operand contract without
; accepting other unsupported operations touching architectural state.
declare void @llvm.tt.bh.sfpu.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.coprocessor.wait()

define void @bound_wait() "tensix-executor"="trisc1" {
; CHECK-LABEL: bound_wait:
; CHECK: lui [[ADDRESS:[a-z][a-z0-9]*]], 1048192
; CHECK-NEXT: sw zero, 4([[ADDRESS]])
; CHECK-NEXT: lw [[RESPONSE:[a-z][a-z0-9]*]], 4([[ADDRESS]])
; CHECK-NEXT: mv zero, [[RESPONSE]]
  call void @llvm.tt.bh.sfpu.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.coprocessor.wait()
  ret void
}
