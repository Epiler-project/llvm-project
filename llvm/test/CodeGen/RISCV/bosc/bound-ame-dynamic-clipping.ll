; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s

declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma(i32 immarg, i32 immarg, i32 immarg)

; The independent LLVM receiver proves range/alignment from real arithmetic.
; Physical matrix bindings remain fixed while scalar extent/stride use GPRs.
; CHECK-LABEL: clipped_shape_and_actual_stride:
; CHECK: msettilem
; CHECK: msettilen
; CHECK: msettilek
; CHECK: mlae8.m tr0,
; CHECK: mlbe8.m tr1,
; CHECK: mlce32.m acc0,
; CHECK: mqma.b.mm acc0, tr0, tr1
; CHECK: msce32.m acc0,
; CHECK: ret
define void @clipped_shape_and_actual_stride(ptr %a, ptr %b, ptr %c, i64 %extent) "trex-bosc-bound-ame"="1" {
  %low = and i64 %extent, 127
  %requested = add i64 %low, 1
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %requested)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %requested)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %requested)
  %stride = mul i64 %n, 4
  call void @llvm.riscv.bosc.ame.load.a(i32 0, ptr %a, i64 %k)
  call void @llvm.riscv.bosc.ame.load.b(i32 1, ptr %b, i64 %k)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %c, i64 %stride)
  call void @llvm.riscv.bosc.ame.mma(i32 0, i32 0, i32 1)
  call void @llvm.riscv.bosc.ame.store.c(i32 0, ptr %c, i64 %stride)
  ret void
}
