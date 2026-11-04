; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s

declare void @llvm.riscv.bosc.ame.setfp8.e4m3()
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.fp.mma(i32 immarg, i32 immarg, i32 immarg)

; The independent LLVM receiver proves range/alignment from real arithmetic.
; Physical matrix bindings remain fixed while scalar extent/stride use GPRs.
; CHECK-LABEL: fp8_e4m3:
; CHECK: msetfp8.e4m3
; CHECK: msettilem
; CHECK: msettilen
; CHECK: msettilek
; CHECK: mlae8.m tr0,
; CHECK: mlbe8.m tr1,
; CHECK: mlce32.m acc0,
; CHECK: mfqma.cf.mm acc0, tr0, tr1
; CHECK: msce32.m acc0,
; CHECK: ret
define void @fp8_e4m3(ptr %a, ptr %b, ptr %c, i64 %extent, i64 %padding) "trex-bosc-bound-ame"="1" {
  %low = and i64 %extent, 63
  %requested = add i64 %low, 1
  %pad = and i64 %padding, 252
  %stride = add i64 %pad, 256
  call void @llvm.riscv.bosc.ame.setfp8.e4m3()
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %requested)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %requested)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %requested)
  call void @llvm.riscv.bosc.ame.load.a(i32 0, ptr %a, i64 64)
  call void @llvm.riscv.bosc.ame.load.b(i32 1, ptr %b, i64 64)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %c, i64 %stride)
  call void @llvm.riscv.bosc.ame.fp.mma(i32 0, i32 0, i32 1)
  call void @llvm.riscv.bosc.ame.store.c(i32 0, ptr %c, i64 %stride)
  ret void
}
