; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma(i32 immarg, i32 immarg, i32 immarg)

; All slots are MLIR-bound. The upper hardware slots must survive scalar
; allocation and no LLVM matrix virtual register may appear after ISel.
; CHECK-LABEL: gemm:
; CHECK: msettilem
; CHECK: msettilen
; CHECK: msettilek
; CHECK: mlae8.m tr14,
; CHECK: mlbe8.m tr15,
; CHECK: mlce32.m acc7,
; CHECK: mqma.b.mm acc7, tr14, tr15
; CHECK: msce32.m acc7,
; CHECK-NOT: call
; CHECK: ret
; MIR-LABEL: name: gemm
; MIR: BOSC_AME_SETM
; MIR: BOSC_AME_SETN
; MIR: BOSC_AME_SETK
; MIR: $bosc_tr14 = BOSC_AME_LOADA
; MIR: $bosc_tr15 = BOSC_AME_LOADB
; MIR: $bosc_acc7 = BOSC_AME_LOADC
; MIR: $bosc_acc7 = BOSC_AME_MMA $bosc_acc7, $bosc_tr14, $bosc_tr15
; MIR: BOSC_AME_STOREC $bosc_acc7
define void @gemm(ptr %a, ptr %b, ptr %c) "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 3)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  call void @llvm.riscv.bosc.ame.load.a(i32 14, ptr %a, i64 4)
  call void @llvm.riscv.bosc.ame.load.b(i32 15, ptr %b, i64 4)
  call void @llvm.riscv.bosc.ame.load.c(i32 7, ptr %c, i64 12)
  call void @llvm.riscv.bosc.ame.mma(i32 7, i32 14, i32 15)
  call void @llvm.riscv.bosc.ame.store.c(i32 7, ptr %c, i64 12)
  ret void
}
