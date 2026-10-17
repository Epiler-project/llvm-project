; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -code-model=medium -O2 -verify-machineinstrs < %s | FileCheck %s

@selector_limit_bias = external global i64

declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b.select(i1, i32 immarg, i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma.b.select(i32 immarg, i32 immarg, i1, i32 immarg, i32 immarg)

; One source loop remains. Each selected instruction emits a local conditional
; choice; all matrix register numbers remain statically bound immediates.
; CHECK-LABEL: gemm:
; CHECK: mlbe8.m tr15,
; CHECK: mlbe8.m tr11,
; CHECK: mqma.b.mm acc7, tr0, tr15
; CHECK: mqma.b.mm acc7, tr0, tr11
; CHECK: msce32.m acc7,
; CHECK-NOT: call
; CHECK: ret
; MIR-LABEL: name: gemm
; MIR: PseudoBOSCAMELoadBSelectBound
; MIR: implicit-def $bosc_tr11, implicit-def $bosc_tr15
; MIR: PseudoBOSCAMEMMABSelectBound
; MIR: implicit $bosc_tr11, implicit $bosc_tr15

define void @gemm(ptr %a, ptr %b, ptr %c, i64 %iterations) "trex-bosc-bound-ame"="1" {
entry:
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 64)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 64)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  call void @llvm.riscv.bosc.ame.load.c(i32 7, ptr %c, i64 256)
  br label %loop
loop:
  %iteration = phi i64 [0, %entry], [%next, %loop]
  %selected = trunc i64 %iteration to i1
  call void @llvm.riscv.bosc.ame.load.a(i32 0, ptr %a, i64 64)
  call void @llvm.riscv.bosc.ame.load.b.select(i1 %selected, i32 11, i32 15, ptr %b, i64 64)
  call void @llvm.riscv.bosc.ame.mma.b.select(i32 7, i32 0, i1 %selected, i32 11, i32 15)
  %next = add i64 %iteration, 1
  %continue = icmp ult i64 %next, %iterations
  br i1 %continue, label %loop, label %exit
exit:
  call void @llvm.riscv.bosc.ame.store.c(i32 7, ptr %c, i64 256)
  ret void
}

; Guarded loads force CodeGenPrepare and SelectionDAG to rematerialize a shared
; selector. Every load/MMA pair must retain value identity through O0, O2 and
; scalar register spills; source iterations remain a single loop.
; CHECK-LABEL: guarded_stages:
; CHECK: mlbe8.m tr12,
; CHECK: mlbe8.m tr8,
; CHECK: mlbe8.m tr13,
; CHECK: mlbe8.m tr9,
; CHECK: mqma.b.mm acc0, tr0, tr12
; CHECK: mqma.b.mm acc0, tr0, tr8
; CHECK: mqma.b.mm acc1, tr0, tr13
; CHECK: mqma.b.mm acc1, tr0, tr9
; CHECK: ret
; MIR-LABEL: name: guarded_stages
; MIR: PseudoBOSCAMELoadBSelectBound
; MIR: PseudoBOSCAMELoadBSelectBound
; MIR: PseudoBOSCAMEMMABSelectBound
; MIR: PseudoBOSCAMEMMABSelectBound

define void @guarded_stages(ptr %a, ptr %b, ptr %c, i64 %iterations, i64 %limit) "trex-bosc-bound-ame"="1" {
entry:
  ; Symbolic ADDI relocation operands are not scalar integer immediates.
  %bias = load volatile i64, ptr @selector_limit_bias
  %bound = add i64 %limit, %bias
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 64)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 64)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %c, i64 512)
  call void @llvm.riscv.bosc.ame.load.c(i32 1, ptr %c, i64 512)
  br label %loop
loop:
  %iteration = phi i64 [0, %entry], [%next, %second]
  %phase = and i64 %iteration, 1
  %selected = icmp ne i64 %phase, 0
  call void @llvm.riscv.bosc.ame.load.a(i32 0, ptr %a, i64 128)
  %first_valid = icmp ult i64 %m, %bound
  br i1 %first_valid, label %first, label %exit
first:
  call void @llvm.riscv.bosc.ame.load.b.select(i1 %selected, i32 8, i32 12, ptr %b, i64 128)
  %second_valid = icmp ult i64 %n, %bound
  br i1 %second_valid, label %second, label %exit
second:
  call void @llvm.riscv.bosc.ame.load.b.select(i1 %selected, i32 9, i32 13, ptr %b, i64 128)
  call void @llvm.riscv.bosc.ame.mma.b.select(i32 0, i32 0, i1 %selected, i32 8, i32 12)
  call void @llvm.riscv.bosc.ame.mma.b.select(i32 1, i32 0, i1 %selected, i32 9, i32 13)
  %next = add i64 %iteration, 1
  %continue = icmp ult i64 %next, %iterations
  br i1 %continue, label %loop, label %exit
exit:
  call void @llvm.riscv.bosc.ame.store.c(i32 0, ptr %c, i64 512)
  ret void
}
