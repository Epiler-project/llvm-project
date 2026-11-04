; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/mismatch.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCH
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/alias.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ALIAS
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/role.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCH
; MISMATCH: selected MMA requires initialized C/A and the same B selector
; ALIAS: selected B stages must bind distinct registers

;--- mismatch.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b.select(i1, i32 immarg, i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma.b.select(i32 immarg, i32 immarg, i1, i32 immarg, i32 immarg)

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
  %other = xor i1 %selected, true
  call void @llvm.riscv.bosc.ame.mma.b.select(i32 7, i32 0, i1 %other, i32 11, i32 15)
  %next = add i64 %iteration, 1
  %continue = icmp ult i64 %next, %iterations
  br i1 %continue, label %loop, label %exit
exit:
  call void @llvm.riscv.bosc.ame.store.c(i32 7, ptr %c, i64 256)
  ret void
}

;--- alias.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b.select(i1, i32 immarg, i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma.b.select(i32 immarg, i32 immarg, i1, i32 immarg, i32 immarg)

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
  call void @llvm.riscv.bosc.ame.load.b.select(i1 %selected, i32 11, i32 11, ptr %b, i64 64)
  call void @llvm.riscv.bosc.ame.mma.b.select(i32 7, i32 0, i1 %selected, i32 11, i32 11)
  %next = add i64 %iteration, 1
  %continue = icmp ult i64 %next, %iterations
  br i1 %continue, label %loop, label %exit
exit:
  call void @llvm.riscv.bosc.ame.store.c(i32 7, ptr %c, i64 256)
  ret void
}

;--- role.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b.select(i1, i32 immarg, i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.mma.b.select(i32 immarg, i32 immarg, i1, i32 immarg, i32 immarg)

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
  call void @llvm.riscv.bosc.ame.load.a(i32 11, ptr %a, i64 64)
  call void @llvm.riscv.bosc.ame.mma.b.select(i32 7, i32 0, i1 %selected, i32 11, i32 15)
  %next = add i64 %iteration, 1
  %continue = icmp ult i64 %next, %iterations
  br i1 %continue, label %loop, label %exit
exit:
  call void @llvm.riscv.bosc.ame.store.c(i32 7, ptr %c, i64 256)
  ret void
}
