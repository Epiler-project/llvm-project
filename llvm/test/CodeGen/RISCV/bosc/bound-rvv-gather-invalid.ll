; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/tail.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TAIL
; TAIL: gather index reads an uninitialized source lane
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/alias.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ALIAS
; ALIAS: gather destination must not overlap its source
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/negative.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEGATIVE
; NEGATIVE: gather index must be a constant in 0..16
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/unbounded.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNBOUNDED
; UNBOUNDED: gather index 16 requires exact VLEN=512
;--- tail.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.gather.scalar(i32 immarg, i32 immarg, i64 immarg)
define void @bad(ptr %input) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 1)
  ret void
}
;--- alias.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.gather.scalar(i32 immarg, i32 immarg, i64 immarg)
define void @bad(ptr %input) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 1, i32 1, i64 1)
  ret void
}
;--- negative.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.gather.scalar(i32 immarg, i32 immarg, i64 immarg)
define void @bad(ptr %input) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 -1)
  ret void
}
;--- unbounded.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.gather.scalar(i32 immarg, i32 immarg, i64 immarg)
define void @bad(ptr %input)  "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 16)
  ret void
}
