; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/seed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SEED
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STORE
; SEED: scalar lane read requires initialized lane zero
; STORE: covering the active setvl lanes
;--- seed.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.freduce.sum(i32 immarg, i32 immarg)
define void @bad_seed(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 5, ptr %p)
  call void @llvm.riscv.bosc.freduce.sum(i32 4, i32 5)
  ret void
}
;--- store.ll
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.fsplat(i32 immarg, float)
declare void @llvm.riscv.bosc.freduce.max(i32 immarg, i32 immarg)
define void @bad_store(ptr %p) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.fsplat(i32 4, float 0.0)
  call void @llvm.riscv.bosc.load(i32 5, ptr %p)
  call void @llvm.riscv.bosc.freduce.max(i32 4, i32 5)
  call void @llvm.riscv.bosc.store(i32 4, ptr %p)
  ret void
}
