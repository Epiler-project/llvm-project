; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/undefined-seed.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/undefined-seed.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/undefined-data.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/undefined-data.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/undefined-tail.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/undefined-tail.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/short-data.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/short-data.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/zero-poison.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/zero-poison.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/agnostic-passthru.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/agnostic-passthru.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/unrelated-vl.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/unrelated-vl.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/growing-vl.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/growing-vl.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: gather source lane is not defined on every incoming path

; A reduction initializes lane zero only, and only from an initialized
; scalar seed plus every actually-read source lane. Positive-VL tails remain
; agnostic; VL=0 preserves the old destination without initializing anything.

;--- undefined-seed.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @undefined_seed(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %full)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> poison, <vscale x 2 x i32> %data, i64 %full)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  ret void
}

;--- undefined-data.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @undefined_data(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %full)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> poison, i64 %full)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  ret void
}

;--- undefined-tail.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @undefined_tail(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %full)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %full)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %data, i64 %full)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 1, i64 %full)
  ret void
}

;--- short-data.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @short_data(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %one)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %one)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %data, i64 %full)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  ret void
}

;--- zero-poison.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @zero_poison(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> poison, <vscale x 2 x i32> poison, i64 0)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  ret void
}

;--- agnostic-passthru.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @agnostic_passthru(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %full)
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> %initial, ptr %input, i64 %one)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %data, i64 %full)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  ret void
}

;--- unrelated-vl.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @unrelated_vl(ptr %input, i64 %m, i64 %n) "trex-bosc-contents-rvv"="1" {
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %one)
  %inputvl = call i64 @llvm.riscv.vsetvli.i64(i64 %m, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %inputvl)
  %reducevl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %data, i64 %reducevl)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  ret void
}

;--- growing-vl.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
; The carried input was loaded under the previous occurrence of the same SSA
; VL instruction. Its shorter prefix does not cover this iteration's VL.
define void @growing_vl(ptr %input, i64 %count) "trex-bosc-contents-rvv"="1" {
entry:
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %initial = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %full)
  br label %loop
loop:
  %i = phi i64 [ 1, %entry ], [ %next, %loop ]
  %carried = phi <vscale x 2 x i32> [ %initial, %entry ], [ %data, %loop ]
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %i, i64 2, i64 0)
  %data = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %initial, <vscale x 2 x i32> %carried, i64 %vl)
  %broadcast = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %full)
  %next = add i64 %i, 1
  %again = icmp ult i64 %next, %count
  br i1 %again, label %loop, label %exit
exit:
  ret void
}
