; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/arbitrary.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/wrong-sew.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/wrong-lmul.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu %t/sfu.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/observe.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/phi-bad.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/phi-poison.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/arithmetic.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: RVV contents require an actual e32,m1 VL result or literal zero

;--- arbitrary.ll
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
define void @arbitrary(ptr %p, i64 %avl) "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %avl)
  ret void
}

;--- wrong-sew.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
define void @wrong_sew(ptr %p, i64 %n) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 0, i64 0)
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  ret void
}

;--- wrong-lmul.ll
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
define void @wrong_lmul(ptr %p) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 1)
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  ret void
}

;--- sfu.ll
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @sfu(i64 %avl) "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer, i64 %avl)
  ret void
}

;--- observe.ll
declare i64 @llvm.riscv.bosc.read.bits.contents(ptr, i64, i32 immarg)
define i64 @observe(ptr %p, i64 %unverified_restore) "trex-bosc-contents-rvv"="1" {
  %bits = call i64 @llvm.riscv.bosc.read.bits.contents(ptr %p, i64 %unverified_restore, i32 32)
  ret i64 %bits
}

;--- phi-bad.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
define void @phi_bad(ptr %p, i64 %n, i64 %bad, i1 %choose) "trex-bosc-contents-rvv"="1" {
entry:
  br i1 %choose, label %good, label %badpath
good:
  %actual = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  br label %join
badpath:
  br label %join
join:
  %vl = phi i64 [ %actual, %good ], [ %bad, %badpath ]
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  ret void
}

;--- phi-poison.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
define void @phi_poison(ptr %p, i64 %n, i1 %choose) "trex-bosc-contents-rvv"="1" {
entry:
  br i1 %choose, label %good, label %badpath
good:
  %actual = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  br label %join
badpath:
  br label %join
join:
  %vl = phi i64 [ %actual, %good ], [ poison, %badpath ]
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  ret void
}

;--- arithmetic.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
define void @arithmetic(ptr %p, i64 %n) "trex-bosc-contents-rvv"="1" {
  %actual = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %not_actual = add i64 %actual, 1
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %not_actual)
  ret void
}
