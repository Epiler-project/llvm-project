; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/shift.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SHIFT
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/index.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDEX
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/tail.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LANE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 %t/tail.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LANE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LANE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 %t/arbitrary-vl.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VL
; SHIFT: e32 shift amount must be in 0..31
; INDEX: gather index must be a constant in 0..16
; WIDTH: gather index 16 requires exact VLEN=512
; LANE: gather source lane is not defined on every incoming path
; VL: RVV contents require an actual e32,m1 VL result or literal zero

;--- shift.ll
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.shl(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @shift() "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.shl(<vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer, i64 32, i64 0)
  ret void
}

;--- index.ll
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @index() "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer, i64 17, i64 0)
  ret void
}

;--- width.ll
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @width() "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> poison, i64 16, i64 0)
  ret void
}

;--- tail.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @tail(ptr %p) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %source = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %source, i64 15, i64 %vl)
  ret void
}

;--- join.ll
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
define void @join(ptr %p, i1 %choose) "trex-bosc-contents-rvv"="1" {
entry:
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  br i1 %choose, label %good, label %bad
good:
  %source = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %p, i64 %vl)
  br label %join
bad:
  br label %join
join:
  %incoming = phi <vscale x 2 x i32> [ %source, %good ], [ poison, %bad ]
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %incoming, i64 15, i64 %vl)
  ret void
}

;--- arbitrary-vl.ll
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.add(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @arbitrary(i64 %n) "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.add(<vscale x 2 x i32> poison, <vscale x 2 x i32> zeroinitializer, <vscale x 2 x i32> zeroinitializer, i64 %n)
  ret void
}
