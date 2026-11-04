; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -verify-machineinstrs %t/exact32.ll -o %t/O0.s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O2 -verify-machineinstrs %t/exact32.ll -o %t/O2.s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -regalloc=fast -verify-machineinstrs %t/exact32.ll -o %t/fast.s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -verify-machineinstrs -filetype=obj %t/exact32.ll -o %t/O0.o
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O2 -verify-machineinstrs -filetype=obj %t/exact32.ll -o %t/O2.o
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -regalloc=fast -verify-machineinstrs -filetype=obj %t/exact32.ll -o %t/fast.o
; RUN: %python %t/check.py %t/O0.s %t/O2.s %t/fast.s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -verify-machineinstrs -filetype=obj %t/overflow33.ll -o %t/overflow.O0.o 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: test ! -e %t/overflow.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O2 -verify-machineinstrs -filetype=obj %t/overflow33.ll -o %t/overflow.O2.o 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: test ! -e %t/overflow.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -regalloc=fast -verify-machineinstrs -filetype=obj %t/overflow33.ll -o %t/overflow.fast.o 2>&1 | FileCheck %s --check-prefix=OVERFLOW
; RUN: test ! -e %t/overflow.fast.o
;
; All 32 ordered loads remain live when SFU consumes the dying x0. Its poison
; passthru must not occupy a 33rd register: InitUndef must be followed by
; ProcessImplicitDefs before PHI elimination and two-address conversion.
; Inspect the physical flow instead of accepting hidden copies or spills.
; The companion 33-value case remains a genuine capacity error in every mode.
;
; OVERFLOW: ran out of registers during register allocation in function 'minimal_sfu_33'

;--- exact32.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

define void @minimal_sfu_32(ptr %base) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %x0 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x1 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x2 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x3 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x4 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x5 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x6 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x7 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x8 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x9 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x10 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x11 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x12 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x13 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x14 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x15 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x16 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x17 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x18 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x19 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x20 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x21 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x22 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x23 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x24 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x25 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x26 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x27 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x28 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x29 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x30 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x31 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %y = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> %x0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %y, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x1, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x2, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x3, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x4, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x5, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x6, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x7, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x8, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x9, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x10, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x11, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x12, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x13, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x14, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x15, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x16, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x17, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x18, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x19, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x20, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x21, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x22, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x23, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x24, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x25, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x26, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x27, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x28, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x29, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x30, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x31, ptr %base, i64 %vl)
  ret void
}

;--- overflow33.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

define void @minimal_sfu_33(ptr %base) "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 16, i64 2, i64 0)
  %x0 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x1 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x2 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x3 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x4 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x5 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x6 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x7 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x8 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x9 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x10 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x11 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x12 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x13 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x14 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x15 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x16 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x17 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x18 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x19 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x20 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x21 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x22 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x23 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x24 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x25 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x26 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x27 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x28 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x29 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x30 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x31 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %x32 = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %base, i64 %vl)
  %y = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> %x0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %y, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x1, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x2, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x3, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x4, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x5, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x6, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x7, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x8, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x9, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x10, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x11, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x12, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x13, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x14, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x15, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x16, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x17, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x18, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x19, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x20, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x21, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x22, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x23, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x24, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x25, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x26, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x27, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x28, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x29, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x30, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x31, ptr %base, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %x32, ptr %base, i64 %vl)
  ret void
}

;--- check.py
from pathlib import Path
import re
import sys


def check(path):
    text = Path(path).read_text()
    assert len(re.findall(r"^minimal_sfu_32:", text, re.M)) == 1, path
    instructions = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        assert not re.match(r"\.(?:insn|word|long|short|byte|[248]byte|quad|zero|space|fill)\b", line), (path, "encoded instruction or data", line)
        if not line or line.startswith(".") or line.endswith(":"):
            continue
        instructions.append(line)

    # Scalar configuration is allowed; stack traffic, helpers, branches, and
    # every vector operation other than the exact contents flow are rejected.
    allowed = {"vsetivli", "vsetvli", "vle32.v", "vse32.v", "vsfu.exp2.v",
               "li", "mv", "addi", "ret"}
    flow = []
    for instruction in instructions:
        opcode = instruction.split()[0]
        assert opcode in allowed, (path, "unexpected transfer or helper", instruction)
        assert not re.search(r"\bsp\b", instruction), (path, "stack use", instruction)
        if opcode in {"vle32.v", "vse32.v", "vsfu.exp2.v"}:
            flow.append(instruction)
    assert len(flow) == 65, (path, "exactly 32 loads, one SFU, and 32 stores", flow)
    assert sum(line == "ret" for line in instructions) == 1, (path, "one return")

    loads = []
    for instruction in flow[:32]:
        match = re.fullmatch(r"vle32\.v\s+(v[0-9]+),\s*\(a0\)", instruction)
        assert match, (path, "32 ordered unmasked input loads before SFU", instruction)
        loads.append(match[1])
    assert set(loads) == {"v" + str(i) for i in range(32)}, (path, "32 distinct resident VRs", loads)
    match = re.fullmatch(r"vsfu\.exp2\.v\s+(v[0-9]+),\s*(v[0-9]+)", flow[32])
    assert match and match[1] == match[2] == loads[0], (path, "SFU must update dying x0 in place", flow[32])
    stores = []
    for instruction in flow[33:]:
        match = re.fullmatch(r"vse32\.v\s+(v[0-9]+),\s*\(a0\)", instruction)
        assert match, (path, "32 ordered unmasked stores after SFU", instruction)
        stores.append(match[1])
    assert stores == loads, (path, "all resident payloads must reach their stores", loads, stores)


for filename in sys.argv[1:]:
    check(filename)
