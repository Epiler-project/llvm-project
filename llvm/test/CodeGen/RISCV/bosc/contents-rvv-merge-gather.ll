; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs %t/positive.ll -o - | FileCheck %s --check-prefix=POS
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs %t/positive.ll -o - | FileCheck %s --check-prefix=POS
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/positive.ll -o %t/positive.O0.o
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/positive.ll -o %t/positive.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/undefined-mask.ll -o %t/undefined-mask.O0.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/undefined-mask.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/undefined-mask.ll -o %t/undefined-mask.O2.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/undefined-mask.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/short-mask.ll -o %t/short-mask.O0.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/short-mask.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/short-mask.ll -o %t/short-mask.O2.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/short-mask.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/agnostic-tail.ll -o %t/agnostic-tail.O0.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/agnostic-tail.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/agnostic-tail.ll -o %t/agnostic-tail.O2.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/agnostic-tail.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/undefined-data.ll -o %t/undefined-data.O0.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/undefined-data.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/undefined-data.ll -o %t/undefined-data.O2.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/undefined-data.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/unrelated-vl.ll -o %t/unrelated-vl.O0.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/unrelated-vl.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/unrelated-vl.ll -o %t/unrelated-vl.O2.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/unrelated-vl.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs -filetype=obj %t/non-actual-vl.ll -o %t/non-actual-vl.O0.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/non-actual-vl.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs -filetype=obj %t/non-actual-vl.ll -o %t/non-actual-vl.O2.o 2>&1 | FileCheck %s --check-prefix=INVALID
; RUN: test ! -e %t/non-actual-vl.O2.o
; INVALID: gather source lane is not defined on every incoming path

; Official integer vmerge reuses allocated contents. Active lanes require a
; defined mask and the selected source; inactive tails use the explicit
; passthru. In particular, positive-VL mask tails remain agnostic.

; POS-LABEL: merge_full:
; POS: vmflt.vv
; POS: vmerge.vvm
; POS: vrgather.vi
; POS: vse32.v
; POS-LABEL: merge_dynamic:
; POS: vmflt.vv
; POS: vmerge.vvm
; POS: vrgather.vi
; POS: vse32.v
; POS-LABEL: merge_undisturbed_tail:
; POS: vrgather.vi
; POS: vse32.v
; POS-LABEL: merge_zero:
; POS: vrgather.vi
; POS: vse32.v
; POS-LABEL: merge_constant_mask:
; POS: vrgather.vi
; POS: vse32.v
; POS-LABEL: merge_reduction:
; POS: vmflt.vv
; POS: vmerge.vvm
; POS: vfredosum.vs
; POS: vrgather.vi
; POS: vse32.v

;--- positive.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @merge_full(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}

define void @merge_dynamic(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}

define void @merge_undisturbed_tail(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %old = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %out, i64 %full)
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %one)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> %old, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %one)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 15, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %full)
  ret void
}

define void @merge_zero(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %full = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %old = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %out, i64 %full)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> %old, <vscale x 2 x i32> poison, <vscale x 2 x i32> poison, <vscale x 2 x i1> poison, i64 0)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 15, i64 %full)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %full)
  ret void
}

define void @merge_constant_mask(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> poison, <vscale x 2 x i1> zeroinitializer, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}

define void @merge_reduction(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %vl)
  %seed = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32> poison, float 0.0, i64 %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32> %seed, <vscale x 2 x i32> %merged, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %sum, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }


;--- undefined-mask.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @undefined_mask(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> poison, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }


;--- short-mask.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @short_mask(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %one)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 15, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }


;--- agnostic-tail.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @agnostic_tail(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %one = call i64 @llvm.riscv.vsetvli.i64(i64 1, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %one)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %one)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 15, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }


;--- undefined-data.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @undefined_data(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> poison, <vscale x 2 x i1> %mask, i64 %vl)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }


;--- unrelated-vl.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @unrelated_vl(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %different = call i64 @llvm.riscv.vsetvli.i64(i64 %other, i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 %different)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %different)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %different)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }


;--- non-actual-vl.ll
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare i64 @llvm.riscv.vsetvlimax.i64(i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i1>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32>, <vscale x 2 x i32>, i64 immarg, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.fsplat(<vscale x 2 x i32>, float, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.freduce.sum(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
define void @non_actual_vl(ptr %a, ptr %b, ptr %out, i64 %n, i64 %other) #0 {
  %vl = call i64 @llvm.riscv.vsetvlimax.i64(i64 2, i64 0)
  %av = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %a, i64 %vl)
  %bv = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %b, i64 %vl)
  %mask = call <vscale x 2 x i1> @llvm.riscv.bosc.rvv.contents.fcmp.lt(<vscale x 2 x i1> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, i64 %vl)
  %merged = call <vscale x 2 x i32> @llvm.riscv.vmerge.nxv2i32.nxv2i32(<vscale x 2 x i32> poison, <vscale x 2 x i32> %av, <vscale x 2 x i32> %bv, <vscale x 2 x i1> %mask, i64 1)
  %g = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.gather.scalar(<vscale x 2 x i32> poison, <vscale x 2 x i32> %merged, i64 0, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %g, ptr %out, i64 %vl)
  ret void
}
attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }
