; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/late.ll -o %t/late.bh.O0.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/late.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/loop.ll -o %t/loop.bh.O0.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/loop.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/branch.ll -o %t/branch.bh.O0.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/branch.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/selector.ll -o %t/selector.bh.O0.o 2>&1 | FileCheck %s --check-prefix=SIGNATURE
; RUN: test ! -s %t/selector.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/old-live.ll -o %t/old-live.bh.O0.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/old-live.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/pressure.bh.O0.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/pressure.bh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/late.ll -o %t/late.bh.O2.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/late.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/loop.ll -o %t/loop.bh.O2.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/loop.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/branch.ll -o %t/branch.bh.O2.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/branch.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/selector.ll -o %t/selector.bh.O2.o 2>&1 | FileCheck %s --check-prefix=SIGNATURE
; RUN: test ! -s %t/selector.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/old-live.ll -o %t/old-live.bh.O2.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/old-live.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/pressure.bh.O2.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/pressure.bh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/late.ll -o %t/late.wh.O0.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/late.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/loop.ll -o %t/loop.wh.O0.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/loop.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/branch.ll -o %t/branch.wh.O0.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/branch.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/selector.ll -o %t/selector.wh.O0.o 2>&1 | FileCheck %s --check-prefix=SIGNATURE
; RUN: test ! -s %t/selector.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/old-live.ll -o %t/old-live.wh.O0.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/old-live.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/pressure.wh.O0.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/pressure.wh.O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/late.ll -o %t/late.wh.O2.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/late.wh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/loop.ll -o %t/loop.wh.O2.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/loop.wh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/branch.ll -o %t/branch.wh.O2.o 2>&1 | FileCheck %s --check-prefix=PREFIX
; RUN: test ! -e %t/branch.wh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/selector.ll -o %t/selector.wh.O2.o 2>&1 | FileCheck %s --check-prefix=SIGNATURE
; RUN: test ! -s %t/selector.wh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/old-live.ll -o %t/old-live.wh.O2.o 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: test ! -e %t/old-live.wh.O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/pressure.ll -o %t/pressure.wh.O2.o 2>&1 | FileCheck %s --check-prefix=PRESSURE
; RUN: test ! -e %t/pressure.wh.O2.o

; A generated-ABI parser error can leave an empty output path, but publishes
; no object bytes. Semantic codegen rejections below must remove their output.
;
; Entry observations are confined to the function entry prefix before actual
; numerical work. They cannot re-seed after a write, in a loop or on a branch,
; accept a selector, evade destructive ties, or supply a ninth ordinary LReg.
; PREFIX: contents entry requires the function entry prefix
; SIGNATURE: intrinsic has incorrect number of args. Expected 0, but got 1
; COPY: Tensix SFPU cannot synthesize a register copy
; PRESSURE: {{ran out of registers during register allocation|Tensix SFPU registers cannot spill or reload}}

;--- late.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @entry_after_numerical_write() "tensix-executor"="trisc1" {
  %first = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %first, <32 x i32> %first, <32 x i32> %first, i32 0)
  %late = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %late, i32 2, i32 0, i32 3)
  ret void
}

;--- loop.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @entry_recreated_inside_loop(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  br label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %reseed = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %reseed, i32 0, i32 0, i32 3)
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- branch.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @entry_on_only_one_predecessor(i1 %choose) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  br i1 %choose, label %observe, label %exit
observe:
  %branch_entry = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %branch_entry, i32 0, i32 0, i32 3)
  br label %exit
exit:
  ret void
}

;--- selector.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry(i32)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @entry_with_physical_selector() "tensix-executor"="trisc1" {
  %selected = call <32 x i32> @llvm.riscv.tt.contents.entry(i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %selected, i32 0, i32 0, i32 3)
  ret void
}

;--- old-live.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @observed_old_still_live_after_tied_write() "tensix-executor"="trisc1" {
  %old = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %rhs = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %old, <32 x i32> %old, <32 x i32> %rhs, i32 0)
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 2, i32 0, i32 3)
  ret void
}

;--- pressure.ll
declare <32 x i32> @llvm.riscv.tt.contents.entry()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @nine_simultaneous_entry_liveins() "tensix-executor"="trisc1" {
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  %v8 = call <32 x i32> @llvm.riscv.tt.contents.entry()
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v1, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v2, i32 4, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v3, i32 6, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v4, i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v5, i32 10, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v6, i32 12, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v7, i32 14, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v8, i32 16, i32 0, i32 3)
  ret void
}
