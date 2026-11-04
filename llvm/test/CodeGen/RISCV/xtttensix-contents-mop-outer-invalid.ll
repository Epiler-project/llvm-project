; RUN: split-file %s %t
; ZERO-OUTER: contents MOP profile requires outer and inner in [1, 127]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/zero_outer.ll -o %t/zero_outer-bh-O0.o 2>&1 | FileCheck %s --check-prefix=ZERO-OUTER
; RUN: not test -e %t/zero_outer-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/zero_outer.ll -o %t/zero_outer-bh-O2.o 2>&1 | FileCheck %s --check-prefix=ZERO-OUTER
; RUN: not test -e %t/zero_outer-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/zero_outer.ll -o %t/zero_outer-wh-O0.o 2>&1 | FileCheck %s --check-prefix=ZERO-OUTER
; RUN: not test -e %t/zero_outer-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/zero_outer.ll -o %t/zero_outer-wh-O2.o 2>&1 | FileCheck %s --check-prefix=ZERO-OUTER
; RUN: not test -e %t/zero_outer-wh-O2.o
; WIDE-OUTER: contents MOP profile requires outer and inner in [1, 127]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/wide_outer.ll -o %t/wide_outer-bh-O0.o 2>&1 | FileCheck %s --check-prefix=WIDE-OUTER
; RUN: not test -e %t/wide_outer-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/wide_outer.ll -o %t/wide_outer-bh-O2.o 2>&1 | FileCheck %s --check-prefix=WIDE-OUTER
; RUN: not test -e %t/wide_outer-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wide_outer.ll -o %t/wide_outer-wh-O0.o 2>&1 | FileCheck %s --check-prefix=WIDE-OUTER
; RUN: not test -e %t/wide_outer-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wide_outer.ll -o %t/wide_outer-wh-O2.o 2>&1 | FileCheck %s --check-prefix=WIDE-OUTER
; RUN: not test -e %t/wide_outer-wh-O2.o
; SLOT8-NO-RECORD: contents MOP binding requires a dominating record
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/slot8_no_record.ll -o %t/slot8_no_record-bh-O0.o 2>&1 | FileCheck %s --check-prefix=SLOT8-NO-RECORD
; RUN: not test -e %t/slot8_no_record-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/slot8_no_record.ll -o %t/slot8_no_record-bh-O2.o 2>&1 | FileCheck %s --check-prefix=SLOT8-NO-RECORD
; RUN: not test -e %t/slot8_no_record-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/slot8_no_record.ll -o %t/slot8_no_record-wh-O0.o 2>&1 | FileCheck %s --check-prefix=SLOT8-NO-RECORD
; RUN: not test -e %t/slot8_no_record-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/slot8_no_record.ll -o %t/slot8_no_record-wh-O2.o 2>&1 | FileCheck %s --check-prefix=SLOT8-NO-RECORD
; RUN: not test -e %t/slot8_no_record-wh-O2.o
; DUPLICATE-SLOT8: contents MOP profile requires unique setup
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/duplicate_slot8.ll -o %t/duplicate_slot8-bh-O0.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SLOT8
; RUN: not test -e %t/duplicate_slot8-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/duplicate_slot8.ll -o %t/duplicate_slot8-bh-O2.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SLOT8
; RUN: not test -e %t/duplicate_slot8-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/duplicate_slot8.ll -o %t/duplicate_slot8-wh-O0.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SLOT8
; RUN: not test -e %t/duplicate_slot8-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/duplicate_slot8.ll -o %t/duplicate_slot8-wh-O2.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SLOT8
; RUN: not test -e %t/duplicate_slot8-wh-O2.o

;--- zero_outer.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

define void @contents_mop_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 0)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}


;--- wide_outer.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

define void @contents_mop_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 128)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}


;--- slot8_no_record.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

define void @contents_mop_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 72, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}


;--- duplicate_slot8.ll
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

define void @contents_mop_outer2_inner4(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 2)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
