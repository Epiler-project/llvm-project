; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/uninitialized.ll -o %t/uninitialized.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/uninitialized.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/half-mask.ll -o %t/half-mask.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/half-mask.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/odd-only-mask.ll -o %t/odd-only-mask.o 2>&1 | FileCheck %s --check-prefix=MASK
; RUN: test ! -e %t/odd-only-mask.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/zero-mask.ll -o %t/zero-mask.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/zero-mask.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/one-branch.ll -o %t/one-branch.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/one-branch.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/backedge-only.ll -o %t/backedge-only.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/backedge-only.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/unknown-store.ll -o %t/unknown-store.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/unknown-store.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/unknown-cc.ll -o %t/unknown-cc.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/unknown-cc.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/reset-only.ll -o %t/reset-only.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/reset-only.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/raw-recorded-config.ll -o %t/raw-recorded-config.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/raw-recorded-config.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/template-recorded-config.ll -o %t/template-recorded-config.o 2>&1 | FileCheck %s --check-prefix=UNINITIALIZED
; RUN: test ! -e %t/template-recorded-config.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/selector-ten.ll -o %t/selector-ten.o 2>&1 | FileCheck %s --check-prefix=SELECTOR
; RUN: test ! -e %t/selector-ten.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/selector-fifteen.ll -o %t/selector-fifteen.o 2>&1 | FileCheck %s --check-prefix=SELECTOR
; RUN: test ! -e %t/selector-fifteen.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/mutable-constant.ll -o %t/mutable-constant.o 2>&1 | FileCheck %s --check-prefix=CONSTANT
; RUN: test ! -e %t/mutable-constant.o

; A mutable CReg snapshot uses unpredicated MOV mode 2 and therefore needs all
; eight columns initialized on every incoming path. Initialization belongs to
; the selected CReg, not just the configuration unit. Accepted SFPCONFIG masks
; use only bits 0, 2, ..., 14; a zero mask initializes no columns. Keep the
; existing rejection of noncanonical odd mask bits. LaneConfig reset, scalar
; stores that might issue configuration, and recording without execution do
; not establish the contents. Each rejection must be ordinary failure, with
; no partially published object; a crash is not accepted by these RUN lines.
;
; UNINITIALIZED: contents CReg snapshot requires all eight columns initialized
; MASK: SFPCONFIG mode 8 mask must select even lane bits
; SELECTOR: contents CReg read requires mutable C11-C14
; CONSTANT: contents constant read requires immutable C8/C9/C10/C15

;--- uninitialized.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @another_creg_is_not_initialization() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 12, i32 0, i32 0)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 11)
  ret void
}

;--- half-mask.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @only_four_columns_initialized() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 11, i32 4369, i32 8)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 11)
  ret void
}

;--- odd-only-mask.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @odd_bits_are_noncanonical() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 12, i32 43690, i32 8)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 12)
  ret void
}

;--- zero-mask.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @zero_mask_initializes_no_column() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 12, i32 0, i32 8)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 12)
  ret void
}

;--- one-branch.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @one_predecessor_initializes(i1 %choose) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br i1 %choose, label %initialize, label %join
initialize:
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 13, i32 0, i32 0)
  br label %join
join:
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 13)
  ret void
}

;--- backedge-only.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @backedge_does_not_initialize_first_iteration(i1 %again) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  br label %loop
loop:
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 14)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 14, i32 0, i32 0)
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- unknown-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @unknown_store_invalidates_creg(ptr %address) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 11, i32 0, i32 0)
  store volatile i32 0, ptr %address, align 4
  ; Reestablish lane-index legality without initializing CReg contents.
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 11)
  ret void
}

;--- unknown-cc.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @unknown_cc_cannot_establish_full_initialization() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %condition = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %condition, i32 6)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 12, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 12)
  ret void
}

;--- reset-only.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
define void @lane_reset_does_not_initialize_creg() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 13)
  ret void
}

;--- raw-recorded-config.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @raw_recording_is_not_initialization() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 14, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.record.end()
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 14)
  ret void
}

;--- template-recorded-config.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
define void @template_recording_is_not_initialization() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.replay.template.begin(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.end(i32 1)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 11)
  ret void
}

;--- selector-ten.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
define void @immutable_ten_is_not_mutable_creg() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 10)
  ret void
}

;--- selector-fifteen.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
define void @immutable_fifteen_is_not_mutable_creg() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 15)
  ret void
}

;--- mutable-constant.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @mutable_eleven_is_not_a_constant() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 11)
  ret void
}
