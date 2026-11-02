; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/no-reset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/masked-reset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/one-branch.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/unknown-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/backedge-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/attribute-only.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/raw-recorded-reset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/template-recorded-reset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG

; Numerical SSA cannot describe the extra SFPLOAD destination-index capture
; write, nor TEN-2932 writes to L4..7 under ENABLE_DEST_INDEX. Actual reaching
; instructions must establish disabled indexing; a function attribute, one
; initialized branch, a masked reset, and a possibly issuing scalar store do
; not establish that fact. CC predicates remain freely usable after setup.
; CONFIG: contents SFPU requires ENABLE_DEST_INDEX known disabled

;--- valid.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @predication_after_setup() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %mask, i32 6)
  ; A later masked reset preserves the already-disabled indexing bits.
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 7, i32 14)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 7, i32 3)
  ret void
}

define void @both_predecessors_initialize(i1 %choose) "tensix-executor"="trisc1" {
entry:
  br i1 %choose, label %left, label %right
left:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %join
right:
  ; Disabling flag predication also makes the first eight columns active.
  call void @llvm.riscv.tt.contents.encc(i32 0, i32 2)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %join
join:
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 7, i32 3)
  ret void
}

define void @initialized_loop(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 7, i32 3)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- no-reset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @no_reset() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- masked-reset.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @masked_reset() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- one-branch.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @one_predecessor_initializes(i1 %choose) "tensix-executor"="trisc1" {
entry:
  br i1 %choose, label %left, label %join
left:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %join
join:
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- unknown-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @possibly_issues_configuration(ptr %address) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  store volatile i32 0, ptr %address, align 4
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- backedge-store.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @backedge_invalidates(ptr %address, i1 %again) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br label %loop
loop:
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  store volatile i32 0, ptr %address, align 4
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- attribute-only.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @attribute_is_not_initialization() "tensix-executor"="trisc1" "tensix-lane-config"="canonical" {
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- raw-recorded-reset.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @raw_recording_is_not_initialization() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 2, i32 0)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.replay.record.end()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- template-recorded-reset.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @template_recording_is_not_initialization() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.replay.template.end(i32 1)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}
