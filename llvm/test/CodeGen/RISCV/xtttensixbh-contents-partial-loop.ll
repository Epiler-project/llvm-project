; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.pushc(i32 immarg)
declare void @llvm.riscv.tt.contents.popc(i32 immarg)

; The low-half load preserves the high half of every enabled lane and all
; inactive lanes; the following unary operation preserves the inactive lanes
; of the same destination. Both previous versions must remain true use/defs.
; One contents recurrence and one dynamic source loop survive allocation.
define void @contents_partial_loop(i32 %limit) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: contents_partial_loop
; ISEL: :sfpr = PHI
; ISEL: :sfpr = TTSFPLOAD %{{[0-9]+}}, 0, 7, 14
; ISEL: :sfpr = TTSFPEXMAN %{{[0-9]+}}, %{{[0-9]+}}, 0
; CHECK-LABEL: <contents_partial_loop>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 7, 14
; CHECK: ttsfpexman [[VALUE]], [[VALUE]], 0
; CHECK: {{bne[z]?}} {{.*}},
; CHECK: ttsfpstore [[VALUE]], 4, 7, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %predicate = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.pushc(i32 0)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %predicate, i32 6)
  br label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %old = phi <32 x i32> [ %initial, %entry ], [ %result, %loop ]
  %partial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 7, i32 14)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %partial, <32 x i32> %partial, i32 0)
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.popc(i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 4, i32 7, i32 3)
  ret void
}

; Stochastic CAST has an explicit PRNG read/def even when its data result
; would otherwise look like an ordinary pure vector conversion.
define void @contents_stochastic_cast() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: contents_stochastic_cast
; ISEL: :sfpr = TTSFPCAST {{.*}}implicit-def{{( dead)?}} $tt_issue, implicit-def{{( dead)?}} $tt_prng, implicit $tt_cc, implicit $tt_config, implicit $tt_issue, implicit $tt_prng
; CHECK-LABEL: <contents_stochastic_cast>:
; CHECK: ttsfpcast
; CHECK: ttsfpstore
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %zero, <32 x i32> %zero, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 7, i32 3)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
