; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.pushc(i32 immarg)
declare void @llvm.riscv.tt.contents.popc(i32 immarg)

; Each load is a partial write: enabled lanes preserve old high halves and
; inactive lanes preserve every old bit. The actual loop carries those contents,
; including the zero-trip path, instead of manufacturing new physical selectors.
define void @wh_predicate_partial_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: wh_predicate_partial_loop
; ISEL: TTSFPPUSHC
; ISEL: TTSFPSETCCEQ
; ISEL: :sfpr = PHI
; ISEL: :sfpr = TTSFPLOAD {{.*}}, 0, 0, 14
; ISEL: :sfpr = TTSFPEXMAN
; ISEL: TTSFPPOPC
; CHECK-LABEL: <wh_predicate_partial_loop>:
; CHECK: ttsfpsetcc.eq0
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 14
; CHECK: ttsfpexman [[VALUE]], [[VALUE]], 0
; CHECK: bne{{z?}}
; CHECK: ttsfppopc
; CHECK: ttsfpstore [[VALUE]], 4, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %predicate = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.pushc(i32 0)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %predicate, i32 6)
  %any = icmp ne i32 %limit, 0
  br i1 %any, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %old = phi <32 x i32> [ %initial, %entry ], [ %result, %loop ]
  %partial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 14)
  %result = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %partial, <32 x i32> %partial, i32 0)
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [ %initial, %entry ], [ %result, %loop ]
  call void @llvm.riscv.tt.contents.popc(i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 4, i32 0, i32 3)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
