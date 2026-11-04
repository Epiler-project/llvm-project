; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr = COPY '
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr = COPY '
;
; One zero-trip-capable loop carries lane contents through three tied updates.
; The independently live source remains a different SFPR. No source iteration
; cloning, numerical copy, helper call, or spill is permitted.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

define void @setfield_contents_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <setfield_contents_loop>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfpmov.all [[SOURCE:l[0-7]]], c10
; CHECK: beqz
; CHECK: ttsfpsetexp [[VALUE]], [[SOURCE]], 0, 2
; CHECK: ttsfpsetman [[VALUE]], [[SOURCE]], 4095, 1
; CHECK: ttsfpsetsgn [[VALUE]], [[SOURCE]], 0, 0
; CHECK: bne
; CHECK: ttsfpstore [[VALUE]], 2, 0, 3
; CHECK: ttsfpstore [[SOURCE]], 3, 0, 3
; ISEL-LABEL: name: setfield_contents_loop
; ISEL: [[OLD:%[0-9]+]]:sfpr = PHI
; ISEL: [[EXP:%[0-9]+]]:sfpr = TTSFPSETEXP [[OLD]], [[SOURCE:%[0-9]+]], 0, 2
; ISEL: [[MAN:%[0-9]+]]:sfpr = TTSFPSETMAN [[EXP]], [[SOURCE]], 4095, 1
; ISEL: [[SIGN:%[0-9]+]]:sfpr = TTSFPSETSGN [[MAN]], [[SOURCE]], 0, 0
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %empty = icmp eq i32 %limit, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %old = phi <32 x i32> [%initial, %entry], [%sign, %loop]
  %exp = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %source, i32 2)
  %man = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %exp, <32 x i32> %source, i32 4095, i32 1)
  %sign = call <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32> %man, <32 x i32> %source, i32 0)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %result = phi <32 x i32> [%initial, %entry], [%sign, %loop]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
