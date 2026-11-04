; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.O0.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh %t.bh.O0.o | FileCheck %s --check-prefix=DIS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh.O2.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh %t.bh.O2.o | FileCheck %s --check-prefix=DIS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.wh.O0.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t.wh.O0.o | FileCheck %s --check-prefix=DIS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh.O2.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t.wh.O2.o | FileCheck %s --check-prefix=DIS

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @direct_common() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: direct_common
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c9
; ISEL: [[SRC:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; ISEL: [[SET:%[0-9]+]]:sfpr = TTSFPDIVP2 [[OLD]], {{(killed )?}}[[SRC]], 255, 0,
; ISEL: [[ADD:%[0-9]+]]:sfpr = TTSFPDIVP2 [[SET]], [[SET]], 255, 1,
; ISEL: [[FPADD:%[0-9]+]]:sfpr = TTSFPADDI [[ADD]], 16256, 0,
; ISEL: [[FPMUL:%[0-9]+]]:sfpr = TTSFPMULI [[FPADD]], 16384, 0,
; ISEL: TTSFPSTORE {{(killed )?}}[[FPMUL]], 0, 0, 4,
; DIS-LABEL: <direct_common>:
; DIS: ttsfpdivp2 [[DST:l[0-7]]], [[SRC:l[0-7]]], 255, 0
; DIS: ttsfpdivp2 [[DST]], [[DST]], 255, 1
; DIS: ttsfpaddi [[DST]], 16256, 0
; DIS: ttsfpmuli [[DST]], 16384, 0
; DIS: ttsfpstore [[DST]], 0, 0, 4
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %set = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %src, i32 255, i32 0)
  %add = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %set, <32 x i32> %set, i32 255, i32 1)
  %fpadd = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %add, i32 16256, i32 0)
  %fpmul = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %fpadd, i32 16384, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %fpmul, i32 0, i32 0, i32 4)
  ret void
}

; Dynamic loop tests actual SFPR PHI/backedge and exact source body count.
define void @direct_common_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: direct_common_loop
; ISEL: [[PHI:%[0-9]+]]:sfpr = PHI
; ISEL: [[DIV:%[0-9]+]]:sfpr = TTSFPDIVP2 [[PHI]], [[PHI]], 1, 1,
; ISEL: [[ADD:%[0-9]+]]:sfpr = TTSFPADDI [[DIV]], 16256, 0,
; ISEL: [[MUL:%[0-9]+]]:sfpr = TTSFPMULI [[ADD]], 16384, 0,
; DIS-LABEL: <direct_common_loop>:
; DIS: ttsfpdivp2
; DIS-NOT: ttsfpdivp2
; DIS: ttsfpaddi
; DIS-NOT: ttsfpaddi
; DIS: ttsfpmuli
; DIS-NOT: ttsfpmuli
; DIS: bne{{z?}}
; DIS: ttsfpstore
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %positive = icmp ugt i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %value = phi <32 x i32> [ %src, %entry ], [ %result, %loop ]
  %div = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %value, <32 x i32> %value, i32 1, i32 1)
  %add = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %div, i32 16256, i32 0)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %add, i32 16384, i32 0)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [ %src, %entry ], [ %result, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 0, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
