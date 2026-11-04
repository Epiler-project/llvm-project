; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 immarg)
declare void @llvm.riscv.tt.contents.config.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)

; Reading mutable architectural state always creates a real ordinary-SFPR
; snapshot. The first contents remain live after C11 is overwritten. Closing
; CC cannot predicate the snapshot: MOV mode 2 reads all columns.
define void @creg_snapshots() "tensix-executor"="trisc1" {
; CHECK-LABEL: <creg_snapshots>:
; CHECK: ttsfpconfig.c11 0, 0
; CHECK: ttsfpencc 1, 10
; CHECK: ttsfpmov.all [[BEFORE:l[0-7]]], c11
; CHECK: ttsfpencc 3, 10
; CHECK: ttsfpconfig.c11 0, 0
; CHECK: ttsfpmov.all [[AFTER:l[0-7]]], c11
; CHECK: ttsfpstore [[BEFORE]], 0, 0, 4
; CHECK: ttsfpstore [[AFTER]], 2, 0, 4
; ISEL-LABEL: name: creg_snapshots
; ISEL: PseudoTTSFPContentsConfigC11 {{.*}}implicit-def {{(dead )?}}$tt_c11
; ISEL: [[BEFORE:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c11
; ISEL: PseudoTTSFPContentsConfigC11 {{.*}}implicit-def {{(dead )?}}$tt_c11
; ISEL: [[AFTER:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c11
; ISEL: TTSFPSTORE {{(killed )?}}[[BEFORE]], 0, 0, 4
; ISEL: TTSFPSTORE {{(killed )?}}[[AFTER]], 2, 0, 4
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %zero, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.encc(i32 1, i32 10)
  %before = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 11)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %one, i32 11, i32 0, i32 0)
  %after = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 11)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %before, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %after, i32 2, i32 0, i32 4)
  ret void
}

; Complementary even Imm16 bits initialize the eight columns. Later partial
; writes preserve initialized columns. Reset changes LaneConfig, not CReg.
define void @creg_partial_loop(i32 noundef %count) "tensix-executor"="trisc1" {
; CHECK-LABEL: <creg_partial_loop>:
; CHECK: ttsfpconfig.c12 4369, 8
; CHECK: ttsfpconfig.c12 17476, 8
; CHECK: ttsfpmov.all {{l[0-7]}}, c12
; CHECK: ttsfpconfig.c12 0, 0
; CHECK: ttsfpmov.all {{l[0-7]}}, c12
; CHECK: ttsfpadd
; CHECK: bne{{z?}}
; CHECK: ttsfpstore
; ISEL-LABEL: name: creg_partial_loop
; ISEL: :sfpr = PHI
; ISEL: PseudoTTSFPContentsConfigC12
; ISEL: :sfpr = TTSFPMOVAll $tt_c12
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %zero, i32 12, i32 4369, i32 8)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %one, i32 12, i32 17476, i32 8)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 12)
  br label %loop
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  %acc = phi <32 x i32> [ %initial, %entry ], [ %sum, %loop ]
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 12, i32 0, i32 0)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 12)
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %acc, <32 x i32> %acc, <32 x i32> %snapshot, i32 0)
  %next = add i32 %remaining, -1
  %again = icmp ne i32 %next, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 4)
  ret void
}

; Both incoming paths define C13. A zero mask selects no C14 column and
; preserves the complete initialization. Odd mask bits remain noncanonical.
define void @creg_cfg(i1 %choose) "tensix-executor"="trisc1" {
; CHECK-LABEL: <creg_cfg>:
; CHECK: ttsfpconfig.c13
; CHECK: ttsfpmov.all {{l[0-7]}}, c13
; CHECK: ttsfpconfig.c14 0, 0
; CHECK: ttsfpconfig.c14 0, 8
; CHECK: ttsfpmov.all {{l[0-7]}}, c14
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  br i1 %choose, label %left, label %right
left:
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %zero, i32 13, i32 0, i32 0)
  br label %join
right:
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %one, i32 13, i32 21845, i32 8)
  br label %join
join:
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 13)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %snapshot, i32 0, i32 0, i32 4)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 14, i32 0, i32 0)
  call void @llvm.riscv.tt.contents.config.creg(<32 x i32> %source, i32 14, i32 0, i32 8)
  %last = call <32 x i32> @llvm.riscv.tt.contents.read.creg(i32 14)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %last, i32 2, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
