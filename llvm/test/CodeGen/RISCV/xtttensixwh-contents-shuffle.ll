; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
;
; Mode 3 is the supported WH/BH eight-lane rotate, not buggy WH mode 4.
; Explicit old/result and source contents retain disabled-destination lanes.
; The fixed SFPU cooldown supplies the official next-cycle gap; source loops
; retain a zero-trip path, scalar backedge and one physical rotate issue.
;
; CHECK-LABEL: <rotate8_lane_values>:
; CHECK: ttsfpmov.all [[OLD:l[0-7]]], c9
; CHECK: ttsfpmov.all [[SRC:l[0-7]]], c15
; CHECK: ttsfpstore [[SRC]], 0, 0, 4
; CHECK: ttsfpshft2 [[OLD]], [[SRC]], 3
; CHECK-NEXT: {{[0-9a-f]+:.*}}ttsfpnop
; CHECK-NEXT: {{[0-9a-f]+:.*}}ttsfpiadd [[OLD]], [[SRC]]
; CHECK: ttsfpstore [[OLD]], 2, 0, 4
; CHECK-NOT: ttsfpmov
; CHECK: ret
; ISEL-LABEL: name: rotate8_lane_values
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c9
; ISEL: [[SRC:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; ISEL: [[ROT:%[0-9]+]]:sfpr = TTSFPSHFT2 [[OLD]], [[SRC]], 3,
; ISEL-SAME: implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
;
; CHECK-LABEL: <rotate8_source_loop>:
; CHECK: ttsfpmov.all [[ACTIVE:l[0-7]]], c15
; CHECK: ttsfpstore [[ACTIVE]], 0, 0, 4
; CHECK: beqz
; CHECK: ttsfpnop
; CHECK: ttsfpshft2 [[ACTIVE]], [[ACTIVE]], 3
; CHECK: {{bne|bltu}}
; CHECK: ttsfpnop
; CHECK: ttsfpstore [[ACTIVE]], 2, 0, 4
; CHECK: ret
; ISEL-LABEL: name: rotate8_source_loop
; ISEL: :sfpr = PHI
; ISEL: :sfpr = TTSFPSHFT2

target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @rotate8_lane_values() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lanes = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lanes, i32 0, i32 0, i32 4)
  %rotated = call <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32> %old, <32 x i32> %lanes, i32 3)
  %sum = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %rotated, <32 x i32> %lanes, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 2, i32 0, i32 4)
  ret void
}

define void @rotate8_source_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %initial, i32 0, i32 0, i32 4)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %iteration = phi i32 [0, %entry], [%next, %loop]
  %active = phi <32 x i32> [%initial, %entry], [%rotated, %loop]
  %rotated = call <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32> %active, <32 x i32> %active, i32 3)
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%initial, %entry], [%rotated, %loop]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
