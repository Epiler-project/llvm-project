; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+m,+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
;
; WH's official SFPSWAP functional model admits modes0..9. The contents
; receiver returns two numerical values, not physical register numbers.
; LaneConfig reset excludes the argmin companion-register side effects.

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)

; CHECK-LABEL: <swap_mode0>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 0
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode0
; ISEL: TTSFPSWAP
; ISEL-SAME: 0
define void @swap_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 0)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode1>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 1
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode1
; ISEL: TTSFPSWAP
; ISEL-SAME: 1
define void @swap_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 1)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode2>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 2
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode2
; ISEL: TTSFPSWAP
; ISEL-SAME: 2
define void @swap_mode2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 2)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode3>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 3
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode3
; ISEL: TTSFPSWAP
; ISEL-SAME: 3
define void @swap_mode3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 3)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode4>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 4
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode4
; ISEL: TTSFPSWAP
; ISEL-SAME: 4
define void @swap_mode4() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 4)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode5>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 5
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode5
; ISEL: TTSFPSWAP
; ISEL-SAME: 5
define void @swap_mode5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 5)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode6>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 6
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode6
; ISEL: TTSFPSWAP
; ISEL-SAME: 6
define void @swap_mode6() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 6)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode7>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 7
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode7
; ISEL: TTSFPSWAP
; ISEL-SAME: 7
define void @swap_mode7() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 7)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode8>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 8
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode8
; ISEL: TTSFPSWAP
; ISEL-SAME: 8
define void @swap_mode8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 8)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_mode9>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 9
; CHECK-NEXT: ttsfpnop
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_mode9
; ISEL: TTSFPSWAP
; ISEL-SAME: 9
define void @swap_mode9() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 9)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_loop>:
; CHECK: ttsfpswap {{l[0-7]}}, {{l[0-7]}}, 0
; CHECK-NOT: ttsfpswap
; CHECK-COUNT-2: ttsfpstore
; ISEL-LABEL: name: swap_loop
; ISEL: PHI
; ISEL: TTSFPSWAP
define void @swap_loop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %v0 = phi <32 x i32> [ %a, %entry ], [ %r0, %loop ]
  %v1 = phi <32 x i32> [ %b, %entry ], [ %r1, %loop ]
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %v0, <32 x i32> %v1, i32 0)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %a, %entry ], [ %r0, %loop ]
  %final1 = phi <32 x i32> [ %b, %entry ], [ %r1, %loop ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
