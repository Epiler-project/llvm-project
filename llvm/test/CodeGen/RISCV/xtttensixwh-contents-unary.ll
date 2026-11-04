; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; The WH EXEXP, EXMAN and LZ functional models are also the BH contracts.
; EXEXP uses mode bits 0/1/3; EXMAN uses bit 0; LZ uses bits 1/2/3.
; All results retain the old destination in disabled lanes. Contents stay in
; SFPRs with actual CC/issue dependencies and no physical selector in LLVM IR.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @wh_exexp_modes() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_exexp_modes>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 0
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 1
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 2
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 3
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 8
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 9
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 10
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 11
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_exexp_modes
; ISEL: [[LOAD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[VALUE:%[0-9]+]]:sfpr = TTSFPEXEXP [[LOAD]], [[LOAD]], 0
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 3)
  %m0 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %loaded, <32 x i32> %loaded, i32 0)
  %m1 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m0, <32 x i32> %m0, i32 1)
  %m2 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m1, <32 x i32> %m1, i32 2)
  %m3 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m2, <32 x i32> %m2, i32 3)
  %m8 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m3, <32 x i32> %m3, i32 8)
  %m9 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m8, <32 x i32> %m8, i32 9)
  %m10 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m9, <32 x i32> %m9, i32 10)
  %m11 = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %m10, <32 x i32> %m10, i32 11)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %m11, i32 0, i32 0, i32 3)
  ret void
}

define void @wh_exman_modes() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_exman_modes>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfpexman [[VALUE]], [[VALUE]], 0
; CHECK: ttsfpexman [[VALUE]], [[VALUE]], 1
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_exman_modes
; ISEL: [[LOAD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[VALUE:%[0-9]+]]:sfpr = TTSFPEXMAN [[LOAD]], [[LOAD]], 0
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 3)
  %m0 = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %loaded, <32 x i32> %loaded, i32 0)
  %m1 = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %m0, <32 x i32> %m0, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %m1, i32 0, i32 0, i32 3)
  ret void
}

define void @wh_lz_modes() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_lz_modes>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 0
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 2
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 4
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 6
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 8
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 10
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 12
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 14
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_lz_modes
; ISEL: [[LOAD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[VALUE:%[0-9]+]]:sfpr = TTSFPLZ [[LOAD]], [[LOAD]], 0
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 3)
  %m0 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %loaded, <32 x i32> %loaded, i32 0)
  %m2 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m0, <32 x i32> %m0, i32 2)
  %m4 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m2, <32 x i32> %m2, i32 4)
  %m6 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m4, <32 x i32> %m4, i32 6)
  %m8 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m6, <32 x i32> %m6, i32 8)
  %m10 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m8, <32 x i32> %m8, i32 10)
  %m12 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m10, <32 x i32> %m10, i32 12)
  %m14 = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %m12, <32 x i32> %m12, i32 14)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %m14, i32 0, i32 0, i32 3)
  ret void
}

; EXEXP mode 2 defines LaneFlags; the following two operations read that
; predicate and preserve the actual previous destination in disabled lanes.
; Source contents remain live in a different SFPR from the tied destination.
define void @wh_unary_predicated_old() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_unary_predicated_old>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfpmov.all [[SOURCE:l[0-7]]], c10
; CHECK: ttsfpexexp [[VALUE]], [[SOURCE]], 2
; CHECK: ttsfpexman [[VALUE]], [[SOURCE]], 1
; CHECK: ttsfplz [[VALUE]], [[SOURCE]], 2
; CHECK: ttsfpencc 3, 10
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_unary_predicated_old
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[SOURCE:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; ISEL: [[EXP:%[0-9]+]]:sfpr = TTSFPEXEXP [[OLD]], [[SOURCE]], 2, implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[MAN:%[0-9]+]]:sfpr = TTSFPEXMAN [[EXP]], [[SOURCE]], 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[LZ:%[0-9]+]]:sfpr = TTSFPLZ [[MAN]], {{(killed )?}}[[SOURCE]], 2, implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPSTORE {{(killed )?}}[[LZ]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %exp = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %loaded, <32 x i32> %source, i32 2)
  %man = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %exp, <32 x i32> %source, i32 1)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %man, <32 x i32> %source, i32 2)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  ret void
}

; WH must insert an SFPU gap before each next-cycle unary MAD consumer.
define void @wh_unary_mad_hazards() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_unary_mad_hazards>:
; CHECK: ttsfpmad [[VALUE:l[0-7]]],
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpexexp [[VALUE]], [[VALUE]], 0
; CHECK: ttsfpmad [[VALUE]],
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpexman [[VALUE]], [[VALUE]], 0
; CHECK: ttsfpmad [[VALUE]],
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfplz [[VALUE]], [[VALUE]], 0
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_unary_mad_hazards
; ISEL: [[MAD1:%[0-9]+]]:sfpr = TTSFPMAD
; ISEL: [[EXP:%[0-9]+]]:sfpr = TTSFPEXEXP [[MAD1]], [[MAD1]], 0
; ISEL: [[MAD2:%[0-9]+]]:sfpr = TTSFPMAD [[EXP]],
; ISEL: [[MAN:%[0-9]+]]:sfpr = TTSFPEXMAN [[MAD2]], [[MAD2]], 0
; ISEL: [[MAD3:%[0-9]+]]:sfpr = TTSFPMAD [[MAN]],
; ISEL: :sfpr = TTSFPLZ [[MAD3]], [[MAD3]], 0
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %mad1 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %old, <32 x i32> %old, <32 x i32> %one, <32 x i32> %one, i32 0)
  %exp = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %mad1, <32 x i32> %mad1, i32 0)
  %mad2 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %exp, <32 x i32> %exp, <32 x i32> %one, <32 x i32> %one, i32 0)
  %man = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %mad2, <32 x i32> %mad2, i32 0)
  %mad3 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %man, <32 x i32> %man, <32 x i32> %one, <32 x i32> %one, i32 0)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %mad3, <32 x i32> %mad3, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 0, i32 3)
  ret void
}

; Partial loading retains the previous high half and disabled lanes. One
; contents PHI and one source loop survive the three tied unary updates.
; Scalar loop-control instructions may be scheduled between SFPU instructions.
define void @wh_unary_contents_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_unary_contents_loop>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 3, 14
; CHECK: ttsfpexexp [[VALUE]], [[VALUE]], 0
; CHECK: ttsfpexman [[VALUE]], [[VALUE]], 1
; CHECK: ttsfplz [[VALUE]], [[VALUE]], 4
; CHECK: bne{{z?}} {{.*}},
; CHECK: ttsfpstore [[VALUE]], 0, 3, 3
; ISEL-LABEL: name: wh_unary_contents_loop
; ISEL: [[OLD:%[0-9]+]]:sfpr = PHI
; ISEL: [[PARTIAL:%[0-9]+]]:sfpr = TTSFPLOAD [[OLD]], 0, 3, 14
; ISEL: [[EXP:%[0-9]+]]:sfpr = TTSFPEXEXP [[PARTIAL]], [[PARTIAL]], 0
; ISEL: [[MAN:%[0-9]+]]:sfpr = TTSFPEXMAN [[EXP]], [[EXP]], 1
; ISEL: :sfpr = TTSFPLZ [[MAN]], [[MAN]], 4
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %old = phi <32 x i32> [%initial, %entry], [%lz, %loop]
  %partial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 3, i32 14)
  %exp = call <32 x i32> @llvm.riscv.tt.contents.exexp(<32 x i32> %partial, <32 x i32> %partial, i32 0)
  %man = call <32 x i32> @llvm.riscv.tt.contents.exman(<32 x i32> %exp, <32 x i32> %exp, i32 1)
  %lz = call <32 x i32> @llvm.riscv.tt.contents.lz(<32 x i32> %man, <32 x i32> %man, i32 4)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lz, i32 0, i32 3, i32 3)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
