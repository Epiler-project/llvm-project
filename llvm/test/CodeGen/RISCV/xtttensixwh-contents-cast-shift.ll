; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=PHYS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.O0.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t.O0.o | FileCheck %s --check-prefixes=CHECK,HAZARD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.O2.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t.O2.o | FileCheck %s --check-prefixes=CHECK,HAZARD

; WH CAST consumes sign-magnitude integer bits and returns FP32 bits. Only
; modes 0/1 exist; the old destination remains an explicit disabled-lane input.
; SHFT mode 0 uses an independent vector shift amount, mode 1 a signed immediate.
; Its data source is always the old destination. No pointer/physical selector
; or generic vector arithmetic is substituted for either operation.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @wh_cast_contents() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_cast_contents>:
; CHECK: ttsfpload [[SRC:l[0-7]]], 0, 0, 3
; CHECK: ttsfpmov.all [[DST:l[0-7]]], c10
; CHECK: ttsfpcast [[DST]], [[SRC]], 0
; CHECK: ttsfpcast [[DST]], [[DST]], 1
; CHECK: ttsfpstore [[DST]], 0, 0, 3
; ISEL-LABEL: name: wh_cast_contents
; ISEL: [[SRC:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; ISEL: [[CAST0:%[0-9]+]]:sfpr = TTSFPCAST {{(killed )?}}[[OLD]], {{(killed )?}}[[SRC]], 0,
; ISEL: [[CAST1:%[0-9]+]]:sfpr = TTSFPCAST [[CAST0]], [[CAST0]], 1, implicit-def{{( dead)?}} $tt_issue, implicit-def{{( dead)?}} $tt_prng, implicit $tt_cc, implicit $tt_config, implicit $tt_issue, implicit $tt_prng
; ISEL: TTSFPSTORE {{(killed )?}}[[CAST1]], 0, 0, 3
; PHYS-LABEL: name: wh_cast_contents
; PHYS: [[SRC:\$tt_l[0-7]]] = TTSFPLOAD
; PHYS: [[DST:\$tt_l[0-7]]] = TTSFPMOVAll $tt_c10
; PHYS: [[DST]] = TTSFPCAST {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC]], 0,
; PHYS: [[DST]] = TTSFPCAST {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[DST]], 1,
; PHYS: TTSFPSTORE {{(killed )?(renamable )?}}[[DST]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %cast0 = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %old, <32 x i32> %src, i32 0)
  %cast1 = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %cast0, <32 x i32> %cast0, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %cast1, i32 0, i32 0, i32 3)
  ret void
}

define void @wh_shift_contents() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_shift_contents>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfpmov.all [[AMOUNT:l[0-7]]], c15
; CHECK: ttsfpshft [[VALUE]], [[AMOUNT]], 0, 0
; CHECK: ttsfpshft [[VALUE]], [[VALUE]], -2048, 1
; CHECK: ttsfpshft [[VALUE]], [[VALUE]], 2047, 1
; CHECK: ttsfpshft [[VALUE]], [[VALUE]], -1, 1
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_shift_contents
; ISEL: [[VALUE:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[AMOUNT:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; ISEL: [[SHIFT0:%[0-9]+]]:sfpr = TTSFPSHFT [[VALUE]], {{(killed )?}}[[AMOUNT]], 0, 0,
; ISEL: [[SHIFT1:%[0-9]+]]:sfpr = TTSFPSHFT [[SHIFT0]], [[SHIFT0]], -2048, 1,
; ISEL: [[SHIFT2:%[0-9]+]]:sfpr = TTSFPSHFT [[SHIFT1]], [[SHIFT1]], 2047, 1,
; ISEL: [[SHIFT3:%[0-9]+]]:sfpr = TTSFPSHFT [[SHIFT2]], [[SHIFT2]], -1, 1,
; ISEL: TTSFPSTORE {{(killed )?}}[[SHIFT3]], 0, 0, 3
; PHYS-LABEL: name: wh_shift_contents
; PHYS: [[VALUE:\$tt_l[0-7]]] = TTSFPLOAD
; PHYS: [[AMOUNT:\$tt_l[0-7]]] = TTSFPMOVAll $tt_c15
; PHYS: [[VALUE]] = TTSFPSHFT {{(killed )?(renamable )?}}[[VALUE]], {{(killed )?(renamable )?}}[[AMOUNT]], 0, 0,
; PHYS: [[VALUE]] = TTSFPSHFT {{(killed )?(renamable )?}}[[VALUE]], {{(killed )?(renamable )?}}[[VALUE]], -2048, 1,
; PHYS: [[VALUE]] = TTSFPSHFT {{(killed )?(renamable )?}}[[VALUE]], {{(killed )?(renamable )?}}[[VALUE]], 2047, 1,
; PHYS: [[VALUE]] = TTSFPSHFT {{(killed )?(renamable )?}}[[VALUE]], {{(killed )?(renamable )?}}[[VALUE]], -1, 1,
; PHYS: TTSFPSTORE {{(killed )?(renamable )?}}[[VALUE]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %amount = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %shift0 = call <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32> %value, <32 x i32> %amount, i32 0)
  %shift1 = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %shift0, i32 -2048, i32 1)
  %shift2 = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %shift1, i32 2047, i32 1)
  %shift3 = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %shift2, i32 -1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %shift3, i32 0, i32 0, i32 3)
  ret void
}

; A real dynamic backedge carries contents; do not copy/unroll the body.
define void @wh_shift_contents_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_shift_contents_loop>:
; CHECK: ttsfpshft [[VALUE:l[0-7]]], [[AMOUNT:l[0-7]]], 0, 0
; CHECK-NOT: ttsfpshft
; CHECK: ttsfpshft [[VALUE]], [[VALUE]], -1, 1
; CHECK-NOT: ttsfpshft
; CHECK: bne{{z?}} {{.*}},
; CHECK-NOT: ttsfpshft
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_shift_contents_loop
; ISEL: [[PHI:%[0-9]+]]:sfpr = PHI
; ISEL: [[SHIFT:%[0-9]+]]:sfpr = TTSFPSHFT [[PHI]], {{.*}}, 0, 0,
; ISEL: [[RESULT:%[0-9]+]]:sfpr = TTSFPSHFT [[SHIFT]], [[SHIFT]], -1, 1,
; ISEL: TTSFPSTORE {{(killed )?}}[[RESULT]], 0, 0, 3
; PHYS-LABEL: name: wh_shift_contents_loop
; PHYS: [[VALUE:\$tt_l[0-7]]] = TTSFPSHFT {{(killed )?(renamable )?}}[[VALUE]], {{.*}}, 0, 0,
; PHYS: [[VALUE]] = TTSFPSHFT {{(killed )?(renamable )?}}[[VALUE]], {{(killed )?(renamable )?}}[[VALUE]], -1, 1,
; PHYS: TTSFPSTORE {{(killed )?(renamable )?}}[[VALUE]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %amount = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br label %loop
loop:
  %iteration = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%initial, %entry], [%result, %loop]
  %shift = call <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32> %value, <32 x i32> %amount, i32 0)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %shift, i32 -1, i32 1)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 3)
  ret void
}

; WH has no MAD forwarding interlock: both new consumers must wait one cycle.
define void @wh_cast_shift_mad_hazards() "tensix-executor"="trisc1" {
; HAZARD-LABEL: <wh_cast_shift_mad_hazards>:
; HAZARD: ttsfpmad [[VALUE:l[0-7]]],
; HAZARD-NEXT: {{.*}}ttsfpnop
; HAZARD-NEXT: {{.*}}ttsfpcast [[VALUE]], [[VALUE]], 0
; HAZARD: ttsfpmad [[VALUE]],
; HAZARD-NEXT: {{.*}}ttsfpnop
; HAZARD-NEXT: {{.*}}ttsfpshft [[VALUE]], [[VALUE]], -1, 1
; HAZARD: ttsfpstore [[VALUE]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %mad = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %value, <32 x i32> %value, <32 x i32> %one, <32 x i32> %one, i32 0)
  %cast = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %mad, <32 x i32> %mad, i32 0)
  %mad2 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %cast, <32 x i32> %cast, <32 x i32> %one, <32 x i32> %one, i32 0)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.immediate(<32 x i32> %mad2, i32 -1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 3)
  ret void
}

; Separate old/source and old/amount roles must also receive the WH MAD wait.
define void @wh_cast_shift_source_hazards() "tensix-executor"="trisc1" {
; HAZARD-LABEL: <wh_cast_shift_source_hazards>:
; HAZARD: ttsfpmad [[PRODUCER:l[0-7]]],
; HAZARD-NEXT: {{.*}}ttsfpnop
; HAZARD-NEXT: {{.*}}ttsfpcast [[CONSUMER:l[0-7]]], [[PRODUCER]], 1
; HAZARD: ttsfpmad [[PRODUCER]],
; HAZARD-NEXT: {{.*}}ttsfpnop
; HAZARD-NEXT: {{.*}}ttsfpshft [[CONSUMER]], [[PRODUCER]], 0, 0
; HAZARD: ttsfpstore [[CONSUMER]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %mad = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %value, <32 x i32> %value, <32 x i32> %one, <32 x i32> %one, i32 0)
  %cast = call <32 x i32> @llvm.riscv.tt.contents.cast(<32 x i32> %old, <32 x i32> %mad, i32 1)
  %mad2 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %mad, <32 x i32> %mad, <32 x i32> %one, <32 x i32> %one, i32 0)
  %result = call <32 x i32> @llvm.riscv.tt.contents.shft.vector(<32 x i32> %cast, <32 x i32> %mad2, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 3)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
