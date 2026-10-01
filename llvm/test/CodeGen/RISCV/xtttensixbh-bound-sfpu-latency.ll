; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefixes=CHECK,O0 --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefixes=CHECK,O2 --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Extend bound_latency in sfpu-bound-optimization.ll, not the old carrier ABI.
; Every move is authored; O2 may only move an independent existing all-lane
; copy into a latency gap. Old/inactive contents, dependence order and the
; real loop must survive LLVM IR -> physical selection -> final object code.

declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpsetcc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()

define void @bound_add_copy_gap() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_add_copy_gap
; ISEL: usesBoundTensixSFPU: true
; ISEL: $tt_l0 = TTSFPADD $tt_l0, $tt_c9, $tt_c10, 1,
; ISEL-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; ISEL-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; CHECK-LABEL: name: bound_add_copy_gap
; CHECK: $tt_l0 = TTSFPADD $tt_l0, $tt_c9, $tt_c10, 1,
; O0-NEXT: TTSFPNOP
; O0-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; O0-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; O2-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; O2-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l4, 1, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 9, i32 10, i32 1)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  ret void
}

define void @bound_mul_copy_gap() "tensix-executor"="trisc2" {
; CHECK-LABEL: name: bound_mul_copy_gap
; CHECK: $tt_l0 = TTSFPMUL $tt_l0, $tt_c10, $tt_c9, 2,
; O0-NEXT: TTSFPNOP
; O0-NEXT: $tt_l0 = TTSFPISUB $tt_l0, $tt_c9,
; O0-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; O2-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; O2-NEXT: $tt_l0 = TTSFPISUB $tt_l0, $tt_c9,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l4, 1, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmul(i32 0, i32 0, i32 10, i32 9, i32 2)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 6)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  ret void
}

; The snapshot must observe the integer consumer, not the preceding FP result.
define void @bound_latency_keep_raw() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_latency_keep_raw
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c10,
; CHECK-NEXT: $tt_l4 = TTSFPMOVAll $tt_l0,
; CHECK-NEXT: TTSFPSTORE $tt_l4, 0, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 10, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 0, i32 0, i32 4)
  ret void
}

define void @bound_latency_keep_war() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_latency_keep_war
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_l2,
; CHECK-NEXT: $tt_l2 = TTSFPMOVAll $tt_c9,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l2, 1, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 2, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 1, i32 0, i32 4)
  ret void
}

define void @bound_latency_keep_waw() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_latency_keep_waw
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c10,
; CHECK-NEXT: $tt_l0 = TTSFPMOVAll $tt_c9,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 10, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

; The authored masked move preserves its old L4 in inactive lanes. It is not
; an all-lane scheduling candidate and cannot be changed into MOVAll.
define void @bound_latency_keep_masked() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_latency_keep_masked
; CHECK: TTSFPSETCCNE $tt_l2,
; CHECK-NEXT: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_l2,
; CHECK-NEXT: $tt_l4 = TTSFPMOV $tt_l4, $tt_c9,
; CHECK-NEXT: TTSFPPOPC
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l4, 1, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 10)
  call void @llvm.riscv.tt.bound.sfpload(i32 2, i32 2, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 2, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 2, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov(i32 4, i32 4, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  ret void
}

define void @bound_latency_keep_cc_consumer() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_latency_keep_cc_consumer
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADDCCLT $tt_l0, $tt_c9,
; CHECK-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l4, 1, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  ret void
}

define void @bound_latency_real_loop(i32 %count) "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_latency_real_loop
; CHECK: bb.{{[0-9]+}}.loop:
; CHECK: $tt_l0 = TTSFPMAD
; O0-NEXT: TTSFPNOP
; O0-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; O0-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; O2-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; O2-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; CHECK: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK: TTSFPSTORE $tt_l4, 1, 0, 4,
; CHECK-NOT: TTSFPMAD
; CHECK-NOT: TTSFPIADD
; CHECK-NOT: TTSFPMOV
; CHECK: BNE {{.*}}%bb.{{[0-9]+}}
; CHECK: bb.{{[0-9]+}}.exit:
; CHECK: PseudoRET
entry:
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  %next = add i32 %remaining, -1
  %more = icmp ne i32 %next, 0
  br i1 %more, label %loop, label %exit, !llvm.loop !0
exit:
  ret void
}

define void @bound_latency_explicit_owner() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: bound_latency_explicit_owner
; CHECK: TTREPLAY 1, 0, 1, 0,
; CHECK-NEXT: TTNOP
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; CHECK-NEXT: $tt_l4 = TTSFPMOVAll $tt_c10,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l4, 1, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 1, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
