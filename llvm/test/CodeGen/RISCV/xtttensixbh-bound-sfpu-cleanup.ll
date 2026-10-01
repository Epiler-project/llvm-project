; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=CHECK --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Extend the basic bound_redundancy case in sfpu-bound-optimization.ll through
; the real bound ingress: every numerical object, old value, snapshot and move
; below is authored. No carrier intrinsic, allocator-created SFPR or hand-built
; MIR supplies an alternative entry into the production cleanup pass.

declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpsetcc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()

define void @bound_enable_modes() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_enable_modes
; ISEL: usesBoundTensixSFPU: true
; ISEL: $tt_l0 = TTSFPMOV $tt_l0, $tt_l0,
; ISEL-NEXT: $tt_l0 = TTSFPMOVAll $tt_l0,
; CHECK-LABEL: name: bound_enable_modes
; CHECK: TTSFPENCC 0, 0,
; CHECK-NEXT: TTSFPENCC 1, 2,
; CHECK-NEXT: TTSFPENCC 2, 8,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: $tt_l0 = TTSFPMOVAll $tt_c9,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 2, i32 8)
  call void @llvm.riscv.tt.bound.sfpencc(i32 2, i32 8)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

; L4 saves the old contents; L6 observes the later masked write. The masked
; move retains L0 in inactive lanes, and an in-place negation is not a self-copy.
; POP restores the outer mask before all three observable Dst stores.
define void @bound_preserve_masked_snapshots() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_preserve_masked_snapshots
; ISEL: usesBoundTensixSFPU: true
; ISEL: $tt_l0 = TTSFPMOV $tt_l0, $tt_l1,
; ISEL-NEXT: $tt_l0 = TTSFPMOV $tt_l0, $tt_l0,
; CHECK-LABEL: name: bound_preserve_masked_snapshots
; CHECK: $tt_l4 = TTSFPMOVAll $tt_l0,
; CHECK-NEXT: TTSFPPUSHC
; CHECK-NEXT: TTSFPSETCCNE $tt_l1,
; CHECK-NEXT: $tt_l0 = TTSFPMOV $tt_l0, $tt_l1,
; CHECK-NEXT: $tt_l6 = TTSFPMOVAll $tt_l0,
; CHECK-NEXT: $tt_l0 = TTSFPMOVNeg $tt_l0, $tt_l0,
; CHECK-NEXT: TTSFPPOPC
; CHECK-NEXT: TTSFPSTORE $tt_l4, 2, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l6, 3, 0, 4,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 4, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 0)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 1, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 3, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 4, i32 0, i32 4)
  ret void
}

define void @bound_keep_cc_transitions() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: bound_keep_cc_transitions
; CHECK: TTSFPENCC 0, 1,
; CHECK-NEXT: TTSFPENCC 0, 1,
; CHECK-NEXT: TTSFPENCC 2, 9,
; CHECK-NEXT: TTSFPENCC 2, 9,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: TTSFPENCC 1, 10,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: TTSFPSETCCNE $tt_c9,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpencc(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpencc(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 9, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  ret void
}

; Removing the redundant copy must not also remove its required issue spacing.
define void @bound_cleanup_repairs_gap() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_cleanup_repairs_gap
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

; The entry ENCC and body ENCC are separated by a real zero-trip edge and a
; backedge. Only the adjacent pair inside the existing body is redundant.
define void @bound_cleanup_loop(i32 %count) "tensix-executor"="trisc1" {
; CHECK-LABEL: name: bound_cleanup_loop
; CHECK: TTSFPENCC 3, 10,
; CHECK: bb.{{[0-9]+}}.loop:
; CHECK: TTSFPENCC 3, 10,
; CHECK-NOT: TTSFPENCC
; CHECK-NOT: TTSFPMOV
; CHECK: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; CHECK-NOT: TTSFPENCC
; CHECK-NOT: TTSFPMOV
; CHECK-NOT: TTSFPIADD
; CHECK: BNE {{.*}}%bb.{{[0-9]+}}
; CHECK: bb.{{[0-9]+}}.exit:
; CHECK: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK: PseudoRET
entry:
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  %next = add i32 %remaining, -1
  %more = icmp ne i32 %next, 0
  br i1 %more, label %loop, label %exit, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

; An explicit ordinary recording owns issue count for the whole function.
; Cleanup must retain otherwise redundant bound operations outside the record.
define void @bound_cleanup_explicit_owner() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: bound_cleanup_explicit_owner
; CHECK: TTREPLAY 1, 0, 1, 0,
; CHECK-NEXT: TTNOP
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: $tt_l0 = TTSFPMOV $tt_l0, $tt_l0,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: TTSFPSTORE $tt_l0, 0, 0, 4,
; CHECK-NEXT: PseudoRET
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
