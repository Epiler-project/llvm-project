; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not='class: sfpr' --implicit-check-not=PseudoTTBound
; RUN: split-file %S/xtttensixbh-sfpu-float-replay.mir %t.check
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-replay-selection %s -o %t.o0.before.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-replay-selection %s -o %t.o2.before.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o %t.o0.selected.mir
; RUN: FileCheck %s --check-prefixes=SELECT,SELECT-O0 < %t.o0.selected.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o %t.o2.selected.mir
; RUN: FileCheck %s --check-prefixes=SELECT,SELECT-O2 < %t.o2.selected.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-hazards,1 %s -o %t.o0.plain.mir
; RUN: FileCheck %s --check-prefix=PLAIN < %t.o0.plain.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-hazards,1 %s -o %t.o2.plain.mir
; RUN: FileCheck %s --check-prefix=PLAIN < %t.o2.plain.mir
; RUN: %python %t.check/check.py O0 %t.o0.before.mir %t.o0.selected.mir %t.o0.plain.mir
; RUN: %python %t.check/check.py O2 %t.o2.before.mir %t.o2.selected.mir %t.o2.plain.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Both sequences below are authored arithmetic, including the L7 snapshot.
; Floating replay keeps physical destination/old ties and the current predicate.
; The second function retains a real, runtime-bounded loop: two authored
; arithmetic sequences per iteration are compressed without copying an iteration.
; Mutable Dst storage, CC changes and the snapshot remain outside recording.
; O2 schedules the scalar loop increment before the last MAD. That scalar
; boundary ends the second candidate at six slots, leaving both mode-2 MADs
; outside replay in their original order. O0 records all seven slots. Including
; both execute fences, the loop saves two words at O2 and three words at O0.

declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpsetcc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @bound_float_replay() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_float_replay
; ISEL: $tt_l7 = TTSFPMOVAll $tt_l0,
; ISEL: $tt_l0 = TTSFPADD $tt_l0, $tt_l1, $tt_c10, 1,
; ISEL-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 2,
; ISEL-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 3,
; ISEL: TTSFPSTORE $tt_l7, 20, 0, 3,
; SELECT-LABEL: name: bound_float_replay
; SELECT: $tt_l7 = TTSFPMOVAll $tt_l0,
; SELECT: TTSFPSETCCNE $tt_l1,
; SELECT: PseudoTTSFPUReplay 1, 1, 7, 0,
; SELECT-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; SELECT-NEXT: $tt_l0 = TTSFPADD $tt_l0, $tt_l1, $tt_c10, 1,
; SELECT-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 2,
; SELECT-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 3,
; SELECT-NEXT: $tt_l0 = TTSFPADD $tt_l0, $tt_l1, $tt_c10, 0,
; SELECT-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 1,
; SELECT-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 2,
; SELECT-NEXT: PseudoTTReplayRecordEnd
; SELECT-NEXT: TTSFPNOP
; SELECT-NEXT: PseudoTTSFPUReplay 0, 0, 7, 0,
; SELECT-NEXT: TTSFPNOP
; SELECT-NOT: TTSFPMOV
; SELECT: TTSFPPOPC
; SELECT: TTSFPSTORE $tt_l7, 20, 0, 3,
; PLAIN-LABEL: name: bound_float_replay
; PLAIN-NOT: PseudoTTSFPUReplay
; PLAIN: TTSFPSETCCNE
; PLAIN: TTSFPIADD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN: TTSFPIADD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN-NOT: PseudoTTSFPUReplay
; PLAIN: TTSFPSTORE $tt_l7, 20, 0, 3,
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 4, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 7, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 1, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 1)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 3)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 2)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 1)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 3)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 2)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 12, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 16, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpstore(i32 7, i32 20, i32 0, i32 3)
  ret void
}

define void @bound_float_loop(i32 %count) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: bound_float_loop
; ISEL: $tt_l7 = TTSFPMOVAll $tt_l0,
; ISEL: $tt_l0 = TTSFPADD $tt_l0, $tt_l1, $tt_c10, 1,
; ISEL-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 2,
; ISEL-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 3,
; ISEL: TTSFPSTORE $tt_l7, 20, 0, 3,
; SELECT-LABEL: name: bound_float_loop
; SELECT: $tt_l7 = TTSFPMOVAll $tt_l0,
; SELECT: TTSFPSETCCNE $tt_l1,
; SELECT: BEQ {{.*}}, $x0, %bb.[[EXIT:[0-9]+]]
; SELECT: bb.[[LOOP:[0-9]+]].loop:
; SELECT-NEXT: successors: %bb.[[LOOP]]{{[^,]*}}, %bb.[[EXIT]]
; SELECT-O0: PseudoTTSFPUReplay 1, 1, 7, 0,
; SELECT-O2: PseudoTTSFPUReplay 1, 1, 6, 0,
; SELECT-NEXT: $tt_l0 = TTSFPIADD $tt_l0, $tt_c9,
; SELECT-NEXT: $tt_l0 = TTSFPADD $tt_l0, $tt_l1, $tt_c10, 1,
; SELECT-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 2,
; SELECT-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 3,
; SELECT-NEXT: $tt_l0 = TTSFPADD $tt_l0, $tt_l1, $tt_c10, 0,
; SELECT-NEXT: $tt_l2 = TTSFPMUL $tt_l2, $tt_l0, $tt_l1, 1,
; SELECT-O0-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 2,
; SELECT-NEXT: PseudoTTReplayRecordEnd
; SELECT-O2-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 2,
; SELECT-NEXT: TTSFPNOP
; SELECT-O0-NEXT: PseudoTTSFPUReplay 0, 0, 7, 0,
; SELECT-O2-NEXT: PseudoTTSFPUReplay 0, 0, 6, 0,
; SELECT-NEXT: TTSFPNOP
; SELECT-O0: BLTU {{.*}}, %bb.[[LOOP]]
; SELECT-O2-NEXT: renamable $[[INDEX:x[0-9]+]] = ADDI killed renamable $[[INDEX]], 1
; SELECT-O2-NEXT: $tt_l4 = TTSFPMAD $tt_l4, $tt_l1, $tt_l2, $tt_c9, 2,
; SELECT-O2-NEXT: BLTU renamable $[[INDEX]], {{.*}}, %bb.[[LOOP]]
; SELECT-NOT: TTSFPMOV
; SELECT: bb.[[EXIT]].exit:
; SELECT-NEXT: TTSFPPOPC
; SELECT: TTSFPSTORE $tt_l7, 20, 0, 3,
; PLAIN-LABEL: name: bound_float_loop
; PLAIN-NOT: PseudoTTSFPUReplay
; PLAIN: TTSFPSETCCNE
; PLAIN: TTSFPIADD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN: TTSFPIADD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN: TTSFPADD
; PLAIN: TTSFPMUL
; PLAIN: TTSFPMAD
; PLAIN-NOT: PseudoTTSFPUReplay
; PLAIN: TTSFPSTORE $tt_l7, 20, 0, 3,
entry:
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 4, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 7, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 1, i32 0, i32 2)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %index = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 1)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 3)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 2)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 1)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 3)
  call void @llvm.riscv.tt.bound.sfpadd(i32 0, i32 0, i32 1, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmul(i32 2, i32 2, i32 0, i32 1, i32 1)
  call void @llvm.riscv.tt.bound.sfpmad(i32 4, i32 4, i32 1, i32 2, i32 9, i32 2)
  %next = add i32 %index, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 12, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 16, i32 0, i32 3)
  call void @llvm.riscv.tt.bound.sfpstore(i32 7, i32 20, i32 0, i32 3)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
