; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/capacity.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAPACITY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/raw-gap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RAW-GAP
;
; CONFIG uses physical L0 outside the MAD forwarding path. Reuse the ordinary
; hazard owner for entry, internal and exit dependencies; finalize length only
; after internal spacing. ENCC is a real issue that fills one native gap, while
; a record-only payload does not advance the current arithmetic pipeline.
;
; CHECK-LABEL: name: lane_internal_gap
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 3, 29,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK: PseudoTTExplicitSFPUReplay 0, 0, 3, 29,
; CHECK-LABEL: name: creg_internal_gap
; CHECK: PseudoTTExplicitSFPUReplay 1, 1, 3, 29,
; CHECK-NEXT: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPCONFIGC12 21844, 8,
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK: PseudoTTExplicitSFPUReplay 0, 0, 3, 29,
; CHECK-LABEL: name: config_execute_entry
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; CHECK-LABEL: name: config_record_and_execute_entry
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 1, 1, 1, 5,
; CHECK-NEXT: TTSFPCONFIGLane 21844, 8,
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-LABEL: name: config_consumes_replay_exit
; CHECK: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPCONFIGC11 21844, 8,
; CHECK-LABEL: name: encc_is_internal_gap
; CHECK: PseudoTTExplicitSFPUReplay 1, 1, 3, 29,
; CHECK-NEXT: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPENCC 3, 10,
; CHECK-NEXT: TTSFPCONFIGLane 21844, 8,
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-LABEL: name: record_only_does_not_clear_pending
; CHECK: $tt_l0 = TTSFPMAD
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 1, 0, 1, 5,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD
; CAPACITY: structured replay fixed placement conflicts or exceeds capacity
; RAW-GAP: explicit SFPU replay template requires an authored hazard gap

;--- valid.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)

define void @lane_internal_gap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 701, i32 0, i32 1, i32 29)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 701)
  call void @llvm.riscv.tt.replay.template.execute(i32 701)
  ret void
}

define void @creg_internal_gap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 702, i32 1, i32 1, i32 29)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 12, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 702)
  call void @llvm.riscv.tt.replay.template.execute(i32 702)
  ret void
}

define void @config_execute_entry() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 703, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 13, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 703)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.replay.template.execute(i32 703)
  ret void
}

define void @config_record_and_execute_entry() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.replay.template.begin(i32 704, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 704)
  ret void
}

define void @config_consumes_replay_exit() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 705, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.replay.template.end(i32 705)
  call void @llvm.riscv.tt.replay.template.execute(i32 705)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 21844, i32 8)
  ret void
}

define void @encc_is_internal_gap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 706, i32 1, i32 1, i32 29)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 706)
  ret void
}

define void @record_only_does_not_clear_pending() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.replay.template.begin(i32 707, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 707)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  ret void
}

;--- capacity.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
define void @config_gap_overflows() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 708, i32 0, i32 1, i32 30)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 708)
  ret void
}

;--- raw-gap.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @raw_config_gap_not_authored() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 2, i32 5)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 1, i32 2, i32 3, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 2, i32 5)
  ret void
}
