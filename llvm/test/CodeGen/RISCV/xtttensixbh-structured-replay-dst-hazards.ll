; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Dst readiness crosses replay execution and recording boundaries. A Matrix
; write needs three issue gaps before SFPLOAD. Required entry gaps belong
; before the replay header, never inside an already counted recording. A
; record-only load has no current Dst read, and a store-only replay must retain
; the remaining readiness delay for a later ordinary load. No source loop or
; body iteration may be duplicated to obtain these native gaps.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mvmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @dst_load_execute_entry() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: dst_load_execute_entry
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 5,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: TTMVMUL
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; CHECK-NEXT: $tt_l1 = TTSFPLOAD
  call void @llvm.riscv.tt.replay.template.begin(i32 211, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 211)
  call void @llvm.riscv.tt.mvmul(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.execute(i32 211)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 40, i32 0, i32 4)
  ret void
}

define void @dst_load_record_and_execute_entry() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: dst_load_record_and_execute_entry
; CHECK: TTMVMUL
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 1, 1, 1, 5,
; CHECK-NEXT: $tt_l0 = TTSFPLOAD
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
  call void @llvm.riscv.tt.mvmul(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.begin(i32 212, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 212)
  call void @llvm.riscv.tt.replay.template.execute(i32 212)
  ret void
}

define void @dst_record_only_has_no_current_read() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: dst_record_only_has_no_current_read
; CHECK: TTMVMUL
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 1, 0, 1, 5,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l1 = TTSFPLOAD
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
  call void @llvm.riscv.tt.mvmul(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.begin(i32 213, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 213)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 213)
  ret void
}

define void @dst_load_after_recorded_issue() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: dst_load_after_recorded_issue
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 2, 5,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: TTMVMUL
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 2, 5,
  call void @llvm.riscv.tt.replay.template.begin(i32 214, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 214)
  call void @llvm.riscv.tt.mvmul(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.execute(i32 214)
  ret void
}

define void @dst_store_replay_preserves_exit_delay() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: dst_store_replay_preserves_exit_delay
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 5,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK-NEXT: TTMVMUL
; CHECK-NEXT: PseudoTTExplicitSFPUReplay 0, 0, 1, 5,
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l1 = TTSFPLOAD
  call void @llvm.riscv.tt.replay.template.begin(i32 215, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 215)
  call void @llvm.riscv.tt.mvmul(i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.replay.template.execute(i32 215)
  call void @llvm.riscv.tt.bound.sfpload(i32 1, i32 1, i32 40, i32 0, i32 4)
  ret void
}

define void @static_dst_retains_internal_gap() "tensix-executor"="trisc1" {
; CHECK-LABEL: name: static_dst_retains_internal_gap
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 4, 28,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK: PseudoTTExplicitSFPUReplay 0, 0, 4, 28,
  call void @llvm.riscv.tt.replay.template.begin(i32 216, i32 0, i32 1, i32 28)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 216)
  call void @llvm.riscv.tt.replay.template.execute(i32 216)
  ret void
}
