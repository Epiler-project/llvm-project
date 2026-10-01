; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/valid.ll -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/capacity.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CAPACITY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/conflict.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFLICT
;
; Source body has two words. MAD -> IADD needs one native issue gap, so length
; must be finalized to THREE. This is instruction legalization, not duplication
; of an algorithm iteration. Slot29 fits; slot30 must fail rather than silently
; retaining length2 or dropping recording. Record-only must capture the NOP;
; record-and-execute must execute it as well as capturing it.
;
; CHECK-LABEL: name: record_only_internal_gap
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 3, 29,
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTSFPURecordWord
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK: PseudoTTExplicitSFPUReplay 0, 0, 3, 29,
; CHECK-LABEL: name: record_and_execute_internal_gap
; CHECK: PseudoTTExplicitSFPUReplay 1, 1, 3, 29,
; CHECK-NEXT: $tt_l0 = TTSFPMAD
; CHECK-NEXT: TTSFPNOP
; CHECK-NEXT: $tt_l0 = TTSFPIADD
; CHECK-NEXT: PseudoTTReplayRecordEnd
; CHECK: PseudoTTExplicitSFPUReplay 0, 0, 3, 29,
; CAPACITY: structured replay fixed placement conflicts or exceeds capacity
; CONFLICT: structured replay fixed placement conflicts or exceeds capacity

;--- valid.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @record_only_internal_gap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 33, i32 0, i32 1, i32 29)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 33)
  call void @llvm.riscv.tt.replay.template.execute(i32 33)
  ret void
}

define void @record_and_execute_internal_gap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 34, i32 1, i32 1, i32 29)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 34)
  call void @llvm.riscv.tt.replay.template.execute(i32 34)
  ret void
}

;--- capacity.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmad(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @gap_overflows_fixed_range() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 35, i32 0, i32 1, i32 30)
  call void @llvm.riscv.tt.bound.sfpmad(i32 0, i32 0, i32 10, i32 9, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 9, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 35)
  call void @llvm.riscv.tt.replay.template.execute(i32 35)
  ret void
}

;--- conflict.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

define void @live_fixed_ranges_overlap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 36, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 36)
  call void @llvm.riscv.tt.replay.template.begin(i32 37, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 37)
  call void @llvm.riscv.tt.replay.template.execute(i32 36)
  call void @llvm.riscv.tt.replay.template.execute(i32 37)
  ret void
}
