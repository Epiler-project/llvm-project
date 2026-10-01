; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=SELECT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=SELECT
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=PREPARED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=PREPARED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=FINAL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=FINAL
;
; Begin/execute are actual issue boundaries, with the same unknown-address
; volatile load/store MMO as ordinary REPLAY. Selection must create that MMO
; before preparation and finalization can preserve it. An end marker has only
; issue ordering, and a record-only payload must not acquire memory effects.
; Scalar volatile stores outside the recording remain on their authored sides
; of both actual controls. No numerical memory effects are assigned to end.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

define void @record_only_memory(ptr %p) "tensix-executor"="trisc1" {
; SELECT-LABEL: name: record_only_memory
; SELECT: SW {{.*}} :: (volatile store (s32) into %ir.p)
; SELECT: PseudoTTReplayTemplateBegin 503, 0, 1, 5, {{.*}} :: (volatile load store (s32))
; SELECT: $tt_l0 = TTSFPMOVAll $tt_l1,{{[^:]*$}}
; SELECT: PseudoTTReplayTemplateEnd 503,{{[^:]*$}}
; SELECT: SW {{.*}} :: (volatile store (s32) into %ir.p)
; SELECT: PseudoTTReplayTemplateExecute 503, {{.*}} :: (volatile load store (s32))
; SELECT: SW {{.*}} :: (volatile store (s32) into %ir.p)
; PREPARED-LABEL: name: record_only_memory
; PREPARED: SW {{.*}} :: (volatile store (s32) into %ir.p)
; PREPARED: PseudoTTReplayTemplateBegin 503, 0, 1, 5, {{.*}} :: (volatile load store (s32))
; PREPARED: PseudoTTReplayTemplateWord 503,{{[^:]*$}}
; PREPARED: PseudoTTReplayTemplateEnd 503,{{[^:]*$}}
; PREPARED: SW {{.*}} :: (volatile store (s32) into %ir.p)
; PREPARED: PseudoTTReplayTemplateExecute 503, {{.*}} :: (volatile load store (s32))
; PREPARED: SW {{.*}} :: (volatile store (s32) into %ir.p)
; FINAL-LABEL: name: record_only_memory
; FINAL: SW {{.*}} :: (volatile store (s32) into %ir.p)
; FINAL: PseudoTTExplicitSFPUReplay 1, 0, 1, 5, {{.*}} :: (volatile load store (s32))
; FINAL: PseudoTTSFPURecordWord {{[^:]*$}}
; FINAL: PseudoTTReplayRecordEnd {{[^:]*$}}
; FINAL: SW {{.*}} :: (volatile store (s32) into %ir.p)
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 1, 5, {{.*}} :: (volatile load store (s32))
; FINAL: SW {{.*}} :: (volatile store (s32) into %ir.p)
  store volatile i32 17, ptr %p, align 4
  call void @llvm.riscv.tt.replay.template.begin(i32 503, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 503)
  store volatile i32 23, ptr %p, align 4
  call void @llvm.riscv.tt.replay.template.execute(i32 503)
  store volatile i32 31, ptr %p, align 4
  ret void
}

define void @record_and_execute_memory(ptr %p) "tensix-executor"="trisc1" {
; SELECT-LABEL: name: record_and_execute_memory
; SELECT: SW {{.*}} :: (volatile store (s32) into %ir.p)
; SELECT: PseudoTTReplayTemplateBegin 509, 1, 1, 5, {{.*}} :: (volatile load store (s32))
; SELECT: $tt_l0 = TTSFPMOVAll $tt_l1,{{[^:]*$}}
; SELECT: PseudoTTReplayTemplateEnd 509,{{[^:]*$}}
; SELECT: SW {{.*}} :: (volatile store (s32) into %ir.p)
; SELECT: PseudoTTReplayTemplateExecute 509, {{.*}} :: (volatile load store (s32))
; SELECT: SW {{.*}} :: (volatile store (s32) into %ir.p)
; PREPARED-LABEL: name: record_and_execute_memory
; PREPARED: SW {{.*}} :: (volatile store (s32) into %ir.p)
; PREPARED: PseudoTTReplayTemplateBegin 509, 1, 1, 5, {{.*}} :: (volatile load store (s32))
; PREPARED: $tt_l0 = TTSFPMOVAll $tt_l1,{{[^:]*$}}
; PREPARED: PseudoTTReplayTemplateEnd 509,{{[^:]*$}}
; PREPARED: SW {{.*}} :: (volatile store (s32) into %ir.p)
; PREPARED: PseudoTTReplayTemplateExecute 509, {{.*}} :: (volatile load store (s32))
; PREPARED: SW {{.*}} :: (volatile store (s32) into %ir.p)
; FINAL-LABEL: name: record_and_execute_memory
; FINAL: SW {{.*}} :: (volatile store (s32) into %ir.p)
; FINAL: PseudoTTExplicitSFPUReplay 1, 1, 1, 5, {{.*}} :: (volatile load store (s32))
; FINAL: $tt_l0 = TTSFPMOVAll $tt_l1,{{[^:]*$}}
; FINAL: PseudoTTReplayRecordEnd {{[^:]*$}}
; FINAL: SW {{.*}} :: (volatile store (s32) into %ir.p)
; FINAL: PseudoTTExplicitSFPUReplay 0, 0, 1, 5, {{.*}} :: (volatile load store (s32))
; FINAL: SW {{.*}} :: (volatile store (s32) into %ir.p)
  store volatile i32 37, ptr %p, align 4
  call void @llvm.riscv.tt.replay.template.begin(i32 509, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 509)
  store volatile i32 41, ptr %p, align 4
  call void @llvm.riscv.tt.replay.template.execute(i32 509)
  store volatile i32 43, ptr %p, align 4
  ret void
}
