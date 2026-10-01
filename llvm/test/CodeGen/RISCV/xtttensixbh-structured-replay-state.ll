; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t.o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t.o2.mir
; RUN: %python %S/Inputs/tensix-replay-state-observer.py %t.o0.s %t.o0.mir
; RUN: %python %S/Inputs/tensix-replay-state-observer.py %t.o2.s %t.o2.mir
; RUN: %python %S/Inputs/tensix-replay-state-effects.py %t.o0.mir %t.effect-mutations llc
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
;
; Explicit physical state is a runtime input, never an instruction-field capture.
; The independent bounded observer checks all 32 lanes for two initial contents
; and exact native effects for each recording/execution mode and ingress.
; ENCC's two authored executes toggle CURRENT enable twice. CONFIG C11..C14
; reads CURRENT L0, uses first-row CC, broadcasts selected columns and preserves
; the other columns. LaneConfig changes STORE/ROW_MASK behavior; reset ignores
; ROW_MASK in its own predicate and preserves the high configuration bits.
; None of these cases depends on an inferred initialization, scratch LReg,
; implicit copy, virtual SFPR, loop expansion or device simulator.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.reset()
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @state_encc_structured_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 601, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 9)
  call void @llvm.riscv.tt.replay.template.end(i32 601)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 601)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.replay.template.execute(i32 601)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 16, i32 0, i32 4)
  ret void
}

define void @state_encc_structured_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 601, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 9)
  call void @llvm.riscv.tt.replay.template.end(i32 601)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 601)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.replay.template.execute(i32 601)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 16, i32 0, i32 4)
  ret void
}

define void @state_encc_raw_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 4)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 9)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 16, i32 0, i32 4)
  ret void
}

define void @state_encc_raw_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 4)
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 9)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 10, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 2)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov(i32 2, i32 2, i32 9, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 2)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 6, i32 16, i32 0, i32 4)
  ret void
}

define void @state_creg_structured_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.begin(i32 607, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 12, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 13, i32 1, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 14, i32 0, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 607)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay.template.execute(i32 607)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 56, i32 0, i32 4)
  ret void
}

define void @state_creg_structured_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.begin(i32 607, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 12, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 13, i32 1, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 14, i32 0, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 607)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay.template.execute(i32 607)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 56, i32 0, i32 4)
  ret void
}

define void @state_creg_raw_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 4, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 12, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 13, i32 1, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 14, i32 0, i32 8)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 4, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 56, i32 0, i32 4)
  ret void
}

define void @state_creg_raw_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 4, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 12, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 13, i32 1, i32 8)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 14, i32 0, i32 8)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 4, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 11)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 12)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 13)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 14)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 48, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 5, i32 56, i32 0, i32 4)
  ret void
}

define void @state_lane_structured_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.begin(i32 613, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 613)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay.template.execute(i32 613)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 617, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.reset()
  call void @llvm.riscv.tt.replay.template.end(i32 617)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 617)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 24, i32 0, i32 4)
  ret void
}

define void @state_lane_structured_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.begin(i32 613, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.template.end(i32 613)
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay.template.execute(i32 613)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 617, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.reset()
  call void @llvm.riscv.tt.replay.template.end(i32 617)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 617)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 24, i32 0, i32 4)
  ret void
}

define void @state_lane_raw_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.reset()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 24, i32 0, i32 4)
  ret void
}

define void @state_lane_raw_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpencc(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 6)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.replay(i32 1, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpconfig.reset()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpstore(i32 2, i32 24, i32 0, i32 4)
  ret void
}
