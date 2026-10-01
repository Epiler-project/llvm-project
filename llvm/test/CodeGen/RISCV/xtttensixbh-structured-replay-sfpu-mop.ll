; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/record_only_binding.ll -o %t/record_only_binding.mir
; RUN: FileCheck %s --check-prefix=RECORD-ONLY-BINDING --input-file=%t/record_only_binding.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/structured_state_once.ll -o %t/structured_state_once.mir
; RUN: FileCheck %s --check-prefix=STRUCTURED-STATE-ONCE --input-file=%t/structured_state_once.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/raw_state_once.ll -o %t/raw_state_once.mir
; RUN: FileCheck %s --check-prefix=RAW-STATE-ONCE --input-file=%t/raw_state_once.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/zero_outer_preserves_lreg.ll -o %t/zero_outer_preserves_lreg.mir
; RUN: FileCheck %s --check-prefix=ZERO-OUTER-PRESERVES-LREG --input-file=%t/zero_outer_preserves_lreg.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/slot_clear_removes_effects.ll -o %t/slot_clear_removes_effects.mir
; RUN: FileCheck %s --check-prefix=SLOT-CLEAR-REMOVES-EFFECTS --input-file=%t/slot_clear_removes_effects.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/template0_skips_body.ll -o %t/template0_skips_body.mir
; RUN: FileCheck %s --check-prefix=TEMPLATE0-SKIPS-BODY --input-file=%t/template0_skips_body.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/persistent_slot_loop.ll -o %t/persistent_slot_loop.mir
; RUN: FileCheck %s --check-prefix=PERSISTENT-SLOT-LOOP --input-file=%t/persistent_slot_loop.mir
;
; Receiver acceptance fixtures. The controls deliberately use the current TT
; typed control-cell ABI, with explicit values rather than firmware defaults.
;
; Template 1 has outer=inner=1: only slot 7 executes; slots 2/3/4/6 are
; plain NOP and slot 5/8 are not selected. Template 0's mask=1, count=0,
; flags=0 selects only SkipA0 (slot 7), so the body bound at A0 never runs.
; Recording and binding alone cannot overwrite L0 or alter CC/CReg/Dst state.
; MOP uses the current SFPU state at actual execution, including real backedges.
; There is one source loop body and one MOP instruction; no unrolling is allowed.
;
; RECORD-ONLY-BINDING-LABEL: name: record_only_binding
; RECORD-ONLY-BINDING: $tt_l0 = TTSFPMOVAll $tt_c10,
; RECORD-ONLY-BINDING: PseudoTTREPLAYMop 7,
; RECORD-ONLY-BINDING-NOT: implicit-def $tt_l0
; RECORD-ONLY-BINDING: $tt_l2 = TTSFPMOVAll $tt_l0,
; STRUCTURED-STATE-ONCE-LABEL: name: structured_state_once
; STRUCTURED-STATE-ONCE: TTMOP 0, 0, 1,
; STRUCTURED-STATE-ONCE-SAME: implicit $tt_l1
; STRUCTURED-STATE-ONCE-SAME: implicit-def $tt_cc
; STRUCTURED-STATE-ONCE-SAME: implicit-def $tt_c11
; STRUCTURED-STATE-ONCE-SAME: implicit-def $tt_l0
; RAW-STATE-ONCE-LABEL: name: raw_state_once
; RAW-STATE-ONCE: TTMOP 0, 0, 1,
; RAW-STATE-ONCE-SAME: implicit $tt_l1
; RAW-STATE-ONCE-SAME: implicit-def $tt_cc
; RAW-STATE-ONCE-SAME: implicit-def $tt_c11
; RAW-STATE-ONCE-SAME: implicit-def $tt_l0
; ZERO-OUTER-PRESERVES-LREG-LABEL: name: zero_outer_preserves_lreg
; ZERO-OUTER-PRESERVES-LREG: $tt_l0 = TTSFPMOVAll $tt_c10,
; ZERO-OUTER-PRESERVES-LREG: TTMOP 0, 0, 1,
; ZERO-OUTER-PRESERVES-LREG-NOT: implicit-def $tt_l0
; ZERO-OUTER-PRESERVES-LREG: $tt_l2 = TTSFPMOVAll $tt_l0,
; SLOT-CLEAR-REMOVES-EFFECTS-LABEL: name: slot_clear_removes_effects
; SLOT-CLEAR-REMOVES-EFFECTS: TTMOP 0, 0, 1,
; SLOT-CLEAR-REMOVES-EFFECTS-NOT: implicit-def $tt_l0
; SLOT-CLEAR-REMOVES-EFFECTS: $tt_l2 = TTSFPMOVAll $tt_l0,
; TEMPLATE0-SKIPS-BODY-LABEL: name: template0_skips_body
; TEMPLATE0-SKIPS-BODY: TTMOP 1, 0, 0,
; TEMPLATE0-SKIPS-BODY-NOT: implicit-def $tt_l0
; TEMPLATE0-SKIPS-BODY: $tt_l2 = TTSFPMOVAll $tt_l0,
; PERSISTENT-SLOT-LOOP-LABEL: name: persistent_slot_loop
; PERSISTENT-SLOT-LOOP: PseudoTTREPLAYMop 7,
; PERSISTENT-SLOT-LOOP: bb.{{[0-9]+}}.loop:
; PERSISTENT-SLOT-LOOP: TTMOP 0, 0, 1,
; PERSISTENT-SLOT-LOOP-SAME: implicit $tt_l1
; PERSISTENT-SLOT-LOOP-SAME: implicit-def $tt_l0
; PERSISTENT-SLOT-LOOP: {{BNE|BLTU|BLT}} {{.*}}%bb.

;--- record_only_binding.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @record_only_binding() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 901, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}


;--- structured_state_once.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @structured_state_once() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 901, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}


;--- raw_state_once.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @raw_state_once() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 5, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpconfig.creg(i32 0, i32 11, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpconfig.lane(i32 0, i32 21844, i32 8)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.mop(i32 7, i32 0, i32 0, i32 5, i32 0)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}


;--- zero_outer_preserves_lreg.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @zero_outer_preserves_lreg() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 0)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 901, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}


;--- slot_clear_removes_effects.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @slot_clear_removes_effects() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 901, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}


;--- template0_skips_body.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @template0_skips_body() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 0)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 901, i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 1, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}


;--- persistent_slot_loop.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.replay.mop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.creg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpconfig.lane(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @persistent_slot_loop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.replay.template.begin(i32 901, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 901)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 901, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 0)
  ret void
}
