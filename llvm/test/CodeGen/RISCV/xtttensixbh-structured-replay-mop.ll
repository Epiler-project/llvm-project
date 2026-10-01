; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/mixed.ll -o - | FileCheck %s --check-prefix=PREPARED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %t/mixed.ll -o - | FileCheck %s --check-prefix=PREPARED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/mixed.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/mixed.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/mop-in-body.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
;
; A structured replay template and an independent MOP owner may share one
; function. The dynamic MOP fields stay scalar SSA values; they are not replay
; words and do not make the template body dynamic or duplicated. MOP slot
; programming remains outside the recording region. There is no loop expansion.
;
; PREPARED-LABEL: name: mixed_structured_dynamic_mop
; PREPARED: PseudoTTReplayTemplateBegin 801, 0, 1, 5,
; PREPARED: PseudoTTReplayTemplateWord 801,
; PREPARED: PseudoTTReplayTemplateEnd 801,
; PREPARED: PseudoTTReplayTemplateExecute 801,
; PREPARED: PseudoTTMOPPort 0,
; PREPARED-LABEL: name: mixed_structured_dynamic_mop_record_and_execute
; PREPARED: PseudoTTReplayTemplateBegin 802, 1, 1, 5,
; PREPARED: $tt_l0 = TTSFPMOVAll $tt_l1,
; PREPARED: PseudoTTReplayTemplateEnd 802,
; PREPARED: PseudoTTReplayTemplateExecute 802,
; PREPARED: PseudoTTMOPPort 0,
;
; REJECT: structured replay recording cannot contain MOP

;--- mixed.ll
declare void @llvm.riscv.tt.nop.mop(i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)

define void @mixed_structured_dynamic_mop(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.nop.mop(i32 2)
  call void @llvm.riscv.tt.nop.mop(i32 3)
  call void @llvm.riscv.tt.nop.mop(i32 4)
  call void @llvm.riscv.tt.nop.mop(i32 5)
  call void @llvm.riscv.tt.nop.mop(i32 6)
  call void @llvm.riscv.tt.nop.mop(i32 7)
  call void @llvm.riscv.tt.nop.mop(i32 8)
  call void @llvm.riscv.tt.replay.template.begin(i32 801, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 801)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %z = and i32 %i, 65535
  %l = and i32 %i, 127
  call void @llvm.riscv.tt.replay.template.execute(i32 801)
  call void @llvm.riscv.tt.mop.port(i32 0, i32 %z, i32 %l, i32 1)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

define void @mixed_structured_dynamic_mop_record_and_execute(i32 %zmask, i32 %loop) "tensix-executor"="trisc1" {
  %fz = freeze i32 %zmask
  %fl = freeze i32 %loop
  %z = and i32 %fz, 65535
  %l = and i32 %fl, 127
  call void @llvm.riscv.tt.nop.mop(i32 2)
  call void @llvm.riscv.tt.nop.mop(i32 3)
  call void @llvm.riscv.tt.nop.mop(i32 4)
  call void @llvm.riscv.tt.nop.mop(i32 5)
  call void @llvm.riscv.tt.nop.mop(i32 6)
  call void @llvm.riscv.tt.nop.mop(i32 7)
  call void @llvm.riscv.tt.nop.mop(i32 8)
  call void @llvm.riscv.tt.replay.template.begin(i32 802, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 802)
  call void @llvm.riscv.tt.replay.template.execute(i32 802)
  call void @llvm.riscv.tt.mop.port(i32 0, i32 %z, i32 %l, i32 1)
  ret void
}

;--- mop-in-body.ll

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop.mop(i32 immarg)

define void @mop_inside_structured_replay() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 803, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.nop.mop(i32 2)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 803)
  ret void
}
