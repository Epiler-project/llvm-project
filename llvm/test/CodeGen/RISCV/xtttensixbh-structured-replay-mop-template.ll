; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=PREPARED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-template-prepare %s -o - | FileCheck %s --check-prefix=PREPARED
;
; A template identity in a MOP slot is retained until that slot is replaced.
; MOP reads the template at its actual execution point, not at configuration.
; The body contains ordinary engine instructions and uses no SFPU value ABI.
; There are no guessed lengths or replay locations in the source.

declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.mop(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.nop.mop(i32 immarg)
declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.unpacr(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.rdcfg(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.adddmareg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.wrcfg(i32 immarg, i32 immarg, i32 immarg)

define void @two_unpack_contexts() "tensix-executor"="trisc0" {
; PREPARED-LABEL: name: two_unpack_contexts
; PREPARED: PseudoTTReplayTemplateBegin 200,
; PREPARED-COUNT-6: PseudoTTReplayTemplateWord 200, {{[0-9, ]+}}implicit-def $tt_issue, implicit $tt_issue{{$}}
; PREPARED-NEXT: PseudoTTReplayTemplateEnd 200, implicit-def $tt_issue, implicit $tt_issue{{$}}
; PREPARED-NEXT: PseudoTTReplayTemplateBegin 201,
; PREPARED-COUNT-6: PseudoTTReplayTemplateWord 201, {{[0-9, ]+}}implicit-def $tt_issue, implicit $tt_issue{{$}}
; PREPARED-NEXT: PseudoTTReplayTemplateEnd 201, implicit-def $tt_issue, implicit $tt_issue{{$}}
; PREPARED: PseudoTTReplayTemplateMop 200, 3, implicit-def $tt_config, implicit-def $tt_issue, implicit $tt_config, implicit $tt_issue :: (volatile store (s32)){{$}}
; PREPARED: PseudoTTReplayTemplateMop 201, 7, implicit-def $tt_config, implicit-def $tt_issue, implicit $tt_config, implicit $tt_issue :: (volatile store (s32)){{$}}
; PREPARED: TTMOP 0, 0, 0, implicit-def $tt_config, implicit-def $tt_dst, implicit-def $tt_issue, implicit $tt_config, implicit $tt_dst, implicit $tt_issue :: (volatile load store (s32)){{$}}
; CHECK-LABEL: name: two_unpack_contexts
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 6, 0,
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 6, 6,
; CHECK: PseudoTTREPLAYMop 3, 0, 0, 6, 0,
; CHECK: PseudoTTREPLAYMop 7, 0, 0, 6, 6,
; CHECK: TTMOP 0, 0, 0,
  call void @llvm.riscv.tt.replay.template.begin(i32 200, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.rdcfg(i32 76, i32 12)
  call void @llvm.riscv.tt.adddmareg(i32 36, i32 12, i32 12, i32 0)
  call void @llvm.riscv.tt.stallwait(i32 1, i32 128)
  call void @llvm.riscv.tt.wrcfg(i32 76, i32 0, i32 12)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 200)
  call void @llvm.riscv.tt.replay.template.begin(i32 201, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.rdcfg(i32 77, i32 12)
  call void @llvm.riscv.tt.adddmareg(i32 36, i32 12, i32 12, i32 0)
  call void @llvm.riscv.tt.stallwait(i32 1, i32 128)
  call void @llvm.riscv.tt.wrcfg(i32 77, i32 0, i32 12)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 201)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 200, i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 201, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 0)
  ret void
}

define void @slot_clear_allows_reuse() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: slot_clear_allows_reuse
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; CHECK: PseudoTTREPLAYMop 3, 0, 0, 1, 31,
; CHECK: TTMOP 0, 0, 1,
; CHECK: PseudoTTMOPClear 3,
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; CHECK: PseudoTTREPLAYMop 3, 0, 0, 1, 31,
; CHECK: TTMOP 0, 0, 1,
  call void @llvm.riscv.tt.replay.template.begin(i32 300, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 300)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 300, i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.replay.template.begin(i32 301, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 301)
  call void @llvm.riscv.tt.replay.template.mop(i32 301, i32 3)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ret void
}

define void @transpose_loop(i32 %count) "tensix-executor"="trisc0" {
; CHECK-LABEL: name: transpose_loop
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 2, 0,
; CHECK: PseudoTTREPLAYMop 5, 0, 0, 2, 0,
; CHECK: PseudoTTREPLAYMop 7, 0, 0, 2, 0,
; CHECK: PseudoTTREPLAYMop 8, 0, 0, 2, 0,
; CHECK: bb.{{[0-9]+}}.loop:
; CHECK: TTMOP 0, 0, 1,
; CHECK: {{BNE|BLTU|BLT}} {{.*}}%bb.
entry:
  call void @llvm.riscv.tt.replay.template.begin(i32 400, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.unpacr.nop(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 0, i32 0, i32 2, i32 0)
  call void @llvm.riscv.tt.replay.template.end(i32 400)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.replay.template.mop(i32 400, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.replay.template.mop(i32 400, i32 7)
  call void @llvm.riscv.tt.replay.template.mop(i32 400, i32 8)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

define void @diamond_slot_selection(i1 noundef %condition) "tensix-executor"="trisc0" {
; CHECK-LABEL: name: diamond_slot_selection
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 0,
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 1,
; CHECK-DAG: PseudoTTREPLAYMop 3, 0, 0, 1, 0,
; CHECK-DAG: PseudoTTREPLAYMop 3, 0, 0, 1, 1,
; CHECK: bb.{{[0-9]+}}.join:
; CHECK: TTMOP 0, 0, 1,
entry:
  call void @llvm.riscv.tt.replay.template.begin(i32 501, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 501)
  call void @llvm.riscv.tt.replay.template.begin(i32 502, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 502)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  br i1 %condition, label %left, label %right
left:
  call void @llvm.riscv.tt.replay.template.mop(i32 501, i32 3)
  br label %join
right:
  call void @llvm.riscv.tt.replay.template.mop(i32 502, i32 3)
  br label %join
join:
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ret void
}

define void @ordinary_slot_overwrite() "tensix-executor"="trisc0" {
; CHECK-LABEL: name: ordinary_slot_overwrite
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; CHECK: PseudoTTREPLAYMop 3, 0, 0, 1, 31,
; CHECK: TTMOP 0, 0, 1,
; CHECK: PseudoTTNOPMop 3,
; CHECK: PseudoTTExplicitSFPUReplay 1, 0, 1, 31,
; CHECK: TTMOP 0, 0, 1,
; CHECK: PseudoTTREPLAYMop 3, 0, 0, 1, 31,
; CHECK: TTMOP 0, 0, 1,
  call void @llvm.riscv.tt.replay.template.begin(i32 601, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 601)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.replay.template.mop(i32 601, i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.nop.mop(i32 3)
  call void @llvm.riscv.tt.replay.template.begin(i32 602, i32 0, i32 1, i32 31)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.template.end(i32 602)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.mop(i32 602, i32 3)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ret void
}
