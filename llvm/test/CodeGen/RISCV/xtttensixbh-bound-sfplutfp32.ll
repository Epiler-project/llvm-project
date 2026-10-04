; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL --implicit-check-not=':sfpr' --implicit-check-not=PseudoTTBound
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=HAZARD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.riscv.tt.bound.sfplutfp32.3r(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfplutfp32.6r(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)

define void @direct_lut_writes_destination() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: direct_lut_writes_destination
; HAZARD-LABEL: name: direct_lut_writes_destination
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 7, i32 9)
; ISEL: $tt_l5 = TTSFPLUTFP32Six $tt_l5, 6, implicit-def $tt_issue, implicit $tt_l0, implicit $tt_l1, implicit $tt_l2, implicit $tt_l3, implicit $tt_l4, implicit $tt_l5, implicit $tt_l6
; ISEL-NEXT: $tt_l5 = TTSFPIADD $tt_l5, $tt_c9
; HAZARD: $tt_l5 = TTSFPLUTFP32Six $tt_l5, 6
; HAZARD-NEXT: TTSFPNOP
; HAZARD-NEXT: $tt_l5 = TTSFPIADD $tt_l5, $tt_c9
  call void @llvm.riscv.tt.bound.sfplutfp32.6r(i32 5, i32 5, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 6)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 5, i32 5, i32 9, i32 4)
  ret void
}

define void @indirect_lut_writes_group() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: indirect_lut_writes_group
; HAZARD-LABEL: name: indirect_lut_writes_group
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 7, i32 9)
; ISEL: TTSFPLUTFP32Three 1, implicit-def $tt_l0, implicit-def $tt_l1, implicit-def $tt_l2, implicit-def $tt_l3, implicit-def $tt_l4, implicit-def $tt_l5, implicit-def $tt_l6, implicit-def $tt_l7, implicit-def $tt_issue
; ISEL-SAME: implicit $tt_l0, implicit $tt_l1, implicit $tt_l2, implicit $tt_l3, implicit $tt_l4, implicit $tt_l5, implicit $tt_l6, implicit $tt_l7
; ISEL-NEXT: $tt_l5 = TTSFPIADD $tt_l5, $tt_c9
; HAZARD: TTSFPLUTFP32Three 1
; HAZARD-NEXT: TTSFPNOP
; HAZARD-NEXT: $tt_l5 = TTSFPIADD $tt_l5, $tt_c9
  call void @llvm.riscv.tt.bound.sfplutfp32.3r(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 14)
  call void @llvm.riscv.tt.bound.sfpiadd(i32 5, i32 5, i32 9, i32 4)
  ret void
}

; One source loop retains one static group action; no resource repair copies
; or duplicated iteration bodies are permitted during physical selection.
define void @loop_lut(i32 %count) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: loop_lut
; ISEL: TTSFPLUTFP32Three 0
; ISEL-NOT: TTSFPLUTFP32Three
; ISEL: PseudoRET
; HAZARD-LABEL: name: loop_lut
; HAZARD: TTSFPLUTFP32Three 0
; HAZARD-NOT: TTSFPLUTFP32Three
; HAZARD: PseudoRET
entry:
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 2, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 3, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 5, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 6, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 7, i32 9)
  br label %test
test:
  %index = phi i32 [ 0, %entry ], [ %next, %body ]
  %more = icmp ult i32 %index, %count
  br i1 %more, label %body, label %exit
body:
  call void @llvm.riscv.tt.bound.sfplutfp32.3r(i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 10)
  %next = add i32 %index, 1
  br label %test
exit:
  ret void
}
