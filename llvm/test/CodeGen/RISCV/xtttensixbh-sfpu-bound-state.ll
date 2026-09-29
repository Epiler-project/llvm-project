; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/snapshot.ll -o - | FileCheck %s --check-prefixes=EXPLICIT-SNAPSHOT,SHARED-CC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/snapshot.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/underflow.ll -o - | FileCheck %s --check-prefix=INCOMING-CC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/underflow.ll -o - | FileCheck %s --check-prefix=INCOMING-CC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/merge.ll -o - | FileCheck %s --check-prefix=CC-LOOP
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/merge.ll -o - | FileCheck %s --check-prefix=CC-LOOP
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/state.ll -o - | FileCheck %s --check-prefix=STATE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/state.ll -o - | FileCheck %s --check-prefix=STATE
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/branch.ll -o - | FileCheck %s --check-prefix=CC-BRANCH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/branch.ll -o - | FileCheck %s --check-prefix=CC-BRANCH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/invalid-push.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CC-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/invalid-pop.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CC-FIELD
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/loop.ll -o - | FileCheck %s --check-prefix=LOOP
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/loop.ll -o /dev/null
; RUN: opt -S -mtriple=riscv32 -mattr=+xtttensixbh -passes='default<O3>' %t/loop.ll | FileCheck %s --check-prefix=LOOP-IR
; RUN: opt -S -mtriple=riscv32 -mattr=+xtttensixbh -passes=loop-unroll -unroll-count=4 -unroll-threshold=100000 %t/loop.ll | FileCheck %s --check-prefix=LOOP-IR
; LOOP-IR-LABEL: define void @loop(
; LOOP-IR: phi i32
; LOOP-IR: call void @llvm.riscv.tt.bound.sfpiadd(
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfpiadd(
; LOOP-IR: br i1
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfpiadd(
; LOOP-IR: call void @llvm.riscv.tt.bound.sfpstore(
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfpiadd(
; LOOP-IR: ret void

;--- snapshot.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpsetcc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
define void @snapshot() "tensix-executor"="trisc2" {
; EXPLICIT-SNAPSHOT-LABEL: name: snapshot
; SHARED-CC: TTSFPENCC 3, 10, {{.*}}implicit-def $tt_cc
; EXPLICIT-SNAPSHOT: $tt_l0 = TTSFPMOVAll $tt_c9
; EXPLICIT-SNAPSHOT-NEXT: $tt_l1 = TTSFPMOVAll $tt_l0
; EXPLICIT-SNAPSHOT-NEXT: $tt_l0 = TTSFPMOVAll $tt_c8
; SHARED-CC: TTSFPPUSHC {{.*}}implicit $tt_cc
; SHARED-CC: TTSFPSETCCNE $tt_l0
; SHARED-CC: TTSFPPOPC {{.*}}implicit-def $tt_cc
; EXPLICIT-SNAPSHOT: TTSFPSTORE $tt_l1, 0, 0, 3
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 0)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 8)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpsetcc(i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpstore(i32 1, i32 0, i32 0, i32 3)
  ret void
}
;--- underflow.ll
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
; Bound ingress preserves the authored physical state. It does not infer an
; empty incoming CC stack or synthesize a matching push.
; INCOMING-CC-LABEL: underflow:
; INCOMING-CC: .word 0x20000002
; INCOMING-CC-NOT: .word
; INCOMING-CC: ret
define void @underflow() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  ret void
}
;--- merge.ll
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
; A source-authored loop need not have a statically balanced CC depth.
; CC-LOOP-LABEL: merge:
; CC-LOOP: .word 0x1c000002
; CC-LOOP-NOT: .word
; CC-LOOP: bnez
; CC-LOOP-NOT: .word
; CC-LOOP: ret
define void @merge(i1 %again) "tensix-executor"="trisc0" {
entry:
  br label %loop
loop:
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  br i1 %again, label %loop, label %exit
exit:
  ret void
}
;--- loop.ll
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpiadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)
define void @loop(i32 %limit) "tensix-executor"="trisc1" {
; LOOP-LABEL: name: loop
; LOOP-NOT: class: sfpr
; LOOP: $tt_l0 = TTSFPMOVAll $tt_c9
; LOOP: PHI
; LOOP: $tt_l0 = TTSFPIADD $tt_l0, $tt_c8
; LOOP: TTSFPSTORE $tt_l0
entry:
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  br label %body
body:
  %i = phi i32 [0, %entry], [%next, %body]
  call void @llvm.riscv.tt.bound.sfpiadd(i32 0, i32 0, i32 8, i32 4)
  %next = add i32 %i, 1
  %again = icmp ult i32 %next, %limit
  br i1 %again, label %body, label %exit
exit:
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 3)
  ret void
}

;--- state.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
; STATE-LABEL: static_state:
; STATE: .word 0xc8000006
; STATE-NEXT: .word 0x08000000
; STATE-NEXT: .word 0x08000000
; STATE-NEXT: .word 0x08000000
; STATE-NEXT: .word 0xf0002409
; STATE-NOT: .word
; STATE: ret
define void @static_state() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setc16(i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  ret void
}
; Port configuration still requires defined, bounded machine fields, but
; its value and target bank are source-controlled.
; STATE-LABEL: dynamic_state:
; STATE: sw
; STATE: .word 0x08000000
; STATE-NEXT: .word 0x08000000
; STATE-NEXT: .word 0x08000000
; STATE-NEXT: .word 0xf0002409
; STATE-NOT: .word
; STATE: ret
define void @dynamic_state(i32 noundef %input) "tensix-executor"="trisc2" {
  %state = and i32 %input, 1
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 %state, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 9)
  ret void
}

;--- branch.ll
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
; CC-BRANCH-LABEL: branch:
; CC-BRANCH: .word 0x1c000002
; CC-BRANCH-NOT: .word
; CC-BRANCH: .word 0x20000002
; CC-BRANCH-NOT: .word
; CC-BRANCH: ret
define void @branch(i1 %condition) "tensix-executor"="trisc0" {
entry:
  br i1 %condition, label %push, label %join
push:
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  br label %join
join:
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  ret void
}

;--- invalid-push.ll
declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
; CC-FIELD: must be in [0, 0]
define void @invalid_push() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfppushc(i32 1, i32 0)
  ret void
}
;--- invalid-pop.ll
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
define void @invalid_pop() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 1)
  ret void
}
