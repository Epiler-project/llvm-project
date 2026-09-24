; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=RA
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=RA
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpmul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.sfpconfig.creg(<32 x i32>, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpconfig.reset()
declare void @llvm.riscv.tt.sfpcompc()
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

; Both inactive-lane passthrough and a later use of old remain observable.
define void @multiply() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: multiply
; ISEL: TTSFPMUL
; ISEL-SAME: 2
; ISEL-SAME: implicit $tt_c9
; RA-LABEL: name: multiply
; RA: TTSFPMOVAll
; RA: TTSFPMUL
; RA: TTSFPSTORE
  %old = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  %p = call <32 x i32> @llvm.riscv.tt.sfpmul(<32 x i32> %old, <32 x i32> %one, <32 x i32> %one, i32 2)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %p, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %old, i32 1, i32 0, i32 4)
  ret void
}

; Snapshot C11 before modifying it; the prior value cannot be reread afterward.
define void @mutable_creg() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: mutable_creg
; ISEL: TTSFPCONFIGC11 {{.*}}21845, 8
; ISEL-SAME: implicit-def {{(dead )?}}$tt_c11
; ISEL-SAME: implicit $tt_l0
; ISEL-SAME: implicit $tt_c11
; RA-LABEL: name: mutable_creg
; RA: TTSFPMOVAll $tt_c11
; RA: $tt_l0 = TTSFPMOVAll
; RA: TTSFPCONFIGC11 {{.*}}21845, 8
; RA: TTSFPMOVAll $tt_c11
  %before = call <32 x i32> @llvm.riscv.tt.creg.read(i32 11)
  %one = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  call void @llvm.riscv.tt.sfpconfig.creg(<32 x i32> %one, i32 11, i32 21845, i32 8)
  %after = call <32 x i32> @llvm.riscv.tt.creg.read(i32 11)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %before, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %after, i32 1, i32 0, i32 4)
  ret void
}

; Architectural L0 state must survive the compiler's private CONFIG staging.
define void @fixed_l0_preserved() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: fixed_l0_preserved
; ISEL: TTSFPCONFIGC12 {{.*}}0, 0
; RA-LABEL: name: fixed_l0_preserved
; RA: [[SAVE:\$tt_l[1-7]]] = TTSFPMOVAll $tt_l0
; RA: $tt_l0 = TTSFPMOVAll
; RA: TTSFPCONFIGC12 {{.*}}0, 0
; RA: $tt_l0 = TTSFPMOVAll {{(killed )?}}[[SAVE]]
; RA: TTSFPMOVAll $tt_l0
  %v = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  call void @llvm.riscv.tt.sfpconfig.creg(<32 x i32> %v, i32 12, i32 0, i32 0)
  %l0 = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %l0, i32 0, i32 0, i32 4)
  ret void
}

define void @condition_and_reset() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: condition_and_reset
; ISEL: TTSFPCONFIGReset
; ISEL-SAME: implicit-def {{(dead )?}}$tt_config
; ISEL: TTSFPPUSHC
; ISEL: TTSFPCOMPC
; ISEL-SAME: implicit $tt_ccstack
; ISEL: TTSFPPOPC
; RA-LABEL: name: condition_and_reset
; RA: TTSFPCONFIGReset
; RA: TTSFPCOMPC
  call void @llvm.riscv.tt.sfpconfig.reset()
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpcompc()
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  ret void
}
