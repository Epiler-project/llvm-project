; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/combined.ll 2>&1 | FileCheck %s --check-prefix=FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/wrong-wait.ll 2>&1 | FileCheck %s --check-prefix=FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/reset.ll 2>&1 | FileCheck %s --check-prefix=RESET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/keep.ll 2>&1 | FileCheck %s --check-prefix=RESET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/both-lanes.ll 2>&1 | FileCheck %s --check-prefix=LANES
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/counter-payload.ll 2>&1 | FileCheck %s --check-prefix=COUNTER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/implicit-flip.ll 2>&1 | FileCheck %s --check-prefix=FLIP
; FORM: Wormhole UNPACR_NOP does not match an admitted logical form
; RESET: Wormhole CLEARDVALID field reset must be in [0, 0]
; LANES: Wormhole CLEARDVALID field cleardvalid must be in [1, 2]
; COUNTER: Wormhole SETRWC field rwc_a must be in [0, 0]
; FLIP: Wormhole SETRWC field clear_ab_vld must be in [0, 0]

;--- combined.ll
declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr.nop(i32 1, i32 0, i32 0, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0)
  ret void
}
;--- wrong-wait.ll
declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr.nop(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}
;--- reset.ll
declare void @llvm.riscv.tt.cleardvalid(i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.cleardvalid(i32 1, i32 1)
  ret void
}
;--- keep.ll
declare void @llvm.riscv.tt.cleardvalid(i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.cleardvalid(i32 2, i32 1)
  ret void
}
;--- both-lanes.ll
declare void @llvm.riscv.tt.cleardvalid(i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.cleardvalid(i32 0, i32 3)
  ret void
}
;--- counter-payload.ll
declare void @llvm.riscv.tt.setrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setrwc(i32 1, i32 1, i32 0, i32 0, i32 0, i32 0)
  ret void
}
;--- implicit-flip.ll
declare void @llvm.riscv.tt.setrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1)
  ret void
}
