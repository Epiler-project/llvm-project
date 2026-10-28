; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/missing-handoff.ll 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/combined-clear.ll 2>&1 | FileCheck %s --check-prefix=CLEAR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/port.ll 2>&1 | FileCheck %s --check-prefix=PORT
; MISSING: Wormhole UNPACR_NOP does not match an admitted logical form
; CLEAR: Wormhole UNPACR_NOP does not match an admitted logical form
; PORT: Tensix intrinsic is not supported on Wormhole

;--- missing-handoff.ll
declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr.nop(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- combined-clear.ll
declare void @llvm.riscv.tt.unpacr.nop(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr.nop(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0)
  ret void
}

;--- port.ll
declare void @llvm.riscv.tt.unpacr.nop.port(i32 immarg, i32, i32, i32, i32, i32, i32, i32, i32, i32)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr.nop.port(i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0)
  ret void
}
