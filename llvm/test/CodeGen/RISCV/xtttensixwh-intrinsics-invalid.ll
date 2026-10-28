; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/field.ll 2>&1 | FileCheck %s --check-prefix=FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/condition.ll 2>&1 | FileCheck %s --check-prefix=CONDITION
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/unsupported.ll 2>&1 | FileCheck %s --check-prefix=REJECTED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/port.ll 2>&1 | FileCheck %s --check-prefix=PORT
; FIELD: Wormhole SEMGET field sem_sel must be in [0, 255]
; CONDITION: Wormhole SEMWAIT field wait_sem_cond must be in [1, 3]
; REJECTED: Tensix intrinsic is not supported on Wormhole
; PORT: Tensix intrinsic is not supported on Wormhole

;--- field.ll
declare void @llvm.riscv.tt.semget(i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.semget(i32 256)
  ret void
}
;--- condition.ll
declare void @llvm.riscv.tt.semwait(i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.semwait(i32 0, i32 1, i32 2)
  ret void
}
;--- unsupported.ll
declare void @llvm.riscv.tt.setdmareg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setdmareg(i32 0, i32 0, i32 0, i32 0)
  ret void
}
;--- port.ll
declare void @llvm.riscv.tt.semget.port(i32 immarg, i32)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.semget.port(i32 0, i32 1)
  ret void
}
