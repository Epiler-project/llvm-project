; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/first-invalid.ll 2>&1 | FileCheck %s --check-prefix=REGISTER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/bh-upper.ll 2>&1 | FileCheck %s --check-prefix=REGISTER
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/value.ll 2>&1 | FileCheck %s --check-prefix=VALUE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/port.ll 2>&1 | FileCheck %s --check-prefix=FORM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh < %t/mop.ll 2>&1 | FileCheck %s --check-prefix=FORM
; REGISTER: Wormhole SETC16 field setc16_reg must be in [0, 56]
; VALUE: SETC16 field setc16_value must be in [0, 65535]
; FORM: Tensix intrinsic is not supported on Wormhole

;--- first-invalid.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.setc16(i32 0, i32 57)
  ret void
}
;--- bh-upper.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.setc16(i32 65535, i32 67)
  ret void
}
;--- value.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.setc16(i32 65536, i32 0)
  ret void
}
;--- port.ll
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 0, i32 0)
  ret void
}
;--- mop.ll
declare void @llvm.riscv.tt.setc16.mop(i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.setc16.mop(i32 2, i32 0, i32 0)
  ret void
}
