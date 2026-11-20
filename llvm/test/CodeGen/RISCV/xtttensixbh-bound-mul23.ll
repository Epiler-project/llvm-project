; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=ARCH
; RUN: sed 's/i32 3, i32 1)/i32 3, i32 4)/' %s | not llc -mtriple=riscv32 -mattr=+xtttensixbh -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
;
; CHECK-LABEL: <product>:
; CHECK: ttsfpmul24 l0, l2, l3, 0
; CHECK: ttsfpmul24 l1, l2, l3, 1
; ARCH: SFPU intrinsic architecture does not match target feature
; MODE: unsupported mode at operand 4

declare void @llvm.tt.bh.sfpu.mul23(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @product() "tensix-executor"="trisc1" {
  call void @llvm.tt.bh.sfpu.mul23(i32 0, i32 0, i32 2, i32 3, i32 0)
  call void @llvm.tt.bh.sfpu.mul23(i32 1, i32 1, i32 2, i32 3, i32 1)
  ret void
}
