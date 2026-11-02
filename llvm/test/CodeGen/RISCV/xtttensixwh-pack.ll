; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s

declare void @llvm.riscv.tt.setadc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.pacr(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

; CHECK-LABEL: <pack_fp32>:
; CHECK: ttsetrwc
; CHECK: ttsetadc
; CHECK: ttpacr
; CHECK: ttpacr
define void @pack_fp32() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setrwc(i32 15, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.setadc(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.setadc(i32 63, i32 0, i32 1, i32 4)
  call void @llvm.riscv.tt.pacr(i32 1, i32 1, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.pacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}
