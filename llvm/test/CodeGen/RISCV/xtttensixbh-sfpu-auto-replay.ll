; Legacy LLVM SSA cannot enter automatic replay: physical SFPU operands must
; be authored upstream. Physical replay selection is covered by
; xtttensixbh-sfpu-auto-replay.mir and the bound-float replay tests.
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; REJECT: unsupported Tensix intrinsic ABI: llvm.riscv.tt.sfpencc

; This is retained as a negative compatibility test for the retired virtual
; SFPU path. It must never be used as evidence for physical replay selection.
; Both inputs are independent Dst values. The original full-lane value remains
; live after the masked arithmetic, so replay must preserve its allocation.
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpand(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpor(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32>, <32 x i32>)
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @masked_arithmetic_with_snapshot() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 0, i32 0, i32 4)
  %rhs = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %rhs, i32 0, i32 2)
  %a0 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %old, <32 x i32> %rhs, i32 4)
  %a1 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %a0, <32 x i32> %rhs)
  %a2 = call <32 x i32> @llvm.riscv.tt.sfpand(<32 x i32> %a1, <32 x i32> %rhs)
  %a3 = call <32 x i32> @llvm.riscv.tt.sfpor(<32 x i32> %a2, <32 x i32> %rhs)
  %a4 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %a3, <32 x i32> %rhs)
  %a5 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %a4, <32 x i32> %rhs, i32 4)
  %b0 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %a5, <32 x i32> %rhs, i32 4)
  %b1 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %b0, <32 x i32> %rhs)
  %b2 = call <32 x i32> @llvm.riscv.tt.sfpand(<32 x i32> %b1, <32 x i32> %rhs)
  %b3 = call <32 x i32> @llvm.riscv.tt.sfpor(<32 x i32> %b2, <32 x i32> %rhs)
  %b4 = call <32 x i32> @llvm.riscv.tt.sfpxor(<32 x i32> %b3, <32 x i32> %rhs)
  %b5 = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %b4, <32 x i32> %rhs, i32 4)
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %b5, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %old, i32 3, i32 0, i32 4)
  ret void
}
