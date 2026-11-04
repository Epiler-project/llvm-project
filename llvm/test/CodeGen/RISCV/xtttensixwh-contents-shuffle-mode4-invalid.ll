; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t-O0.o 2>&1 | FileCheck %s
; RUN: not test -e %t-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t-O2.o 2>&1 | FileCheck %s
; RUN: not test -e %t-O2.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=BH
; CHECK: Wormhole SFPSHFT2 operand 3 must be in [0, 3]
;
; BH: ttsfpshft2 {{l[0-7]}}, {{l[0-7]}}, 4
; BH-NEXT: {{[0-9a-f]+:.*}}ttsfpnop
; BH-NEXT: {{[0-9a-f]+:.*}}ttsfpiadd
;
; WH mode 4 has architecturally discouraged/undefined source lanes. Mode 3
; remains the supported rotate replacement; do not admit mode 4 with it.
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @rotate8_lane_values() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %lanes = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %lanes, i32 0, i32 0, i32 4)
  %rotated = call <32 x i32> @llvm.riscv.tt.contents.shft2(<32 x i32> %old, <32 x i32> %lanes, i32 4)
  %sum = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %rotated, <32 x i32> %lanes, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 2, i32 0, i32 4)
  ret void
}
