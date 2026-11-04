; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %s -o %t.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.o
; CHECK: contents swap mode must be in 0..9

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
define void @invalid_swap_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %a, <32 x i32> %b, i32 10)
  %r = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  ret void
}
