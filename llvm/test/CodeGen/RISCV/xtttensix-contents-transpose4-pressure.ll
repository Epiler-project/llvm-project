; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %s -o %t.bh-o0.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.bh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -filetype=obj %s -o %t.bh-o2.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.bh-o2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -filetype=obj %s -o %t.wh-o0.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.wh-o0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -filetype=obj %s -o %t.wh-o2.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.wh-o2.o

; Four extra actual values stay live across TRANSP. Low L0..L3 hold its
; tied inputs/results while L4..L7 are clobbered: none is legal storage for
; the extra values. Reject rather than spill, duplicate, or silently overwrite.
; CHECK: {{ran out of registers during register allocation|Tensix SFPU cannot synthesize a register copy}}
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

define void @scratch_is_not_live_capacity() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %c = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %d = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %extra0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %extra1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %extra2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %extra3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose4(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %extra0, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %extra1, i32 10, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %extra2, i32 12, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %extra3, i32 14, i32 0, i32 4)
  ret void
}
