; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t-O0.o 2>&1 | FileCheck %s
; RUN: test ! -e %t-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t-O2.o 2>&1 | FileCheck %s
; RUN: test ! -e %t-O2.o
; CHECK: contents copy4 mode must be 0 or 1
;
; copy4 owns only modes0/1. Mode2 must use the complete rotate4 operand
; tuple; accepting it here would silently omit its VC numerical dependency.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @invalid_copy_mode(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %raw0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed0 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw0, i32 0, i32 5)
  %raw1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed1 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw1, i32 64, i32 5)
  %raw2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed2 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw2, i32 128, i32 5)
  %raw3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed3 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %raw3, i32 192, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %seed0, i32 0, i32 0, i32 4)
  %has.iterations = icmp ugt i32 %limit, 0
  br i1 %has.iterations, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %latch ]
  %active0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %active1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %active2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %active3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %active0, <32 x i32> %active1, <32 x i32> %active2, <32 x i32> %active3, i32 2)
  %updated0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %updated1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %updated2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %updated3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %finished = icmp eq i32 %next, %limit
  br i1 %finished, label %exit, label %loop, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %final1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %final2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %final3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 38, i32 0, i32 4)
  call void @llvm.riscv.tt.stallwait(i32 16384, i32 511)
  call void @llvm.riscv.tt.nop()
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
