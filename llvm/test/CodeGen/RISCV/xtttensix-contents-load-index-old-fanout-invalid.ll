; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh-O0.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh-O2.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t.wh-O0.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t.wh-O2.o 2>&1 | FileCheck %s
; RUN: test ! -e %t.wh-O2.o
;
; Reusing one old SSA for two distinct destructive initializations requires
; an actual register copy. The contents ABI forbids synthesizing that copy.
; Separate explicit constant reads define independent initialized contents;
; the positive dynamic fixture exercises that form on both architectures.
; CHECK: Tensix SFPU cannot synthesize a register copy

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @dynamic_pair(i32 noundef %offset) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  %bounded = and i32 %offset, 1023
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 %bounded, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

define void @dynamic_pair_loop(i32 noundef %offset, i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %initial, i32 21845, i32 8)
  %empty = icmp eq i32 %count, 0
  br i1 %empty, label %exit, label %loop
loop:
  %oldvalue = phi <32 x i32> [ %initial, %entry ], [ %value, %loop ]
  %oldindex = phi <32 x i32> [ %index, %entry ], [ %captured, %loop ]
  %coordinate = phi i32 [ %offset, %entry ], [ %next, %loop ]
  %remaining = phi i32 [ %count, %entry ], [ %decremented, %loop ]
  %bounded = and i32 %coordinate, 1023
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %oldvalue, <32 x i32> %oldindex, i32 %bounded, i32 0, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  %next = add i32 %coordinate, 4
  %decremented = add i32 %remaining, -1
  %again = icmp ne i32 %decremented, 0
  br i1 %again, label %loop, label %exit, !llvm.loop !0
exit:
  %finalvalue = phi <32 x i32> [ %initial, %entry ], [ %value, %loop ]
  %finalindex = phi <32 x i32> [ %index, %entry ], [ %captured, %loop ]
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %finalvalue, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %finalindex, i32 2, i32 0, i32 4)
  ret void
}

define void @dynamic_pair_addr3(i32 noundef %offset) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %config = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %index = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 42330, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %config, i32 21845, i32 8)
  %bounded = and i32 %offset, 1023
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %config, <32 x i32> %index, i32 %bounded, i32 3, i32 3)
  %value = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  %captured = extractvalue { <32 x i32>, <32 x i32> } %pair, 1
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %captured, i32 2, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
