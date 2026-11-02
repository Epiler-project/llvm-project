; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -riscv-rvv-regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -enable-new-pm -O0 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -enable-new-pm -O2 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM

; NEWPM: Tensix SFPU requires the SelectionDAG legacy codegen pipeline

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

; Both branch results reach one contents PHI. Placement may coalesce the three
; versions; a predecessor's source value must not be reconstructed from a
; physical selector or transported through ordinary memory.
define void @contents_diamond(i1 %choose_add) "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_diamond>:
; CHECK: ttsfpmov.all [[VALUE:l[0-7]]], c9
; CHECK: ttsfpmov.all [[ONE:l[0-7]]], c10
; CHECK: b{{.*}}
; CHECK-DAG: ttsfpadd [[VALUE]], [[VALUE]], [[ONE]], 0
; CHECK-DAG: ttsfpmul [[VALUE]], [[VALUE]], [[ONE]], 0
; CHECK: ttsfpstore [[VALUE]], 0, 0, 4
; CHECK: ret
; ISEL-LABEL: name: contents_diamond
; ISEL: :sfpr = TTSFPADD
; ISEL: :sfpr = TTSFPMUL
; ISEL: :sfpr = PHI
; ISEL: TTSFPSTORE
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br i1 %choose_add, label %add, label %mul
add:
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %initial, <32 x i32> %initial, <32 x i32> %one, i32 0)
  br label %join
mul:
  %product = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %initial, <32 x i32> %initial, <32 x i32> %one, i32 0)
  br label %join
join:
  %result = phi <32 x i32> [ %sum, %add ], [ %product, %mul ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

; Two independent loop-carried values retain their simultaneous numerical
; lifetimes. The CFG survives; iterations are not duplicated to replace PHIs.
define void @two_loop_contents(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <two_loop_contents>:
; CHECK: ttsfpmov.all [[FIRST:l[0-7]]], c9
; CHECK: ttsfpmov.all [[SECOND:l[0-7]]], c10
; CHECK: ttsfpmov.all [[STEP:l[0-7]]], c10
; CHECK: ttsfpadd [[FIRST]], [[FIRST]], [[STEP]], 0
; CHECK: ttsfpmul [[SECOND]], [[SECOND]], [[STEP]], 0
; CHECK: bne{{z?}} {{.*}},
; CHECK: ttsfpstore [[FIRST]], 0, 0, 4
; CHECK: ttsfpstore [[SECOND]], 2, 0, 4
; CHECK: ret
; ISEL-LABEL: name: two_loop_contents
; ISEL-COUNT-2: :sfpr = PHI
; ISEL: :sfpr = TTSFPADD
; ISEL: :sfpr = TTSFPMUL
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial.first = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial.second = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %step = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %first = phi <32 x i32> [ %initial.first, %entry ], [ %sum, %loop ]
  %second = phi <32 x i32> [ %initial.second, %entry ], [ %product, %loop ]
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %first, <32 x i32> %first, <32 x i32> %step, i32 0)
  %product = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %second, <32 x i32> %second, <32 x i32> %step, i32 0)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %product, i32 2, i32 0, i32 4)
  ret void
}
