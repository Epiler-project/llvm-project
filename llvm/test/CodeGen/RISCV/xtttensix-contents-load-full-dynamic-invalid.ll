; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/undefined.ll -o %t/undefined-bh.o 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not test -e %t/undefined-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/undefined.ll -o %t/undefined-wh.o 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not test -e %t/undefined-wh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/unbounded.ll -o %t/unbounded-bh.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not test -e %t/unbounded-bh.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/unbounded.ll -o %t/unbounded-wh.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not test -e %t/unbounded-wh.o
; DEFINED: contents Dst offset must be defined and non-poison
; RANGE: contents Dst offset must be proven in [0, 1023]

; Full numerical definition does not relax instruction-port field safety.
;--- undefined.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @undefined_coordinate() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 poison, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}

;--- unbounded.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.load.full(i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @unbounded_coordinate(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %value = call <32 x i32> @llvm.riscv.tt.contents.load.full(i32 %offset, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %value, i32 4, i32 0, i32 3)
  ret void
}
