; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/lane.ll -o %t/lane.o 2>&1 | FileCheck %s --check-prefix=LANE
; RUN: test ! -e %t/lane.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/lut.ll -o %t/lut.o 2>&1 | FileCheck %s --check-prefix=LUT
; RUN: test ! -e %t/lut.o

; LaneConfig uses the admitted masked assignment mode on either target.
; Invalid modes still fail before object emission, as do incomplete LUT forms.
; LANE: contents lane configuration mode must be 8
; LUT: contents LUT mode must be 0 or 4

;--- lane.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
define void @invalid_lane_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %source, i32 21845, i32 7)
  ret void
}

;--- lut.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lut(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @unsupported_lut_indirect() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.lut(<32 x i32> %source, <32 x i32> %source, <32 x i32> %source, <32 x i32> %source, <32 x i32> %source, i32 8)
  ret void
}
