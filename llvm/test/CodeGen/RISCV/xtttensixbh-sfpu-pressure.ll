; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/eight.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/eight.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/nine.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/nine.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/fixed.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: {{register allocation|register pressure}}

;--- eight.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %v0 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v2 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v1, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v2, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v3, i32 3, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v4, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v5, i32 5, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v6, i32 6, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v7, i32 7, i32 0, i32 4)
  ret void
}

;--- nine.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %v0 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v2 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v8 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v1, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v2, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v3, i32 3, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v4, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v5, i32 5, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v6, i32 6, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v7, i32 7, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v8, i32 8, i32 0, i32 4)
  ret void
}

;--- fixed.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  %v0 = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 7)
  %v1 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v2 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v1, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v2, i32 2, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v3, i32 3, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v4, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v5, i32 5, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v6, i32 6, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v7, i32 7, i32 0, i32 4)
  ret void
}
