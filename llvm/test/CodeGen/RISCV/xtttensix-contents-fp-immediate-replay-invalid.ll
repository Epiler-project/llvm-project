; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/stale-old.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/stale-old.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/undeclared-source.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %t/undeclared-source.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: contents replay requires current mutable contents and declared read-only captures

;--- stale-old.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @stale_old() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 73, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old)
  %exponent = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %old, i32 1, i32 1)
  %add = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %old, i32 16256, i32 0)
  %mul = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %add, i32 16384, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 73, <32 x i32> %mul)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 73, i32 1, <32 x i32> %mul)
  %result = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

;--- undeclared-source.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)

define void @undeclared_source() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %foreign = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 73, i32 1, i32 0, i32 0, i32 1, <32 x i32> %old)
  %exponent = call <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32> %old, <32 x i32> %foreign, i32 1, i32 1)
  %add = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %exponent, i32 16256, i32 0)
  %mul = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %add, i32 16384, i32 0)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 73, <32 x i32> %mul)
  %updated = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 73, i32 1, <32 x i32> %mul)
  %result = extractvalue { <32 x i32> } %updated, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}
