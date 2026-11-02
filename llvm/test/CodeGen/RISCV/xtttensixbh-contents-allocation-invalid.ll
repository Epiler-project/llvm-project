; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/old-live.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=COPY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 %t/pressure.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SPILL

; Predication requires the new contents to share the old destination. Keeping
; the old version live cannot silently insert a physical register copy.
; COPY: Tensix SFPU cannot synthesize a register copy
; SPILL: ran out of registers during register allocation

;--- old-live.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @old_still_live() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %new = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %old, <32 x i32> %old, <32 x i32> %one, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %new, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 2, i32 0, i32 3)
  ret void
}

; Nine simultaneously live ordinary values exceed L0-L7. CRegs, macro staging
; and host memory cannot become extra capacity or an implicit spill destination.
;--- pressure.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @nine_live_values() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v8 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v1, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v2, i32 4, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v3, i32 6, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v4, i32 8, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v5, i32 10, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v6, i32 12, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v7, i32 14, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v8, i32 16, i32 0, i32 3)
  ret void
}
