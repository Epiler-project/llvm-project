; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -stop-after=finalize-isel -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ISEL

declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.lut(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.lutfp32.six(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.lutfp32.three(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)


; Fixed inputs/results have the same contents contract on WH and BH.
; WH native LUTFP32 encodings intentionally have no Mod1Mirror.

; CHECK-LABEL: <lut_direct>:
; CHECK: ttsfplut {{l[0-7]}}, 4
; CHECK: ttsfpstore
; ISEL-LABEL: name: lut_direct
; ISEL: PseudoTTSFPContentsLUT
define void @lut_direct() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %r = call <32 x i32> @llvm.riscv.tt.contents.lut(<32 x i32> %v4, <32 x i32> %v0, <32 x i32> %v1, <32 x i32> %v2, <32 x i32> %v3, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 0, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <lutfp32_six_direct>:
; CHECK: ttsfplutfp32.6r {{l[0-7]}}, 4
; CHECK: ttsfpstore
; ISEL-LABEL: name: lutfp32_six_direct
; ISEL: PseudoTTSFPContentsLUTSix
define void @lutfp32_six_direct() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %r = call <32 x i32> @llvm.riscv.tt.contents.lutfp32.six(<32 x i32> %v7, <32 x i32> %v0, <32 x i32> %v1, <32 x i32> %v2, <32 x i32> %v3, <32 x i32> %v4, <32 x i32> %v5, <32 x i32> %v6, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 0, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <lutfp32_three_indirect>:
; CHECK: ttsfplutfp32.3r 1
; CHECK-COUNT-8: ttsfpstore
; ISEL-LABEL: name: lutfp32_three_indirect
; ISEL: PseudoTTSFPContentsLUTThree
define void @lutfp32_three_indirect() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %v0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %v2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %v4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %v5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %v6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %v7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.lutfp32.three(<32 x i32> %v0, <32 x i32> %v1, <32 x i32> %v2, <32 x i32> %v3, <32 x i32> %v4, <32 x i32> %v5, <32 x i32> %v6, <32 x i32> %v7, i32 1)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  %r4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r4, i32 8, i32 0, i32 4)
  %r5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 5
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r5, i32 10, i32 0, i32 4)
  %r6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 6
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r6, i32 12, i32 0, i32 4)
  %r7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 7
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r7, i32 14, i32 0, i32 4)
  ret void
}
