; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s

; A compiler-authored math guard is legal without a DMA instruction. Its
; failure edge terminates immediately; the successful edge retains bound VRF.
declare void @llvm.trap()
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)

; CHECK-LABEL: guarded_math:
; CHECK: beqz {{[a-z][0-9]+}},
; CHECK: vset{{i?}}vli
; CHECK: vle32.v v4,
; CHECK: vse32.v v4,
; CHECK: unimp
define void @guarded_math(ptr %input, ptr %output, i1 %valid)
    "trex-bosc-bound-rvv"="1" {
entry:
  br i1 %valid, label %compute, label %invalid
compute:
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %input)
  call void @llvm.riscv.bosc.store(i32 4, ptr %output)
  ret void
invalid:
  call void @llvm.trap()
  unreachable
}
