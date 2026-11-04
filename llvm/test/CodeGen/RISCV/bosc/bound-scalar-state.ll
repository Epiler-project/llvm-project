; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -filetype=obj -verify-machineinstrs < %s -o %t.o
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s

; Scalar state must not manufacture VRF contents in fixed-resource functions.
; Keep the stores ordinary: their volatility is not an allocation contract.
declare void @llvm.riscv.bosc.dma.fence()
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)

; CHECK-LABEL: fixed_scalar_constants:
; CHECK-NOT: vmv
; CHECK-NOT: vse
; CHECK: sd zero, 0(a0)
; CHECK: sd zero, 8(a0)
; CHECK: sd zero, 16(a0)
; CHECK: sd zero, 24(a0)
; CHECK-NOT: vmv
; CHECK-NOT: vse
; CHECK: ret
define void @fixed_scalar_constants(ptr %state) "trex-bosc-bound-rvv"="1" {
  %p1 = getelementptr i64, ptr %state, i64 1
  %p2 = getelementptr i64, ptr %state, i64 2
  %p3 = getelementptr i64, ptr %state, i64 3
  store i64 0, ptr %state, align 8
  store i64 0, ptr %p1, align 8
  store i64 0, ptr %p2, align 8
  store i64 0, ptr %p3, align 8
  ret void
}

; A DMA-only function has no bound-RVV attribute, but the same machine gate
; forbids implicit vector state. Its actual DMA instructions determine scope.
; CHECK-LABEL: dma_scalar_constants:
; CHECK-NOT: vmv
; CHECK-NOT: vse
; CHECK: sd zero, 0(a0)
; CHECK: sd zero, 8(a0)
; CHECK: sd zero, 16(a0)
; CHECK: sd zero, 24(a0)
; CHECK: fence
; CHECK: ret
define void @dma_scalar_constants(ptr %state) {
  %p1 = getelementptr i64, ptr %state, i64 1
  %p2 = getelementptr i64, ptr %state, i64 2
  %p3 = getelementptr i64, ptr %state, i64 3
  store i64 0, ptr %state, align 8
  store i64 0, ptr %p1, align 8
  store i64 0, ptr %p2, align 8
  store i64 0, ptr %p3, align 8
  call void @llvm.riscv.bosc.dma.fence()
  ret void
}

; Fixed instructions retain their explicitly assigned VRF operands while
; scalar protocol storage uses GPR memory actions.
; CHECK-LABEL: fixed_rvv_and_scalar_constants:
; CHECK-NOT: vmv
; CHECK: sd zero, 0(a0)
; CHECK: sd zero, 8(a0)
; CHECK: sd zero, 16(a0)
; CHECK: sd zero, 24(a0)
; CHECK: vle32.v v4,
; CHECK: vse32.v v4,
; CHECK-NOT: vmv
; CHECK: ret
define void @fixed_rvv_and_scalar_constants(ptr %state, ptr %input, ptr %output)
    "trex-bosc-bound-rvv"="1" {
  %p1 = getelementptr i64, ptr %state, i64 1
  %p2 = getelementptr i64, ptr %state, i64 2
  %p3 = getelementptr i64, ptr %state, i64 3
  store i64 0, ptr %state, align 8
  store i64 0, ptr %p1, align 8
  store i64 0, ptr %p2, align 8
  store i64 0, ptr %p3, align 8
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %input)
  call void @llvm.riscv.bosc.store(i32 4, ptr %output)
  ret void
}

; noimplicitfloat prevents implicit FP/vector use. Explicit scalar floating
; point arithmetic and the fixed vector operations above retain their meaning.
; CHECK-LABEL: fixed_scalar_float:
; CHECK: fadd.s
; CHECK: ret
define float @fixed_scalar_float(float %a, float %b)
    "trex-bosc-bound-rvv"="1" {
  %sum = fadd float %a, %b
  ret float %sum
}

; The allocated contents ABI has real vector SSA. Even when it also issues
; DMA, it must keep its explicit vector definitions and uses.
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
; CHECK-LABEL: contents_with_dma:
; CHECK: vle32.v [[CONTENTS:v[0-9]+]],
; CHECK: fence
; CHECK: vse32.v [[CONTENTS]],
; CHECK: ret
define void @contents_with_dma(ptr %input, ptr %output, i64 %n)
    "trex-bosc-contents-rvv"="1" {
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %contents = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %input, i64 %vl)
  call void @llvm.riscv.bosc.dma.fence()
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %contents, ptr %output, i64 %vl)
  ret void
}
