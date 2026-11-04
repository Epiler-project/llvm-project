; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/bound-memset.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/bound-memcpy.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/dma-memset.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/dma-memcpy.ll -o /dev/null 2>&1 | FileCheck %s

; The implicit-vector machine policy does not admit source memory intrinsics.
; Neither fixed operations nor the presence of an actual DMA instruction can
; turn an unreceived intrinsic into typed native protocol storage.
; CHECK: calls, inline assembly and unbound intrinsics are forbidden in bound scope

;--- bound-memset.ll
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
define void @bound_memset(ptr %state) "trex-bosc-bound-rvv"="1" {
  call void @llvm.memset.p0.i64(ptr align 8 %state, i8 0, i64 128, i1 false)
  ret void
}

;--- bound-memcpy.ll
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
define void @bound_memcpy(ptr %state, ptr %source) "trex-bosc-bound-rvv"="1" {
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %state, ptr align 8 %source, i64 128, i1 false)
  ret void
}

;--- dma-memset.ll
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
declare void @llvm.riscv.bosc.dma.fence()
define void @dma_memset(ptr %state) {
  call void @llvm.memset.p0.i64(ptr align 8 %state, i8 0, i64 128, i1 false)
  call void @llvm.riscv.bosc.dma.fence()
  ret void
}

;--- dma-memcpy.ll
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @llvm.riscv.bosc.dma.fence()
define void @dma_memcpy(ptr %state, ptr %source) {
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %state, ptr align 8 %source, i64 128, i1 false)
  call void @llvm.riscv.bosc.dma.fence()
  ret void
}
