; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/missing-scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/width.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDTH
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/pure-memory.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MEMORY
; SCOPE: ordered RVV contents require RVV contents scope
; WIDTH: bit observation width must be 8, 32, or 64
; MEMORY: RVV contents memory and FP operations require ordered intrinsics

;--- missing-scope.ll
declare i64 @llvm.riscv.bosc.read.bits.contents(ptr, i64, i32 immarg)
define i64 @missing_scope(ptr %p) {
  %bits = call i64 @llvm.riscv.bosc.read.bits.contents(ptr %p, i64 0, i32 32)
  ret i64 %bits
}

;--- width.ll
declare i64 @llvm.riscv.bosc.read.bits.contents(ptr, i64, i32 immarg)
define i64 @invalid_width(ptr %p) "trex-bosc-contents-rvv"="1" {
  %bits = call i64 @llvm.riscv.bosc.read.bits.contents(ptr %p, i64 0, i32 16)
  ret i64 %bits
}

;--- pure-memory.ll
declare <vscale x 2 x i32> @llvm.riscv.vle.nxv2i32(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.vse.nxv2i32(<vscale x 2 x i32>, ptr, i64)
define void @pure_memory(ptr %p) "trex-bosc-contents-rvv"="1" {
  %v = call <vscale x 2 x i32> @llvm.riscv.vle.nxv2i32(<vscale x 2 x i32> poison, ptr %p, i64 16)
  call void @llvm.riscv.vse.nxv2i32(<vscale x 2 x i32> %v, ptr %p, i64 16)
  ret void
}
