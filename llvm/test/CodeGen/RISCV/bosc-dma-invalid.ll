; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/trap-resume.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TRAP
; RUN: not llc -mtriple=riscv64 -mattr=+v,+zvl512b %t/call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: not llc -mtriple=riscv32 %t/rv32.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RV32
; TRAP: trap must be immediately followed by unreachable
; CALL: calls, inline assembly and unbound intrinsics are forbidden in bound scope
; RV32: patch7 DMA requires RV64

;--- trap-resume.ll
declare void @llvm.riscv.bosc.dma.fence()
declare void @llvm.trap()
define void @trap_resume() "trex-bosc-bound-rvv"="1" {
  call void @llvm.riscv.bosc.dma.fence()
  call void @llvm.trap()
  ret void
}

;--- call.ll
declare void @llvm.riscv.bosc.dma.fence()
declare void @external()
define void @dma_call() "trex-bosc-bound-rvv"="1" {
  call void @llvm.riscv.bosc.dma.fence()
  call void @external()
  ret void
}

;--- rv32.ll
declare void @llvm.riscv.bosc.dma.fence()
define void @dma_rv32() {
  call void @llvm.riscv.bosc.dma.fence()
  ret void
}
