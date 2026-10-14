; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s > %t.o0
; RUN: FileCheck %s < %t.o0
; RUN: FileCheck %s --check-prefix=SYMBOLS < %t.o0
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s > %t.o2
; RUN: FileCheck %s < %t.o2
; RUN: FileCheck %s --check-prefix=SYMBOLS < %t.o2
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s > %t.stress
; RUN: FileCheck %s < %t.stress
; RUN: FileCheck %s --check-prefix=SYMBOLS < %t.stress

declare void @llvm.riscv.bosc.dma.writes()
declare void @llvm.riscv.bosc.dma.reads()
declare void @llvm.riscv.bosc.dma.fence()
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.add(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.trap()

; PMA writes uses the exact patch7 support ranges and linker-owned mutable DDR.
; Its scalar temporaries must never clobber MLIR-owned VRF or call support code.
; Address materialization may move across independent scalar instructions;
; CSR writes and fences retain their architectural order below.
; SYMBOLS-LABEL: dma_rvv:
; SYMBOLS-DAG: __bosc_data_start
; SYMBOLS-DAG: __bosc_data_end
; SYMBOLS: ret
; CHECK-LABEL: dma_rvv:
; CHECK: vsetivli {{[a-z0-9]+}}, 16, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: fence
; CHECK: csrw pmpaddr0,
; CHECK: csrw pmpaddr1,
; CHECK: csrw pmpaddr2,
; CHECK: csrw pmpaddr3,
; CHECK: csrw pmpaddr4,
; CHECK: csrw pmpaddr5,
; CHECK: csrw pmpaddr6,
; CHECK: csrw pmpcfg0,
; CHECK: csrw 3008,
; CHECK: fence
; CHECK: sd
; CHECK: fence
; CHECK: fence
; CHECK: csrw 3008,
; CHECK: fence
; CHECK: vle32.v v5
; CHECK: vadd.vv v6, v4, v5
; CHECK: vse32.v v6
; CHECK: fence
; CHECK: ret
define void @dma_rvv(ptr %a, ptr %b, ptr %out, ptr %mmio, i64 %value) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.dma.writes()
  store volatile i64 %value, ptr %mmio, align 8
  call void @llvm.riscv.bosc.dma.fence()
  call void @llvm.riscv.bosc.dma.reads()
  call void @llvm.riscv.bosc.load(i32 5, ptr %b)
  call void @llvm.riscv.bosc.add(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.store(i32 6, ptr %out)
  call void @llvm.riscv.bosc.dma.fence()
  ret void
}

; A failed DMA guard terminates; no call, resume or unmodeled continuation.
; CHECK-LABEL: dma_guard:
; CHECK: fence
; CHECK: ret
; CHECK: unimp
define void @dma_guard(i1 %valid) "trex-bosc-bound-rvv"="1" {
entry:
  br i1 %valid, label %ready, label %fail
ready:
  call void @llvm.riscv.bosc.dma.fence()
  ret void
fail:
  call void @llvm.trap()
  unreachable
}

; Standalone DMA images have no VRF attribute or requirement for +v.
; CHECK-LABEL: dma_only:
; CHECK: fence
; CHECK: ret
define void @dma_only() {
  call void @llvm.riscv.bosc.dma.fence()
  ret void
}
