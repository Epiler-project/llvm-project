; RUN: llc -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c,+v,+zvl512b -target-abi=lp64d -code-model=medium -relocation-model=static -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c,+v,+zvl512b -target-abi=lp64d -code-model=medium -relocation-model=static -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c,+v,+zvl512b -target-abi=lp64d -code-model=medium -relocation-model=static -O2 -stress-regalloc=4 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64-unknown-elf -mcpu=generic-rv64 -mattr=+m,+a,+f,+d,+c,+v,+zvl512b -target-abi=lp64d -code-model=medium -relocation-model=static -O2 -stop-after=machine-sink %s -o - | FileCheck %s --check-prefix=MIR
;
; Constant patch7 MMIO addresses are shared across two CFG transactions.
; Machine sinking folds address ADDIs into volatile LD/SD instructions;
; the rebuilt instructions must keep the received physical issue dependency.
; Production target features, ABI and exact VLEN match native object codegen.
; ASM-LABEL: dma_mmio:
; ASM: sd {{[a-z0-9]+}}, 8(
; ASM: sd {{[a-z0-9]+}}, 16(
; ASM: sd {{[a-z0-9]+}}, 24(
; ASM: ld {{[a-z0-9]+}}, 32(
; ASM: sd {{[a-z0-9]+}}, 8(
; ASM: sd {{[a-z0-9]+}}, 16(
; ASM: sd {{[a-z0-9]+}}, 24(
; ASM: ld {{[a-z0-9]+}}, 32(
; ASM-NOT: bosc_issue
; ASM: ret
; MIR-LABEL: name: dma_mmio
; MIR: SD {{.*}}, 16, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile store
; MIR: SD {{.*}}, 24, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile store
; MIR: LD {{.*}}, 32, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile load
; MIR: SD {{.*}}, 8, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile store
; MIR: SD {{.*}}, 16, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile store
; MIR: SD {{.*}}, 24, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile store
; MIR: LD {{.*}}, 32, implicit $bosc_issue, implicit-def $bosc_issue :: (volatile load

declare void @llvm.riscv.bosc.dma.writes()
declare void @llvm.riscv.bosc.dma.reads()
declare void @llvm.riscv.bosc.dma.fence()

define void @dma_mmio(i64 %source, i1 %second) "trex-bosc-bound-rvv"="1" vscale_range(8,8) {
entry:
  call void @llvm.riscv.bosc.dma.writes()
  store volatile i64 %source, ptr inttoptr (i64 1610612736 to ptr), align 8
  store volatile i64 536870912, ptr inttoptr (i64 1610612744 to ptr), align 8
  store volatile i64 64, ptr inttoptr (i64 1610612752 to ptr), align 8
  call void @llvm.riscv.bosc.dma.fence()
  store volatile i64 1, ptr inttoptr (i64 1610612760 to ptr), align 8
  call void @llvm.riscv.bosc.dma.fence()
  %status = load volatile i64, ptr inttoptr (i64 1610612768 to ptr), align 8
  call void @llvm.riscv.bosc.dma.fence()
  br i1 %second, label %next, label %done
next:
  call void @llvm.riscv.bosc.dma.reads()
  store volatile i64 %source, ptr inttoptr (i64 1610612736 to ptr), align 8
  store volatile i64 536870976, ptr inttoptr (i64 1610612744 to ptr), align 8
  store volatile i64 64, ptr inttoptr (i64 1610612752 to ptr), align 8
  call void @llvm.riscv.bosc.dma.fence()
  store volatile i64 1, ptr inttoptr (i64 1610612760 to ptr), align 8
  call void @llvm.riscv.bosc.dma.fence()
  %status2 = load volatile i64, ptr inttoptr (i64 1610612768 to ptr), align 8
  call void @llvm.riscv.bosc.dma.fence()
  br label %done
done:
  ret void
}
