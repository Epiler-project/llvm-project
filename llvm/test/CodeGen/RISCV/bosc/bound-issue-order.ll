; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zvl512b,+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zvl512b,+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zvl512b,+xboscame -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zvl512b,+xboscame -O2 -stop-after=machine-scheduler < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+m,+v,+zvl512b,+xboscame -O2 -stop-after=machine-scheduler < %s | FileCheck %s --check-prefix=SCALAR
;
; MLIR's issue stream is one sequence across RVV, AME, DMA and volatile MMIO.
; Explicit physical data and old-destination dependencies remain unchanged.
; The machine-only state must not appear as an encoded operand in assembly.
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.fmax(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fmacc(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fcmp.lt(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.merge(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.dma.fence()
declare void @llvm.riscv.bosc.dma.reads()
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)

; ASM-LABEL: common_issue:
; ASM: vle32.v v4,
; ASM: vle32.v v5,
; ASM: vfmax.vv v6, v4, v5
; ASM: vfmacc.vv v6, v4, v5
; ASM: vmflt.vv v0, v4, v5
; ASM: fence
; ASM: msettilem
; ASM: msettilen
; ASM: msettilek
; ASM: mlae8.m tr14,
; ASM: vmerge.vvm v6, v4, v5, v0
; ASM: sd
; ASM: fence
; ASM: csrw
; ASM: fence
; ASM: vse32.v v6,
; ASM-NOT: bosc_issue
; ASM: ret
; MIR-LABEL: name: common_issue
; MIR: VSETIVLI {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v4 = VLE32_V {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v5 = VLE32_V {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v6 = VFMAX_VV {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v6 = VFMACC_VV $v6, $v4, $v5, $noreg, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v0 = VMFLT_VV {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: FENCE 15, 15, implicit $bosc_issue, implicit-def $bosc_issue
; MIR: BOSC_AME_SETM {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: BOSC_AME_SETN {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: BOSC_AME_SETK {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $bosc_tr14 = BOSC_AME_LOADA {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v6 = VMERGE_VVM {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: SD {{.*}}implicit $bosc_issue, implicit-def $bosc_issue :: (volatile store
; MIR: FENCE 15, 15, implicit $bosc_issue, implicit-def $bosc_issue
; MIR: CSRRW {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: FENCE 15, 15, implicit $bosc_issue, implicit-def $bosc_issue
; MIR: VSE32_V {{.*}}implicit $bosc_issue, implicit-def $bosc_issue

define void @common_issue(ptr %a, ptr %b, ptr %out, ptr %mmio, i64 %value)
    "trex-bosc-bound-rvv"="1" "trex-bosc-bound-ame"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.load(i32 5, ptr %b)
  call void @llvm.riscv.bosc.fmax(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.fmacc(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.fcmp.lt(i32 0, i32 4, i32 5)
  call void @llvm.riscv.bosc.dma.fence()
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 3)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  call void @llvm.riscv.bosc.ame.load.a(i32 14, ptr %a, i64 4)
  call void @llvm.riscv.bosc.merge(i32 6, i32 4, i32 5, i32 0)
  store volatile i64 %value, ptr %mmio, align 8
  call void @llvm.riscv.bosc.dma.reads()
  call void @llvm.riscv.bosc.store(i32 6, ptr %out)
  ret void
}

; Plain scalar arithmetic and nonvolatile scalar memory have no issue effects.
; This checks the dependency fact, not a scheduler heuristic or a fixed GPR.
; MIR-LABEL: name: scalar_remains_schedulable
; MIR-DAG: = LD {{.*}}{{$}}
; MIR-DAG: = MUL {{.*}}{{$}}
; MIR: = ADD {{.*}}{{$}}
; MIR: SD {{.*}}{{$}}
; MIR: PseudoRET
; SCALAR-LABEL: name: scalar_remains_schedulable
; SCALAR-NOT: bosc_issue
; SCALAR: PseudoRET
; ASM-LABEL: scalar_remains_schedulable:
; ASM: mul
; ASM: ret

define void @scalar_remains_schedulable(ptr %input, ptr %out, i64 %x, i64 %y)
    "trex-bosc-bound-rvv"="1" {
  %loaded = load i64, ptr %input, align 8
  %product = mul i64 %x, %y
  %sum = add i64 %loaded, %product
  store i64 %sum, ptr %out, align 8
  ret void
}
