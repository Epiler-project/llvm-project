; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=call --implicit-check-not=tail --implicit-check-not=fence
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM --implicit-check-not=call --implicit-check-not=tail --implicit-check-not=fence
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame -O2 -stop-after=machine-scheduler -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIR
; RUN: not llc -mtriple=riscv32 -mattr=+v,+zvl512b,+xboscame -O0 < %s 2>&1 | FileCheck %s --check-prefix=RV32
;
; BOSC observations read the machine counter through a fixed typed intrinsic.
; Scalar-only observation needs neither RVV binding nor a vector configuration.
; The standard LLVM counter retains its ordinary RISC-V CYCLE selection.
declare i64 @llvm.readcyclecounter()
declare i64 @llvm.riscv.bosc.read.mcycle()
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.add(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)

; ASM-LABEL: generic_counter:
; ASM: rdcycle
; ASM: ret
; MIR-LABEL: name: generic_counter
; MIR: CSRRS 3072,
define i64 @generic_counter() {
  %cycles = call i64 @llvm.readcyclecounter()
  ret i64 %cycles
}

; Scalar memory and counter ordering survive codegen. Even the unused sample
; remains an observation, and no runtime helper or hardware fence is inserted.
; ASM-LABEL: scalar_counter:
; ASM: sd
; ASM: csrr {{[a-z][0-9]+}}, mcycle
; ASM: sd
; ASM: csrr {{[a-z][0-9]+}}, mcycle
; ASM: sd
; ASM: csrr {{zero|[a-z][0-9]+}}, mcycle
; ASM: ret
; MIR-LABEL: name: scalar_counter
; MIR: SD {{.*}} :: (volatile store
; MIR: CSRRS 2816,
; MIR: SD {{.*}} :: (volatile store
; MIR: CSRRS 2816,
; MIR: SD {{.*}} :: (volatile store
; MIR: CSRRS 2816,
; RV32: BOSC observation counter requires RV64
define i64 @scalar_counter(ptr %out) {
  store volatile i64 11, ptr %out, align 8
  %begin = call i64 @llvm.riscv.bosc.read.mcycle()
  store volatile i64 22, ptr %out, align 8
  %end = call i64 @llvm.riscv.bosc.read.mcycle()
  %delta = sub i64 %end, %begin
  store volatile i64 %delta, ptr %out, align 8
  %unused = call i64 @llvm.riscv.bosc.read.mcycle()
  ret i64 %delta
}

; A physical vector remains live across both samples. CSRRS participates in
; the same issue dependency as the accelerator operations after selection.
; ASM-LABEL: vector_counter:
; ASM: vle32.v v4,
; ASM: csrr {{[a-z][0-9]+}}, mcycle
; ASM: vadd.vv v4, v4, v4
; ASM: csrr {{[a-z][0-9]+}}, mcycle
; ASM: vse32.v v4,
; ASM: ret
; MIR-LABEL: name: vector_counter
; MIR: $v4 = VLE32_V {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: CSRRS 2816, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: $v4 = VADD_VV $v4, $v4, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: CSRRS 2816, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: VSE32_V $v4, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
define i64 @vector_counter(ptr %input, ptr %output)
    "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %input)
  %begin = call i64 @llvm.riscv.bosc.read.mcycle()
  call void @llvm.riscv.bosc.add(i32 4, i32 4, i32 4)
  %end = call i64 @llvm.riscv.bosc.read.mcycle()
  call void @llvm.riscv.bosc.store(i32 4, ptr %output)
  %delta = sub i64 %end, %begin
  ret i64 %delta
}

; Scalar observation has no AME slot or configuration effect either.
; ASM-LABEL: matrix_counter:
; ASM: mlce32.m acc7,
; ASM: csrr {{[a-z][0-9]+}}, mcycle
; ASM: msce32.m acc7,
; ASM: ret
; MIR-LABEL: name: matrix_counter
; MIR: $bosc_acc7 = BOSC_AME_LOADC {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: CSRRS 2816, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
; MIR: BOSC_AME_STOREC $bosc_acc7, {{.*}}implicit $bosc_issue, implicit-def $bosc_issue
define i64 @matrix_counter(ptr %input, ptr %output)
    "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 2)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 3)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 4)
  call void @llvm.riscv.bosc.ame.load.c(i32 7, ptr %input, i64 12)
  %cycles = call i64 @llvm.riscv.bosc.read.mcycle()
  call void @llvm.riscv.bosc.ame.store.c(i32 7, ptr %output, i64 12)
  ret i64 %cycles
}
