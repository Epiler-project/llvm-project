; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -regalloc=fast -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -stop-after=twoaddressinstruction -verify-machineinstrs < %s | FileCheck %s --check-prefix=TIED
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -filetype=obj -verify-machineinstrs %s -o %t.o0.o
; RUN: llvm-objdump -d --mattr=+xboscame %t.o0.o | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -filetype=obj -verify-machineinstrs %s -o %t.o2.o
; RUN: llvm-objdump -d --mattr=+xboscame %t.o2.o | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -regalloc=fast -filetype=obj -verify-machineinstrs %s -o %t.fast.o
; RUN: llvm-objdump -d --mattr=+xboscame %t.fast.o | FileCheck %s --check-prefix=OBJ

; FP8 uses the same opaque Tr/Acc contents and actual dimension results as
; integer MMA. The format dependency, accumulator tie and operand dataflow must
; survive selection and allocation into a real floating matrix instruction.
; This proves offline code generation, not a particular hardware FP8 format's
; numerical behavior or admission of AMEFP8 images for execution.
declare void @llvm.riscv.bosc.ame.setfp8.e4m3()
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.fp.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @fp8_contents(ptr %a, ptr %b, ptr %c, ptr %out) "trex-bosc-contents-ame"="1" {
; CHECK-LABEL: fp8_contents:
; CHECK: msetfp8.e4m3 zero
; CHECK: msettilem
; CHECK: msettilen
; CHECK: msettilek
; CHECK: mlae8.m [[A:tr[0-9]+]],
; CHECK: mlbe8.m [[B:tr[0-9]+]],
; CHECK: mlce32.m [[C:acc[0-9]+]],
; CHECK-NEXT: mfqma.cf.mm [[C]], [[A]], [[B]]
; CHECK-NEXT: msce32.m [[C]],
; CHECK-NEXT: ret
;
; MIR-LABEL: name: fp8_contents
; MIR: BOSC_AME_SETFP8_E4M3 implicit-def $bosc_mtype
; MIR: BOSC_AME_SETM {{.*}}implicit-def $bosc_m
; MIR: BOSC_AME_SETN {{.*}}implicit-def $bosc_n
; MIR: BOSC_AME_SETK {{.*}}implicit-def $bosc_k
; MIR: [[MA:%[0-9]+]]:bosctrreg = BOSC_AME_LOADA {{.*}} :: (volatile load unknown-size, align 1)
; MIR: [[MB:%[0-9]+]]:bosctrreg = BOSC_AME_LOADB {{.*}} :: (volatile load unknown-size, align 1)
; MIR: [[MC:%[0-9]+]]:boscaccreg = BOSC_AME_LOADC {{.*}} :: (volatile load unknown-size, align 1)
; MIR-NEXT: [[NEXT:%[0-9]+]]:boscaccreg = BOSC_AME_FPMMA [[MC]], killed [[MA]], killed [[MB]], implicit $bosc_m, implicit $bosc_n, implicit $bosc_k, implicit $bosc_mtype, implicit $bosc_issue, implicit-def $bosc_issue
; MIR-NEXT: BOSC_AME_STOREC killed [[NEXT]], {{.*}} :: (volatile store unknown-size, align 1)
;
; TIED-LABEL: name: fp8_contents
; TIED: [[TC:%[0-9]+]]:boscaccreg = BOSC_AME_FPMMA [[TC]],
;
; OBJ-LABEL: <fp8_contents>:
; OBJ: 020b6077 {{.*}}msetfp8.e4m3 zero
; OBJ: msettilem
; OBJ: msettilen
; OBJ: msettilek
; OBJ: mlae8.m [[OA:tr[0-9]+]],
; OBJ: mlbe8.m [[OB:tr[0-9]+]],
; OBJ: mlce32.m [[OC:acc[0-9]+]],
; OBJ-NEXT: {{[0-9a-f]+}}: 2a{{[0-9a-f]{6}[[:space:]]+}}mfqma.cf.mm [[OC]], [[OA]], [[OB]]
; OBJ-NEXT: msce32.m [[OC]],
; OBJ-NEXT: ret
  call void @llvm.riscv.bosc.ame.setfp8.e4m3()
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %n, i64 %k)
  %bv = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b, i64 16, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 64, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.fp.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %out, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}
