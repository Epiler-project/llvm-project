; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s --check-prefix=PRESSURE
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.add(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)

; CHECK-LABEL: bound_add:
; CHECK: vsetvli {{[a-z][0-9]+}}, {{[a-z][0-9]+}}, e32, m1, ta, ma
; CHECK-DAG: vle32.v v4, ({{[a-z][0-9]+}})
; CHECK-DAG: vle32.v v5, ({{[a-z][0-9]+}})
; CHECK: vadd.vv v6, v4, v5
; CHECK: vse32.v v6, ({{[a-z][0-9]+}})
; CHECK-NOT: vmv
; CHECK-NOT: call
; PRESSURE-LABEL: bound_add:
; PRESSURE: vle32.v v4
; PRESSURE: vle32.v v5
; PRESSURE: vadd.vv v6, v4, v5
; PRESSURE: vse32.v v6
; PRESSURE-NOT: vmv
; PRESSURE-NOT: vs1r
; PRESSURE-NOT: vl1r
; MIR-LABEL: name: bound_add
; MIR: $v4 = VLE32_V
; MIR: $v5 = VLE32_V
; MIR: $v6 = VADD_VV $v4, $v5
; MIR: VSE32_V $v6

define i64 @bound_add(ptr %a, ptr %b, ptr %c, i64 %avl) #0 {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %avl)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.load(i32 5, ptr %b)
  call void @llvm.riscv.bosc.add(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.store(i32 6, ptr %c)
  ret i64 %vl
}
attributes #0 = { "trex-bosc-bound-rvv"="1" }

; AVL=0 must mean zero active lanes, never the VSETVLI rs1=x0 VLMAX form.
; CHECK-LABEL: zero_avl:
; CHECK: vsetivli {{[a-z][0-9]+}}, 0, e32, m1, ta, ma
; CHECK: ret
define i64 @zero_avl() #0 {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 0)
  ret i64 %vl
}

; Unused actual VL still retains the explicit configuration effect.
; CHECK-LABEL: config_only:
; CHECK: vsetivli {{zero|[a-z][0-9]+}}, 16, e32, m1, ta, ma
; CHECK: ret
define void @config_only() #0 {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  ret void
}
