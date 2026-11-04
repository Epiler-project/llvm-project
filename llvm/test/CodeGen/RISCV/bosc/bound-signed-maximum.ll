; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.maximum(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.isplat(i32 immarg, i64)

; CHECK-LABEL: signed_maximum:
; CHECK: vsetvli {{[a-z][0-9]+}}, {{[a-z][0-9]+}}, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vle32.v v5
; CHECK: vmax.vv v4, v4, v5
; CHECK: vse32.v v4
; CHECK-NOT: vmaxu
; CHECK-NOT: vfmax
; CHECK-NOT: vmv
; CHECK-NOT: call
; MIR-LABEL: name: signed_maximum
; MIR: $v4 = VLE32_V
; MIR: $v5 = VLE32_V
; MIR: $v4 = VMAX_VV $v4, $v5
; MIR: VSE32_V $v4
define i64 @signed_maximum(ptr %a, ptr %b, ptr %out, i64 %avl) #0 {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %avl)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.load(i32 5, ptr %b)
  call void @llvm.riscv.bosc.maximum(i32 4, i32 4, i32 5)
  call void @llvm.riscv.bosc.store(i32 4, ptr %out)
  ret i64 %vl
}
attributes #0 = { "trex-bosc-bound-rvv"="1" }

; CHECK-LABEL: signed_threshold:
; CHECK: vsetvli {{[a-z][0-9]+}}, {{[a-z][0-9]+}}, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vmv.v.x v5, {{[a-z][0-9]+}}
; CHECK: vmax.vv v4, v4, v5
; CHECK: vse32.v v4
; CHECK-NOT: vf
; CHECK-NOT: fcvt
; CHECK-NOT: call
; MIR-LABEL: name: signed_threshold
; MIR: $v4 = VLE32_V
; MIR: $v5 = VMV_V_X
; MIR: $v4 = VMAX_VV $v4, $v5
; MIR: VSE32_V $v4
define i64 @signed_threshold(ptr %a, ptr %out, i64 %avl, i64 %threshold) #0 {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 %avl)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.isplat(i32 5, i64 %threshold)
  call void @llvm.riscv.bosc.maximum(i32 4, i32 4, i32 5)
  call void @llvm.riscv.bosc.store(i32 4, ptr %out)
  ret i64 %vl
}
