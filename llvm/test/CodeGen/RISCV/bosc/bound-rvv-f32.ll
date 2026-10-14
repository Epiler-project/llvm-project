; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.fadd(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fsub(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fmul(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.fdiv(i32 immarg, i32 immarg, i32 immarg)

; CHECK-LABEL: fp32_four:
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vle32.v v5
; CHECK: vfadd.vv v6, v4, v5
; CHECK: vfsub.vv v6, v6, v5
; CHECK: vfmul.vv v6, v6, v5
; CHECK: vfdiv.vv v6, v6, v5
; CHECK: vse32.v v6
; CHECK-NOT: vmv
; CHECK-NOT: vs1r
; CHECK-NOT: vl1r
; CHECK-NOT: call
; MIR-LABEL: name: fp32_four
; MIR: $v6 = VFADD_VV $v4, $v5, $noreg, implicit $frm, implicit $vl, implicit $vtype
; MIR: $v6 = VFSUB_VV $v6, $v5, $noreg, implicit $frm, implicit $vl, implicit $vtype
; MIR: $v6 = VFMUL_VV $v6, $v5, $noreg, implicit $frm, implicit $vl, implicit $vtype
; MIR: $v6 = VFDIV_VV $v6, $v5, $noreg, implicit $frm, implicit $vl, implicit $vtype
define void @fp32_four(ptr %a, ptr %b, ptr %out) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  call void @llvm.riscv.bosc.load(i32 5, ptr %b)
  call void @llvm.riscv.bosc.fadd(i32 6, i32 4, i32 5)
  call void @llvm.riscv.bosc.fsub(i32 6, i32 6, i32 5)
  call void @llvm.riscv.bosc.fmul(i32 6, i32 6, i32 5)
  call void @llvm.riscv.bosc.fdiv(i32 6, i32 6, i32 5)
  call void @llvm.riscv.bosc.store(i32 6, ptr %out)
  ret void
}

; CHECK-LABEL: fp32_tails:
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vle32.v v4
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vle32.v v5
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vfadd.vv v4, v4, v4
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vfdiv.vv v5, v5, v5
; CHECK: vsetivli {{.*}}, 16, e32, m1, ta, ma
; CHECK: vse32.v v4
; CHECK: vsetivli {{.*}}, 15, e32, m1, ta, ma
; CHECK: vse32.v v5
; MIR-LABEL: name: fp32_tails
; MIR: $v4 = VFADD_VV $v4, $v4, $noreg, implicit $frm, implicit $vl, implicit $vtype
; MIR: $v5 = VFDIV_VV $v5, $v5, $noreg, implicit $frm, implicit $vl, implicit $vtype
define void @fp32_tails(ptr %a, ptr %out) "trex-bosc-bound-rvv"="1" {
  %vl0 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 4, ptr %a)
  %vl1 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.load(i32 5, ptr %a)
  %vl2 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.fadd(i32 4, i32 4, i32 4)
  %vl3 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.fdiv(i32 5, i32 5, i32 5)
  %vl4 = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.store(i32 4, ptr %out)
  %vl5 = call i64 @llvm.riscv.bosc.setvl(i64 15)
  call void @llvm.riscv.bosc.store(i32 5, ptr %out)
  ret void
}
