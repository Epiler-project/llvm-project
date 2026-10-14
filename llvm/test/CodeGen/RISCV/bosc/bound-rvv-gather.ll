; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.gather.scalar(i32 immarg, i32 immarg, i64 immarg)

; CHECK-LABEL: lane0:
; CHECK: vrgather.vx v2, v1,
; CHECK: vse32.v v2,
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: lane0
; MIR: $v2 = VRGATHER_VX $v1, {{.*}}, $noreg, implicit $vl, implicit $vtype
define void @lane0(ptr %input, ptr %output) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 0)
  call void @llvm.riscv.bosc.store(i32 2, ptr %output)
  ret void
}

; CHECK-LABEL: lane1:
; CHECK: vrgather.vx v2, v1,
; CHECK: vse32.v v2,
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: lane1
; MIR: $v2 = VRGATHER_VX $v1, {{.*}}, $noreg, implicit $vl, implicit $vtype
define void @lane1(ptr %input, ptr %output) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 1)
  call void @llvm.riscv.bosc.store(i32 2, ptr %output)
  ret void
}

; CHECK-LABEL: lane14:
; CHECK: vrgather.vx v2, v1,
; CHECK: vse32.v v2,
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: lane14
; MIR: $v2 = VRGATHER_VX $v1, {{.*}}, $noreg, implicit $vl, implicit $vtype
define void @lane14(ptr %input, ptr %output) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 14)
  call void @llvm.riscv.bosc.store(i32 2, ptr %output)
  ret void
}

; CHECK-LABEL: lane15:
; CHECK: vrgather.vx v2, v1,
; CHECK: vse32.v v2,
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: lane15
; MIR: $v2 = VRGATHER_VX $v1, {{.*}}, $noreg, implicit $vl, implicit $vtype
define void @lane15(ptr %input, ptr %output) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %full = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 1, ptr %input)
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 15)
  call void @llvm.riscv.bosc.store(i32 2, ptr %output)
  ret void
}

; CHECK-LABEL: lane16:
; CHECK: vrgather.vx v2, v1,
; CHECK: vse32.v v2,
; CHECK-NOT: vl1r
; CHECK-NOT: vs1r
; MIR-LABEL: name: lane16
; MIR: $v2 = VRGATHER_VX $v1, {{.*}}, $noreg, implicit $vl, implicit $vtype
define void @lane16(ptr %input, ptr %output) vscale_range(8,8) "trex-bosc-bound-rvv"="1" {
  %one = call i64 @llvm.riscv.bosc.setvl(i64 1)
  call void @llvm.riscv.bosc.gather.scalar(i32 2, i32 1, i64 16)
  call void @llvm.riscv.bosc.store(i32 2, ptr %output)
  ret void
}
