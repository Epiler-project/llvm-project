; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O2 -stress-regalloc=4 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR

declare i64 @llvm.riscv.bosc.setvl(i64)
declare void @llvm.riscv.bosc.load(i32 immarg, ptr)
declare void @llvm.riscv.bosc.store(i32 immarg, ptr)
declare void @llvm.riscv.bosc.sfu.exp2(i32 immarg, i32 immarg)
declare void @llvm.riscv.bosc.sfu.rcp(i32 immarg, i32 immarg)

; CHECK-LABEL: sfu:
; CHECK: vle32.v v31,
; CHECK: vsfu.exp2.v v30, v31
; CHECK: vsfu.rcp.v v29, v30
; CHECK: vse32.v v29,
; CHECK-NOT: call
; CHECK: ret
; MIR-LABEL: name: sfu
; MIR: $v30 = BOSC_SFU_EXP2 $v31
; MIR: $v29 = BOSC_SFU_RCP $v30
define void @sfu(ptr %source, ptr %destination) "trex-bosc-bound-rvv"="1" {
  %vl = call i64 @llvm.riscv.bosc.setvl(i64 16)
  call void @llvm.riscv.bosc.load(i32 31, ptr %source)
  call void @llvm.riscv.bosc.sfu.exp2(i32 30, i32 31)
  call void @llvm.riscv.bosc.sfu.rcp(i32 29, i32 30)
  call void @llvm.riscv.bosc.store(i32 29, ptr %destination)
  ret void
}
