; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O0 -verify-machineinstrs -filetype=obj %s -o %t.O0.o
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscsfu -O2 -verify-machineinstrs -filetype=obj %s -o %t.O2.o

declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.rcp.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

define void @contents_sfu(ptr %in, ptr %out, i64 %n) #0 {
; CHECK-LABEL: contents_sfu:
; CHECK: vsetvli {{.*}}, e32, m1
; CHECK: vle32.v [[X:v[0-9]+]],
; CHECK: vsfu.exp2.v [[Y:v[0-9]+]], [[X]]
; CHECK: vsfu.rcp.v [[Z:v[0-9]+]], [[Y]]
; CHECK: vse32.v [[Z]],
; ISEL-LABEL: name: contents_sfu
; ISEL: class: vr
; ISEL: PseudoBOSC_SFU_EXP2_M1
; ISEL: PseudoBOSC_SFU_RCP_M1
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %x = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %in, i64 %vl)
  %y = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> %x, i64 %vl)
  %z = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.rcp.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> %y, i64 %vl)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %z, ptr %out, i64 %vl)
  ret void
}

attributes #0 = { "trex-bosc-contents-rvv"="1" vscale_range(8,8) }
