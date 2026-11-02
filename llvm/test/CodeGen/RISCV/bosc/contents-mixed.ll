; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame,+xboscsfu -O0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame,+xboscsfu -O2 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame,+xboscsfu -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl512b,+xboscame,+xboscsfu -filetype=obj -verify-machineinstrs %s -o %t.o

; A source-selected B tile is an ordinary CFG PHI of contents, sharing one
; function with real RVV/SFU contents. There is no physical register selector,
; AME copy, auxiliary spill or duplicated matrix operation.
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
declare i64 @llvm.riscv.vsetvli.i64(i64, i64 immarg, i64 immarg)
declare <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32>, ptr, i64)
declare void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32>, ptr, i64)
declare <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32>, <vscale x 2 x i32>, i64)

; CHECK-LABEL: mixed_contents_selected_b:
; CHECK: vle32.v
; CHECK: vsfu.exp2.v
; CHECK: mlae8.m
; CHECK: mlce32.m
; CHECK: beqz
; CHECK: mlbe8.m
; CHECK: mlbe8.m
; CHECK: mqma.b.mm
; CHECK: msce32.m
; CHECK: vse32.v
; MIR-LABEL: name: mixed_contents_selected_b
; MIR: PseudoBOSC_SFU_EXP2_M1
; MIR: BOSC_AME_LOADA
; MIR: BOSC_AME_LOADC
; MIR: BOSC_AME_LOADB
; MIR: BOSC_AME_LOADB
; MIR: :bosctrreg = PHI
; MIR: BOSC_AME_MMA
define void @mixed_contents_selected_b(ptr %a, ptr %b0, ptr %b1, ptr %c,
                                      ptr %matrix_out, ptr %vector_in,
                                      ptr %vector_out, i64 %n, i1 %choose) #0 {
entry:
  %vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
  %x = call <vscale x 2 x i32> @llvm.riscv.bosc.rvv.contents.load(<vscale x 2 x i32> poison, ptr %vector_in, i64 %vl)
  %y = call <vscale x 2 x i32> @llvm.riscv.bosc.sfu.exp2.contents(<vscale x 2 x i32> poison, <vscale x 2 x i32> %x, i64 %vl)
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %cols = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %cols, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 64, i64 %m, i64 %cols, i64 %k)
  br i1 %choose, label %yes, label %no
yes:
  %bv0 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b0, i64 16, i64 %m, i64 %cols, i64 %k)
  br label %join
no:
  %bv1 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b1, i64 16, i64 %m, i64 %cols, i64 %k)
  br label %join
join:
  %bv = phi target("riscv.bosc.ame.tr") [ %bv0, %yes ], [ %bv1, %no ]
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv, i64 %m, i64 %cols, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %matrix_out, i64 64, i64 %m, i64 %cols, i64 %k)
  call void @llvm.riscv.bosc.rvv.contents.store(<vscale x 2 x i32> %y, ptr %vector_out, i64 %vl)
  ret void
}

attributes #0 = { "trex-bosc-contents-rvv"="1" "trex-bosc-contents-ame"="1" }
