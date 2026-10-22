; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 %t/no-mode.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 %t/partial-mode.ll -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 %t/partial-mode.ll -o /dev/null 2>&1 | FileCheck %s
; CHECK: floating MMA requires E4M3 configuration on every incoming path

;--- no-mode.ll
declare void @llvm.riscv.bosc.ame.setfp8.e4m3()
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.store.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.fp.mma(i32 immarg, i32 immarg, i32 immarg)

; All matrix inputs are initialized, but no E4M3 mode was established.
define void @fp8_e4m3(ptr %a, ptr %b, ptr %c, i64 %extent, i64 %padding) "trex-bosc-bound-ame"="1" {
  %low = and i64 %extent, 63
  %requested = add i64 %low, 1
  %pad = and i64 %padding, 252
  %stride = add i64 %pad, 256
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 %requested)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 %requested)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 %requested)
  call void @llvm.riscv.bosc.ame.load.a(i32 0, ptr %a, i64 64)
  call void @llvm.riscv.bosc.ame.load.b(i32 1, ptr %b, i64 64)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %c, i64 %stride)
  call void @llvm.riscv.bosc.ame.fp.mma(i32 0, i32 0, i32 1)
  call void @llvm.riscv.bosc.ame.store.c(i32 0, ptr %c, i64 %stride)
  ret void
}

;--- partial-mode.ll
declare void @llvm.riscv.bosc.ame.setfp8.e4m3()
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare void @llvm.riscv.bosc.ame.load.a(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.b(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.load.c(i32 immarg, ptr, i64)
declare void @llvm.riscv.bosc.ame.fp.mma(i32 immarg, i32 immarg, i32 immarg)

; A must-state analysis must not publish a mode set on only one predecessor.
define void @mode_missing_on_one_path(ptr %a, ptr %b, ptr %c, i1 %choose) "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 1)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 1)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 1)
  call void @llvm.riscv.bosc.ame.load.a(i32 0, ptr %a, i64 1)
  call void @llvm.riscv.bosc.ame.load.b(i32 1, ptr %b, i64 1)
  call void @llvm.riscv.bosc.ame.load.c(i32 0, ptr %c, i64 4)
  br i1 %choose, label %setmode, label %merge
setmode:
  call void @llvm.riscv.bosc.ame.setfp8.e4m3()
  br label %merge
merge:
  call void @llvm.riscv.bosc.ame.fp.mma(i32 0, i32 0, i32 1)
  ret void
}
