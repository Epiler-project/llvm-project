; RUN: not llc -mtriple=riscv64 -mattr=+xboscame < %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: ran out of registers during register allocation
;
; Nine distinct accumulator definitions are simultaneously live in an eight-slot
; file. The receiver must diagnose pressure, never invent a matrix stack spill.
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @acc_pressure(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %c0 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c1 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c2 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c3 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c4 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c5 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c6 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c7 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c8 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c0, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c1, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c2, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c3, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c4, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c5, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c6, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c7, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c8, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}
