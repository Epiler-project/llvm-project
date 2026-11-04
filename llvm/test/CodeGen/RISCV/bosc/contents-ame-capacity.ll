; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/acc8.ll -o %t/acc8.O0.o
; RUN: llvm-objdump -d --mattr=+xboscame %t/acc8.O0.o > %t/acc8.O0.disasm
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/acc8.ll -o %t/acc8.O2.o
; RUN: llvm-objdump -d --mattr=+xboscame %t/acc8.O2.o > %t/acc8.O2.disasm
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/tr16.ll -o %t/tr16.O0.o
; RUN: llvm-objdump -d --mattr=+xboscame %t/tr16.O0.o > %t/tr16.O0.disasm
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/tr16.ll -o %t/tr16.O2.o
; RUN: llvm-objdump -d --mattr=+xboscame %t/tr16.O2.o > %t/tr16.O2.disasm
; RUN: %python %t/check.py %t/acc8.O0.disasm %t/acc8.O2.disasm %t/tr16.O0.disasm %t/tr16.O2.disasm
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/acc9.ll -o %t/acc9.O0.o 2>&1 | FileCheck %s --check-prefix=ACC
; RUN: test ! -e %t/acc9.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/acc9.ll -o %t/acc9.O2.o 2>&1 | FileCheck %s --check-prefix=ACC
; RUN: test ! -e %t/acc9.O2.o
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/tr17.ll -o %t/tr17.O0.o 2>&1 | FileCheck %s --check-prefix=TR
; RUN: test ! -e %t/tr17.O0.o
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/tr17.ll -o %t/tr17.O2.o 2>&1 | FileCheck %s --check-prefix=TR
; RUN: test ! -e %t/tr17.O2.o
;
; ACC: ran out of registers during register allocation in function 'acc9'
; TR: ran out of registers during register allocation in function 'tr17'
;
; Test both sides of the real AME files, with every distinct ordered load
; simultaneously live. Tr16 uses one A plus fifteen B values before fifteen
; destructive MMAs; Acc8 keeps all eight accumulators live before any store.
; Full physical slot coverage and the exact final matrix stream rule out hidden
; transfers, copies, spills or rematerialization. Overflow must publish no object.

;--- acc8.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @acc8(ptr %base) "trex-bosc-contents-ame"="1" {
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
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c0, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c1, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c2, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c3, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c4, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c5, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c6, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c7, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- acc9.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @acc9(ptr %base) "trex-bosc-contents-ame"="1" {
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

;--- tr16.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @tr16(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b0 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b1 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b2 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b3 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b4 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b5 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b6 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b7 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b8 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b9 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b10 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b11 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b12 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b13 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b14 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c0 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b0, i64 %m, i64 %n, i64 %k)
  %c1 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c0, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b1, i64 %m, i64 %n, i64 %k)
  %c2 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c1, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b2, i64 %m, i64 %n, i64 %k)
  %c3 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c2, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b3, i64 %m, i64 %n, i64 %k)
  %c4 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c3, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b4, i64 %m, i64 %n, i64 %k)
  %c5 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c4, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b5, i64 %m, i64 %n, i64 %k)
  %c6 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c5, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b6, i64 %m, i64 %n, i64 %k)
  %c7 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c6, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b7, i64 %m, i64 %n, i64 %k)
  %c8 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c7, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b8, i64 %m, i64 %n, i64 %k)
  %c9 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c8, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b9, i64 %m, i64 %n, i64 %k)
  %c10 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c9, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b10, i64 %m, i64 %n, i64 %k)
  %c11 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c10, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b11, i64 %m, i64 %n, i64 %k)
  %c12 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c11, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b12, i64 %m, i64 %n, i64 %k)
  %c13 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c12, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b13, i64 %m, i64 %n, i64 %k)
  %c14 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c13, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b14, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c14, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- tr17.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)
define void @tr17(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b0 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b1 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b2 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b3 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b4 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b5 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b6 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b7 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b8 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b9 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b10 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b11 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b12 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b13 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b14 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %b15 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %c0 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b0, i64 %m, i64 %n, i64 %k)
  %c1 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c0, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b1, i64 %m, i64 %n, i64 %k)
  %c2 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c1, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b2, i64 %m, i64 %n, i64 %k)
  %c3 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c2, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b3, i64 %m, i64 %n, i64 %k)
  %c4 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c3, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b4, i64 %m, i64 %n, i64 %k)
  %c5 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c4, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b5, i64 %m, i64 %n, i64 %k)
  %c6 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c5, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b6, i64 %m, i64 %n, i64 %k)
  %c7 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c6, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b7, i64 %m, i64 %n, i64 %k)
  %c8 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c7, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b8, i64 %m, i64 %n, i64 %k)
  %c9 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c8, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b9, i64 %m, i64 %n, i64 %k)
  %c10 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c9, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b10, i64 %m, i64 %n, i64 %k)
  %c11 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c10, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b11, i64 %m, i64 %n, i64 %k)
  %c12 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c11, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b12, i64 %m, i64 %n, i64 %k)
  %c13 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c12, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b13, i64 %m, i64 %n, i64 %k)
  %c14 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c13, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b14, i64 %m, i64 %n, i64 %k)
  %c15 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c14, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b15, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c15, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- check.py
from collections import Counter
from pathlib import Path
import re
import sys

for filename in sys.argv[1:]:
    path = Path(filename)
    instructions = []
    for line in path.read_text().splitlines():
        match = re.match(r"\s*[0-9a-f]+:\s+[0-9a-f]+\s+([^\s]+)\s*(.*)", line)
        if match:
            instructions.append(match.groups())
    assert instructions, (path, "missing disassembly")
    matrix = Counter(op for op, _ in instructions if op.startswith("m") and op != "mv")
    if path.name.startswith("acc8."):
        assert matrix == {
            "msettilem": 1, "msettilen": 1, "msettilek": 1,
            "mlce32.m": 8, "msce32.m": 8,
        }, (path, matrix)
        loaded = [args.split(",")[0] for op, args in instructions if op == "mlce32.m"]
        stored = [args.split(",")[0] for op, args in instructions if op == "msce32.m"]
        assert set(loaded) == {f"acc{i}" for i in range(8)}, (path, loaded)
        assert stored == loaded, (path, "accumulator contents reordered", loaded, stored)
        last_load = max(i for i, (op, _) in enumerate(instructions) if op == "mlce32.m")
        first_store = next(i for i, (op, _) in enumerate(instructions) if op == "msce32.m")
        assert last_load < first_store, (path, "accumulators not simultaneously live")
    else:
        assert matrix == {
            "msettilem": 1, "msettilen": 1, "msettilek": 1,
            "mlae8.m": 1, "mlbe8.m": 15, "mlce32.m": 1,
            "mqma.b.mm": 15, "msce32.m": 1,
        }, (path, matrix)
        a = next(args.split(",")[0] for op, args in instructions if op == "mlae8.m")
        bs = [args.split(",")[0] for op, args in instructions if op == "mlbe8.m"]
        assert set([a] + bs) == {f"tr{i}" for i in range(16)}, (path, a, bs)
        c = next(args.split(",")[0] for op, args in instructions if op == "mlce32.m")
        updates = [tuple(part.strip() for part in args.split(","))
                   for op, args in instructions if op == "mqma.b.mm"]
        assert updates == [(c, a, b) for b in bs], (path, updates)
        last_load = max(i for i, (op, _) in enumerate(instructions)
                        if op in {"mlae8.m", "mlbe8.m"})
        first_mma = next(i for i, (op, _) in enumerate(instructions) if op == "mqma.b.mm")
        assert last_load < first_mma, (path, "Tr values not simultaneously live")
        stored = [args.split(",")[0] for op, args in instructions if op == "msce32.m"]
        assert stored == [c], (path, stored)
    # Ordinary GPR copies/stack saves are legal; exact matrix counts above
    # exclude implicit restricted-register transfers. No helper is required.
    assert not any(op in {"call", "tail", "jal", "jalr", "jr"}
                   for op, _ in instructions), (path, "unexpected helper")
