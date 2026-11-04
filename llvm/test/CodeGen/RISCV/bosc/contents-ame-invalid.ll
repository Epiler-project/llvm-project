; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/role.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ROLE
; ROLE: AME contents have incompatible role or configuration
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/stale-config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; CONFIG: AME contents require the actual reaching M/N/K configuration
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/stride.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STRIDE
; STRIDE: AME contents stride does not cover its configured row
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/mixed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MIXED
; MIXED: bound and contents AME ABIs cannot share a function
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/missing-scope.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SCOPE
; SCOPE: contents intrinsic requires trex-bosc-contents-ame=1
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame %t/old-live.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=COPY
; COPY: cannot synthesize a matrix register copy

;--- role.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @role(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c, target("riscv.bosc.ame.tr") %b, target("riscv.bosc.ame.tr") %a, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- stale-config.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @stale_config(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %new = call i64 @llvm.riscv.bosc.ame.setn(i64 32)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- stride.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @stride(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- mixed.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @mixed(ptr %base) "trex-bosc-contents-ame"="1" "trex-bosc-bound-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- missing-scope.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @missing_scope(ptr %base)  {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

;--- old-live.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @old_live(ptr %base) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %b = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %base, i64 16, i64 %m, i64 %n, i64 %k)
  %c = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  %next = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %c, target("riscv.bosc.ame.tr") %a, target("riscv.bosc.ame.tr") %b, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %c, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %next, ptr %base, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}
