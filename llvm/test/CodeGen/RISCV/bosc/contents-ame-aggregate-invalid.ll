; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 %t/return.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ABI
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 %t/argument.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ABI
; RUN: not llc -mtriple=riscv64 -mattr=+xboscame -O0 %t/nested.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NESTED
; ABI: AME contents cannot cross a function ABI
; NESTED: AME contents cannot use aggregate carriers

;--- return.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
define { target("riscv.bosc.ame.tr"), i64 } @aggregate_return(ptr %p) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %p, i64 64, i64 %m, i64 %n, i64 %k)
  %result = insertvalue { target("riscv.bosc.ame.tr"), i64 } poison, target("riscv.bosc.ame.tr") %a, 0
  ret { target("riscv.bosc.ame.tr"), i64 } %result
}

;--- argument.ll
define void @aggregate_argument({ i64, [2 x target("riscv.bosc.ame.acc")] } %x) "trex-bosc-contents-ame"="1" {
  ret void
}

;--- nested.ll
declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
define void @nested_contents(ptr %p) "trex-bosc-contents-ame"="1" {
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %a = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %p, i64 64, i64 %m, i64 %n, i64 %k)
  %aggregate = insertvalue { i64, [1 x target("riscv.bosc.ame.tr")] } poison, target("riscv.bosc.ame.tr") %a, 1, 0
  %readback = extractvalue { i64, [1 x target("riscv.bosc.ame.tr")] } %aggregate, 1, 0
  ret void
}
