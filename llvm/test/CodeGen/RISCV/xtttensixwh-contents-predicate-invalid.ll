; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 %t/replace.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 %t/replace.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/replace.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/replace.ll -o /dev/null
; WH: Tensix instruction is not supported on Wormhole

; The same typed contents call still selects Blackhole's existing replacement
; form. It must not acquire WH support when plain PUSHC is shared.
;--- replace.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.pushc(i32 immarg)
declare void @llvm.riscv.tt.contents.popc(i32 immarg)
define void @replace_top_bh_only() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.pushc(i32 0)
  call void @llvm.riscv.tt.contents.pushc(i32 1)
  call void @llvm.riscv.tt.contents.popc(i32 0)
  ret void
}
