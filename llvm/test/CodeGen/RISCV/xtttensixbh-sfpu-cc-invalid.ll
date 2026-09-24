; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/underflow.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNDERFLOW
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/backedge.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MERGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MERGE
; UNDERFLOW: CC stack underflow
; MERGE: CC stack depth disagrees at CFG join or backedge

;--- underflow.ll
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  ret void
}
;--- backedge.ll
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
define void @test(i1 %again) "tensix-executor"="trisc1" {
entry:
  br label %loop
loop:
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  br label %loop
}
;--- join.ll
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
define void @test(i1 %condition) "tensix-executor"="trisc1" {
entry:
  br i1 %condition, label %push, label %join
push:
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  br label %join
join:
  ret void
}
