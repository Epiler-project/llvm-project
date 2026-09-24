; RUN: split-file %s %t
; RUN: not opt -passes=verify -disable-output %t/direct.ll 2>&1 | FileCheck %s
; RUN: not opt -passes=verify -disable-output %t/port.ll 2>&1 | FileCheck %s
; CHECK: immarg operand has non-immediate parameter

;--- direct.ll
declare void @llvm.riscv.tt.setc16(i32, i32)
define void @test(i32 %value) {
  call void @llvm.riscv.tt.setc16(i32 %value, i32 1)
  ret void
}

;--- port.ll
declare void @llvm.riscv.tt.setc16.port(i32, i32, i32)
define void @test(i32 %port, i32 %value) {
  call void @llvm.riscv.tt.setc16.port(i32 %port, i32 %value, i32 1)
  ret void
}
