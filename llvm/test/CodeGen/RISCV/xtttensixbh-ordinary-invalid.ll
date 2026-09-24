; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/static.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/undef.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/flagged.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFINED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PORT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; STATIC: immediate operand 0 must be in [0, 32767]
; RANGE: STALLWAIT field wait_res must be proven in [0, 32767]
; DEFINED: STALLWAIT field wait_res must be defined and non-poison
; PORT: remote Tensix instruction ports require brisc
; CONFIG: SETC16 field setc16_reg must be proven in [0, 67]

;--- static.ll
declare void @llvm.riscv.tt.stallwait(i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.stallwait(i32 32768, i32 0)
  ret void
}
;--- range.ll
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
define void @test(i32 noundef %value) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.stallwait.port(i32 0, i32 %value, i32 0)
  ret void
}
;--- poison.ll
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
define void @test(i32 %input) "tensix-executor"="trisc1" {
  %value = and i32 %input, 32767
  call void @llvm.riscv.tt.stallwait.port(i32 0, i32 %value, i32 0)
  ret void
}
;--- undef.ll
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.stallwait.port(i32 0, i32 undef, i32 0)
  ret void
}
;--- flagged.ll
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
define void @test(i32 noundef %input) "tensix-executor"="trisc1" {
  %overflow = add nuw i32 %input, 1
  %value = and i32 %overflow, 32767
  call void @llvm.riscv.tt.stallwait.port(i32 0, i32 %value, i32 0)
  ret void
}
;--- port.ll
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.stallwait.port(i32 1, i32 0, i32 0)
  ret void
}
;--- config.ll
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test(i32 noundef %input) "tensix-executor"="trisc1" {
  %config = and i32 %input, 255
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 1, i32 %config)
  ret void
}
