; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/missing.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
; MISSING: requires a tensix-executor function attribute
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; EXECUTOR: invalid tensix-executor value
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/ncrisc.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NCRISC
; NCRISC: ncrisc cannot issue Tensix instructions
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; CONFIG: immediate operand 1 must be in [0, 67]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PORT
; PORT: unknown Tensix instruction port
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/remote.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=REMOTE
; REMOTE: remote Tensix instruction ports require brisc
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/ncport.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NCPORT
; NCPORT: ncrisc cannot issue Tensix instructions

;--- missing.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test()  {
  call void @llvm.riscv.tt.setc16(i32 0, i32 1)
  ret void
}

;--- executor.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="math" {
  call void @llvm.riscv.tt.setc16(i32 0, i32 1)
  ret void
}

;--- ncrisc.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="ncrisc" {
  call void @llvm.riscv.tt.setc16(i32 0, i32 1)
  ret void
}

;--- config.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setc16(i32 0, i32 68)
  ret void
}

;--- port.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="brisc" {
  call void @llvm.riscv.tt.setc16.port(i32 3, i32 0, i32 1)
  ret void
}

;--- remote.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setc16.port(i32 1, i32 0, i32 1)
  ret void
}

;--- ncport.ll
declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setc16.port(i32 immarg, i32, i32)
define void @test() "tensix-executor"="ncrisc" {
  call void @llvm.riscv.tt.setc16.port(i32 0, i32 0, i32 1)
  ret void
}
