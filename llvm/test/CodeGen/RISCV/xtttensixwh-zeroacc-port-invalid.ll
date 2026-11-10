; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/range16.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE16
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/range32.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE32
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/config.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/other.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OTHER
; RANGE16: ZEROACC field where must be proven in [0, 63]
; RANGE32: ZEROACC field where must be proven in [0, 31]
; POISON: ZEROACC field where must be defined and non-poison
; CONFIG: expected an immediate operand
; OTHER: Tensix intrinsic is not supported on Wormhole

; Dynamic where does not relax the physical face range or permit dynamic
; configuration, unrelated instruction ports or potentially poison values.
;--- range16.ll
declare void @llvm.riscv.tt.zeroacc.port(i32 immarg, i32, i32, i32, i32, i32)
define void @wide(i32 noundef %face) "tensix-executor"="trisc1" {
  %bounded = and i32 %face, 127
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 %bounded, i32 0, i32 0, i32 0, i32 1)
  ret void
}
;--- range32.ll
declare void @llvm.riscv.tt.zeroacc.port(i32 immarg, i32, i32, i32, i32, i32)
define void @wide(i32 noundef %face) "tensix-executor"="trisc1" {
  %bounded = and i32 %face, 63
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 %bounded, i32 0, i32 0, i32 1, i32 1)
  ret void
}
;--- poison.ll
declare void @llvm.riscv.tt.zeroacc.port(i32 immarg, i32, i32, i32, i32, i32)
define void @possibly_poison(i32 %face) "tensix-executor"="trisc1" {
  %bounded = and i32 %face, 63
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 %bounded, i32 0, i32 0, i32 0, i32 1)
  ret void
}
;--- config.ll
declare void @llvm.riscv.tt.zeroacc.port(i32 immarg, i32, i32, i32, i32, i32)
define void @dynamic_configuration(i32 noundef %mode) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 0, i32 0, i32 0, i32 0, i32 %mode)
  ret void
}
;--- other.ll
declare void @llvm.riscv.tt.stallwait.port(i32 immarg, i32, i32)
define void @other_port() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.stallwait.port(i32 0, i32 0, i32 0)
  ret void
}
