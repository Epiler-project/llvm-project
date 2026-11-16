; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/cast.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/shift.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/shuffle.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/push.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-STACK
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/round-vector.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WH-FIELD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/descale.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DESCALE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/tie.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TIE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/group.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GROUP
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/slot16.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SLOT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/gt.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/le.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LE
; WH-FIELD: Wormhole {{.*}} operand {{[0-9]+}} must be in
; WH-STACK: Tensix instruction is not supported on Wormhole
; DESCALE: Tensix floating conversion requires zero descale
; TIE: bound SFPU destructive tie requires identical physical registers
; GROUP: bound SFPU fixed register group does not match the instruction
; SLOT: bound SFPU invalid read register operand class
; POISON: defined and non-poison
; RANGE: bound Dst offset must be proven in [0, 1023]
; EXECUTOR: ncrisc cannot issue Tensix instructions
; GT: unsupported Tensix intrinsic ABI: llvm.tt.wh.sfpu.gt
; LE: unsupported Tensix intrinsic ABI: llvm.tt.wh.sfpu.le

; These cases distinguish legal BH fields from WH admission. A plain external
; call, silent field truncation, or receiver chosen only by suffix is incorrect.
;--- cast.ll
declare void @llvm.tt.wh.sfpu.cast(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @cast() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.cast(i32 0, i32 0, i32 9, i32 2)
  ret void
}
;--- shift.ll
declare void @llvm.tt.wh.sfpu.shft.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @shift() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.shft.v(i32 0, i32 0, i32 9, i32 2)
  ret void
}
;--- shuffle.ll
declare void @llvm.tt.wh.sfpu.shft2(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @shuffle() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.shft2(i32 0, i32 0, i32 1, i32 0, i32 4)
  ret void
}
;--- push.ll
declare void @llvm.tt.wh.sfpu.pushc(i32 immarg, i32 immarg)
define void @push() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.pushc(i32 0, i32 1)
  ret void
}
;--- load.ll
declare void @llvm.tt.wh.sfpu.load(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @load() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.load(i32 0, i32 0, i32 1023, i32 4, i32 4)
  ret void
}
;--- store.ll
declare void @llvm.tt.wh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
define void @store() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.store(i32 0, i32 1023, i32 4, i32 4)
  ret void
}
;--- round-vector.ll
declare void @llvm.tt.wh.sfpu.stochrnd.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @round_vector() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.stochrnd.v(i32 0, i32 0, i32 9, i32 10, i32 4, i32 2)
  ret void
}
;--- descale.ll
declare void @llvm.tt.wh.sfpu.stochrnd.i(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @descale() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.stochrnd.i(i32 0, i32 0, i32 9, i32 1, i32 0, i32 0)
  ret void
}
;--- tie.ll
declare void @llvm.tt.wh.sfpu.setexp.v(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @tie() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.setexp.v(i32 0, i32 0, i32 10, i32 1, i32 0)
  ret void
}
;--- group.ll
declare void @llvm.tt.wh.sfpu.shft2.copy4(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @group() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.shft2.copy4(i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 4, i32 0)
  ret void
}
;--- slot16.ll
declare void @llvm.tt.wh.sfpu.mov.all(i32 immarg, i32 immarg)
define void @slot16() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.mov.all(i32 0, i32 16)
  ret void
}
;--- poison.ll
declare void @llvm.tt.wh.sfpu.load(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @poison(i32 %offset) "tensix-executor"="trisc1" {
  %bounded = and i32 %offset, 1023
  call void @llvm.tt.wh.sfpu.load(i32 0, i32 0, i32 %bounded, i32 0, i32 4)
  ret void
}
;--- range.ll
declare void @llvm.tt.wh.sfpu.store(i32 immarg, i32, i32 immarg, i32 immarg)
define void @range(i32 noundef %offset) "tensix-executor"="trisc1" {
  %bounded = and i32 %offset, 2047
  call void @llvm.tt.wh.sfpu.store(i32 0, i32 %bounded, i32 0, i32 4)
  ret void
}
;--- executor.ll
declare void @llvm.tt.wh.sfpu.nop()
define void @executor() "tensix-executor"="ncrisc" {
  call void @llvm.tt.wh.sfpu.nop()
  ret void
}
;--- gt.ll
declare void @llvm.tt.wh.sfpu.gt(i32, i32, i32)
define void @gt() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.gt(i32 0, i32 9, i32 1)
  ret void
}
;--- le.ll
declare void @llvm.tt.wh.sfpu.le(i32, i32, i32)
define void @le() "tensix-executor"="trisc1" {
  call void @llvm.tt.wh.sfpu.le(i32 0, i32 9, i32 1)
  ret void
}
