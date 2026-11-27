; RUN: llc -mtriple=riscv32-unknown-elf -mattr=+m -verify-machineinstrs %s -o - | FileCheck %s --implicit-check-not=__udivdi3 --implicit-check-not=__moddi3 --implicit-check-not=__umoddi3
; RUN: llc -mtriple=riscv32-unknown-elf -mattr=+m,+xtttensixwh -verify-machineinstrs %s -o - | FileCheck %s --implicit-check-not=__udivdi3 --implicit-check-not=__moddi3 --implicit-check-not=__umoddi3
; RUN: llc -mtriple=riscv32-unknown-elf -mattr=+m,+xtttensixbh -verify-machineinstrs %s -o - | FileCheck %s --implicit-check-not=__udivdi3 --implicit-check-not=__moddi3 --implicit-check-not=__umoddi3

; Firmware supplies no general-purpose division library. NCRISC shares its
; ordinary RV32IM ISA with hosted code, but uses the Tensix executor ABI.

; CHECK-LABEL: generic_before:
; CHECK: call __divdi3
define i64 @generic_before(i64 %a, i64 %b) {
  %result = sdiv i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: reader:
; CHECK-NOT: __divdi3
; CHECK: ret
define i64 @reader(i64 %a, i64 %b) "tensix-executor"="ncrisc" {
  %result = sdiv i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: writer:
; CHECK-NOT: __divdi3
; CHECK: ret
define i64 @writer(i64 %a, i64 %b) "tensix-executor"="brisc" {
  %result = udiv i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: unpack:
; CHECK-NOT: __divdi3
; CHECK: ret
define i64 @unpack(i64 %a, i64 %b) "tensix-executor"="trisc0" {
  %result = srem i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: math:
; CHECK-NOT: __divdi3
; CHECK: ret
define i64 @math(i64 %a, i64 %b) "tensix-executor"="trisc1" {
  %result = urem i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: pack:
; CHECK-NOT: __divdi3
; CHECK: ret
define i64 @pack(i64 %a, i64 %b) "tensix-executor"="trisc2" {
  %result = sdiv i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: generic_after:
; CHECK: call __divdi3
define i64 @generic_after(i64 %a, i64 %b) {
  %result = sdiv i64 %a, %b
  ret i64 %result
}

; CHECK-LABEL: native_divrem:
; CHECK-DAG: div
; CHECK-DAG: remu
; CHECK: ret
define i32 @native_divrem(i32 %x, i32 %y) "tensix-executor"="ncrisc" {
  %q = sdiv i32 %x, %y
  %r = urem i32 %x, %y
  %sum = add i32 %q, %r
  ret i32 %sum
}
