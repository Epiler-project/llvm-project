; RUN: not llc -mtriple=riscv32 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING
; RUN: not llc -mtriple=riscv64 -mattr=+xtttensixbh %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=RV32
; RUN: not llc -mtriple=riscv32be -mattr=+xtttensixbh %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=RV32
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh,+c %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMPRESSED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh,+zca %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=COMPRESSED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh,+v %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh,+zve32x %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR
; MISSING: Tensix intrinsic requires +xtttensixbh
; RV32: XTTTensixBH requires RV32 little-endian
; COMPRESSED: XTTTensixBH is incompatible with C/Zc extensions
; VECTOR: XTTTensixBH is incompatible with V/Zve extensions

declare void @llvm.riscv.tt.setc16(i32 immarg, i32 immarg)
define void @test() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.setc16(i32 0, i32 1)
  ret void
}
