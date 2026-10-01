; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-before=riscv-isel %t/retired.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETIRED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -stop-before=riscv-isel %t/retired.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETIRED
; RUN: llc -mtriple=riscv32 -mattr=+v -O0 -verify-machineinstrs %t/vector.ll -o - | FileCheck %s --check-prefix=VECTOR
; RUN: llc -mtriple=riscv32 -mattr=+v -O2 -verify-machineinstrs %t/vector.ll -o - | FileCheck %s --check-prefix=VECTOR

; A retired reserved intrinsic is rejected before selection, including forms
; with no vector result. Ordinary RVV values are not SFPU numerical carriers.
; RETIRED: unsupported Tensix intrinsic ABI: llvm.riscv.tt.sfpencc
; RETIRED-NOT: PLEASE submit a bug report
; VECTOR-LABEL: ordinary_vector:
; VECTOR: vadd.vv
; VECTOR-NOT: ttsfp
; VECTOR: ret

;--- retired.ll
declare void @llvm.riscv.tt.sfpencc(i32, i32)
define void @retired() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}

;--- vector.ll
define <32 x i32> @ordinary_vector(<32 x i32> %a, <32 x i32> %b) {
  %sum = add <32 x i32> %a, %b
  ret <32 x i32> %sum
}
