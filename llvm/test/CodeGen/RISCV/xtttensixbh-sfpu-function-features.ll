; The module/default target has no Tensix feature. Register allocation must
; follow each function's subtarget while preserving ordinary scalar code.
; Gating the entire SFPU allocation pipeline on the default target is invalid.
; RUN: llc -mtriple=riscv32 -O0 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -stop-after=postrapseudos %s -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -O0 -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

; MIR-LABEL: name: ordinary
; MIR: LW
; MIR: ADD
; MIR: SW
; MIR: PseudoRET
define i32 @ordinary(ptr %p, i32 %increment) {
  %old = load volatile i32, ptr %p, align 4
  %value = add i32 %old, %increment
  store volatile i32 %value, ptr %p, align 4
  ret i32 %value
}
