; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=WH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ

; These direct native issues preserve all five authored logical fields.
; Actual source ownership, payload initialization and completion remain
; independent compiler contracts; this file grants none of those facts.
declare void @llvm.tt.elwadd(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.elwsub(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.elwmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)

define void @address_only() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: address_only
; ISEL: TTELWADD 0, 1, 0, 0, 0
; ISEL-NOT: TTELWADD
; WH-LABEL: address_only:
; WH: .word 0xa0020000
; WH-NOT: .word
; WH-NOT: call
; WH: ret
; BH-LABEL: address_only:
; BH: .word 0xa0010000
; BH-NOT: .word
; BH-NOT: call
; BH: ret
; OBJ-LABEL: <address_only>:
; OBJ: ttelwadd 0, 1, 0, 0, 0
; OBJ-NOT: ttelwadd
; OBJ: ret
  call void @llvm.tt.elwadd(i32 0, i32 1, i32 0, i32 0, i32 0)
  ret void
}

define void @all_selectors() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: all_selectors
; ISEL: TTELWADD 1023, 3, 3, 1, 3
; ISEL-NOT: TTELWADD
; WH-LABEL: all_selectors:
; WH: .word 0xa3e60ffc
; WH-NOT: .word
; WH-NOT: call
; WH: ret
; BH-LABEL: all_selectors:
; BH: .word 0xa3e30ffc
; BH-NOT: .word
; BH-NOT: call
; BH: ret
; OBJ-LABEL: <all_selectors>:
; OBJ: ttelwadd 1023, 3, 3, 1, 3
; OBJ-NOT: ttelwadd
; OBJ: ret
  call void @llvm.tt.elwadd(i32 1023, i32 3, i32 3, i32 1, i32 3)
  ret void
}

define void @sub_mul() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: sub_mul
; ISEL: TTELWSUB 24, 3, 2, 1, 3
; ISEL: TTELWMUL 24, 3, 2, 0, 3
; WH-LABEL: sub_mul:
; WH: .word 0xc3c60060
; WH: .word 0x9f460060
; WH-NEXT: ret
; BH-LABEL: sub_mul:
; BH: .word 0xc3c30060
; BH: .word 0x9f430060
; BH-NEXT: ret
; OBJ-LABEL: <sub_mul>:
; OBJ: ttelwsub 24, 3, 2, 1, 3
; OBJ-NEXT: {{.*}}ttelwmul 24, 3, 2, 0, 3
; OBJ-NEXT: {{.*}}ret
  call void @llvm.tt.elwsub(i32 24, i32 3, i32 2, i32 1, i32 3)
  call void @llvm.tt.elwmul(i32 24, i32 3, i32 2, i32 0, i32 3)
  ret void
}

declare void @llvm.tt.elwmul.port(i32 immarg, i32, i32, i32, i32, i32)

define void @mul_dynamic_mode(i32 noundef %raw) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: mul_dynamic_mode
; ISEL: PseudoTTELWMULPort
; WH-LABEL: mul_dynamic_mode:
; WH: slli {{[ast][0-9]+}}, {{[ast][0-9]+}}, 15
; WH: sw
; WH: ret
; BH-LABEL: mul_dynamic_mode:
; BH: slli {{[ast][0-9]+}}, {{[ast][0-9]+}}, 14
; BH: sw
; BH: ret
  %bounded = and i32 %raw, 3
  call void @llvm.tt.elwmul.port(i32 0, i32 0, i32 %bounded, i32 0, i32 0, i32 0)
  ret void
}

declare void @llvm.tt.mvmul.port(i32 immarg, i32, i32, i32, i32)

define void @matrix_dynamic_mode(i32 noundef %raw) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: matrix_dynamic_mode
; ISEL: PseudoTTMVMULPort
; WH-LABEL: matrix_dynamic_mode:
; WH: slli {{[ast][0-9]+}}, {{[ast][0-9]+}}, 15
; WH: sw
; WH: ret
; BH-LABEL: matrix_dynamic_mode:
; BH: slli {{[ast][0-9]+}}, {{[ast][0-9]+}}, 14
; BH: sw
; BH: ret
  %bounded = and i32 %raw, 3
  call void @llvm.tt.mvmul.port(i32 0, i32 0, i32 %bounded, i32 0, i32 0)
  ret void
}
