; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs < %t/common.ll | FileCheck %s --check-prefix=COMMON
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs < %t/common.ll | FileCheck %s --check-prefix=COMMON
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -stop-after=finalize-isel < %t/common.ll | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs < %t/common.ll | FileCheck %s --check-prefix=COMMON
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs < %t/bh-upper.ll | FileCheck %s --check-prefix=BH
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -filetype=obj %t/port.ll -o - | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefix=PORT

; COMMON-LABEL: configure_thread:
; COMMON: .word 0xc8000002
; COMMON-NEXT: .word 0xc8e3fffe
; COMMON-NEXT: .word 0xc8100002
; COMMON-NEXT: .word 0xc89c0002
; COMMON-NOT: call
; COMMON: ret
; ISEL-LABEL: name: configure_thread
; ISEL: TTSETC16 0, 0
; ISEL-NEXT: {{.*}}TTSETC16 65535, 56
; ISEL-NEXT: {{.*}}TTSETC16 0, 4
; ISEL-NEXT: {{.*}}TTSETC16 0, 39
; BH-LABEL: configure_blackhole_upper:
; BH: .word 0xc90ffffe
; BH: ret
; PORT-LABEL: <configure_dest_offset>:
; PORT: sw
; PORT: ret

;--- port.ll
declare void @llvm.tt.setc16.port(i32 immarg, i32, i32)
define void @configure_dest_offset(i32 noundef %ordinal) "tensix-executor"="trisc1" {
  %offset = and i32 %ordinal, 512
  call void @llvm.tt.setc16.port(i32 0, i32 %offset, i32 4)
  ret void
}

;--- common.ll
declare void @llvm.tt.setc16(i32 immarg, i32 immarg)
define void @configure_thread() "tensix-executor"="trisc0" {
  call void @llvm.tt.setc16(i32 0, i32 0)
  call void @llvm.tt.setc16(i32 65535, i32 56)
  call void @llvm.tt.setc16(i32 0, i32 4)
  call void @llvm.tt.setc16(i32 0, i32 39)
  ret void
}

;--- bh-upper.ll
declare void @llvm.tt.setc16(i32 immarg, i32 immarg)
define void @configure_blackhole_upper() "tensix-executor"="trisc0" {
  call void @llvm.tt.setc16(i32 65535, i32 67)
  ret void
}
