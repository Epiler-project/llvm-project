; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -filetype=obj %t/unpack.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/wide.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=WIDE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ISSUE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/mop.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ISSUE

;--- unpack.ll
declare void @llvm.tt.dmanop()
declare void @llvm.tt.adddmareg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.muldmareg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)

; CHECK-LABEL: <unpack_offset>:
; CHECK: ttdmanop
; CHECK-NEXT: {{.*}}ttadddmareg 1, 2, 3, 0
; CHECK-NEXT: {{.*}}ttmuldmareg 3, 31, 63, 1
define void @unpack_offset() "tensix-executor"="trisc0" {
  call void @llvm.tt.dmanop()
  call void @llvm.tt.adddmareg(i32 1, i32 2, i32 3, i32 0)
  call void @llvm.tt.muldmareg(i32 3, i32 31, i32 63, i32 1)
  ret void
}

;--- wide.ll
declare void @llvm.tt.adddmareg(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
; WIDE: Wormhole ADDDMAREG field resultregindex must be in [0, 63]
define void @wide() "tensix-executor"="trisc0" {
  call void @llvm.tt.adddmareg(i32 1, i32 2, i32 64, i32 0)
  ret void
}

;--- port.ll
declare void @llvm.tt.adddmareg.port(i32 immarg, i32, i32, i32, i32)
; ISSUE: Tensix intrinsic is not supported on Wormhole
define void @port() "tensix-executor"="trisc0" {
  call void @llvm.tt.adddmareg.port(i32 0, i32 1, i32 2, i32 3, i32 0)
  ret void
}

;--- mop.ll
declare void @llvm.tt.dmanop.mop(i32 immarg)
define void @mop() "tensix-executor"="trisc0" {
  call void @llvm.tt.dmanop.mop(i32 2)
  ret void
}
