; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -filetype=obj %t/valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/thread-override.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVRD
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/wide-cr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/port.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PORT

;--- valid.ll
declare void @llvm.riscv.tt.addrcrxy(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.addrcrzw(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.incadcxy(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.incadczw(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.incrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setadcxx(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setadcxy(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.setadczw(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.trnspsrcb()

; CHECK-LABEL: <llk_ordinary>:
; CHECK: ttaddrcrxy 1, 0, 0, 0, 0, 1
; CHECK-NEXT: {{.*}}ttaddrcrzw 1, 0, 0, 0, 0, 1
; CHECK-NEXT: {{.*}}ttincadcxy 0, 1, 0, 0, 1
; CHECK-NEXT: {{.*}}ttincadczw 1, 0, 0, 0, 2
; CHECK-NEXT: {{.*}}ttincrwc 0, 0, 8, 4
; CHECK: ttsetadcxx 0, 255, 1
; CHECK-NEXT: {{.*}}ttsetadcxy 11, 0, 0, 0, 0, 3
; CHECK-NEXT: {{.*}}ttsetadczw 15, 0, 0, 0, 0, 3
; CHECK-NEXT: {{.*}}tttrnspsrcb
define void @llk_ordinary() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.addrcrxy(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.addrcrzw(i32 1, i32 0, i32 0, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.incadcxy(i32 0, i32 1, i32 0, i32 0, i32 1)
  call void @llvm.riscv.tt.incadczw(i32 1, i32 0, i32 0, i32 0, i32 2)
  call void @llvm.riscv.tt.incrwc(i32 0, i32 0, i32 8, i32 4)
  call void @llvm.riscv.tt.setadcxx(i32 0, i32 255, i32 1)
  call void @llvm.riscv.tt.setadcxy(i32 11, i32 0, i32 0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.setadczw(i32 15, i32 0, i32 0, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.trnspsrcb()
  ret void
}

;--- thread-override.ll
declare void @llvm.riscv.tt.setadczw(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
; OVRD: Wormhole SETADCZW field ch1_y must be in [0, 7]
define void @override_rejected() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.setadczw(i32 1, i32 0, i32 0, i32 0, i32 8, i32 1)
  ret void
}

;--- wide-cr.ll
declare void @llvm.riscv.tt.incrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
; CR: Wormhole INCRWC field rwc_cr must be in [0, 7]
define void @wide_cr() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.incrwc(i32 0, i32 0, i32 2, i32 8)
  ret void
}

;--- port.ll
declare void @llvm.riscv.tt.trnspsrcb.port(i32 immarg)
; PORT: Tensix intrinsic is not supported on Wormhole
define void @port_rejected() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.trnspsrcb.port(i32 0)
  ret void
}
