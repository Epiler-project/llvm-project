; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -filetype=obj %t/wh.ll -o - | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -verify-machineinstrs -filetype=obj %t/bh.ll -o - | llvm-objdump -d --mattr=+xtttensixbh --no-print-imm-hex - | FileCheck %s --check-prefix=OBJ
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/wh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCH
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/bh.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISMATCH

; OBJ-LABEL: <kernel_main>:
; OBJ:      ttreplay 1, 0, 1, 5
; OBJ-NEXT: {{.*}}ttnop
; OBJ:      ttreplay 0, 0, 1, 5
; MISMATCH: Replay intrinsic architecture does not match target feature

;--- wh.ll
declare void @llvm.tt.wh.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.wh.replay.template.end(i32 immarg)
declare void @llvm.tt.wh.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.nop()

define void @kernel_main() "tensix-executor"="trisc0" {
  call void @llvm.tt.wh.replay.template.begin(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.nop()
  call void @llvm.tt.wh.replay.template.end(i32 0)
  call void @llvm.tt.wh.replay.template.execute(i32 0)
  ret void
}

;--- bh.ll
declare void @llvm.tt.bh.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.tt.bh.replay.template.end(i32 immarg)
declare void @llvm.tt.bh.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.nop()

define void @kernel_main() "tensix-executor"="trisc0" {
  call void @llvm.tt.bh.replay.template.begin(i32 0, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.nop()
  call void @llvm.tt.bh.replay.template.end(i32 0)
  call void @llvm.tt.bh.replay.template.execute(i32 0)
  ret void
}
