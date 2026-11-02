; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -riscv-rvv-regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -enable-new-pm -O0 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -enable-new-pm -O2 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NEWPM

; The contents receiver currently supports SelectionDAG with the legacy codegen
; pipeline only. NewPM must diagnose before selecting any numerical carrier.
; NEWPM: Tensix SFPU requires the SelectionDAG legacy codegen pipeline

; Contents are numerical SSA, not physical selectors. Each explicit constant
; read is one all-lane initialization instruction, never an allocator COPY.
; The old-value input keeps inactive lanes and partial Dst loads observable.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @dst_add_mul() "tensix-executor"="trisc1" {
; CHECK-LABEL: <dst_add_mul>:
; CHECK: ttsfpmov.all [[A:l[0-7]]], c9
; CHECK: ttsfpload [[A]], 0, 0, 3
; CHECK: ttsfpmov.all [[B:l[0-7]]], c9
; CHECK: ttsfpload [[B]], 2, 0, 3
; CHECK: ttsfpadd [[A]], [[A]], [[B]], 0
; CHECK: ttsfpmul [[A]], [[A]], [[B]], 0
; CHECK: ttsfpstore [[A]], 4, 0, 3
; CHECK: ret
; ISEL-LABEL: name: dst_add_mul
; ISEL: %{{[0-9]+}}:sfpr = TTSFPMOVAll $tt_c9
; ISEL: %{{[0-9]+}}:sfpr = TTSFPLOAD
; ISEL: %{{[0-9]+}}:sfpr = TTSFPADD
; ISEL: %{{[0-9]+}}:sfpr = TTSFPMUL
; ISEL: TTSFPSTORE
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %zero.a = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %zero.a, i32 0, i32 0, i32 3)
  %zero.b = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %zero.b, i32 2, i32 0, i32 3)
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %a, <32 x i32> %a, <32 x i32> %b, i32 0)
  %product = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %sum, <32 x i32> %sum, <32 x i32> %b, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %product, i32 4, i32 0, i32 3)
  ret void
}

; Loop-carried contents must remain a real CFG loop through allocation.
define void @contents_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <contents_loop>:
; CHECK: ttsfpmov.all [[ACC:l[0-7]]], c9
; CHECK: ttsfpmov.all [[ONE:l[0-7]]], c10
; CHECK: ttsfpadd [[ACC]], [[ACC]], [[ONE]], 0
; CHECK: bne{{z?}} {{.*}},
; CHECK: ttsfpstore [[ACC]], 0, 0, 3
; ISEL-LABEL: name: contents_loop
; ISEL: :sfpr = PHI
; ISEL: :sfpr = TTSFPADD
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %acc = phi <32 x i32> [ %initial, %entry ], [ %sum, %loop ]
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %acc, <32 x i32> %acc, <32 x i32> %one, i32 0)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 3)
  ret void
}
