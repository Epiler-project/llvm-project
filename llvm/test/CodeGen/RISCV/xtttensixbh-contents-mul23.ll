; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t.bh.O0.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh %t.bh.O0.o | FileCheck %s --check-prefix=DIS
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t.bh.O2.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh %t.bh.O2.o | FileCheck %s --check-prefix=DIS

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.divp2(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @blackhole_direct() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: blackhole_direct
; ISEL: [[NEGADD:%[0-9]+]]:sfpr = TTSFPADDI {{.*}}, 16256, 2,
; ISEL: [[NEGMUL:%[0-9]+]]:sfpr = TTSFPMULI [[NEGADD]], 16384, 2,
; ISEL: [[LOW:%[0-9]+]]:sfpr = TTSFPMUL24 [[NEGMUL]], [[NEGMUL]], {{.*}}, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_c9, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[HIGH:%[0-9]+]]:sfpr = TTSFPMUL24 [[LOW]], [[LOW]], {{.*}}, 1,
; DIS-LABEL: <blackhole_direct>:
; DIS: ttsfpaddi [[DST:l[0-7]]], 16256, 2
; DIS: ttsfpmuli [[DST]], 16384, 2
; DIS: ttsfpmul24 [[DST]], [[DST]], [[SRC:l[0-7]]], 0
; DIS: ttsfpmul24 [[DST]], [[DST]], [[SRC]], 1
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %src = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %negadd = call <32 x i32> @llvm.riscv.tt.contents.add.immediate(<32 x i32> %old, i32 16256, i32 2)
  %negmul = call <32 x i32> @llvm.riscv.tt.contents.mul.immediate(<32 x i32> %negadd, i32 16384, i32 2)
  %low = call <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32> %negmul, <32 x i32> %negmul, <32 x i32> %src, i32 0)
  %high = call <32 x i32> @llvm.riscv.tt.contents.mul23(<32 x i32> %low, <32 x i32> %low, <32 x i32> %src, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %high, i32 0, i32 0, i32 4)
  ret void
}
