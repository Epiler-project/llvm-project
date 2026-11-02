; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; WH has the same 32 x 32-bit LReg carrier, but only two address-modifier
; bits and no hardware interlock for next-cycle MAD consumers. These are
; real allocating contents values; no physical selector enters LLVM IR.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.abs(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32>, <32 x i32>)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

; FP32 2 -> 3 -> 3 -> 10 -> -10 -> 10, with explicit WH MAD gaps.
; The FP32 ABS mode is 1, whereas mode 0 is two's-complement integer ABS.
define void @wh_lane_contents() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_lane_contents>:
; CHECK: ttsfpencc 3, 10
; CHECK: ttsfpconfig.reset
; CHECK: ttsfploadi [[VALUE:l[0-7]]], 16384, 0
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; CHECK: ttsfpload [[VALUE]], 0, 0, 3
; CHECK: ttsfpmov.all [[ONE:l[0-7]]], c10
; CHECK: ttsfpadd [[VALUE]], [[VALUE]], [[ONE]], 0
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpmul [[VALUE]], [[VALUE]], [[ONE]], 0
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpmad [[VALUE]], [[VALUE]], [[VALUE]], [[ONE]], 0
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpmov.neg [[VALUE]], [[VALUE]]
; CHECK: ttsfpabs [[VALUE]], [[VALUE]], 1
; CHECK: ttsfpxor [[VALUE]], {{l[0-7]}}
; CHECK: ttsfpstore [[VALUE]], 2, 3, 3
; ISEL-LABEL: name: wh_lane_contents
; ISEL: :sfpr = TTSFPLOADI
; ISEL: :sfpr = TTSFPLOAD
; ISEL: :sfpr = TTSFPMAD
; ISEL: :sfpr = TTSFPABS
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %two = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %old, i32 16384, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %two, i32 0, i32 0, i32 3)
  %loaded = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %two, i32 0, i32 0, i32 3)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %loaded, <32 x i32> %loaded, <32 x i32> %one, i32 0)
  %product = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %sum, <32 x i32> %sum, <32 x i32> %one, i32 0)
  %madd = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %product, <32 x i32> %product, <32 x i32> %product, <32 x i32> %one, i32 0)
  %negative = call <32 x i32> @llvm.riscv.tt.contents.mov(<32 x i32> %madd, <32 x i32> %madd, i32 1)
  %positive = call <32 x i32> @llvm.riscv.tt.contents.abs(<32 x i32> %negative, <32 x i32> %negative, i32 1)
  %zero = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %positive, <32 x i32> %zero)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 3, i32 3)
  ret void
}

define void @wh_contents_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_contents_loop>:
; CHECK: ttsfpmov.all [[ACC:l[0-7]]], c9
; CHECK: ttsfpmov.all [[ONE:l[0-7]]], c10
; CHECK: ttsfpadd [[ACC]], [[ACC]], [[ONE]], 0
; CHECK: bne{{z?}} {{.*}},
; CHECK: ttsfpstore [[ACC]], 0, 0, 3
; ISEL-LABEL: name: wh_contents_loop
; ISEL: :sfpr = PHI
; ISEL: :sfpr = TTSFPADD
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%initial, %entry], [%sum, %loop]
  %sum = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %value, <32 x i32> %value, <32 x i32> %one, i32 0)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %sum, i32 0, i32 0, i32 3)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
