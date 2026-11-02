; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL

; WH AND/OR/NOT consume real LReg contents. The destination passthrough is
; retained for disabled lanes, then tied to the allocated result. No physical
; LReg selector or ordinary RISC-V vector operation enters the contents ABI.
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @wh_bitwise_contents() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_bitwise_contents>:
; CHECK: ttsfpload [[VALUE:l[0-7]]], 0, 0, 3
; CHECK: ttsfpmov.all [[MASK:l[0-7]]], c10
; CHECK: ttsfpand [[VALUE]], [[MASK]]
; CHECK: ttsfpor [[VALUE]], [[MASK]]
; CHECK: ttsfpnot [[VALUE]], [[VALUE]]
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_bitwise_contents
; ISEL: [[LOAD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[MASK:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c10
; ISEL: [[AND:%[0-9]+]]:sfpr = TTSFPAND [[LOAD]], [[MASK]]
; ISEL: [[OR:%[0-9]+]]:sfpr = TTSFPOR [[AND]], {{(killed )?}}[[MASK]]
; ISEL: [[NOT:%[0-9]+]]:sfpr = TTSFPNOT [[OR]], [[OR]]
; ISEL: TTSFPSTORE {{(killed )?}}[[NOT]], 0, 0, 3
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 3)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %and = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %value, <32 x i32> %mask)
  %or = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %and, <32 x i32> %mask)
  %not = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %or)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %not, i32 0, i32 0, i32 3)
  ret void
}

; Each bitwise operation reads the immediately preceding MAD's actual
; allocated destination. WH lacks the interlock and needs one SFPU NOP.
define void @wh_bitwise_mad_hazards() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_bitwise_mad_hazards>:
; CHECK: ttsfpmad [[VALUE:l[0-7]]],
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpand [[VALUE]],
; CHECK: ttsfpmad [[VALUE]],
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpor [[VALUE]],
; CHECK: ttsfpmad [[VALUE]],
; CHECK-NEXT: {{.*}}ttsfpnop
; CHECK-NEXT: {{.*}}ttsfpnot [[VALUE]], [[VALUE]]
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_bitwise_mad_hazards
; ISEL: [[MAD1:%[0-9]+]]:sfpr = TTSFPMAD
; ISEL: [[AND:%[0-9]+]]:sfpr = TTSFPAND [[MAD1]],
; ISEL: [[MAD2:%[0-9]+]]:sfpr = TTSFPMAD [[AND]],
; ISEL: [[OR:%[0-9]+]]:sfpr = TTSFPOR [[MAD2]],
; ISEL: [[MAD3:%[0-9]+]]:sfpr = TTSFPMAD [[OR]],
; ISEL: :sfpr = TTSFPNOT [[MAD3]], [[MAD3]]
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %mad1 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %old, <32 x i32> %old, <32 x i32> %one, <32 x i32> %one, i32 0)
  %and = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %mad1, <32 x i32> %one)
  %mad2 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %and, <32 x i32> %and, <32 x i32> %one, <32 x i32> %one, i32 0)
  %or = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %mad2, <32 x i32> %one)
  %mad3 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %or, <32 x i32> %or, <32 x i32> %one, <32 x i32> %one, i32 0)
  %not = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %mad3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %not, i32 0, i32 0, i32 3)
  ret void
}

; Keep one real loop body and carry numerical contents through the backedge.
; Independent scalar loop control may be scheduled between bitwise operations.
define void @wh_bitwise_contents_loop(i32 %limit) "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_bitwise_contents_loop>:
; CHECK: ttsfpmov.all [[VALUE:l[0-7]]], c9
; CHECK: ttsfpmov.all [[MASK:l[0-7]]], c10
; CHECK: ttsfpand [[VALUE]], [[MASK]]
; CHECK: ttsfpor [[VALUE]], [[MASK]]
; CHECK: ttsfpnot [[VALUE]], [[VALUE]]
; CHECK: bne{{z?}} {{.*}},
; CHECK: ttsfpstore [[VALUE]], 0, 0, 3
; ISEL-LABEL: name: wh_bitwise_contents_loop
; ISEL: [[VALUE:%[0-9]+]]:sfpr = PHI
; ISEL: [[AND:%[0-9]+]]:sfpr = TTSFPAND [[VALUE]],
; ISEL: [[OR:%[0-9]+]]:sfpr = TTSFPOR [[AND]],
; ISEL: :sfpr = TTSFPNOT [[OR]], [[OR]]
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%initial, %entry], [%not, %loop]
  %and = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %value, <32 x i32> %mask)
  %or = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %and, <32 x i32> %mask)
  %not = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %or)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %not, i32 0, i32 0, i32 3)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
