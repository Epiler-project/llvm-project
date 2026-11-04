; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/good.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=MC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/good.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=MC
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/good.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/good.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -filetype=obj %t/bad-low.ll -o %t/bad-low.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/bad-low.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -filetype=obj %t/bad-high.ll -o %t/bad-high.o 2>&1 | FileCheck %s --check-prefix=RANGE
; RUN: test ! -e %t/bad-high.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -filetype=obj %t/bad-mode.ll -o %t/bad-mode.o 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: test ! -e %t/bad-mode.o

; WH/BH share SFPIADD's VC +/- old VD semantics, lane predication, CC modes,
; and signed immediate. Check allocating numerical contents through actual
; object emission, precise CC def/use in ISel, and next-cycle MAD hazards.
; No fixed numerical register selector or implicit staging copy is introduced.

;--- good.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @wh_iadd_modes() "tensix-executor"="trisc1" {
; MC-LABEL: <wh_iadd_modes>:
; MC: ttsfpload [[VALUE:l[0-7]]], 0, 0, 4
; MC: ttsfpmov.all [[SOURCE:l[0-7]]], c15
; MC: ttsfpiadd.lt0 [[VALUE]], [[SOURCE]]
; MC: ttsfpiadd.sub.lt0 [[VALUE]], [[SOURCE]]
; MC: ttsfpiadd [[VALUE]], [[SOURCE]]
; MC: ttsfpiadd.sub [[VALUE]], [[SOURCE]]
; MC: ttsfpiadd.ge0 [[VALUE]], [[SOURCE]]
; MC: ttsfpiadd.sub.ge0 [[VALUE]], [[SOURCE]]
; MC: ttsfpstore [[VALUE]], 0, 0, 4
; ISEL-LABEL: name: wh_iadd_modes
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[SOURCE:%[0-9]+]]:sfpr = TTSFPMOVAll $tt_c15
; ISEL: [[R0:%[0-9]+]]:sfpr = TTSFPIADDCCLT [[OLD]], [[SOURCE]], implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R2:%[0-9]+]]:sfpr = TTSFPISUBCCLT {{(killed )?}}[[R0]], [[SOURCE]], implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R4:%[0-9]+]]:sfpr = TTSFPIADD {{(killed )?}}[[R2]], [[SOURCE]], implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R6:%[0-9]+]]:sfpr = TTSFPISUB {{(killed )?}}[[R4]], [[SOURCE]], implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R8:%[0-9]+]]:sfpr = TTSFPIADDCCGE {{(killed )?}}[[R6]], [[SOURCE]], implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R10:%[0-9]+]]:sfpr = TTSFPISUBCCGE {{(killed )?}}[[R8]], {{(killed )?}}[[SOURCE]], implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 4)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %r0 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %value, <32 x i32> %source, i32 0)
  %r2 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %r0, <32 x i32> %source, i32 2)
  %r4 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %r2, <32 x i32> %source, i32 4)
  %r6 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %r4, <32 x i32> %source, i32 6)
  %r8 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %r6, <32 x i32> %source, i32 8)
  %r10 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %r8, <32 x i32> %source, i32 10)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r10, i32 0, i32 0, i32 4)
  ret void
}

define void @wh_iadd_immediate_modes() "tensix-executor"="trisc1" {
; MC-LABEL: <wh_iadd_immediate_modes>:
; MC: ttsfpload [[VALUE:l[0-7]]], 0, 0, 4
; MC: ttsfpiadd.i [[VALUE]], [[VALUE]], -2048, 1
; MC: ttsfpiadd.i [[VALUE]], [[VALUE]], 2047, 5
; MC: ttsfpiadd.i [[VALUE]], [[VALUE]], 0, 9
; MC: ttsfpstore [[VALUE]], 0, 0, 4
; ISEL-LABEL: name: wh_iadd_immediate_modes
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: [[R1:%[0-9]+]]:sfpr = TTSFPIADDI [[OLD]], [[OLD]], -2048, 1, implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R5:%[0-9]+]]:sfpr = TTSFPIADDI [[R1]], {{(killed )?}}[[R1]], 2047, 5, implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: [[R9:%[0-9]+]]:sfpr = TTSFPIADDI [[R5]], {{(killed )?}}[[R5]], 0, 9, implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 4)
  %r1 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %value, i32 -2048, i32 1)
  %r5 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %r1, i32 2047, i32 5)
  %r9 = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %r5, i32 0, i32 9)
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r9, i32 0, i32 0, i32 4)
  ret void
}

define void @wh_iadd_same_contents() "tensix-executor"="trisc1" {
; MC-LABEL: <wh_iadd_same_contents>:
; MC: ttsfpload [[VALUE:l[0-7]]], 0, 0, 4
; MC: ttsfpiadd [[VALUE]], [[VALUE]]
; MC: ttsfpstore [[VALUE]], 0, 0, 4
; ISEL-LABEL: name: wh_iadd_same_contents
; ISEL: [[VALUE:%[0-9]+]]:sfpr = TTSFPLOAD
; ISEL: :sfpr = TTSFPIADD [[VALUE]], {{(killed )?}}[[VALUE]], implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %value = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %old, i32 0, i32 0, i32 4)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %value, <32 x i32> %value, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

define void @wh_iadd_mad_hazards() "tensix-executor"="trisc1" {
; MC-LABEL: <wh_iadd_mad_hazards>:
; MC: ttsfpmad [[VALUE:l[0-7]]],
; MC-NEXT: {{.*}}ttsfpnop
; MC-NEXT: {{.*}}ttsfpiadd [[VALUE]],
; MC: ttsfpmad [[VALUE]],
; MC-NEXT: {{.*}}ttsfpnop
; MC-NEXT: {{.*}}ttsfpiadd.sub [[VALUE]],
; MC: ttsfpmad [[VALUE]],
; MC-NEXT: {{.*}}ttsfpnop
; MC-NEXT: {{.*}}ttsfpiadd.i [[VALUE]], [[VALUE]], -1, 5
; MC: ttsfpstore [[VALUE]], 0, 0, 4
; ISEL-LABEL: name: wh_iadd_mad_hazards
; ISEL: [[MAD:%[0-9]+]]:sfpr = TTSFPMAD
; ISEL-NEXT: {{.*}}:sfpr = TTSFPIADD {{(killed )?}}[[MAD]],
; ISEL: [[MAD2:%[0-9]+]]:sfpr = TTSFPMAD
; ISEL-NEXT: {{.*}}:sfpr = TTSFPISUB {{(killed )?}}[[MAD2]],
; ISEL: [[MAD3:%[0-9]+]]:sfpr = TTSFPMAD
; ISEL-NEXT: {{.*}}:sfpr = TTSFPIADDI [[MAD3]], {{(killed )?}}[[MAD3]], -1, 5,
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %mad = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %old, <32 x i32> %one, <32 x i32> %one, <32 x i32> %one, i32 0)
  %add = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %mad, <32 x i32> %one, i32 4)
  %mad2 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %add, <32 x i32> %one, <32 x i32> %one, <32 x i32> %one, i32 0)
  %sub = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %mad2, <32 x i32> %one, i32 6)
  %mad3 = call <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32> %sub, <32 x i32> %one, <32 x i32> %one, <32 x i32> %one, i32 0)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %mad3, i32 -1, i32 5)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

; RANGE: Tensix immediate operand must fit in 12 signed bits
;--- bad-low.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @bad_low() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %old, i32 -2049, i32 5)
  ret void
}

;--- bad-high.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @bad_high() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %old, i32 2048, i32 5)
  ret void
}

; MODE: SFPIADDI mode must be one of {1, 5, 9}
;--- bad-mode.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32>, i32 immarg, i32 immarg)
define void @bad_mode() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.iadd.immediate(<32 x i32> %old, i32 0, i32 13)
  ret void
}
