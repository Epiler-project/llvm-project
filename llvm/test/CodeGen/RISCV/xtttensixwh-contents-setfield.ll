; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --implicit-check-not='ttsfpmov{{[ ]}}' --implicit-check-not=call
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=' COPY '
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=' COPY '
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=' COPY '
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=MIR --implicit-check-not=' COPY '
;
; These three BH instruction pages explicitly adopt identical WH behavior.
; Vector field bits come from tied old VD, remaining bits from distinct VC.
; Exact admitted modifier masks and 8/12/1-bit immediate boundaries stay owned
; by the existing native descriptors and shared field verifier. Zero is the
; canonical unused immediate in every vector form. No new intrinsic ABI.

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mad(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32>, <32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32>, <32 x i32>, i32 immarg)

; CHECK-LABEL: <setexp_vector_m0_v0>:
; CHECK: ttsfpsetexp [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 0
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setexp_vector_m0_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETEXP {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setexp_vector_m0_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %source, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setexp_immediate_m1_v0>:
; CHECK: ttsfpsetexp [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 1
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setexp_immediate_m1_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETEXP {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setexp_immediate_m1_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setexp_immediate_m1_v255>:
; CHECK: ttsfpsetexp [[DST:l[0-7]]], [[SRC:l[0-7]]], 255, 1
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setexp_immediate_m1_v255
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETEXP {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 255, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setexp_immediate_m1_v255() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 255, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setexp_vector_m2_v0>:
; CHECK: ttsfpsetexp [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 2
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setexp_vector_m2_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETEXP {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 2, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setexp_vector_m2_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.vector(<32 x i32> %old, <32 x i32> %source, i32 2)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setexp_immediate_m3_v0>:
; CHECK: ttsfpsetexp [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 3
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setexp_immediate_m3_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETEXP {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 3, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setexp_immediate_m3_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setexp_immediate_m3_v255>:
; CHECK: ttsfpsetexp [[DST:l[0-7]]], [[SRC:l[0-7]]], 255, 3
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setexp_immediate_m3_v255
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETEXP {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 255, 3, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setexp_immediate_m3_v255() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setexp.immediate(<32 x i32> %old, <32 x i32> %source, i32 255, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setman_vector_m0_v0>:
; CHECK: ttsfpsetman [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 0
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setman_vector_m0_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETMAN {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setman_vector_m0_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.vector(<32 x i32> %old, <32 x i32> %source, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setman_immediate_m1_v0>:
; CHECK: ttsfpsetman [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 1
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setman_immediate_m1_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETMAN {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setman_immediate_m1_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setman_immediate_m1_v4095>:
; CHECK: ttsfpsetman [[DST:l[0-7]]], [[SRC:l[0-7]]], 4095, 1
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setman_immediate_m1_v4095
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETMAN {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 4095, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setman_immediate_m1_v4095() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setman.immediate(<32 x i32> %old, <32 x i32> %source, i32 4095, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setsgn_vector_m0_v0>:
; CHECK: ttsfpsetsgn [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 0
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setsgn_vector_m0_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETSGN {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 0, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setsgn_vector_m0_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.vector(<32 x i32> %old, <32 x i32> %source, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setsgn_immediate_m1_v0>:
; CHECK: ttsfpsetsgn [[DST:l[0-7]]], [[SRC:l[0-7]]], 0, 1
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setsgn_immediate_m1_v0
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETSGN {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 0, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setsgn_immediate_m1_v0() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %source, i32 0, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}

; CHECK-LABEL: <setsgn_immediate_m1_v1>:
; CHECK: ttsfpsetsgn [[DST:l[0-7]]], [[SRC:l[0-7]]], 1, 1
; CHECK: ttsfpstore [[DST]], 2, 0, 3
; CHECK: ttsfpstore [[SRC]], 3, 0, 3
; MIR-LABEL: name: setsgn_immediate_m1_v1
; MIR: [[DST:\$tt_l[0-7]]] = TTSFPSETSGN {{(killed )?(renamable )?}}[[DST]], {{(killed )?(renamable )?}}[[SRC:\$tt_l[0-7]]], 1, 1, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
define void @setsgn_immediate_m1_v1() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %source = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %result = call <32 x i32> @llvm.riscv.tt.contents.setsgn.immediate(<32 x i32> %old, <32 x i32> %source, i32 1, i32 1)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 2, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %source, i32 3, i32 0, i32 3)
  ret void
}
