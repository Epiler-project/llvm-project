; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/valid.ll -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -stop-after=finalize-isel -verify-machineinstrs %t/valid.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/add.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/add.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/mov.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/mov.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/swap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/swap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONFIG

; SFPTRANSP is exempt from TEN-2932 even with ENABLE_DEST_INDEX set. The
; complete eight-value tuple already names every old and new LReg contents.
; Requiring a reset here would reject a legal operation after LaneConfig.
; The enabled case writes raw 0x0004 (all rows enabled, indexing enabled); the unknown
; case uses actual Dst data. Both keep L0's configuration source as group input
; and require all eight outputs without an extra live numerical register.
; Ordinary writes and the single-result LOAD/two-result SWAP ABI must retain
; their disabled-index requirement: they do not receive companion updates.
; CONFIG: contents SFPU requires ENABLE_DEST_INDEX known disabled

; CHECK-LABEL: <transpose_index_enabled>:
; CHECK: ttsfpconfig.lane 21845, 8
; CHECK: ttsfptransp
; CHECK: ttsfpstore l0,
; CHECK: ttsfpstore l1,
; CHECK: ttsfpstore l2,
; CHECK: ttsfpstore l3,
; CHECK: ttsfpstore l4,
; CHECK: ttsfpstore l5,
; CHECK: ttsfpstore l6,
; CHECK: ttsfpstore l7,
; ISEL-LABEL: name: transpose_index_enabled
; ISEL: class: sfpr0
; ISEL: class: sfpr7
; ISEL: PseudoTTSFPContentsTranspose8

; CHECK-LABEL: <transpose_index_unknown>:
; CHECK: ttsfpconfig.lane 21845, 8
; CHECK: ttsfptransp
; CHECK: ttsfpstore l0,
; CHECK: ttsfpstore l1,
; CHECK: ttsfpstore l2,
; CHECK: ttsfpstore l3,
; CHECK: ttsfpstore l4,
; CHECK: ttsfpstore l5,
; CHECK: ttsfpstore l6,
; CHECK: ttsfpstore l7,
; ISEL-LABEL: name: transpose_index_unknown
; ISEL: class: sfpr0
; ISEL: class: sfpr7
; ISEL: PseudoTTSFPContentsTranspose8

;--- valid.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

define void @transpose_index_enabled() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 4, i32 2)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %a4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %a0, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, <32 x i32> %a4, <32 x i32> %a5, <32 x i32> %a6, <32 x i32> %a7)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  %r4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r4, i32 8, i32 0, i32 4)
  %r5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 5
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r5, i32 10, i32 0, i32 4)
  %r6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 6
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r6, i32 12, i32 0, i32 4)
  %r7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 7
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r7, i32 14, i32 0, i32 4)
  ret void
}

define void @transpose_index_unknown() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %seed, i32 0, i32 0, i32 3)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %a4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %a0, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, <32 x i32> %a4, <32 x i32> %a5, <32 x i32> %a6, <32 x i32> %a7)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  %r4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 4
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r4, i32 8, i32 0, i32 4)
  %r5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 5
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r5, i32 10, i32 0, i32 4)
  %r6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 6
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r6, i32 12, i32 0, i32 4)
  %r7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 7
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r7, i32 14, i32 0, i32 4)
  ret void
}

;--- add.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

define void @index_still_rejects_add() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %other = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %r = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %other, <32 x i32> %other, <32 x i32> %other, i32 0)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 0, i32 0, i32 4)
  ret void
}

;--- mov.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

define void @index_still_rejects_mov() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %other = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %r = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 0, i32 0, i32 4)
  ret void
}

;--- load.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

define void @index_still_rejects_load() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %other = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %r = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %other, i32 0, i32 0, i32 3)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 0, i32 0, i32 4)
  ret void
}

;--- swap.ll
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32>, <32 x i32>, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.transpose8(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)

define void @index_still_rejects_swap() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %seed, i32 12, i32 2)
  %other = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %pair = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap(<32 x i32> %cfg, <32 x i32> %other, i32 1)
  %r = extractvalue { <32 x i32>, <32 x i32> } %pair, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r, i32 0, i32 0, i32 4)
  ret void
}
