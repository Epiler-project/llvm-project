; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST

; Explicit index-enabled config is supported on both targets. The four-result receiver
; remains valid for unknown per-lane configuration and alternating row masks.
; ROW_MASK bits disable rows: 0x0004 enables every row, 0x5004 masks rows
; 0/2 and 0xa004 masks rows1/3. Reset before STORE observes all four results.
; WH LaneConfig is received through the same complete native configuration.

; CHECK-LABEL: <swap_index_enabled>:
; CHECK: ttsfpconfig.lane 21845, 8
; CHECK: ttsfpswap l[[#D0:]], l[[#C0:]], 1
; CHECK: ttsfpstore l[[#D0]],
; CHECK: ttsfpstore l[[#C0]],
; CHECK: ttsfpstore l[[#D0+4]],
; CHECK: ttsfpstore l[[#C0+4]],
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD0:]], {{(renamable )?}}$tt_l[[#PC0:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD0+4]], {{(renamable )?}}$tt_l[[#PC0+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD0]], {{(killed )?(renamable )?}}$tt_l[[#PC0]], {{(killed )?(renamable )?}}$tt_l[[#PD0+4]], {{(killed )?(renamable )?}}$tt_l[[#PC0+4]], 1
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.config.lane(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)

define void @swap_index_enabled() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %a0, i32 4, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %cfg, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 1)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}


; CHECK-LABEL: <swap_index_unknown>:
; CHECK: ttsfpconfig.lane 21845, 8
; CHECK: ttsfpswap l[[#D1:]], l[[#C1:]], 1
; CHECK: ttsfpstore l[[#D1]],
; CHECK: ttsfpstore l[[#C1]],
; CHECK: ttsfpstore l[[#D1+4]],
; CHECK: ttsfpstore l[[#C1+4]],
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD1:]], {{(renamable )?}}$tt_l[[#PC1:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD1+4]], {{(renamable )?}}$tt_l[[#PC1+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD1]], {{(killed )?(renamable )?}}$tt_l[[#PC1]], {{(killed )?(renamable )?}}$tt_l[[#PD1+4]], {{(killed )?(renamable )?}}$tt_l[[#PC1+4]], 1
define void @swap_index_unknown() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %a0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %cfg, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 1)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}


; CHECK-LABEL: <swap_index_row_mask_5>:
; CHECK: ttsfpconfig.lane 21845, 8
; CHECK: ttsfpswap l[[#D2:]], l[[#C2:]], 1
; CHECK: ttsfpconfig.reset
; CHECK: ttsfpstore l[[#D2]],
; CHECK: ttsfpstore l[[#C2]],
; CHECK: ttsfpstore l[[#D2+4]],
; CHECK: ttsfpstore l[[#C2+4]],
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD2:]], {{(renamable )?}}$tt_l[[#PC2:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD2+4]], {{(renamable )?}}$tt_l[[#PC2+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD2]], {{(killed )?(renamable )?}}$tt_l[[#PC2]], {{(killed )?(renamable )?}}$tt_l[[#PD2+4]], {{(killed )?(renamable )?}}$tt_l[[#PC2+4]], 1
define void @swap_index_row_mask_5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %a0, i32 20484, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %cfg, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 1)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}

; CHECK-LABEL: <swap_index_row_mask_a>:
; CHECK: ttsfpconfig.lane 21845, 8
; CHECK: ttsfpswap l[[#D3:]], l[[#C3:]], 1
; CHECK: ttsfpconfig.reset
; CHECK: ttsfpstore l[[#D3]],
; CHECK: ttsfpstore l[[#C3]],
; CHECK: ttsfpstore l[[#D3+4]],
; CHECK: ttsfpstore l[[#C3+4]],
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD3:]], {{(renamable )?}}$tt_l[[#PC3:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD3+4]], {{(renamable )?}}$tt_l[[#PC3+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD3]], {{(killed )?(renamable )?}}$tt_l[[#PC3]], {{(killed )?(renamable )?}}$tt_l[[#PD3+4]], {{(killed )?(renamable )?}}$tt_l[[#PC3+4]], 1
define void @swap_index_row_mask_a() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %cfg = call <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32> %a0, i32 40964, i32 2)
  call void @llvm.riscv.tt.contents.config.lane(<32 x i32> %cfg, i32 21845, i32 8)
  call void @llvm.riscv.tt.contents.nop()
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %cfg, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 1)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 2, i32 0, i32 4)
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 4, i32 0, i32 4)
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 6, i32 0, i32 4)
  ret void
}
