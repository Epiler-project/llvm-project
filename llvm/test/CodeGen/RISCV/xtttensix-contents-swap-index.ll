; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o - | FileCheck %s --check-prefix=POST

; D,C,D-index,C-index are complete old/result contents; allocation chooses
; distinct low primaries and their hardware +4 companion registers.
; Modes 0..9 share official BH/WH encoding. Index-off preserves both index
; values; enabled or masked lanes can only affect this complete four-value set.
; The native view has four def/use ties and no staging or helper register.

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load.immediate(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)

; CHECK-LABEL: <swap_index_mode0>:
; CHECK: ttsfpswap l[[#D0:]], l[[#C0:]], 0
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D0]], 0, 0, 4
; CHECK: ttsfpstore l[[#C0]], 2, 0, 4
; CHECK: ttsfpstore l[[#D0+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C0+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode0
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode0
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD0:]], {{(renamable )?}}$tt_l[[#PC0:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD0+4]], {{(renamable )?}}$tt_l[[#PC0+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD0]], {{(killed )?(renamable )?}}$tt_l[[#PC0]], {{(killed )?(renamable )?}}$tt_l[[#PD0+4]], {{(killed )?(renamable )?}}$tt_l[[#PC0+4]], 0
define void @swap_index_mode0() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 0)
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

; CHECK-LABEL: <swap_index_mode1>:
; CHECK: ttsfpswap l[[#D1:]], l[[#C1:]], 1
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D1]], 0, 0, 4
; CHECK: ttsfpstore l[[#C1]], 2, 0, 4
; CHECK: ttsfpstore l[[#D1+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C1+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode1
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode1
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD1:]], {{(renamable )?}}$tt_l[[#PC1:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD1+4]], {{(renamable )?}}$tt_l[[#PC1+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD1]], {{(killed )?(renamable )?}}$tt_l[[#PC1]], {{(killed )?(renamable )?}}$tt_l[[#PD1+4]], {{(killed )?(renamable )?}}$tt_l[[#PC1+4]], 1
define void @swap_index_mode1() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 1)
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

; CHECK-LABEL: <swap_index_mode2>:
; CHECK: ttsfpswap l[[#D2:]], l[[#C2:]], 2
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D2]], 0, 0, 4
; CHECK: ttsfpstore l[[#C2]], 2, 0, 4
; CHECK: ttsfpstore l[[#D2+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C2+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode2
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode2
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD2:]], {{(renamable )?}}$tt_l[[#PC2:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD2+4]], {{(renamable )?}}$tt_l[[#PC2+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD2]], {{(killed )?(renamable )?}}$tt_l[[#PC2]], {{(killed )?(renamable )?}}$tt_l[[#PD2+4]], {{(killed )?(renamable )?}}$tt_l[[#PC2+4]], 2
define void @swap_index_mode2() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 2)
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

; CHECK-LABEL: <swap_index_mode3>:
; CHECK: ttsfpswap l[[#D3:]], l[[#C3:]], 3
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D3]], 0, 0, 4
; CHECK: ttsfpstore l[[#C3]], 2, 0, 4
; CHECK: ttsfpstore l[[#D3+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C3+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode3
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode3
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD3:]], {{(renamable )?}}$tt_l[[#PC3:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD3+4]], {{(renamable )?}}$tt_l[[#PC3+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD3]], {{(killed )?(renamable )?}}$tt_l[[#PC3]], {{(killed )?(renamable )?}}$tt_l[[#PD3+4]], {{(killed )?(renamable )?}}$tt_l[[#PC3+4]], 3
define void @swap_index_mode3() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 3)
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

; CHECK-LABEL: <swap_index_mode4>:
; CHECK: ttsfpswap l[[#D4:]], l[[#C4:]], 4
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D4]], 0, 0, 4
; CHECK: ttsfpstore l[[#C4]], 2, 0, 4
; CHECK: ttsfpstore l[[#D4+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C4+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode4
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode4
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD4:]], {{(renamable )?}}$tt_l[[#PC4:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD4+4]], {{(renamable )?}}$tt_l[[#PC4+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD4]], {{(killed )?(renamable )?}}$tt_l[[#PC4]], {{(killed )?(renamable )?}}$tt_l[[#PD4+4]], {{(killed )?(renamable )?}}$tt_l[[#PC4+4]], 4
define void @swap_index_mode4() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 4)
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

; CHECK-LABEL: <swap_index_mode5>:
; CHECK: ttsfpswap l[[#D5:]], l[[#C5:]], 5
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D5]], 0, 0, 4
; CHECK: ttsfpstore l[[#C5]], 2, 0, 4
; CHECK: ttsfpstore l[[#D5+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C5+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode5
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode5
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD5:]], {{(renamable )?}}$tt_l[[#PC5:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD5+4]], {{(renamable )?}}$tt_l[[#PC5+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD5]], {{(killed )?(renamable )?}}$tt_l[[#PC5]], {{(killed )?(renamable )?}}$tt_l[[#PD5+4]], {{(killed )?(renamable )?}}$tt_l[[#PC5+4]], 5
define void @swap_index_mode5() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 5)
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

; CHECK-LABEL: <swap_index_mode6>:
; CHECK: ttsfpswap l[[#D6:]], l[[#C6:]], 6
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D6]], 0, 0, 4
; CHECK: ttsfpstore l[[#C6]], 2, 0, 4
; CHECK: ttsfpstore l[[#D6+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C6+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode6
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode6
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD6:]], {{(renamable )?}}$tt_l[[#PC6:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD6+4]], {{(renamable )?}}$tt_l[[#PC6+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD6]], {{(killed )?(renamable )?}}$tt_l[[#PC6]], {{(killed )?(renamable )?}}$tt_l[[#PD6+4]], {{(killed )?(renamable )?}}$tt_l[[#PC6+4]], 6
define void @swap_index_mode6() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 6)
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

; CHECK-LABEL: <swap_index_mode7>:
; CHECK: ttsfpswap l[[#D7:]], l[[#C7:]], 7
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D7]], 0, 0, 4
; CHECK: ttsfpstore l[[#C7]], 2, 0, 4
; CHECK: ttsfpstore l[[#D7+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C7+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode7
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode7
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD7:]], {{(renamable )?}}$tt_l[[#PC7:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD7+4]], {{(renamable )?}}$tt_l[[#PC7+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD7]], {{(killed )?(renamable )?}}$tt_l[[#PC7]], {{(killed )?(renamable )?}}$tt_l[[#PD7+4]], {{(killed )?(renamable )?}}$tt_l[[#PC7+4]], 7
define void @swap_index_mode7() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 7)
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

; CHECK-LABEL: <swap_index_mode8>:
; CHECK: ttsfpswap l[[#D8:]], l[[#C8:]], 8
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D8]], 0, 0, 4
; CHECK: ttsfpstore l[[#C8]], 2, 0, 4
; CHECK: ttsfpstore l[[#D8+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C8+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode8
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode8
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD8:]], {{(renamable )?}}$tt_l[[#PC8:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD8+4]], {{(renamable )?}}$tt_l[[#PC8+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD8]], {{(killed )?(renamable )?}}$tt_l[[#PC8]], {{(killed )?(renamable )?}}$tt_l[[#PD8+4]], {{(killed )?(renamable )?}}$tt_l[[#PC8+4]], 8
define void @swap_index_mode8() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 8)
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

; CHECK-LABEL: <swap_index_mode9>:
; CHECK: ttsfpswap l[[#D9:]], l[[#C9:]], 9
; CHECK-NOT: ttsfpmov
; CHECK: ttsfpstore l[[#D9]], 0, 0, 4
; CHECK: ttsfpstore l[[#C9]], 2, 0, 4
; CHECK: ttsfpstore l[[#D9+4]], 4, 0, 4
; CHECK: ttsfpstore l[[#C9+4]], 6, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_mode9
; ISEL: class: sfprindexpair
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_mode9
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD9:]], {{(renamable )?}}$tt_l[[#PC9:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD9+4]], {{(renamable )?}}$tt_l[[#PC9+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD9]], {{(killed )?(renamable )?}}$tt_l[[#PC9]], {{(killed )?(renamable )?}}$tt_l[[#PD9+4]], {{(killed )?(renamable )?}}$tt_l[[#PC9+4]], 9
define void @swap_index_mode9() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %a0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %a1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %a2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %a3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, i32 9)
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

; CHECK-LABEL: <swap_index_loop>:
; CHECK: beqz
; CHECK: ttsfpswap l[[#D10:]], l[[#C10:]], 1
; CHECK-NOT: ttsfpswap
; CHECK: ttsfpstore l[[#D10]], 32, 0, 4
; CHECK: ttsfpstore l[[#C10]], 34, 0, 4
; CHECK: ttsfpstore l[[#D10+4]], 36, 0, 4
; CHECK: ttsfpstore l[[#C10+4]], 38, 0, 4
; CHECK: ret
; ISEL-LABEL: name: swap_index_loop
; ISEL: PHI
; ISEL: PseudoTTSFPContentsSwapIndex
; POST-LABEL: name: swap_index_loop
; POST: BEQ
; POST: {{^ *}}{{(renamable )?}}$tt_l[[#PD10:]], {{(renamable )?}}$tt_l[[#PC10:]]
; POST-SAME: , {{(renamable )?}}$tt_l[[#PD10+4]], {{(renamable )?}}$tt_l[[#PC10+4]] = TTSFPSWAPIndex {{(killed )?(renamable )?}}$tt_l[[#PD10]], {{(killed )?(renamable )?}}$tt_l[[#PC10]], {{(killed )?(renamable )?}}$tt_l[[#PD10+4]], {{(killed )?(renamable )?}}$tt_l[[#PC10+4]], 1
define void @swap_index_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %seed0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %seed1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %seed2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %seed3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %has = icmp ugt i32 %limit, 0
  br i1 %has, label %loop, label %exit
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %latch ]
  %active0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %active1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %active2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %active3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %active0, <32 x i32> %active1, <32 x i32> %active2, <32 x i32> %active3, i32 1)
  %updated0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  %updated1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 1
  %updated2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 2
  %updated3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 3
  br label %latch
latch:
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final0 = phi <32 x i32> [ %seed0, %entry ], [ %updated0, %latch ]
  %final1 = phi <32 x i32> [ %seed1, %entry ], [ %updated1, %latch ]
  %final2 = phi <32 x i32> [ %seed2, %entry ], [ %updated2, %latch ]
  %final3 = phi <32 x i32> [ %seed3, %entry ], [ %updated3, %latch ]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final1, i32 34, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final2, i32 36, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final3, i32 38, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
