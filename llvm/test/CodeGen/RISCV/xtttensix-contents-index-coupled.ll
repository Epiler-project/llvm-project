; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-hazards,1 %s -o - | FileCheck %s --check-prefix=POST --implicit-check-not=" COPY "

; Two simultaneously live complete groups must fit the actual eight leaves.
; Each old has an independent explicit initializer; no source numbering,
; shared-old repair, extra capacity or dead companion models this case.

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare void @llvm.riscv.tt.contents.nop()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32>, <32 x i32>, i32, i32 immarg, i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)


; OBJ-LABEL: <load_two_live_static>:
; OBJ: ttsfpload l[[#S0:]], 4, 0, 3
; OBJ: ttsfpload l[[#S1:]], 8, 0, 3
; OBJ: ttsfpstore l[[#S0]], 16, 0, 4
; OBJ: ttsfpstore l[[#S0+4]], 18, 0, 4
; OBJ: ttsfpstore l[[#S1]], 20, 0, 4
; OBJ: ttsfpstore l[[#S1+4]], 22, 0, 4
; POST-LABEL: name: load_two_live_static
; POST: TTSFPLOADIndex
; POST: TTSFPLOADIndex
; POST: TTSFPSTORE
; POST: TTSFPSTORE
; POST: TTSFPSTORE
; POST: TTSFPSTORE
define void @load_two_live_static(i32 noundef %coord) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %bounded = and i32 %coord, 1023
  %s0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p0 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s0, <32 x i32> %s1, i32 4, i32 0, i32 3)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %p0, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %p0, 1
  %p1 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s2, <32 x i32> %s3, i32 8, i32 0, i32 3)
  %r2 = extractvalue { <32 x i32>, <32 x i32> } %p1, 0
  %r3 = extractvalue { <32 x i32>, <32 x i32> } %p1, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 18, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 20, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 22, i32 0, i32 4)
  ret void
}

; OBJ-LABEL: <load_two_live_dynamic>:
; OBJ: sw
; OBJ: sw
; OBJ: ttsfpstore l[[#D0:]], 16, 0, 4
; OBJ: ttsfpstore l[[#D0+4]], 18, 0, 4
; OBJ: ttsfpstore l[[#D1:]], 20, 0, 4
; OBJ: ttsfpstore l[[#D1+4]], 22, 0, 4
; POST-LABEL: name: load_two_live_dynamic
; POST: PseudoTTSFPLOADIndex
; POST-SAME: :: (volatile load store (s32))
; POST: PseudoTTSFPLOADIndex
; POST-SAME: :: (volatile load store (s32))
; POST: TTSFPSTORE
; POST: TTSFPSTORE
; POST: TTSFPSTORE
; POST: TTSFPSTORE
define void @load_two_live_dynamic(i32 noundef %coord) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %bounded = and i32 %coord, 1023
  %s0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %p0 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s0, <32 x i32> %s1, i32 %bounded, i32 0, i32 3)
  %r0 = extractvalue { <32 x i32>, <32 x i32> } %p0, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32> } %p0, 1
  %p1 = call { <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.load.index(<32 x i32> %s2, <32 x i32> %s3, i32 %bounded, i32 0, i32 3)
  %r2 = extractvalue { <32 x i32>, <32 x i32> } %p1, 0
  %r3 = extractvalue { <32 x i32>, <32 x i32> } %p1, 1
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 18, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 20, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 22, i32 0, i32 4)
  ret void
}

; OBJ-LABEL: <swap_two_live_groups>:
; OBJ: ttsfpswap l[[#P0:]], l[[#P1:]], 1
; OBJ: ttsfpswap l[[#Q0:]], l[[#Q1:]], 1
; OBJ: ttsfpstore l[[#P0]], 16, 0, 4
; OBJ: ttsfpstore l[[#P1]], 18, 0, 4
; OBJ: ttsfpstore l[[#P0+4]], 20, 0, 4
; OBJ: ttsfpstore l[[#P1+4]], 22, 0, 4
; OBJ: ttsfpstore l[[#Q0]], 24, 0, 4
; OBJ: ttsfpstore l[[#Q1]], 26, 0, 4
; OBJ: ttsfpstore l[[#Q0+4]], 28, 0, 4
; OBJ: ttsfpstore l[[#Q1+4]], 30, 0, 4
; POST-LABEL: name: swap_two_live_groups
; POST: TTSFPSWAPIndex
; POST: TTSFPNOP
; POST: TTSFPSWAPIndex
; POST: TTSFPNOP
; POST: TTSFPSTORE
define void @swap_two_live_groups() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  call void @llvm.riscv.tt.contents.nop()
  %s0 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s1 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %s2 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %s3 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %s4 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %s5 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %s6 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %s7 = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 8)
  %g0 = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %s0, <32 x i32> %s1, <32 x i32> %s2, <32 x i32> %s3, i32 1)
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g0, 0
  %r1 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g0, 1
  %r2 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g0, 2
  %r3 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g0, 3
  %g1 = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.swap.index(<32 x i32> %s4, <32 x i32> %s5, <32 x i32> %s6, <32 x i32> %s7, i32 1)
  %r4 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g1, 0
  %r5 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g1, 1
  %r6 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g1, 2
  %r7 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %g1, 3
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r1, i32 18, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r2, i32 20, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r3, i32 22, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r4, i32 24, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r5, i32 26, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r6, i32 28, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r7, i32 30, i32 0, i32 4)
  ret void
}
