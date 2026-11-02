; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/constant.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONSTANT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/format.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FORMAT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OFFSET
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/unknown-offset.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/mixed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MIXED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/executor.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EXECUTOR
; RUN: not llc -mtriple=riscv32 %t/feature.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FEATURE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/abi.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ABI
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/staging.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONSTANT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/indirect-source.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDIRECT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/indirect-destination.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=INDIRECT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/addrmode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ADDRMODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/vector-arithmetic.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VECTOR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/return.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETURN

; CONSTANT: contents constant read requires immutable C8/C9/C10/C15
; POISON: SFPU carrier constants require explicit target initialization
; FORMAT: unsupported contents Dst format
; OFFSET: contents Dst offset must be proven in [0, 1023]
; UNKNOWN: contents Dst offset must be defined and non-poison
; MIXED: bound SFPU and contents SSA cannot mix in one function
; EXECUTOR: Tensix SFPU execution requires tensix-executor=trisc1
; FEATURE: Tensix intrinsic requires +xtttensixbh
; ABI: SFPU carrier cannot cross the argument ABI
; INDIRECT: immediate operand 3 must be in [0, 3]
; ADDRMODE: immediate operand 2 must be in [0, 7]
; VECTOR: SFPU carrier is only legal in target intrinsics and PHI
; RETURN: SFPU carrier cannot cross the return ABI

;--- constant.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @mutable_constant() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 11)
  ret void
}

;--- poison.ll
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @poison_seed() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> poison, i32 0, i32 0, i32 3)
  ret void
}

;--- format.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @reserved_format() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v, i32 0, i32 0, i32 7)
  ret void
}

;--- offset.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @overflow_offset() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v, i32 1024, i32 0, i32 3)
  ret void
}

;--- unknown-offset.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @unknown_offset(i32 %offset) "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %bounded = and i32 %offset, 1023
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v, i32 %bounded, i32 0, i32 3)
  ret void
}

;--- mixed.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)
define void @mixed_ingress() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 0)
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- executor.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @wrong_executor() "tensix-executor"="trisc0" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- feature.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @wrong_feature() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret void
}

;--- abi.ll
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @carrier_argument(<32 x i32> %v) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v, i32 0, i32 0, i32 3)
  ret void
}

; Macro staging is not an ordinary readable constant or a ninth SFPR slot.
;--- staging.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define void @macro_staging() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 16)
  ret void
}

; Physical indirect modes address LRegs through L7 lane bits. Numerical SSA
; operands cannot stand in for these independently observable locations.
;--- indirect-source.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @indirect_source() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.mul(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 4)
  ret void
}

;--- indirect-destination.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @indirect_destination() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = call <32 x i32> @llvm.riscv.tt.contents.add(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 8)
  ret void
}

;--- addrmode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @overflow_addrmode() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %v, i32 0, i32 8, i32 3)
  ret void
}

; A bit carrier does not authorize ordinary LLVM vector arithmetic, scalar
; decomposition or RVV allocation in place of the received SFPU instructions.
;--- vector-arithmetic.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @ordinary_vector_add() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %result = add <32 x i32> %v, %v
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 3)
  ret void
}

;--- return.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
define <32 x i32> @carrier_return() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  ret <32 x i32> %v
}
