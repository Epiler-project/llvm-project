; The copy/condition cleanup pass only sees bound physical SFPU operands.
; Virtual SSA values are rejected before it runs.
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=REJECT
; REJECT: Tensix SFPU requires bound physical registers

; These authored virtual-register cases are retained as a rejection fixture;
; cleanup of bound physical operands is covered by the MIR ingress tests.
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpnop()

define void @duplicate_encc() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpencc(i32 1, i32 2)
  call void @llvm.riscv.tt.sfpencc(i32 1, i32 2)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 8)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 8)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}

define void @self_copy_mode0() "tensix-executor"="trisc1" {
  %x = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %copy = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %x, <32 x i32> %x, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %copy, i32 0, i32 0, i32 4)
  ret void
}

define void @self_copy_all() "tensix-executor"="trisc1" {
  %x = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %copy = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %x, <32 x i32> %x, i32 2)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %copy, i32 0, i32 0, i32 4)
  ret void
}

define void @keep_negation() "tensix-executor"="trisc1" {
  %x = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %neg = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %x, <32 x i32> %x, i32 1)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %neg, i32 0, i32 0, i32 4)
  ret void
}

define void @keep_toggle_and_barrier() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.sfpencc(i32 0, i32 1)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 9)
  call void @llvm.riscv.tt.sfpencc(i32 2, i32 9)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}

; Fixed reads are stable snapshots. The write between reads and the old-value
; use after a masked assignment must remain distinct in the final stream.
define void @keep_snapshots_and_masked_later_use() "tensix-executor"="trisc1" {
  %before = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 3)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %zero)
  %after = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 3)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %after, i32 0, i32 2)
  %selected = call <32 x i32> @llvm.riscv.tt.sfpmov(<32 x i32> %before, <32 x i32> %after, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %selected, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %before, i32 1, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  ret void
}
