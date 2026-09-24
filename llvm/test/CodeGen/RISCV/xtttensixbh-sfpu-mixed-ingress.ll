; RUN: not llc -mtriple=riscv32 %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CALL
; RUN: opt -S -passes=always-inline %s -o %t.ll
; RUN: FileCheck %s --check-prefix=INLINE < %t.ll
; RUN: llc -mtriple=riscv32 -O0 -verify-machineinstrs -stop-after=postrapseudos %t.ll -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -stop-after=postrapseudos %t.ll -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -O0 -verify-machineinstrs -filetype=obj %t.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -filetype=obj %t.ll -o /dev/null
; RUN: opt -S -passes='always-inline,default<O2>' %s -o %t.opt.ll
; RUN: FileCheck %s --check-prefix=INLINE < %t.opt.ll
; RUN: llc -mtriple=riscv32 -O2 -verify-machineinstrs -filetype=obj %t.opt.ll -o /dev/null
;
; The production ingress has already inlined supported helpers while retaining
; their CFG. Model that boundary with LLVM's ordinary inliner, then verify the
; complete mixed body at O0 and O2. Unlowered calls remain an explicit error;
; an alwaysinline hint alone is not a register-preservation ABI.
;
; The import is the TRISC1 firmware data symbol used by native Math lowering.
; This test proves symbol/memory transport, not SDK linking or device behavior.
; CALL: call has no verified SFPU preservation ABI

target triple = "riscv32"
@math_sync_tile_dst_index = external global i32

declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32>, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.lreg.write(i32 immarg, <32 x i32>)
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)

define internal i32 @iteration_limit() alwaysinline {
  %index = load volatile i32, ptr @math_sync_tile_dst_index, align 4
  %bounded = and i32 %index, 15
  %limit = add i32 %bounded, 1
  ret i32 %limit
}

; INLINE-LABEL: define void @mixed(
; INLINE: load volatile i32, ptr @math_sync_tile_dst_index
; INLINE-NOT: call i32 @iteration_limit
; INLINE: phi <32 x i32>
; INLINE: call void @llvm.riscv.tt.sfppushc
; INLINE: call void @llvm.riscv.tt.sfpsetcc
; INLINE: call <32 x i32> @llvm.riscv.tt.sfpiadd
; INLINE: call void @llvm.riscv.tt.sfppopc
; INLINE: store volatile i32
; INLINE: br i1
; INLINE: ret void
; MIR-LABEL: name: mixed
; MIR-NOT: PseudoCALL
; MIR: LW {{.*}}@math_sync_tile_dst_index
; MIR: TTSFPPUSHC
; MIR: TTSFPSETCC
; MIR: TTSFPIADD
; MIR: TTSFPPOPC
; MIR: SW
; MIR: BNE
; MIR-NOT: PseudoCALL
; MIR: PseudoRET
define void @mixed(ptr %observed) "target-features"="+xtttensixbh" "tensix-executor"="trisc1" {
entry:
  %limit = call i32 @iteration_limit()
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %one = call <32 x i32> @llvm.riscv.tt.sfploadi(<32 x i32> %zero, i32 1, i32 2)
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  %value = phi <32 x i32> [%zero, %entry], [%sum, %loop]
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %value, i32 0, i32 2)
  %sum = call <32 x i32> @llvm.riscv.tt.sfpiadd(<32 x i32> %value, <32 x i32> %one, i32 4)
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  ; Keep the pre-update lanes live independently of the tied result.
  call void @llvm.riscv.tt.lreg.write(i32 3, <32 x i32> %value)
  store volatile i32 %i, ptr %observed, align 4
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.lreg.write(i32 4, <32 x i32> %sum)
  ret void
}
