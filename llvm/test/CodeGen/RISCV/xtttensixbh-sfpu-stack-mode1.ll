; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O3 -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ASM
; RUN: opt -S -mtriple=riscv32 -mattr=+xtttensixbh -passes='default<O3>' %s | FileCheck %s --check-prefix=LOOP-IR
; RUN: opt -S -mtriple=riscv32 -mattr=+xtttensixbh -passes=loop-unroll -unroll-count=4 -unroll-threshold=100000 %s | FileCheck %s --check-prefix=LOOP-IR

declare void @llvm.riscv.tt.bound.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfppopc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpencc(i32 immarg, i32 immarg)

; Physical ingress keeps an unknown incoming stack. It must neither invent a
; matching push nor let mode one reuse a depth-changing machine descriptor.
define void @incoming_replace_peek() "tensix-executor"="trisc1" {
; ISEL-LABEL: name: incoming_replace_peek
; ISEL: TTSFPPUSHCReplaceTop implicit-def $tt_ccstack, implicit-def $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue{{$}}
; ISEL-NEXT: TTSFPPOPCPeek implicit-def $tt_cc, implicit-def $tt_issue, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue{{$}}
; ASM-LABEL: incoming_replace_peek:
; ASM-NOT: .word
; ASM: .word 0x1c000006
; ASM-NEXT: .word 0x20000006
; ASM-NOT: .word
; ASM: ret
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 1)
  ret void
}

define void @replace_peek_loop(i32 %count) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: replace_peek_loop
; ISEL: TTSFPPUSHC implicit
; ISEL: PHI
; ISEL: TTSFPPUSHCReplaceTop
; ISEL-NEXT: TTSFPPOPCPeek
; ISEL: TTSFPPOPC implicit
; ASM-LABEL: replace_peek_loop:
; ASM: .word 0x1c000002
; ASM-NOT: .word
; ASM: [[LOOP:.LBB[0-9_]+]]:
; ASM: .word 0x1c000006
; ASM-NOT: .word
; ASM: .word 0x20000006
; ASM-NOT: .word
; ASM: {{bnez?}} {{.*}}, [[LOOP]]
; ASM-NOT: .word
; ASM: .word 0x20000002
; ASM-NOT: .word
; ASM: ret
; LOOP-IR-LABEL: define void @replace_peek_loop(
; LOOP-IR: call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
; LOOP-IR: phi i32
; LOOP-IR: call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 1)
; LOOP-IR-NEXT: call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 1)
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfppushc
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfppopc
; LOOP-IR: br i1
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfppushc
; LOOP-IR: call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfppushc
; LOOP-IR-NOT: call void @llvm.riscv.tt.bound.sfppopc
; LOOP-IR: ret void
entry:
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  br label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 1)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %count
  br i1 %done, label %exit, label %loop
exit:
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  ret void
}

; Both dynamic alternatives preserve depth. Peek after the join reads the
; actual top produced on that path rather than a lexical predecessor's CC.
define void @replace_peek_diamond(i1 %choose) "tensix-executor"="trisc1" {
; ISEL-LABEL: name: replace_peek_diamond
; ISEL: TTSFPPUSHC implicit
; ISEL: TTSFPENCC 1, 10
; ISEL-NEXT: TTSFPPUSHCReplaceTop
; ISEL: TTSFPENCC 3, 10
; ISEL-NEXT: TTSFPPUSHCReplaceTop
; ISEL: TTSFPPOPCPeek
; ISEL-NEXT: TTSFPPOPC implicit
; ASM-LABEL: replace_peek_diamond:
; ASM-NOT: .word
; ASM: .word 0x1c000002
; ASM-NOT: .word
; ASM: beqz {{.*}}, [[RIGHT:.LBB[0-9_]+]]
; ASM-NOT: .word
; ASM: .word 0x2800402a
; ASM-NOT: .word
; ASM: j [[JOIN:.LBB[0-9_]+]]
; ASM: [[RIGHT]]:
; ASM-NOT: .word
; ASM: .word 0x2800c02a
; ASM: [[JOIN]]:
; ASM-NOT: .word
; ASM: .word 0x1c000006
; ASM-NOT: .word
; ASM: .word 0x20000006
; ASM-NOT: .word
; ASM: .word 0x20000002
; ASM-NOT: .word
; ASM: ret
entry:
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 0)
  br i1 %choose, label %left, label %right
left:
  call void @llvm.riscv.tt.bound.sfpencc(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 1)
  br label %join
right:
  call void @llvm.riscv.tt.bound.sfpencc(i32 3, i32 10)
  call void @llvm.riscv.tt.bound.sfppushc(i32 0, i32 1)
  br label %join
join:
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 1)
  call void @llvm.riscv.tt.bound.sfppopc(i32 0, i32 0)
  ret void
}
