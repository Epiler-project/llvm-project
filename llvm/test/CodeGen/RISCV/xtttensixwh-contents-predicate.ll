; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o - | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -regalloc=fast -verify-machineinstrs -filetype=obj %s -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o /dev/null

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare void @llvm.riscv.tt.contents.setcc(<32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.pushc(i32 immarg)
declare void @llvm.riscv.tt.contents.popc(i32 immarg)
declare void @llvm.riscv.tt.contents.compc()

; These are stateful contents operations. The fixed CC and CCSTACK use/defs
; retain raw saved flags, enable state and true push/pop versus peeking effects.
; All stack depths are one or zero, avoiding WH's full-stack peek hardware bug.
define void @wh_predicate_modes() "tensix-executor"="trisc1" {
; CHECK-LABEL: <wh_predicate_modes>:
; CHECK: ttsfppushc
; CHECK: ttsfpsetcc.lt0
; CHECK: ttsfpcompc
; CHECK: ttsfppopc.peek
; CHECK: ttsfpsetcc.ne0
; CHECK: ttsfppopc.and_top
; CHECK: ttsfpsetcc.ge0
; CHECK: ttsfppopc.or_top
; CHECK: ttsfpsetcc.eq0
; CHECK: ttsfppopc{{$}}
; ISEL-LABEL: name: wh_predicate_modes
; ISEL: TTSFPPUSHC implicit-def{{( dead)?}} $tt_ccstack, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPSETCCLT {{.*}}implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPCOMPC implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPPOPCPeek implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPSETCCNE
; ISEL: TTSFPPOPCAndTop implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPSETCCGE
; ISEL: TTSFPPOPCOrTop implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_issue, implicit $tt_cc, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue
; ISEL: TTSFPSETCCEQ
; ISEL: TTSFPPOPC implicit-def{{( dead)?}} $tt_cc, implicit-def{{( dead)?}} $tt_ccstack, implicit-def{{( dead)?}} $tt_issue, implicit $tt_ccstack, implicit $tt_config, implicit $tt_issue
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %predicate = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void @llvm.riscv.tt.contents.pushc(i32 0)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %predicate, i32 0)
  call void @llvm.riscv.tt.contents.compc()
  call void @llvm.riscv.tt.contents.popc(i32 1)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %predicate, i32 2)
  call void @llvm.riscv.tt.contents.popc(i32 3)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %predicate, i32 4)
  call void @llvm.riscv.tt.contents.popc(i32 4)
  call void @llvm.riscv.tt.contents.setcc(<32 x i32> %predicate, i32 6)
  call void @llvm.riscv.tt.contents.popc(i32 0)
  ret void
}
