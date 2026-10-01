; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/count.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=COUNT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/no-end.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=END
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/join.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=JOIN
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/slot.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=SLOT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/dynamic.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYNAMIC
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -run-pass=riscv-tensix-hazards %t/sfpu.mir -o /dev/null 2>&1 | FileCheck %s --check-prefix=SFPU
; COUNT: Tensix replay final issue count 2 differs from declared length 1
; END: Tensix replay recording cannot cross scalar control flow
; JOIN: Tensix replay execution references slots not recorded on every incoming path
; SLOT: immediate operand 0 must be in [2, 8]
; DYNAMIC: expected an immediate operand
; SFPU: SFPU replay recording requires a typed SSA preservation ABI

;--- count.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  ret void
}
;--- no-end.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  ret void
}
;--- join.ll
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.nop()
define void @bad(i1 %condition) "tensix-executor"="trisc1" {
entry:
  br i1 %condition, label %record, label %run
record:
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  br label %run
run:
  call void @llvm.riscv.tt.replay(i32 0, i32 0, i32 1, i32 0)
  ret void
}
;--- slot.ll
declare void @llvm.riscv.tt.nop.mop(i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.nop.mop(i32 1)
  ret void
}
;--- dynamic.ll
declare void @llvm.riscv.tt.replay.port(i32 immarg, i32, i32, i32, i32)
define void @bad(i32 noundef %len) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.port(i32 0, i32 0, i32 0, i32 %len, i32 0)
  ret void
}
;--- sfpu.mir
# An unbound SFPU recording is a machine-contract rejection. Do not fabricate
# this input through a retired LLVM intrinsic which becomes an external call.
--- |
  define void @bad() "target-features"="+xtttensixbh" "tensix-executor"="trisc1" { ret void }
...
---
name: bad
tracksRegLiveness: false
body: |
  bb.0:
    TTREPLAY 1, 0, 1, 0, implicit $tt_config, implicit $tt_dst, implicit $tt_issue, implicit-def $tt_config, implicit-def $tt_dst, implicit-def $tt_issue
    TTSFPNOP implicit $tt_issue, implicit-def $tt_issue
    PseudoTTReplayRecordEnd implicit $tt_issue, implicit-def $tt_issue
    PseudoRET
...
