; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t.selected
; RUN: FileCheck %s --check-prefixes=SELECT,LOOPS < %t.selected
; RUN: FileCheck %s --check-prefix=SCALAR < %t.selected
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -riscv-tensix-enable-replay-selection=false %s -o %t.plain
; RUN: FileCheck %s --check-prefixes=PLAIN,LOOPS < %t.plain
; RUN: FileCheck %s --check-prefix=SCALAR < %t.plain

declare void @llvm.riscv.tt.setrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.incrwc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.nop()
declare void @llvm.riscv.tt.replay(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.record.end()
declare void @llvm.riscv.tt.sfpnop()

; The source contains six existing instructions. Mining saves one word:
; record+execute, the three original words, then one replay word. It creates
; no source iteration and preserves the ordinary instruction order.
define void @existing_repeat() "tensix-executor"="trisc1" {
; SELECT-LABEL: existing_repeat:
; SELECT: .word 0x100000cc
; SELECT-NEXT: .word
; SELECT-NEXT: .word
; SELECT-NEXT: .word
; SELECT-NEXT: .word 0x100000c0
; SELECT-NEXT: ret
; PLAIN-LABEL: existing_repeat:
; PLAIN-NOT: .word 0x100000c
; PLAIN-COUNT-6: .word
; PLAIN-NEXT: ret
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  ret void
}

; A structured loop remains a loop. Three words written once in the source
; are not duplicated to manufacture the repeated sequence above.
define void @preserve_loop(i32 %n) "tensix-executor"="trisc1" {
; LOOPS-LABEL: preserve_loop:
; LOOPS-NOT: .word
; LOOPS: [[LOOP:.LBB[0-9_]+]]:
; LOOPS-NOT: .word
; LOOPS: .word 0xdc000004
; LOOPS-NEXT: .word 0xe0000100
; LOOPS-NEXT: .word 0x08000000
; LOOPS-NOT: .word
; LOOPS: bltu {{[a-z0-9]+}}, {{[a-z0-9]+}}, [[LOOP]]
; LOOPS-NOT: .word
; LOOPS: ret
; LOOPS-NOT: .word
; LOOPS: .Lfunc_end
  br label %loop
loop:
  %i = phi i32 [0, %0], [%next, %loop]
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %n
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

; Known trip counts must not manufacture a candidate either. The non-unit
; induction step and volatile scalar store are part of each original iteration.
define void @preserve_static_loop(ptr %sink) "tensix-executor"="trisc1" {
; LOOPS-LABEL: preserve_static_loop:
; LOOPS-NOT: .word
; LOOPS: .word 0xdc000004
; LOOPS-NEXT: .word 0xe0000100
; LOOPS-NEXT: .word 0x08000000
; LOOPS-NOT: .word
; LOOPS: {{bne|bltu}} {{.*}}.LBB
; LOOPS-NOT: .word
; LOOPS: ret
; LOOPS-NOT: .word
; LOOPS: .Lfunc_end
; SCALAR-LABEL: preserve_static_loop:
; SCALAR-NOT: {{^[ \t]*sh[ \t]}}
; SCALAR: {{^[ \t]*sh[ \t]}}
; SCALAR-NOT: {{^[ \t]*sh[ \t]}}
; SCALAR: .Lfunc_end
entry:
  br label %loop
loop:
  %i = phi i32 [1, %entry], [%next, %loop]
  %half = trunc i32 %i to i16
  store volatile i16 %half, ptr %sink, align 2
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  %next = add i32 %i, 2
  %more = icmp ult i32 %next, 11
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

; A multi-block inner loop is nested in a static outer loop. Count issue words
; over the complete function, including any peeled prologue or epilogue. The
; scalar halfword side effect is checked separately so scheduling is free.
define void @preserve_nested_scalar_effects(i32 %n, ptr %sink) "tensix-executor"="trisc1" {
; LOOPS-LABEL: preserve_nested_scalar_effects:
; LOOPS-NOT: .word
; LOOPS: .word 0xdc000004
; LOOPS-NEXT: .word 0xe0000100
; LOOPS-NEXT: .word 0x08000000
; LOOPS-NOT: .word
; LOOPS: ret
; LOOPS-NOT: .word
; LOOPS: .Lfunc_end
; SCALAR-LABEL: preserve_nested_scalar_effects:
; SCALAR-NOT: {{^[ \t]*sh[ \t]}}
; SCALAR: {{^[ \t]*sh[ \t]}}
; SCALAR-NOT: {{^[ \t]*sh[ \t]}}
; SCALAR: .Lfunc_end
entry:
  br label %outer
outer:
  %row = phi i32 [0, %entry], [%next.row, %outer.latch]
  br label %inner
inner:
  %column = phi i32 [1, %outer], [%next.column, %body]
  %more = icmp ult i32 %column, %n
  br i1 %more, label %body, label %outer.latch
body:
  %row.bits = shl i32 %row, 8
  %coordinate = add i32 %row.bits, %column
  %half = trunc i32 %coordinate to i16
  store volatile i16 %half, ptr %sink, align 2
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  %next.column = add i32 %column, 2
  br label %inner
outer.latch:
  %next.row = add i32 %row, 1
  %more.rows = icmp ult i32 %next.row, 3
  br i1 %more.rows, label %outer, label %exit
exit:
  ret void
}

define void @explicit_occupancy() "tensix-executor"="trisc1" {
; SELECT-LABEL: explicit_occupancy:
; SELECT: .word 0x10000044
; SELECT-NEXT: .word 0x08000000
; SELECT-NOT: .word 0x100000cc
; SELECT: ret
  call void @llvm.riscv.tt.replay(i32 1, i32 0, i32 1, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.replay.record.end()
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  call void @llvm.riscv.tt.setrwc(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.incrwc(i32 1, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.nop()
  ret void
}

define void @sfpu_is_not_mined() "tensix-executor"="trisc1" {
; SELECT-LABEL: sfpu_is_not_mined:
; SELECT-NOT: .word 0x100000
; SELECT-COUNT-6: .word 0x3c000002
; SELECT-NEXT: ret
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpnop()
  call void @llvm.riscv.tt.sfpnop()
  ret void
}
