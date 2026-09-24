; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=finalize-isel %t/masked-lut.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/masked-lut.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-after=postrapseudos %t/masked-lut.ll -o %t/ra-o0.mir
; RUN: FileCheck %s --check-prefix=RA < %t/ra-o0.mir
; RUN: FileCheck %s --check-prefix=PRESERVE < %t/ra-o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-after=postrapseudos %t/masked-lut.ll -o %t/ra-o2.mir
; RUN: FileCheck %s --check-prefix=RA < %t/ra-o2.mir
; RUN: FileCheck %s --check-prefix=PRESERVE < %t/ra-o2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/masked-lut.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/masked-lut.ll -o /dev/null
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/transpose-live.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LIVE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -filetype=obj %t/transpose-live.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LIVE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -filetype=obj %t/transpose-fixed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FIXED
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -filetype=obj %t/transpose-fixed.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=FIXED

; The old destination is independent of all four fixed-group LUT inputs. Its
; inactive lanes must survive the issue, and its original full-lane value is
; still observed after the predicate is popped. Dst loads make the five input
; values independent; no constant-register equality can conceal bad staging.
; ISEL-LABEL: name: masked_independent_old_lut
; ISEL: [[OLD:%[0-9]+]]:sfpr = TTSFPLOAD {{%[0-9]+}}, 0, 0, 4,
; ISEL: [[A:%[0-9]+]]:sfpr = TTSFPLOAD {{%[0-9]+}}, 1, 0, 4,
; ISEL: [[B:%[0-9]+]]:sfpr = TTSFPLOAD {{%[0-9]+}}, 2, 0, 4,
; ISEL: [[C:%[0-9]+]]:sfpr = TTSFPLOAD {{%[0-9]+}}, 3, 0, 4,
; ISEL: [[X:%[0-9]+]]:sfpr = TTSFPLOAD {{%[0-9]+}}, 4, 0, 4,
; ISEL: TTSFPPUSHC
; ISEL-NEXT: TTSFPSETCCNE [[X]],
; ISEL-NEXT: $tt_l0 = COPY [[A]]
; ISEL-NEXT: $tt_l1 = COPY [[B]]
; ISEL-NEXT: $tt_l2 = COPY [[C]]
; ISEL-NEXT: $tt_l3 = COPY [[X]]
; ISEL-NEXT: [[RESULT:%[0-9]+]]:sfpr = TTSFPLUT [[OLD]], 4,
; ISEL-SAME: implicit $tt_l0, implicit $tt_l1, implicit $tt_l2, implicit $tt_l3, implicit $tt_cc
; ISEL-NEXT: TTSFPPOPC
; ISEL-NEXT: TTSFPSTORE killed [[RESULT]], 10, 0, 4,
; ISEL-NEXT: TTSFPSTORE [[OLD]], 11, 0, 4,

; The tied result receives an all-lane copy while old remains live. The move
; may legally occur on either side of SETCC, so do not freeze that scheduling.
; RA-LABEL: name: masked_independent_old_lut
; RA: [[OLDREG:\$tt_l[4-7]]] = TTSFPLOAD {{.*}}, 0, 0, 4,
; RA: TTSFPPUSHC
; RA-DAG: [[RESULTREG:\$tt_l[4-7]]] = TTSFPMOVAll [[OLDREG]],
; RA-DAG: TTSFPSETCCNE
; RA: [[RESULTREG]] = TTSFPLUT killed renamable [[RESULTREG]], 4,
; RA-SAME: implicit {{(killed )?}}$tt_l0, implicit {{(killed )?}}$tt_l1, implicit {{(killed )?}}$tt_l2, implicit {{(killed )?}}$tt_l3, implicit $tt_cc
; RA-NEXT: TTSFPPOPC
; RA-NEXT: TTSFPSTORE killed renamable [[RESULTREG]], 10, 0, 4,
; RA-NEXT: TTSFPSTORE killed renamable [[OLDREG]], 11, 0, 4,
; RA-NEXT: PseudoRET

; A separate check spans the whole masked region, proving result and old do
; not merely match the same register in the positive allocation checks.
; PRESERVE-LABEL: name: masked_independent_old_lut
; PRESERVE: [[OLDREG:\$tt_l[4-7]]] = TTSFPLOAD {{.*}}, 0, 0, 4,
; PRESERVE-NOT: [[OLDREG]] = TTSFPLUT
; PRESERVE: PseudoRET

; SFPTRANSP clobbers both L0-L3 and L4-L7. An ordinary SSA value live across
; it, or a fixed binding requiring restoration, cannot be hidden in another
; LReg. The no-spill contract must fail instead of corrupting either value.
; LIVE: error: {{.*}}ran out of registers during register allocation in function 'transpose_live'
; FIXED: error: {{.*}}ran out of registers during register allocation in function 'transpose_fixed'

;--- masked-lut.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpencc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppushc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfpsetcc(<32 x i32>, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.sfppopc(i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfplut(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @masked_independent_old_lut() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.sfpencc(i32 3, i32 10)
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %old = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 0, i32 0, i32 4)
  %a = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 1, i32 0, i32 4)
  %b = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 2, i32 0, i32 4)
  %c = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 3, i32 0, i32 4)
  %x = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 4, i32 0, i32 4)
  call void @llvm.riscv.tt.sfppushc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpsetcc(<32 x i32> %x, i32 0, i32 2)
  %v = call <32 x i32> @llvm.riscv.tt.sfplut(<32 x i32> %old, <32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %x, i32 4)
  call void @llvm.riscv.tt.sfppopc(i32 0, i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %v, i32 10, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %old, i32 11, i32 0, i32 4)
  ret void
}

;--- transpose-live.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32>, i32, i32 immarg, i32 immarg)
declare {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} @llvm.riscv.tt.sfptransp(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @transpose_live() "tensix-executor"="trisc1" {
  %zero = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %a = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 0, i32 0, i32 4)
  %b = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 1, i32 0, i32 4)
  %c = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 2, i32 0, i32 4)
  %d = call <32 x i32> @llvm.riscv.tt.sfpload(<32 x i32> %zero, i32 3, i32 0, i32 4)
  %four = call {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} @llvm.riscv.tt.sfptransp(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d)
  %r0 = extractvalue {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} %four, 0
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %r0, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %a, i32 9, i32 0, i32 4)
  ret void
}

;--- transpose-fixed.ll
declare <32 x i32> @llvm.riscv.tt.creg.read(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.lreg.read(i32 immarg)
declare {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} @llvm.riscv.tt.sfptransp(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>)
declare void @llvm.riscv.tt.sfpstore(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @transpose_fixed() "tensix-executor"="trisc1" {
  %a = call <32 x i32> @llvm.riscv.tt.creg.read(i32 9)
  %b = call <32 x i32> @llvm.riscv.tt.creg.read(i32 10)
  %c = call <32 x i32> @llvm.riscv.tt.creg.read(i32 11)
  %d = call <32 x i32> @llvm.riscv.tt.creg.read(i32 15)
  %four = call {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} @llvm.riscv.tt.sfptransp(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d)
  %r0 = extractvalue {<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>} %four, 0
  %fixed = call <32 x i32> @llvm.riscv.tt.lreg.read(i32 0)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %r0, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.sfpstore(<32 x i32> %fixed, i32 9, i32 0, i32 4)
  ret void
}
