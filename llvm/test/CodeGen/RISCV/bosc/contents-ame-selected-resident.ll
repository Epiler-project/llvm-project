; RUN: split-file %s %t
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -stop-after=finalize-isel %t/input.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -stop-after=finalize-isel %t/input.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs %t/input.ll -o %t/O0.s
; RUN: FileCheck %s --check-prefix=ASM < %t/O0.s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs %t/input.ll -o %t/O2.s
; RUN: FileCheck %s --check-prefix=ASM < %t/O2.s
; RUN: %python %t/check.py %t/O0.s %t/O2.s
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O0 -verify-machineinstrs -filetype=obj %t/input.ll -o %t/O0.o
; RUN: llc -mtriple=riscv64 -mattr=+xboscame -O2 -verify-machineinstrs -filetype=obj %t/input.ll -o %t/O2.o
;
; Unlike mutually exclusive B loads, both B contents are resident before the
; condition and remain live after its Acc PHI. The allocator must preserve
; their distinct contents without matrix copies, spills or physical selectors.
; O0 may spill ordinary scalar parameters; these are not matrix contents.
;
; ISEL-LABEL: name: resident_b_choice
; ISEL: [[A:%[0-9]+]]:bosctrreg = BOSC_AME_LOADA
; ISEL: [[B0:%[0-9]+]]:bosctrreg = BOSC_AME_LOADB
; ISEL: [[B1:%[0-9]+]]:bosctrreg = BOSC_AME_LOADB
; ISEL: [[C:%[0-9]+]]:boscaccreg = BOSC_AME_LOADC
; ISEL: BEQ {{.*}}, $x0, %bb.2
; ISEL: bb.1.yes:
; ISEL: [[YES:%[0-9]+]]:boscaccreg = BOSC_AME_MMA [[C]], [[A]], [[B0]],
; ISEL: bb.2.no:
; ISEL: [[NO:%[0-9]+]]:boscaccreg = BOSC_AME_MMA [[C]], [[A]], [[B1]],
; ISEL: bb.3.join:
; ISEL: [[SELECT:%[0-9]+]]:boscaccreg = PHI [[NO]], %bb.2, [[YES]], %bb.1
; ISEL: [[NEXT:%[0-9]+]]:boscaccreg = BOSC_AME_MMA [[SELECT]], [[A]], [[B0]],
; ISEL: [[LAST:%[0-9]+]]:boscaccreg = BOSC_AME_MMA [[NEXT]], [[A]], [[B1]],
; ISEL: BOSC_AME_STOREC {{(killed )?}}[[LAST]],
;
; ASM-LABEL: resident_b_choice:
; ASM: mlae8.m [[A:tr[0-9]+]],
; ASM: mlbe8.m [[B0:tr[0-9]+]],
; ASM: mlbe8.m [[B1:tr[0-9]+]],
; ASM: mlce32.m [[C:acc[0-9]+]],
; ASM: beqz {{[a-z0-9]+}}, [[NO:.LBB[0-9]+_2]]
; ASM: # %yes
; ASM-NEXT: mqma.b.mm [[C]], [[A]], [[B0]]
; ASM-NEXT: j [[JOIN:.LBB[0-9]+_3]]
; ASM: [[NO]]:
; ASM-NEXT: mqma.b.mm [[C]], [[A]], [[B1]]
; ASM: [[JOIN]]:
; ASM: mqma.b.mm [[C]], [[A]], [[B0]]
; ASM-NEXT: mqma.b.mm [[C]], [[A]], [[B1]]
; ASM: msce32.m [[C]],
; ASM: ret

;--- input.ll
; Two B values coexist before a branch and both remain live after the join.
; No physical register selectors, loop duplication, or memory round-trip is
; available to make the selection work.
target triple = "riscv64"

declare i64 @llvm.riscv.bosc.ame.setm(i64)
declare i64 @llvm.riscv.bosc.ame.setn(i64)
declare i64 @llvm.riscv.bosc.ame.setk(i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr, i64, i64, i64, i64)
declare target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc"), target("riscv.bosc.ame.tr"), target("riscv.bosc.ame.tr"), i64, i64, i64)
declare void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc"), ptr, i64, i64, i64, i64)

define void @resident_b_choice(ptr %a, ptr %b0, ptr %b1, ptr %c, ptr %out, i1 %choose) #0 {
entry:
  %m = call i64 @llvm.riscv.bosc.ame.setm(i64 16)
  %n = call i64 @llvm.riscv.bosc.ame.setn(i64 16)
  %k = call i64 @llvm.riscv.bosc.ame.setk(i64 64)
  %av = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.a(ptr %a, i64 64, i64 %m, i64 %n, i64 %k)
  %bv0 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b0, i64 16, i64 %m, i64 %n, i64 %k)
  %bv1 = call target("riscv.bosc.ame.tr") @llvm.riscv.bosc.ame.contents.load.b(ptr %b1, i64 16, i64 %m, i64 %n, i64 %k)
  %cv = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.load.c(ptr %c, i64 64, i64 %m, i64 %n, i64 %k)
  br i1 %choose, label %yes, label %no

yes:
  %cy = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv0, i64 %m, i64 %n, i64 %k)
  br label %join

no:
  %cn = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %cv, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv1, i64 %m, i64 %n, i64 %k)
  br label %join

join:
  %selected = phi target("riscv.bosc.ame.acc") [ %cy, %yes ], [ %cn, %no ]
  %after0 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %selected, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv0, i64 %m, i64 %n, i64 %k)
  %after1 = call target("riscv.bosc.ame.acc") @llvm.riscv.bosc.ame.contents.mma(target("riscv.bosc.ame.acc") %after0, target("riscv.bosc.ame.tr") %av, target("riscv.bosc.ame.tr") %bv1, i64 %m, i64 %n, i64 %k)
  call void @llvm.riscv.bosc.ame.contents.store.c(target("riscv.bosc.ame.acc") %after1, ptr %out, i64 64, i64 %m, i64 %n, i64 %k)
  ret void
}

attributes #0 = { "trex-bosc-contents-ame"="1" }

;--- check.py
import collections
from pathlib import Path
import re
import sys

for path in sys.argv[1:]:
    text = Path(path).read_text()
    a = re.findall(r"\bmlae8\.m\s+(tr\d+),", text)
    bs = re.findall(r"\bmlbe8\.m\s+(tr\d+),", text)
    assert len(a) == 1 and len(bs) == 2 and len(set(a + bs)) == 3, "resident A/B contents alias"
    # Catch extra matrix transfers/helpers as well as lost operations. Scalar
    # 'mv' and ordinary GPR stack accesses remain permitted at O0.
    matrix = re.findall(r"^\s+(m(?!v\b)[\w.]+)\s", text, re.MULTILINE)
    assert collections.Counter(matrix) == {
        "msettilem": 1, "msettilen": 1, "msettilek": 1,
        "mlae8.m": 1, "mlbe8.m": 2, "mlce32.m": 1,
        "mqma.b.mm": 4, "msce32.m": 1,
    }, (path, matrix)
    assert not re.search(r"^\s+(?:call|tail|jal|jalr|jr)\b", text, re.MULTILINE), "unexpected helper"
    labels = set(re.findall(r"^(\.LBB[0-9]+_[0-9]+):", text, re.MULTILINE))
    jumps = re.findall(r"^\s+j\s+([^\s#]+)", text, re.MULTILINE)
    assert all(target in labels for target in jumps), "jump leaves function"
