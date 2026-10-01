; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/loops.ll -o %t/o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/loops.ll -o %t/o2.s
; RUN: FileCheck %s < %t/o0.s
; RUN: FileCheck %s < %t/o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/loops.ll -o %t/o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/loops.ll -o %t/o2.mir
; RUN: %python %t/check_loops.py %t/o0.s %t/o0.mir %S/Inputs/tensix-explicit-replay-observer.py
; RUN: %python %t/check_loops.py %t/o2.s %t/o2.mir %S/Inputs/tensix-explicit-replay-observer.py
;
; A record outside a runtime loop executes zero times in mode 0 and once in
; mode 1 when count=0. The same record is used by each real backedge. Subsequent
; executions read the L1 updated by the preceding iteration. A final store
; exposes the zero-trip difference; checking only loop stores would miss it.
; Exact static emitted words reject loop peeling, duplicated template bodies,
; hidden SFPR copies, or a separate execute substituted for mode 1.
;
; FileCheck proves the scalar branch/backedge shape. The embedded test applies
; the existing independent word observer to those emitted block words for
; paths of length 0, 1, and 7. It is bounded codegen evidence, not a scalar ISA
; interpreter or hardware execution. No production interpreter is introduced.
;
; CHECK-LABEL: structured_loop_record_only:
; CHECK: .word 0x10050044
; CHECK-NEXT: .word 0xf0000409
; CHECK: {{beq|beqz|bne|bnez}}
; CHECK: .LBB{{[0-9_]+}}:
; CHECK: .word 0x10050040
; CHECK: .word 0xf0002849
; CHECK: .word 0xc8100021
; CHECK: {{bne|bltu|blt|bnez}} {{.*}}.LBB
; CHECK: .word 0xc8100001
; CHECK: ret
; CHECK-LABEL: structured_loop_record_and_execute:
; CHECK: .word 0x1005004c
; CHECK-NEXT: .word 0xf0000409
; CHECK: {{beq|beqz|bne|bnez}}
; CHECK: .LBB{{[0-9_]+}}:
; CHECK: .word 0x10050040
; CHECK: .word 0xf0002849
; CHECK: .word 0xc8100021
; CHECK: {{bne|bltu|blt|bnez}} {{.*}}.LBB
; CHECK: .word 0xc8100001
; CHECK: ret

;--- loops.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @structured_loop_record_only(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay.template.begin(i32 23, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 23)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.replay.template.execute(i32 23)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

define void @structured_loop_record_and_execute(i32 %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 9)
  call void @llvm.riscv.tt.replay.template.begin(i32 71, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 1)
  call void @llvm.riscv.tt.replay.template.end(i32 71)
  %nonzero = icmp ne i32 %count, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %loop]
  call void @llvm.riscv.tt.replay.template.execute(i32 71)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 10)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 8, i32 0, i32 4)
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  ret void
}

;--- check_loops.py
import pathlib
import re
import runpy
import sys

assembly_path, mir_path, observer_path = sys.argv[1:]
assembly = pathlib.Path(assembly_path).read_text()
mir = pathlib.Path(mir_path).read_text()
# Load the existing test-only oracle without asking its command-line driver
# to check a differently named function. No production helper computes wants.
sys.argv = [observer_path, assembly_path, mir_path]
observer = runpy.run_path(observer_path)
zero = (0,) * 32
one = (0x3F800000,) * 32

for name, mode in (("structured_loop_record_only", 0),
                   ("structured_loop_record_and_execute", 1)):
    block = re.search(r"^" + name + r":.*?^\.Lfunc_end\d+:", assembly,
                      re.M | re.S)
    assert block, (name, "missing assembly function")
    lines = block[0].splitlines()
    execute_line = next(i for i, line in enumerate(lines)
                        if re.search(r"\.word\s+0x10050040\b", line))
    labels = [match[1] for line in lines[:execute_line]
              if (match := re.match(r"(\.LBB[0-9_]+):", line))]
    assert labels, (name, "execute has no scalar loop block")
    loop_label = labels[-1]
    assert any(re.search(r"\b(?:bne|bltu|blt|bnez)\b.*" +
                         re.escape(loop_label) + r"(?:\s|$)", line)
               for line in lines[execute_line + 1:]), (name, "lost real backedge")
    words = observer["words"](assembly, name)
    useful = [word for word in words if word != 0x8F000000]
    expected = [0x7C000A02, 0x7C000912,
                0x04014013 if mode else 0x04014011, 0x7C000102,
                0x04014010, 0x7C000A12, 0x72040008, 0x72040000]
    assert useful == expected, (name, "loop or captured body duplicated/changed")
    observer["check_machine_effects"](mir, name, mode)
    for count in (0, 1, 7):
        # FileCheck independently establishes entry -> loop* -> exit. Reuse
        # actual emitted words from those blocks rather than a source recipe.
        path_words = useful[:4] + useful[4:7] * count + useful[7:]
        if count == 0:
            want = [(0, zero if mode else one)]
        else:
            want = [(8, zero)] + [(8, one)] * (count - 1)
            want += [(0, zero if count == 1 else one)]
        for seed in (37, 911):
            assert observer["observe"](path_words, seed) == want, (name, count, seed)
