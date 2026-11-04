; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=finalize-isel %t/input.ll -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-replay-selection %t/input.ll -o %t/O0-before.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-after=riscv-tensix-replay-selection %t/input.ll -o %t/O0-selected.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-replay-selection %t/input.ll -o %t/O0-plain.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -run-pass=riscv-tensix-hazards %t/O0-selected.mir -o %t/O0-final.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/input.ll -o %t/O0-selected.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -filetype=obj %t/input.ll -o %t/O0-plain.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t/O0-selected.o > %t/O0-selected.dis
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t/O0-plain.o > %t/O0-plain.dis
; RUN: %python %t/check-replay.py %t/O0-before.mir %t/O0-selected.mir %t/O0-plain.mir %t/O0-final.mir %t/O0-selected.dis %t/O0-plain.dis
; RUN: not %python %t/check-replay.py %t/O0-before.mir %t/O0-plain.mir %t/O0-plain.mir %t/O0-final.mir %t/O0-selected.dis %t/O0-plain.dis 2>&1 | FileCheck %s --check-prefix=DISABLED
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-replay-selection %t/input.ll -o %t/O2-before.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-after=riscv-tensix-replay-selection %t/input.ll -o %t/O2-selected.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -stop-after=riscv-tensix-replay-selection %t/input.ll -o %t/O2-plain.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -verify-machineinstrs -run-pass=riscv-tensix-hazards %t/O2-selected.mir -o %t/O2-final.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/input.ll -o %t/O2-selected.o
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -riscv-tensix-enable-replay-selection=false -filetype=obj %t/input.ll -o %t/O2-plain.o
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t/O2-selected.o > %t/O2-selected.dis
; RUN: llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh %t/O2-plain.o > %t/O2-plain.dis
; RUN: %python %t/check-replay.py %t/O2-before.mir %t/O2-selected.mir %t/O2-plain.mir %t/O2-final.mir %t/O2-selected.dis %t/O2-plain.dis
; RUN: not %python %t/check-replay.py %t/O2-before.mir %t/O2-plain.mir %t/O2-plain.mir %t/O2-final.mir %t/O2-selected.dis %t/O2-plain.dis 2>&1 | FileCheck %s --check-prefix=DISABLED
;
; The author supplies two existing arithmetic sequences, not a replay recipe
; or an unrolled source loop. This checks genuine contents SSA through isel,
; allocation, automatic replay and final object. An earlier snapshot must
; remain live and distinct while the active numerical contents are updated.
; The separate dynamic loop retains its one source body and real backedge.
; The checker independently expands the 32-word hardware bank and compares
; complete MIR issue/CFG/memory effects, physical Replay use/def, and the
; final object arithmetic stream. Disabled Replay is a sensitivity control.
; Existing auto-replay.mir owns missing-use/def, length, predicate, backedge
; and capacity mutations; no authored Replay/MOP bound guard is relaxed.
;
; ISEL-LABEL: name:{{ *}}contents_auto_replay
; ISEL: :sfpr = TTSFPLOAD
; ISEL: :sfpr = TTSFPIADD
; ISEL: :sfpr = TTSFPXOR
; ISEL-LABEL: name:{{ *}}contents_real_loop
; ISEL: :sfpr = PHI
; ISEL: :sfpr = TTSFPIADD
; DISABLED: automatic arithmetic replay was not selected

;--- input.ll
; A genuine numerical contents source. The author explicitly wrote the two
; arithmetic sequences below; no compiler or preparation tool copied a loop.
; Snapshot remains live until after both sequences and must retain its own
; allocated contents. This file never calls the bound or authored Replay ABI.
target triple = "riscv32"

declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32>, i32, i32 immarg, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32>)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)

define void @contents_auto_replay() "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %snapshot.old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %snapshot = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %snapshot.old, i32 6, i32 0, i32 4)
  %initial.old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %initial.old, i32 0, i32 0, i32 4)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  %first.1 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %initial, <32 x i32> %mask, i32 4)
  %first.2 = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %first.1, <32 x i32> %mask)
  %first.3 = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %first.2, <32 x i32> %mask)
  %first.4 = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %first.3, <32 x i32> %mask)
  %first.5 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %first.4)
  %first.6 = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %first.5, <32 x i32> %mask)
  %second.1 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %first.6, <32 x i32> %mask, i32 4)
  %second.2 = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %second.1, <32 x i32> %mask)
  %second.3 = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %second.2, <32 x i32> %mask)
  %second.4 = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %second.3, <32 x i32> %mask)
  %second.5 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %second.4)
  %second.6 = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %second.5, <32 x i32> %mask)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second.6, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %snapshot, i32 6, i32 0, i32 4)
  ret void
}

; A separate negative boundary: the author supplied one body with a true
; dynamic backedge, never two explicit occurrences eligible for Replay.
define void @contents_real_loop(i32 %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %initial.old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %initial = call <32 x i32> @llvm.riscv.tt.contents.load(<32 x i32> %initial.old, i32 0, i32 0, i32 4)
  %mask = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  br label %loop
loop:
  %iteration = phi i32 [ 0, %entry ], [ %next, %loop ]
  %old = phi <32 x i32> [ %initial, %entry ], [ %result, %loop ]
  %step.1 = call <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32> %old, <32 x i32> %mask, i32 4)
  %step.2 = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %step.1, <32 x i32> %mask)
  %step.3 = call <32 x i32> @llvm.riscv.tt.contents.and(<32 x i32> %step.2, <32 x i32> %mask)
  %step.4 = call <32 x i32> @llvm.riscv.tt.contents.or(<32 x i32> %step.3, <32 x i32> %mask)
  %step.5 = call <32 x i32> @llvm.riscv.tt.contents.not(<32 x i32> %step.4)
  %result = call <32 x i32> @llvm.riscv.tt.contents.xor(<32 x i32> %step.5, <32 x i32> %mask)
  %next = add i32 %iteration, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  call void @llvm.riscv.tt.contents.store(<32 x i32> %result, i32 0, i32 0, i32 4)
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}

;--- check-replay.py
"""Check the receiver seam without reusing the compiler's mining/effect logic."""
import json
import re
import sys
from pathlib import Path


def read_mir(path):
    functions = {}
    current = None
    in_body = False
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"name:\s+(\w+)\s*", line.strip())
        if match:
            current = match.group(1)
            functions[current] = []
            in_body = False
        if re.fullmatch(r"body:\s+\|", line):
            in_body = True
            continue
        if not in_body or current is None:
            continue
        text = line.strip()
        if text == "...":
            in_body = False
            continue
        if not text or text.startswith(("liveins:", "successors:")):
            continue
        if text.startswith("bb."):
            functions[current].append(("block", (), (text,), (), (), ""))
            continue
        # Liveness annotations do not change instruction bits or actual effects.
        text = re.sub(r"\b(?:killed|dead|renamable|frame-setup|frame-destroy)\s+", "", text)
        code, _, memory = text.partition(" :: ")
        definition, equal, instruction = code.partition(" = ")
        if not equal:
            definition, instruction = "", code
        match = re.fullmatch(r"(\w+)(?:\s+(.*))?", instruction)
        assert match, (path, current, text)
        opcode, operands = match.group(1), match.group(2) or ""
        explicit = operands.split(", implicit", 1)[0]
        args = tuple(argument.strip() for argument in explicit.split(",") if argument.strip())
        defs = tuple(re.findall(r"\$\w+", definition))
        uses = tuple(re.findall(r"implicit (\$\w+)", operands))
        clobbers = tuple(re.findall(r"implicit-def (\$\w+)", operands))
        functions[current].append((opcode, defs, args, uses, clobbers, memory))
    assert functions, path
    return functions


def expand_mir(instructions):
    # Hardware has one 32-word Replay bank. A header changes issue handling;
    # record-and-execute runs the retained words now, execute fetches old words.
    bank = [None] * 32
    recording = None
    result = []
    records = []
    for instruction in instructions:
        opcode, defs, args, uses, clobbers, memory = instruction
        if opcode == "PseudoTTSFPUReplay":
            load, execute, length, start = map(int, args)
            assert 0 < length <= 32 and 0 <= start <= 32 - length
            assert recording is None
            if load:
                recording = [start, length, execute, 0]
            else:
                payload = bank[start:start + length]
                assert all(word is not None for word in payload)
                result.extend(payload)
                records.append((instruction, payload))
            continue
        if opcode == "PseudoTTReplayRecordEnd":
            assert recording is not None and recording[1] == recording[3]
            recording = None
            continue
        if recording is not None:
            assert opcode.startswith("TTSFP"), instruction
            start, length, execute, count = recording
            assert count < length
            bank[start + count] = instruction
            recording[3] += 1
            if execute:
                result.append(instruction)
        else:
            result.append(instruction)
    assert recording is None
    # Arithmetic/control/CFG/memory effects must match completely. Extra SFPU
    # fences only alter spacing; the separate LLVM hazard verifier checks them.
    return [word for word in result if word[0] != "TTSFPNOP"], records


def issued_words(instructions):
    return sum(word[0].startswith("TTSFP") or word[0] == "PseudoTTSFPUReplay"
               for word in instructions)


def physical_effects(payload):
    reads, written = set(), set()
    for _, defs, args, uses, clobbers, _ in payload:
        operands = {reg for argument in args for reg in re.findall(r"\$\w+", argument)} | set(uses)
        reads |= operands - written
        written |= set(defs) | set(clobbers)
    return reads, written


def read_disassembly(path):
    functions = {}
    current = None
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"[0-9a-f]+ <(\w+)>:", line)
        if match:
            current = match.group(1)
            functions[current] = []
            continue
        match = re.match(r"\s*[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)(?:\s+(.*))?$", line)
        if match and current:
            opcode, operands = match.group(1), match.group(2) or ""
            # Compression moves subsequent functions. Compare symbol-relative
            # branch targets, retaining the actual destination and offset.
            operands = re.sub(r"0x[0-9a-f]+ (<[^>]+>)", r"\1", operands)
            functions[current].append((opcode, tuple(arg.strip() for arg in operands.split(",") if arg.strip())))
    return functions


def expand_object(words):
    bank = [None] * 32
    recording = None
    output = []
    for word in words:
        opcode, args = word
        if opcode == "ttreplay":
            load, execute, length, start = map(int, args)
            assert recording is None and 0 < length <= 32 and start + length <= 32
            if load:
                recording = [start, length, execute, 0]
            else:
                payload = bank[start:start + length]
                assert all(value is not None for value in payload)
                output.extend(payload)
            continue
        if recording:
            start, length, execute, count = recording
            assert opcode.startswith("ttsfp")
            bank[start + count] = word
            recording[3] += 1
            if execute:
                output.append(word)
            if recording[3] == length:
                recording = None
        else:
            output.append(word)
    assert recording is None
    return [word for word in output if word[0] != "ttsfpnop"]


before_path, selected_path, plain_path, final_path, selected_dis, plain_dis = sys.argv[1:]
before, selected, plain, final = map(read_mir, (before_path, selected_path, plain_path, final_path))
assert before.keys() == selected.keys() == plain.keys() == final.keys()
summary = {}
for name, original in before.items():
    expected, original_records = expand_mir(original)
    assert not original_records
    for stage, functions in (("selected", selected), ("plain", plain), ("final", final)):
        actual, records = expand_mir(functions[name])
        assert actual == expected, (name, stage, "changed complete issued stream or CFG")
    assert not any(word[0] == "PseudoTTSFPUReplay" for word in plain[name])
    if name == "contents_real_loop":
        assert not any(word[0] == "PseudoTTSFPUReplay" for word in selected[name])
        assert sum(word[0] == "TTSFPIADD" for word in original) == 1
        assert sum(word[0] == "BNE" and word[2][-1] == "%bb.1" for word in original) == 1
        assert sum(word[0] == "block" for word in original) == 3
        summary[name] = {"replay": False, "real_backedge": True, "one_authored_body": True}
        continue
    assert name == "contents_auto_replay", name
    for functions in (selected, final):
        headers = [word for word in functions[name] if word[0] == "PseudoTTSFPUReplay"]
        assert [word[2] for word in headers] == [("1", "1", "6", "0"), ("0", "0", "6", "0")], "automatic arithmetic replay was not selected"
        assert issued_words(functions[name]) == issued_words(original) - 2
        _, executions = expand_mir(functions[name])
        assert len(executions) == 1
        execution, payload = executions[0]
        expected_uses, expected_defs = physical_effects(payload)
        assert set(execution[3]) == expected_uses, (execution, expected_uses)
        assert set(execution[4]) == expected_defs, (execution, expected_defs)
    loads = [word for word in original if word[0] == "TTSFPLOAD"]
    stores = [word for word in original if word[0] == "TTSFPSTORE"]
    assert len(loads) == len(stores) == 2
    snapshot = loads[0][1][0]
    assert loads[0][2][1] == stores[1][2][1] == "6"
    assert stores[1][2][0] == snapshot
    assert snapshot not in expected_uses | expected_defs
    assert all(snapshot not in word[1] for word in original[original.index(loads[0]) + 1:])
    # Numerical SSA really crossed isel/allocation, rather than bound ingress.
    text = Path(selected_path).read_text()
    assert re.search(r"usesBoundTensixSFPU:\s+false", text) and re.search(r"noVRegs:\s+true", text)
    summary[name] = {"record_words": len(payload), "saved_issue_words": 2,
                     "execute_uses": sorted(expected_uses), "execute_defs": sorted(expected_defs),
                     "preserved_snapshot": snapshot}

selected_object, plain_object = map(read_disassembly, (selected_dis, plain_dis))
assert selected_object.keys() == plain_object.keys() == selected.keys()
headers = [word[1] for word in selected_object["contents_auto_replay"] if word[0] == "ttreplay"]
assert headers == [("1", "1", "6", "0"), ("0", "0", "6", "0")], "final object lost automatic arithmetic replay"
def object_issue_words(words):
    return sum(word[0].startswith("ttsfp") or word[0] == "ttreplay" for word in words)
assert object_issue_words(selected_object["contents_auto_replay"]) == object_issue_words(plain_object["contents_auto_replay"]) - 2
assert expand_object(selected_object["contents_auto_replay"]) == expand_object(plain_object["contents_auto_replay"])
assert selected_object["contents_real_loop"] == plain_object["contents_real_loop"], "final object changed the real loop or its output"
assert not any(word[0] == "ttreplay" for word in selected_object["contents_real_loop"])
assert sum(word[0] == "ttsfpiadd" for word in selected_object["contents_real_loop"]) == 1
assert sum(word[0].startswith("bne") for word in selected_object["contents_real_loop"]) == 1
assert not any(word[0] == "ttreplay" for words in plain_object.values() for word in words)
summary["object"] = {"expanded_positive_stream_equal": True, "plain_has_no_replay": True,
                      "loop_body_and_branch_retained": True}
print(json.dumps(summary, indent=2, sort_keys=True))
