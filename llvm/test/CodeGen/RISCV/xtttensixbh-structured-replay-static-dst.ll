; RUN: split-file %s %t
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %t/dst.ll -o %t/o0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %t/dst.ll -o %t/o2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/dst.ll -o %t/o0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %t/dst.ll -o %t/o2.mir
; RUN: %python %t/check_dst.py %t/o0.s %t/o0.mir
; RUN: %python %t/check_dst.py %t/o2.s %t/o2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/dst.ll -o /dev/null
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/dst.ll -o /dev/null
;
; Required static direct Dst receiver. Each two-word template loads Dst[32]
; into authored physical L0 and stores it to Dst[40]. Record-only must change
; neither location nor L0; record-and-execute must perform both operations.
; Changing Dst[32] between recording and execution distinguishes encoded
; address capture from an incorrect record-time snapshot of Dst contents.
;
; The independent emitted-word observer below covers raw32 rows only, under
; all-enabled CC and neutral address modifiers. It does not model Dst packing,
; timing, conversion formats, or arbitrary scalar instructions and is software
; codegen evidence, not hardware acceptance. MIR separately verifies the exact
; external physical effects and absence of virtual SFPR allocation.

;--- dst.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.replay.template.execute(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpmov.all(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.bound.sfpstore(i32 immarg, i32, i32 immarg, i32 immarg)

define void @structured_dst_record_only() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpstore(i32 1, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 103, i32 0, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 103)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 5)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 103)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 24, i32 0, i32 4)
  ret void
}

define void @structured_dst_record_and_execute() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 0, i32 10)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 1, i32 2)
  call void @llvm.riscv.tt.bound.sfpstore(i32 1, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.begin(i32 107, i32 1, i32 1, i32 5)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 107)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 8, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpmov.all(i32 4, i32 5)
  call void @llvm.riscv.tt.bound.sfpstore(i32 4, i32 32, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.execute(i32 107)
  call void @llvm.riscv.tt.bound.sfpstore(i32 0, i32 16, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpload(i32 3, i32 3, i32 40, i32 0, i32 4)
  call void @llvm.riscv.tt.bound.sfpstore(i32 3, i32 24, i32 0, i32 4)
  ret void
}

;--- check_dst.py
import collections
import pathlib
import re
import sys


def words(assembly, name):
    body = re.search(r"^" + name + r":.*?^\.Lfunc_end\d+:", assembly, re.M | re.S)
    assert body, (name, "missing assembly function")
    encoded = [int(word, 16) for word in re.findall(r"\.word\s+(0x[0-9a-fA-F]+)", body[0])]
    return [((word >> 2) | (word << 30)) & 0xFFFFFFFF for word in encoded]


def effects(line):
    uses, definitions = [], []
    if " = " in line:
        lhs, line = line.split(" = ", 1)
        definitions.extend(re.findall(r"\$(tt_\w+)", lhs))
    for operand in line.split(","):
        registers = re.findall(r"\$(tt_\w+)", operand)
        (definitions if "implicit-def" in operand else uses).extend(registers)
    return collections.Counter(uses), collections.Counter(definitions)


def observe(stream, seed):
    registers = {reg: tuple(seed + 100 * reg + lane for lane in range(32)) for reg in range(16)}
    registers[10] = (0x3F800000,) * 32
    dst = {address: tuple(seed + 10000 + 100 * address + lane for lane in range(32))
           for address in (0, 8, 16, 24, 32, 40)}
    slots, observations = {}, []
    recording = None

    def issue(word):
        opcode = word >> 24
        if opcode == 0x7C:
            destination, source = (word >> 4) & 15, (word >> 8) & 15
            assert word == 0x7C000002 | (source << 8) | (destination << 4)
            registers[destination] = registers[source]
        elif opcode in (0x70, 0x72):
            register, address = (word >> 20) & 15, word & 1023
            assert word == (opcode << 24) | 0x40000 | (register << 20) | address
            if opcode == 0x70:
                registers[register] = dst[address]
            else:
                dst[address] = registers[register]
                observations.append((address, dst[address]))
        else:
            assert word == 0x8F000000, f"unexpected issued word {word:#x}"

    for word in stream:
        if recording is not None:
            start, remaining, execute = recording
            assert word >> 24 != 0x04, "nested replay recording"
            slots[start] = word
            if execute:
                issue(word)
            recording = (start + 1, remaining - 1, execute) if remaining > 1 else None
        elif word >> 24 == 0x04:
            load, execute = word & 1, (word >> 1) & 7
            length, start = (word >> 4) & 1023, (word >> 14) & 1023
            assert length == 2 and start == 5 and execute in (0, 1)
            if load:
                recording = (start, length, execute)
            else:
                assert execute == 0
                for index in range(start, start + length):
                    assert index in slots, "executing an unrecorded slot"
                    issue(slots[index])
        else:
            issue(word)
    assert recording is None, "truncated recording"
    return observations


assembly, mir = (pathlib.Path(path).read_text() for path in sys.argv[1:])
for name, mode in (("structured_dst_record_only", 0),
                   ("structured_dst_record_and_execute", 1)):
    stream = words(assembly, name)
    useful = [word for word in stream if word != 0x8F000000]
    assert useful == [0x7C000A02, 0x7C000212, 0x72140020,
                      0x04014023 if mode else 0x04014021,
                      0x70040020, 0x72040028, 0x72040000,
                      0x70340028, 0x72340008, 0x7C000542,
                      0x72440020, 0x04014020, 0x72040010,
                      0x70340028, 0x72340018], (name, "authored words or replay mode changed")
    block = re.search(r"^name:\s*" + name + r"\s*$.*?^\.\.\.", mir, re.M | re.S)
    assert block, (name, "missing machine function")
    assert not re.search(r"(?:class:\s*sfpr|:sfpr\b)", block[0]), "virtual SFPR introduced"
    lines = block[0].splitlines()
    headers = [(number, match, line) for number, line in enumerate(lines)
               if (match := re.search(r"\b\w+ ([01]), ([01]), 2, 5,", line))]
    assert len(headers) == 2, (name, "changed static replay count")
    record, execute = headers
    assert tuple(map(int, record[1].groups())) == (1, mode)
    assert tuple(map(int, execute[1].groups())) == (0, 0)
    assert effects(record[2]) == (collections.Counter(["tt_config", "tt_issue"]),
                                  collections.Counter(["tt_issue"]))
    assert effects(execute[2]) == (
        collections.Counter(["tt_cc", "tt_config", "tt_dst", "tt_issue", "tt_l0"]),
        collections.Counter(["tt_dst", "tt_issue", "tt_l0"])), "execute lost actual Dst/old-value effects"
    for header in (record, execute):
        assert "::" in header[2] and "volatile" in header[2]
        assert "load" in header[2] and "store" in header[2]
    end = next(index for index in range(record[0] + 1, len(lines))
               if "PseudoTTReplayRecordEnd" in lines[index])
    captured = [line for line in lines[record[0] + 1:end] if line.strip()]
    assert len(captured) == 2, "recorded instruction count changed"
    if not mode:
        for line in captured:
            assert effects(line) == (collections.Counter(["tt_issue"]),
                                      collections.Counter(["tt_issue"])), "record-only changed Dst or LReg"
    else:
        assert "TTSFPLOAD" in captured[0] and "TTSFPSTORE" in captured[1]
    for seed in (37, 911):
        before = tuple(seed + 200 + lane for lane in range(32))
        after = tuple(seed + 500 + lane for lane in range(32))
        old_dst = tuple(seed + 14000 + lane for lane in range(32))
        want = [(32, before)] + ([(40, before)] if mode else [])
        want += [(0, before if mode else (0x3F800000,) * 32),
                 (8, before if mode else old_dst), (32, after),
                 (40, after), (16, after), (24, after)]
        assert observe(stream, seed) == want, (name, seed)
