"""Bounded emitted-code observer, independent of the LLVM replay representation.

Implements only scalar RV32 ops used by these fixtures, local port stores,
raw32 SFPLOAD/STORE, all-lane MOV, SFPNOP and REPLAY. This is software codegen
evidence, not general ISA emulation, timing, conversions or hardware acceptance.
"""
import collections
import pathlib
import re
import sys

PORT = 1048128 << 12
MASK = (1 << 32) - 1


def function(assembly, name):
    body = re.search(r"^" + name + r":.*?^\.Lfunc_end\d+:", assembly, re.M | re.S)
    assert body, (name, "missing function")
    instructions, labels = [], {}
    for line in body[0].splitlines()[1:]:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith(":"):
            labels[line[:-1]] = len(instructions)
        elif line.startswith(".word") or not line.startswith("."):
            instructions.append(line)
    return instructions, labels


def observe(assembly, mode, read_offset, write_offset, iterations, epochs=None):
    name = ("dynamic_dst_record_" + ("and_execute" if mode else "only")
            if epochs is None else
            ("conditional_record" if epochs else "rerecord") + f"_mode{mode}")
    instructions, labels = function(assembly, name)
    gpr = collections.defaultdict(int, a0=read_offset, a1=write_offset,
                                  a2=iterations, sp=0x100000)
    if epochs is not None:
        gpr["a1"] = iterations
    memory, dst, slots = {}, {}, {}
    lreg = {i: 0x10000000 + i for i in range(16)}
    lreg[10] = 0x3F800000
    initial = 0x42000000 + read_offset
    dst[read_offset] = initial
    dst[write_offset] = 0xDEADBEEF
    recording = None
    captures, captured_gprs, writes = [], set(), []
    executions, records = 0, 0
    epoch_observations = []
    pc, steps = 0, 0
    before_execute = None

    def get(reg):
        return 0 if reg in ("zero", "x0") else gpr[reg]

    def put(reg, value):
        if reg not in ("zero", "x0"):
            gpr[reg] = value & MASK

    def epoch_value(offset, ordinal):
        return 0x43000000 + 1024 * ordinal + offset

    def issue(word):
        nonlocal before_execute
        opcode = word >> 24
        if opcode == 0x7C:
            destination, source = (word >> 4) & 15, (word >> 8) & 15
            assert word == 0x7C000002 | (source << 8) | (destination << 4)
            lreg[destination] = lreg[source]
        elif opcode in (0x70, 0x72):
            reg, offset = (word >> 20) & 15, word & 1023
            assert word == (opcode << 24) | 0x40000 | (reg << 20) | offset
            if opcode == 0x70:
                if epochs is not None:
                    epoch_observations.append(offset)
                    dst[offset] = epoch_value(offset, len(epoch_observations))
                lreg[reg] = dst[offset]
            else:
                dst[offset] = lreg[reg]
                writes.append((offset, lreg[reg]))
                if offset == 0:
                    before_execute = lreg[reg]
                    dst[read_offset] = initial + 17
        else:
            assert word == 0x8F000000, f"unexpected Tensix word {word:#x}"

    def incoming(word):
        nonlocal recording, executions, records
        if recording:
            start, remaining, execute = recording
            assert word >> 24 != 4, "nested replay"
            slots[start] = word
            captures.append(word)
            if execute:
                issue(word)
            recording = (start + 1, remaining - 1, execute) if remaining > 1 else None
            if recording is None and epochs is None:
                # Captured GPRs are dead after the final port store, and execute
                # must never read them. Change them before later scalar work.
                for reg in captured_gprs:
                    put(reg, 999)
        elif word >> 24 == 4:
            load, execute = word & 1, (word >> 1) & 7
            length, start = (word >> 4) & 1023, (word >> 14) & 1023
            assert length == 2 and start == 5, "scalar instructions counted as words"
            if load:
                assert execute == mode
                records += 1
                recording = (start, length, execute)
            else:
                assert execute == 0
                executions += 1
                if epochs is None:
                    dst[read_offset] = initial + 17 * executions
                for index in range(start, start + length):
                    issue(slots[index])
        else:
            issue(word)

    while pc < len(instructions):
        steps += 1
        assert steps < 1000, "loop failed to terminate"
        line = instructions[pc]
        pc += 1
        op, _, operand_text = line.partition("\t")
        if not operand_text:
            op, _, operand_text = line.partition(" ")
        args = [x.strip() for x in operand_text.split(",")]
        if op == ".word":
            encoded = int(args[0], 0)
            incoming(((encoded >> 2) | (encoded << 30)) & MASK)
        elif op == "ret":
            break
        elif op in ("li", "lui"):
            put(args[0], int(args[1], 0) << (12 if op == "lui" else 0))
        elif op == "mv":
            put(args[0], get(args[1]))
        elif op in ("addi", "andi", "ori", "slli", "srli"):
            a, b = get(args[1]), int(args[2], 0)
            result = {"addi": lambda: a + b, "andi": lambda: a & b,
                      "ori": lambda: a | b, "slli": lambda: a << b,
                      "srli": lambda: a >> b}[op]()
            put(args[0], result)
        elif op in ("add", "sub", "or", "and"):
            a, b = get(args[1]), get(args[2])
            if op == "or" and recording:
                captured_gprs.add(args[2])
            put(args[0], {"add": lambda: a + b, "sub": lambda: a - b,
                          "or": lambda: a | b, "and": lambda: a & b}[op]())
        elif op in ("lw", "sw"):
            match = re.fullmatch(r"(-?\d+)\((\w+)\)", args[1])
            assert match, line
            address = (get(match[2]) + int(match[1])) & MASK
            if op == "lw":
                put(args[0], memory[address])
            elif address == PORT:
                incoming(get(args[0]))
            else:
                memory[address] = get(args[0])
        elif op == "j":
            pc = labels[args[0]]
        elif op in ("beqz", "bnez"):
            if (get(args[0]) == 0) == (op == "beqz"):
                pc = labels[args[1]]
        elif op in ("beq", "bne", "bltu", "bgeu"):
            a, b = get(args[0]), get(args[1])
            taken = {"beq": a == b, "bne": a != b,
                     "bltu": a < b, "bgeu": a >= b}[op]
            if taken:
                pc = labels[args[2]]
        else:
            raise AssertionError(f"unmodeled scalar instruction: {line}")
    assert executions == iterations and recording is None
    if epochs is None:
        assert records == 1
        assert captures == [0x70040000 | read_offset, 0x72040000 | write_offset]
        assert before_execute == (initial if mode else 0x3F800000)
        expected = initial + 17 * iterations if iterations else before_execute
        assert dst[8] == expected, (mode, read_offset, iterations, dst[8], expected)
        results = [value for offset, value in writes if offset == write_offset]
        assert results == ([initial] if mode else []) + [
            initial + 17 * i for i in range(1, iterations + 1)]
    else:
        record_iterations = [i for i in range(iterations)
                             if not epochs or i == 0 or i % 2]
        assert records == len(record_iterations)
        expected_captures = []
        expected_observations = []
        actual_offset = None
        for iteration in range(iterations):
            if iteration in record_iterations:
                actual_offset = (read_offset + 8 * iteration) & 1023
                expected_captures += [0x70040000 | actual_offset, 0x72040100]
                if mode:
                    expected_observations.append(actual_offset)
            expected_observations.append(actual_offset)
        assert captures == expected_captures, (name, captures, expected_captures)
        assert epoch_observations == expected_observations
        assert writes == [(256, epoch_value(offset, ordinal + 1))
                          for ordinal, offset in enumerate(expected_observations)]
    # Both instructions must remain single static sites regardless of source
    # iteration count. The runtime trace above separately verifies exact counts.
    def static_words():
        for line in instructions:
            if line.startswith(".word"):
                encoded = int(line.split()[-1], 0)
                yield ((encoded >> 2) | (encoded << 30)) & MASK
    replay_words = [word for word in static_words() if word >> 24 == 4]
    assert len(replay_words) == 2, "record or execute loop was duplicated"
    assert replay_words.count(0x04014020) == 1


def machine_effects(text, mode):
    assert not re.search(r"(?:class:\s*sfpr|:sfpr\b)", text)
    executes = [line for line in text.splitlines()
                if "PseudoTTReplayTemplateExecute 701," in line or
                "PseudoTTExplicitSFPUReplay 0, 0, 2, 5," in line]
    assert len(executes) == 1
    line = executes[0]
    assert not re.search(r"\$(?:x\d+|a\d+|t\d+|s\d+)|%\d+", line)
    for reg in ("tt_l0", "tt_cc", "tt_config", "tt_dst", "tt_issue"):
        assert re.search(r"implicit(?: killed)? \$" + reg + r"\b", line), (reg, line)
    for reg in ("tt_l0", "tt_dst", "tt_issue"):
        assert re.search(r"implicit-def(?: dead)? \$" + reg + r"\b", line), (reg, line)
    payloads = [line for line in text.splitlines()
                if "PseudoTTReplayTemplateDstWord" in line or
                "PseudoTTSFPUDstRecordWord" in line]
    if mode:
        assert not payloads
    else:
        assert len(payloads) == 2
        for line in payloads:
            assert "volatile store (s32)" in line, line
            assert line.count("early-clobber") == 2, line
            assert not any(reg in line for reg in ("$tt_l", "$tt_dst", "$tt_cc", "$tt_config"))
            assert re.search(r", (?:killed )?(?:%\d+(?::gpr)?|\$x\d+), 0, 4,", line), line


if __name__ == "__main__":
    assembly, prepared, final = [pathlib.Path(p).read_text() for p in sys.argv[1:4]]
    mode = int(sys.argv[4])
    machine_effects(prepared, mode)
    machine_effects(final, mode)
    for input_offset, output_offset in ((32, 40), (64, 72)):
        for count in (0, 1, 3):
            observe(assembly, mode, input_offset, output_offset, count)
