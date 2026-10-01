"""Small emitted-word oracle; software evidence, not a hardware simulator.

Only REPLAY, MOVAll, raw32 SFPSTORE and SFPNOP are received. Stores observe
their input register under the test's all-enabled, neutral-address premise;
Dst packing, execution timing, and arbitrary scalar instructions are not
simulated. Exact static words plus FileCheck backedges cover the real loop.
"""

import collections
import pathlib
import re
import sys


def words(assembly, name):
    body = re.search(r"^" + name + r":.*?^\.Lfunc_end\d+:", assembly, re.M | re.S)
    assert body, f"missing assembly function {name}"
    encoded = [int(word, 16) for word in re.findall(r"\.word\s+(0x[0-9a-fA-F]+)", body[0])]
    # Direct-issue words rotate the architectural word left by two bits.
    return [((word >> 2) | (word << 30)) & 0xFFFFFFFF for word in encoded]


def observe(stream, seed):
    registers = {reg: tuple(seed + 100 * reg + lane for lane in range(32)) for reg in range(16)}
    registers[9] = (0,) * 32
    registers[10] = (0x3F800000,) * 32
    slots, observations = {}, []
    recording = None

    def issue(word):
        opcode = word >> 24
        if opcode == 0x7C:
            assert word & 15 == 2, "only authored MOVAll is received"
            destination, source = (word >> 4) & 15, (word >> 8) & 15
            assert word == 0x7C000002 | (source << 8) | (destination << 4)
            registers[destination] = registers[source]
        elif opcode == 0x72:
            source, address = (word >> 20) & 15, word & 1023
            assert word == 0x72040000 | (source << 20) | address
            observations.append((address, registers[source]))
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
            assert length == 1 and start == 5 and execute in (0, 1)
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


def effects(line):
    uses, definitions = [], []
    if " = " in line:
        lhs, line = line.split(" = ", 1)
        definitions.extend(re.findall(r"\$(tt_\w+)", lhs))
    for operand in line.split(","):
        registers = re.findall(r"\$(tt_\w+)", operand)
        (definitions if "implicit-def" in operand else uses).extend(registers)
    return collections.Counter(uses), collections.Counter(definitions)


def check_machine_effects(mir, name, execute_while_loading):
    block = re.search(r"^name:\s*" + name + r"\s*$.*?^\.\.\.", mir, re.M | re.S)
    assert block, f"missing machine function {name}"
    body = block[0]
    assert not re.search(r"(?:class:\s*sfpr|:sfpr\b)", body), "virtual SFPR introduced"
    headers = []
    for number, line in enumerate(body.splitlines()):
        # Match the public field tuple, not an unfrozen internal opcode name.
        match = re.search(r"\b\w+ ([01]), ([01]), 1, 5,", line)
        if match:
            headers.append((number, int(match[1]), int(match[2]), line))
    assert len(headers) == 2, f"changed static replay count in {name}"
    record, execute = headers
    assert record[1:3] == (1, execute_while_loading) and execute[1:3] == (0, 0)
    assert effects(record[3]) == (
        collections.Counter(["tt_config", "tt_issue"]), collections.Counter(["tt_issue"])
    ), "record header acquired numerical effects"
    assert effects(execute[3]) == (
        collections.Counter(["tt_config", "tt_issue", "tt_l1"]),
        collections.Counter(["tt_issue", "tt_l0"]),
    ), "execute lost its actual L1 read or L0 write, or invented an old/CC input"
    for header in (record, execute):
        assert "::" in header[3] and "volatile" in header[3]
        assert "load" in header[3] and "store" in header[3], "replay lost its memory barrier"
    lines = body.splitlines()
    end = next(index for index in range(record[0] + 1, len(lines)) if "PseudoTTReplayRecordEnd" in lines[index])
    captured = [line for line in lines[record[0] + 1:end] if line.strip()]
    assert len(captured) == 1, "recorded instruction count changed"
    if execute_while_loading:
        assert effects(captured[0]) == (
            collections.Counter(["tt_l1", "tt_config", "tt_issue"]),
            collections.Counter(["tt_l0", "tt_issue"]),
        )
    else:
        assert effects(captured[0]) == (
            collections.Counter(["tt_issue"]), collections.Counter(["tt_issue"])
        ), "record-only body falsely executes its numerical effects"


assembly, mir = (pathlib.Path(path).read_text() for path in sys.argv[1:3])
for name in sys.argv[3:]:
    stream = words(assembly, name)
    useful = [word for word in stream if word != 0x8F000000]
    if name == "replay_runtime_loop":
        assert useful == [0x7C000A02, 0x7C000912, 0x04014011, 0x7C000102,
                          0x04014010, 0x7C000A12, 0x72040008], "loop body was rewritten or duplicated"
        check_machine_effects(mir, name, 0)
        # This checks one dynamic iteration's actual word order; FileCheck
        # separately requires the scalar runtime backedge in emitted code.
        for seed in (37, 911):
            assert observe(stream, seed) == [(8, (0,) * 32)]
    else:
        executing = name == "record_and_execute"
        assert name in ("record_and_execute", "record_only")
        expected_prefix = [0x7C000A02, 0x7C000212]
        assert useful == expected_prefix + [0x04014013 if executing else 0x04014011,
                                            0x7C000102, 0x72040000, 0x7C000912,
                                            0x04014010, 0x72040008], "authored words or replay mode changed"
        check_machine_effects(mir, name, int(executing))
        for seed in (37, 911):
            before = tuple(seed + 200 + lane for lane in range(32)) if executing else (0x3F800000,) * 32
            assert observe(stream, seed) == [(0, before), (8, (0,) * 32)], name
