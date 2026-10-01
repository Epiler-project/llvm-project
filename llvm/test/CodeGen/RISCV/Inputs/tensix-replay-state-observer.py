"""Independent bounded emitted-word observer for explicit state replay.

Receives ENCC, CONFIG C11..14/Lane/reset, MOV/move-all, raw32 neutral-address
STORE, NOP and direct local REPLAY. Initial physical contents are explicit test
premises. No timing, general scalar ISA, other Dst formats, MOP, stack, or
backdoor-load operation is simulated. This is software codegen evidence.
"""

import collections
import pathlib
import re
import sys


def emitted_words(assembly, name):
    body = re.search(r"^" + name + r":.*?^\.Lfunc_end\d+:", assembly, re.M | re.S)
    assert body, (name, "missing assembly function")
    rotated = [int(word, 16) for word in re.findall(r"\.word\s+(0x[0-9a-fA-F]+)", body[0])]
    return [((word >> 2) | (word << 30)) & 0xFFFFFFFF for word in rotated]


def effects(line):
    uses, definitions = [], []
    if " = " in line:
        lhs, line = line.split(" = ", 1)
        definitions.extend(re.findall(r"\$(tt_\w+)", lhs))
    for operand in line.split(","):
        registers = re.findall(r"\$(tt_\w+)", operand)
        (definitions if "implicit-def" in operand else uses).extend(registers)
    return collections.Counter(uses), collections.Counter(definitions)


def state_effects(kind):
    uses = ["tt_cc", "tt_config", "tt_issue"]
    definitions = ["tt_issue"]
    if kind == "encc":
        definitions += ["tt_cc"]
    elif kind == "creg":
        uses += ["tt_l0"] + [f"tt_c{reg}" for reg in range(11, 15)]
        definitions += ["tt_config"] + [f"tt_c{reg}" for reg in range(11, 15)]
    elif kind == "lane":
        uses += ["tt_l0"]
        definitions += ["tt_config"]
    else:
        assert kind == "reset"
        definitions += ["tt_config"]
    return collections.Counter(uses), collections.Counter(definitions)


def check_effects(mir, name, kind, mode):
    block = re.search(r"^name:\s*" + name + r"\s*$.*?^\.\.\.", mir, re.M | re.S)
    assert block, (name, "missing machine function")
    assert not re.search(r"(?:class:\s*sfpr|:sfpr\b)", block[0]), "virtual SFPR introduced"
    lines = block[0].splitlines()
    controls = [(index, match, line) for index, line in enumerate(lines)
                if (match := re.search(r"\bPseudoTTExplicitSFPUReplay ([01]), ([01]), ([0-9]+), 5,", line))]
    if kind == "encc":
        wants = [(1, mode, 1, "encc"), (0, 0, 1, "encc"), (0, 0, 1, "encc")]
    elif kind == "creg":
        wants = [(1, mode, 4, "creg"), (0, 0, 4, "creg")]
    else:
        assert kind == "lane"
        wants = [(1, mode, 1, "lane"), (0, 0, 1, "lane"),
                 (1, mode, 1, "reset"), (0, 0, 1, "reset")]
    assert len(controls) == len(wants), (name, "changed authored control count")
    for (index, match, line), (load, execute, length, operation) in zip(controls, wants):
        assert tuple(map(int, match.groups())) == (load, execute, length)
        assert ":: (volatile load store (s32))" in line, "lost issue memory boundary"
        if not load:
            assert effects(line) == state_effects(operation), (name, operation, line)
            continue
        assert effects(line) == (collections.Counter(["tt_config", "tt_issue"]),
                                  collections.Counter(["tt_issue"]))
        end = next(i for i in range(index + 1, len(lines)) if "PseudoTTReplayRecordEnd" in lines[i])
        captured = [body for body in lines[index + 1:end] if body.strip()]
        assert len(captured) == length, (name, "changed recorded issue count")
        for position, body in enumerate(captured):
            assert "::" not in body, "state payload acquired memory effects"
            if not mode:
                assert "PseudoTTSFPURecordWord" in body
                assert effects(body) == (collections.Counter(["tt_issue"]),
                                          collections.Counter(["tt_issue"]))
            elif operation == "creg":
                reg = 11 + position
                assert f"TTSFPCONFIGC{reg}" in body
                assert effects(body) == (
                    collections.Counter(["tt_l0", f"tt_c{reg}", "tt_cc", "tt_config", "tt_issue"]),
                    collections.Counter([f"tt_c{reg}", "tt_config", "tt_issue"]))
            else:
                assert effects(body) == state_effects(operation), (name, operation, body)


def initial_registers(seed, kind):
    registers = {reg: [seed + 100 * reg + lane for lane in range(32)] for reg in range(16)}
    registers[9] = [0] * 32
    registers[10] = [0x3F800000] * 32
    if kind == "lane":
        # First-row configuration values have no EnableDestIndex bit. Higher
        # source rows deliberately differ and must never become configuration.
        for column in range(8):
            registers[1][column] = (1 << (12 + ((column + seed) % 4))) | ((column % 2) << 4) | 0x10000
            registers[6][column] = (1 << (12 + ((column + seed + 1) % 4))) | (((column + 1) % 2) << 4) | 0x20000
    return registers


def observe(words, seed, kind):
    registers = initial_registers(seed, kind)
    enable = [True] * 32
    flags = [(lane + seed) % 3 != 0 for lane in range(32)]
    config = [0] * 32
    dst = {address: [seed + 10000 + 100 * address + lane for lane in range(32)]
           for address in range(0, 64, 8)}
    touched, slots = set(), {}
    recording = None

    def lane_enabled(lane):
        return not (config[lane & 7] & (1 << (12 + lane // 8))) and (not enable[lane] or flags[lane])

    def issue(word):
        opcode = word >> 24
        if opcode == 0x8A:
            value, mode = (word >> 12) & 3, word & 15
            assert word == 0x8A000000 | (value << 12) | mode
            assert mode in (0, 1, 2, 8, 9, 10)
            for lane in range(32):
                if mode & 2:
                    enable[lane] = bool(value & 1)
                elif mode & 1:
                    enable[lane] = not enable[lane]
                flags[lane] = bool(value & 2) if mode & 8 else True
        elif opcode == 0x91:
            mask, destination, mode = (word >> 8) & 0xFFFF, (word >> 4) & 15, word & 15
            assert destination in (11, 12, 13, 14, 15)
            assert mode in ((0, 8) if destination < 15 else (1, 8))
            if mode == 8:
                assert mask & 0xAAAA == 0, "bound CONFIG mask requires even bits"
            if mode == 1:
                assert mask == 0
            for column in range(8):
                if mode & 8 and not mask & (1 << (2 * column)):
                    continue
                # CONFIG uses first-row CC, not LaneEnabled/ROW_MASK.
                if enable[column] and not flags[column]:
                    continue
                for lane in range(column, 32, 8):
                    if destination < 15:
                        registers[destination][lane] = registers[0][column]
                    elif mode == 1:
                        config[lane] &= 0x30000
                    else:
                        config[lane] = registers[0][column] & 0x3FFFF
        elif opcode == 0x7C:
            destination, source, mode = (word >> 4) & 15, (word >> 8) & 15, word & 15
            assert destination < 8 and mode in (0, 2)
            assert word == 0x7C000000 | (source << 8) | (destination << 4) | mode
            for lane in range(32):
                if mode == 2 or lane_enabled(lane):
                    registers[destination][lane] = registers[source][lane]
        elif opcode == 0x72:
            source, address = (word >> 20) & 15, word & 1023
            assert source < 8 and address in dst
            assert word == 0x72040000 | (source << 20) | address
            touched.add(address)
            for lane in range(32):
                if not config[lane] & 16 and lane_enabled(lane):
                    dst[address][lane] = registers[source][lane]
        else:
            assert word == 0x8F000000, f"unexpected word {word:#x}"

    for word in words:
        if recording is not None:
            start, remaining, execute = recording
            assert word >> 24 != 0x04, "nested recording"
            slots[start] = word
            if execute:
                issue(word)
            recording = (start + 1, remaining - 1, execute) if remaining > 1 else None
        elif word >> 24 == 0x04:
            load, execute = word & 1, (word >> 1) & 7
            length, start = (word >> 4) & 1023, (word >> 14) & 1023
            assert start == 5 and length in (1, 4) and execute in (0, 1)
            if load:
                recording = (start, length, execute)
            else:
                assert execute == 0
                for slot in range(start, start + length):
                    assert slot in slots, "execution before recording"
                    issue(slots[slot])
        else:
            issue(word)
    assert recording is None
    return {address: tuple(dst[address]) for address in touched}, config


def expected(seed, kind, mode):
    # Direct source-level expectations; do not replay expected words or use
    # the observer's instruction functions to manufacture wanted results.
    if kind == "encc":
        first = (0x3F800000,) * 32 if mode else tuple(seed + 400 + lane for lane in range(32))
        return {0: first, 8: first, 16: (0,) * 32}
    if kind == "creg":
        wanted = {}
        for reg in range(11, 15):
            def selected(column):
                return reg == 11 or (reg == 12 and column != 0) or (reg == 13 and column == 0)

            before, after = [], []
            for lane in range(32):
                column = lane % 8
                old = seed + 100 * reg + lane
                changed = selected(column)
                before.append(seed + 100 + column if mode and changed and (column + seed) % 3 != 0 else old)
                after.append(seed + 600 + column if changed else old)
            wanted[(reg - 11) * 8] = tuple(before)
            wanted[32 + (reg - 11) * 8] = tuple(after)
        return wanted
    assert kind == "lane"
    before, after = [], []
    for column in range(8):
        before.append((1 << (12 + ((column + seed) % 4))) | ((column % 2) << 4)
                      if mode and column != 0 and (column + seed) % 3 != 0 else 0)
        after.append((1 << (12 + ((column + seed + 1) % 4))) | (((column + 1) % 2) << 4)
                     if column != 0 else 0)
    wanted = {}
    for address, columns, value in ((0, before, 0x3F800000), (8, after, 0),
                                     (16, [0] * 8 if mode else after, 0x3F800000),
                                     (24, [0] * 8, 0)):
        wanted[address] = tuple(
            seed + 10000 + 100 * address + lane
            if columns[lane % 8] & (16 | (1 << (12 + lane // 8))) else value
            for lane in range(32))
    return wanted


def main(paths):
    assembly, mir = (pathlib.Path(path).read_text() for path in paths)
    for kind in ("encc", "creg", "lane"):
        for route in ("structured", "raw"):
            for mode in (0, 1):
                name = f"state_{kind}_{route}_mode{mode}"
                words = emitted_words(assembly, name)
                counts = {"encc": {0x8A: 4, 0x7C: 7, 0x72: 3, 0x04: 3},
                          "creg": {0x8A: 1, 0x7C: 10, 0x72: 8, 0x04: 2, 0x91: 4},
                          "lane": {0x8A: 1, 0x7C: 6, 0x72: 4, 0x04: 4, 0x91: 2}}
                assert collections.Counter(word >> 24 for word in words if word != 0x8F000000) == counts[kind], (name, "added or deleted authored work")
                check_effects(mir, name, kind, mode)
                for seed in (37, 911):
                    actual, config = observe(words, seed, kind)
                    assert actual == expected(seed, kind, mode), (name, seed, actual)
                    if kind == "lane":
                        # Reset clears low configuration under first-row CC even
                        # when ROW_MASK had disabled the corresponding row.
                        assert config == [0 if lane % 8 == 0 else 0x20000 for lane in range(32)]


if __name__ == "__main__":
    main(sys.argv[1:])
