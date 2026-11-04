#!/usr/bin/env python3
"""Check nonexecuting Replay words and actual allocated execution effects.

This is a small emitted-word oracle for the explicit test profiles, not a
hardware simulator. It checks exact integer/bitwise roles and every real
store in straight-line functions. The loop check uses concrete machine CFG
and leaves numerical device execution to the controller's hardware gate.
"""
import argparse
import collections
import importlib.util
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def words(assembly, name):
    match = re.search(r'^' + re.escape(name) + r':.*?^\.Lfunc_end\d+:',
                      assembly, re.M | re.S)
    require(match, 'missing assembly function ' + name)
    require(not re.search(r'\b(?:call|jal|jalr|tail)\b', match[0]),
            'unexpected helper call')
    encoded = [int(value, 16) for value in
               re.findall(r'\.word\s+(0x[0-9a-fA-F]+)', match[0])]
    return [((word >> 2) | (word << 30)) & 0xffffffff for word in encoded]


def effects(line):
    uses, definitions = [], []
    if ' = ' in line:
        lhs, line = line.split(' = ', 1)
        definitions.extend(re.findall(r'\$(tt_\w+)', lhs))
    for operand in line.split(','):
        registers = re.findall(r'\$(tt_\w+)', operand)
        (definitions if 'implicit-def' in operand else uses).extend(registers)
    return collections.Counter(uses), collections.Counter(definitions)


def body(mir, name):
    match = re.search(r'^name:\s*' + re.escape(name) + r'\s*$.*?^body:\s*\|\s*\n'
                      r'(.*?)(?=^\.\.\.|\Z)', mir, re.M | re.S)
    require(match, 'missing machine function ' + name)
    require(not re.search(r'(?:class:\s*sfpr|:sfpr\b)', match[0]),
            'virtual SFPR survived allocation')
    require(not re.search(r'\b(?:SPILL|RELOAD|PseudoCALL|JALR)\b|\$tt_l[0-7]\s*=\s*(?:COPY|REG_SEQUENCE)\b|\bCOPY\s+\$tt_l[0-7]', match[1]),
            'hidden repair copy/spill/helper introduced')
    return match[1].splitlines()


def check_mir(mir, name, expected_executes):
    lines = body(mir, name)
    records, executes = [], []
    for index, line in enumerate(lines):
        match = re.search(r'\bPseudoTTContentsSFPUReplay ([01]), ([01]), (\d+), (\d+),', line)
        if not match:
            continue
        values = tuple(map(int, match.groups()))
        (records if values[0] else executes).append((index, values, line))
    require(len(records) == 1 and len(executes) == expected_executes,
            'record or execute count changed in ' + name)
    at, record, header = records[0]
    require(record[0:2] == (1, 0), 'recording executed numerical work')
    require(0 < record[2] <= 32 and record[3] == 0,
            'record capacity/automatic placement changed')
    require(effects(header) == (collections.Counter(['tt_config', 'tt_issue']),
                                collections.Counter(['tt_issue'])),
            'record header acquired numerical effects')
    ends = [index for index, line in enumerate(lines) if 'PseudoTTReplayRecordEnd' in line]
    require(len(ends) == 1 and at < ends[0], 'record end missing or reordered')
    captured = [line for line in lines[at + 1:ends[0]] if line.strip()]
    payload = [line for line in captured if 'PseudoTTSFPURecordWord ' in line]
    require(len(payload) == record[2], 'recorded instruction count changed')
    for line in captured:
        if 'PseudoTTSFPURecordWord ' not in line:
            require(not re.search(r'\$tt_|\b(?:TT\w+|PseudoTT\w+|BEQ|BNE|BLT|BGE|BLTU|BGEU|PseudoBR|PseudoRET|CALL|PseudoCALL|JALR|INLINEASM)\b|::|^\s*bb\.', line),
                    'record-only body contains an executing issue or control/memory effect')
    for line in payload:
        require(effects(line) == (collections.Counter(['tt_issue']),
                                  collections.Counter(['tt_issue'])),
                'record-only payload acquired numerical/CC/Dst effects')
        require('::' not in line, 'record-only payload acquired memory effects')
    seeds = {}
    for index, line in enumerate(lines):
        match = re.search(r'(\$tt_l[0-7]) = TTSFPMOVAll \$tt_c(9|15)\b', line)
        if match:
            require(match[2] not in seeds, 'extra seed/repair issue')
            require(index < at, 'seed moved into/after recording')
            seeds[match[2]] = match[1][1:]
    require(set(seeds) == {'9', '15'} and seeds['9'] != seeds['15'],
            'lost distinct mutable/capture allocation roles')
    allowed = {'TTSFPENCC', 'TTSFPCONFIGReset', 'TTSFPMOVAll',
               'TTSFPSTORE', 'PseudoTTContentsSFPUReplay',
               'PseudoTTSFPURecordWord', 'PseudoTTReplayRecordEnd'}
    for line in lines:
        candidate = line.strip().split(' = ', 1)[-1].split()
        if not candidate:
            continue
        opcode = candidate[0]
        if opcode.startswith(('TT', 'PseudoTT')):
            require(opcode in allowed, 'hidden accelerator transport/spill/reload')
    stores = [line for line in lines if re.search(r'\bTTSFPSTORE\b', line)]
    expected_stores = 1 if name == 'record_only_loop' else expected_executes + 1
    require(len(stores) == expected_stores, 'extra accelerator store/spill')
    for index, fields, line in executes:
        require(index > ends[0] and fields == (0, 0, record[2], record[3]),
                'execute lost its actual recording')
        require(effects(line) ==
                (collections.Counter(['tt_config', 'tt_issue', 'tt_cc',
                                      seeds['9'], seeds['15']]),
                 collections.Counter(['tt_issue', seeds['9']])),
                'execute lost its exact old/capture/result effects')
        require('volatile load store' in line, 'execute lost issue memory barrier')
    return lines, records[0][0], executes


def check_stream(assembly, name, recipe, execute_count):
    stream = words(assembly, name)
    registers = {9: (0,) * 32, 15: tuple(0x1020304 + 17 * lane for lane in range(32))}
    initial, capture = registers[9], registers[15]
    slots, observations = {}, []
    recording = None
    records = 0
    executes = 0
    mutable_reg = capture_reg = None
    issued_recipe = []

    def arithmetic_roles(word):
        opcode = word >> 24
        destination, source = (word >> 4) & 15, (word >> 8) & 15
        require(opcode in (0x79, 0x7e, 0x7f, 0x8d, 0x80),
                'unexpected recorded arithmetic operation')
        require(destination == mutable_reg and source ==
                (mutable_reg if opcode == 0x80 else capture_reg),
                'recorded operation lost the allocated roles')
        require(word == (opcode << 24) | (source << 8) |
                (destination << 4) | (4 if opcode == 0x79 else 0),
                'recorded opcode/mode/unused fields changed')
        return opcode, destination, source

    def issue(word):
        nonlocal mutable_reg, capture_reg
        opcode = word >> 24
        if opcode == 0x7c:
            destination, source = (word >> 4) & 15, (word >> 8) & 15
            require(word == 0x7c000002 | (source << 8) | (destination << 4),
                    'unsupported MOV/repair instruction')
            require(source in (9, 15) and destination < 8,
                    'MOV is not the authored constant snapshot')
            if source == 9:
                require(mutable_reg is None, 'duplicate mutable snapshot')
                mutable_reg = destination
            else:
                require(capture_reg is None, 'duplicate capture snapshot')
                capture_reg = destination
            registers[destination] = registers[source]
        elif opcode in (0x79, 0x7e, 0x7f, 0x8d, 0x80):
            opcode, destination, source = arithmetic_roles(word)
            old, rhs = registers[destination], registers[source]
            fn = {0x79: lambda a, b: (a + b) & 0xffffffff,
                  0x7e: lambda a, b: a & b,
                  0x7f: lambda a, b: a | b,
                  0x8d: lambda a, b: a ^ b,
                  0x80: lambda a, b: (~b) & 0xffffffff}[opcode]
            registers[destination] = tuple(fn(a, b) for a, b in zip(old, rhs))
        elif opcode == 0x72:
            source, address = (word >> 20) & 15, word & 1023
            require(word == 0x72040000 | (source << 20) | address,
                    'store changed address mode/format')
            require(source == mutable_reg, 'store lost mutable role')
            observations.append((address, registers[source]))
        else:
            require(word in (0x8f000000, 0x8a00300a, 0x910000f1),
                    'unexpected emitted native issue: ' + hex(word))

    for word in stream:
        if recording is not None:
            start, remaining = recording
            require(word >> 24 != 0x04, 'nested recording')
            slots[start] = word
            if word != 0x8f000000:
                arithmetic_roles(word)
                issued_recipe.append(word >> 24)
            recording = (start + 1, remaining - 1) if remaining > 1 else None
            continue
        if word >> 24 == 0x04:
            load, execute = word & 1, (word >> 1) & 7
            length, start = (word >> 4) & 1023, (word >> 14) & 1023
            require(word == 0x04000000 | load | (execute << 1) |
                    (length << 4) | (start << 14), 'noncanonical Replay')
            require(execute == 0 and start == 0 and 0 < length <= 32,
                    'record-only mode/placement changed')
            if load:
                require(records == 0, 'template was rerecorded')
                records += 1
                recording = (start, length)
            else:
                executes += 1
                for index in range(start, start + length):
                    require(index in slots, 'execute reads unrecorded slot')
                    issue(slots[index])
        else:
            issue(word)
    require(recording is None and records == 1 and executes == execute_count,
            'truncated recording or wrong real execute count')
    require(issued_recipe == recipe, 'recorded recipe lost/reordered an operation')
    expected = [(0, initial)]
    current = initial
    for step in range(execute_count):
        for opcode in recipe:
            fn = {0x79: lambda a, b: (a + b) & 0xffffffff,
                  0x7e: lambda a, b: a & b,
                  0x7f: lambda a, b: a | b,
                  0x8d: lambda a, b: a ^ b,
                  0x80: lambda a, b: (~a) & 0xffffffff}[opcode]
            current = tuple(fn(a, b) for a, b in zip(current, capture))
        expected.append((8 * (step + 1), current))
    require(observations == expected,
            'recording changed a numerical snapshot or execute result')


def check_loop(mir):
    lines, record_at, executes = check_mir(mir, 'record_only_loop', 1)
    helper = Path(__file__).with_name('check-tensix-contents-replay-phi.py')
    spec = importlib.util.spec_from_file_location('replay_cfg', helper)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    (blocks, owners, instructions, graph, conditional, reaches, reachable,
     dominators, before, after) = cfg.machine_cfg(lines)
    execute_at = executes[0][0]
    record_block, execute_block = owners[record_at], owners[execute_at]
    require(record_block != execute_block and before(record_at, execute_at),
            'record does not dominate loop execute')
    require(not any(reaches(successor, record_block) for successor in graph[record_block]),
            'record moved into a dynamic cycle')
    require(any(reaches(successor, execute_block) for successor in graph[execute_block]),
            'real execute loop backedge lost')
    exits = {bid for bid in reachable if not graph[bid]}
    require(any(reaches(next(iter(blocks)), end, {execute_block}) for end in exits),
            'zero-trip path was lost')


def check(assembly, mir):
    for name, recipe, count in (('record_only', [0x79], 2),
                                 ('record_all_recipes', [0x79, 0x7e, 0x7f, 0x8d, 0x80], 1),
                                 ('record_only_unused', [0x79], 0)):
        check_mir(mir, name, count)
        check_stream(assembly, name, recipe, count)
    check_loop(mir)


def check_mutations(assembly, mir):
    # Mutate actual passing output. Every targeted string must be present;
    # ineffective parser tests are failures rather than counted coverage.
    changes = []
    changes.append(('record executes', assembly.replace('0x10000044', '0x1000004c', 1), mir))
    changes.append(('hidden helper', assembly.replace('record_only:', 'record_only:\n  call __hidden', 1), mir))
    changes.append(('record MIR executes', assembly,
                    mir.replace('PseudoTTContentsSFPUReplay 1, 0, 1, 0,',
                                'PseudoTTContentsSFPUReplay 1, 1, 1, 0,', 1)))
    changes.append(('executing MIR body', assembly,
                    mir.replace('PseudoTTSFPURecordWord ', 'TTSFPIADD ', 1)))
    match = re.search(r'PseudoTTSFPURecordWord[^\n]*', mir)
    require(match, 'no payload available for mutation controls')
    for operand in ('implicit-def $tt_l0', 'implicit $tt_cc', 'implicit-def $tt_dst'):
        mutated = mir[:match.end()] + ', ' + operand + mir[match.end():]
        changes.append(('payload acquired ' + operand, assembly, mutated))
    for name, changed_assembly, changed_mir in changes:
        require(changed_assembly != assembly or changed_mir != mir,
                'mutation did not change input: ' + name)
        try:
            check(changed_assembly, changed_mir)
        except ValueError:
            continue
        raise ValueError('checker accepted mutation: ' + name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly', type=Path, required=True)
    parser.add_argument('--mir', type=Path, required=True)
    args = parser.parse_args()
    assembly, mir = args.assembly.read_text(), args.mir.read_text()
    check(assembly, mir)
    check_mutations(assembly, mir)
