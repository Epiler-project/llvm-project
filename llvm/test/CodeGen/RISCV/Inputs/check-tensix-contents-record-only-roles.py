#!/usr/bin/env python3
"""Check emitted role-indexed Replay words and actual machine effects.

This finite fixture oracle checks backend output, not device execution. Reuse
existing word/CFG readers; no compiler legality or source-loop expansion lives
here. Real numerical hardware reception remains a separate controller gate.
"""
import argparse
import collections
import importlib.util
from pathlib import Path
import re


def module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem.replace('-', '_'), path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


base = module('check-tensix-contents-record-only.py')
require = base.require


def check_mir(mir, name, mutables, captures, stores):
    lines = base.body(mir, name)
    records, executes, ends, seeds = [], [], [], {}
    allowed = {'TTSFPENCC', 'TTSFPCONFIGReset', 'TTSFPMOVAll',
               'TTSFPSTORE', 'PseudoTTContentsSFPUReplay',
               'PseudoTTSFPURecordWord', 'PseudoTTReplayRecordEnd'}
    for index, line in enumerate(lines):
        match = re.search(r'\bPseudoTTContentsSFPUReplay ([01]), ([01]), (\d+), (\d+),', line)
        if match:
            ((records if match[1] == '1' else executes)
             .append((index, tuple(map(int, match.groups())), line)))
        if 'PseudoTTReplayRecordEnd' in line:
            ends.append(index)
        seed = re.search(r'(\$tt_l[0-7]) = TTSFPMOVAll \$tt_c(9|10|15)\b', line)
        if seed:
            role = int(seed[2])
            require(role not in seeds, 'duplicate constant snapshot/repair')
            seeds[role] = seed[1][1:]
        opcode = line.strip().split(' = ', 1)[-1].split()
        if opcode and opcode[0].startswith(('TT', 'PseudoTT')):
            require(opcode[0] in allowed, 'hidden accelerator instruction/transport')
    require(len(records) == len(executes) == len(ends) == 1,
            'record/execute topology changed in ' + name)
    at, control, header = records[0]
    require(control[0:2] == (1, 0) and 0 < control[2] <= 32,
            'record-only mode or capacity changed')
    require(base.effects(header) == (collections.Counter(['tt_config', 'tt_issue']),
                                     collections.Counter(['tt_issue'])),
            'recording acquired numerical effects')
    payload = [line for line in lines[at + 1:ends[0]] if line.strip()]
    require(len(payload) == control[2], 'recorded words missing or reordered')
    for line in payload:
        require('PseudoTTSFPURecordWord ' in line and '::' not in line and
                base.effects(line) == (collections.Counter(['tt_issue']),
                                       collections.Counter(['tt_issue'])),
                'recorded payload executes numerical work')
    require(set(seeds) == set(mutables + captures) and
            len(set(seeds.values())) == len(seeds),
            'independent allocation roles collapsed')
    index, fields, execution = executes[0]
    require(index > ends[0] and fields == (0, 0, control[2], control[3]),
            'execute does not retain its record')
    require(base.effects(execution) ==
            (collections.Counter(['tt_config', 'tt_issue', 'tt_cc'] +
                                 [seeds[key] for key in mutables + captures]),
             collections.Counter(['tt_issue'] + [seeds[key] for key in mutables])),
            'execute lost old/capture/result effects')
    require('volatile load store' in execution, 'execute lost issue ordering')
    require(sum('TTSFPSTORE ' in line for line in lines) == stores,
            'extra SFPR store/spill')
    return lines, at, index


def check_stream(assembly, name, seed_roles, recipe, expected_stores):
    constants = {9: tuple(0x125701 + 17 * lane for lane in range(32)),
                 10: tuple(0x443311 + 29 * lane for lane in range(32)),
                 15: tuple(0x213405 + 41 * lane for lane in range(32))}
    registers, physical, slots, observed, recorded = dict(constants), {}, {}, [], []
    recording = None
    records = executes = 0
    mask = 0xffffffff

    def arithmetic(word):
        opcode, destination, source = word >> 24, (word >> 4) & 15, (word >> 8) & 15
        require(opcode in (0x79, 0x7e, 0x7f, 0x8d, 0x80), 'unexpected arithmetic')
        require(word == ((opcode << 24) | (source << 8) | (destination << 4) |
                         (4 if opcode == 0x79 else 0)), 'arithmetic fields changed')
        operation = {0x79: lambda a, b: (a + b) & mask,
                     0x7e: lambda a, b: a & b,
                     0x7f: lambda a, b: a | b,
                     0x8d: lambda a, b: a ^ b,
                     0x80: lambda a, b: (~b) & mask}[opcode]
        registers[destination] = tuple(operation(a, b) for a, b in
                                       zip(registers[destination], registers[source]))

    for word in base.words(assembly, name):
        opcode = word >> 24
        if recording:
            slot, remaining = recording
            slots[slot] = word
            if word != 0x8f000000:
                require(len(physical) == len(seed_roles), 'unallocated recorded role')
                require(len(recorded) < len(recipe), 'extra recorded arithmetic')
                role_opcode, destination, source = recipe[len(recorded)]
                require(word == ((role_opcode << 24) |
                                 (physical[seed_roles[source]] << 8) |
                                 (physical[seed_roles[destination]] << 4) |
                                 (4 if role_opcode == 0x79 else 0)),
                        'recorded word lost its ordered mutable/capture roles')
                recorded.append(word)
            recording = (slot + 1, remaining - 1) if remaining > 1 else None
        elif opcode == 0x04:
            load, execute = word & 1, (word >> 1) & 7
            length, start = (word >> 4) & 1023, (word >> 14) & 1023
            require(execute == 0 and start == 0 and 0 < length <= 32,
                    'record-only controls changed')
            require(word == (0x04000000 | load | (length << 4)),
                    'noncanonical Replay fields')
            if load:
                records += 1
                recording = (start, length)
            else:
                executes += 1
                for slot in range(start, start + length):
                    item = slots[slot]
                    if item != 0x8f000000:
                        arithmetic(item)
        elif opcode == 0x7c:
            destination, source = (word >> 4) & 15, (word >> 8) & 15
            require(source in seed_roles and source not in physical and destination < 8,
                    'unexpected constant snapshot/repair copy')
            require(word == (0x7c000002 | (source << 8) | (destination << 4)),
                    'constant snapshot fields changed')
            physical[source] = destination
            registers[destination] = constants[source]
        elif opcode == 0x72:
            source, address = (word >> 20) & 15, word & 1023
            require(word == (0x72040000 | (source << 20) | address),
                    'store mode/format changed')
            observed.append((address, registers[source]))
        else:
            require(word in (0x8f000000, 0x8a00300a, 0x910000f1),
                    'unexpected native issue ' + hex(word))
    require(recording is None and records == executes == 1 and
            len(recorded) == len(recipe), 'record/execute count changed')
    require(observed == expected_stores(constants),
            'recording changed old contents or execute lost role ordering')


def check(assembly, mir):
    check_mir(mir, 'record_two_mutable', [9, 10], [15], 4)
    check_stream(assembly, 'record_two_mutable', [9, 10, 15],
                 [(0x79, 0, 2), (0x8d, 1, 0), (0x80, 1, 1)],
                 lambda c: [(0, c[9]), (8, c[10]),
                            (16, tuple((a + b) & 0xffffffff for a, b in zip(c[9], c[15]))),
                            (24, tuple((~(a ^ ((b + d) & 0xffffffff))) & 0xffffffff
                                       for a, b, d in zip(c[10], c[9], c[15])))])
    check_mir(mir, 'record_two_capture', [9], [10, 15], 2)
    check_stream(assembly, 'record_two_capture', [9, 10, 15],
                 [(0x7e, 0, 2), (0x7f, 0, 1)],
                 lambda c: [(0, c[9]), (8, tuple((a & b) | d for a, b, d in
                                                zip(c[9], c[15], c[10])))])
    check_mir(mir, 'record_no_capture', [9, 10], [], 2)
    check_stream(assembly, 'record_no_capture', [9, 10],
                 [(0x80, 1, 1), (0x8d, 0, 1)],
                 lambda c: [(0, tuple(a ^ ((~b) & 0xffffffff) for a, b in zip(c[9], c[10]))),
                            (8, tuple((~b) & 0xffffffff for b in c[10]))])
    lines, record_at, execute_at = check_mir(mir, 'record_two_mutable_loop', [9, 10], [15], 2)
    cfg = module('check-tensix-contents-replay-phi.py')
    (blocks, owners, instructions, graph, conditional, reaches, reachable,
     dominators, before, after) = cfg.machine_cfg(lines)
    record_block, execute_block = owners[record_at], owners[execute_at]
    require(record_block != execute_block and before(record_at, execute_at),
            'loop record does not dominate execute')
    require(not any(reaches(successor, record_block) for successor in graph[record_block]),
            'record moved into dynamic cycle')
    require(any(reaches(successor, execute_block) for successor in graph[execute_block]),
            'actual loop backedge lost')
    exits = {bid for bid in reachable if not graph[bid]}
    require(any(reaches(next(iter(blocks)), end, {execute_block}) for end in exits),
            'zero-trip path was lost')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly', type=Path, required=True)
    parser.add_argument('--mir', type=Path, required=True)
    args = parser.parse_args()
    check(args.assembly.read_text(), args.mir.read_text())
