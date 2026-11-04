#!/usr/bin/env python3
"""Check actual MOP1 outer/Loop1Last setup, roles, native words and source CFG."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import re


def load_owner():
    path = Path(__file__).with_name('check-tensix-contents-mop.py')
    spec = importlib.util.spec_from_file_location('mop_owner', path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    return owner


owner = load_owner()
require = owner.require
PROFILES = [(2, 4), (2, 7), (3, 4), (3, 7), (2, 1), (3, 1), (127, 127), (1, 1)]


def check_function(assembly, mir, name, outer, inner, bind8=True):
    lines = owner.record.body(mir, name)
    records, mops, controls, bindings, clears = [], [], [], [], []
    for index, line in enumerate(lines):
        match = re.search(r'\bPseudoTTContentsSFPUReplay ([01]), ([01]), (\d+), (\d+),', line)
        if match:
            require(match.groups()[:2] == ('1', '0'), 'ordinary execute or recording executed arithmetic')
            records.append((index, tuple(map(int, match.groups()))))
        if re.search(r'\bTTMOP\b', line):
            require(re.search(r'\bTTMOP 0, 0, 1,', line), 'MOP fields or count override changed')
            mops.append((index, line))
        match = re.search(r'\bPseudoTTMOPControlWriteImm (\d+), (\d+),', line)
        if match:
            controls.append((index, tuple(map(int, match.groups()))))
        match = re.search(r'\bPseudoTTREPLAYMop (\d+), 0, 0, (\d+), (\d+),', line)
        if match:
            bindings.append((index, tuple(map(int, match.groups()))))
            require(owner.record.effects(line) == (Counter(['tt_config', 'tt_issue']),
                                                   Counter(['tt_config', 'tt_issue'])),
                    'MOP programming acquired numerical effects')
        match = re.search(r'\bPseudoTTMOPClear (\d+),', line)
        if match:
            clears.append((index, int(match[1])))
    require(len(records) == len(mops) == 1, 'source loop/record/MOP duplicated or lost')
    start, header = records[0]
    require(header[:2] == (1, 0) and 0 < header[2] <= 32 and header[3] == 0,
            'recording range/mode changed')
    slots = [5, 7, 8] if bind8 else [5, 7]
    require(sorted(value for _, value in controls) == [(0, outer), (1, inner)],
            'hardware outer/inner count changed')
    require(sorted(value for _, value in bindings) == [(slot, header[2], 0) for slot in slots],
            'LoopOp/Loop0Last/Loop1Last lost the actual recorded identity')
    require(sorted(value for _, value in clears) == ([2, 3, 4, 6] if bind8 else [2, 3, 4, 6, 8]),
            'unused instruction slot is not plain NOP')
    end = [i for i, line in enumerate(lines) if 'PseudoTTReplayRecordEnd' in line]
    require(len(end) == 1 and end[0] > start, 'record-only boundary lost')
    payload = [line for line in lines[start + 1:end[0]] if 'PseudoTTSFPURecordWord ' in line]
    require(len(payload) == header[2], 'recorded payload length changed')
    for line in payload:
        require(owner.record.effects(line) == (Counter(['tt_issue']), Counter(['tt_issue'])) and
                '::' not in line, 'record-only payload acquired numerical or memory effects')
    seeds = {}
    for index, line in enumerate(lines):
        match = re.search(r'(\$tt_l[0-7]) = TTSFPMOVAll \$tt_c(9|15)\b', line)
        if match:
            require(match[2] not in seeds and index < start, 'seed duplicated or moved')
            seeds[match[2]] = match[1][1:]
    require(set(seeds) == {'9', '15'} and seeds['9'] != seeds['15'], 'mutable/capture alias')
    expected_effects = (Counter(['tt_cc', 'tt_config', 'tt_dst', 'tt_issue', seeds['9'], seeds['15']]),
                        Counter(['tt_config', 'tt_dst', 'tt_issue', seeds['9']]))
    require(owner.record.effects(mops[0][1]) == expected_effects and
            'volatile load store' in mops[0][1], 'MOP lost old/result/readonly-capture effects')
    require(not any(re.search(r'\b(?:TTSFPLOAD|TTSFPMOV|PseudoCALL)\b', line) for line in lines),
            'hidden SFPR copy/spill/helper')
    require(not any('tt_l' in line and ('%stack' in line or 'fixed-stack' in line) for line in lines),
            'SFPR stack traffic')
    require(sum('TTSFPMOVAll' in line for line in lines) == 2 and
            not any(' = COPY' in line and 'tt_l' in line for line in lines),
            'numerical repair transport')
    stores = [(i, line) for i, line in enumerate(lines) if re.search(r'\bTTSFPSTORE\b', line)]
    require(len(stores) == 2, 'before/after snapshots changed')
    for (_, line), address in zip(stores, [0, 2]):
        match = re.search(r'\bTTSFPSTORE\s+(?:killed\s+|renamable\s+)*(\$tt_l[0-7]),\s*(\d+),\s*0,\s*4,', line)
        require(match and match[1] == '$' + seeds['9'] and int(match[2]) == address,
                'snapshot changed the real mutable role/address')
    (_, owners, _, graph, _, reaches, reachable, _, before, after) = owner.cfg.machine_cfg(lines)
    setup = [index for index, _ in controls + bindings + clears]
    for site in setup:
        require(before(end[0], site) and before(site, mops[0][0]), 'record/setup dominance lost')
        require(not any(reaches(successor, owners[site]) for successor in graph[owners[site]]),
                'setup moved into source cycle')
    execute = mops[0][0]
    require(any(reaches(successor, owners[execute]) for successor in graph[owners[execute]]),
            'source backedge lost')
    exits = {bid for bid in reachable if not graph[bid]}
    require(any(reaches(next(iter(owners.values())), exit, {owners[execute]}) for exit in exits),
            'zero-trip path lost')
    require(before(stores[0][0], execute) and after(stores[1][0], execute),
            'snapshots escaped source execution boundary')

    cells = owner.setup_words(assembly, name)
    require(cells[0] == outer and cells[1] == inner, 'actual MMIO counts changed')
    require(all(cells[slot] == 0x02000000 for slot in ([2, 3, 4, 6] if bind8 else [2, 3, 4, 6, 8])),
            'actual inactive slot changed')
    expected_replay = 0x04000000 | (header[2] << 4)
    require(all(cells[slot] == expected_replay for slot in slots), 'actual Loop1Last lost Replay range')
    stream = owner.record.words(assembly, name)
    record_at = [index for index, word in enumerate(stream) if word >> 24 == 4]
    require(len(record_at) == 1, 'record header duplicated or ordinary Replay substituted')
    record_at = record_at[0]
    require(stream[record_at] == expected_replay | 1, 'actual record mode/range changed')
    body = stream[record_at + 1:record_at + 1 + header[2]]
    mutable, capture = int(seeds['9'][-1]), int(seeds['15'][-1])
    iadd = 0x79000004 | (capture << 8) | (mutable << 4)
    require(body.count(iadd) == 1 and all(word in [iadd, 0x8f000000] for word in body),
            'recorded arithmetic duplicated or physical roles changed')
    require(stream.count(0x01800000) == 1, 'actual MOP1 instruction lost')
    # The ISA replaces each inner tail: nonfinal outer uses slot8, final outer slot7.
    slot_counts = {5: outer * (inner - 1), 7: 1, 8: outer - 1}
    require(sum(slot_counts.values()) == outer * inner, 'MOP expansion count mismatch')
    return {'outer': outer, 'inner': inner, 'slot_counts': slot_counts, 'mutable': mutable,
            'capture': capture, 'recorded_words': header[2]}


def check(assembly, mir, object_path, disassembly, single=None):
    profiles = single or [(f'contents_mop_outer{outer}_inner{inner}', outer, inner, True)
                          for outer, inner in PROFILES]
    owner.check_object(assembly, object_path, disassembly, [p[0] for p in profiles])
    return [check_function(assembly, mir, *profile) for profile in profiles]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['mir', 'assembly', 'object', 'disassembly']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--control', action='store_true')
    args = parser.parse_args()
    single = [('contents_mop_loop', 1, 7, False)] if args.control else None
    check(args.assembly.read_text(), args.mir.read_text(), args.object,
          args.disassembly.read_text(), single)
