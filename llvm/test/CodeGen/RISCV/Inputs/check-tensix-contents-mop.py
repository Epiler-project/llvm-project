#!/usr/bin/env python3
"""Check real MOP encodings, setup, allocated effects and preserved machine CFG.

The arithmetic oracle uses symbolic lane seeds; it is an emitted-word contract
check, not a substitute for independent device numeric verification.
"""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import re
import struct


def load_helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


record = load_helper('check-tensix-contents-record-only.py', 'contents_record')
cfg = load_helper('check-tensix-contents-replay-phi.py', 'contents_cfg')
require = record.require


def function_assembly(assembly, name):
    match = re.search(r'^' + re.escape(name) + r':.*?^\.Lfunc_end\d+:', assembly, re.M | re.S)
    require(match, 'missing assembly function ' + name)
    return match[0]


def setup_lines(lines):
    # Unknown GPR definitions kill constant facts. Never let a prior LI/LUI
    # survive an unmodelled ADD/load/CSR write (the mutation tests cover this).
    aliases = ['zero', 'ra', 'sp', 'gp', 'tp', 't0', 't1', 't2', 's0', 's1',
               'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 's2', 's3',
               's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 't3', 't4', 't5', 't6']
    names = {name: index for index, name in enumerate(aliases)}
    names.update({'x' + str(index): index for index in range(32)})
    names['fp'] = 8
    registers = {0: 0}
    cells = {}
    immediate = r'(-?(?:0x[0-9a-fA-F]+|[0-9]+))'
    def value(token):
        return int(token, 0) if token.lower().lstrip('-').startswith('0x') else int(token)
    def assign(name, result):
        number = names.get(name)
        if number:
            if result is None:
                registers.pop(number, None)
            else:
                registers[number] = result & 0xffffffff
    for line in lines:
        text = line.split('#', 1)[0].strip()
        match = re.fullmatch(r'lui\s+(\w+),\s*' + immediate, text)
        if match:
            assign(match[1], value(match[2]) << 12)
            continue
        match = re.fullmatch(r'addi\s+(\w+),\s*(\w+),\s*' + immediate, text)
        if match:
            source = registers.get(names.get(match[2]))
            assign(match[1], None if source is None else source + value(match[3]))
            continue
        match = re.fullmatch(r'li\s+(\w+),\s*' + immediate, text)
        if match:
            assign(match[1], value(match[2]))
            continue
        match = re.fullmatch(r'mv\s+(\w+),\s*(\w+)', text)
        if match:
            assign(match[1], registers.get(names.get(match[2])))
            continue
        match = re.fullmatch(r'sw\s+(\w+),\s*' + immediate + r'\((\w+)\)', text)
        if match:
            base = registers.get(names.get(match[3]))
            # Unknown store addresses cannot certify a write-only MMIO profile.
            if base is None and names.get(match[3]) in (2, 8):
                continue  # ordinary fixed-stack GPR spills are not MMIO
            if base is None and names.get(match[3]) in (2, 8):
                continue  # ordinary fixed-stack GPR spills are not MMIO
            require(base is not None, 'unknown store base in MOP setup receiver')
            address = (base + value(match[2])) & 0xffffffff
            if 0xffb80000 <= address < 0xffb80024:
                data = registers.get(names.get(match[1]))
                require((address & 3) == 0 and data is not None,
                        'MOP setup lost its known aligned value')
                cell = (address - 0xffb80000) // 4
                require(cell not in cells, 'MOP setup cell was rewritten')
                cells[cell] = data
            continue
        if re.match(r'(?:call|tail|jal|jalr)\b', text):
            registers = {0: 0}
            continue
        # Stores, branches and side-effect-only instructions have no explicit
        # destination GPR. Other unknown first-GPR operands conservatively kill.
        if re.match(r'(?:s[bhwd]|f[sw]s|b(?:eq|ne|lt|ge|ltu|geu|eqz|nez|lez|gez|ltz|gtz)|j|jr|ret)\b', text):
            continue
        match = re.match(r'[A-Za-z][A-Za-z0-9_.]*\s+(\w+)(?:,|$)', text)
        if match and match[1] in names:
            assign(match[1], None)
    require(set(cells) == set(range(9)), 'actual MMIO setup does not cover all nine cells')
    return cells


def setup_words(assembly, name):
    return setup_lines(function_assembly(assembly, name).splitlines())


def object_functions(path):
    data = path.read_bytes()
    require(data[:6] == b'\x7fELF\x01\x01', 'expected little-endian ELF32 object')
    shoff = struct.unpack_from('<I', data, 32)[0]
    shentsize, shnum = struct.unpack_from('<HH', data, 46)
    require(shentsize == 40 and shnum > 0, 'unsupported ELF section table')
    sections = [struct.unpack_from('<IIIIIIIIII', data, shoff + index * 40)
                for index in range(shnum)]
    def section_bytes(index):
        section = sections[index]
        return data[section[4]:section[4] + section[5]]
    functions = {}
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        require(section[9] == 16, 'unsupported ELF32 symbol entry')
        strings = section_bytes(section[6])
        for offset in range(section[4], section[4] + section[5], 16):
            name, address, size, info, _, owner = struct.unpack_from('<IIIBBH', data, offset)
            if info & 15 != 2 or not size or owner >= len(sections):
                continue
            end = strings.find(b'\0', name)
            label = strings[name:end].decode('ascii')
            contents = section_bytes(owner)
            require(address + size <= len(contents), 'function outside object section')
            functions[label] = (address, contents[address:address + size])
    return functions


def check_object(assembly, object_path, disassembly, names):
    functions = object_functions(object_path)
    entries = []
    for line in disassembly.splitlines():
        match = re.match(r'^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{8})\s+(\S+)(?:\s+(.*))?$', line)
        if match:
            entries.append((int(match[1], 16), int(match[2], 16), match[3], match[4] or ''))
    for name in names:
        require(name in functions, 'missing object function ' + name)
        address, data = functions[name]
        require(len(data) % 4 == 0, 'unexpected compressed/truncated object function')
        actual = [entry for entry in entries if address <= entry[0] < address + len(data)]
        require([entry[0] for entry in actual] == list(range(address, address + len(data), 4)),
                'object disassembly omitted/duplicated an instruction')
        for position, word, _, _ in actual:
            require(word == struct.unpack_from('<I', data, position - address)[0],
                    'disassembly raw word differs from actual object bytes')
        native = [((word >> 2) | (word << 30)) & 0xffffffff
                  for _, word, mnemonic, _ in actual if mnemonic.startswith('tt')]
        require(native == record.words(assembly, name),
                'actual object native issue/recorded words differ from assembly')
        lines = [mnemonic + ' ' + operands for _, _, mnemonic, operands in actual]
        require(setup_lines(lines) == setup_words(assembly, name),
                'actual object MMIO setup differs from assembly')


def check_mir(mir, name, inner, expected_mops, expected_replays, mode):
    lines = record.body(mir, name)
    records, replays, mops, controls, bindings, clears = [], [], [], [], [], []
    for index, line in enumerate(lines):
        match = re.search(r'\bPseudoTTContentsSFPUReplay ([01]), ([01]), (\d+), (\d+),', line)
        if match:
            fields = tuple(map(int, match.groups()))
            (records if fields[0] else replays).append((index, fields, line))
        if re.search(r'\bTTMOP\b', line):
            require(re.search(r'\bTTMOP 0, 0, 1,', line), 'noncanonical native MOP')
            mops.append((index, line))
        match = re.search(r'\bPseudoTTMOPControlWriteImm (\d+), (\d+),', line)
        if match:
            controls.append((index, tuple(map(int, match.groups()))))
        match = re.search(r'\bPseudoTTREPLAYMop (\d+), 0, 0, (\d+), (\d+),', line)
        if match:
            bindings.append((index, tuple(map(int, match.groups()))))
            require(record.effects(line) == (Counter(['tt_config', 'tt_issue']),
                                              Counter(['tt_config', 'tt_issue'])),
                    'slot programming acquired numerical effects')
        match = re.search(r'\bPseudoTTMOPClear (\d+),', line)
        if match:
            clears.append((index, int(match[1])))
    require(len(records) == 1 and len(mops) == expected_mops and len(replays) == expected_replays,
            'record/MOP/ordinary Replay count changed')
    at, header, _ = records[0]
    require(header[:2] == (1, mode) and 0 < header[2] <= 32 and header[3] == 0,
            'record mode/length/placement changed')
    require(sorted(value for _, value in controls) == [(0, 1), (1, inner)], 'MOP counts changed')
    require(sorted(value for _, value in bindings) == [(5, header[2], 0), (7, header[2], 0)],
            'MOP loop/last slots lost the same actual recorded range')
    require(sorted(value for _, value in clears) == [2, 3, 4, 6, 8], 'unused slots are not plain NOP')
    ends = [i for i, line in enumerate(lines) if 'PseudoTTReplayRecordEnd' in line]
    require(len(ends) == 1 and ends[0] > at, 'record end lost')
    seeds = {}
    for i, line in enumerate(lines):
        match = re.search(r'(\$tt_l[0-7]) = TTSFPMOVAll \$tt_c(9|15)\b', line)
        if match:
            require(match[2] not in seeds and i < at, 'duplicate/moved seed')
            seeds[match[2]] = match[1][1:]
    require(set(seeds) == {'9', '15'} and seeds['9'] != seeds['15'], 'mutable/capture roles lost')
    captured = [line for line in lines[at + 1:ends[0]] if line.strip()]
    if mode == 0:
        payload = [line for line in captured if 'PseudoTTSFPURecordWord ' in line]
        require(len(payload) == header[2], 'record-only payload length changed')
        for line in payload:
            require(record.effects(line) == (Counter(['tt_issue']), Counter(['tt_issue'])) and '::' not in line,
                    'record-only payload acquired numerical/memory effects')
        require(not any(re.search(r'\bTTSFPIADD\b', line) for line in captured),
                'record-only body performs numerical work')
    else:
        require(sum(bool(re.search(r'\bTTSFPIADD\b', line)) for line in captured) == 1,
                'record-and-execute body duplicated/lost arithmetic')
    mop_effects = (Counter(['tt_cc', 'tt_config', 'tt_dst', 'tt_issue', seeds['9'], seeds['15']]),
                   Counter(['tt_config', 'tt_dst', 'tt_issue', seeds['9']]))
    replay_effects = (Counter(['tt_cc', 'tt_config', 'tt_issue', seeds['9'], seeds['15']]),
                      Counter(['tt_issue', seeds['9']]))
    for index, line in mops:
        require(index > ends[0] and record.effects(line) == mop_effects,
                'MOP lost its exact old/result/readonly-capture effects')
        require('volatile load store' in line, 'MOP lost issue memory barrier')
    for index, fields, line in replays:
        require(index > ends[0] and fields == (0, 0, header[2], 0) and
                record.effects(line) == replay_effects, 'ordinary Replay ancestry/effects changed')
    allowed = {'TTSFPENCC', 'TTSFPCONFIGReset', 'TTSFPMOVAll', 'TTSFPIADD', 'TTSFPNOP',
               'TTSFPSTORE', 'TTMOP', 'PseudoTTContentsSFPUReplay', 'PseudoTTSFPURecordWord',
               'PseudoTTReplayRecordEnd', 'PseudoTTMOPControlWriteImm', 'PseudoTTMOPClear', 'PseudoTTREPLAYMop'}
    for line in lines:
        tokens = line.strip().split(' = ', 1)[-1].split()
        if tokens and tokens[0].startswith(('TT', 'PseudoTT')):
            require(tokens[0] in allowed, 'hidden accelerator transport/spill/helper')
    stores = [(i, line) for i, line in enumerate(lines) if re.search(r'\bTTSFPSTORE\b', line)]
    require(len(stores) == (2 if name == 'contents_mop_loop' else 3), 'snapshot count changed')
    snapshot_addresses = [0, 2] if name == 'contents_mop_loop' else [0, 2, 4]
    for (_, line), address in zip(stores, snapshot_addresses):
        store = re.search(r'\bTTSFPSTORE\s+(?:killed\s+|renamable\s+)*'
                          r'(\$tt_l[0-7]),\s*(\d+),\s*0,\s*4,', line)
        require(store and store[1] == '$' + seeds['9'] and int(store[2]) == address,
                'snapshot changed mutable role or exact ordered addressing')
    (blocks, owners, instructions, graph, conditional, reaches, reachable, dom, before, after) = cfg.machine_cfg(lines)
    setup = [index for index, _ in controls + bindings + clears]
    require(all(before(ends[0], index) for index in setup), 'setup escaped record dominance')
    for site in setup:
        require(not any(reaches(successor, owners[site]) for successor in graph[owners[site]]),
                'setup moved into dynamic cycle')
        require(all(before(site, execute) for execute, _ in mops), 'setup does not dominate every MOP')
    if name == 'contents_mop_loop':
        execute = mops[0][0]
        require(any(reaches(successor, owners[execute]) for successor in graph[owners[execute]]),
                'source loop backedge was lost')
        exits = {bid for bid in reachable if not graph[bid]}
        require(any(reaches(next(iter(blocks)), end, {owners[execute]}) for end in exits),
                'source zero-trip path was lost')
        require(before(stores[0][0], execute) and after(stores[1][0], execute),
                'source snapshots lost loop dominance/postdominance')
    return header


def check_stream(assembly, name, inner, expected_mops, expected_replays, mode):
    cells = setup_words(assembly, name)
    require(cells[0] == 1 and cells[1] == inner, 'actual hardware counts changed')
    require(all(cells[slot] == 0x02000000 for slot in (2, 3, 4, 6, 8)), 'actual unused slot is not NOP')
    stream = record.words(assembly, name)
    recorded, recording = [], None
    mutable = capture = None
    lane_capture = tuple(0x01020304 + 17 * lane for lane in range(32))
    values, snapshots = {}, []
    mops = replays = records = arithmetic = 0
    def issue(word):
        nonlocal mutable, capture, arithmetic
        opcode = word >> 24
        if opcode == 0x7c:
            dst, src = (word >> 4) & 15, (word >> 8) & 15
            require(word == 0x7c000002 | (src << 8) | (dst << 4) and src in (9, 15) and dst < 8,
                    'unexpected physical move/repair')
            if src == 9:
                require(mutable is None, 'extra mutable seed'); mutable = dst; values[dst] = (0,) * 32
            else:
                require(capture is None, 'extra capture seed'); capture = dst; values[dst] = lane_capture
        elif opcode == 0x79:
            dst, src = (word >> 4) & 15, (word >> 8) & 15
            require(dst == mutable and src == capture and word == 0x79000004 | (src << 8) | (dst << 4),
                    'IADD lost recorded physical old/capture roles')
            values[dst] = tuple((a + b) & 0xffffffff for a, b in zip(values[dst], values[src])); arithmetic += 1
        elif opcode == 0x72:
            src, address = (word >> 20) & 15, word & 1023
            require(src == mutable and word == 0x72040000 | (src << 20) | address, 'unexpected store/spill')
            snapshots.append((address, values[src]))
        else:
            require(word in (0x8f000000, 0x8a00300a, 0x910000f1), 'unexpected native word ' + hex(word))
    for word in stream:
        if recording is not None:
            recorded.append(word)
            if mode:
                issue(word)
            recording -= 1
            if recording == 0:
                recording = None
            continue
        opcode = word >> 24
        if opcode == 0x04:
            load, execute, length, start = word & 1, (word >> 1) & 7, (word >> 4) & 1023, (word >> 14) & 1023
            require(0 < length <= 32 and start == 0, 'Replay range changed')
            if load:
                require(records == 0 and execute == mode, 'record mode/count changed')
                records += 1; recording = length
                expected = 0x04000000 | (length << 4)
                require(cells[5] == cells[7] == expected, 'MMIO slots do not name actual recorded body')
            else:
                require(execute == 0 and length == len(recorded), 'ordinary Replay changed range')
                replays += 1
                for item in recorded:
                    issue(item)
        elif opcode == 0x01:
            require(word == 0x01800000, 'MOP changed native fields/override')
            mops += 1
            require(recorded, 'MOP read no actual recording')
            for _ in range(inner):
                for item in recorded:
                    issue(item)
        else:
            issue(word)
    require(recording is None and records == 1 and mops == expected_mops and replays == expected_replays,
            'truncated/missing/duplicate actual issue')
    require(sum(word >> 24 == 0x79 for word in recorded) == 1 and
            all(word >> 24 == 0x79 or word == 0x8f000000 for word in recorded),
            'recorded body was duplicated or contains unreceived arithmetic')
    if name != 'contents_mop_loop':
        coefficients = [mode, mode + inner, mode + inner + (inner if expected_mops == 2 else 1)]
        expected = [(address, tuple((coefficient * x) & 0xffffffff for x in lane_capture))
                    for address, coefficient in zip((0, 2, 4), coefficients)]
        require(snapshots == expected, 'before/between/after numerical snapshots changed')
    require(values[capture] == lane_capture, 'readonly capture was modified')


def check(assembly, mir, object_path, disassembly):
    check_object(assembly, object_path, disassembly,
                 ['contents_mop_loop', 'contents_mop_twice', 'contents_mop_and_replay'])
    for profile in [('contents_mop_loop', 7, 1, 0, 0), ('contents_mop_twice', 4, 2, 0, 0),
                    ('contents_mop_and_replay', 1, 1, 1, 1)]:
        check_mir(mir, *profile)
        check_stream(assembly, *profile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mir', required=True, type=Path)
    parser.add_argument('--assembly', required=True, type=Path)
    parser.add_argument('--object', required=True, type=Path)
    parser.add_argument('--disassembly', required=True, type=Path)
    args = parser.parse_args()
    check(args.assembly.read_text(), args.mir.read_text(), args.object, args.disassembly.read_text())
