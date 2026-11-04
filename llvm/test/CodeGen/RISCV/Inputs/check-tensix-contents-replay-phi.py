#!/usr/bin/env python3
"""Check finalized contents Replay roles, real CFG, and source snapshots."""
import argparse
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise ValueError(message)


def machine_cfg(lines):
    # Build edges from actual terminators and layout fallthrough, never numeric
    # block order or successor annotations standing in for emitted branches.
    blocks, owners, instructions, declared = {}, {}, [], {}
    active = None
    flags = {'frame-setup', 'frame-destroy', 'nofpexcept', 'nnan', 'ninf',
             'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
    for i, line in enumerate(lines):
        label = re.match(r'\s*bb\.(\d+)[^:]*:', line)
        if label:
            active = int(label.group(1))
            require(active not in blocks, 'duplicate machine block')
            blocks[active] = []
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if stripped.startswith('successors:'):
            declared[active] = set(map(int, re.findall(r'%bb\.(\d+)', line)))
            continue
        if re.match(r'(?:liveins|alignment|address-taken|landing-pad):', stripped):
            continue
        require(active is not None, 'instruction has no machine block')
        tokens = stripped.split(' = ', 1)[-1].split()
        while tokens and tokens[0] in flags:
            tokens.pop(0)
        require(tokens, 'missing machine opcode')
        op = tokens[0]
        owners[i] = active
        instructions.append((i, op, line))
        blocks[active].append((i, op, line))
    require(blocks, 'machine CFG is empty')
    order = list(blocks)
    graph = {bid: set() for bid in blocks}
    conditional = {}
    branch_ops = {'BEQ', 'BNE', 'BLT', 'BGE', 'BLTU', 'BGEU'}
    for n, bid in enumerate(order):
        falls_through = True
        for i, op, line in blocks[bid]:
            if op in branch_ops or op == 'PseudoBR':
                targets = list(map(int, re.findall(r'%bb\.(\d+)', line)))
                require(len(targets) == 1, 'branch lost its concrete target')
                graph[bid].add(targets[0])
                if op in branch_ops:
                    conditional.setdefault(bid, []).append(i)
                else:
                    falls_through = False
            elif op in {'PseudoRET', 'RET'}:
                falls_through = False
        if falls_through and n + 1 < len(order):
            graph[bid].add(order[n + 1])
        require(graph[bid] <= blocks.keys(), 'branch targets a missing block')
        if bid in declared:
            require(graph[bid] == declared[bid], 'successors differ from actual CFG edges')

    def reaches(start, target, forbidden=frozenset()):
        seen, pending = set(), [start]
        while pending:
            node = pending.pop()
            if node in seen or node in forbidden:
                continue
            if node == target:
                return True
            seen.add(node)
            pending.extend(graph[node])
        return False

    entry = order[0]
    reachable = {bid for bid in blocks if reaches(entry, bid)}
    preds = {bid: {other for other in reachable if bid in graph[other]}
             for bid in reachable}
    dom = {bid: ({entry} if bid == entry else set(reachable)) for bid in reachable}
    changed = True
    while changed:
        changed = False
        for bid in reachable - {entry}:
            value = {bid} | set.intersection(*(dom[pred] for pred in preds[bid]))
            if value != dom[bid]:
                dom[bid] = value
                changed = True
    exits = {bid for bid in reachable if not graph[bid]}
    require(exits, 'machine CFG has no exit')
    require(all(any(reaches(bid, end) for end in exits) for bid in reachable),
            'machine CFG contains a path with no exit')
    postdom = {bid: ({bid} if bid in exits else set(reachable)) for bid in reachable}
    changed = True
    while changed:
        changed = False
        for bid in reachable - exits:
            value = {bid} | set.intersection(*(postdom[succ] for succ in graph[bid]))
            if value != postdom[bid]:
                postdom[bid] = value
                changed = True

    def before(left, right):
        a, b = owners[left], owners[right]
        return a in reachable and b in reachable and (left < right if a == b else a in dom[b])

    def after(left, right):
        a, b = owners[left], owners[right]
        return a in reachable and b in reachable and (left > right if a == b else a in postdom[b])

    return blocks, owners, instructions, graph, conditional, reaches, reachable, dom, before, after


def check(text, kind):
    # Inspect only machine bodies. Embedded IR is observation, not a second
    # emitted instruction stream or proof of target replay behavior.
    bodies = re.findall(r'^body:\s*\|\s*\n(.*?)(?=^\.\.\.|\Z)', text,
                        re.M | re.S)
    require(len(bodies) == 1, 'expected one complete machine body')
    body = bodies[0]
    lines = body.splitlines()
    (blocks, owners, instructions, graph, conditional, reaches, reachable,
     dom, before, after) = machine_cfg(lines)
    expected_executes = 2 if kind == 'diamond' else 1
    records = [(i, line) for i, line in enumerate(lines)
               if re.search(r'\bPseudoTTContentsSFPUReplay 1, 1, 1, 0\b', line)]
    executes = [(i, line) for i, line in enumerate(lines)
                if re.search(r'\bPseudoTTContentsSFPUReplay 0, 0, 1, 0\b', line)]
    require(len(records) == 1, 'record must appear once')
    require(len(executes) == expected_executes, 'execute count changed')
    require(body.count('PseudoTTContentsSFPUReplay') == 1 + expected_executes,
            'extra or malformed Replay control')
    arithmetic = [(i, line) for i, line in enumerate(lines)
                  if re.search(r'\bTTSFPIADD\b', line)]
    require(len(arithmetic) == 1, 'recorded arithmetic body was duplicated or lost')
    at, operation = arithmetic[0]
    role = re.search(r'(\$tt_l[0-7])\s*=\s*TTSFPIADD\s+'
                     r'(?:killed\s+|renamable\s+)*(\$tt_l[0-7]),\s*'
                     r'(?:killed\s+|renamable\s+)*(\$tt_l[0-7])', operation)
    require(role is not None, 'arithmetic lost concrete ordinary LReg roles')
    mutable, old, capture = role.groups()
    require(mutable == old and mutable != capture,
            'mutable tie or independent capture role lost')
    ends = [i for i, line in enumerate(lines) if 'PseudoTTReplayRecordEnd' in line]
    require(len(ends) == 1, 'record end was lost or duplicated')
    record = records[0][0]
    require(owners[record] == owners[at] == owners[ends[0]] and
            record < at < ends[0], 'record/body/end left their original issue region')
    for reg, constant in ((mutable, '$tt_c9'), (capture, '$tt_c15')):
        seeds = [i for i, line in enumerate(lines)
                 if re.search(re.escape(reg) + r'\s*=\s*TTSFPMOVAll ' +
                              re.escape(constant) + r'\b', line)]
        require(len(seeds) == 1 and before(seeds[0], record),
                'contents seed does not dominate the recorded arithmetic')
    for _, execute in executes:
        uses = re.findall(r'implicit (?:killed |renamable )*(\$tt_l[0-7])\b', execute)
        defs = re.findall(r'implicit-def (?:dead |renamable )*(\$tt_l[0-7])\b', execute)
        require(sorted(uses) == sorted([mutable, capture]), 'execute use effects mismatch')
        require(defs == [mutable], 'execute must define its mutable role only')
    stores = []
    for i, line in enumerate(lines):
        if 'TTSFPSTORE' in line:
            store = re.search(r'TTSFPSTORE\s+(?:killed\s+|renamable\s+)*'
                              r'(\$tt_l[0-7]),\s*(\d+),\s*0,\s*4\b', line)
            require(store is not None, 'unexpected snapshot addressing or role')
            stores.append((i, store.group(1), int(store.group(2))))
    require(len(stores) == 2 and sorted(s[2] for s in stores) == [0, 8],
            'first/final snapshots must remain exact')
    require(all(s[1] == mutable for s in stores), 'snapshot role differs from mutable')
    first = next(i for i, _, offset in stores if offset == 0)
    final = next(i for i, _, offset in stores if offset == 8)
    require(before(ends[0], first), 'first snapshot precedes the actual recording')
    require(all(before(first, i) and after(final, i) for i, _ in executes),
            'first/final snapshots do not surround every destructive execute')
    require(not re.search(r'%\d+\b|\bCOPY\b|\bTTSFPLOAD\w*\b', body),
            'unallocated carrier, repair copy, or accelerator spill/reload remains')
    require(not any(re.match(r'(?:PseudoCALL|PseudoTAIL|INLINEASM|JAL|CALL|TAIL)', op)
                    for _, op, _ in instructions), 'unexpected helper or opaque assembler')
    allowed = {'TTSFPENCC', 'TTSFPCONFIGReset', 'TTSFPMOVAll', 'TTSFPIADD',
               'TTSFPSTORE', 'PseudoTTContentsSFPUReplay', 'PseudoTTReplayRecordEnd'}
    require(all(op in allowed for _, op, line in instructions
                if op.startswith(('TT', 'PseudoTT')) or re.search(r'\$tt_l[0-7]\b', line)),
            'unexpected native issue or hidden contents transport')
    require(not re.search(r'PseudoTTContentsReplayTemplate(?:Begin|End|Execute)', body),
            'template pseudo escaped finalization')
    require(body.count('TTSFPMOVAll') == 2, 'extra hidden contents transport')
    setup = {}
    for opcode in ('TTSFPENCC', 'TTSFPCONFIGReset'):
        sites = [i for i, op, _ in instructions if op == opcode]
        require(len(sites) == 1 and before(sites[0], record),
                'initial native setup was duplicated or moved into execution')
        setup[opcode] = sites[0]
    require(before(setup['TTSFPENCC'], setup['TTSFPCONFIGReset']),
            'native setup order changed')
    semantic = [record, at, ends[0], first, final] + [i for i, _ in executes]
    require(all(owners[i] in reachable for i in semantic), 'semantic instruction is unreachable')
    record_block = owners[record]
    require(not any(reaches(succ, record_block) for succ in graph[record_block]),
            'record-once template was moved into a cycle')
    require(all(before(ends[0], i) for i, _ in executes),
            'record does not dominate every execute')
    if kind in ('loop', 'capture-phi'):
        execute_block = owners[executes[0][0]]
        component = {bid for bid in reachable
                     if reaches(execute_block, bid) and reaches(bid, execute_block)}
        require(len(component) > 1 or execute_block in graph[execute_block],
                'execute is not in a real dynamic loop')
        require(record_block not in component, 'record must remain outside the loop')
        final_block = owners[final]
        require(final_block not in component, 'final snapshot must remain outside the loop')
        guards = [bid for bid in conditional if bid not in component and
                  bid in dom[execute_block] and len(graph[bid]) == 2]
        require(any(any(reaches(succ, execute_block, {final_block}) for succ in graph[guard]) and
                    any(reaches(succ, final_block, component) and
                        not reaches(succ, execute_block) for succ in graph[guard])
                    for guard in guards), 'zero-trip guard lost its loop and bypass edges')
        require(any(bid in conditional and len(graph[bid]) == 2 and
                    any(succ in component for succ in graph[bid]) and
                    any(succ not in component and reaches(succ, final_block, component)
                        for succ in graph[bid]) for bid in component),
                'dynamic latch lost its backedge or exit edge')
    elif kind == 'diamond':
        sites = [owners[i] for i, _ in executes]
        require(sites[0] != sites[1], 'diamond execute sites were collapsed')
        merges = [bid for bid in sites if bid in dom[owners[final]] and
                  all(reaches(other, bid) for other in sites)]
        require(len(merges) == 1, 'diamond lost its common execute after the PHI')
        merge = merges[0]
        optional = next(bid for bid in sites if bid != merge)
        require(optional not in dom[merge], 'optional execute became unconditional')
        require(any(bid in dom[optional] and bid in dom[merge] and
                    len(graph[bid]) == 2 and
                    any(reaches(succ, optional, {merge}) for succ in graph[bid]) and
                    any(reaches(succ, merge, {optional}) and not reaches(succ, optional)
                        for succ in graph[bid]) for bid in conditional),
                'conditional diamond lost its execute and bypass edges')
    return {'mutable': mutable, 'capture': capture, 'records': len(records),
            'executes': len(executes), 'stores': sorted(s[2] for s in stores),
            'kind': kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mir', required=True, type=Path)
    parser.add_argument('--kind', required=True, choices=('loop', 'diamond', 'capture-phi', 'straight'))
    args = parser.parse_args()
    try:
        result = check(args.mir.read_text(), args.kind)
    except ValueError as error:
        parser.exit(1, str(error) + '\n')
    print(result)


if __name__ == '__main__':
    main()
