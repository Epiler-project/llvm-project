#!/usr/bin/env python3
"""Mutation tests for the Replay evidence checker, not compiler/device acceptance."""
import argparse
import importlib.util
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mir', required=True, type=Path)
args = parser.parse_args()
checker_path = Path(__file__).with_name('check-tensix-contents-replay-phi.py')
spec = importlib.util.spec_from_file_location('checker', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
base = args.mir.read_text()
roles = checker.check(base, 'straight')
mutable, capture = roles['mutable'], roles['capture']
foreign = next(f'$tt_l{i}' for i in range(8) if f'$tt_l{i}' not in (mutable, capture))
lines = base.splitlines(True)
execute = next(line for line in lines if 'PseudoTTContentsSFPUReplay 0,' in line)
add = next(line for line in lines if '= TTSFPIADD' in line)
end = next(line for line in lines if 'PseudoTTReplayRecordEnd' in line)
first = next(line for line in lines if 'TTSFPSTORE' in line and ', 0, 0, 4,' in line)
seed = next(line for line in lines if 'TTSFPMOVAll $tt_c9' in line)
setup = next(line for line in lines if 'TTSFPCONFIGReset' in line)

# These CFG scaffolds exercise the parser using real compiled issue/effect
# lines. They are deliberately not described as compiler-generated loops.
loop = base.replace(execute,
    '    BGE $x0, $x10, %bb.2\n    PseudoBR %bb.1\n  bb.1:\n' + execute +
    '    BLT $x11, $x10, %bb.1\n    PseudoBR %bb.2\n  bb.2:\n')
diamond = base.replace(execute,
    '    BEQ $x10, $x0, %bb.2\n    PseudoBR %bb.1\n  bb.1:\n' + execute +
    '    PseudoBR %bb.2\n  bb.2:\n' + execute)
checker.check(loop, 'loop')
checker.check(loop, 'capture-phi')
checker.check(diamond, 'diamond')


def reorder(text):
    chunks = re.split(r'(?=^  bb\.)', text, flags=re.M)
    prefix, entry, middle, last = chunks
    tail = last[last.index('...'):]
    last = last[:last.index('...')]
    text = prefix + entry + last + middle + tail
    return re.sub(r'\bbb\.(\d+)',
                  lambda match: 'bb.' + {'0': '7', '1': '42', '2': '2'}[match.group(1)],
                  text)


# Equivalent explicit-edge CFGs must survive arbitrary block numbering and
# layout. A numeric/layout backedge heuristic is insufficient.
checker.check(reorder(loop), 'loop')
checker.check(reorder(diamond), 'diamond')
mutations = {
    'missing-first-snapshot': (base.replace(first, ''), 'straight'),
    'snapshot-after-execute': (base.replace(first, '').replace(execute, execute + first), 'straight'),
    'wrong-execute-use': (base.replace(execute, execute.replace(f'implicit {mutable},', f'implicit {foreign},')), 'straight'),
    'missing-capture-use': (base.replace(execute, execute.replace(f'implicit {capture},', '')), 'straight'),
    'wrong-execute-def': (base.replace(execute, execute.replace(f'implicit-def {mutable}', f'implicit-def {capture}')), 'straight'),
    'duplicated-arithmetic': (base.replace(add, add + add), 'straight'),
    'wrong-mutable-tie': (base.replace(add, add.replace(f'TTSFPIADD {mutable},', f'TTSFPIADD {foreign},')), 'straight'),
    'wrong-word-count': (base.replace('SFPUReplay 0, 0, 1, 0', 'SFPUReplay 0, 0, 2, 0'), 'straight'),
    'extra-replay': (base.replace(execute, execute + execute), 'straight'),
    'repair-copy': (base.replace(execute, f'    {foreign} = COPY {mutable}\n' + execute), 'straight'),
    'reload': (base.replace(execute, f'    {foreign} = TTSFPLOAD 0, 0, 4\n' + execute), 'straight'),
    'direct-helper': (base.replace(execute, '    PseudoCALL @helper\n' + execute), 'straight'),
    'indirect-helper': (base.replace(execute, '    PseudoCALLIndirect $x10\n' + execute), 'straight'),
    'hidden-native-write': (base.replace(execute, f'    {mutable} = TTSFPXOR {mutable}, {capture}\n' + execute), 'straight'),
    'missing-record-end': (base.replace(end, ''), 'straight'),
    'record-end-after-snapshot': (base.replace(end, '').replace(first, first + end), 'straight'),
    'wrong-capture-seed': (base.replace('TTSFPMOVAll $tt_c15', 'TTSFPMOVAll $tt_c14'), 'straight'),
    'late-mutable-seed': (base.replace(seed, '').replace(execute, seed + execute), 'straight'),
    'late-setup': (base.replace(setup, '').replace(execute, setup + execute), 'straight'),
    'duplicate-setup': (base.replace(setup, setup + setup), 'straight'),
    'loop-absent': (base, 'loop'),
    'zero-trip-enters-loop': (loop.replace('BGE $x0, $x10, %bb.2', 'BGE $x0, $x10, %bb.1'), 'loop'),
    'backedge-enters-exit': (loop.replace('BLT $x11, $x10, %bb.1', 'BLT $x11, $x10, %bb.2'), 'loop'),
    'diamond-bypass-executes': (diamond.replace('BEQ $x10, $x0, %bb.2', 'BEQ $x10, $x0, %bb.1'), 'diamond'),
}
# Keep all issue lines in a dynamic loop: lexical record count remains one,
# but recording, seeds and both snapshots now repeat every iteration.
body_match = re.search(r'^body:\s*\|\s*\n(.*?)(?=^\.\.\.|\Z)', base, re.M | re.S)
body = body_match.group(1)
body = re.sub(r'^  bb\.[^\n]+\n', '', body, count=1, flags=re.M)
body = body.replace('    PseudoRET\n', '')
rerecord = ('  bb.0:\n    BGE $x0, $x10, %bb.2\n    PseudoBR %bb.1\n'
            '  bb.1:\n' + body + '    BLT $x11, $x10, %bb.1\n'
            '    PseudoBR %bb.2\n  bb.2:\n    PseudoRET\n')
mutations['record-inside-loop'] = (base[:body_match.start(1)] + rerecord +
                                   base[body_match.end(1):], 'loop')
for name, (text, kind) in mutations.items():
    try:
        checker.check(text, kind)
    except ValueError:
        continue
    raise AssertionError('incorrectly accepted mutation: ' + name)
print('actual compiled baseline accepted; 5 CFG parser controls accepted; '
      f'{len(mutations)} mutations rejected')
