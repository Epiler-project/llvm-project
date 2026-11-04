#!/usr/bin/env python3
"""Mutation controls on the actual ELF, independently re-disassembled each time."""
import argparse
import importlib.util
from pathlib import Path
import re
import struct
import subprocess
import tempfile


def load_checker():
    path = Path(__file__).with_name('check-tensix-contents-mop.py')
    spec = importlib.util.spec_from_file_location('mop_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mutate_object(data, old, new):
    pattern = struct.pack('<I', old)
    # Restrict the search to executable ELF sections, never metadata/debug info.
    shoff = struct.unpack_from('<I', data, 32)[0]
    shentsize, shnum = struct.unpack_from('<HH', data, 46)
    assert shentsize == 40
    for index in range(shnum):
        section = struct.unpack_from('<IIIIIIIIII', data, shoff + index * 40)
        if not section[2] & 4:  # SHF_EXECINSTR
            continue
        for offset in range(section[4], section[4] + section[5], 4):
            if data[offset:offset + 4] == pattern:
                result = bytearray(data)
                struct.pack_into('<I', result, offset, new)
                return bytes(result)
    raise ValueError('mutation target missing from actual object: ' + hex(old))


def assert_rejected(action, label):
    try:
        action()
    except ValueError:
        return
    raise ValueError('mutation was falsely accepted: ' + label)


def check_mutations(checker, assembly, object_path, objdump, mattr, names):
    data = object_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix='contents-mop-mutations-') as directory:
        output = Path(directory) / 'mutation.o'
        def receive(contents):
            output.write_bytes(contents)
            disassembly = subprocess.check_output(
                [str(objdump), '-d', '--mattr=' + mattr, str(output)], text=True)
            checker.check_object(assembly, output, disassembly, names)
        receive(data)
        # Native MOP, actual recorded arithmetic, scalar count, MMIO base and
        # store displacement are independent semantic corruption controls.
        native = checker.record.words(assembly, names[0])
        recorded = next((word for word in native if word >> 24 == 0x79), native[1])
        rotate = lambda word: ((word << 2) | (word >> 30)) & 0xffffffff
        changes = [('native-mop', 0x06000000, 0x08000000),
                   ('recorded-word-role', rotate(recorded), rotate(recorded ^ 0x10))]
        # Locate genuine RISC-V scalar instructions by their opcode/fields.
        instructions = []
        shoff = struct.unpack_from('<I', data, 32)[0]
        shnum = struct.unpack_from('<H', data, 48)[0]
        for index in range(shnum):
            sec = struct.unpack_from('<IIIIIIIIII', data, shoff + 40 * index)
            if sec[2] & 4:
                instructions.extend(struct.unpack_from('<I', data, off)[0]
                                    for off in range(sec[4], sec[4] + sec[5], 4))
        count = next(word for word in instructions if word & 0x707f == 0x13 and
                     (word >> 20) & 4095 in (1, 4, 7))
        base = next(word for word in instructions if word & 0xfffff07f == 0xffb80037)
        # Select an actual MMIO store, not an earlier O0 fixed-stack spill.
        store = next(instructions[index + 1] for index, word in enumerate(instructions[:-1])
                     if word & 0xfffff07f == 0xffb80037 and
                     instructions[index + 1] & 0x707f == 0x2023 and
                     (instructions[index + 1] >> 15) & 31 == (word >> 7) & 31)
        changes.extend([('hardware-count', count, count ^ (1 << 20)),
                        ('mmio-base', base, base ^ (1 << 12)),
                        ('mmio-store-offset', store, store ^ (1 << 7))])
        for label, old, new in changes:
            changed = mutate_object(data, old, new)
            assert_rejected(lambda: receive(changed), label)
    # Unknown scalar definitions may not preserve an earlier constant fact.
    source = checker.function_assembly(assembly, names[0])
    match = re.search(r'^\s*addi\s+(\w+),\s*\1,\s*1\s*$', source, re.M)
    if match is None:
        raise ValueError('scalar stale-definition mutation target missing')
    for instruction in ['add {0}, zero, zero', 'lw {0}, 0(sp)', 'xor {0}, {0}, {0}']:
        mutated = assembly.replace(match[0], match[0] + '\n\t' + instruction.format(match[1]), 1)
        assert_rejected(lambda: checker.setup_words(mutated, names[0]), instruction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly', required=True, type=Path)
    parser.add_argument('--object', required=True, type=Path)
    parser.add_argument('--objdump', required=True, type=Path)
    parser.add_argument('--mattr', required=True)
    args = parser.parse_args()
    check_mutations(load_checker(), args.assembly.read_text(), args.object,
                    args.objdump, args.mattr,
                    ['contents_mop_loop', 'contents_mop_twice', 'contents_mop_and_replay'])
