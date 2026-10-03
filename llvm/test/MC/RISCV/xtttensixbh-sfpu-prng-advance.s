# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=FEATURE

# Fixed SFPMOV opcode 0x7c, special selector 9, exact mode 8; only VD varies.
# Direct instruction words are the raw ISA words rotated left by two bits.
# ENC: ttsfpmov.prng.advance l0 # encoding: [0x21,0x24,0x00,0xf0]
# ENC: ttsfpmov.prng.advance l7 # encoding: [0xe1,0x25,0x00,0xf0]
# DIS: ttsfpmov.prng.advance l0
# DIS: ttsfpmov.prng.advance l7
# FEATURE: instruction requires the following: 'XTTTensixBH'
ttsfpmov.prng.advance l0
ttsfpmov.prng.advance l7
