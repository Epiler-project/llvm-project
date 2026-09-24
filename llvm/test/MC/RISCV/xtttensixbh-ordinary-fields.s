# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s
# One-field and asymmetric words catch operand permutations that all-maxima
# encodings cannot distinguish. Independent raw-word reference: TT-Metal
# 0473bf273aaca3ad0757fd29a1b6ccd6232bb6aa, Blackhole common/inc/ckernel_ops.h
# TT_OP_MVMUL / TT_OP_SETADC / TT_OP_SETRWC / TT_OP_REPLAY. Direct instruction
# fetch rotates the raw word left by two bits. These are fixed oracle bytes;
# the test does not read a production encoder or generate an opcode schema.

# Raw word: 0x26000001
# CHECK: ttmvmul 1, 0, 0, 0{{ *}}# encoding: [0x04,0x00,0x00,0x98]
ttmvmul 1, 0, 0, 0

# Raw word: 0x26004000
# CHECK: ttmvmul 0, 1, 0, 0{{ *}}# encoding: [0x00,0x00,0x01,0x98]
ttmvmul 0, 1, 0, 0

# Raw word: 0x26080000
# CHECK: ttmvmul 0, 0, 1, 0{{ *}}# encoding: [0x00,0x00,0x20,0x98]
ttmvmul 0, 0, 1, 0

# Raw word: 0x26400000
# CHECK: ttmvmul 0, 0, 0, 1{{ *}}# encoding: [0x00,0x00,0x00,0x99]
ttmvmul 0, 0, 0, 1

# Raw word: 0x26888011
# CHECK: ttmvmul 17, 2, 1, 2{{ *}}# encoding: [0x44,0x00,0x22,0x9a]
ttmvmul 17, 2, 1, 2

# Raw word: 0x50000001
# CHECK: ttsetadc 1, 0, 0, 0{{ *}}# encoding: [0x05,0x00,0x00,0x40]
ttsetadc 1, 0, 0, 0

# Raw word: 0x50040000
# CHECK: ttsetadc 0, 1, 0, 0{{ *}}# encoding: [0x01,0x00,0x10,0x40]
ttsetadc 0, 1, 0, 0

# Raw word: 0x50100000
# CHECK: ttsetadc 0, 0, 1, 0{{ *}}# encoding: [0x01,0x00,0x40,0x40]
ttsetadc 0, 0, 1, 0

# Raw word: 0x50200000
# CHECK: ttsetadc 0, 0, 0, 1{{ *}}# encoding: [0x01,0x00,0x80,0x40]
ttsetadc 0, 0, 0, 1

# Raw word: 0x50b81234
# CHECK: ttsetadc 4660, 2, 1, 5{{ *}}# encoding: [0xd1,0x48,0xe0,0x42]
ttsetadc 4660, 2, 1, 5

# Raw word: 0x37000001
# CHECK: ttsetrwc 1, 0, 0, 0, 0, 0{{ *}}# encoding: [0x04,0x00,0x00,0xdc]
ttsetrwc 1, 0, 0, 0, 0, 0

# Raw word: 0x37000040
# CHECK: ttsetrwc 0, 1, 0, 0, 0, 0{{ *}}# encoding: [0x00,0x01,0x00,0xdc]
ttsetrwc 0, 1, 0, 0, 0, 0

# Raw word: 0x37000400
# CHECK: ttsetrwc 0, 0, 1, 0, 0, 0{{ *}}# encoding: [0x00,0x10,0x00,0xdc]
ttsetrwc 0, 0, 1, 0, 0, 0

# Raw word: 0x37004000
# CHECK: ttsetrwc 0, 0, 0, 1, 0, 0{{ *}}# encoding: [0x00,0x00,0x01,0xdc]
ttsetrwc 0, 0, 0, 1, 0, 0

# Raw word: 0x37040000
# CHECK: ttsetrwc 0, 0, 0, 0, 1, 0{{ *}}# encoding: [0x00,0x00,0x10,0xdc]
ttsetrwc 0, 0, 0, 0, 1, 0

# Raw word: 0x37400000
# CHECK: ttsetrwc 0, 0, 0, 0, 0, 1{{ *}}# encoding: [0x00,0x00,0x00,0xdd]
ttsetrwc 0, 0, 0, 0, 0, 1

# Raw word: 0x379a54cd
# CHECK: ttsetrwc 13, 3, 5, 9, 6, 2{{ *}}# encoding: [0x34,0x53,0x69,0xde]
ttsetrwc 13, 3, 5, 9, 6, 2

# Raw word: 0x04000001
# CHECK: ttreplay 1, 0, 0, 0{{ *}}# encoding: [0x04,0x00,0x00,0x10]
ttreplay 1, 0, 0, 0

# Raw word: 0x04000002
# CHECK: ttreplay 0, 1, 0, 0{{ *}}# encoding: [0x08,0x00,0x00,0x10]
ttreplay 0, 1, 0, 0

# Raw word: 0x04000010
# CHECK: ttreplay 0, 0, 1, 0{{ *}}# encoding: [0x40,0x00,0x00,0x10]
ttreplay 0, 0, 1, 0

# Raw word: 0x04004000
# CHECK: ttreplay 0, 0, 0, 1{{ *}}# encoding: [0x00,0x00,0x01,0x10]
ttreplay 0, 0, 0, 1

# Raw word: 0x04014033
# CHECK: ttreplay 1, 1, 3, 5{{ *}}# encoding: [0xcc,0x00,0x05,0x10]
ttreplay 1, 1, 3, 5
