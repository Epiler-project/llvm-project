# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/good.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/good.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/bad.s 2>&1 | FileCheck %s --check-prefix=BAD
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved2.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved3.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved10.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved11.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved12.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved13.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved19.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved20.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved21.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved22.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-reserved23.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-count0.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-count33.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-range.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-exec-without-load.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-bh-wide.txt 2>&1 | FileCheck %s --check-prefix=RAW
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-good.txt 2>&1 | FileCheck %s --check-prefix=RAW-GOOD
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/raw-bh-wide.txt 2>&1 | FileCheck %s --check-prefix=BH-RAW
#
# Literal raw words follow official REPLAY.md / Bits32_REPLAY: Load bit0,
# Exec bit1, Count bits9:4, Index bits18:14 and opcode0x04. Direct issue rotates
# the complete word left by two, then stores it little endian. The eleven
# reserved-bit fixtures each change exactly one of bits2-3,10-13,19-23 in a
# legal record-and-execute word. Bit19 is also the first out-of-range Index.
# Count0/33, crossing the last slot and Exec without Load are noncontractual.
# Each rejection is followed by a legal REPLAY and ret: consume all four bytes
# and do not silently accept, mask, or resynchronize within the invalid word.
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning: invalid instruction encoding
# RAW-NOT: ttreplay
# RAW-NOT: {{^[[:space:]]*[a-z]}}
# RAW: ttreplay 1, 1, 1, 0
# RAW-NEXT: ret
# RAW-NOT: warning: invalid instruction encoding
#
# Positive raw decoding prevents an unsupported WH REPLAY feature from passing
# the rejection checks. Cover execute, record-only at Index31 and Count32.
# RAW-GOOD-NOT: warning: invalid instruction encoding
# RAW-GOOD: ttreplay 1, 1, 1, 0
# RAW-GOOD-NEXT: ttreplay 0, 0, 1, 0
# RAW-GOOD-NEXT: ttreplay 1, 0, 1, 31
# RAW-GOOD-NEXT: ttreplay 1, 1, 32, 0
# RAW-GOOD-NEXT: ret
# RAW-GOOD-NOT: warning: invalid instruction encoding
#
# Keep BH's broad raw field envelope. WH-only bounds must not narrow it.
# BH-RAW-NOT: warning: invalid instruction encoding
# BH-RAW: ttreplay 1, 7, 1023, 1023
# BH-RAW-NEXT: ttreplay 1, 1, 1, 0
# BH-RAW-NEXT: ret
# BH-RAW-NOT: warning: invalid instruction encoding
#--- good.s
# ENC: ttreplay 1, 1, 1, 0{{ *}}# encoding: [0x4c,0x00,0x00,0x10]
# DIS: ttreplay 1, 1, 1, 0
ttreplay 1, 1, 1, 0
# ENC: ttreplay 0, 0, 1, 0{{ *}}# encoding: [0x40,0x00,0x00,0x10]
# DIS: ttreplay 0, 0, 1, 0
ttreplay 0, 0, 1, 0
# ENC: ttreplay 1, 0, 1, 31{{ *}}# encoding: [0x44,0x00,0x1f,0x10]
# DIS: ttreplay 1, 0, 1, 31
ttreplay 1, 0, 1, 31
# ENC: ttreplay 1, 1, 32, 0{{ *}}# encoding: [0x0c,0x08,0x00,0x10]
# DIS: ttreplay 1, 1, 32, 0
ttreplay 1, 1, 32, 0
#--- bad.s
ttreplay 1, 1, 0, 0
# BAD: error: Wormhole REPLAY field len must be in [1, 32]
ttreplay 1, 1, 33, 0
# BAD: error: Wormhole REPLAY field len must be in [1, 32]
ttreplay 1, 1, 1, 32
# BAD: error: Wormhole REPLAY field start_idx must be in [0, 31]
ttreplay 1, 2, 1, 0
# BAD: error: Wormhole REPLAY field execute_while_loading must be in [0, 1]
ttreplay 0, 1, 1, 0
# BAD: error: Wormhole REPLAY does not match an admitted logical form
ttreplay 1, 1, 2, 31
# BAD: error: Tensix replay range must fit 32 instruction slots

#--- raw-good.txt
0x4c 0x00 0x00 0x10 0x40 0x00 0x00 0x10 0x44 0x00 0x1f 0x10 0x0c 0x08 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved2.txt
0x5c 0x00 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved3.txt
0x6c 0x00 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved10.txt
0x4c 0x10 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved11.txt
0x4c 0x20 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved12.txt
0x4c 0x40 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved13.txt
0x4c 0x80 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved19.txt
0x4c 0x00 0x20 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved20.txt
0x4c 0x00 0x40 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved21.txt
0x4c 0x00 0x80 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved22.txt
0x4c 0x00 0x00 0x11 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-reserved23.txt
0x4c 0x00 0x00 0x12 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-count0.txt
0x0c 0x00 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-count33.txt
0x4c 0x08 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-range.txt
0x8c 0x00 0x1f 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-exec-without-load.txt
0x48 0x00 0x00 0x10 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00

#--- raw-bh-wide.txt
0xfc 0xff 0xff 0x13 0x4c 0x00 0x00 0x10 0x67 0x80 0x00 0x00
