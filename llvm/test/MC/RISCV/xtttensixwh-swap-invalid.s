# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/asm.s 2>&1 | FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw.txt 2>&1 | FileCheck %s --check-prefix=RAW
# ASM: error: SFPSWAP mode must be one of {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
# RAW-NOT: ttsfpswap
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning:
# RAW-NOT: ttsfpswap
# RAW: ttsfpswap l0, l1, 0
# RAW-NOT: warning:
# RAW-NEXT: ret
# RAW-NOT: warning:

#--- asm.s
ttsfpswap l0, l1, 10

#--- raw.txt
0x2a 0x04 0x00 0x48
0x02 0x04 0x00 0x48
0x67 0x80 0x00 0x00
