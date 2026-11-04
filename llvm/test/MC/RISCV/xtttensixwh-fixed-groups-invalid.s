# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/copy.s 2>&1 | FileCheck %s --check-prefix=COPY
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/mode4.s 2>&1 | FileCheck %s --check-prefix=MODE4
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw.txt 2>&1 | FileCheck %s --check-prefix=RAW
# COPY: error: immediate must be an integer in the range [0, 1]
# MODE4: error: Wormhole SFPSHFT2 operand 3 must be in [0, 3]
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning:
# RAW-NOT: ttsfpshft2 l0, l1, 4
# RAW: ttsfpshft2.copy4 0
# RAW-NEXT: ret
# RAW-NOT: warning:
#
# Copy4 cannot stand in for rotate4's missing VC. Buggy WH unary mode4 is
# not admitted by opening the distinct fixed mode0/1 and mode2 forms.
#--- copy.s
ttsfpshft2.copy4 2
#--- mode4.s
ttsfpshft2 l0, l1, 4
#--- raw.txt
0x12 0x04 0x00 0x50
0x02 0x00 0x00 0x50
0x67 0x80 0x00 0x00
