# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/asm.s 2>&1 | FileCheck %s --check-prefix=CHECK
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-cast2.txt 2>&1 | FileCheck %s --check-prefix=RAWCAST
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-cast3.txt 2>&1 | FileCheck %s --check-prefix=RAWCAST
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-shift2.txt 2>&1 | FileCheck %s --check-prefix=RAWSHIFT
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-shift3.txt 2>&1 | FileCheck %s --check-prefix=RAWSHIFT
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-shift5.txt 2>&1 | FileCheck %s --check-prefix=RAWSHIFT
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-shift7.txt 2>&1 | FileCheck %s --check-prefix=RAWSHIFT
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw-good.txt 2>&1 | FileCheck %s --check-prefix=RAWGOOD

# RAWCAST-NOT: ttsfpcast
# RAWCAST: warning: invalid instruction encoding
# RAWCAST-NOT: warning:
# RAWCAST-NOT: ttsfpcast
# RAWCAST: ttsfpcast l0, l1, 0
# RAWCAST-NOT: warning:
# RAWCAST-NEXT: ret
# RAWCAST-NOT: warning:
# RAWSHIFT-NOT: ttsfpshft
# RAWSHIFT: warning: invalid instruction encoding
# RAWSHIFT-NOT: warning:
# RAWSHIFT-NOT: ttsfpshft
# RAWSHIFT: ttsfpshft l0, l1, 0, 0
# RAWSHIFT-NOT: warning:
# RAWSHIFT-NEXT: ret
# RAWSHIFT-NOT: warning:
# RAWGOOD-NOT: warning:
# RAWGOOD: ttsfpcast l0, l1, 0
# RAWGOOD-NOT: warning:
# RAWGOOD: ttsfpshft l0, l1, 0, 0
# RAWGOOD-NOT: warning:
# RAWGOOD: ret
# RAWGOOD-NOT: warning:

#--- asm.s

# These BH modes must stay unavailable on WH after modes0/1 are received.
ttsfpcast l0, l1, 2
# CHECK: error: Wormhole SFPCAST operand 3 must be in [0, 1]
ttsfpcast l0, l1, 3
# CHECK: error: Wormhole SFPCAST operand 3 must be in [0, 1]
ttsfpshft l0, l1, 0, 2
# CHECK: error: Wormhole SFPSHFT operand 4 must be in [0, 1]
ttsfpshft l0, l1, 0, 3
# CHECK: error: Wormhole SFPSHFT operand 4 must be in [0, 1]
ttsfpshft l0, l1, 0, 5
# CHECK: error: Wormhole SFPSHFT operand 4 must be in [0, 1]
ttsfpshft l0, l1, 0, 7
# CHECK: error: Wormhole SFPSHFT operand 4 must be in [0, 1]

# Literal encoded forms pair each rejected WH mode with a legal neighbor.
#--- raw-cast2.txt
0x0a 0x04 0x00 0x40
0x02 0x04 0x00 0x40
0x67 0x80 0x00 0x00
#--- raw-cast3.txt
0x0e 0x04 0x00 0x40
0x02 0x04 0x00 0x40
0x67 0x80 0x00 0x00
#--- raw-shift2.txt
0x09 0x04 0x00 0xe8
0x01 0x04 0x00 0xe8
0x67 0x80 0x00 0x00
#--- raw-shift3.txt
0x0d 0x04 0x00 0xe8
0x01 0x04 0x00 0xe8
0x67 0x80 0x00 0x00
#--- raw-shift5.txt
0x15 0x04 0x00 0xe8
0x01 0x04 0x00 0xe8
0x67 0x80 0x00 0x00
#--- raw-shift7.txt
0x1d 0x04 0x00 0xe8
0x01 0x04 0x00 0xe8
0x67 0x80 0x00 0x00
#--- raw-good.txt
0x02 0x04 0x00 0x40
0x01 0x04 0x00 0xe8
0x67 0x80 0x00 0x00
