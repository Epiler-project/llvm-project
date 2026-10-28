# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/mixed.txt | FileCheck %s --check-prefix=MIXED
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/reserved.txt 2>&1 | FileCheck %s --check-prefix=RESERVED
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/unsupported.txt 2>&1 | FileCheck %s --check-prefix=REJECTED
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/truncated.txt 2>&1 | FileCheck %s --check-prefix=TRUNCATED
# MIXED: addi a0, a0, 1
# MIXED-NEXT: ttsemget 255
# MIXED-NEXT: ret
# RESERVED: warning: invalid instruction encoding
# RESERVED-NOT: ttsemget
# RESERVED: ret
# REJECTED: warning: invalid instruction encoding
# REJECTED-NOT: ttsetc16
# REJECTED: ret
# TRUNCATED: warning: invalid instruction encoding
# TRUNCATED-NOT: ttsemget

#--- mixed.txt
0x13 0x05 0x15 0x00 0xf2 0x0f 0x00 0x94 0x67 0x80 0x00 0x00
#--- reserved.txt
0x02 0x10 0x00 0x94 0x67 0x80 0x00 0x00
#--- unsupported.txt
0x0a 0x00 0xe4 0xc8 0x67 0x80 0x00 0x00
#--- truncated.txt
0xf2 0x0f 0x00
