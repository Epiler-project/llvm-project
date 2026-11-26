# RUN: split-file %s %t
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/replace.s 2>&1 | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/replace.s | FileCheck %s --check-prefix=BH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/raw.txt 2>&1 | FileCheck %s --check-prefix=RAW
# WH: instruction requires the following: 'XTTTensixBH'
# BH: ttsfppushc.replace_top # encoding: [0x06,0x00,0x00,0x1c]
# RAW: warning: invalid instruction encoding
# RAW-NOT: warning:
# RAW: ttsfppushc{{$}}
# RAW-NEXT: ttsfppopc{{$}}
# RAW-NOT: warning:

# WH PUSHC Mod1!=0 is NonContractualBehavior; the BH replacement form stays
# BH-only in both assembly feature admission and feature-aware disassembly.
#--- replace.s
ttsfppushc.replace_top
# Direct bytes exercise instruction decoding without .word data mapping symbols.
#--- raw.txt
0x02 0x00 0x00 0x1c
0x06 0x00 0x00 0x1c
0x02 0x00 0x00 0x20
