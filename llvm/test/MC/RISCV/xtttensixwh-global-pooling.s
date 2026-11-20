# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --mattr=+xtttensixwh --no-print-imm-hex - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=BAD

# Both LLK assembly.yaml files assign GMPOOL 0x33 and ELWSUB 0x30.
# llk_math_reduce.h uses dimension 1 (16x16) with indices disabled.
# The documented single-row GAPOOL and GMPOOL column argmax forms exercise
# the independent dimension and index bits in the same five-field ABI.
# ENC: ttgmpool 13, 0, 2, 1, 3{{ *}}# encoding: [0x34,0x00,0x24,0xcf]
# ENC: ttgmpool 13, 1, 2, 1, 3{{ *}}# encoding: [0x34,0x00,0x25,0xcf]
# ENC: ttgapool 13, 0, 2, 0, 3{{ *}}# encoding: [0x34,0x00,0x04,0xd3]
# ENC: ttgapool 13, 0, 2, 1, 3{{ *}}# encoding: [0x34,0x00,0x24,0xd3]
# DIS: ttgmpool 13, 0, 2, 1, 3
# DIS-NEXT: {{.*}}ttgmpool 13, 1, 2, 1, 3
# DIS-NEXT: {{.*}}ttgapool 13, 0, 2, 0, 3
# DIS-NEXT: {{.*}}ttgapool 13, 0, 2, 1, 3
# BAD: error: Wormhole GAPOOL field max_pool_index_en must be in [0, 0]
# BAD: error: Wormhole GMPOOL field pool_addr_mode must be in [0, 3]
# BAD: error: Wormhole GAPOOL field dst must be in [0, 1023]
# BAD: error: {{.*}}GMPOOL{{.*}}
#--- valid.s
ttgmpool 13, 0, 2, 1, 3
ttgmpool 13, 1, 2, 1, 3
ttgapool 13, 0, 2, 0, 3
ttgapool 13, 0, 2, 1, 3
#--- invalid.s
ttgapool 0, 1, 0, 1, 0
ttgmpool 0, 0, 4, 1, 0
ttgapool 1024, 0, 0, 0, 0
ttgmpool 0, 0, 0, 2, 0
