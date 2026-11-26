# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=FEATURE

# ISA raw words are 0x87000000/1 and 0x88000000/1. Direct issue rotates
# each word left two bits; mode one must not decode as the plain operation.
# ENC: ttsfppushc # encoding: [0x02,0x00,0x00,0x1c]
# DIS: 0: {{.*}}ttsfppushc{{$}}
# FEATURE: instruction requires the following: 'XTTTensixBH'
ttsfppushc
# ENC: ttsfppushc.replace_top # encoding: [0x06,0x00,0x00,0x1c]
# DIS: 4: {{.*}}ttsfppushc.replace_top
ttsfppushc.replace_top
# ENC: ttsfppopc # encoding: [0x02,0x00,0x00,0x20]
# DIS: 8: {{.*}}ttsfppopc{{$}}
ttsfppopc
# ENC: ttsfppopc.peek # encoding: [0x06,0x00,0x00,0x20]
# DIS: c: {{.*}}ttsfppopc.peek
ttsfppopc.peek
