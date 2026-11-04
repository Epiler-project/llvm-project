# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=FEATURE
# FEATURE: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'

# WH/BH share these contractual raw forms. Independent expected bytes use
# raw opcode/VC/Mod1 from the official WH ISA, rotated left two for direct issue.
# Nonzero POPC modes peek; only mode zero retires a stack entry.
# ENC: ttsfpsetcc.lt0 l0 # encoding: [0x01,0x00,0x00,0xec]
# DIS: 0: {{.*}}ttsfpsetcc.lt0 l0{{$}}
ttsfpsetcc.lt0 l0
# ENC: ttsfpsetcc.ne0 l1 # encoding: [0x09,0x04,0x00,0xec]
# DIS: 4: {{.*}}ttsfpsetcc.ne0 l1{{$}}
ttsfpsetcc.ne0 l1
# ENC: ttsfpsetcc.ge0 c10 # encoding: [0x11,0x28,0x00,0xec]
# DIS: 8: {{.*}}ttsfpsetcc.ge0 c10{{$}}
ttsfpsetcc.ge0 c10
# ENC: ttsfpsetcc.eq0 c15 # encoding: [0x19,0x3c,0x00,0xec]
# DIS: c: {{.*}}ttsfpsetcc.eq0 c15{{$}}
ttsfpsetcc.eq0 c15
# ENC: ttsfppushc # encoding: [0x02,0x00,0x00,0x1c]
# DIS: 10: {{.*}}ttsfppushc{{$}}
ttsfppushc
# ENC: ttsfppopc.peek # encoding: [0x06,0x00,0x00,0x20]
# DIS: 14: {{.*}}ttsfppopc.peek{{$}}
ttsfppopc.peek
# ENC: ttsfppopc.and_top # encoding: [0x0e,0x00,0x00,0x20]
# DIS: 18: {{.*}}ttsfppopc.and_top{{$}}
ttsfppopc.and_top
# ENC: ttsfppopc.or_top # encoding: [0x12,0x00,0x00,0x20]
# DIS: 1c: {{.*}}ttsfppopc.or_top{{$}}
ttsfppopc.or_top
# ENC: ttsfpcompc # encoding: [0x02,0x00,0x00,0x2c]
# DIS: 20: {{.*}}ttsfpcompc{{$}}
ttsfpcompc
# ENC: ttsfppopc # encoding: [0x02,0x00,0x00,0x20]
# DIS: 24: {{.*}}ttsfppopc{{$}}
ttsfppopc
