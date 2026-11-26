# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS

# Blackhole SFPPOPC modes 3 and 4 combine the current LaneFlags with the
# saved stack top without retiring it. They retain the same opcode and encode
# the operation in Mod1 (the low four bits of the raw word).
# ENC: ttsfppopc.and_top # encoding: [0x0e,0x00,0x00,0x20]
# DIS: 0: {{.*}}ttsfppopc.and_top
ttsfppopc.and_top
# ENC: ttsfppopc.or_top # encoding: [0x12,0x00,0x00,0x20]
# DIS: 4: {{.*}}ttsfppopc.or_top
ttsfppopc.or_top
