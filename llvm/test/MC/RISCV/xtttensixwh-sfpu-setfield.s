# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# Shared WH/BH setter field positions; rotate raw word left by two for issue.

# ENC: ttsfpsetexp l2, c9, 0, 0 # encoding: [0x82,0x24,0x00,0x08]
# DIS: {{.*}}ttsfpsetexp l2, c9, 0, 0
ttsfpsetexp l2, c9, 0, 0

# ENC: ttsfpsetexp l2, c9, 0, 1 # encoding: [0x86,0x24,0x00,0x08]
# DIS: {{.*}}ttsfpsetexp l2, c9, 0, 1
ttsfpsetexp l2, c9, 0, 1

# ENC: ttsfpsetexp l2, c9, 255, 1 # encoding: [0x86,0xe4,0x3f,0x08]
# DIS: {{.*}}ttsfpsetexp l2, c9, 255, 1
ttsfpsetexp l2, c9, 255, 1

# ENC: ttsfpsetexp l2, c9, 0, 2 # encoding: [0x8a,0x24,0x00,0x08]
# DIS: {{.*}}ttsfpsetexp l2, c9, 0, 2
ttsfpsetexp l2, c9, 0, 2

# ENC: ttsfpsetexp l2, c9, 0, 3 # encoding: [0x8e,0x24,0x00,0x08]
# DIS: {{.*}}ttsfpsetexp l2, c9, 0, 3
ttsfpsetexp l2, c9, 0, 3

# ENC: ttsfpsetexp l2, c9, 255, 3 # encoding: [0x8e,0xe4,0x3f,0x08]
# DIS: {{.*}}ttsfpsetexp l2, c9, 255, 3
ttsfpsetexp l2, c9, 255, 3

# ENC: ttsfpsetman l2, c9, 0, 0 # encoding: [0x82,0x24,0x00,0x0c]
# DIS: {{.*}}ttsfpsetman l2, c9, 0, 0
ttsfpsetman l2, c9, 0, 0

# ENC: ttsfpsetman l2, c9, 0, 1 # encoding: [0x86,0x24,0x00,0x0c]
# DIS: {{.*}}ttsfpsetman l2, c9, 0, 1
ttsfpsetman l2, c9, 0, 1

# ENC: ttsfpsetman l2, c9, 4095, 1 # encoding: [0x86,0xe4,0xff,0x0f]
# DIS: {{.*}}ttsfpsetman l2, c9, 4095, 1
ttsfpsetman l2, c9, 4095, 1

# ENC: ttsfpsetsgn l2, c9, 0, 0 # encoding: [0x82,0x24,0x00,0x24]
# DIS: {{.*}}ttsfpsetsgn l2, c9, 0, 0
ttsfpsetsgn l2, c9, 0, 0

# ENC: ttsfpsetsgn l2, c9, 0, 1 # encoding: [0x86,0x24,0x00,0x24]
# DIS: {{.*}}ttsfpsetsgn l2, c9, 0, 1
ttsfpsetsgn l2, c9, 0, 1

# ENC: ttsfpsetsgn l2, c9, 1, 1 # encoding: [0x86,0x64,0x00,0x24]
# DIS: {{.*}}ttsfpsetsgn l2, c9, 1, 1
ttsfpsetsgn l2, c9, 1, 1
