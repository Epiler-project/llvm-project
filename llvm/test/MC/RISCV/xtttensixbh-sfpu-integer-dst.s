# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/good.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/good.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %t/bad.s 2>&1 | FileCheck %s --check-prefix=BAD
# Each literal mode has its own direction-specific Dst conversion.
# These cases verify encoding, not data initialization or completed access.
#--- good.s
# ENC: ttsfpload l3, 513, 7, 5 # encoding: [0x05,0x88,0xd7,0xc0]
# DIS: ttsfpload l3, 513, 7, 5
ttsfpload l3, 513, 7, 5
# ENC: ttsfpstore l3, 1023, 7, 5 # encoding: [0xfd,0x8f,0xd7,0xc8]
# DIS: ttsfpstore l3, 1023, 7, 5
ttsfpstore l3, 1023, 7, 5
# ENC: ttsfpload l3, 513, 7, 6 # encoding: [0x05,0x88,0xdb,0xc0]
# DIS: ttsfpload l3, 513, 7, 6
ttsfpload l3, 513, 7, 6
# ENC: ttsfpstore l3, 1023, 7, 6 # encoding: [0xfd,0x8f,0xdb,0xc8]
# DIS: ttsfpstore l3, 1023, 7, 6
ttsfpstore l3, 1023, 7, 6
# ENC: ttsfpload l3, 513, 7, 8 # encoding: [0x05,0x88,0xe3,0xc0]
# DIS: ttsfpload l3, 513, 7, 8
ttsfpload l3, 513, 7, 8
# ENC: ttsfpstore l3, 1023, 7, 8 # encoding: [0xfd,0x8f,0xe3,0xc8]
# DIS: ttsfpstore l3, 1023, 7, 8
ttsfpstore l3, 1023, 7, 8
#--- bad.s
ttsfpload l3, 1024, 7, 5
# BAD: immediate must be an integer in the range [0, 1023]
ttsfpstore l3, 0, 8, 6
# BAD: immediate must be an integer in the range [0, 7]
ttsfpload l3, 0, 7, 7
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l3, 0, 7, 7
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpload l3, 0, 7, 9
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l3, 0, 7, 9
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpload l3, 0, 7, 10
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l3, 0, 7, 10
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpload l3, 0, 7, 11
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l3, 0, 7, 11
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpload l3, 0, 7, 12
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l3, 0, 7, 12
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpload l3, 0, 7, 13
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l3, 0, 7, 13
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
