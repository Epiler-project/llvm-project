# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/good.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/good.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixbh %t/bad.s 2>&1 | FileCheck %s --check-prefix=BAD
#--- good.s
# ENC: ttsfpload l3, 513, 7, 14 # encoding: [0x05,0x88,0xfb,0xc0]
# DIS: ttsfpload l3, 513, 7, 14
ttsfpload l3, 513, 7, 14
# ENC: ttsfpstore l3, 1023, 7, 14 # encoding: [0xfd,0x8f,0xfb,0xc8]
# DIS: ttsfpstore l3, 1023, 7, 14
ttsfpstore l3, 1023, 7, 14
# ENC: ttsfpload l3, 514, 7, 15 # encoding: [0x09,0x88,0xff,0xc0]
# DIS: ttsfpload l3, 514, 7, 15
ttsfpload l3, 514, 7, 15
# ENC: ttsfpstore l3, 1022, 7, 15 # encoding: [0xf9,0x8f,0xff,0xc8]
# DIS: ttsfpstore l3, 1022, 7, 15
ttsfpstore l3, 1022, 7, 15
#--- bad.s
ttsfpload l0, 1024, 7, 14
# BAD: immediate must be an integer in the range [0, 1023]
ttsfpstore l0, 0, 8, 15
# BAD: immediate must be an integer in the range [0, 7]
ttsfpload l0, 0, 7, 10
# BAD: SFPLOAD format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
ttsfpstore l0, 0, 7, 13
# BAD: SFPSTORE format must be one of {0, 1, 2, 3, 4, 5, 6, 8, 14, 15}
