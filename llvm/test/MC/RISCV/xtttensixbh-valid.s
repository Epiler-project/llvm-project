# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 %s 2>&1 | FileCheck %s --check-prefix=NOFEATURE

# DIS: 0: {{.*}}addi a0, a0, 1
# DIS: 4: {{.*}}ttsetc16 2, 1
# DIS: 8: {{.*}}ttsetc16 65535, 67
# DIS: c: {{.*}}ret
addi a0, a0, 1

# CHECK: ttsetc16 2, 1 # encoding: [0x0a,0x00,0x04,0xc8]
# NOFEATURE: instruction requires the following: 'XTTTensixBH'
ttsetc16 2, 1

# CHECK: ttsetc16 65535, 67 # encoding: [0xfe,0xff,0x0f,0xc9]
ttsetc16 65535, 67
ret
