# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/good.s | FileCheck %s --check-prefix=ENC
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -filetype=obj %t/good.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixbh - | FileCheck %s --check-prefix=DIS
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/overlap.txt | FileCheck %s --check-prefix=OVERLAP
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/reserved-mode.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/vector-high-descale.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/float-descale.txt 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -disassemble %t/reserved-round.txt 2>&1 | FileCheck %s --check-prefix=INVALID

# Immediate descale zero overlaps the vector form's fixed high field. Mode
# selects the form: vector modes are 4/5; the remaining supported modes use an
# immediate. Check every supported mode with zero descale, then nonzero fields.
# Literal bytes follow the Blackhole raw fields rotated left by two.
#--- good.s
# ENC: ttsfpstochrnd.i l3, l0, 0, 0, 0 # encoding: [0xc2,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 0, 0
ttsfpstochrnd.i l3, l0, 0, 0, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 1, 0 # encoding: [0xc6,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 1, 0
ttsfpstochrnd.i l3, l0, 0, 1, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 2, 0 # encoding: [0xca,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 2, 0
ttsfpstochrnd.i l3, l0, 0, 2, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 3, 0 # encoding: [0xce,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 3, 0
ttsfpstochrnd.i l3, l0, 0, 3, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 6, 0 # encoding: [0xda,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 6, 0
ttsfpstochrnd.i l3, l0, 0, 6, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 7, 0 # encoding: [0xde,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 7, 0
ttsfpstochrnd.i l3, l0, 0, 7, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 12, 0 # encoding: [0xf2,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 12, 0
ttsfpstochrnd.i l3, l0, 0, 12, 0
# ENC: ttsfpstochrnd.i l3, l0, 0, 13, 0 # encoding: [0xf6,0x00,0x00,0x38]
# DIS: ttsfpstochrnd.i l3, l0, 0, 13, 0
ttsfpstochrnd.i l3, l0, 0, 13, 0
# ENC: ttsfpstochrnd.v l2, l3, l0, 4, 0 # encoding: [0x92,0x0c,0x00,0x38]
# DIS: ttsfpstochrnd.v l2, l3, l0, 4, 0
ttsfpstochrnd.v l2, l3, l0, 4, 0
# ENC: ttsfpstochrnd.v l2, l3, l4, 5, 2 # encoding: [0x96,0x0c,0x01,0x39]
# DIS: ttsfpstochrnd.v l2, l3, l4, 5, 2
ttsfpstochrnd.v l2, l3, l4, 5, 2
# ENC: ttsfpstochrnd.i l2, l3, 31, 12, 1 # encoding: [0xb2,0xcc,0xfc,0x38]
# DIS: ttsfpstochrnd.i l2, l3, 31, 12, 1
ttsfpstochrnd.i l2, l3, 31, 12, 1
# ENC: ttsfpstochrnd.i l2, l3, 31, 13, 2 # encoding: [0xb6,0xcc,0x7c,0x39]
# DIS: ttsfpstochrnd.i l2, l3, 31, 13, 2
ttsfpstochrnd.i l2, l3, 31, 13, 2

# These literal words also occur in native clamped conversion, signed/unsigned
# affine and BF16 reciprocal kernels. They must decode as immediate forms.
# OVERLAP: ttsfpstochrnd.i l3, l0, 0, 6, 0
# OVERLAP-NEXT: ttsfpstochrnd.i l2, l1, 0, 7, 0
# OVERLAP-NEXT: ttsfpstochrnd.i l2, l0, 0, 6, 0
# OVERLAP-NEXT: ttsfpstochrnd.i l3, l3, 0, 1, 0
#--- overlap.txt
0xda 0x00 0x00 0x38 0x9e 0x44 0x00 0x38 0x9a 0x00 0x00 0x38 0xc6 0xcc 0x00 0x38

# Reject malformed words as one four-byte instruction and retain the following
# RV32 return. Form discrimination must not weaken mode/descale/round checks.
# INVALID: warning: invalid instruction encoding
# INVALID-NOT: ttsfp
# INVALID: ret
#--- reserved-mode.txt
0xe2 0x00 0x00 0x38 0x67 0x80 0x00 0x00
#--- vector-high-descale.txt
0xd2 0x00 0x04 0x38 0x67 0x80 0x00 0x00
#--- float-descale.txt
0xda 0x00 0x04 0x38 0x67 0x80 0x00 0x00
#--- reserved-round.txt
0xda 0x00 0x80 0x39 0x67 0x80 0x00 0x00
