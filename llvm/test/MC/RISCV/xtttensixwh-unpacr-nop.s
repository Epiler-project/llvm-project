# RUN: split-file %s %t
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -show-encoding %t/valid.s | FileCheck %s --check-prefix=WH
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %t/valid.s | FileCheck %s --check-prefix=BH
# RUN: not llvm-mc -triple=riscv32 %t/valid.s 2>&1 | FileCheck %s --check-prefix=NOFEATURE
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -filetype=obj %t/valid.s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=DIS
# RUN: not llvm-mc -triple=riscv32 -mattr=+xtttensixwh %t/invalid.s 2>&1 | FileCheck %s --check-prefix=INVALID
# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixwh -disassemble %t/bh-encoding.txt 2>&1 | FileCheck %s --check-prefix=REJECT-BH

# WormholeB0/UNPACR_NOP_SETDVALID.md defines raw 0x43000007 or 0x43800007.
# This is only the sequenced source-bank handoff. It performs no wait itself.
# Its direct issue rotates the raw word left by two, independently of BH's
# separated Set_Dvalid field at bits 11:8. The LLVM formal operand ABI stays
# unchanged; selected-bank wait-and-zero has a separate exact logical form.
# WH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0{{ *}}# encoding: [0x1d,0x00,0x00,0x0c]
# WH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1{{ *}}# encoding: [0x1d,0x00,0x00,0x0e]
# BH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0{{ *}}# encoding: [0x01,0x04,0x00,0x0c]
# BH: ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1{{ *}}# encoding: [0x01,0x04,0x00,0x0e]
# NOFEATURE: error: instruction requires the following: 'XTTTensixBH' or 'XTTTensixWH'
# DIS: 0: {{.*}}ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0
# DIS: 4: {{.*}}ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1
# INVALID: error: Wormhole UNPACR_NOP does not match an admitted logical form
# INVALID: error: Wormhole UNPACR_NOP field src_clrval_ctrl must be in [0, 0]
# INVALID: error: Wormhole UNPACR_NOP field bank_clr_ctrl must be in [0, 0]
# INVALID: error: Wormhole UNPACR_NOP does not match an admitted logical form
# INVALID: error: Wormhole UNPACR_NOP field clr_to1_fmt_ctrl must be in [0, 0]
# INVALID: error: Wormhole UNPACR_NOP does not match an admitted logical form
# INVALID: error: Wormhole UNPACR_NOP field set_dvalid must be in [0, 1]
# INVALID: error: Wormhole UNPACR_NOP field msg_clr_cnt must be in [0, 0]
# INVALID: error: Wormhole UNPACR_NOP field stream_id must be in [0, 0]
# REJECT-BH: warning: invalid instruction encoding
# REJECT-BH-NOT: ttunpacr_nop
# REJECT-BH: ret

#--- valid.s
ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 0
ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 0, 1

#--- invalid.s
ttunpacr_nop 1, 0, 0, 0, 0, 1, 0, 0, 0
ttunpacr_nop 0, 1, 0, 0, 0, 1, 0, 0, 0
ttunpacr_nop 0, 0, 1, 0, 0, 1, 0, 0, 0
ttunpacr_nop 0, 0, 0, 1, 0, 1, 0, 0, 0
ttunpacr_nop 0, 0, 0, 0, 1, 1, 0, 0, 0
ttunpacr_nop 0, 0, 0, 0, 0, 0, 0, 0, 0
ttunpacr_nop 0, 0, 0, 0, 0, 2, 0, 0, 0
ttunpacr_nop 0, 0, 0, 0, 0, 1, 1, 0, 0
ttunpacr_nop 0, 0, 0, 0, 0, 1, 0, 1, 0

#--- bh-encoding.txt
0x01 0x04 0x00 0x0c 0x67 0x80 0x00 0x00
