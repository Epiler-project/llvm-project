# RUN: llvm-mc -triple=riscv32 -mattr=+xtttensixbh -show-encoding %s | FileCheck %s
# Every ordinary instruction has a formal MC record. Boundary encodings are
# independent constants; no runtime instruction-schema generator is required.

# CHECK: ttadddmareg 63, 63, 2047, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x63]
ttadddmareg 63, 63, 2047, 1

# CHECK: ttaddrcrxy 63, 7, 7, 7, 63, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x4f]
ttaddrcrxy 63, 7, 7, 7, 63, 7

# CHECK: ttaddrcrzw 63, 7, 7, 7, 63, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x5b]
ttaddrcrzw 63, 7, 7, 7, 63, 7

# CHECK: ttapool3s1 16383, 1, 127, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x97]
ttapool3s1 16383, 1, 127, 3

# CHECK: ttapool3s2 16383, 1, 127, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xcb]
ttapool3s2 16383, 1, 127, 3

# CHECK: ttatcas 63, 63, 3, 15, 31, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x93]
ttatcas 63, 63, 3, 15, 31, 1

# CHECK: ttatgetm 16777215{{ *}}# encoding: [0xfe,0xff,0xff,0x83]
ttatgetm 16777215

# CHECK: ttatincget 63, 63, 3, 511, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x87]
ttatincget 63, 63, 3, 511, 1

# CHECK: ttatincgetptr 63, 63, 3, 15, 15, 1, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x8b]
ttatincgetptr 63, 63, 3, 15, 15, 1, 1

# CHECK: ttatrelm 16777215{{ *}}# encoding: [0xfe,0xff,0xff,0x87]
ttatrelm 16777215

# CHECK: ttatswap 63, 255, 511, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x8f]
ttatswap 63, 255, 511, 1

# CHECK: ttbitwopdmareg 63, 63, 63, 31, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x6f]
ttbitwopdmareg 63, 63, 63, 31, 1

# CHECK: ttcfgshiftmask 255, 3, 31, 31, 7, 1{{ *}}# encoding: [0xfe,0xff,0xff,0xe3]
ttcfgshiftmask 255, 3, 31, 31, 7, 1

# CHECK: ttcleardvalid 4194303, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xdb]
ttcleardvalid 4194303, 3

# CHECK: ttclrexphist{{ *}}# encoding: [0x00,0x00,0x00,0x84]
ttclrexphist

# CHECK: ttcmpdmareg 63, 63, 63, 31, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x77]
ttcmpdmareg 63, 63, 63, 31, 1

# CHECK: ttconv3s1 16383, 7, 31, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x8b]
ttconv3s1 16383, 7, 31, 3

# CHECK: ttconv3s2 16383, 7, 31, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x8f]
ttconv3s2 16383, 7, 31, 3

# CHECK: ttdmanop{{ *}}# encoding: [0x01,0x00,0x00,0x80]
ttdmanop

# CHECK: ttdotpv 16383, 31, 3, 1, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xa7]
ttdotpv 16383, 31, 3, 1, 3

# CHECK: ttelwadd 16383, 31, 3, 1, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xa3]
ttelwadd 16383, 31, 3, 1, 3

# CHECK: ttelwmul 16383, 31, 3, 1, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x9f]
ttelwmul 16383, 31, 3, 1, 3

# CHECK: ttelwsub 16383, 31, 3, 1, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xc3]
ttelwsub 16383, 31, 3, 1, 3

# CHECK: ttflushdma 16777215{{ *}}# encoding: [0xfd,0xff,0xff,0x1b]
ttflushdma 16777215

# CHECK: ttgapool 16383, 1, 15, 7, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xd3]
ttgapool 16383, 1, 15, 7, 3

# CHECK: ttgatesrcrst 1, 8388607{{ *}}# encoding: [0xfc,0xff,0xff,0xd7]
ttgatesrcrst 1, 8388607

# CHECK: ttgmpool 16383, 1, 15, 7, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xcf]
ttgmpool 16383, 1, 15, 7, 3

# CHECK: ttincadcxy 7, 7, 7, 63, 7{{ *}}# encoding: [0x01,0xff,0xff,0x4b]
ttincadcxy 7, 7, 7, 63, 7

# CHECK: ttincadczw 7, 7, 7, 63, 7{{ *}}# encoding: [0x01,0xff,0xff,0x57]
ttincadczw 7, 7, 7, 63, 7

# CHECK: ttincrwc 15, 15, 15, 63{{ *}}# encoding: [0x00,0xff,0xff,0xe3]
ttincrwc 15, 15, 15, 63

# CHECK: ttloadind 63, 63, 3, 255, 3{{ *}}# encoding: [0xfd,0xff,0xff,0x27]
ttloadind 63, 63, 3, 255, 3

# CHECK: ttloadreg 262143, 63{{ *}}# encoding: [0xfd,0xff,0xff,0xa3]
ttloadreg 262143, 63

# CHECK: ttmfconv3s1 16383, 7, 31, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xeb]
ttmfconv3s1 16383, 7, 31, 3

# CHECK: ttmop 65535, 127, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x07]
ttmop 65535, 127, 1

# CHECK: ttmop_cfg 16777215{{ *}}# encoding: [0xfc,0xff,0xff,0x0f]
ttmop_cfg 16777215

# CHECK: ttmova2d 4095, 3, 7, 63, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x4b]
ttmova2d 4095, 3, 7, 63, 1

# CHECK: ttmovb2a 4095, 3, 7, 127{{ *}}# encoding: [0xfc,0xff,0xff,0x2f]
ttmovb2a 4095, 3, 7, 127

# CHECK: ttmovb2d 2047, 7, 7, 63, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x4f]
ttmovb2d 2047, 7, 7, 63, 1

# CHECK: ttmovd2a 4095, 3, 7, 63, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x23]
ttmovd2a 4095, 3, 7, 63, 1

# CHECK: ttmovd2b 4095, 3, 7, 63, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x2b]
ttmovd2b 4095, 3, 7, 63, 1

# CHECK: ttmovdbga2d 4095, 3, 7, 63, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x27]
ttmovdbga2d 4095, 3, 7, 63, 1

# CHECK: ttmovdbgb2d 2047, 7, 7, 63, 1{{ *}}# encoding: [0xfc,0xff,0xff,0x33]
ttmovdbgb2d 2047, 7, 7, 63, 1

# CHECK: ttmpool3s1 16383, 1, 127, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x93]
ttmpool3s1 16383, 1, 127, 3

# CHECK: ttmpool3s2 16383, 1, 127, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xc7]
ttmpool3s2 16383, 1, 127, 3

# CHECK: ttmuldmareg 63, 63, 2047, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x6b]
ttmuldmareg 63, 63, 2047, 1

# CHECK: ttmvmul 16383, 31, 7, 3{{ *}}# encoding: [0xfc,0xff,0xff,0x9b]
ttmvmul 16383, 31, 7, 3

# CHECK: ttnop{{ *}}# encoding: [0x00,0x00,0x00,0x08]
ttnop

# CHECK: ttpacr 1, 1, 3, 7, 1, 15, 1, 3, 3, 1, 7, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x07]
ttpacr 1, 1, 3, 7, 1, 15, 1, 3, 3, 1, 7, 7

# CHECK: ttpacr_setreg 1, 1, 63, 3, 3, 1023, 1, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x2b]
ttpacr_setreg 1, 1, 63, 3, 3, 1023, 1, 1

# CHECK: ttrareb{{ *}}# encoding: [0x00,0x00,0x00,0x54]
ttrareb

# CHECK: ttrdcfg 65535, 255{{ *}}# encoding: [0xfe,0xff,0xff,0xc7]
ttrdcfg 65535, 255

# CHECK: ttreg2flop 63, 1023, 3, 3, 3, 3{{ *}}# encoding: [0xfd,0xff,0xff,0x23]
ttreg2flop 63, 1023, 3, 3, 3, 3

# CHECK: ttreplay 1, 7, 1023, 1023{{ *}}# encoding: [0xfc,0xff,0xff,0x13]
ttreplay 1, 7, 1023, 1023

# CHECK: ttresourcedecl 15, 511, 2047{{ *}}# encoding: [0xfc,0xff,0xff,0x17]
ttresourcedecl 15, 511, 2047

# CHECK: ttrmwcib0 255, 255, 255{{ *}}# encoding: [0xfe,0xff,0xff,0xcf]
ttrmwcib0 255, 255, 255

# CHECK: ttrmwcib1 255, 255, 255{{ *}}# encoding: [0xfe,0xff,0xff,0xd3]
ttrmwcib1 255, 255, 255

# CHECK: ttrmwcib2 255, 255, 255{{ *}}# encoding: [0xfe,0xff,0xff,0xd7]
ttrmwcib2 255, 255, 255

# CHECK: ttrmwcib3 255, 255, 255{{ *}}# encoding: [0xfe,0xff,0xff,0xdb]
ttrmwcib3 255, 255, 255

# CHECK: ttrstdma{{ *}}# encoding: [0x01,0x00,0x00,0x10]
ttrstdma

# CHECK: ttsemget 4194303{{ *}}# encoding: [0xf2,0xff,0xff,0x97]
ttsemget 4194303

# CHECK: ttseminit 16383, 15, 15{{ *}}# encoding: [0xf2,0xff,0xff,0x8f]
ttseminit 16383, 15, 15

# CHECK: ttsempost 4194303{{ *}}# encoding: [0xf2,0xff,0xff,0x93]
ttsempost 4194303

# CHECK: ttsemwait 3, 8191, 511{{ *}}# encoding: [0xfe,0xff,0xff,0x9b]
ttsemwait 3, 8191, 511

# CHECK: ttsetadc 262143, 3, 1, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x43]
ttsetadc 262143, 3, 1, 7

# CHECK: ttsetadcxx 1023, 2047, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x7b]
ttsetadcxx 1023, 2047, 7

# CHECK: ttsetadcxy 63, 7, 7, 7, 63, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x47]
ttsetadcxy 63, 7, 7, 7, 63, 7

# CHECK: ttsetadczw 63, 7, 7, 7, 63, 7{{ *}}# encoding: [0xfd,0xff,0xff,0x53]
ttsetadczw 63, 7, 7, 7, 63, 7

# CHECK: ttsetashrmh 1, 8388607{{ *}}# encoding: [0xfc,0xff,0xff,0x7b]
ttsetashrmh 1, 8388607

# CHECK: ttsetashrmh0 1, 8388607{{ *}}# encoding: [0xfc,0xff,0xff,0x6b]
ttsetashrmh0 1, 8388607

# CHECK: ttsetashrmh1 1, 8388607{{ *}}# encoding: [0xfc,0xff,0xff,0x6f]
ttsetashrmh1 1, 8388607

# CHECK: ttsetashrmv 16777215{{ *}}# encoding: [0xfc,0xff,0xff,0x73]
ttsetashrmv 16777215

# CHECK: ttsetc16 65535, 67{{ *}}# encoding: [0xfe,0xff,0x0f,0xc9]
ttsetc16 65535, 67

# CHECK: ttsetdmareg 127, 1, 16383, 3{{ *}}# encoding: [0xfd,0xff,0xff,0x17]
ttsetdmareg 127, 1, 16383, 3

# CHECK: ttsetdvalid 16777215{{ *}}# encoding: [0xfd,0xff,0xff,0x5f]
ttsetdvalid 16777215

# CHECK: ttsetibrwc 63, 4095, 63{{ *}}# encoding: [0xfc,0xff,0xff,0xe7]
ttsetibrwc 63, 4095, 63

# CHECK: ttsetpkedgof 15, 15, 15, 4095{{ *}}# encoding: [0xfc,0xff,0xff,0x77]
ttsetpkedgof 15, 15, 15, 4095

# CHECK: ttsetrwc 63, 15, 15, 15, 15, 3{{ *}}# encoding: [0xfc,0xff,0xff,0xdf]
ttsetrwc 63, 15, 15, 15, 15, 3

# CHECK: ttshiftdmareg 63, 63, 63, 31, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x73]
ttshiftdmareg 63, 63, 63, 31, 1

# CHECK: ttshiftxa 3, 4194303{{ *}}# encoding: [0xfc,0xff,0xff,0x5f]
ttshiftxa 3, 4194303

# CHECK: ttshiftxb 1023, 15, 1023{{ *}}# encoding: [0xfc,0xff,0xff,0x63]
ttshiftxb 1023, 15, 1023

# CHECK: ttstallwait 32767, 511{{ *}}# encoding: [0xfe,0xff,0xff,0x8b]
ttstallwait 32767, 511

# CHECK: ttstoreind 63, 63, 3, 127, 1, 1, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x9b]
ttstoreind 63, 63, 3, 127, 1, 1, 1

# CHECK: ttstorereg 262143, 63{{ *}}# encoding: [0xfd,0xff,0xff,0x9f]
ttstorereg 262143, 63

# CHECK: ttstreamwait 7, 1, 2047, 511{{ *}}# encoding: [0xfe,0xff,0xff,0x9f]
ttstreamwait 7, 1, 2047, 511

# CHECK: ttstreamwrcfg 2047, 1023, 7{{ *}}# encoding: [0xfe,0xff,0xff,0xdf]
ttstreamwrcfg 2047, 1023, 7

# CHECK: ttsubdmareg 63, 63, 2047, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x67]
ttsubdmareg 63, 63, 2047, 1

# CHECK: tttbufcmd{{ *}}# encoding: [0x01,0x00,0x00,0x2c]
tttbufcmd

# CHECK: tttrnspsrca{{ *}}# encoding: [0x00,0x00,0x00,0x50]
tttrnspsrca

# CHECK: tttrnspsrcb{{ *}}# encoding: [0x00,0x00,0x00,0x58]
tttrnspsrcb

# CHECK: ttunpacr 1, 1, 1, 1, 1, 1, 1, 1, 3, 7, 3, 255, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x0b]
ttunpacr 1, 1, 1, 1, 1, 1, 1, 1, 3, 7, 3, 255, 1

# CHECK: ttunpacr_nop 3, 3, 1, 1, 3, 15, 15, 127, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x0f]
ttunpacr_nop 3, 3, 1, 1, 3, 15, 15, 127, 1

# CHECK: ttwrcfg 32767, 1, 255{{ *}}# encoding: [0xfe,0xff,0xff,0xc3]
ttwrcfg 32767, 1, 255

# CHECK: ttxmov 8388607, 1{{ *}}# encoding: [0xfd,0xff,0xff,0x03]
ttxmov 8388607, 1

# CHECK: ttzeroacc 16383, 7, 1, 1, 31{{ *}}# encoding: [0xfc,0xff,0xff,0x43]
ttzeroacc 16383, 7, 1, 1, 31

# CHECK: ttzerosrc 3, 1, 1, 1048575{{ *}}# encoding: [0xfc,0xff,0xff,0x47]
ttzerosrc 3, 1, 1, 1048575
