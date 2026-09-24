//===-- RISCVInstrInfoTensixSFPU.td ------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SSA destination passthroughs model predication. The machine allocator owns
// all physical LReg choices and ties; scalar integers never carry LReg IDs.
class TensixUnary<bits<8> Opcode, string Name, bits<16> Modes>
    : TensixSSAInst<Opcode, (outs SFPR:$dst),
                    (ins SFPR:$old, SFPRRead:$src, uimm4:$mode),
                    Name, "$dst, $src, $mode"> {
  bits<4> dst;
  bits<4> src;
  bits<4> mode;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 3;
  let AllowedModes = Modes;
}
def TTSFPARECIP : TensixUnary<0x99, "ttsfparecip", 0x0007> {
  // Conditional reciprocal reads the old destination as its predicate source.
  let RawWord{15-12} = dst;
}
def TTSFPEXEXP : TensixUnary<0x77, "ttsfpexexp", 0x0f0f> {
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPEXMAN : TensixUnary<0x78, "ttsfpexman", 0x0003>;
def TTSFPABS : TensixUnary<0x7d, "ttsfpabs", 0x0003>;
def TTSFPLZ : TensixUnary<0x81, "ttsfplz", 0x5555> {
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPCAST : TensixUnary<0x90, "ttsfpcast", 0x000f>;

class TensixSetField<bits<8> Opcode, string Name, bits<16> Modes>
    : TensixSSAInst<Opcode, (outs SFPR:$dst),
                    (ins SFPR:$old, SFPRRead:$src, uimm16:$value, uimm4:$mode),
                    Name, "$dst, $src, $value, $mode"> {
  bits<4> dst;
  bits<4> src;
  bits<12> value;
  bits<4> mode;
  let RawWord{23-12} = value;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 4;
  let AllowedModes = Modes;
}
def TTSFPSETEXP : TensixSetField<0x82, "ttsfpsetexp", 0x000f>;
def TTSFPSETMAN : TensixSetField<0x83, "ttsfpsetman", 0x0003>;
def TTSFPSETSGN : TensixSetField<0x89, "ttsfpsetsgn", 0x0003>;

class TensixBitwise<bits<8> Opcode, string Name>
    : TensixSSAInst<Opcode, (outs SFPR:$dst),
                    (ins SFPR:$old, SFPRRead:$src), Name, "$dst, $src"> {
  bits<4> dst;
  bits<4> src;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let Constraints = "$dst = $old";
}
def TTSFPAND : TensixBitwise<0x7e, "ttsfpand">;
def TTSFPOR : TensixBitwise<0x7f, "ttsfpor">;
def TTSFPXOR : TensixBitwise<0x8d, "ttsfpxor">;
def TTSFPNOT : TensixBitwise<0x80, "ttsfpnot">;

class TensixSignedImmediate<bits<8> Opcode, string Name, bits<16> Modes>
    : TensixSSAInst<Opcode, (outs SFPR:$dst),
                    (ins SFPR:$old, SFPRRead:$src, simm12:$value, uimm4:$mode),
                    Name, "$dst, $src, $value, $mode"> {
  bits<4> dst;
  bits<4> src;
  bits<12> value;
  bits<4> mode;
  let RawWord{23-12} = value;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 4;
  let AllowedModes = Modes;
}
def TTSFPIADDI : TensixSignedImmediate<0x79, "ttsfpiadd.i", 0x0222> {
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPSHFT : TensixSignedImmediate<0x7a, "ttsfpshft", 0x00af>;

def TTSFPSHFT2 : TensixUnary<0x94, "ttsfpshft2", 0x0018>;

class TensixStochRound<string Name, bit Immediate>
    : TensixSSAInst<0x8e, (outs SFPR:$dst),
        !if(Immediate,
          (ins SFPR:$old, SFPRRead:$src, uimm8:$descale, uimm4:$mode, uimm2:$round),
          (ins SFPR:$old, SFPRRead:$src, SFPRRead:$descale, uimm4:$mode, uimm2:$round)),
        Name, "$dst, $src, $descale, $mode, $round"> {
  bits<4> dst;
  bits<4> src;
  bits<5> descale;
  bits<4> mode;
  bits<2> round;
  let RawWord{22-21} = round;
  let RawWord{20-16} = !if(Immediate, descale, 0);
  let RawWord{15-12} = !if(Immediate, src, descale{3-0});
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 4;
  let AllowedModes = !if(Immediate, 0x30cf, 0x0030);
}
def TTSFPSTOCHRNDI : TensixStochRound<"ttsfpstochrnd.i", 1>;
def TTSFPSTOCHRNDV : TensixStochRound<"ttsfpstochrnd.v", 0>;

def TTSFPSWAP : TensixSSAInst<0x92, (outs SFPR:$dst, SFPR:$other),
    (ins SFPR:$old, SFPR:$oldother, uimm4:$mode),
    "ttsfpswap", "$dst, $other, $mode"> {
  bits<4> dst;
  bits<4> other;
  bits<4> mode;
  let RawWord{11-8} = other;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old,$other = $oldother";
  let ModeOperandIndex = 4;
  let AllowedModes = 0x03ff;
}

// Group instructions name their entire fixed-register footprint. The selector
// stages SSA inputs with register copies, then reads results after the issue.
def TTSFPLUT : TensixSSAInst<0x73, (outs SFPR:$dst),
    (ins SFPR:$old, uimm4:$mode), "ttsfplut", "$dst, $mode"> {
  bits<4> dst;
  bits<4> mode;
  let RawWord{23-20} = dst;
  let RawWord{19-16} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 2;
  let AllowedModes = 0x0011;
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_CC, TT_CONFIG, TT_ISSUE];
}
def TTSFPTRANSP : TensixSSAInst<0x8c, (outs), (ins), "ttsfptransp", ""> {
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_L0, TT_L1, TT_L2, TT_L3, TT_L4, TT_L5, TT_L6, TT_L7, TT_ISSUE];
}
