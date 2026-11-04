//===-- RISCVInstrInfoTensixSFPU.td ------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Destination passthroughs model predication for both ingress paths. Legacy
// numerical SSA delegates placement to the allocator; bound ingress directly
// constructs these native instructions with verified physical defs and uses.
class TensixUnary<bits<8> Opcode, string Name, bits<16> Modes>
    : TensixPhysicalInst<Opcode, (outs SFPR:$dst),
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
// Simple exponent adjustment: unused immediate and modifier bits stay zero.
// Explicit old/result tying preserves disabled lanes without fixed placement.
def TTSFPDIVP2 : TensixPhysicalInst<0x76, (outs SFPR:$dst),
    (ins SFPR:$old, SFPRRead:$src, uimm8:$value, uimm1:$mode),
    "ttsfpdivp2", "$dst, $src, $value, $mode"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  bits<4> dst;
  bits<4> src;
  bits<8> value;
  bits<1> mode;
  let RawWord{19-12} = value;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let RawWord{0} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 4;
  let AllowedModes = 0x0003;
}

// BF16 immediate MAD-unit operations. Only direct VD is received. Bit1
// negates old VD on BH; WH reserves all modifier bits other than indirect VD,
// which is deliberately absent from this single-result contents contract.
class TensixFPImmediate<bits<8> Opcode, string Name>
    : TensixPhysicalInst<Opcode, (outs SFPR:$dst),
        (ins SFPR:$old, uimm16:$value, uimm4:$mode),
        Name, "$dst, $value, $mode"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let WormholeOperandMaxima = [-1, -1, -1, 0];
  bits<4> dst;
  bits<16> value;
  bits<4> mode;
  let RawWord{23-8} = value;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 3;
  let AllowedModes = 0x0005;
}
def TTSFPADDI : TensixFPImmediate<0x75, "ttsfpaddi">;
def TTSFPMULI : TensixFPImmediate<0x74, "ttsfpmuli">;

// BH-only 23x23 integer multiply. The fixed C9 zero suppresses the undefined
// exponent-dependent shift/add datapath. No indirect VA/VD mode is exposed.
def TTSFPMUL24 : TensixMAD<0x98,
    (ins SFPR:$old, SFPRRead:$a, SFPRRead:$b, uimm1:$mode),
    "ttsfpmul24", "$dst, $a, $b, $mode"> {
  let RawWord{11-8} = 9;
  let ModeOperandIndex = 4;
  let AllowedModes = 0x0003;
  let Uses = [TT_C9, TT_CC, TT_CONFIG, TT_ISSUE];
}

def TTSFPARECIP : TensixUnary<0x99, "ttsfparecip", 0x0007> {
  // Conditional reciprocal reads the old destination as its predicate source.
  let RawWord{15-12} = dst;
}
def TTSFPEXEXP : TensixUnary<0x77, "ttsfpexexp", 0x0f0f> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPEXMAN : TensixUnary<0x78, "ttsfpexman", 0x0003> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPABS : TensixUnary<0x7d, "ttsfpabs", 0x0003> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPLZ : TensixUnary<0x81, "ttsfplz", 0x5555> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPCAST : TensixUnary<0x90, "ttsfpcast", 0x000f> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  // WH only converts sign-magnitude integer to FP32, nearest or stochastic.
  let WormholeOperandMaxima = [-1, -1, -1, 1];
  // The shared descriptor conservatively includes the stochastic mode's PRNG.
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE, TT_PRNG];
  let Defs = [TT_ISSUE, TT_PRNG];
}

class TensixSetField<bits<8> Opcode, string Name, bits<16> Modes>
    : TensixPhysicalInst<Opcode, (outs SFPR:$dst),
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
def TTSFPSETEXP : TensixSetField<0x82, "ttsfpsetexp", 0x000f> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPSETMAN : TensixSetField<0x83, "ttsfpsetman", 0x0003> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPSETSGN : TensixSetField<0x89, "ttsfpsetsgn", 0x0003> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}

class TensixBitwise<bits<8> Opcode, string Name>
    : TensixPhysicalInst<Opcode, (outs SFPR:$dst),
                    (ins SFPR:$old, SFPRRead:$src), Name, "$dst, $src"> {
  bits<4> dst;
  bits<4> src;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let Constraints = "$dst = $old";
}
def TTSFPAND : TensixBitwise<0x7e, "ttsfpand"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPOR : TensixBitwise<0x7f, "ttsfpor"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPXOR : TensixBitwise<0x8d, "ttsfpxor"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}
def TTSFPNOT : TensixBitwise<0x80, "ttsfpnot"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
}

class TensixSignedImmediate<bits<8> Opcode, string Name, bits<16> Modes>
    : TensixPhysicalInst<Opcode, (outs SFPR:$dst),
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
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPSHFT : TensixSignedImmediate<0x7a, "ttsfpshft", 0x00af> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  // WH shifts the old destination logically, by a vector or signed Imm12.
  let WormholeOperandMaxima = [-1, -1, -1, -1, 1];
}

def TTSFPSHFT2 : TensixUnary<0x94, "ttsfpshft2", 0x0018> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  // WH mode 3 rotates each subvector of eight lanes from the pre-issue VC.
  // Mode 4 has a hardware bug on WH; retain its existing BH-only contract.
  let WormholeOperandMaxima = [-1, -1, -1, 3];
}

// Group forms read the complete previous L0..L3 state before any write. VD=0
// selects the normal enabled-lane path; it is not an extra destination.
def TTSFPSHFT2Copy4 : TensixPhysicalInst<0x94, (outs),
    (ins uimm1:$mode), "ttsfpshft2.copy4", "$mode"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  bits<1> mode;
  let RawWord{0} = mode;
  let ModeOperandIndex = 0;
  let AllowedModes = 0x0003;
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_L0, TT_L1, TT_L2, TT_L3, TT_ISSUE];
}
def TTSFPSHFT2Rotate4 : TensixPhysicalInst<0x94, (outs),
    (ins SFPRRead:$value), "ttsfpshft2.rotate4", "$value"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  bits<4> value;
  let RawWord{11-8} = value;
  let RawWord{3-0} = 2;
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_L0, TT_L1, TT_L2, TT_L3, TT_ISSUE];
}

class TensixStochRound<string Name, bit Immediate>
    : TensixPhysicalInst<0x8e, (outs SFPR:$dst),
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
  // Blackhole advances PRNG before selecting even nearest/zero rounding.
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE, TT_PRNG];
  let Defs = [TT_ISSUE, TT_PRNG];
}
def TTSFPSTOCHRNDI : TensixStochRound<"ttsfpstochrnd.i", 1>;
def TTSFPSTOCHRNDV : TensixStochRound<"ttsfpstochrnd.v", 0> {
  // Only modes 4/5 use a vector descale. Distinguish the forms before the
  // post-decode operand checks when the immediate descale field is zero.
  let mode{3-1} = 0b010;
}

def TTSFPSWAP : TensixPhysicalInst<0x92, (outs SFPR:$dst, SFPR:$other),
    (ins SFPR:$old, SFPR:$oldother, uimm4:$mode),
    "ttsfpswap", "$dst, $other, $mode"> {
  // Wormhole and Blackhole share the official SFPSWAP modes and encoding.
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
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

// Complete contents view of the same SFPSWAP word. D/C and their derived
// companion positions are separate numerical old/result values. Allocation
// uses native pair superregisters; the independent MC receiver also enforces
// primary/+4 companions and distinct low primaries. Disabled CC and index-off
// lanes retain the actual old contents.
// The ordinary instruction remains the decoder/assembler authority.
def TTSFPSWAPIndex : TensixPhysicalInst<0x92,
    (outs SFPRLow:$dst, SFPRLow:$other,
          SFPRHigh:$index, SFPRHigh:$otherindex),
    (ins SFPRLow:$old, SFPRLow:$oldother, SFPRHigh:$oldindex,
         SFPRHigh:$oldotherindex, uimm4:$mode),
    "ttsfpswap", "$dst, $other, $mode"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let isCodeGenOnly = 1;
  bits<4> dst;
  bits<4> other;
  bits<4> mode;
  let RawWord{11-8} = other;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = mode;
  let Constraints = "$dst = $old,$other = $oldother,"
                    "$index = $oldindex,$otherindex = $oldotherindex";
  let ModeOperandIndex = 8;
  let AllowedModes = 0x03ff;
}

// Group instructions name their entire fixed-register footprint. The selector
// stages legacy SSA inputs with copies and captures results. Bound ingress
// requires the complete authored group and never performs that staging.
def TTSFPLUT : TensixPhysicalInst<0x73, (outs SFPR:$dst),
    (ins SFPR:$old, uimm4:$mode), "ttsfplut", "$dst, $mode"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  bits<4> dst;
  bits<4> mode;
  let RawWord{23-20} = dst;
  let RawWord{19-16} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 2;
  let AllowedModes = 0x0011;
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_CC, TT_CONFIG, TT_ISSUE];
}
// Both architectures share the numerical and fixed-register contract, but
// only Blackhole has Mod1Mirror[19:16]. Keep this difference in the native
// encoding owner, just like the architecture-specific Dst address fields.
class TensixLUTFP32SixEncoding<bit Mirror> : InstructionEncoding {
  let Size = 4;
  let DecoderNamespace = "Tensix";
  bits<32> Inst;
  bits<32> SoftFail = 0;
  bits<32> RawWord = 0x95000000;
  bits<4> dst;
  bits<3> mode;
  let RawWord{7-4} = dst;
  let RawWord{2-0} = mode;
  let RawWord{18-16} = !if(Mirror, mode, 0);
  let Inst{31-2} = RawWord{29-0};
  let Inst{1-0} = RawWord{31-30};
}
class TensixLUTFP32ThreeEncoding<bit Mirror> : InstructionEncoding {
  let Size = 4;
  let DecoderNamespace = "Tensix";
  bits<32> Inst;
  bits<32> SoftFail = 0;
  bits<32> RawWord = 0x95000000;
  bits<1> sign;
  let RawWord{3-0} = {1, sign, 1, 0};
  let RawWord{19-16} = !if(Mirror, {1, sign, 1, 0}, 0);
  let Inst{31-2} = RawWord{29-0};
  let Inst{1-0} = RawWord{31-30};
}
def TTSFPLUTFP32SixBlackholeEncoding : TensixLUTFP32SixEncoding<1>;
def TTSFPLUTFP32SixWormholeEncoding : TensixLUTFP32SixEncoding<0>;
def TTSFPLUTFP32ThreeBlackholeEncoding : TensixLUTFP32ThreeEncoding<1>;
def TTSFPLUTFP32ThreeWormholeEncoding : TensixLUTFP32ThreeEncoding<0>;

// Direct SFPLUTFP32 forms read seven fixed LRegs: six table
// registers plus L3. Mode bit3 stays clear in both compute and scheduler mirror.
def TTSFPLUTFP32Six : TensixPhysicalInst<0x95, (outs SFPR:$dst),
    (ins SFPR:$old, uimm3:$mode), "ttsfplutfp32.6r", "$dst, $mode"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let EncodingInfos = EncodingByHwMode<
      [DefaultMode, TensixWormholeEncoding],
      [TTSFPLUTFP32SixBlackholeEncoding, TTSFPLUTFP32SixWormholeEncoding]>;
  bits<4> dst;
  bits<3> mode;
  let RawWord{7-4} = dst;
  let RawWord{2-0} = mode;
  let RawWord{18-16} = mode;
  let Constraints = "$dst = $old";
  let ModeOperandIndex = 2;
  let AllowedModes = 0x00dd; // FP32-3entry, LUT16-6entry table1/table2, sign.
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_L4, TT_L5, TT_L6,
              TT_CC, TT_CONFIG, TT_ISSUE];
}
// Modes10/14 select the destination from each lane's old L7. Model the
// conditional whole-group update, never a fabricated single static write.
def TTSFPLUTFP32Three : TensixPhysicalInst<0x95, (outs),
    (ins uimm1:$sign), "ttsfplutfp32.3r", "$sign"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let EncodingInfos = EncodingByHwMode<
      [DefaultMode, TensixWormholeEncoding],
      [TTSFPLUTFP32ThreeBlackholeEncoding, TTSFPLUTFP32ThreeWormholeEncoding]>;
  bits<1> sign;
  let RawWord{3-0} = {1, sign, 1, 0};
  let RawWord{19-16} = {1, sign, 1, 0};
  let ModeOperandIndex = 0;
  let AllowedModes = 0x0003;
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_L4, TT_L5, TT_L6, TT_L7,
              TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_L0, TT_L1, TT_L2, TT_L3, TT_L4, TT_L5, TT_L6, TT_L7,
              TT_ISSUE];
}
def TTSFPTRANSP : TensixPhysicalInst<0x8c, (outs), (ins), "ttsfptransp", ""> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let Uses = [TT_L0, TT_L1, TT_L2, TT_L3, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_L0, TT_L1, TT_L2, TT_L3, TT_L4, TT_L5, TT_L6, TT_L7, TT_ISSUE];
}

include "RISCVInstrInfoTensixContentsGroup.td"
