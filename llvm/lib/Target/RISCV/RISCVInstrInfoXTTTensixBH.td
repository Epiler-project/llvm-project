//===-- RISCVInstrInfoXTTTensixBH.td -----------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One raw-word authority serves direct issue (rotate left by two) and scalar
// stores to an instruction port (unrotated). This table is private to RISCV.
class TensixEncoding<bits<8> opcode, bits<8> configShift,
                     bits<8> configCount, bits<8> valueBits> {
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<8> RawOpcode = opcode;
  bits<8> OpcodeShift = 24;
  bits<8> ConfigShift = configShift;
  bits<8> ConfigCount = configCount;
  bits<8> ValueBits = valueBits;
  bits<8> ModeOperandIndex = 0;
  bits<16> AllowedModes = 0;
}

def RISCVTensixEncodingTable : GenericTable {
  let FilterClass = "TensixEncoding";
  let CppTypeName = "TensixEncoding";
  let Fields = ["Opcode", "RawOpcode", "OpcodeShift", "ConfigShift",
                "ConfigCount", "ValueBits", "ModeOperandIndex", "AllowedModes"];
  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getTensixEncoding";
}

let Predicates = [HasVendorXTTTensixBH], hasSideEffects = 1, isNotDuplicable = 1,
    mayLoad = 0, mayStore = 1, hasNoSchedulingInfo = 1 in {
// This instruction consumes the loaded scalar even though its result is zero.
// It must survive DAG folding and preserve the ordering of device reads.
def PseudoTTDependentUse : Pseudo<(outs), (ins GPR:$value), []>;
}

include "RISCVInstrInfoTensixOrdinary.td"

class TensixSSAInst<bits<8> RawOpcodeValue, dag Outputs, dag Inputs,
                    string Mnemonic, string Operands>
    : RVInst<Outputs, Inputs, Mnemonic, Operands, [], InstFormatOther>,
      TensixEncoding<RawOpcodeValue, 0, 0, 0> {
  bits<32> RawWord = 0;
  let RawWord{31-24} = RawOpcodeValue;
  let Inst{31-2} = RawWord{29-0};
  let Inst{1-0} = RawWord{31-30};
  let DecoderNamespace = "Tensix";
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
}

def TTSFPLOADI : TensixSSAInst<0x71, (outs SFPR:$dst),
    (ins SFPR:$old, uimm16:$value, uimm4:$mode),
    "ttsfploadi", "$dst, $value, $mode"> {
  bits<4> dst;
  bits<16> value;
  bits<4> mode;
  let ModeOperandIndex = 3;
  let AllowedModes = 0x0517;
  let RawWord{23-20} = dst;
  let RawWord{19-16} = mode;
  let RawWord{15-0} = value;
  let Constraints = "$dst = $old";
}

class TensixMove<bits<4> Mode, dag Inputs, string Mnemonic, string Operands>
    : TensixSSAInst<0x7c, (outs SFPR:$dst), Inputs, Mnemonic, Operands> {
  bits<4> dst;
  bits<4> src;
  let RawWord{11-8} = src;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = Mode;
}
def TTSFPMOV : TensixMove<0, (ins SFPR:$old, SFPRRead:$src),
    "ttsfpmov", "$dst, $src"> {
  let Constraints = "$dst = $old";
}
def TTSFPMOVNeg : TensixMove<1, (ins SFPR:$old, SFPRRead:$src),
    "ttsfpmov.neg", "$dst, $src"> {
  let Constraints = "$dst = $old";
}
// Allocator copies, fixed reads and fixed writes all use this exact instruction.
// It has no old-destination input and does not read CC lane enable state.
def TTSFPMOVAll : TensixMove<2, (ins SFPRRead:$src),
    "ttsfpmov.all", "$dst, $src"> {
  let Uses = [TT_CONFIG, TT_ISSUE];
}

class TensixMAD<bits<8> RawOpcodeValue, dag Inputs, string Mnemonic, string Operands>
    : TensixSSAInst<RawOpcodeValue, (outs SFPR:$dst), Inputs, Mnemonic, Operands> {
  bits<4> dst;
  bits<4> a;
  bits<4> b;
  bits<4> c;
  bits<2> mode;
  let RawWord{19-16} = a;
  let RawWord{15-12} = b;
  let RawWord{11-8} = c;
  let RawWord{7-4} = dst;
  let RawWord{1-0} = mode;
  let Constraints = "$dst = $old";
}
def TTSFPMAD : TensixMAD<0x84,
    (ins SFPR:$old, SFPRRead:$a, SFPRRead:$b, SFPRRead:$c, uimm2:$mode),
    "ttsfpmad", "$dst, $a, $b, $c, $mode">;
def TTSFPMUL : TensixMAD<0x86,
    (ins SFPR:$old, SFPRRead:$a, SFPRRead:$b, uimm2:$mode),
    "ttsfpmul", "$dst, $a, $b, $mode"> {
  let RawWord{11-8} = 9;
  // The tied old destination occupies operand 1 in MCInst.
  let ModeOperandIndex = 4;
  let AllowedModes = 0x000f;
  let Uses = [TT_C9, TT_CC, TT_CONFIG, TT_ISSUE];
}
def TTSFPADD : TensixMAD<0x85,
    (ins SFPR:$old, SFPRRead:$b, SFPRRead:$c, uimm2:$mode),
    "ttsfpadd", "$dst, $b, $c, $mode"> {
  let a = 10;
  let Uses = [TT_C10, TT_CC, TT_CONFIG, TT_ISSUE];
}

class TensixIAdd<bits<4> Mode, string Suffix>
    : TensixSSAInst<0x79, (outs SFPR:$dst),
                    (ins SFPR:$old, SFPRRead:$c),
                    "ttsfpiadd"#Suffix, "$dst, $c"> {
  bits<4> dst;
  bits<4> c;
  let RawWord{11-8} = c;
  let RawWord{7-4} = dst;
  let RawWord{3-0} = Mode;
  let Constraints = "$dst = $old";
  let Defs = !if(!eq(!and(Mode, 4), 0), [TT_CC, TT_ISSUE], [TT_ISSUE]);
}
def TTSFPIADDCCLT : TensixIAdd<0, ".lt0">;
def TTSFPISUBCCLT : TensixIAdd<2, ".sub.lt0">;
def TTSFPIADD : TensixIAdd<4, "">;
def TTSFPISUB : TensixIAdd<6, ".sub">;
def TTSFPIADDCCGE : TensixIAdd<8, ".ge0">;
def TTSFPISUBCCGE : TensixIAdd<10, ".sub.ge0">;

// Plain stack operations only. Encodings with peek/mutate behavior are distinct
// future contracts and cannot enter these nodes with hidden mode integers.
def TTSFPPUSHC : TensixSSAInst<0x87, (outs), (ins), "ttsfppushc", ""> {
  let Uses = [TT_CC, TT_CCSTACK, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_CCSTACK, TT_ISSUE];
}
def TTSFPPOPC : TensixSSAInst<0x88, (outs), (ins), "ttsfppopc", ""> {
  let Uses = [TT_CCSTACK, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_CC, TT_CCSTACK, TT_ISSUE];
}
def TTSFPNOP : TensixSSAInst<0x8f, (outs), (ins), "ttsfpnop", ""> {
  let Uses = [TT_ISSUE];
}

def TTSFPCOMPC : TensixSSAInst<0x8b, (outs), (ins), "ttsfpcompc", ""> {
  let Uses = [TT_CC, TT_CCSTACK, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_CC, TT_ISSUE];
}

// The admitted CONFIG forms only write programmable CRegs.  They consume an
// explicit L0 staging value and retain precise mutable CReg effects.  Macro
// and arbitrary LaneConfig destinations are intentionally absent.
class TensixConfigCReg<bits<4> Dest, Register CReg, string Suffix>
    : TensixSSAInst<0x91, (outs),
                    (ins uimm16:$mask, uimm4:$mode),
                    "ttsfpconfig."#Suffix, "$mask, $mode"> {
  bits<16> mask;
  bits<4> mode;
  let RawWord{23-8} = mask;
  let RawWord{7-4} = Dest;
  let RawWord{3-0} = mode;
  let ModeOperandIndex = 1;
  let AllowedModes = 0x0101;
  let Uses = [TT_L0, CReg, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [CReg, TT_CONFIG, TT_ISSUE];
}
def TTSFPCONFIGC11 : TensixConfigCReg<11, TT_C11, "c11">;
def TTSFPCONFIGC12 : TensixConfigCReg<12, TT_C12, "c12">;
def TTSFPCONFIGC13 : TensixConfigCReg<13, TT_C13, "c13">;
def TTSFPCONFIGC14 : TensixConfigCReg<14, TT_C14, "c14">;
def TTSFPCONFIGReset : TensixSSAInst<0x91, (outs), (ins),
    "ttsfpconfig.reset", ""> {
  let RawWord{23-8} = 0;
  let RawWord{7-4} = 15;
  let RawWord{3-0} = 1;
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_CONFIG, TT_ISSUE];
}

def TTSFPENCC : TensixSSAInst<0x8a, (outs), (ins uimm2:$value, uimm4:$mode),
                             "ttsfpencc", "$value, $mode"> {
  bits<2> value;
  bits<4> mode;
  let ModeOperandIndex = 1;
  let AllowedModes = 0x0707;
  let RawWord{13-12} = value;
  let RawWord{3-0} = mode;
  let Defs = [TT_CC, TT_ISSUE];
}
class TensixSetCC<bits<4> Mode, string Suffix>
    : TensixSSAInst<0x7b, (outs), (ins SFPRRead:$c),
                    "ttsfpsetcc."#Suffix, "$c"> {
  bits<4> c;
  let RawWord{11-8} = c;
  let RawWord{3-0} = Mode;
  let Defs = [TT_CC, TT_ISSUE];
}
def TTSFPSETCCLT : TensixSetCC<0, "lt0">;
def TTSFPSETCCNE : TensixSetCC<2, "ne0">;
def TTSFPSETCCGE : TensixSetCC<4, "ge0">;
def TTSFPSETCCEQ : TensixSetCC<6, "eq0">;

// Dst uses a logical 10-bit offset in this compiler contract. The ISA field
// reserves 13 bits; high fields are never invented by masking a scalar value.
def tensix_dst_offset : RISCVUImmLeafOp<10>;
class TensixDst<bits<8> RawOpcodeValue, dag Outputs, dag Inputs,
                string Mnemonic, string Operands>
    : TensixSSAInst<RawOpcodeValue, Outputs, Inputs, Mnemonic, Operands> {
  bits<4> reg;
  bits<4> format;
  bits<3> addrmode;
  bits<10> offset;
  let RawWord{23-20} = reg;
  let RawWord{19-16} = format;
  let RawWord{15-13} = addrmode;
  let RawWord{9-0} = offset;
  let ValueBits = 10;
  let Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_DST, TT_ISSUE];
}
def TTSFPLOAD : TensixDst<0x70, (outs SFPR:$reg),
    (ins SFPR:$old, tensix_dst_offset:$offset, uimm3:$addrmode, uimm4:$format),
    "ttsfpload", "$reg, $offset, $addrmode, $format"> {
  let Constraints = "$reg = $old";
  let mayLoad = 1;
  let ModeOperandIndex = 4;
  // Format 0 selects the runtime-configured Dst format (SRCB).
  let AllowedModes = 0x001d;
}
def TTSFPSTORE : TensixDst<0x72, (outs),
    (ins SFPR:$reg, tensix_dst_offset:$offset, uimm3:$addrmode, uimm4:$format),
    "ttsfpstore", "$reg, $offset, $addrmode, $format"> {
  let ModeOperandIndex = 3;
  // Format 0 selects the runtime-configured Dst format (SRCB).
  let AllowedModes = 0x001d;
  let mayStore = 1;
}
// Scalar offset issue keeps SFPU SSA and all scalar scratch registers visible
// until final MC emission. Word/address never overlap any live input.
let Predicates = [HasVendorXTTTensixBH], hasSideEffects = 1,
    isNotDuplicable = 1, hasNoSchedulingInfo = 1,
    Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE],
    Defs = [TT_DST, TT_ISSUE], Size = 16 in {
def PseudoTTSFPLOAD : Pseudo<
    (outs SFPR:$reg, GPR:$word, GPR:$address),
    (ins SFPR:$old, GPR:$offset, uimm3:$addrmode, uimm4:$format), []> {
  let Constraints = "$reg = $old,@earlyclobber $word,@earlyclobber $address";
  let mayLoad = 1;
  let mayStore = 1;
}
def PseudoTTSFPSTORE : Pseudo<(outs GPR:$word, GPR:$address),
    (ins SFPR:$reg, GPR:$offset, uimm3:$addrmode, uimm4:$format), []> {
  let Constraints = "@earlyclobber $word,@earlyclobber $address";
  let mayLoad = 0;
  let mayStore = 1;
}
}

include "RISCVInstrInfoTensixSFPU.td"
