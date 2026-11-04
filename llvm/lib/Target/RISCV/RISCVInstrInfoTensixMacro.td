//===-- RISCVInstrInfoTensixMacro.td ---------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared FP32/AddrMod0 macro profile with architecture-specific address setup.
// These are actual SFPCONFIG forms, not an independently encoded word ABI.
def uimm12 : RISCVUImmOp<12>;

class TensixMacroConfig<dag Inputs, string Suffix, string Operands>
    : TensixPhysicalInst<0x91, (outs), Inputs,
                         "ttsfpconfig.macro." # Suffix, Operands> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let Defs = [TT_CONFIG, TT_ISSUE];
}
def TTSFPCONFIGMacroTemplate
    : TensixMacroConfig<(ins uimm2:$index), "template", "$index"> {
  bits<2> index;
  let RawWord{5-4} = index;
  let Uses = [TT_L0, TT_CC, TT_CONFIG, TT_ISSUE];
}
def TTSFPCONFIGMacroSequence
    : TensixMacroConfig<(ins uimm2:$index), "sequence", "$index"> {
  bits<2> index;
  let RawWord{7-6} = 1;
  let RawWord{5-4} = index;
  let Uses = [TT_L0, TT_CC, TT_CONFIG, TT_ISSUE];
}
def TTSFPCONFIGMacroMisc
    : TensixMacroConfig<(ins uimm12:$value), "misc", "$value"> {
  bits<12> value;
  let RawWord{19-8} = value;
  let RawWord{7-4} = 8;
  let RawWord{3-0} = 1;
}

// The native offset is Imm9. The intrinsic receives the byte-independent Dst
// word offset and proves it even before shifting. No VDHi bit is discarded.
def TTSFPLOADMACROFP32Even : TensixPhysicalInst<0x93,
    (outs SFPRLow:$dst),
    (ins SFPRLow:$old, uimm9:$offset, uimm2:$index),
    "ttsfploadmacro.fp32.even", "$dst, $offset, $index"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  bits<2> dst;
  bits<9> offset;
  bits<2> index;
  let RawWord{23-22} = index;
  let RawWord{21-20} = dst;
  let RawWord{19-16} = 3;
  let RawWord{9-1} = offset;
  let Constraints = "$dst = $old";
  let Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_DST, TT_ISSUE];
  let mayLoad = 1;
  let mayStore = 1;
}

let Predicates = [HasVendorXTTTensix], hasSideEffects = 1,
    isNotDuplicable = 1, isConvergent = 1, hasNoSchedulingInfo = 1,
    Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE],
    Defs = [TT_CONFIG, TT_DST, TT_ISSUE] in {
  def PseudoTTSFPMacroAddressReset : Pseudo<(outs), (ins), []> {
    // BH has twelve native setup instructions; WH retains seven.
    let Size = 48;
    let mayLoad = 1;
    let mayStore = 1;
  }
  def PseudoTTSFPMacroConfigureADDI
      : Pseudo<(outs), (ins uimm2:$index, uimm16:$value), []> {
    let Defs = [TT_L0, TT_CONFIG, TT_ISSUE];
    let Size = 40;
    let mayLoad = 0;
    let mayStore = 0;
  }
  def PseudoTTSFPContentsLoadMacroADDI : Pseudo<(outs SFPRLow:$dst),
      (ins SFPRLow:$old, uimm9:$offset, uimm2:$index), []> {
    let Constraints = "$dst = $old";
    let Defs = [TT_DST, TT_ISSUE];
    let Size = 20;
    let mayLoad = 1;
    let mayStore = 1;
  }
}
