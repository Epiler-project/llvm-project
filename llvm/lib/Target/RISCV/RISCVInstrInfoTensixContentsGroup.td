//===-- RISCVInstrInfoTensixContentsGroup.td -------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The paired view emits the exact existing LOAD word but exposes both actual
// conditional definitions to allocation. It is not a distinct hardware opcode
// or an assembler surface; decoding remains owned by ordinary TTSFPLOAD.
// Inactive lanes and LOAD14/15's preserved half consume the real tied olds.
def TTSFPLOADIndexBlackholeEncoding : TensixDstBlackholeEncoding<0x70>;
def TTSFPLOADIndexWormholeEncoding : TensixDstWormholeEncoding<0x70>;
def TTSFPLOADIndex : TensixDst<0x70,
    (outs SFPRLow:$reg, SFPRHigh:$index),
    (ins SFPRLow:$old, SFPRHigh:$oldindex, tensix_dst_offset:$offset,
         uimm3:$addrmode, uimm4:$format),
    "ttsfpload", "$reg, $offset, $addrmode, $format"> {
  let Predicates = [HasVendorXTTTensix];
  let WormholeSupported = 1;
  let WormholeOperandMaxima = [-1, -1, -1, -1, -1, 3];
  let Constraints = "$reg = $old,$index = $oldindex";
  let isCodeGenOnly = 1;
  let mayLoad = 1;
  let ModeOperandIndex = 6;
  let AllowedModes = 0xc17f;
}

// Scalar issue retains both real SFPU defs/olds and ordinary scalar scratch
// until final emission. The same static native record owns the word encoding;
// no replay profile or additional architectural register is introduced.
def PseudoTTSFPLOADIndex : Pseudo<
    (outs SFPRLow:$reg, SFPRHigh:$index, GPR:$word, GPR:$address),
    (ins SFPRLow:$old, SFPRHigh:$oldindex, GPR:$offset, uimm3:$addrmode,
         uimm4:$format), []> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$reg = $old,$index = $oldindex,"
                    "@earlyclobber $word,@earlyclobber $address";
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_DST, TT_ISSUE];
  let mayLoad = 1;
  let mayStore = 1;
  let Size = 16;
}

// Allocation receives the entire primary/+4 pair as one native register
// choice. REG_SEQUENCE/subregister views retain both real numerical contents;
// the post-RA expansion below projects only the allocated leaf registers.
class TensixContentsIndexPseudo<dag Outputs, dag Inputs>
    : Pseudo<Outputs, Inputs, []> {
  let Predicates = [HasVendorXTTTensix];
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
}
def PseudoTTSFPContentsLoadIndex : TensixContentsIndexPseudo<
    (outs SFPRIndexPair:$pair),
    (ins SFPRIndexPair:$oldpair, tensix_dst_offset:$offset,
         uimm3:$addrmode, uimm4:$format)> {
  let Constraints = "$pair = $oldpair";
  let Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_DST, TT_ISSUE];
  let mayLoad = 1;
  let Size = 4;
}
def PseudoTTSFPContentsLoadIndexDynamic : TensixContentsIndexPseudo<
    (outs SFPRIndexPair:$pair, GPR:$word, GPR:$address),
    (ins SFPRIndexPair:$oldpair, GPR:$offset, uimm3:$addrmode,
         uimm4:$format)> {
  let Constraints = "$pair = $oldpair,"
                    "@earlyclobber $word,@earlyclobber $address";
  let Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_DST, TT_ISSUE];
  let mayLoad = 1;
  let mayStore = 1;
  let Size = 16;
}
def PseudoTTSFPContentsSwapIndex : TensixContentsIndexPseudo<
    (outs SFPRIndexPair:$pair, SFPRIndexPair:$other),
    (ins SFPRIndexPair:$oldpair, SFPRIndexPair:$oldother, uimm4:$mode)> {
  // Early-clobber excludes the other pair's actual old contents even when
  // a result is dead. Each own tied old is the intentional overlap.
  let Constraints = "$pair = $oldpair,$other = $oldother,"
                    "@earlyclobber $pair,@earlyclobber $other";
  let mayLoad = 0;
  let mayStore = 0;
  let Size = 4;
}

// Singleton register classes constrain numerical SSA to the actual hardware
// groups. There is no ingress staging, extra register capacity, or hidden move.
// Expansion occurs after allocation, when these ties and fixed positions can
// be checked before selecting the existing instruction encoding.
class TensixContentsGroupPseudo<dag Outputs, dag Inputs>
    : Pseudo<Outputs, Inputs, []> {
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
}

def PseudoTTSFPContentsTranspose4 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3";
  // These real hardware writes must constrain allocation before expansion.
  // Their previous contents are neither read nor returned.
  let Defs = [TT_L4, TT_L5, TT_L6, TT_L7, TT_ISSUE];
}

def PseudoTTSFPContentsTranspose8 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3,
          SFPR4:$r4, SFPR5:$r5, SFPR6:$r6, SFPR7:$r7),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         SFPR4:$old4, SFPR5:$old5, SFPR6:$old6, SFPR7:$old7)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3,"
                    "$r4 = $old4,$r5 = $old5,$r6 = $old6,$r7 = $old7";
}

def PseudoTTSFPContentsCopy4 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         uimm1:$mode)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3";
}

def PseudoTTSFPContentsRotate4 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         SFPRRead:$src)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3";
}

def PseudoTTSFPContentsLUT : TensixContentsGroupPseudo<
    (outs SFPR:$dst),
    (ins SFPR:$old, SFPR0:$t0, SFPR1:$t1, SFPR2:$t2, SFPR3:$x,
         uimm4:$mode)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$dst = $old";
}

def PseudoTTSFPContentsLUTSix : TensixContentsGroupPseudo<
    (outs SFPR:$dst),
    (ins SFPR:$old, SFPR0:$t0, SFPR1:$t1, SFPR2:$t2, SFPR3:$x,
         SFPR4:$t4, SFPR5:$t5, SFPR6:$t6, uimm3:$mode)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$dst = $old";
}

def PseudoTTSFPContentsLUTThree : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3,
          SFPR4:$r4, SFPR5:$r5, SFPR6:$r6, SFPR7:$r7),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         SFPR4:$old4, SFPR5:$old5, SFPR6:$old6, SFPR7:$old7,
         uimm1:$sign)> {
  let Predicates = [HasVendorXTTTensix];
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3,"
                    "$r4 = $old4,$r5 = $old5,$r6 = $old6,$r7 = $old7";
}

class TensixContentsConfigCReg<Register CReg> : TensixContentsGroupPseudo<
    (outs), (ins SFPR0:$source, uimm16:$mask, uimm4:$mode)> {
  let Predicates = [HasVendorXTTTensix];
  let Uses = [CReg, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [CReg, TT_CONFIG, TT_ISSUE];
}
def PseudoTTSFPContentsConfigC11 : TensixContentsConfigCReg<TT_C11>;
def PseudoTTSFPContentsConfigC12 : TensixContentsConfigCReg<TT_C12>;
def PseudoTTSFPContentsConfigC13 : TensixContentsConfigCReg<TT_C13>;
def PseudoTTSFPContentsConfigC14 : TensixContentsConfigCReg<TT_C14>;

def PseudoTTSFPContentsConfigLane : TensixContentsGroupPseudo<
    (outs), (ins SFPR0:$source, uimm16:$mask, uimm4:$mode)> {
  let Predicates = [HasVendorXTTTensix];
  let Defs = [TT_CONFIG, TT_ISSUE];
}
