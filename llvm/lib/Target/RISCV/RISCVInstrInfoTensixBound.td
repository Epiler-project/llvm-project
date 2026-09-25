//===-- RISCVInstrInfoTensixBound.td ---------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// These ingress pseudos carry already-selected physical register numbers.
// FinalizeISel creates the existing native instruction with physical defs/uses.
// No SFPR result ever passes through InstrEmitter's virtual-result creation.
// Native instruction records remain the only register/effect/encoding owner.
class TensixBoundPseudo<dag Inputs, dag Outputs = (outs)>
    : Pseudo<Outputs, Inputs, []> {
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let usesCustomInserter = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let Uses = [TT_ISSUE];
  let Defs = [TT_ISSUE];
}

def PseudoTTBoundSFPLOADI : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm16:$value, uimm4:$mode)>;
def PseudoTTBoundSFPLOAD : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, tensix_dst_offset:$offset,
         uimm3:$addrmode, uimm4:$format)> {
  let mayLoad = 1;
}
def PseudoTTBoundSFPSTORE : TensixBoundPseudo<
    (ins uimm3:$src, tensix_dst_offset:$offset, uimm3:$addrmode,
         uimm4:$format)> {
  let mayStore = 1;
}
def PseudoTTBoundSFPLOADDynamic : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, GPR:$offset, uimm3:$addrmode,
         uimm4:$format), (outs GPR:$word, GPR:$address)> {
  let Constraints = "@earlyclobber $word,@earlyclobber $address";
  let mayLoad = 1;
  let mayStore = 1;
}
def PseudoTTBoundSFPSTOREDynamic : TensixBoundPseudo<
    (ins uimm3:$src, GPR:$offset, uimm3:$addrmode, uimm4:$format),
    (outs GPR:$word, GPR:$address)> {
  let Constraints = "@earlyclobber $word,@earlyclobber $address";
  let mayStore = 1;
}

class TensixBoundBinary : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$lhs, uimm4:$rhs, uimm2:$mode)>;
def PseudoTTBoundSFPADD : TensixBoundBinary;
def PseudoTTBoundSFPMUL : TensixBoundBinary;
def PseudoTTBoundSFPMAD : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$a, uimm4:$b, uimm4:$c,
         uimm2:$mode)>;
def PseudoTTBoundSFPIADD : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$rhs, uimm4:$mode)>;

class TensixBoundUnary : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$src, uimm4:$mode)>;
def PseudoTTBoundSFPMOV : TensixBoundUnary;
def PseudoTTBoundSFPMOVAll : TensixBoundPseudo<
    (ins uimm3:$dst, uimm4:$src)>;
def PseudoTTBoundSFPARECIP : TensixBoundUnary;
def PseudoTTBoundSFPEXEXP : TensixBoundUnary;
def PseudoTTBoundSFPEXMAN : TensixBoundUnary;
def PseudoTTBoundSFPABS : TensixBoundUnary;
def PseudoTTBoundSFPLZ : TensixBoundUnary;
def PseudoTTBoundSFPCAST : TensixBoundUnary;

class TensixBoundSetImmediate : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$base, uimm16:$value,
         uimm4:$mode)>;
class TensixBoundSetVector : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$base, uimm3:$field,
         uimm4:$mode)>;
def PseudoTTBoundSFPSETEXPI : TensixBoundSetImmediate;
def PseudoTTBoundSFPSETMANI : TensixBoundSetImmediate;
def PseudoTTBoundSFPSETSGNI : TensixBoundSetImmediate;
def PseudoTTBoundSFPSETEXPV : TensixBoundSetVector;
def PseudoTTBoundSFPSETMANV : TensixBoundSetVector;
def PseudoTTBoundSFPSETSGNV : TensixBoundSetVector;

class TensixBoundImmediate : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$value, simm12:$immediate, uimm4:$mode)>;
def PseudoTTBoundSFPIADDI : TensixBoundImmediate;
def PseudoTTBoundSFPSHFTI : TensixBoundImmediate;
def PseudoTTBoundSFPSHFTV : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$value, uimm4:$amount, uimm4:$mode)>;
class TensixBoundBitwise : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$lhs, uimm4:$rhs)>;
def PseudoTTBoundSFPAND : TensixBoundBitwise;
def PseudoTTBoundSFPOR : TensixBoundBitwise;
def PseudoTTBoundSFPXOR : TensixBoundBitwise;
def PseudoTTBoundSFPNOT : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$value)>;
def PseudoTTBoundSFPSHFT2 : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$value, uimm16:$immediate,
         uimm4:$mode)>;
def PseudoTTBoundSFPSTOCHRNDI : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$value, uimm8:$descale,
         uimm4:$mode, uimm2:$round)>;
def PseudoTTBoundSFPSTOCHRNDV : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm4:$value, uimm4:$descale,
         uimm4:$mode, uimm2:$round)>;

def PseudoTTBoundSFPLUT : TensixBoundPseudo<
    (ins uimm3:$dst, uimm3:$old, uimm3:$table0, uimm3:$table1,
         uimm3:$table2, uimm3:$input, uimm4:$mode)>;
def PseudoTTBoundSFPSWAP : TensixBoundPseudo<
    (ins uimm3:$dst0, uimm3:$dst1, uimm3:$lhs, uimm3:$rhs, uimm4:$mode)>;
def PseudoTTBoundSFPTRANSP : TensixBoundPseudo<
    (ins uimm3:$out0, uimm3:$out1, uimm3:$out2, uimm3:$out3,
         uimm3:$clobber0, uimm3:$clobber1, uimm3:$clobber2, uimm3:$clobber3,
         uimm3:$in0, uimm3:$in1, uimm3:$in2, uimm3:$in3)>;
def PseudoTTBoundSFPSETCC : TensixBoundPseudo<
    (ins uimm4:$src, uimm16:$immediate, uimm4:$mode)>;
class TensixBoundCondition : TensixBoundPseudo<
    (ins uimm16:$immediate, uimm4:$mode)>;
def PseudoTTBoundSFPENCC : TensixBoundCondition;
def PseudoTTBoundSFPPUSHC : TensixBoundCondition;
def PseudoTTBoundSFPPOPC : TensixBoundCondition;
def PseudoTTBoundSFPCONFIGCReg : TensixBoundPseudo<
    (ins uimm3:$stage, uimm4:$creg, uimm16:$mask, uimm4:$mode)>;
def PseudoTTBoundSFPNOP : TensixBoundPseudo<(ins)>;
def PseudoTTBoundSFPCOMPC : TensixBoundPseudo<(ins)>;
def PseudoTTBoundSFPCONFIGReset : TensixBoundPseudo<(ins)>;
