//===-- RISCVInstrInfoBOSC.td -----------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// FinalizeISel replaces ingress pseudos with standard RVV instructions bearing
// physical VRF operands. There is no virtual vector result or vector allocator.
class BOSCBoundPseudo<dag Outputs, dag Inputs> : Pseudo<Outputs, Inputs, []> {
  let Predicates = [IsRV64, HasVInstructions];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let usesCustomInserter = 1;
  let mayLoad = 0;
  let mayStore = 0;
}
def PseudoBOSCSetVL : BOSCBoundPseudo<(outs GPR:$vl), (ins GPRNoX0:$avl)> {
  let Defs = [VL, VTYPE];
}
def PseudoBOSCSetVLI : BOSCBoundPseudo<(outs GPR:$vl), (ins uimm5:$avl)> {
  let Defs = [VL, VTYPE];
}
let Uses = [VL, VTYPE] in {
def PseudoBOSCLoad : BOSCBoundPseudo<(outs), (ins uimm5:$vd, GPR:$base)> {
  let mayLoad = 1;
}
def PseudoBOSCAdd : BOSCBoundPseudo<(outs),
                                   (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
let Uses = [FRM, VL, VTYPE], mayRaiseFPException = true in {
def PseudoBOSCFAdd : BOSCBoundPseudo<(outs),
                                    (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
def PseudoBOSCFSub : BOSCBoundPseudo<(outs),
                                    (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
def PseudoBOSCFMul : BOSCBoundPseudo<(outs),
                                    (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
def PseudoBOSCFDiv : BOSCBoundPseudo<(outs),
                                    (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
def PseudoBOSCFMacc : BOSCBoundPseudo<(outs),
                                    (ins uimm5:$vd, uimm5:$vs1, uimm5:$vs2)>;
def PseudoBOSCFCvtToI32 : BOSCBoundPseudo<(outs), (ins uimm5:$vd, uimm5:$vs)>;
def PseudoBOSCFCvtFromI32 : BOSCBoundPseudo<(outs), (ins uimm5:$vd, uimm5:$vs)>;
}
let mayRaiseFPException = true in {
def PseudoBOSCFMax : BOSCBoundPseudo<(outs),
                                    (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
def PseudoBOSCFCmpLt : BOSCBoundPseudo<(outs),
                                     (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
def PseudoBOSCFCmpEq : BOSCBoundPseudo<(outs),
                                     (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1)>;
}
def PseudoBOSCMerge : BOSCBoundPseudo<(outs),
    (ins uimm5:$vd, uimm5:$vs2, uimm5:$vs1, uimm5:$mask)>;
def PseudoBOSCIAddScalar : BOSCBoundPseudo<(outs),
    (ins uimm5:$vd, uimm5:$vs2, GPR:$value)>;
def PseudoBOSCGatherScalar : BOSCBoundPseudo<(outs),
    (ins uimm5:$vd, uimm5:$vs2, GPR:$index)>;
def PseudoBOSCShl : BOSCBoundPseudo<(outs),
    (ins uimm5:$vd, uimm5:$vs2, uimm5:$amount)>;
def PseudoBOSCStore : BOSCBoundPseudo<(outs), (ins uimm5:$vs, GPR:$base)> {
  let mayStore = 1;
}
def PseudoBOSCFSplat : BOSCBoundPseudo<(outs), (ins uimm5:$vd, FPR32:$value)>;
let mayRaiseFPException = true in
def PseudoBOSCFReduceMax : BOSCBoundPseudo<(outs), (ins uimm5:$vd, uimm5:$vs)>;
let Uses = [FRM, VL, VTYPE], mayRaiseFPException = true in
def PseudoBOSCFReduceSum : BOSCBoundPseudo<(outs), (ins uimm5:$vd, uimm5:$vs)>;
}
let Uses = [VTYPE] in
def PseudoBOSCFExtract : BOSCBoundPseudo<(outs FPR32:$value), (ins uimm5:$vs)>;

// Patch7 DMA PMA changes use only ordinary scalar registers. The custom
// inserter emits the received platform sequence before register allocation.
class BOSCDMAPseudo : Pseudo<(outs), (ins), []> {
  let Predicates = [IsRV64];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let usesCustomInserter = 1;
  let mayLoad = 1;
  let mayStore = 1;
}
def PseudoBOSCDMAWrites : BOSCDMAPseudo;
def PseudoBOSCDMAReads : BOSCDMAPseudo;
def PseudoBOSCDMAFence : BOSCDMAPseudo;

//===----------------------------------------------------------------------===//
// BOSC AME fixed-profile receiver
//===----------------------------------------------------------------------===//

class BOSCAMEPseudo<dag outs, dag ins> : Pseudo<outs, ins, []> {
  let Predicates = [HasVendorXBOSCAME];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let isConvergent = 1;
  let hasNoSchedulingInfo = 1;
  let usesCustomInserter = 1;
  let mayLoad = 0;
  let mayStore = 0;
}

def PseudoBOSCAMESetM : BOSCAMEPseudo<(outs GPR:$rd), (ins GPR:$value)>;
def PseudoBOSCAMESetN : BOSCAMEPseudo<(outs GPR:$rd), (ins GPR:$value)>;
def PseudoBOSCAMESetK : BOSCAMEPseudo<(outs GPR:$rd), (ins GPR:$value)>;
def PseudoBOSCAMELoadA : BOSCAMEPseudo<(outs),
    (ins uimm5:$tr, GPR:$base, GPR:$stride)> { let mayLoad = 1; let mayStore = 0; }
def PseudoBOSCAMELoadB : BOSCAMEPseudo<(outs),
    (ins uimm5:$tr, GPR:$base, GPR:$stride)> { let mayLoad = 1; let mayStore = 0; }
def PseudoBOSCAMELoadC : BOSCAMEPseudo<(outs),
    (ins uimm3:$acc, GPR:$base, GPR:$stride)> { let mayLoad = 1; let mayStore = 0; }
def PseudoBOSCAMEStoreC : BOSCAMEPseudo<(outs),
    (ins uimm3:$acc, GPR:$base, GPR:$stride)> { let mayLoad = 0; let mayStore = 1; }
def PseudoBOSCAMEMMA : BOSCAMEPseudo<(outs),
    (ins uimm3:$acc, uimm5:$lhs, uimm5:$rhs)>;

let Predicates = [HasVendorXBOSCAME], isNotDuplicable = 1,
    isConvergent = 1, hasNoSchedulingInfo = 1 in {
class BOSCAMEConfig<bits<3> F3, string Name>
    : RVInst<(outs GPR:$rd), (ins GPR:$rs1), Name, "$rd, $rs1", [],
             InstFormatOther> {
  bits<5> rd; bits<5> rs1;
  let Inst{31-26} = 0b000001;
  let Inst{25} = 0b0;
  let Inst{24-20} = 0b00000;
  let Inst{19-15} = rs1;
  let Inst{14-12} = F3;
  let Inst{11-7} = rd;
  let Inst{6-0} = 0b1110111;
  let hasSideEffects = 1;
  let mayLoad = 0;
  let mayStore = 0;
}

let Defs = [BOSC_M] in
def BOSC_AME_SETM : BOSCAMEConfig<0b101, "msettilem">;
let Defs = [BOSC_N] in
def BOSC_AME_SETN : BOSCAMEConfig<0b100, "msettilen">;
let Defs = [BOSC_K] in
def BOSC_AME_SETK : BOSCAMEConfig<0b110, "msettilek">;

let Uses = [BOSC_M, BOSC_N, BOSC_K] in {
class BOSCAMELoad<bits<6> F6, bits<3> EEW, bit Transpose,
                  RegisterClass RC, string Name>
    : RVInst<(outs RC:$md), (ins GPR:$rs1, GPR:$rs2), Name,
             "$md, (${rs1}), $rs2", [], InstFormatOther> {
  bits<5> rs2; bits<5> rs1; bits<4> md;
  let Inst{31-26} = F6; let Inst{25} = 0b0;
  let Inst{24-20} = rs2; let Inst{19-15} = rs1;
  let Inst{14-12} = EEW; let Inst{11} = Transpose;
  let Inst{10-7} = md; let Inst{6-0} = 0b1110111;
  let hasSideEffects = 1; let mayLoad = 1; let mayStore = 0;
}

class BOSCAMEStore<bits<6> F6, bits<3> EEW, bit Transpose,
                   RegisterClass RC, string Name>
    : RVInst<(outs), (ins RC:$ms3, GPR:$rs1, GPR:$rs2), Name,
             "$ms3, (${rs1}), $rs2", [], InstFormatOther> {
  bits<5> rs2; bits<5> rs1; bits<4> ms3;
  let Inst{31-26} = F6; let Inst{25} = 0b1;
  let Inst{24-20} = rs2; let Inst{19-15} = rs1;
  let Inst{14-12} = EEW; let Inst{11} = Transpose;
  let Inst{10-7} = ms3; let Inst{6-0} = 0b1110111;
  let hasSideEffects = 1; let mayLoad = 0; let mayStore = 1;
}

def BOSC_AME_LOADA : BOSCAMELoad<0b000001, 0b000, 0, BOSCTrReg,
                                  "mlae8.m">;
def BOSC_AME_LOADB : BOSCAMELoad<0b000010, 0b000, 0, BOSCTrReg,
                                  "mlbe8.m">;
def BOSC_AME_LOADC : BOSCAMELoad<0b000000, 0b010, 0, BOSCAccReg,
                                  "mlce32.m">;
def BOSC_AME_STOREC : BOSCAMEStore<0b000000, 0b010, 0, BOSCAccReg,
                                    "msce32.m">;

class BOSCAMEMMA<bits<6> F6, bits<3> EEW, string Name>
    : RVInst<(outs BOSCAccReg:$md),
             (ins BOSCAccReg:$md_in, BOSCTrReg:$ms1, BOSCTrReg:$ms2), Name,
             "$md, $ms1, $ms2", [], InstFormatOther> {
  bits<4> md; bits<4> md_in; bits<4> ms1; bits<4> ms2;
  let Constraints = "$md = $md_in";
  let Inst{31-26} = F6; let Inst{25} = 0b0;
  let Inst{24} = 0b0; let Inst{23-20} = ms2; let Inst{19} = 0b1;
  let Inst{18-15} = ms1; let Inst{14-12} = EEW; let Inst{11} = 0b1;
  let Inst{10-7} = md; let Inst{6-0} = 0b1110111;
  let hasSideEffects = 1;
  let mayLoad = 0; let mayStore = 0;
}
def BOSC_AME_MMA : BOSCAMEMMA<0b001010, 0b000, "mqma.b.mm">;
}
}
