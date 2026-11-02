//===-- RISCVInstrInfoTensixContentsGroup.td -------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

def PseudoTTSFPContentsTranspose8 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3,
          SFPR4:$r4, SFPR5:$r5, SFPR6:$r6, SFPR7:$r7),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         SFPR4:$old4, SFPR5:$old5, SFPR6:$old6, SFPR7:$old7)> {
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3,"
                    "$r4 = $old4,$r5 = $old5,$r6 = $old6,$r7 = $old7";
}

def PseudoTTSFPContentsCopy4 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         uimm1:$mode)> {
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3";
}

def PseudoTTSFPContentsRotate4 : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         SFPRRead:$src)> {
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3";
}

def PseudoTTSFPContentsLUT : TensixContentsGroupPseudo<
    (outs SFPR:$dst),
    (ins SFPR:$old, SFPR0:$t0, SFPR1:$t1, SFPR2:$t2, SFPR3:$x,
         uimm4:$mode)> {
  let Constraints = "$dst = $old";
}

def PseudoTTSFPContentsLUTSix : TensixContentsGroupPseudo<
    (outs SFPR:$dst),
    (ins SFPR:$old, SFPR0:$t0, SFPR1:$t1, SFPR2:$t2, SFPR3:$x,
         SFPR4:$t4, SFPR5:$t5, SFPR6:$t6, uimm3:$mode)> {
  let Constraints = "$dst = $old";
}

def PseudoTTSFPContentsLUTThree : TensixContentsGroupPseudo<
    (outs SFPR0:$r0, SFPR1:$r1, SFPR2:$r2, SFPR3:$r3,
          SFPR4:$r4, SFPR5:$r5, SFPR6:$r6, SFPR7:$r7),
    (ins SFPR0:$old0, SFPR1:$old1, SFPR2:$old2, SFPR3:$old3,
         SFPR4:$old4, SFPR5:$old5, SFPR6:$old6, SFPR7:$old7,
         uimm1:$sign)> {
  let Constraints = "$r0 = $old0,$r1 = $old1,$r2 = $old2,$r3 = $old3,"
                    "$r4 = $old4,$r5 = $old5,$r6 = $old6,$r7 = $old7";
}

class TensixContentsConfigCReg<Register CReg> : TensixContentsGroupPseudo<
    (outs), (ins SFPR0:$source, uimm16:$mask, uimm4:$mode)> {
  let Uses = [CReg, TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [CReg, TT_CONFIG, TT_ISSUE];
}
def PseudoTTSFPContentsConfigC11 : TensixContentsConfigCReg<TT_C11>;
def PseudoTTSFPContentsConfigC12 : TensixContentsConfigCReg<TT_C12>;
def PseudoTTSFPContentsConfigC13 : TensixContentsConfigCReg<TT_C13>;
def PseudoTTSFPContentsConfigC14 : TensixContentsConfigCReg<TT_C14>;

def PseudoTTSFPContentsConfigLane : TensixContentsGroupPseudo<
    (outs), (ins SFPR0:$source, uimm16:$mask, uimm4:$mode)> {
  let Defs = [TT_CONFIG, TT_ISSUE];
}
