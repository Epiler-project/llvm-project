//===-- RISCVRegisterInfoTensix.td ------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// These are 32 physical lanes of 32 raw bits, never RVV or scalar GPR aliases.
let Namespace = "RISCV" in {
  class TensixRegister<bits<4> Encoding, string Name> : Register<Name> {
    let HWEncoding{3-0} = Encoding;
    let DwarfNumbers = [-1];
    // Hardware-defined contents, independent of software setup and numerical
    // value. C15 is initialized even though its lanes contain different values.
    bit ImmutableInitialized = 0;
  }
  foreach I = 0-7 in
    def TT_L#I : TensixRegister<I, "l"#I>;
  foreach I = 8-15 in
    def TT_C#I : TensixRegister<I, "c"#I> {
      let ImmutableInitialized =
          !or(!eq(I, 8), !eq(I, 9), !eq(I, 10), !eq(I, 15));
    }
  // Macro staging is not ninth-register allocation capacity.
  def TT_STAGE : Register<"ttstage">;
  def TT_CC : Register<"ttcc">;
  def TT_CCSTACK : Register<"ttccstack">;
  def TT_CONFIG : Register<"ttconfig">;
  def TT_DST : Register<"ttdst">;
  def TT_ISSUE : Register<"ttissue">;
  def TT_PRNG : Register<"ttprng">;
}

def SFPR : RISCVRegisterClass<[v32i32], 1024,
                             (add (sequence "TT_L%u", 0, 7))> {
  // Only ordinary writable contents participate in LLVM allocation. Constants,
  // macro staging and shared hardware state remain fixed nonallocatable banks.
  let Size = 1024;
  let isAllocatable = 1;
  let DecoderMethod = "DecodeTensixLReg";
}
// SFPLOADMACRO encodes VDLo and derives VDHi from effective Dst parity.
// The initial profile receives even addresses, so allocation uses L0..L3.
def SFPRLow : RISCVRegisterClass<[v32i32], 1024,
                                (add (sequence "TT_L%u", 0, 3))> {
  let Size = 1024;
  let isAllocatable = 1;
  let DecoderMethod = "DecodeTensixLReg";
}
// The index companion is selected by hardware from a low primary. These
// aggregate registers are allocation identities, not extra hardware capacity.
// Their leaf regunits alias the same eight ordinary writable LRegs.
def SFPRHigh : RISCVRegisterClass<[v32i32], 1024,
                                 (add (sequence "TT_L%u", 4, 7))> {
  let Size = 1024;
  let isAllocatable = 1;
}
let Namespace = "RISCV" in {
  def sub_sfpu_value : SubRegIndex<1024, 0>;
  def sub_sfpu_index : SubRegIndex<1024, 1024>;
}
def SFPRIndexPairs : RegisterTuples<
    [sub_sfpu_value, sub_sfpu_index],
    [(add TT_L0, TT_L1, TT_L2, TT_L3),
     (add TT_L4, TT_L5, TT_L6, TT_L7)]>;
def SFPRIndexPair : RISCVRegisterClass<[untyped], 1024,
                                      (add SFPRIndexPairs)> {
  let Size = 2048;
  let isAllocatable = 1;
}
foreach I = 0-7 in {
  def SFPR#I : RISCVRegisterClass<[v32i32], 1024,
                                  (add !cast<Register>("TT_L"#I))> {
    let Size = 1024;
    let isAllocatable = 1;
  }
}
def SFPCR : RISCVRegisterClass<[v32i32], 1024,
                              (add (sequence "TT_C%u", 8, 15))> {
  let Size = 1024;
  let isAllocatable = 0;
  let DecoderMethod = "DecodeTensixCReg";
}
def SFPRRead : RISCVRegisterClass<[v32i32], 1024, (add SFPR, SFPCR)> {
  let Size = 1024;
  let isAllocatable = 0;
  let DecoderMethod = "DecodeTensixReadReg";
}
def SFPStage : RISCVRegisterClass<[v32i32], 1024, (add TT_STAGE)> {
  let Size = 1024;
  let isAllocatable = 0;
}
def SFPState : RISCVRegisterClass<[untyped], 32,
                                (add TT_CC, TT_CCSTACK, TT_CONFIG,
                                     TT_DST, TT_ISSUE, TT_PRNG)> {
  let isAllocatable = 0;
}
