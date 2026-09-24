//===-- RISCVRegisterInfoTensix.td ------------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// These are 32 physical lanes of 32 raw bits, never RVV or scalar GPR aliases.
let Namespace = "RISCV" in {
  class TensixRegister<bits<4> Encoding, string Name> : Register<Name> {
    let HWEncoding{3-0} = Encoding;
    let DwarfNumbers = [-1];
  }
  foreach I = 0-7 in
    def TT_L#I : TensixRegister<I, "l"#I>;
  foreach I = 8-15 in
    def TT_C#I : TensixRegister<I, "c"#I>;
  // Macro staging is not ninth-register allocation capacity.
  def TT_STAGE : Register<"ttstage">;
  def TT_CC : Register<"ttcc">;
  def TT_CCSTACK : Register<"ttccstack">;
  def TT_CONFIG : Register<"ttconfig">;
  def TT_DST : Register<"ttdst">;
  def TT_ISSUE : Register<"ttissue">;
}

def SFPR : RISCVRegisterClass<[v32i32], 1024,
                             (add (sequence "TT_L%u", 0, 7))> {
  let Size = 1024;
  let DecoderMethod = "DecodeTensixLReg";
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
                                     TT_DST, TT_ISSUE)> {
  let isAllocatable = 0;
}
