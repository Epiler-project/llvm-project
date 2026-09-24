//===-- RISCVInstrInfoTensixOrdinary.td --------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The logical operand order is also the formal intrinsic order. These are
// target instruction definitions, not an imported instruction schema.
// Dynamic issue uses the same MC encoding with each field at zero or one to
// derive its position; no second mask/shift table exists.
class TensixOrdinaryInst<string LogicalName, bits<8> RawOpcodeValue,
                         dag Inputs, string Operands, list<string> Names,
                         list<int> Maxima>
    : RVInst<(outs), Inputs, "tt" # !tolower(LogicalName), Operands, [],
             InstFormatOther>,
      TensixEncoding<RawOpcodeValue, 0, 0, 0> {
  string Name = LogicalName;
  Intrinsic IntrinsicID = !cast<Intrinsic>("int_riscv_tt_" # !tolower(LogicalName));
  Intrinsic PortIntrinsicID =
      !cast<Intrinsic>("int_riscv_tt_" # !tolower(LogicalName) # "_port");
  Intrinsic MopIntrinsicID =
      !cast<Intrinsic>("int_riscv_tt_" # !tolower(LogicalName) # "_mop");
  Instruction MopOpcode = !cast<Instruction>("PseudoTT" # LogicalName # "Mop");
  Instruction PortOpcode = !cast<Instruction>("PseudoTT" # LogicalName # "Port");
  bit WritesDst = !or(!eq(LogicalName, "APOOL3S1"), !eq(LogicalName, "APOOL3S2"),
      !eq(LogicalName, "CONV3S1"), !eq(LogicalName, "CONV3S2"),
      !eq(LogicalName, "DOTPV"), !eq(LogicalName, "ELWADD"),
      !eq(LogicalName, "ELWMUL"), !eq(LogicalName, "ELWSUB"),
      !eq(LogicalName, "GAPOOL"), !eq(LogicalName, "GMPOOL"),
      !eq(LogicalName, "MFCONV3S1"), !eq(LogicalName, "MOVA2D"),
      !eq(LogicalName, "MOVB2D"), !eq(LogicalName, "MOVDBGA2D"),
      !eq(LogicalName, "MOVDBGB2D"), !eq(LogicalName, "MPOOL3S1"),
      !eq(LogicalName, "MPOOL3S2"), !eq(LogicalName, "MVMUL"),
      !eq(LogicalName, "ZEROACC"));
  list<string> FieldNames = Names;
  list<int> FieldMaxima = Maxima;
  bits<8> NumFields = !size(Names);
  bits<32> RawWord = 0;
  let RawWord{31-24} = RawOpcodeValue;
  let Inst{31-2} = RawWord{29-0};
  let Inst{1-0} = RawWord{31-30};
  let DecoderNamespace = "Tensix";
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_CONFIG, TT_DST, TT_ISSUE];
}

class TensixOrdinaryPort<dag Inputs>
    : Pseudo<(outs GPR:$word, GPR:$address, GPR:$field), Inputs, []> {
  let Constraints = "@earlyclobber $word,@earlyclobber $address,@earlyclobber $field";
  // Conservative expansion bound for branch relaxation, including port store.
  let Size = !mul(4, !add(3, !mul(2, !sub(!size(Inputs), 1))));
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_CONFIG, TT_DST, TT_ISSUE];
}

// MOP slots hold unrotated instruction words. The two GPRs belong to the
// scalar allocator and survive until MC is the sole encoding authority.
class TensixOrdinaryMop<dag Inputs>
    : Pseudo<(outs GPR:$word, GPR:$address), Inputs, []> {
  let Constraints = "@earlyclobber $word,@earlyclobber $address";
  let Size = 16;
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 0;
  let mayStore = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CONFIG, TT_ISSUE];
  let Defs = [TT_CONFIG, TT_ISSUE];
}
def PseudoTTMOPClear : TensixOrdinaryMop<(ins uimm8:$slot)>;
// Post-RA automatic SFPU compression only. Physical value effects are derived
// from the retained record and verified again before emission. This is not an
// LLVM IR ingress for explicit SFPU recording.
def PseudoTTSFPUReplay : Pseudo<(outs),
    (ins uimm32:$load_mode, uimm32:$execute_while_loading,
         uimm32:$len, uimm32:$start_idx), []> {
  let Size = 4;
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
}
def PseudoTTReplayRecordEnd : Pseudo<(outs), (ins), []> {
  let Size = 0;
  let Predicates = [HasVendorXTTTensixBH];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_ISSUE];
  let Defs = [TT_ISSUE];
}

def PseudoTTADDDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$oparegindex, GPR:$opbregindex, GPR:$resultregindex, GPR:$opbisconst)>;
def PseudoTTADDDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opbisconst)>;
def TTADDDMAREG : TensixOrdinaryInst<"ADDDMAREG", 0x58,
    (ins uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opbisconst),
    "$oparegindex, $opbregindex, $resultregindex, $opbisconst",
    ["oparegindex", "opbregindex", "resultregindex", "opbisconst"],
    [63, 63, 2047, 1]> {
  bits<6> oparegindex;
  bits<6> opbregindex;
  bits<11> resultregindex;
  bits<1> opbisconst;
  let RawWord{5-0} = oparegindex;
  let RawWord{11-6} = opbregindex;
  let RawWord{22-12} = resultregindex;
  let RawWord{23} = opbisconst;
}

def PseudoTTADDRCRXYPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$bitmask, GPR:$ch0_x, GPR:$ch0_y, GPR:$ch1_x, GPR:$ch1_y, GPR:$cntsetmask)>;
def PseudoTTADDRCRXYMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask)>;
def TTADDRCRXY : TensixOrdinaryInst<"ADDRCRXY", 0x53,
    (ins uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask),
    "$bitmask, $ch0_x, $ch0_y, $ch1_x, $ch1_y, $cntsetmask",
    ["bitmask", "ch0_x", "ch0_y", "ch1_x", "ch1_y", "cntsetmask"],
    [63, 7, 7, 7, 63, 7]> {
  bits<6> bitmask;
  bits<3> ch0_x;
  bits<3> ch0_y;
  bits<3> ch1_x;
  bits<6> ch1_y;
  bits<3> cntsetmask;
  let RawWord{5-0} = bitmask;
  let RawWord{8-6} = ch0_x;
  let RawWord{11-9} = ch0_y;
  let RawWord{14-12} = ch1_x;
  let RawWord{20-15} = ch1_y;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTADDRCRZWPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$bitmask, GPR:$ch0_x, GPR:$ch0_y, GPR:$ch1_x, GPR:$ch1_y, GPR:$cntsetmask)>;
def PseudoTTADDRCRZWMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask)>;
def TTADDRCRZW : TensixOrdinaryInst<"ADDRCRZW", 0x56,
    (ins uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask),
    "$bitmask, $ch0_x, $ch0_y, $ch1_x, $ch1_y, $cntsetmask",
    ["bitmask", "ch0_x", "ch0_y", "ch1_x", "ch1_y", "cntsetmask"],
    [63, 7, 7, 7, 63, 7]> {
  bits<6> bitmask;
  bits<3> ch0_x;
  bits<3> ch0_y;
  bits<3> ch1_x;
  bits<6> ch1_y;
  bits<3> cntsetmask;
  let RawWord{5-0} = bitmask;
  let RawWord{8-6} = ch0_x;
  let RawWord{11-9} = ch0_y;
  let RawWord{14-12} = ch1_x;
  let RawWord{20-15} = ch1_y;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTAPOOL3S1Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$index_en, GPR:$pool_addr_mode, GPR:$clear_dvalid)>;
def PseudoTTAPOOL3S1Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid)>;
def TTAPOOL3S1 : TensixOrdinaryInst<"APOOL3S1", 0x25,
    (ins uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid),
    "$dst, $index_en, $pool_addr_mode, $clear_dvalid",
    ["dst", "index_en", "pool_addr_mode", "clear_dvalid"],
    [16383, 1, 127, 3]> {
  bits<14> dst;
  bits<1> index_en;
  bits<7> pool_addr_mode;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{14} = index_en;
  let RawWord{21-15} = pool_addr_mode;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTAPOOL3S2Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$index_en, GPR:$pool_addr_mode, GPR:$clear_dvalid)>;
def PseudoTTAPOOL3S2Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid)>;
def TTAPOOL3S2 : TensixOrdinaryInst<"APOOL3S2", 0x32,
    (ins uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid),
    "$dst, $index_en, $pool_addr_mode, $clear_dvalid",
    ["dst", "index_en", "pool_addr_mode", "clear_dvalid"],
    [16383, 1, 127, 3]> {
  bits<14> dst;
  bits<1> index_en;
  bits<7> pool_addr_mode;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{14} = index_en;
  let RawWord{21-15} = pool_addr_mode;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTATCASPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$addrregindex, GPR:$dataregindex, GPR:$sel32b, GPR:$cmpval, GPR:$swapval, GPR:$memhiersel)>;
def PseudoTTATCASMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$sel32b, uimm32:$cmpval, uimm32:$swapval, uimm32:$memhiersel)>;
def TTATCAS : TensixOrdinaryInst<"ATCAS", 0x64,
    (ins uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$sel32b, uimm32:$cmpval, uimm32:$swapval, uimm32:$memhiersel),
    "$addrregindex, $dataregindex, $sel32b, $cmpval, $swapval, $memhiersel",
    ["addrregindex", "dataregindex", "sel32b", "cmpval", "swapval", "memhiersel"],
    [63, 63, 3, 15, 31, 1]> {
  bits<6> addrregindex;
  bits<6> dataregindex;
  bits<2> sel32b;
  bits<4> cmpval;
  bits<5> swapval;
  bits<1> memhiersel;
  let RawWord{5-0} = addrregindex;
  let RawWord{11-6} = dataregindex;
  let RawWord{13-12} = sel32b;
  let RawWord{17-14} = cmpval;
  let RawWord{22-18} = swapval;
  let RawWord{23} = memhiersel;
}

def PseudoTTATGETMPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$mutex_index)>;
def PseudoTTATGETMMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$mutex_index)>;
def TTATGETM : TensixOrdinaryInst<"ATGETM", 0xa0,
    (ins uimm32:$mutex_index),
    "$mutex_index",
    ["mutex_index"],
    [16777215]> {
  bits<24> mutex_index;
  let RawWord{23-0} = mutex_index;
}

def PseudoTTATINCGETPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$addrregindex, GPR:$dataregindex, GPR:$sel32b, GPR:$wrapval, GPR:$memhiersel)>;
def PseudoTTATINCGETMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$sel32b, uimm32:$wrapval, uimm32:$memhiersel)>;
def TTATINCGET : TensixOrdinaryInst<"ATINCGET", 0x61,
    (ins uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$sel32b, uimm32:$wrapval, uimm32:$memhiersel),
    "$addrregindex, $dataregindex, $sel32b, $wrapval, $memhiersel",
    ["addrregindex", "dataregindex", "sel32b", "wrapval", "memhiersel"],
    [63, 63, 3, 511, 1]> {
  bits<6> addrregindex;
  bits<6> dataregindex;
  bits<2> sel32b;
  bits<9> wrapval;
  bits<1> memhiersel;
  let RawWord{5-0} = addrregindex;
  let RawWord{11-6} = dataregindex;
  let RawWord{13-12} = sel32b;
  let RawWord{22-14} = wrapval;
  let RawWord{23} = memhiersel;
}

def PseudoTTATINCGETPTRPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$addrregindex, GPR:$dataregindex, GPR:$sel32b, GPR:$wrapval, GPR:$incrval, GPR:$noincr, GPR:$memhiersel)>;
def PseudoTTATINCGETPTRMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$sel32b, uimm32:$wrapval, uimm32:$incrval, uimm32:$noincr, uimm32:$memhiersel)>;
def TTATINCGETPTR : TensixOrdinaryInst<"ATINCGETPTR", 0x62,
    (ins uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$sel32b, uimm32:$wrapval, uimm32:$incrval, uimm32:$noincr, uimm32:$memhiersel),
    "$addrregindex, $dataregindex, $sel32b, $wrapval, $incrval, $noincr, $memhiersel",
    ["addrregindex", "dataregindex", "sel32b", "wrapval", "incrval", "noincr", "memhiersel"],
    [63, 63, 3, 15, 15, 1, 1]> {
  bits<6> addrregindex;
  bits<6> dataregindex;
  bits<2> sel32b;
  bits<4> wrapval;
  bits<4> incrval;
  bits<1> noincr;
  bits<1> memhiersel;
  let RawWord{5-0} = addrregindex;
  let RawWord{11-6} = dataregindex;
  let RawWord{13-12} = sel32b;
  let RawWord{17-14} = wrapval;
  let RawWord{21-18} = incrval;
  let RawWord{22} = noincr;
  let RawWord{23} = memhiersel;
}

def PseudoTTATRELMPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$mutex_index)>;
def PseudoTTATRELMMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$mutex_index)>;
def TTATRELM : TensixOrdinaryInst<"ATRELM", 0xa1,
    (ins uimm32:$mutex_index),
    "$mutex_index",
    ["mutex_index"],
    [16777215]> {
  bits<24> mutex_index;
  let RawWord{23-0} = mutex_index;
}

def PseudoTTATSWAPPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$addrregindex, GPR:$dataregindex, GPR:$swapmask, GPR:$memhiersel)>;
def PseudoTTATSWAPMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$swapmask, uimm32:$memhiersel)>;
def TTATSWAP : TensixOrdinaryInst<"ATSWAP", 0x63,
    (ins uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$swapmask, uimm32:$memhiersel),
    "$addrregindex, $dataregindex, $swapmask, $memhiersel",
    ["addrregindex", "dataregindex", "swapmask", "memhiersel"],
    [63, 255, 511, 1]> {
  bits<6> addrregindex;
  bits<8> dataregindex;
  bits<9> swapmask;
  bits<1> memhiersel;
  let RawWord{5-0} = addrregindex;
  let RawWord{13-6} = dataregindex;
  let RawWord{22-14} = swapmask;
  let RawWord{23} = memhiersel;
}

def PseudoTTBITWOPDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$oparegindex, GPR:$opbregindex, GPR:$resultregindex, GPR:$opsel, GPR:$opbisconst)>;
def PseudoTTBITWOPDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opsel, uimm32:$opbisconst)>;
def TTBITWOPDMAREG : TensixOrdinaryInst<"BITWOPDMAREG", 0x5b,
    (ins uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opsel, uimm32:$opbisconst),
    "$oparegindex, $opbregindex, $resultregindex, $opsel, $opbisconst",
    ["oparegindex", "opbregindex", "resultregindex", "opsel", "opbisconst"],
    [63, 63, 63, 31, 1]> {
  bits<6> oparegindex;
  bits<6> opbregindex;
  bits<6> resultregindex;
  bits<5> opsel;
  bits<1> opbisconst;
  let RawWord{5-0} = oparegindex;
  let RawWord{11-6} = opbregindex;
  let RawWord{17-12} = resultregindex;
  let RawWord{22-18} = opsel;
  let RawWord{23} = opbisconst;
}

def PseudoTTCFGSHIFTMASKPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgreg, GPR:$scratch_sel, GPR:$right_cshift_amt, GPR:$mask_width, GPR:$operation, GPR:$disable_mask_on_old_val)>;
def PseudoTTCFGSHIFTMASKMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgreg, uimm32:$scratch_sel, uimm32:$right_cshift_amt, uimm32:$mask_width, uimm32:$operation, uimm32:$disable_mask_on_old_val)>;
def TTCFGSHIFTMASK : TensixOrdinaryInst<"CFGSHIFTMASK", 0xb8,
    (ins uimm32:$cfgreg, uimm32:$scratch_sel, uimm32:$right_cshift_amt, uimm32:$mask_width, uimm32:$operation, uimm32:$disable_mask_on_old_val),
    "$cfgreg, $scratch_sel, $right_cshift_amt, $mask_width, $operation, $disable_mask_on_old_val",
    ["cfgreg", "scratch_sel", "right_cshift_amt", "mask_width", "operation", "disable_mask_on_old_val"],
    [255, 3, 31, 31, 7, 1]> {
  bits<8> cfgreg;
  bits<2> scratch_sel;
  bits<5> right_cshift_amt;
  bits<5> mask_width;
  bits<3> operation;
  bits<1> disable_mask_on_old_val;
  let RawWord{7-0} = cfgreg;
  let RawWord{9-8} = scratch_sel;
  let RawWord{14-10} = right_cshift_amt;
  let RawWord{19-15} = mask_width;
  let RawWord{22-20} = operation;
  let RawWord{23} = disable_mask_on_old_val;
}

def PseudoTTCLEARDVALIDPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$reset, GPR:$cleardvalid)>;
def PseudoTTCLEARDVALIDMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$reset, uimm32:$cleardvalid)>;
def TTCLEARDVALID : TensixOrdinaryInst<"CLEARDVALID", 0x36,
    (ins uimm32:$reset, uimm32:$cleardvalid),
    "$reset, $cleardvalid",
    ["reset", "cleardvalid"],
    [4194303, 3]> {
  bits<22> reset;
  bits<2> cleardvalid;
  let RawWord{21-0} = reset;
  let RawWord{23-22} = cleardvalid;
}

def PseudoTTCLREXPHISTPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTCLREXPHISTMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTCLREXPHIST : TensixOrdinaryInst<"CLREXPHIST", 0x21,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTCMPDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$oparegindex, GPR:$opbregindex, GPR:$resultregindex, GPR:$opsel, GPR:$opbisconst)>;
def PseudoTTCMPDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opsel, uimm32:$opbisconst)>;
def TTCMPDMAREG : TensixOrdinaryInst<"CMPDMAREG", 0x5d,
    (ins uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opsel, uimm32:$opbisconst),
    "$oparegindex, $opbregindex, $resultregindex, $opsel, $opbisconst",
    ["oparegindex", "opbregindex", "resultregindex", "opsel", "opbisconst"],
    [63, 63, 63, 31, 1]> {
  bits<6> oparegindex;
  bits<6> opbregindex;
  bits<6> resultregindex;
  bits<5> opsel;
  bits<1> opbisconst;
  let RawWord{5-0} = oparegindex;
  let RawWord{11-6} = opbregindex;
  let RawWord{17-12} = resultregindex;
  let RawWord{22-18} = opsel;
  let RawWord{23} = opbisconst;
}

def PseudoTTCONV3S1Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$rotate_weights, GPR:$clear_dvalid)>;
def PseudoTTCONV3S1Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$rotate_weights, uimm32:$clear_dvalid)>;
def TTCONV3S1 : TensixOrdinaryInst<"CONV3S1", 0x22,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$rotate_weights, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $rotate_weights, $clear_dvalid",
    ["dst", "addr_mode", "rotate_weights", "clear_dvalid"],
    [16383, 7, 31, 3]> {
  bits<14> dst;
  bits<3> addr_mode;
  bits<5> rotate_weights;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{16-14} = addr_mode;
  let RawWord{21-17} = rotate_weights;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTCONV3S2Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$rotate_weights, GPR:$clear_dvalid)>;
def PseudoTTCONV3S2Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$rotate_weights, uimm32:$clear_dvalid)>;
def TTCONV3S2 : TensixOrdinaryInst<"CONV3S2", 0x23,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$rotate_weights, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $rotate_weights, $clear_dvalid",
    ["dst", "addr_mode", "rotate_weights", "clear_dvalid"],
    [16383, 7, 31, 3]> {
  bits<14> dst;
  bits<3> addr_mode;
  bits<5> rotate_weights;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{16-14} = addr_mode;
  let RawWord{21-17} = rotate_weights;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTDMANOPPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTDMANOPMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTDMANOP : TensixOrdinaryInst<"DMANOP", 0x60,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTDOTPVPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$instr_mod19, GPR:$dest_accum_en, GPR:$clear_dvalid)>;
def PseudoTTDOTPVMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid)>;
def TTDOTPV : TensixOrdinaryInst<"DOTPV", 0x29,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $instr_mod19, $dest_accum_en, $clear_dvalid",
    ["dst", "addr_mode", "instr_mod19", "dest_accum_en", "clear_dvalid"],
    [16383, 31, 3, 1, 3]> {
  bits<14> dst;
  bits<5> addr_mode;
  bits<2> instr_mod19;
  bits<1> dest_accum_en;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{18-14} = addr_mode;
  let RawWord{20-19} = instr_mod19;
  let RawWord{21} = dest_accum_en;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTELWADDPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$instr_mod19, GPR:$dest_accum_en, GPR:$clear_dvalid)>;
def PseudoTTELWADDMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid)>;
def TTELWADD : TensixOrdinaryInst<"ELWADD", 0x28,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $instr_mod19, $dest_accum_en, $clear_dvalid",
    ["dst", "addr_mode", "instr_mod19", "dest_accum_en", "clear_dvalid"],
    [16383, 31, 3, 1, 3]> {
  bits<14> dst;
  bits<5> addr_mode;
  bits<2> instr_mod19;
  bits<1> dest_accum_en;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{18-14} = addr_mode;
  let RawWord{20-19} = instr_mod19;
  let RawWord{21} = dest_accum_en;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTELWMULPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$instr_mod19, GPR:$dest_accum_en, GPR:$clear_dvalid)>;
def PseudoTTELWMULMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid)>;
def TTELWMUL : TensixOrdinaryInst<"ELWMUL", 0x27,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $instr_mod19, $dest_accum_en, $clear_dvalid",
    ["dst", "addr_mode", "instr_mod19", "dest_accum_en", "clear_dvalid"],
    [16383, 31, 3, 1, 3]> {
  bits<14> dst;
  bits<5> addr_mode;
  bits<2> instr_mod19;
  bits<1> dest_accum_en;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{18-14} = addr_mode;
  let RawWord{20-19} = instr_mod19;
  let RawWord{21} = dest_accum_en;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTELWSUBPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$instr_mod19, GPR:$dest_accum_en, GPR:$clear_dvalid)>;
def PseudoTTELWSUBMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid)>;
def TTELWSUB : TensixOrdinaryInst<"ELWSUB", 0x30,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$dest_accum_en, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $instr_mod19, $dest_accum_en, $clear_dvalid",
    ["dst", "addr_mode", "instr_mod19", "dest_accum_en", "clear_dvalid"],
    [16383, 31, 3, 1, 3]> {
  bits<14> dst;
  bits<5> addr_mode;
  bits<2> instr_mod19;
  bits<1> dest_accum_en;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{18-14} = addr_mode;
  let RawWord{20-19} = instr_mod19;
  let RawWord{21} = dest_accum_en;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTFLUSHDMAPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$flushspec)>;
def PseudoTTFLUSHDMAMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$flushspec)>;
def TTFLUSHDMA : TensixOrdinaryInst<"FLUSHDMA", 0x46,
    (ins uimm32:$flushspec),
    "$flushspec",
    ["flushspec"],
    [16777215]> {
  bits<24> flushspec;
  let RawWord{23-0} = flushspec;
}

def PseudoTTGAPOOLPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$max_pool_index_en, GPR:$pool_addr_mode, GPR:$instr_mod19, GPR:$clear_dvalid)>;
def PseudoTTGAPOOLMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$max_pool_index_en, uimm32:$pool_addr_mode, uimm32:$instr_mod19, uimm32:$clear_dvalid)>;
def TTGAPOOL : TensixOrdinaryInst<"GAPOOL", 0x34,
    (ins uimm32:$dst, uimm32:$max_pool_index_en, uimm32:$pool_addr_mode, uimm32:$instr_mod19, uimm32:$clear_dvalid),
    "$dst, $max_pool_index_en, $pool_addr_mode, $instr_mod19, $clear_dvalid",
    ["dst", "max_pool_index_en", "pool_addr_mode", "instr_mod19", "clear_dvalid"],
    [16383, 1, 15, 7, 3]> {
  bits<14> dst;
  bits<1> max_pool_index_en;
  bits<4> pool_addr_mode;
  bits<3> instr_mod19;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{14} = max_pool_index_en;
  let RawWord{18-15} = pool_addr_mode;
  let RawWord{21-19} = instr_mod19;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTGATESRCRSTPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$reset_srca_gate_control, GPR:$reset_srcb_gate_control)>;
def PseudoTTGATESRCRSTMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$reset_srca_gate_control, uimm32:$reset_srcb_gate_control)>;
def TTGATESRCRST : TensixOrdinaryInst<"GATESRCRST", 0x35,
    (ins uimm32:$reset_srca_gate_control, uimm32:$reset_srcb_gate_control),
    "$reset_srca_gate_control, $reset_srcb_gate_control",
    ["reset_srca_gate_control", "reset_srcb_gate_control"],
    [1, 8388607]> {
  bits<1> reset_srca_gate_control;
  bits<23> reset_srcb_gate_control;
  let RawWord{0} = reset_srca_gate_control;
  let RawWord{23-1} = reset_srcb_gate_control;
}

def PseudoTTGMPOOLPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$max_pool_index_en, GPR:$pool_addr_mode, GPR:$instr_mod19, GPR:$clear_dvalid)>;
def PseudoTTGMPOOLMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$max_pool_index_en, uimm32:$pool_addr_mode, uimm32:$instr_mod19, uimm32:$clear_dvalid)>;
def TTGMPOOL : TensixOrdinaryInst<"GMPOOL", 0x33,
    (ins uimm32:$dst, uimm32:$max_pool_index_en, uimm32:$pool_addr_mode, uimm32:$instr_mod19, uimm32:$clear_dvalid),
    "$dst, $max_pool_index_en, $pool_addr_mode, $instr_mod19, $clear_dvalid",
    ["dst", "max_pool_index_en", "pool_addr_mode", "instr_mod19", "clear_dvalid"],
    [16383, 1, 15, 7, 3]> {
  bits<14> dst;
  bits<1> max_pool_index_en;
  bits<4> pool_addr_mode;
  bits<3> instr_mod19;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{14} = max_pool_index_en;
  let RawWord{18-15} = pool_addr_mode;
  let RawWord{21-19} = instr_mod19;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTINCADCXYPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$ch0_x, GPR:$ch0_y, GPR:$ch1_x, GPR:$ch1_y, GPR:$cntsetmask)>;
def PseudoTTINCADCXYMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask)>;
def TTINCADCXY : TensixOrdinaryInst<"INCADCXY", 0x52,
    (ins uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask),
    "$ch0_x, $ch0_y, $ch1_x, $ch1_y, $cntsetmask",
    ["ch0_x", "ch0_y", "ch1_x", "ch1_y", "cntsetmask"],
    [7, 7, 7, 63, 7]> {
  bits<3> ch0_x;
  bits<3> ch0_y;
  bits<3> ch1_x;
  bits<6> ch1_y;
  bits<3> cntsetmask;
  let RawWord{8-6} = ch0_x;
  let RawWord{11-9} = ch0_y;
  let RawWord{14-12} = ch1_x;
  let RawWord{20-15} = ch1_y;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTINCADCZWPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$ch0_x, GPR:$ch0_y, GPR:$ch1_x, GPR:$ch1_y, GPR:$cntsetmask)>;
def PseudoTTINCADCZWMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask)>;
def TTINCADCZW : TensixOrdinaryInst<"INCADCZW", 0x55,
    (ins uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask),
    "$ch0_x, $ch0_y, $ch1_x, $ch1_y, $cntsetmask",
    ["ch0_x", "ch0_y", "ch1_x", "ch1_y", "cntsetmask"],
    [7, 7, 7, 63, 7]> {
  bits<3> ch0_x;
  bits<3> ch0_y;
  bits<3> ch1_x;
  bits<6> ch1_y;
  bits<3> cntsetmask;
  let RawWord{8-6} = ch0_x;
  let RawWord{11-9} = ch0_y;
  let RawWord{14-12} = ch1_x;
  let RawWord{20-15} = ch1_y;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTINCRWCPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$rwc_a, GPR:$rwc_b, GPR:$rwc_d, GPR:$rwc_cr)>;
def PseudoTTINCRWCMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$rwc_a, uimm32:$rwc_b, uimm32:$rwc_d, uimm32:$rwc_cr)>;
def TTINCRWC : TensixOrdinaryInst<"INCRWC", 0x38,
    (ins uimm32:$rwc_a, uimm32:$rwc_b, uimm32:$rwc_d, uimm32:$rwc_cr),
    "$rwc_a, $rwc_b, $rwc_d, $rwc_cr",
    ["rwc_a", "rwc_b", "rwc_d", "rwc_cr"],
    [15, 15, 15, 63]> {
  bits<4> rwc_a;
  bits<4> rwc_b;
  bits<4> rwc_d;
  bits<6> rwc_cr;
  let RawWord{9-6} = rwc_a;
  let RawWord{13-10} = rwc_b;
  let RawWord{17-14} = rwc_d;
  let RawWord{23-18} = rwc_cr;
}

def PseudoTTLOADINDPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$addrregindex, GPR:$dataregindex, GPR:$autoincspec, GPR:$offsetindex, GPR:$sizesel)>;
def PseudoTTLOADINDMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$autoincspec, uimm32:$offsetindex, uimm32:$sizesel)>;
def TTLOADIND : TensixOrdinaryInst<"LOADIND", 0x49,
    (ins uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$autoincspec, uimm32:$offsetindex, uimm32:$sizesel),
    "$addrregindex, $dataregindex, $autoincspec, $offsetindex, $sizesel",
    ["addrregindex", "dataregindex", "autoincspec", "offsetindex", "sizesel"],
    [63, 63, 3, 255, 3]> {
  bits<6> addrregindex;
  bits<6> dataregindex;
  bits<2> autoincspec;
  bits<8> offsetindex;
  bits<2> sizesel;
  let RawWord{5-0} = addrregindex;
  let RawWord{11-6} = dataregindex;
  let RawWord{13-12} = autoincspec;
  let RawWord{21-14} = offsetindex;
  let RawWord{23-22} = sizesel;
}

def PseudoTTLOADREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$regaddr, GPR:$tdmadataregindex)>;
def PseudoTTLOADREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$regaddr, uimm32:$tdmadataregindex)>;
def TTLOADREG : TensixOrdinaryInst<"LOADREG", 0x68,
    (ins uimm32:$regaddr, uimm32:$tdmadataregindex),
    "$regaddr, $tdmadataregindex",
    ["regaddr", "tdmadataregindex"],
    [262143, 63]> {
  bits<18> regaddr;
  bits<6> tdmadataregindex;
  let RawWord{17-0} = regaddr;
  let RawWord{23-18} = tdmadataregindex;
}

def PseudoTTMFCONV3S1Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$rotate_weights, GPR:$clear_dvalid)>;
def PseudoTTMFCONV3S1Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$rotate_weights, uimm32:$clear_dvalid)>;
def TTMFCONV3S1 : TensixOrdinaryInst<"MFCONV3S1", 0x3a,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$rotate_weights, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $rotate_weights, $clear_dvalid",
    ["dst", "addr_mode", "rotate_weights", "clear_dvalid"],
    [16383, 7, 31, 3]> {
  bits<14> dst;
  bits<3> addr_mode;
  bits<5> rotate_weights;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{16-14} = addr_mode;
  let RawWord{21-17} = rotate_weights;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTMOPPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$zmask_lo16_or_loop_count, GPR:$loop_count, GPR:$mop_type)>;
def PseudoTTMOPMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$zmask_lo16_or_loop_count, uimm32:$loop_count, uimm32:$mop_type)>;
def TTMOP : TensixOrdinaryInst<"MOP", 0x01,
    (ins uimm32:$zmask_lo16_or_loop_count, uimm32:$loop_count, uimm32:$mop_type),
    "$zmask_lo16_or_loop_count, $loop_count, $mop_type",
    ["zmask_lo16_or_loop_count", "loop_count", "mop_type"],
    [65535, 127, 1]> {
  bits<16> zmask_lo16_or_loop_count;
  bits<7> loop_count;
  bits<1> mop_type;
  let RawWord{15-0} = zmask_lo16_or_loop_count;
  let RawWord{22-16} = loop_count;
  let RawWord{23} = mop_type;
}

def PseudoTTMOP_CFGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$zmask_hi16)>;
def PseudoTTMOP_CFGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$zmask_hi16)>;
def TTMOP_CFG : TensixOrdinaryInst<"MOP_CFG", 0x03,
    (ins uimm32:$zmask_hi16),
    "$zmask_hi16",
    ["zmask_hi16"],
    [16777215]> {
  bits<24> zmask_hi16;
  let RawWord{23-0} = zmask_hi16;
}

def PseudoTTMOVA2DPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$instr_mod, GPR:$addr_mode, GPR:$src, GPR:$dest_32b_lo)>;
def PseudoTTMOVA2DMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo)>;
def TTMOVA2D : TensixOrdinaryInst<"MOVA2D", 0x12,
    (ins uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo),
    "$dst, $instr_mod, $addr_mode, $src, $dest_32b_lo",
    ["dst", "instr_mod", "addr_mode", "src", "dest_32b_lo"],
    [4095, 3, 7, 63, 1]> {
  bits<12> dst;
  bits<2> instr_mod;
  bits<3> addr_mode;
  bits<6> src;
  bits<1> dest_32b_lo;
  let RawWord{11-0} = dst;
  let RawWord{13-12} = instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{22-17} = src;
  let RawWord{23} = dest_32b_lo;
}

def PseudoTTMOVB2APort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$srcb, GPR:$instr_mod, GPR:$addr_mode, GPR:$srca)>;
def PseudoTTMOVB2AMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$srcb, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$srca)>;
def TTMOVB2A : TensixOrdinaryInst<"MOVB2A", 0x0b,
    (ins uimm32:$srcb, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$srca),
    "$srcb, $instr_mod, $addr_mode, $srca",
    ["srcb", "instr_mod", "addr_mode", "srca"],
    [4095, 3, 7, 127]> {
  bits<12> srcb;
  bits<2> instr_mod;
  bits<3> addr_mode;
  bits<7> srca;
  let RawWord{11-0} = srcb;
  let RawWord{13-12} = instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{23-17} = srca;
}

def PseudoTTMOVB2DPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$movb2d_instr_mod, GPR:$addr_mode, GPR:$src, GPR:$dest_32b_lo)>;
def PseudoTTMOVB2DMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$movb2d_instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo)>;
def TTMOVB2D : TensixOrdinaryInst<"MOVB2D", 0x13,
    (ins uimm32:$dst, uimm32:$movb2d_instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo),
    "$dst, $movb2d_instr_mod, $addr_mode, $src, $dest_32b_lo",
    ["dst", "movb2d_instr_mod", "addr_mode", "src", "dest_32b_lo"],
    [2047, 7, 7, 63, 1]> {
  bits<11> dst;
  bits<3> movb2d_instr_mod;
  bits<3> addr_mode;
  bits<6> src;
  bits<1> dest_32b_lo;
  let RawWord{10-0} = dst;
  let RawWord{13-11} = movb2d_instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{22-17} = src;
  let RawWord{23} = dest_32b_lo;
}

def PseudoTTMOVD2APort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$instr_mod, GPR:$addr_mode, GPR:$src, GPR:$dest_32b_lo)>;
def PseudoTTMOVD2AMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo)>;
def TTMOVD2A : TensixOrdinaryInst<"MOVD2A", 0x08,
    (ins uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo),
    "$dst, $instr_mod, $addr_mode, $src, $dest_32b_lo",
    ["dst", "instr_mod", "addr_mode", "src", "dest_32b_lo"],
    [4095, 3, 7, 63, 1]> {
  bits<12> dst;
  bits<2> instr_mod;
  bits<3> addr_mode;
  bits<6> src;
  bits<1> dest_32b_lo;
  let RawWord{11-0} = dst;
  let RawWord{13-12} = instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{22-17} = src;
  let RawWord{23} = dest_32b_lo;
}

def PseudoTTMOVD2BPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$instr_mod, GPR:$addr_mode, GPR:$src, GPR:$dest_32b_lo)>;
def PseudoTTMOVD2BMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo)>;
def TTMOVD2B : TensixOrdinaryInst<"MOVD2B", 0x0a,
    (ins uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo),
    "$dst, $instr_mod, $addr_mode, $src, $dest_32b_lo",
    ["dst", "instr_mod", "addr_mode", "src", "dest_32b_lo"],
    [4095, 3, 7, 63, 1]> {
  bits<12> dst;
  bits<2> instr_mod;
  bits<3> addr_mode;
  bits<6> src;
  bits<1> dest_32b_lo;
  let RawWord{11-0} = dst;
  let RawWord{13-12} = instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{22-17} = src;
  let RawWord{23} = dest_32b_lo;
}

def PseudoTTMOVDBGA2DPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$instr_mod, GPR:$addr_mode, GPR:$src, GPR:$dest_32b_lo)>;
def PseudoTTMOVDBGA2DMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo)>;
def TTMOVDBGA2D : TensixOrdinaryInst<"MOVDBGA2D", 0x09,
    (ins uimm32:$dst, uimm32:$instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo),
    "$dst, $instr_mod, $addr_mode, $src, $dest_32b_lo",
    ["dst", "instr_mod", "addr_mode", "src", "dest_32b_lo"],
    [4095, 3, 7, 63, 1]> {
  bits<12> dst;
  bits<2> instr_mod;
  bits<3> addr_mode;
  bits<6> src;
  bits<1> dest_32b_lo;
  let RawWord{11-0} = dst;
  let RawWord{13-12} = instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{22-17} = src;
  let RawWord{23} = dest_32b_lo;
}

def PseudoTTMOVDBGB2DPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$movb2d_instr_mod, GPR:$addr_mode, GPR:$src, GPR:$dest_32b_lo)>;
def PseudoTTMOVDBGB2DMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$movb2d_instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo)>;
def TTMOVDBGB2D : TensixOrdinaryInst<"MOVDBGB2D", 0x0c,
    (ins uimm32:$dst, uimm32:$movb2d_instr_mod, uimm32:$addr_mode, uimm32:$src, uimm32:$dest_32b_lo),
    "$dst, $movb2d_instr_mod, $addr_mode, $src, $dest_32b_lo",
    ["dst", "movb2d_instr_mod", "addr_mode", "src", "dest_32b_lo"],
    [2047, 7, 7, 63, 1]> {
  bits<11> dst;
  bits<3> movb2d_instr_mod;
  bits<3> addr_mode;
  bits<6> src;
  bits<1> dest_32b_lo;
  let RawWord{10-0} = dst;
  let RawWord{13-11} = movb2d_instr_mod;
  let RawWord{16-14} = addr_mode;
  let RawWord{22-17} = src;
  let RawWord{23} = dest_32b_lo;
}

def PseudoTTMPOOL3S1Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$index_en, GPR:$pool_addr_mode, GPR:$clear_dvalid)>;
def PseudoTTMPOOL3S1Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid)>;
def TTMPOOL3S1 : TensixOrdinaryInst<"MPOOL3S1", 0x24,
    (ins uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid),
    "$dst, $index_en, $pool_addr_mode, $clear_dvalid",
    ["dst", "index_en", "pool_addr_mode", "clear_dvalid"],
    [16383, 1, 127, 3]> {
  bits<14> dst;
  bits<1> index_en;
  bits<7> pool_addr_mode;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{14} = index_en;
  let RawWord{21-15} = pool_addr_mode;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTMPOOL3S2Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$index_en, GPR:$pool_addr_mode, GPR:$clear_dvalid)>;
def PseudoTTMPOOL3S2Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid)>;
def TTMPOOL3S2 : TensixOrdinaryInst<"MPOOL3S2", 0x31,
    (ins uimm32:$dst, uimm32:$index_en, uimm32:$pool_addr_mode, uimm32:$clear_dvalid),
    "$dst, $index_en, $pool_addr_mode, $clear_dvalid",
    ["dst", "index_en", "pool_addr_mode", "clear_dvalid"],
    [16383, 1, 127, 3]> {
  bits<14> dst;
  bits<1> index_en;
  bits<7> pool_addr_mode;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{14} = index_en;
  let RawWord{21-15} = pool_addr_mode;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTMULDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$oparegindex, GPR:$opbregindex, GPR:$resultregindex, GPR:$opbisconst)>;
def PseudoTTMULDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opbisconst)>;
def TTMULDMAREG : TensixOrdinaryInst<"MULDMAREG", 0x5a,
    (ins uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opbisconst),
    "$oparegindex, $opbregindex, $resultregindex, $opbisconst",
    ["oparegindex", "opbregindex", "resultregindex", "opbisconst"],
    [63, 63, 2047, 1]> {
  bits<6> oparegindex;
  bits<6> opbregindex;
  bits<11> resultregindex;
  bits<1> opbisconst;
  let RawWord{5-0} = oparegindex;
  let RawWord{11-6} = opbregindex;
  let RawWord{22-12} = resultregindex;
  let RawWord{23} = opbisconst;
}

def PseudoTTMVMULPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$dst, GPR:$addr_mode, GPR:$instr_mod19, GPR:$clear_dvalid)>;
def PseudoTTMVMULMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$clear_dvalid)>;
def TTMVMUL : TensixOrdinaryInst<"MVMUL", 0x26,
    (ins uimm32:$dst, uimm32:$addr_mode, uimm32:$instr_mod19, uimm32:$clear_dvalid),
    "$dst, $addr_mode, $instr_mod19, $clear_dvalid",
    ["dst", "addr_mode", "instr_mod19", "clear_dvalid"],
    [16383, 31, 7, 3]> {
  bits<14> dst;
  bits<5> addr_mode;
  bits<3> instr_mod19;
  bits<2> clear_dvalid;
  let RawWord{13-0} = dst;
  let RawWord{18-14} = addr_mode;
  let RawWord{21-19} = instr_mod19;
  let RawWord{23-22} = clear_dvalid;
}

def PseudoTTNOPPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTNOPMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTNOP : TensixOrdinaryInst<"NOP", 0x02,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTPACRPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$last, GPR:$flush, GPR:$ctxtctrl, GPR:$concat, GPR:$ovrdthreadid, GPR:$readintfsel, GPR:$zerowrite, GPR:$addrcntcontext, GPR:$addrmode, GPR:$dstaccessmode, GPR:$rowpadzero, GPR:$cfgcontext)>;
def PseudoTTPACRMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$last, uimm32:$flush, uimm32:$ctxtctrl, uimm32:$concat, uimm32:$ovrdthreadid, uimm32:$readintfsel, uimm32:$zerowrite, uimm32:$addrcntcontext, uimm32:$addrmode, uimm32:$dstaccessmode, uimm32:$rowpadzero, uimm32:$cfgcontext)>;
def TTPACR : TensixOrdinaryInst<"PACR", 0x41,
    (ins uimm32:$last, uimm32:$flush, uimm32:$ctxtctrl, uimm32:$concat, uimm32:$ovrdthreadid, uimm32:$readintfsel, uimm32:$zerowrite, uimm32:$addrcntcontext, uimm32:$addrmode, uimm32:$dstaccessmode, uimm32:$rowpadzero, uimm32:$cfgcontext),
    "$last, $flush, $ctxtctrl, $concat, $ovrdthreadid, $readintfsel, $zerowrite, $addrcntcontext, $addrmode, $dstaccessmode, $rowpadzero, $cfgcontext",
    ["last", "flush", "ctxtctrl", "concat", "ovrdthreadid", "readintfsel", "zerowrite", "addrcntcontext", "addrmode", "dstaccessmode", "rowpadzero", "cfgcontext"],
    [1, 1, 3, 7, 1, 15, 1, 3, 3, 1, 7, 7]> {
  bits<1> last;
  bits<1> flush;
  bits<2> ctxtctrl;
  bits<3> concat;
  bits<1> ovrdthreadid;
  bits<4> readintfsel;
  bits<1> zerowrite;
  bits<2> addrcntcontext;
  bits<2> addrmode;
  bits<1> dstaccessmode;
  bits<3> rowpadzero;
  bits<3> cfgcontext;
  let RawWord{0} = last;
  let RawWord{1} = flush;
  let RawWord{3-2} = ctxtctrl;
  let RawWord{6-4} = concat;
  let RawWord{7} = ovrdthreadid;
  let RawWord{11-8} = readintfsel;
  let RawWord{12} = zerowrite;
  let RawWord{14-13} = addrcntcontext;
  let RawWord{16-15} = addrmode;
  let RawWord{17} = dstaccessmode;
  let RawWord{20-18} = rowpadzero;
  let RawWord{23-21} = cfgcontext;
}

def PseudoTTPACR_SETREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$last, GPR:$flush, GPR:$streamid, GPR:$addrsel, GPR:$disablestall, GPR:$unused, GPR:$modesel, GPR:$push)>;
def PseudoTTPACR_SETREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$last, uimm32:$flush, uimm32:$streamid, uimm32:$addrsel, uimm32:$disablestall, uimm32:$unused, uimm32:$modesel, uimm32:$push)>;
def TTPACR_SETREG : TensixOrdinaryInst<"PACR_SETREG", 0x4a,
    (ins uimm32:$last, uimm32:$flush, uimm32:$streamid, uimm32:$addrsel, uimm32:$disablestall, uimm32:$unused, uimm32:$modesel, uimm32:$push),
    "$last, $flush, $streamid, $addrsel, $disablestall, $unused, $modesel, $push",
    ["last", "flush", "streamid", "addrsel", "disablestall", "unused", "modesel", "push"],
    [1, 1, 63, 3, 3, 1023, 1, 1]> {
  bits<1> last;
  bits<1> flush;
  bits<6> streamid;
  bits<2> addrsel;
  bits<2> disablestall;
  bits<10> unused;
  bits<1> modesel;
  bits<1> push;
  let RawWord{0} = last;
  let RawWord{1} = flush;
  let RawWord{7-2} = streamid;
  let RawWord{9-8} = addrsel;
  let RawWord{11-10} = disablestall;
  let RawWord{21-12} = unused;
  let RawWord{22} = modesel;
  let RawWord{23} = push;
}

def PseudoTTRAREBPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTRAREBMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTRAREB : TensixOrdinaryInst<"RAREB", 0x15,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTRDCFGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgreg, GPR:$gpraddress)>;
def PseudoTTRDCFGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgreg, uimm32:$gpraddress)>;
def TTRDCFG : TensixOrdinaryInst<"RDCFG", 0xb1,
    (ins uimm32:$cfgreg, uimm32:$gpraddress),
    "$cfgreg, $gpraddress",
    ["cfgreg", "gpraddress"],
    [65535, 255]> {
  bits<16> cfgreg;
  bits<8> gpraddress;
  let RawWord{15-0} = cfgreg;
  let RawWord{23-16} = gpraddress;
}

def PseudoTTREG2FLOPPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$regindex, GPR:$flopindex, GPR:$contextid_2, GPR:$byteoffset, GPR:$targetsel, GPR:$sizesel)>;
def PseudoTTREG2FLOPMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$regindex, uimm32:$flopindex, uimm32:$contextid_2, uimm32:$byteoffset, uimm32:$targetsel, uimm32:$sizesel)>;
def TTREG2FLOP : TensixOrdinaryInst<"REG2FLOP", 0x48,
    (ins uimm32:$regindex, uimm32:$flopindex, uimm32:$contextid_2, uimm32:$byteoffset, uimm32:$targetsel, uimm32:$sizesel),
    "$regindex, $flopindex, $contextid_2, $byteoffset, $targetsel, $sizesel",
    ["regindex", "flopindex", "contextid_2", "byteoffset", "targetsel", "sizesel"],
    [63, 1023, 3, 3, 3, 3]> {
  bits<6> regindex;
  bits<10> flopindex;
  bits<2> contextid_2;
  bits<2> byteoffset;
  bits<2> targetsel;
  bits<2> sizesel;
  let RawWord{5-0} = regindex;
  let RawWord{15-6} = flopindex;
  let RawWord{17-16} = contextid_2;
  let RawWord{19-18} = byteoffset;
  let RawWord{21-20} = targetsel;
  let RawWord{23-22} = sizesel;
}

def PseudoTTREPLAYPort : TensixOrdinaryPort<
    (ins uimm8:$port, uimm32:$load_mode, uimm32:$execute_while_loading, uimm32:$len, uimm32:$start_idx)>;
def PseudoTTREPLAYMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$load_mode, uimm32:$execute_while_loading, uimm32:$len, uimm32:$start_idx)>;
def TTREPLAY : TensixOrdinaryInst<"REPLAY", 0x04,
    (ins uimm32:$load_mode, uimm32:$execute_while_loading, uimm32:$len, uimm32:$start_idx),
    "$load_mode, $execute_while_loading, $len, $start_idx",
    ["load_mode", "execute_while_loading", "len", "start_idx"],
    [1, 7, 1023, 1023]> {
  bits<1> load_mode;
  bits<3> execute_while_loading;
  bits<10> len;
  bits<10> start_idx;
  let RawWord{0} = load_mode;
  let RawWord{3-1} = execute_while_loading;
  let RawWord{13-4} = len;
  let RawWord{23-14} = start_idx;
}

def PseudoTTRESOURCEDECLPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$op_class, GPR:$resources, GPR:$linger_time)>;
def PseudoTTRESOURCEDECLMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$op_class, uimm32:$resources, uimm32:$linger_time)>;
def TTRESOURCEDECL : TensixOrdinaryInst<"RESOURCEDECL", 0x05,
    (ins uimm32:$op_class, uimm32:$resources, uimm32:$linger_time),
    "$op_class, $resources, $linger_time",
    ["op_class", "resources", "linger_time"],
    [15, 511, 2047]> {
  bits<4> op_class;
  bits<9> resources;
  bits<11> linger_time;
  let RawWord{3-0} = op_class;
  let RawWord{12-4} = resources;
  let RawWord{23-13} = linger_time;
}

def PseudoTTRMWCIB0Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgregaddr, GPR:$data, GPR:$mask)>;
def PseudoTTRMWCIB0Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask)>;
def TTRMWCIB0 : TensixOrdinaryInst<"RMWCIB0", 0xb3,
    (ins uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask),
    "$cfgregaddr, $data, $mask",
    ["cfgregaddr", "data", "mask"],
    [255, 255, 255]> {
  bits<8> cfgregaddr;
  bits<8> data;
  bits<8> mask;
  let RawWord{7-0} = cfgregaddr;
  let RawWord{15-8} = data;
  let RawWord{23-16} = mask;
}

def PseudoTTRMWCIB1Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgregaddr, GPR:$data, GPR:$mask)>;
def PseudoTTRMWCIB1Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask)>;
def TTRMWCIB1 : TensixOrdinaryInst<"RMWCIB1", 0xb4,
    (ins uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask),
    "$cfgregaddr, $data, $mask",
    ["cfgregaddr", "data", "mask"],
    [255, 255, 255]> {
  bits<8> cfgregaddr;
  bits<8> data;
  bits<8> mask;
  let RawWord{7-0} = cfgregaddr;
  let RawWord{15-8} = data;
  let RawWord{23-16} = mask;
}

def PseudoTTRMWCIB2Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgregaddr, GPR:$data, GPR:$mask)>;
def PseudoTTRMWCIB2Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask)>;
def TTRMWCIB2 : TensixOrdinaryInst<"RMWCIB2", 0xb5,
    (ins uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask),
    "$cfgregaddr, $data, $mask",
    ["cfgregaddr", "data", "mask"],
    [255, 255, 255]> {
  bits<8> cfgregaddr;
  bits<8> data;
  bits<8> mask;
  let RawWord{7-0} = cfgregaddr;
  let RawWord{15-8} = data;
  let RawWord{23-16} = mask;
}

def PseudoTTRMWCIB3Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgregaddr, GPR:$data, GPR:$mask)>;
def PseudoTTRMWCIB3Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask)>;
def TTRMWCIB3 : TensixOrdinaryInst<"RMWCIB3", 0xb6,
    (ins uimm32:$cfgregaddr, uimm32:$data, uimm32:$mask),
    "$cfgregaddr, $data, $mask",
    ["cfgregaddr", "data", "mask"],
    [255, 255, 255]> {
  bits<8> cfgregaddr;
  bits<8> data;
  bits<8> mask;
  let RawWord{7-0} = cfgregaddr;
  let RawWord{15-8} = data;
  let RawWord{23-16} = mask;
}

def PseudoTTRSTDMAPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTRSTDMAMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTRSTDMA : TensixOrdinaryInst<"RSTDMA", 0x44,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTSEMGETPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$sem_sel)>;
def PseudoTTSEMGETMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$sem_sel)>;
def TTSEMGET : TensixOrdinaryInst<"SEMGET", 0xa5,
    (ins uimm32:$sem_sel),
    "$sem_sel",
    ["sem_sel"],
    [4194303]> {
  bits<22> sem_sel;
  let RawWord{23-2} = sem_sel;
}

def PseudoTTSEMINITPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$sem_sel, GPR:$init_value, GPR:$max_value)>;
def PseudoTTSEMINITMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$sem_sel, uimm32:$init_value, uimm32:$max_value)>;
def TTSEMINIT : TensixOrdinaryInst<"SEMINIT", 0xa3,
    (ins uimm32:$sem_sel, uimm32:$init_value, uimm32:$max_value),
    "$sem_sel, $init_value, $max_value",
    ["sem_sel", "init_value", "max_value"],
    [16383, 15, 15]> {
  bits<14> sem_sel;
  bits<4> init_value;
  bits<4> max_value;
  let RawWord{15-2} = sem_sel;
  let RawWord{19-16} = init_value;
  let RawWord{23-20} = max_value;
}

def PseudoTTSEMPOSTPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$sem_sel)>;
def PseudoTTSEMPOSTMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$sem_sel)>;
def TTSEMPOST : TensixOrdinaryInst<"SEMPOST", 0xa4,
    (ins uimm32:$sem_sel),
    "$sem_sel",
    ["sem_sel"],
    [4194303]> {
  bits<22> sem_sel;
  let RawWord{23-2} = sem_sel;
}

def PseudoTTSEMWAITPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$wait_sem_cond, GPR:$sem_sel, GPR:$stall_res)>;
def PseudoTTSEMWAITMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$wait_sem_cond, uimm32:$sem_sel, uimm32:$stall_res)>;
def TTSEMWAIT : TensixOrdinaryInst<"SEMWAIT", 0xa6,
    (ins uimm32:$wait_sem_cond, uimm32:$sem_sel, uimm32:$stall_res),
    "$wait_sem_cond, $sem_sel, $stall_res",
    ["wait_sem_cond", "sem_sel", "stall_res"],
    [3, 8191, 511]> {
  bits<2> wait_sem_cond;
  bits<13> sem_sel;
  bits<9> stall_res;
  let RawWord{1-0} = wait_sem_cond;
  let RawWord{14-2} = sem_sel;
  let RawWord{23-15} = stall_res;
}

def PseudoTTSETADCPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$value, GPR:$dimensionindex, GPR:$channelindex, GPR:$cntsetmask)>;
def PseudoTTSETADCMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$value, uimm32:$dimensionindex, uimm32:$channelindex, uimm32:$cntsetmask)>;
def TTSETADC : TensixOrdinaryInst<"SETADC", 0x50,
    (ins uimm32:$value, uimm32:$dimensionindex, uimm32:$channelindex, uimm32:$cntsetmask),
    "$value, $dimensionindex, $channelindex, $cntsetmask",
    ["value", "dimensionindex", "channelindex", "cntsetmask"],
    [262143, 3, 1, 7]> {
  bits<18> value;
  bits<2> dimensionindex;
  bits<1> channelindex;
  bits<3> cntsetmask;
  let RawWord{17-0} = value;
  let RawWord{19-18} = dimensionindex;
  let RawWord{20} = channelindex;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTSETADCXXPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$x_start, GPR:$x_end2, GPR:$cntsetmask)>;
def PseudoTTSETADCXXMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$x_start, uimm32:$x_end2, uimm32:$cntsetmask)>;
def TTSETADCXX : TensixOrdinaryInst<"SETADCXX", 0x5e,
    (ins uimm32:$x_start, uimm32:$x_end2, uimm32:$cntsetmask),
    "$x_start, $x_end2, $cntsetmask",
    ["x_start", "x_end2", "cntsetmask"],
    [1023, 2047, 7]> {
  bits<10> x_start;
  bits<11> x_end2;
  bits<3> cntsetmask;
  let RawWord{9-0} = x_start;
  let RawWord{20-10} = x_end2;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTSETADCXYPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$bitmask, GPR:$ch0_x, GPR:$ch0_y, GPR:$ch1_x, GPR:$ch1_y, GPR:$cntsetmask)>;
def PseudoTTSETADCXYMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask)>;
def TTSETADCXY : TensixOrdinaryInst<"SETADCXY", 0x51,
    (ins uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask),
    "$bitmask, $ch0_x, $ch0_y, $ch1_x, $ch1_y, $cntsetmask",
    ["bitmask", "ch0_x", "ch0_y", "ch1_x", "ch1_y", "cntsetmask"],
    [63, 7, 7, 7, 63, 7]> {
  bits<6> bitmask;
  bits<3> ch0_x;
  bits<3> ch0_y;
  bits<3> ch1_x;
  bits<6> ch1_y;
  bits<3> cntsetmask;
  let RawWord{5-0} = bitmask;
  let RawWord{8-6} = ch0_x;
  let RawWord{11-9} = ch0_y;
  let RawWord{14-12} = ch1_x;
  let RawWord{20-15} = ch1_y;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTSETADCZWPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$bitmask, GPR:$ch0_x, GPR:$ch0_y, GPR:$ch1_x, GPR:$ch1_y, GPR:$cntsetmask)>;
def PseudoTTSETADCZWMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask)>;
def TTSETADCZW : TensixOrdinaryInst<"SETADCZW", 0x54,
    (ins uimm32:$bitmask, uimm32:$ch0_x, uimm32:$ch0_y, uimm32:$ch1_x, uimm32:$ch1_y, uimm32:$cntsetmask),
    "$bitmask, $ch0_x, $ch0_y, $ch1_x, $ch1_y, $cntsetmask",
    ["bitmask", "ch0_x", "ch0_y", "ch1_x", "ch1_y", "cntsetmask"],
    [63, 7, 7, 7, 63, 7]> {
  bits<6> bitmask;
  bits<3> ch0_x;
  bits<3> ch0_y;
  bits<3> ch1_x;
  bits<6> ch1_y;
  bits<3> cntsetmask;
  let RawWord{5-0} = bitmask;
  let RawWord{8-6} = ch0_x;
  let RawWord{11-9} = ch0_y;
  let RawWord{14-12} = ch1_x;
  let RawWord{20-15} = ch1_y;
  let RawWord{23-21} = cntsetmask;
}

def PseudoTTSETASHRMHPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$halo_mask, GPR:$reg_mask)>;
def PseudoTTSETASHRMHMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$halo_mask, uimm32:$reg_mask)>;
def TTSETASHRMH : TensixOrdinaryInst<"SETASHRMH", 0x1e,
    (ins uimm32:$halo_mask, uimm32:$reg_mask),
    "$halo_mask, $reg_mask",
    ["halo_mask", "reg_mask"],
    [1, 8388607]> {
  bits<1> halo_mask;
  bits<23> reg_mask;
  let RawWord{0} = halo_mask;
  let RawWord{23-1} = reg_mask;
}

def PseudoTTSETASHRMH0Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$halo_mask, GPR:$reg_mask)>;
def PseudoTTSETASHRMH0Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$halo_mask, uimm32:$reg_mask)>;
def TTSETASHRMH0 : TensixOrdinaryInst<"SETASHRMH0", 0x1a,
    (ins uimm32:$halo_mask, uimm32:$reg_mask),
    "$halo_mask, $reg_mask",
    ["halo_mask", "reg_mask"],
    [1, 8388607]> {
  bits<1> halo_mask;
  bits<23> reg_mask;
  let RawWord{0} = halo_mask;
  let RawWord{23-1} = reg_mask;
}

def PseudoTTSETASHRMH1Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$halo_mask, GPR:$reg_mask)>;
def PseudoTTSETASHRMH1Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$halo_mask, uimm32:$reg_mask)>;
def TTSETASHRMH1 : TensixOrdinaryInst<"SETASHRMH1", 0x1b,
    (ins uimm32:$halo_mask, uimm32:$reg_mask),
    "$halo_mask, $reg_mask",
    ["halo_mask", "reg_mask"],
    [1, 8388607]> {
  bits<1> halo_mask;
  bits<23> reg_mask;
  let RawWord{0} = halo_mask;
  let RawWord{23-1} = reg_mask;
}

def PseudoTTSETASHRMVPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$reg_mask2)>;
def PseudoTTSETASHRMVMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$reg_mask2)>;
def TTSETASHRMV : TensixOrdinaryInst<"SETASHRMV", 0x1c,
    (ins uimm32:$reg_mask2),
    "$reg_mask2",
    ["reg_mask2"],
    [16777215]> {
  bits<24> reg_mask2;
  let RawWord{23-0} = reg_mask2;
}

def PseudoTTSETC16Port : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$setc16_value, GPR:$setc16_reg)>;
def PseudoTTSETC16Mop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$setc16_value, uimm32:$setc16_reg)>;
def TTSETC16 : TensixOrdinaryInst<"SETC16", 0xb2,
    (ins uimm32:$setc16_value, uimm32:$setc16_reg),
    "$setc16_value, $setc16_reg",
    ["setc16_value", "setc16_reg"],
    [65535, 67]> {
  bits<16> setc16_value;
  bits<8> setc16_reg;
  let RawWord{15-0} = setc16_value;
  let RawWord{23-16} = setc16_reg;
}

def PseudoTTSETDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$regindex16b, GPR:$setsignalsmode, GPR:$payload_sigsel, GPR:$payload_sigselsize)>;
def PseudoTTSETDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$regindex16b, uimm32:$setsignalsmode, uimm32:$payload_sigsel, uimm32:$payload_sigselsize)>;
def TTSETDMAREG : TensixOrdinaryInst<"SETDMAREG", 0x45,
    (ins uimm32:$regindex16b, uimm32:$setsignalsmode, uimm32:$payload_sigsel, uimm32:$payload_sigselsize),
    "$regindex16b, $setsignalsmode, $payload_sigsel, $payload_sigselsize",
    ["regindex16b", "setsignalsmode", "payload_sigsel", "payload_sigselsize"],
    [127, 1, 16383, 3]> {
  bits<7> regindex16b;
  bits<1> setsignalsmode;
  bits<14> payload_sigsel;
  bits<2> payload_sigselsize;
  let RawWord{6-0} = regindex16b;
  let RawWord{7} = setsignalsmode;
  let RawWord{21-8} = payload_sigsel;
  let RawWord{23-22} = payload_sigselsize;
}

def PseudoTTSETDVALIDPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$setvalid)>;
def PseudoTTSETDVALIDMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$setvalid)>;
def TTSETDVALID : TensixOrdinaryInst<"SETDVALID", 0x57,
    (ins uimm32:$setvalid),
    "$setvalid",
    ["setvalid"],
    [16777215]> {
  bits<24> setvalid;
  let RawWord{23-0} = setvalid;
}

def PseudoTTSETIBRWCPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$set_inc_ctrl, GPR:$rwc_bias, GPR:$rwc_cr)>;
def PseudoTTSETIBRWCMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$set_inc_ctrl, uimm32:$rwc_bias, uimm32:$rwc_cr)>;
def TTSETIBRWC : TensixOrdinaryInst<"SETIBRWC", 0x39,
    (ins uimm32:$set_inc_ctrl, uimm32:$rwc_bias, uimm32:$rwc_cr),
    "$set_inc_ctrl, $rwc_bias, $rwc_cr",
    ["set_inc_ctrl", "rwc_bias", "rwc_cr"],
    [63, 4095, 63]> {
  bits<6> set_inc_ctrl;
  bits<12> rwc_bias;
  bits<6> rwc_cr;
  let RawWord{5-0} = set_inc_ctrl;
  let RawWord{17-6} = rwc_bias;
  let RawWord{23-18} = rwc_cr;
}

def PseudoTTSETPKEDGOFPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$x_start, GPR:$x_end, GPR:$y_start, GPR:$y_end)>;
def PseudoTTSETPKEDGOFMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$x_start, uimm32:$x_end, uimm32:$y_start, uimm32:$y_end)>;
def TTSETPKEDGOF : TensixOrdinaryInst<"SETPKEDGOF", 0x1d,
    (ins uimm32:$x_start, uimm32:$x_end, uimm32:$y_start, uimm32:$y_end),
    "$x_start, $x_end, $y_start, $y_end",
    ["x_start", "x_end", "y_start", "y_end"],
    [15, 15, 15, 4095]> {
  bits<4> x_start;
  bits<4> x_end;
  bits<4> y_start;
  bits<12> y_end;
  let RawWord{3-0} = x_start;
  let RawWord{7-4} = x_end;
  let RawWord{11-8} = y_start;
  let RawWord{23-12} = y_end;
}

def PseudoTTSETRWCPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$bitmask, GPR:$rwc_a, GPR:$rwc_b, GPR:$rwc_d, GPR:$rwc_cr, GPR:$clear_ab_vld)>;
def PseudoTTSETRWCMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$bitmask, uimm32:$rwc_a, uimm32:$rwc_b, uimm32:$rwc_d, uimm32:$rwc_cr, uimm32:$clear_ab_vld)>;
def TTSETRWC : TensixOrdinaryInst<"SETRWC", 0x37,
    (ins uimm32:$bitmask, uimm32:$rwc_a, uimm32:$rwc_b, uimm32:$rwc_d, uimm32:$rwc_cr, uimm32:$clear_ab_vld),
    "$bitmask, $rwc_a, $rwc_b, $rwc_d, $rwc_cr, $clear_ab_vld",
    ["bitmask", "rwc_a", "rwc_b", "rwc_d", "rwc_cr", "clear_ab_vld"],
    [63, 15, 15, 15, 15, 3]> {
  bits<6> bitmask;
  bits<4> rwc_a;
  bits<4> rwc_b;
  bits<4> rwc_d;
  bits<4> rwc_cr;
  bits<2> clear_ab_vld;
  let RawWord{5-0} = bitmask;
  let RawWord{9-6} = rwc_a;
  let RawWord{13-10} = rwc_b;
  let RawWord{17-14} = rwc_d;
  let RawWord{21-18} = rwc_cr;
  let RawWord{23-22} = clear_ab_vld;
}

def PseudoTTSHIFTDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$oparegindex, GPR:$opbregindex, GPR:$resultregindex, GPR:$opsel, GPR:$opbisconst)>;
def PseudoTTSHIFTDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opsel, uimm32:$opbisconst)>;
def TTSHIFTDMAREG : TensixOrdinaryInst<"SHIFTDMAREG", 0x5c,
    (ins uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opsel, uimm32:$opbisconst),
    "$oparegindex, $opbregindex, $resultregindex, $opsel, $opbisconst",
    ["oparegindex", "opbregindex", "resultregindex", "opsel", "opbisconst"],
    [63, 63, 63, 31, 1]> {
  bits<6> oparegindex;
  bits<6> opbregindex;
  bits<6> resultregindex;
  bits<5> opsel;
  bits<1> opbisconst;
  let RawWord{5-0} = oparegindex;
  let RawWord{11-6} = opbregindex;
  let RawWord{17-12} = resultregindex;
  let RawWord{22-18} = opsel;
  let RawWord{23} = opbisconst;
}

def PseudoTTSHIFTXAPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$shift_mode, GPR:$log2_amount2)>;
def PseudoTTSHIFTXAMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$shift_mode, uimm32:$log2_amount2)>;
def TTSHIFTXA : TensixOrdinaryInst<"SHIFTXA", 0x17,
    (ins uimm32:$shift_mode, uimm32:$log2_amount2),
    "$shift_mode, $log2_amount2",
    ["shift_mode", "log2_amount2"],
    [3, 4194303]> {
  bits<2> shift_mode;
  bits<22> log2_amount2;
  let RawWord{1-0} = shift_mode;
  let RawWord{23-2} = log2_amount2;
}

def PseudoTTSHIFTXBPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$shift_row, GPR:$rot_shift, GPR:$addr_mode)>;
def PseudoTTSHIFTXBMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$shift_row, uimm32:$rot_shift, uimm32:$addr_mode)>;
def TTSHIFTXB : TensixOrdinaryInst<"SHIFTXB", 0x18,
    (ins uimm32:$shift_row, uimm32:$rot_shift, uimm32:$addr_mode),
    "$shift_row, $rot_shift, $addr_mode",
    ["shift_row", "rot_shift", "addr_mode"],
    [1023, 15, 1023]> {
  bits<10> shift_row;
  bits<4> rot_shift;
  bits<10> addr_mode;
  let RawWord{9-0} = shift_row;
  let RawWord{13-10} = rot_shift;
  let RawWord{23-14} = addr_mode;
}

def PseudoTTSTALLWAITPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$wait_res, GPR:$stall_res)>;
def PseudoTTSTALLWAITMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$wait_res, uimm32:$stall_res)>;
def TTSTALLWAIT : TensixOrdinaryInst<"STALLWAIT", 0xa2,
    (ins uimm32:$wait_res, uimm32:$stall_res),
    "$wait_res, $stall_res",
    ["wait_res", "stall_res"],
    [32767, 511]> {
  bits<15> wait_res;
  bits<9> stall_res;
  let RawWord{14-0} = wait_res;
  let RawWord{23-15} = stall_res;
}

def PseudoTTSTOREINDPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$addrregindex, GPR:$dataregindex, GPR:$autoincspec, GPR:$offsetindex, GPR:$regsizesel, GPR:$sizesel, GPR:$memhiersel)>;
def PseudoTTSTOREINDMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$autoincspec, uimm32:$offsetindex, uimm32:$regsizesel, uimm32:$sizesel, uimm32:$memhiersel)>;
def TTSTOREIND : TensixOrdinaryInst<"STOREIND", 0x66,
    (ins uimm32:$addrregindex, uimm32:$dataregindex, uimm32:$autoincspec, uimm32:$offsetindex, uimm32:$regsizesel, uimm32:$sizesel, uimm32:$memhiersel),
    "$addrregindex, $dataregindex, $autoincspec, $offsetindex, $regsizesel, $sizesel, $memhiersel",
    ["addrregindex", "dataregindex", "autoincspec", "offsetindex", "regsizesel", "sizesel", "memhiersel"],
    [63, 63, 3, 127, 1, 1, 1]> {
  bits<6> addrregindex;
  bits<6> dataregindex;
  bits<2> autoincspec;
  bits<7> offsetindex;
  bits<1> regsizesel;
  bits<1> sizesel;
  bits<1> memhiersel;
  let RawWord{5-0} = addrregindex;
  let RawWord{11-6} = dataregindex;
  let RawWord{13-12} = autoincspec;
  let RawWord{20-14} = offsetindex;
  let RawWord{21} = regsizesel;
  let RawWord{22} = sizesel;
  let RawWord{23} = memhiersel;
}

def PseudoTTSTOREREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$regaddr, GPR:$tdmadataregindex)>;
def PseudoTTSTOREREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$regaddr, uimm32:$tdmadataregindex)>;
def TTSTOREREG : TensixOrdinaryInst<"STOREREG", 0x67,
    (ins uimm32:$regaddr, uimm32:$tdmadataregindex),
    "$regaddr, $tdmadataregindex",
    ["regaddr", "tdmadataregindex"],
    [262143, 63]> {
  bits<18> regaddr;
  bits<6> tdmadataregindex;
  let RawWord{17-0} = regaddr;
  let RawWord{23-18} = tdmadataregindex;
}

def PseudoTTSTREAMWAITPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$wait_stream_sel, GPR:$target_sel, GPR:$target_value, GPR:$stall_res)>;
def PseudoTTSTREAMWAITMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$wait_stream_sel, uimm32:$target_sel, uimm32:$target_value, uimm32:$stall_res)>;
def TTSTREAMWAIT : TensixOrdinaryInst<"STREAMWAIT", 0xa7,
    (ins uimm32:$wait_stream_sel, uimm32:$target_sel, uimm32:$target_value, uimm32:$stall_res),
    "$wait_stream_sel, $target_sel, $target_value, $stall_res",
    ["wait_stream_sel", "target_sel", "target_value", "stall_res"],
    [7, 1, 2047, 511]> {
  bits<3> wait_stream_sel;
  bits<1> target_sel;
  bits<11> target_value;
  bits<9> stall_res;
  let RawWord{2-0} = wait_stream_sel;
  let RawWord{3} = target_sel;
  let RawWord{14-4} = target_value;
  let RawWord{23-15} = stall_res;
}

def PseudoTTSTREAMWRCFGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgreg, GPR:$streamregaddr, GPR:$stream_id_sel)>;
def PseudoTTSTREAMWRCFGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgreg, uimm32:$streamregaddr, uimm32:$stream_id_sel)>;
def TTSTREAMWRCFG : TensixOrdinaryInst<"STREAMWRCFG", 0xb7,
    (ins uimm32:$cfgreg, uimm32:$streamregaddr, uimm32:$stream_id_sel),
    "$cfgreg, $streamregaddr, $stream_id_sel",
    ["cfgreg", "streamregaddr", "stream_id_sel"],
    [2047, 1023, 7]> {
  bits<11> cfgreg;
  bits<10> streamregaddr;
  bits<3> stream_id_sel;
  let RawWord{10-0} = cfgreg;
  let RawWord{20-11} = streamregaddr;
  let RawWord{23-21} = stream_id_sel;
}

def PseudoTTSUBDMAREGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$oparegindex, GPR:$opbregindex, GPR:$resultregindex, GPR:$opbisconst)>;
def PseudoTTSUBDMAREGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opbisconst)>;
def TTSUBDMAREG : TensixOrdinaryInst<"SUBDMAREG", 0x59,
    (ins uimm32:$oparegindex, uimm32:$opbregindex, uimm32:$resultregindex, uimm32:$opbisconst),
    "$oparegindex, $opbregindex, $resultregindex, $opbisconst",
    ["oparegindex", "opbregindex", "resultregindex", "opbisconst"],
    [63, 63, 2047, 1]> {
  bits<6> oparegindex;
  bits<6> opbregindex;
  bits<11> resultregindex;
  bits<1> opbisconst;
  let RawWord{5-0} = oparegindex;
  let RawWord{11-6} = opbregindex;
  let RawWord{22-12} = resultregindex;
  let RawWord{23} = opbisconst;
}

def PseudoTTTBUFCMDPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTTBUFCMDMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTTBUFCMD : TensixOrdinaryInst<"TBUFCMD", 0x4b,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTTRNSPSRCAPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTTRNSPSRCAMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTTRNSPSRCA : TensixOrdinaryInst<"TRNSPSRCA", 0x14,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTTRNSPSRCBPort : TensixOrdinaryPort<
    (ins uimm8:$port)>;
def PseudoTTTRNSPSRCBMop : TensixOrdinaryMop<
    (ins uimm8:$slot)>;
def TTTRNSPSRCB : TensixOrdinaryInst<"TRNSPSRCB", 0x16,
    (ins ),
    "",
    [],
    []> {
}

def PseudoTTUNPACRPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$last, GPR:$searchcacheflush, GPR:$rowsearch, GPR:$autoinccontextid, GPR:$zerowrite2, GPR:$srcb_bcast, GPR:$setdatvalid, GPR:$ovrdthreadid, GPR:$addrcntcontextid, GPR:$cfgcontextid, GPR:$cfgcontextcntinc, GPR:$addrmode, GPR:$unpack_block_selection)>;
def PseudoTTUNPACRMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$last, uimm32:$searchcacheflush, uimm32:$rowsearch, uimm32:$autoinccontextid, uimm32:$zerowrite2, uimm32:$srcb_bcast, uimm32:$setdatvalid, uimm32:$ovrdthreadid, uimm32:$addrcntcontextid, uimm32:$cfgcontextid, uimm32:$cfgcontextcntinc, uimm32:$addrmode, uimm32:$unpack_block_selection)>;
def TTUNPACR : TensixOrdinaryInst<"UNPACR", 0x42,
    (ins uimm32:$last, uimm32:$searchcacheflush, uimm32:$rowsearch, uimm32:$autoinccontextid, uimm32:$zerowrite2, uimm32:$srcb_bcast, uimm32:$setdatvalid, uimm32:$ovrdthreadid, uimm32:$addrcntcontextid, uimm32:$cfgcontextid, uimm32:$cfgcontextcntinc, uimm32:$addrmode, uimm32:$unpack_block_selection),
    "$last, $searchcacheflush, $rowsearch, $autoinccontextid, $zerowrite2, $srcb_bcast, $setdatvalid, $ovrdthreadid, $addrcntcontextid, $cfgcontextid, $cfgcontextcntinc, $addrmode, $unpack_block_selection",
    ["last", "searchcacheflush", "rowsearch", "autoinccontextid", "zerowrite2", "srcb_bcast", "setdatvalid", "ovrdthreadid", "addrcntcontextid", "cfgcontextid", "cfgcontextcntinc", "addrmode", "unpack_block_selection"],
    [1, 1, 1, 1, 1, 1, 1, 1, 3, 7, 3, 255, 1]> {
  bits<1> last;
  bits<1> searchcacheflush;
  bits<1> rowsearch;
  bits<1> autoinccontextid;
  bits<1> zerowrite2;
  bits<1> srcb_bcast;
  bits<1> setdatvalid;
  bits<1> ovrdthreadid;
  bits<2> addrcntcontextid;
  bits<3> cfgcontextid;
  bits<2> cfgcontextcntinc;
  bits<8> addrmode;
  bits<1> unpack_block_selection;
  let RawWord{0} = last;
  let RawWord{1} = searchcacheflush;
  let RawWord{2} = rowsearch;
  let RawWord{3} = autoinccontextid;
  let RawWord{4} = zerowrite2;
  let RawWord{5} = srcb_bcast;
  let RawWord{6} = setdatvalid;
  let RawWord{7} = ovrdthreadid;
  let RawWord{9-8} = addrcntcontextid;
  let RawWord{12-10} = cfgcontextid;
  let RawWord{14-13} = cfgcontextcntinc;
  let RawWord{22-15} = addrmode;
  let RawWord{23} = unpack_block_selection;
}

def PseudoTTUNPACR_NOPPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$unpack_pop, GPR:$src_clrval_ctrl, GPR:$bank_clr_ctrl, GPR:$stall_clr_cntrl, GPR:$clr_to1_fmt_ctrl, GPR:$set_dvalid, GPR:$msg_clr_cnt, GPR:$stream_id, GPR:$unpacker_select)>;
def PseudoTTUNPACR_NOPMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$unpack_pop, uimm32:$src_clrval_ctrl, uimm32:$bank_clr_ctrl, uimm32:$stall_clr_cntrl, uimm32:$clr_to1_fmt_ctrl, uimm32:$set_dvalid, uimm32:$msg_clr_cnt, uimm32:$stream_id, uimm32:$unpacker_select)>;
def TTUNPACR_NOP : TensixOrdinaryInst<"UNPACR_NOP", 0x43,
    (ins uimm32:$unpack_pop, uimm32:$src_clrval_ctrl, uimm32:$bank_clr_ctrl, uimm32:$stall_clr_cntrl, uimm32:$clr_to1_fmt_ctrl, uimm32:$set_dvalid, uimm32:$msg_clr_cnt, uimm32:$stream_id, uimm32:$unpacker_select),
    "$unpack_pop, $src_clrval_ctrl, $bank_clr_ctrl, $stall_clr_cntrl, $clr_to1_fmt_ctrl, $set_dvalid, $msg_clr_cnt, $stream_id, $unpacker_select",
    ["unpack_pop", "src_clrval_ctrl", "bank_clr_ctrl", "stall_clr_cntrl", "clr_to1_fmt_ctrl", "set_dvalid", "msg_clr_cnt", "stream_id", "unpacker_select"],
    [3, 3, 1, 1, 3, 15, 15, 127, 1]> {
  bits<2> unpack_pop;
  bits<2> src_clrval_ctrl;
  bits<1> bank_clr_ctrl;
  bits<1> stall_clr_cntrl;
  bits<2> clr_to1_fmt_ctrl;
  bits<4> set_dvalid;
  bits<4> msg_clr_cnt;
  bits<7> stream_id;
  bits<1> unpacker_select;
  let RawWord{1-0} = unpack_pop;
  let RawWord{3-2} = src_clrval_ctrl;
  let RawWord{4} = bank_clr_ctrl;
  let RawWord{5} = stall_clr_cntrl;
  let RawWord{7-6} = clr_to1_fmt_ctrl;
  let RawWord{11-8} = set_dvalid;
  let RawWord{15-12} = msg_clr_cnt;
  let RawWord{22-16} = stream_id;
  let RawWord{23} = unpacker_select;
}

def PseudoTTWRCFGPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$cfgreg, GPR:$wr128b, GPR:$gpraddress)>;
def PseudoTTWRCFGMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$cfgreg, uimm32:$wr128b, uimm32:$gpraddress)>;
def TTWRCFG : TensixOrdinaryInst<"WRCFG", 0xb0,
    (ins uimm32:$cfgreg, uimm32:$wr128b, uimm32:$gpraddress),
    "$cfgreg, $wr128b, $gpraddress",
    ["cfgreg", "wr128b", "gpraddress"],
    [32767, 1, 255]> {
  bits<15> cfgreg;
  bits<1> wr128b;
  bits<8> gpraddress;
  let RawWord{14-0} = cfgreg;
  let RawWord{15} = wr128b;
  let RawWord{23-16} = gpraddress;
}

def PseudoTTXMOVPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$last, GPR:$mov_block_selection)>;
def PseudoTTXMOVMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$last, uimm32:$mov_block_selection)>;
def TTXMOV : TensixOrdinaryInst<"XMOV", 0x40,
    (ins uimm32:$last, uimm32:$mov_block_selection),
    "$last, $mov_block_selection",
    ["last", "mov_block_selection"],
    [8388607, 1]> {
  bits<23> last;
  bits<1> mov_block_selection;
  let RawWord{22-0} = last;
  let RawWord{23} = mov_block_selection;
}

def PseudoTTZEROACCPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$where, GPR:$addr_mode, GPR:$clear_zero_flags, GPR:$use_32_bit_mode, GPR:$clear_mode)>;
def PseudoTTZEROACCMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$where, uimm32:$addr_mode, uimm32:$clear_zero_flags, uimm32:$use_32_bit_mode, uimm32:$clear_mode)>;
def TTZEROACC : TensixOrdinaryInst<"ZEROACC", 0x10,
    (ins uimm32:$where, uimm32:$addr_mode, uimm32:$clear_zero_flags, uimm32:$use_32_bit_mode, uimm32:$clear_mode),
    "$where, $addr_mode, $clear_zero_flags, $use_32_bit_mode, $clear_mode",
    ["where", "addr_mode", "clear_zero_flags", "use_32_bit_mode", "clear_mode"],
    [16383, 7, 1, 1, 31]> {
  bits<14> where;
  bits<3> addr_mode;
  bits<1> clear_zero_flags;
  bits<1> use_32_bit_mode;
  bits<5> clear_mode;
  let RawWord{13-0} = where;
  let RawWord{16-14} = addr_mode;
  let RawWord{17} = clear_zero_flags;
  let RawWord{18} = use_32_bit_mode;
  let RawWord{23-19} = clear_mode;
}

def PseudoTTZEROSRCPort : TensixOrdinaryPort<
    (ins uimm8:$port, GPR:$src_mask, GPR:$bank_mask, GPR:$write_mode, GPR:$zero_val)>;
def PseudoTTZEROSRCMop : TensixOrdinaryMop<
    (ins uimm8:$slot, uimm32:$src_mask, uimm32:$bank_mask, uimm32:$write_mode, uimm32:$zero_val)>;
def TTZEROSRC : TensixOrdinaryInst<"ZEROSRC", 0x11,
    (ins uimm32:$src_mask, uimm32:$bank_mask, uimm32:$write_mode, uimm32:$zero_val),
    "$src_mask, $bank_mask, $write_mode, $zero_val",
    ["src_mask", "bank_mask", "write_mode", "zero_val"],
    [3, 1, 1, 1048575]> {
  bits<2> src_mask;
  bits<1> bank_mask;
  bits<1> write_mode;
  bits<20> zero_val;
  let RawWord{1-0} = src_mask;
  let RawWord{2} = bank_mask;
  let RawWord{3} = write_mode;
  let RawWord{23-4} = zero_val;
}

// Public logical metadata and private machine mapping share these records.
def RISCVTensixInstructionTable : GenericTable {
  let FilterClass = "TensixOrdinaryInst";
  let CppTypeName = "TensixInstructionInfo";
  let Fields = ["Name", "IntrinsicID", "PortIntrinsicID", "MopIntrinsicID", "NumFields"];
  let PrimaryKey = ["IntrinsicID"];
  let PrimaryKeyName = "lookupTensixInstruction";
}
def lookupTensixInstructionByName : SearchIndex {
  let Table = RISCVTensixInstructionTable;
  let Key = ["Name"];
}
def lookupTensixInstructionByPort : SearchIndex {
  let Table = RISCVTensixInstructionTable;
  let Key = ["PortIntrinsicID"];
}
def lookupTensixInstructionByMop : SearchIndex {
  let Table = RISCVTensixInstructionTable;
  let Key = ["MopIntrinsicID"];
}
def RISCVTensixMachineTable : GenericTable {
  let FilterClass = "TensixOrdinaryInst";
  let CppTypeName = "TensixMachineInfo";
  let Fields = ["Opcode", "PortOpcode", "MopOpcode", "IntrinsicID", "WritesDst"];
  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getTensixMachineInfo";
}
def getTensixMachineInfoByPort : SearchIndex {
  let Table = RISCVTensixMachineTable;
  let Key = ["PortOpcode"];
}
def getTensixMachineInfoByMop : SearchIndex {
  let Table = RISCVTensixMachineTable;
  let Key = ["MopOpcode"];
}
def getTensixMachineInfoByIntrinsic : SearchIndex {
  let Table = RISCVTensixMachineTable;
  let Key = ["IntrinsicID"];
}

class TensixLogicalField<Intrinsic ID, int I, string N, int Maximum> {
  Intrinsic IntrinsicID = ID;
  bits<8> Index = I;
  string Name = N;
  bits<32> MaxValue = Maximum;
}
foreach Op = !instances<TensixOrdinaryInst>() in
  foreach I = !range(!size(Op.FieldNames)) in
    def : TensixLogicalField<Op.IntrinsicID, I, Op.FieldNames[I],
                             Op.FieldMaxima[I]>;
def RISCVTensixFieldTable : GenericTable {
  let FilterClass = "TensixLogicalField";
  let CppTypeName = "TensixInstructionField";
  let Fields = ["IntrinsicID", "Index", "Name", "MaxValue"];
  let PrimaryKey = ["IntrinsicID", "Index"];
  let PrimaryKeyName = "lookupTensixInstructionField";
}
