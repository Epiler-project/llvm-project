//===-- RISCVInstrInfoTensixOrdinary.td --------------------*- tablegen -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Post-RA automatic SFPU compression only. Physical value effects are derived
// from the retained record and verified again before emission. This is not an
// LLVM IR ingress for explicit SFPU recording.
def PseudoTTSFPUReplay : Pseudo<(outs),
    (ins uimm32:$load_mode, uimm32:$execute_while_loading,
         uimm32:$len, uimm32:$start_idx), []> {
  let Size = 4;
  let Predicates = [HasVendorXTTTensix];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
}
// Only the contents template finalizer produces this allocated receiver.
// Keep it distinct from raw/bound recordings, whose ingress remains unchanged.
def PseudoTTContentsSFPUReplay : Pseudo<(outs),
    (ins uimm32:$load_mode, uimm32:$execute_while_loading,
         uimm32:$len, uimm32:$start_idx), []> {
  let Size = 4;
  let Predicates = [HasVendorXTTTensix];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
}
def PseudoTTContentsReplayTemplateBegin : TensixReplayTemplateControl<
    (ins variable_ops), 4> {
  let Predicates = [HasVendorXTTTensix];
}
def PseudoTTContentsReplayTemplateEnd : TensixReplayTemplateControl<
    (ins variable_ops), 0> {
  let Predicates = [HasVendorXTTTensix];
  let mayLoad = 0;
  let mayStore = 0;
  let Uses = [TT_ISSUE];
}
// The result arity is bounded by the eight allocatable SFPR roles. Use normal
// static defs: variadicOpsAreDefs is not an instruction-emitter contract for
// targets that allocate physical registers. Captures alone remain variadic.
class TensixContentsReplayExecute<int Count> : Pseudo<
    !dag(outs, !listsplat(SFPR, Count),
         !foreach(I, !range(Count), "dst" # I)),
    !con((ins uimm32:$identity, uimm32:$mutable_count),
         !dag(ins, !listsplat(SFPR, Count),
              !foreach(I, !range(Count), "old" # I)),
         (ins variable_ops)), []> {
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<8> MutableCount = Count;
  let Size = 4;
  let Predicates = [HasVendorXTTTensix];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CC, TT_CONFIG, TT_ISSUE];
  let Defs = [TT_ISSUE];
  // A body can write one role before consuming another. Every independent
  // input must survive all early definitions, while each old value is tied.
  let Constraints = !interleave(!foreach(I, !range(Count),
      "$dst" # I # " = $old" # I # ",@earlyclobber $dst" # I), ",");
}
foreach Count = 1-8 in
  def PseudoTTContentsReplayTemplateExecute # Count
      : TensixContentsReplayExecute<Count>;

def RISCVTensixContentsReplayExecuteTable : GenericTable {
  let FilterClass = "TensixContentsReplayExecute";
  let CppTypeName = "TensixContentsReplayExecuteInfo";
  let Fields = ["Opcode", "MutableCount"];
  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getTensixContentsReplayExecuteInfo";
}
def getTensixContentsReplayExecuteInfoByCount : SearchIndex {
  let Table = RISCVTensixContentsReplayExecuteTable;
  let Key = ["MutableCount"];
}
// These instructions record an operation over begin's allocation roles.
// Role indices survive to final physical encoding; they are not register IDs.
// They do not execute that operation and carry no SFPR/CC/Dst effects.
class TensixContentsReplayRecipe<bit Binary> : Pseudo<(outs),
    !if(Binary, (ins uimm32:$identity, uimm32:$mutable_role,
                    uimm32:$source_role),
                (ins uimm32:$identity, uimm32:$mutable_role)), []> {
  let Size = 4;
  let Predicates = [HasVendorXTTTensix];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_ISSUE];
  let Defs = [TT_ISSUE];
}
def PseudoTTContentsReplayTemplateRecordIAdd : TensixContentsReplayRecipe<1>;
def PseudoTTContentsReplayTemplateRecordAnd : TensixContentsReplayRecipe<1>;
def PseudoTTContentsReplayTemplateRecordOr : TensixContentsReplayRecipe<1>;
def PseudoTTContentsReplayTemplateRecordXor : TensixContentsReplayRecipe<1>;
def PseudoTTContentsReplayTemplateRecordNot : TensixContentsReplayRecipe<0>;
// MOP executes the same flat mutable/capture roles as ordinary replay. Its
// additional native configuration and Dst effects remain explicit until the
// final MOP receiver replaces this symbolic execution with actual effects.
class TensixContentsMOPExecute<int Count> : Pseudo<
    !dag(outs, !listsplat(SFPR, Count),
         !foreach(I, !range(Count), "dst" # I)),
    !con((ins uimm32:$identity, uimm32:$mutable_count),
         !dag(ins, !listsplat(SFPR, Count),
              !foreach(I, !range(Count), "old" # I)),
         (ins variable_ops)), []> {
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<8> MutableCount = Count;
  let Size = 4;
  let Predicates = [HasVendorXTTTensix];
  let hasSideEffects = 1;
  let isNotDuplicable = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasNoSchedulingInfo = 1;
  let Uses = [TT_CC, TT_CONFIG, TT_DST, TT_ISSUE];
  let Defs = [TT_CONFIG, TT_DST, TT_ISSUE];
  let Constraints = !interleave(!foreach(I, !range(Count),
      "$dst" # I # " = $old" # I # ",@earlyclobber $dst" # I), ",");
}
foreach Count = 1-8 in
  def PseudoTTContentsReplayTemplateMopExecute # Count
      : TensixContentsMOPExecute<Count>;

def RISCVTensixContentsMOPExecuteTable : GenericTable {
  let FilterClass = "TensixContentsMOPExecute";
  let CppTypeName = "TensixContentsMOPExecuteInfo";
  let Fields = ["Opcode", "MutableCount"];
  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getTensixContentsMOPExecuteInfo";
}
def getTensixContentsMOPExecuteInfoByCount : SearchIndex {
  let Table = RISCVTensixContentsMOPExecuteTable;
  let Key = ["MutableCount"];
}
