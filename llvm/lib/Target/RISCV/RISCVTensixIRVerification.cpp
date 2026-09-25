//===-- RISCVTensixIRVerification.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCVTensixIRVerification.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "RISCVTensixBoundLowering.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <functional>
#include <initializer_list>
#include <optional>

using namespace llvm;

bool llvm::isTensixSFPUIntrinsic(Intrinsic::ID ID) {
  if (isTensixBoundSFPUIntrinsic(ID))
    return true;
  switch (ID) {
  case Intrinsic::riscv_tt_sfparecip:
  case Intrinsic::riscv_tt_sfpexexp:
  case Intrinsic::riscv_tt_sfpexman:
  case Intrinsic::riscv_tt_sfpabs:
  case Intrinsic::riscv_tt_sfplz:
  case Intrinsic::riscv_tt_sfpcast:
  case Intrinsic::riscv_tt_sfpsetexp_i:
  case Intrinsic::riscv_tt_sfpsetexp_v:
  case Intrinsic::riscv_tt_sfpsetman_i:
  case Intrinsic::riscv_tt_sfpsetman_v:
  case Intrinsic::riscv_tt_sfpsetsgn_i:
  case Intrinsic::riscv_tt_sfpsetsgn_v:
  case Intrinsic::riscv_tt_sfpand:
  case Intrinsic::riscv_tt_sfpor:
  case Intrinsic::riscv_tt_sfpxor:
  case Intrinsic::riscv_tt_sfpnot:
  case Intrinsic::riscv_tt_sfpiadd_i:
  case Intrinsic::riscv_tt_sfpshft_i:
  case Intrinsic::riscv_tt_sfpshft_v:
  case Intrinsic::riscv_tt_sfpshft2:
  case Intrinsic::riscv_tt_sfpstochrnd_i:
  case Intrinsic::riscv_tt_sfpstochrnd_v:
  case Intrinsic::riscv_tt_sfplut:
  case Intrinsic::riscv_tt_sfpswap:
  case Intrinsic::riscv_tt_sfptransp:
  case Intrinsic::riscv_tt_sfploadi:
  case Intrinsic::riscv_tt_sfpload:
  case Intrinsic::riscv_tt_sfpstore:
  case Intrinsic::riscv_tt_sfpadd:
  case Intrinsic::riscv_tt_sfpmad:
  case Intrinsic::riscv_tt_sfpmul:
  case Intrinsic::riscv_tt_sfpiadd:
  case Intrinsic::riscv_tt_sfpmov:
  case Intrinsic::riscv_tt_lreg_read:
  case Intrinsic::riscv_tt_lreg_write:
  case Intrinsic::riscv_tt_creg_read:
  case Intrinsic::riscv_tt_sfpencc:
  case Intrinsic::riscv_tt_sfpsetcc:
  case Intrinsic::riscv_tt_sfppushc:
  case Intrinsic::riscv_tt_sfppopc:
  case Intrinsic::riscv_tt_sfpnop:
  case Intrinsic::riscv_tt_sfpcompc:
  case Intrinsic::riscv_tt_sfpconfig_creg:
  case Intrinsic::riscv_tt_sfpconfig_reset:
    return true;
  default:
    return false;
  }
}

bool llvm::RISCV::isTensixIntrinsic(Intrinsic::ID ID) {
  return isTensixSFPUIntrinsic(ID) || getTensixInstructionByIntrinsic(ID) ||
         ID == Intrinsic::riscv_tt_dependent_use ||
         ID == Intrinsic::riscv_tt_mop_clear ||
         ID == Intrinsic::riscv_tt_replay_record_end;
}

namespace {
bool isCarrier(Type *Ty) {
  auto *VT = dyn_cast<FixedVectorType>(Ty);
  return VT && VT->getNumElements() == 32 && VT->getElementType()->isIntegerTy(32);
}

bool containsCarrier(Type *Ty) {
  if (isCarrier(Ty))
    return true;
  if (auto *ST = dyn_cast<StructType>(Ty))
    return llvm::any_of(ST->elements(), containsCarrier);
  if (auto *AT = dyn_cast<ArrayType>(Ty))
    return containsCarrier(AT->getElementType());
  return false;
}


Error invalid(const Function &F, const Twine &Detail) {
  return createStringError(Twine("Tensix function '") + F.getName() +
                           "': " + Detail);
}

Expected<uint64_t> immediate(const IntrinsicInst &II, unsigned Index) {
  auto *CI = dyn_cast<ConstantInt>(II.getArgOperand(Index));
  if (!CI || CI->getValue().getActiveBits() > 32)
    return invalid(*II.getFunction(), "expected an immediate operand");
  return CI->getZExtValue();
}

Error range(const IntrinsicInst &II, unsigned Index, uint64_t Lower,
            uint64_t Upper) {
  auto Value = immediate(II, Index);
  if (!Value)
    return Value.takeError();
  if (*Value < Lower || *Value > Upper)
    return invalid(*II.getFunction(), "immediate operand " + Twine(Index) +
                                         " must be in [" + Twine(Lower) +
                                         ", " + Twine(Upper) + "]");
  return Error::success();
}

Error signedRange(const IntrinsicInst &II, unsigned Index, int64_t Lower,
                  int64_t Upper) {
  const auto *CI = dyn_cast<ConstantInt>(II.getArgOperand(Index));
  if (!CI || CI->getSExtValue() < Lower || CI->getSExtValue() > Upper)
    return invalid(*II.getFunction(), "signed immediate operand " + Twine(Index) +
        " must be in [" + Twine(Lower) + ", " + Twine(Upper) + "]");
  return Error::success();
}

Error mode(const IntrinsicInst &II, unsigned Index,
           std::initializer_list<uint64_t> Modes) {
  auto Value = immediate(II, Index);
  if (!Value)
    return Value.takeError();
  if (!llvm::is_contained(Modes, *Value))
    return invalid(*II.getFunction(), "unsupported instruction mode " +
                                         Twine(*Value) + " for '" +
                                         II.getCalledFunction()->getName() +
                                         "'");
  return Error::success();
}

// Prove definedness separately from the address range. Range information alone
// cannot distinguish a ten-bit value from a ten-bit poison value.
static bool preservesDefinedScalar(const Instruction &I, ScalarEvolution &SE) {
  if (!I.getType()->isIntegerTy() ||
      !(isa<PHINode>(I) || isa<BinaryOperator>(I) || isa<CastInst>(I) ||
        isa<ICmpInst>(I) || isa<SelectInst>(I)))
    return false;
  const auto *Op = cast<Operator>(&I);
  if (!canCreateUndefOrPoison(Op))
    return true;

  // A flag is an obligation, never evidence. Check the actual operand ranges
  // with LLVM's overflow operations before admitting flagged arithmetic.
  if (const auto *BO = dyn_cast<OverflowingBinaryOperator>(Op)) {
    ConstantRange LU = SE.getUnsignedRange(SE.getSCEV(I.getOperand(0)));
    ConstantRange RU = SE.getUnsignedRange(SE.getSCEV(I.getOperand(1)));
    ConstantRange LS = SE.getSignedRange(SE.getSCEV(I.getOperand(0)));
    ConstantRange RS = SE.getSignedRange(SE.getSCEV(I.getOperand(1)));
    using OR = ConstantRange::OverflowResult;
    bool UnsignedSafe = false, SignedSafe = false;
    switch (I.getOpcode()) {
    case Instruction::Add:
      UnsignedSafe = LU.unsignedAddMayOverflow(RU) == OR::NeverOverflows;
      SignedSafe = LS.signedAddMayOverflow(RS) == OR::NeverOverflows;
      break;
    case Instruction::Sub:
      UnsignedSafe = LU.unsignedSubMayOverflow(RU) == OR::NeverOverflows;
      SignedSafe = LS.signedSubMayOverflow(RS) == OR::NeverOverflows;
      break;
    case Instruction::Mul:
    case Instruction::Shl: {
      unsigned Width = I.getType()->getIntegerBitWidth();
      APInt RMin = RS.getSignedMin(), RMax = RS.getSignedMax();
      if (I.getOpcode() == Instruction::Shl) {
        const auto *Shift = dyn_cast<ConstantInt>(I.getOperand(1));
        if (!Shift || Shift->getValue().uge(Width))
          return false;
        APInt Factor = APInt::getOneBitSet(Width, Shift->getZExtValue());
        RU = ConstantRange(Factor);
        // Widen the positive factor: shift by width-1 is not multiplication
        // by a negative number for the signed no-wrap obligation.
        RMin = RMax = Factor.zext(Width + 1);
      } else {
        RMin = RMin.sext(Width + 1);
        RMax = RMax.sext(Width + 1);
      }
      UnsignedSafe = LU.unsignedMulMayOverflow(RU) == OR::NeverOverflows;
      SignedSafe = true;
      for (const APInt &L : {LS.getSignedMin(), LS.getSignedMax()})
        for (const APInt &R : {RMin, RMax}) {
          bool Overflow = false;
          APInt Product = L.sext(Width + 1).smul_ov(R, Overflow);
          SignedSafe &= !Overflow && Product.isSignedIntN(Width);
        }
      break;
    }
    default:
      return false;
    }
    return (!BO->hasNoUnsignedWrap() || UnsignedSafe) &&
           (!BO->hasNoSignedWrap() || SignedSafe);
  }
  if (isa<PossiblyNonNegInst>(I) && I.hasNonNeg())
    return !SE.getSignedRange(SE.getSCEV(I.getOperand(0)))
                .getSignedMin().isNegative();
  return false;
}

static bool isDefinedOffset(const Value *Offset, const Instruction &Use,
                            ScalarEvolution &SE, AssumptionCache &AC,
                            DominatorTree &DT) {
  auto KnownDefined = [&](const Value *V) {
    return isGuaranteedNotToBeUndefOrPoison(V, &AC, &Use, &DT);
  };
  if (KnownDefined(Offset))
    return true;
  struct Node {
    const Instruction *Inst;
    SmallVector<unsigned, 2> Inputs;
    bool DefinedInput = false;
  };
  constexpr unsigned NodeBudget = 128;
  SmallVector<Node, 16> Nodes;
  DenseMap<const Value *, unsigned> Numbers;
  auto Insert = [&](const Value *V) -> std::optional<unsigned> {
    auto It = Numbers.find(V);
    if (It != Numbers.end())
      return It->second; // Deduplication only; SCC certification is below.
    auto *I = dyn_cast<Instruction>(V);
    if (!I || Nodes.size() == NodeBudget || !preservesDefinedScalar(*I, SE))
      return std::nullopt;
    unsigned Number = Nodes.size();
    Nodes.push_back({I, {}, false});
    Numbers[V] = Number;
    return Number;
  };
  if (!Insert(Offset))
    return false;
  for (unsigned N = 0; N != Nodes.size(); ++N) {
    const Instruction *I = Nodes[N].Inst;
    for (const Value *Input : I->operands()) {
      if (KnownDefined(Input)) {
        Nodes[N].DefinedInput = true;
        continue;
      }
      auto Number = Insert(Input);
      if (!Number)
        return false;
      Nodes[N].Inputs.push_back(*Number);
    }
  }

  // Tarjan emits dependency SCCs first. A cycle is admitted as an induction
  // only when its operations preserve defined inputs, its external inputs have
  // already been proved, and a PHI has a defined incoming seed outside it.
  SmallVector<int> Index(Nodes.size(), -1), Low(Nodes.size(), -1);
  SmallVector<unsigned> Stack, Component(Nodes.size(), ~0u);
  SmallVector<bool> OnStack(Nodes.size(), false), Proven;
  unsigned Next = 0;
  bool Valid = true;
  std::function<void(unsigned)> Visit = [&](unsigned N) {
    Index[N] = Low[N] = Next++;
    Stack.push_back(N);
    OnStack[N] = true;
    for (unsigned Input : Nodes[N].Inputs) {
      if (Index[Input] == -1) {
        Visit(Input);
        Low[N] = std::min(Low[N], Low[Input]);
      } else if (OnStack[Input]) {
        Low[N] = std::min(Low[N], Index[Input]);
      }
    }
    if (Low[N] != Index[N])
      return;
    SmallVector<unsigned> Members;
    unsigned C = Proven.size();
    do {
      unsigned Member = Stack.pop_back_val();
      OnStack[Member] = false;
      Component[Member] = C;
      Members.push_back(Member);
    } while (Members.back() != N);
    bool Cyclic = Members.size() > 1, Seed = false;
    for (unsigned Member : Members) {
      bool ExternalDefined = Nodes[Member].DefinedInput;
      for (unsigned Input : Nodes[Member].Inputs) {
        Cyclic |= Input == Member;
        if (Component[Input] == C)
          continue;
        if (Component[Input] >= Proven.size() || !Proven[Component[Input]])
          Valid = false;
        ExternalDefined = true;
      }
      Seed |= isa<PHINode>(Nodes[Member].Inst) && ExternalDefined;
    }
    if (Cyclic && !Seed)
      Valid = false;
    Proven.push_back(Valid);
  };
  Visit(0);
  return Valid;
}

// Strength reduction can turn C - IV into a second PHI. LVI then loses the
// expression's def-use relation to the guarded IV, while SCEV retains it.
// Recover only exact constant sum/difference identities in the same loop;
// ConstantRange performs their modular arithmetic without assuming no-wrap.
ConstantRange refineWithRelatedInductions(const SCEV *Expr,
                                          ConstantRange Bounds,
                                          IntrinsicInst &Use,
                                          ScalarEvolution &SE,
                                          LazyValueInfo &LVI,
                                          AssumptionCache &AC,
                                          DominatorTree &DT) {
  const auto *Recurrence = dyn_cast<SCEVAddRecExpr>(Expr);
  if (!Recurrence || !Recurrence->isAffine() ||
      !Recurrence->getLoop()->contains(Use.getParent()))
    return Bounds;
  for (PHINode &Phi : Recurrence->getLoop()->getHeader()->phis()) {
    if (Phi.getType() != Expr->getType() || !DT.dominates(&Phi, &Use) ||
        !isDefinedOffset(&Phi, Use, SE, AC, DT))
      continue;
    const SCEV *Other = SE.getSCEV(&Phi);
    const auto *OtherRecurrence = dyn_cast<SCEVAddRecExpr>(Other);
    if (Other == Expr || !OtherRecurrence || !OtherRecurrence->isAffine() ||
        OtherRecurrence->getLoop() != Recurrence->getLoop())
      continue;
    const auto *Sum = dyn_cast<SCEVConstant>(SE.getAddExpr(Expr, Other));
    const auto *Difference =
        dyn_cast<SCEVConstant>(SE.getMinusSCEV(Expr, Other));
    if (!Sum && !Difference)
      continue;
    ConstantRange OtherBounds = SE.getUnsignedRange(Other).intersectWith(
        LVI.getConstantRange(&Phi, &Use, /*UndefAllowed=*/false),
        ConstantRange::Unsigned);
    if (OtherBounds.isEmptySet())
      continue;
    ConstantRange Related = Sum ? ConstantRange(Sum->getAPInt()).sub(OtherBounds)
                                : OtherBounds.add(
                                      ConstantRange(Difference->getAPInt()));
    Bounds = Bounds.intersectWith(Related, ConstantRange::Unsigned);
  }
  return Bounds;
}

Error verifyIssue(Function &F, bool HasFeature, ScalarEvolution &SE,
                  AssumptionCache &AC, DominatorTree &DT) {
  LazyValueInfo LVI(&F, &AC);
  Attribute ExecutorAttr = F.getFnAttribute("tensix-executor");
  StringRef Executor = ExecutorAttr.isStringAttribute()
                           ? ExecutorAttr.getValueAsString() : StringRef();
  bool UsesSFPU = llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && isTensixSFPUIntrinsic(II->getIntrinsicID());
  });
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !RISCV::isTensixIntrinsic(II->getIntrinsicID()))
      continue;
    if (!HasFeature)
      return invalid(F, "Tensix intrinsic requires +xtttensixbh");
    if (Executor.empty())
      return invalid(F, "Tensix intrinsic requires a tensix-executor function attribute");
    if (Executor != "ncrisc" && Executor != "brisc" && Executor != "trisc0" &&
        Executor != "trisc1" && Executor != "trisc2")
      return invalid(F, "invalid tensix-executor value: " + Executor);
    if (II->hasOperandBundles() || II->getCallingConv() != CallingConv::C)
      return invalid(F, "Tensix intrinsic requires a direct C call without operand bundles");
    Intrinsic::ID ID = II->getIntrinsicID();
    if (ID == Intrinsic::riscv_tt_dependent_use)
      continue;
    if (Executor == "ncrisc")
      return invalid(F, "ncrisc cannot issue Tensix instructions");
    if (isTensixSFPUIntrinsic(ID)) {
      if (isTensixBoundSFPUIntrinsic(ID)) {
        if (Executor != "trisc0" && Executor != "trisc1" &&
            Executor != "trisc2")
          return invalid(F, "bound SFPU execution requires a TRISC executor");
      } else if (Executor != "trisc1") {
        return invalid(F, "Tensix SFPU execution requires tensix-executor=trisc1");
      }
      continue;
    }
    if (ID == Intrinsic::riscv_tt_replay_record_end)
      continue;
    if (ID == Intrinsic::riscv_tt_mop_clear) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 0, 2, 8))
        return E;
      continue;
    }
    const auto *Info = RISCV::getTensixInstructionByIntrinsic(ID);
    bool Mop = ID == Info->MopIntrinsicID;
    bool Replay = Info->IntrinsicID == Intrinsic::riscv_tt_replay;
    bool Port = ID == Info->PortIntrinsicID;
    unsigned First = Port || Mop ? 1 : 0;
    if (Mop) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 0, 2, 8))
        return E;
      if (Info->IntrinsicID == Intrinsic::riscv_tt_mop ||
          Info->IntrinsicID == Intrinsic::riscv_tt_mop_cfg)
        return invalid(F, "MOP control cannot be nested in a MOP instruction slot");
    }
    if (Port) {
      auto PortID = immediate(*II, 0);
      if (!PortID)
        return PortID.takeError();
      if (*PortID > 2)
        return invalid(F, "unknown Tensix instruction port");
      if (*PortID != 0 && Executor != "brisc")
        return invalid(F, "remote Tensix instruction ports require brisc");
    }
    for (unsigned N = 0; N != Info->NumFields; ++N) {
      const auto *Field = RISCV::getTensixInstructionField(*Info, N);
      if (!Port || Replay) {
        if (Error E = range(*II, First + N, 0, Field->MaxValue))
          return E;
        continue;
      }
      const Value *V = II->getArgOperand(First + N);
      if (!isDefinedOffset(V, *II, SE, AC, DT))
        return invalid(F, Twine(Info->getName()) + " field " + Field->getName() +
                          " must be defined and non-poison");
      const SCEV *FieldValue = SE.getSCEV(const_cast<Value *>(V));
      // A loop-header value also exists on the exiting edge. Admit a dynamic
      // field only when its unsigned bound holds at the actual issuing use,
      // including a dominating body guard; retain the separate poison proof.
      // Combine the recurrence's global bounds with edge constraints at this
      // use. Neither alone need prove derived fields such as 1 - induction.
      ConstantRange Bounds = SE.getUnsignedRange(FieldValue);
      if (!Bounds.isEmptySet() &&
          Bounds.getUnsignedMax().ugt(Field->MaxValue))
        Bounds = Bounds.intersectWith(
            LVI.getConstantRangeAtUse(II->getArgOperandUse(First + N),
                                      /*UndefAllowed=*/false),
            ConstantRange::Unsigned);
      if (!Bounds.isEmptySet() &&
          Bounds.getUnsignedMax().ugt(Field->MaxValue))
        Bounds = refineWithRelatedInductions(FieldValue, Bounds, *II, SE, LVI,
                                              AC, DT);
      if (Bounds.isEmptySet() ||
          (Bounds.getUnsignedMax().ugt(Field->MaxValue) &&
           !SE.isKnownPredicateAt(
               ICmpInst::ICMP_ULE, FieldValue,
               SE.getConstant(V->getType(), Field->MaxValue), II)))
        return invalid(F, Twine(Info->getName()) + " field " + Field->getName() +
                          " must be proven in [0, " +
                          Twine(Field->MaxValue) + "]");
    }
    if (Replay) {
      if (Error E = range(*II, First + 1, 0, 1))
        return E;
      if (Error E = range(*II, First + 2, 1, 32))
        return E;
      if (Error E = range(*II, First + 3, 0, 31))
        return E;
      auto Length = cast<ConstantInt>(II->getArgOperand(First + 2))->getZExtValue();
      auto Start = cast<ConstantInt>(II->getArgOperand(First + 3))->getZExtValue();
      if (Length > 32 - Start)
        return invalid(F, "replay range must fit 32 instruction slots");
      if (Mop && !cast<ConstantInt>(II->getArgOperand(First))->isZero())
        return invalid(F, "MOP replay slot must execute, not record");
    }
    // StateID changes alter the meaning of subsequent SFPU instructions. Until
    // a typed state propagation contract exists, admit only explicit state 0.
    if (UsesSFPU && Info->IntrinsicID == Intrinsic::riscv_tt_setc16) {
      const auto *Config = dyn_cast<ConstantInt>(II->getArgOperand(First + 1));
      const auto *Value = dyn_cast<ConstantInt>(II->getArgOperand(First));
      if (!Config || (Config->isZero() && (!Value || !Value->isZero())))
        return invalid(F, "Tensix SFPU StateID configuration must be static zero");
    }
  }
  return Error::success();
}

Error verifyIntrinsic(const IntrinsicInst &II, ScalarEvolution &SE,
                      AssumptionCache &AC, DominatorTree &DT,
                      TensixSFPUFunctionFacts &Facts) {
  if (isTensixBoundSFPUIntrinsic(II.getIntrinsicID())) {
    if (Error E = verifyTensixBoundSFPUIntrinsic(II))
      return E;
    if (auto Index = getTensixBoundDstOffsetOperand(II.getIntrinsicID())) {
      Value *OffsetValue = II.getArgOperand(*Index);
      if (!isDefinedOffset(OffsetValue, II, SE, AC, DT))
        return invalid(*II.getFunction(),
                       "bound Dst offset must be defined and non-poison");
      const SCEV *Offset = SE.getSCEV(OffsetValue);
      if (SE.getUnsignedRange(Offset).getUnsignedMax().ugt(1023))
        return invalid(*II.getFunction(),
                       "bound Dst offset must be proven in [0, 1023]");
    }
    return Error::success();
  }
  switch (II.getIntrinsicID()) {
  case Intrinsic::riscv_tt_sfparecip:
    return mode(II, 2, {0, 1, 2});
  case Intrinsic::riscv_tt_sfpexexp:
    return mode(II, 2, {0, 1, 2, 3, 8, 9, 10, 11});
  case Intrinsic::riscv_tt_sfpexman:
  case Intrinsic::riscv_tt_sfpabs:
    return mode(II, 2, {0, 1});
  case Intrinsic::riscv_tt_sfplz:
    return mode(II, 2, {0, 2, 4, 6, 8, 10, 12, 14});
  case Intrinsic::riscv_tt_sfpcast:
    return range(II, 2, 0, 3);
  case Intrinsic::riscv_tt_sfpsetexp_i:
    if (Error E = range(II, 2, 0, 255))
      return E;
    return mode(II, 3, {1, 3});
  case Intrinsic::riscv_tt_sfpsetman_i:
    if (Error E = range(II, 2, 0, 4095))
      return E;
    return mode(II, 3, {1});
  case Intrinsic::riscv_tt_sfpsetsgn_i:
    if (Error E = range(II, 2, 0, 1))
      return E;
    return mode(II, 3, {1});
  case Intrinsic::riscv_tt_sfpsetexp_v:
    return mode(II, 3, {0, 2});
  case Intrinsic::riscv_tt_sfpsetman_v:
  case Intrinsic::riscv_tt_sfpsetsgn_v:
    return mode(II, 3, {0});
  case Intrinsic::riscv_tt_sfpand:
  case Intrinsic::riscv_tt_sfpor:
  case Intrinsic::riscv_tt_sfpxor:
  case Intrinsic::riscv_tt_sfpnot:
  case Intrinsic::riscv_tt_sfptransp:
    return Error::success();
  case Intrinsic::riscv_tt_sfpiadd_i:
    if (Error E = signedRange(II, 1, -2048, 2047))
      return E;
    return mode(II, 2, {1, 5, 9});
  case Intrinsic::riscv_tt_sfpshft_i:
    if (Error E = signedRange(II, 1, -2048, 2047))
      return E;
    return mode(II, 2, {1, 3, 5, 7});
  case Intrinsic::riscv_tt_sfpshft_v:
    return mode(II, 2, {0, 2});
  case Intrinsic::riscv_tt_sfpshft2:
    if (Error E = range(II, 2, 0, 0))
      return E;
    return mode(II, 3, {3, 4});
  case Intrinsic::riscv_tt_sfpstochrnd_i: {
    if (Error E = range(II, 3, 0, 7))
      return E;
    if (Error E = range(II, 4, 0, 2))
      return E;
    auto Mod = cast<ConstantInt>(II.getArgOperand(3))->getZExtValue();
    return range(II, 2, 0, Mod == 4 || Mod == 5 ? 31 : 0);
  }
  case Intrinsic::riscv_tt_sfpstochrnd_v:
    if (Error E = mode(II, 3, {4, 5}))
      return E;
    return range(II, 4, 0, 2);
  case Intrinsic::riscv_tt_sfplut:
    return mode(II, 5, {0, 4});
  case Intrinsic::riscv_tt_sfpswap:
    return range(II, 2, 0, 9);
  case Intrinsic::riscv_tt_sfploadi:
    if (Error E = range(II, 1, 0, 65535))
      return E;
    return mode(II, 2, {0, 1, 2, 4, 8, 10});
  case Intrinsic::riscv_tt_sfpload:
  case Intrinsic::riscv_tt_sfpstore: {
    if (Error E = range(II, 2, 0, 7))
      return E;
    // SRCB format 0 selects the configured SFPU destination format. Raw
    // format 4 preserves payload bits required by Dst copies. Neither may be
    // collapsed to FP32 format 3: their conversion semantics differ.
    if (Error E = mode(II, 3, {0, 2, 3, 4}))
      return E;
    if (!isDefinedOffset(II.getArgOperand(1), II, SE, AC, DT))
      return invalid(*II.getFunction(), "Dst offset must be defined and non-poison");
    const SCEV *Offset = SE.getSCEV(II.getArgOperand(1));
    if (SE.getUnsignedRange(Offset).getUnsignedMax().ugt(1023))
      return invalid(*II.getFunction(),
                     "Dst offset must be proven in [0, 1023]");
    return Error::success();
  }
  case Intrinsic::riscv_tt_sfpadd:
    return range(II, 3, 0, 3);
  case Intrinsic::riscv_tt_sfpmad:
    return range(II, 4, 0, 3);
  case Intrinsic::riscv_tt_sfpmul:
    return range(II, 3, 0, 3);
  case Intrinsic::riscv_tt_sfpiadd:
    return mode(II, 2, {0, 2, 4, 6, 8, 10});
  case Intrinsic::riscv_tt_sfpmov:
    return mode(II, 2, {0, 1, 2});
  case Intrinsic::riscv_tt_lreg_read:
  case Intrinsic::riscv_tt_lreg_write: {
    if (Error E = range(II, 0, 0, 7))
      return E;
    Facts.ExplicitFixedLRegs |=
        uint8_t(1u << cast<ConstantInt>(II.getArgOperand(0))->getZExtValue());
    return Error::success();
  }
  case Intrinsic::riscv_tt_creg_read:
    return range(II, 0, 8, 15);
  case Intrinsic::riscv_tt_sfpencc:
    if (Error E = range(II, 0, 0, 3))
      return E;
    return mode(II, 1, {0, 1, 2, 8, 9, 10});
  case Intrinsic::riscv_tt_sfpsetcc:
    if (Error E = range(II, 1, 0, 0))
      return E;
    return mode(II, 2, {0, 2, 4, 6});
  case Intrinsic::riscv_tt_sfppushc:
  case Intrinsic::riscv_tt_sfppopc:
    if (Error E = range(II, 0, 0, 0))
      return E;
    return range(II, 1, 0, 0);
  case Intrinsic::riscv_tt_sfpnop:
    return Error::success();
  case Intrinsic::riscv_tt_sfpcompc:
  case Intrinsic::riscv_tt_sfpconfig_reset:
    return Error::success();
  case Intrinsic::riscv_tt_sfpconfig_creg: {
    auto Dest = immediate(II, 1);
    if (!Dest)
      return Dest.takeError();
    if (*Dest < 11 || *Dest > 14)
      return invalid(*II.getFunction(),
                       "SFPCONFIG CReg destination must be in [11, 14]");
    if (Error E = range(II, 2, 0, 65535))
      return E;
    auto Mask = immediate(II, 2);
    if (!Mask)
      return Mask.takeError();
    auto Mod = immediate(II, 3);
    if (!Mod)
      return Mod.takeError();
    if (*Mod == 0) {
      if (*Mask != 0)
        return invalid(*II.getFunction(),
                       "SFPCONFIG mode 0 requires a zero mask");
    } else if (*Mod == 8) {
      if ((*Mask & ~uint64_t(0x5555)) != 0)
        return invalid(*II.getFunction(),
                       "SFPCONFIG mode 8 mask must select even lane bits");
    } else {
      return invalid(*II.getFunction(),
                     "unsupported SFPCONFIG CReg mode");
    }
    return Error::success();
  }
  default:
    llvm_unreachable("expected admitted SFPU intrinsic");
  }
}

Error verifyCCStack(Function &F) {
  if (F.empty())
    return Error::success();
  DenseMap<const BasicBlock *, unsigned> EntryDepth;
  SmallVector<const BasicBlock *> Worklist;
  EntryDepth.try_emplace(&F.getEntryBlock(), 0);
  Worklist.push_back(&F.getEntryBlock());
  while (!Worklist.empty()) {
    const BasicBlock *BB = Worklist.pop_back_val();
    unsigned Depth = EntryDepth.lookup(BB);
    for (const Instruction &I : *BB) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      if (II->getIntrinsicID() == Intrinsic::riscv_tt_sfppushc ||
          II->getIntrinsicID() == Intrinsic::riscv_tt_bound_sfppushc) {
        if (Depth == 8)
          return invalid(F, "CC stack depth exceeds 8");
        ++Depth;
      } else if (II->getIntrinsicID() == Intrinsic::riscv_tt_sfppopc ||
                 II->getIntrinsicID() == Intrinsic::riscv_tt_bound_sfppopc) {
        if (Depth == 0)
          return invalid(F, "CC stack underflow");
        --Depth;
      }
    }
    if (isa<ReturnInst>(BB->getTerminator()) && Depth != 0)
      return invalid(F, "CC stack must be empty at return");
    for (const BasicBlock *Successor : successors(BB)) {
      auto [It, Inserted] = EntryDepth.try_emplace(Successor, Depth);
      if (Inserted)
        Worklist.push_back(Successor);
      else if (It->second != Depth)
        return invalid(F, "CC stack depth disagrees at CFG join or backedge");
    }
  }
  return Error::success();
}
} // namespace

Expected<TensixSFPUFunctionFacts>
llvm::verifyTensixSFPUFunction(Function &F, ScalarEvolution &SE,
                              AssumptionCache &AC, DominatorTree &DT) {
  TensixSFPUFunctionFacts Facts;
  bool Bound = false, Legacy = false;
  for (const Instruction &I : instructions(F)) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !isTensixSFPUIntrinsic(II->getIntrinsicID()))
      continue;
    if (isTensixBoundSFPUIntrinsic(II->getIntrinsicID()))
      Bound = true;
    else
      Legacy = true;
  }
  if (Bound && Legacy)
    return invalid(F, "bound and legacy SFPU ingress cannot share a function");
  if (containsCarrier(F.getReturnType()))
    return invalid(F, "SFPU carrier cannot cross the return ABI");
  for (const Argument &Arg : F.args())
    if (containsCarrier(Arg.getType()))
      return invalid(F, "SFPU carrier cannot cross the argument ABI");

  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    bool SFPU = II && isTensixSFPUIntrinsic(II->getIntrinsicID());
    Facts.UsesSFPU |= SFPU;
    bool AggregateResult = SFPU &&
        (II->getIntrinsicID() == Intrinsic::riscv_tt_sfpswap ||
         II->getIntrinsicID() == Intrinsic::riscv_tt_sfptransp);
    bool Extract = false;
    if (const auto *EV = dyn_cast<ExtractValueInst>(&I))
      if (const auto *Source = dyn_cast<IntrinsicInst>(EV->getAggregateOperand()))
        Extract = EV->getNumIndices() == 1 &&
            (Source->getIntrinsicID() == Intrinsic::riscv_tt_sfpswap ||
             Source->getIntrinsicID() == Intrinsic::riscv_tt_sfptransp);
    bool Carrier = containsCarrier(I.getType());
    for (Value *Input : I.operands()) {
      if (!containsCarrier(Input->getType()))
        continue;
      Carrier = true;
      // The initial ingress requires explicit initialized snapshots. It does
      // not invent a full-lane proof to justify poison/undef passthroughs.
      if (isa<Constant>(Input))
        return invalid(F, "SFPU carrier constants require explicit target initialization");
      if (!isCarrier(Input->getType()) && !Extract)
        return invalid(F, "SFPU carrier cannot be nested in an aggregate");
    }
    if (auto *AI = dyn_cast<AllocaInst>(&I))
      Carrier |= containsCarrier(AI->getAllocatedType());
    if (Bound && Carrier)
      return invalid(
          F, "bound SFPU ingress cannot contain an unbound carrier or PHI");
    if (Carrier && !SFPU && !isa<PHINode>(I) && !Extract)
      return invalid(F, "SFPU carrier is only legal in target intrinsics and PHI");
    if (containsCarrier(I.getType()) && !isCarrier(I.getType()) &&
        !AggregateResult)
      return invalid(F, "SFPU carrier cannot be nested in an aggregate");
    if (SFPU) {
      if (II->hasOperandBundles())
        return invalid(F, "SFPU intrinsics do not admit operand bundles");
      if (Error E = verifyIntrinsic(*II, SE, AC, DT, Facts))
        return std::move(E);
    }
  }

  if (!Facts.UsesSFPU)
    return Facts;
  for (Instruction &I : instructions(F)) {
    // Includes indirect calls, inline assembly, invokes and callbr. Scalar ABI
    // does not prove preservation of the fixed/CC/SFPU state of this function.
    if (!isa<CallBase>(I))
      continue;
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (II && RISCV::isTensixIntrinsic(II->getIntrinsicID()) &&
        !II->hasOperandBundles())
      continue;
    if (II && !II->hasOperandBundles()) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::riscv_tt_setc16:
      case Intrinsic::riscv_tt_setc16_port:
      case Intrinsic::riscv_tt_dependent_use:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::assume:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::dbg_label:
        continue;
      default:
        break;
      }
    }
    return invalid(F, "call has no verified SFPU preservation ABI");
  }
  if (Error E = verifyCCStack(F))
    return std::move(E);
  return Facts;
}

Error llvm::RISCV::verifyTensixModule(Module &M) {
  std::string Details;
  raw_string_ostream OS(Details);
  if (verifyModule(M, &OS))
    return createStringError("invalid LLVM module: " + OS.str());
  for (Function &Decl : M) {
    if (!isTensixIntrinsic(Decl.getIntrinsicID()))
      continue;
    for (const User *U : Decl.users()) {
      const auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &Decl || CI->hasOperandBundles() ||
          CI->getCallingConv() != CallingConv::C)
        return createStringError("Tensix intrinsic requires a direct C call without operand bundles");
    }
  }
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    bool Uses = llvm::any_of(instructions(F), [](const Instruction &I) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && isTensixIntrinsic(II->getIntrinsicID());
    });
    if (!Uses)
      continue;
    Attribute CPU = F.getFnAttribute("target-cpu");
    Attribute Features = F.getFnAttribute("target-features");
    auto HasFeature = verifyTensixTargetFeatures(
        M.getTargetTriple(), CPU.isStringAttribute() ? CPU.getValueAsString() : "",
        Features.isStringAttribute() ? Features.getValueAsString() : "");
    if (!HasFeature)
      return HasFeature.takeError();
    AssumptionCache AC(F);
    DominatorTree DT(F);
    LoopInfo LI(DT);
    TargetLibraryInfoImpl TLII(M.getTargetTriple());
    TargetLibraryInfo TLI(TLII);
    ScalarEvolution SE(F, TLI, AC, DT, LI);
    if (Error E = verifyIssue(F, *HasFeature, SE, AC, DT))
      return E;
    auto Facts = verifyTensixSFPUFunction(F, SE, AC, DT);
    if (!Facts)
      return Facts.takeError();
  }
  return Error::success();
}

namespace {
class RISCVTensixIRVerification : public FunctionPass {
  bool SupportedPipeline;
public:
  static char ID;
  explicit RISCVTensixIRVerification(bool Supported = true)
      : FunctionPass(ID), SupportedPipeline(Supported) {
    initializeRISCVTensixIRVerificationPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
  bool runOnFunction(Function &F) override {
    auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
    const auto &ST = TM.getSubtarget<RISCVSubtarget>(F);
    bool UsesSFPU = llvm::any_of(instructions(F), [](const Instruction &I) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && isTensixSFPUIntrinsic(II->getIntrinsicID());
    });
    bool UsesTensix = llvm::any_of(instructions(F), [](const Instruction &I) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && RISCV::isTensixIntrinsic(II->getIntrinsicID());
    });
    if (!ST.hasVendorXTTTensixBH() && !UsesTensix)
      return false;
    if (Error E = verifyIssue(
            F, ST.hasVendorXTTTensixBH(),
            getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
            getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
            getAnalysis<DominatorTreeWrapperPass>().getDomTree()))
      reportFatalUsageError(Twine(toString(std::move(E))));
    if (UsesSFPU && !SupportedPipeline)
      reportFatalUsageError("Tensix SFPU requires the SelectionDAG legacy codegen pipeline");
    auto Facts = verifyTensixSFPUFunction(
        F, getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
        getAnalysis<DominatorTreeWrapperPass>().getDomTree());
    if (!Facts)
      reportFatalUsageError(Twine(toString(Facts.takeError())));
    return false;
  }
};
} // namespace
char RISCVTensixIRVerification::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVTensixIRVerification, "riscv-tensix-ir-verify",
                      "Verify Tensix SFPU IR", false, true)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(RISCVTensixIRVerification, "riscv-tensix-ir-verify",
                    "Verify Tensix SFPU IR", false, true)
FunctionPass *llvm::createRISCVTensixIRVerificationPass(bool Supported) {
  return new RISCVTensixIRVerification(Supported);
}

PreservedAnalyses RISCVTensixNewPMGate::run(Function &F,
                                         FunctionAnalysisManager &FAM) {
  bool UsesSFPU = llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && isTensixSFPUIntrinsic(II->getIntrinsicID());
  });
  const auto &ST = TM->getSubtarget<RISCVSubtarget>(F);
  bool UsesTensix = llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && RISCV::isTensixIntrinsic(II->getIntrinsicID());
  });
  if (!ST.hasVendorXTTTensixBH() && !UsesTensix)
    return PreservedAnalyses::all();
  if (Error E = verifyIssue(F, ST.hasVendorXTTTensixBH(),
          FAM.getResult<ScalarEvolutionAnalysis>(F),
          FAM.getResult<AssumptionAnalysis>(F),
          FAM.getResult<DominatorTreeAnalysis>(F)))
    reportFatalUsageError(Twine(toString(std::move(E))));
  auto Facts = verifyTensixSFPUFunction(
      F, FAM.getResult<ScalarEvolutionAnalysis>(F),
      FAM.getResult<AssumptionAnalysis>(F),
      FAM.getResult<DominatorTreeAnalysis>(F));
  if (!Facts)
    reportFatalUsageError(Twine(toString(Facts.takeError())));
  if (UsesSFPU)
    reportFatalUsageError("Tensix SFPU requires the SelectionDAG legacy codegen pipeline");
  return PreservedAnalyses::all();
}
