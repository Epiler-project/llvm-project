//===-- RISCVTensixIRVerification.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCVTensixIRVerification.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "RISCVTensixBoundLowering.h"
#include "RISCVTensixExplicitReplay.h"
#include "RISCVTensixMacro.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
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
#include "llvm/IR/IntrinsicsTT.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/RISCV/RISCVTensix.h"
#include "llvm/TargetParser/Triple.h"
#include <array>
#include <functional>
#include <initializer_list>
#include <optional>

using namespace llvm;

static std::optional<RISCV::TensixSFPUArchitecture>
getBoundArchitecture(Intrinsic::ID ID) {
  if (auto Macro = getTensixBoundMacroArchitecture(ID))
    return Macro;
  return RISCV::getTensixBoundSFPUArchitecture(ID);
}

bool llvm::isTensixSFPUIntrinsic(Intrinsic::ID ID) {
  return isTensixBoundSFPUIntrinsic(ID);
}

bool llvm::isRetiredTensixSFPUIntrinsicName(StringRef Name) {
  // Historical vector-carrier, contents, and implicit LReg/CReg forms share
  // the retired legacy namespace; classify them before intrinsic selection so
  // textual IR cannot silently become an external call.
  if (!Name.consume_front("llvm.riscv.") || !Name.consume_front("tt."))
    return false;
  return Name.starts_with("bound.") || Name.starts_with("sfp") ||
         Name.starts_with("lreg.") || Name.starts_with("creg.") ||
         Name.starts_with("contents.");
}

Intrinsic::ID llvm::getTensixReplayControl(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::tt_wh_mop_execute:
    return Intrinsic::tt_mop;
  case Intrinsic::tt_wh_replay_template_begin:
  case Intrinsic::tt_bh_replay_template_begin:
    return Intrinsic::tt_replay_template_begin;
  case Intrinsic::tt_wh_replay_template_end:
  case Intrinsic::tt_bh_replay_template_end:
    return Intrinsic::tt_replay_template_end;
  case Intrinsic::tt_wh_replay_template_execute:
  case Intrinsic::tt_bh_replay_template_execute:
    return Intrinsic::tt_replay_template_execute;
  case Intrinsic::tt_wh_replay_template_mop:
  case Intrinsic::tt_bh_replay_template_mop:
    return Intrinsic::tt_replay_template_mop;
  default:
    return ID;
  }
}

std::optional<RISCV::TensixSFPUArchitecture>
llvm::getTensixReplayControlArchitecture(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::tt_wh_mop_execute:
  case Intrinsic::tt_wh_replay_template_begin:
  case Intrinsic::tt_wh_replay_template_end:
  case Intrinsic::tt_wh_replay_template_execute:
  case Intrinsic::tt_wh_replay_template_mop:
    return RISCV::TensixSFPUArchitecture::Wormhole;
  case Intrinsic::tt_bh_replay_template_begin:
  case Intrinsic::tt_bh_replay_template_end:
  case Intrinsic::tt_bh_replay_template_execute:
  case Intrinsic::tt_bh_replay_template_mop:
    return RISCV::TensixSFPUArchitecture::Blackhole;
  default:
    return std::nullopt;
  }
}

bool llvm::RISCV::isTensixIntrinsic(Intrinsic::ID ID) {
  ID = getTensixReplayControl(ID);
  return isTensixSFPUIntrinsic(ID) || getTensixInstructionByIntrinsic(ID) ||
         ID == Intrinsic::tt_dependent_use ||
         ID == Intrinsic::tt_coprocessor_wait ||
         ID == Intrinsic::tt_mop_clear ||
         ID == Intrinsic::tt_mop_control_write ||
         ID == Intrinsic::tt_replay_template_begin ||
         ID == Intrinsic::tt_replay_template_end ||
         ID == Intrinsic::tt_replay_template_execute ||
         ID == Intrinsic::tt_replay_template_mop ||
         ID == Intrinsic::tt_replay_record_end;
}

namespace {
bool isUnsupportedTensixDeclaration(const Function &F) {
  return isRetiredTensixSFPUIntrinsicName(F.getName()) ||
         (F.getName().starts_with("llvm.tt.") &&
          F.getIntrinsicID() != Intrinsic::tt_read_response &&
          F.getIntrinsicID() != Intrinsic::tt_issue_word &&
          !RISCV::isTensixIntrinsic(F.getIntrinsicID()));
}

bool isCarrier(Type *Ty) {
  auto *VT = dyn_cast<FixedVectorType>(Ty);
  return VT && VT->getNumElements() == 32 &&
         VT->getElementType()->isIntegerTy(32);
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

// read.response is available to RV32 roles without a Tensix issue extension.
// coprocessor.wait additionally requires a supported TRISC issue owner below.
Error verifyScalarTensixSynchronization(Function &F, bool IsRV32) {
  for (Instruction &I : instructions(F)) {
    auto *Call = dyn_cast<CallBase>(&I);
    const Function *Callee = Call ? Call->getCalledFunction() : nullptr;
    if (!Callee)
      continue;
    Intrinsic::ID ID = Callee->getIntrinsicID();
    bool Wait = ID == Intrinsic::tt_coprocessor_wait;
    if (!Wait && ID != Intrinsic::tt_read_response)
      continue;
    StringRef Name = Wait ? "coprocessor wait" : "scalar device read response";
    if (!isa<CallInst>(Call) || Call->hasOperandBundles() ||
        Call->getCallingConv() != CallingConv::C)
      return invalid(
          F, Name + " requires a direct C call without operand bundles");
    if (Callee->getMemoryEffects() != MemoryEffects::unknown() ||
        Call->getMemoryEffects() != MemoryEffects::unknown())
      return invalid(F, Name + " requires full memory effects");
    if (Wait && (!Callee->isConvergent() || !Call->isConvergent() ||
                 !Callee->cannotDuplicate() || !Call->cannotDuplicate() ||
                 Callee->hasFnAttribute(Attribute::NoSync) ||
                 Call->hasFnAttr(Attribute::NoSync)))
      return invalid(F, "coprocessor wait requires convergent nonduplicable "
                        "synchronization");
    if (!IsRV32)
      return invalid(F, Name + " requires RV32");
  }
  return Error::success();
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

static ConstantRange refineWithScalarExpressions(Value *Root,
                                                 ConstantRange Bounds,
                                                 Instruction &Use,
                                                 ScalarEvolution &SE,
                                                 LazyValueInfo &LVI,
                                                 AssumptionCache &AC,
                                                 DominatorTree &DT,
                                                 bool AllowSameSignGuards = true);

static unsigned computeModularTrailingZeros(const SCEV *S,
                                            ScalarEvolution &SE,
                                            unsigned Depth = 0) {
  unsigned Width = SE.getTypeSizeInBits(S->getType());
  if (Depth >= 8 || Width == 0)
    return 0;
  if (const auto *C = dyn_cast<SCEVConstant>(S))
    return C->getAPInt().countTrailingZeros();
  if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    unsigned MinTZ = Width;
    for (const SCEV *Op : Add->operands()) {
      MinTZ = std::min(MinTZ, computeModularTrailingZeros(Op, SE, Depth + 1));
      if (MinTZ == 0)
        return 0;
    }
    return MinTZ;
  }
  if (const auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
    unsigned SumTZ = 0;
    for (const SCEV *Op : Mul->operands()) {
      SumTZ = std::min(
          Width, SumTZ + computeModularTrailingZeros(Op, SE, Depth + 1));
      if (SumTZ >= Width)
        return Width;
    }
    return SumTZ;
  }
  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    unsigned MinTZ = Width;
    for (const SCEV *Op : AR->operands()) {
      MinTZ = std::min(MinTZ, computeModularTrailingZeros(Op, SE, Depth + 1));
      if (MinTZ == 0)
        return 0;
    }
    return MinTZ;
  }
  if (const auto *ZExt = dyn_cast<SCEVZeroExtendExpr>(S))
    return computeModularTrailingZeros(ZExt->getOperand(), SE, Depth + 1);
  if (const auto *SExt = dyn_cast<SCEVSignExtendExpr>(S))
    return computeModularTrailingZeros(SExt->getOperand(), SE, Depth + 1);
  if (const auto *Trunc = dyn_cast<SCEVTruncateExpr>(S))
    return std::min(
        Width, computeModularTrailingZeros(Trunc->getOperand(), SE, Depth + 1));
  return 0;
}

static bool preservesDefinedScalar(const Instruction &I, ScalarEvolution &SE,
                                   AssumptionCache &AC, DominatorTree &DT,
                                   LazyValueInfo &LVI) {
  if (!I.getType()->isIntegerTy() ||
      !(isa<PHINode>(I) || isa<BinaryOperator>(I) || isa<CastInst>(I) ||
        isa<ICmpInst>(I) || isa<SelectInst>(I)))
    return false;
  const auto *Op = cast<Operator>(&I);
  if (!canCreateUndefOrPoison(Op))
    return true;

  auto RefinedUnsigned = [&](const Value *V, bool AllowSameSign = true) {
    Value *MV = const_cast<Value *>(V);
    ConstantRange U = SE.getUnsignedRange(SE.getSCEV(MV));
    return refineWithScalarExpressions(MV, U, const_cast<Instruction &>(I), SE,
                                       LVI, AC, DT, AllowSameSign);
  };

  // Plain shifts preserve defined operands when every possible shift count
  // is below the scalar width. The dependency graph still proves both input
  // values independently; a bounded poison count is not a defined count.
  if (I.isShift()) {
    ConstantRange Counts = RefinedUnsigned(I.getOperand(1));
    if (Counts.isEmptySet() ||
        Counts.getUnsignedMax().uge(I.getType()->getIntegerBitWidth()))
      return false;
    // Exact right shifts preserve defined inputs only when every discarded
    // bit is zero. Prove that from the input without using poison-generating
    // flags as evidence; the dependency graph still checks both operands.
    if (const auto *Exact = dyn_cast<PossiblyExactOperator>(Op))
      if (Exact->isExact()) {
        const DataLayout &DL = I.getModule()->getDataLayout();
        KnownBits Input =
            computeKnownBits(I.getOperand(0), DL, nullptr, nullptr, nullptr,
                             /*UseInstrInfo=*/false);
        unsigned TZ = std::max(
            Input.countMinTrailingZeros(),
            computeModularTrailingZeros(SE.getSCEV(I.getOperand(0)), SE));
        return TZ >= Counts.getUnsignedMax().getZExtValue();
      }
    if (I.getOpcode() != Instruction::Shl ||
        (!I.hasNoUnsignedWrap() && !I.hasNoSignedWrap()))
      return true;
    // Flagged left shifts retain the overflow obligations below.
  }

  if (I.getOpcode() == Instruction::UDiv ||
      I.getOpcode() == Instruction::URem) {
    unsigned Width = I.getType()->getIntegerBitWidth();
    ConstantRange RU = RefinedUnsigned(I.getOperand(1))
                           .intersectWith(
                               SE.getSignedRange(SE.getSCEV(I.getOperand(1))),
                               ConstantRange::Unsigned);
    if (RU.isEmptySet() || RU.contains(APInt(Width, 0)))
      return false;
    if (I.getOpcode() == Instruction::UDiv) {
      if (const auto *Exact = dyn_cast<PossiblyExactOperator>(Op);
          Exact && Exact->isExact()) {
        const auto *DivC = dyn_cast<ConstantInt>(I.getOperand(1));
        if (!DivC || !DivC->getValue().isPowerOf2())
          return false;
        const DataLayout &DL = I.getModule()->getDataLayout();
        KnownBits Input =
            computeKnownBits(I.getOperand(0), DL, &AC, &I, &DT,
                             /*UseInstrInfo=*/false);
        unsigned TZ = std::max(
            Input.countMinTrailingZeros(),
            computeModularTrailingZeros(SE.getSCEV(I.getOperand(0)), SE));
        return TZ >= static_cast<unsigned>(DivC->getValue().exactLogBase2());
      }
    }
    return true;
  }

  if (const auto *Disjoint = dyn_cast<PossiblyDisjointInst>(&I);
      Disjoint && Disjoint->isDisjoint()) {
    // InstCombine can express bounded field addition as a disjoint OR.
    // The OR's own flag is not evidence. Combine plain bit arithmetic with
    // the operand ranges, including durable ABI load domains. Range facts
    // are conditional on defined inputs; isDefinedOffset independently
    // checks every dependency and its poison-generating flags below.
    const DataLayout &DL = I.getModule()->getDataLayout();
    DenseMap<const Value *, KnownBits> Cache;
    unsigned Remaining = 128;
    std::function<KnownBits(const Value *)> Bits = [&](const Value *V) {
      if (auto It = Cache.find(V); It != Cache.end())
        return It->second;
      unsigned Width = V->getType()->getIntegerBitWidth();
      if (!Remaining)
        return KnownBits(Width);
      --Remaining;
      KnownBits Result = computeKnownBits(V, DL, &AC, &I, &DT,
                                          /*UseInstrInfo=*/false);
      ConstantRange Bounds = RefinedUnsigned(V);
      if (!Bounds.isEmptySet())
        Result = Result.unionWith(Bounds.toKnownBits());
      Result.Zero.setLowBits(computeModularTrailingZeros(
          SE.getSCEV(const_cast<Value *>(V)), SE));
      // A range can span gaps between fields or the arms of a select. Keep
      // those zero bits when composing offsets. Model plain operations here;
      // the dependency walk checks each nested operation's flags separately.
      if (const auto *BO = dyn_cast<BinaryOperator>(V)) {
        if (BO->getOpcode() == Instruction::Or)
          Result = Result.unionWith(Bits(BO->getOperand(0)) |
                                    Bits(BO->getOperand(1)));
        else if (BO->getOpcode() == Instruction::Shl)
          Result = Result.unionWith(KnownBits::shl(Bits(BO->getOperand(0)),
                                                   Bits(BO->getOperand(1))));
      } else if (const auto *SI = dyn_cast<SelectInst>(V)) {
        Result = Result.unionWith(Bits(SI->getTrueValue())
                                      .intersectWith(Bits(SI->getFalseValue())));
      }
      if (Result.hasConflict())
        Result = KnownBits(Width);
      Cache.try_emplace(V, Result);
      return Result;
    };
    auto Left = Bits(I.getOperand(0)), Right = Bits(I.getOperand(1));
    if (KnownBits::haveNoCommonBitsSet(Left, Right))
      return true;
    // A header PHI includes its exiting value. Inside a guarded loop body a
    // dominating comparison can give a tighter bound than the PHI's range.
    for (unsigned N = 0; N != 2; ++N) {
      auto *Mask = dyn_cast<ConstantInt>(I.getOperand(N));
      if (!Mask || Mask->isZero())
        continue;
      APInt FirstSetBit = Mask->getValue() & -Mask->getValue();
      if (SE.isKnownPredicateAt(ICmpInst::ICMP_ULT,
                                SE.getSCEV(I.getOperand(1 - N)),
                                SE.getConstant(FirstSetBit), &I))
        return true;
    }
    return false;
  }

  // A flag is an obligation, never evidence. Check the actual operand ranges
  // with LLVM's overflow operations before admitting flagged arithmetic.
  if (const auto *BO = dyn_cast<OverflowingBinaryOperator>(Op)) {
    ConstantRange LU = RefinedUnsigned(I.getOperand(0));
    ConstantRange RU = RefinedUnsigned(I.getOperand(1));
    ConstantRange LS = SE.getSignedRange(SE.getSCEV(I.getOperand(0)));
    ConstantRange RS = SE.getSignedRange(SE.getSCEV(I.getOperand(1)));
    // These are two conservative ranges over the same operand bits. SCEV's
    // signed range can lose a remainder bound retained by its unsigned range
    // (and vice versa); use both facts before checking no-wrap obligations.
    LU = LU.intersectWith(LS, ConstantRange::Unsigned);
    RU = RU.intersectWith(RS, ConstantRange::Unsigned);
    LS = LS.intersectWith(LU, ConstantRange::Signed);
    RS = RS.intersectWith(RU, ConstantRange::Signed);
    if (LU.isEmptySet() || RU.isEmptySet() || LS.isEmptySet() ||
        RS.isEmptySet())
      return false;
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
  if (const auto *TI = dyn_cast<TruncInst>(&I)) {
    unsigned DstWidth = TI->getType()->getIntegerBitWidth();
    const SCEV *Src = SE.getSCEV(TI->getOperand(0));
    ConstantRange U = RefinedUnsigned(TI->getOperand(0));
    ConstantRange S = SE.getSignedRange(Src);
    U = U.intersectWith(S, ConstantRange::Unsigned);
    S = S.intersectWith(U, ConstantRange::Signed);
    if (U.isEmptySet() || S.isEmptySet())
      return false;
    bool UnsignedSafe = U.getUnsignedMax().isIntN(DstWidth);
    bool SignedSafe = S.getSignedMin().isSignedIntN(DstWidth) &&
                      S.getSignedMax().isSignedIntN(DstWidth);
    return (!TI->hasNoUnsignedWrap() || UnsignedSafe) &&
           (!TI->hasNoSignedWrap() || SignedSafe);
  }
  if (const auto *IC = dyn_cast<ICmpInst>(&I); IC && IC->hasSameSign()) {
    ConstantRange L = SE.getSignedRange(SE.getSCEV(IC->getOperand(0)))
                          .intersectWith(RefinedUnsigned(IC->getOperand(0),
                                                         /*AllowSameSign=*/false),
                                         ConstantRange::Signed);
    ConstantRange R = SE.getSignedRange(SE.getSCEV(IC->getOperand(1)))
                          .intersectWith(RefinedUnsigned(IC->getOperand(1),
                                                         /*AllowSameSign=*/false),
                                         ConstantRange::Signed);
    if (L.isEmptySet() || R.isEmptySet())
      return false;
    return (L.getSignedMin().isNonNegative() &&
            R.getSignedMin().isNonNegative()) ||
           (L.getSignedMax().isNegative() && R.getSignedMax().isNegative());
  }
  if (isa<PossiblyNonNegInst>(I) && I.hasNonNeg()) {
    ConstantRange S = SE.getSignedRange(SE.getSCEV(I.getOperand(0)))
                          .intersectWith(RefinedUnsigned(I.getOperand(0)),
                                         ConstantRange::Signed);
    return !S.isEmptySet() && !S.getSignedMin().isNegative();
  }
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
  LazyValueInfo LVI(const_cast<Function *>(Use.getFunction()), &AC);
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
    if (!I || Nodes.size() == NodeBudget ||
        !preservesDefinedScalar(*I, SE, AC, DT, LVI))
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
ConstantRange
refineWithRelatedInductions(const SCEV *Expr, ConstantRange Bounds,
                            IntrinsicInst &Use, ScalarEvolution &SE,
                            LazyValueInfo &LVI, AssumptionCache &AC,
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
    ConstantRange Related =
        Sum ? ConstantRange(Sum->getAPInt()).sub(OtherBounds)
            : OtherBounds.add(ConstantRange(Difference->getAPInt()));
    Bounds = Bounds.intersectWith(Related, ConstantRange::Unsigned);
  }
  return Bounds;
}

// SCEV can bound an induction and the operands of a bitwise expression while
// representing the expression itself as unknown. LVI alone need not recover
// the induction's bounds. Compose their facts at this issuing instruction,
// without expanding PHI backedges or relying on later IR optimization.
static ConstantRange refineWithScalarExpressions(Value *Root, ConstantRange Bounds,
                                          Instruction &Use,
                                          ScalarEvolution &SE,
                                          LazyValueInfo &LVI,
                                          AssumptionCache &AC,
                                          DominatorTree &DT,
                                          bool AllowSameSignGuards) {
  constexpr unsigned NodeBudget = 128;
  unsigned Remaining = NodeBudget;
  // This cache is local to one issuing use. Path constraints must not be
  // reused at another instruction or on a different incoming edge.
  DenseMap<Value *, ConstantRange> Ranges;
  std::function<ConstantRange(Value *)> Visit = [&](Value *V) -> ConstantRange {
    auto It = Ranges.find(V);
    if (It != Ranges.end())
      return It->second;
    ConstantRange Unknown =
        ConstantRange::getFull(V->getType()->getIntegerBitWidth());
    if (!Remaining)
      return Unknown;
    --Remaining;
    // A revisited active node is conservative even if the supported scalar
    // expression set is extended in the future. PHIs are leaves below.
    Ranges.try_emplace(V, Unknown);
    ConstantRange Result =
        SE.getUnsignedRange(SE.getSCEV(V))
            .intersectWith(
                LVI.getConstantRange(V, &Use, /*UndefAllowed=*/false),
                ConstantRange::Unsigned);
    if (isa<Instruction>(V) || isa<Argument>(V)) {
      for (const User *U : V->users()) {
        const auto *Cmp = dyn_cast<ICmpInst>(U);
        if (!Cmp)
          continue;
        if (Cmp->hasSameSign() &&
            (!AllowSameSignGuards ||
             !preservesDefinedScalar(*Cmp, SE, AC, DT, LVI)))
          continue;
        auto *RHS = dyn_cast<ConstantInt>(
            Cmp->getOperand(0) == V ? Cmp->getOperand(1) : Cmp->getOperand(0));
        if (!RHS)
          continue;
        ICmpInst::Predicate Pred = Cmp->getOperand(0) == V
                                       ? Cmp->getPredicate()
                                       : Cmp->getSwappedPredicate();
        for (const User *CU : Cmp->users()) {
          const auto *BI = dyn_cast<CondBrInst>(CU);
          if (!BI || BI->getCondition() != Cmp)
            continue;
          BasicBlock *BB = const_cast<BasicBlock *>(BI->getParent());
          if (DT.dominates(BasicBlockEdge(BB, BI->getSuccessor(0)),
                           Use.getParent())) {
            ConstantRange Guard = ConstantRange::makeAllowedICmpRegion(
                Pred, ConstantRange(RHS->getValue()));
            Result = Result.intersectWith(Guard, ConstantRange::Unsigned);
          } else if (DT.dominates(BasicBlockEdge(BB, BI->getSuccessor(1)),
                                  Use.getParent())) {
            ConstantRange Guard = ConstantRange::makeAllowedICmpRegion(
                ICmpInst::getInversePredicate(Pred),
                ConstantRange(RHS->getValue()));
            Result = Result.intersectWith(Guard, ConstantRange::Unsigned);
          }
        }
      }
    }
    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
      switch (BO->getOpcode()) {
      case Instruction::Add:
      case Instruction::Sub:
      case Instruction::Mul:
      case Instruction::Shl:
      case Instruction::LShr:
      case Instruction::AShr:
      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor: {
        ConstantRange Left = Visit(BO->getOperand(0));
        ConstantRange Right = Visit(BO->getOperand(1));
        // Use modular transfer, not the overload that assumes arithmetic
        // flags. Definedness, including shift validity and poison, remains
        // the independent obligation checked before this range refinement.
        Result = Result.intersectWith(Left.binaryOp(BO->getOpcode(), Right),
                                      ConstantRange::Unsigned);
        break;
      }
      default:
        break;
      }
    } else if (auto *SI = dyn_cast<SelectInst>(V)) {
      Result = Result.intersectWith(
          Visit(SI->getTrueValue())
              .unionWith(Visit(SI->getFalseValue()), ConstantRange::Unsigned),
          ConstantRange::Unsigned);
    } else if (auto *ZI = dyn_cast<ZExtInst>(V)) {
      if (ZI->getOperand(0)->getType()->isIntegerTy())
        Result = Result.intersectWith(
            Visit(ZI->getOperand(0))
                .zeroExtend(V->getType()->getIntegerBitWidth()),
            ConstantRange::Unsigned);
    } else if (auto *TI = dyn_cast<TruncInst>(V)) {
      if (TI->getOperand(0)->getType()->isIntegerTy())
        Result = Result.intersectWith(
            Visit(TI->getOperand(0))
                .truncate(V->getType()->getIntegerBitWidth()),
            ConstantRange::Unsigned);
    } else if (auto *BC = dyn_cast<BitCastInst>(V)) {
      if (BC->getOperand(0)->getType() == V->getType())
        Result = Result.intersectWith(Visit(BC->getOperand(0)),
                                      ConstantRange::Unsigned);
    } else if (auto *PN = dyn_cast<PHINode>(V);
               PN && PN->getNumIncomingValues() == 2 &&
               V->getType()->isIntegerTy()) {
      for (unsigned InitIdx : {0u, 1u}) {
        auto *InitC = dyn_cast<ConstantInt>(PN->getIncomingValue(InitIdx));
        auto *SI = dyn_cast<SelectInst>(PN->getIncomingValue(1u - InitIdx));
        if (!InitC || !SI)
          continue;
        auto *Cmp = dyn_cast<ICmpInst>(SI->getCondition());
        if (!Cmp || !isa<ConstantInt>(Cmp->getOperand(1)))
          continue;
        ICmpInst::Predicate Pred = Cmp->getPredicate();
        const Value *ResetVal = SI->getTrueValue();
        const Value *StepVal = SI->getFalseValue();
        if (Pred == ICmpInst::ICMP_NE || Pred == ICmpInst::ICMP_ULT) {
          Pred = ICmpInst::getInversePredicate(Pred);
          std::swap(ResetVal, StepVal);
        }
        auto *ResetC = dyn_cast<ConstantInt>(ResetVal);
        auto *IncBO = dyn_cast<BinaryOperator>(StepVal);
        if (!ResetC || !IncBO || IncBO->getOpcode() != Instruction::Add)
          continue;
        const Value *StepOp = IncBO->getOperand(0) == PN
                                  ? IncBO->getOperand(1)
                                  : (IncBO->getOperand(1) == PN
                                         ? IncBO->getOperand(0)
                                         : nullptr);
        auto *StepC = dyn_cast_or_null<ConstantInt>(StepOp);
        if (!StepC || StepC->isZero())
          continue;
        auto *BoundC = cast<ConstantInt>(Cmp->getOperand(1));
        if (Cmp->getOperand(0) == IncBO &&
            (Pred == ICmpInst::ICMP_UGE ||
             (Pred == ICmpInst::ICMP_EQ && StepC->isOne())) &&
            !BoundC->isZero() && InitC->getValue().ult(BoundC->getValue()) &&
            ResetC->getValue().ult(BoundC->getValue())) {
          bool Ov = false;
          (void)(BoundC->getValue() - 1).uadd_ov(StepC->getValue(), Ov);
          if (!Ov) {
            APInt Lower =
                APIntOps::umin(InitC->getValue(), ResetC->getValue());
            Result = Result.intersectWith(
                ConstantRange(Lower, BoundC->getValue()),
                ConstantRange::Unsigned);
            break;
          }
        }
        if (Cmp->getOperand(0) == PN && Pred == ICmpInst::ICMP_EQ &&
            StepC->isOne() && !BoundC->isMaxValue(false) &&
            InitC->getValue().ule(BoundC->getValue()) &&
            ResetC->getValue().ule(BoundC->getValue())) {
          APInt Lower = APIntOps::umin(InitC->getValue(), ResetC->getValue());
          Result = Result.intersectWith(
              ConstantRange(Lower, BoundC->getValue() + 1),
              ConstantRange::Unsigned);
          break;
        }
      }
    }
    Ranges.find(V)->second = Result;
    return Result;
  };
  return Bounds.intersectWith(Visit(Root), ConstantRange::Unsigned);
}

Error verifyReplayTemplates(Function &F, DominatorTree &DT) {
  DenseMap<uint32_t, const IntrinsicInst *> Producers, Ends;
  SmallVector<std::pair<uint32_t, const IntrinsicInst *>, 4> Executions;
  for (const BasicBlock &BB : F) {
    const IntrinsicInst *Active = nullptr;
    uint32_t ActiveID = 0;
    for (const Instruction &I : BB) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      auto Kind = II ? getTensixReplayControl(II->getIntrinsicID())
                     : Intrinsic::not_intrinsic;
      if (Kind == Intrinsic::tt_replay_template_begin ||
          Kind == Intrinsic::tt_replay_template_end ||
          Kind == Intrinsic::tt_replay_template_execute ||
          Kind == Intrinsic::tt_replay_template_mop) {
        auto Identity = immediate(*II, 0);
        if (!Identity)
          return Identity.takeError();
        uint32_t ID = *Identity;
        if (Kind == Intrinsic::tt_replay_template_begin) {
          if (Active || !Producers.try_emplace(ID, II).second)
            return invalid(F,
                           "replay template requires one non-nested producer");
          Active = II;
          ActiveID = ID;
        } else if (Kind == Intrinsic::tt_replay_template_end) {
          if (!Active || ActiveID != ID)
            return invalid(F,
                           "replay template end does not match its producer");
          Ends.try_emplace(ID, II);
          Active = nullptr;
        } else {
          if (Active)
            return invalid(
                F,
                "replay template cannot contain an execute or MOP reference");
          Executions.emplace_back(ID, II);
        }
        continue;
      }
      if (Active &&
          (I.isTerminator() || (isa<CallBase>(I) && !II) ||
           (II && !RISCV::isTensixIntrinsic(Kind) && I.mayHaveSideEffects()) ||
           (!II && I.mayReadOrWriteMemory())))
        return invalid(F, "replay template body cannot cross scalar control or "
                          "observable memory/call effects");
    }
    if (Active)
      return invalid(F, "replay template requires an end in its basic block");
  }
  for (auto [ID, Execute] : Executions) {
    auto End = Ends.find(ID);
    if (End == Ends.end() || !DT.dominates(End->second, Execute))
      return invalid(F, "replay template execute requires a dominating record");
  }
  return Error::success();
}

Error verifyIssue(Function &F, bool HasFeature, bool Wormhole,
                  ScalarEvolution &SE, AssumptionCache &AC, DominatorTree &DT) {
  if (Error E = verifyReplayTemplates(F, DT))
    return E;
  LazyValueInfo LVI(&F, &AC);
  for (Instruction &I : instructions(F)) {
    auto *Call = dyn_cast<CallBase>(&I);
    const Function *Callee = Call ? Call->getCalledFunction() : nullptr;
    if (Callee && isUnsupportedTensixDeclaration(*Callee))
      return invalid(F,
                     "unsupported Tensix intrinsic ABI: " + Callee->getName());
  }
  Attribute ExecutorAttr = F.getFnAttribute("tensix-executor");
  StringRef Executor = ExecutorAttr.isStringAttribute()
                           ? ExecutorAttr.getValueAsString()
                           : StringRef();
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !RISCV::isTensixIntrinsic(II->getIntrinsicID()))
      continue;
    if (!HasFeature) {
      if (II->getIntrinsicID() == Intrinsic::tt_coprocessor_wait)
        return invalid(
            F, "coprocessor wait requires +xtttensixbh or +xtttensixwh");
      return invalid(F,
                     "Tensix intrinsic requires +xtttensixbh or +xtttensixwh");
    }
    // The scalar readback dependency uses ordinary RISC-V instructions rather
    // than a model-specific Tensix instruction encoding.
    if (auto Architecture =
            getTensixReplayControlArchitecture(II->getIntrinsicID())) {
      if ((*Architecture == RISCV::TensixSFPUArchitecture::Wormhole) != Wormhole)
        return invalid(F, "Replay intrinsic architecture does not match target feature");
    } else if (auto Architecture =
                   getBoundArchitecture(II->getIntrinsicID())) {
      if ((*Architecture == RISCV::TensixSFPUArchitecture::Wormhole) !=
          Wormhole)
        return invalid(
            F, "SFPU intrinsic architecture does not match target feature");
    } else if (Wormhole &&
               II->getIntrinsicID() != Intrinsic::tt_dependent_use &&
               II->getIntrinsicID() != Intrinsic::tt_coprocessor_wait &&
               (II->getIntrinsicID() == Intrinsic::tt_replay ||
                II->getIntrinsicID() == Intrinsic::tt_mop ||
                (II->getIntrinsicID() != Intrinsic::tt_mop_control_write &&
                 II->getIntrinsicID() != Intrinsic::tt_mop_clear &&
                 !RISCV::isTensixWormholeIntrinsic(II->getIntrinsicID())))) {
      return invalid(F, "Tensix intrinsic is not supported on Wormhole");
    }
    if (Executor.empty())
      return invalid(
          F, "Tensix intrinsic requires a tensix-executor function attribute");
    if (Executor != "ncrisc" && Executor != "brisc" && Executor != "trisc0" &&
        Executor != "trisc1" && Executor != "trisc2")
      return invalid(F, "invalid tensix-executor value: " + Executor);
    if (II->hasOperandBundles() || II->getCallingConv() != CallingConv::C)
      return invalid(
          F,
          "Tensix intrinsic requires a direct C call without operand bundles");
    Intrinsic::ID ID = getTensixReplayControl(II->getIntrinsicID());
    if (ID == Intrinsic::tt_dependent_use)
      continue;
    if (ID == Intrinsic::tt_coprocessor_wait) {
      if (Executor != "trisc0" && Executor != "trisc1" && Executor != "trisc2")
        return invalid(F, "coprocessor wait requires a TRISC executor");
      continue;
    }
    if (Executor == "ncrisc")
      return invalid(F, "ncrisc cannot issue Tensix instructions");
    if (isTensixSFPUIntrinsic(ID)) {
      if (getTensixBoundMacroArchitecture(ID)) {
        if (Executor != "trisc1")
          return invalid(F, "SFPLOADMACRO requires the TRISC1 executor");
      } else if (Executor != "trisc0" && Executor != "trisc1" &&
                 Executor != "trisc2") {
        return invalid(F, "bound SFPU execution requires a TRISC executor");
      }
      continue;
    }
    if (ID == Intrinsic::tt_replay_record_end)
      continue;
    if (ID == Intrinsic::tt_replay_template_begin) {
      if (Error E = range(*II, 1, 0, 1))
        return E;
      if (Error E = range(*II, 2, 0, 1))
        return E;
      auto Placement = immediate(*II, 2);
      auto Start = immediate(*II, 3);
      if (!Placement)
        return Placement.takeError();
      if (!Start)
        return Start.takeError();
      if ((*Placement == 0 && *Start != 0) || *Start >= TensixReplaySlotCount)
        return invalid(F, "replay template requires canonical automatic "
                          "placement or a valid fixed start");
      continue;
    }
    if (ID == Intrinsic::tt_replay_template_end ||
        ID == Intrinsic::tt_replay_template_execute)
      continue;
    if (ID == Intrinsic::tt_replay_template_mop) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 1, 2, 8))
        return E;
      continue;
    }
    if (ID == Intrinsic::tt_mop_control_write) {
      if (Executor == "brisc")
        return invalid(F, "MOP control programming requires a TRISC executor");
      if (Error E = range(*II, 0, 0, 1))
        return E;
      if (!isDefinedOffset(II->getArgOperand(1), *II, SE, AC, DT))
        return invalid(F, "MOP control value must be defined and non-poison");
      continue;
    }
    if (ID == Intrinsic::tt_mop_clear) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 0, 2, 8))
        return E;
      continue;
    }
    const auto *Info = RISCV::getTensixInstructionByIntrinsic(ID);
    if (II->getIntrinsicID() == Intrinsic::tt_wh_mop_execute &&
        Executor == "brisc")
      return invalid(F, "MOP execution requires a TRISC executor");
    bool Mop = ID == Info->MopIntrinsicID;
    bool Replay = Info->IntrinsicID == Intrinsic::tt_replay;
    bool Port = ID == Info->PortIntrinsicID;
    unsigned First = Port || Mop ? 1 : 0;
    if (Wormhole) {
      SmallVector<uint32_t, 16> Fields;
      for (unsigned N = 0; N != Info->NumFields; ++N) {
        // Use the minimum of a dynamic field for static form validation.
        // The issuing-use range check below receives the actual value.
        if (!isa<ConstantInt>(II->getArgOperand(First + N))) {
          if (auto Minimum =
                  RISCV::getTensixWormholeDynamicFieldMinimum(ID, N)) {
            Fields.push_back(*Minimum);
            continue;
          }
        }
        auto Value = immediate(*II, First + N);
        if (!Value)
          return Value.takeError();
        Fields.push_back(static_cast<uint32_t>(*Value));
      }
      if (Error E = RISCV::verifyTensixWormholeInstructionFields(ID, Fields))
        return invalid(F, toString(std::move(E)));
    } else if (RISCV::isTensixBlackholeMutexIntrinsic(Info->IntrinsicID)) {
      // Encodable mutex indices outside this set cause an indefinite wait.
      auto Index = immediate(*II, First);
      if (!Index)
        return Index.takeError();
      if (Error E = RISCV::verifyTensixBlackholeMutexIndex(*Index))
        return invalid(F, toString(std::move(E)));
    }
    if (Mop) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 0, 2, 8))
        return E;
      if (Info->IntrinsicID == Intrinsic::tt_mop ||
          Info->IntrinsicID == Intrinsic::tt_mop_cfg)
        return invalid(
            F, "MOP control cannot be nested in a MOP instruction slot");
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
      uint64_t Minimum = 0;
      uint64_t Maximum = Field->MaxValue;
      if (Wormhole && Port) {
        const auto *Machine =
            RISCV::getTensixMachineInfoByIntrinsic(Info->IntrinsicID);
        const auto *WormholeField =
            RISCV::getTensixWormholeField(Machine->Opcode, N);
        Minimum = WormholeField->MinValue;
        Maximum = WormholeField->MaxValue;
        if (ID == Intrinsic::tt_zeroacc_port && N == 0) {
          const auto Mode =
              cast<ConstantInt>(II->getArgOperand(First + 4))->getZExtValue();
          const auto Use32 =
              cast<ConstantInt>(II->getArgOperand(First + 3))->getZExtValue();
          Maximum = Mode == 2 ? 1 : Maximum >> Use32;
        }
      }
      if (!Port || Replay) {
        if (Error E = range(*II, First + N, Minimum, Maximum))
          return E;
        continue;
      }
      const Value *V = II->getArgOperand(First + N);
      if (!isDefinedOffset(V, *II, SE, AC, DT))
        return invalid(F, Twine(Info->getName()) + " field " +
                              Field->getName() +
                              " must be defined and non-poison");
      const SCEV *FieldValue = SE.getSCEV(const_cast<Value *>(V));
      auto NeedsRefinement = [Minimum, Maximum](const ConstantRange &Bounds) {
        return !Bounds.isEmptySet() && (Bounds.getUnsignedMin().ult(Minimum) ||
                                        Bounds.getUnsignedMax().ugt(Maximum));
      };
      ConstantRange Bounds = SE.getUnsignedRange(FieldValue);
      if (NeedsRefinement(Bounds))
        Bounds = Bounds.intersectWith(
            LVI.getConstantRangeAtUse(II->getArgOperandUse(First + N),
                                      /*UndefAllowed=*/false),
            ConstantRange::Unsigned);
      if (NeedsRefinement(Bounds))
        Bounds = refineWithRelatedInductions(FieldValue, Bounds, *II, SE, LVI,
                                             AC, DT);
      if (NeedsRefinement(Bounds))
        Bounds = refineWithScalarExpressions(const_cast<Value *>(V), Bounds,
                                             *II, SE, LVI, AC, DT);
      if (Bounds.isEmptySet() ||
          (Bounds.getUnsignedMin().ult(Minimum) &&
           !SE.isKnownPredicateAt(ICmpInst::ICMP_UGE, FieldValue,
                                  SE.getConstant(V->getType(), Minimum), II)) ||
          (Bounds.getUnsignedMax().ugt(Maximum) &&
           !SE.isKnownPredicateAt(ICmpInst::ICMP_ULE, FieldValue,
                                  SE.getConstant(V->getType(), Maximum), II)))
        return invalid(F, Twine(Info->getName()) + " field " +
                              Field->getName() + " must be proven in [" +
                              Twine(Minimum) + ", " + Twine(Maximum) + "]");
    }
    if (Replay) {
      if (Error E = range(*II, First + 1, 0, 1))
        return E;
      if (Error E = range(*II, First + 2, 1, 32))
        return E;
      if (Error E = range(*II, First + 3, 0, 31))
        return E;
      auto Length =
          cast<ConstantInt>(II->getArgOperand(First + 2))->getZExtValue();
      auto Start =
          cast<ConstantInt>(II->getArgOperand(First + 3))->getZExtValue();
      if (Length > 32 - Start)
        return invalid(F, "replay range must fit 32 instruction slots");
      if (Mop && !cast<ConstantInt>(II->getArgOperand(First))->isZero())
        return invalid(F, "MOP replay slot must execute, not record");
    }
  }
  return Error::success();
}

Error verifyIntrinsic(const IntrinsicInst &II, ScalarEvolution &SE,
                      AssumptionCache &AC, DominatorTree &DT) {
  if (!isTensixBoundSFPUIntrinsic(II.getIntrinsicID()))
    return invalid(*II.getFunction(), "unsupported Tensix intrinsic ABI: " +
                                          II.getCalledFunction()->getName());
  if (Error E = verifyTensixBoundSFPUIntrinsic(II))
    return E;
  if (auto Index = getTensixBoundDstOffsetOperand(II.getIntrinsicID())) {
    Value *OffsetValue = II.getArgOperand(*Index);
    if (!isDefinedOffset(OffsetValue, II, SE, AC, DT))
      return invalid(*II.getFunction(),
                     "bound Dst offset must be defined and non-poison");
    const SCEV *Offset = SE.getSCEV(OffsetValue);
    ConstantRange Bounds = SE.getUnsignedRange(Offset);
    if (Bounds.isEmptySet() || Bounds.getUnsignedMax().ugt(1023)) {
      LazyValueInfo LVI(const_cast<Function *>(II.getFunction()), &AC);
      Bounds = refineWithScalarExpressions(OffsetValue, Bounds,
                                           const_cast<IntrinsicInst &>(II), SE,
                                           LVI, AC, DT);
    }
    if (Bounds.isEmptySet() || Bounds.getUnsignedMax().ugt(1023))
      return invalid(*II.getFunction(),
                     "bound Dst offset must be proven in [0, 1023]");
  }
  return Error::success();
}

} // namespace

Expected<TensixSFPUFunctionFacts>
llvm::verifyTensixSFPUFunction(Function &F, ScalarEvolution &SE,
                               AssumptionCache &AC, DominatorTree &DT,
                               RISCV::TensixSFPUArchitecture Architecture) {
  TensixSFPUFunctionFacts Facts;
  bool Bound = false;
  for (const Instruction &I : instructions(F)) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !isTensixSFPUIntrinsic(II->getIntrinsicID()))
      continue;
    if (auto IntrinsicArchitecture = getBoundArchitecture(II->getIntrinsicID()))
      if (*IntrinsicArchitecture != Architecture)
        return invalid(
            F, "SFPU intrinsic architecture does not match target feature");
    Bound = isTensixBoundSFPUIntrinsic(II->getIntrinsicID()) || Bound;
  }

  // The physical bound ABI cannot leak an old vector carrier through the
  // function boundary. Ordinary RVV functions are not sent through this
  // verifier unless they also contain a Tensix intrinsic.
  if (containsCarrier(F.getReturnType()))
    return invalid(F, "SFPU carrier cannot cross the return ABI");
  for (const Argument &Arg : F.args())
    if (containsCarrier(Arg.getType()))
      return invalid(F, "SFPU carrier cannot cross the argument ABI");

  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    bool SFPU = II && isTensixSFPUIntrinsic(II->getIntrinsicID());
    Facts.UsesSFPU |= SFPU;
    bool Carrier = containsCarrier(I.getType());
    for (Value *Input : I.operands()) {
      if (!containsCarrier(Input->getType()))
        continue;
      Carrier = true;
      if (isa<Constant>(Input))
        return invalid(
            F, "SFPU carrier constants require explicit target initialization");
    }
    if (Carrier)
      return invalid(
          F, "bound SFPU ingress cannot contain an unbound carrier or PHI");
    if (SFPU) {
      if (II->hasOperandBundles())
        return invalid(F, "SFPU intrinsics do not admit operand bundles");
      if (Error E = verifyIntrinsic(*II, SE, AC, DT))
        return std::move(E);
    }
  }

  if (!Facts.UsesSFPU)
    return Facts;
  for (Instruction &I : instructions(F)) {
    // Includes indirect calls, inline assembly, invokes and callbr. Scalar
    // calls do not prove preservation of the fixed SFPU state of this
    // function, so only direct Tensix intrinsics and inert LLVM intrinsics
    // are admitted around a bound SFPU region.
    if (!isa<CallBase>(I))
      continue;
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (II && RISCV::isTensixIntrinsic(II->getIntrinsicID()) &&
        !II->hasOperandBundles())
      continue;
    if (II && !II->hasOperandBundles()) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::smax:
      case Intrinsic::smin:
      case Intrinsic::umax:
      case Intrinsic::umin:
        if (II->getType()->isIntegerTy(32))
          continue;
        break;
      case Intrinsic::tt_setc16:
      case Intrinsic::tt_setc16_port:
      case Intrinsic::tt_dependent_use:
      case Intrinsic::tt_read_response:
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
  return Facts;
}

Error llvm::RISCV::verifyTensixModule(Module &M) {
  std::string Details;
  raw_string_ostream OS(Details);
  if (verifyModule(M, &OS))
    return createStringError("invalid LLVM module: " + OS.str());
  for (Function &Decl : M) {
    if (isUnsupportedTensixDeclaration(Decl))
      return createStringError("unsupported Tensix intrinsic ABI: " +
                               Decl.getName());
    if (!isTensixIntrinsic(Decl.getIntrinsicID()))
      continue;
    for (const User *U : Decl.users()) {
      const auto *CI = dyn_cast<CallInst>(U);
      if (!CI || CI->getCalledFunction() != &Decl || CI->hasOperandBundles() ||
          CI->getCallingConv() != CallingConv::C)
        return createStringError("Tensix intrinsic requires a direct C call "
                                 "without operand bundles");
    }
  }
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    if (Error E = verifyScalarTensixSynchronization(
            F, M.getTargetTriple().isRISCV32()))
      return E;
    bool Uses = llvm::any_of(instructions(F), [](const Instruction &I) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && isTensixIntrinsic(II->getIntrinsicID());
    });
    if (!Uses)
      continue;
    Attribute CPU = F.getFnAttribute("target-cpu");
    Attribute Features = F.getFnAttribute("target-features");
    auto FeaturesBits = getTensixTargetFeatureBits(
        M.getTargetTriple(),
        CPU.isStringAttribute() ? CPU.getValueAsString() : "",
        Features.isStringAttribute() ? Features.getValueAsString() : "");
    if (!FeaturesBits)
      return FeaturesBits.takeError();
    bool Wormhole = (*FeaturesBits)[RISCV::FeatureVendorXTTTensixWH];
    bool HasFeature =
        (*FeaturesBits)[RISCV::FeatureVendorXTTTensixBH] || Wormhole;
    AssumptionCache AC(F);
    DominatorTree DT(F);
    LoopInfo LI(DT);
    TargetLibraryInfoImpl TLII(M.getTargetTriple());
    TargetLibraryInfo TLI(TLII);
    ScalarEvolution SE(F, TLI, AC, DT, LI);
    if (Error E = verifyIssue(F, HasFeature, Wormhole, SE, AC, DT))
      return E;
    auto Facts =
        verifyTensixSFPUFunction(F, SE, AC, DT,
                                 Wormhole ? TensixSFPUArchitecture::Wormhole
                                          : TensixSFPUArchitecture::Blackhole);
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
    if (Error E = verifyScalarTensixSynchronization(F, !ST.is64Bit()))
      reportFatalUsageError(Twine(toString(std::move(E))));
    bool UsesSFPU = llvm::any_of(instructions(F), [](const Instruction &I) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && isTensixSFPUIntrinsic(II->getIntrinsicID());
    });
    bool UsesTensix = llvm::any_of(instructions(F), [](const Instruction &I) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      const auto *Call = dyn_cast<CallBase>(&I);
      const Function *Callee = Call ? Call->getCalledFunction() : nullptr;
      return (II && RISCV::isTensixIntrinsic(II->getIntrinsicID())) ||
             (Callee && isUnsupportedTensixDeclaration(*Callee));
    });
    if (!ST.hasVendorXTTTensixBH() && !ST.hasVendorXTTTensixWH() && !UsesTensix)
      return false;
    if (Error E = verifyIssue(
            F, ST.hasVendorXTTTensixBH() || ST.hasVendorXTTTensixWH(),
            ST.hasVendorXTTTensixWH(),
            getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
            getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
            getAnalysis<DominatorTreeWrapperPass>().getDomTree()))
      reportFatalUsageError(Twine(toString(std::move(E))));
    if (UsesSFPU && !SupportedPipeline)
      reportFatalUsageError(
          "Tensix SFPU requires the SelectionDAG legacy codegen pipeline");
    auto Facts = verifyTensixSFPUFunction(
        F, getAnalysis<ScalarEvolutionWrapperPass>().getSE(),
        getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F),
        getAnalysis<DominatorTreeWrapperPass>().getDomTree(),
        ST.hasVendorXTTTensixWH() ? RISCV::TensixSFPUArchitecture::Wormhole
                                  : RISCV::TensixSFPUArchitecture::Blackhole);
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
  if (Error E = verifyScalarTensixSynchronization(F, !ST.is64Bit()))
    reportFatalUsageError(Twine(toString(std::move(E))));
  bool UsesTensix = llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    const auto *Call = dyn_cast<CallBase>(&I);
    const Function *Callee = Call ? Call->getCalledFunction() : nullptr;
    return (II && RISCV::isTensixIntrinsic(II->getIntrinsicID())) ||
           (Callee && isUnsupportedTensixDeclaration(*Callee));
  });
  if (!ST.hasVendorXTTTensixBH() && !ST.hasVendorXTTTensixWH() && !UsesTensix)
    return PreservedAnalyses::all();
  if (Error E = verifyIssue(
          F, ST.hasVendorXTTTensixBH() || ST.hasVendorXTTTensixWH(),
          ST.hasVendorXTTTensixWH(), FAM.getResult<ScalarEvolutionAnalysis>(F),
          FAM.getResult<AssumptionAnalysis>(F),
          FAM.getResult<DominatorTreeAnalysis>(F)))
    reportFatalUsageError(Twine(toString(std::move(E))));
  auto Facts = verifyTensixSFPUFunction(
      F, FAM.getResult<ScalarEvolutionAnalysis>(F),
      FAM.getResult<AssumptionAnalysis>(F),
      FAM.getResult<DominatorTreeAnalysis>(F),
      ST.hasVendorXTTTensixWH() ? RISCV::TensixSFPUArchitecture::Wormhole
                                : RISCV::TensixSFPUArchitecture::Blackhole);
  if (!Facts)
    reportFatalUsageError(Twine(toString(Facts.takeError())));
  if (UsesSFPU)
    reportFatalUsageError(
        "Tensix SFPU requires the SelectionDAG legacy codegen pipeline");
  return PreservedAnalyses::all();
}
