//===-- RISCVTensixIRVerification.cpp -------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCVTensixIRVerification.h"
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVTargetMachine.h"
#include "RISCVTensixBoundLowering.h"
#include "RISCVTensixContentsGroup.h"
#include "RISCVTensixMacro.h"
#include "RISCVTensixLowering.h"
#include "RISCVTensixReplay.h"
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
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <array>
#include <functional>
#include <initializer_list>
#include <optional>

using namespace llvm;

bool llvm::isTensixSFPUIntrinsic(Intrinsic::ID ID) {
  return isTensixBoundSFPUIntrinsic(ID) || isTensixContentsIntrinsic(ID);
}

bool llvm::isTensixContentsIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_tt_contents_read_constant:
  case Intrinsic::riscv_tt_contents_read_creg:
  case Intrinsic::riscv_tt_contents_load:
  case Intrinsic::riscv_tt_contents_load_full:
  case Intrinsic::riscv_tt_contents_add:
  case Intrinsic::riscv_tt_contents_mul:
  case Intrinsic::riscv_tt_contents_store:
  case Intrinsic::riscv_tt_contents_load_immediate:
  case Intrinsic::riscv_tt_contents_divp2:
  case Intrinsic::riscv_tt_contents_add_immediate:
  case Intrinsic::riscv_tt_contents_mul_immediate:
  case Intrinsic::riscv_tt_contents_mul23:
  case Intrinsic::riscv_tt_contents_mad:
  case Intrinsic::riscv_tt_contents_iadd:
  case Intrinsic::riscv_tt_contents_mov:
  case Intrinsic::riscv_tt_contents_mov_all:
  case Intrinsic::riscv_tt_contents_prng_advance:
  case Intrinsic::riscv_tt_contents_arecip:
  case Intrinsic::riscv_tt_contents_exexp:
  case Intrinsic::riscv_tt_contents_exman:
  case Intrinsic::riscv_tt_contents_abs:
  case Intrinsic::riscv_tt_contents_lz:
  case Intrinsic::riscv_tt_contents_cast:
  case Intrinsic::riscv_tt_contents_setexp_immediate:
  case Intrinsic::riscv_tt_contents_setman_immediate:
  case Intrinsic::riscv_tt_contents_setsgn_immediate:
  case Intrinsic::riscv_tt_contents_setexp_vector:
  case Intrinsic::riscv_tt_contents_setman_vector:
  case Intrinsic::riscv_tt_contents_setsgn_vector:
  case Intrinsic::riscv_tt_contents_iadd_immediate:
  case Intrinsic::riscv_tt_contents_shft_immediate:
  case Intrinsic::riscv_tt_contents_shft_vector:
  case Intrinsic::riscv_tt_contents_and:
  case Intrinsic::riscv_tt_contents_or:
  case Intrinsic::riscv_tt_contents_xor:
  case Intrinsic::riscv_tt_contents_not:
  case Intrinsic::riscv_tt_contents_shft2:
  case Intrinsic::riscv_tt_contents_stochrnd_immediate:
  case Intrinsic::riscv_tt_contents_stochrnd_vector:
  case Intrinsic::riscv_tt_contents_encc:
  case Intrinsic::riscv_tt_contents_pushc:
  case Intrinsic::riscv_tt_contents_popc:
  case Intrinsic::riscv_tt_contents_setcc:
  case Intrinsic::riscv_tt_contents_gt:
  case Intrinsic::riscv_tt_contents_le:
  case Intrinsic::riscv_tt_contents_compc:
  case Intrinsic::riscv_tt_contents_nop:
  case Intrinsic::riscv_tt_contents_config_reset:
  case Intrinsic::riscv_tt_contents_replay_template_begin:
  case Intrinsic::riscv_tt_contents_replay_template_end:
  case Intrinsic::riscv_tt_contents_replay_template_mop:
  case Intrinsic::riscv_tt_contents_replay_template_mop_execute:
  case Intrinsic::riscv_tt_contents_replay_template_execute:
  case Intrinsic::riscv_tt_contents_replay_template_record_iadd:
  case Intrinsic::riscv_tt_contents_replay_template_record_and:
  case Intrinsic::riscv_tt_contents_replay_template_record_or:
  case Intrinsic::riscv_tt_contents_replay_template_record_xor:
  case Intrinsic::riscv_tt_contents_replay_template_record_not:
    return true;
  default:
    return isTensixContentsGroupIntrinsic(ID) || isTensixMacroIntrinsic(ID);
  }
}

bool llvm::isRetiredTensixSFPUIntrinsicName(StringRef Name) {
  constexpr StringLiteral TensixPrefix = "llvm.riscv.tt.";
  if (!Name.starts_with(TensixPrefix))
    return false;
  // Historical vector-carrier and implicit LReg/CReg forms share these
  // prefixes; classify them before intrinsic selection so textual IR cannot
  // silently become an external call or a legacy selector input.
  StringRef Suffix = Name.drop_front(TensixPrefix.size());
  if (Suffix.starts_with("bound."))
    return false;
  return Suffix.starts_with("sfp") || Suffix.starts_with("lreg.") ||
         Suffix.starts_with("creg.");
}

bool llvm::RISCV::isTensixIntrinsic(Intrinsic::ID ID) {
  return isTensixSFPUIntrinsic(ID) || getTensixInstructionByIntrinsic(ID) ||
         ID == Intrinsic::riscv_tt_dependent_use ||
         ID == Intrinsic::riscv_tt_coprocessor_wait ||
         ID == Intrinsic::riscv_tt_mop_clear ||
         ID == Intrinsic::riscv_tt_mop_control_write ||
         ID == Intrinsic::riscv_tt_replay_template_begin ||
         ID == Intrinsic::riscv_tt_replay_template_end ||
         ID == Intrinsic::riscv_tt_replay_template_execute ||
         ID == Intrinsic::riscv_tt_replay_template_mop ||
         ID == Intrinsic::riscv_tt_replay_record_end;
}

namespace {
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
    bool Wait = ID == Intrinsic::riscv_tt_coprocessor_wait;
    if (!Wait && ID != Intrinsic::riscv_tt_read_response)
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

static bool preservesDefinedScalar(const Instruction &I, ScalarEvolution &SE) {
  if (!I.getType()->isIntegerTy() ||
      !(isa<PHINode>(I) || isa<BinaryOperator>(I) || isa<CastInst>(I) ||
        isa<ICmpInst>(I) || isa<SelectInst>(I)))
    return false;
  const auto *Op = cast<Operator>(&I);
  if (!canCreateUndefOrPoison(Op))
    return true;

  // Plain shifts preserve defined operands when every possible shift count
  // is below the scalar width. The dependency graph still proves both input
  // values independently; a bounded poison count is not a defined count.
  if (I.isShift()) {
    ConstantRange Counts = SE.getUnsignedRange(SE.getSCEV(I.getOperand(1)));
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
        return Input.countMinTrailingZeros() >=
               Counts.getUnsignedMax().getZExtValue();
      }
    if (I.getOpcode() != Instruction::Shl ||
        (!I.hasNoUnsignedWrap() && !I.hasNoSignedWrap()))
      return true;
    // Flagged left shifts retain the overflow obligations below.
  }

  if (const auto *Disjoint = dyn_cast<PossiblyDisjointInst>(&I);
      Disjoint && Disjoint->isDisjoint()) {
    // InstCombine can express bounded field addition as a disjoint OR.
    // Prove the flag's obligation from its operands, never from the OR's
    // result or poison-generating flags elsewhere in their dependency graph.
    // isDefinedOffset still independently proves both operands are defined.
    const DataLayout &DL = I.getModule()->getDataLayout();
    auto Bits = [&](const Value *V) {
      return computeKnownBits(V, DL, nullptr, nullptr, nullptr,
                              /*UseInstrInfo=*/false);
    };
    return KnownBits::haveNoCommonBitsSet(Bits(I.getOperand(0)),
                                          Bits(I.getOperand(1)));
  }

  // A flag is an obligation, never evidence. Check the actual operand ranges
  // with LLVM's overflow operations before admitting flagged arithmetic.
  if (const auto *BO = dyn_cast<OverflowingBinaryOperator>(Op)) {
    ConstantRange LU = SE.getUnsignedRange(SE.getSCEV(I.getOperand(0)));
    ConstantRange RU = SE.getUnsignedRange(SE.getSCEV(I.getOperand(1)));
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
  if (isa<PossiblyNonNegInst>(I) && I.hasNonNeg())
    return !SE.getSignedRange(SE.getSCEV(I.getOperand(0)))
                .getSignedMin()
                .isNegative();
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
ConstantRange refineWithScalarExpressions(Value *Root, ConstantRange Bounds,
                                          IntrinsicInst &Use,
                                          ScalarEvolution &SE,
                                          LazyValueInfo &LVI) {
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
    }
    Ranges.find(V)->second = Result;
    return Result;
  };
  return Bounds.intersectWith(Visit(Root), ConstantRange::Unsigned);
}

// Receive transport of one recorded value without requiring its backedge
// producer to dominate the header PHI. All reachable nodes must be permitted
// transport and must reach the actual seed. In particular, discovering a grey
// node or one seeded input must not admit a separate unseeded cycle.
bool isRecordedContents(const Value *Root, const Value *Seed,
                        const Value *ExecuteIdentity, unsigned MutableIndex = 0) {
  SmallPtrSet<const Value *, 32> Visited, Seeded;
  DenseMap<const Value *, SmallVector<const Value *, 4>> Users;
  SmallVector<const Value *, 16> Pending{Root};
  while (!Pending.empty()) {
    const Value *V = Pending.pop_back_val();
    if (!Visited.insert(V).second || V == Seed)
      continue;
    SmallVector<const Value *, 4> Inputs;
    if (const auto *Phi = dyn_cast<PHINode>(V)) {
      for (const Value *Incoming : Phi->incoming_values())
        Inputs.push_back(Incoming);
    } else if (const auto *Extract = dyn_cast<ExtractValueInst>(V);
               ExecuteIdentity && Extract && Extract->getNumIndices() == 1 &&
               Extract->getIndices()[0] == MutableIndex) {
      const auto *II =
          dyn_cast<IntrinsicInst>(Extract->getAggregateOperand());
      if (!II || (II->getIntrinsicID() !=
                      Intrinsic::riscv_tt_contents_replay_template_execute &&
                  II->getIntrinsicID() !=
                      Intrinsic::riscv_tt_contents_replay_template_mop_execute) ||
          II->arg_size() < 2 || II->getArgOperand(0) != ExecuteIdentity)
        return false;
      const auto *Count = dyn_cast<ConstantInt>(II->getArgOperand(1));
      if (!Count || Count->getZExtValue() <= MutableIndex ||
          Count->getZExtValue() > 8 ||
          II->arg_size() < 2 + Count->getZExtValue())
        return false;
      Inputs.push_back(II->getArgOperand(2 + MutableIndex));
    } else {
      return false;
    }
    if (Inputs.empty())
      return false;
    for (const Value *Input : Inputs) {
      Users[Input].push_back(V);
      Pending.push_back(Input);
    }
  }
  if (!Visited.contains(Seed))
    return false;
  Pending.push_back(Seed);
  while (!Pending.empty()) {
    const Value *V = Pending.pop_back_val();
    if (!Seeded.insert(V).second)
      continue;
    auto Found = Users.find(V);
    if (Found != Users.end())
      append_range(Pending, Found->second);
  }
  return Seeded.size() == Visited.size();
}

bool isContentsReplayRecipe(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::riscv_tt_contents_replay_template_record_iadd:
  case Intrinsic::riscv_tt_contents_replay_template_record_and:
  case Intrinsic::riscv_tt_contents_replay_template_record_or:
  case Intrinsic::riscv_tt_contents_replay_template_record_xor:
  case Intrinsic::riscv_tt_contents_replay_template_record_not:
    return true;
  default:
    return false;
  }
}

// A Replay tuple only packs independent SFPR results. It is not a fixed
// hardware group, an aggregate input, or permission for aggregate transport.
bool isContentsReplayTuple(const IntrinsicInst &II) {
  if ((II.getIntrinsicID() !=
           Intrinsic::riscv_tt_contents_replay_template_execute &&
       II.getIntrinsicID() !=
           Intrinsic::riscv_tt_contents_replay_template_mop_execute) ||
      II.arg_size() < 3)
    return false;
  const auto *Count = dyn_cast<ConstantInt>(II.getArgOperand(1));
  const auto *Tuple = dyn_cast<StructType>(II.getType());
  if (!Count || Count->getZExtValue() < 1 || Count->getZExtValue() > 8 ||
      !Tuple || !Tuple->isLiteral() || Tuple->isPacked() ||
      Tuple->getNumElements() != Count->getZExtValue())
    return false;
  return all_of(Tuple->elements(), [](Type *Ty) { return isCarrier(Ty); });
}

Error verifyContentsReplayTemplates(Function &F, DominatorTree &DT) {
  struct Template {
    const IntrinsicInst *Begin;
    const IntrinsicInst *End = nullptr;
    SmallVector<const Value *, 8> Mutables;
    SmallVector<const Value *, 8> Captures;
    unsigned Words = 0;
    bool RecordOnly = false;
  };
  DenseMap<uint32_t, Template> Templates;
  SmallVector<const IntrinsicInst *, 4> Executions, MOPBindings;
  for (const BasicBlock &BB : F) {
    Template *Active = nullptr;
    uint32_t ActiveID = 0;
    for (const Instruction &I : BB) {
      if (I.isDebugOrPseudoInst())
        continue;
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      auto Kind = II ? II->getIntrinsicID() : Intrinsic::not_intrinsic;
      if (Kind == Intrinsic::riscv_tt_contents_replay_template_begin) {
        if (II->arg_size() < 6 || II->arg_size() > 13)
          return invalid(F,
              "contents replay begin requires one to eight flat contents roles");
        auto ID = immediate(*II, 0);
        if (!ID)
          return ID.takeError();
        if (Active || Templates.contains(*ID))
          return invalid(F, "contents replay requires one non-nested producer");
        if (Error E = range(*II, 1, 0, 1))
          return E;
        if (Error E = range(*II, 2, 0, 1))
          return E;
        if (Error E = range(*II, 3, 0, 31))
          return E;
        auto Placement = immediate(*II, 2);
        auto Start = immediate(*II, 3);
        if (!Placement || !Start)
          return invalid(F, "contents replay requires static placement");
        if (!*Placement && *Start)
          return invalid(F,
              "contents replay automatic placement requires zero start");
        auto Count = immediate(*II, 4);
        if (!Count)
          return Count.takeError();
        if (!*Count || *Count > II->arg_size() - 5)
          return invalid(F,
              "contents replay mutable count differs from its flat roles");
        bool RecordOnly = cast<ConstantInt>(II->getArgOperand(1))->isZero();
        SmallPtrSet<const Value *, 8> Roles;
        for (unsigned R = 5; R < II->arg_size(); ++R) {
          if (!isCarrier(II->getArgOperand(R)->getType()))
            return invalid(F,
                "contents replay roles require flat full contents values");
          if (!Roles.insert(II->getArgOperand(R)).second)
            return invalid(F,
                "contents replay capture and mutable roles must be distinct");
        }
        ActiveID = *ID;
        Active = &Templates.try_emplace(*ID, Template{II}).first->second;
        for (unsigned R = 5; R < II->arg_size(); ++R)
          (R < 5 + *Count ? Active->Mutables : Active->Captures)
              .push_back(II->getArgOperand(R));
        Active->RecordOnly = RecordOnly;
        continue;
      }
      if (Kind == Intrinsic::riscv_tt_contents_replay_template_end) {
        if (II->arg_size() < 2)
          return invalid(F,
              "contents replay end requires its actual final contents");
        auto ID = immediate(*II, 0);
        if (!ID)
          return ID.takeError();
        if (!Active || *ID != ActiveID || !Active->Words ||
            II->arg_size() != 1 + Active->Mutables.size())
          return invalid(F,
              "contents replay end requires its actual final contents");
        for (auto [R, Final] : enumerate(Active->Mutables))
          if (II->getArgOperand(1 + R) != Final)
            return invalid(F,
                "contents replay end requires its actual final contents");
        Active->End = II;
        Active = nullptr;
        continue;
      }
      if (Kind == Intrinsic::riscv_tt_contents_replay_template_mop && !Active) {
        auto Slot = immediate(*II, 1);
        if (!Slot)
          return Slot.takeError();
        if (*Slot != 5 && *Slot != 7 && *Slot != 8)
          return invalid(F, "contents MOP binding requires slot5, slot7 or slot8");
        MOPBindings.push_back(II);
        continue;
      }
      if ((Kind == Intrinsic::riscv_tt_contents_replay_template_execute ||
           Kind == Intrinsic::riscv_tt_contents_replay_template_mop_execute) &&
          !Active) {
        if (!isContentsReplayTuple(*II))
          return invalid(F,
              "contents replay execute requires an unpacked literal tuple matching mutable count");
        Executions.push_back(II);
        continue;
      }
      if (isContentsReplayRecipe(Kind)) {
        auto ID = immediate(*II, 0);
        if (!ID)
          return ID.takeError();
        if (!Active || !Active->RecordOnly || *ID != ActiveID)
          return invalid(F,
              "contents replay recipe requires its active record-only owner");
        auto Mutable = immediate(*II, 1);
        if (!Mutable)
          return Mutable.takeError();
        if (*Mutable >= Active->Mutables.size())
          return invalid(F,
              "contents replay recipe destination requires a declared mutable role");
        if (Kind != Intrinsic::riscv_tt_contents_replay_template_record_not) {
          auto Source = immediate(*II, 2);
          if (!Source)
            return Source.takeError();
          if (*Source >= Active->Mutables.size() + Active->Captures.size())
            return invalid(F,
                "contents replay recipe source requires a declared contents role");
        }
        if (++Active->Words > TensixReplaySlotCount)
          return invalid(F, "contents replay body exceeds replay capacity");
        continue;
      }
      if (!Active)
        continue;
      if (Active->RecordOnly)
        return invalid(F,
            "contents record-only body requires a non-executing recipe");
      // The first operand is the destructive old destination. Other contents
      // operands read current mutable roles or declared captures; native
      // instruction projection checks every immediate and its effects.
      unsigned Sources = 0;
      if (Kind == Intrinsic::riscv_tt_contents_iadd ||
          Kind == Intrinsic::riscv_tt_contents_mov ||
          Kind == Intrinsic::riscv_tt_contents_and ||
          Kind == Intrinsic::riscv_tt_contents_or ||
          Kind == Intrinsic::riscv_tt_contents_xor ||
          Kind == Intrinsic::riscv_tt_contents_divp2)
        Sources = 1;
      else if (Kind == Intrinsic::riscv_tt_contents_add ||
               Kind == Intrinsic::riscv_tt_contents_mul)
        Sources = 2;
      else if (Kind == Intrinsic::riscv_tt_contents_mad)
        Sources = 3;
      else if (Kind != Intrinsic::riscv_tt_contents_not &&
               Kind != Intrinsic::riscv_tt_contents_add_immediate &&
               Kind != Intrinsic::riscv_tt_contents_mul_immediate &&
               Kind != Intrinsic::riscv_tt_contents_load_immediate)
        return invalid(F,
            "contents replay requires current mutable contents and declared read-only captures");
      auto Mutable = find(Active->Mutables, II->getArgOperand(0));
      if (Mutable == Active->Mutables.end())
        return invalid(F,
            "contents replay requires current mutable contents and declared read-only captures");
      for (unsigned R = 1; R <= Sources; ++R)
        if (!is_contained(Active->Mutables, II->getArgOperand(R)) &&
            !is_contained(Active->Captures, II->getArgOperand(R)))
          return invalid(F,
              "contents replay requires current mutable contents and declared read-only captures");
      *Mutable = II;
      if (++Active->Words > TensixReplaySlotCount)
        return invalid(F, "contents replay body exceeds replay capacity");
    }
    if (Active)
      return invalid(F, "contents replay requires an end in its basic block");
  }
  for (const IntrinsicInst *Bind : MOPBindings) {
    auto ID = immediate(*Bind, 0);
    if (!ID)
      return ID.takeError();
    auto Found = Templates.find(*ID);
    if (Found == Templates.end() || !Found->second.End ||
        !DT.dominates(Found->second.End, Bind))
      return invalid(F, "contents MOP binding requires a dominating record");
  }
  for (const IntrinsicInst *Execute : Executions) {
    auto ID = immediate(*Execute, 0);
    if (!ID)
      return ID.takeError();
    auto Found = Templates.find(*ID);
    if (Found == Templates.end() || !Found->second.End ||
        !DT.dominates(Found->second.End, Execute))
      return invalid(F, "contents replay execute requires a dominating record");
    const Template &T = Found->second;
    unsigned First = 2;
    if (Execute->arg_size() != First + T.Mutables.size() + T.Captures.size())
      return invalid(F,
          "contents replay execute arity differs from its recorded roles");
    auto Count = immediate(*Execute, 1);
    if (!Count)
      return Count.takeError();
    if (*Count != T.Mutables.size())
      return invalid(F,
          "contents replay execute mutable count differs from its record");
    for (unsigned R = First; R < Execute->arg_size(); ++R)
      if (!isCarrier(Execute->getArgOperand(R)->getType()))
        return invalid(F,
            "contents replay roles require flat full contents values");
    for (auto [R, Capture] : enumerate(T.Captures))
      if (!isRecordedContents(
              Execute->getArgOperand(First + T.Mutables.size() + R), Capture,
              nullptr))
        return invalid(F,
            "contents replay execute requires its recorded read-only capture");
    for (auto [R, Mutable] : enumerate(T.Mutables))
      if (!isRecordedContents(Execute->getArgOperand(First + R), Mutable,
                              Execute->getArgOperand(0), R))
        return invalid(F,
            "contents replay execute requires its current mutable contents");
  }
  return Error::success();
}

Error verifyFullLoadsOutsideRecording(Function &F) {
  for (const BasicBlock &BB : F) {
    bool TemplateActive = false;
    unsigned RawRemaining = 0;
    for (const Instruction &I : BB) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      if (!II)
        continue;
      Intrinsic::ID ID = II->getIntrinsicID();
      if (ID == Intrinsic::riscv_tt_contents_replay_template_begin ||
          ID == Intrinsic::riscv_tt_replay_template_begin) {
        TemplateActive = true;
        continue;
      }
      if (ID == Intrinsic::riscv_tt_contents_replay_template_end ||
          ID == Intrinsic::riscv_tt_replay_template_end) {
        TemplateActive = false;
        continue;
      }
      if (ID == Intrinsic::riscv_tt_replay ||
          ID == Intrinsic::riscv_tt_replay_port) {
        unsigned First = ID == Intrinsic::riscv_tt_replay_port ? 1 : 0;
        const auto *Port =
            First ? dyn_cast<ConstantInt>(II->getArgOperand(0)) : nullptr;
        const auto *Record = dyn_cast<ConstantInt>(II->getArgOperand(First));
        const auto *Length =
            dyn_cast<ConstantInt>(II->getArgOperand(First + 2));
        // Full LOAD issues on the current thread's local port. Other port
        // recordings do not capture it; malformed fields retain their normal
        // verifier rather than becoming a guessed recording boundary.
        if ((!First || (Port && Port->isZero())) && Record && Record->isOne() &&
            Length && Length->getZExtValue() <= 32)
          RawRemaining = Length->getZExtValue();
        continue;
      }
      if (ID == Intrinsic::riscv_tt_contents_load_full &&
          (TemplateActive || RawRemaining))
        return invalid(
            F, "contents full LOAD is not supported in replay recording");
      const auto *Issue = RISCV::getTensixInstructionByIntrinsic(ID);
      if (RawRemaining && (isTensixContentsIntrinsic(ID) ||
                           (Issue && ID != Issue->MopIntrinsicID)))
        --RawRemaining;
    }
  }
  return Error::success();
}

Error verifyReplayTemplates(Function &F, DominatorTree &DT) {
  if (Error E = verifyFullLoadsOutsideRecording(F))
    return E;
  if (Error E = verifyContentsReplayTemplates(F, DT))
    return E;
  DenseMap<uint32_t, const IntrinsicInst *> Producers, Ends;
  SmallVector<std::pair<uint32_t, const IntrinsicInst *>, 4> Executions;
  for (const BasicBlock &BB : F) {
    const IntrinsicInst *Active = nullptr;
    uint32_t ActiveID = 0;
    for (const Instruction &I : BB) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      auto Kind = II ? II->getIntrinsicID() : Intrinsic::not_intrinsic;
      if (Kind == Intrinsic::riscv_tt_replay_template_begin ||
          Kind == Intrinsic::riscv_tt_replay_template_end ||
          Kind == Intrinsic::riscv_tt_replay_template_execute ||
          Kind == Intrinsic::riscv_tt_replay_template_mop) {
        auto Identity = immediate(*II, 0);
        if (!Identity)
          return Identity.takeError();
        uint32_t ID = *Identity;
        if (Kind == Intrinsic::riscv_tt_replay_template_begin) {
          if (Active || !Producers.try_emplace(ID, II).second)
            return invalid(F,
                           "replay template requires one non-nested producer");
          Active = II;
          ActiveID = ID;
        } else if (Kind == Intrinsic::riscv_tt_replay_template_end) {
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
    if (Callee && isRetiredTensixSFPUIntrinsicName(Callee->getName()))
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
      if (II->getIntrinsicID() == Intrinsic::riscv_tt_coprocessor_wait)
        return invalid(
            F, "coprocessor wait requires +xtttensixbh or +xtttensixwh");
      return invalid(F, "Tensix intrinsic requires +xtttensixbh");
    }
    // The scalar readback dependency uses ordinary RISC-V instructions rather
    // than a model-specific Tensix instruction encoding.
    if (Wormhole && isTensixContentsIntrinsic(II->getIntrinsicID())) {
      if (Error E = verifyTensixContentsIntrinsicFields(*II, true))
        return invalid(F, toString(std::move(E)));
    } else if (Wormhole &&
               II->getIntrinsicID() != Intrinsic::riscv_tt_dependent_use &&
               II->getIntrinsicID() != Intrinsic::riscv_tt_coprocessor_wait &&
               (II->getIntrinsicID() == Intrinsic::riscv_tt_replay ||
                II->getIntrinsicID() == Intrinsic::riscv_tt_mop ||
                (II->getIntrinsicID() !=
                     Intrinsic::riscv_tt_mop_control_write &&
                 II->getIntrinsicID() != Intrinsic::riscv_tt_mop_clear &&
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
    Intrinsic::ID ID = II->getIntrinsicID();
    if (ID == Intrinsic::riscv_tt_dependent_use)
      continue;
    if (ID == Intrinsic::riscv_tt_coprocessor_wait) {
      if (Executor != "trisc0" && Executor != "trisc1" && Executor != "trisc2")
        return invalid(F, "coprocessor wait requires a TRISC executor");
      continue;
    }
    if (Executor == "ncrisc")
      return invalid(F, "ncrisc cannot issue Tensix instructions");
    if (isTensixSFPUIntrinsic(ID)) {
      if (isTensixBoundSFPUIntrinsic(ID)) {
        if (Executor != "trisc0" && Executor != "trisc1" &&
            Executor != "trisc2")
          return invalid(F, "bound SFPU execution requires a TRISC executor");
      } else if (Executor != "trisc1") {
        return invalid(F,
                       "Tensix SFPU execution requires tensix-executor=trisc1");
      }
      continue;
    }
    if (ID == Intrinsic::riscv_tt_replay_record_end)
      continue;
    if (ID == Intrinsic::riscv_tt_replay_template_begin) {
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
    if (ID == Intrinsic::riscv_tt_replay_template_end ||
        ID == Intrinsic::riscv_tt_replay_template_execute)
      continue;
    if (ID == Intrinsic::riscv_tt_replay_template_mop) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 1, 2, 8))
        return E;
      continue;
    }
    if (ID == Intrinsic::riscv_tt_mop_control_write) {
      if (Executor == "brisc")
        return invalid(F, "MOP control programming requires a TRISC executor");
      if (Error E = range(*II, 0, 0, 1))
        return E;
      if (!isDefinedOffset(II->getArgOperand(1), *II, SE, AC, DT))
        return invalid(F, "MOP control value must be defined and non-poison");
      continue;
    }
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
    if (Wormhole) {
      SmallVector<uint32_t, 16> Fields;
      for (unsigned N = 0; N != Info->NumFields; ++N) {
        auto Value = immediate(*II, N);
        if (!Value)
          return Value.takeError();
        Fields.push_back(static_cast<uint32_t>(*Value));
      }
      if (Error E = RISCV::verifyTensixWormholeInstructionFields(ID, Fields))
        return invalid(F, toString(std::move(E)));
    }
    if (Mop) {
      if (Executor == "brisc")
        return invalid(F, "MOP slot programming requires a TRISC executor");
      if (Error E = range(*II, 0, 2, 8))
        return E;
      if (Info->IntrinsicID == Intrinsic::riscv_tt_mop ||
          Info->IntrinsicID == Intrinsic::riscv_tt_mop_cfg)
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
      if (!Port || Replay) {
        if (Error E = range(*II, First + N, 0, Field->MaxValue))
          return E;
        continue;
      }
      const Value *V = II->getArgOperand(First + N);
      if (!isDefinedOffset(V, *II, SE, AC, DT))
        return invalid(F, Twine(Info->getName()) + " field " +
                              Field->getName() +
                              " must be defined and non-poison");
      const SCEV *FieldValue = SE.getSCEV(const_cast<Value *>(V));
      // A loop-header value also exists on the exiting edge. Admit a dynamic
      // field only when its unsigned bound holds at the actual issuing use,
      // including a dominating body guard; retain the separate poison proof.
      // Combine the recurrence's global bounds with edge constraints at this
      // use. Neither alone need prove derived fields such as 1 - induction.
      ConstantRange Bounds = SE.getUnsignedRange(FieldValue);
      if (!Bounds.isEmptySet() && Bounds.getUnsignedMax().ugt(Field->MaxValue))
        Bounds = Bounds.intersectWith(
            LVI.getConstantRangeAtUse(II->getArgOperandUse(First + N),
                                      /*UndefAllowed=*/false),
            ConstantRange::Unsigned);
      if (!Bounds.isEmptySet() && Bounds.getUnsignedMax().ugt(Field->MaxValue))
        Bounds = refineWithRelatedInductions(FieldValue, Bounds, *II, SE, LVI,
                                             AC, DT);
      if (!Bounds.isEmptySet() && Bounds.getUnsignedMax().ugt(Field->MaxValue))
        Bounds = refineWithScalarExpressions(const_cast<Value *>(V), Bounds,
                                             *II, SE, LVI);
      if (Bounds.isEmptySet() ||
          (Bounds.getUnsignedMax().ugt(Field->MaxValue) &&
           !SE.isKnownPredicateAt(ICmpInst::ICMP_ULE, FieldValue,
                                  SE.getConstant(V->getType(), Field->MaxValue),
                                  II)))
        return invalid(F, Twine(Info->getName()) + " field " +
                              Field->getName() + " must be proven in [0, " +
                              Twine(Field->MaxValue) + "]");
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
  if (isTensixContentsIntrinsic(II.getIntrinsicID())) {
    switch (II.getIntrinsicID()) {
    case Intrinsic::riscv_tt_contents_replay_template_begin:
    case Intrinsic::riscv_tt_contents_replay_template_end:
    case Intrinsic::riscv_tt_contents_replay_template_mop:
    case Intrinsic::riscv_tt_contents_replay_template_mop_execute:
    case Intrinsic::riscv_tt_contents_replay_template_execute:
    case Intrinsic::riscv_tt_contents_replay_template_record_iadd:
    case Intrinsic::riscv_tt_contents_replay_template_record_and:
    case Intrinsic::riscv_tt_contents_replay_template_record_or:
    case Intrinsic::riscv_tt_contents_replay_template_record_xor:
    case Intrinsic::riscv_tt_contents_replay_template_record_not:
      return verifyTensixContentsIntrinsicFields(II);
    case Intrinsic::riscv_tt_contents_read_creg:
    case Intrinsic::riscv_tt_contents_read_constant: {
      auto Selector = immediate(II, 0);
      if (!Selector)
        return Selector.takeError();
      if (Error E = verifyTensixContentsIntrinsicFields(II))
        return invalid(*II.getFunction(), toString(std::move(E)));
      return Error::success();
    }
    case Intrinsic::riscv_tt_contents_add:
    case Intrinsic::riscv_tt_contents_mul:
      return range(II, 3, 0, 3);
    case Intrinsic::riscv_tt_contents_load_index: {
      if (Error E = verifyTensixContentsIntrinsicFields(II))
        return E;
      Value *OffsetValue = II.getArgOperand(2);
      if (!isDefinedOffset(OffsetValue, II, SE, AC, DT))
        return invalid(
            *II.getFunction(),
            "contents paired LOAD Dst offset must be defined and non-poison");
      const SCEV *Offset = SE.getSCEV(OffsetValue);
      if (SE.getUnsignedRange(Offset).getUnsignedMax().ugt(1023))
        return invalid(
            *II.getFunction(),
            "contents paired LOAD Dst offset must be proven in [0, 1023]");
      return Error::success();
    }
    case Intrinsic::riscv_tt_contents_load_full:
    case Intrinsic::riscv_tt_contents_load:
    case Intrinsic::riscv_tt_contents_store: {
      bool Full = II.getIntrinsicID() == Intrinsic::riscv_tt_contents_load_full;
      if (Full) {
        if (Error E = verifyTensixContentsIntrinsicFields(II))
          return E;
      } else {
        if (Error E = range(II, 2, 0, 7))
          return E;
        auto Format = immediate(II, 3);
        if (!Format)
          return Format.takeError();
        if (*Format > 15 || !(0xc17fu & (1u << *Format)))
          return invalid(*II.getFunction(), "unsupported contents Dst format");
      }
      Value *OffsetValue = II.getArgOperand(Full ? 0 : 1);
      if (!isDefinedOffset(OffsetValue, II, SE, AC, DT))
        return invalid(*II.getFunction(),
                       "contents Dst offset must be defined and non-poison");
      const SCEV *Offset = SE.getSCEV(OffsetValue);
      if (SE.getUnsignedRange(Offset).getUnsignedMax().ugt(1023))
        return invalid(*II.getFunction(),
                       "contents Dst offset must be proven in [0, 1023]");
      return Error::success();
    }
    default:
      return verifyTensixContentsIntrinsicFields(II);
    }
  }
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
    if (SE.getUnsignedRange(Offset).getUnsignedMax().ugt(1023))
      return invalid(*II.getFunction(),
                     "bound Dst offset must be proven in [0, 1023]");
  }
  return Error::success();
}

// Scalar numerical results may occupy either half of SFPR, so they cannot
// hide SFPLOAD's paired destination capture or TEN-2932 writes to L4..7.
// SFPLOADI has one explicit destination and is exempt from TEN-2932.
// Transpose is also architecturally exempt: Transpose8 receives every old
// contents value/result in L0..7; Transpose4 exposes L4..7 as machine scratch
// clobbers. Indexing cannot add a hidden write to either form.
// The paired LOAD view likewise explicitly receives the conditional L4
// companion and its old contents instead of hiding that second write.
// SWAPIndex similarly receives both values and both companion indices.
bool requiresDisabledDestIndex(const IntrinsicInst &II) {
  return containsCarrier(II.getType()) &&
         II.getIntrinsicID() != Intrinsic::riscv_tt_contents_load_immediate &&
         II.getIntrinsicID() != Intrinsic::riscv_tt_contents_transpose4 &&
         II.getIntrinsicID() != Intrinsic::riscv_tt_contents_transpose8 &&
         II.getIntrinsicID() != Intrinsic::riscv_tt_contents_load_index &&
         II.getIntrinsicID() != Intrinsic::riscv_tt_contents_swap_index;
}

struct ContentsLaneConfig {
  std::optional<bool> Enable;
  std::optional<bool> Flags;
  bool DestIndexDisabled = false;
  bool RowsUnmasked = false;
  bool DstReadUnblocked = false;
  std::array<uint8_t, 4> CRegColumns = {};
  SmallPtrSet<const Value *, 8> IndexBitZeroValues;

  bool allColumnsEnabled() const { return Enable == false || Flags == true; }
  bool operator==(const ContentsLaneConfig &Other) const {
    return Enable == Other.Enable && Flags == Other.Flags &&
           DestIndexDisabled == Other.DestIndexDisabled &&
           RowsUnmasked == Other.RowsUnmasked &&
           DstReadUnblocked == Other.DstReadUnblocked &&
           CRegColumns == Other.CRegColumns &&
           IndexBitZeroValues == Other.IndexBitZeroValues;
  }
  bool operator!=(const ContentsLaneConfig &Other) const {
    return !(*this == Other);
  }
  void meet(const ContentsLaneConfig &Other) {
    if (Enable != Other.Enable)
      Enable.reset();
    if (Flags != Other.Flags)
      Flags.reset();
    DestIndexDisabled &= Other.DestIndexDisabled;
    RowsUnmasked &= Other.RowsUnmasked;
    DstReadUnblocked &= Other.DstReadUnblocked;
    for (unsigned I = 0; I != CRegColumns.size(); ++I)
      CRegColumns[I] &= Other.CRegColumns[I];
    SmallPtrSet<const Value *, 8> Common;
    for (const Value *V : IndexBitZeroValues)
      if (Other.IndexBitZeroValues.contains(V))
        Common.insert(V);
    IndexBitZeroValues = std::move(Common);
  }
};

bool preservesContentsLaneConfig(const Value *Pointer, Type *AccessType,
                                 const DataLayout &DL, bool IsLoad) {
  // Only classify complete accesses to known hardware owners. In particular,
  // excluding just the instruction-port base would miss its 64 KiB aliases.
  if (Pointer->getType()->getPointerAddressSpace() != 0 ||
      DL.getPointerSizeInBits() != 32 || !AccessType->isSized())
    return false;
  TypeSize Size = DL.getTypeStoreSize(AccessType);
  if (Size.isScalable() || Size.isZero())
    return false;

  int64_t Offset;
  const Value *Base = GetPointerBaseWithConstantOffset(Pointer, Offset, DL);
  APInt Address(32, 0);
  if (!isa<ConstantPointerNull>(Base)) {
    const auto *Cast = dyn_cast<Operator>(Base);
    if (!Cast || Cast->getOpcode() != Instruction::IntToPtr)
      return false;
    // ConstantHoisting may have materialized the integer in scalar SSA.
    // Accept only an exact known value, never a guessed address range.
    KnownBits Integer = computeKnownBits(Cast->getOperand(0), DL);
    if (!Integer.isConstant())
      return false;
    Address = Integer.getConstant().zextOrTrunc(32);
  }
  Address += APInt(32, Offset);
  uint64_t Start = Address.getZExtValue();
  uint64_t Bytes = Size.getFixedValue();
  auto Within = [&](uint64_t Begin, uint64_t End) {
    return Start >= Begin && Start < End && Bytes <= End - Start;
  };

  // Wormhole's L1 extent is also present on Blackhole. Ordinary data accesses
  // there do not issue instructions or write the coprocessor configuration.
  if (Within(0, 0x0016e000))
    return true;
  // Manual TTSync has two donecheck words on both machines. Stores discard
  // their data; loads wait for the corresponding completion condition.
  if (Within(0xffe80004, 0xffe8000c))
    return true;
  // The wallclock low, current-high and latched-high words are read-only here.
  // Other debug registers can issue Tensix instructions and remain unknown.
  return IsLoad && Within(0xffb121f0, 0xffb121fc);
}

void updateContentsLaneConfig(const Instruction &I, ContentsLaneConfig &State,
                              bool &RecordOnly, unsigned &RawRecordRemaining) {
  const auto *II = dyn_cast<IntrinsicInst>(&I);
  if (!II) {
    // Volatile loads are ordered effects in LLVM, so mayWriteToMemory alone
    // cannot distinguish completion observation from an instruction issue.
    if (const auto *Load = dyn_cast<LoadInst>(&I)) {
      if (!preservesContentsLaneConfig(Load->getPointerOperand(),
                                       Load->getType(),
                                       I.getModule()->getDataLayout(), true))
        State = {};
      return;
    }
    if (const auto *Store = dyn_cast<StoreInst>(&I)) {
      if (!preservesContentsLaneConfig(Store->getPointerOperand(),
                                       Store->getValueOperand()->getType(),
                                       I.getModule()->getDataLayout(), false))
        State = {};
      return;
    }
    // Unknown accesses and opaque calls may change LaneConfig, including
    // through an alias of a Tensix instruction port.
    if (I.mayWriteToMemory() || isa<CallBase>(I))
      State = {};
    return;
  }
  Intrinsic::ID ID = II->getIntrinsicID();
  if (ID == Intrinsic::riscv_tt_replay_template_begin) {
    RecordOnly = cast<ConstantInt>(II->getArgOperand(1))->isZero();
    return;
  }
  if (ID == Intrinsic::riscv_tt_replay_template_end) {
    RecordOnly = false;
    return;
  }
  if (RecordOnly)
    return;
  const auto *Issue = RISCV::getTensixInstructionByIntrinsic(ID);
  if (ID == Intrinsic::riscv_tt_replay ||
      ID == Intrinsic::riscv_tt_replay_port) {
    unsigned First = ID == Intrinsic::riscv_tt_replay_port ? 1 : 0;
    bool Record = !cast<ConstantInt>(II->getArgOperand(First))->isZero();
    bool Execute = !cast<ConstantInt>(II->getArgOperand(First + 1))->isZero();
    if (Record && !Execute) {
      RawRecordRemaining =
          cast<ConstantInt>(II->getArgOperand(First + 2))->getZExtValue();
      return;
    }
  }
  if (RawRecordRemaining && (isTensixContentsIntrinsic(ID) ||
                             (Issue && ID != Issue->MopIntrinsicID))) {
    // These admitted operations each issue one native Tensix word. Scalar
    // address/field computation and the record-end marker issue no word.
    --RawRecordRemaining;
    return;
  }
  if (ID == Intrinsic::riscv_tt_contents_encc) {
    unsigned Value = cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
    unsigned Mode = cast<ConstantInt>(II->getArgOperand(1))->getZExtValue();
    if (Mode & 2)
      State.Enable = bool(Value & 1);
    else if ((Mode & 1) && State.Enable)
      State.Enable = !*State.Enable;
    State.Flags = (Mode & 8) ? bool(Value & 2) : true;
    return;
  }
  if (ID == Intrinsic::riscv_tt_contents_config_reset) {
    // SFPCONFIG reads the first eight columns' effective CC, independently
    // of ROW_MASK, and broadcasts the selected configuration vertically.
    // Clearing selected low bits preserves already-known zero configuration,
    // but only a full broadcast establishes every lane from unknown state.
    State.DestIndexDisabled |= State.allColumnsEnabled();
    State.RowsUnmasked |= State.allColumnsEnabled();
    State.DstReadUnblocked |= State.allColumnsEnabled();
    return;
  }
  switch (ID) {
  case Intrinsic::riscv_tt_contents_read_constant: {
    unsigned Selector = cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
    // MOVAll snapshots every lane independently of CC and ROW_MASK. The zero
    // and one constants both have bit 2 clear; C15's lane indices do not.
    if (Selector == 9 || Selector == 10)
      State.IndexBitZeroValues.insert(II);
    return;
  }
  case Intrinsic::riscv_tt_contents_load_immediate: {
    if (!State.IndexBitZeroValues.contains(II->getArgOperand(0)))
      return;
    unsigned Value = cast<ConstantInt>(II->getArgOperand(1))->getZExtValue();
    unsigned Mode = cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
    // Both inactive old contents and the written value must have bit 2 clear.
    // This deliberately does not infer full overwrite from the current CC:
    // numerical writes also observe ROW_MASK, unlike SFPCONFIG's broadcast.
    if (Mode == 0 || Mode == 1 || Mode == 8 ||
        ((Mode == 2 || Mode == 4 || Mode == 10) && !(Value & 4)))
      State.IndexBitZeroValues.insert(II);
    return;
  }
  case Intrinsic::riscv_tt_contents_config_creg: {
    if (!State.allColumnsEnabled())
      return;
    unsigned Selector = cast<ConstantInt>(II->getArgOperand(1))->getZExtValue();
    unsigned Mask = cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
    unsigned Mode = cast<ConstantInt>(II->getArgOperand(3))->getZExtValue();
    // Mode 8 uses only even Imm16 bits, one for each of the first eight
    // columns. SFPCONFIG broadcasts these columns vertically, ignoring ROW_MASK.
    uint8_t Columns = 0xff;
    if (Mode == 8) {
      Columns = 0;
      for (unsigned I = 0; I != 8; ++I)
        Columns |= ((Mask >> (2 * I)) & 1) << I;
    }
    State.CRegColumns[Selector - 11] |= Columns;
    return;
  }
  case Intrinsic::riscv_tt_contents_config_lane:
    // Admitted mode 8 assigns the selected columns from L0. A known-zero bit
    // cannot enable indexing in any selected/effective column. Keep unknown
    // incoming hardware state unknown; this is preservation, not initialization.
    State.DestIndexDisabled &=
        State.IndexBitZeroValues.contains(II->getArgOperand(0));
    // This source may independently set ROW_MASK and Dst read-block bits.
    // An index-bit proof does not establish either separate write condition.
    State.RowsUnmasked = false;
    State.DstReadUnblocked = false;
    return;
  case Intrinsic::riscv_tt_contents_setcc:
  case Intrinsic::riscv_tt_contents_gt:
  case Intrinsic::riscv_tt_contents_le:
  case Intrinsic::riscv_tt_contents_exexp:
  case Intrinsic::riscv_tt_contents_lz:
  case Intrinsic::riscv_tt_contents_iadd:
  case Intrinsic::riscv_tt_contents_iadd_immediate:
    State.Flags.reset();
    return;
  case Intrinsic::riscv_tt_contents_popc:
  case Intrinsic::riscv_tt_contents_compc:
    // Saved raw CC may differ from both the current flags and enable field.
    State.Enable.reset();
    State.Flags.reset();
    return;
  case Intrinsic::riscv_tt_replay_template_execute:
  case Intrinsic::riscv_tt_mop:
  case Intrinsic::riscv_tt_mop_port:
  case Intrinsic::riscv_tt_replay:
  case Intrinsic::riscv_tt_replay_port:
    // These execution effects have no contents LaneConfig summary yet.
    State = {};
    return;
  default:
    break;
  }
  if (isTensixContentsIntrinsic(ID) ||
      RISCV::getTensixInstructionByIntrinsic(ID) ||
      ID == Intrinsic::riscv_tt_dependent_use ||
      ID == Intrinsic::riscv_tt_read_response ||
      ID == Intrinsic::riscv_tt_coprocessor_wait ||
      ID == Intrinsic::riscv_tt_replay_record_end ||
      ID == Intrinsic::riscv_tt_replay_template_mop ||
      ID == Intrinsic::riscv_tt_mop_clear ||
      ID == Intrinsic::riscv_tt_mop_control_write || !II->mayWriteToMemory())
    return;
  State = {};
}

Error verifyContentsLaneConfig(Function &F) {
  // Absent entries are not-yet-reached CFG nodes. Entry has unknown hardware
  // configuration. Iterate the actual CFG to convergence; no source iteration
  // or replay sequence is expanded to approximate a loop's reaching facts.
  DenseMap<const BasicBlock *, ContentsLaneConfig> Outputs;
  bool Changed;
  do {
    Changed = false;
    for (const BasicBlock &BB : F) {
      std::optional<ContentsLaneConfig> Incoming;
      if (&BB == &F.getEntryBlock())
        Incoming.emplace();
      for (const BasicBlock *Pred : predecessors(&BB)) {
        auto It = Outputs.find(Pred);
        if (It == Outputs.end())
          continue;
        if (Incoming)
          Incoming->meet(It->second);
        else
          Incoming = It->second;
      }
      if (!Incoming)
        continue;
      ContentsLaneConfig State = *Incoming;
      bool RecordOnly = false;
      unsigned RawRecordRemaining = 0;
      for (const Instruction &I : BB)
        updateContentsLaneConfig(I, State, RecordOnly, RawRecordRemaining);
      if (RawRecordRemaining)
        return invalid(F, "contents SFPU raw replay recording must end in its "
                          "basic block");
      auto [It, Inserted] = Outputs.try_emplace(&BB, State);
      if (Inserted || It->second != State) {
        It->second = State;
        Changed = true;
      }
    }
  } while (Changed);

  for (const BasicBlock &BB : F) {
    if (!Outputs.count(&BB))
      continue;
    std::optional<ContentsLaneConfig> Incoming;
    if (&BB == &F.getEntryBlock())
      Incoming.emplace();
    for (const BasicBlock *Pred : predecessors(&BB)) {
      auto It = Outputs.find(Pred);
      if (It == Outputs.end())
        continue;
      if (Incoming)
        Incoming->meet(It->second);
      else
        Incoming = It->second;
    }
    ContentsLaneConfig State = *Incoming;
    bool RecordOnly = false;
    unsigned RawRecordRemaining = 0;
    for (const Instruction &I : BB) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      if (II &&
          II->getIntrinsicID() == Intrinsic::riscv_tt_contents_load_full &&
          (!State.allColumnsEnabled() || !State.DestIndexDisabled ||
           !State.RowsUnmasked || !State.DstReadUnblocked))
        return invalid(
            F, "contents full LOAD requires all lanes writable with unmasked "
               "rows, unblocked Dst reads and disabled destination indexing");
      if (II && isTensixContentsIntrinsic(II->getIntrinsicID()) &&
          requiresDisabledDestIndex(*II) && !State.DestIndexDisabled)
        return invalid(F, "contents SFPU requires ENABLE_DEST_INDEX known "
                          "disabled before a potentially affected write");
      if (II && II->getIntrinsicID() == Intrinsic::riscv_tt_contents_read_creg) {
        unsigned Selector =
            cast<ConstantInt>(II->getArgOperand(0))->getZExtValue();
        // MOV mode 2 snapshots every column, even when effective CC is off.
        if (State.CRegColumns[Selector - 11] != 0xff)
          return invalid(F, "contents CReg snapshot requires all eight columns "
                            "initialized");
      }
      updateContentsLaneConfig(I, State, RecordOnly, RawRecordRemaining);
    }
  }
  return Error::success();
}

} // namespace

Expected<TensixSFPUFunctionFacts>
llvm::verifyTensixSFPUFunction(Function &F, ScalarEvolution &SE,
                               AssumptionCache &AC, DominatorTree &DT) {
  TensixSFPUFunctionFacts Facts;
  bool Bound = false, Contents = false;
  for (const Instruction &I : instructions(F)) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || !isTensixSFPUIntrinsic(II->getIntrinsicID()))
      continue;
    Bound = isTensixBoundSFPUIntrinsic(II->getIntrinsicID()) || Bound;
    Contents = isTensixContentsIntrinsic(II->getIntrinsicID()) || Contents;
  }

  if (Bound && Contents)
    return invalid(F, "bound SFPU and contents SSA cannot mix in one function");

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
    if (II && II->getIntrinsicID() ==
                  Intrinsic::riscv_tt_contents_replay_template_execute &&
        !isContentsReplayTuple(*II))
      return invalid(F,
          "contents replay execute requires an unpacked literal tuple matching mutable count");
    bool GroupTuple =
        II && (getTensixContentsGroupResultCount(II->getIntrinsicID()) > 1 ||
               isContentsReplayTuple(*II));
    auto *Extract = dyn_cast<ExtractValueInst>(&I);
    auto *Tuple = Extract
                      ? dyn_cast<IntrinsicInst>(Extract->getAggregateOperand())
                      : nullptr;
    bool GroupExtract =
        Tuple &&
        (getTensixContentsGroupResultCount(Tuple->getIntrinsicID()) > 1 ||
         isContentsReplayTuple(*Tuple)) &&
        Extract->getNumIndices() == 1 && isCarrier(I.getType());
    Facts.UsesSFPU |= SFPU;
    bool Carrier = containsCarrier(I.getType());
    for (Value *Input : I.operands()) {
      if (!containsCarrier(Input->getType()))
        continue;
      Carrier = true;
      if (isa<Constant>(Input))
        return invalid(
            F, "SFPU carrier constants require explicit target initialization");
      if (!isCarrier(Input->getType()) && !GroupExtract)
        return invalid(F, "SFPU carrier cannot be nested in an aggregate");
    }
    if (Bound && Carrier)
      return invalid(
          F, "bound SFPU ingress cannot contain an unbound carrier or PHI");
    if (Carrier && !SFPU && !isa<PHINode>(I) && !GroupExtract)
      return invalid(F,
                     "SFPU carrier is only legal in target intrinsics and PHI");
    if (containsCarrier(I.getType()) && !isCarrier(I.getType()) && !GroupTuple)
      return invalid(F, "SFPU carrier cannot be nested in an aggregate");
    if (SFPU) {
      if (II->hasOperandBundles())
        return invalid(F, "SFPU intrinsics do not admit operand bundles");
      if (Error E = verifyIntrinsic(*II, SE, AC, DT))
        return std::move(E);
    }
  }

  if (!Facts.UsesSFPU)
    return Facts;
  if (Contents)
    if (Error E = verifyContentsLaneConfig(F))
      return std::move(E);
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
      case Intrinsic::riscv_tt_setc16:
      case Intrinsic::riscv_tt_setc16_port:
      case Intrinsic::riscv_tt_dependent_use:
      case Intrinsic::riscv_tt_read_response:
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
    if (Error E = verifyScalarTensixSynchronization(F, !ST.is64Bit()))
      reportFatalUsageError(Twine(toString(std::move(E))));
    bool UsesSFPU = llvm::any_of(instructions(F), [](const Instruction &I) {
      auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && isTensixSFPUIntrinsic(II->getIntrinsicID());
    });
    bool UsesTensix = llvm::any_of(instructions(F), [](const Instruction &I) {
      const auto *II = dyn_cast<IntrinsicInst>(&I);
      return II && RISCV::isTensixIntrinsic(II->getIntrinsicID());
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
  if (Error E = verifyScalarTensixSynchronization(F, !ST.is64Bit()))
    reportFatalUsageError(Twine(toString(std::move(E))));
  bool UsesTensix = llvm::any_of(instructions(F), [](const Instruction &I) {
    const auto *II = dyn_cast<IntrinsicInst>(&I);
    return II && RISCV::isTensixIntrinsic(II->getIntrinsicID());
  });
  if (!ST.hasVendorXTTTensixBH() && !ST.hasVendorXTTTensixWH() && !UsesTensix)
    return PreservedAnalyses::all();
  if (Error E = verifyIssue(
          F, ST.hasVendorXTTTensixBH() || ST.hasVendorXTTTensixWH(),
          ST.hasVendorXTTTensixWH(), FAM.getResult<ScalarEvolutionAnalysis>(F),
          FAM.getResult<AssumptionAnalysis>(F),
          FAM.getResult<DominatorTreeAnalysis>(F)))
    reportFatalUsageError(Twine(toString(std::move(E))));
  auto Facts =
      verifyTensixSFPUFunction(F, FAM.getResult<ScalarEvolutionAnalysis>(F),
                               FAM.getResult<AssumptionAnalysis>(F),
                               FAM.getResult<DominatorTreeAnalysis>(F));
  if (!Facts)
    reportFatalUsageError(Twine(toString(Facts.takeError())));
  if (UsesSFPU)
    reportFatalUsageError(
        "Tensix SFPU requires the SelectionDAG legacy codegen pipeline");
  return PreservedAnalyses::all();
}
