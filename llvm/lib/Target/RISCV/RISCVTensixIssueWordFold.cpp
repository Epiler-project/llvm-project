//===-- RISCVTensixIssueWordFold.cpp ---------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Encode ordinary Tensix instruction-port issue into its complete word before
// instruction selection. Field ranges were already proven by IR verification,
// so each field contributes `value << shift` without overlap. Constant fields
// fold into the opcode base, and a select tree with constant leaves becomes
// branches on its conditions, outermost first, each yielding a complete word.
// The word and the port address are then ordinary scalar values that
// SelectionDAG, MachineLICM and MachineCSE can optimize.
// REPLAY and MOP control keep their field form for replay/MOP analysis.
//===----------------------------------------------------------------------===//
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/MC/MCContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <functional>
using namespace llvm;

#define DEBUG_TYPE "riscv-tensix-issue-word-fold"

namespace {
// Bounded so a pathological expression cannot grow the select tree.
constexpr unsigned MaxSelectDepth = 8;
constexpr unsigned MaxDecisionLeaves = 16;

bool isConstantTree(const Value *V, unsigned Depth) {
  if (isa<ConstantInt>(V))
    return true;
  const auto *S = dyn_cast<SelectInst>(V);
  return S && Depth && isConstantTree(S->getTrueValue(), Depth - 1) &&
         isConstantTree(S->getFalseValue(), Depth - 1);
}

unsigned countLeaves(const Value *V) {
  const auto *S = dyn_cast<SelectInst>(V);
  return S ? countLeaves(S->getTrueValue()) + countLeaves(S->getFalseValue())
           : 1;
}

// Recompute a side-effect-free condition next to the branch that consumes it,
// so a condition is evaluated only on the path that needs it. Values defined
// outside Origin, PHIs and memory reads are used in place; they dominate the
// issue point.
Value *sinkCondition(Value *V, BasicBlock *Origin, IRBuilder<> &B,
                     DenseMap<Value *, Value *> &Cache, unsigned Depth) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !Depth || I->getParent() != Origin || isa<PHINode>(I) ||
      I->mayHaveSideEffects() || I->mayReadFromMemory())
    return V;
  if (Value *Known = Cache.lookup(V))
    return Known;
  Instruction *Copy = I->clone();
  for (Use &U : Copy->operands())
    U.set(sinkCondition(U.get(), Origin, B, Cache, Depth - 1));
  B.Insert(Copy);
  Cache[V] = Copy;
  return Copy;
}

// Lower a select tree of constant fields as branches that test the outermost
// condition first, as the select semantics allow. Each leaf supplies a
// complete word to a PHI at the issue point. Returns that PHI.
Value *emitDecisionTree(IntrinsicInst *II, Value *Field, unsigned Shift,
                        uint32_t Or) {
  BasicBlock *Head = II->getParent();
  Function &F = *Head->getParent();
  LLVMContext &Context = F.getContext();
  BasicBlock *Tail = SplitBlock(Head, II);
  Head->getTerminator()->eraseFromParent();
  PHINode *Word = PHINode::Create(Type::getInt32Ty(Context),
                                  countLeaves(Field), "tt.word");
  Word->insertBefore(Tail->begin());
  std::function<void(Value *, BasicBlock *)> Emit = [&](Value *V,
                                                        BasicBlock *Block) {
    IRBuilder<> B(Block);
    if (auto *C = dyn_cast<ConstantInt>(V)) {
      B.CreateBr(Tail);
      Word->addIncoming(
          B.getInt32(uint32_t(C->getZExtValue() << Shift) | Or), Block);
      return;
    }
    auto *S = cast<SelectInst>(V);
    DenseMap<Value *, Value *> Cache;
    Value *Condition = sinkCondition(S->getCondition(), Head, B, Cache, 4);
    auto *True = BasicBlock::Create(Context, "tt.word.t", &F, Tail);
    auto *False = BasicBlock::Create(Context, "tt.word.f", &F, Tail);
    B.CreateCondBr(Condition, True, False);
    Emit(S->getTrueValue(), True);
    Emit(S->getFalseValue(), False);
  };
  Emit(Field, Head);
  return Word;
}

// Requires isConstantTree(V). Every leaf becomes (leaf << Shift) | Or.
Value *encodeTree(IRBuilder<> &B, Value *V, unsigned Shift, uint32_t Or) {
  if (auto *C = dyn_cast<ConstantInt>(V))
    return B.getInt32(uint32_t(C->getZExtValue() << Shift) | Or);
  auto *S = cast<SelectInst>(V);
  return B.CreateSelect(S->getCondition(),
                        encodeTree(B, S->getTrueValue(), Shift, Or),
                        encodeTree(B, S->getFalseValue(), Shift, Or));
}

Expected<bool> foldFunction(Function &F, const RISCVTargetMachine &TM) {
  const auto &ST = TM.getSubtarget<RISCVSubtarget>(F);
  SmallVector<IntrinsicInst *> Issues;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    if (II->getIntrinsicID() == Intrinsic::riscv_tt_issue_word)
      return createStringError("compiler-private Tensix issue word in input");
    const auto *Info = RISCV::getTensixInstructionByIntrinsic(
        II->getIntrinsicID());
    if (!Info || II->getIntrinsicID() != Info->PortIntrinsicID)
      continue;
    const auto *Machine = RISCV::getTensixMachineInfoByIntrinsic(
        Info->IntrinsicID);
    if (Machine && !RISCV::isTensixFieldPortOpcode(Machine->Opcode))
      Issues.push_back(II);
  }
  if (Issues.empty())
    return false;
  MCContext Context(TM.getTargetTriple(), TM.getMCAsmInfo(),
                    TM.getMCRegisterInfo(), ST);
  Function *IssueWord = Intrinsic::getOrInsertDeclaration(
      F.getParent(), Intrinsic::riscv_tt_issue_word);
  DenseMap<unsigned, RISCV::TensixPortLayout> Layouts;
  for (IntrinsicInst *II : Issues) {
    const auto *Info =
        RISCV::getTensixInstructionByIntrinsic(II->getIntrinsicID());
    const auto *Machine =
        RISCV::getTensixMachineInfoByIntrinsic(Info->IntrinsicID);
    auto Found = Layouts.find(Machine->Opcode);
    if (Found == Layouts.end()) {
      auto Layout = RISCV::getTensixPortLayout(
          Machine->Opcode, *ST.getInstrInfo(), ST, Context);
      if (!Layout)
        return Layout.takeError();
      Found = Layouts.try_emplace(Machine->Opcode, std::move(*Layout)).first;
    }
    const RISCV::TensixPortLayout &Layout = Found->second;
    IRBuilder<> B(II);
    uint32_t Constant = Layout.Base;
    SmallVector<unsigned> Dynamic;
    for (unsigned I = 0; I != Info->NumFields; ++I) {
      Value *Field = II->getArgOperand(1 + I);
      if (auto *C = dyn_cast<ConstantInt>(Field))
        Constant |= uint32_t(C->getZExtValue() << Layout.Shifts[I]);
      else
        Dynamic.push_back(I);
    }
    // Fold the constant part into the first select tree with constant leaves,
    // so a selected field yields complete words.
    Value *Word = nullptr;
    int Folded = -1;
    for (unsigned I : Dynamic) {
      Value *Field = II->getArgOperand(1 + I);
      if (!isConstantTree(Field, MaxSelectDepth))
        continue;
      Folded = I;
      if (isa<SelectInst>(Field) && countLeaves(Field) <= MaxDecisionLeaves) {
        Word = emitDecisionTree(II, Field, Layout.Shifts[I], Constant);
        B.SetInsertPoint(II);
      } else {
        Word = encodeTree(B, Field, Layout.Shifts[I], Constant);
      }
      break;
    }
    if (!Word)
      Word = B.getInt32(Constant);
    for (unsigned I : Dynamic) {
      if (int(I) == Folded)
        continue;
      Value *Field = II->getArgOperand(1 + I);
      if (Layout.Shifts[I])
        Field = B.CreateShl(Field, Layout.Shifts[I], "", /*HasNUW=*/true,
                            /*HasNSW=*/false);
      Word = B.CreateOr(Word, Field, "", /*IsDisjoint=*/true);
    }
    B.CreateCall(IssueWord, {II->getArgOperand(0),
                             B.getInt32(Machine->RawOpcode), Word});
    SmallVector<Value *> Fields(II->args());
    II->eraseFromParent();
    for (Value *Field : Fields)
      RecursivelyDeleteTriviallyDeadInstructions(Field);
  }
  return true;
}
} // namespace

namespace {
class RISCVTensixIssueWordFold : public FunctionPass {
public:
  static char ID;
  RISCVTensixIssueWordFold() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "RISC-V Tensix issue-word fold";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    // Only Tensix issue calls, their scalar field expressions and the
    // branches choosing a field word change; no alloca or memory access.
    AU.addPreserved<AAResultsWrapperPass>();
    AU.addPreserved<GCModuleInfo>();
    AU.addPreserved<StackProtector>();
  }
  bool runOnFunction(Function &F) override {
    auto &TM = getAnalysis<TargetPassConfig>().getTM<RISCVTargetMachine>();
    auto Changed = foldFunction(F, TM);
    if (!Changed)
      reportFatalUsageError(Twine(toString(Changed.takeError())));
    return *Changed;
  }
};
} // namespace

char RISCVTensixIssueWordFold::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVTensixIssueWordFold, DEBUG_TYPE,
                      "RISC-V Tensix issue-word fold", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVTensixIssueWordFold, DEBUG_TYPE,
                    "RISC-V Tensix issue-word fold", false, false)

FunctionPass *llvm::createRISCVTensixIssueWordFoldPass() {
  return new RISCVTensixIssueWordFold();
}

PreservedAnalyses RISCVTensixIssueWordFoldPass::run(Function &F,
                                                   FunctionAnalysisManager &) {
  auto Changed = foldFunction(F, *TM);
  if (!Changed)
    reportFatalUsageError(Twine(toString(Changed.takeError())));
  if (!*Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}
