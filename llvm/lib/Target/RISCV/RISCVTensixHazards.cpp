//===-- RISCVTensixHazards.cpp ---------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "RISCVTensixReplay.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

namespace {
constexpr unsigned SFPUCooldown = 1u << 16;
bool isSFPU(const MachineInstr &MI) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
      MI.getOpcode() == RISCV::PseudoTTSFPSTORE)
    return true;
  return RISCV::getTensixEncoding(MI.getOpcode()) &&
         !RISCV::getTensixMachineInfo(MI.getOpcode());
}
bool readsMatrixDst(const MachineInstr &MI) {
  return MI.getOpcode() == RISCV::TTSFPLOAD ||
         MI.getOpcode() == RISCV::PseudoTTSFPLOAD;
}
bool isIAdd(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::TTSFPIADDCCLT: case RISCV::TTSFPISUBCCLT:
  case RISCV::TTSFPIADD: case RISCV::TTSFPISUB:
  case RISCV::TTSFPIADDCCGE: case RISCV::TTSFPISUBCCGE:
  case RISCV::TTSFPIADDI:
    return true;
  default: return false;
  }
}
bool readsL0ForConfig(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::TTSFPCONFIGC11:
  case RISCV::TTSFPCONFIGC12:
  case RISCV::TTSFPCONFIGC13:
  case RISCV::TTSFPCONFIGC14:
  case RISCV::TTSFPCONFIGLane:
    return true;
  default:
    return false;
  }
}
unsigned lregBit(const MachineOperand &MO, const TargetRegisterInfo &TRI) {
  if (!MO.isReg() || !MO.getReg().isPhysical() ||
      !RISCV::SFPRRegClass.contains(MO.getReg()))
    return 0;
  return 1u << TRI.getEncodingValue(MO.getReg());
}
bool isReplayExecution(const MachineInstr &MI) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
      MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay)
    return MI.getOperand(0).getImm() == 0;
  if (MI.getOpcode() == RISCV::TTREPLAY)
    return MI.getOperand(0).getImm() == 0;
  if (MI.getOpcode() == RISCV::PseudoTTREPLAYPort)
    return MI.getOperand(4).getImm() == 0;
  return MI.getOpcode() == RISCV::TTMOP ||
         MI.getOpcode() == RISCV::PseudoTTMOPPort;
}
unsigned readMask(const MachineInstr &MI, const TargetRegisterInfo &TRI) {
  unsigned Mask = 0;
  for (const auto &MO : MI.operands())
    if (MO.isReg() && MO.isUse())
      Mask |= lregBit(MO, TRI);
  return Mask;
}
bool needsSFPUWait(const MachineInstr &MI, const TargetRegisterInfo &TRI,
                   unsigned Pending) {
  bool ReadsPendingL0 = readsL0ForConfig(MI) && (Pending & 1u);
  bool ReadsPendingDest = isIAdd(MI) &&
      (Pending & lregBit(MI.getOperand(0), TRI));
  bool ReadsPendingShift = MI.getOpcode() == RISCV::TTSFPSHFT &&
      (Pending & lregBit(MI.getOperand(0), TRI));
  bool ReadsPendingSwap = MI.getOpcode() == RISCV::TTSFPSWAP &&
      MI.getOperand(4).getImm() != 0 &&
      (Pending & (lregBit(MI.getOperand(0), TRI) |
                  lregBit(MI.getOperand(1), TRI)));
  bool ReadsPendingShuffle =
      (MI.getOpcode() == RISCV::TTSFPSHFT2 ||
       MI.getOpcode() == RISCV::TTSFPSHFT2Copy4 ||
       MI.getOpcode() == RISCV::TTSFPSHFT2Rotate4) &&
      (Pending & readMask(MI, TRI));
  bool ReadsPendingLUT = isSFPU(MI) &&
      ((Pending >> 8) & readMask(MI, TRI));
  bool ReplayBoundary = isReplayExecution(MI) && Pending;
  bool PipelineGap = (Pending & SFPUCooldown) && isSFPU(MI) &&
                     MI.getOpcode() != RISCV::TTSFPNOP;
  return ReadsPendingL0 || ReadsPendingDest || ReadsPendingShift ||
         ReadsPendingSwap || ReadsPendingShuffle || ReadsPendingLUT ||
         ReplayBoundary || PipelineGap;
}
unsigned afterIssue(const MachineInstr &MI, const TargetRegisterInfo &TRI,
                    unsigned Pending,
                    const TensixSFPUReplayExecutionEffects *ReplayEffects =
                        nullptr) {
  // Reuse the replay verifier's native exit state. A union of physical defs
  // cannot describe the pending MAD/LUT result or a cross-lane cooldown.
  if ((MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
       MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay) &&
      MI.getOperand(0).getImm() == 0) {
    assert(ReplayEffects && "SFPU replay requires verified effects");
    auto It = ReplayEffects->find(&MI);
    assert(It != ReplayEffects->end() && "missing SFPU replay effects");
    return It->second.PendingSFPU;
  }
  // Replay and MOP have finite validated instruction streams, but their final
  // SFPU producer may depend on control values. Fence the SFPU handoff instead
  // of guessing a producer from a source recipe or duplicating iterations.
  if (isReplayExecution(MI))
    return 0xffff;
  if (!isSFPU(MI))
    return Pending;
  // SFPADD uses the MAD datapath with the fixed multiplicand C10.
  if (MI.getOpcode() == RISCV::TTSFPMAD || MI.getOpcode() == RISCV::TTSFPMUL ||
      MI.getOpcode() == RISCV::TTSFPADD)
    return lregBit(MI.getOperand(0), TRI);
  // Unlike MAD's partially tracked dependencies, LUT requires a gap before
  // any next-cycle read of its result.
  if (MI.getOpcode() == RISCV::TTSFPLUTFP32Three)
    return 0xffu << 8;
  if (MI.getOpcode() == RISCV::TTSFPLUT ||
      MI.getOpcode() == RISCV::TTSFPLUTFP32Six)
    return lregBit(MI.getOperand(0), TRI) << 8;
  if (MI.getOpcode() == RISCV::TTSFPSWAP ||
      MI.getOpcode() == RISCV::TTSFPSHFT2 ||
      MI.getOpcode() == RISCV::TTSFPSHFT2Rotate4)
    return SFPUCooldown;
  return 0;
}
unsigned afterDstIssue(const MachineInstr &MI, unsigned Cycles) {
  if ((MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
       MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay) &&
      MI.getOperand(0).getImm() == 0) {
    unsigned Length = MI.getOperand(2).getImm();
    return Cycles > Length ? Cycles - Length : 0;
  }
  if (isReplayExecution(MI))
    return 3;
  const auto *Ordinary = RISCV::getTensixMachineInfo(MI.getOpcode());
  if (!Ordinary)
    if (const auto *Port = RISCV::getTensixMachineInfoByPort(MI.getOpcode()))
      if (MI.getOperand(RISCV::getTensixPortOperandIndex(MI.getDesc()))
              .getImm() == 0)
        Ordinary = Port;
  if (Ordinary && Ordinary->WritesDst)
    return 3;
  if (Ordinary || isSFPU(MI))
    return Cycles ? Cycles - 1 : 0;
  return Cycles;
}

// An explicit recording or an opaque issuer can observe instruction order or
// issue count outside the modeled SFPU register effects. Do not schedule any
// part of such a function, including instructions outside a recorded region.
bool hasExternalIssueOwner(const MachineFunction &MF) {
  for (const auto &BB : MF)
    for (const auto &MI : BB)
      if (MI.isCall() || MI.isInlineAsm() ||
          RISCV::getTensixMachineInfoByPort(MI.getOpcode()) ||
          RISCV::getTensixMachineInfoByMop(MI.getOpcode()) ||
          MI.getOpcode() == RISCV::TTREPLAY ||
          MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
          MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay ||
          MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
          MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord ||
          MI.getOpcode() == RISCV::PseudoTTReplayTemplateBegin ||
          MI.getOpcode() == RISCV::PseudoTTReplayTemplateEnd ||
          MI.getOpcode() == RISCV::PseudoTTReplayTemplateExecute ||
          MI.getOpcode() == RISCV::PseudoTTReplayTemplateMop ||
          MI.getOpcode() == RISCV::PseudoTTReplayTemplateWord ||
          MI.getOpcode() == RISCV::PseudoTTReplayTemplateDstWord ||
          MI.getOpcode() == RISCV::TTMOP ||
          MI.getOpcode() == RISCV::TTMOP_CFG ||
          MI.getOpcode() == RISCV::PseudoTTMOPClear ||
          MI.getOpcode() == RISCV::PseudoTTMOPControlWrite ||
          MI.getOpcode() == RISCV::PseudoTTMOPControlWriteImm ||
          MI.getOpcode() == RISCV::PseudoTTReplayRecordEnd)
        return true;
  return false;
}

// The opcode proof below covers exactly the declared physical operands. An
// extra clobber, memory annotation, subregister or debug instruction reference
// needs its own proof and must not silently inherit this commutation rule.
bool hasDeclaredPhysicalEffects(const MachineInstr &MI) {
  if (MI.isBundled() || !MI.memoperands_empty() || MI.peekDebugInstrNum())
    return false;
  const MCInstrDesc &Desc = MI.getDesc();
  if (MI.getNumOperands() != Desc.getNumOperands() +
                                 Desc.implicit_uses().size() +
                                 Desc.implicit_defs().size())
    return false;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg()) {
      if (!MO.getReg().isPhysical() || MO.getSubReg())
        return false;
    } else if (!MO.isImm()) {
      return false;
    }
  }
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isReg() && MO.isUse() && MO.getReg() == Reg;
        }) != 1)
      return false;
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isReg() && MO.isDef() && MO.getReg() == Reg;
        }) != 1)
      return false;
  return true;
}

bool touchesFixedLReg(const MachineInstr &MI, unsigned Fixed,
                      const TargetRegisterInfo &TRI) {
  return any_of(MI.operands(), [&](const MachineOperand &MO) {
    return lregBit(MO, TRI) & Fixed;
  });
}

bool hasRegisterConflict(const MachineInstr &A, const MachineInstr &B,
                         const TargetRegisterInfo &TRI) {
  for (const MachineOperand &Left : A.operands()) {
    // TT_ISSUE represents the default total issue order. Only the selected
    // opcode/effect proof may commute it; every other implicit dependency is
    // checked just like an explicit physical register, including aliases.
    if (!Left.isReg() || Left.getReg() == RISCV::TT_ISSUE)
      continue;
    for (const MachineOperand &Right : B.operands())
      if (Right.isReg() && (Left.isDef() || Right.isDef()) &&
          TRI.regsOverlap(Left.getReg(), Right.getReg()))
        return true;
  }
  return false;
}

bool scheduleIndependentCopies(MachineFunction &MF,
                               const TargetRegisterInfo &TRI) {
  if (hasExternalIssueOwner(MF))
    return false;
  unsigned Fixed = MF.getInfo<RISCVMachineFunctionInfo>()->getTensixFixedLRegs();
  bool Changed = false;
  for (auto &BB : MF) {
    for (auto It = BB.begin(); It != BB.end(); ++It) {
      MachineInstr &Producer = *It;
      if (Producer.getOpcode() != RISCV::TTSFPMAD &&
          Producer.getOpcode() != RISCV::TTSFPMUL &&
          Producer.getOpcode() != RISCV::TTSFPADD)
        continue;
      auto ConsumerIt = std::next(It);
      if (ConsumerIt == BB.end())
        continue;
      auto CopyIt = std::next(ConsumerIt);
      if (CopyIt == BB.end())
        continue;
      MachineInstr &Consumer = *ConsumerIt;
      MachineInstr &Copy = *CopyIt;
      if ((Consumer.getOpcode() != RISCV::TTSFPIADD &&
           Consumer.getOpcode() != RISCV::TTSFPISUB) ||
          Copy.getOpcode() != RISCV::TTSFPMOVAll ||
          !hasDeclaredPhysicalEffects(Producer) ||
          !hasDeclaredPhysicalEffects(Consumer) ||
          !hasDeclaredPhysicalEffects(Copy) ||
          touchesFixedLReg(Producer, Fixed, TRI) ||
          touchesFixedLReg(Consumer, Fixed, TRI) ||
          touchesFixedLReg(Copy, Fixed, TRI))
        continue;
      unsigned Pending = afterIssue(Producer, TRI, 0);
      if (!(Pending & lregBit(Consumer.getOperand(0), TRI)) ||
          (Pending & readMask(Copy, TRI)) ||
          Copy.getOperand(0).getReg() == Copy.getOperand(1).getReg() ||
          hasRegisterConflict(Consumer, Copy, TRI))
        continue;

      // Blackhole MAD/ADD/MUL need two SFPU cycles; IADD's tied destination
      // read is not covered by its hardware scoreboard. Mode-2 MOV ignores
      // lane enable and has one-cycle latency. Independent P,C,F -> P,F,C
      // therefore fills the gap with an existing useful issue, while leaving
      // P's predicate, C's unchanged CC and all final register values intact.
      // C and F both clear the SFPU pending state and consume one Dst cycle,
      // so their output states agree in either order. SWAP/SHFT2 cooldowns,
      // Dst loads and scalar instructions are deliberately outside this rule.
      BB.splice(ConsumerIt, &BB, CopyIt);
      for (MachineInstr *MI : {&Consumer, &Copy})
        for (MachineOperand &MO : MI->operands())
          if (MO.isReg()) {
            if (MO.isDef())
              MO.setIsDead(false);
            else
              MO.setIsKill(false);
          }
      Changed = true;
    }
  }
  return Changed;
}

Error verifyCC(const MachineFunction &MF) {
  if (MF.empty())
    return Error::success();
  DenseMap<const MachineBasicBlock *, unsigned> Depths;
  SmallVector<const MachineBasicBlock *> Worklist{&MF.front()};
  Depths.try_emplace(&MF.front(), 0);
  while (!Worklist.empty()) {
    const auto *BB = Worklist.pop_back_val();
    unsigned Depth = Depths.lookup(BB);
    for (const MachineInstr &MI : *BB) {
      if (MI.getOpcode() == RISCV::TTSFPPUSHC) {
        if (Depth == 8)
          return createStringError("Tensix SFPU machine CC stack exceeds 8");
        ++Depth;
      } else if (MI.getOpcode() == RISCV::TTSFPPOPC) {
        if (Depth == 0)
          return createStringError("Tensix SFPU machine CC stack underflow");
        --Depth;
      }
      if (MI.isReturn() && Depth)
        return createStringError("Tensix SFPU machine CC stack is not empty at return");
    }
    for (const MachineBasicBlock *Succ : BB->successors()) {
      auto [It, Inserted] = Depths.try_emplace(Succ, Depth);
      if (Inserted)
        Worklist.push_back(Succ);
      else if (It->second != Depth)
        return createStringError("Tensix SFPU machine CC depth disagrees at CFG join or backedge");
    }
  }
  return Error::success();
}
class RISCVTensixHazards : public MachineFunctionPass {
  bool Repair;
public:
  static char ID;
  explicit RISCVTensixHazards(bool Repair = false)
      : MachineFunctionPass(ID), Repair(Repair) {
    initializeRISCVTensixHazardsPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
    if (Info->hasTensixCodegenFailed() || MF.getProperties().hasFailedRegAlloc())
      return false;
    auto Fail = [&](const Twine &Message) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(
          DiagnosticInfoUnsupported(MF.getFunction(), Message));
      return false;
    };
    TensixSFPUReplayExecutionEffects ReplayEffects;
    bool HasAutomaticReplay = any_of(MF, [](const MachineBasicBlock &BB) {
      return any_of(BB, [](const MachineInstr &MI) {
        return MI.getOpcode() == RISCV::PseudoTTSFPUReplay;
      });
    });
    // Normal selection follows repair. A supplied automatic record still
    // needs its exact effect summary before either repair or verification;
    // final verification will reject any record length changed by repair.
    if (!Repair || HasAutomaticReplay)
      if (Error E = verifyTensixReplay(MF, &ReplayEffects))
        return Fail(toString(std::move(E)));
    auto Explicit = analyzeTensixExplicitReplay(MF);
    if (!Explicit)
      return Fail(toString(Explicit.takeError()));
    for (const auto &Entry : Explicit->ExecutionEffects)
      ReplayEffects.try_emplace(Entry.first, Entry.second);
    const auto &MOPExecutions = Explicit->MOPExecutions;
    SmallPtrSet<const MachineInstr *, 32> RecordedBodies;
    SmallPtrSet<const MachineInstr *, 8> ExecutedRecordHeaders;
    for (const auto &Record : Explicit->Records) {
      RecordedBodies.insert(Record.Body.begin(), Record.Body.end());
      if (Record.ExecuteWhileLoading)
        ExecutedRecordHeaders.insert(Record.Header);
    }
    bool Bound = Info->usesBoundTensixSFPU();
    bool Uses = Info->usesTensixSFPU();
    for (const auto &BB : MF)
      for (const auto &MI : BB.instrs())
        Uses |= isSFPU(MI);
    if (!Uses)
      return false;
    auto &ST = MF.getSubtarget<RISCVSubtarget>();
    const auto &TRI = *ST.getRegisterInfo();
    const auto &TII = *ST.getInstrInfo();
    // Record-only payloads are not executed now, but their future issue order
    // has the same hardware latency rules. Reconstruct through the native
    // descriptor, then reuse exactly the normal transfer and gap predicates.
    // The first Dst read's native issue position determines how much incoming
    // Matrix latency the body can absorb. Derive it from actual final words,
    // including authored/legalized NOPs; do not infer it from union effects.
    // All currently admitted words are SFPU issues and none starts a Matrix
    // write, so the existing length-based outgoing Dst transfer remains exact.
    DenseMap<const MachineInstr *, unsigned> DstReadPrefixes;
    auto verifyFutureHazards =
        [&](ArrayRef<TensixRecordedWord> Words) -> Expected<unsigned> {
      TensixReplayHazardState State;
      unsigned DstPrefix = TensixReplaySlotCount;
      for (auto [Index, Word] : enumerate(Words)) {
        auto Rebuilt = createTensixRecordedInstruction(Word, MF);
        if (!Rebuilt)
          return Rebuilt.takeError();
        MachineInstr *Future = *Rebuilt;
        if (readsMatrixDst(*Future))
          DstPrefix = std::min(DstPrefix, unsigned(Index));
        bool NeedsGap = needsTensixReplayHazardGap(*Future, TRI, State);
        State = advanceTensixReplayHazard(*Future, TRI, State);
        MF.deleteMachineInstr(Future);
        if (NeedsGap)
          return createStringError(
              "explicit SFPU replay template requires an authored hazard gap");
      }
      return DstPrefix;
    };
    for (const auto &Record : Explicit->Records) {
      auto Prefix = verifyFutureHazards(Record.Words);
      if (!Prefix)
        return Fail(toString(Prefix.takeError()));
      if (Record.ExecuteWhileLoading)
        DstReadPrefixes.try_emplace(Record.Header, *Prefix);
    }
    // A selected range may combine words from partial overwrites. Checking
    // each original recording alone does not establish the executed order.
    for (const auto &Execution : Explicit->ExecutionWords) {
      auto Prefix = verifyFutureHazards(Execution.second);
      if (!Prefix)
        return Fail(toString(Prefix.takeError()));
      DstReadPrefixes.try_emplace(Execution.first, *Prefix);
    }
    for (const auto &[Instruction, Execution] : MOPExecutions)
      if (Execution.Hazards.apply({}).RequiresGap)
        return Fail("SFPU MOP selected words require an authored internal "
                    "hazard gap across slots or hardware-loop edges");
    for (const auto &BB : MF)
      for (const auto &MI : BB.instrs()) {
        if (MI.isBundled())
          return Fail("bundled instructions have no verified Tensix SFPU issue order");
        if (MI.isInlineAsm())
          return Fail("inline assembly has no verified Tensix SFPU preservation ABI");
        if (MI.isCall())
          return Fail("machine call has no verified Tensix SFPU preservation ABI");
        if (!Bound && MI.getOpcode() == RISCV::TTSETC16 &&
            MI.getOperand(1).getImm() == 0 && MI.getOperand(0).getImm() != 0)
          return Fail("Tensix SFPU fused StateID issue is not implemented");
        // Legacy transformations require state zero, which the IR verifier
        // checks before port fields become scalar GPRs. Bound ingress keeps
        // source-authored state changes; both forms retain full state effects.
        for (const auto &MO : MI.operands())
          if (MO.isReg() && MO.getReg().isVirtual() &&
              RISCV::SFPRReadRegClass.hasSubClassEq(
                  MF.getRegInfo().getRegClass(MO.getReg())))
            return Fail("unallocated SFPU register reached final legality");
      }
    // Stack lifecycle is source-owned for bound operations. Keep the legacy
    // local-stack precondition without imposing it on physical issue streams.
    if (!Bound)
      if (Error E = verifyCC(MF))
        return Fail(toString(std::move(E)));
    bool Modified = false;
    const auto Options = static_cast<const RISCVTargetMachine &>(MF.getTarget())
                             .getTensixOptimizationOptions();
    if (Repair && Options.LatencyScheduling && ST.hasVendorXTTTensixBH() &&
        MF.getTarget().getOptLevel() != CodeGenOptLevel::None &&
        !skipFunction(MF.getFunction()))
      Modified = scheduleIndependentCopies(MF, TRI);
    DenseMap<const MachineBasicBlock *, unsigned> Entry, Exit, DstEntry, DstExit;
    bool Changed;
    do {
      Changed = false;
      for (const auto &BB : MF) {
        unsigned Incoming = 0;
        unsigned DstIncoming = 0;
        for (const auto *Pred : BB.predecessors())
          Incoming |= Exit.lookup(Pred);
        for (const auto *Pred : BB.predecessors())
          DstIncoming = std::max(DstIncoming, DstExit.lookup(Pred));
        Entry[&BB] = Incoming;
        DstEntry[&BB] = DstIncoming;
        unsigned Outgoing = Incoming;
        for (const auto &MI : BB) {
          if (auto MOP = MOPExecutions.find(&MI); MOP != MOPExecutions.end()) {
            auto Next = MOP->second.Hazards.apply({Outgoing, DstIncoming});
            Outgoing = Next.State.SFPU;
            DstIncoming = Next.State.DstCycles;
            continue;
          }
          Outgoing = afterIssue(MI, TRI, Outgoing, &ReplayEffects);
          DstIncoming = afterDstIssue(MI, DstIncoming);
        }
        if (Exit.lookup(&BB) != Outgoing) {
          Exit[&BB] = Outgoing;
          Changed = true;
        }
        if (DstExit.lookup(&BB) != DstIncoming) {
          DstExit[&BB] = DstIncoming;
          Changed = true;
        }
      }
    } while (Changed);
    for (auto &BB : MF) {
      unsigned Pending = Entry.lookup(&BB);
      unsigned DstPending = DstEntry.lookup(&BB);
      for (auto It = BB.begin(); It != BB.end(); ++It) {
        MachineInstr &MI = *It;
        if (auto MOP = MOPExecutions.find(&MI); MOP != MOPExecutions.end()) {
          auto Next = MOP->second.Hazards.apply({Pending, DstPending});
          if (Next.RequiresGap && !Repair)
            return Fail("unresolved native pipeline dependency at SFPU MOP entry");
          // Empty-entry internal hazards were rejected above. Repair only
          // actual incoming dependencies; an outer fence cannot repair an
          // unsafe Replay-to-slot or hardware-loop edge inside the expander.
          while (Next.RequiresGap && (Pending || DstPending)) {
            BuildMI(BB, It, MI.getDebugLoc(), TII.get(RISCV::TTSFPNOP));
            Pending = 0;
            DstPending = DstPending ? DstPending - 1 : 0;
            Modified = true;
            Next = MOP->second.Hazards.apply({Pending, DstPending});
          }
          if (Next.RequiresGap)
            return Fail("SFPU MOP selected words require an authored internal "
                        "hazard gap across slots or hardware-loop edges");
          Pending = Next.State.SFPU;
          DstPending = Next.State.DstCycles;
          continue;
        }
        // The immutable authored body cannot absorb a newly inserted issue.
        // Fence an executed record at its header, before capturing starts.
        bool RecordBoundary = ExecutedRecordHeaders.contains(&MI) && Pending;
        if (needsSFPUWait(MI, TRI, Pending) || RecordBoundary) {
          if (RecordedBodies.contains(&MI))
            return Fail("explicit SFPU replay template requires an authored hazard gap");
          if (!Repair)
            return Fail("unresolved Tensix SFPU MAD to IADD destination hazard");
          BuildMI(BB, It, MI.getDebugLoc(), TII.get(RISCV::TTSFPNOP));
          Pending = 0;
          DstPending = DstPending ? DstPending - 1 : 0;
          Modified = true;
        }
        unsigned DstPrefix = readsMatrixDst(MI) ? 0 : TensixReplaySlotCount;
        if (auto Prefix = DstReadPrefixes.find(&MI);
            Prefix != DstReadPrefixes.end())
          DstPrefix = Prefix->second;
        if (DstPending > DstPrefix) {
          if (RecordedBodies.contains(&MI))
            return Fail("explicit SFPU replay template requires an authored "
                        "Dst hazard gap");
          if (!Repair)
            return Fail("unresolved Tensix matrix Dst write to SFPLOAD hazard");
          while (DstPending > DstPrefix) {
            BuildMI(BB, It, MI.getDebugLoc(), TII.get(RISCV::TTSFPNOP));
            --DstPending;
          }
          Pending = 0;
          Modified = true;
        }
        Pending = afterIssue(MI, TRI, Pending, &ReplayEffects);
        DstPending = afterDstIssue(MI, DstPending);
      }
    }
    return Modified;
  }
};
} // namespace
bool llvm::needsTensixReplayHazardGap(const MachineInstr &MI,
                                     const TargetRegisterInfo &TRI,
                                     TensixReplayHazardState State) {
  return needsSFPUWait(MI, TRI, State.SFPU) ||
         (readsMatrixDst(MI) && State.DstCycles);
}

TensixReplayHazardState llvm::advanceTensixReplayHazard(
    const MachineInstr &MI, const TargetRegisterInfo &TRI,
    TensixReplayHazardState State) {
  return {afterIssue(MI, TRI, State.SFPU),
          afterDstIssue(MI, State.DstCycles)};
}

char RISCVTensixHazards::ID = 0;
INITIALIZE_PASS(RISCVTensixHazards, "riscv-tensix-hazards",
                "Repair and verify Tensix SFPU machine hazards", false, false)
FunctionPass *llvm::createRISCVTensixHazardsPass(bool Repair) {
  return new RISCVTensixHazards(Repair);
}
