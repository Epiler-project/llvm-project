//===-- RISCVTensixHazards.cpp ---------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixReplay.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
unsigned afterIssue(const MachineInstr &MI, const TargetRegisterInfo &TRI,
                    unsigned Pending) {
  // Automatic SFPU records contain only the verified one-cycle arithmetic
  // subset. Their exact physical effects and length are rechecked before this
  // final hazard analysis; explicit replay retains its conservative fence.
  if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay &&
      MI.getOperand(0).getImm() == 0)
    return 0;
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
  if (MI.getOpcode() == RISCV::TTSFPLUT)
    return lregBit(MI.getOperand(0), TRI) << 8;
  if (MI.getOpcode() == RISCV::TTSFPSWAP ||
      MI.getOpcode() == RISCV::TTSFPSHFT2)
    return SFPUCooldown;
  return 0;
}
unsigned afterDstIssue(const MachineInstr &MI, unsigned Cycles) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPUReplay &&
      MI.getOperand(0).getImm() == 0) {
    unsigned Length = MI.getOperand(2).getImm();
    return Cycles > Length ? Cycles - Length : 0;
  }
  if (isReplayExecution(MI))
    return 3;
  const auto *Ordinary = RISCV::getTensixMachineInfo(MI.getOpcode());
  if (!Ordinary)
    if (const auto *Port = RISCV::getTensixMachineInfoByPort(MI.getOpcode()))
      if (MI.getOperand(3).getImm() == 0)
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
          MI.getOpcode() == RISCV::TTMOP ||
          MI.getOpcode() == RISCV::TTMOP_CFG ||
          MI.getOpcode() == RISCV::PseudoTTMOPClear ||
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
    if (!Repair)
      if (Error E = verifyTensixReplay(MF))
        return Fail(toString(std::move(E)));
    bool Uses = MF.getInfo<RISCVMachineFunctionInfo>()->usesTensixSFPU();
    for (const auto &BB : MF)
      for (const auto &MI : BB)
        Uses |= isSFPU(MI);
    if (!Uses)
      return false;
    auto &ST = MF.getSubtarget<RISCVSubtarget>();
    const auto &TRI = *ST.getRegisterInfo();
    const auto &TII = *ST.getInstrInfo();
    for (const auto &BB : MF)
      for (const auto &MI : BB) {
        if (MI.isCall())
          return Fail("machine call has no verified Tensix SFPU preservation ABI");
        if (MI.getOpcode() == RISCV::TTSETC16 &&
            MI.getOperand(1).getImm() == 0 && MI.getOperand(0).getImm() != 0)
          return Fail("Tensix SFPU fused StateID issue is not implemented");
        // The IR verifier proves StateID legality before port field constants
        // become scalar GPRs. The port pseudo retains its full state effects.
        for (const auto &MO : MI.operands())
          if (MO.isReg() && MO.getReg().isVirtual() &&
              RISCV::SFPRReadRegClass.hasSubClassEq(
                  MF.getRegInfo().getRegClass(MO.getReg())))
            return Fail("unallocated SFPU register reached final legality");
      }
    if (Error E = verifyCC(MF))
      return Fail(toString(std::move(E)));
    bool Modified = false;
    if (Repair && ST.hasVendorXTTTensixBH() &&
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
          Outgoing = afterIssue(MI, TRI, Outgoing);
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
        bool ReadsPendingL0 = readsL0ForConfig(MI) && (Pending & 1u);
        bool ReadsPendingDest = isIAdd(MI) &&
            (Pending & lregBit(MI.getOperand(0), TRI));
        bool ReadsPendingShift = MI.getOpcode() == RISCV::TTSFPSHFT &&
            (Pending & lregBit(MI.getOperand(0), TRI));
        bool ReadsPendingSwap = MI.getOpcode() == RISCV::TTSFPSWAP &&
            MI.getOperand(4).getImm() != 0 &&
            (Pending & (lregBit(MI.getOperand(0), TRI) |
                        lregBit(MI.getOperand(1), TRI)));
        bool ReadsPendingShuffle = MI.getOpcode() == RISCV::TTSFPSHFT2 &&
            (Pending & readMask(MI, TRI));
        bool ReadsPendingLUT = isSFPU(MI) &&
            ((Pending >> 8) & readMask(MI, TRI));
        bool ReplayBoundary = isReplayExecution(MI) && Pending;
        bool PipelineGap = (Pending & SFPUCooldown) && isSFPU(MI) &&
                           MI.getOpcode() != RISCV::TTSFPNOP;
        if (ReadsPendingL0 || ReadsPendingDest || ReadsPendingShift ||
            ReadsPendingSwap || ReadsPendingShuffle || ReadsPendingLUT ||
            ReplayBoundary || PipelineGap) {
          if (!Repair)
            return Fail("unresolved Tensix SFPU MAD to IADD destination hazard");
          BuildMI(BB, It, MI.getDebugLoc(), TII.get(RISCV::TTSFPNOP));
          Pending = 0;
          DstPending = DstPending ? DstPending - 1 : 0;
          Modified = true;
        }
        if ((MI.getOpcode() == RISCV::TTSFPLOAD ||
             MI.getOpcode() == RISCV::PseudoTTSFPLOAD) && DstPending) {
          if (!Repair)
            return Fail("unresolved Tensix matrix Dst write to SFPLOAD hazard");
          while (DstPending) {
            BuildMI(BB, It, MI.getDebugLoc(), TII.get(RISCV::TTSFPNOP));
            --DstPending;
          }
          Pending = 0;
          Modified = true;
        }
        Pending = afterIssue(MI, TRI, Pending);
        DstPending = afterDstIssue(MI, DstPending);
      }
    }
    return Modified;
  }
};
} // namespace
char RISCVTensixHazards::ID = 0;
INITIALIZE_PASS(RISCVTensixHazards, "riscv-tensix-hazards",
                "Repair and verify Tensix SFPU machine hazards", false, false)
FunctionPass *llvm::createRISCVTensixHazardsPass(bool Repair) {
  return new RISCVTensixHazards(Repair);
}
