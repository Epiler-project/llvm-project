//===-- RISCVTensixReplaySelection.cpp --------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCVTensixReplaySelection.h"
#include "RISCVTensixReplay.h"
#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include <optional>

using namespace llvm;
static cl::opt<bool> EnableTensixReplaySelection(
    "riscv-tensix-enable-replay-selection", cl::init(true), cl::Hidden,
    cl::desc("Compress existing repeated Tensix machine sequences"));

namespace {
// These issue operations affect only typed Tensix arithmetic/address state.
// No memory transfer, completion, semaphore, predicate, macro, port or replay
// control is admitted. A candidate is uninterrupted native issue, so replacing
// it does not move an operation across any scalar or device effect boundary.
bool isCandidate(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::TTSETRWC:
  case RISCV::TTINCRWC:
  case RISCV::TTSETADC:
  case RISCV::TTSETADCXX:
  case RISCV::TTSETADCXY:
  case RISCV::TTSETADCZW:
  case RISCV::TTADDRCRXY:
  case RISCV::TTADDRCRZW:
  case RISCV::TTINCADCXY:
  case RISCV::TTINCADCZW:
  case RISCV::TTSETDVALID:
  case RISCV::TTCLEARDVALID:
  case RISCV::TTZEROACC:
  case RISCV::TTZEROSRC:
  case RISCV::TTMOVA2D:
  case RISCV::TTMOVB2D:
  case RISCV::TTMOVB2A:
  case RISCV::TTMOVD2A:
  case RISCV::TTMOVD2B:
  case RISCV::TTMOVDBGA2D:
  case RISCV::TTMOVDBGB2D:
  case RISCV::TTELWADD:
  case RISCV::TTELWMUL:
  case RISCV::TTELWSUB:
  case RISCV::TTMVMUL:
  case RISCV::TTGAPOOL:
  case RISCV::TTGMPOOL:
  case RISCV::TTSHIFTXA:
  case RISCV::TTSHIFTXB:
  case RISCV::TTTRNSPSRCB:
  case RISCV::TTNOP:
    break;
  default:
    return false;
  }
  // The opcode proof covers only its declared effects. In particular, a
  // bundled header does not account for the issues hidden inside its bundle.
  const MCInstrDesc &Desc = MI.getDesc();
  if (MI.isBundled() || MI.peekDebugInstrNum() ||
      MI.getNumExplicitOperands() != Desc.getNumOperands() ||
      MI.getNumOperands() != Desc.getNumOperands() +
                                 Desc.implicit_uses().size() +
                                 Desc.implicit_defs().size())
    return false;
  for (unsigned I = 0; I != Desc.getNumOperands(); ++I)
    if (!MI.getOperand(I).isImm())
      return false;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical() || MO.getSubReg() ||
        MO.isUndef() || MO.isEarlyClobber() || MO.isInternalRead())
      return false;
    if (MO.isDef() ? !is_contained(Desc.implicit_defs(), MO.getReg().id())
                   : !is_contained(Desc.implicit_uses(), MO.getReg().id()))
      return false;
  }
  for (MCPhysReg Reg : Desc.implicit_uses())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isUse() && MO.getReg() == Reg;
        }) != 1)
      return false;
  for (MCPhysReg Reg : Desc.implicit_defs())
    if (count_if(MI.implicit_operands(), [Reg](const MachineOperand &MO) {
          return MO.isDef() && MO.getReg() == Reg;
        }) != 1)
      return false;
  // Ordinary intrinsic lowering carries volatile memory references. Retain
  // them on the execute replacement rather than treating them as extra state.
  return true;
}

bool equalInstruction(const MachineInstr &A, const MachineInstr &B) {
  if (A.getOpcode() != B.getOpcode() ||
      A.getNumExplicitOperands() != B.getNumExplicitOperands())
    return false;
  for (unsigned I = 0; I != A.getNumExplicitOperands(); ++I) {
    const MachineOperand &LHS = A.getOperand(I), &RHS = B.getOperand(I);
    if (LHS.getType() != RHS.getType())
      return false;
    if (LHS.isImm()) {
      if (LHS.getImm() != RHS.getImm())
        return false;
    } else if (!LHS.isReg() || LHS.getReg() != RHS.getReg() ||
               LHS.isDef() != RHS.isDef() ||
               LHS.getSubReg() != RHS.getSubReg()) {
      return false;
    }
  }
  // Both whitelists separately prove that every implicit effect is the
  // exact declared physical use/def. Kill/dead annotations are allocation
  // bookkeeping, not instruction bits or permissions to change the result.
  return true;
}

struct Candidate {
  SmallVector<MachineInstr *> Instructions;
  SmallVector<unsigned> Occurrences;
  unsigned Length = 0;
  unsigned SavedBytes = 0;
  bool SFPU = false;
};

void consider(ArrayRef<MachineInstr *> Run, Candidate &Best, bool SFPU) {
  // Bound compile time independently of the replay capacity. Larger straight
  // line regions remain correct without optimization; no truncation changes
  // their semantics and no source iteration is ever synthesized.
  if (Run.size() > 256)
    return;
  for (unsigned Start = 0; Start + 6 <= Run.size(); ++Start) {
    for (unsigned Length = 3; Length <= 32 &&
         Start + 2 * Length <= Run.size(); ++Length) {
      // A sequence of spacing NOPs alone is not the arithmetic optimization
      // promised by this pass and is deliberately left unchanged.
      if (SFPU && !any_of(Run.slice(Start, Length), [](const MachineInstr *MI) {
            return MI->getDesc().getNumDefs() != 0;
          }))
        continue;
      SmallVector<unsigned> Occurrences{Start};
      for (unsigned Next = Start + Length; Next + Length <= Run.size();) {
        bool Equal = true;
        for (unsigned I = 0; I != Length && Equal; ++I)
          Equal = equalInstruction(*Run[Start + I], *Run[Next + I]);
        if (Equal) {
          Occurrences.push_back(Next);
          Next += Length;
        } else {
          ++Next;
        }
      }
      // The first sequence stays in place and gains one record+execute word;
      // each subsequent sequence becomes one execute word. SFPU execution
      // additionally has two explicit fences; charge them before accepting a
      // candidate. End is zero bytes and no source iteration is manufactured.
      unsigned Count = Occurrences.size();
      if (Count < 2)
        continue;
      unsigned OldWords = Count * Length;
      unsigned NewWords = Length + Count + (SFPU ? 2 * (Count - 1) : 0);
      if (OldWords <= NewWords || 4 * (OldWords - NewWords) <= Best.SavedBytes)
        continue;
      Best.Instructions.assign(Run.begin(), Run.end());
      Best.Occurrences = std::move(Occurrences);
      Best.Length = Length;
      Best.SavedBytes = 4 * (OldWords - NewWords);
      Best.SFPU = SFPU;
    }
  }
}
} // namespace

bool llvm::selectTensixReplay(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!EnableTensixReplaySelection || !ST.hasVendorXTTTensixBH() ||
      MF.getInfo<RISCVMachineFunctionInfo>()->hasTensixCodegenFailed() ||
      MF.getProperties().hasFailedRegAlloc())
    return false;
  bool HasSFPU = MF.getInfo<RISCVMachineFunctionInfo>()->usesTensixSFPU();
  // Initial ownership policy allocates bank zero only if the function has no
  // explicit or implicit replay/MOP owner. In particular a call cannot silently
  // invalidate a recording between its first and subsequent occurrences.
  for (const auto &BB : MF)
    for (const auto &MI : BB.instrs()) {
      if (any_of(MI.operands(), [](const MachineOperand &MO) {
            return MO.isReg() && MO.getReg().isVirtual();
          }))
        return false;
      const auto *Port = RISCV::getTensixMachineInfoByPort(MI.getOpcode());
      HasSFPU |= (RISCV::getTensixEncoding(MI.getOpcode()) &&
                  !RISCV::getTensixMachineInfo(MI.getOpcode())) ||
                 MI.getOpcode() == RISCV::PseudoTTSFPLOAD ||
                 MI.getOpcode() == RISCV::PseudoTTSFPSTORE;
      if (MI.isCall() || MI.isInlineAsm() || Port ||
          RISCV::getTensixMachineInfoByMop(MI.getOpcode()) ||
          MI.getOpcode() == RISCV::TTREPLAY ||
          MI.getOpcode() == RISCV::PseudoTTSFPUReplay ||
          MI.getOpcode() == RISCV::TTMOP ||
          MI.getOpcode() == RISCV::TTMOP_CFG ||
          MI.getOpcode() == RISCV::PseudoTTMOPClear ||
          MI.getOpcode() == RISCV::PseudoTTReplayRecordEnd)
        return false;
    }
  Candidate Best;
  for (auto &BB : MF) {
    SmallVector<MachineInstr *> Run;
    for (auto &MI : BB) {
      if (HasSFPU ? isTensixSFPUReplayCandidate(MI) : isCandidate(MI)) {
        // Merging a known memory reference with an unannotated instruction
        // drops all references. Keep that boundary so volatile annotations
        // survive and the replacement never understates unknown accesses.
        if (!Run.empty() &&
            Run.back()->memoperands_empty() != MI.memoperands_empty()) {
          consider(Run, Best, HasSFPU);
          Run.clear();
        }
        Run.push_back(&MI);
        continue;
      }
      consider(Run, Best, HasSFPU);
      Run.clear();
    }
    consider(Run, Best, HasSFPU);
  }
  if (!Best.SavedBytes)
    return false;
  const auto &TII = *ST.getInstrInfo();
  MachineInstr *First = Best.Instructions[Best.Occurrences.front()];
  auto &BB = *First->getParent();
  std::optional<TensixSFPUReplayEffects> Effects;
  if (Best.SFPU) {
    SmallVector<const MachineInstr *, 32> Body;
    for (unsigned I = 0; I != Best.Length; ++I)
      Body.push_back(Best.Instructions[Best.Occurrences.front() + I]);
    auto Analyzed = getTensixSFPUReplayEffects(Body);
    if (!Analyzed) {
      consumeError(Analyzed.takeError());
      return false;
    }
    Effects = std::move(*Analyzed);
    // The replacement has the same physical live-in/live-out relation, but
    // an interior last use or dead def no longer sits at the same instruction.
    // Conservatively remove those annotations; never invent a dead result.
    for (MCRegister Reg : Effects->Uses)
      MF.getRegInfo().clearKillFlags(Reg);
    for (MCRegister Reg : Effects->Defs) {
      MF.getRegInfo().clearKillFlags(Reg);
      for (MachineOperand &MO : MF.getRegInfo().def_operands(Reg))
        MO.setIsDead(false);
    }
  }
  unsigned Opcode = Best.SFPU ? RISCV::PseudoTTSFPUReplay : RISCV::TTREPLAY;
  BuildMI(BB, First, First->getDebugLoc(), TII.get(Opcode))
      .addImm(1).addImm(1).addImm(Best.Length).addImm(0);
  MachineInstr *Last = Best.Instructions[
      Best.Occurrences.front() + Best.Length - 1];
  BuildMI(BB, std::next(Last->getIterator()), Last->getDebugLoc(),
          TII.get(RISCV::PseudoTTReplayRecordEnd));
  for (unsigned Occurrence : llvm::drop_begin(Best.Occurrences)) {
    MachineInstr *At = Best.Instructions[Occurrence];
    if (Best.SFPU)
      BuildMI(BB, At, At->getDebugLoc(), TII.get(RISCV::TTSFPNOP));
    MachineInstrBuilder Execute =
        BuildMI(BB, At, At->getDebugLoc(), TII.get(Opcode))
            .addImm(0).addImm(0).addImm(Best.Length).addImm(0);
    if (Best.SFPU) {
      const MCInstrDesc &Desc = TII.get(Opcode);
      for (MCRegister Reg : Effects->Uses)
        if (!is_contained(Desc.implicit_uses(), Reg.id()))
          Execute.addReg(Reg, RegState::Implicit);
      for (MCRegister Reg : Effects->Defs)
        if (!is_contained(Desc.implicit_defs(), Reg.id()))
          Execute.addReg(Reg, RegState::Implicit | RegState::Define);
      BuildMI(BB, At, At->getDebugLoc(), TII.get(RISCV::TTSFPNOP));
    } else {
      SmallVector<const MachineInstr *, 32> Replaced;
      for (unsigned I = 0; I != Best.Length; ++I)
        Replaced.push_back(Best.Instructions[Occurrence + I]);
      Execute.cloneMergedMemRefs(Replaced);
    }
    for (unsigned I = 0; I != Best.Length; ++I)
      Best.Instructions[Occurrence + I]->eraseFromParent();
  }
  return true;
}

namespace {
class RISCVTensixReplaySelection final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixReplaySelection() : MachineFunctionPass(ID) {
    initializeRISCVTensixReplaySelectionPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return selectTensixReplay(MF);
  }
};
}
char RISCVTensixReplaySelection::ID = 0;
INITIALIZE_PASS(RISCVTensixReplaySelection, "riscv-tensix-replay-selection",
                "Select existing repeated Tensix sequences for replay", false,
                false)
FunctionPass *llvm::createRISCVTensixReplaySelectionPass() {
  return new RISCVTensixReplaySelection();
}
PreservedAnalyses RISCVTensixReplaySelectionPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &) {
  return selectTensixReplay(MF) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
}
PreservedAnalyses RISCVTensixReplayVerificationPass::run(
    MachineFunction &MF, MachineFunctionAnalysisManager &) {
  if (Error E = verifyTensixReplay(MF)) {
    MF.getInfo<RISCVMachineFunctionInfo>()->setTensixCodegenFailed();
    MF.getFunction().getContext().diagnose(
        DiagnosticInfoUnsupported(MF.getFunction(), toString(std::move(E))));
  }
  return PreservedAnalyses::all();
}
