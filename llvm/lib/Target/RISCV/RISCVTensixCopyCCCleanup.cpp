//===-- RISCVTensixCopyCCCleanup.cpp ---------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace {

// Removing an issue from an explicitly recorded stream changes its length.
// Keep the entire function unchanged when an external/explicit owner could
// observe that issue count, rather than reconstructing recording ownership.
bool hasExternalIssueOwner(const MachineFunction &MF) {
  for (const auto &MBB : MF)
    for (const auto &MI : MBB)
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

// Do not erase a copy carrying an extra clobber, memory effect or instruction
// reference. Such a node needs a separate proof beyond its opcode semantics.
bool hasOnlyDeclaredEffects(const MachineInstr &MI) {
  if (MI.isBundled() || !MI.memoperands_empty() || MI.peekDebugInstrNum())
    return false;
  const MCInstrDesc &Desc = MI.getDesc();
  if (MI.getNumOperands() != Desc.getNumOperands() +
                                 Desc.implicit_uses().size() +
                                 Desc.implicit_defs().size())
    return false;
  for (const MachineOperand &MO : MI.implicit_operands()) {
    if (!MO.isReg() || MO.getSubReg())
      return false;
    if (MO.isDef() ? !is_contained(Desc.implicit_defs(), MO.getReg())
                   : !is_contained(Desc.implicit_uses(), MO.getReg()))
      return false;
  }
  return true;
}

bool isPhysicalSelfCopy(const MachineInstr &MI) {
  unsigned Source;
  switch (MI.getOpcode()) {
  case RISCV::TTSFPMOV:
    Source = 2;
    break;
  case RISCV::TTSFPMOVAll:
    Source = 1;
    break;
  default:
    return false;
  }
  if (!hasOnlyDeclaredEffects(MI))
    return false;
  const MachineOperand &Def = MI.getOperand(0);
  if (!Def.isReg() || !Def.getReg().isPhysical() || Def.getSubReg() ||
      !RISCV::SFPRRegClass.contains(Def.getReg()))
    return false;
  for (unsigned I = 1; I <= Source; ++I) {
    const MachineOperand &Use = MI.getOperand(I);
    if (!Use.isReg() || Use.getSubReg() || Use.getReg() != Def.getReg())
      return false;
  }
  // Mode 0 writes the same raw bits in enabled lanes, preserving the others.
  // Mode 2 writes those same bits in every lane. Mode 1 negates the sign bit
  // and is deliberately absent above, even if src and dst are identical.
  return true;
}

bool isIdempotentEnable(const MachineInstr &MI) {
  if (MI.getOpcode() != RISCV::TTSFPENCC || !hasOnlyDeclaredEffects(MI) ||
      !MI.getOperand(0).isImm() || !MI.getOperand(1).isImm())
    return false;
  switch (MI.getOperand(1).getImm()) {
  case 0:
  case 2:
  case 8:
  case 10:
    return true;
  default:
    // Modes 1 and 9 complement UseLaneFlagsForLaneEnable. Two executions
    // restore that bit, so replacing them with one would change the mask.
    return false;
  }
}

class RISCVTensixCopyCCCleanup final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixCopyCCCleanup() : MachineFunctionPass(ID) {
    initializeRISCVTensixCopyCCCleanupPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getSubtarget<RISCVSubtarget>().hasVendorXTTTensixBH() ||
        MF.getInfo<RISCVMachineFunctionInfo>()->hasTensixCodegenFailed() ||
        MF.getProperties().hasFailedRegAlloc() || hasExternalIssueOwner(MF))
      return false;

    bool Changed = false;
    for (MachineBasicBlock &MBB : MF) {
      MachineInstr *PreviousEnable = nullptr;
      for (MachineInstr &MI : make_early_inc_range(MBB)) {
        bool Remove = isPhysicalSelfCopy(MI);
        bool Enable = isIdempotentEnable(MI);
        if (Enable && PreviousEnable &&
            MI.getOperand(0).getImm() ==
                PreviousEnable->getOperand(0).getImm() &&
            MI.getOperand(1).getImm() ==
                PreviousEnable->getOperand(1).getImm())
          Remove = true;

        if (Remove) {
          // A removed physical self-def no longer starts a fresh live range.
          // Conservatively discard kill markers on all of its register uses;
          // this also preserves implicit CC/issue dependencies after deletion.
          for (const MachineOperand &MO : MI.operands())
            if (MO.isReg() && MO.isUse())
              MF.getRegInfo().clearKillFlags(MO.getReg());
          MI.eraseFromParent();
          Changed = true;
          continue;
        }
        // Only adjacency in the surviving block is used. No state is carried
        // through another instruction, scalar branch, join or backedge.
        PreviousEnable = Enable ? &MI : nullptr;
      }
    }
    // The pipeline must run SFPU hazard repair after this pass: a removed
    // no-op may have supplied issue spacing to an otherwise dependent pair.
    return Changed;
  }
};

} // namespace

char RISCVTensixCopyCCCleanup::ID = 0;
INITIALIZE_PASS(RISCVTensixCopyCCCleanup, "riscv-tensix-copy-cc-cleanup",
                "Remove redundant Tensix SFPU copies and condition enables",
                false, false)

FunctionPass *llvm::createRISCVTensixCopyCCCleanupPass() {
  return new RISCVTensixCopyCCCleanup();
}
