//===-- RISCVTensixExplicitReplay.cpp ------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVTensixReplay.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

namespace {

void retainLivePhysicalRegister(MachineFunction &MF, MCRegister Reg) {
  auto &MRI = MF.getRegInfo();
  MRI.clearKillFlags(Reg);
  for (MachineOperand &MO : MRI.def_operands(Reg))
    MO.setIsDead(false);
}

void retainLivePhysicalEffects(MachineFunction &MF,
                               const TensixSFPUReplayEffects &Effects) {
  for (MCRegister Reg : Effects.Uses)
    retainLivePhysicalRegister(MF, Reg);
  for (MCRegister Reg : Effects.Defs)
    retainLivePhysicalRegister(MF, Reg);
}

void normalizeControl(MachineInstr &MI, const RISCVInstrInfo &TII,
                      const TensixSFPUReplayEffects *Effects) {
  if (MI.getOpcode() == RISCV::PseudoTTExplicitSFPUReplay)
    return;
  assert(MI.getOpcode() == RISCV::TTREPLAY && "requires direct local control");
  const MCInstrDesc &Desc = TII.get(RISCV::PseudoTTExplicitSFPUReplay);
  auto Builder = BuildMI(*MI.getParent(), MI, MI.getDebugLoc(), Desc);
  for (unsigned I = 0; I != 4; ++I)
    Builder.addImm(MI.getOperand(I).getImm());
  Builder.cloneMemRefs(MI).setMIFlags(MI.getFlags());
  if (Effects) {
    for (MCRegister Reg : Effects->Uses)
      if (!is_contained(Desc.implicit_uses(), Reg.id()))
        Builder.addReg(Reg, RegState::Implicit);
    for (MCRegister Reg : Effects->Defs)
      if (!is_contained(Desc.implicit_defs(), Reg.id()))
        Builder.addReg(Reg, RegState::Define | RegState::Implicit);
  }
  MI.eraseFromParent();
}

void normalizeRecordWord(MachineInstr &MI, const TensixRecordedWord &Word,
                         const RISCVInstrInfo &TII) {
  if (MI.getOpcode() == RISCV::PseudoTTSFPURecordWord ||
      MI.getOpcode() == RISCV::PseudoTTSFPUDstRecordWord)
    return;
  insertTensixRecordedPayload(MI, Word, TII, std::nullopt)
      .setMIFlags(MI.getFlags());
  MI.eraseFromParent();
}

class RISCVTensixExplicitReplay final : public MachineFunctionPass {
public:
  static char ID;
  RISCVTensixExplicitReplay() : MachineFunctionPass(ID) {
    initializeRISCVTensixExplicitReplayPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *Info = MF.getInfo<RISCVMachineFunctionInfo>();
    if (!Info->usesBoundTensixSFPU() || Info->hasTensixCodegenFailed())
      return false;
    auto Analysis = analyzeTensixExplicitReplay(MF);
    if (!Analysis) {
      Info->setTensixCodegenFailed();
      MF.getFunction().getContext().diagnose(DiagnosticInfoUnsupported(
          MF.getFunction(), toString(Analysis.takeError())));
      return false;
    }
    if (Analysis->Records.empty() && Analysis->ExecutionWords.empty() &&
        Analysis->MOPExecutions.empty())
      return false;
    bool NeedsNormalization =
        any_of(Analysis->Records,
               [](const auto &Record) {
                 return Record.Header->getOpcode() !=
                            RISCV::PseudoTTExplicitSFPUReplay ||
                        (!Record.ExecuteWhileLoading &&
                         any_of(Record.Body, [](const MachineInstr *MI) {
                           return MI->getOpcode() != RISCV::PseudoTTSFPURecordWord &&
                                  MI->getOpcode() != RISCV::PseudoTTSFPUDstRecordWord;
                         }));
               }) ||
        any_of(Analysis->ExecutionWords, [](const auto &Execution) {
          return Execution.first->getOpcode() !=
                 RISCV::PseudoTTExplicitSFPUReplay;
        }) || any_of(Analysis->MOPExecutions, [](const auto &Execution) {
          if (Error E = verifyTensixMOPExecutionEffects(*Execution.first,
                                                       Execution.second)) {
            consumeError(std::move(E));
            return true;
          }
          return false;
        });
    if (!NeedsNormalization)
      return false;
    const auto &TII = *MF.getSubtarget<RISCVSubtarget>().getInstrInfo();
    bool Changed = false;
    // Analysis of every actual incoming path must succeed before mutation.
    // Recording stores instruction fields; execution reads current registers.
    for (const auto &Record : Analysis->Records) {
      if (!Record.ExecuteWhileLoading)
        for (auto [Original, Word] : zip(Record.Body, Record.Words)) {
          if (Original->getOpcode() == RISCV::PseudoTTSFPURecordWord ||
              Original->getOpcode() == RISCV::PseudoTTSFPUDstRecordWord)
            continue;
          // Removing a falsely executing def can expose an earlier value.
          // Old last-use/dead annotations cannot survive that correction,
          // including a recording that is never subsequently replayed.
          // Dst/config/CC dependencies are implicit in the native descriptor,
          // not encoded operand fields. Removing their false record-time
          // effects must also invalidate any stale liveness flags.
          for (const MachineOperand &Operand : Original->operands())
            if (Operand.isReg() && Operand.getReg().isPhysical())
              retainLivePhysicalRegister(MF, Operand.getReg().asMCReg());
          auto &MI = *const_cast<MachineInstr *>(Original);
          Changed = true;
          normalizeRecordWord(MI, Word, TII);
        }
      auto &Header = *const_cast<MachineInstr *>(Record.Header);
      Changed |= Header.getOpcode() != RISCV::PseudoTTExplicitSFPUReplay;
      normalizeControl(Header, TII, nullptr);
    }
    for (const auto &Execution : Analysis->ExecutionWords) {
      auto &MI = *const_cast<MachineInstr *>(Execution.first);
      const auto &Effects = Analysis->ExecutionEffects.find(&MI)->second;
      retainLivePhysicalEffects(MF, Effects);
      Changed |= MI.getOpcode() != RISCV::PseudoTTExplicitSFPUReplay;
      normalizeControl(MI, TII, &Effects);
    }
    for (const auto &[Instruction, Summary] : Analysis->MOPExecutions) {
      auto *Original = const_cast<MachineInstr *>(Instruction);
      if (Error E = verifyTensixMOPExecutionEffects(*Original, Summary))
        consumeError(std::move(E));
      else
        continue;
      retainLivePhysicalEffects(MF, Summary.Physical);
      auto Builder = BuildMI(*Original->getParent(), *Original,
                             Original->getDebugLoc(), Original->getDesc());
      for (const auto &Operand : Original->explicit_operands())
        Builder.add(Operand);
      Builder.cloneMemRefs(*Original).setMIFlags(Original->getFlags());
      const auto &Desc = Original->getDesc();
      for (MCRegister Reg : Summary.Physical.Uses)
        if (!is_contained(Desc.implicit_uses(), Reg.id()))
          Builder.addReg(Reg, RegState::Implicit);
      for (MCRegister Reg : Summary.Physical.Defs)
        if (!is_contained(Desc.implicit_defs(), Reg.id()))
          Builder.addReg(Reg, RegState::Define | RegState::Implicit);
      Original->eraseFromParent();
      Changed = true;
    }
    return Changed;
  }
};
} // namespace

char RISCVTensixExplicitReplay::ID = 0;
INITIALIZE_PASS(RISCVTensixExplicitReplay, "riscv-tensix-explicit-replay",
                "Normalize authored bound Tensix SFPU recordings", false, false)
FunctionPass *llvm::createRISCVTensixExplicitReplayPass() {
  return new RISCVTensixExplicitReplay();
}
