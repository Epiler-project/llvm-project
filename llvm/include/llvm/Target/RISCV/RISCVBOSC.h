//===-- RISCVBOSC.h - BOSC compiler ingress ---------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_TARGET_RISCV_RISCVBOSC_H
#define LLVM_TARGET_RISCV_RISCVBOSC_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
class TargetMachine;
namespace RISCV {

// Recoverable IR preflight for compiler library clients. The same verifier
// receives legacy/codegen ingress. The actual target machine supplies feature
// and vector geometry facts; no caller-reconstructed register plan is accepted.
// Per-function target CPU/features must be absent or match that machine.
// TM's CPU/features and global vector bounds must already be valid; function
// vector bounds are received recoverably against its actual ISA requirements.
// Successful IR reception does not guarantee register pressure or machine
// legality; code generation reports those failures through LLVM diagnostics.
Error verifyBOSCModule(Module &M, TargetMachine &TM);

} // namespace RISCV
} // namespace llvm
#endif
