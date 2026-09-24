//===-- RISCVTensixIRVerification.h -----------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_LIB_TARGET_RISCV_RISCVTENSIXIRVERIFICATION_H
#define LLVM_LIB_TARGET_RISCV_RISCVTENSIXIRVERIFICATION_H

#include "llvm/Support/Error.h"
#include "llvm/IR/Intrinsics.h"
#include <cstdint>

namespace llvm {
class Function;
class ScalarEvolution;
class AssumptionCache;
class DominatorTree;
bool isTensixSFPUIntrinsic(Intrinsic::ID ID);

struct TensixSFPUFunctionFacts {
  bool UsesSFPU = false;
  uint8_t ExplicitFixedLRegs = 0;
};

// Run on each Tensix function before either selector, including at -O0.
// Selection/ABI integration consumes these facts; no string metadata is trusted.
Expected<TensixSFPUFunctionFacts>
verifyTensixSFPUFunction(Function &F, ScalarEvolution &SE,
                          AssumptionCache &AC, DominatorTree &DT);
} // namespace llvm
#endif
