//===-- RISCVTensix.h - Public Tensix target contracts -----------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef LLVM_TARGET_RISCV_RISCVTENSIX_H
#define LLVM_TARGET_RISCV_RISCVTENSIX_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <optional>
#include <variant>

namespace llvm {
class Module;
class Triple;
namespace RISCV {

// Logical instruction contracts deliberately expose no bit positions or masks.
struct TensixInstructionInfo {
  StringTable::Offset Name;
  unsigned IntrinsicID;
  unsigned PortIntrinsicID;
  unsigned MopIntrinsicID;
  uint8_t NumFields;
  StringRef getName() const;
};
struct TensixInstructionField {
  unsigned IntrinsicID;
  uint8_t Index;
  StringTable::Offset Name;
  uint32_t MaxValue;
  StringRef getName() const;
};

const TensixInstructionInfo *getTensixInstructionByName(StringRef Name);
const TensixInstructionInfo *getTensixInstructionByIntrinsic(Intrinsic::ID ID);
const TensixInstructionField *
getTensixInstructionField(const TensixInstructionInfo &Info, unsigned Index);
bool isTensixIntrinsic(Intrinsic::ID ID);

// Non-encoding SFPU facts shared by the compiler's explicit-object binder and
// native ingress. Logical fields select the actual native descriptor; an
// intrinsic ID alone does not determine all condition/configuration effects.
enum class TensixBoundValueKind : uint8_t {
  Constant,
  UnboundRegister,
  DynamicScalar,
};
struct TensixBoundValue {
  TensixBoundValueKind Kind;
  int64_t Constant = 0;
};
enum class TensixBoundArgumentRole : uint8_t {
  WriteLReg,
  ReadLReg,
  ReadRegister,
  OldDestination,
  FixedGroupRead,
  FixedGroupWrite,
  FixedGroupClobber,
  WriteCReg,
  LogicalImmediate,
  DstOffset,
};
enum class TensixSFPUAccess : uint8_t { Read, Write, ReadWrite, Clobber };
struct TensixBoundRegisterArgument {
  unsigned Argument;
  TensixBoundArgumentRole Role;
  SmallVector<uint32_t, 16> AllowedRegisters;
  std::optional<uint32_t> FixedRegister;
};
enum class TensixBoundConstraintKind : uint8_t {
  SameLocation,
  DistinctLocation,
  EarlyClobber,
};
struct TensixBoundRegisterConstraint {
  TensixBoundConstraintKind Kind;
  unsigned FirstArgument;
  unsigned SecondArgument;
};
enum class TensixSFPUState : uint8_t {
  CC,
  CCStack,
  Configuration,
  Dst,
  Issue,
};
struct TensixBoundArgumentRef {
  unsigned Argument;
};
struct TensixFixedRegisterRef {
  uint32_t Number;
};
using TensixSFPUResource =
    std::variant<TensixBoundArgumentRef, TensixFixedRegisterRef, TensixSFPUState>;
struct TensixSFPUArchitecturalEffect {
  TensixSFPUResource Resource;
  TensixSFPUAccess Access;
};

class TensixBoundSFPUContract final {
public:
  Intrinsic::ID getIntrinsicID() const { return ID; }
  unsigned getArgumentCount() const { return Roles.size(); }
  ArrayRef<TensixBoundArgumentRole> getArgumentRoles() const { return Roles; }
  ArrayRef<TensixBoundRegisterArgument> getRegisterArguments() const {
    return Registers;
  }
  ArrayRef<TensixBoundRegisterConstraint> getRegisterConstraints() const {
    return Constraints;
  }
  ArrayRef<TensixSFPUArchitecturalEffect> getArchitecturalEffects() const {
    return Effects;
  }

private:
  friend Expected<TensixBoundSFPUContract>
  getTensixBoundSFPUContract(Intrinsic::ID, ArrayRef<TensixBoundValue>);
  TensixBoundSFPUContract() = default;
  Intrinsic::ID ID = Intrinsic::not_intrinsic;
  SmallVector<TensixBoundArgumentRole, 12> Roles;
  SmallVector<TensixBoundRegisterArgument, 12> Registers;
  SmallVector<TensixBoundRegisterConstraint, 8> Constraints;
  SmallVector<TensixSFPUArchitecturalEffect, 16> Effects;
};

// The query permits explicitly unbound register arguments but requires actual
// logical fields. It never chooses registers, constructs IR, or grants program
// admission. DynamicScalar is allowed only at a dynamic Dst-offset position.
Expected<TensixBoundSFPUContract>
getTensixBoundSFPUContract(Intrinsic::ID ID,
                         ArrayRef<TensixBoundValue> Arguments);
// Complete bound ingress additionally rejects every UnboundRegister.
Error verifyTensixBoundSFPUOperands(Intrinsic::ID ID,
                                  ArrayRef<TensixBoundValue> Arguments);
// Architectural numbers use the formal ABI, not internal MC register IDs.
Expected<bool> tensixSFPURegistersOverlap(uint32_t First, uint32_t Second);

// Check width and Tensix feature compatibility without constructing a target
// subtarget whose CLI diagnostics terminate the process.
Expected<bool> verifyTensixTargetFeatures(const Triple &TT, StringRef CPU,
                                        StringRef Features);

// Recoverable preflight for compiler library clients. Function target-features
// must describe the same feature set used to construct the TargetMachine.
// Checks LLVM IR validity and the target's logical field, issue, feature,
// definedness, carrier ABI and condition-stack contracts before code generation.
Error verifyTensixModule(Module &M);

} // namespace RISCV
} // namespace llvm
#endif
