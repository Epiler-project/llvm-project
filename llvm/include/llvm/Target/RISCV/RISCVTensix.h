//===-- RISCVTensix.h - Public Tensix target contracts -----------*- C++
//-*-===//
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
// Coverage of an actual selected native form, not an initialization proof.
// StateDependent includes unresolved lane selection, cross-lane reads and
// partial-bit writes. It does not promise preservation of inactive lanes.
enum class TensixSFPURegisterFootprint : uint8_t {
  Unsupported,
  StateDependent,
  WholeRegister,
};
struct TensixBoundRegisterArgument {
  unsigned Argument;
  TensixBoundArgumentRole Role;
  SmallVector<uint32_t, 16> AllowedRegisters;
  std::optional<uint32_t> FixedRegister;
  TensixSFPURegisterFootprint Footprint =
      TensixSFPURegisterFootprint::Unsupported;
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
  PRNG,
};
struct TensixBoundArgumentRef {
  unsigned Argument;
};
struct TensixFixedRegisterRef {
  uint32_t Number;
};
using TensixSFPUResource =
    std::variant<TensixBoundArgumentRef, TensixFixedRegisterRef,
                 TensixSFPUState>;
struct TensixSFPUArchitecturalEffect {
  // Conservative descriptor effects, not exact per-lane read/write coverage.
  TensixSFPUResource Resource;
  TensixSFPUAccess Access;
};

enum class TensixSFPUTransferKind : uint8_t {
  Unsupported,
  Identity,
  ImmediateFull,
  ImmediateUpperHalf,
  ImmediateLowerHalf,
  LaneWise,
  LanePermutation,
};
struct TensixSFPUTransferInput {
  unsigned Argument;
  uint32_t DemandedBits;
};
struct TensixSFPUTransferFacts {
  TensixSFPUTransferKind Kind = TensixSFPUTransferKind::Unsupported;
  // Bits preserved from the old destination in active lanes. Numerical
  // dependencies on the old value belong in Inputs, never in this mask.
  uint32_t DemandedOldBits = 0;
  uint32_t GeneratedBits = 0;
  uint32_t GuaranteedWriteBits = 0;
  bool RequiresLaneEnable = true;
  bool PreservesInactiveLanes = false;
  // CC and ROW_MASK are independent semantic inputs to lane selection.  The
  // native register descriptor currently exposes CC effects only; this pair
  // keeps the unresolved ROW_MASK dependency explicit without inventing a
  // physical LLVM state register.
  bool RequiresConditionState = true;
  bool RequiresRowMaskState = true;
  // Each entry retains one actual bound ABI argument. LaneWise results are
  // known only when every demanded input bit in that active lane is known.
  SmallVector<TensixSFPUTransferInput, 3> Inputs;
  // For LanePermutation only: destination lane -> source lane. This complete
  // permutation is owned by the native instruction semantics. Inactive
  // destinations preserve their own old lane, not the mapped source lane.
  SmallVector<uint32_t, 0> SourceLanes;
};

enum class TensixSFPUConditionEnableUpdate : uint8_t {
  Preserve,
  Invert,
  Disable,
  Enable,
};
struct TensixSFPUConditionUpdate {
  TensixSFPUConditionEnableUpdate Enable;
  bool Flags;
};
struct TensixSFPULaneConfigReset {
  // Fields are projected by the native owner; no client decodes bit positions.
  // Only CC-active lanes among the first Columns predicates clear the column;
  // ROW_MASK does not predicate SFPCONFIG. High configuration bits survive.
  unsigned Columns;
  bool ClearsRowMask = true;
  bool ClearsDestIndex = false;
  bool ClearsDstReadBlock = false;
  bool ClearsDstWriteBlock = false;
  bool ClearsDstReadColumnExchange = false;
  bool ClearsDstWriteColumnExchange = false;
  bool ClearsDstIndexCapture = false;
};
using TensixSFPULaneControl =
    std::variant<std::monostate, TensixSFPUConditionUpdate,
                 TensixSFPULaneConfigReset>;

enum class TensixSFPUConfigurationField : uint8_t { EnableDestIndex };
struct TensixSFPUConfigurationRequirement {
  unsigned Argument;
  TensixSFPUConfigurationField Field;
  SmallVector<uint32_t, 4> Registers;
};
struct TensixSFPUConfigurationFacts {
  // A requirement applies only when this actual ABI operand selects one of
  // Registers. The named field must be disabled in every native lane.
  SmallVector<TensixSFPUConfigurationRequirement, 2> Requirements;
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
  const TensixSFPUTransferFacts &getTransferFacts() const {
    return Transfer;
  }
  // Exact transitions of supported state-control forms, not entry-state or
  // execution proofs. An empty projection does not mean no state effects.
  const TensixSFPULaneControl &getLaneControl() const { return LaneControl; }
  // Absent means this native form's configuration legality is unreceived.
  // Present with no requirements is a proved empty set, never the default.
  const std::optional<TensixSFPUConfigurationFacts> &
  getConfigurationFacts() const { return Configuration; }

private:
  friend Expected<TensixBoundSFPUContract>
      getTensixBoundSFPUContract(Intrinsic::ID, ArrayRef<TensixBoundValue>);
  TensixBoundSFPUContract() = default;
  Intrinsic::ID ID = Intrinsic::not_intrinsic;
  SmallVector<TensixBoundArgumentRole, 12> Roles;
  SmallVector<TensixBoundRegisterArgument, 12> Registers;
  SmallVector<TensixBoundRegisterConstraint, 8> Constraints;
  SmallVector<TensixSFPUArchitecturalEffect, 16> Effects;
  TensixSFPUTransferFacts Transfer;
  TensixSFPULaneControl LaneControl;
  std::optional<TensixSFPUConfigurationFacts> Configuration;
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

struct TensixSFPURegisterGeometry {
  uint32_t NumLanes;
  uint32_t BitsPerLane;
};
// Raw physical geometry from the native register class, not a tensor layout.
// A successful query says nothing about contents, masks or initialization.
Expected<TensixSFPURegisterGeometry>
getTensixSFPURegisterGeometry(uint32_t Number);

// A hardware property of one actual physical register, not an initialization
// seed supplied by a caller or inferred from a nonallocatable register class.
// ImmutableInitialized guarantees defined bits in every native lane. It does
// not describe numerical values or imply that lanes contain equal values.
enum class TensixSFPURegisterContents : uint8_t {
  Unknown,
  ImmutableInitialized,
};
Expected<TensixSFPURegisterContents>
getTensixSFPURegisterContents(uint32_t Number);

// Check width and Tensix feature compatibility without constructing a target
// subtarget whose CLI diagnostics terminate the process.
Expected<bool> verifyTensixTargetFeatures(const Triple &TT, StringRef CPU,
                                          StringRef Features);

// Recoverable preflight for compiler library clients. Function target-features
// must describe the same feature set used to construct the TargetMachine.
// Checks LLVM IR validity and the target's logical field, issue, feature,
// definedness, carrier ABI and condition-stack contracts before code
// generation.
Error verifyTensixModule(Module &M);

} // namespace RISCV
} // namespace llvm
#endif
