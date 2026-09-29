//===-- RISCVTensixRegisterContentsTest.cpp -----------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/Target/RISCV/RISCVTensix.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <limits>

using namespace llvm;
using namespace llvm::RISCV;

namespace {

class RISCVTensixImmutableRegisterContentsTest
    : public testing::TestWithParam<uint32_t> {};

TEST_P(RISCVTensixImmutableRegisterContentsTest,
       HardwareContentsAreInitialized) {
  auto Contents = getTensixSFPURegisterContents(GetParam());
  ASSERT_TRUE(bool(Contents)) << toString(Contents.takeError());
  EXPECT_EQ(*Contents, TensixSFPURegisterContents::ImmutableInitialized);
}

// C8/C9/C10 and C15 are hardware read-only contents. In particular, C15's
// lanes differ: initialized does not promise a uniform vector or a value.
INSTANTIATE_TEST_SUITE_P(ReadOnlyHardwareRegisters,
                         RISCVTensixImmutableRegisterContentsTest,
                         testing::Values(8u, 9u, 10u, 15u));

class RISCVTensixMutableRegisterContentsTest
    : public testing::TestWithParam<uint32_t> {};

TEST_P(RISCVTensixMutableRegisterContentsTest, EntryContentsRemainUnknown) {
  auto Contents = getTensixSFPURegisterContents(GetParam());
  ASSERT_TRUE(bool(Contents)) << toString(Contents.takeError());
  EXPECT_EQ(*Contents, TensixSFPURegisterContents::Unknown);
}

// L0-L7 need actual program writes. C11-C14 need actual configuration writes;
// SFPI's software reservation of C11 is not a hardware initialization fact.
INSTANTIATE_TEST_SUITE_P(WritableRegisters,
                         RISCVTensixMutableRegisterContentsTest,
                         testing::Values(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 11u,
                                         12u, 13u, 14u));

class RISCVTensixInvalidRegisterContentsTest
    : public testing::TestWithParam<uint32_t> {};

TEST_P(RISCVTensixInvalidRegisterContentsTest, RejectsUnsupportedRegister) {
  auto Contents = getTensixSFPURegisterContents(GetParam());
  ASSERT_FALSE(bool(Contents));
  consumeError(Contents.takeError());
}

// The macro-only hardware location 16 is outside the published native SFPU
// register geometry, just like an otherwise out-of-range number.
INSTANTIATE_TEST_SUITE_P(OutsideNativeRegisterGeometry,
                         RISCVTensixInvalidRegisterContentsTest,
                         testing::Values(16u,
                                         std::numeric_limits<uint32_t>::max()));

} // namespace
