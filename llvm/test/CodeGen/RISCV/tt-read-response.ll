; RUN: llc -mtriple=riscv32 -mattr=+m -O0 -riscv-no-aliases \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM \
; RUN:   --implicit-check-not=call --implicit-check-not=fence
; RUN: llc -mtriple=riscv32 -mattr=+m -O2 -riscv-no-aliases \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM \
; RUN:   --implicit-check-not=call --implicit-check-not=fence
; RUN: opt -passes='default<O2>' -S < %s | \
; RUN:   llc -mtriple=riscv32 -mattr=+m -O2 -riscv-no-aliases \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=ASM \
; RUN:   --implicit-check-not=call --implicit-check-not=fence
; RUN: llc -mtriple=riscv32 -mattr=+m -stop-after=finalize-isel \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=ISEL

; Wormhole NCRISC has no Tensix issue extension. Consuming a returned scalar
; uses an ordinary RISC-V instruction, but must survive optimization, DCE and
; register allocation. The load destination and consumed source must be the
; same nonzero GPR. This establishes read-response ordering, not completion of
; an asynchronous operation elsewhere on the device.
declare void @llvm.riscv.tt.read.response(i32)

define void @same_address_readback(ptr %address, ptr noalias %next) {
; ASM-LABEL: same_address_readback:
; ASM: sw {{[a-z0-9]+}}, 0([[ADDR:[a-z0-9]+]])
; ASM: lw [[READ:[ast][0-9]+]], 0([[ADDR]])
; ASM: addi zero, [[READ]], 0
; ASM: sw {{[a-z0-9]+}}, 0({{[a-z0-9]+}})
; ASM: jalr zero, 0(ra)
; ISEL-LABEL: name: same_address_readback
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.address)
; ISEL: [[REG:%[0-9]+]]:gpr = LW {{.*}} :: (volatile load (s32) from %ir.address)
; ISEL: PseudoTTReadResponse {{(killed )?}}[[REG]]
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.next)
  store volatile i32 1, ptr %address, align 4
  %value = load volatile i32, ptr %address, align 4
  call void @llvm.riscv.tt.read.response(i32 %value)
  store volatile i32 2, ptr %next, align 4
  ret void
}

define void @cross_region_readback(ptr %address, ptr noalias %other,
                                  i32 %value) {
; ASM-LABEL: cross_region_readback:
; ASM: sw {{[a-z0-9]+}}, 0([[ADDR:[a-z0-9]+]])
; ASM: lw [[READ:[ast][0-9]+]], 0([[ADDR]])
; ASM: addi zero, [[READ]], 0
; ASM: lw [[OTHER:[ast][0-9]+]], 0({{[a-z0-9]+}})
; ASM: addi zero, [[OTHER]], 0
; ASM: sw {{[a-z0-9]+}}, 0([[ADDR]])
; ASM: jalr zero, 0(ra)
; ISEL-LABEL: name: cross_region_readback
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.address)
; ISEL: [[REG:%[0-9]+]]:gpr = LW {{.*}} :: (volatile load (s32) from %ir.address)
; ISEL: PseudoTTReadResponse {{(killed )?}}[[REG]]
; ISEL: [[OTHER:%[0-9]+]]:gpr = LW {{.*}} :: (volatile load (s32) from %ir.other)
; ISEL: PseudoTTReadResponse {{(killed )?}}[[OTHER]]
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.address)
  store volatile i32 %value, ptr %address, align 4
  %returned = load volatile i32, ptr %address, align 4
  call void @llvm.riscv.tt.read.response(i32 %returned)
  %other_returned = load volatile i32, ptr %other, align 4
  call void @llvm.riscv.tt.read.response(i32 %other_returned)
  store volatile i32 %value, ptr %address, align 4
  ret void
}

; Full memory effects also prevent dead-store elimination across this handoff.
; These stores are ordinary memory actions in a disjoint region, so volatile
; semantics alone cannot preserve the first store if the intrinsic is readnone.
define void @memory_effects(ptr %address, ptr noalias %other) {
; ASM-LABEL: memory_effects:
; ASM: sw {{[a-z0-9]+}}, 0([[ADDR:[a-z0-9]+]])
; ASM: lw [[READ:[ast][0-9]+]], 0({{[a-z0-9]+}})
; ASM: addi zero, [[READ]], 0
; ASM: sw {{[a-z0-9]+}}, 0([[ADDR]])
; ASM: jalr zero, 0(ra)
; ISEL-LABEL: name: memory_effects
; ISEL: SW {{.*}} :: (store (s32) into %ir.address)
; ISEL: [[REG:%[0-9]+]]:gpr = LW {{.*}} :: (volatile load (s32) from %ir.other)
; ISEL: PseudoTTReadResponse {{(killed )?}}[[REG]]
; ISEL: SW {{.*}} :: (store (s32) into %ir.address)
  store i32 1, ptr %address, align 4
  %returned = load volatile i32, ptr %other, align 4
  call void @llvm.riscv.tt.read.response(i32 %returned)
  store i32 2, ptr %address, align 4
  ret void
}
