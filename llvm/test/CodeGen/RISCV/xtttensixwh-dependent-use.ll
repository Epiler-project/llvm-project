; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj < %s | llvm-objdump -d --no-print-imm-hex --mattr=+xtttensixwh - | FileCheck %s --check-prefix=OBJ

; This scalar ordering intrinsic is shared by the supported Tensix models and
; does not issue a Tensix instruction. It must consume the actual readback GPR
; before a subsequent store; a fence or an unused load is not a replacement.
; The sequence alone does not establish asynchronous hardware completion.
declare void @llvm.riscv.tt.dependent.use(i32)

define void @ncrisc_readback(ptr %address, ptr %output) "tensix-executor"="ncrisc" {
; ASM-LABEL: ncrisc_readback:
; ASM: sw {{[a-z0-9]+}}, 0([[NADDR:[a-z0-9]+]])
; ASM: lw [[NREAD:[a-z0-9]+]], 0([[NADDR]])
; ASM: and zero, zero, [[NREAD]]
; ASM-NOT: and zero, zero,
; ASM-NOT: fence
; ASM-NOT: call
; ASM: sw
; ASM-NOT: and zero, zero,
; ASM: ret
; ISEL-LABEL: name: ncrisc_readback
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.address)
; ISEL: [[NREG:%[0-9]+]]:gpr = LW {{.*}} :: (volatile load (s32) from %ir.address)
; ISEL: PseudoTTDependentUse {{(killed )?}}[[NREG]]
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.output)
; OBJ-LABEL: <ncrisc_readback>:
; OBJ: sw {{[a-z0-9]+}}, 0([[ONADDR:[a-z0-9]+]])
; OBJ: lw [[ONREAD:[a-z0-9]+]], 0([[ONADDR]])
; OBJ: and zero, zero, [[ONREAD]]
; OBJ-NOT: and zero, zero,
; OBJ: sw
; OBJ: ret
  store volatile i32 1, ptr %address, align 4
  %value = load volatile i32, ptr %address, align 4
  call void @llvm.riscv.tt.dependent.use(i32 %value)
  store volatile i32 2, ptr %output, align 4
  ret void
}

define void @trisc_readback(ptr %address, ptr %output) "tensix-executor"="trisc1" {
; ASM-LABEL: trisc_readback:
; ASM: sw {{[a-z0-9]+}}, 0([[TADDR:[a-z0-9]+]])
; ASM: lw [[TREAD:[a-z0-9]+]], 0([[TADDR]])
; ASM: and zero, zero, [[TREAD]]
; ASM-NOT: and zero, zero,
; ASM-NOT: fence
; ASM-NOT: call
; ASM: sw
; ASM-NOT: and zero, zero,
; ASM: ret
; ISEL-LABEL: name: trisc_readback
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.address)
; ISEL: [[TREG:%[0-9]+]]:gpr = LW {{.*}} :: (volatile load (s32) from %ir.address)
; ISEL: PseudoTTDependentUse {{(killed )?}}[[TREG]]
; ISEL: SW {{.*}} :: (volatile store (s32) into %ir.output)
; OBJ-LABEL: <trisc_readback>:
; OBJ: sw {{[a-z0-9]+}}, 0([[OTADDR:[a-z0-9]+]])
; OBJ: lw [[OTREAD:[a-z0-9]+]], 0([[OTADDR]])
; OBJ: and zero, zero, [[OTREAD]]
; OBJ-NOT: and zero, zero,
; OBJ: sw
; OBJ: ret
  store volatile i32 1, ptr %address, align 4
  %value = load volatile i32, ptr %address, align 4
  call void @llvm.riscv.tt.dependent.use(i32 %value)
  store volatile i32 2, ptr %output, align 4
  ret void
}
