; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %s -o %t.O0.s && FileCheck %s --check-prefixes=ASM,OR-FACE --implicit-check-not=ttsfpload --implicit-check-not=ttsfpstore < %t.O0.s && FileCheck %s --check-prefixes=ASM,OR-BASE < %t.O0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o %t.O2.s && FileCheck %s --check-prefixes=ASM,OR-FACE < %t.O2.s && FileCheck %s --check-prefixes=ASM,OR-BASE < %t.O2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -stop-after=finalize-isel -verify-machineinstrs %s -o - | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -stop-after=riscv-tensix-issue-word-fold %s -o - | FileCheck %s --check-prefix=FOLD

; Dynamic face selection must retain WH Mode16, use32 and actual face bits.
; It is not a full-Dst clear and does not materialize any SFPU payload.
; Check both OR inputs independently so either commutative order is legal,
; while both checks retain the exact register feeding the instruction store.
declare void @llvm.riscv.tt.zeroacc.port(i32 immarg, i32, i32, i32, i32, i32)

define void @face_fp16(i32 noundef %face) "tensix-executor"="trisc1" {
; FOLD-LABEL: define void @face_fp16
; FOLD: [[WORD:%.*]] = or disjoint i32 268959744, {{%.*}}
; FOLD: call void @llvm.riscv.tt.issue.word(i32 0, i32 16, i32 [[WORD]])
; ISEL-LABEL: name: face_fp16
; ISEL: PseudoTTZEROACCPort 0,
; ASM-LABEL: face_fp16:
; ASM-DAG: andi [[FACE:[a-z0-9]+]], {{[a-z0-9]+}}, 63
; ASM-DAG: lui [[BASE:[a-z0-9]+]], 65664
; OR-FACE: or [[WORD:[a-z0-9]+]], {{([a-z0-9]+, )?}}[[FACE]]{{(, [a-z0-9]+)?}}{{$}}
; OR-BASE: or [[WORD:[a-z0-9]+]], {{([a-z0-9]+, )?}}[[BASE]]{{(, [a-z0-9]+)?}}{{$}}
; ASM: lui [[ADDRESS:[a-z0-9]+]], 1048128
; ASM: sw [[WORD]], 0([[ADDRESS]])
; ASM: ret
  %bounded = and i32 %face, 63
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 %bounded, i32 0, i32 0, i32 0, i32 1)
  ret void
}

define void @face_fp32(i32 noundef %face) "tensix-executor"="trisc1" {
; FOLD-LABEL: define void @face_fp32
; FOLD: [[WORD32:%.*]] = or disjoint i32 271056896, {{%.*}}
; FOLD: call void @llvm.riscv.tt.issue.word(i32 0, i32 16, i32 [[WORD32]])
; ISEL-LABEL: name: face_fp32
; ISEL: PseudoTTZEROACCPort 0,
; ASM-LABEL: face_fp32:
; ASM-DAG: andi [[FACE32:[a-z0-9]+]], {{[a-z0-9]+}}, 31
; ASM-DAG: lui [[BASE32:[a-z0-9]+]], 66176
; OR-FACE: or [[WORD32:[a-z0-9]+]], {{([a-z0-9]+, )?}}[[FACE32]]{{(, [a-z0-9]+)?}}{{$}}
; OR-BASE: or [[WORD32:[a-z0-9]+]], {{([a-z0-9]+, )?}}[[BASE32]]{{(, [a-z0-9]+)?}}{{$}}
; ASM: lui [[ADDRESS32:[a-z0-9]+]], 1048128
; ASM: sw [[WORD32]], 0([[ADDRESS32]])
; ASM: ret
  %bounded = and i32 %face, 31
  call void @llvm.riscv.tt.zeroacc.port(i32 0, i32 %bounded, i32 0, i32 0, i32 1, i32 1)
  ret void
}
