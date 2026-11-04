; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/mvmul-addr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ADDR
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/unpack-legacy-context.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=LEGACY
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/unpack-context-index.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=CONTEXT
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh %t/zerosrc-high.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=ZERO
; ADDR: Wormhole MVMUL field addr_mode must be in [0, 3]
; LEGACY: Wormhole UNPACR field ovrdthreadid must be in [1, 1]
; CONTEXT: Wormhole UNPACR field cfgcontextid must be in [0, 0]
; ZERO: Wormhole ZEROSRC field zero_val must be in [0, 1]

;--- mvmul-addr.ll
declare void @llvm.riscv.tt.mvmul(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.mvmul(i32 0, i32 4, i32 0, i32 0)
  ret void
}
;--- unpack-legacy-context.ll
declare void @llvm.riscv.tt.unpacr(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}
;--- unpack-context-index.ll
declare void @llvm.riscv.tt.unpacr(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc0" {
  call void @llvm.riscv.tt.unpacr(i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0, i32 1, i32 0, i32 0, i32 0)
  ret void
}
;--- zerosrc-high.ll
declare void @llvm.riscv.tt.zerosrc(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
define void @bad() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.zerosrc(i32 1, i32 0, i32 0, i32 2)
  ret void
}
