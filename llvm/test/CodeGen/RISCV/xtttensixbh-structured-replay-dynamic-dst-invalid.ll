; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/poison.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=POISON
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 %t/range.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RANGE
; POISON: bound Dst offset must be defined and non-poison
; RANGE: bound Dst offset must be proven in [0, 1023]

;--- poison.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @bad(i10 %offset) "tensix-executor"="trisc1" {
  %address = zext i10 %offset to i32
  call void @llvm.riscv.tt.replay.template.begin(i32 709, i32 0, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %address, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 709)
  ret void
}

;--- range.ll
declare void @llvm.riscv.tt.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.replay.template.end(i32 immarg)
declare void @llvm.riscv.tt.bound.sfpload(i32 immarg, i32 immarg, i32, i32 immarg, i32 immarg)
define void @bad(i32 noundef %offset) "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.replay.template.begin(i32 719, i32 1, i32 0, i32 0)
  call void @llvm.riscv.tt.bound.sfpload(i32 0, i32 0, i32 %offset, i32 0, i32 4)
  call void @llvm.riscv.tt.replay.template.end(i32 719)
  ret void
}
