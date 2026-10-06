; The issue-word form is compiler-private; only the fold may create it.
declare void @llvm.riscv.tt.issue.word(i32 immarg, i32 immarg, i32)
define void @private() "tensix-executor"="trisc1" {
  call void @llvm.riscv.tt.issue.word(i32 0, i32 38, i32 637534208)
  ret void
}
