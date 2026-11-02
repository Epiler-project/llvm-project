; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/mode.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MODE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=AGGREGATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=AGGREGATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/insert.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=AGGREGATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/phi.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=AGGREGATE
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh %t/return.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=RETURN

; A formal multi-result intrinsic does not authorize a general aggregate ABI.
; Only its direct, single-level field extractions are numerical SSA contents.
; MODE: contents copy4 mode must be 0 or 1
; AGGREGATE: SFPU carrier cannot be nested in an aggregate
; RETURN: SFPU carrier cannot cross the return ABI

;--- mode.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @invalid_mode() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 2)
  ret void
}

;--- store.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define void @tuple_store(ptr %out) "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 0)
  store { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, ptr %out
  ret void
}

;--- call.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @escape({ <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> })
define void @tuple_call() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 0)
  call void @escape({ <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group)
  ret void
}

;--- insert.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @escape({ <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> })
define void @tuple_insert() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 0)
  %changed = insertvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, <32 x i32> %v, 0
  call void @escape({ <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %changed)
  ret void
}

;--- phi.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
define void @tuple_phi(i1 %condition) "tensix-executor"="trisc1" {
entry:
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  br i1 %condition, label %left, label %right
left:
  %a = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 0)
  br label %join
right:
  %b = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 1)
  br label %join
join:
  %group = phi { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } [ %a, %left ], [ %b, %right ]
  %r0 = extractvalue { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %r0, i32 0, i32 0, i32 4)
  ret void
}

;--- return.ll
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, i32 immarg)
define { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @tuple_return() "tensix-executor"="trisc1" {
  %v = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %group = call { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } @llvm.riscv.tt.contents.copy4(<32 x i32> %v, <32 x i32> %v, <32 x i32> %v, <32 x i32> %v, i32 0)
  ret { <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32> } %group
}
