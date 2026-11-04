; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t-bh-O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs %s -o %t-bh-O0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %s -o %t-bh-O0.o
; RUN: llvm-objdump -d --mattr=+xtttensixbh %t-bh-O0.o > %t-bh-O0.dis
; RUN: %python %S/Inputs/check-tensix-contents-mop-outer.py --mir %t-bh-O0.mir --assembly %t-bh-O0.s --object %t-bh-O0.o --disassembly %t-bh-O0.dis
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t-bh-O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs %s -o %t-bh-O2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %s -o %t-bh-O2.o
; RUN: llvm-objdump -d --mattr=+xtttensixbh %t-bh-O2.o > %t-bh-O2.dis
; RUN: %python %S/Inputs/check-tensix-contents-mop-outer.py --mir %t-bh-O2.mir --assembly %t-bh-O2.s --object %t-bh-O2.o --disassembly %t-bh-O2.dis
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t-wh-O0.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs %s -o %t-wh-O0.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %s -o %t-wh-O0.o
; RUN: llvm-objdump -d --mattr=+xtttensixwh %t-wh-O0.o > %t-wh-O0.dis
; RUN: %python %S/Inputs/check-tensix-contents-mop-outer.py --mir %t-wh-O0.mir --assembly %t-wh-O0.s --object %t-wh-O0.o --disassembly %t-wh-O0.dis
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -stop-before=riscv-tensix-hazards %s -o %t-wh-O2.mir
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs %s -o %t-wh-O2.s
; RUN: llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %s -o %t-wh-O2.o
; RUN: llvm-objdump -d --mattr=+xtttensixwh %t-wh-O2.o > %t-wh-O2.dis
; RUN: %python %S/Inputs/check-tensix-contents-mop-outer.py --mir %t-wh-O2.mir --assembly %t-wh-O2.s --object %t-wh-O2.o --disassembly %t-wh-O2.dis
; MOP1 hardware outer/inner loops retain scalar contents execute and source CFG.
target triple = "riscv32-unknown-elf"
declare void @llvm.riscv.tt.contents.encc(i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.config.reset()
declare <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 immarg)
declare <32 x i32> @llvm.riscv.tt.contents.iadd(<32 x i32>, <32 x i32>, i32 immarg)
declare void @llvm.riscv.tt.contents.store(<32 x i32>, i32, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.contents.replay.template.begin(i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.end(i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 immarg, i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.contents.replay.template.mop(i32 immarg, i32 immarg)
declare { <32 x i32> } @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 immarg, i32 immarg, ...)
declare void @llvm.riscv.tt.mop.control.write(i32 immarg, i32)
declare void @llvm.riscv.tt.mop.clear(i32 immarg)
declare void @llvm.riscv.tt.mop(i32 immarg, i32 immarg, i32 immarg)
declare void @llvm.riscv.tt.mop.port(i32 immarg, i32, i32, i32)

define void @contents_mop_outer2_inner4(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 2)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer2_inner7(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 2)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer3_inner4(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 3)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer3_inner7(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 3)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer2_inner1(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 2)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer3_inner1(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 3)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer127_inner127(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 127)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 127)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}

define void @contents_mop_outer1_inner1(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 1)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 8)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  %updated.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %active, <32 x i32> %cap)
  %updated = extractvalue { <32 x i32> } %updated.mop, 0
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  %final = phi <32 x i32> [%old, %entry], [%updated, %latch]
  call void @llvm.riscv.tt.contents.store(<32 x i32> %final, i32 2, i32 0, i32 4)
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
