; RUN: split-file %s %t
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/brisc_executor.ll -o %t/brisc_executor-bh-O0.o 2>&1 | FileCheck %s --check-prefix=BRISC-EXECUTOR
; RUN: not test -e %t/brisc_executor-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/brisc_executor.ll -o %t/brisc_executor-bh-O2.o 2>&1 | FileCheck %s --check-prefix=BRISC-EXECUTOR
; RUN: not test -e %t/brisc_executor-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/brisc_executor.ll -o %t/brisc_executor-wh-O0.o 2>&1 | FileCheck %s --check-prefix=BRISC-EXECUTOR
; RUN: not test -e %t/brisc_executor-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/brisc_executor.ll -o %t/brisc_executor-wh-O2.o 2>&1 | FileCheck %s --check-prefix=BRISC-EXECUTOR
; RUN: not test -e %t/brisc_executor-wh-O2.o
; BRISC-EXECUTOR: Tensix SFPU execution requires tensix-executor=trisc1
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/trisc0_executor.ll -o %t/trisc0_executor-bh-O0.o 2>&1 | FileCheck %s --check-prefix=TRISC0-EXECUTOR
; RUN: not test -e %t/trisc0_executor-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/trisc0_executor.ll -o %t/trisc0_executor-bh-O2.o 2>&1 | FileCheck %s --check-prefix=TRISC0-EXECUTOR
; RUN: not test -e %t/trisc0_executor-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/trisc0_executor.ll -o %t/trisc0_executor-wh-O0.o 2>&1 | FileCheck %s --check-prefix=TRISC0-EXECUTOR
; RUN: not test -e %t/trisc0_executor-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/trisc0_executor.ll -o %t/trisc0_executor-wh-O2.o 2>&1 | FileCheck %s --check-prefix=TRISC0-EXECUTOR
; RUN: not test -e %t/trisc0_executor-wh-O2.o
; TRISC0-EXECUTOR: Tensix SFPU execution requires tensix-executor=trisc1
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/missing_setup.ll -o %t/missing_setup-bh-O0.o 2>&1 | FileCheck %s --check-prefix=MISSING-SETUP
; RUN: not test -e %t/missing_setup-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/missing_setup.ll -o %t/missing_setup-bh-O2.o 2>&1 | FileCheck %s --check-prefix=MISSING-SETUP
; RUN: not test -e %t/missing_setup-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/missing_setup.ll -o %t/missing_setup-wh-O0.o 2>&1 | FileCheck %s --check-prefix=MISSING-SETUP
; RUN: not test -e %t/missing_setup-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/missing_setup.ll -o %t/missing_setup-wh-O2.o 2>&1 | FileCheck %s --check-prefix=MISSING-SETUP
; RUN: not test -e %t/missing_setup-wh-O2.o
; MISSING-SETUP: contents MOP profile requires all nine setup cells
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/nonunit_outer.ll -o %t/nonunit_outer-bh-O0.o 2>&1 | FileCheck %s --check-prefix=NONUNIT-OUTER
; RUN: not test -e %t/nonunit_outer-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/nonunit_outer.ll -o %t/nonunit_outer-bh-O2.o 2>&1 | FileCheck %s --check-prefix=NONUNIT-OUTER
; RUN: not test -e %t/nonunit_outer-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/nonunit_outer.ll -o %t/nonunit_outer-wh-O0.o 2>&1 | FileCheck %s --check-prefix=NONUNIT-OUTER
; RUN: not test -e %t/nonunit_outer-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/nonunit_outer.ll -o %t/nonunit_outer-wh-O2.o 2>&1 | FileCheck %s --check-prefix=NONUNIT-OUTER
; RUN: not test -e %t/nonunit_outer-wh-O2.o
; NONUNIT-OUTER: contents MOP outer>1 requires slot8 bound to its template
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/zero_inner.ll -o %t/zero_inner-bh-O0.o 2>&1 | FileCheck %s --check-prefix=ZERO-INNER
; RUN: not test -e %t/zero_inner-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/zero_inner.ll -o %t/zero_inner-bh-O2.o 2>&1 | FileCheck %s --check-prefix=ZERO-INNER
; RUN: not test -e %t/zero_inner-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/zero_inner.ll -o %t/zero_inner-wh-O0.o 2>&1 | FileCheck %s --check-prefix=ZERO-INNER
; RUN: not test -e %t/zero_inner-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/zero_inner.ll -o %t/zero_inner-wh-O2.o 2>&1 | FileCheck %s --check-prefix=ZERO-INNER
; RUN: not test -e %t/zero_inner-wh-O2.o
; ZERO-INNER: contents MOP profile requires outer and inner in [1, 127]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/wide_inner.ll -o %t/wide_inner-bh-O0.o 2>&1 | FileCheck %s --check-prefix=WIDE-INNER
; RUN: not test -e %t/wide_inner-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/wide_inner.ll -o %t/wide_inner-bh-O2.o 2>&1 | FileCheck %s --check-prefix=WIDE-INNER
; RUN: not test -e %t/wide_inner-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wide_inner.ll -o %t/wide_inner-wh-O0.o 2>&1 | FileCheck %s --check-prefix=WIDE-INNER
; RUN: not test -e %t/wide_inner-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wide_inner.ll -o %t/wide_inner-wh-O2.o 2>&1 | FileCheck %s --check-prefix=WIDE-INNER
; RUN: not test -e %t/wide_inner-wh-O2.o
; WIDE-INNER: contents MOP profile requires outer and inner in [1, 127]
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/dynamic_count.ll -o %t/dynamic_count-bh-O0.o 2>&1 | FileCheck %s --check-prefix=DYNAMIC-COUNT
; RUN: not test -e %t/dynamic_count-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/dynamic_count.ll -o %t/dynamic_count-bh-O2.o 2>&1 | FileCheck %s --check-prefix=DYNAMIC-COUNT
; RUN: not test -e %t/dynamic_count-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/dynamic_count.ll -o %t/dynamic_count-wh-O0.o 2>&1 | FileCheck %s --check-prefix=DYNAMIC-COUNT
; RUN: not test -e %t/dynamic_count-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/dynamic_count.ll -o %t/dynamic_count-wh-O2.o 2>&1 | FileCheck %s --check-prefix=DYNAMIC-COUNT
; RUN: not test -e %t/dynamic_count-wh-O2.o
; DYNAMIC-COUNT: contents MOP requires explicit contents execution and setup
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/duplicate_setup.ll -o %t/duplicate_setup-bh-O0.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SETUP
; RUN: not test -e %t/duplicate_setup-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/duplicate_setup.ll -o %t/duplicate_setup-bh-O2.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SETUP
; RUN: not test -e %t/duplicate_setup-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/duplicate_setup.ll -o %t/duplicate_setup-wh-O0.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SETUP
; RUN: not test -e %t/duplicate_setup-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/duplicate_setup.ll -o %t/duplicate_setup-wh-O2.o 2>&1 | FileCheck %s --check-prefix=DUPLICATE-SETUP
; RUN: not test -e %t/duplicate_setup-wh-O2.o
; DUPLICATE-SETUP: contents MOP profile requires unique setup
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/wrong_slot.ll -o %t/wrong_slot-bh-O0.o 2>&1 | FileCheck %s --check-prefix=WRONG-SLOT
; RUN: not test -e %t/wrong_slot-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/wrong_slot.ll -o %t/wrong_slot-bh-O2.o 2>&1 | FileCheck %s --check-prefix=WRONG-SLOT
; RUN: not test -e %t/wrong_slot-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wrong_slot.ll -o %t/wrong_slot-wh-O0.o 2>&1 | FileCheck %s --check-prefix=WRONG-SLOT
; RUN: not test -e %t/wrong_slot-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wrong_slot.ll -o %t/wrong_slot-wh-O2.o 2>&1 | FileCheck %s --check-prefix=WRONG-SLOT
; RUN: not test -e %t/wrong_slot-wh-O2.o
; WRONG-SLOT: contents MOP binding requires slot5, slot7 or slot8
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/wrong_binding_id.ll -o %t/wrong_binding_id-bh-O0.o 2>&1 | FileCheck %s --check-prefix=WRONG-BINDING-ID
; RUN: not test -e %t/wrong_binding_id-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/wrong_binding_id.ll -o %t/wrong_binding_id-bh-O2.o 2>&1 | FileCheck %s --check-prefix=WRONG-BINDING-ID
; RUN: not test -e %t/wrong_binding_id-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/wrong_binding_id.ll -o %t/wrong_binding_id-wh-O0.o 2>&1 | FileCheck %s --check-prefix=WRONG-BINDING-ID
; RUN: not test -e %t/wrong_binding_id-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/wrong_binding_id.ll -o %t/wrong_binding_id-wh-O2.o 2>&1 | FileCheck %s --check-prefix=WRONG-BINDING-ID
; RUN: not test -e %t/wrong_binding_id-wh-O2.o
; WRONG-BINDING-ID: contents MOP binding requires a dominating record
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/changed_capture.ll -o %t/changed_capture-bh-O0.o 2>&1 | FileCheck %s --check-prefix=CHANGED-CAPTURE
; RUN: not test -e %t/changed_capture-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/changed_capture.ll -o %t/changed_capture-bh-O2.o 2>&1 | FileCheck %s --check-prefix=CHANGED-CAPTURE
; RUN: not test -e %t/changed_capture-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/changed_capture.ll -o %t/changed_capture-wh-O0.o 2>&1 | FileCheck %s --check-prefix=CHANGED-CAPTURE
; RUN: not test -e %t/changed_capture-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/changed_capture.ll -o %t/changed_capture-wh-O2.o 2>&1 | FileCheck %s --check-prefix=CHANGED-CAPTURE
; RUN: not test -e %t/changed_capture-wh-O2.o
; CHANGED-CAPTURE: contents replay execute requires its recorded read-only capture
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/foreign_mutable.ll -o %t/foreign_mutable-bh-O0.o 2>&1 | FileCheck %s --check-prefix=FOREIGN-MUTABLE
; RUN: not test -e %t/foreign_mutable-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/foreign_mutable.ll -o %t/foreign_mutable-bh-O2.o 2>&1 | FileCheck %s --check-prefix=FOREIGN-MUTABLE
; RUN: not test -e %t/foreign_mutable-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/foreign_mutable.ll -o %t/foreign_mutable-wh-O0.o 2>&1 | FileCheck %s --check-prefix=FOREIGN-MUTABLE
; RUN: not test -e %t/foreign_mutable-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/foreign_mutable.ll -o %t/foreign_mutable-wh-O2.o 2>&1 | FileCheck %s --check-prefix=FOREIGN-MUTABLE
; RUN: not test -e %t/foreign_mutable-wh-O2.o
; FOREIGN-MUTABLE: contents replay execute requires its current mutable contents
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/remote_mop.ll -o %t/remote_mop-bh-O0.o 2>&1 | FileCheck %s --check-prefix=REMOTE-MOP
; RUN: not test -e %t/remote_mop-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/remote_mop.ll -o %t/remote_mop-bh-O2.o 2>&1 | FileCheck %s --check-prefix=REMOTE-MOP
; RUN: not test -e %t/remote_mop-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/remote_mop.ll -o %t/remote_mop-wh-O0.o 2>&1 | FileCheck %s --check-prefix=RAW-MOP-WH
; RUN: not test -e %t/remote_mop-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/remote_mop.ll -o %t/remote_mop-wh-O2.o 2>&1 | FileCheck %s --check-prefix=RAW-MOP-WH
; RUN: not test -e %t/remote_mop-wh-O2.o
; REMOTE-MOP: remote Tensix instruction ports require brisc
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/raw_mop.ll -o %t/raw_mop-bh-O0.o 2>&1 | FileCheck %s --check-prefix=RAW-MOP
; RUN: not test -e %t/raw_mop-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/raw_mop.ll -o %t/raw_mop-bh-O2.o 2>&1 | FileCheck %s --check-prefix=RAW-MOP
; RUN: not test -e %t/raw_mop-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/raw_mop.ll -o %t/raw_mop-wh-O0.o 2>&1 | FileCheck %s --check-prefix=RAW-MOP-WH
; RUN: not test -e %t/raw_mop-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/raw_mop.ll -o %t/raw_mop-wh-O2.o 2>&1 | FileCheck %s --check-prefix=RAW-MOP-WH
; RUN: not test -e %t/raw_mop-wh-O2.o
; RAW-MOP: contents MOP requires explicit contents execution and setup
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/unknown_scalar_write.ll -o %t/unknown_scalar_write-bh-O0.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-SCALAR-WRITE
; RUN: not test -e %t/unknown_scalar_write-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/unknown_scalar_write.ll -o %t/unknown_scalar_write-bh-O2.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-SCALAR-WRITE
; RUN: not test -e %t/unknown_scalar_write-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/unknown_scalar_write.ll -o %t/unknown_scalar_write-wh-O0.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-SCALAR-WRITE
; RUN: not test -e %t/unknown_scalar_write-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/unknown_scalar_write.ll -o %t/unknown_scalar_write-wh-O2.o 2>&1 | FileCheck %s --check-prefix=UNKNOWN-SCALAR-WRITE
; RUN: not test -e %t/unknown_scalar_write-wh-O2.o
; UNKNOWN-SCALAR-WRITE: contents MOP profile cannot admit unknown scalar writes
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/loop_setup.ll -o %t/loop_setup-bh-O0.o 2>&1 | FileCheck %s --check-prefix=LOOP-SETUP
; RUN: not test -e %t/loop_setup-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/loop_setup.ll -o %t/loop_setup-bh-O2.o 2>&1 | FileCheck %s --check-prefix=LOOP-SETUP
; RUN: not test -e %t/loop_setup-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/loop_setup.ll -o %t/loop_setup-wh-O0.o 2>&1 | FileCheck %s --check-prefix=LOOP-SETUP
; RUN: not test -e %t/loop_setup-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/loop_setup.ll -o %t/loop_setup-wh-O2.o 2>&1 | FileCheck %s --check-prefix=LOOP-SETUP
; RUN: not test -e %t/loop_setup-wh-O2.o
; LOOP-SETUP: contents MOP setup cannot be inside a CFG cycle
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O0 -verify-machineinstrs -filetype=obj %t/missing_dominance.ll -o %t/missing_dominance-bh-O0.o 2>&1 | FileCheck %s --check-prefix=MISSING-DOMINANCE
; RUN: not test -e %t/missing_dominance-bh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixbh -O2 -verify-machineinstrs -filetype=obj %t/missing_dominance.ll -o %t/missing_dominance-bh-O2.o 2>&1 | FileCheck %s --check-prefix=MISSING-DOMINANCE
; RUN: not test -e %t/missing_dominance-bh-O2.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O0 -verify-machineinstrs -filetype=obj %t/missing_dominance.ll -o %t/missing_dominance-wh-O0.o 2>&1 | FileCheck %s --check-prefix=MISSING-DOMINANCE
; RUN: not test -e %t/missing_dominance-wh-O0.o
; RUN: not llc -mtriple=riscv32 -mattr=+xtttensixwh -O2 -verify-machineinstrs -filetype=obj %t/missing_dominance.ll -o %t/missing_dominance-wh-O2.o 2>&1 | FileCheck %s --check-prefix=MISSING-DOMINANCE
; RUN: not test -e %t/missing_dominance-wh-O2.o
; MISSING-DOMINANCE: contents MOP setup must dominate every execution
; RAW-MOP-WH: Tensix intrinsic is not supported on Wormhole

;--- brisc_executor.ll
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

define void @invalid_mop() "tensix-executor"="brisc" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- trisc0_executor.ll
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

define void @invalid_mop() "tensix-executor"="trisc0" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- missing_setup.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- nonunit_outer.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- zero_inner.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 0)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- wide_inner.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 128)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- dynamic_count.ll
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

define void @invalid_mop(i32 noundef %count) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 %count)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- duplicate_setup.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- wrong_slot.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 6)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- wrong_binding_id.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 72, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- changed_capture.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %old)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- foreign_mutable.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %foreign = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 10)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %foreign, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- remote_mop.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop.port(i32 1, i32 0, i32 0, i32 1)
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- raw_mop.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  call void @llvm.riscv.tt.mop(i32 0, i32 0, i32 1)
  ; Restore SFPU configuration so the independent mixed-MOP owner check,
  ; rather than an earlier unknown-configuration diagnostic, rejects this case.
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- unknown_scalar_write.ll
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

define void @invalid_mop() "tensix-executor"="trisc1" {
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
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  store volatile i32 0, ptr inttoptr (i32 -4718592 to ptr), align 4
  ; Keep lane configuration defined while testing the independent MMIO alias
  ; restriction on the actual MOP control cells.
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}

;--- loop_setup.ll
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

define void @contents_mop_loop(i32 noundef %limit) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  %positive = icmp ne i32 %limit, 0
  br i1 %positive, label %loop, label %exit
loop:
  %i = phi i32 [0, %entry], [%next, %latch]
  %active = phi <32 x i32> [%old, %entry], [%updated, %latch]
  %cap = phi <32 x i32> [%capture, %entry], [%cap, %latch]
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
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

;--- missing_dominance.ll
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

define void @invalid_mop(i1 noundef %condition) "tensix-executor"="trisc1" {
entry:
  call void @llvm.riscv.tt.contents.encc(i32 3, i32 10)
  call void @llvm.riscv.tt.contents.config.reset()
  %old = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 9)
  %capture = call <32 x i32> @llvm.riscv.tt.contents.read.constant(i32 15)
  call void (i32, i32, i32, i32, i32, ...) @llvm.riscv.tt.contents.replay.template.begin(i32 71, i32 0, i32 0, i32 0, i32 1, <32 x i32> %old, <32 x i32> %capture)
  call void @llvm.riscv.tt.contents.replay.template.record.iadd(i32 71, i32 0, i32 1)
  call void (i32, ...) @llvm.riscv.tt.contents.replay.template.end(i32 71, <32 x i32> %old)
  call void @llvm.riscv.tt.contents.store(<32 x i32> %old, i32 0, i32 0, i32 4)
  br i1 %condition, label %setup, label %execute
setup:
  call void @llvm.riscv.tt.mop.control.write(i32 0, i32 1)
  call void @llvm.riscv.tt.mop.control.write(i32 1, i32 4)
  call void @llvm.riscv.tt.mop.clear(i32 2)
  call void @llvm.riscv.tt.mop.clear(i32 3)
  call void @llvm.riscv.tt.mop.clear(i32 4)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 5)
  call void @llvm.riscv.tt.mop.clear(i32 6)
  call void @llvm.riscv.tt.contents.replay.template.mop(i32 71, i32 7)
  call void @llvm.riscv.tt.mop.clear(i32 8)
  br label %execute
execute:
  %first.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %old, <32 x i32> %capture)
  %first = extractvalue { <32 x i32> } %first.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %first, i32 2, i32 0, i32 4)
  %second.mop = call { <32 x i32> } (i32, i32, ...) @llvm.riscv.tt.contents.replay.template.mop.execute.sl_v32i32s(i32 71, i32 1, <32 x i32> %first, <32 x i32> %capture)
  %second = extractvalue { <32 x i32> } %second.mop, 0
  call void @llvm.riscv.tt.contents.store(<32 x i32> %second, i32 4, i32 0, i32 4)
  ret void
}
