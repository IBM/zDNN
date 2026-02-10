# Quality Enhancements TODO (zDNN)

This document tracks a set of focused quality enhancements identified during a
repo review.

Principles for this work:
- One enhancement per branch.
- Each branch must include:
  - the fix/improvement
  - targeted tests for the new/changed behavior
  - doc updates where they reduce future misuse
  - commit messages that clearly state: what changed, why, expected impact,
    and justification.
- Avoid staging/committing unrelated untracked files.

## 1. Safety / Correctness (Highest Priority)

### 1.1 Fix Potential Buffer Overflow in Status Formatting
- Problem:
  - `set_zdnn_status()` builds a `full_fmtstr` by prefixing `ZDNN_...: ` then
    `strncat()` of caller format without accounting for the existing prefix.
  - This can overflow `full_fmtstr` for long `format` strings.
- Files:
  - `zdnn/status.c`
- Tasks:
  - Replace unsafe concat with a size-safe approach (`snprintf` or
    `strncat` with remaining capacity).
  - Add regression test that exercises long format strings and ensures:
    - no crash
    - output remains NUL-terminated
- Branch:
  - `qe/status-safe-format`

### 1.2 Fix Possible Out-of-Bounds Write in `zdnn_reshape_ztensor()` Fallback
- Problem:
  - Fallback path allocates a 1MB stack buffer; if required size `s` is larger
    and `malloc(s)` fails, it still falls back to the 1MB stack buffer and can
    write past it.
- Files:
  - `zdnn/reshape_ztensor.c`
- Tasks:
  - If `s > STACK_TMPBUF_SIZE` and heap allocation fails, return
    `ZDNN_ALLOCATION_FAILURE`.
  - Remove the expensive implicit 1MB memset (`= {0}`) where not needed.
  - Add regression test:
    - forces the fallback path
    - forces malloc failure
    - asserts clean error return (no stack-buffer fallback)
- Branch:
  - `qe/reshape-fallback-alloc-safety`

### 1.3 Fix NNPA Detection Edge Case When `ZDNN_ENABLE_PRECHECK=true`
- Problem:
  - Linux STFLE path checks for doubleword alignment of a local `unsigned char`
    array. That array is not guaranteed 8-byte aligned; with precheck enabled it
    can spuriously fail.
- Files:
  - `zdnn/zdnn_init.c`
- Tasks:
  - Make the STFLE facility list storage explicitly 8-byte aligned (e.g.
    `uint64_t facilities64[STFLE_LENGTH/8]`).
  - (Optional) add a defensive comment explaining why.
  - Add build-time assertions or tests where feasible.
- Branch:
  - `qe/stfle-alignment-precheck`

### 1.4 Ensure Logger Copy Path is Always NUL-Terminated
- Problem:
  - `log_message()` uses `strncpy()` for the "no formatting" path and may not
    NUL-terminate, then calls `strlen(msg_buf)`.
- Files:
  - `zdnn/logger.c`
- Tasks:
  - Replace `strncpy` with a NUL-terminating write (`snprintf` or explicit
    terminator).
  - Add regression test covering the `arg == NULL` path with a long message.
- Branch:
  - `qe/logger-nul-termination`

### 1.5 Fix Incorrect Field Used in `ztensor_slice_dim4()` Error Message
- Problem:
  - Error path reports `transformed_desc->layout` although the switch is on the
    *pre-transformed* layout.
- Files:
  - `zdnn/tensor_desc.c`
- Tasks:
  - Fix the message to report the correct layout value.
  - Add test that triggers this error path and asserts message contains the
    expected layout value.
- Branch:
  - `qe/slice-dim4-error-message`

### 1.6 Harden `malloc_aligned_4k()`
- Problems:
  - Overflow check is ineffective (`size > SIZE_MAX` is always false).
  - Allocator prints to stderr on failure, which is undesirable for a library.
- Files:
  - `zdnn/malloc4k.c`
- Tasks:
  - Correct overflow guard (`size > SIZE_MAX - extra_allocation`).
  - Remove unconditional stderr printing; let callers map NULL to
    `ZDNN_ALLOCATION_FAILURE`.
  - Add tests for overflow and failure behavior.
- Branch:
  - `qe/malloc4k-hardening`

## 2. Usability Improvements

### 2.1 Add Typed Transform Wrappers to Avoid Varargs Footguns
- Motivation:
  - `zdnn_transform_ztensor(ztensor, ...)` is easy to misuse for concatenated
    layouts (FICO/ZRH/BIDIR_*). Typed wrappers make call sites clearer and
    safer.
- Files:
  - `zdnn/zdnn.h`, `zdnn/zdnn.map`, `README.md`, implementation file
- Tasks:
  - Add wrappers (examples):
    - `zdnn_transform_ztensor_fico(ztensor, f, i, c, o)`
    - `zdnn_transform_ztensor_zrh(ztensor, z, r, h)`
  - Implement as thin wrappers over existing varargs.
  - Document usage and pointer ordering.
  - Add tests validating wrappers pass through correctly.
- Branch:
  - `qe/transform-typed-wrappers`

### 2.2 Provide First-Class Work-Area Size Helpers
- Motivation:
  - README documents formulas for `work_area` sizing (LSTM/GRU). Codifying these
    as APIs reduces integration errors and avoids repeated re-derivation.
- Files:
  - `zdnn/zdnn.h`, `zdnn/zdnn.map`, `README.md`, new `*.c` implementation
- Tasks:
  - Add APIs like:
    - `zdnn_get_lstm_work_area_size(num_timesteps, num_batches, num_hidden, direction)`
    - `zdnn_get_gru_work_area_size(num_timesteps, num_batches, num_hidden, direction)`
  - Add tests matching README formulas.
  - Update docs to recommend caller-managed work areas for performance.
- Branch:
  - `qe/work-area-size-apis`

### 2.3 Provide Save-Area Size Helper for Softmax/Reduce
- Motivation:
  - Callers either need to know the constant or rely on implicit malloc/free.
  - A helper makes integration clearer and avoids hidden allocations.
- Files:
  - `zdnn/zdnn.h`, `zdnn/zdnn.map`, `README.md`
- Tasks:
  - Add e.g. `zdnn_get_save_area_size(nnpa_function_code)` (or two explicit
    helpers).
  - Add tests.
- Branch:
  - `qe/save-area-size-apis`

## 3. Build/Integration Quality

### 3.1 Remove Unused C++ Include in Initializer
- Problem:
  - `zdnn/initializer.cpp` includes `<iostream>` but doesn’t use it.
- Tasks:
  - Remove include.
  - (Optional follow-up) consider a C-only auto-init mechanism for Linux to
    remove the C++ toolchain dependency; keep behavior identical.
- Branch:
  - `qe/initializer-cleanup`

## 4. Test Coverage Additions

Each enhancement branch is expected to add or extend a dedicated Unity test
binary (new `tests/testDriver_*.c` where possible) so failures are easy to
attribute to the enhancement.

