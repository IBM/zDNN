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
- Status: DONE
- Problem:
  - `set_zdnn_status()` builds a `full_fmtstr` by prefixing `ZDNN_...: ` then
    `strncat()` of caller format without accounting for the existing prefix.
  - This can overflow `full_fmtstr` for long `format` strings.
- Files:
  - `zdnn/status.c`
- Tasks:
  - [x] Replace unsafe concat with a size-safe approach (`snprintf`, or a bounded
    append using remaining capacity).
  - [x] Add regression test for long format strings (no crash; output remains
    NUL-terminated).
- Branch:
  - `qe/status-safe-format` (DONE)

### 1.2 Fix Possible Out-of-Bounds Write in `zdnn_reshape_ztensor()` Fallback
- Status: DONE
- Problem:
  - Fallback path allocates a 1MB stack buffer; if required size `s` is larger
    and `malloc(s)` fails, it still falls back to the 1MB stack buffer and can
    write past it.
- Files:
  - `zdnn/reshape_ztensor.c`
- Tasks:
  - [x] If `s > STACK_TMPBUF_SIZE` and heap allocation fails, return
    `ZDNN_ALLOCATION_FAILURE`.
  - [x] Remove the expensive implicit 1MB memset (`= {0}`) where not needed.
  - [x] Add regression test that forces the fallback path, forces malloc failure,
    and asserts a clean error return (no stack-buffer fallback).
- Branch:
  - `qe/reshape-fallback-alloc-safety` (DONE)

### 1.3 Fix NNPA Detection Edge Case When `ZDNN_ENABLE_PRECHECK=true`
- Status: DONE
- Problem:
  - Linux STFLE path checks for doubleword alignment of a local `unsigned char`
    array. That array is not guaranteed 8-byte aligned; with precheck enabled it
    can spuriously fail.
- Files:
  - `zdnn/zdnn_init.c`
- Tasks:
  - [x] Make the STFLE facility list storage explicitly 8-byte aligned (e.g.
    `uint64_t facilities64[STFLE_LENGTH/8]`).
  - [x] Add a defensive comment explaining why the alignment is required.
  - [x] Add regression coverage for the alignment requirement where feasible.
- Branch:
  - `qe/stfle-alignment-precheck` (DONE)

### 1.4 Ensure Logger Copy Path is Always NUL-Terminated
- Status: DONE
- Problem:
  - `log_message()` uses `strncpy()` for the "no formatting" path and may not
    NUL-terminate, then calls `strlen(msg_buf)`.
- Files:
  - `zdnn/logger.c`
- Tasks:
  - [x] Replace `strncpy` with a NUL-terminating write (`snprintf` or explicit
    terminator).
  - [x] Add regression test covering the `arg == NULL` path with a long message.
- Branch:
  - `qe/logger-nul-termination` (DONE)

### 1.5 Fix Incorrect Field Used in `ztensor_slice_dim4()` Error Message
- Status: DONE
- Problem:
  - Error path reports `transformed_desc->layout` although the switch is on the
    *pre-transformed* layout.
- Files:
  - `zdnn/tensor_desc.c`
- Tasks:
  - [x] Fix the message to report the correct layout value.
  - [x] Add test that triggers this error path and asserts message contains the
    expected layout value.
- Branch:
  - `qe/slice-dim4-error-message` (DONE)

### 1.6 Harden `malloc_aligned_4k()`
- Status: DONE
- Problems:
  - Overflow check is ineffective (`size > SIZE_MAX` is always false).
  - Allocator prints to stderr on failure, which is undesirable for a library.
- Files:
  - `zdnn/malloc4k.c`
- Tasks:
  - [x] Correct overflow guard (`size > SIZE_MAX - extra_allocation`).
  - [x] Remove unconditional stderr printing; let callers map NULL to
    `ZDNN_ALLOCATION_FAILURE`.
  - [x] Add tests for overflow and failure behavior.
- Branch:
  - `qe/malloc4k-hardening` (DONE)

## 2. Usability Improvements

### 2.1 Add Typed Transform Wrappers to Avoid Varargs Footguns
- Status: DONE
- Motivation:
  - `zdnn_transform_ztensor(ztensor, ...)` is easy to misuse for concatenated
    layouts (FICO/ZRH/BIDIR_*). Typed wrappers make call sites clearer and
    safer.
- Files:
  - `zdnn/zdnn.h`, `zdnn/zdnn.map`, `README.md`, implementation file
- Tasks:
  - [x] Add typed wrappers (FICO and ZRH) to avoid varargs argument-order
    mistakes.
  - [x] Implement wrappers as thin shims over the existing varargs API.
  - [x] Document usage and pointer ordering.
  - [x] Add tests validating wrappers pass through correctly.
- Branch:
  - `qe/transform-typed-wrappers` (DONE)

### 2.2 Provide First-Class Work-Area Size Helpers
- Status: DONE
- Motivation:
  - README documents formulas for `work_area` sizing (LSTM/GRU). Codifying these
    as APIs reduces integration errors and avoids repeated re-derivation.
- Files:
  - `zdnn/zdnn.h`, `zdnn/zdnn.map`, `README.md`, new `*.c` implementation
- Tasks:
  - [x] Add work-area sizing APIs:
    - `zdnn_get_lstm_work_area_size(num_timesteps, num_batches, num_hidden, direction)`
    - `zdnn_get_gru_work_area_size(num_timesteps, num_batches, num_hidden, direction)`
  - [x] Add tests matching the documented formulas.
  - [x] Update docs to recommend caller-managed work areas for performance.
- Branch:
  - `qe/work-area-size-apis` (DONE)

### 2.3 Provide Save-Area Size Helper for Softmax/Reduce
- Status: DONE
- Motivation:
  - Callers either need to know the constant or rely on implicit malloc/free.
  - A helper makes integration clearer and avoids hidden allocations.
- Files:
  - `zdnn/zdnn.h`, `zdnn/zdnn.map`, `README.md`
- Tasks:
  - [x] Add `zdnn_get_save_area_size(nnpa_function_code)`.
  - [x] Add tests.
- Branch:
  - `qe/save-area-size-apis` (DONE)

## 3. Build/Integration Quality

### 3.1 Remove Unused C++ Include in Initializer
- Status: DONE (optional follow-up remains)
- Problem:
  - `zdnn/initializer.cpp` includes `<iostream>` but doesn’t use it.
- Tasks:
  - [x] Remove include.
  - [ ] (Optional follow-up) consider a C-only auto-init mechanism for Linux to
    remove the C++ toolchain dependency; keep behavior identical.
- Branch:
  - `qe/initializer-cleanup` (DONE)

## 4. Test Coverage Additions

Each enhancement branch is expected to add or extend a dedicated Unity test
binary (new `tests/testDriver_*.c` where possible) so failures are easy to
attribute to the enhancement.
