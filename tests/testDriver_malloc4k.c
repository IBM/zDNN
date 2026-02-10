// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021, 2024
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "testsupport.h"

void setUp(void) {}

void tearDown(void) {}

// test 0-byte allocation
void malloc4k_zero() {
  void *ptr = malloc_aligned_4k(0);
  TEST_ASSERT_MESSAGE(
      ptr == NULL,
      "malloc_aligned_4k() returned non-zero for 0-byte allocation");
}

// test absolute hardware max + 1 byte allocation
void malloc4k_overflow_guard() {
  // malloc_aligned_4k() internally allocates `size + extra_allocation`.
  // Verify we fail safely (return NULL) when `size + extra_allocation` would
  // overflow size_t.
  size_t extra_allocation = (AIU_PAGESIZE_IN_BYTES - 1) + sizeof(void *);
  void *ptr = malloc_aligned_4k(SIZE_MAX - extra_allocation + 1);
  TEST_ASSERT_MESSAGE(ptr == NULL,
                      "malloc_aligned_4k() returned non-NULL for a size that "
                      "would overflow internal calculations");
}

// test different happy-path allocation sizes and make sure the return address
// is on 4k boundary
void malloc4k_check_boundary() {

#define PLUS_AND_MINUS 2

  // 1K, 4K, 32K, 64K, 256K, 1M, 1G, 2G
  // 5 allocations (-2, -1, +0, +1, +2) of each
  unsigned int allocations[] = {1,   4,    32,          64,
                                256, 1024, 1024 * 1024, 2 * 1024 * 1024};

  for (int i = 0; i < sizeof(allocations) / sizeof(allocations[0]); i++) {
    for (size_t j = allocations[i] * 1024 - PLUS_AND_MINUS;
         j <= allocations[i] * 1024 + PLUS_AND_MINUS; j++) {
      void *ptr = malloc_aligned_4k(j);

      LOG_DEBUG(
          "malloc_aligned_4k() returned location = %016lx\n, size = %zu\n",
          (uintptr_t)ptr, j);

      TEST_ASSERT_MESSAGE_FORMATTED(
          ptr,
          "detected NULL return from malloc_aligned_4k(), size = %zu, "
          "location = %016lx\n",
          j, (uintptr_t)ptr);

      TEST_ASSERT_MESSAGE_FORMATTED(
          !((uintptr_t)ptr % AIU_PAGESIZE_IN_BYTES),
          "detected non-4k aligned return from malloc_aligned_4k(), size = "
          "%zu, "
          "location = %016lx\n",
          j, (uintptr_t)ptr);

      free_aligned_4k(ptr);
    }
  }

  TEST_PASS();
}

// test that allocation failures do not print from within malloc_aligned_4k()
void malloc4k_failure_is_quiet() {
  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  void *ptr = malloc_aligned_4k((size_t)1 << 63); // absurdly large, should fail
  restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_MESSAGE(ptr == NULL,
                      "malloc_aligned_4k() unexpectedly succeeded for a huge "
                      "allocation");

  TEST_ASSERT_MESSAGE(strstr(buf_stderr, "Error during malloc") == NULL,
                      "malloc_aligned_4k() printed to stderr unexpectedly");
  TEST_ASSERT_MESSAGE(strstr(buf_stderr, "errno") == NULL,
                      "malloc_aligned_4k() printed errno to stderr unexpectedly");
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(malloc4k_zero);
  RUN_TEST(malloc4k_overflow_guard);
  RUN_TEST(malloc4k_check_boundary);
  RUN_TEST(malloc4k_failure_is_quiet);
  return UNITY_END();
}
