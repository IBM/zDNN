// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
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

void setUp(void) { /* This is run before EACH TEST */
}

void tearDown(void) {}

// test 0-byte allocation
void malloc4k_zero() {
  void *ptr = malloc_aligned_4k(0);
  TEST_ASSERT_MESSAGE(
      ptr == NULL,
      "malloc_aligned_4k() returned non-zero for 0-byte allocation");
}

// test absolute hardware max + 1 byte allocation
// SIZE_MAX is 18446744073709551615UL (2^64)
void malloc4k_size_max_plus_one() {
  void *ptr = malloc_aligned_4k(SIZE_MAX + 1);
  TEST_ASSERT_MESSAGE(
      ptr == NULL,
      "malloc_aligned_4k() returned non-zero SIZE_MAX+1 bytes allocation");
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

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(malloc4k_zero);
  RUN_TEST(malloc4k_size_max_plus_one);
  RUN_TEST(malloc4k_check_boundary);
  return UNITY_END();
}
