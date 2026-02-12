// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024
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

#include "testsupport.h"

#include <string.h>

/*
 * Regression coverage for a safety bug in zdnn_reshape_ztensor()'s last-resort
 * path.
 *
 * Historical behavior (bug):
 * - When the reshaped tensor required an unstick/restick temp buffer larger
 *   than the 1MB stack buffer, the implementation attempted malloc(s).
 * - If malloc(s) failed, it silently fell back to the 1MB stack buffer anyway,
 *   risking an out-of-bounds write.
 *
 * Expected behavior (fixed):
 * - If s > 1MB and malloc(s) fails, return ZDNN_ALLOCATION_FAILURE.
 */

void setUp(void) {}
void tearDown(void) {}

void test_large_tmpbuf_malloc_failure_returns_error(void) {
  zdnn_tensor_desc pre_src = {0}, pre_dest = {0};
  zdnn_tensor_desc tfrmd_src = {0}, tfrmd_dest = {0};
  zdnn_ztensor src = {0}, dest = {0};

  // Force a huge unstickified size `s` so malloc(s) is expected to fail on any
  // realistic system. Use a product that stays within uint64_t (avoid wrap).
  // s = num_elements * sizeof(FP32) ~= 2^62 bytes (4 exabytes).
  const uint32_t big = (1U << 30);
  zdnn_init_pre_transformed_desc(ZDNN_4D, FP32, &pre_src, big, 1, 1, big);
  zdnn_init_pre_transformed_desc(ZDNN_4D, FP32, &pre_dest, big, 1, 1, big);

  // Set transformed desc layouts to satisfy reshape preconditions, but make
  // dim1 differ so we take the last-resort unstick/restick path.
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &tfrmd_src, 1, 1, 1, 64);
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &tfrmd_dest, 1, 1, 2, 32);

  zdnn_init_ztensor(&pre_src, &tfrmd_src, &src);
  zdnn_init_ztensor(&pre_dest, &tfrmd_dest, &dest);

  // Provide dummy buffers; we should fail before any unstick/restick touches.
  src.buffer = malloc_aligned_4k(4096);
  dest.buffer = malloc_aligned_4k(4096);
  TEST_ASSERT_NOT_NULL_MESSAGE(src.buffer, "src.buffer allocation failed");
  TEST_ASSERT_NOT_NULL_MESSAGE(dest.buffer, "dest.buffer allocation failed");

  src.is_transformed = true;
  dest.is_transformed = false;

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  zdnn_status st = zdnn_reshape_ztensor(&src, &dest);
  restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ZDNN_ALLOCATION_FAILURE, st,
                                  "Expected allocation failure status");
  TEST_ASSERT_NOT_NULL_MESSAGE(
      strstr(buf_stderr, "reshape tmp buffer"),
      "Expected reshape tmp buffer allocation failure message");

  free_aligned_4k(src.buffer);
  free_aligned_4k(dest.buffer);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_large_tmpbuf_malloc_failure_returns_error);

  return UNITY_END();
}
