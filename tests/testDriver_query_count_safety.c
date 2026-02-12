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

#include <limits.h>
#include <stdint.h>
#include <string.h>

/*
 * Regression tests for query varargs count handling.
 *
 * Before this hardening, these loops used small unsigned counters (uint16_t
 * and uint8_t) against an int `count`. Very large count values could cause
 * counter wraparound and non-terminating loops.
 */

void setUp(void) {
  memset(&nnpa_query_result, 0, sizeof(nnpa_query_result));

  // Mark all functions/formats as installed so negative checks in this file are
  // specifically validating count/input guards.
  nnpa_query_result.installed_functions_vector.bits_0to63 = UINT64_MAX;
  nnpa_query_result.installed_functions_vector.bits_64to127 = UINT64_MAX;
  nnpa_query_result.installed_functions_vector.bits_128to191 = UINT64_MAX;
  nnpa_query_result.installed_functions_vector.bits_192to255 = UINT64_MAX;
  nnpa_query_result.installed_parameter_block_formats.bits_0to63 = UINT64_MAX;
  nnpa_query_result.installed_parameter_block_formats.bits_64to127 =
      UINT64_MAX;
}

void tearDown(void) {}

void test_function_query_rejects_non_positive_count(void) {
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed(0));
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed(-1));
}

void test_function_query_rejects_count_above_bitmap_capacity(void) {
  int too_many =
      (int)BIT_SIZEOF(nnpa_query_result.installed_functions_vector) + 1;
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed(too_many));
}

void test_function_query_rejects_extreme_count(void) {
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed(INT_MAX));
}

void test_function_query_rejects_negative_function_code(void) {
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed(1, -1));
}

void test_function_query_valid_input_still_succeeds(void) {
  TEST_ASSERT_TRUE(zdnn_is_nnpa_function_installed(2, NNPA_ADD, NNPA_SOFTMAX));
}

void test_parmblk_query_rejects_non_positive_count(void) {
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed(0));
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed(-1));
}

void test_parmblk_query_rejects_count_above_bitmap_capacity(void) {
  int too_many =
      (int)BIT_SIZEOF(nnpa_query_result.installed_parameter_block_formats) + 1;
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed(too_many));
}

void test_parmblk_query_rejects_extreme_count(void) {
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed(INT_MAX));
}

void test_parmblk_query_rejects_negative_format_code(void) {
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed(1, -1));
}

void test_parmblk_query_valid_input_still_succeeds(void) {
  TEST_ASSERT_TRUE(
      zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_function_query_rejects_non_positive_count);
  RUN_TEST(test_function_query_rejects_count_above_bitmap_capacity);
  RUN_TEST(test_function_query_rejects_extreme_count);
  RUN_TEST(test_function_query_rejects_negative_function_code);
  RUN_TEST(test_function_query_valid_input_still_succeeds);
  RUN_TEST(test_parmblk_query_rejects_non_positive_count);
  RUN_TEST(test_parmblk_query_rejects_count_above_bitmap_capacity);
  RUN_TEST(test_parmblk_query_rejects_extreme_count);
  RUN_TEST(test_parmblk_query_rejects_negative_format_code);
  RUN_TEST(test_parmblk_query_valid_input_still_succeeds);

  return UNITY_END();
}
