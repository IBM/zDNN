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

#include <stdint.h>
#include <string.h>

void setUp(void) {
  memset(&nnpa_query_result, 0, sizeof(nnpa_query_result));

  // Start with all bits installed so each test can remove only what it needs.
  nnpa_query_result.installed_functions_vector.bits_0to63 = UINT64_MAX;
  nnpa_query_result.installed_functions_vector.bits_64to127 = UINT64_MAX;
  nnpa_query_result.installed_functions_vector.bits_128to191 = UINT64_MAX;
  nnpa_query_result.installed_functions_vector.bits_192to255 = UINT64_MAX;
  nnpa_query_result.installed_parameter_block_formats.bits_0to63 = UINT64_MAX;
  nnpa_query_result.installed_parameter_block_formats.bits_64to127 =
      UINT64_MAX;
}

void tearDown(void) {}

void test_function_list_rejects_null_or_zero_count(void) {
  nnpa_function_code funcs[] = {NNPA_ADD};

  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed_list(NULL, 1));
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed_list(funcs, 0));
}

void test_function_list_rejects_invalid_entries(void) {
  nnpa_function_code invalid_negative[] = {(nnpa_function_code)-1};
  nnpa_function_code invalid_large[] = {(nnpa_function_code)999};

  TEST_ASSERT_FALSE(
      zdnn_is_nnpa_function_installed_list(invalid_negative, 1));
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed_list(invalid_large, 1));
}

void test_function_list_matches_varargs_for_valid_input(void) {
  nnpa_function_code funcs[] = {NNPA_ADD, NNPA_SOFTMAX, NNPA_TRANSFORM};

  bool typed = zdnn_is_nnpa_function_installed_list(funcs, 3);
  bool variadic = zdnn_is_nnpa_function_installed(3, NNPA_ADD, NNPA_SOFTMAX,
                                                   NNPA_TRANSFORM);

  TEST_ASSERT_TRUE(typed);
  TEST_ASSERT_EQUAL(variadic, typed);
}

void test_function_list_reports_missing_function(void) {
  nnpa_query_result.installed_functions_vector.bits_0to63 = 0;
  nnpa_query_result.installed_functions_vector.bits_64to127 = 0;
  nnpa_query_result.installed_functions_vector.bits_128to191 = 0;
  nnpa_query_result.installed_functions_vector.bits_192to255 = 0;

  nnpa_function_code funcs[] = {NNPA_ADD};
  TEST_ASSERT_FALSE(zdnn_is_nnpa_function_installed_list(funcs, 1));
}

void test_parmblk_list_rejects_null_or_zero_count(void) {
  nnpa_parmblk_format formats[] = {NNPA_PARMBLKFORMAT_0};

  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed_list(NULL, 1));
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed_list(formats, 0));
}

void test_parmblk_list_rejects_invalid_entries(void) {
  nnpa_parmblk_format invalid_negative[] = {(nnpa_parmblk_format)-1};
  nnpa_parmblk_format invalid_large[] = {(nnpa_parmblk_format)999};

  TEST_ASSERT_FALSE(
      zdnn_is_nnpa_parmblk_fmt_installed_list(invalid_negative, 1));
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed_list(invalid_large, 1));
}

void test_parmblk_list_matches_varargs_for_valid_input(void) {
  nnpa_parmblk_format formats[] = {NNPA_PARMBLKFORMAT_0, NNPA_PARMBLKFORMAT_1};

  bool typed = zdnn_is_nnpa_parmblk_fmt_installed_list(formats, 2);
  bool variadic =
      zdnn_is_nnpa_parmblk_fmt_installed(2, NNPA_PARMBLKFORMAT_0,
                                         NNPA_PARMBLKFORMAT_1);

  TEST_ASSERT_TRUE(typed);
  TEST_ASSERT_EQUAL(variadic, typed);
}

void test_parmblk_list_reports_missing_format(void) {
  nnpa_query_result.installed_parameter_block_formats.bits_0to63 = 0;
  nnpa_query_result.installed_parameter_block_formats.bits_64to127 = 0;

  nnpa_parmblk_format formats[] = {NNPA_PARMBLKFORMAT_0};
  TEST_ASSERT_FALSE(zdnn_is_nnpa_parmblk_fmt_installed_list(formats, 1));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_function_list_rejects_null_or_zero_count);
  RUN_TEST(test_function_list_rejects_invalid_entries);
  RUN_TEST(test_function_list_matches_varargs_for_valid_input);
  RUN_TEST(test_function_list_reports_missing_function);
  RUN_TEST(test_parmblk_list_rejects_null_or_zero_count);
  RUN_TEST(test_parmblk_list_rejects_invalid_entries);
  RUN_TEST(test_parmblk_list_matches_varargs_for_valid_input);
  RUN_TEST(test_parmblk_list_reports_missing_format);

  return UNITY_END();
}
