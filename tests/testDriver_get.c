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

#include "testsupport.h"

void setUp(void) {}

void tearDown(void) {}

//=================================================================================================
// tests for zdnn_get_max_limit and zdnn_get_min_limit

void test_max_limit_dlf16_fp32() {
  zdnn_status expected_status = ZDNN_OK;
  float expected_value = DLF16_MAX_AS_FP32;
  float my_data = 0;
  zdnn_status return_status =
      zdnn_get_max_limit(ZDNN_DLFLOAT16, FP32, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_FLOAT(expected_value, my_data);
}

void test_max_limit_dlf16_fp16() {
  zdnn_status expected_status = ZDNN_OK;
  uint16_t expected_value = FP16_MAX;
  uint16_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_max_limit(ZDNN_DLFLOAT16, FP16, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_UINT16(expected_value, my_data);
}

void test_max_limit_dlf16_bfloat() {
  zdnn_status expected_status = ZDNN_OK;
  uint16_t expected_value = DLF16_MAX_AS_BFLOAT;
  uint16_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_max_limit(ZDNN_DLFLOAT16, BFLOAT, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_UINT16(expected_value, my_data);
}

void test_min_limit_int8_fp32() {
  zdnn_status expected_status = ZDNN_OK;
  float expected_value = INT8_MIN_AS_FP32;
  float my_data = 0;
  zdnn_status return_status =
      zdnn_get_min_limit(ZDNN_BINARY_INT8, FP32, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_FLOAT(expected_value, my_data);
}

void test_min_limit_int8_fp16() {
  zdnn_status expected_status = ZDNN_OK;
  uint16_t expected_value = INT8_MIN_AS_FP16;
  uint16_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_min_limit(ZDNN_BINARY_INT8, FP16, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_UINT16(expected_value, my_data);
}

void test_min_limit_int8_bfloat() {
  zdnn_status expected_status = ZDNN_OK;
  uint16_t expected_value = INT8_MIN_AS_BFLOAT;
  uint16_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_min_limit(ZDNN_BINARY_INT8, BFLOAT, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_UINT16(expected_value, my_data);
}

void test_min_limit_int8_int8() {
  zdnn_status expected_status = ZDNN_OK;
  int8_t expected_value = INT8_MIN;
  int8_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_min_limit(ZDNN_BINARY_INT8, INT8, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_INT8(expected_value, my_data);
}

void test_min_limit_int32_int32() {
  zdnn_status expected_status = ZDNN_OK;
  int32_t expected_value = INT32_MIN;
  int32_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_min_limit(ZDNN_BINARY_INT32, INT32, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
  TEST_ASSERT_EQUAL_INT32(expected_value, my_data);
}

void test_invalid_limit_int32_int8() {
  zdnn_status expected_status = ZDNN_INVALID_TYPE;
  int32_t my_data = 0;
  zdnn_status return_status =
      zdnn_get_min_limit(ZDNN_BINARY_INT32, INT8, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
}

void test_invalid_transformed_type() {
  zdnn_status expected_status = ZDNN_INVALID_TYPE;
  float my_data = 0;
  zdnn_status return_status = zdnn_get_max_limit(999, FP32, &my_data);

  TEST_ASSERT_EQUAL(expected_status, return_status);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST(test_max_limit_dlf16_fp32);
  RUN_TEST(test_max_limit_dlf16_fp16);
  RUN_TEST(test_max_limit_dlf16_bfloat);
  RUN_TEST(test_min_limit_int8_fp32);
  RUN_TEST(test_min_limit_int8_fp16);
  RUN_TEST(test_min_limit_int8_bfloat);
  RUN_TEST(test_min_limit_int8_int8);
  RUN_TEST(test_min_limit_int32_int32);
  RUN_TEST(test_invalid_limit_int32_int8);
  RUN_TEST(test_invalid_transformed_type);
  return UNITY_END();
}