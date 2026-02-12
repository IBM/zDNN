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

/*
 * Tests for quantized matmul work-area size helper API.
 */

void setUp(void) {}
void tearDown(void) {}

void test_quantized_matmul_work_area_size_matches_internal_qc_tilde_shape(void) {
  // Use dim1=65 so size differs between int8 source tensor and DLF16 qc_tilde.
  zdnn_tensor_desc input_c_desc;
  init_transformed_desc(ZDNN_NHWC, ZDNN_BINARY_INT8, ZDNN_FORMAT_4DFEATURE,
                        &input_c_desc, 1, 1, 1, 65);

  zdnn_tensor_desc qc_tilde_desc;
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &qc_tilde_desc, 1, 1, 1, 65);

  zdnn_ztensor input_c;
  zdnn_init_ztensor(&input_c_desc, &input_c_desc, &input_c);

  uint64_t expected = zdnn_getsize_ztensor(&qc_tilde_desc);
  uint64_t actual = zdnn_get_quantized_matmul_work_area_size(&input_c, false);

  TEST_ASSERT_EQUAL_UINT64(expected, actual);
}

void test_quantized_matmul_work_area_size_returns_zero_for_precomputed(void) {
  zdnn_tensor_desc input_c_desc;
  init_transformed_desc(ZDNN_NHWC, ZDNN_BINARY_INT8, ZDNN_FORMAT_4DFEATURE,
                        &input_c_desc, 1, 1, 2, 17);

  zdnn_ztensor input_c;
  zdnn_init_ztensor(&input_c_desc, &input_c_desc, &input_c);

  TEST_ASSERT_EQUAL_UINT64(
      0, zdnn_get_quantized_matmul_work_area_size(&input_c, true));
}

void test_quantized_matmul_work_area_size_rejects_invalid_input(void) {
  TEST_ASSERT_EQUAL_UINT64(0,
                           zdnn_get_quantized_matmul_work_area_size(NULL, false));

  zdnn_ztensor missing_desc = {0};
  TEST_ASSERT_EQUAL_UINT64(
      0, zdnn_get_quantized_matmul_work_area_size(&missing_desc, false));
}

void test_quantized_matmul_work_area_size_rejects_non_int8_input_c(void) {
  zdnn_tensor_desc invalid_desc;
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &invalid_desc, 1, 1, 1, 64);

  zdnn_ztensor input_c;
  zdnn_init_ztensor(&invalid_desc, &invalid_desc, &input_c);

  TEST_ASSERT_EQUAL_UINT64(
      0, zdnn_get_quantized_matmul_work_area_size(&input_c, false));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_quantized_matmul_work_area_size_matches_internal_qc_tilde_shape);
  RUN_TEST(test_quantized_matmul_work_area_size_returns_zero_for_precomputed);
  RUN_TEST(test_quantized_matmul_work_area_size_rejects_invalid_input);
  RUN_TEST(test_quantized_matmul_work_area_size_rejects_non_int8_input_c);

  return UNITY_END();
}
