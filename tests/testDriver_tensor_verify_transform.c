// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2023, 2024
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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}

void tearDown(void) {}

/// Common test routine for transform tensors
///
/// \param[in] input_shape    Pointer to input dim array
/// \param[in] input_format   Input format
/// \param[in] input_type     Input type
/// \param[in] output_shape   Pointer to output dim array
/// \param[in] output_format  Output format
/// \param[in] output_type    Output type
/// \param[in] toc            transformation-operation code
/// \param[in] min_clipping   minimum clipping
/// \param[in] max_clipping   maximum clipping
/// \param[in] exp_status     Expected status
/// \param[in] error_msg      Error message to prepend to the standard error
///                           message
///
void test_transform(uint32_t input_shape[], zdnn_data_formats input_format,
                    zdnn_data_types input_type, uint32_t output_shape[],
                    zdnn_data_formats output_format,
                    zdnn_data_types output_type, uint32_t toc,
                    int8_t min_clipping, int8_t max_clipping,
                    zdnn_status exp_status, char *error_msg) {
  zdnn_status status = ZDNN_OK;

  zdnn_ztensor input, output;

  zdnn_tensor_desc tfrmd_desc_input, tfrmd_desc_output;

  input.transformed_desc = &tfrmd_desc_input;
  output.transformed_desc = &tfrmd_desc_output;

  init_transformed_desc(ZDNN_NHWC, input_type, input_format,
                        input.transformed_desc, input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]);

  init_transformed_desc(ZDNN_NHWC, output_type, output_format,
                        output.transformed_desc, output_shape[0],
                        output_shape[1], output_shape[2], output_shape[3]);

  func_sp_parm1_transform transform_parm1;
  memset(&transform_parm1, 0, sizeof(func_sp_parm1_transform));
  transform_parm1.toc = toc;

  func_sp_parm4_transform transform_parm4;
  memset(&transform_parm4, 0, sizeof(func_sp_parm4_transform));
  transform_parm4.clip_min = min_clipping;

  func_sp_parm5_transform transform_parm5;
  memset(&transform_parm5, 0, sizeof(func_sp_parm5_transform));
  transform_parm5.clip_max = max_clipping;

  status = verify_transform_tensors(&input, &output, &transform_parm1,
                                    &transform_parm4, &transform_parm5);

  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_status == status, "%s  Expected status = %08x, actual status = %08x",
      error_msg, exp_status, status);
}

void transform_verify_pass_fp32_dlfloat() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16,
                 NNPA_TOC_STICK_DLFLOAT, 0, 0, ZDNN_OK,
                 "DLFloat transform tensors are different.");
}

void transform_verify_pass_fp32_int8() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_BINARY_INT8,
                 NNPA_TOC_STICK_DLFLOAT, 2, 3, ZDNN_OK,
                 "DLFloat transform tensors are different.");
}

void transform_verify_pass_dlfloat_fp32() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16,
                 output_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 NNPA_TOC_STICK_DLFLOAT, 0, 0, ZDNN_OK,
                 "DLFloat transform tensors are different.");
}

void transform_verify_fail_shape_dim1() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 2};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16,
                 NNPA_TOC_STICK_DLFLOAT, 0, 0, ZDNN_INVALID_SHAPE,
                 "Failed to fail on different shapes.");
}

void transform_verify_fail_shape_dim2() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16,
                 NNPA_TOC_STICK_DLFLOAT, 0, 0, ZDNN_INVALID_SHAPE,
                 "Failed to fail on different shapes.");
}

void transform_verify_fail_shape_dim3() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 2, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_BINARY_INT8,
                 NNPA_TOC_STICK_INT8, 2, 3, ZDNN_INVALID_SHAPE,
                 "Failed to fail on different shapes.");
}

void transform_verify_fail_shape_dim4() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {2, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_BINARY_INT8,
                 NNPA_TOC_STICK_INT8, 2, 3, ZDNN_INVALID_SHAPE,
                 "Failed to fail on different shapes.");
}

void transform_verify_fail_clips_equal() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_BINARY_INT8,
                 NNPA_TOC_STICK_INT8, 3, 3, ZDNN_INVALID_CLIPPING_VALUE,
                 "Failed to fail on invalid clipping value.");
}

void transform_verify_fail_invalid_clip() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_transform(input_shape, ZDNN_FORMAT_4DGENERIC, ZDNN_BINARY_FP32,
                 output_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_BINARY_INT8,
                 NNPA_TOC_STICK_INT8, 4, 3, ZDNN_INVALID_CLIPPING_VALUE,
                 "Failed to fail on invalid clipping value.");
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(transform_verify_pass_fp32_dlfloat);
  RUN_TEST(transform_verify_pass_fp32_int8);
  RUN_TEST(transform_verify_pass_dlfloat_fp32);
  RUN_TEST(transform_verify_fail_shape_dim1);
  RUN_TEST(transform_verify_fail_shape_dim2);
  RUN_TEST(transform_verify_fail_shape_dim3);
  RUN_TEST(transform_verify_fail_shape_dim4);
  RUN_TEST(transform_verify_fail_clips_equal);
  RUN_TEST(transform_verify_fail_invalid_clip);

  return UNITY_END();
}
