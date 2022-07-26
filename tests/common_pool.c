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

#include "common_pool.h"

/// Call public API and checks returned status and values matches expected.
///
/// \return nothing but throws test failure if actual status status doesn't
/// match expected. An error is also thrown if expected status is ZDNN_OK but
/// actual output values to not match expected input values.
///
void test_pool_function(nnpa_function_code function_code, uint32_t *input_shape,
                        zdnn_data_layouts input_layout,
                        bool repeat_first_input_value, float *input_values,
                        zdnn_pool_padding padding_type, uint32_t kernel_height,
                        uint32_t kernel_width, uint32_t stride_height,
                        uint32_t stride_width, uint32_t *output_shape,
                        zdnn_data_layouts output_layout,
                        zdnn_status expected_status,
                        bool repeat_first_expected_value,
                        float *expected_values) {

// Test requires AIU
#ifdef TEST_AIU

  // Create input and output ztensors
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      input_shape, input_layout, test_datatype, NO_CONCAT,
      repeat_first_input_value, input_values);
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      output_shape, output_layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  char api_method[AIU_METHOD_STR_LENGTH] = "zdnn_<tbd>";
  zdnn_status status = GENERAL_TESTCASE_FAILURE;

  // Call public NNPA method
  switch (function_code) {
  case NNPA_AVGPOOL2D:
    strcpy(api_method, "zdnn_avgpool2d");
    status =
        zdnn_avgpool2d(input_ztensor, padding_type, kernel_height, kernel_width,
                       stride_height, stride_width, output_ztensor);
    break;
  case NNPA_MAXPOOL2D:
    strcpy(api_method, "zdnn_maxpool2d");
    status =
        zdnn_maxpool2d(input_ztensor, padding_type, kernel_height, kernel_width,
                       stride_height, stride_width, output_ztensor);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("unsupported function_code: %d", function_code);
    break;
  }

  // Assert returned status matches expected
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to %s() to returned status %08x \"%s\" but expected %08x \"%s\"",
      api_method, status, zdnn_get_status_message(status), expected_status,
      zdnn_get_status_message(expected_status));

  fp_tolerance *tol = NULL;

  switch (output_ztensor->pre_transformed_desc->type) {
  case BFLOAT:
    tol = &tol_bfloat;
    break;
  case FP16:
    tol = &tol_fp16;
    break;
  case FP32:
    tol = &tol_fp32;
    break;
  default:
    break;
    // should never get here
  }

  // If expected status is ZDNN_OK, assert output values matches expected
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values_adv(output_ztensor, repeat_first_expected_value,
                              expected_values, *tol);
  }

  // Cleanup test ztensors
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
#endif
}
