// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "common_pool.h"

void setUp(void) { /* This is run before EACH TEST */

  tol_bfloat.ulps = 64;
  tol_bfloat.epsilon_mult = (0.1 / EPSILON_BFLOAT) + 1;

  tol_fp16.ulps = 64;
  tol_fp16.epsilon_mult = (0.1 / EPSILON_FP16) + 1;

  tol_fp32.ulps = 64 * 16384;
  tol_fp32.epsilon_mult = (0.1 / EPSILON_FLOAT) + 1;

  VERIFY_HW_ENV;
}

void tearDown(void) { /* This is run after EACH TEST */
}

void test_meanreduce2d(uint32_t *input_shape, zdnn_data_layouts input_layout,
                       bool repeat_first_input_value, float *input_values,
                       uint32_t *output_shape, zdnn_data_layouts output_layout,
                       zdnn_status expected_status,
                       bool repeat_first_expected_value,
                       float *expected_values) {

  // Create input and output ztensors
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      input_shape, input_layout, test_datatype, NO_CONCAT,
      repeat_first_input_value, input_values);
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      output_shape, output_layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

// Test requires AIU
#ifdef TEST_AIU
  // Call public NNPA method
  zdnn_status status = zdnn_meanreduce2d(input_ztensor, output_ztensor);

  // Assert returned status matches expected
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_meanreduce2d to returned status %08x but expected "
      "%08x\n",
      status, expected_status);

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
#endif

  // Cleanup test ztensors
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
 * Simple test of basic mean reduce
 */
void zdnn_meanreduce2d_basic() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  /* Visualization of input values
    [[
      [[1, 10], [2, 20], [3, 30]],
      [[4, 40], [5, 50], [6, 60]],
      [[7, 70], [8, 80], [9, 90]]
    ]]
  */
  uint32_t input_shape[] = {1, 3, 3, 2};
  float input_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                          50, 6,  60, 7,  70, 8,  80, 9,  90};

  /* Visualization of expected values
    [[
      [[5, 50]]
    ]]
  */
  uint32_t output_shape[] = {1, 1, 1, 2};
  float expected_values[] = {5, 50};

  test_meanreduce2d(input_shape, layout, false, input_values, output_shape,
                    layout, ZDNN_OK, false, expected_values);
}

/*
 * Check that we don't hit a condition code when Height and Width dimensions are
 * at the largest size allowed.
 */
void zdnn_meanreduce2d_max_height_width_dims_pass() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  uint32_t input_shape[] = {1, MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE,
                            MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE, 2};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  uint32_t output_shape[] = {1, 1, 1, 2};
  // Since all input values are the same, they should average to the same.
  float *expected_values = input_values;

  test_meanreduce2d(input_shape, layout, true, input_values, output_shape,
                    layout, ZDNN_OK, true, expected_values);
}

/*
 * Check that we hit the expected condition code when height is over the
 * largest size.
 */
void zdnn_meanreduce2d_over_max_height_fail() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_max_dim is a valid tensor dimension size but is too large for a
  // meanreduce dimension. This should lead to a condition code from the NNPA.
  // If not, update the test constant and the API documentation.
  uint32_t over_max_dim = MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE + 1;

  uint32_t input_shape[] = {1, over_max_dim, 3, 2};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  uint32_t output_shape[] = {1, 1, 1, 2};
  // Output values don't really matter as we expect failure status.
  float *expected_values = input_values;

  test_meanreduce2d(input_shape, layout, true, input_values, output_shape,
                    layout, ZDNN_FUNC_RC_F001, true, expected_values);
}

/*
 * Check that we hit the expected condition code when width is over the
 * largest size.
 */
void zdnn_meanreduce2d_over_max_width_fail() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_max_dim is a valid tensor dimension size but is too large for a
  // meanreduce dimension. This should lead to a condition code from the NNPA.
  // If not, update the test constant and the API documentation.
  uint32_t over_max_dim = MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE + 1;

  uint32_t input_shape[] = {1, 3, over_max_dim, 2};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  uint32_t output_shape[] = {1, 1, 1, 2};
  // Output values don't really matter as we expect failure status.
  float *expected_values = input_values;

  test_meanreduce2d(input_shape, layout, true, input_values, output_shape,
                    layout, ZDNN_FUNC_RC_F001, true, expected_values);
}

int main() {

  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(zdnn_meanreduce2d_basic);
  RUN_TEST_ALL_DATATYPES(zdnn_meanreduce2d_max_height_width_dims_pass);
  RUN_TEST_ALL_DATATYPES(zdnn_meanreduce2d_over_max_height_fail);
  RUN_TEST_ALL_DATATYPES(zdnn_meanreduce2d_over_max_width_fail);
  return UNITY_END();
}
