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

#include "common_act.h"

void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

float approximate_gelu(float x) {
  return 0.5 * x * (1.0 + tanhf(x * 0.7978845608 * (1.0 + 0.044715 * x * x)));
}

/**
 * zdnn_gelu_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_gelu_test(uint32_t *io_dims, zdnn_data_layouts layout, float *input,
                    zdnn_status expected_status, float *expected_values) {

  /*
   * Input Tensor
   */
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      io_dims, layout, test_datatype, NO_CONCAT, false, input);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      io_dims, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Begin Testing!
   */
  zdnn_status status = zdnn_gelu(input_ztensor, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_gelu() to returned status %08x but expected  %08x\n",
      status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
  -------------------------------------------------------------------------------
                                  GeLU Basic
                                  Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_gelu_basic_nhwc_basic
 *
 * Simple test of all 0  input values
 * Expect a mirror of the Input values as the Output values
 *
 * Input values as NHWC
 *  [[
 *    [[0], [0], [0]],
 *    [[0], [0], [0]],
 *    [[0], [0], [0]]
 *  ]]
 *
 * Expected Output values as NHWC
 * [[
 *    [[0], [0], [0]],
 *    [[0], [0], [0]],
 *    [[0], [0], [0]]
 *  ]]
 *
 */
void zdnn_gelu_basic_zeros_nhwc() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[0] * shape[1] * shape[2];

  float input_values[num_io_buffer_values];
  gen_float_array_zeros(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  gen_float_array_zeros(num_io_buffer_values, expected_values);

  zdnn_gelu_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_gelu_basic_negatives_nhwc
 *
 * Simple test of all negative input values
 *
 * Input values as NHWC
 *  [[
 *    [[-1.1], [-1.2], [-1.3]],
 *    [[-1.4], [-1.5], [-1.6]],
 *    [[-1.7], [-1.8], [-1.9]]
 *  ]]
 *
 */
void zdnn_gelu_basic_negatives_nhwc() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];
  float input_values[] = {-1.1, -1.2, -1.3, -1.4, -1.5, -1.6, -1.7, -1.8, -1.9};
  float expected_values[num_io_buffer_values];
  generate_expected_output(approximate_gelu, input_values, num_io_buffer_values,
                           expected_values);
  zdnn_gelu_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_gelu_basic_random_large_nhwc
 *
 * Simple test of all random input values
 */
void zdnn_gelu_basic_random_large_nhwc() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 10, 30, 60}; // Will be same for in and out dim.
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];
  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);
  float expected_values[num_io_buffer_values];
  generate_expected_output(approximate_gelu, input_values, num_io_buffer_values,
                           expected_values);
  zdnn_gelu_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  GeLU Basic
                                  Layout: 3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_gelu_basic_random_neg_large_3d
 *
 * Simple test of all random negative input values
 */
void zdnn_gelu_basic_random_neg_large_3d() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {20, 30, 40}; // Will be same for in and out dim.
  int num_io_buffer_values = shape[0] * shape[1] * shape[2];
  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);
  float expected_values[num_io_buffer_values];
  generate_expected_output(approximate_gelu, input_values, num_io_buffer_values,
                           expected_values);
  zdnn_gelu_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_gelu_basic_random_large_nhwc
 *
 * Simple test of all random input values
 */
void zdnn_gelu_basic_random_large_3d() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {20, 30, 40}; // Will be same for in and out dim.
  int num_io_buffer_values = shape[0] * shape[1] * shape[2];
  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);
  float expected_values[num_io_buffer_values];
  generate_expected_output(approximate_gelu, input_values, num_io_buffer_values,
                           expected_values);
  zdnn_gelu_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_gelu_basic_zeros_nhwc);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_gelu_basic_negatives_nhwc);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_gelu_basic_random_large_nhwc);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_gelu_basic_random_neg_large_3d);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_gelu_basic_random_large_3d);
  UNITY_END();
}
