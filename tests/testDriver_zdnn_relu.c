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

#include "common_act.h"

// -----------------------------------------------------------------------------
// ReLU Unit Testing, for convenience, recall the following:
//     relu(x) -> if (x>0) {return x; else return 0;}
// -----------------------------------------------------------------------------

void setUp(void) { VERIFY_HW_ENV; }

void tearDown(void) {}
/**
 * zdnn_relu_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_relu_test(uint32_t *io_dims, zdnn_data_layouts layout, float *input,
                    float *clipping_value, zdnn_status expected_status,
                    float *expected_values) {

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
  zdnn_status status = zdnn_relu(input_ztensor, clipping_value, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_relu() to returned status %08x but expected  %08x\n",
      status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Basic
                                  Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_basic_nhwc_basic
 *
 * Simple test of all positive input values
 * Expect a mirror of the Input values as the Output values
 *
 * Input values as NHWC
 *  [[
 *    [[1], [2], [3]],
 *    [[4], [5], [6]],
 *    [[7], [8], [9]]
 *  ]]
 *
 * Expected Output values as NHWC
 *  [[
 *    [[1], [2], [3]],
 *    [[4], [5], [6]],
 *    [[7], [8], [9]]
 *  ]]
 */
void zdnn_relu_basic_nhwc_basic() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_expected_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float clip_value = 0;
  zdnn_relu_test(shape, ZDNN_NHWC, input_expected_values, &clip_value, ZDNN_OK,
                 input_expected_values);
}

/**
 * zdnn_relu_basic_nhwc_basic_clip6
 *
 * Simple test of all positive input values
 * Expect a mirror of the Input values as the Output values
 *
 * Input values as NHWC
 *  [[
 *    [[1], [2], [3]],
 *    [[4], [5], [6]],
 *    [[7], [8], [9]]
 *  ]]
 *
 * Expected Output values as NHWC
 *  [[
 *    [[1], [2], [3]],
 *    [[4], [5], [6]],
 *    [[6], [6], [6]]
 *  ]]
 */
void zdnn_relu_basic_nhwc_basic_clip6() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_expected_values[] = {1, 2, 3, 4, 5, 6, 6, 6, 6};
  float clip_value = 6;
  zdnn_relu_test(shape, ZDNN_NHWC, input_expected_values, &clip_value, ZDNN_OK,
                 input_expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Basic
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_deadneuron_3d_basic
 *
 * Simple test of all negative input values
 * Expect a dead neuron
 *
 * Input values as NWC sized (3,3,2):
 *  [[
 *    [[-1, -10], [-2, -20], [-3, -30]],
 *    [[-4, -40], [-5, -50], [-6, -60]],
 *    [[-7, -70], [-8, -80], [-9, -90]]
 *  ]]
 *
 * Expected Output values as NWC sized (3,3,2):
 *  [[
 *    [[0, 0], [0, 0], [0, 0]],
 *    [[0, 0], [0, 0], [0, 0]],
 *    [[0, 0], [0, 0], [0, 0]]
 *  ]]
 */
void zdnn_relu_deadneuron_3d_basic() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {3, 3, 2}; // Will be same for in and out dim.
  float input_values[] = {-1,  -10, -2,  -20, -3,  -30, -4,  -40, -5,
                          -50, -6,  -60, -7,  -70, -8,  -80, -9,  -90};
  float expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0};
  zdnn_relu_test(shape, ZDNN_3D, input_values, NULL, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Basic
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_balance_nhwc_basic
 *
 * Simple test of half positive and half negative input values
 * Expect 50% zeroed 50% valued
 *
 * Input values as NHWC
 *  [[
 *    [[10, -10], [20, -20], [30, -30]],
 *    [[40, -40], [50, -50], [60, -60]],
 *    [[70, -70], [80, -80], [90, -90]],
 *  ]]
 *
 * Expected Output values as NHWC
 *  [[
 *    [[10, 0], [20, 0], [30, 0]],
 *    [[40, 0], [50, 0], [60, 0]],
 *    [[70, 0], [80, 0], [90, 0]],
 *  ]]
 */
void zdnn_relu_balance_nhwc_basic() {
  // Initialize the dimensions for our input tensor
  uint32_t shape[] = {1, 3, 3, 2}; // Will be same for in and out dim.

  float input_values[] = {10,  -10, 20,  -20, 30,  -30, 40,  -40, 50,
                          -50, 60,  -60, 70,  -70, 80,  -80, 90,  -90};
  float expected_values[] = {10, 0,  20, 0,  30, 0,  40, 0,  50,
                             0,  60, 0,  70, 0,  80, 0,  90, 0};
  zdnn_relu_test(shape, ZDNN_NHWC, input_values, NULL, ZDNN_OK,
                 expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Basic
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_balance_nhwc_basic_clip50
 *
 * Simple test of half positive and half negative input values
 * Expect 50% zeroed 50% valued
 *
 * Input values as NHWC
 *  [[
 *    [[10, -10], [20, -20], [30, -30]],
 *    [[40, -40], [50, -50], [60, -60]],
 *    [[70, -70], [80, -80], [90, -90]],
 *  ]]
 *
 * Expected Output values as NHWC
 *  [[
 *    [[10, 0], [20, 0], [30, 0]],
 *    [[40, 0], [50, 0], [50, 0]],
 *    [[50, 0], [50, 0], [50, 0]],
 *  ]]
 */
void zdnn_relu_balance_nhwc_basic_clip50() {
  // Initialize the dimensions for our input tensor
  uint32_t shape[] = {1, 3, 3, 2}; // Will be same for in and out dim.

  float input_values[] = {10,  -10, 20,  -20, 30,  -30, 40,  -40, 50,
                          -50, 60,  -60, 70,  -70, 80,  -80, 90,  -90};
  float expected_values[] = {10, 0,  20, 0,  30, 0,  40, 0,  50,
                             0,  50, 0,  50, 0,  50, 0,  50, 0};
  float clip_value = 50;
  zdnn_relu_test(shape, ZDNN_NHWC, input_values, &clip_value, ZDNN_OK,
                 expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Large
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_basic_nhwc_large
 *
 * Simple test of all positive input values
 * Expect a mirror of the Input values as the Output values
 *
 * Input values as NHWC
 *  [[
 *    [[65000, 65100, 65200], [64000, 64100, 64200], [63000, 63100, 63200]],
 *    [[62000, 62100, 62200], [61000, 61100, 61200], [60000, 60100, 60200]],
 *    [[59000, 59100, 59200], [58000, 58100, 58200], [57000, 57100, 57200]]
 *  ]]
 *
 * Expected Output values as NHWC
 *  [[
 *    [[65000, 65100, 65200], [64000, 64100, 64200], [63000, 63100, 63200]],
 *    [[62000, 62100, 62200], [61000, 61100, 61200], [60000, 60100, 60200]],
 *    [[59000, 59100, 59200], [58000, 58100, 58200], [57000, 57100, 57200]]
 *  ]]
 *
 */
void zdnn_relu_basic_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 3}; // Will be same for in and out dim.
  float input_expected_values[] = {
      65000, 65100, 65200, 64000, 64100, 64200, 63000, 63100, 63200,
      62000, 62100, 62200, 61000, 61100, 61200, 60000, 60100, 60200,
      59000, 59100, 59200, 58000, 58100, 58200, 57000, 57100, 57200};
  zdnn_relu_test(shape, ZDNN_NHWC, input_expected_values, NULL, ZDNN_OK,
                 input_expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Large
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_deadneuron_3d_large
 *
 * Simple test of all negative input values
 * Expect a dead neuron
 *
 * Generate a test that is of size 8x8x8
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 8x8x8
 * with all 0 zeros.
 */
void zdnn_relu_deadneuron_3d_large() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {8, 8, 8}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[0] * shape[1] * shape[2];

  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  gen_float_array_zeros(num_io_buffer_values, expected_values);

  zdnn_relu_test(shape, ZDNN_3D, input_values, NULL, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  ReLU Large
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_relu_balance_nhwc_large
 *
 * Simple test of half positive and half negative input values
 * Expect 50% zeroed 50% valued
 *
 * Generate a test that is of size 50x25x10x1
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 50x25x10x1
 * with 50% zeros 50% valued.
 *
 *
 */
void zdnn_relu_balance_nhwc_large() {
  // Initialize the dimensions for our input tensor
  uint32_t shape[] = {1, 10, 25, 50}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array_pos_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  copy_to_array(num_io_buffer_values, input_values, expected_values);
  fill_everyother_with_zero_float_array(num_io_buffer_values, expected_values);

  zdnn_relu_test(shape, ZDNN_NHWC, input_values, NULL, ZDNN_OK,
                 expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_basic_nhwc_basic);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_basic_nhwc_large);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_deadneuron_3d_basic);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_balance_nhwc_basic);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_deadneuron_3d_large);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_balance_nhwc_large);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_basic_nhwc_basic_clip6);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_relu_balance_nhwc_basic_clip50);
  return UNITY_END();
}
