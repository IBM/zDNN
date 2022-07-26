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

#include "common_act.h"
#include <math.h>

// -----------------------------------------------------------------------------
// Sigmoid Unit Testing, for convenience, recall the following:
//     sigmoid(x) -> [0,1]
//     For some value x, we squash that  value to some real-valued number within
//     range [0,1].
//     For the behind the scenes:
//          sigmoid(x) -> ( 1 / (1 + e(-x)) )
//          https://mathworld.wolfram.com/SigmoidFunction.html
// -----------------------------------------------------------------------------

void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

void tearDown(void) { /* This is run after EACH TEST */
}

/**
 * Helper function to compute output tensor values using activation
 * sigmoid
 */
void act_sigmoid(float input[], float output[], int num_elems) {
  for (long i = 0; i < num_elems; i++) {
    output[i] = 1 / (1 + exp(-input[i]));
  }
}

/**
 * zdnn_sigmoid_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_sigmoid_test(uint32_t *shape, zdnn_data_layouts layout,
                       float *input_values, zdnn_status expected_status,
                       float *expected_values) {
  /*
   * Input Tensor
   */
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, false, input_values);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Begin Testing!
   */
  zdnn_status status = zdnn_sigmoid(input_ztensor, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_sigmoid() to returned status %08x but expected %08x\n",
      status, expected_status);

#ifdef TEST_AIU
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }
#endif

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
  -------------------------------------------------------------------------------
                                 Sigmoid Basic
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_sigmoid_basic_nhwc
 *
 * Simple test to demonstrate tanh
 *
 * Input values as NHWC sized (1,3,3,1):
 * [[
 *   [[0], [1], [2]],
 *   [[3], [4], [5]],
 *   [[6], [7], [8]]
 * ]]
 *
 * Expected Output values as NHWC sized (1,3,3,1):
 * [[
 *   [[0.5],          [0.7310585786], [0.880797078]],
 *   [[0.9525741268], [0.98201379],   [0.9933071491]],
 *   [[0.9975273768], [0.9990889488], [0.9996646499]
 * ]]
 *
 */
void zdnn_sigmoid_basic_nhwc() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  float expected_values[] = {
      0.5,          0.7310585786, 0.880797078,  0.9525741268, 0.98201379,
      0.9933071491, 0.9975273768, 0.9990889488, 0.9996646499,
  };
  zdnn_sigmoid_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                 Sigmoid Basic
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_sigmoid_balanced_nhwc
 *
 * Balanced (pos and neg inputs) test to demonstrate sigmoid
 *
 *
 * Input values as NHWC sized (1,3,3,2):
 *  [[
 *    [[-1, 1], [-2, 2], [-3, 3]],
 *    [[-4, 4], [-5, 5], [-6, 6]],
 *    [[-7, 7], [-8, 8], [-9, 9]],
 *  ]]
 *
 * Expected Output values as NHWC sized 1,3,3,2:
 *  [[
 *    [[0.2689414214, 0.7310585786], [0.119202922 , 0.880797078], [0.0474258732,
 * 0.9525741268]],
 *    [[0.01798621, 0.98201379],     [0.0066928509, 0.9933071491],[0.0024726232,
 * 0.9975273768]],
 *    [[0.0009110512, 0.9990889488], [0.0003353501, 0.9996646499],[0.0001233946,
 * 0.9998766054]],
 *  ]]
 */
void zdnn_sigmoid_balanced_nhwc() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 2}; // Will be same for in and out dim.
  float input_values[] = {-1, 1,  -2, 2,  -3, 3,  -4, 4,  -5,
                          5,  -6, 6,  -7, 7,  -8, 8,  -9, 9};
  float expected_values[] = {
      0.2689414214, 0.7310585786, 0.119202922,  0.880797078,  0.0474258732,
      0.9525741268, 0.01798621,   0.98201379,   0.0066928509, 0.9933071491,
      0.0024726232, 0.9975273768, 0.0009110512, 0.9990889488, 0.0003353501,
      0.9996646499, 0.0001233946, 0.9998766054,
  };
  zdnn_sigmoid_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                 Sigmoid Basic
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_sigmoid_negative_3d
 *
 * Simple test to demonstrate tanh
 *
 * Input values as NWC sized (1,2,4):
 *  [[
 *    [[-1, -2, -3, -4], [-5, -6, -7, -8]],
 *  ]]
 *
 * Expected Output values as NWC sized (1,2,4):
 *  [[
 *    [[0.2689414214, 0.119202922, 0.0474258732, 0.01798621],
 *     [0.0066928509, 0.0024726232, 0.0009110512 , 0.0003353501]],
 *  ]]
 */
void zdnn_sigmoid_negative_3d() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 2, 4}; // Will be same for in and out dim.
  float input_values[] = {-1, -2, -3, -4, -5, -6, -7, -8};
  float expected_values[] = {
      0.2689414214, 0.119202922,  0.0474258732, 0.01798621,
      0.0066928509, 0.0024726232, 0.0009110512, 0.0003353501,
  };
  zdnn_sigmoid_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                 Sigmoid Large
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_sigmoid_basic_nhwc_large
 *
 * Simple test of all positive input values
 *
 * Input values as NHWC sized (3,3,3,1):
 *  [[
 *    [[65000, 65100, 65200], [64000, 64100, 64200], [63000, 63100, 63200]],
 *    [[62000, 62100, 62200], [61000, 61100, 61200], [60000, 60100, 60200]],
 *    [[59000, 59100, 59200], [58000, 58100, 58200], [57000, 57100, 57200]]
 *  ]]
 *
 * Expected Output values as NHWC sized (3,3,3,1):
 *
 */
void zdnn_sigmoid_basic_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 3}; // Will be same for in and out dim.
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  float input_values[] = {65000, 65100, 65200, 64000, 64100, 64200, 63000,
                          63100, 63200, 62000, 62100, 62200, 61000, 61100,
                          61200, 60000, 60100, 60200, 59000, 59100, 59200,
                          58000, 58100, 58200, 57000, 57100, 57200};

  float expected_values[num_io_buffer_values];
  act_sigmoid(input_values, expected_values, num_io_buffer_values);

  zdnn_sigmoid_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                 Sigmoid Large
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_sigmoid_balanced_nhwc_large
 *
 * Simple test of half positive and half negative input values
 *
 * Generate a test that is of size 53x30x11x1
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 53x30x11x1
 */
void zdnn_sigmoid_balanced_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 4, 20, 12}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array_pos_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_sigmoid(input_values, expected_values, num_io_buffer_values);

  zdnn_sigmoid_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                 Sigmoid Large
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_sigmoid_negative_3d_large
 *
 * Simple test of all negative input values
 *
 * Generate a test that is of size 78x45x30
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 78x45x30
 */
void zdnn_sigmoid_negative_3d_large() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {10, 6, 22}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[0] * shape[1] * shape[2];

  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_sigmoid(input_values, expected_values, num_io_buffer_values);

  zdnn_sigmoid_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(zdnn_sigmoid_basic_nhwc);
  RUN_TEST_ALL_DATATYPES(zdnn_sigmoid_basic_nhwc_large);
  RUN_TEST_ALL_DATATYPES(zdnn_sigmoid_balanced_nhwc);
  RUN_TEST_ALL_DATATYPES(zdnn_sigmoid_negative_3d);
  RUN_TEST_ALL_DATATYPES(zdnn_sigmoid_balanced_nhwc_large);
  RUN_TEST_ALL_DATATYPES(zdnn_sigmoid_negative_3d_large);
  return UNITY_END();
}
