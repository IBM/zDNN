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

// -----------------------------------------------------------------------------
// TanH Unit Testing, for convenience, recall the following:
//     tanh(x) -> [-1,1]
//     For some value x, we squash that  value to some real-valued number within
//     range [-1,1]. Negative inputs are mapped strongly negative and zero
//     inputs are mapped near zero.
//     For the behind the scenes:
//         tanh(x) -> ( 1 - e(-2(x)) ) /  ( 1 + e(-2(x)) )
//     https://functions.wolfram.com/ElementaryFunctions/Tanh/
//                                               introductions/Tanh/ShowAll.html
// -----------------------------------------------------------------------------

void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

void tearDown(void) { /* This is run after EACH TEST */
}

/**
 * Helper function to compute output tensor values using activation
 * tanh
 */
void act_tanh(float input[], float output[], int num_elems) {
  for (long i = 0; i < num_elems; i++) {
    output[i] = (2 / (1 + exp(-2 * input[i]))) - 1;
  }
}

/**
 * zdnn_tanh_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_tanh_test(uint32_t *shape, zdnn_data_layouts layout, float *input,
                    zdnn_status expected_status, float *expected_values) {

  /*
   * Input Tensor
   */
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, false, input);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Begin Testing!
   */
  zdnn_status status = zdnn_tanh(input_ztensor, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_tanh() to returned status %08x but expected "
      "%08x\n",
      status, expected_status);

#ifdef TEST_AIU
  // Only check expected values if we expected the NNPA call to be successful
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

#endif

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
  -------------------------------------------------------------------------------
                                  TanH Basic
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_tanh_basic_nhwc
 *
 * Simple test to demonstrate tanh
 *
 * Input values as NHWC sized (1,3,3,1):
 * [[
 *   [[0.01], [0.02], [0.03]],
 *   [[0.04], [0.05], [0.06]],
 *   [[0.07], [0.08], [0.09]]
 * ]]
 *
 * Expected Output values as NHWC sized (1,3,3,1):
 * [[
 *   [[0.00999966667999946],  [0.019997333759930933], [0.029991003238820143]],
 *   [[0.03997868031116357], [0.04995837495787998], [0.059928103529143496]],
 *   [[0.06988589031642899], [0.07982976911113136], [0.0897577847471601]
 * ]]
 */
void zdnn_tanh_basic_nhwc_1() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};
  float expected_values[] = {
      0.00999966667999946, 0.019997333759930933, 0.029991003238820143,
      0.03997868031116357, 0.04995837495787998,  0.059928103529143496,
      0.06988589031642899, 0.07982976911113136,  0.0897577847471601};
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_zeros_nhwc
 *
 * Zero test to demonstrate tanh
 *
 * Input values as NHWC sized (1, 3, 3, 3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 *
 * Expected Output values as NHWC sized (1, 3, 3, 3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 */
void zdnn_tanh_zeros_nhwc_1() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 3}; // Will be same for in and out dim.
  float input_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_negative_nhwc
 *
 * Negative test to demonstrate tanh
 *
 * Input values as NHWC sized (1,3,3,1):
 * [[
 *   [[-0.01], [-0.02], [-0.03]],
 *   [[-0.04], [-0.05], [-0.06]],
 *   [[-0.07], [-0.08], [-0.09]]
 * ]]
 *
 * Expected Output values as NHWC sized (1,3,3,1):
 * [[
 *   [[-0.00999966667999946],  [-0.019997333759930933],
 * [-0.029991003238820143]],
 *   [[-0.03997868031116357], [-0.04995837495787998], [-0.059928103529143496]],
 *   [[-0.06988589031642899], [-0.07982976911113136], [-0.0897577847471601]
 * ]]
 */
void zdnn_tanh_negative_nhwc_1() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {-0.01, -0.02, -0.03, -0.04, -0.05,
                          -0.06, -0.07, -0.08, -0.09};
  float expected_values[] = {
      -0.00999966667999946, -0.019997333759930933, -0.029991003238820143,
      -0.03997868031116357, -0.04995837495787998,  -0.059928103529143496,
      -0.06988589031642899, -0.07982976911113136,  -0.0897577847471601};
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}
/**
 * zdnn_tanh_positive_nhwc
 *
 * Positive test to demonstrate tanh
 *
 * Input values as NHWC sized (4, 1, 1, 1)
 *  [[
 *    [[0.01]],
 *    [[0.02]],
 *    [[0.03]],
 *    [[0.04]],
 *  ]]
 *
 * Expected Output values as NHWC sized (4, 1, 1, 1):
 *  [[
 *    [[0.00999966667999946]],
 *    [[0.019997333759930933]],
 *    [[0.029991003238820143]],
 *    [[0.03997868031116357]],
 *  ]]
 */
void zdnn_tanh_positive_nhwc_1() {
  uint32_t shape[] = {4, 1, 1, 1}; // Will be same for in and out dim.
  float input_values[] = {0.01, 0.02, 0.03, 0.04};
  float expected_values[] = {0.00999966667999946, 0.019997333759930933,
                             0.029991003238820143, 0.03997868031116357};
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_balanced_nhwc
 *
 * Balanced (pos and neg inputs) test to demonstrate tanh
 *
 * Input values as NHWC sized (1, 1, 2, 6)
 * [[
 *   [[-0.05, -0.04, -0.03, -0.02, -0.01, -0.00],
 *    [0.01,  0.02,  0.03,  0.04, 0.05,  0.06]]
 * ]]
 *
 * Expected Output values as NHWC sized (1, 1, 2, 6):
 * [[
 *   [[-0.04995837495787998, -0.03997868031116357, -0.029991003238820143,
 *     -0.019997333759930933, -0.00999966667999946, 0.0],
 *    [0.00999966667999946, 0.019997333759930933, 0.029991003238820143,
 *     0.03997868031116357, 0.04995837495787998, 0.059928103529143496]]
 * ]]
 */
void zdnn_tanh_balanced_nhwc_1() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 1, 2, 6}; // Will be same for in and out dim.
  float input_values[] = {-0.05, -0.04, -0.03, -0.02, -0.01, 0.0,
                          0.01,  0.02,  0.03,  0.04,  0.05,  0.06};
  float expected_values[] = {
      -0.04995837495787998,  -0.03997868031116357, -0.029991003238820143,
      -0.019997333759930933, -0.00999966667999946, 0.0,
      0.00999966667999946,   0.019997333759930933, 0.029991003238820143,
      0.03997868031116357,   0.04995837495787998,  0.059928103529143496};
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  TanH Basic
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_tanh_basic_3d
 *
 * Simple test to demonstrate tanh
 *
 * Input values as NWC sized (1,3,1):
 * [[
 *   [[0.01], [0.02], [0.03]],
 *   [[0.04], [0.05], [0.06]],
 *   [[0.07], [0.08], [0.09]]
 * ]]
 *
 * Expected Output values as NWC sized (1,3,1):
 * [[
 *   [[0.00999966667999946],  [0.019997333759930933], [0.029991003238820143]],
 *   [[0.03997868031116357], [0.04995837495787998], [0.059928103529143496]],
 *   [[0.06988589031642899], [0.07982976911113136], [0.0897577847471601]
 * ]]
 */
void zdnn_tanh_basic_3d_1() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};
  float expected_values[] = {
      0.00999966667999946, 0.019997333759930933, 0.029991003238820143,
      0.03997868031116357, 0.04995837495787998,  0.059928103529143496,
      0.06988589031642899, 0.07982976911113136,  0.0897577847471601};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_zeros_3d
 *
 * Zero test to demonstrate tanh
 *
 * Input values as NWC sized (1,3,3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 *
 * Expected Output values as NWC sized (1,3,3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 */
void zdnn_tanh_zeros_3d_1() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 3}; // Will be same for in and out dim.
  float input_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_negative_3d
 *
 * Negative test to demonstrate tanh
 *
 * Input values as NWC sized (1,3,3):
 * [[
 *   [[-0.01], [-0.02], [-0.03]],
 *   [[-0.04], [-0.05], [-0.06]],
 *   [[-0.07], [-0.08], [-0.09]]
 * ]]
 *
 * Expected Output values as NWC sized (1,3,3):
 * [[
 *   [[-0.00999966667999946],  [-0.019997333759930933],
 *    [-0.029991003238820143]],
 *   [[-0.03997868031116357], [-0.04995837495787998], [-0.059928103529143496]],
 *   [[-0.06988589031642899], [-0.07982976911113136], [-0.0897577847471601]
 * ]]
 */
void zdnn_tanh_negative_3d_1() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 3}; // Will be same for in and out dim.
  float input_values[] = {-0.01, -0.02, -0.03, -0.04, -0.05,
                          -0.06, -0.07, -0.08, -0.09};
  float expected_values[] = {
      -0.00999966667999946, -0.019997333759930933, -0.029991003238820143,
      -0.03997868031116357, -0.04995837495787998,  -0.059928103529143496,
      -0.06988589031642899, -0.07982976911113136,  -0.0897577847471601};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_positive_3d
 *
 * Positive test to demonstrate tanh
 *
 *
 * Input values as NWC sized (4, 1, 1)
 *  [[
 *    [[0.01]],
 *    [[0.02]],
 *    [[0.03]],
 *    [[0.04]],
 *  ]]
 *
 * Expected Output values as NWC sized (4, 1, 1):
 *  [[
 *    [[0.00999966667999946]],
 *    [[0.019997333759930933]],
 *    [[0.029991003238820143]],
 *    [[0.03997868031116357]],
 *  ]]
 */
void zdnn_tanh_positive_3d_1() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {4, 1, 1}; // Will be same for in and out dim.
  float input_values[] = {0.01, 0.02, 0.03, 0.04};
  float expected_values[] = {0.00999966667999946, 0.019997333759930933,
                             0.029991003238820143, 0.03997868031116357};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_balanced_3d
 *
 * Balanced (pos and neg inputs) test to demonstrate tanh
 *
 * Input values as NWC sized (1, 2, 6)
 * [[
 *   [[-0.05, -0.04, -0.03, -0.02, -0.01, -0.00],
 *    [0.01,  0.02,  0.03,  0.04, 0.05,  0.06]]
 * ]]
 *
 * Expected Output values as NWC sized (1, 2, 6):
 * [[
 *   [[-0.04995837495787998, -0.03997868031116357, -0.029991003238820143,
 *     -0.019997333759930933, -0.00999966667999946, 0.0],
 *    [0.00999966667999946, 0.019997333759930933, 0.029991003238820143,
 *     0.03997868031116357, 0.04995837495787998, 0.059928103529143496]]
 * ]]
 *
 */
void zdnn_tanh_balanced_3d_1() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 2, 6}; // Will be same for in and out dim.
  float input_values[] = {-0.05, -0.04, -0.03, -0.02, -0.01, 0.0,
                          0.01,  0.02,  0.03,  0.04,  0.05,  0.06};
  float expected_values[] = {
      -0.04995837495787998,  -0.03997868031116357, -0.029991003238820143,
      -0.019997333759930933, -0.00999966667999946, 0.0,
      0.00999966667999946,   0.019997333759930933, 0.029991003238820143,
      0.03997868031116357,   0.04995837495787998,  0.059928103529143496};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  TanH Basic
                                Layout: NHWC
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_tanh_basic_nhwc
 *
 * Simple test to demonstrate tanh
 *
 * Input values as NHWC sized (1,3,3,1):
 * [[
 *   [[1], [2], [3]],
 *   [[4], [5], [6]],
 *   [[7], [8], [9]]
 * ]]
 *
 * Expected Output values as NHWC sized (1,3,3,1):
 * [[
 *   [[0.761594156],  [0.9640275801], [0.9950547537]],
 *   [[0.9993292997], [0.9999092043], [0.9999877117]],
 *   [[0.9999983369], [0.9999997749], [0.9999999695]
 * ]]
 */
void zdnn_tanh_basic_nhwc_2() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float expected_values[] = {
      0.761594156,  0.9640275801, 0.9950547537, 0.9993292997, 0.9999092043,
      0.9999877117, 0.9999983369, 0.9999997749, 0.9999999695,
  };
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_zeros_nhwc
 *
 * Zero test to demonstrate tanh
 *
 * Input values as NHWC sized (1, 3, 3, 3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 *
 * Expected Output values as NHWC sized (1, 3, 3, 3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 */
void zdnn_tanh_zeros_nhwc_2() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 3}; // Will be same for in and out dim.
  float input_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_negative_nhwc
 *
 * Negative test to demonstrate tanh
 *
 * Input values as NHWC sized (1, 3, 3, 1):
 *  [[
 *    [[-1], [-2], [-3]],
 *    [[-4], [-5], [-6]],
 *    [[-7], [-8], [-9]]
 *  ]]
 *
 * Expected Output values as NHWC sized (1, 3, 3, 1):
 *  [[
 *    [[-0.761594156], [-0.9640275801], [-0.9950547537]],
 *    [[-0.9993292997], [-0.9999092043], [-0.9999877117]],
 *    [[-0.9999983369], [-0.9999997749], [-0.9999999695]]
 *  ]]
 */
void zdnn_tanh_negative_nhwc_2() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {-1, -2, -3, -4, -5, -6, -7, -8, -9};
  float expected_values[] = {
      -0.761594156,  -0.9640275801, -0.9950547537, -0.9993292997, -0.9999092043,
      -0.9999877117, -0.9999983369, -0.9999997749, -0.9999999695,
  };
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_positive_nhwc
 *
 * Positive test to demonstrate tanh
 *
 * Input values as NHWC sized (9, 1, 1, 1)
 *  [[
 *    [[1]],
 *    [[2]],
 *    [[3]],
 *    [[4]],
 *    [[5]],
 *    [[6]],
 *    [[7]],
 *    [[8]],
 *    [[9]],
 *  ]]
 *
 * Expected Output values as NHWC sized (9, 1, 1, 1):
 *  [[
 *    [[0.761594156]],
 *    [[0.9640275801]],
 *    [[0.9950547537]],
 *    [[0.9993292997]],
 *    [[0.9999092043]],
 *    [[0.9999877117]],
 *    [[0.9999983369]],
 *    [[0.9999997749]],
 *    [[0.9999999695]],
 *  ]]
 */
void zdnn_tanh_positive_nhwc_2() {
  uint32_t shape[] = {9, 1, 1, 1}; // Will be same for in and out dim.
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float expected_values[] = {
      0.761594156,  0.9640275801, 0.9950547537, 0.9993292997, 0.9999092043,
      0.9999877117, 0.9999983369, 0.9999997749, 0.9999999695,
  };
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_balanced_nhwc
 *
 * Balanced (pos and neg inputs) test to demonstrate tanh
 *
 * Input values as NHWC sized (1, 1, 3, 5)
 * [[
 *   [[-4, -2, 0, 2, 4], [-3, -1, 0, 1, 3], [-8, -6, 0, 6, 8]]
 * ]]
 *
 * Expected Output values as NHWC sized (1, 1, 3, 5):
 * [[
 *   [[ -0.9993292997, -0.9640275801, 0.0, 0.9640275801, 0.9993292997],
 *    [-0.9950547537, -0.761594156,  0.0, 0.761594156,  0.9950547537],
 *    [-0.9999997749, -0.9999877117, 0.0, 0.9999877117, 0.9999997749]]
 * ]]
 */
void zdnn_tanh_balanced_nhwc_2() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 1, 3, 5}; // Will be same for in and out dim.
  float input_values[] = {-4, -2, 0, 2, 4, -3, -1, 0, 1, 3, -8, -6, 0, 6, 8};
  float expected_values[] = {
      -0.9993292997, -0.9640275801, 0.0, 0.9640275801, 0.9993292997,
      -0.9950547537, -0.761594156,  0.0, 0.761594156,  0.9950547537,
      -0.9999997749, -0.9999877117, 0.0, 0.9999877117, 0.9999997749,
  };
  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  TanH Basic
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_tanh_basic_3d
 *
 * Simple test to demonstrate tanh
 *
 * Input values as NWC sized (1,3,1):
 * [[
 *   [[1], [2], [3]],
 *   [[4], [5], [6]],
 *   [[7], [8], [9]]
 * ]]
 *
 * Expected Output values as NWC sized (1,3,1):
 * [[
 *   [[0.761594156],  [0.9640275801], [0.9950547537]],
 *   [[0.9993292997], [0.9999092043], [0.9999877117]],
 *   [[0.9999983369], [0.9999997749], [0.9999999695]
 * ]]
 */
void zdnn_tanh_basic_3d_2() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  float expected_values[] = {
      0.761594156,  0.9640275801, 0.9950547537, 0.9993292997, 0.9999092043,
      0.9999877117, 0.9999983369, 0.9999997749, 0.9999999695,
  };
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_zeros_3d
 *
 * Zero test to demonstrate tanh
 *
 * Input values as NWC sized (1,3,3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 *
 * Expected Output values as NWC sized (1,3,3):
 * [[
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]],
 *   [[0,0,0], [0,0,0], [0,0,0]]
 * ]]
 */
void zdnn_tanh_zeros_3d_2() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 3}; // Will be same for in and out dim.
  float input_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_negative_3d
 *
 * Negative test to demonstrate tanh
 *
 * Input values as NWC sized (1,3,3):
 *  [[
 *    [[-1.0], [-2.1], [-3.2]],
 *    [[-4.3], [-5.4], [-6.5]],
 *    [[-7.6], [-8.7], [-9.8]]
 *  ]]
 *
 * Expected Output values as NWC sized (1,3,3):
 *  [[
 *    [[-0.761594156], [-0.9704519366], [-0.9966823978]],
 *    [[-0.9996318562], [-0.9999592018], [-0.9999954794]],
 *    [[-0.9999994991], [-0.9999999445], [-0.9999999939]]
 *  ]]
 */
void zdnn_tanh_negative_3d_2() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 3}; // Will be same for in and out dim.
  float input_values[] = {-1.0, -2.1, -3.2, -4.3, -5.4, -6.5, -7.6, -8.7, -9.8};
  float expected_values[] = {
      -0.761594156,  -0.9704519366, -0.9966823978, -0.9996318562, -0.9999592018,
      -0.9999954794, -0.9999994991, -0.9999999445, -0.9999999939,
  };
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_positive_3d
 *
 * Positive test to demonstrate tanh
 *
 *
 *  * Input values as NWC sized (8, 1, 1)
 *  [[
 *    [[1.0]],
 *    [[2.1]],
 *    [[3.2]],
 *    [[4.3]],
 *    [[5.4]],
 *    [[6.5]],
 *    [[7.6]],
 *    [[8.7]]
 *  ]]
 *
 * Expected Output values as NWC sized (8, 1, 1):
 *  [[
 *    [[0.761594156]],
 *    [[0.9704519366]],
 *    [[0.9966823978]],
 *    [[0.9996318562]],
 *    [[0.9999592018]],
 *    [[0.9999954794]],
 *    [[0.9999994991]],
 *    [[0.9999999445]]
 *  ]]
 */
void zdnn_tanh_positive_3d_2() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {8, 1, 1}; // Will be same for in and out dim.
  float input_values[] = {1.0, 2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7};
  float expected_values[] = {0.761594156,  0.9704519366, 0.9966823978,
                             0.9996318562, 0.9999592018, 0.9999954794,
                             0.9999994991, 0.9999999445};
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_balanced_3d
 *
 * Balanced (pos and neg inputs) test to demonstrate tanh
 *
 * Input values as NWC sized (1, 3, 5)
 * [[
 *   [[-4, -2, 0, 2, 4], [-3, -1, 0, 1, 3], [-8, -6, 0, 6, 8]]
 * ]]
 *
 * Expected Output values as NWC sized (1 3, 5):
 * [[
 *   [[ -0.9993292997, -0.9640275801, 0.0, 0.9640275801, 0.9993292997],
 *    [-0.9950547537, -0.761594156,  0.0, 0.761594156,  0.9950547537],
 *    [-0.9999997749, -0.9999877117, 0.0, 0.9999877117, 0.9999997749]]
 * ]]
 *
 */
void zdnn_tanh_balanced_3d_2() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {1, 3, 5}; // Will be same for in and out dim.
  float input_values[] = {-4, -2, 0, 2, 4, -3, -1, 0, 1, 3, -8, -6, 0, 6, 8};
  float expected_values[] = {
      -0.9993292997, -0.9640275801, 0.0, 0.9640275801, 0.9993292997,
      -0.9950547537, -0.761594156,  0.0, 0.761594156,  0.9950547537,
      -0.9999997749, -0.9999877117, 0.0, 0.9999877117, 0.9999997749,
  };
  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  TANH Large
                                Layout: NCHW
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_tanh_basic_nhwc_large
 *
 * - ZDNN_3D

 * Simple test of positive input.
 *
 * Generate a test that is of size 40x30x15x1
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 40x30x15x1.
 */
void zdnn_tanh_basic_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 15, 30, 43}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[3] * shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_zeros_nhwc_large
 *
 * Simple test of all zero input.
 *
 * Generate a test that is of size 80x40x20x1
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 80x40x20x1
 */
void zdnn_tanh_zeros_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 20, 40, 80}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[3] * shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  fill_all_with_zero_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_negative_nhwc_large
 *
 * Simple test of all negative input values.
 *
 * Generate a test that is of size 80x23x10x1
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 80x23x10x1
 */
void zdnn_tanh_negative_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 10, 28, 83}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[3] * shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_balanced_nhwc_large
 *
 * Simple test of half negative and positive inputs.
 *
 * Generate a test that is of size 56x12x10x1
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 56x12x10x1
 */
void zdnn_tanh_balanced_nhwc_large() {
  // Initialize the dimensions for our input tensor ZDNN_NHWC
  uint32_t shape[] = {1, 10, 12, 56}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[3] * shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  gen_random_float_array_pos_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  TANH Large
                                Layout: ZDNN_3D
  -------------------------------------------------------------------------------
*/

/**
 * zdnn_tanh_basic_3d_large
 *
 * Simple test of positive input.
 *
 * Generate a test that is of size 10x10x10.
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 10x10x10.
 */
void zdnn_tanh_basic_3d_large() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {10, 10, 10}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_zeros_3d_large
 *
 * Simple test of all zero input.
 *
 * Generate a test that is of size 15x5x3
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 15x5x3
 */
void zdnn_tanh_zeros_3d_large() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {3, 5, 13}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  fill_all_with_zero_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_negative_3d_large
 *
 * Simple test of all negative input values.
 *
 * Generate a test that is of size 20x15x10
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 20x15x10
 */
void zdnn_tanh_negative_3d_large() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {20, 15, 10}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/**
 * zdnn_tanh_balanced_3d_large
 *
 * Simple test of half negative and positive inputs.
 *
 * Generate a test that is of size 30x3x3
 * and use automatic float generator to create
 * input values.
 *
 * Output will contain tensor of size size 30x3x3
 */
void zdnn_tanh_balanced_3d_large() {
  // Initialize the dimensions for our input tensor ZDNN_3D
  uint32_t shape[] = {3, 3, 30}; // Will be same for in and out dim.

  int num_io_buffer_values = shape[2] * shape[1] * shape[0];

  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  act_tanh(input_values, expected_values, num_io_buffer_values);

  zdnn_tanh_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST_ALL_DATATYPES(zdnn_tanh_basic_nhwc_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_zeros_nhwc_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_negative_nhwc_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_positive_nhwc_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_balanced_nhwc_1);

  RUN_TEST_ALL_DATATYPES(zdnn_tanh_basic_3d_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_zeros_3d_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_negative_3d_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_positive_3d_1);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_balanced_3d_1);

  RUN_TEST_ALL_DATATYPES(zdnn_tanh_basic_nhwc_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_zeros_nhwc_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_negative_nhwc_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_positive_nhwc_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_balanced_nhwc_2);

  RUN_TEST_ALL_DATATYPES(zdnn_tanh_basic_nhwc_large);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_zeros_nhwc_large);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_negative_nhwc_large);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_balanced_nhwc_large);

  RUN_TEST_ALL_DATATYPES(zdnn_tanh_basic_3d_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_zeros_3d_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_negative_3d_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_positive_3d_2);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_balanced_3d_2);

  RUN_TEST_ALL_DATATYPES(zdnn_tanh_basic_3d_large);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_zeros_3d_large);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_negative_3d_large);
  RUN_TEST_ALL_DATATYPES(zdnn_tanh_balanced_3d_large);

  return UNITY_END();
}
