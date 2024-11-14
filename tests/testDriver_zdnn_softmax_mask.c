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

// -----------------------------------------------------------------------------
// SoftmaxMask Unit Testing, for convenience, recall the following:
//     softmax(x) -> [0,1]
//     For some value x, we squash that  value to some real-valued number within
//     range [0,1] -- all components will indeed add up to one, this is mainly
//     so thar they can be interpreted as probabilities.
//     For the behind the scenes:
//          softmax(x) -> ( e(x) /  e(x)  +e(x)    +...+e(x)      +e(x) )
//                                   sub 1    sub 2        sub n-1    sub n
//          https://en.wikipedia.org/wiki/Softmax_function
//
//     When mask > 0, only inidices (0, mask) are used for the righ-most dim.
// -----------------------------------------------------------------------------

void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

/**
 * zdnn_softmax_mask_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_softmax_mask_test(uint32_t *shape, zdnn_data_layouts layout,
                            float *input, zdnn_softmax_act act_func,
                            uint32_t softmax_mask, zdnn_status expected_status,
                            float *expected_values) {
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

  zdnn_status status;

  /*
   * Begin Testing!
   */

  /* once with NULL workarea, once with self-allocated */

  status = zdnn_softmax_mask(input_ztensor, NULL, act_func, softmax_mask,
                             output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_softmax_mask() with activation function %d and mask %u "
      "returned status %08x but expected %08x\n",
      act_func, softmax_mask, status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  zdnn_reset_ztensor(output_ztensor);

  void *self_workarea = malloc_aligned_4k(ZDNN_SOFTMAX_SAVEAREA_SIZE);
  TEST_ASSERT_MESSAGE_FORMATTED(
      self_workarea, "%s() - can't allocate SOFTMAX workarea\n", __func__);

  status = zdnn_softmax_mask(input_ztensor, self_workarea, act_func,
                             softmax_mask, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_softmax_mask() with activation function %d, mask %u, and "
      "provided work_area returned status %08x but expected %08x\n",
      act_func, softmax_mask, status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }
  free_aligned_4k(self_workarea);
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
  ------------------------------------------------------------------------------
                                SoftmaxMask Basic
                                Layout: 3DS
  ------------------------------------------------------------------------------
*/

/**
 * zdnn_softmax_mask_basic_3ds_
 *
 * Simple test of all positive input values
 * Expect a mirror of the Input values as the Output values
 *
 * Input values as 3DS
 *  [[
 *    [[0.5], [1.0], [1.5]],
 *    [[2.0], [2.5], [3.0]],
 *    [[3.5], [4.0], [4.5]]
 *  ]]
 *
 * Expected Output values as 3DS with no activation
 *  [[
 *    [[1.0], [1.0], [1.0]],
 *    [[1.0], [1.0], [1.0]],
 *    [[1.0], [1.0], [1.0]]
 *  ]]
 *
 * Expected Output values as 3DS with log activation
 *  [[
 *    [[0.0], [0.0], [0.0]],
 *    [[0.0], [0.0], [0.0]],
 *    [[0.0], [0.0], [0.0]]
 *  ]]
 */
void zdnn_softmax_mask_basic_3ds() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
  float expected_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Mask of 0 will use all of dimension 1
  uint32_t softmax_mask = 0;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_basic_3ds_1() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
  float expected_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 1;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

/*
  ------------------------------------------------------------------------------
                                SoftmaxMask Basic
                                Layout: 3DS
  ------------------------------------------------------------------------------
*/

/**
 * zdnn_softmax_mask_balanced_3ds_
 *
 * Balanced (pos and neg inputs) test to demonstrate softmax
 *
 * Input values as 3DS
 *  [[
 *    [[-2, -1.5], [-1, -0.5]],
 *    [[0.5, 1.0], [1.5, 2.0]],
 *  ]]
 *
 * Expected Output values as 3DS with no activation
 *  [[
 *    [[0.37754068, 0.62245935], [0.37754068, 0.62245935]],
 *    [[0.37754068, 0.62245935], [0.37754068, 0.62245935]],
 *  ]]
 *
 * Expected Output values as 3DS with log activation
 *  [[
 *    [[-0.974077   -0.47407693], [-0.974077   -0.47407693]]
 *    [[-0.974077   -0.47407693], [-0.974077   -0.47407693]]
 *  ]]
 */
void zdnn_softmax_mask_balanced_3ds() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {2, 2, 2}; // Will be same for in and out dim.
  float input_values[] = {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2};
  float expected_values[] = {0.37754068, 0.62245935, 0.37754068, 0.62245935,
                             0.37754068, 0.62245935, 0.37754068, 0.62245935};
  float log_expected_values[] = {-0.974077,   -0.47407693, -0.974077,
                                 -0.47407693, -0.974077,   -0.47407693,
                                 -0.974077,   -0.47407693};

  // Mask of 0 will use all of dimension 1
  uint32_t softmax_mask = 0;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_balanced_3ds_1() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {2, 2, 2}; // Will be same for in and out dim.
  float input_values[] = {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2};
  float expected_values[] = {1.0, 0, 1.0, 0, 1.0, 0, 1.0, 0};
  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 1;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_balanced_3ds_2() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {2, 2, 2}; // Will be same for in and out dim.
  float input_values[] = {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2};
  float expected_values[] = {0.37754068, 0.62245935, 0.37754068, 0.62245935,
                             0.37754068, 0.62245935, 0.37754068, 0.62245935};
  float log_expected_values[] = {-0.974077,   -0.47407693, -0.974077,
                                 -0.47407693, -0.974077,   -0.47407693,
                                 -0.974077,   -0.47407693};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 2;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

/*
  ------------------------------------------------------------------------------
                                SoftmaxMask Basic
                                Layout: ZDNN_3D
  ------------------------------------------------------------------------------
*/

/**
 * zdnn_softmax_mask_negative_3ds_
 *
 * Negative test to demonstrate tanh
 *
 * Input values as NWC sized (1,1,8):
 *  [[
 *    [[-1.4, -2.8, -3.12, -4.16, -5.20, -6.24, -7.28, -8.32]],
 *  ]]
 *
 * Expected Output values as NWC sized (1,1,8) with no activation:
 *  [[
 *    [[0.656592, 0.161914, 0.117573,
 *      0.041557, 0.014688, 0.005192,
 *      0.001835 , 0.000649]],
 *  ]]
 *
 * Expected Output values as NWC sized (1,1,8) with log activation:
 *  [[
 *    [[-0.42069218, -1.8206921, -2.140692,
 *      -3.180692, -4.2206917, -5.260692,
 *      -6.300692, -7.3406916]],
 *  ]]
 *

 */
void zdnn_softmax_mask_negative_3ds() {
  // Initialize the dimensions for our input tensor--ZDNN_3DS [C,W,N]
  uint32_t shape[] = {1, 1, 8}; // Will be same for in and out dim.
  float input_values[] = {-1.4, -2.8, -3.12, -4.16, -5.20, -6.24, -7.28, -8.32};
  float expected_values[] = {0.656592, 0.161914, 0.117573, 0.041557,
                             0.014688, 0.005192, 0.001835, 0.000649};
  float log_expected_values[] = {-0.42069218, -1.8206921, -2.140692,
                                 -3.180692,   -4.2206917, -5.260692,
                                 -6.300692,   -7.3406916};

  // Mask of 0 will use all of dimension 1
  uint32_t softmax_mask = 0;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_negative_3ds_1() {
  // Initialize the dimensions for our input tensor--ZDNN_3DS [C,W,N]
  uint32_t shape[] = {1, 1, 8}; // Will be same for in and out dim.
  float input_values[] = {-1.4, -2.8, -3.12, -4.16, -5.20, -6.24, -7.28, -8.32};
  float expected_values[] = {1.0, 0, 0, 0, 0, 0, 0, 0};
  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 1;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_negative_3ds_2() {
  // Initialize the dimensions for our input tensor--ZDNN_3DS [C,W,N]
  uint32_t shape[] = {1, 1, 8}; // Will be same for in and out dim.
  float input_values[] = {-1.4, -2.8, -3.12, -4.16, -5.20, -6.24, -7.28, -8.32};
  float expected_values[] = {0.802184, 0.197816, 0, 0, 0, 0, 0, 0};
  float log_expected_values[] = {-0.220417, -1.620417, 0, 0, 0, 0, 0, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 2;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_negative_3ds_3() {
  // Initialize the dimensions for our input tensor--ZDNN_3DS [C,W,N]
  uint32_t shape[] = {1, 1, 8}; // Will be same for in and out dim.
  float input_values[] = {-1.4, -2.8, -3.12, -4.16, -5.20, -6.24, -7.28, -8.32};
  float expected_values[] = {0.701172, 0.172852, 0.125488, 0, 0, 0, 0, 0};
  float log_expected_values[] = {-0.354492, -1.753906, -2.074219, 0,
                                 0,         0,         0,         0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 3;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

/*
  ------------------------------------------------------------------------------
                                SoftmaxMask Large
                                Layout: 3DS
  ------------------------------------------------------------------------------
*/

/**
 * zdnn_softmax_mask_basic_3ds_large
 *
 * Simple test of all positive input values
 * Expect a mirror of the Input values as the Output values
 *
 * Input values as 3DS
 *  [[
 *    [[0.65536, 0.65100, 0.65200],
 *     [0.64000, 0.64100, 0.64200],
 *     [0.63000, 0.63100, 0.63200]],
 *    [[0.62000, 0.62100, 0.62200],
 *     [0.61000, 0.61100, 0.61200],
 *     [0.60000, 0.60100, 0.60200]],
 *    [[0.59000, 0.59100, 0.59200],
 *     [0.58000, 0.58100, 0.58200],
 *     [0.57000, 0.57100, 0.57200]]
 *  ]]
 *
 * Expected Output values as 3DS with no activation
 *  [[
 *     [[0.33419162, 0.3327377, 0.33307064]
 *      [0.33300006, 0.33333322, 0.33366674]
 *      [0.33300006, 0.33333322, 0.33366674]]
 *     [[0.33300006, 0.3333332, 0.3336667]
 *      [0.33300006, 0.3333332, 0.3336667]
 *      [0.33300006, 0.3333332, 0.3336667]]
 *     [[0.33300003, 0.3333332, 0.3336667]
 *      [0.33300006, 0.33333322, 0.33366674]
 *      [0.33300006, 0.33333322, 0.33366674]]
 *  ]]
 *
 * Expected Output values as 3DS with log activation
 *  [[
 *     [[-1.0960407 -1.1004007 -1.0994008]
 *      [-1.0996126 -1.0986125 -1.0976126]
 *      [-1.0996126 -1.0986125 -1.0976126]]
 *     [[-1.0996126 -1.0986127 -1.0976126]
 *      [-1.0996126 -1.0986127 -1.0976126]
 *      [-1.0996126 -1.0986127 -1.0976126]]
 *     [[-1.0996127 -1.0986127 -1.0976126]
 *      [-1.0996126 -1.0986125 -1.0976126]
 *      [-1.0996126 -1.0986127 -1.0976126]]
 *  ]]
 */
void zdnn_softmax_mask_basic_3ds_large() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 3};

  float input_values[] = {0.65536, 0.65100, 0.65200, 0.64000, 0.64100, 0.64200,
                          0.63000, 0.63100, 0.63200, 0.62000, 0.62100, 0.62200,
                          0.61000, 0.61100, 0.61200, 0.60000, 0.60100, 0.60200,
                          0.59000, 0.59100, 0.59200, 0.58000, 0.58100, 0.58200,
                          0.57000, 0.57100, 0.57200};

  float expected_values[] = {
      0.33419162, 0.3327377,  0.33307064, 0.33300006, 0.33333322, 0.33366674,
      0.33300006, 0.33333322, 0.33366674, 0.33300006, 0.3333332,  0.3336667,
      0.33300006, 0.3333332,  0.3336667,  0.33300006, 0.3333332,  0.3336667,
      0.33300003, 0.3333332,  0.3336667,  0.33300006, 0.33333322, 0.33366674,
      0.33300006, 0.33333322, 0.33366674};

  float log_expected_values[] = {
      -1.0960407, -1.1004007, -1.0994008, -1.0996126, -1.0986125, -1.0976126,
      -1.0996126, -1.0986125, -1.0976126, -1.0996126, -1.0986127, -1.0976126,
      -1.0996126, -1.0986127, -1.0976126, -1.0996126, -1.0986127, -1.0976126,
      -1.0996127, -1.0986127, -1.0976126, -1.0996126, -1.0986125, -1.0976126,
      -1.0996126, -1.0986127, -1.0976126};

  // Mask of 0 will use all of dimension 1
  uint32_t softmax_mask = 0;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_basic_3ds_large_1() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 3};

  float input_values[] = {0.65536, 0.65100, 0.65200, 0.64000, 0.64100, 0.64200,
                          0.63000, 0.63100, 0.63200, 0.62000, 0.62100, 0.62200,
                          0.61000, 0.61100, 0.61200, 0.60000, 0.60100, 0.60200,
                          0.59000, 0.59100, 0.59200, 0.58000, 0.58100, 0.58200,
                          0.57000, 0.57100, 0.57200};

  float expected_values[] = {1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0,
                             1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0,
                             1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0};

  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 1;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_basic_3ds_large_2() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 3};

  float input_values[] = {0.65536, 0.65100, 0.65200, 0.64000, 0.64100, 0.64200,
                          0.63000, 0.63100, 0.63200, 0.62000, 0.62100, 0.62200,
                          0.61000, 0.61100, 0.61200, 0.60000, 0.60100, 0.60200,
                          0.59000, 0.59100, 0.59200, 0.58000, 0.58100, 0.58200,
                          0.57000, 0.57100, 0.57200};

  float expected_values[] = {
      0.500977, 0.499023, 0, 0.500000, 0.500000, 0, 0.500000, 0.500000, 0,
      0.500000, 0.500000, 0, 0.500000, 0.500000, 0, 0.500000, 0.500000, 0,
      0.500000, 0.500000, 0, 0.500000, 0.500000, 0, 0.500000, 0.500000, 0};

  float log_expected_values[] = {
      -0.691406, -0.695312, 0, -0.693359, -0.692383, 0,
      -0.693359, -0.692383, 0, -0.693359, -0.692383, 0,
      -0.693359, -0.692383, 0, -0.693359, -0.692383, 0,
      -0.693359, -0.692383, 0, -0.693359, -0.692383, 0,
      -0.693359, -0.692383, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 2;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_basic_3ds_large_3() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 3};

  float input_values[] = {0.65536, 0.65100, 0.65200, 0.64000, 0.64100, 0.64200,
                          0.63000, 0.63100, 0.63200, 0.62000, 0.62100, 0.62200,
                          0.61000, 0.61100, 0.61200, 0.60000, 0.60100, 0.60200,
                          0.59000, 0.59100, 0.59200, 0.58000, 0.58100, 0.58200,
                          0.57000, 0.57100, 0.57200};

  float expected_values[] = {
      0.33419162, 0.3327377,  0.33307064, 0.33300006, 0.33333322, 0.33366674,
      0.33300006, 0.33333322, 0.33366674, 0.33300006, 0.3333332,  0.3336667,
      0.33300006, 0.3333332,  0.3336667,  0.33300006, 0.3333332,  0.3336667,
      0.33300003, 0.3333332,  0.3336667,  0.33300006, 0.33333322, 0.33366674,
      0.33300006, 0.33333322, 0.33366674};

  float log_expected_values[] = {
      -1.0960407, -1.1004007, -1.0994008, -1.0996126, -1.0986125, -1.0976126,
      -1.0996126, -1.0986125, -1.0976126, -1.0996126, -1.0986127, -1.0976126,
      -1.0996126, -1.0986127, -1.0976126, -1.0996126, -1.0986127, -1.0976126,
      -1.0996127, -1.0986127, -1.0976126, -1.0996126, -1.0986125, -1.0976126,
      -1.0996126, -1.0986127, -1.0976126};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 3;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

/*
  ------------------------------------------------------------------------------
                                SoftmaxMask Large
                                Layout: 3DS
  ------------------------------------------------------------------------------
*/

/**
 * zdnn_softmax_mask_balanced_3ds_large
 *
 * Input values as 3DS
 *  [[[ 0.9356609 ,  1.0854305 , -0.93788373],
 *    [-0.5061547 ,  1.3169702 ,  0.7137579 ]],
 *    [[-0.4126717 , -0.40257987,  2.0713255 ],
 *    [-0.35911667,  0.3861619 ,  1.9897066 ]],
 *    [[-0.2823396 , -0.5135972 , -0.8962833 ],
 *    [-0.0901652 , -0.73964226, -0.46269894]],
 *    [[ 0.42379895,  1.1180195 ,  1.4442351 ],
 *    [-1.0771092 ,  0.9014347 , -0.14529487]],
 *    [[ 1.173365  ,  1.510687  , -0.46714714],
 *    [ 1.3281798 ,  1.7365712 , -1.5435543 ]],
 *    [[ 0.35064182,  0.5708492 , -1.8452454 ],
 *    [ 0.9243176 ,  0.57233644, -1.0959795 ]],
 *    [[-0.62557054,  0.686686  ,  0.4222773 ],
 *    [-0.2146352 , -0.81243026, -1.1678637 ]],
 *    [[ 1.6384528 ,  1.187959  , -2.5538385 ],
 *    [-0.39338952,  0.233341  , -1.6181145 ]],
 *    [[-0.8736809 ,  0.05150718,  2.2328985 ],
 *    [ 2.8749912 ,  0.08306922, -0.9871888 ]],
 *    [[ 0.47143334, -1.7806206 , -0.27681163],
 *    [-0.9240901 ,  1.3088665 ,  0.7826533 ]]]
 *
 * Expected Output values as 3DS with no activation
 * [[
 *    [[0.43193838, 0.5017252,  0.06633637],
 *    [0.09453523, 0.5852842,  0.32018057]],
 *    [[0.07143247, 0.07215702, 0.85641056],
 *    [0.07363626, 0.15515368, 0.7712101 ]],
 *    [[0.42831188, 0.3398805,  0.23180765],
 *    [0.45222163, 0.23620388, 0.31157458]],
 *    [[0.17311363, 0.3465991,  0.48028725],
 *    [0.09283915, 0.67143184, 0.23572904]],
 *    [[0.38534594, 0.5399429,  0.07471115],
 *    [0.390473,   0.58742595, 0.02210104]],
 *    [[0.42416108, 0.5286468,  0.04719208],
 *    [0.5446892,  0.38307628, 0.07223454]],
 *    [[0.13216929, 0.49094895, 0.37688172],
 *    [0.51665765, 0.28417364, 0.19916865]],
 *    [[0.6051712,  0.3856837,  0.00914512],
 *    [0.31592378, 0.5912456,  0.09283058]],
 *    [[0.03865956, 0.09751265, 0.86382776],
 *    [0.9239366,  0.05664035, 0.01942311]],
 *    [[0.6335613,  0.06663986, 0.29979888],
 *    [0.06313774, 0.5889111,  0.34795114]]
 *  ]]
 *
 * Expected Output values as 3DS with log activation
 * [[
 *    [[-0.83947235 -0.68970275 -2.713017  ]
 *    [-2.3587828  -0.53565776 -1.1388701 ]]
 *    [[-2.6390028  -2.6289108  -0.1550054 ]
 *    [-2.6086178  -1.8633392  -0.25979444]]
 *    [[-0.84790367 -1.0791612  -1.4618473 ]
 *    [-0.79358286 -1.4430599  -1.1661166 ]]
 *    [[-1.7538071  -1.0595865  -0.7333709 ]
 *    [-2.3768868  -0.39834276 -1.4450723 ]]
 *    [[-0.9536138  -0.6162919  -2.594126  ]
 *    [-0.9403964  -0.5320051  -3.8121307 ]]
 *    [[-0.857642   -0.6374347  -3.0535293 ]
 *    [-0.60753995 -0.9595212  -2.627837  ]]
 *    [[-2.0236716  -0.7114151  -0.9758239 ]
 *    [-0.6603748  -1.2581699  -1.6136034 ]]
 *    [[-0.5022439  -0.9527377  -4.6945353 ]
 *    [-1.1522543  -0.5255238  -2.376979  ]]
 *    [[-3.2529612  -2.327773   -0.14638188]
 *    [-0.07911182 -2.8710337  -3.9412918 ]]
 *    [[-0.4563985  -2.7084525  -1.2046435 ]
 *    [-2.7624366  -0.52948    -1.0556931 ]]]
 *  ]]
 */
void zdnn_softmax_mask_balanced_3ds_large() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {10, 2, 3}; // Will be same for in and out dim.

  float input_values[] = {
      0.9356609,   1.0854305,   -0.93788373, -0.5061547,  1.3169702,
      0.7137579,   -0.4126717,  -0.40257987, 2.0713255,   -0.35911667,
      0.3861619,   1.9897066,   -0.2823396,  -0.5135972,  -0.8962833,
      -0.0901652,  -0.73964226, -0.46269894, 0.42379895,  1.1180195,
      1.4442351,   -1.0771092,  0.9014347,   -0.14529487, 1.173365,
      1.510687,    -0.46714714, 1.3281798,   1.7365712,   -1.5435543,
      0.35064182,  0.5708492,   -1.8452454,  0.9243176,   0.57233644,
      -1.0959795,  -0.62557054, 0.686686,    0.4222773,   -0.2146352,
      -0.81243026, -1.1678637,  1.6384528,   1.187959,    -2.5538385,
      -0.39338952, 0.233341,    -1.6181145,  -0.8736809,  0.05150718,
      2.2328985,   2.8749912,   0.08306922,  -0.9871888,  0.47143334,
      -1.7806206,  -0.27681163, -0.9240901,  1.3088665,   0.7826533};

  float expected_values[] = {
      0.43193838, 0.5017252,  0.06633637, 0.09453523, 0.5852842,  0.32018057,
      0.07143247, 0.07215702, 0.85641056, 0.07363626, 0.15515368, 0.7712101,
      0.42831188, 0.3398805,  0.23180765, 0.45222163, 0.23620388, 0.31157458,
      0.17311363, 0.3465991,  0.48028725, 0.09283915, 0.67143184, 0.23572904,
      0.38534594, 0.5399429,  0.07471115, 0.390473,   0.58742595, 0.02210104,
      0.42416108, 0.5286468,  0.04719208, 0.5446892,  0.38307628, 0.07223454,
      0.13216929, 0.49094895, 0.37688172, 0.51665765, 0.28417364, 0.19916865,
      0.6051712,  0.3856837,  0.00914512, 0.31592378, 0.5912456,  0.09283058,
      0.03865956, 0.09751265, 0.86382776, 0.9239366,  0.05664035, 0.01942311,
      0.6335613,  0.06663986, 0.29979888, 0.06313774, 0.5889111,  0.34795114};

  float log_expected_values[] = {
      -0.83947235, -0.68970275, -2.713017,   -2.3587828,  -0.53565776,
      -1.1388701,  -2.6390028,  -2.6289108,  -0.1550054,  -2.6086178,
      -1.8633392,  -0.25979444, -0.84790367, -1.0791612,  -1.4618473,
      -0.79358286, -1.4430599,  -1.1661166,  -1.7538071,  -1.0595865,
      -0.7333709,  -2.3768868,  -0.39834276, -1.4450723,  -0.9536138,
      -0.6162919,  -2.594126,   -0.9403964,  -0.5320051,  -3.8121307,
      -0.857642,   -0.6374347,  -3.0535293,  -0.60753995, -0.9595212,
      -2.627837,   -2.0236716,  -0.7114151,  -0.9758239,  -0.6603748,
      -1.2581699,  -1.6136034,  -0.5022439,  -0.9527377,  -4.6945353,
      -1.1522543,  -0.5255238,  -2.376979,   -3.2529612,  -2.327773,
      -0.14638188, -0.07911182, -2.8710337,  -3.9412918,  -0.4563985,
      -2.7084525,  -1.2046435,  -2.7624366,  -0.52948,    -1.0556931};

  // Mask of 0 will use all of dimension 1
  uint32_t softmax_mask = 0;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_balanced_3ds_large_1() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {10, 2, 3}; // Will be same for in and out dim.

  float input_values[] = {
      0.9356609,   1.0854305,   -0.93788373, -0.5061547,  1.3169702,
      0.7137579,   -0.4126717,  -0.40257987, 2.0713255,   -0.35911667,
      0.3861619,   1.9897066,   -0.2823396,  -0.5135972,  -0.8962833,
      -0.0901652,  -0.73964226, -0.46269894, 0.42379895,  1.1180195,
      1.4442351,   -1.0771092,  0.9014347,   -0.14529487, 1.173365,
      1.510687,    -0.46714714, 1.3281798,   1.7365712,   -1.5435543,
      0.35064182,  0.5708492,   -1.8452454,  0.9243176,   0.57233644,
      -1.0959795,  -0.62557054, 0.686686,    0.4222773,   -0.2146352,
      -0.81243026, -1.1678637,  1.6384528,   1.187959,    -2.5538385,
      -0.39338952, 0.233341,    -1.6181145,  -0.8736809,  0.05150718,
      2.2328985,   2.8749912,   0.08306922,  -0.9871888,  0.47143334,
      -1.7806206,  -0.27681163, -0.9240901,  1.3088665,   0.7826533};

  float expected_values[] = {1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0,
                             1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0,
                             1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0,
                             1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0,
                             1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 1.0, 0, 0};

  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 1;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_balanced_3ds_large_2() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {10, 2, 3}; // Will be same for in and out dim.

  float input_values[] = {
      0.9356609,   1.0854305,   -0.93788373, -0.5061547,  1.3169702,
      0.7137579,   -0.4126717,  -0.40257987, 2.0713255,   -0.35911667,
      0.3861619,   1.9897066,   -0.2823396,  -0.5135972,  -0.8962833,
      -0.0901652,  -0.73964226, -0.46269894, 0.42379895,  1.1180195,
      1.4442351,   -1.0771092,  0.9014347,   -0.14529487, 1.173365,
      1.510687,    -0.46714714, 1.3281798,   1.7365712,   -1.5435543,
      0.35064182,  0.5708492,   -1.8452454,  0.9243176,   0.57233644,
      -1.0959795,  -0.62557054, 0.686686,    0.4222773,   -0.2146352,
      -0.81243026, -1.1678637,  1.6384528,   1.187959,    -2.5538385,
      -0.39338952, 0.233341,    -1.6181145,  -0.8736809,  0.05150718,
      2.2328985,   2.8749912,   0.08306922,  -0.9871888,  0.47143334,
      -1.7806206,  -0.27681163, -0.9240901,  1.3088665,   0.7826533};

  float expected_values[] = {
      0.462402, 0.537109, 0, 0.139160, 0.860352, 0, 0.497559, 0.502930, 0,
      0.321777, 0.677734, 0, 0.557617, 0.442383, 0, 0.657227, 0.343262, 0,
      0.333496, 0.666992, 0, 0.121582, 0.878906, 0, 0.416992, 0.583008, 0,
      0.399414, 0.600586, 0, 0.444824, 0.554688, 0, 0.586914, 0.412598, 0,
      0.212158, 0.788086, 0, 0.645508, 0.354980, 0, 0.611328, 0.389160, 0,
      0.348145, 0.651367, 0, 0.283691, 0.715820, 0, 0.942383, 0.057739, 0,
      0.905273, 0.095093, 0, 0.096924, 0.903320, 0};

  float log_expected_values[] = {
      -0.771484, -0.621094, 0, -1.972656, -0.149902, 0, -0.698242, -0.688477, 0,
      -1.132812, -0.388672, 0, -0.583984, -0.815430, 0, -0.420410, -1.070312, 0,
      -1.099609, -0.405273, 0, -2.105469, -0.129639, 0, -0.875000, -0.539062, 0,
      -0.917969, -0.509766, 0, -0.809570, -0.588867, 0, -0.532227, -0.884766, 0,
      -1.550781, -0.238281, 0, -0.438477, -1.037109, 0, -0.492676, -0.944336, 0,
      -1.054688, -0.428223, 0, -1.259766, -0.333984, 0, -0.059509, -2.851562, 0,
      -0.099976, -2.351562, 0, -2.335938, -0.101929, 0};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 2;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_balanced_3ds_large_3() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {10, 2, 3}; // Will be same for in and out dim.

  float input_values[] = {
      0.9356609,   1.0854305,   -0.93788373, -0.5061547,  1.3169702,
      0.7137579,   -0.4126717,  -0.40257987, 2.0713255,   -0.35911667,
      0.3861619,   1.9897066,   -0.2823396,  -0.5135972,  -0.8962833,
      -0.0901652,  -0.73964226, -0.46269894, 0.42379895,  1.1180195,
      1.4442351,   -1.0771092,  0.9014347,   -0.14529487, 1.173365,
      1.510687,    -0.46714714, 1.3281798,   1.7365712,   -1.5435543,
      0.35064182,  0.5708492,   -1.8452454,  0.9243176,   0.57233644,
      -1.0959795,  -0.62557054, 0.686686,    0.4222773,   -0.2146352,
      -0.81243026, -1.1678637,  1.6384528,   1.187959,    -2.5538385,
      -0.39338952, 0.233341,    -1.6181145,  -0.8736809,  0.05150718,
      2.2328985,   2.8749912,   0.08306922,  -0.9871888,  0.47143334,
      -1.7806206,  -0.27681163, -0.9240901,  1.3088665,   0.7826533};

  float expected_values[] = {
      0.43193838, 0.5017252,  0.06633637, 0.09453523, 0.5852842,  0.32018057,
      0.07143247, 0.07215702, 0.85641056, 0.07363626, 0.15515368, 0.7712101,
      0.42831188, 0.3398805,  0.23180765, 0.45222163, 0.23620388, 0.31157458,
      0.17311363, 0.3465991,  0.48028725, 0.09283915, 0.67143184, 0.23572904,
      0.38534594, 0.5399429,  0.07471115, 0.390473,   0.58742595, 0.02210104,
      0.42416108, 0.5286468,  0.04719208, 0.5446892,  0.38307628, 0.07223454,
      0.13216929, 0.49094895, 0.37688172, 0.51665765, 0.28417364, 0.19916865,
      0.6051712,  0.3856837,  0.00914512, 0.31592378, 0.5912456,  0.09283058,
      0.03865956, 0.09751265, 0.86382776, 0.9239366,  0.05664035, 0.01942311,
      0.6335613,  0.06663986, 0.29979888, 0.06313774, 0.5889111,  0.34795114};

  float log_expected_values[] = {
      -0.83947235, -0.68970275, -2.713017,   -2.3587828,  -0.53565776,
      -1.1388701,  -2.6390028,  -2.6289108,  -0.1550054,  -2.6086178,
      -1.8633392,  -0.25979444, -0.84790367, -1.0791612,  -1.4618473,
      -0.79358286, -1.4430599,  -1.1661166,  -1.7538071,  -1.0595865,
      -0.7333709,  -2.3768868,  -0.39834276, -1.4450723,  -0.9536138,
      -0.6162919,  -2.594126,   -0.9403964,  -0.5320051,  -3.8121307,
      -0.857642,   -0.6374347,  -3.0535293,  -0.60753995, -0.9595212,
      -2.627837,   -2.0236716,  -0.7114151,  -0.9758239,  -0.6603748,
      -1.2581699,  -1.6136034,  -0.5022439,  -0.9527377,  -4.6945353,
      -1.1522543,  -0.5255238,  -2.376979,   -3.2529612,  -2.327773,
      -0.14638188, -0.07911182, -2.8710337,  -3.9412918,  -0.4563985,
      -2.7084525,  -1.2046435,  -2.7624366,  -0.52948,    -1.0556931};

  // Mask greater than 0 will use elements [0, mask] (inclusive) of dimension 1
  uint32_t softmax_mask = 3;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_OK, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_OK, log_expected_values);
}

void zdnn_softmax_mask_invalid_mask() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape[] = {3, 3, 1}; // Will be same for in and out dim.
  float input_values[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
  float expected_values[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  float log_expected_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  // Mask must be less than or equal to dimension 1
  uint32_t softmax_mask = 2;

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_NONE,
                         softmax_mask, ZDNN_FUNC_RC_F002, expected_values);

  zdnn_softmax_mask_test(shape, ZDNN_3DS, input_values, SOFTMAX_ACT_LOG,
                         softmax_mask, ZDNN_FUNC_RC_F002, log_expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_basic_3ds);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_basic_3ds_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_basic_3ds_large);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_basic_3ds_large_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_basic_3ds_large_2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_basic_3ds_large_3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds_2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_negative_3ds);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_negative_3ds_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_negative_3ds_2)
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_negative_3ds_3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds_large);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds_large_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds_large_2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_softmax_mask_balanced_3ds_large_3);
  RUN_TEST(zdnn_softmax_mask_invalid_mask);
  return UNITY_END();
}
