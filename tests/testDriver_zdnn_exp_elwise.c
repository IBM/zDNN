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

#include "common_elwise.h"

void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

/**********************************************************
 * FP16 tops out at 65504, so no input number larger than
 * 11.089866488461016 should be used
 **********************************************************/

void tearDown(void) {}
/*
 * Simple test to drive a full exp api.
 */
void api_exp_basic() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 4], [6, 7]],
    [[10, 12], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 4, 6, 7, 10, 9, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,2)
  [[
    [[20.085536923, 54.598150033], [403.42879349, 1096.6331584]],
    [[22026.465794, 8103.083926], [20.085536923, 22026.465794]]
  ]]
  */

  test_elwise_api_1_input(shape, ZDNN_NHWC, input_values, NNPA_EXP, ZDNN_OK);
}

// test to drive input tensors with 280 values in their buffer.
void api_exp_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  test_elwise_api_1_input(shape, ZDNN_NHWC, input_values, NNPA_EXP, ZDNN_OK);
}

// test to drive an input tensor with 6825 values in its buffer
void api_exp_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  test_elwise_api_1_input(shape, ZDNN_NHWC, input_values, NNPA_EXP, ZDNN_OK);
}

/*
 * Simple test to drive a full exp api using the  data type
 * and 3D layout
 */
void api_exp_3D() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 4], [6, 7]],
    [[10, 8], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2, 2};
  float input_values[] = {3, 4, 6, 7, 10, 5, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,2)
    [[
      [[20.085536923, 54.598150033], [403.42879349, 1096.6331584]],
      [[22026.465794, 148.41315910], [20.085536923, 22026.465794]]
    ]]
  */

  test_elwise_api_1_input(shape, ZDNN_3D, input_values, NNPA_EXP, ZDNN_OK);
}

/*
 * Simple test to drive a full exp api using the  data type
 * and 2 dimensional tensors
 */
void api_exp_2D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[1, 10], [2, 6]]
  ]]
*/
  float input_values[] = {1, 10, 2, 6};

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[2.718281828, 22026.465794807],   [7.3890560989, 403.42879349]]
    ]]
  */

  test_elwise_api_1_input(shape, ZDNN_2D, input_values, NNPA_EXP, ZDNN_OK);
}

/*
 * Simple test to drive a full exp api using the  data type
 * and 1 dimensional tensors
 */
void api_exp_1D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[6, 7]]
  ]]
*/
  float input_values[] = {6, 7};

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[403.42879349, 1096.6331584]]
    ]]
  */

  test_elwise_api_1_input(shape, ZDNN_1D, input_values, NNPA_EXP, ZDNN_OK);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(api_exp_basic);
  RUN_TEST_ALL_DATATYPES(api_exp_med_dims);
  RUN_TEST_ALL_DATATYPES(api_exp_high_dims);
  RUN_TEST_ALL_DATATYPES(api_exp_3D);
  RUN_TEST_ALL_DATATYPES(api_exp_2D);
  RUN_TEST_ALL_DATATYPES(api_exp_1D);
  return UNITY_END();
}
