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

#include "common_elwise.h"

void setUp(void) { VERIFY_HW_ENV; }

void tearDown(void) {}
/*
 * Simple test to drive a full log api.
 */
void api_log_basic() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,2)
     [[
       [[1.09861228, 3.40119738], [1.79175946, 4.09434456]],
       [[2.07944154, 4.38202663], [1.09861228,  2.30258509]]
     ]]
  */

  test_elwise_api_1_input(shape, ZDNN_NHWC, input_values, NNPA_LOG, ZDNN_OK);
}

// test to drive input tensors with 280 values in their buffer.
void api_log_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  test_elwise_api_1_input(shape, ZDNN_NHWC, input_values, NNPA_LOG, ZDNN_OK);
}

// test to drive an input tensor with 6825 values in its buffer
void api_log_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  test_elwise_api_1_input(shape, ZDNN_NHWC, input_values, NNPA_LOG, ZDNN_OK);
}

/*
 * Simple test to drive a full log api using  Data type and a
 * 3D layout
 */
void api_log_3D() {

  /* Input 1 values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Expected values as true NHWC sized (1,2,2,2)
    [[
      [[1.09861228, 3.40119738], [1.79175946, 4.09434456]],
      [[2.07944154, 4.38202663], [2.19722457, 4.49980967]]
    ]]
  */

  test_elwise_api_1_input(shape, ZDNN_3D, input_values, NNPA_LOG, ZDNN_OK);
}

/*
 * Simple test to drive a full log api using the  data type
 * and 2 dimensional tensors
 */
void api_log_2D() {

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
      [[0, 2.30258509],   [0.69314718, 1.79175946]]
    ]]
  */

  test_elwise_api_1_input(shape, ZDNN_2D, input_values, NNPA_LOG, ZDNN_OK);
}

/*
 * Simple test to drive a full log api using the  data type
 * and 1 dimensional tensors
 */
void api_log_1D() {

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
      [[1.79175946, 1.94591014]]
    ]]
  */

  test_elwise_api_1_input(shape, ZDNN_1D, input_values, NNPA_LOG, ZDNN_OK);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_log_basic);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_log_med_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_log_high_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_log_3D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_log_2D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_log_1D);
  return UNITY_END();
}
