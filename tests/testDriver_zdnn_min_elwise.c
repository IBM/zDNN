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

void tearDown(void) {}
/*
 * Simple test to drive a full min api. Input tensor 1 has values greater than
 * those in input tensor 2.
 */
void api_min_basic() {

  /* Input 1 values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input1_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Input 2 values as true NHWC sized (1,2,2,2)
  [[
    [[1, 15], [3, 12]],
    [[4, 40], [4.5, 45]]
  ]]
  */

  // Values in ZDNN_NHWC order
  float input2_values[] = {1, 15, 3, 12, 4, 40, 4.5, 15};

  /* Expected values as true NHWC sized (1,2,2,2)
    [[
    [[1, 15], [3, 12]],
    [[4, 40], [3, 10]]
  ]]
  */
  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_MIN, ZDNN_OK);
}

// test to drive input tensors with 280 values in their buffer. All randomly
// generated numbers in first input tensor will be greater than or equal to
// those in the second input tensor to avoid negatives in the output tensor
void api_min_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC

  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_MIN, ZDNN_OK);
}

// test to drive input tensors with 6825 values in their buffer
void api_min_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC

  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_MIN, ZDNN_OK);
}

/*
 * Simple test to drive a full min api.
 */
void api_min_3D() {

  /* Input 1 values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC
  uint32_t shape[] = {2, 2, 2};
  float input1_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Input 2 values as true NHWC sized (1,2,2,2)
  [[
    [[1, 10], [2, 20]],
    [[4, 40], [5, 50]]
  ]]
  */

  // Values in ZDNN_NHWC
  float input2_values[] = {1, 5, 2, 20, 4, 40, 5, 50};

  /* Expected values as true NHWC sized (1,2,2,2)
    [[
      [[1, 10], [2, 20]],
      [[4, 40], [5, 50]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_3D, input1_values, input2_values,
                           NNPA_MIN, ZDNN_OK);
}

/*
 * Simple test to drive a full min api using the  data type
 * and 2 dimensional tensors
 */
void api_min_2D() {

  // Values in ZDNN_NHWC
  uint32_t shape[] = {2, 2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[1, 10], [2, 20]]
  ]]
*/
  float input1_values[] = {1, 10, 2, 20};

  /* Input 2 values as true NHWC sized (1,1,2,2)
  [[
    [[3, 20], [2, 5]]
  ]]
*/
  float input2_values[] = {3, 20, 2, 5};

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[1, 10],   [2, 5]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_2D, input1_values, input2_values,
                           NNPA_MIN, ZDNN_OK);
}

/*
 * Simple test to drive a full min api using the  data type
 * and 1 dimensional tensors
 */
void api_min_1D() {

  // Values in ZDNN_NHWC
  uint32_t shape[] = {2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[10000, 12000]]
  ]]
*/
  float input1_values[] = {10000, 12000};

  /* Input 2 values as true NHWC sized (1,1,2,2)
  [[
    [[2.5, 4000]]
  ]]
*/
  float input2_values[] = {2.5, 4000};

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[2.5, 4000]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_1D, input1_values, input2_values,
                           NNPA_MIN, ZDNN_OK);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(api_min_basic);
  RUN_TEST_ALL_DATATYPES(api_min_med_dims);
  RUN_TEST_ALL_DATATYPES(api_min_high_dims);
  RUN_TEST_ALL_DATATYPES(api_min_3D);
  RUN_TEST_ALL_DATATYPES(api_min_2D);
  RUN_TEST_ALL_DATATYPES(api_min_1D);
  return UNITY_END();
}
