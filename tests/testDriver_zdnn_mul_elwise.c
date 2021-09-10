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

#include "common_elwise.h"

void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

void tearDown(void) {}
/*
 * Simple test to drive a full mul api.
 */
void api_mul_basic() {

  /* Input 1 values as true NHWC
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input1_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Input 2 values as true NHWC
  [[
    [[1, 10], [2, 20]],
    [[4, 40], [5, 50]]
  ]]
  */

  // Values in ZDNN_NHWC order
  float input2_values[] = {1, 10, 2, 20, 4, 40, 5, 50};

  /* Expected values as true NHWC
    [[
      [[3, 300],   [12, 1200]],
      [[32, 3200], [45, 4500]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_MUL, ZDNN_OK);
}

// test to drive input tensors with 280 values in their buffer.
void api_mul_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC order
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_MUL, ZDNN_OK);
}

// test to drive input tensors with 6825 values in their buffer
void api_mul_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC order
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_MUL, ZDNN_OK);
}

/*
 * Simple test to drive a full mul api.
 */
void api_mul_3D() {

  /* Input 1 values as true NHWC
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2, 2};
  float input1_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Input 2 values as true NHWC
  [[
    [[1, 10], [2, 20]],
    [[4, 40], [5, 50]]
  ]]
  */

  // Values in ZDNN_NHWC order
  float input2_values[] = {1, 5, 2, 20, 4, 40, 5, 50};

  // Create ztensor with input1_values
  /* Expected values as true NHWC
    [[
      [[3, 300],   [12, 1200]],
      [[32, 3200], [45, 1400]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_3D, input1_values, input2_values,
                           NNPA_MUL, ZDNN_OK);
}

/*
 * Simple test to drive a full mul api using the  data type
 * and 2 dimensional tensors
 */
void api_mul_2D() {

  // Values in ZDNN_NHWC order
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
      [[3, 200],   [4, 100]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_2D, input1_values, input2_values,
                           NNPA_MUL, ZDNN_OK);
}

/*
 * Simple test to drive a full mul api using the  data type
 * and 1 dimensional tensors
 */
void api_mul_1D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[8, 12]]
  ]]
*/
  float input1_values[] = {8, 12};

  /* Input 2 values as true NHWC sized (1,1,2,2)
  [[
    [[2.5, 4000]]
  ]]
*/
  float input2_values[] = {2.5, 4000};

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[20, 48000]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_1D, input1_values, input2_values,
                           NNPA_MUL, ZDNN_OK);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST_ALL_DATATYPES(api_mul_basic);
  RUN_TEST_ALL_DATATYPES(api_mul_med_dims);
  RUN_TEST_ALL_DATATYPES(api_mul_high_dims);
  RUN_TEST_ALL_DATATYPES(api_mul_3D);
  RUN_TEST_ALL_DATATYPES(api_mul_2D);
  RUN_TEST_ALL_DATATYPES(api_mul_1D);

  return UNITY_END();
}
