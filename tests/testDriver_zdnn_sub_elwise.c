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
 * Simple test to drive a full sub api. Input tensor 1 has values greater than
 * those in input tensor 2, so the result values will not be negative.
 */
void api_sub_basic() {

  /* Input 1 values as true NHWC
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input1_values[] = {3, 8, 6, 9, 30, 80, 60, 90};

  /* Input 2 values as true NHWC
  [[
    [[1, 10], [2, 20]],
    [[4, 40], [5, 50]]
  ]]
  */

  // Values in ZDNN_NHWC order
  float input2_values[] = {1, 4, 2, 5, 10, 40, 20, 50};

  /* Expected values as true NHWC
    [[
      [[2, 20],   [4, 40]],
      [[4, 40], [4, 40]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_SUB, ZDNN_OK);
}

// test to drive input tensors with 280 values in their buffer. All randomly
// generated numbers in first input tensor will be greater than or equal to
// those in the second input tensor to avoid negatives in the output tensor
void api_sub_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC order
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_SUB, ZDNN_OK);
}

// test to drive input tensors with 6825 values in their buffer
void api_sub_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC order
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_SUB, ZDNN_OK);
}

/*
 * Simple test to drive a full sub api.
 */
void api_sub_3D() {

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
  float input2_values[] = {1, 10, 2, 20, 4, 40, 5, 50};

  /* Expected values as true NHWC
    [[
      [[2, 20],   [4, 40]],
      [[4, 40], [4, 40]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_3D, input1_values, input2_values,
                           NNPA_SUB, ZDNN_OK);
}

/*
 * Simple test to drive a full sub api using the  data type
 * and 2 dimensional tensors
 */
void api_sub_2D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2};

  /* Input 1 values as true NHWC
  [[
    [[3, 20], [2, 20]]
  ]]
*/
  float input1_values[] = {3, 20, 2, 20};

  /* Input 2 values as true NHWC
  [[
    [[1, 10], [2, 5]]
  ]]
*/
  float input2_values[] = {1, 10, 2, 5};

  /* Expected values as true NHWC
    [[
      [[2, 10],   [0, 15]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_2D, input1_values, input2_values,
                           NNPA_SUB, ZDNN_OK);
}

/*
 * Simple test to drive a full sub api using the  data type
 * and 1 dimensional tensors
 */
void api_sub_1D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2};

  /* Input 1 values as true NHWC
  [[
    [[8, 4000]]
  ]]
*/
  float input1_values[] = {8, 4000};

  /* Input 2 values as true NHWC
  [[
    [[2.5, 12]]
  ]]
*/
  float input2_values[] = {2.5, 12};

  /* Expected values as true NHWC
    [[
      [[5.5, 3988]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_1D, input1_values, input2_values,
                           NNPA_SUB, ZDNN_OK);
}

/*
 * Simple test to drive a full sub api, resulting in underflow.
 * Input tensors 1 and 2 have negative values, such that when tensor 2
 * is subtracted from tensor 1, the result values will be negative, and
 * one value will be exceed the DLFloat16 capability.
 */
void api_sub_underflow() {

  /* Input 1 values as true NHWC
  [[
    [[3, 30], [-MAX_DLF16 * 0.75, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input1_values[] = {3, 8, -MAX_DLF16 * 0.75, 9, 30, 80, 60, 90};

  /* Input 2 values as true NHWC
  [[
    [[1, 10], [-MAX_DLF16 * 0.75, 20]],
    [[4, 40], [5, 50]]
  ]]
  */

  // Values in ZDNN_NHWC order
  float input2_values[] = {1, 4, MAX_DLF16 * 0.75, 5, 10, 40, 20, 50};

  /* Expected values as true NHWC
    [[
      [[2, 20], [UNDERFLOW, 40]],
      [[4, 40], [4, 40]]
    ]]
  */

  // when overflow/underflow happens, AIU sets range violation flag

  test_elwise_api_2_inputs_adv(shape, ZDNN_NHWC, FP32, input1_values,
                               input2_values, NNPA_SUB,
                               ZDNN_ELEMENT_RANGE_VIOLATION);
  test_elwise_api_2_inputs_adv(shape, ZDNN_NHWC, BFLOAT, input1_values,
                               input2_values, NNPA_SUB,
                               ZDNN_ELEMENT_RANGE_VIOLATION);

  // Note: We can't create an add/sub overflow/underflow with values that
  // originate as FP16s, since FP16's max is way below the DLFloat max.
}
int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(api_sub_basic);
  RUN_TEST_ALL_DATATYPES(api_sub_med_dims);
  RUN_TEST_ALL_DATATYPES(api_sub_high_dims);
  RUN_TEST_ALL_DATATYPES(api_sub_3D);
  RUN_TEST_ALL_DATATYPES(api_sub_2D);
  RUN_TEST_ALL_DATATYPES(api_sub_1D);
  RUN_TEST(api_sub_underflow);

  return UNITY_END();
}
