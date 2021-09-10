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
 * Simple test to drive a full add api.
 */
void api_add_basic() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {1, 2, 2, 2};

  /* Input 1 values as NHWC
  [[
    [[1, 10], [2, 20]],
    [[4, 40], [5, 50]]
  ]]
  */
  float input1_values[] = {1, 10, 2, 20, 4, 40, 5, 50};

  /* Input 2 values as NHWC
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */
  float input2_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Expected values as NHWC (test method will generate this array)
    [[
      [[4, 40],   [8, 80]],
      [[12, 120], [14, 140]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_ADD, ZDNN_OK);
}

// test to drive input tensors with 320 values in their buffer
void api_add_med_dims() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {1, 8, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order
  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC order
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_ADD, ZDNN_OK);
}

// test to drive input tensors with 6825 values in their buffer
void api_add_high_dims() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order
  float input1_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input1_values);

  // Values in ZDNN_NHWC order
  float input2_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input2_values);

  test_elwise_api_2_inputs(shape, ZDNN_NHWC, input1_values, input2_values,
                           NNPA_ADD, ZDNN_OK);
}

/*
 * Simple test to drive a full add api using the  data type
 * and 3 dimensional tensors
 */
void api_add_3D() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {2, 2, 2};

  /* Input 1 values as NHWC
    [[
      [[1, 10], [2, 20]],
      [[4, 40], [5, 50]]
    ]]
  */
  float input1_values[] = {1, 10, 2, 20, 4, 40, 5, 50};

  /* Input 2 values as NHWC
    [[
      [[3, 30], [6, 60]],
      [[8, 80], [9, 90]]
    ]]
  */
  float input2_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Expected values as NHWC (test method will generate this array)
    [[
      [[4, 40],   [8, 80]],
      [[12, 120], [14, 140]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_3D, input1_values, input2_values,
                           NNPA_ADD, ZDNN_OK);
}

/*
 * Simple test to drive a full add api using the  data type
 * and 2 dimensional tensors
 */
void api_add_2D() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {2, 2};

  /* Input 1 values as NHWC
  [[
    [[1, 10], [2, 20]]
  ]]
*/
  float input1_values[] = {1, 10, 2, 20};

  /* Input 2 values as NHWC
  [[
    [[3, 30], [6, 60]]
  ]]
*/
  float input2_values[] = {3, 30, 6, 60, 8, 80, 9, 90};

  /* Expected values as NHWC (test method will generate this array)
    [[
      [[4, 40],   [8, 80]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_2D, input1_values, input2_values,
                           NNPA_ADD, ZDNN_OK);
}

/*
 * Simple test to drive a full add api using the  data type
 * and 1 dimensional tensors
 */
void api_add_1D() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {2};

  /* Input 1 values as NHWC
  [[
    [[10000, 12000]]
  ]]
*/
  float input1_values[] = {10000, 12000};

  /* Input 2 values as NHWC
  [[
    [[860, 1400]]
  ]]
*/
  float input2_values[] = {860, 1400};

  /* Expected values as NHWC (test method will generate this array)
    [[
      [[10860, 13400]]
    ]]
  */

  test_elwise_api_2_inputs(shape, ZDNN_1D, input1_values, input2_values,
                           NNPA_ADD, ZDNN_OK);
}
/*
 * Simple test to drive a full add api that hits an overflow.
 */
void api_add_overflow() {

  // Input and outputs expect the same shape so just define it once
  uint32_t shape[] = {1, 2, 2, 2};

  /* Input 1 values as NHWC
  [[
    [[1, 10], [MAX_DLF16 * 0.75, 20]],
    [[4, 40], [5, 50]]
  ]]
  */
  float input1_values[] = {1, 10, MAX_DLF16 * 0.75, 20, 4, 40, 5, 50};

  /* Input 2 values as NHWC
  [[
    [[3, 30], [MAX_DLF16 * 0.75, 60]],
    [[8, 80], [9, 90]]
  ]]
  */
  float input2_values[] = {3, 30, MAX_DLF16 * 0.75 + 1.0, 60, 8, 80, 9, 90};

  /* Expected values as NHWC (test method will generate this array)
    [[
      [[4, 40],   [OVERFLOW, 80]],
      [[12, 120], [14, 140]]
    ]]
  */

  // when overflow/underflow happens, AIU sets range violation flag

  test_elwise_api_2_inputs_adv(shape, ZDNN_NHWC, FP32, input1_values,
                               input2_values, NNPA_ADD,
                               ZDNN_ELEMENT_RANGE_VIOLATION);
  test_elwise_api_2_inputs_adv(shape, ZDNN_NHWC, BFLOAT, input1_values,
                               input2_values, NNPA_ADD,
                               ZDNN_ELEMENT_RANGE_VIOLATION);

  // Note: We can't create an add/sub overflow/underflow with values that
  // originate as FP16s, since FP16's max is way below the DLFloat max.
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(api_add_basic);
  RUN_TEST_ALL_DATATYPES(api_add_med_dims);
  RUN_TEST_ALL_DATATYPES(api_add_high_dims);
  RUN_TEST_ALL_DATATYPES(api_add_3D);
  RUN_TEST_ALL_DATATYPES(api_add_2D);
  RUN_TEST_ALL_DATATYPES(api_add_1D);
  RUN_TEST(api_add_overflow);

  return UNITY_END();
}
