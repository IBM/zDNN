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

#include "common_elwise.h"
#include <math.h>

void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}
/*
 * Simple test to drive a full sqrt api.
 */
void zdnn_sqrt_test(uint32_t *io_dims, zdnn_data_layouts layout, float *input,
                    zdnn_status expected_status, float *expected_values) {
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
  zdnn_status status = zdnn_sqrt(input_ztensor, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_sqrt() returned status %08x but expected  %08x\n", status,
      expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

void api_sqrt_basic() {

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
       [[1.732422, 5.476562], [2.449219, 7.742188]],
       [[2.828125, 8.937500], [1.732422, 3.164062]]
     ]]
  */
  float expected_values[] = {1.732422, 5.476562, 2.449219, 7.742188,
                             2.828125, 8.937500, 1.732422, 3.164062};

  zdnn_sqrt_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

// test to drive input tensors with 280 values in their buffer.
void api_sqrt_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  generate_expected_output(sqrtf, input_values, num_io_buffer_values,
                           expected_values);

  zdnn_sqrt_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

// test to drive an input tensor with 6825 values in its buffer
void api_sqrt_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];
  generate_expected_output(sqrtf, input_values, num_io_buffer_values,
                           expected_values);

  zdnn_sqrt_test(shape, ZDNN_NHWC, input_values, ZDNN_OK, expected_values);
}

/*
 * Simple test to drive a full sqrt api using data type and a 3D layout
 */
void api_sqrt_3D() {

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
       [[1.732422, 5.476562], [2.449219, 7.742188]],
       [[2.828125, 8.937500], [3, 9.484375]]
     ]]
  */
  float expected_values[] = {1.732422, 5.476562, 2.449219, 7.742188,
                             2.828125, 8.937500, 3,        9.484375};

  zdnn_sqrt_test(shape, ZDNN_3D, input_values, ZDNN_OK, expected_values);
}

/*
 * Simple test to drive a full sqrt api using the data type and a 2D layout
 */
void api_sqrt_2D() {

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
      [[1, 3.164062], [1.414062, 2.449219]]
    ]]
  */
  float expected_values[] = {1, 3.164062, 1.414062, 2.449219};

  zdnn_sqrt_test(shape, ZDNN_2D, input_values, ZDNN_OK, expected_values);
}

/*
 * Simple test to drive a full sqrt api using the data type and a 1D layout
 */
void api_sqrt_1D() {

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
      [[2.449219, 2.644531]]
    ]]
  */
  float expected_values[] = {2.449219, 2.644531};

  zdnn_sqrt_test(shape, ZDNN_1D, input_values, ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_sqrt_basic);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_sqrt_med_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_sqrt_high_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_sqrt_3D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_sqrt_2D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_sqrt_1D);
  return UNITY_END();
}
