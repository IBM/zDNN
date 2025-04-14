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

  tol_bfloat.ulps = MAX_ULPS_BFLOAT;
  tol_bfloat.epsilon_mult = MAX_EPSILON_MULT_BFLOAT;

  // note: api_invsqrt_med_dims     (FP16)
  //       api_invsqrt_med_dims_1   (FP16)
  //       api_invsqrt_high_dims    (FP16)
  //       api_invsqrt_high_dims_1  (FP16)
  // need custom tolerance
  tol_fp16.ulps = MAX_ULPS_FP16;
  tol_fp16.epsilon_mult = (0.63 / EPSILON_FP16) + 1;

  tol_fp32.ulps = MAX_ULPS_FLOAT;
  tol_fp32.epsilon_mult = MAX_EPSILON_MULT_FLOAT;
}

void tearDown(void) {}

float invsqrtf(float x, float e) { return 1.0 / sqrtf(x + e); }

/*
 * Simple test to drive a full invsqrt api.
 */
void zdnn_invsqrt_test(uint32_t *io_dims, zdnn_data_layouts layout,
                       float *input, float epsilon, zdnn_status expected_status,
                       float *expected_values) {
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
  zdnn_status status = zdnn_invsqrt(input_ztensor, epsilon, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_invsqrt() returned status %08x but expected  %08x\n",
      status, expected_status);

  // To allow for unique tolerance
  fp_tolerance *tol = NULL;
  switch (output_ztensor->pre_transformed_desc->type) {
  case BFLOAT:
    tol = &tol_bfloat;
    break;
  case FP16:
    tol = &tol_fp16;
    break;
  case FP32:
    tol = &tol_fp32;
    break;
  default:
    break;
    // should never get here
  }
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values_adv(output_ztensor, false, expected_values, *tol);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

void api_invsqrt_basic() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};
  float epsilon = 0;

  /* Expected values as true NHWC sized (1,2,2,2)
     [[
       [[0.577148, 0.182617], [0.408203, 0.129150]],
       [[0.353516, 0.111816], [0.577148, 0.316406]]
     ]]
  */
  float expected_values[] = {0.577148, 0.182617, 0.408203, 0.129150,
                             0.353516, 0.111816, 0.577148, 0.316406};

  zdnn_invsqrt_test(shape, ZDNN_NHWC, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

void api_invsqrt_basic_1() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};
  float epsilon = 0.001;

  /* Expected values as true NHWC sized (1,2,2,2)
     [[
       [[0.577148, 0.182617], [0.408203, 0.129150]],
       [[0.353516, 0.111816], [0.577148, 0.316406]]
     ]]
  */
  float expected_values[] = {0.577148, 0.182617, 0.408203, 0.129150,
                             0.353516, 0.111816, 0.577148, 0.316406};

  zdnn_invsqrt_test(shape, ZDNN_NHWC, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

// test to drive input tensors with 280 values in their buffer.
void api_invsqrt_med_dims() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];
  float epsilon = 0;

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];

  for (int i = 0; i < num_io_buffer_values; i++) {
    expected_values[i] = invsqrtf(input_values[i], epsilon);
  }

  zdnn_invsqrt_test(shape, ZDNN_NHWC, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

void api_invsqrt_med_dims_1() {

  uint32_t shape[] = {1, 7, 10, 4};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];
  float epsilon = 0.001;

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];

  for (int i = 0; i < num_io_buffer_values; i++) {
    expected_values[i] = invsqrtf(input_values[i], epsilon);
  }

  zdnn_invsqrt_test(shape, ZDNN_NHWC, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

// test to drive an input tensor with 6825 values in its buffer
void api_invsqrt_high_dims() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];
  float epsilon = 0;

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];

  for (int i = 0; i < num_io_buffer_values; i++) {
    expected_values[i] = invsqrtf(input_values[i], epsilon);
  }

  zdnn_invsqrt_test(shape, ZDNN_NHWC, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

void api_invsqrt_high_dims_1() {

  uint32_t shape[] = {1, 3, 33, 65};
  int num_io_buffer_values = shape[0] * shape[1] * shape[2] * shape[3];
  float epsilon = 0.001;

  // Values in ZDNN_NHWC order

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  float expected_values[num_io_buffer_values];

  for (int i = 0; i < num_io_buffer_values; i++) {
    expected_values[i] = invsqrtf(input_values[i], epsilon);
  }

  zdnn_invsqrt_test(shape, ZDNN_NHWC, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

/*
 * Simple test to drive a full invsqrt api using data type and a 3D layout
 */
void api_invsqrt_3D() {

  /* Input 1 values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 9, 90};
  float epsilon = 0;

  /* Expected values as true NHWC sized (1,2,2,2)
     [[
       [[0.577148, 0.182617], [0.408203, 0.129150]],
       [[0.353516, 0.111816], [0.333496, 0.105469]]
     ]]
  */
  float expected_values[] = {0.577148, 0.182617, 0.408203, 0.129150,
                             0.353516, 0.111816, 0.333496, 0.105469};

  zdnn_invsqrt_test(shape, ZDNN_3D, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

void api_invsqrt_3D_1() {

  /* Input 1 values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [9, 90]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 9, 90};
  float epsilon = 0.001;

  /* Expected values as true NHWC sized (1,2,2,2)
     [[
       [[0.577148, 0.182617], [0.408203, 0.129150]],
       [[0.353516, 0.111816], [0.333496, 0.105469]]
     ]]
  */
  float expected_values[] = {0.577148, 0.182617, 0.408203, 0.129150,
                             0.353516, 0.111816, 0.333496, 0.105469};

  zdnn_invsqrt_test(shape, ZDNN_3D, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

/*
 * Simple test to drive a full invsqrt api using the data type and a 2D layout
 */
void api_invsqrt_2D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[1, 10], [2, 6]]
  ]]
  */
  float input_values[] = {1, 10, 2, 6};
  float epsilon = 0;

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[1, 0.316406], [0.707031, 0.408203]]
    ]]
  */
  float expected_values[] = {1, 0.316406, 0.707031, 0.408203};

  zdnn_invsqrt_test(shape, ZDNN_2D, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

void api_invsqrt_2D_1() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2, 2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[1, 10], [2, 6]]
  ]]
  */
  float input_values[] = {1, 10, 2, 6};
  float epsilon = 0.001;

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[1, 0.316406], [0.707031, 0.408203]]
    ]]
  */
  float expected_values[] = {1, 0.316406, 0.707031, 0.408203};

  zdnn_invsqrt_test(shape, ZDNN_2D, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

/*
 * Simple test to drive a full invsqrt api using the data type and a 1D layout
 */
void api_invsqrt_1D() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[6, 7]]
  ]]
  */
  float input_values[] = {6, 7};
  float epsilon = 0;

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[0.408203, 0.377930]]
    ]]
  */
  float expected_values[] = {0.408203, 0.377930};

  zdnn_invsqrt_test(shape, ZDNN_1D, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

void api_invsqrt_1D_1() {

  // Values in ZDNN_NHWC order
  uint32_t shape[] = {2};

  /* Input 1 values as true NHWC sized (1,1,2,2)
  [[
    [[6, 7]]
  ]]
  */
  float input_values[] = {6, 7};
  float epsilon = 0.001;

  /* Expected values as true NHWC sized (1,1,2,2)
    [[
      [[0.408203, 0.377930]]
    ]]
  */
  float expected_values[] = {0.408203, 0.377930};

  zdnn_invsqrt_test(shape, ZDNN_1D, input_values, epsilon, ZDNN_OK,
                    expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_basic);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_basic_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_med_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_med_dims_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_high_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_high_dims_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_3D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_3D_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_2D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_2D_1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_1D);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_invsqrt_1D_1);
  return UNITY_END();
}
