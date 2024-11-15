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

#include <stdlib.h>
#include <string.h>

#include "testsupport.h"

void setUp(void) {

  tol_bfloat.ulps = 64;
  tol_bfloat.epsilon_mult = (0.1 / EPSILON_BFLOAT) + 1;

  tol_fp16.ulps = 64;
  tol_fp16.epsilon_mult = (0.1 / EPSILON_FP16) + 1;

  tol_fp32.ulps = 64 * 16384;
  tol_fp32.epsilon_mult = (0.1 / EPSILON_FLOAT) + 1;

  VERIFY_HW_ENV;
}

void tearDown(void) {}

/**
 * Helper function to compute expected output tensor from randomly generated
 * test input arrays.
 *
 * | input_a         | input_b  | input_c  | result        |
 * | (n, h, w, c)    | (c)      | (c)      | (n, h, w, c)  |
 *
 * formula: output(*, *, *, c) = input_a(*, *, *, c) * input_b(c) + input_c(c)
 *
 */
void gen_test_expected_fp32_array(uint32_t *shape, zdnn_data_types type,
                                  float *input_a, float *input_b,
                                  float *input_c, float *result) {

  uint32_t c = shape[3];
  for (uint64_t i = 0; i < (uint64_t)shape[0] * shape[1] * shape[2] * c; i++) {

    float cleansed_input_a = 0;
    float cleansed_input_b = 0;
    float cleansed_input_c = 0;

    switch (type) {
    case (BFLOAT):
      cleansed_input_a = CLEANSE_BFLOAT(input_a[i]);
      cleansed_input_b = CLEANSE_BFLOAT(input_b[i % c]);
      cleansed_input_c = CLEANSE_BFLOAT(input_c[i % c]);
      break;
    case (FP16):
      cleansed_input_a = CLEANSE_FP16(input_a[i]);
      cleansed_input_b = CLEANSE_FP16(input_b[i % c]);
      cleansed_input_c = CLEANSE_FP16(input_c[i % c]);
      break;
    case (FP32):
      cleansed_input_a = CLEANSE_FP32(input_a[i]);
      cleansed_input_b = CLEANSE_FP32(input_b[i % c]);
      cleansed_input_c = CLEANSE_FP32(input_c[i % c]);
      break;
    default:
      break;
    }

    result[i] = cleansed_input_a * cleansed_input_b + cleansed_input_c;
  }
}

void do_test(uint32_t *input_a_shape, uint32_t *input_b_shape,
             uint32_t *input_c_shape, uint32_t *output_shape,
             zdnn_data_types dtype, float *input_a_values,
             float *input_b_values, float *input_c_values,
             zdnn_status expected_status, float *expected_values) {

  zdnn_ztensor *input_a_ztensor = alloc_ztensor_with_values(
      input_a_shape, ZDNN_NHWC, dtype, NO_CONCAT, false, input_a_values);

  zdnn_ztensor *input_b_ztensor = alloc_ztensor_with_values(
      input_b_shape, ZDNN_1D, dtype, NO_CONCAT, false, input_b_values);

  zdnn_ztensor *input_c_ztensor = alloc_ztensor_with_values(
      input_c_shape, ZDNN_1D, dtype, NO_CONCAT, false, input_c_values);

  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      output_shape, ZDNN_NHWC, dtype, NO_CONCAT, true, ZERO_ARRAY);

  // Call public NNPA method
  zdnn_status status = zdnn_batchnorm(input_a_ztensor, input_b_ztensor,
                                      input_c_ztensor, output_ztensor);

  // Assert returned status matches expected
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_batchnorm to returned status %08x but expected "
      "%08x\n",
      status, expected_status);

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

  // If expected status is ZDNN_OK, assert output values matches expected
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values_adv(output_ztensor, false, expected_values, *tol);
  }

  // Cleanup test ztensors
  free_ztensor_buffers(4, input_a_ztensor, input_b_ztensor, input_c_ztensor,
                       output_ztensor);
}

void zdnn_batchnorm_small_values() {

  uint32_t shape[] = {1, 3, 3, 2};
  float input_a_values[] = {0.1, 1,   0.2, 2,   0.3, 3,   0.4, 4,   0.5,
                            5,   0.6, 6,   0.7, 7,   0.8, 8,   0.9, 9};
  uint32_t input_b_shape[] = {2};
  float input_b_values[] = {0.45, 0.55};
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {0.75, 0.45};
  float output_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0};

  gen_test_expected_fp32_array(shape, test_datatype, input_a_values,
                               input_b_values, input_c_values, output_values);

  do_test(shape, input_b_shape, input_c_shape, shape, test_datatype,
          input_a_values, input_b_values, input_c_values, ZDNN_OK,
          output_values);
}

void zdnn_batchnorm_high_values() {

  uint32_t shape[] = {1, 3, 3, 2};
  float input_a_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                            50, 6,  60, 7,  70, 8,  80, 9,  90};
  uint32_t input_b_shape[] = {2};
  float input_b_values[] = {4.5, 5.5};
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {7.5, 4.5};
  float output_values[] = {0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0};

  gen_test_expected_fp32_array(shape, test_datatype, input_a_values,
                               input_b_values, input_c_values, output_values);

  do_test(shape, input_b_shape, input_c_shape, shape, test_datatype,
          input_a_values, input_b_values, input_c_values, ZDNN_OK,
          output_values);
}

void test_batchnorm_random_values(uint32_t n, uint32_t h, uint32_t w,
                                  uint32_t c) {

  uint32_t shape[] = {n, h, w, c};
  uint64_t num_values = (uint64_t)n * h * w * c;

  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  uint32_t input_b_shape[] = {c};
  float input_b_values[c];
  gen_random_float_array_pos_neg(c, input_b_values);

  uint32_t input_c_shape[] = {c};
  float input_c_values[c];
  gen_random_float_array_pos_neg(c, input_c_values);

  float output_values[num_values];

  gen_test_expected_fp32_array(shape, test_datatype, input_a_values,
                               input_b_values, input_c_values, output_values);

  do_test(shape, input_b_shape, input_c_shape, shape, test_datatype,
          input_a_values, input_b_values, input_c_values, ZDNN_OK,
          output_values);
}

void zdnn_batchnorm_random_values_low_dims() {
  test_batchnorm_random_values(2, 3, 4, 5);
}

void zdnn_batchnorm_random_values_high_dims() {
  test_batchnorm_random_values(2, 3, 4, 100);
}

int main() {

  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_batchnorm_small_values);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_batchnorm_high_values);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_batchnorm_random_values_low_dims);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_batchnorm_random_values_high_dims);
  return UNITY_END();
}
