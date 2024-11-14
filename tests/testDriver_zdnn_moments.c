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

void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

/**
 * zdnn_norm_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_moments_test(uint32_t *i_dims, uint32_t *o_a_dims, uint32_t *o_b_dims,
                       zdnn_data_layouts layout, float *input_a,
                       uint32_t bessel_correction, zdnn_status expected_status,
                       float *expected_values_a, float *expected_values_b) {

  /*
   * Input Tensor a
   */
  zdnn_ztensor *input_ztensor_a = alloc_ztensor_with_values(
      i_dims, layout, test_datatype, NO_CONCAT, false, input_a);

  /*
   * Output Tensor a
   */

  zdnn_ztensor *output_ztensor_a = alloc_ztensor_with_values(
      o_a_dims, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Output Tensor b
   */

  zdnn_ztensor *output_ztensor_b = alloc_ztensor_with_values(
      o_b_dims, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Begin Testing!
   */
  zdnn_status status = zdnn_moments(input_ztensor_a, bessel_correction,
                                    output_ztensor_a, output_ztensor_b);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_moments() to returned status %08x but expected  %08x\n",
      status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor_a, false, expected_values_a);
    assert_ztensor_values(output_ztensor_b, false, expected_values_b);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(3, input_ztensor_a, output_ztensor_a, output_ztensor_b);
}

// Calculate values to approximate zDNN LayerNorm
void generate_moments_output(const float input_values[],
                             const uint32_t input_shape[],
                             uint32_t bessel_correction, int num_values,
                             float expected_values_a[],
                             float expected_values_b[]) {

  uint64_t l =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];

  float summation = 0.0;
  float summation_sq = 0.0;

  for (int i = 0; i < num_values; i++) {
    summation += input_values[i];
    summation_sq += powf(input_values[i], 2);
  }

  expected_values_a[0] = summation / l;
  expected_values_b[0] =
      (summation_sq - (powf(summation, 2) / l)) / (l - bessel_correction);
}

void zdnn_moments_basic_small_nhwc_pos() {

  uint32_t shape_i[] = {1, 5, 12, 1};
  uint32_t shape_o_a[] = {1, 1, 1, 1};
  uint32_t shape_o_b[] = {1, 1, 1, 1};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  uint32_t bessel_correction = 0;

  float expected_values_a[shape_o_a[3]];
  float expected_values_b[shape_o_b[3]];

  generate_moments_output(input_values, shape_i, bessel_correction,
                          num_io_buffer_values, expected_values_a,
                          expected_values_b);

  zdnn_moments_test(shape_i, shape_o_a, shape_o_b, ZDNN_NHWC, input_values,
                    bessel_correction, ZDNN_OK, expected_values_a,
                    expected_values_b);
}

void zdnn_moments_basic_large_nhwc_pos() {

  uint32_t shape_i[] = {1, 56, 70, 3};
  uint32_t shape_o_a[] = {1, 1, 1, 1};
  uint32_t shape_o_b[] = {1, 1, 1, 1};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array(num_io_buffer_values, input_values);

  uint32_t bessel_correction = 0;

  float expected_values_a[shape_o_a[3]];
  float expected_values_b[shape_o_b[3]];

  generate_moments_output(input_values, shape_i, bessel_correction,
                          num_io_buffer_values, expected_values_a,
                          expected_values_b);

  zdnn_moments_test(shape_i, shape_o_a, shape_o_b, ZDNN_NHWC, input_values,
                    bessel_correction, ZDNN_OK, expected_values_a,
                    expected_values_b);
}

void zdnn_moments_basic_large_nhwc_pos_neg() {

  uint32_t shape_i[] = {1, 40, 30, 20};
  uint32_t shape_o_a[] = {1, 1, 1, 1};
  uint32_t shape_o_b[] = {1, 1, 1, 1};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array_pos_neg(num_io_buffer_values, input_values);

  uint32_t bessel_correction = 1;

  float expected_values_a[shape_o_a[3]];
  float expected_values_b[shape_o_b[3]];

  generate_moments_output(input_values, shape_i, bessel_correction,
                          num_io_buffer_values, expected_values_a,
                          expected_values_b);

  zdnn_moments_test(shape_i, shape_o_a, shape_o_b, ZDNN_NHWC, input_values,
                    bessel_correction, ZDNN_OK, expected_values_a,
                    expected_values_b);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_moments_basic_small_nhwc_pos);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_moments_basic_large_nhwc_pos);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_moments_basic_large_nhwc_pos_neg);

  UNITY_END();
}
