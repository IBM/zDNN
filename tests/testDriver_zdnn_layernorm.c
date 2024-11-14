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
void zdnn_layernorm_test(uint32_t *i_dims, uint32_t *bc_dims, uint32_t *o_dims,
                         zdnn_data_layouts layout, float *input_a,
                         float *input_b, float *input_c, const float beta_value,
                         const float gamma_value, const float epsilon_value,
                         zdnn_status expected_status, float *expected_values) {

  /*
   * Input Tensor a
   */
  zdnn_ztensor *input_ztensor_a = alloc_ztensor_with_values(
      i_dims, layout, test_datatype, NO_CONCAT, false, input_a);

  /*
   * Input Tensor b
   */
  zdnn_ztensor *input_ztensor_b = alloc_ztensor_with_values(
      bc_dims, layout, test_datatype, NO_CONCAT, false, input_b);

  /*
   * Input Tensor c
   */
  zdnn_ztensor *input_ztensor_c = alloc_ztensor_with_values(
      bc_dims, layout, test_datatype, NO_CONCAT, false, input_c);

  /*
   * Output Tensor
   */

  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      o_dims, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Begin Testing!
   */
  zdnn_status status =
      zdnn_layernorm(input_ztensor_a, input_ztensor_b, input_ztensor_c,
                     beta_value, gamma_value, epsilon_value, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_layernorm() to returned status %08x but expected  %08x\n",
      status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(4, input_ztensor_a, input_ztensor_b, input_ztensor_c,
                       output_ztensor);
}

// Calculate values to approximate zDNN LayerNorm
void generate_layernorm_output(const float input_values[], const float mean[],
                               const float variance[], const float beta,
                               const float gamma, const float epsilon,
                               int num_values, float expected_values[]) {
  float sum = variance[0] + epsilon;
  sum = (sum <= 0.0f) ? 1e-2f : sum;
  float invsqrt_val = 1.0 / sqrtf(sum);
  for (int i = 0; i < num_values; i++) {
    expected_values[i] =
        (input_values[i] - mean[0]) * invsqrt_val * gamma + beta;
  }
}

void zdnn_layernorm_basic_small_nhwc() {

  uint32_t shape_i[] = {1, 1, 2, 5};
  uint32_t shape_bc[] = {1, 1, 1, 1};
  uint32_t shape_o[] = {1, 1, 2, 5};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];

  float input_values[] = {0.10, 0.15, 0.20, 0.25, 0.30,
                          0.35, 0.40, 0.45, 0.50, 0.55};
  float mean[] = {0.325};
  float variance[] = {0.45};

  const float beta = 0.089;
  const float gamma = 0.67;
  const float epsilon = 0.0001;

  float expected_values[num_io_buffer_values];

  generate_layernorm_output(input_values, mean, variance, beta, gamma, epsilon,
                            num_io_buffer_values, expected_values);
  zdnn_layernorm_test(shape_i, shape_bc, shape_o, ZDNN_NHWC, input_values, mean,
                      variance, beta, gamma, epsilon, ZDNN_OK, expected_values);
}

void zdnn_layernorm_basic_large_nhwc_pos_neg() {

  uint32_t shape_i[] = {1, 1, 40, 80};
  uint32_t shape_bc[] = {1, 1, 1, 1};
  uint32_t shape_o[] = {1, 1, 40, 80};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array_pos_neg(num_io_buffer_values, input_values);

  float mean[] = {0.729};
  float variance[] = {0.25};

  const float beta = 0.089;
  const float gamma = 0.67;
  const float epsilon = 0.0001;

  float expected_values[num_io_buffer_values];

  generate_layernorm_output(input_values, mean, variance, beta, gamma, epsilon,
                            num_io_buffer_values, expected_values);
  zdnn_layernorm_test(shape_i, shape_bc, shape_o, ZDNN_NHWC, input_values, mean,
                      variance, beta, gamma, epsilon, ZDNN_OK, expected_values);
}

void zdnn_layernorm_basic_large_nhwc_neg() {

  uint32_t shape_i[] = {1, 1, 50, 20};
  uint32_t shape_bc[] = {1, 1, 1, 1};
  uint32_t shape_o[] = {1, 1, 50, 20};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];

  float input_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_values);

  float mean[] = {0.2};
  float variance[] = {0.25};

  const float beta = 0.089;
  const float gamma = 0.67;
  const float epsilon = 0.0001;

  float expected_values[num_io_buffer_values];

  generate_layernorm_output(input_values, mean, variance, beta, gamma, epsilon,
                            num_io_buffer_values, expected_values);
  zdnn_layernorm_test(shape_i, shape_bc, shape_o, ZDNN_NHWC, input_values, mean,
                      variance, beta, gamma, epsilon, ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_layernorm_basic_small_nhwc);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_layernorm_basic_large_nhwc_pos_neg);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_layernorm_basic_large_nhwc_neg);

  UNITY_END();
}
