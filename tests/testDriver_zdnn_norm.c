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
void zdnn_norm_test(uint32_t *i_dims, uint32_t *o_dims,
                    zdnn_data_layouts layout, float *input_a, float *input_b,
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
      i_dims, layout, test_datatype, NO_CONCAT, false, input_b);

  /*
   * Output Tensor
   */

  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      o_dims, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Begin Testing!
   */
  zdnn_status status =
      zdnn_norm(input_ztensor_a, input_ztensor_b, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_norm() to returned status %08x but expected  %08x\n",
      status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(3, input_ztensor_a, input_ztensor_b, output_ztensor);
}

// Calculate values to approximate zDNN Norm
void approximate_norm(const float input_a_values[],
                      const float input_b_values[], float expected_values[],
                      const uint32_t shape_i[], uint32_t input_shape_size) {

  // Check if we were passed in all dims we need...for example
  // 3D we can just assume N = 1, but for 4D we will need that...
  uint32_t N, H, W, C;
  if (input_shape_size > 3) {
    N = shape_i[0];
    H = shape_i[1];
    W = shape_i[2];
    C = shape_i[3];
  } else {
    N = 1;
    H = shape_i[0];
    W = shape_i[1];
    C = shape_i[2];
  }

  for (uint32_t n = 0; n < N; n++) {
    for (uint32_t h = 0; h < H; h++) {
      for (uint32_t w = 0; w < W; w++) {
        float sum = 0.0;
        for (uint32_t c = 0; c < C; c++) {
          // The expression n * H * W * C + h * W * C + w * C + c is calculating
          // an index for a 1D array that represents a 4D tensor. This is
          // "flattening" a multi-dimensional tensor into a 1D array.
          //
          // Then n * H * W * C term:
          //   For each batch n, there are C * H * W elements.
          //
          // The h * W * C term:
          //   For each channel c, there are H * W elements.
          //
          // The  w * C term:
          //   For each height h, there are W elements (width).
          //
          // The c term:
          //   Represents the width position inside the h-th row.
          //
          // After summing up the terms, you get the 4D position of the
          // flattened tensor in (n, c, h, w) as a 1D tensor.
          uint32_t index = n * H * W * C + h * W * C + w * C + c;
          sum += powf(input_a_values[index] - input_b_values[index], 2);
        }
        expected_values[w] = sqrtf(sum);
      }
    }
  }
}

/*
  -------------------------------------------------------------------------------
                                  Norm Basic
                                  Layout: 3D
  -------------------------------------------------------------------------------
*/

void zdnn_norm_basic_small_3d() {
  uint32_t shape_i[] = {1, 1, 6};
  uint32_t shape_o[] = {1, 1, 1};

  float input_a_values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  float input_b_values[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  float expected_values[shape_o[1]];
  approximate_norm(input_a_values, input_b_values, expected_values, shape_i,
                   sizeof(shape_i) / sizeof(shape_i[0]));
  zdnn_norm_test(shape_i, shape_o, ZDNN_3D, input_a_values, input_b_values,
                 ZDNN_OK, expected_values);
}

void zdnn_norm_basic_large_3d__pos_neg() {
  uint32_t shape_i[] = {1, 10, 70};
  uint32_t shape_o[] = {1, 10, 1};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2];
  float input_a_values[num_io_buffer_values];
  float input_b_values[num_io_buffer_values];
  gen_random_float_array_pos_neg(num_io_buffer_values, input_a_values);
  gen_random_float_array_pos_neg(num_io_buffer_values, input_b_values);
  float expected_values[shape_o[1]];
  approximate_norm(input_a_values, input_b_values, expected_values, shape_i,
                   sizeof(shape_i) / sizeof(shape_i[0]));
  zdnn_norm_test(shape_i, shape_o, ZDNN_3D, input_a_values, input_b_values,
                 ZDNN_OK, expected_values);
}

void zdnn_norm_basic_large_3d_neg() {
  uint32_t shape_i[] = {1, 10, 70};
  uint32_t shape_o[] = {1, 10, 1};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2];
  float input_a_values[num_io_buffer_values];
  float input_b_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_a_values);
  gen_random_float_array_neg(num_io_buffer_values, input_b_values);
  float expected_values[shape_o[1]];
  approximate_norm(input_a_values, input_b_values, expected_values, shape_i,
                   sizeof(shape_i) / sizeof(shape_i[0]));
  zdnn_norm_test(shape_i, shape_o, ZDNN_3D, input_a_values, input_b_values,
                 ZDNN_OK, expected_values);
}

/*
  -------------------------------------------------------------------------------
                                  Norm Basic
                                  Layout: NHWC
  -------------------------------------------------------------------------------
*/

void zdnn_norm_basic_small_nhwc() {
  uint32_t shape_i[] = {1, 1, 2, 6};
  uint32_t shape_o[] = {1, 1, 2, 1};

  float input_a_values[] = {1, 2, 3, 4, 5, 6, 5, 10, 15, 20, 25, 30};
  float input_b_values[] = {0, 1, 2, 3, 4, 5, 35, 40, 45, 50, 55, 60};
  float expected_values[shape_o[2]];
  approximate_norm(input_a_values, input_b_values, expected_values, shape_i,
                   sizeof(shape_i) / sizeof(shape_i[0]));
  zdnn_norm_test(shape_i, shape_o, ZDNN_NHWC, input_a_values, input_b_values,
                 ZDNN_OK, expected_values);
}

void zdnn_norm_basic_large_nhwc() {
  // Initialize the dimensions for our input tensor ZDNN_3DS
  uint32_t shape_i[] = {1, 1, 70, 180};
  uint32_t shape_o[] = {1, 1, 70, 1};

  int num_io_buffer_values = shape_i[0] * shape_i[1] * shape_i[2] * shape_i[3];
  float input_a_values[num_io_buffer_values];
  float input_b_values[num_io_buffer_values];
  gen_random_float_array_neg(num_io_buffer_values, input_a_values);
  gen_random_float_array_neg(num_io_buffer_values, input_b_values);
  float expected_values[shape_o[2]];
  approximate_norm(input_a_values, input_b_values, expected_values, shape_i,
                   sizeof(shape_i) / sizeof(shape_i[0]));
  zdnn_norm_test(shape_i, shape_o, ZDNN_NHWC, input_a_values, input_b_values,
                 ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_norm_basic_small_3d);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_norm_basic_large_3d__pos_neg);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_norm_basic_large_3d_neg);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_norm_basic_small_nhwc);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_norm_basic_large_nhwc);
  UNITY_END();
}
