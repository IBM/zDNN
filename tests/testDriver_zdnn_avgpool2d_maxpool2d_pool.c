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

#include "common_pool.h"

void setUp(void) { /* This is run before EACH TEST */

  // note: maxpool2d is actually OK with default tolerance values, but avgpool2d
  // needs custom tolerance

  tol_bfloat.ulps = 64;
  tol_bfloat.epsilon_mult = (0.1 / EPSILON_BFLOAT) + 1;

  tol_fp16.ulps = 64;
  tol_fp16.epsilon_mult = (0.1 / EPSILON_FP16) + 1;

  tol_fp32.ulps = 64 * 16384;
  tol_fp32.epsilon_mult = (0.1 / EPSILON_FLOAT) + 1;

  VERIFY_HW_ENV;
}

void tearDown(void) { /* This is run after EACH TEST */
}

/*
 * Simple test of basic pool with non-zero strides and SAME_PADDING
 */
void maxpool2d_same_basic() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  /* Visualization of input values
    [[
      [[1, 10], [2, 20], [3, 30]],
      [[4, 40], [5, 50], [6, 60]],
      [[7, 70], [8, 80], [9, 90]]
    ]]
  */
  uint32_t input_shape[] = {1, 3, 3, 2};
  float input_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                          50, 6,  60, 7,  70, 8,  80, 9,  90};

  // Input pooling arguments
  zdnn_pool_padding padding_type = SAME_PADDING;
  uint32_t kernel_height = 2;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 2;
  uint32_t stride_width = 2;

  /* Visualization of expected values
    [[
      [[5, 50], [6, 60]],
      [[8, 80], [9, 90]]
    ]]
  */
  uint32_t output_shape[] = {1, 2, 2, 2};
  float expected_values[] = {5, 50, 6, 60, 8, 80, 9, 90};

  test_pool_function(NNPA_MAXPOOL2D, input_shape, layout, false, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, false,
                     expected_values);
}

/*
 * Simple test of basic pool with non-zero strides and VALID_PADDING
 */
void maxpool2d_valid_basic() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  /* Visualization of input values
    [[
      [[1, 10], [2, 20], [3, 30]],
      [[4, 40], [5, 50], [6, 60]],
      [[7, 70], [8, 80], [9, 90]]
    ]]
  */
  uint32_t input_shape[] = {1, 3, 3, 2};
  float input_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                          50, 6,  60, 7,  70, 8,  80, 9,  90};

  // Input pooling arguments
  zdnn_pool_padding padding_type = VALID_PADDING;
  uint32_t kernel_height = 2;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 2;
  uint32_t stride_width = 2;

  /* Visualization of expected values
    [[
      [[5, 50]],
    ]]
  */
  uint32_t output_shape[] = {1, 1, 1, 2};
  float expected_values[] = {5.0, 50.0};

  test_pool_function(NNPA_MAXPOOL2D, input_shape, layout, false, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, false,
                     expected_values);
}

/*
 * Simple test of basic pool with non-zero strides and SAME_PADDING
 */
void avgpool2d_same_basic() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  /* Visualization of input values
    [[
      [[1, 10], [2, 20], [3, 30]],
      [[4, 40], [5, 50], [6, 60]],
      [[7, 70], [8, 80], [9, 90]]
    ]]
  */
  uint32_t input_shape[] = {1, 3, 3, 2};
  float input_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                          50, 6,  60, 7,  70, 8,  80, 9,  90};

  // Input pooling arguments
  zdnn_pool_padding padding_type = SAME_PADDING;
  uint32_t kernel_height = 2;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 2;
  uint32_t stride_width = 2;

  /* Visualization of expected values
    [[
         [[ 3, 30],   [ 4.5, 45]],
         [[ 7.5, 75], [ 9, 90]]
    ]]
  */
  uint32_t output_shape[] = {1, 2, 2, 2};
  float expected_values[] = {3.0, 30.0, 4.5, 45, 7.5, 75, 9, 90};

  test_pool_function(NNPA_AVGPOOL2D, input_shape, layout, false, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, false,
                     expected_values);
}

/*
 * Simple test of basic pool with non-zero strides and VALID_PADDING
 */
void avgpool2d_valid_basic() {
  zdnn_data_layouts layout = ZDNN_NHWC;

  /* Visualization of input values
    [[
      [[1, 10], [2, 20], [3, 30]],
      [[4, 40], [5, 50], [6, 60]],
      [[7, 70], [8, 80], [9, 90]]
    ]]
  */
  uint32_t input_shape[] = {1, 3, 3, 2};
  float input_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                          50, 6,  60, 7,  70, 8,  80, 9,  90};

  // Input pooling arguments
  zdnn_pool_padding padding_type = VALID_PADDING;
  uint32_t kernel_height = 2;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 2;
  uint32_t stride_width = 2;

  /* Visualization of expected values
    [[
      [[3, 30]],
    ]]
  */
  uint32_t output_shape[] = {1, 1, 1, 2};
  float expected_values[] = {3.0, 30.0};

  test_pool_function(NNPA_AVGPOOL2D, input_shape, layout, false, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, false,
                     expected_values);
}

/*
 * Simple test of basic pool with zero strides
 */
void zero_strides(nnpa_function_code function_code) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  /* Visualization of input values
    [[
      [[1, 10], [2, 20], [3, 30]],
      [[4, 40], [5, 50], [6, 60]],
      [[7, 70], [8, 80], [9, 90]]
    ]]
  */
  uint32_t input_shape[] = {1, 3, 3, 2};
  float input_values[] = {1,  10, 2,  20, 3,  30, 4,  40, 5,
                          50, 6,  60, 7,  70, 8,  80, 9,  90};

  // Input pooling arguments
  zdnn_pool_padding padding_type = VALID_PADDING;
  uint32_t kernel_height = 3;
  uint32_t kernel_width = 3;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  /* Visualization of expected values
    [[
      [[9, 90]]
    ]]
  */
  uint32_t output_shape[] = {1, 1, 1, 2};

  /* Visualization of MAXPOOL2D expected values
   [[
     [[9, 90]]
   ]]
 */

  /* Visualization of AVGPOOL2D expected values
    [[
      [[5, 50]]
    ]]
  */

  float expected_values[] = {0, 0};
  if (function_code == NNPA_MAXPOOL2D) {
    expected_values[0] = 9;
    expected_values[1] = 90;
  } else {
    expected_values[0] = 5;
    expected_values[1] = 50;
  }

  test_pool_function(function_code, input_shape, layout, false, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, false,
                     expected_values);
}

void maxpool2d_zero_strides() { zero_strides(NNPA_MAXPOOL2D); }

void avgpool2d_zero_strides() { zero_strides(NNPA_AVGPOOL2D); }

/*
 * Check that we don't hit a condition code when using an unexpected padding
 * type.
 */
void unexpected_padding_fail(nnpa_function_code function_code) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  uint32_t input_shape[] = {1, 3, 3, 2};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  // Set this to the first unused padding type. Then if a new one is
  // supported, this should fail and we remember to update our code and
  // documentation.
  zdnn_pool_padding padding_type = 2;
  uint32_t kernel_height = 1;
  uint32_t kernel_width = 1;
  uint32_t stride_height = 1;
  uint32_t stride_width = 1;

  // kernel and strides of 1 should basically copy the input (if the padding
  // type was valid)
  uint32_t *output_shape = input_shape;
  float *expected_values = input_values;

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_FUNC_RC_F000,
                     true, expected_values);
}

void maxpool2d_unexpected_padding_fail() {
  unexpected_padding_fail(NNPA_MAXPOOL2D);
}

void avgpool2d_unexpected_padding_fail() {
  unexpected_padding_fail(NNPA_AVGPOOL2D);
}

/*
 * Check that we don't hit a condition code when using 0 strides and the
 * largest kernel size.
 */
void zero_strides_max_kernel_dims_pass(nnpa_function_code function_code) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  uint32_t input_shape[] = {1, MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE,
                            MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  zdnn_pool_padding padding_type = VALID_PADDING;
  uint32_t kernel_height = input_shape[1];
  uint32_t kernel_width = input_shape[2];
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  uint32_t output_shape[] = {1, 1, 1, 1};
  // Since all input values are the same, they should average to the same.
  float *expected_values = input_values;

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, true,
                     expected_values);
}

void maxpool2d_zero_strides_max_kernel_dims_pass() {
  zero_strides_max_kernel_dims_pass(NNPA_MAXPOOL2D);
}

void avgpool2d_zero_strides_max_kernel_dims_pass() {
  zero_strides_max_kernel_dims_pass(NNPA_AVGPOOL2D);
}

/*
 * Check that we hit the expected condition code when using 0 strides and the
 * over the largest kernel size.
 */
void zero_strides_max_kernel_height_fail(nnpa_function_code function_code) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_kernel_max is a valid tensor dimension size but is too large for a
  // kernel. This should lead to a condition code from the NNPA. If not,
  // update the test constant and the API documentation to the new value.
  uint32_t over_kernel_max = MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE + 1;

  uint32_t input_shape[] = {1, over_kernel_max, 5, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  zdnn_pool_padding padding_type = VALID_PADDING;
  uint32_t kernel_height = input_shape[1];
  uint32_t kernel_width = input_shape[2];
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  uint32_t output_shape[] = {1, 1, 1, 1};
  // Output values don't really matter as we expect failure status.
  float *expected_values = input_values;

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_FUNC_RC_F001,
                     true, expected_values);
}

void maxpool2d_zero_strides_max_kernel_height_fail() {
  zero_strides_max_kernel_height_fail(NNPA_MAXPOOL2D);
}

void avgpool2d_zero_strides_max_kernel_height_fail() {
  zero_strides_max_kernel_height_fail(NNPA_AVGPOOL2D);
}

/*
 * Check that we hit the expected condition code when using 0 strides and the
 * over the largest kernel size.
 */
void zero_strides_max_kernel_width_fail(nnpa_function_code function_code) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_kernel_max is a valid tensor dimension size but is too large for a
  // kernel. This should lead to a condition code from the NNPA. If not,
  // update the test constant and the API documentation to the new value.
  uint32_t over_kernel_max = MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE + 1;

  uint32_t input_shape[] = {1, 8, over_kernel_max, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  zdnn_pool_padding padding_type = VALID_PADDING;
  uint32_t kernel_height = input_shape[1];
  uint32_t kernel_width = input_shape[2];
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  uint32_t output_shape[] = {1, 1, 1, 1};
  // Output values don't really matter as we expect failure status.
  float *expected_values = input_values;

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_FUNC_RC_F001,
                     true, expected_values);
}

void maxpool2d_zero_strides_max_kernel_width_fail() {
  zero_strides_max_kernel_width_fail(NNPA_MAXPOOL2D);
}

void avgpool2d_zero_strides_max_kernel_width_fail() {
  zero_strides_max_kernel_width_fail(NNPA_AVGPOOL2D);
}

/*
 * Check that we don't hit a condition code when using nonzero strides and the
 * largest kernel size.
 */
void max_kernel_pass(nnpa_function_code function_code,
                     zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  uint32_t input_shape[] = {1, MAXIMUM_POOL_NONZERO_STRIDES_KERNEL_SIZE,
                            MAXIMUM_POOL_NONZERO_STRIDES_KERNEL_SIZE, 1};

  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  uint32_t kernel_height = input_shape[1];
  uint32_t kernel_width = input_shape[2];
  uint32_t stride_height = 1;
  uint32_t stride_width = 1;

  uint32_t output_shape[] = {1, 1, 1, 1};

  // Since all input values are the same, they should average to the same.
  float *expected_values = input_values;

  // use input_shape[] as output shape if SAME_PADDING since stride
  // height/width are 1
  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width,
                     padding_type == SAME_PADDING ? input_shape : output_shape,
                     layout, ZDNN_OK, true, expected_values);
}

void maxpool2d_max_kernel_valid_padding_pass() {
  max_kernel_pass(NNPA_MAXPOOL2D, VALID_PADDING);
}

void maxpool2d_max_kernel_same_padding_pass() {
  max_kernel_pass(NNPA_MAXPOOL2D, SAME_PADDING);
}

void avgpool2d_max_kernel_valid_padding_pass() {
  max_kernel_pass(NNPA_AVGPOOL2D, VALID_PADDING);
}

void avgpool2d_max_kernel_same_padding_pass() {
  max_kernel_pass(NNPA_AVGPOOL2D, SAME_PADDING);
}

/*
 * Check that we hit the expected condition code when using 0 strides and the
 * over the largest kernel size.
 */
void max_kernel_height_fail(nnpa_function_code function_code,
                            zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_kernel_max is a valid tensor dimension size but is too large for a
  // kernel. This should lead to a condition code from the NNPA. If not,
  // update the test constant and the API documentation to the new value.
  uint32_t over_kernel_max = MAXIMUM_POOL_NONZERO_STRIDES_KERNEL_SIZE + 1;

  uint32_t input_shape[] = {1, over_kernel_max, 5, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  uint32_t kernel_height = input_shape[1];
  uint32_t kernel_width = input_shape[2];
  uint32_t stride_height = 1;
  uint32_t stride_width = 1;

  uint32_t output_shape[] = {1, 1, 1, 1};
  // Output values don't really matter as we expect failure status.
  float *expected_values = input_values;

  // use input_shape[] as output shape if SAME_PADDING since stride
  // height/width are 1
  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width,
                     padding_type == SAME_PADDING ? input_shape : output_shape,
                     layout, ZDNN_FUNC_RC_F002, true, expected_values);
}

void maxpool2d_max_kernel_valid_padding_height_fail() {
  max_kernel_height_fail(NNPA_MAXPOOL2D, VALID_PADDING);
}

void maxpool2d_max_kernel_same_padding_height_fail() {
  max_kernel_height_fail(NNPA_MAXPOOL2D, SAME_PADDING);
}

void avgpool2d_max_kernel_valid_padding_height_fail() {
  max_kernel_height_fail(NNPA_AVGPOOL2D, VALID_PADDING);
}

void avgpool2d_max_kernel_same_padding_height_fail() {
  max_kernel_height_fail(NNPA_AVGPOOL2D, SAME_PADDING);
}

/*
 * Check that we hit the expected condition code when using 0 strides and the
 * over the largest kernel size.
 */
void max_kernel_width_fail(nnpa_function_code function_code,
                           zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_kernel_max is a valid tensor dimension size but is too large for a
  // kernel. This should lead to a condition code from the NNPA. If not,
  // update the test constant and the API documentation to the new value.
  uint32_t over_kernel_max = MAXIMUM_POOL_NONZERO_STRIDES_KERNEL_SIZE + 1;

  uint32_t input_shape[] = {1, 8, over_kernel_max, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  uint32_t kernel_height = input_shape[1];
  uint32_t kernel_width = input_shape[2];
  uint32_t stride_height = 1;
  uint32_t stride_width = 1;

  uint32_t output_shape[] = {1, 1, 1, 1};
  // Output values don't really matter as we expect failure status.
  float *expected_values = input_values;

  // use input_shape[] as output shape if SAME_PADDING since stride
  // height/width are 1
  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width,
                     padding_type == SAME_PADDING ? input_shape : output_shape,
                     layout, ZDNN_FUNC_RC_F002, true, expected_values);
}

void maxpool2d_max_kernel_valid_padding_width_fail() {
  max_kernel_width_fail(NNPA_MAXPOOL2D, VALID_PADDING);
}

void maxpool2d_max_kernel_same_padding_width_fail() {
  max_kernel_width_fail(NNPA_MAXPOOL2D, SAME_PADDING);
}

void avgpool2d_max_kernel_valid_padding_width_fail() {
  max_kernel_width_fail(NNPA_AVGPOOL2D, VALID_PADDING);
}

void avgpool2d_max_kernel_same_padding_width_fail() {
  max_kernel_width_fail(NNPA_AVGPOOL2D, SAME_PADDING);
}

/*
 * Check that we don't hit a condition code when using nonzero strides and the
 * largest stride size.
 */
void max_stride_pass(nnpa_function_code function_code,
                     zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  uint32_t input_shape[] = {1, 2 * MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE,
                            2 * MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  uint32_t kernel_height = input_shape[1] / 2;
  uint32_t kernel_width = input_shape[2] / 2;
  uint32_t stride_height = input_shape[1] / 2;
  uint32_t stride_width = input_shape[2] / 2;

  // With stride and kernel set to exactly 1/2 of input, we'd expect output to
  // end with a height and width of exactly 2.
  // These dimensions work for both VALID_PADDING and VALID_PADDING
  uint32_t output_shape[] = {1, 2, 2, 1};
  // Since all input values are the same, they should average to the same.
  float expected_values[] = {input_values[0], input_values[0], input_values[0],
                             input_values[0]};

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_OK, true,
                     expected_values);
}

void maxpool2d_max_stride_valid_padding_pass() {
  max_stride_pass(NNPA_MAXPOOL2D, VALID_PADDING);
}

void maxpool2d_max_stride_same_padding_pass() {
  max_stride_pass(NNPA_MAXPOOL2D, SAME_PADDING);
}

void avgpool2d_max_stride_valid_padding_pass() {
  max_stride_pass(NNPA_AVGPOOL2D, VALID_PADDING);
}

void avgpool2d_max_stride_same_padding_pass() {
  max_stride_pass(NNPA_AVGPOOL2D, SAME_PADDING);
}

/*
 * Check that we hit the expected condition code when using just over the
 * largest nonzero strides allowed
 */
void max_stride_height_fail(nnpa_function_code function_code,
                            zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_stride_max is a valid tensor dimension size but is too large for a
  // stride. This should lead to a condition code from the AIU. If not, update
  // the test constant and the API documentation to the new value.
  uint32_t over_stride_max = MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE + 1;

  // Use 2 * X here to make determining exected shape and values easier.
  uint32_t input_shape[] = {1, 2 * over_stride_max,
                            2 * MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  uint32_t kernel_height = input_shape[1] / 2;
  uint32_t kernel_width = input_shape[2] / 2;
  uint32_t stride_height = input_shape[1] / 2;
  uint32_t stride_width = input_shape[2] / 2;

  // With stride and kernel set to exactly 1/2 of input, we'd expect output to
  // end with a height and width of exactly 2.
  uint32_t output_shape[] = {1, 2, 2, 1};
  // Output values don't really matter as we expect failure status.
  float expected_values[] = {input_values[0], input_values[0], input_values[0],
                             input_values[0]};

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_FUNC_RC_F003,
                     true, expected_values);
}

void maxpool2d_max_stride_valid_padding_height_fail() {
  max_stride_height_fail(NNPA_MAXPOOL2D, VALID_PADDING);
}

void maxpool2d_max_stride_same_padding_height_fail() {
  max_stride_height_fail(NNPA_MAXPOOL2D, SAME_PADDING);
}

void avgpool2d_max_stride_valid_padding_height_fail() {
  max_stride_height_fail(NNPA_AVGPOOL2D, VALID_PADDING);
}

void avgpool2d_max_stride_same_padding_height_fail() {
  max_stride_height_fail(NNPA_AVGPOOL2D, SAME_PADDING);
}

/*
 * Check that we hit the expected condition code when using just over the
 * largest nonzero strides allowed
 */
void max_stride_width_fail(nnpa_function_code function_code,
                           zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  // over_stride_max is a valid tensor dimension size but is too large for a
  // stride. This should lead to a condition code from the AIU. If not, update
  // the test constant and the API documentation to the new value.
  uint32_t over_stride_max = MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE + 1;

  // Use 2 * X here to make determining exected shape and values easier.
  uint32_t input_shape[] = {1, 2 * MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE,
                            2 * over_stride_max, 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  // Input pooling arguments
  uint32_t kernel_height = input_shape[1] / 2;
  uint32_t kernel_width = input_shape[2] / 2;
  uint32_t stride_height = input_shape[1] / 2;
  uint32_t stride_width = input_shape[2] / 2;

  // With stride and kernel set to exactly 1/2 of input, we'd expect output to
  // end with a height and width of exactly 2.
  uint32_t output_shape[] = {1, 2, 2, 1};
  // Output values don't really matter as we expect failure status.
  float expected_values[] = {input_values[0], input_values[0], input_values[0],
                             input_values[0]};

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, output_shape, layout, ZDNN_FUNC_RC_F003,
                     true, expected_values);
}

void maxpool2d_max_stride_valid_padding_width_fail() {
  max_stride_width_fail(NNPA_MAXPOOL2D, VALID_PADDING);
}

void maxpool2d_max_stride_same_padding_width_fail() {
  max_stride_width_fail(NNPA_MAXPOOL2D, SAME_PADDING);
}

void avgpool2d_max_stride_valid_padding_width_fail() {
  max_stride_width_fail(NNPA_AVGPOOL2D, VALID_PADDING);
}

void avgpool2d_max_stride_same_padding_width_fail() {
  max_stride_width_fail(NNPA_AVGPOOL2D, SAME_PADDING);
}

/*
 * Check that we hit the expected condition code when using just over the
 * largest input height/width allowed when strides are non-zero
 */
void nonzero_strides_bad_height_or_width_fail(nnpa_function_code function_code,
                                              bool bad_height, bool bad_width,
                                              zdnn_pool_padding padding_type) {
  zdnn_data_layouts layout = ZDNN_NHWC;

  uint32_t input_shape[] = {
      1, MAXIMUM_POOL_NONZERO_STRIDES_HEIGHT_WIDTH + (bad_height ? 1 : 0),
      MAXIMUM_POOL_NONZERO_STRIDES_HEIGHT_WIDTH + (bad_width ? 1 : 0), 1};
  // Just repeat the same value rather than try and genarate a unique array of
  // values for this test.
  float input_values[] = {42};

  uint32_t kernel_height = 1;
  uint32_t kernel_width = 1;
  uint32_t stride_height = 1;
  uint32_t stride_width = 1;

  // when kernel height/width and stride height/width are all 1, output shape is
  // same as input's

  // Output values don't really matter as we expect failure status.

  test_pool_function(function_code, input_shape, layout, true, input_values,
                     padding_type, kernel_height, kernel_width, stride_height,
                     stride_width, input_shape, layout, ZDNN_FUNC_RC_F004, true,
                     ZERO_ARRAY);
}

void maxpool2d_non_zero_strides_valid_padding_height_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_MAXPOOL2D, true, false,
                                           VALID_PADDING);
}

void maxpool2d_non_zero_strides_same_padding_height_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_MAXPOOL2D, true, false,
                                           SAME_PADDING);
}

void avgpool2d_non_zero_strides_valid_padding_height_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_AVGPOOL2D, true, false,
                                           VALID_PADDING);
}

void avgpool2d_non_zero_strides_same_padding_height_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_AVGPOOL2D, true, false,
                                           SAME_PADDING);
}

void maxpool2d_non_zero_strides_valid_padding_width_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_MAXPOOL2D, false, true,
                                           VALID_PADDING);
}

void maxpool2d_non_zero_strides_same_padding_width_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_MAXPOOL2D, false, true,
                                           SAME_PADDING);
}

void avgpool2d_non_zero_strides_valid_padding_width_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_AVGPOOL2D, false, true,
                                           VALID_PADDING);
}

void avgpool2d_non_zero_strides_same_padding_width_fail() {
  nonzero_strides_bad_height_or_width_fail(NNPA_AVGPOOL2D, false, true,
                                           SAME_PADDING);
}

int main(int argc, char *argv[]) {
  UNITY_BEGIN();

  RUN_TEST_ALL_DATATYPES(maxpool2d_same_basic);
  RUN_TEST_ALL_DATATYPES(maxpool2d_valid_basic);
  RUN_TEST_ALL_DATATYPES(avgpool2d_same_basic);
  RUN_TEST_ALL_DATATYPES(avgpool2d_valid_basic);

  RUN_TEST_ALL_DATATYPES(maxpool2d_zero_strides);
  RUN_TEST_ALL_DATATYPES(avgpool2d_zero_strides);

  // Tests to confirm we get the expected condition codes from the NNPA.
  // Technically these don't test our library. However we document these
  // in our API. These tests should fail if hardware changes the underlying
  // conditions meaning we need to update our documentation (and tests).
  {
    RUN_TEST_ALL_DATATYPES(maxpool2d_unexpected_padding_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_unexpected_padding_fail);

    RUN_TEST_ALL_DATATYPES(maxpool2d_zero_strides_max_kernel_dims_pass);
    RUN_TEST_ALL_DATATYPES(maxpool2d_zero_strides_max_kernel_height_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_zero_strides_max_kernel_width_fail);

    RUN_TEST_ALL_DATATYPES(avgpool2d_zero_strides_max_kernel_dims_pass);
    RUN_TEST_ALL_DATATYPES(avgpool2d_zero_strides_max_kernel_height_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_zero_strides_max_kernel_width_fail);

    RUN_TEST_ALL_DATATYPES(maxpool2d_max_kernel_valid_padding_pass);
    RUN_TEST_ALL_DATATYPES(maxpool2d_max_kernel_same_padding_pass);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_kernel_valid_padding_pass);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_kernel_same_padding_pass);

    RUN_TEST_ALL_DATATYPES(maxpool2d_max_kernel_valid_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_max_kernel_same_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_kernel_valid_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_kernel_same_padding_height_fail);

    RUN_TEST_ALL_DATATYPES(maxpool2d_max_kernel_valid_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_max_kernel_same_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_kernel_valid_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_kernel_same_padding_width_fail);

    RUN_TEST_ALL_DATATYPES(maxpool2d_max_stride_valid_padding_pass);
    RUN_TEST_ALL_DATATYPES(maxpool2d_max_stride_same_padding_pass);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_stride_valid_padding_pass);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_stride_same_padding_pass);

    RUN_TEST_ALL_DATATYPES(maxpool2d_max_stride_valid_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_max_stride_same_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_stride_valid_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_stride_same_padding_height_fail);

    RUN_TEST_ALL_DATATYPES(maxpool2d_max_stride_valid_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_max_stride_same_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_stride_valid_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_max_stride_same_padding_width_fail);

    RUN_TEST_ALL_DATATYPES(
        maxpool2d_non_zero_strides_valid_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_non_zero_strides_same_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(
        avgpool2d_non_zero_strides_valid_padding_height_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_non_zero_strides_same_padding_height_fail);

    RUN_TEST_ALL_DATATYPES(maxpool2d_non_zero_strides_valid_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(maxpool2d_non_zero_strides_same_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_non_zero_strides_valid_padding_width_fail);
    RUN_TEST_ALL_DATATYPES(avgpool2d_non_zero_strides_same_padding_width_fail);
  }

  return UNITY_END();
}
