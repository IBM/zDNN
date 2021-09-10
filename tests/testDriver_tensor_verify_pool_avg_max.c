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
  VERIFY_HW_ENV;
}

void tearDown(void) { /* This is run after EACH TEST */
}

#define NON_EXISTENT_FORMAT -1
#define NON_EXISTENT_DTYPE -1

void run_verify_pool_avg_max_tensors(
    uint32_t *input_shape, zdnn_data_layouts input_layout,
    zdnn_data_types input_dtype, uint32_t *output_shape,
    zdnn_data_layouts output_layout, zdnn_data_types output_dtype,
    zdnn_pool_padding padding_type, uint32_t kernel_height,
    uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,
    bool use_mismatch_dtype, zdnn_status expected_status) {

  // Create status to check status after verify calls
  zdnn_status status;

  // We don't care about the values for these tests so just pass the zero array
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      input_shape, input_layout, input_dtype, NO_CONCAT, true, ZERO_ARRAY);
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      output_shape, output_layout, output_dtype, NO_CONCAT, true, ZERO_ARRAY);

  // Special scenario. Test is checking what happens when input and output data
  // types don't match. alloc_ztensor_with_values() above transforms into real
  // ztensors, with ZDNN_DLFLOAT16. Forcibly break that for such tests.
  if (use_mismatch_dtype) {
    input_ztensor->transformed_desc->type = NON_EXISTENT_DTYPE;
  }

  // Make call to verify with our newly created ztensors and other inputs
  if ((status = verify_pool_avg_max_tensors(
           input_ztensor, padding_type, kernel_height, kernel_width,
           stride_height, stride_width, output_ztensor)) != expected_status) {
    TEST_FAIL_MESSAGE_FORMATTED(
        "Call to verify_pool_avg_max_tensors() returned zdnn_status %08x "
        "\"%s\" but we expected %08x \"%s\"",
        status, zdnn_get_status_message(status), expected_status,
        zdnn_get_status_message(expected_status));
  }

  // Cleanup
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/*
 * Simple test to confirm verification does not return any known error codes
 * with valid SAME_PADDING values
 */
void verify_same_pass() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 3, 3, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(
      input_shape, ZDNN_NHWC, FP32, output_shape, ZDNN_NHWC, FP32, SAME_PADDING,
      kernel_height, kernel_width, stride_height, stride_width, false, ZDNN_OK);
}

/*
 * Simple test to confirm verification passes with valid VALID_PADDING values
 */
void verify_valid_pass() {

  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 2, 2, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_OK);
}

/*
 * Verifying the input tensor with output. Should fail
 * because the input and output tensors have different dtypes
 */
void verify_dtype_mismatch_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 3, 3, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  // Setting output dtype to FP16 instead of FP32 should cause failure
  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP16, SAME_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  true, ZDNN_INVALID_TYPE);
}

/*
 * Verifying the input tensor with output. Should fail
 * because the input and output tensor have different formats.
 */
void verify_format_mismatch_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 2, 2, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  // Setting input format to ZDNN_HWCK instead of NHWC should cause failure
  run_verify_pool_avg_max_tensors(input_shape, ZDNN_HWCK, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_FORMAT);
}

/*
 * Verifying the input tensor with output. Should fail
 * because the innermost dimension of the input and output are different
 */
void verify_bad_c_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[3] to 4 instead of 1 should cause failure
  uint32_t output_shape[] = {1, 3, 3, 4};

  uint32_t kernel_height = 4;
  uint32_t kernel_width = 4;
  uint32_t stride_height = 3;
  uint32_t stride_width = 3;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, SAME_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the input tensor with output. Should fail
 * because the outermost dimension of the input and output are different
 */
void verify_bad_n_fail() {

  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[0] to 4 instead of 1 should cause failure
  uint32_t output_shape[] = {4, 3, 3, 1};

  uint32_t kernel_height = 4;
  uint32_t kernel_width = 4;
  uint32_t stride_height = 3;
  uint32_t stride_width = 3;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, SAME_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Simple test to confirm verification does not return any known error codes
 * with valid SAME_PADDING values when strides are 0
 */
void verify_0_strides_pass() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 1, 1, 1};

  uint32_t kernel_height = 8;
  uint32_t kernel_width = 5;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_OK);
}

/*
 * Verifying the 0 stride values. Should fail
 * because the the padding_type must be VALID_PADDING when strides are 0
 */
void verify_0_strides_same_padding_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 1, 1, 1};

  uint32_t kernel_height = 8;
  uint32_t kernel_width = 5;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, SAME_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_STRIDE_PADDING);
}

/*
 * Verifying the 0 stride values. Should fail
 * because the second dimension stride value is greater than 0,
 * and the third dimension stride value is 0.
 */
void verify_0_strides_stride_width_not_zero_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 3, 3, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 0;
  // Setting stride_width to 1 instead of 0 should cause failure
  uint32_t stride_width = 1;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_STRIDES);
}

/*
 * Verifying the stride values. Should fail because the third dimension stride
 * value is greater than 0, and the second dimension stride value is 0.
 */
void verify_0_strides_stride_height_not_zero_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 3, 3, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  // Setting stride_height to 1 instead of 0 should cause failure
  uint32_t stride_height = 1;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_STRIDES);
}

/*
 * Verifying the input tensor with output. Should fail
 * because stride values are both 0 and input dimension 2 is not equal to
 * window dim 2
 */
void verify_0_strides_bad_kernel_width_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 1, 1, 1};

  uint32_t kernel_height = 8;
  // Setting kernel_width to 4 instead of 5 should cause failure
  uint32_t kernel_width = 4;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the input tensor with output. Should fail
 * because stride values are both 0 and input dimension 3 is not equal
 * to window_size dimension 3
 */
void verify_0_strides_bad_kernel_height_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 1, 1, 1};

  // Setting kernel_height to 7 instead of 8 should cause failure
  uint32_t kernel_height = 7;
  uint32_t kernel_width = 5;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the output tensor. Should fail because stride values are both 0 and
 * output dimensions 2 and 3 are not equal to 1
 */
void verify_0_strides_bad_out_width_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[2] to 2 instead of 1 should cause failure
  uint32_t output_shape[] = {1, 1, 2, 1};

  uint32_t kernel_height = 8;
  uint32_t kernel_width = 5;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the output tensor. Should fail because stride values are both 0 and
 * output dimensions 2 and 3 are not equal to 1
 */
void verify_0_strides_bad_out_height_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[1] to 2 instead of 1 should cause failure
  uint32_t output_shape[] = {1, 2, 1, 1};

  uint32_t kernel_height = 8;
  uint32_t kernel_width = 5;
  uint32_t stride_height = 0;
  uint32_t stride_width = 0;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the input and window values. Should fail
 * because the second dimension window value is greater than the
 * second dimension of the input tensor and the padding is VALID.
 */
void verify_valid_bad_kernel_width_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 2, 2, 1};

  uint32_t kernel_height = 3;
  // Setting kernel_width to 6 instead of 2 should cause failure
  uint32_t kernel_width = 6;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the input and window values. Should fail
 * because the third dimension window value is greater than the
 * third dimension of the input tensor and the padding is VALID.
 */
void verify_valid_bad_kernel_height_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  uint32_t output_shape[] = {1, 2, 2, 1};

  // Setting kernel_width to 9 instead of 3 should cause failure
  uint32_t kernel_height = 9;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the output tensor has the correct shape given the padding.
 * This test should fail because the dimension 3 of the output tensor is not
 * equal to the expected value and the padding is VALID_PADDING
 */
void verify_valid_bad_out_width_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[2] to 3 instead of 2 should cause expected failure
  uint32_t output_shape[] = {1, 2, 3, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the output tensor has the correct shape given the padding. This
 * test should fail because the dimension 2 of the output tensor is not equal to
 * the expected value and the padding is VALID_PADDING
 */
void verify_valid_bad_out_height_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[1] to 3 instead of 2 should cause expected failure
  uint32_t output_shape[] = {1, 3, 2, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, VALID_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the output tensor has the correct shape given the padding. This
 * test should fail because the dimension 3 of the output tensor is not equal to
 * the expected value and the padding is SAME_PADDING
 */
void verify_same_bad_out_width_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[2] to 4 instead of 3 should cause expected failure
  uint32_t output_shape[] = {1, 3, 4, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, SAME_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

/*
 * Verifying the output tensor has the correct shape given the padding. This
 * test should fail because the dimension 2 of the output tensor is not equal to
 * the expected value and the padding is SAME_PADDING
 */
void verify_same_bad_out_height_fail() {
  uint32_t input_shape[] = {1, 8, 5, 1};
  // Setting shape[1] to 4 instead of 3 should cause expected failure
  uint32_t output_shape[] = {1, 4, 3, 1};

  uint32_t kernel_height = 3;
  uint32_t kernel_width = 2;
  uint32_t stride_height = 3;
  uint32_t stride_width = 2;

  run_verify_pool_avg_max_tensors(input_shape, ZDNN_NHWC, FP32, output_shape,
                                  ZDNN_NHWC, FP32, SAME_PADDING, kernel_height,
                                  kernel_width, stride_height, stride_width,
                                  false, ZDNN_INVALID_SHAPE);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(verify_same_pass);
  RUN_TEST(verify_valid_pass);

  RUN_TEST(verify_format_mismatch_fail);
  RUN_TEST(verify_dtype_mismatch_fail);

  RUN_TEST(verify_bad_c_fail);
  RUN_TEST(verify_bad_n_fail);

  RUN_TEST(verify_0_strides_pass);
  RUN_TEST(verify_0_strides_same_padding_fail);
  RUN_TEST(verify_0_strides_stride_width_not_zero_fail);
  RUN_TEST(verify_0_strides_stride_height_not_zero_fail);
  RUN_TEST(verify_0_strides_bad_kernel_width_fail);
  RUN_TEST(verify_0_strides_bad_kernel_height_fail);
  RUN_TEST(verify_0_strides_bad_out_width_fail);
  RUN_TEST(verify_0_strides_bad_out_height_fail);

  RUN_TEST(verify_valid_bad_kernel_width_fail);
  RUN_TEST(verify_valid_bad_kernel_height_fail);
  RUN_TEST(verify_valid_bad_out_width_fail);
  RUN_TEST(verify_valid_bad_out_height_fail);

  RUN_TEST(verify_same_bad_out_width_fail);
  RUN_TEST(verify_same_bad_out_height_fail);

  return UNITY_END();
}
