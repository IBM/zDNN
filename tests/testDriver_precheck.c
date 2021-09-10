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

#include "testsupport.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

zdnn_ztensor ztensor_input1, ztensor_input2, ztensor_input3, ztensor_output1,
    ztensor_output2;

zdnn_tensor_desc pre_tfrmd_desc, input1_tfrmd_desc, input2_tfrmd_desc,
    input3_tfrmd_desc, output1_tfrmd_desc, output2_tfrmd_desc;

void create_garbage_tensors();

void setUp(void) { /* This is run before EACH TEST */
  create_garbage_tensors();
}

void tearDown(void) {}

/*********************************************************************
 * The goal is to verify if the verifier routine is invoked when
 * precheck_enabled = true, not if the verifier routine returns the
 * correct status code (which is testDriver_tensor_verify*.c's job).
 *
 * On environment equipped with AIU, all testcases will cause program
 * termination due to DXG rather than issuing a non-ZDNN_OK status.
 * *******************************************************************/

/// Create garbage input/output tensors that are guaranteed to fail any AIU op
void create_garbage_tensors() {
  precheck_enabled = true;

  uint32_t dim4 = 1, dim3 = 1, dim2 = 1, dim1 = 1;
  zdnn_data_layouts layout = ZDNN_NHWC;
  zdnn_data_types type = FP16;

  zdnn_init_pre_transformed_desc(layout, type, &pre_tfrmd_desc, dim4, dim3,
                                 dim2, dim1);

  // all inputs and outputs same shape
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &input1_tfrmd_desc);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &input2_tfrmd_desc);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &input3_tfrmd_desc);

  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &output1_tfrmd_desc);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &output2_tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &input1_tfrmd_desc,
                                &ztensor_input1);
  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &input2_tfrmd_desc,
                                &ztensor_input2);
  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &input3_tfrmd_desc,
                                &ztensor_input3);
  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &output1_tfrmd_desc,
                                &ztensor_output1);
  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &output2_tfrmd_desc,
                                &ztensor_output2);

  // all input tensors are features, all output tensors are kernels.
  ztensor_output1.transformed_desc->format = ZDNN_FORMAT_4DKERNEL;
  ztensor_output2.transformed_desc->format = ZDNN_FORMAT_4DKERNEL;
}

void bad_element_wise() {
  zdnn_status status =
      zdnn_add(&ztensor_input1, &ztensor_input2, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(status != ZDNN_OK,
                                "Expected failure status but got %d \"%s\"",
                                status, zdnn_get_status_message(status));
}

void bad_batchnorm() {
  zdnn_status status = zdnn_batchnorm(&ztensor_input1, &ztensor_input2,
                                      &ztensor_input3, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(status != ZDNN_OK,
                                "Expected failure status but got %d \"%s\"",
                                status, zdnn_get_status_message(status));
}

void bad_lstm() {
  // Force a type mismatch so we should fail precheck verification.
  ztensor_input1.transformed_desc->type = ZDNN_DLFLOAT16;
  ztensor_input2.transformed_desc->type = FP32;

  zdnn_status exp_status = ZDNN_INVALID_TYPE;
  zdnn_status status =
      zdnn_lstm(&ztensor_input1, &ztensor_input2, &ztensor_input3,
                &ztensor_input1, &ztensor_input2, &ztensor_input3,
                &ztensor_input1, FWD, NULL, &ztensor_output1, &ztensor_output2);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

void bad_matmul_op_with_bias_addition() {
  zdnn_status status =
      zdnn_matmul_op(&ztensor_input1, &ztensor_input2, &ztensor_input3,
                     MATMUL_OP_ADDITION, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(status != ZDNN_OK,
                                "Expected failure status but got %d \"%s\"",
                                status, zdnn_get_status_message(status));
}

void bad_matmul_bcast_op_with_bias_addition() {
  zdnn_status status =
      zdnn_matmul_bcast_op(&ztensor_input1, &ztensor_input2, &ztensor_input3,
                           MATMUL_BCAST_OP_ADDITION, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(status != ZDNN_OK,
                                "Expected failure status but got %d \"%s\"",
                                status, zdnn_get_status_message(status));
}

void bad_pool() {
  zdnn_status status =
      zdnn_avgpool2d(&ztensor_input1, 1, 1, 1, 1, 1, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(status != ZDNN_OK,
                                "Expected failure status but got %d \"%s\"",
                                status, zdnn_get_status_message(status));
}

void negative_relu_clipping() {
  VERIFY_HW_ENV; // zdnn_relu drives HW conversion before precheck
  ztensor_output1.transformed_desc->format = ZDNN_FORMAT_4DFEATURE;
  zdnn_status exp_status = ZDNN_INVALID_CLIPPING_VALUE;
  float clip_value = -1;
  zdnn_status status =
      zdnn_relu(&ztensor_input1, (void *)&clip_value, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

void nan_relu_clipping() {
  VERIFY_HW_ENV; // zdnn_relu drives HW conversion before precheck
  ztensor_output1.transformed_desc->format = ZDNN_FORMAT_4DFEATURE;
  zdnn_status exp_status = ZDNN_INVALID_CLIPPING_VALUE;
  uint32_t clip_value = 0x7FFFFFFF;
  zdnn_status status =
      zdnn_relu(&ztensor_input1, (void *)&clip_value, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

void negative_nan_relu_clipping() {
  VERIFY_HW_ENV; // zdnn_relu drives HW conversion before precheck
  ztensor_output1.transformed_desc->format = ZDNN_FORMAT_4DFEATURE;
  zdnn_status exp_status = ZDNN_INVALID_CLIPPING_VALUE;
  uint32_t clip_value = 0xFFFFFFFF;
  zdnn_status status =
      zdnn_relu(&ztensor_input1, (void *)&clip_value, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

// Make all tensor and other values correct.
void setup_conv2d_tensors() {
  ztensor_output1.transformed_desc->format = ZDNN_FORMAT_4DFEATURE;
  ztensor_input1.transformed_desc->dim4 = 1;
  ztensor_input1.transformed_desc->dim3 = 4;
  ztensor_input1.transformed_desc->dim2 = 3;
  ztensor_input1.transformed_desc->dim1 = 5;
  ztensor_input2.transformed_desc->dim4 = 2;
  ztensor_input2.transformed_desc->dim3 = 2;
  ztensor_input2.transformed_desc->dim2 = 5;
  ztensor_input2.transformed_desc->dim1 = 2;
  ztensor_input3.transformed_desc->dim4 = 1;
  ztensor_input3.transformed_desc->dim3 = 1;
  ztensor_input3.transformed_desc->dim2 = 1;
  ztensor_input3.transformed_desc->dim1 = 2;
  ztensor_output1.transformed_desc->dim4 = 1;
  ztensor_output1.transformed_desc->dim3 = 3;
  ztensor_output1.transformed_desc->dim2 = 2;
  ztensor_output1.transformed_desc->dim1 = 2;
}

void negative_conv2d_clipping() {
  VERIFY_HW_ENV; // zdnn_conv2d drives HW conversion before precheck
  setup_conv2d_tensors();
  zdnn_status exp_status = ZDNN_INVALID_CLIPPING_VALUE;
  float clip_value = -1;
  zdnn_status status = zdnn_conv2d(
      &ztensor_input1, &ztensor_input2, &ztensor_input3, VALID_PADDING, 1, 1,
      CONV2D_ACT_RELU, (void *)&clip_value, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

void nan_conv2d_clipping() {
  VERIFY_HW_ENV; // zdnn_conv2d drives HW conversion before precheck
  setup_conv2d_tensors();
  zdnn_status exp_status = ZDNN_INVALID_CLIPPING_VALUE;
  uint32_t clip_value = 0x7FFFFFFF;
  zdnn_status status = zdnn_conv2d(
      &ztensor_input1, &ztensor_input2, &ztensor_input3, VALID_PADDING, 1, 1,
      CONV2D_ACT_RELU, (void *)&clip_value, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

void negative_nan_conv2d_clipping() {
  VERIFY_HW_ENV; // zdnn_conv2d drives HW conversion before precheck
  setup_conv2d_tensors();
  zdnn_status exp_status = ZDNN_INVALID_CLIPPING_VALUE;
  uint32_t clip_value = 0xFFFFFFFF;
  zdnn_status status = zdnn_conv2d(
      &ztensor_input1, &ztensor_input2, &ztensor_input3, VALID_PADDING, 1, 1,
      CONV2D_ACT_RELU, (void *)&clip_value, &ztensor_output1);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status, "Got status %d \"%s\" but expected %d \"%s\"",
      status, zdnn_get_status_message(status), exp_status,
      (zdnn_get_status_message(exp_status)));
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(bad_element_wise);
  RUN_TEST(bad_batchnorm);
  RUN_TEST(bad_lstm);
  RUN_TEST(bad_matmul_op_with_bias_addition);
  RUN_TEST(bad_matmul_bcast_op_with_bias_addition);
  RUN_TEST(bad_pool);
  RUN_TEST(negative_relu_clipping);
  RUN_TEST(nan_relu_clipping);
  RUN_TEST(negative_nan_relu_clipping);
  RUN_TEST(negative_conv2d_clipping);
  RUN_TEST(nan_conv2d_clipping);
  RUN_TEST(negative_nan_conv2d_clipping);
  return UNITY_END();
}
