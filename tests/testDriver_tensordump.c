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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "testsupport.h"

bool suppress_output;

void setUp(void) {
  VERIFY_HW_ENV;
  if (log_level == LOGLEVEL_DEBUG) {
    suppress_output = false;
  } else {
    suppress_output = true;
  }
}

void tearDown(void) {}

// Create pre and post descriptions
void init_tensor_descriptors(uint32_t dim4, uint32_t dim3, uint32_t dim2,
                             uint32_t dim1, zdnn_data_layouts layout,
                             zdnn_data_types data_type,
                             zdnn_tensor_desc *pre_tfrmd_desc,
                             zdnn_tensor_desc *tfrmd_desc) {
  switch (layout) {
  case ZDNN_1D:
    zdnn_init_pre_transformed_desc(layout, data_type, pre_tfrmd_desc, dim1);
    break;
  case ZDNN_2D:
  case ZDNN_2DS:
    zdnn_init_pre_transformed_desc(layout, data_type, pre_tfrmd_desc, dim2,
                                   dim1);
    break;
  case ZDNN_3D:
  case ZDNN_3DS:
    zdnn_init_pre_transformed_desc(layout, data_type, pre_tfrmd_desc, dim3,
                                   dim2, dim1);
    break;
  default:
    zdnn_init_pre_transformed_desc(layout, data_type, pre_tfrmd_desc, dim4,
                                   dim3, dim2, dim1);
  }
  zdnn_status status =
      zdnn_generate_transformed_desc(pre_tfrmd_desc, tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc failed (status = %08x)", status);
}

void test_origtensor_dump(uint32_t dim4, uint32_t dim3, uint32_t dim2,
                          uint32_t dim1, zdnn_data_layouts layout,
                          zdnn_data_types data_type, dump_mode mode) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  init_tensor_descriptors(dim4, dim3, dim2, dim1, layout, data_type,
                          &pre_tfrmd_desc, &tfrmd_desc);

  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc failed (status = %08x)",
      status);

  void *data = create_and_fill_random_fp_data(&ztensor);

  printf("\n--- Pre-Transformed Tensor Dump (%s) ---\n",
         get_data_type_str(data_type));
  dumpdata_origtensor(&pre_tfrmd_desc, data, mode);

  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_tensor_data_dump(uint32_t dim4, uint32_t dim3, uint32_t dim2,
                           uint32_t dim1, zdnn_data_layouts layout,
                           zdnn_data_types data_type, dump_mode mode) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  init_tensor_descriptors(dim4, dim3, dim2, dim1, layout, data_type,
                          &pre_tfrmd_desc, &tfrmd_desc);

  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc failed (status = %08x)",
      status);

  void *data = create_and_fill_random_fp_data(&ztensor);

  // Transform the tensor
  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_ztensor() failed, status = %08x (%s)",
      status, zdnn_get_status_message(status));

  // Print transformed tensor dump
  printf("\n--- Transformed (Stickified) Tensor Dump (%s) ---\n",
         get_data_type_str(data_type));
  dumpdata_ztensor(&ztensor, mode, false);

  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_tensor_dump_int8(uint32_t dim4, uint32_t dim3, uint32_t dim2,
                           uint32_t dim1, dump_mode mode) {

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  int8_t *data;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, INT8, &pre_tfrmd_desc, dim4, dim3,
                                 dim2, dim1);

  status = zdnn_generate_quantized_transformed_desc(
      &pre_tfrmd_desc, QUANTIZED_INT8, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_quantized_transformed_desc() failed (status = %08x)",
      status);

  status = zdnn_init_quantized_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc,
                                                   0, 0, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_quantized_ztensor_with_malloc() failed (status = %08x)",
      status);

  data = create_and_fill_random_int8_data(&ztensor);

  status =
      zdnn_transform_quantized_ztensor(&ztensor, false, 0, 0, (void *)data);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_transform_quantized_ztensor() failed, status = %08x "
      "(%s)",
      status, zdnn_get_status_message(status));

  printf("\n--- Transformed (Stickified) Tensor Dump (%s) ---\n",
         get_data_type_str(INT8));
  dumpdata_ztensor(&ztensor, mode, false);

  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_tensor_dump_no_page_break() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  init_tensor_descriptors(1, 1, 1, 100, ZDNN_NHWC, FP32, &pre_tfrmd_desc,
                          &tfrmd_desc);
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc failed (status = %08x)",
      status);

  void *data = create_and_fill_random_fp_data(&ztensor);
  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_ztensor() failed, status = %08x (%s)",
      status, zdnn_get_status_message(status));

  dumpdata_ztensor(&ztensor, AS_HEX, false);

  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_tensor_dump_with_page_break() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  init_tensor_descriptors(1, 1, 1, 2150, ZDNN_NHWC, FP32, &pre_tfrmd_desc,
                          &tfrmd_desc);
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc failed (status = %08x)",
      status);

  void *data = create_and_fill_random_fp_data(&ztensor);
  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_ztensor() failed, status = %08x (%s)",
      status, zdnn_get_status_message(status));
  dumpdata_ztensor(&ztensor, AS_HEX, false);

  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

// Only if log_level is set to debug, print output
void run_test_suppress_output(void (*test_func)(void)) {
  // save origin stdout
  FILE *original_stdout = stdout;
  // create null area for std about to printed
  FILE *null_out = fopen("/dev/null", "w");
  if (!suppress_output) {
    test_func();
    fclose(null_out);
    return;
  }
  // if log_level not set, print output to null area
  stdout = null_out;
  test_func();
  // restore original stdout
  stdout = original_stdout;
  fclose(null_out);
}

void test_simple_1D_hex_bfloat_dump() {
  test_origtensor_dump(1, 1, 1, 1, ZDNN_1D, BFLOAT, AS_HEX);
  test_tensor_data_dump(1, 1, 1, 1, ZDNN_1D, BFLOAT, AS_HEX);
}
void test_simple_1D_float_bfloat_dump() {
  test_origtensor_dump(1, 1, 1, 1, ZDNN_1D, BFLOAT, AS_FLOAT);
  test_tensor_data_dump(1, 1, 1, 1, ZDNN_1D, BFLOAT, AS_FLOAT);
}
void test_simple_hex_fp16_dump() {
  test_origtensor_dump(1, 1, 1, 1, ZDNN_NHWC, FP16, AS_HEX);
  test_tensor_data_dump(1, 1, 1, 1, ZDNN_NHWC, FP16, AS_HEX);
}
void test_simple_float_fp16_dump() {
  test_origtensor_dump(1, 1, 1, 1, ZDNN_NHWC, FP16, AS_FLOAT);
  test_tensor_data_dump(1, 1, 1, 1, ZDNN_NHWC, FP16, AS_FLOAT);
}
void test_simple_hex_fp32_dump() {
  test_origtensor_dump(1, 1, 1, 1, ZDNN_NHWC, FP32, AS_HEX);
  test_tensor_data_dump(1, 1, 1, 1, ZDNN_NHWC, FP32, AS_HEX);
}
void test_simple_float_fp32_dump() {
  test_origtensor_dump(1, 1, 1, 1, ZDNN_NHWC, FP32, AS_FLOAT);
  test_tensor_data_dump(1, 1, 1, 1, ZDNN_NHWC, FP32, AS_FLOAT);
}
void test_simple_hex_int8_dump() { test_tensor_dump_int8(1, 1, 1, 1, AS_HEX); }
void test_simple_float_int8_dump() {
  test_tensor_dump_int8(1, 1, 1, 1, AS_FLOAT);
}

// Wrapper functions for Unity to run
void test_simple_1D_hex_bfloat_dump_with_suppression() {
  run_test_suppress_output(test_simple_1D_hex_bfloat_dump);
}
void test_simple_1D_float_bfloat_dump_with_suppression() {
  run_test_suppress_output(test_simple_1D_float_bfloat_dump);
}
void test_simple_hex_fp16_dump_with_suppression() {
  run_test_suppress_output(test_simple_hex_fp16_dump);
}
void test_simple_float_fp16_dump_with_suppression() {
  run_test_suppress_output(test_simple_float_fp16_dump);
}
void test_simple_hex_fp32_dump_with_suppression() {
  run_test_suppress_output(test_simple_hex_fp32_dump);
}
void test_simple_float_fp32_dump_with_suppression() {
  run_test_suppress_output(test_simple_float_fp32_dump);
}
void test_simple_hex_int8_dump_with_suppression() {
  run_test_suppress_output(test_simple_hex_int8_dump);
}
void test_simple_float_int8_dump_with_suppression() {
  run_test_suppress_output(test_simple_float_int8_dump);
}
void test_tensor_dump_no_page_break_with_suppression() {
  run_test_suppress_output(test_tensor_dump_no_page_break);
}
void test_tensor_dump_with_page_break_with_suppression() {
  run_test_suppress_output(test_tensor_dump_with_page_break);
}

int main(void) {
  UNITY_BEGIN();

  // If log_level is set to debug, output will printed.
  // Otherwise, nothing will be printed but tests will still run

  // BFLOAT
  RUN_TEST(test_simple_1D_hex_bfloat_dump_with_suppression);
  RUN_TEST(test_simple_1D_float_bfloat_dump_with_suppression);
  // FP16
  RUN_TEST(test_simple_hex_fp16_dump_with_suppression);
  RUN_TEST(test_simple_float_fp16_dump_with_suppression);
  // FP32
  RUN_TEST(test_simple_hex_fp32_dump_with_suppression);
  RUN_TEST(test_simple_float_fp32_dump_with_suppression);
  // INT8 Quantized
  RUN_TEST(test_simple_hex_int8_dump_with_suppression);
  RUN_TEST(test_simple_float_int8_dump_with_suppression);
  // Page Break
  RUN_TEST(test_tensor_dump_no_page_break_with_suppression);
  RUN_TEST(test_tensor_dump_with_page_break_with_suppression);

  return UNITY_END();
}
