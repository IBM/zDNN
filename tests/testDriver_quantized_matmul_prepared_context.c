// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024
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

#include "testsupport.h"

#include <stdint.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

static void init_quantized_matmul_tensors(zdnn_ztensor *input_a,
                                          zdnn_ztensor *input_b,
                                          zdnn_ztensor *input_c,
                                          zdnn_ztensor *output,
                                          zdnn_tensor_desc *a_desc,
                                          zdnn_tensor_desc *b_desc,
                                          zdnn_tensor_desc *c_desc,
                                          zdnn_tensor_desc *out_desc) {
  // Matmul shape family:
  // input_a  [S,1,M,N] = [2,1,4,3]
  // input_b  [S,1,N,P] = [2,1,3,5]
  // input_c  [S,1,1,P] = [2,1,1,5]
  // output   [S,1,M,P] = [2,1,4,5]
  init_transformed_desc(ZDNN_NHWC, ZDNN_BINARY_INT8, ZDNN_FORMAT_4DFEATURE,
                        a_desc, 2, 1, 4, 3);
  init_transformed_desc(ZDNN_NHWC, ZDNN_BINARY_INT8, ZDNN_FORMAT_4DWEIGHTS,
                        b_desc, 2, 1, 3, 5);
  init_transformed_desc(ZDNN_NHWC, ZDNN_BINARY_INT8, ZDNN_FORMAT_4DFEATURE,
                        c_desc, 2, 1, 1, 5);
  init_transformed_desc(ZDNN_NHWC, ZDNN_BINARY_INT8, ZDNN_FORMAT_4DFEATURE,
                        out_desc, 2, 1, 4, 5);

  memset(input_a, 0, sizeof(*input_a));
  memset(input_b, 0, sizeof(*input_b));
  memset(input_c, 0, sizeof(*input_c));
  memset(output, 0, sizeof(*output));

  input_a->transformed_desc = a_desc;
  input_b->transformed_desc = b_desc;
  input_c->transformed_desc = c_desc;
  output->transformed_desc = out_desc;
}

void test_prepare_context_rejects_null_or_missing_descriptors(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_INVALID_BUFFER,
      zdnn_prepare_quantized_matmul_context(&input_a, &input_b, &input_c, false,
                                            NULL));

  input_b.transformed_desc = NULL;
  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_INVALID_BUFFER,
      zdnn_prepare_quantized_matmul_context(&input_a, &input_b, &input_c, false,
                                            &context));
}

void test_prepare_context_allocates_required_work_area(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  // qc_tilde uses input_c's transformed dims/layout/format but promotes the
  // element type to DLFLOAT16.
  zdnn_tensor_desc qc_tilde_desc = {0};
  init_transformed_desc(c_desc.layout, ZDNN_DLFLOAT16, c_desc.format,
                        &qc_tilde_desc, c_desc.dim4, c_desc.dim3, c_desc.dim2,
                        c_desc.dim1);
  uint64_t expected_work_area = zdnn_getsize_ztensor(&qc_tilde_desc);
  TEST_ASSERT_TRUE(expected_work_area > 0);

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_OK, zdnn_prepare_quantized_matmul_context(
                   &input_a, &input_b, &input_c, false, &context));

  TEST_ASSERT_EQUAL_UINT32(NNPA_MATMUL_OP, context.function_code);
  TEST_ASSERT_EQUAL_UINT64(expected_work_area, context.required_work_area_size);
  TEST_ASSERT_NOT_NULL(context.work_area_buffer);
  TEST_ASSERT_EQUAL_UINT64(expected_work_area, context.work_area_buffer_size);
  TEST_ASSERT_EQUAL_UINT64(0, (uintptr_t)context.work_area_buffer %
                                  AIU_PAGESIZE_IN_BYTES);

  zdnn_release_quantized_matmul_context(&context);
  TEST_ASSERT_NULL(context.work_area_buffer);
  TEST_ASSERT_EQUAL_UINT64(0, context.work_area_buffer_size);
  TEST_ASSERT_EQUAL_UINT64(0, context.required_work_area_size);
}

void test_prepare_context_precomputed_path_has_no_work_area(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_OK, zdnn_prepare_quantized_matmul_context(
                   &input_a, &input_b, &input_c, true, &context));

  TEST_ASSERT_TRUE(context.pre_computed);
  TEST_ASSERT_EQUAL_UINT64(0, context.required_work_area_size);
  TEST_ASSERT_NULL(context.work_area_buffer);
  TEST_ASSERT_EQUAL_UINT64(0, context.work_area_buffer_size);

  zdnn_release_quantized_matmul_context(&context);
}

void test_prepare_context_rejects_non_int8_bias_when_not_precomputed(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  c_desc.type = ZDNN_DLFLOAT16;

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_INVALID_TYPE,
      zdnn_prepare_quantized_matmul_context(&input_a, &input_b, &input_c, false,
                                            &context));
}

void test_prepared_op_rejects_unprepared_context(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_INVALID_BUFFER,
      zdnn_quantized_matmul_op_prepared(&context, &input_a, &input_b, &input_c,
                                        MATMUL_OP_ADDITION, 0, 0, false, false,
                                        &output));
}

void test_prepared_op_rejects_shape_drift(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_OK, zdnn_prepare_quantized_matmul_context(
                   &input_a, &input_b, &input_c, false, &context));

  zdnn_tensor_desc a_desc_mismatch = a_desc;
  a_desc_mismatch.dim2 += 1;
  zdnn_ztensor input_a_mismatch = input_a;
  input_a_mismatch.transformed_desc = &a_desc_mismatch;

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_INVALID_SHAPE,
      zdnn_quantized_matmul_op_prepared(
          &context, &input_a_mismatch, &input_b, &input_c, MATMUL_OP_ADDITION,
          0, 0, false, false, &output));

  zdnn_release_quantized_matmul_context(&context);
}

void test_prepared_op_rejects_precomputed_offset_violation(void) {
  zdnn_quantized_matmul_prepared_context context = {0};
  zdnn_ztensor input_a = {0}, input_b = {0}, input_c = {0}, output = {0};
  zdnn_tensor_desc a_desc = {0}, b_desc = {0}, c_desc = {0}, out_desc = {0};

  init_quantized_matmul_tensors(&input_a, &input_b, &input_c, &output, &a_desc,
                                &b_desc, &c_desc, &out_desc);

  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_OK, zdnn_prepare_quantized_matmul_context(
                   &input_a, &input_b, &input_c, true, &context));

  input_b.offset = 1.f;
  TEST_ASSERT_EQUAL_UINT32(
      ZDNN_INVALID_OFFSET,
      zdnn_quantized_matmul_op_prepared(&context, &input_a, &input_b, &input_c,
                                        MATMUL_OP_ADDITION, 0, 0, false, false,
                                        &output));

  zdnn_release_quantized_matmul_context(&context);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_prepare_context_rejects_null_or_missing_descriptors);
  RUN_TEST(test_prepare_context_allocates_required_work_area);
  RUN_TEST(test_prepare_context_precomputed_path_has_no_work_area);
  RUN_TEST(test_prepare_context_rejects_non_int8_bias_when_not_precomputed);
  RUN_TEST(test_prepared_op_rejects_unprepared_context);
  RUN_TEST(test_prepared_op_rejects_shape_drift);
  RUN_TEST(test_prepared_op_rejects_precomputed_offset_violation);

  return UNITY_END();
}
