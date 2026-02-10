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

#include <string.h>

/*
 * Tests for typed wrappers over the varargs zdnn_transform_ztensor() API.
 *
 * Goals:
 * - The wrappers must reject incorrect transformed layouts (avoid silent
 *   varargs misuse).
 * - For valid concatenated tensors, wrapper output must match the equivalent
 *   varargs call byte-for-byte.
 */

void setUp(void) {}
void tearDown(void) {}

static void fill_fp16(uint16_t *buf, size_t n, uint16_t v) {
  for (size_t i = 0; i < n; i++) {
    buf[i] = v;
  }
}

void test_fico_wrapper_rejects_non_fico_layout(void) {
  zdnn_tensor_desc pre = {0}, tfrmd = {0};
  zdnn_ztensor z = {0};

  zdnn_init_pre_transformed_desc(ZDNN_1D, FP16, &pre, 4);
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_generate_transformed_desc(&pre, &tfrmd));

  zdnn_init_ztensor(&pre, &tfrmd, &z);

  // Should fail fast before dereferencing any gate pointers.
  TEST_ASSERT_EQUAL_UINT32(ZDNN_INVALID_LAYOUT,
                           zdnn_transform_ztensor_fico(&z, NULL, NULL, NULL, NULL));
}

void test_zrh_wrapper_rejects_non_zrh_layout(void) {
  zdnn_tensor_desc pre = {0}, tfrmd = {0};
  zdnn_ztensor z = {0};

  zdnn_init_pre_transformed_desc(ZDNN_1D, FP16, &pre, 4);
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_generate_transformed_desc(&pre, &tfrmd));

  zdnn_init_ztensor(&pre, &tfrmd, &z);

  TEST_ASSERT_EQUAL_UINT32(ZDNN_INVALID_LAYOUT,
                           zdnn_transform_ztensor_zrh(&z, NULL, NULL, NULL));
}

void test_fico_wrapper_matches_varargs(void) {
  const uint32_t num_dirs = 1;
  const uint32_t num_hidden = 7;
  const size_t gate_el = (size_t)num_dirs * num_hidden;

  zdnn_tensor_desc pre = {0}, tfrmd = {0};

  // Concatenated LSTM biases.
  zdnn_concat_info info = RNN_TYPE_LSTM | USAGE_BIASES | PREV_LAYER_NONE;

  // For biases: pre-transformed shape per gate is (num_dirs, num_hidden) using
  // ZDNN_2DS.
  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP16, &pre, num_dirs, num_hidden);
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_generate_transformed_desc_concatenated(&pre, info,
                                                                        &tfrmd));

  zdnn_ztensor z_varargs = {0}, z_wrapper = {0};

  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_init_ztensor_with_malloc(&pre, &tfrmd, &z_varargs));
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_init_ztensor_with_malloc(&pre, &tfrmd, &z_wrapper));

  // Distinct FP16 values per gate to catch ordering mistakes.
  // (IEEE-754 half): 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400.
  uint16_t f[gate_el], i[gate_el], c[gate_el], o[gate_el];
  fill_fp16(f, gate_el, 0x3C00);
  fill_fp16(i, gate_el, 0x4000);
  fill_fp16(c, gate_el, 0x4200);
  fill_fp16(o, gate_el, 0x4400);

  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_transform_ztensor(&z_varargs, f, i, c, o));
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_transform_ztensor_fico(&z_wrapper, f, i, c, o));

  TEST_ASSERT_EQUAL_UINT64_MESSAGE(z_varargs.buffer_size, z_wrapper.buffer_size,
                                  "Buffer sizes unexpectedly differ");

  TEST_ASSERT_EQUAL_INT_MESSAGE(0,
                                memcmp(z_varargs.buffer, z_wrapper.buffer,
                                       z_varargs.buffer_size),
                                "Wrapper output differs from varargs output");

  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_free_ztensor_buffer(&z_varargs));
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_free_ztensor_buffer(&z_wrapper));
}

void test_zrh_wrapper_matches_varargs(void) {
  const uint32_t num_dirs = 1;
  const uint32_t num_hidden = 7;
  const size_t gate_el = (size_t)num_dirs * num_hidden;

  zdnn_tensor_desc pre = {0}, tfrmd = {0};

  // Concatenated GRU biases.
  zdnn_concat_info info = RNN_TYPE_GRU | USAGE_BIASES | PREV_LAYER_NONE;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP16, &pre, num_dirs, num_hidden);
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_generate_transformed_desc_concatenated(&pre, info,
                                                                        &tfrmd));

  zdnn_ztensor z_varargs = {0}, z_wrapper = {0};

  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_init_ztensor_with_malloc(&pre, &tfrmd, &z_varargs));
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK,
                           zdnn_init_ztensor_with_malloc(&pre, &tfrmd, &z_wrapper));

  uint16_t z[gate_el], r[gate_el], h[gate_el];
  fill_fp16(z, gate_el, 0x3C00);
  fill_fp16(r, gate_el, 0x4000);
  fill_fp16(h, gate_el, 0x4200);

  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_transform_ztensor(&z_varargs, z, r, h));
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_transform_ztensor_zrh(&z_wrapper, z, r, h));

  TEST_ASSERT_EQUAL_UINT64_MESSAGE(z_varargs.buffer_size, z_wrapper.buffer_size,
                                  "Buffer sizes unexpectedly differ");

  TEST_ASSERT_EQUAL_INT_MESSAGE(0,
                                memcmp(z_varargs.buffer, z_wrapper.buffer,
                                       z_varargs.buffer_size),
                                "Wrapper output differs from varargs output");

  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_free_ztensor_buffer(&z_varargs));
  TEST_ASSERT_EQUAL_UINT32(ZDNN_OK, zdnn_free_ztensor_buffer(&z_wrapper));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_fico_wrapper_rejects_non_fico_layout);
  RUN_TEST(test_zrh_wrapper_rejects_non_zrh_layout);
  RUN_TEST(test_fico_wrapper_matches_varargs);
  RUN_TEST(test_zrh_wrapper_matches_varargs);

  return UNITY_END();
}
