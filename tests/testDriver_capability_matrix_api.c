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
 * Tests for capability-matrix snapshot API.
 */

void setUp(void) {
  memset(&nnpa_query_result, 0, sizeof(nnpa_query_result));
  aiu_lib_vernum = 0;
}

void tearDown(void) {}

void test_get_capability_matrix_rejects_null_output(void) {
  TEST_ASSERT_EQUAL(ZDNN_INVALID_BUFFER, zdnn_get_capability_matrix(NULL));
}

void test_get_capability_matrix_returns_cached_query_snapshot(void) {
  nnpa_query_result.installed_functions_vector.bits_0to63 = 0x0123456789ABCDEF;
  nnpa_query_result.installed_functions_vector.bits_64to127 =
      0x89ABCDEF01234567;
  nnpa_query_result.installed_functions_vector.bits_128to191 =
      0xAAAAAAAA55555555;
  nnpa_query_result.installed_functions_vector.bits_192to255 =
      0x13579BDF2468ACE0;

  nnpa_query_result.installed_parameter_block_formats.bits_0to63 =
      0x00FF00FF00FF00FF;
  nnpa_query_result.installed_parameter_block_formats.bits_64to127 =
      0x0F0F0F0F0F0F0F0F;

  nnpa_query_result.installed_data_types = 0xA5A5;
  nnpa_query_result.installed_dt1_conversions_vector = 0x5A5A;
  nnpa_query_result.installed_data_layout_formats = 0xCC33CC33;
  nnpa_query_result.maximum_dimension_index_size = 32768;
  nnpa_query_result.max_dim4_index_size = 16384;
  nnpa_query_result.max_dim3_index_size = 8192;
  nnpa_query_result.max_dim2_index_size = 4096;
  nnpa_query_result.max_dim1_index_size = 2048;
  nnpa_query_result.maximum_tensor_size = 0x0000000180000000ULL;

  aiu_lib_vernum = 0x01020A;

  zdnn_capability_matrix caps;
  memset(&caps, 0, sizeof(caps));

  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_get_capability_matrix(&caps));

  TEST_ASSERT_EQUAL_HEX64(0x0123456789ABCDEFULL, caps.installed_functions[0]);
  TEST_ASSERT_EQUAL_HEX64(0x89ABCDEF01234567ULL, caps.installed_functions[1]);
  TEST_ASSERT_EQUAL_HEX64(0xAAAAAAAA55555555ULL, caps.installed_functions[2]);
  TEST_ASSERT_EQUAL_HEX64(0x13579BDF2468ACE0ULL, caps.installed_functions[3]);

  TEST_ASSERT_EQUAL_HEX64(0x00FF00FF00FF00FFULL, caps.installed_parmblk_fmts[0]);
  TEST_ASSERT_EQUAL_HEX64(0x0F0F0F0F0F0F0F0FULL, caps.installed_parmblk_fmts[1]);

  TEST_ASSERT_EQUAL_HEX16(0xA5A5, caps.installed_data_types);
  TEST_ASSERT_EQUAL_HEX16(0x5A5A, caps.installed_dt1_conversions);
  TEST_ASSERT_EQUAL_HEX32(0xCC33CC33, caps.installed_data_layout_formats);

  TEST_ASSERT_EQUAL_UINT32(32768, caps.maximum_dimension_index_size);
  TEST_ASSERT_EQUAL_UINT32(16384, caps.max_dim4_index_size);
  TEST_ASSERT_EQUAL_UINT32(8192, caps.max_dim3_index_size);
  TEST_ASSERT_EQUAL_UINT32(4096, caps.max_dim2_index_size);
  TEST_ASSERT_EQUAL_UINT32(2048, caps.max_dim1_index_size);
  TEST_ASSERT_EQUAL_HEX64(0x0000000180000000ULL, caps.maximum_tensor_size);
  TEST_ASSERT_EQUAL_HEX32(0x01020A, caps.aiu_library_version);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_get_capability_matrix_rejects_null_output);
  RUN_TEST(test_get_capability_matrix_returns_cached_query_snapshot);

  return UNITY_END();
}
