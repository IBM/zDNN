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

#include "testsupport.h"

#include <stdio.h>

#define NNPA_OP_FAKE 255
#define NNPA_PARMBLKFORMAT_FAKE 127
#define QUERY_DATATYPE_FAKE (1 << 0)
#define QUERY_LAYOUTFMT_FAKE (10 << 0)
#define QUERY_BFPFMT_FAKE (1 << 0)

void setUp(void) { VERIFY_HW_ENV; }

void tearDown(void) {}

void test_function_available() {

  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_function_installed(3, NNPA_ADD, NNPA_BATCHNORMALIZATION,
                                      NNPA_SOFTMAX) == true,
      "One or more of the requested functions is not detected as available");
}

void test_function_not_available() {

  TEST_ASSERT_MESSAGE(zdnn_is_nnpa_function_installed(3, NNPA_ADD,
                                                      NNPA_BATCHNORMALIZATION,
                                                      NNPA_OP_FAKE) == false,
                      "NNPA_OP_FAKE is not detected as unavailable");
}

void test_parm_blk_fmt_installed() {
  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_0) == true,
      "NNPA_PARMBLKFORMAT_TENSORDESC is not detected as available");
}

void test_parm_blk_fmt_not_installed() {
  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_parmblk_fmt_installed(2, NNPA_PARMBLKFORMAT_FAKE,
                                         NNPA_PARMBLKFORMAT_0) == false,
      "NNPA_PARMBLKFORMAT_FAKE is not detected as unavailable");
}

void test_datatype_installed() {
  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_datatype_installed(QUERY_DATATYPE_INTERNAL1) == true,
      "NNPA_QAF_DATATYPE_INTERNAL1 is not detected as available");
}

void test_datatype_not_installed() {
  TEST_ASSERT_MESSAGE(zdnn_is_nnpa_datatype_installed(QUERY_DATATYPE_INTERNAL1 |
                                                      QUERY_DATATYPE_FAKE) ==
                          false,
                      "QUERY_DATATYPE_FAKE is not detected as unavailable");
}

void test_datalayout_installed() {
  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_layout_fmt_installed(QUERY_LAYOUTFMT_4DFEATURE |
                                        QUERY_LAYOUTFMT_4DKERNEL) == true,
      "NNPA_QAF_DATALAYOUT_4DFEATURETENSOR is not detected as available");
}

void test_datalayout_not_installed() {
  TEST_ASSERT_MESSAGE(zdnn_is_nnpa_layout_fmt_installed(
                          QUERY_LAYOUTFMT_4DFEATURE | QUERY_LAYOUTFMT_4DKERNEL |
                          QUERY_LAYOUTFMT_FAKE) == false,
                      "QUERY_LAYOUTFMT_FAKE is not detected as unavailable");
}

void test_datatype_conversion_installed() {
  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_conversion_installed(
          NNPA_DATATYPE_1, QUERY_BFPFMT_TINY | QUERY_BFPFMT_SHORT) == true,
      "QUERY_BFPFMT_TINY | QUERY_BFPFMT_SHORT is not detected as available");
}

void test_datatype_conversion_not_installed() {
  TEST_ASSERT_MESSAGE(
      zdnn_is_nnpa_conversion_installed(NNPA_DATATYPE_1,
                                        QUERY_BFPFMT_TINY | QUERY_BFPFMT_SHORT |
                                            QUERY_BFPFMT_FAKE) == false,
      "QUERY_BFPFMT_FAKE is not detected as unavailable");
}

// Values from AR11010-12
#define MAXIMUM_DIMENSION_INDEX_SIZE ((uint32_t)1 << 15) // 32768
#define MAX_DIM4_INDEX_SIZE ((uint32_t)1 << 15)          // 32768
#define MAX_DIM3_INDEX_SIZE ((uint32_t)1 << 15)          // 32768
#define MAX_DIM2_INDEX_SIZE ((uint32_t)1 << 20)          // 1048576
#define MAX_DIM1_INDEX_SIZE ((uint32_t)1 << 21)          // 2097152
#define MAXIMUM_TENSOR_SIZE ((uint64_t)1 << 32)          // 4294967296

void test_get_max_dim_idx_size() {
  TEST_ASSERT_MESSAGE_FORMATTED(
      zdnn_get_nnpa_max_dim_idx_size() == MAXIMUM_DIMENSION_INDEX_SIZE,
      "zdnn_get_nnpa_max_dim_idx_size() %u did not return %u",
      zdnn_get_nnpa_max_dim_idx_size(), MAXIMUM_DIMENSION_INDEX_SIZE);
}

void test_get_max_dim4_idx_size() {
  uint32_t expected_index_size = MAX_DIM4_INDEX_SIZE;
  TEST_ASSERT_MESSAGE_FORMATTED(zdnn_get_max_for_dim(4) == expected_index_size,
                                "zdnn_get_max_for_dim() %u did not return %u",
                                zdnn_get_max_for_dim(4), expected_index_size);
}

void test_get_max_dim3_idx_size() {
  uint32_t expected_index_size = MAX_DIM3_INDEX_SIZE;
  TEST_ASSERT_MESSAGE_FORMATTED(zdnn_get_max_for_dim(3) == expected_index_size,
                                "zdnn_get_max_for_dim(3) %u did not return %u",
                                zdnn_get_max_for_dim(3), expected_index_size);
}

void test_get_max_dim2_idx_size() {
  uint32_t expected_index_size = nnpa_query_result.max_dim2_index_size
                                     ? MAX_DIM2_INDEX_SIZE
                                     : MAXIMUM_DIMENSION_INDEX_SIZE;
  TEST_ASSERT_MESSAGE_FORMATTED(zdnn_get_max_for_dim(2) == expected_index_size,
                                "zdnn_get_max_for_dim(2) %u did not return %u",
                                zdnn_get_max_for_dim(2), expected_index_size);
}

void test_get_max_dim1_idx_size() {
  uint32_t expected_index_size = nnpa_query_result.max_dim1_index_size
                                     ? MAX_DIM1_INDEX_SIZE
                                     : MAXIMUM_DIMENSION_INDEX_SIZE;
  TEST_ASSERT_MESSAGE_FORMATTED(zdnn_get_max_for_dim(1) == expected_index_size,
                                "zdnn_get_max_for_dim(1) %u did not return %u",
                                zdnn_get_max_for_dim(1), expected_index_size);
}

void test_get_max_tensor_size() {
  TEST_ASSERT_MESSAGE_FORMATTED(
      zdnn_get_nnpa_max_tensor_size() == MAXIMUM_TENSOR_SIZE,
      "zdnn_get_nnpa_max_tensor_size() %" PRIu64 " did not return %" PRIu64,
      zdnn_get_nnpa_max_tensor_size(), MAXIMUM_TENSOR_SIZE);
}

// eyeball inspection
void test_print_version() {
  printf("version = %04x\n", zdnn_get_library_version());
  printf("version string = %s\n", zdnn_get_library_version_str());
}

// ------------------------------------------------------------------------------------------------

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_function_available);
  RUN_TEST(test_function_not_available);

  RUN_TEST(test_parm_blk_fmt_installed);
  RUN_TEST(test_parm_blk_fmt_not_installed);

  RUN_TEST(test_datatype_installed);
  RUN_TEST(test_datatype_not_installed);

  RUN_TEST(test_datalayout_installed);
  RUN_TEST(test_datalayout_not_installed);

  RUN_TEST(test_datatype_conversion_installed);
  RUN_TEST(test_datatype_conversion_not_installed);

  RUN_TEST(test_get_max_dim_idx_size);
  RUN_TEST(test_get_max_dim4_idx_size);
  RUN_TEST(test_get_max_dim3_idx_size);
  RUN_TEST(test_get_max_dim2_idx_size);
  RUN_TEST(test_get_max_dim1_idx_size);
  RUN_TEST(test_get_max_tensor_size);

  RUN_TEST(test_print_version);

  return UNITY_END();
}
