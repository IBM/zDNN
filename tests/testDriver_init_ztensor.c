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

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "testsupport.h"

void setUp(void) { /* This is run before EACH TEST */

  VERIFY_HW_ENV;
}

void tearDown(void) {}

// Helper method for tests that check the boundaries of the maximum dim1 index.
// Concatenated ztensors introduce padding that must be determined to test this.
// See zdnn_generate_transformed_desc_concatenated() to see padding equation.
uint32_t max_concat_dim1(uint32_t num_concats) {
  uint32_t temp = zdnn_get_nnpa_max_dim_idx_size() / num_concats;
  uint32_t max_concat_dim1 = temp - (temp % AIU_2BYTE_CELLS_PER_STICK);
  LOG_TRACE("returning %d\n", max_concat_dim1);
  return max_concat_dim1;
}

// test if we can zdnn_init_ztensor_with_malloc() correctly with the supplied
// pre-transformed and transformed descriptors
void test_main(zdnn_tensor_desc *pre_tfrmd_desc, zdnn_tensor_desc *tfrmd_desc,
               uint64_t exp_size, zdnn_status exp_status_allochelper) {
  zdnn_ztensor ztensor;
  zdnn_status status;

  status = zdnn_init_ztensor_with_malloc(pre_tfrmd_desc, tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status_allochelper,
      "zdnn_init_ztensor_with_malloc() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), exp_status_allochelper,
      zdnn_get_status_message(exp_status_allochelper));

  // check and free buffer but only if expected
  // zdnn_init_ztensor_with_malloc() to work
  if (exp_status_allochelper == ZDNN_OK) {
    TEST_ASSERT_MESSAGE_FORMATTED(
        ztensor.buffer_size == exp_size,
        "zdnn_init_ztensor_with_malloc() returns incorrect size: %" PRIu64
        " (expects %" PRIu64 ")",
        ztensor.buffer_size, exp_size);

    zdnn_free_ztensor_buffer(&ztensor);
  }
}

void test_normal(zdnn_tensor_desc *pre_tfrmd_desc, uint64_t exp_size) {

  zdnn_tensor_desc tfrmd_desc;
  zdnn_status status;

  status = zdnn_generate_transformed_desc(pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), ZDNN_OK,
      zdnn_get_status_message(ZDNN_OK));

  test_main(pre_tfrmd_desc, &tfrmd_desc, exp_size, ZDNN_OK);
}

/// Drive the creation of a FICO/ZRH ztensor with the provided pre-transformed
/// layout, data type and dims, and transformed layout (FICO/ZRH).  Then drive
/// allocation and compare to an expected value.
///
/// \param[in] pre_tfrmd_layout          pre-transformed layout
/// \param[in] concat_type               concatenation type
/// \param[in] exp_size                  expected allocation size
/// \param[in] exp_status_gen_concat     expected status of _desc_concatenated()
/// \param[in] exp_status_allochelper    expected status of _allochelper()
/// \param[in] ...                       dimensions, outermost -> innermost
///                                      order  (ie shape order)
///
/// \return None - Fails test assertion if actual values don't match specified
///          exp values
///
void test_concat(zdnn_data_layouts pre_tfrmd_layout,
                 zdnn_ztensor_concat_types concat_type, uint64_t exp_size,
                 zdnn_status exp_status_gen_concat,
                 zdnn_status exp_status_allochelper, ...) {

  zdnn_status status;
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;

  va_list v_list;
  va_start(v_list, exp_status_allochelper);
  switch (pre_tfrmd_layout) {
  case ZDNN_2DS:
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, va_arg(v_list, uint32_t),
                                   va_arg(v_list, uint32_t));
    break;
  case ZDNN_3DS:
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, va_arg(v_list, uint32_t),
                                   va_arg(v_list, uint32_t),
                                   va_arg(v_list, uint32_t));
    break;
  // for driving an "invalid layout" testcase
  default:
    zdnn_init_pre_transformed_desc(
        pre_tfrmd_layout, test_datatype, &pre_tfrmd_desc,
        va_arg(v_list, uint32_t), va_arg(v_list, uint32_t),
        va_arg(v_list, uint32_t), va_arg(v_list, uint32_t));
    break;
  }
  va_end(v_list);

  status = zdnn_generate_transformed_desc_concatenated(
      &pre_tfrmd_desc, concat_type, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status_gen_concat,
      "zdnn_generate_transformed_desc_concatenated() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), exp_status_gen_concat,
      zdnn_get_status_message(exp_status_gen_concat));

  // do the rest if expected zdnn_generate_transformed_desc_concatenated() to
  // work
  if (exp_status_gen_concat == ZDNN_OK) {
    test_main(&pre_tfrmd_desc, &tfrmd_desc, exp_size, exp_status_allochelper);
  }
}

void test_NHWC(unsigned int n, unsigned int h, unsigned int w, unsigned int c,
               uint64_t exp_size) {
  zdnn_tensor_desc pre_tfrmd_desc;
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &pre_tfrmd_desc, n,
                                 h, w, c);

  test_normal(&pre_tfrmd_desc, exp_size);
}

void test_2D(unsigned int dim2, unsigned int dim1, uint64_t exp_size) {
  zdnn_tensor_desc pre_tfrmd_desc;
  zdnn_init_pre_transformed_desc(ZDNN_2D, test_datatype, &pre_tfrmd_desc, dim2,
                                 dim1);

  test_normal(&pre_tfrmd_desc, exp_size);
}

/// Drive the creation of a tensor descriptor with the layout
/// ZDNN_2DS and passed in dimensions. This will then call
/// the test_main function to drive allocation and compare to
/// an expected value.
///
/// \param[in] e1 dimension 1
/// \param[in] e2 dimension 2
/// \param[in] exp_size expected allocation size
///
/// \return None
///
void test_2DS(uint32_t dim2, uint32_t dim1, uint64_t exp_size) {
  zdnn_tensor_desc pre_tfrmd_desc;
  zdnn_init_pre_transformed_desc(ZDNN_2DS, test_datatype, &pre_tfrmd_desc, dim2,
                                 dim1);

  test_normal(&pre_tfrmd_desc, exp_size);
}

/// Drive the creation of a tensor descriptor with the layout
/// ZDNN_3DS and passed in dimensions. This will then call
/// the test_main function to drive allocation and compare to
/// an expected value.
///
/// \param[in] e1 dimension 1
/// \param[in] e2 dimension 2
/// \param[in] e3 dimension 3
/// \param[in] exp_size expected allocation size
///
/// \return None
///
void test_3DS(uint32_t dim3, uint32_t dim2, uint32_t dim1, uint64_t exp_size) {
  zdnn_tensor_desc pre_tfrmd_desc;
  zdnn_init_pre_transformed_desc(ZDNN_3DS, test_datatype, &pre_tfrmd_desc, dim3,
                                 dim2, dim1);

  test_normal(&pre_tfrmd_desc, exp_size);
}

void test_NHWC_1x3x3x5() { test_NHWC(1, 3, 3, 5, 12288); }
void test_NHWC_5x32x32x3() { test_NHWC(5, 32, 32, 3, 655360); }
void test_NHWC_1x64x64x64() { test_NHWC(1, 64, 64, 64, 524288); }
void test_NHWC_1x8x8x1() { test_NHWC(1, 8, 8, 1, 32768); }
void test_NHWC_1x256x256x1() { test_NHWC(1, 256, 256, 1, 8388608); }
void test_NHWC_1x1x256x1() { test_NHWC(1, 1, 256, 1, 32768); }

void test_2D_8x8() { test_2D(8, 8, 4096); }

void test_2DS_1x8() { test_2DS(1, 8, 4096); }
void test_2DS_8x1() { test_2DS(8, 1, 32768); }
void test_2DS_8x8() { test_2DS(8, 8, 32768); }
void test_2DS_32x8() { test_2DS(32, 8, 131072); }
void test_2DS_64x8() { test_2DS(64, 8, 262144); }
void test_2DS_64x64() { test_2DS(64, 64, 262144); }
void test_2DS_256x32() { test_2DS(256, 32, 1048576); }
void test_2DS_256x256() { test_2DS(256, 256, 4194304); }

void test_3DS_1x8x1() { test_3DS(1, 8, 1, 4096); }
void test_3DS_8x8x1() { test_3DS(8, 8, 1, 32768); }
void test_3DS_8x8x8() { test_3DS(8, 8, 8, 32768); }
void test_3DS_16x32x8() { test_3DS(16, 32, 8, 65536); }
void test_3DS_16x64x8() { test_3DS(16, 64, 8, 131072); }
void test_3DS_16x256x32() { test_3DS(16, 256, 32, 524288); }
void test_3DS_16x64x64() { test_3DS(16, 64, 64, 131072); }
void test_3DS_16x256x256() { test_3DS(16, 256, 256, 2097152); }

#define TEST_CONCAT_2DS(mode, size, status1, status2, a, b)                    \
  void test_##mode##_##a##x##b() {                                             \
    test_concat(ZDNN_2DS, mode, size, status1, status2, a, b);                 \
  }

#define TEST_CONCAT_3DS(mode, size, status1, status2, a, b, c)                 \
  void test_##mode##_##a##x##b##x##c() {                                       \
    test_concat(ZDNN_3DS, mode, size, status1, status2, a, b, c);              \
  }

TEST_CONCAT_2DS(CONCAT_LSTM, 16384, ZDNN_OK, ZDNN_OK, 1, 8)
TEST_CONCAT_2DS(CONCAT_LSTM, 32768, ZDNN_OK, ZDNN_OK, 2, 32)
TEST_CONCAT_2DS(CONCAT_LSTM, 16384, ZDNN_OK, ZDNN_OK, 1, 64)
TEST_CONCAT_2DS(CONCAT_LSTM, 65536, ZDNN_OK, ZDNN_OK, 2, 70)
TEST_CONCAT_2DS(CONCAT_LSTM, 32768, ZDNN_OK, ZDNN_OK, 1, 128)
TEST_CONCAT_2DS(CONCAT_LSTM, 98304, ZDNN_OK, ZDNN_OK, 2, 150)

TEST_CONCAT_3DS(CONCAT_LSTM, 16384, ZDNN_OK, ZDNN_OK, 1, 2, 8)
TEST_CONCAT_3DS(CONCAT_LSTM, 32768, ZDNN_OK, ZDNN_OK, 2, 5, 32)
TEST_CONCAT_3DS(CONCAT_LSTM, 16384, ZDNN_OK, ZDNN_OK, 1, 3, 64)
TEST_CONCAT_3DS(CONCAT_LSTM, 65536, ZDNN_OK, ZDNN_OK, 2, 10, 70)
TEST_CONCAT_3DS(CONCAT_LSTM, 65536, ZDNN_OK, ZDNN_OK, 1, 34, 128)
TEST_CONCAT_3DS(CONCAT_LSTM, 196608, ZDNN_OK, ZDNN_OK, 2, 50, 150)

void test_CONCAT_LSTM_fail_unsupported_layout() {
  // bad layout: ZDNN_4D as pre-transformed yields ZDNN_INVALID_LAYOUT
  test_concat(ZDNN_4D, CONCAT_LSTM, 0, ZDNN_INVALID_LAYOUT, 1, 2, 3, 4);
}

void test_CONCAT_LSTM_max_dim1() {
  // Confirm we pass when at the maximum number of dim1 elements
  // LSTM concatenates 4 gates.
  uint32_t max_dim1 = max_concat_dim1(4);
  test_concat(ZDNN_2DS, CONCAT_LSTM, 2097152, ZDNN_OK, ZDNN_OK, 1, max_dim1);
}

void test_CONCAT_LSTM_fail_dim1_too_big() {
  // zdnn_generate_transformed_desc_concatenated() yields no error but
  // zdnn_allochelper() yields ZDNN_DATA_ERROR during it's checks.
  // LSTM concatenates 4 gates.
  uint32_t max_dim1 = max_concat_dim1(4);
  test_concat(ZDNN_2DS, CONCAT_LSTM, 0, ZDNN_OK, ZDNN_INVALID_SHAPE, 1,
              max_dim1 + 1);
}

void test_CONCAT_LSTM_max_dim1_API_doc() {
  // This value is hardcoded in our API documentation. If the hardware makes
  // a change that alters the max value, this UT will fail.
  uint64_t max_dim1 = max_concat_dim1(4);
  uint64_t doc_max_dim1 = 8192;

  TEST_ASSERT_MESSAGE_FORMATTED(
      max_dim1 == doc_max_dim1,
      "hardware returned a maximum dim1 of %" PRIu64
      " but our LSTM API documents the hidden_stat_size limit %" PRIu64
      ". Update documentation and this test to match new value.",
      max_dim1, doc_max_dim1);
}

// test_CONCAT_GRU_* tests are based off test_CONCAT_LSTM_*, with smaller
// expected sizes ( = 3/4 of test_CONCAT_LSTM_*'s )

TEST_CONCAT_2DS(CONCAT_GRU, 12288, ZDNN_OK, ZDNN_OK, 1, 8)
TEST_CONCAT_2DS(CONCAT_GRU, 24576, ZDNN_OK, ZDNN_OK, 2, 32)
TEST_CONCAT_2DS(CONCAT_GRU, 12288, ZDNN_OK, ZDNN_OK, 1, 64)
TEST_CONCAT_2DS(CONCAT_GRU, 49152, ZDNN_OK, ZDNN_OK, 2, 70)
TEST_CONCAT_2DS(CONCAT_GRU, 24576, ZDNN_OK, ZDNN_OK, 1, 128)
TEST_CONCAT_2DS(CONCAT_GRU, 73728, ZDNN_OK, ZDNN_OK, 2, 150)

TEST_CONCAT_3DS(CONCAT_GRU, 12288, ZDNN_OK, ZDNN_OK, 1, 2, 8)
TEST_CONCAT_3DS(CONCAT_GRU, 24576, ZDNN_OK, ZDNN_OK, 2, 5, 32)
TEST_CONCAT_3DS(CONCAT_GRU, 12288, ZDNN_OK, ZDNN_OK, 1, 3, 64)
TEST_CONCAT_3DS(CONCAT_GRU, 49152, ZDNN_OK, ZDNN_OK, 2, 10, 70)
TEST_CONCAT_3DS(CONCAT_GRU, 49152, ZDNN_OK, ZDNN_OK, 1, 34, 128)
TEST_CONCAT_3DS(CONCAT_GRU, 147456, ZDNN_OK, ZDNN_OK, 2, 50, 150)

void test_CONCAT_GRU_fail_unsupported_layout() {
  // bad layout: ZDNN_4D as pre-transformed yields ZDNN_INVALID_LAYOUT
  test_concat(ZDNN_4D, CONCAT_GRU, 0, ZDNN_INVALID_LAYOUT, 1, 2, 3, 4);
}

void test_CONCAT_GRU_max_dim1() {
  // Confirm we pass when at the maximum number of dim1 elements
  // GRU concatenates 3 gates.
  uint64_t max_dim1 = max_concat_dim1(3);
  test_concat(ZDNN_2DS, CONCAT_GRU, 2088960, ZDNN_OK, ZDNN_OK, 1, max_dim1);
}

void test_CONCAT_GRU_fail_dim1_too_big() {
  // zdnn_generate_transformed_desc_concatenated() yields no error but
  // zdnn_allochelper() yields ZDNN_DATA_ERROR during it's checks.
  // GRU concatenates 3 gates.
  uint64_t max_dim1 = max_concat_dim1(3);
  test_concat(ZDNN_2DS, CONCAT_GRU, 0, ZDNN_OK, ZDNN_INVALID_SHAPE, 1,
              max_dim1 + 1);
}

void test_CONCAT_GRU_max_dim1_API_doc() {
  // This value is hardcoded in our API documentation. If the hardware makes
  // a change that alters this, this UT will fail.
  uint64_t max_dim1 = max_concat_dim1(3);
  uint64_t doc_max_dim1 = 10880;

  TEST_ASSERT_MESSAGE_FORMATTED(
      max_dim1 == doc_max_dim1,
      "hardware returned a maximum dim1 of %" PRIu64
      " but our GRU API documents the hidden_stat_size limit %" PRIu64
      ". Update documentation and this test to match new value.",
      max_dim1, doc_max_dim1);
}

TEST_CONCAT_3DS(CONCAT_BIDIR_OUTPUT, 8192, ZDNN_OK, ZDNN_OK, 1, 2, 8)
TEST_CONCAT_3DS(CONCAT_BIDIR_OUTPUT, 16384, ZDNN_OK, ZDNN_OK, 2, 5, 32)
TEST_CONCAT_3DS(CONCAT_BIDIR_OUTPUT, 8192, ZDNN_OK, ZDNN_OK, 1, 3, 64)
TEST_CONCAT_3DS(CONCAT_BIDIR_OUTPUT, 32768, ZDNN_OK, ZDNN_OK, 2, 10, 70)
TEST_CONCAT_3DS(CONCAT_BIDIR_OUTPUT, 32768, ZDNN_OK, ZDNN_OK, 1, 34, 128)
TEST_CONCAT_3DS(CONCAT_BIDIR_OUTPUT, 98304, ZDNN_OK, ZDNN_OK, 2, 50, 150)

void test_CONCAT_BIDIR_OUTPUT_fail_unsupported_layout() {
  // bad layout: ZDNN_2DS as pre-transformed yields ZDNN_INVALID_LAYOUT
  test_concat(ZDNN_2DS, CONCAT_BIDIR_OUTPUT, 0, ZDNN_INVALID_LAYOUT, 1, 2);
}

void test_CONCAT_BIDIR_OUTPUT_max_dim1() {
  // Confirm we pass when at the maximum number of dim1 elements
  uint64_t max_dim1 = max_concat_dim1(2);
  test_concat(ZDNN_3DS, CONCAT_BIDIR_OUTPUT, 2097152, ZDNN_OK, ZDNN_OK, 1, 2,
              max_dim1);
}

void test_CONCAT_BIDIR_OUTPUT_fail_dim1_too_big() {
  // zdnn_generate_transformed_desc_concatenated() yields no error but
  // zdnn_allochelper() yields ZDNN_DATA_ERROR during it's checks.
  uint64_t max_dim1 = max_concat_dim1(2);
  test_concat(ZDNN_3DS, CONCAT_BIDIR_OUTPUT, 0, ZDNN_OK, ZDNN_INVALID_SHAPE, 1,
              2, max_dim1 + 1);
}

int main(void) {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(test_NHWC_1x3x3x5);
  RUN_TEST_ALL_DATATYPES(test_NHWC_5x32x32x3);
  RUN_TEST_ALL_DATATYPES(test_NHWC_1x64x64x64);
  RUN_TEST_ALL_DATATYPES(test_NHWC_1x8x8x1);
  RUN_TEST_ALL_DATATYPES(test_NHWC_1x256x256x1);
  RUN_TEST_ALL_DATATYPES(test_NHWC_1x1x256x1);

  RUN_TEST_ALL_DATATYPES(test_2D_8x8);

  RUN_TEST_ALL_DATATYPES(test_2DS_1x8);
  RUN_TEST_ALL_DATATYPES(test_2DS_8x1);
  RUN_TEST_ALL_DATATYPES(test_2DS_8x8);
  RUN_TEST_ALL_DATATYPES(test_2DS_32x8);
  RUN_TEST_ALL_DATATYPES(test_2DS_64x8);
  RUN_TEST_ALL_DATATYPES(test_2DS_256x32);
  RUN_TEST_ALL_DATATYPES(test_2DS_64x64);
  RUN_TEST_ALL_DATATYPES(test_2DS_256x256);

  RUN_TEST_ALL_DATATYPES(test_3DS_1x8x1);
  RUN_TEST_ALL_DATATYPES(test_3DS_8x8x1);
  RUN_TEST_ALL_DATATYPES(test_3DS_8x8x8);
  RUN_TEST_ALL_DATATYPES(test_3DS_16x32x8);
  RUN_TEST_ALL_DATATYPES(test_3DS_16x64x8);
  RUN_TEST_ALL_DATATYPES(test_3DS_16x256x32);
  RUN_TEST_ALL_DATATYPES(test_3DS_16x64x64);
  RUN_TEST_ALL_DATATYPES(test_3DS_16x256x256);

  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_1x8);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_2x32);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_1x64);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_2x70);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_1x128);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_2x150);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_1x2x8);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_2x5x32);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_1x3x64);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_2x10x70);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_1x34x128);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_2x50x150);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_max_dim1);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_fail_unsupported_layout);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_fail_dim1_too_big);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_LSTM_max_dim1_API_doc);

  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_1x8);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_2x32);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_1x64);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_2x70);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_1x128);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_2x150);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_1x2x8);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_2x5x32);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_1x3x64);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_2x10x70);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_1x34x128);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_2x50x150);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_max_dim1);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_fail_unsupported_layout);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_fail_dim1_too_big);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_GRU_max_dim1_API_doc);

  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_1x2x8);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_2x5x32);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_1x3x64);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_2x10x70);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_1x34x128);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_2x50x150);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_max_dim1);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_fail_unsupported_layout);
  RUN_TEST_ALL_DATATYPES(test_CONCAT_BIDIR_OUTPUT_fail_dim1_too_big);

  return UNITY_END();
}
