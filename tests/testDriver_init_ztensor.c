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
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "testsupport.h"

void setUp(void) { VERIFY_HW_ENV; }

void tearDown(void) {}

// Helper method for tests that check the boundaries of the maximum dim1 index.
// Concatenated ztensors introduce padding that must be determined to test this.
// See zdnn_generate_transformed_desc_concatenated() to see padding equation.
uint32_t max_concat_dim1(uint32_t num_concats) {
  uint32_t temp = zdnn_get_max_for_dim(1) / num_concats;
  uint32_t max_concat_dim1 = temp - (temp % AIU_2BYTE_CELLS_PER_STICK);
  LOG_TRACE("returning %d\n", max_concat_dim1);
  return max_concat_dim1;
}

// test if we can zdnn_init_ztensor_with_malloc() correctly with the supplied
// pre-transformed and transformed descriptors
void test_main(zdnn_tensor_desc *pre_tfrmd_desc, zdnn_tensor_desc *tfrmd_desc,
               zdnn_concat_info info, uint64_t exp_size,
               zdnn_status exp_status_allochelper) {
  zdnn_ztensor ztensor;
  zdnn_status status;

  status = zdnn_init_ztensor_with_malloc(pre_tfrmd_desc, tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status_allochelper,
      "zdnn_init_ztensor_with_malloc() status is %08x (%s) "
      "but expects %08x (%s) (concat info = %08x)",
      status, zdnn_get_status_message(status), exp_status_allochelper,
      zdnn_get_status_message(exp_status_allochelper), info);

  // check and free buffer but only if expected
  // zdnn_init_ztensor_with_malloc() to work
  if (exp_status_allochelper == ZDNN_OK) {
    TEST_ASSERT_MESSAGE_FORMATTED(
        ztensor.buffer_size == exp_size,
        "zdnn_init_ztensor_with_malloc() returns incorrect size: %" PRIu64
        " (expects %" PRIu64 ") (concat info = %08x)",
        ztensor.buffer_size, exp_size, info);

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

  test_main(pre_tfrmd_desc, &tfrmd_desc, NO_CONCAT, exp_size, ZDNN_OK);
}

// test if we can zdnn_init_quantized_ztensor_with_malloc() correctly with the
// supplied pre-transformed and quantized transformed descriptors
void test_quantized_main(zdnn_tensor_desc *pre_tfrmd_desc,
                         zdnn_tensor_desc *tfrmd_desc, float scale,
                         float offset, uint64_t exp_size,
                         zdnn_status exp_status_allochelper) {
  zdnn_ztensor ztensor;
  zdnn_status status;

  status = zdnn_init_quantized_ztensor_with_malloc(pre_tfrmd_desc, tfrmd_desc,
                                                   scale, offset, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status_allochelper,
      "zdnn_init_quantized_ztensor_with_malloc() status is %08x (%s) "
      "but expects %08x (%s)",
      status, zdnn_get_status_message(status), exp_status_allochelper,
      zdnn_get_status_message(exp_status_allochelper));

  // check and free buffer but only if expected
  // zdnn_init_ztensor_with_malloc() to work
  if (exp_status_allochelper == ZDNN_OK) {
    TEST_ASSERT_MESSAGE_FORMATTED(ztensor.buffer_size == exp_size,
                                  "zdnn_init_quantized_ztensor_with_malloc() "
                                  "returns incorrect size: %" PRIu64
                                  " (expects %" PRIu64 ")",
                                  ztensor.buffer_size, exp_size);

    zdnn_free_ztensor_buffer(&ztensor);
  }
}

void test_quantized(zdnn_quantized_transform_types type, unsigned int n,
                    unsigned int h, unsigned int w, unsigned int c, float scale,
                    float offset, uint64_t exp_size) {

  zdnn_tensor_desc pre_tfrmd_desc;
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &pre_tfrmd_desc, n,
                                 h, w, c);

  zdnn_tensor_desc tfrmd_desc;
  zdnn_status status;

  status = zdnn_generate_quantized_transformed_desc(&pre_tfrmd_desc, type,
                                                    &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), ZDNN_OK,
      zdnn_get_status_message(ZDNN_OK));

  test_quantized_main(&pre_tfrmd_desc, &tfrmd_desc, scale, offset, exp_size,
                      ZDNN_OK);
}

/// Drive the creation of a FICO/ZRH ztensor with the provided pre-transformed
/// layout, data type and dims, and transformed layout (FICO/ZRH).  Then drive
/// allocation and compare to an expected value.
///
/// \param[in] pre_tfrmd_layout          pre-transformed layout
/// \param[in] info                      concatenation info
/// \param[in] exp_size                  expected allocation size
/// \param[in] exp_status_gen_concat     expected status of _desc_concatenated()
/// \param[in] exp_status_allochelper    expected status of _allochelper()
/// \param[in] ...                       dimensions, outermost -> innermost
///                                      order  (ie shape order)
///
/// \return None - Fails test assertion if actual values don't match specified
///          exp values
///
void test_concat(zdnn_data_layouts pre_tfrmd_layout, zdnn_concat_info info,
                 uint64_t exp_size, zdnn_status exp_status_gen_concat,
                 zdnn_status exp_status_allochelper, ...) {

  zdnn_status status;
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;

  uint32_t num_things;

  switch (pre_tfrmd_layout) {
  case ZDNN_2DS:
  case ZDNN_3DS:
    num_things = get_data_layout_dims(pre_tfrmd_layout);
    break;
  default: // for driving an "invalid layout" testcase
    num_things = 4;
    break;
  }

  va_list v_list;
  va_start(v_list, exp_status_allochelper);
  uint32_t dim_nums[num_things];
  for (uint32_t i = 0; i < num_things; i++) {
    dim_nums[i] = va_arg(v_list, uint32_t);
  }
  va_end(v_list);

  switch (pre_tfrmd_layout) {
  case ZDNN_2DS:
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, dim_nums[0], dim_nums[1]);
    break;
  case ZDNN_3DS:
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, dim_nums[0], dim_nums[1],
                                   dim_nums[2]);
    break;
  default:
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, dim_nums[0], dim_nums[1],
                                   dim_nums[2], dim_nums[3]);
    break;
  }

  status = zdnn_generate_transformed_desc_concatenated(&pre_tfrmd_desc, info,
                                                       &tfrmd_desc);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status_gen_concat,
      "zdnn_generate_transformed_desc_concatenated() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), exp_status_gen_concat,
      zdnn_get_status_message(exp_status_gen_concat));

  // do the rest if expected zdnn_generate_transformed_desc_concatenated() to
  // work
  if (exp_status_gen_concat == ZDNN_OK) {
    test_main(&pre_tfrmd_desc, &tfrmd_desc, info, exp_size,
              exp_status_allochelper);
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

// Different quantized types have different cells per stick. Focus on innermost
// dimension limits.
void test_quantized_DLFLOAT_1x3x3x5() {
  test_quantized(QUANTIZED_DLFLOAT16, 1, 3, 3, 5, 5, 6, 12288);
}
void test_quantized_DLFLOAT_1x3x3x64() {
  test_quantized(QUANTIZED_DLFLOAT16, 1, 3, 3, 64, 7, 8, 12288);
}
void test_quantized_DLFLOAT_1x3x3x65() {
  test_quantized(QUANTIZED_DLFLOAT16, 1, 3, 3, 65, 9, 10, 24576);
}
void test_quantized_INT8_1x3x3x5() {
  test_quantized(QUANTIZED_INT8, 1, 3, 3, 5, 5, 6, 12288);
}
void test_quantized_INT8_1x3x3x128() {
  test_quantized(QUANTIZED_INT8, 1, 3, 3, 128, 7, 8, 12288);
}
void test_quantized_INT8_1x3x3x129() {
  test_quantized(QUANTIZED_INT8, 1, 3, 3, 129, 9, 10, 24576);
}
void test_quantized_WEIGHTS_INT8_1x3x3x5() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 3, 5, 5, 6, 12288);
}
void test_quantized_WEIGHTS_INT8_1x3x3x64() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 3, 64, 7, 8, 12288);
}
void test_quantized_WEIGHTS_INT8_1x3x3x65() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 3, 65, 9, 10, 24576);
}
void test_quantized_WEIGHTS_INT8_1x3x32x64() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 32, 64, 9, 10, 12288);
}
void test_quantized_WEIGHTS_INT8_1x3x33x64() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 33, 64, 9, 10, 12288);
}
void test_quantized_WEIGHTS_INT8_1x3x64x64() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 64, 64, 9, 10, 12288);
}
void test_quantized_WEIGHTS_INT8_1x3x65x64() {
  test_quantized(QUANTIZED_WEIGHTS_INT8, 1, 3, 65, 64, 9, 10, 24576);
}

// TODO, will need to drive INT32 scenarios

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

//------------------------------------------------------------

// any combination of PREV_ UNI/BIDIR + BIASES/HIDDEN_BIASES should yield the
// same results

void test_lstm_biases_1x8() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j],
                  16384, ZDNN_OK, ZDNN_OK, 1, 8);
    }
  }
}

void test_lstm_biases_2x32() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j],
                  32768, ZDNN_OK, ZDNN_OK, 2, 32);
    }
  }
}

void test_lstm_biases_1x64() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j],
                  16384, ZDNN_OK, ZDNN_OK, 1, 64);
    }
  }
}

void test_lstm_biases_2x70() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j],
                  65536, ZDNN_OK, ZDNN_OK, 2, 70);
    }
  }
}

void test_lstm_biases_1x128() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j],
                  32768, ZDNN_OK, ZDNN_OK, 1, 128);
    }
  }
}

void test_lstm_biases_2x150() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j],
                  98304, ZDNN_OK, ZDNN_OK, 2, 150);
    }
  }
}

//------------------------------------------------------------

// PREV_ UNI/BIDIR + HIDDEN_WEIGHTS and UNI + WEIGHTS should yield the same
// results

void test_lstm_no_vconcat_weights_1x2x8() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i], 16384, ZDNN_OK,
                ZDNN_OK, 1, 2, 8);
  }
}

void test_lstm_no_vconcat_weights_2x5x32() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i], 32768, ZDNN_OK,
                ZDNN_OK, 2, 5, 32);
  }
}

void test_lstm_no_vconcat_weights_1x3x64() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i], 16384, ZDNN_OK,
                ZDNN_OK, 1, 3, 64);
  }
}

void test_lstm_no_vconcat_weights_2x10x70() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i], 65536, ZDNN_OK,
                ZDNN_OK, 2, 10, 70);
  }
}

void test_lstm_no_vconcat_weights_1x34x128() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i], 65536, ZDNN_OK,
                ZDNN_OK, 1, 34, 128);
  }
}

void test_lstm_no_vconcat_weights_2x50x150() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i], 196608, ZDNN_OK,
                ZDNN_OK, 2, 50, 150);
  }
}

//------------------------------------------------------------

// lstm_prev_bidir_weights expected size:
//     dim3 * (2 * PADDED(dim2/2) / AIU_STICKS_PER_PAGE) *
//     ceil(dim1/AIU_2BYTE_CELLS_PER_STICK) *
//     * AIU_PAGESIZE_IN_BYTES * 4

void test_lstm_prev_bidir_weights_1x2x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 65536,
              ZDNN_OK, ZDNN_OK, 1, 2, 8);
}

void test_lstm_prev_bidir_weights_2x2x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 2, 2, 8);
}

void test_lstm_prev_bidir_weights_1x34x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 65536,
              ZDNN_OK, ZDNN_OK, 1, 34, 8);
}

void test_lstm_prev_bidir_weights_2x34x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 2, 34, 8);
}

void test_lstm_prev_bidir_weights_1x64x10() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 65536,
              ZDNN_OK, ZDNN_OK, 1, 64, 10);
}

void test_lstm_prev_bidir_weights_2x64x10() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 2, 64, 10);
}

void test_lstm_prev_bidir_weights_1x70x20() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 65536,
              ZDNN_OK, ZDNN_OK, 1, 70, 20);
}

void test_lstm_prev_bidir_weights_2x70x20() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 2, 70, 20);
}

void test_lstm_prev_bidir_weights_1x10x32() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 65536,
              ZDNN_OK, ZDNN_OK, 1, 10, 32);
}

void test_lstm_prev_bidir_weights_2x10x32() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 2, 10, 32);
}

void test_lstm_prev_bidir_weights_1x6x64() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 65536,
              ZDNN_OK, ZDNN_OK, 1, 6, 64);
}

void test_lstm_prev_bidir_weights_2x6x64() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 2, 6, 64);
}

void test_lstm_prev_bidir_weights_1x10x70() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 1, 10, 70);
}

void test_lstm_prev_bidir_weights_2x10x70() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              262144, ZDNN_OK, ZDNN_OK, 2, 10, 70);
}

void test_lstm_prev_bidir_weights_1x34x128() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              131072, ZDNN_OK, ZDNN_OK, 1, 34, 128);
}

void test_lstm_prev_bidir_weights_2x34x128() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              262144, ZDNN_OK, ZDNN_OK, 2, 34, 128);
}

void test_lstm_prev_bidir_weights_1x50x150() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              196608, ZDNN_OK, ZDNN_OK, 1, 50, 150);
}

void test_lstm_prev_bidir_weights_2x50x150() {
  test_concat(ZDNN_3DS, RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
              393216, ZDNN_OK, ZDNN_OK, 2, 50, 150);
}

//------------------------------------------------------------

void test_CONCAT_LSTM_fail_unsupported_layout() {
  // bad layout: ZDNN_4D as pre-transformed yields ZDNN_INVALID_LAYOUT
  test_concat(ZDNN_4D, RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_WEIGHTS, 0,
              ZDNN_INVALID_LAYOUT, 0, 1, 2, 3, 4);
}

void test_CONCAT_LSTM_max_dim1() {
  // Confirm we pass when at the maximum number of dim1 elements
  // LSTM concatenates 4 gates.
  uint32_t max_dim1 = max_concat_dim1(4);
  // If MDnIS exists, use larger number; otherwise keep Telum I value.
  uint64_t expected_size =
      nnpa_query_result.max_dim1_index_size ? 134217728 : 2097152;

  test_concat(ZDNN_2DS, USAGE_BIASES | RNN_TYPE_LSTM | PREV_LAYER_UNI,
              expected_size, ZDNN_OK, ZDNN_OK, 1, max_dim1);
}

void test_CONCAT_LSTM_fail_dim1_too_big() {
  // zdnn_generate_transformed_desc_concatenated() yields no error but
  // zdnn_allochelper() yields ZDNN_DATA_ERROR during it's checks.
  // LSTM concatenates 4 gates.
  uint32_t max_dim1 = max_concat_dim1(4);
  test_concat(ZDNN_2DS, USAGE_BIASES | RNN_TYPE_LSTM | PREV_LAYER_UNI, 0,
              ZDNN_OK, ZDNN_INVALID_SHAPE, 1, max_dim1 + 1);
}

//------------------------------------------------------------

// test_gru_* tests are based off test_lstm_*, with smaller expected sizes ( =
// 3/4 of test_lstm__*'s )

void test_gru_biases_1x8() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                  12288, ZDNN_OK, ZDNN_OK, 1, 8);
    }
  }
}

void test_gru_biases_2x32() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                  24576, ZDNN_OK, ZDNN_OK, 2, 32);
    }
  }
}

void test_gru_biases_1x64() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                  12288, ZDNN_OK, ZDNN_OK, 1, 64);
    }
  }
}

void test_gru_biases_2x70() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                  49152, ZDNN_OK, ZDNN_OK, 2, 70);
    }
  }
}

void test_gru_biases_1x128() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                  24576, ZDNN_OK, ZDNN_OK, 1, 128);
    }
  }
}

void test_gru_biases_2x150() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat(ZDNN_2DS, RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                  73728, ZDNN_OK, ZDNN_OK, 2, 150);
    }
  }
}

//------------------------------------------------------------

// PREV_ UNI/BIDIR + HIDDEN_WEIGHTS and UNI + WEIGHTS should yield the same
// results

void test_gru_no_vconcat_weights_1x2x8() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i], 12288, ZDNN_OK,
                ZDNN_OK, 1, 2, 8);
  }
}

void test_gru_no_vconcat_weights_2x5x32() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i], 24576, ZDNN_OK,
                ZDNN_OK, 2, 5, 32);
  }
}

void test_gru_no_vconcat_weights_1x3x64() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i], 12288, ZDNN_OK,
                ZDNN_OK, 1, 3, 64);
  }
}

void test_gru_no_vconcat_weights_2x10x70() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i], 49152, ZDNN_OK,
                ZDNN_OK, 2, 10, 70);
  }
}

void test_gru_no_vconcat_weights_1x34x128() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i], 49152, ZDNN_OK,
                ZDNN_OK, 1, 34, 128);
  }
}

void test_gru_no_vconcat_weights_2x50x150() {
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i], 147456, ZDNN_OK,
                ZDNN_OK, 2, 50, 150);
  }
}

//------------------------------------------------------------

// gru_prev_bidir_weights expected size:
//     dim3 * (2 * PADDED(dim2/2) / AIU_STICKS_PER_PAGE) *
//     ceil(dim1/AIU_2BYTE_CELLS_PER_STICK) *
//     * AIU_PAGESIZE_IN_BYTES * 3

void test_gru_prev_bidir_weights_1x2x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 49152,
              ZDNN_OK, ZDNN_OK, 1, 2, 8);
}

void test_gru_prev_bidir_weights_2x2x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 2, 2, 8);
}

void test_gru_prev_bidir_weights_1x34x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 49152,
              ZDNN_OK, ZDNN_OK, 1, 34, 8);
}

void test_gru_prev_bidir_weights_2x34x8() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 2, 34, 8);
}

void test_gru_prev_bidir_weights_1x64x10() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 49152,
              ZDNN_OK, ZDNN_OK, 1, 64, 10);
}

void test_gru_prev_bidir_weights_2x64x10() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 2, 64, 10);
}

void test_gru_prev_bidir_weights_1x70x20() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 49152,
              ZDNN_OK, ZDNN_OK, 1, 70, 20);
}

void test_gru_prev_bidir_weights_2x70x20() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 2, 70, 20);
}

void test_gru_prev_bidir_weights_1x10x32() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 49152,
              ZDNN_OK, ZDNN_OK, 1, 10, 32);
}

void test_gru_prev_bidir_weights_2x10x32() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 2, 10, 32);
}

void test_gru_prev_bidir_weights_1x6x64() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 49152,
              ZDNN_OK, ZDNN_OK, 1, 6, 64);
}

void test_gru_prev_bidir_weights_2x6x64() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 2, 6, 64);
}

void test_gru_prev_bidir_weights_1x10x70() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 1, 10, 70);
}

void test_gru_prev_bidir_weights_2x10x70() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 196608,
              ZDNN_OK, ZDNN_OK, 2, 10, 70);
}

void test_gru_prev_bidir_weights_1x34x128() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 98304,
              ZDNN_OK, ZDNN_OK, 1, 34, 128);
}

void test_gru_prev_bidir_weights_2x34x128() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 196608,
              ZDNN_OK, ZDNN_OK, 2, 34, 128);
}

void test_gru_prev_bidir_weights_1x50x150() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 147456,
              ZDNN_OK, ZDNN_OK, 1, 50, 150);
}

void test_gru_prev_bidir_weights_2x50x150() {
  test_concat(ZDNN_3DS, RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 294912,
              ZDNN_OK, ZDNN_OK, 2, 50, 150);
}

//------------------------------------------------------------

void test_CONCAT_GRU_fail_unsupported_layout() {
  // bad layout: ZDNN_4D as pre-transformed yields ZDNN_INVALID_LAYOUT
  test_concat(ZDNN_4D, RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_WEIGHTS, 0,
              ZDNN_INVALID_LAYOUT, 1, 2, 3, 4);
}

void test_CONCAT_GRU_max_dim1() {
  // Confirm we pass when at the maximum number of dim1 elements
  // GRU concatenates 3 gates.
  uint64_t max_dim1 = max_concat_dim1(3);
  // If MDnIS exists, use larger number; otherwise keep Telum I value.
  uint64_t expected_size =
      nnpa_query_result.max_dim1_index_size ? 134209536 : 2088960;
  test_concat(ZDNN_2DS, RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_BIASES,
              expected_size, ZDNN_OK, ZDNN_OK, 1, max_dim1);
}

void test_CONCAT_GRU_fail_dim1_too_big() {
  // zdnn_generate_transformed_desc_concatenated() yields no error but
  // zdnn_allochelper() yields ZDNN_DATA_ERROR during it's checks.
  // GRU concatenates 3 gates.
  uint64_t max_dim1 = max_concat_dim1(3);
  test_concat(ZDNN_2DS, RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_BIASES, 0,
              ZDNN_OK, ZDNN_INVALID_SHAPE, 1, max_dim1 + 1);
}

//------------------------------------------------------------

void test_rnn_output(uint32_t dim4, uint32_t dim3, uint32_t dim2, uint32_t dim1,
                     uint64_t exp_size) {
  zdnn_tensor_desc pre_tfrmd_desc;
  zdnn_init_pre_transformed_desc(ZDNN_4DS, test_datatype, &pre_tfrmd_desc, dim4,
                                 dim3, dim2, dim1);

  test_normal(&pre_tfrmd_desc, exp_size);
}

void test_uni_output_1x1x2x8() { test_rnn_output(1, 1, 2, 8, 4096); }

void test_uni_output_2x1x5x32() { test_rnn_output(2, 1, 5, 32, 8192); }

void test_uni_output_1x1x3x64() { test_rnn_output(1, 1, 3, 64, 4096); }

void test_uni_output_2x1x10x70() { test_rnn_output(2, 1, 10, 70, 16384); }

void test_uni_output_1x1x34x128() { test_rnn_output(1, 1, 34, 128, 16384); }

void test_uni_output_2x1x50x150() { test_rnn_output(2, 1, 50, 150, 49152); }

void test_bidir_output_1x2x2x8() { test_rnn_output(1, 2, 2, 8, 8192); }

void test_bidir_output_2x2x5x32() { test_rnn_output(2, 2, 5, 32, 16384); }

void test_bidir_output_1x2x3x64() { test_rnn_output(1, 2, 3, 64, 8192); }

void test_bidir_output_2x2x10x70() { test_rnn_output(2, 2, 10, 70, 32768); }

void test_bidir_output_1x2x34x128() { test_rnn_output(1, 2, 34, 128, 32768); }

void test_bidir_output_2x2x50x150() { test_rnn_output(2, 2, 50, 150, 98304); }

//------------------------------------------------------------

void test_bidir_output_max_dim1() {
  // Confirm we pass when at the maximum number of dim1 elements

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  uint64_t max_dim1 = max_concat_dim1(2);
  zdnn_init_pre_transformed_desc(ZDNN_4DS, test_datatype, &pre_tfrmd_desc, 1, 2,
                                 2, max_dim1);

  zdnn_status status =
      zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), ZDNN_OK,
      zdnn_get_status_message(ZDNN_OK));

  // If MDnIS exists, use larger number; otherwise keep Telum I value.
  uint64_t expected_size =
      nnpa_query_result.max_dim1_index_size ? 134217728 : 2097152;
  test_main(&pre_tfrmd_desc, &tfrmd_desc, NO_CONCAT, expected_size, ZDNN_OK);
}

void test_bidir_output_fail_dim1_too_big() {

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  uint64_t max_dim1 = max_concat_dim1(2);
  zdnn_init_pre_transformed_desc(ZDNN_4DS, test_datatype, &pre_tfrmd_desc, 1, 2,
                                 3, max_dim1 + 1);

  // zdnn_generate_transformed_desc_concatenated() yields no error but
  // zdnn_allochelper() yields ZDNN_DATA_ERROR during it's checks.

  zdnn_status status =
      zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), ZDNN_OK,
      zdnn_get_status_message(ZDNN_OK));

  test_main(&pre_tfrmd_desc, &tfrmd_desc, NO_CONCAT, 9999, ZDNN_INVALID_SHAPE);
}

void test_zdnn_init_ztensor_function() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  // Set ztensor to all 1s prior to function call.
  memset(&ztensor, 1, sizeof(ztensor));

  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  TEST_ASSERT_MESSAGE(
      ztensor.pre_transformed_desc == &pre_tfrmd_desc,
      "Expected ztensor to point to passed in pre-transformed descriptor.");
  TEST_ASSERT_MESSAGE(
      ztensor.transformed_desc == &tfrmd_desc,
      "Expected ztensor to point to passed in transformed descriptor.");
  TEST_ASSERT_MESSAGE(
      false == ztensor.is_transformed,
      "Expected ztensor to have is_transformed initialized as false.");

  // We expect reserved area to be all zeros, create variable for memcmp
  char expected_reserved[sizeof(ztensor.reserved)] = {0};

  TEST_ASSERT_MESSAGE(
      memcmp(expected_reserved, ztensor.reserved, sizeof(expected_reserved)) ==
          0,
      "Expected ztensor reserved area not initialized to zeroes.");

  // We expect reserved2 area to be all zeros, create variable for memcmp
  char expected_reserved2[sizeof(ztensor.reserved2)] = {0};

  TEST_ASSERT_MESSAGE(
      memcmp(expected_reserved2, ztensor.reserved2,
             sizeof(expected_reserved2)) == 0,
      "Expected ztensor reserved2 area not initialized to zeroes.");
}

void test_zdnn_init_ztensor_via_malloc_function() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  // Create a very basic descriptors to satisfy malloc portion of init function
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, 1, 1, 1, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  // Set ztensor to all 1s prior to function call.
  memset(&ztensor, 1, sizeof(ztensor));

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  TEST_ASSERT_MESSAGE(
      ztensor.pre_transformed_desc == &pre_tfrmd_desc,
      "Expected ztensor to point to passed in pre-transformed descriptor.");
  TEST_ASSERT_MESSAGE(
      ztensor.transformed_desc == &tfrmd_desc,
      "Expected ztensor to point to passed in transformed descriptor.");
  TEST_ASSERT_MESSAGE(
      false == ztensor.is_transformed,
      "Expected ztensor to have is_transformed initialized as false.");

  // We expect reserved area to be all zeros, create variable for memcmp
  char expected_reserved[sizeof(ztensor.reserved)] = {0};

  TEST_ASSERT_MESSAGE(
      memcmp(expected_reserved, ztensor.reserved, sizeof(expected_reserved)) ==
          0,
      "Expected ztensor reserved area not initialized to zeroes.");

  // We expect reserved2 area to be all zeros, create variable for memcmp
  char expected_reserved2[sizeof(ztensor.reserved2)] = {0};

  TEST_ASSERT_MESSAGE(
      memcmp(expected_reserved2, ztensor.reserved2,
             sizeof(expected_reserved2)) == 0,
      "Expected ztensor reserved2 area not initialized to zeroes.");

  zdnn_free_ztensor_buffer(&ztensor);
}

void test_zdnn_is_quantized_ztensor_scale() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, 1, 1, 1, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  ztensor.rec_scale = 0.2;

  TEST_ASSERT_MESSAGE(zdnn_is_quantized_ztensor(&ztensor) == true,
                      "Expected ztensor not indicated as a quantized ztensor.");
}

void test_zdnn_is_quantized_ztensor_false() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, 1, 1, 1, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  ztensor.rec_scale = 0;

  TEST_ASSERT_MESSAGE(zdnn_is_quantized_ztensor(&ztensor) == false,
                      "Expected ztensor indicated as a quantized ztensor.");
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_NHWC_1x3x3x5);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_NHWC_5x32x32x3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_NHWC_1x64x64x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_NHWC_1x8x8x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_NHWC_1x256x256x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_NHWC_1x1x256x1);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_quantized_DLFLOAT_1x3x3x5);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_quantized_DLFLOAT_1x3x3x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_quantized_DLFLOAT_1x3x3x65);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_quantized_INT8_1x3x3x5);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_quantized_INT8_1x3x3x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_quantized_INT8_1x3x3x129);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x3x5);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x3x64);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x3x65);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x32x64);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x33x64);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x64x64);
  RUN_TEST(test_quantized_WEIGHTS_INT8_1x3x65x64);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2D_8x8);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_1x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_8x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_8x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_32x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_64x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_256x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_64x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2DS_256x256);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_1x8x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_8x8x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_8x8x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_16x32x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_16x64x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_16x256x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_16x64x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_3DS_16x256x256);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_biases_1x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_biases_2x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_biases_1x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_biases_2x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_biases_1x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_biases_2x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_no_vconcat_weights_1x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_no_vconcat_weights_2x5x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x34x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x34x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x64x10);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x64x10);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x70x20);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x70x20);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x10x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x10x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_no_vconcat_weights_1x3x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_no_vconcat_weights_2x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_no_vconcat_weights_1x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_no_vconcat_weights_2x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x6x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x6x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_1x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_lstm_prev_bidir_weights_2x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_CONCAT_LSTM_max_dim1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      test_CONCAT_LSTM_fail_unsupported_layout);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_CONCAT_LSTM_fail_dim1_too_big);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_biases_1x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_biases_2x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_biases_1x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_biases_2x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_biases_1x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_biases_2x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_no_vconcat_weights_1x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_no_vconcat_weights_2x5x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x34x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x34x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x64x10);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x64x10);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x70x20);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x70x20);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x10x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x10x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_no_vconcat_weights_1x3x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_no_vconcat_weights_2x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_no_vconcat_weights_1x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_no_vconcat_weights_2x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x6x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x6x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_1x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_gru_prev_bidir_weights_2x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_CONCAT_GRU_max_dim1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_CONCAT_GRU_fail_unsupported_layout);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_CONCAT_GRU_fail_dim1_too_big);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_uni_output_1x1x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_uni_output_2x1x5x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_1x2x2x8);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_2x2x5x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_uni_output_1x1x3x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_uni_output_2x1x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_uni_output_1x1x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_uni_output_2x1x50x150);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_1x2x3x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_2x2x10x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_1x2x34x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_2x2x50x150);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_max_dim1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_bidir_output_fail_dim1_too_big);

  RUN_TEST(test_zdnn_init_ztensor_function);
  RUN_TEST(test_zdnn_init_ztensor_via_malloc_function);

  RUN_TEST(test_zdnn_is_quantized_ztensor_scale);
  RUN_TEST(test_zdnn_is_quantized_ztensor_false);

  return UNITY_END();
}
