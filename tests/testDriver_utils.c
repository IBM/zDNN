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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}

void tearDown(void) {}

/*
 * Checks that each mode of get_num_elements returns the expected value for each
 * supported pre_tfrmd data type
 */
void test_num_elements_concat(zdnn_data_layouts layout,
                              unsigned char zdnn_ztensor_concat_type,
                              uint32_t *shape, uint64_t exp_all,
                              uint64_t exp_single, uint64_t exp_no_pad) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  switch (layout) {
  case ZDNN_1D:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   shape[0]);
    break;
  case ZDNN_2D:
  case ZDNN_2DS:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   shape[0], shape[1]);
    break;
  case ZDNN_3D:
  case ZDNN_3DS:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   shape[0], shape[1], shape[2]);
    break;
  default:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   shape[0], shape[1], shape[2], shape[3]);
    break;
  }

  // This supports the not_concat helper can just call this helper
  if (zdnn_ztensor_concat_type == NO_CONCAT) {
    zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
    // The concat cases use this
  } else {
    zdnn_generate_transformed_desc_concatenated(
        &pre_tfrmd_desc, zdnn_ztensor_concat_type, &tfrmd_desc);
  }
  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  // Get output from each mode
  uint64_t all = get_num_elements(&ztensor, ELEMENTS_ALL);
  uint64_t single_concat = get_num_elements(&ztensor, ELEMENTS_CONCAT_SINGLE);
  uint64_t without_pad = get_num_elements(&ztensor, ELEMENTS_CONCAT_WO_PAD);

  // Check each mode's output matches the expected value.
  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_all == all,
      "For %s tfrmd_desc tensor we expected %" PRIu64
      " elements but ELEMENTS_ALL returned %" PRIu64 " elements",
      get_data_layout_str(tfrmd_desc.layout), exp_all, all);

  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_single == single_concat,
      "For %s tfrmd_desc tensor we expected %" PRIu64
      " elements but ELEMENTS_CONCAT_SINGLE returned %" PRIu64 " elements",
      get_data_layout_str(tfrmd_desc.layout), exp_single, single_concat);

  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_no_pad == without_pad,
      "For %s tfrmd_desc tensor we expected %" PRIu64
      " elements but ELEMENTS_CONCAT_WO_PAD returned %" PRIu64 " elements",
      get_data_layout_str(tfrmd_desc.layout), exp_no_pad, without_pad);
}

/*
 * Convenience non-concatenated helper that expects the output of each mode to
 * be the same value.
 */
void test_num_elements_not_concat(zdnn_data_layouts layout, uint32_t *shape,
                                  uint64_t expected) {
  test_num_elements_concat(layout, NO_CONCAT, shape, expected, expected,
                           expected);
}

/*
 * Test to ensure get_num_elements works with a NHWC tensor.
 */
void get_num_elements_nhwc() {
  uint32_t shape[] = {1, 4, 4, 1};
  test_num_elements_not_concat(ZDNN_NHWC, shape, 16);
}

/*
 * Test to ensure get_num_elements works with a 4D tensor.
 */
void get_num_elements_4d() {
  uint32_t shape[] = {1, 32, 15, 5};
  test_num_elements_not_concat(ZDNN_4D, shape, 2400);
}

/*
 * Test to ensure get_num_elements works with a 3DS tensor.
 */
void get_num_elements_3ds() {
  uint32_t shape[] = {3, 4, 4};
  test_num_elements_not_concat(ZDNN_3DS, shape, 48);
}

/*
 * Test to ensure get_num_elements works with a 3D tensor.
 */
void get_num_elements_3d() {
  uint32_t shape[] = {15, 4, 2};
  test_num_elements_not_concat(ZDNN_3D, shape, 120);
}

/*
 * Test to ensure get_num_elements works with a 2DS tensor.
 */
void get_num_elements_2ds() {
  uint32_t shape[] = {4, 4};
  test_num_elements_not_concat(ZDNN_2DS, shape, 16);
}

/*
 * Test to ensure get_num_elements works with a 2D tensor.
 */
void get_num_elements_2d() {
  uint32_t shape[] = {15, 4};
  test_num_elements_not_concat(ZDNN_2D, shape, 60);
}

/*
 * Test to ensure get_num_elements works with a 1D tensor.
 */
void get_num_elements_1d() {
  uint32_t shape[] = {16};
  test_num_elements_not_concat(ZDNN_1D, shape, 16);
}

/*
 * Test to ensure get_num_elements works with a 3DS LSTM tensor.
 */
void get_num_elements_lstm_3DS_input_concat() {
  uint32_t shape[] = {2, 3, 4};
  test_num_elements_concat(ZDNN_3DS, CONCAT_LSTM, shape, 1536, 24, 96);
}

/*
 * Test to ensure get_num_elements works with a 2DS LSTM tensor.
 */
void get_num_elements_lstm_2DS_input_concat() {
  uint32_t shape[] = {2, 3};
  test_num_elements_concat(ZDNN_2DS, CONCAT_LSTM, shape, 512, 6, 24);
}

/*
 * Test to ensure get_num_elements works with a 3DS GRU tensor.
 */
void get_num_elements_gru_3DS_input_concat() {
  uint32_t shape[] = {2, 3, 4};
  test_num_elements_concat(ZDNN_3DS, CONCAT_GRU, shape, 1152, 24, 72);
}

/*
 * Test to ensure get_num_elements works with a 2DS GRU tensor.
 */
void get_num_elements_gru_2DS_input_concat() {
  uint32_t shape[] = {2, 3};
  test_num_elements_concat(ZDNN_2DS, CONCAT_GRU, shape, 384, 6, 18);
}

/*
 * Test to ensure get_num_elements works with an RNN bidir output tensor.
 */
void get_num_elements_bidir_output_concat() {
  uint32_t shape[] = {2, 3, 4};
  test_num_elements_concat(ZDNN_3DS, CONCAT_BIDIR_OUTPUT, shape, 768, 24, 48);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(get_num_elements_nhwc);
  RUN_TEST_ALL_DATATYPES(get_num_elements_4d);
  RUN_TEST_ALL_DATATYPES(get_num_elements_3ds);
  RUN_TEST_ALL_DATATYPES(get_num_elements_3d);
  RUN_TEST_ALL_DATATYPES(get_num_elements_2ds);
  RUN_TEST_ALL_DATATYPES(get_num_elements_2d);
  RUN_TEST_ALL_DATATYPES(get_num_elements_1d);

  RUN_TEST_ALL_DATATYPES(get_num_elements_lstm_3DS_input_concat);
  RUN_TEST_ALL_DATATYPES(get_num_elements_lstm_2DS_input_concat);

  RUN_TEST_ALL_DATATYPES(get_num_elements_gru_3DS_input_concat);
  RUN_TEST_ALL_DATATYPES(get_num_elements_gru_2DS_input_concat);

  RUN_TEST_ALL_DATATYPES(get_num_elements_bidir_output_concat);

  return UNITY_END();
}
