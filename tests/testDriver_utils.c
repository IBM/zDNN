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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}

void tearDown(void) {}

void test_num_elements(zdnn_data_layouts layout, uint32_t *shape,
                       uint64_t exp_pre, uint64_t exp_aiu) {
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

  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  // Get output from each mode
  uint64_t num_elements_pre = get_num_elements(&ztensor, ELEMENTS_PRE);
  uint64_t num_elements_aiu = get_num_elements(&ztensor, ELEMENTS_AIU);

  // Check each mode's output matches the expected value.
  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_pre == num_elements_pre,
      "For %s tensor we expected %" PRIu64
      " elements but ELEMENTS_PRE returned %" PRIu64 " elements",
      get_data_layout_str(tfrmd_desc.layout), exp_pre, num_elements_pre);

  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_aiu == num_elements_aiu,
      "For %s tensor we expected %" PRIu64
      " elements but ELEMENTS_AIU returned %" PRIu64 " elements",
      get_data_layout_str(tfrmd_desc.layout), exp_aiu, num_elements_aiu);
}

void test_num_elements_concat(zdnn_data_layouts layout, zdnn_concat_info info,
                              uint32_t *shape, uint64_t exp_single_gate,
                              uint64_t exp_all_gates, uint64_t exp_aiu) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  switch (layout) {
  case ZDNN_2DS:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   shape[0], shape[1]);
    break;
  case ZDNN_3DS:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   shape[0], shape[1], shape[2]);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("invalid pre-transformed layout: %s",
                                get_data_layout_str(layout));
  }

  zdnn_generate_transformed_desc_concatenated(&pre_tfrmd_desc, info,
                                              &tfrmd_desc);
  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  // Get output from each mode
  uint64_t num_elements_single_gate =
      get_num_elements(&ztensor, ELEMENTS_PRE_SINGLE_GATE);
  uint64_t num_elements_all_gates =
      get_num_elements(&ztensor, ELEMENTS_PRE_ALL_GATES);
  uint64_t num_elements_aiu = get_num_elements(&ztensor, ELEMENTS_AIU);

  // Check each mode's output matches the expected value.
  TEST_ASSERT_MESSAGE_FORMATTED(
      num_elements_single_gate == exp_single_gate,
      "For %s tensor we expected %" PRIu64
      " elements but ELEMENTS_PRE_SINGLE_GATE returned %" PRIu64
      " elements (info = %08x)",
      get_data_layout_str(tfrmd_desc.layout), exp_single_gate,
      num_elements_single_gate, info);

  TEST_ASSERT_MESSAGE_FORMATTED(
      "For %s tensor we expected %" PRIu64
      " elements but ELEMENTS_PRE_ALL_GATES returned %" PRIu64
      " elements (info = %08x)",
      get_data_layout_str(tfrmd_desc.layout), exp_all_gates,
      num_elements_all_gates, info);

  TEST_ASSERT_MESSAGE_FORMATTED(
      "For %s tensor we expected %" PRIu64
      " elements but ELEMENTS_AIU returned %" PRIu64 " elements (info = %08x)",
      get_data_layout_str(tfrmd_desc.layout), exp_aiu, num_elements_aiu, info);
}

/*
 * Test to ensure get_num_elements works with a NHWC tensor.
 */
void get_num_elements_nhwc() {
  uint32_t shape[] = {1, 4, 4, 1};
  test_num_elements(ZDNN_NHWC, shape, 16, 16);
}

/*
 * Test to ensure get_num_elements works with a 4D tensor.
 */
void get_num_elements_4d() {
  uint32_t shape[] = {1, 32, 15, 5};
  test_num_elements(ZDNN_4D, shape, 2400, 2400);
}

/*
 * Test to ensure get_num_elements works with a 3DS tensor.
 */
void get_num_elements_3ds() {
  uint32_t shape[] = {3, 4, 4};
  test_num_elements(ZDNN_3DS, shape, 48, 48);
}

/*
 * Test to ensure get_num_elements works with a 3D tensor.
 */
void get_num_elements_3d() {
  uint32_t shape[] = {15, 4, 2};
  test_num_elements(ZDNN_3D, shape, 120, 120);
}

/*
 * Test to ensure get_num_elements works with a 2DS tensor.
 */
void get_num_elements_2ds() {
  uint32_t shape[] = {4, 4};
  test_num_elements(ZDNN_2DS, shape, 16, 16);
}

/*
 * Test to ensure get_num_elements works with a 2D tensor.
 */
void get_num_elements_2d() {
  uint32_t shape[] = {15, 4};
  test_num_elements(ZDNN_2D, shape, 60, 60);
}

/*
 * Test to ensure get_num_elements works with a 1D tensor.
 */
void get_num_elements_1d() {
  uint32_t shape[] = {16};
  test_num_elements(ZDNN_1D, shape, 16, 16);
}

/*
 * Test to ensure get_num_elements works with a 3DS LSTM tensor that doesn't
 * require vertical concatenation.
 */
void get_num_elements_lstm_no_vconcat_weights() {
  uint32_t shape[] = {2, 3, 4};
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_num_elements_concat(ZDNN_3DS, RNN_TYPE_LSTM | no_vconcat_infos[i],
                             shape, 24, 96, 1536);
  }
}

/*
 * Test to ensure get_num_elements works with a 3DS LSTM tensor that requires
 * vertical concatenation.
 */
void get_num_elements_lstm_prev_bidir_weights() {
  uint32_t shape[] = {2, 6, 4};
  test_num_elements_concat(ZDNN_3DS,
                           RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
                           shape, 48, 192, 65536);
}

/*
 * Test to ensure get_num_elements works with a (hidden-)biases 2DS LSTM
 * tensor.
 */
void get_num_elements_lstm_biases() {
  uint32_t shape[] = {2, 3};
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_num_elements_concat(
          ZDNN_2DS, RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j], shape, 6,
          24, 512);
    }
  }
}

/*
 * Test to ensure get_num_elements works with a 3DS GRU tensor that doesn't
 * require vertical concatenation.
 */
void get_num_elements_gru_no_vconcat_weights() {
  uint32_t shape[] = {2, 3, 4};
  for (int i = 0; i < NUM_NO_VCONCAT_INFOS; i++) {
    test_num_elements_concat(ZDNN_3DS, RNN_TYPE_GRU | no_vconcat_infos[i],
                             shape, 24, 72, 1152);
  }
}

/*
 * Test to ensure get_num_elements works with a 3DS GRU tensor that requires
 * vertical concatenation.
 */
void get_num_elements_gru_prev_bidir_weights() {
  uint32_t shape[] = {2, 6, 4};
  test_num_elements_concat(ZDNN_3DS,
                           RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS,
                           shape, 48, 144, 49152);
}

/*
 * Test to ensure get_num_elements works with a (hidden-)biases 2DS GRU
 * tensor.
 */
void get_num_elements_gru_biases() {
  uint32_t shape[] = {2, 3};
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_num_elements_concat(ZDNN_2DS,
                               RNN_TYPE_GRU | prev_layers[i] | biases_usages[j],
                               shape, 6, 18, 384);
    }
  }
}

/*
 * Test to ensure get_num_elements works with an RNN uni output tensor, which
 * the ELEMENTS_AIU result will not have any padding
 */
void get_num_elements_uni_output() {
  uint32_t shape[] = {2, 1, 3, 4};
  test_num_elements(ZDNN_4DS, shape, 24, 24);
}

/*
 * Test to ensure get_num_elements works with an RNN bidir output tensor, which
 * the ELEMENTS_AIU result WILL have paddings
 */
void get_num_elements_bidir_output() {
  uint32_t shape[] = {2, 2, 3, 4};
  test_num_elements(ZDNN_4DS, shape, 48, 768);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_nhwc);
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_4d);
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_3ds);
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_3d);
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_2ds);
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_2d);
  RUN_TEST_ALL_PRE_DATATYPES(get_num_elements_1d);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      get_num_elements_lstm_no_vconcat_weights);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      get_num_elements_lstm_prev_bidir_weights);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(get_num_elements_lstm_biases);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(get_num_elements_gru_no_vconcat_weights);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(get_num_elements_gru_prev_bidir_weights);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(get_num_elements_gru_biases);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(get_num_elements_uni_output);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(get_num_elements_bidir_output);

  return UNITY_END();
}
