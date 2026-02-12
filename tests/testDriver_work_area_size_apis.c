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

/*
 * Tests for RNN work-area size helper APIs.
 *
 * These helpers codify the README formulas:
 * - LSTM: dim4 = (4 * num_timesteps) + 6
 * - GRU:  dim4 = (3 * num_timesteps) + 5
 * and multiply by 2 for BIDIR.
 */

void setUp(void) {}
void tearDown(void) {}

static uint64_t expected_size(uint32_t dim4, uint32_t num_batches,
                              uint32_t num_hidden,
                              lstm_gru_direction direction) {
  zdnn_tensor_desc desc;
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE, &desc,
                        dim4, 1, num_batches, num_hidden);
  uint64_t size = zdnn_getsize_ztensor(&desc);
  if (direction == BIDIR) {
    size *= 2;
  }
  return size;
}

void test_lstm_work_area_size_matches_formula(void) {
  uint32_t ts = 7, b = 4, h = 64;
  uint32_t dim4 = (4 * ts) + 6;

  TEST_ASSERT_EQUAL_UINT64(expected_size(dim4, b, h, FWD),
                           zdnn_get_lstm_work_area_size(ts, b, h, FWD));
  TEST_ASSERT_EQUAL_UINT64(expected_size(dim4, b, h, BWD),
                           zdnn_get_lstm_work_area_size(ts, b, h, BWD));
  TEST_ASSERT_EQUAL_UINT64(expected_size(dim4, b, h, BIDIR),
                           zdnn_get_lstm_work_area_size(ts, b, h, BIDIR));
}

void test_gru_work_area_size_matches_formula(void) {
  uint32_t ts = 7, b = 4, h = 64;
  uint32_t dim4 = (3 * ts) + 5;

  TEST_ASSERT_EQUAL_UINT64(expected_size(dim4, b, h, FWD),
                           zdnn_get_gru_work_area_size(ts, b, h, FWD));
  TEST_ASSERT_EQUAL_UINT64(expected_size(dim4, b, h, BWD),
                           zdnn_get_gru_work_area_size(ts, b, h, BWD));
  TEST_ASSERT_EQUAL_UINT64(expected_size(dim4, b, h, BIDIR),
                           zdnn_get_gru_work_area_size(ts, b, h, BIDIR));
}

void test_invalid_inputs_return_zero(void) {
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_lstm_work_area_size(0, 1, 1, FWD));
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_lstm_work_area_size(1, 0, 1, FWD));
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_lstm_work_area_size(1, 1, 0, FWD));
  TEST_ASSERT_EQUAL_UINT64(0,
                           zdnn_get_lstm_work_area_size(1, 1, 1,
                                                        (lstm_gru_direction)99));

  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_gru_work_area_size(0, 1, 1, FWD));
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_gru_work_area_size(1, 0, 1, FWD));
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_gru_work_area_size(1, 1, 0, FWD));
  TEST_ASSERT_EQUAL_UINT64(0,
                           zdnn_get_gru_work_area_size(1, 1, 1,
                                                       (lstm_gru_direction)99));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_lstm_work_area_size_matches_formula);
  RUN_TEST(test_gru_work_area_size_matches_formula);
  RUN_TEST(test_invalid_inputs_return_zero);

  return UNITY_END();
}
