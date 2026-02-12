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
 * Tests for save-area size helper API.
 */

void setUp(void) {}
void tearDown(void) {}

void test_save_area_size_softmax_reduce(void) {
  TEST_ASSERT_EQUAL_UINT64(ZDNN_8K_SAVEAREA_SIZE,
                           zdnn_get_save_area_size(NNPA_SOFTMAX));
  TEST_ASSERT_EQUAL_UINT64(ZDNN_8K_SAVEAREA_SIZE,
                           zdnn_get_save_area_size(NNPA_REDUCE));
}

void test_save_area_size_non_savearea_ops_return_zero(void) {
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_save_area_size(NNPA_ADD));
  TEST_ASSERT_EQUAL_UINT64(0, zdnn_get_save_area_size((nnpa_function_code)999));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_save_area_size_softmax_reduce);
  RUN_TEST(test_save_area_size_non_savearea_ops_return_zero);

  return UNITY_END();
}
