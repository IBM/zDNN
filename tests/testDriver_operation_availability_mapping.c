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
 * Regression tests for operation-availability mapping.
 *
 * These tests avoid hardware dependency by setting nnpa_query_result directly
 * and asserting query_nnpa_op() selects the expected parmblock/function pairs.
 */

static void set_bit_128(bit128_t *field, uint8_t bit_pos) {
  if (bit_pos < 64) {
    field->bits_0to63 |= (uint64_t)1 << ((uint8_t)(63 - bit_pos));
  } else if (bit_pos < 128) {
    field->bits_64to127 |= (uint64_t)1 << ((uint8_t)(63 - (bit_pos - 64)));
  }
}

static void set_bit_256(bit256_t *field, uint16_t bit_pos) {
  if (bit_pos < 64) {
    field->bits_0to63 |= (uint64_t)1 << ((uint16_t)(63 - bit_pos));
  } else if (bit_pos < 128) {
    field->bits_64to127 |= (uint64_t)1 << ((uint16_t)(63 - (bit_pos - 64)));
  } else if (bit_pos < 192) {
    field->bits_128to191 |= (uint64_t)1 << ((uint16_t)(63 - (bit_pos - 128)));
  } else if (bit_pos < 256) {
    field->bits_192to255 |= (uint64_t)1 << ((uint16_t)(63 - (bit_pos - 192)));
  }
}

void setUp(void) { memset(&nnpa_query_result, 0, sizeof(nnpa_query_result)); }

void tearDown(void) {}

void test_conv2d_mapping_requires_parmblkformat_0(void) {
  set_bit_256(&nnpa_query_result.installed_functions_vector, NNPA_CONVOLUTION);

  set_bit_128(&nnpa_query_result.installed_parameter_block_formats,
              NNPA_PARMBLKFORMAT_0);
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_CONV2D));

  memset(&nnpa_query_result.installed_parameter_block_formats, 0,
         sizeof(nnpa_query_result.installed_parameter_block_formats));
  set_bit_128(&nnpa_query_result.installed_parameter_block_formats,
              NNPA_PARMBLKFORMAT_1);
  TEST_ASSERT_FALSE(query_nnpa_op(ZDNN_CONV2D));
}

void test_relu_mapping_distinguishes_parmblock_versions(void) {
  set_bit_256(&nnpa_query_result.installed_functions_vector, NNPA_RELU);

  set_bit_128(&nnpa_query_result.installed_parameter_block_formats,
              NNPA_PARMBLKFORMAT_0);
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_RELU));
  TEST_ASSERT_FALSE(query_nnpa_op(ZDNN_LEAKY_RELU));

  set_bit_128(&nnpa_query_result.installed_parameter_block_formats,
              NNPA_PARMBLKFORMAT_1);
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_RELU));
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_LEAKY_RELU));
}

void test_softmax_mapping_distinguishes_parmblock_versions(void) {
  set_bit_256(&nnpa_query_result.installed_functions_vector, NNPA_SOFTMAX);

  set_bit_128(&nnpa_query_result.installed_parameter_block_formats,
              NNPA_PARMBLKFORMAT_0);
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_SOFTMAX));
  TEST_ASSERT_FALSE(query_nnpa_op(ZDNN_SOFTMAX_MASK));

  set_bit_128(&nnpa_query_result.installed_parameter_block_formats,
              NNPA_PARMBLKFORMAT_1);
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_SOFTMAX));
  TEST_ASSERT_TRUE(query_nnpa_op(ZDNN_SOFTMAX_MASK));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_conv2d_mapping_requires_parmblkformat_0);
  RUN_TEST(test_relu_mapping_distinguishes_parmblock_versions);
  RUN_TEST(test_softmax_mapping_distinguishes_parmblock_versions);

  return UNITY_END();
}
