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
  Common routine for testing dimension translation
  Transformed dimensions must match the expected
*/
void test_tfrmd_dims(zdnn_data_layouts pre_tfrmd_layout,
                     uint32_t pre_tfrmd_dim4, uint32_t pre_tfrmd_dim3,
                     uint32_t pre_tfrmd_dim2, uint32_t pre_tfrmd_dim1,
                     uint32_t tfrmd_dim4, uint32_t tfrmd_dim3,
                     uint32_t tfrmd_dim2, uint32_t tfrmd_dim1) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_status status;

  switch (pre_tfrmd_layout) {
  case (ZDNN_1D):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, pre_tfrmd_dim1);
    break;
  case (ZDNN_2D):
  case (ZDNN_2DS):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, pre_tfrmd_dim2,
                                   pre_tfrmd_dim1);
    break;
  case (ZDNN_3D):
  case (ZDNN_3DS):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, pre_tfrmd_dim3,
                                   pre_tfrmd_dim2, pre_tfrmd_dim1);
    break;
  default:
    zdnn_init_pre_transformed_desc(
        pre_tfrmd_layout, test_datatype, &pre_tfrmd_desc, pre_tfrmd_dim4,
        pre_tfrmd_dim3, pre_tfrmd_dim2, pre_tfrmd_dim1);
  }

  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() failed (status = %08x)", status);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim4 == tfrmd_dim4,
      "tfrmd_desc.dim4 (%u) doesn't match expected (%u)", tfrmd_desc.dim4,
      tfrmd_dim4);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim3 == tfrmd_dim3,
      "tfrmd_desc.dim3 (%u) doesn't match expected (%u)", tfrmd_desc.dim3,
      tfrmd_dim3);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim2 == tfrmd_dim2,
      "tfrmd_desc.dim2 (%u) doesn't match expected (%u)", tfrmd_desc.dim2,
      tfrmd_dim4);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim1 == tfrmd_dim1,
      "tfrmd_desc.dim1 (%u) doesn't match expected (%u)", tfrmd_desc.dim1,
      tfrmd_dim4);
}

/*
  Common routine for testing dimension translation (concatenated types)
  Transformed dimensions must match the expected
  pre_tfrmd_dim3 is ignored when pre_tfrmd_layout is ZDNN_2DS
*/
void test_tfrmd_concat_dims(zdnn_data_layouts pre_tfrmd_layout,

                            uint32_t pre_tfrmd_dim3, uint32_t pre_tfrmd_dim2,
                            uint32_t pre_tfrmd_dim1,
                            zdnn_ztensor_concat_types concat_type) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_status status;

  uint32_t expected_dim4 = 0, expected_dim3 = 0, expected_dim2 = 0,
           expected_dim1 = 0;

  uint8_t num_concats = 0;
  switch (concat_type) {
  case CONCAT_LSTM:
    num_concats = 4;
    break;
  case CONCAT_GRU:
    num_concats = 3;
    break;
  case CONCAT_BIDIR_OUTPUT:
    num_concats = 2;
    break;
  default:
    TEST_FAIL_MESSAGE("unknown concat_type");
  }

  switch (pre_tfrmd_layout) {
  case (ZDNN_2DS):
    expected_dim4 = pre_tfrmd_dim2;
    expected_dim2 = 1;
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, pre_tfrmd_dim2,
                                   pre_tfrmd_dim1);
    break;
  case (ZDNN_3DS):
    expected_dim4 = pre_tfrmd_dim3;
    expected_dim2 = pre_tfrmd_dim2;
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, test_datatype,
                                   &pre_tfrmd_desc, pre_tfrmd_dim3,
                                   pre_tfrmd_dim2, pre_tfrmd_dim1);
    break;
  default:
    TEST_FAIL_MESSAGE("unknown pre_tfrmd_layout");
    break;
  }
  expected_dim3 = 1;
  expected_dim1 = CEIL(pre_tfrmd_dim1, AIU_2BYTE_CELLS_PER_STICK) *
                  AIU_2BYTE_CELLS_PER_STICK * num_concats;

  status = zdnn_generate_transformed_desc_concatenated(
      &pre_tfrmd_desc, concat_type, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() status is %08x (%s) "
      "but expects %08x (%s))",
      status, zdnn_get_status_message(status), ZDNN_OK,
      zdnn_get_status_message(ZDNN_OK));
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim4 == expected_dim4,
      "tfrmd_desc.dim4 (%u) doesn't match expected (%u)", tfrmd_desc.dim4,
      expected_dim4);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim3 == expected_dim3,
      "tfrmd_desc.dim3 (%u) doesn't match expected (%u)", tfrmd_desc.dim3,
      expected_dim3);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim2 == expected_dim2,
      "tfrmd_desc.dim2 (%u) doesn't match expected (%u)", tfrmd_desc.dim2,
      expected_dim2);
  TEST_ASSERT_MESSAGE_FORMATTED(
      tfrmd_desc.dim1 == expected_dim1,
      "tfrmd_desc.dim1 (%u) doesn't match expected (%u)", tfrmd_desc.dim1,
      expected_dim1);
}

void test_tfrmd_dims_nhwc_1() {
  test_tfrmd_dims(ZDNN_NHWC, 1, 1, 1, 3, 1, 1, 1, 3);
}

void test_tfrmd_dims_nhwc_2() {
  test_tfrmd_dims(ZDNN_NHWC, 4, 3, 2, 7, 4, 3, 2, 7);
}

void test_tfrmd_dims_4d() { test_tfrmd_dims(ZDNN_4D, 2, 3, 2, 3, 2, 3, 2, 3); }

void test_tfrmd_dims_3ds_1() {
  test_tfrmd_dims(ZDNN_3DS, 0, 5, 1, 3, 5, 1, 1, 3);
}

void test_tfrmd_dims_3ds_2() {
  test_tfrmd_dims(ZDNN_3DS, 0, 3, 4, 2, 3, 1, 4, 2);
}

void test_tfrmd_dims_3d() {
  test_tfrmd_dims(ZDNN_3D, 0, 16, 32, 5, 1, 16, 32, 5);
}

void test_tfrmd_dims_2ds() {
  test_tfrmd_dims(ZDNN_2DS, 0, 0, 4, 2, 4, 1, 1, 2);
}

void test_tfrmd_dims_2d() { test_tfrmd_dims(ZDNN_2D, 0, 0, 2, 5, 1, 1, 2, 5); }

void test_tfrmd_dims_1d() { test_tfrmd_dims(ZDNN_1D, 0, 0, 0, 5, 1, 1, 1, 5); }

void test_tfrmd_dims_concat_lstm_2ds() {
  test_tfrmd_concat_dims(ZDNN_2DS, 0, 2, 16, CONCAT_LSTM);
}

void test_tfrmd_dims_concat_lstm_3ds() {
  test_tfrmd_concat_dims(ZDNN_3DS, 2, 15, 72, CONCAT_LSTM);
}

void test_tfrmd_dims_concat_gru_2ds() {
  test_tfrmd_concat_dims(ZDNN_2DS, 0, 2, 16, CONCAT_GRU);
}

void test_tfrmd_dims_concat_gru_3ds() {
  test_tfrmd_concat_dims(ZDNN_3DS, 2, 15, 72, CONCAT_GRU);
}

void test_tfrmd_dims_concat_rnn_bidir_output_3ds() {
  test_tfrmd_concat_dims(ZDNN_3DS, 2, 15, 72, CONCAT_BIDIR_OUTPUT);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_nhwc_1);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_nhwc_2);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_4d);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_3ds_1);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_3ds_2);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_3d);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_2ds);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_2d);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_1d);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_concat_lstm_2ds);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_concat_lstm_3ds);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_concat_gru_2ds);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_concat_gru_3ds);
  RUN_TEST_ALL_DATATYPES(test_tfrmd_dims_concat_rnn_bidir_output_3ds);
  return UNITY_END();
}
