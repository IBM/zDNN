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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "testsupport.h"

void setUp(void) {}

void tearDown(void) {}

//=================================================================================================
// tests for get_stick_offset

void test_offset(uint32_t dim4, uint32_t dim3, uint32_t dim2, uint32_t dim1,
                 zdnn_data_layouts layout) {

  zdnn_status status;
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;

  zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc, dim4,
                                 dim3, dim2, dim1);
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_generate_transformed_desc() returned %d \"%s\"",
      status, zdnn_get_status_message(status));
  zdnn_init_ztensor(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  size_t *correct_offset = alloc_offsets(&ztensor);

  uint64_t *offsets_calculated =
      malloc(sizeof(uint64_t) * get_num_elements(&ztensor, ELEMENTS_PRE));

  uint64_t c = 0;

  for (uint32_t e4x = 0; e4x < pre_tfrmd_desc.dim4; e4x++) {
    for (uint32_t e3x = 0; e3x < pre_tfrmd_desc.dim3; e3x++) {
      for (uint32_t e2x = 0; e2x < pre_tfrmd_desc.dim2; e2x++) {
        for (uint32_t e1x = 0; e1x < pre_tfrmd_desc.dim1; e1x++) {
          offsets_calculated[c] =
              get_stick_offset(e4x, e3x, e2x, e1x, &pre_tfrmd_desc);
          TEST_ASSERT_MESSAGE_FORMATTED(
              offsets_calculated[c] == correct_offset[c],
              "element (%d, %d, %d, %d) has wrong offset of %" PRIu64
              ", (expects %" PRIu64 ")",
              e4x, e3x, e2x, e1x, offsets_calculated[c], correct_offset[c]);
          c++;
        }
      }
    }
  }
  free(offsets_calculated);
  free(correct_offset);
}

// offsets for a 1,4,4,1,NHWC
void test_nhwc_1x4x4x1() { test_offset(1, 4, 4, 1, ZDNN_NHWC); }

void test_nhwc_1x2x2x4() { test_offset(1, 2, 2, 4, ZDNN_NHWC); }

// offsets for 1,32,32,3,NHWC
void test_nhwc_1x32x32x3() { test_offset(1, 32, 32, 3, ZDNN_NHWC); }

void test_nhwc_1x4x33x64() { test_offset(1, 4, 33, 64, ZDNN_NHWC); }

void test_nhwc_1x4x32x65() { test_offset(1, 4, 32, 65, ZDNN_NHWC); }

void test_nhwc_1x4x33x65() { test_offset(1, 4, 33, 65, ZDNN_NHWC); }

void test_nhwc_1x2x3x4() { test_offset(1, 2, 3, 4, ZDNN_NHWC); }

void test_nhwc_1x1x31x64() { test_offset(1, 1, 31, 64, ZDNN_NHWC); }
void test_nhwc_1x1x32x64() { test_offset(1, 1, 32, 64, ZDNN_NHWC); }
void test_nhwc_1x1x33x64() { test_offset(1, 1, 33, 64, ZDNN_NHWC); }
void test_nhwc_1x1x32x63() { test_offset(1, 1, 32, 63, ZDNN_NHWC); }
void test_nhwc_1x1x32x65() { test_offset(1, 1, 32, 65, ZDNN_NHWC); }
void test_nhwc_1x1x4x127() { test_offset(1, 1, 4, 127, ZDNN_NHWC); }
void test_nhwc_1x1x4x128() { test_offset(1, 1, 4, 128, ZDNN_NHWC); }
void test_nhwc_1x1x4x129() { test_offset(1, 1, 4, 129, ZDNN_NHWC); }
void test_nhwc_1x1x63x4() { test_offset(1, 1, 63, 4, ZDNN_NHWC); }
void test_nhwc_1x1x64x4() { test_offset(1, 1, 64, 4, ZDNN_NHWC); }
void test_nhwc_1x1x65x4() { test_offset(1, 1, 65, 4, ZDNN_NHWC); }
void test_nhwc_2x3x33x129() { test_offset(2, 3, 33, 129, ZDNN_NHWC); }

void test_nchw_1x1x4x4() { test_offset(1, 1, 4, 4, ZDNN_NCHW); }
void test_nchw_1x4x2x3() { test_offset(1, 4, 2, 3, ZDNN_NCHW); }
void test_nchw_1x3x32x32() { test_offset(1, 3, 32, 32, ZDNN_NCHW); }
void test_nchw_2x129x3x33() { test_offset(2, 129, 3, 33, ZDNN_NCHW); }
void test_nchw_1x64x1x31() { test_offset(1, 64, 1, 31, ZDNN_NCHW); }
void test_nchw_1x64x1x32() { test_offset(1, 64, 1, 32, ZDNN_NCHW); }
void test_nchw_1x64x1x33() { test_offset(1, 64, 1, 33, ZDNN_NCHW); }
void test_nchw_1x63x1x32() { test_offset(1, 63, 1, 32, ZDNN_NCHW); }
void test_nchw_1x65x1x32() { test_offset(1, 65, 1, 32, ZDNN_NCHW); }
void test_nchw_1x127x1x4() { test_offset(1, 127, 1, 4, ZDNN_NCHW); }
void test_nchw_1x128x1x4() { test_offset(1, 128, 1, 4, ZDNN_NCHW); }
void test_nchw_1x129x1x4() { test_offset(1, 129, 1, 4, ZDNN_NCHW); }
void test_nchw_1x4x1x63() { test_offset(1, 4, 1, 63, ZDNN_NCHW); }
void test_nchw_1x4x1x64() { test_offset(1, 4, 1, 64, ZDNN_NCHW); }
void test_nchw_1x4x1x65() { test_offset(1, 4, 1, 65, ZDNN_NCHW); }

void test_hwck_1x4x4x1() { test_offset(1, 4, 4, 1, ZDNN_HWCK); }
void test_hwck_1x2x3x4() { test_offset(1, 2, 3, 4, ZDNN_HWCK); }
void test_hwck_2x3x33x129() { test_offset(2, 3, 33, 129, ZDNN_HWCK); }
void test_hwck_1x32x32x3() { test_offset(1, 32, 32, 3, ZDNN_HWCK); }
void test_hwck_1x1x32x63() { test_offset(1, 1, 32, 63, ZDNN_HWCK); }
void test_hwck_1x1x31x64() { test_offset(1, 1, 31, 64, ZDNN_HWCK); }
void test_hwck_1x1x32x64() { test_offset(1, 1, 32, 64, ZDNN_HWCK); }
void test_hwck_1x1x33x64() { test_offset(1, 1, 33, 64, ZDNN_HWCK); }
void test_hwck_1x1x32x65() { test_offset(1, 1, 32, 65, ZDNN_HWCK); }
void test_hwck_1x1x4x127() { test_offset(1, 1, 4, 127, ZDNN_HWCK); }
void test_hwck_1x1x4x128() { test_offset(1, 1, 4, 128, ZDNN_HWCK); }
void test_hwck_1x1x4x129() { test_offset(1, 1, 4, 129, ZDNN_HWCK); }
void test_hwck_1x1x63x4() { test_offset(1, 1, 63, 4, ZDNN_HWCK); }
void test_hwck_1x1x64x4() { test_offset(1, 1, 64, 4, ZDNN_HWCK); }
void test_hwck_1x1x65x4() { test_offset(1, 1, 65, 4, ZDNN_HWCK); }

int main(void) {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x4x4x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x2x2x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x32x32x3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x4x33x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x4x32x65);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x4x33x65);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_2x3x33x129);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x31x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x32x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x33x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x32x63);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x32x65);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x4x127);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x4x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x4x129);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x63x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x64x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nhwc_1x1x65x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x1x4x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x4x2x3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x3x32x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_2x129x3x33);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x63x1x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x64x1x31);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x64x1x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x64x1x33);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x65x1x32);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x127x1x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x128x1x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x129x1x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x4x1x63);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x4x1x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_nchw_1x4x1x65);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x4x4x1);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x2x3x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x32x32x3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_2x3x33x129);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x32x63);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x31x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x32x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x33x64);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x32x65);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x4x127);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x4x128);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x4x129);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x63x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x64x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_hwck_1x1x65x4);

  return UNITY_END();
}
