// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
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

void setUp(void) { /* This is run before EACH TEST */
}

void tearDown(void) {}

//=================================================================================================
// tests for get_stick_offset

void test_offset(uint32_t dim4, uint32_t dim3, uint32_t dim2, uint32_t dim1,
                 zdnn_data_layouts layout, const uint64_t correct_offset[]) {
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
}

// offsets for a 1,4,4,1,NHWC
void test_nhwc_1x4x4x1() {
  uint32_t dim4 = 1, dim3 = 4, dim2 = 4, dim1 = 1;
  uint64_t correct_offset[dim3 * dim2 * dim1];
  int correct_offset_template[4] = {0, 128, 256, 384};

  /*
   * Each element is the only one on a stick (C == 1), so that each W takes one
   * stick. There are then only 4 sticks used in a page, each H taking one page.
   */

  for (int h = 0; h < dim3; h++) {
    for (int w = 0; w < dim2; w++) {
      correct_offset[h * dim2 + w] =
          h * AIU_PAGESIZE_IN_BYTES + correct_offset_template[w];
    }
  }

  test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);
}

void test_nhwc_1x2x2x4() {
  uint32_t dim4 = 1, dim3 = 2, dim2 = 2, dim1 = 4;

  /*
   * 16 elements total, with 4 cells used per stick, and 2 sticks per page.
   * At the H boundary (==2) switch to a new page
   */
  uint64_t correct_offset[16] = {0,    2,    4,    6,    128,  130,
                                 132,  134,  4096, 4098, 4100, 4102,
                                 4224, 4226, 4228, 4230};

  test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);
}

// offsets for 1,32,32,3,NHWC
void test_nhwc_1x32x32x3() {
  uint32_t dim4 = 1, dim3 = 32, dim2 = 32, dim1 = 3;

  uint64_t correct_offset[dim3 * dim2 * dim1];

  uint32_t correct_offset_template[3] = {0, 2, 4};
  uint32_t vertical_pages_per_h =
      (dim2 + AIU_STICKS_PER_PAGE - 1) / AIU_STICKS_PER_PAGE;

  uint16_t index = 0;
  for (int hx = 0; hx < dim3; hx++) {
    for (int wx = 0; wx < dim2; wx++) {
      for (int cx = 0; cx < dim1; cx++) {
        correct_offset[index++] =
            correct_offset_template[cx] +
            hx * vertical_pages_per_h * AIU_PAGESIZE_IN_BYTES +
            wx * AIU_BYTES_PER_STICK;
      }
    }
  }

  test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);
}

void test_nhwc_1x4x33x64() {
  uint32_t dim4 = 1, dim3 = 4, dim2 = 33, dim1 = 64;

  uint64_t correct_offset[dim3 * dim2 * dim1];

  uint32_t *template_stick = malloc(sizeof(uint32_t) * 64);
  for (uint32_t i = 0; i < 64; i++)
    template_stick[i] = i * AIU_2BYTE_CELL_SIZE;

  int index = 0;
  for (uint32_t e4x = 0; e4x < dim4; e4x++) {
    for (uint32_t e3x = 0; e3x < dim3; e3x++) {
      uint32_t page_num = e3x * 2;
      for (uint32_t e2x = 0; e2x < dim2; e2x++) {
        uint64_t stickOffset = e2x * AIU_BYTES_PER_STICK;
        for (int e1x = 0; e1x < dim1; e1x++) {
          correct_offset[index] = page_num * AIU_PAGESIZE_IN_BYTES +
                                  stickOffset + template_stick[e1x];
          index++;
        }
      }
    }
  }

  test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);

  free(template_stick);
}

void test_nhwc_1x4x32x65() {
  uint32_t dim4 = 1, dim3 = 4, dim2 = 32, dim1 = 65;

  uint64_t correct_offset[dim3 * dim2 * dim1];

  uint32_t *template_stick =
      malloc(sizeof(uint32_t) * AIU_2BYTE_CELLS_PER_STICK);
  for (uint32_t i = 0; i < AIU_2BYTE_CELLS_PER_STICK; i++)
    template_stick[i] = i * AIU_2BYTE_CELL_SIZE;

  int index = 0;
  for (uint32_t e4x = 0; e4x < dim4; e4x++) {
    for (uint32_t e3x = 0; e3x < dim3; e3x++) {
      for (uint32_t e2x = 0; e2x < dim2; e2x++) {
        for (uint32_t e1x = 0; e1x < dim1; e1x++) {
          // h + floor(c/AIU_2BYTE_CELLS_PER_STICK) * H;
          uint32_t page_num = e3x + (e1x / AIU_2BYTE_CELLS_PER_STICK) * dim3;
          correct_offset[index] =
              page_num * AIU_PAGESIZE_IN_BYTES + e2x * AIU_BYTES_PER_STICK +
              template_stick[e1x % AIU_2BYTE_CELLS_PER_STICK];
          index++;
        }
      }
    }
  }

  test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);

  free(template_stick);
}

void test_nhwc_1x4x33x65() {
  uint32_t dim4 = 1, dim3 = 4, dim2 = 33, dim1 = 65;

  uint64_t correct_offset[dim3 * dim2 * dim1];

  uint32_t *template_stick =
      malloc(sizeof(uint32_t) * AIU_2BYTE_CELLS_PER_STICK);
  for (uint32_t i = 0; i < AIU_2BYTE_CELLS_PER_STICK; i++)
    template_stick[i] = i * AIU_2BYTE_CELL_SIZE;

  int index = 0;
  for (int e4x = 0; e4x < dim4; e4x++) {
    for (int e3x = 0; e3x < dim3; e3x++) {
      for (int e2x = 0; e2x < dim2; e2x++) {
        for (int e1x = 0; e1x < dim1; e1x++) {
          uint32_t page_num =
              e3x * 2 + (e1x / AIU_2BYTE_CELLS_PER_STICK) * dim3 * 2;
          correct_offset[index] =
              page_num * AIU_PAGESIZE_IN_BYTES + e2x * AIU_BYTES_PER_STICK +
              template_stick[e1x % AIU_2BYTE_CELLS_PER_STICK];

          index++;
        }
      }
    }
  }

  test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);

  free(template_stick);
}

#define NHWC_TEST_WITH_FILE(n, h, w, c)                                        \
  void test_nhwc_##n##x##h##x##w##x##c() {                                     \
    uint32_t dim4 = n, dim3 = h, dim2 = w, dim1 = c;                           \
    uint64_t correct_offset[n * h * w * c];                                    \
    TEST_ASSERT_MESSAGE(get_offsets_from_file(OFFSET_FILE(nhwc, n, h, w, c),   \
                                              correct_offset) > 0,             \
                        "get_offsets_from_file() failed");                     \
    test_offset(dim4, dim3, dim2, dim1, ZDNN_NHWC, correct_offset);            \
  }

NHWC_TEST_WITH_FILE(1, 2, 3, 4)
NHWC_TEST_WITH_FILE(1, 1, 31, 64)
NHWC_TEST_WITH_FILE(1, 1, 32, 64)
NHWC_TEST_WITH_FILE(1, 1, 33, 64)
NHWC_TEST_WITH_FILE(1, 1, 32, 63)
NHWC_TEST_WITH_FILE(1, 1, 32, 65)
NHWC_TEST_WITH_FILE(1, 1, 4, 127)
NHWC_TEST_WITH_FILE(1, 1, 4, 128)
NHWC_TEST_WITH_FILE(1, 1, 4, 129)
NHWC_TEST_WITH_FILE(1, 1, 63, 4)
NHWC_TEST_WITH_FILE(1, 1, 64, 4)
NHWC_TEST_WITH_FILE(1, 1, 65, 4)
NHWC_TEST_WITH_FILE(2, 3, 33, 129)

#define NCHW_TEST_WITH_FILE(n, c, h, w)                                        \
  void test_nchw_##n##x##c##x##h##x##w() {                                     \
    uint32_t dim4 = 1, dim3 = c, dim2 = h, dim1 = w;                           \
    uint64_t correct_offset[n * c * h * w];                                    \
    TEST_ASSERT_MESSAGE(get_offsets_from_file(OFFSET_FILE(nchw, n, c, h, w),   \
                                              correct_offset) > 0,             \
                        "get_offsets_from_file() failed");                     \
    test_offset(dim4, dim3, dim2, dim1, ZDNN_NCHW, correct_offset);            \
  }

NCHW_TEST_WITH_FILE(1, 1, 4, 4)
NCHW_TEST_WITH_FILE(1, 4, 2, 3)
NCHW_TEST_WITH_FILE(1, 3, 32, 32)
NCHW_TEST_WITH_FILE(2, 129, 3, 33)
NCHW_TEST_WITH_FILE(1, 64, 1, 31)
NCHW_TEST_WITH_FILE(1, 64, 1, 32)
NCHW_TEST_WITH_FILE(1, 64, 1, 33)
NCHW_TEST_WITH_FILE(1, 63, 1, 32)
NCHW_TEST_WITH_FILE(1, 65, 1, 32)
NCHW_TEST_WITH_FILE(1, 127, 1, 4)
NCHW_TEST_WITH_FILE(1, 128, 1, 4)
NCHW_TEST_WITH_FILE(1, 129, 1, 4)
NCHW_TEST_WITH_FILE(1, 4, 1, 63)
NCHW_TEST_WITH_FILE(1, 4, 1, 64)
NCHW_TEST_WITH_FILE(1, 4, 1, 65)

#define HWCK_TEST_WITH_FILE(h, w, c, k)                                        \
  void test_hwck_##h##x##w##x##c##x##k() {                                     \
    uint32_t dim4 = h, dim3 = w, dim2 = c, dim1 = k;                           \
    uint64_t correct_offset[h * w * c * k];                                    \
    TEST_ASSERT_MESSAGE(get_offsets_from_file(OFFSET_FILE(hwck, h, w, c, k),   \
                                              correct_offset) > 0,             \
                        "get_offsets_from_file() failed");                     \
    test_offset(dim4, dim3, dim2, dim1, ZDNN_HWCK, correct_offset);            \
  }

HWCK_TEST_WITH_FILE(1, 4, 4, 1)
HWCK_TEST_WITH_FILE(1, 2, 3, 4)
HWCK_TEST_WITH_FILE(2, 3, 33, 129)
HWCK_TEST_WITH_FILE(1, 32, 32, 3)
HWCK_TEST_WITH_FILE(1, 1, 32, 63)
HWCK_TEST_WITH_FILE(1, 1, 31, 64)
HWCK_TEST_WITH_FILE(1, 1, 32, 64)
HWCK_TEST_WITH_FILE(1, 1, 33, 64)
HWCK_TEST_WITH_FILE(1, 1, 32, 65)
HWCK_TEST_WITH_FILE(1, 1, 4, 127)
HWCK_TEST_WITH_FILE(1, 1, 4, 128)
HWCK_TEST_WITH_FILE(1, 1, 4, 129)
HWCK_TEST_WITH_FILE(1, 1, 63, 4)
HWCK_TEST_WITH_FILE(1, 1, 64, 4)
HWCK_TEST_WITH_FILE(1, 1, 65, 4)

int main(void) {
  UNITY_BEGIN();
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x2x2x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x32x32x3);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x4x33x64);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x4x32x65);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x4x33x65);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x33x129);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x31x64);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x32x64);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x33x64);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x32x63);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x32x65);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x127);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x128);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x129);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x63x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x64x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x65x4);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x1x4x4);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x4x2x3);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x3x32x32);
  RUN_TEST_ALL_DATATYPES(test_nchw_2x129x3x33);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x63x1x32);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x64x1x31);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x64x1x32);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x64x1x33);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x65x1x32);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x127x1x4);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x128x1x4);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x129x1x4);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x4x1x63);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x4x1x64);
  RUN_TEST_ALL_DATATYPES(test_nchw_1x4x1x65);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x2x3x4);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x32x32x3);
  RUN_TEST_ALL_DATATYPES(test_hwck_2x3x33x129);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x32x63);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x31x64);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x32x64);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x33x64);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x32x65);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x4x127);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x4x128);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x4x129);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x63x4);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x64x4);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x1x65x4);

  return UNITY_END();
}
