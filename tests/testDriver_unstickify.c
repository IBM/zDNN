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
  VERIFY_HW_ENV;
}

void tearDown(void) {}

//=================================================================================================
// tests for unstickify

/*

  Use 1x4x4x1 as example:

  1) Create the input tensor descriptor
  2) Create the raw (i.e., dense) input tensor data with random
     FP16/FP32/BFLOAT values 1 >= x > SMALLEST_RANDOM_FP.
     For 1x4x4x1 we have 16 elements.
  3) Create a zTensor with that.
  4a) If caller wants to use offsets, we'll "stickify" the
     input tensor data by putting things in ztensor.buffer directly:
     stick_area[offsets[n] = fp16_to_dlf16(input_data[n]).
  4b) If NO_OFFSETS, we'll use the official stickify routine.
  5) Send that zTensor to unstickify, result goes to "data_unstickified"
  6) compare the raw input tensor data against that "data_unstickified" array.

  The rationale is since we're using random FP data, if there's something wrong
  with the unstickify routine then it's very unlikely to match 100% with the
  raw input data.

*/

void test_unstickify(uint32_t dim4, uint32_t dim3, uint32_t dim2, uint32_t dim1,
                     zdnn_data_layouts layout, offset_mode offset_mode,
                     const char *path) {

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  void *data, *data_unstickified;

  switch (layout) {
  case (ZDNN_1D):
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc,
                                   dim1);
    break;
  case (ZDNN_2D):
  case (ZDNN_2DS):
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc, dim2,
                                   dim1);
    break;
  case (ZDNN_3D):
  case (ZDNN_3DS):
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc, dim3,
                                   dim2, dim1);
    break;
  default:
    zdnn_init_pre_transformed_desc(layout, test_datatype, &pre_tfrmd_desc, dim4,
                                   dim3, dim2, dim1);
  }

  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() failed (status = %08x)", status);

  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc() failed (status = %08x)", status);

  uint64_t num_elements = get_num_elements(&ztensor, ELEMENTS_PRE);
  data = create_and_fill_random_fp_data(&ztensor);
  data_unstickified =
      malloc(num_elements * get_data_type_size(pre_tfrmd_desc.type));

  if (offset_mode == NO_OFFSETS) {
    // Stickify tensor using the official API
    status = zdnn_transform_ztensor(&ztensor, data);
    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_OK, "zdnn_transform_ztensor failed (status = %08x)",
        status);
  } else {
    // "stickify" by converting input values to DLFLOAT16s and writing directly
    // to the ztensor's buffer.
    size_t *offsets;

    if (layout != ZDNN_4DS) {
      offsets = alloc_offsets(&ztensor, offset_mode, path);
    } else {
      offsets = alloc_rnn_output_offsets(&ztensor);
    }

    for (uint64_t i = 0; i < num_elements; i++) {
      uint16_t stickified_input_value = 0;

      switch (test_datatype) {
      case BFLOAT:
        stickified_input_value = cnvt_1_bfloat_to_dlf16(((uint16_t *)data)[i]);
        break;
      case FP16:
        stickified_input_value = cnvt_1_fp16_to_dlf16(((uint16_t *)data)[i]);
        break;
      case FP32:
        stickified_input_value = cnvt_1_fp32_to_dlf16(((float *)data)[i]);
        break;
      default:
        TEST_FAIL_MESSAGE("Unsupported data type");
        free(data_unstickified);
        return;
      }

      // offsets[i] is in # of bytes
      // ztensor.buffer is void*
      // stickified_input_value is uint16_t
      *(uint16_t *)((uintptr_t)(ztensor.buffer) + offsets[i]) =
          stickified_input_value;
    }
    free(offsets);
    // hack, since we never actually stickified anything
    ztensor.is_transformed = true;
  }

  status = zdnn_transform_origtensor(&ztensor, data_unstickified);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_origtensor failed (status = %08x)",
      status);

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    dumpdata_origtensor(&pre_tfrmd_desc, data, AS_FLOAT);
    dumpdata_ztensor(&ztensor, AS_FLOAT, false);
    dumpdata_origtensor(&pre_tfrmd_desc, data_unstickified, AS_FLOAT);
  }

  char *error_fmt = "Incorrect value at element %" PRIu64 ": Unstickified: "
                    "%.6f, Expected: %.6f";

  // the zdnn_transform_origtensor() values went through a
  // FP16/32/BFLOAT16 -> DLFLOAT16 -> FP16/32/BFLOAT16 roundtrip, so we can't
  // just compare them with like a memcmp() because we could have lost precision
  // during the process
  for (uint64_t i = 0; i < num_elements; i++) {

    switch (test_datatype) {
    case BFLOAT: {
      // raw tensor value, this is the "expected" value
      uint16_t data_val = ((uint16_t *)data)[i];

      // BFLOAT -> DLFLOAT16 -> BFLOAT roundtrip'd tensor
      // value
      uint16_t data_unstickified_val = ((uint16_t *)data_unstickified)[i];

      TEST_ASSERT_MESSAGE_FORMATTED(
          almost_equal_bfloat(data_unstickified_val, data_val), error_fmt, i,
          cnvt_1_bfloat_to_fp32(data_unstickified_val),
          cnvt_1_bfloat_to_fp32(data_val));
      break;
    }
    case FP16: {
      // raw tensor value
      uint16_t data_val = ((uint16_t *)data)[i];

      // FP16 -> DLFLOAT16 -> FP16 roundtrip'd tensor value
      uint16_t data_unstickified_val = ((uint16_t *)data_unstickified)[i];

      TEST_ASSERT_MESSAGE_FORMATTED(
          almost_equal_fp16(data_unstickified_val, data_val), error_fmt, i,
          cnvt_1_fp16_to_fp32(data_unstickified_val),
          cnvt_1_fp16_to_fp32(data_val));
      break;
    }
    case FP32: {
      // raw tensor value
      float data_val = ((float *)data)[i];

      // FP32 -> DLFLOAT16 -> FP32 roundtrip'd tensor value
      float data_unstickified_val = ((float *)data_unstickified)[i];

      TEST_ASSERT_MESSAGE_FORMATTED(
          almost_equal_float(data_unstickified_val, data_val), error_fmt, i,
          data_unstickified_val, data_val);
      break;
    }
    default:
      TEST_FAIL_MESSAGE("Unsupported data type");
      return;
    }
  }

  free(data);
  free(data_unstickified);
  zdnn_free_ztensor_buffer(&ztensor);
}

/**************************************************************
 * NHWC
 **************************************************************/

#define NHWC_TEST_BASIC(n, h, w, c)                                            \
  void test_nhwc_##n##x##h##x##w##x##c() {                                     \
    test_unstickify(n, h, w, c, ZDNN_NHWC, QUICK_OFFSETS, NULL);               \
  }

/*
 * Tensor with 16 entries, NHWC
 * 1,4,4,1 NHWC will use one cell per stick, 4 sticks per page and a total of
 * 4 pages.
 */
NHWC_TEST_BASIC(1, 4, 4, 1);

NHWC_TEST_BASIC(1, 4, 4, 2);

/*
 * Tensor with 1024 entries, NHWC
 * 1,32,32,1 NHWC will use 1 cell per stick, all sticks in the page,
 * and 32 pages.
 */
NHWC_TEST_BASIC(1, 32, 32, 1);

NHWC_TEST_BASIC(1, 32, 32, 2);
NHWC_TEST_BASIC(1, 32, 32, 3);

NHWC_TEST_BASIC(1, 1, 2, 1);
NHWC_TEST_BASIC(1, 1, 2, 2);
NHWC_TEST_BASIC(1, 1, 2, 4);
NHWC_TEST_BASIC(1, 1, 2, 7);
NHWC_TEST_BASIC(1, 1, 4, 1);
NHWC_TEST_BASIC(1, 1, 4, 2);
NHWC_TEST_BASIC(1, 1, 4, 4);
NHWC_TEST_BASIC(1, 1, 4, 7);
NHWC_TEST_BASIC(1, 1, 7, 1);
NHWC_TEST_BASIC(1, 1, 7, 2);
NHWC_TEST_BASIC(1, 1, 7, 4);
NHWC_TEST_BASIC(1, 1, 7, 7);
NHWC_TEST_BASIC(1, 1, 8, 1);
NHWC_TEST_BASIC(1, 1, 8, 2);
NHWC_TEST_BASIC(1, 1, 8, 4);
NHWC_TEST_BASIC(1, 1, 8, 7);
NHWC_TEST_BASIC(1, 1, 13, 1);
NHWC_TEST_BASIC(1, 1, 13, 2);
NHWC_TEST_BASIC(1, 1, 13, 4);
NHWC_TEST_BASIC(1, 1, 13, 7);
NHWC_TEST_BASIC(1, 1, 100, 1);
NHWC_TEST_BASIC(1, 1, 100, 2);
NHWC_TEST_BASIC(1, 1, 100, 4);
NHWC_TEST_BASIC(1, 1, 100, 7);

NHWC_TEST_BASIC(2, 3, 2, 1);
NHWC_TEST_BASIC(2, 3, 2, 2);
NHWC_TEST_BASIC(2, 3, 2, 4);
NHWC_TEST_BASIC(2, 3, 2, 7);
NHWC_TEST_BASIC(2, 3, 4, 1);
NHWC_TEST_BASIC(2, 3, 4, 2);
NHWC_TEST_BASIC(2, 3, 4, 4);
NHWC_TEST_BASIC(2, 3, 4, 7);
NHWC_TEST_BASIC(2, 3, 7, 1);
NHWC_TEST_BASIC(2, 3, 7, 2);
NHWC_TEST_BASIC(2, 3, 7, 4);
NHWC_TEST_BASIC(2, 3, 7, 7);
NHWC_TEST_BASIC(2, 3, 8, 1);
NHWC_TEST_BASIC(2, 3, 8, 2);
NHWC_TEST_BASIC(2, 3, 8, 4);
NHWC_TEST_BASIC(2, 3, 8, 7);
NHWC_TEST_BASIC(2, 3, 13, 1);
NHWC_TEST_BASIC(2, 3, 13, 2);
NHWC_TEST_BASIC(2, 3, 13, 4);
NHWC_TEST_BASIC(2, 3, 13, 7);
NHWC_TEST_BASIC(2, 3, 100, 1);
NHWC_TEST_BASIC(2, 3, 100, 2);
NHWC_TEST_BASIC(2, 3, 100, 4);
NHWC_TEST_BASIC(2, 3, 100, 7);

NHWC_TEST_BASIC(3, 2, 2, 1);
NHWC_TEST_BASIC(3, 2, 2, 2);
NHWC_TEST_BASIC(3, 2, 2, 4);
NHWC_TEST_BASIC(3, 2, 2, 7);
NHWC_TEST_BASIC(3, 2, 4, 1);
NHWC_TEST_BASIC(3, 2, 4, 2);
NHWC_TEST_BASIC(3, 2, 4, 4);
NHWC_TEST_BASIC(3, 2, 4, 7);
NHWC_TEST_BASIC(3, 2, 7, 1);
NHWC_TEST_BASIC(3, 2, 7, 2);
NHWC_TEST_BASIC(3, 2, 7, 4);
NHWC_TEST_BASIC(3, 2, 7, 7);
NHWC_TEST_BASIC(3, 2, 8, 1);
NHWC_TEST_BASIC(3, 2, 8, 2);
NHWC_TEST_BASIC(3, 2, 8, 4);
NHWC_TEST_BASIC(3, 2, 8, 7);
NHWC_TEST_BASIC(3, 2, 13, 1);
NHWC_TEST_BASIC(3, 2, 13, 2);
NHWC_TEST_BASIC(3, 2, 13, 4);
NHWC_TEST_BASIC(3, 2, 13, 7);
NHWC_TEST_BASIC(3, 2, 100, 1);
NHWC_TEST_BASIC(3, 2, 100, 2);
NHWC_TEST_BASIC(3, 2, 100, 4);
NHWC_TEST_BASIC(3, 2, 100, 7);

void test_nhwc_1x1x1xe1(int e1) {
  test_unstickify(1, 1, 1, e1, ZDNN_NHWC, QUICK_OFFSETS, NULL);
}

void test_nhwc_1x1x1x4() { test_nhwc_1x1x1xe1(4); }
void test_nhwc_1x1x1x5() { test_nhwc_1x1x1xe1(5); }
void test_nhwc_1x1x1x8() { test_nhwc_1x1x1xe1(8); }
void test_nhwc_1x1x1x9() { test_nhwc_1x1x1xe1(9); }
void test_nhwc_1x1x1x63() { test_nhwc_1x1x1xe1(63); }
void test_nhwc_1x1x1x64() { test_nhwc_1x1x1xe1(64); }
void test_nhwc_1x1x1x65() { test_nhwc_1x1x1xe1(65); }
void test_nhwc_1x1x1x127() { test_nhwc_1x1x1xe1(127); }
void test_nhwc_1x1x1x128() { test_nhwc_1x1x1xe1(128); }

/*
 * Tensor with 16 entries, 3DS
 * 4,4,1 3DS will use one cell per stick, 4 sticks per page and a total of 4
 * pages.
 */
void test_3ds_4x4x1() {
  // first entry doesn't matter
  test_unstickify(9999, 4, 4, 1, ZDNN_3DS, QUICK_OFFSETS, NULL);
}

/*
 * Tensor with 3072 entries, 3DS
 * 32,32,3 3DS will use 3 cells per stick, all sticks in the page,
 * and 32 pages.
 */
void test_3ds_32x32x3() {
  // first entry doesn't matter
  test_unstickify(9999, 32, 32, 3, ZDNN_3DS, QUICK_OFFSETS, NULL);
}

/*
 * Tensor with 8 entries, 2DS
 * 4,2 2DS will use two cells per stick, (implied 1 stick per page) and a
 * total of 4 pages.
 */
void test_2ds_4x2() {
  // first two entries don't matter in 2DS
  test_unstickify(9999, 9999, 4, 2, ZDNN_2DS, QUICK_OFFSETS, NULL);
}

/*
 * Tensor with 4k entries, 2DS
 * We expect this to require 4 pages total. Each dim2 will require 2 pages.
 * The first page will have all 64 cells of all 32 sticks filled holding 2048
 * values. A second page will have 1 stick with 1 cell filled to hold val
 * 2049.
 */
void test_2ds_2x2049() {
  // first two entries don't matter in 2DS
  test_unstickify(9999, 9999, 2, 2049, ZDNN_2DS, QUICK_OFFSETS, NULL);
}

/**************************************************************
 * NCHW
 **************************************************************/

#define NCHW_TEST_WITH_FILE(n, c, h, w)                                        \
  void test_nchw_##n##x##c##x##h##x##w() {                                     \
    test_unstickify(n, c, h, w, ZDNN_NCHW, FILE_OFFSETS,                       \
                    OFFSET_FILE(nchw, n, c, h, w));                            \
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

/**************************************************************
 * RNN OUTPUT
 **************************************************************/

#define RNN_OUT_TEST(d4, d3, d2, d1)                                           \
  void test_rnn_output_##d4##x##d3##x##d2##x##d1() {                           \
    test_unstickify(d4, d3, d2, d1, ZDNN_4DS, QUICK_OFFSETS, NULL);            \
  }

RNN_OUT_TEST(5, 1, 4, 3)
RNN_OUT_TEST(1, 1, 4, 3)
RNN_OUT_TEST(5, 1, 4, 64)
RNN_OUT_TEST(1, 1, 4, 64)
RNN_OUT_TEST(5, 1, 4, 65)
RNN_OUT_TEST(1, 1, 4, 65)
RNN_OUT_TEST(5, 1, 31, 5)
RNN_OUT_TEST(1, 1, 31, 5)
RNN_OUT_TEST(5, 1, 60, 5)
RNN_OUT_TEST(1, 1, 60, 5)
RNN_OUT_TEST(5, 2, 4, 3)
RNN_OUT_TEST(1, 2, 4, 3)
RNN_OUT_TEST(5, 2, 4, 64)
RNN_OUT_TEST(1, 2, 4, 64)
RNN_OUT_TEST(5, 2, 4, 65)
RNN_OUT_TEST(1, 2, 4, 65)
RNN_OUT_TEST(5, 2, 31, 5)
RNN_OUT_TEST(1, 2, 31, 5)
RNN_OUT_TEST(5, 2, 60, 5)
RNN_OUT_TEST(1, 2, 60, 5)

void test_unstickify_4dfeature_twice() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &pre_tfrmd_desc, 1,
                                 4, 4, 1);

  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() failed (status = %08x)", status);

  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc() failed (status = %08x)", status);

  unsigned char *data_unstickified =
      malloc(get_num_elements(&ztensor, ELEMENTS_PRE) *
             get_data_type_size(pre_tfrmd_desc.type));

  ztensor.is_transformed = true; // hack, since we never actually
                                 // stickified anything
  status = zdnn_transform_origtensor(&ztensor, data_unstickified);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "First unstickify: expected status = %08x, actual status = %08x", ZDNN_OK,
      status);

  // second one should still be OK
  status = zdnn_transform_origtensor(&ztensor, data_unstickified);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "Second unstickify: expected status = %08x, actual status = %08x",
      ZDNN_OK, status);
}

void test_stickify_unstickify(uint32_t dim4, uint32_t dim3, uint32_t dim2,
                              uint32_t dim1, zdnn_data_layouts layout) {
  test_unstickify(dim4, dim3, dim2, dim1, layout, NO_OFFSETS, NULL);
}

/*
 * Tensor with 16 entries, NHWC
 * 1,4,4,1 NHWC will use one cell per stick, 4 sticks per page and a total of
 * 4 pages.
 */
//
void test_stickify_unstickify_nhwc_1x4x4x1() {
  test_stickify_unstickify(1, 4, 4, 1, ZDNN_NHWC);
}

void test_stickify_unstickify_nhwc_1x4x4x2() {
  test_stickify_unstickify(1, 4, 4, 2, ZDNN_NHWC);
}

/*
 * Tensor with 3072 entries, NHWC
 * 1,32,32,1 NHWC will use 1 cell per stick, all sticks in the page,
 * and 32 pages.
 */
//
void test_stickify_unstickify_nhwc_1x32x32x1() {
  test_stickify_unstickify(1, 32, 32, 1, ZDNN_NHWC);
}

void test_stickify_unstickify_nhwc_1x32x32x2() {
  test_stickify_unstickify(1, 32, 32, 2, ZDNN_NHWC);
}

void test_stickify_unstickify_nhwc_1x32x32x3() {
  test_stickify_unstickify(1, 32, 32, 3, ZDNN_NHWC);
}

void test_stickify_unstickify_nhwc_1x2x33x65() {
  test_stickify_unstickify(1, 2, 33, 65, ZDNN_NHWC);
}

void test_stickify_unstickify_nchw_1x4x4x1() {
  test_stickify_unstickify(1, 4, 4, 1, ZDNN_NCHW);
}

void test_stickify_unstickify_nchw_1x32x32x3() {
  test_stickify_unstickify(1, 32, 32, 3, ZDNN_NCHW);
}

void test_stickify_unstickify_nchw_1x2x33x65() {
  test_stickify_unstickify(1, 2, 33, 65, ZDNN_NCHW);
}

// This routine tests the conversion from DLF to FP16.
// Input: a "bad" value in DLFloat, which will "trip" the
//        floating point exception trigger on VCFN
void test_ztensor_bad_value_FP16(uint16_t bad_value) {
#define TOO_LARGE_DLF16_POS 0x7E00
#define TOO_LARGE_DLF16_NEG 0xFE00
#define TOO_SMALL_DLF16_POS 0x0001
#define TOO_SMALL_DLF16_NEG 0x8001
  // Note:  Ninf = "NaN or INF"
#define NINF_DLF16_POS 0x7FFF
#define NINF_DLF16_NEG 0xFFFF

#define STICK_ENTRIES_FP16 7

  uint32_t stick_entries_to_try[STICK_ENTRIES_FP16] = {0, 1, 7, 8, 9, 62, 63};

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;
  zdnn_status status;

  uint16_t *array; // Alternate view on the stickified_data (ztensor.buffer)
  unsigned char *unstickified_data;

  // Build a transformed ztensor with valid data
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 1, 1, 64);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);

  // Transform the data to an is_stickified ztensor, so we can test
  // unstickification later
  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(status == ZDNN_OK,
                                "zdnn_transform_ztensor failed (status = %08x)",
                                status);

  // Create an area to unstickify/convert back to
  uint64_t num_elements = get_num_elements(&ztensor, ELEMENTS_PRE);
  zdnn_data_types dtype = ztensor.pre_transformed_desc->type;
  unstickified_data = malloc(num_elements * get_data_type_size(dtype));
  array = (uint16_t *)ztensor.buffer; /* use stickified_data as an array */

  for (int i = 0; i < STICK_ENTRIES_FP16; i++) {
    array[stick_entries_to_try[i]] = bad_value;
    status = zdnn_transform_origtensor(&ztensor, unstickified_data);

    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_CONVERT_FAILURE,
        "zdnn_transform_origtensor() succeeded (status = %08x, expects = "
        "%08x, i = %d, value = %04x)",
        status, ZDNN_CONVERT_FAILURE, i, bad_value);

    array[stick_entries_to_try[i]] = 0; // set entry to 0 for next iteration
  }
  // Free allocated storage
  free(data);
  free(unstickified_data);
  zdnn_free_ztensor_buffer(&ztensor);
}

// Test unstickify conversions DLFloat to FP16 (VCFN)
void test_ztensor_fp16_bad_values() {

#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE("needs NNPA to trigger overflow");
#endif

  test_ztensor_bad_value_FP16(
      TOO_LARGE_DLF16_POS); // is not a number, will cause overflow
  test_ztensor_bad_value_FP16(
      TOO_LARGE_DLF16_NEG); // is not a number, will cause overflow
  // TODO:
  // The following look valid in the documentation, but do not happen on test
  // system at this time
  //  test_ztensor_bad_value_FP16(
  //      TOO_SMALL_DLF16_POS); // is not a number, will cause overflow
  //  test_ztensor_bad_value_FP16(
  //      TOO_SMALL_DLF16_NEG); // is not a number, will cause overflow
  test_ztensor_bad_value_FP16(
      NINF_DLF16_POS); // is not a number, will cause invalid op
  test_ztensor_bad_value_FP16(NINF_DLF16_NEG); // is not a number, will cause
                                               // invalid op
}

// This routine tests the conversion from DLF to FP32.
// Input: a "bad" value in DLFloat, which will "trip" the
//        floating point exception trigger on VCLFNH/VCLFNL
// NOTE:  Only Not-A-Number values will trip the exception.
//        "Anything DLFLOAT16 can represent, FP32 can do better." -TinTo

void test_ztensor_bad_value_FP32(uint16_t bad_value) {
#define NAN_DL16_POS 0x7FFF
#define NAN_DL16_NEG 0xFFFF
#define STICK_ENTRIES_FP32 9

  uint32_t stick_entries_to_try[STICK_ENTRIES_FP32] = {0, 1, 3,  4, 7,
                                                       8, 9, 15, 63};

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;
  uint16_t *array;
  zdnn_status status;

  unsigned char *unstickified_data;

  // Build a transformed ztensor with valid data
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, 1, 1, 1, 64);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);

  // Transform the data to an stickified ztensor, so we can test
  // unstickification later
  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(status == ZDNN_OK,
                                "zdnn_transform_ztensor failed (status = %08x)",
                                status);

  // Create an area to unstickify/convert back to
  uint64_t num_elements = get_num_elements(&ztensor, ELEMENTS_PRE);
  zdnn_data_types dtype = ztensor.pre_transformed_desc->type;
  unstickified_data = malloc(num_elements * get_data_type_size(dtype));
  array = (uint16_t *)ztensor.buffer; /* use stickified_data as an array */

  for (int i = 0; i < STICK_ENTRIES_FP32; i++) {
    array[stick_entries_to_try[i]] = bad_value;

    status = zdnn_transform_origtensor(&ztensor, unstickified_data);
    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_CONVERT_FAILURE,
        "zdnn_transform_origtensor() succeeded (status = %08x, expects = "
        "%08x, i = %d, value = %04x)",
        status, ZDNN_CONVERT_FAILURE, i, bad_value);

    array[stick_entries_to_try[i]] = 0; // set entry to 0 for next iteration
  }
  // Free allocated storage
  free(data);
  free(unstickified_data);
  zdnn_free_ztensor_buffer(&ztensor);
}

// Test unstickify conversions DLFloat to FP32 (VCLFNx
void test_ztensor_fp32_bad_values() {

#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE("needs NNPA to trigger overflow");
#endif

  // too large or too small not possible,

  test_ztensor_bad_value_FP32(
      NAN_DL16_POS); // is not a number, will cause overflow
  test_ztensor_bad_value_FP32(
      NAN_DL16_NEG); // is not a number, will cause overflow
}

// Test unstickify invalid transform type
void test_unstickify_transform_desc_invalid_type() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  unsigned char *unstickified_data;

  // Create descriptors and ztensor
  // For test, pre_transformed desc must be valid. All other transformed desc
  // options must be valid. Type will be changed.
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, 1, 1, 1, 64);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  // Allocate storage for unstickified data. Although not required for test, if
  // expected status doesn't occur, this space may be touched and would require
  // to be allocated or it may blow up.
  uint64_t num_elements = get_num_elements(&ztensor, ELEMENTS_PRE);
  unstickified_data =
      malloc(num_elements * get_data_type_size(ztensor.transformed_desc->type));

  // Set is_transformed to true as this check occurs prior to type check
  ztensor.is_transformed = true;

  // Update type to an invalid type.
  ztensor.transformed_desc->type = test_datatype;

  status = zdnn_transform_origtensor(&ztensor, unstickified_data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_INVALID_TYPE,
      "zdnn_transform_origtensor() unexpected status (status = %08x, "
      "expects = %08x)",
      status, ZDNN_INVALID_TYPE);

  free(unstickified_data);
  zdnn_free_ztensor_buffer(&ztensor);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST_ALL_DATATYPES(test_nhwc_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x4x4x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x32x32x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x32x32x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x32x32x3);

  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x2x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x2x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x2x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x2x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x4x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x7x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x7x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x7x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x7x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x8x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x8x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x8x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x8x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x13x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x13x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x13x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x13x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x100x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x100x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x100x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x100x7);

  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x2x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x2x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x2x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x2x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x4x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x4x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x4x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x4x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x7x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x7x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x7x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x7x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x8x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x8x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x8x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x8x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x13x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x13x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x13x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x13x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x100x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x100x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x100x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x100x7);

  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x2x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x2x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x2x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x2x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x4x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x4x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x4x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x4x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x7x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x7x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x7x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x7x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x8x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x8x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x8x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x8x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x13x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x13x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x13x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x13x7);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x100x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x100x2);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x100x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_3x2x100x7);

  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x4);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x5);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x8);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x9);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x63);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x64);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x65);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x127);
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x1x1x128);

  RUN_TEST_ALL_DATATYPES(test_3ds_4x4x1);
  RUN_TEST_ALL_DATATYPES(test_3ds_32x32x3);

  RUN_TEST_ALL_DATATYPES(test_2ds_4x2);
  RUN_TEST_ALL_DATATYPES(test_2ds_2x2049);

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

  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x1x4x3);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x1x4x3);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x1x4x64);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x1x4x64);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x1x4x65);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x1x4x65);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x1x31x5);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x1x31x5);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x1x60x5);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x1x60x5);

  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x2x4x3);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x2x4x3);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x2x4x64);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x2x4x64);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x2x4x65);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x2x4x65);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x2x31x5);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x2x31x5);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_5x2x60x5);
  RUN_TEST_ALL_DATATYPES(test_rnn_output_1x2x60x5);

  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nhwc_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nhwc_1x4x4x2);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nhwc_1x32x32x1);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nhwc_1x32x32x2);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nhwc_1x32x32x3);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nhwc_1x2x33x65);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nchw_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nchw_1x32x32x3);
  RUN_TEST_ALL_DATATYPES(test_stickify_unstickify_nchw_1x2x33x65);

  RUN_TEST_ALL_DATATYPES(test_unstickify_4dfeature_twice);

  RUN_TEST_ALL_DATATYPES(test_unstickify_transform_desc_invalid_type);

  RUN_TEST(test_ztensor_fp16_bad_values);
  RUN_TEST(test_ztensor_fp32_bad_values);

  return UNITY_END();
}
