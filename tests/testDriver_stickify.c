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
// tests for stickify

void test_stickify(uint32_t dim4, uint32_t dim3, uint32_t dim2, uint32_t dim1,
                   zdnn_data_layouts layout, offset_mode offset_mode,
                   const char *path) {

  /*
    Use 1x4x4x1 as example:

    1) Create the input tensor descriptor
    2) Create the raw (i.e., dense) input tensor data with random
       FP16/FP32/BFLOAT values 1 >= x > SMALLEST_RANDOM_FP.
       For 1x4x4x1 we have 16 elements.
    3) Stickify the data to ztensor.  Now ztensor.buffer has 16 DLFLOAT16
       elements with all the necessary paddings.
    4) get the array of address offsets where the values are expected to be in
       the stickified buffer.
    5) Perform the check:
       fp16_to_dlf16(input_data[n]) == output_data[n]
                                    (i.e., stick_area[offsets[n]])?
  */

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  void *data;

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

  data = create_and_fill_random_fp_data(&ztensor);

  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_transform_ztensor() failed, status = %08x "
      "(%s)",
      status, zdnn_get_status_message(status));

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    printf("%s(): dumpdata_origtensor\n", __func__);
    dumpdata_origtensor(&pre_tfrmd_desc, data, AS_HEX);
    dumpdata_origtensor(&pre_tfrmd_desc, data, AS_FLOAT);

    printf("%s(): dumpdata_ztensor\n", __func__);
    dumpdata_ztensor(&ztensor, AS_HEX, false);
    dumpdata_ztensor(&ztensor, AS_FLOAT, false);
  }

  uint64_t num_elements = get_num_elements(&ztensor, ELEMENTS_PRE);
  size_t *offsets = alloc_offsets(&ztensor, offset_mode, path);

  for (uint64_t i = 0; i < num_elements; i++) {

    // value in stick area, stickified
    uint16_t output_stickified_value =
        *(uint16_t *)((uintptr_t)ztensor.buffer + offsets[i]);

    // input value converted to DLFLOAT16, this is the "expected" value
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
      return;
    }

    TEST_ASSERT_MESSAGE_FORMATTED(
        almost_equal_dlf16(output_stickified_value, stickified_input_value),
        "Incorrect value at element %" PRIu64 ": Stickified: "
        "%.6f, Expected: %.6f",
        i, cnvt_1_dlf16_to_fp32(output_stickified_value),
        cnvt_1_dlf16_to_fp32(stickified_input_value));
  }

  // Free allocated storage
  free(offsets);
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

/**************************************************************
 * NHWC
 **************************************************************/

#define NHWC_TEST_BASIC(n, h, w, c)                                            \
  void test_nhwc_##n##x##h##x##w##x##c() {                                     \
    test_stickify(n, h, w, c, ZDNN_NHWC, QUICK_OFFSETS, NULL);                 \
  }

/*
 * Tensor with 16 entries, NHWC
 * 1,4,4,1 NHWC will use one cell per stick, 4 sticks per page and a total of 4
 * pages
 *
 *  [0, 128, 256, 384,          (H = 0)
 *  4096, 4224, 4352, 4480,     (H = 1)
 *  8192, 8320, 8448, 8576,     (H = 2)
 *  12288, 12416, 12544, 12672] (H = 3)
 */
NHWC_TEST_BASIC(1, 4, 4, 1);

NHWC_TEST_BASIC(1, 4, 4, 2);

NHWC_TEST_BASIC(1, 32, 32, 1);
NHWC_TEST_BASIC(1, 32, 32, 2);

/*
 * 3K entries in tensor, send to NHWC sticks
 * Each stick uses 3 cells, and all 32 sticks of the page are used.
 * 32 pages are used to store the values.
 */
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

/*
 * This routine is a generic test routine, allowing various 'e1' values
 * to be input. It tests stickification conversion (X -> DLFLOAT).
 * It assumes the e4-e2 values are 1 in order to
 * allow simpler assignment of the "offset" variable for
 * examining values stored in the stick.  e1 can range from 1 to 128,
 * i.e. one or two pages of 64 values per stick.
 */
void test_nhwc_1x1x1xe1(uint32_t e1) {
  test_stickify(1, 1, 1, e1, ZDNN_NHWC, QUICK_OFFSETS, NULL);
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

#define NHWC_TEST_WITH_FILE(n, h, w, c)                                        \
  void test_nhwc_##n##x##h##x##w##x##c() {                                     \
    test_stickify(n, h, w, c, ZDNN_NHWC, FILE_OFFSETS,                         \
                  OFFSET_FILE(nhwc, n, h, w, c));                              \
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

/*
 * Tensor with 16 entries, 3DS
 * 4,4,1 3DS will use one cell per stick, 4 sticks per page and a total of 4
 * pages.
 */
void test_3ds_4x4x1() {
  // first entry doesn't matter
  test_stickify(9999, 4, 4, 1, ZDNN_3DS, QUICK_OFFSETS, NULL);
}

/*
 * 3K entries in tensor, send to 3DS sticks
 * Each stick uses 3 cells, and all 32 sticks of the page are used.
 * 32 pages are used to store the values.
 *
 */
void test_3ds_32x32x3() {
  // first entry doesn't matter
  test_stickify(9999, 32, 32, 3, ZDNN_3DS, QUICK_OFFSETS, NULL);
}

/*
 * Tensor with 8 entries, 2DS
 * 4,2 2DS will use two cells per stick, (implied 1 stick per page) and a total
 * of 4 pages.
 */
void test_2ds_4x2() {
  // first two entries don't matter in 2DS
  test_stickify(9999, 9999, 4, 2, ZDNN_2DS, QUICK_OFFSETS, NULL);
}

/*
 * Tensor with 4k entries, 2DS
 * We expect this to require 4 pages total. Each dim2 will require 2 pages.
 * The first page will have all 64 cells of all 32 sticks filled holding 2048
 * values. A second page will have 1 stick with 1 cell filled to hold val 2049.
 */
void test_2ds_2x2049() {
  // first two entries don't matter in 2DS
  test_stickify(9999, 9999, 2, 2049, ZDNN_2DS, QUICK_OFFSETS, NULL);
}

void test_concat_stickify(zdnn_concat_info info, uint32_t dim3, uint32_t dim2,
                          uint32_t dim1) {

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  void *data[] = {NULL, NULL, NULL, NULL};
  uint8_t num_concats = 0;

  if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
    num_concats = 4;
  } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
    num_concats = 3;
  } else {
    TEST_FAIL_MESSAGE_FORMATTED("bad concat info: %08x\n", info);
  }

  // Fill in pre_transformed_desc. If dim3 is set, we're concatenating a 3DS
  // tensor otherwise assume 2DS.
  if (dim3) {
    // Initialize tensor descriptor
    zdnn_init_pre_transformed_desc(ZDNN_3DS, test_datatype, &pre_tfrmd_desc,
                                   dim3, dim2, dim1);
  } else {
    zdnn_init_pre_transformed_desc(ZDNN_2DS, test_datatype, &pre_tfrmd_desc,
                                   dim2, dim1);
  }

  // Fill in transformed_desc.
  status = zdnn_generate_transformed_desc_concatenated(&pre_tfrmd_desc, info,
                                                       &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc_concatenated() failed, status = %08x "
      "(%s) (concat info = %08x)",
      status, zdnn_get_status_message(status), info);

  // Create ztensor and allocate space for it's buffer
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(status == ZDNN_OK,
                                "zdnn_init_ztensor_with_malloc() failed, "
                                "status =  %08x (%s) (concat info = %08x)",
                                status, zdnn_get_status_message(status), info);

  // Fill in random data for each gate's original values
  for (uint8_t i = 0; i < num_concats; i++) {
    data[i] = create_and_fill_random_fp_data(&ztensor);
  }

  // Transform the original data values into the stickified ztensor
  switch (num_concats) {
  case 4:
    status =
        zdnn_transform_ztensor(&ztensor, data[0], data[1], data[2], data[3]);
    break;
  case 3:
    status = zdnn_transform_ztensor(&ztensor, data[0], data[1], data[2]);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("num_concats of %d is not supported (concat "
                                "info = %08x)",
                                num_concats, info);
    break;
  }
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_transform_ztensor() failed, status = %08x "
      "(%s) (concat info = %08x)",
      status, zdnn_get_status_message(status), info);

  // Print the original data and stickified buffer
  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    // Each gate will have it's own input data so dump each one. Each will
    // have the same dimensions/pre-tfrmd_desc
    for (uint8_t i = 0; i < num_concats; i++) {
      printf("%s(): dumpdata_origtensor for gate %d\n", __func__, i);
      dumpdata_origtensor(ztensor.pre_transformed_desc, data[i], AS_HEX);
      dumpdata_origtensor(ztensor.pre_transformed_desc, data[i], AS_FLOAT);
    }

    // The gates are concatenated into one ztensor so there's only one to dump
    printf("%s(): dumpdata_ztensor (concatenated)\n", __func__);
    dumpdata_ztensor(&ztensor, AS_HEX, false);
    dumpdata_ztensor(&ztensor, AS_FLOAT, false);
  }

  uint64_t elements_per_concat =
      get_num_elements(&ztensor, ELEMENTS_PRE_SINGLE_GATE);
  uint64_t slices_per_concat = ztensor.transformed_desc->dim4;
  uint64_t elements_per_concat_slice = elements_per_concat / slices_per_concat;

  LOG_DEBUG("elements_per_concat = %ld, slices_per_concat = %ld, "
            "elements_per_concat_slice = %ld",
            elements_per_concat, slices_per_concat, elements_per_concat_slice);

  size_t *offsets = alloc_offsets(&ztensor, QUICK_OFFSETS, NULL);

  uint16_t input_stickified_value = 0;
  uint16_t output_stickified_value;
  uint32_t offset_index = 0;

  // Loop through each offset in order and confirm the stickified value there
  // matches the correct original input value. The loop handles the difference
  // in output vs input element order caused by support of ztensor slicing.
  for (uint32_t slice = 0; slice < ztensor.transformed_desc->dim4; slice++) {
    size_t slice_offset =
        slice * elements_per_concat_slice * get_data_type_size(test_datatype);
    for (uint32_t concat = 0; concat < num_concats; concat++) {
      void *concat_slice_data =
          (void *)((uintptr_t)data[concat] + slice_offset);
      for (uint32_t elm_i = 0; elm_i < elements_per_concat_slice; elm_i++) {
        output_stickified_value =
            *(uint16_t *)((uintptr_t)ztensor.buffer + offsets[offset_index]);
        switch (test_datatype) {
        // Convert input to stickified values for comparison to output.
        case BFLOAT:
          input_stickified_value =
              cnvt_1_bfloat_to_dlf16(((uint16_t *)concat_slice_data)[elm_i]);
          LOG_TRACE(
              "offsets[%d] (native %s) = %04x vs %04x for input from "
              "slice %d of concat %d at element index %d (%s converted to %s)",
              offset_index, get_data_type_str(ZDNN_DLFLOAT16),
              output_stickified_value, input_stickified_value, slice, concat,
              elm_i, get_data_type_str(test_datatype),
              get_data_type_str(ZDNN_DLFLOAT16));
          break;
        case FP16:
          input_stickified_value =
              cnvt_1_fp16_to_dlf16(((uint16_t *)concat_slice_data)[elm_i]);
          LOG_TRACE(
              "offsets[%d] (native %s) = %04x vs %04x for input from "
              "slice %d of concat %d at element index %d (%s converted to %s)",
              offset_index, get_data_type_str(ZDNN_DLFLOAT16),
              output_stickified_value, input_stickified_value, slice, concat,
              elm_i, get_data_type_str(test_datatype),
              get_data_type_str(ZDNN_DLFLOAT16));
          break;
        case FP32:
          input_stickified_value =
              cnvt_1_fp32_to_dlf16(((float *)concat_slice_data)[elm_i]);
          LOG_TRACE(
              "offsets[%d] (%s converted to %s) = %4f vs %4f for input from "
              "slice %d of concat %d at element index %d (native %s)",
              offset_index, get_data_type_str(ZDNN_DLFLOAT16),
              get_data_type_str(test_datatype),
              cnvt_1_dlf16_to_fp32(output_stickified_value),
              ((float *)concat_slice_data)[elm_i], slice, concat, elm_i,
              get_data_type_str(test_datatype));
          break;
        default:
          TEST_FAIL_MESSAGE_FORMATTED("Unsupported data type %d (%s)",
                                      test_datatype,
                                      get_data_type_str(test_datatype));
          break;
        }
        TEST_ASSERT_MESSAGE_FORMATTED(
            output_stickified_value == input_stickified_value,
            "offsets[%u] = %04x (native %s) but expected %04x (%s "
            "converted to %s)",
            offset_index, output_stickified_value,
            get_data_type_str(ZDNN_DLFLOAT16), input_stickified_value,
            get_data_type_str(test_datatype),
            get_data_type_str(ZDNN_DLFLOAT16));
        offset_index++;
      }
    }
  }

  // Free allocated storage
  free(offsets);
  for (uint8_t i = 0; i < num_concats; i++) {
    free(data[i]);
  }
  zdnn_free_ztensor_buffer(&ztensor);
}

/*
 * Create a FICO bias ztensor with 16 entries:
 * 4 gates each having 1 direction each having 4 elements
 */
void test_lstm_biases_1x4() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j], 0,
                           1, 4);
    }
  }
}

/*
 * Create a FICO bias ztensor with 32 entries:
 * 4 gates each having 2 directions each having 4 elements
 */
void test_lstm_biases_2x4() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j], 0,
                           2, 4);
    }
  }
}

/*
 * Create a FICO bias ztensor with 520 entries:
 * 4 gates each having 2 directions each having 65 elements
 */
void test_lstm_biases_2x65() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j], 0,
                           2, 65);
    }
  }
}

/*
 * Create a FICO bias ztensor with 16392 entries:
 * 4 gates each having 2 directions each having 2049 elements
 * 2049 = 64 max cells per stick * 32 max sticks per page + 1. This means each
 * direction will require two 4K pages to stickify.
 */
void test_lstm_biases_2x2049() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_LSTM | prev_layers[i] | biases_usages[j], 0,
                           2, 2049);
    }
  }
}

/*
 * Create a FICO weights ztensor (PREV_LAYER_UNI) with 48 entries:
 * 4 gates each having 1 direction each having 3 rows with 4 elements
 */
void test_lstm_no_vconcat_weights_1x3x4() {
  test_concat_stickify(RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_WEIGHTS, 1, 3, 4);
}

/*
 * Create a FICO weights ztensor (PREV_LAYER_UNI) with 96 entries:
 * 4 gates each having 2 directions each having 3 rows with 4 elements
 */
void test_lstm_no_vconcat_weights_2x3x4() {
  test_concat_stickify(RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_WEIGHTS, 2, 3, 4);
}

/*
 * Create a FICO weights ztensor (PREV_LAYER_UNI) with 17160 entries:
 * 4 gates each having 2 directions each having 33 rows with 65 elements
 * Each direction will require two 4k pages to stickify as each cell has a max
 * of 64 elements and each page has a max of 32 sticks.
 */
void test_lstm_no_vconcat_weights_2x33x65() {
  test_concat_stickify(RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_WEIGHTS, 2, 33,
                       65);
}

/*
 * Create a FICO weights ztensor (PREV_LAYER_BIDIR) with 96 entries:
 * 4 gates each having 1 direction each having 6 rows with 4 elements
 */
void test_lstm_prev_bidir_weights_1x6x4() {
  test_concat_stickify(RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 1, 6,
                       4);
}

/*
 * Create a FICO weights ztensor (PREV_LAYER_BIDIR) with 192 entries:
 * 4 gates each having 2 directions each having 6 rows with 4 elements
 */
void test_lstm_prev_bidir_weights_2x6x4() {
  test_concat_stickify(RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 2, 6,
                       4);
}

/*
 * Create a FICO weights ztensor with (PREV_LAYER_BIDIR) 34320 entries:
 * 4 gates each having 2 directions each having 66 rows with 65 elements
 * Each direction will require eight 4k pages to stickify as each cell has a max
 * of 64 elements and each page has a max of 32 sticks.
 */
void test_lstm_prev_bidir_weights_2x66x65() {
  test_concat_stickify(RNN_TYPE_LSTM | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 2, 66,
                       65);
}

/*
 * Create a GRU bias ztensor with 12 entries:
 * 3 gates each having 1 direction each having 4 elements
 */
void test_gru_biases_1x4() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_GRU | prev_layers[i] | biases_usages[j], 0,
                           1, 4);
    }
  }
}

/*
 * Create a GRU bias ztensor with 24 entries:
 * 3 gates each having 2 directions each having 4 elements
 */
void test_gru_biases_2x4() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_GRU | prev_layers[i] | biases_usages[j], 0,
                           2, 4);
    }
  }
}

/*
 * Create a GRU bias ztensor with 390 entries:
 * 3 gates each having 2 directions each having 65 elements
 */
void test_gru_biases_2x65() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_GRU | prev_layers[i] | biases_usages[j], 0,
                           2, 65);
    }
  }
}

/*
 * Create a GRU bias ztensor with 12294 entries:
 * 3 gates each having 2 directions each having 2049 elements
 * 2049 = 64 max cells per stick * 32 max sticks per page + 1. This means each
 * direction will require two 4K pages to stickify.
 */
void test_gru_biases_2x2049() {
  for (int i = 0; i < NUM_PREV_LAYERS; i++) {
    for (int j = 0; j < NUM_BIASES_USAGES; j++) {
      test_concat_stickify(RNN_TYPE_GRU | prev_layers[i] | biases_usages[j], 0,
                           2, 2049);
    }
  }
}

/*
 * Create a ZRH weights ztensor (PREV_LAYER_UNI) with 36 entries:
 * 3 gates each having 1 direction each having 3 rows with 4 elements
 */
void test_gru_no_vconcat_weights_1x3x4() {
  test_concat_stickify(RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_WEIGHTS, 1, 3, 4);
}

/*
 * Create a ZRH weights ztensor (PREV_LAYER_UNI) with 72 entries:
 * 3 gates each having 2 directions each having 3 rows with 4 elements
 */
void test_gru_no_vconcat_weights_2x3x4() {
  test_concat_stickify(RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_WEIGHTS, 2, 3, 4);
}

/*
 * Create a ZRH weights ztensor (PREV_LAYER_UNI) with 12870 entries:
 * 3 gates each having 2 directions each having 33 rows with 65 elements
 * Each direction will require two 4k pages to stickify as each cell has a max
 * of 64 elements and each page has a max of 32 sticks.
 */
void test_gru_no_vconcat_weights_2x33x65() {
  test_concat_stickify(RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_WEIGHTS, 2, 33,
                       65);
}

/*
 * Create a ZRH weights ztensor (PREV_LAYER_BIDIR) with 72 entries:
 * 3 gates each having 1 direction each having 6 rows with 4 elements
 */
void test_gru_prev_bidir_weights_1x6x4() {
  test_concat_stickify(RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 1, 6,
                       4);
}

/*
 * Create a ZRH weights ztensor (PREV_LAYER_BIDIR) with 144 entries:
 * 3 gates each having 2 directions each having 6 rows with 4 elements
 */
void test_gru_prev_bidir_weights_2x6x4() {
  test_concat_stickify(RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 2, 6,
                       4);
}

/*
 * Create a ZRH weights ztensor with (PREV_LAYER_BIDIR) 25740 entries:
 * 3 gates each having 2 directions each having 66 rows with 65 elements
 * Each direction will require six 4k pages to stickify as each cell has a max
 * of 64 elements and each page has a max of 32 sticks.
 */
void test_gru_prev_bidir_weights_2x66x65() {
  test_concat_stickify(RNN_TYPE_GRU | PREV_LAYER_BIDIR | USAGE_WEIGHTS, 2, 66,
                       65);
}

void test_concat_weights_dim2(zdnn_concat_info info, uint32_t dim3,
                              uint32_t dim2, uint32_t dim1,
                              zdnn_status exp_status) {

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  void *data[] = {NULL, NULL, NULL, NULL};
  uint8_t num_concats = 0;

  if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
    num_concats = 4;
  } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
    num_concats = 3;
  } else {
    TEST_FAIL_MESSAGE_FORMATTED("bad concat info: %08x\n", info);
  }

  // if dim2 is odd number coming in, +1 so we create a valid dim2 and create
  // a valid ztensor with that.  else use it as is
  zdnn_init_pre_transformed_desc(ZDNN_3DS, test_datatype, &pre_tfrmd_desc, dim3,
                                 ((dim2 & 1) ? dim2 + 1 : dim2), dim1);

  status = zdnn_generate_transformed_desc_concatenated(&pre_tfrmd_desc, info,
                                                       &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc_concatenated() failed, status = %08x "
      "(%s) (concat info = %08x)",
      status, zdnn_get_status_message(status), info);

  // Create ztensor and allocate space for it's buffer
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(status == ZDNN_OK,
                                "zdnn_init_ztensor_with_malloc() failed, "
                                "status = %08x (%s) (concat info = %08x)",
                                status, zdnn_get_status_message(status), info);

  // Fill in random data for each gate's original values
  for (uint8_t i = 0; i < num_concats; i++) {
    data[i] = create_and_fill_random_fp_data(&ztensor);
  }

  // put back the incoming dim2 into pre-transformed desc as caller intended
  ztensor.pre_transformed_desc->dim2 = dim2;

  // Transform the original data values into the stickified ztensor
  switch (num_concats) {
  case 4:
    status =
        zdnn_transform_ztensor(&ztensor, data[0], data[1], data[2], data[3]);
    break;
  case 3:
    status = zdnn_transform_ztensor(&ztensor, data[0], data[1], data[2]);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "num_concats of %d is not supported (concat info = %08x)", num_concats,
        info);
    break;
  }
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == exp_status,
      "zdnn_transform_origtensor() unexpected status (status = %08x, "
      "expects = %08x)",
      status, exp_status);

  for (uint8_t i = 0; i < num_concats; i++) {
    free(data[i]);
  }
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_lstm_no_vconcat_weights_odd_dim2_pass() {
  test_concat_weights_dim2(RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_UNI, 3, 9,
                           10, ZDNN_OK);
}

void test_lstm_prev_bidir_weights_odd_dim2_fail() {
  test_concat_weights_dim2(RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_BIDIR, 3,
                           9, 10, ZDNN_INVALID_SHAPE);
}

void test_gru_no_vconcat_weights_odd_dim2_pass() {
  test_concat_weights_dim2(RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_UNI, 3, 9,
                           10, ZDNN_OK);
}

void test_gru_prev_bidir_weights_odd_dim2_fail() {
  test_concat_weights_dim2(RNN_TYPE_GRU | USAGE_WEIGHTS | PREV_LAYER_BIDIR, 3,
                           9, 10, ZDNN_INVALID_SHAPE);
}

/**************************************************************
 * NCHW
 **************************************************************/

#define NCHW_TEST_WITH_FILE(n, c, h, w)                                        \
  void test_nchw_##n##x##c##x##h##x##w() {                                     \
    test_stickify(n, c, h, w, ZDNN_NCHW, FILE_OFFSETS,                         \
                  OFFSET_FILE(nchw, n, c, h, w));                              \
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

// a simple (dumb) routine to convert a NHWC datastream to NCHW
void nhwc_2_nchw(void *nhwc_ptr, uint32_t n, uint32_t h, uint32_t w, uint32_t c,
                 int element_size, void *nchw_ptr) {
  uint32_t nx, hx, wx, cx;

  for (nx = 0; nx < n; nx++) {
    for (hx = 0; hx < h; hx++) {
      for (wx = 0; wx < w; wx++) {
        for (cx = 0; cx < c; cx++) {
          uint64_t nhwc_idx = nx * (h * w * c) + hx * (w * c) + wx * (c) + cx;
          uint64_t nchw_idx = nx * (c * h * w) + cx * (h * w) + hx * (w) + wx;
          if (element_size == 2) {
            ((uint16_t *)nchw_ptr)[nchw_idx] = ((uint16_t *)nhwc_ptr)[nhwc_idx];
          } else if (element_size == 4) {
            ((uint32_t *)nchw_ptr)[nchw_idx] = ((uint32_t *)nhwc_ptr)[nhwc_idx];
          }
        }
      }
    }
  }
}

/* create a NHWC input tensor data stream, then create a NCHW-copy of it via
 * matrix-rotate, then stickify both.  Compare the stickified data areas via
 * memcmp and they should match 100% */
void nhwc_nchw_comp(uint32_t n, uint32_t h, uint32_t w, uint32_t c) {

  zdnn_tensor_desc pre_tfrmd_desc_nhwc, pre_tfrmd_desc_nchw;
  zdnn_tensor_desc tfrmd_desc_nhwc, tfrmd_desc_nchw;

  zdnn_ztensor ztensor_nhwc, ztensor_nchw;
  zdnn_status status;
  void *data_nhwc, *data_nchw;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &pre_tfrmd_desc_nhwc,
                                 n, h, w, c);
  zdnn_init_pre_transformed_desc(ZDNN_NCHW, test_datatype, &pre_tfrmd_desc_nchw,
                                 n, c, h, w);

  zdnn_generate_transformed_desc(&pre_tfrmd_desc_nhwc, &tfrmd_desc_nhwc);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc_nchw, &tfrmd_desc_nchw);

  status = zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc_nhwc, &tfrmd_desc_nhwc,
                                         &ztensor_nhwc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc NHWC failed (status = %08x)", status);

  status = zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc_nchw, &tfrmd_desc_nchw,
                                         &ztensor_nchw);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc NCHW failed (status = %08x)", status);

  // create NHWC data stream, then matrix-rotate it to NCHW to another data
  // stream
  data_nhwc = create_and_fill_random_fp_data(&ztensor_nhwc);
  data_nchw = malloc(pre_tfrmd_desc_nhwc.dim4 * pre_tfrmd_desc_nhwc.dim3 *
                     pre_tfrmd_desc_nhwc.dim2 * pre_tfrmd_desc_nhwc.dim1 *
                     get_data_type_size(pre_tfrmd_desc_nhwc.type));
  nhwc_2_nchw(data_nhwc, n, h, w, c,
              get_data_type_size(pre_tfrmd_desc_nhwc.type), data_nchw);

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    printf("NHWC DATA  "
           "================================================================="
           "\n");

    dumpdata_origtensor(&pre_tfrmd_desc_nhwc, data_nhwc, AS_FLOAT);

    printf("NCHW DATA  "
           "================================================================="
           "\n");

    dumpdata_origtensor(&pre_tfrmd_desc_nchw, data_nchw, AS_FLOAT);
  }

  memset(ztensor_nhwc.buffer, 0, ztensor_nhwc.buffer_size);
  memset(ztensor_nchw.buffer, 0, ztensor_nchw.buffer_size);

  status = zdnn_transform_ztensor(&ztensor_nhwc, data_nhwc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_ztensor NHWC failed (status = %08x)",
      status);

  status = zdnn_transform_ztensor(&ztensor_nchw, data_nchw);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_ztensor NCHW failed (status = %08x)",
      status);

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    printf("NHWC STICK "
           "================================================================="
           "\n");

    dumpdata_ztensor(&ztensor_nhwc, AS_FLOAT, false);

    printf("NCHW STICK "
           "================================================================="
           "\n");

    dumpdata_ztensor(&ztensor_nchw, AS_FLOAT, false);
  }

  TEST_ASSERT_MESSAGE(memcmp(ztensor_nchw.buffer, ztensor_nhwc.buffer,
                             ztensor_nhwc.buffer_size) == 0,
                      "Stickified NHWC and NCHW don't match");

  free(data_nchw);
}

void test_nhwc_nchw_comp_1x4x4x1() { nhwc_nchw_comp(1, 4, 4, 1); }

void test_nhwc_nchw_comp_1x32x32x3() { nhwc_nchw_comp(1, 32, 32, 3); }

void test_nhwc_nchw_comp_2x3x33x129() { nhwc_nchw_comp(2, 3, 33, 129); }

// Reuse zdnn_ztensor without resetting is_transformed, expects
// ZDNN_BAD_PARAMETER
void test_ztensor_reuse_with_reset() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  unsigned char *data, *data2;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 4, 4, 1);

  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() failed (status = %08x)", status);

  TEST_ASSERT_MESSAGE(ZDNN_OK == zdnn_init_ztensor_with_malloc(
                                     &pre_tfrmd_desc, &tfrmd_desc, &ztensor),
                      "Unsuccessful zdnn_init_ztensor_with_malloc");

  data = create_and_fill_random_fp_data(&ztensor);
  data2 = create_and_fill_random_fp_data(&ztensor);

  TEST_ASSERT_MESSAGE(ZDNN_OK == zdnn_transform_ztensor(&ztensor, data),
                      "Unsuccessful first zdnn_transform_ztensor");

  zdnn_reset_ztensor(&ztensor);

  TEST_ASSERT_MESSAGE(ZDNN_OK == zdnn_transform_ztensor(&ztensor, data2),
                      "Unsuccessful second zdnn_transform_ztensor");

  // Free allocated storage
  free(data);
  free(data2);
  zdnn_free_ztensor_buffer(&ztensor);
}

// Reuse zdnn_ztensor without resetting is_transformed, expects
// ZDNN_BAD_PARAMETER
void test_ztensor_reuse_without_reset() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  unsigned char *data, *data2;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 4, 4, 1);

  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() failed (status = %08x)", status);

  TEST_ASSERT_MESSAGE(ZDNN_OK == zdnn_init_ztensor_with_malloc(
                                     &pre_tfrmd_desc, &tfrmd_desc, &ztensor),
                      "Unsuccessful zdnn_init_ztensor_with_malloc");

  data = create_and_fill_random_fp_data(&ztensor);
  data2 = create_and_fill_random_fp_data(&ztensor);

  TEST_ASSERT_MESSAGE(ZDNN_OK == zdnn_transform_ztensor(&ztensor, data),
                      "Unsuccessful first zdnn_transform_ztensor");

  TEST_ASSERT_MESSAGE(
      ZDNN_INVALID_STATE == zdnn_transform_ztensor(&ztensor, data2),
      "Second zdnn_transform_ztensor does not yield ZDNN_INVALID_STATE");

  // Free allocated storage
  free(data);
  free(data2);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_format_after_stickify_4dfeature_success() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;

  // sabotage ztensor with crap values
  memset(&ztensor, 0xFF, sizeof(ztensor));

  // doing all these steps absolutely barebone, as the normal testcases should
  // have covered verifying the status
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 4, 4, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);

  zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE(ztensor.is_transformed == true,
                      "Expected is_transformed to be set to true, it is not.");

  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_format_after_stickify_4dfeature_fail() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;

  // sabotage ztensor with crap values
  memset(&ztensor, 0xFF, sizeof(ztensor));

  // doing all these steps absolutely barebone, as the normal testcases should
  // have covered verifying the status
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 4, 4, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);

  // sabotage ztensor.pre_transformed_desc so it would fail
  ztensor.pre_transformed_desc->type = ZDNN_DLFLOAT16;

  zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE(ztensor.is_transformed == false,
                      "Expected is_transformed to be set to false, it is not.");

  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_ztensor_null_buffer() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;
  zdnn_status status;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 4, 4, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);

  ztensor.buffer = NULL;

  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_INVALID_BUFFER,
      "zdnn_transform_ztensor() failed (status = %08x, expects = %08x)", status,
      ZDNN_DATA_ERROR);

  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_ztensor_not_enough_buffersize() {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;
  zdnn_status status;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 4, 1, 1, 1);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);

  // (4, 1, 1, 1) needs 4 * 4096 bytes
  ztensor.buffer_size = 4096;

  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_INVALID_BUFFER,
      "zdnn_transform_ztensor() failed (status = %08x, expects = %08x)", status,
      ZDNN_DATA_ERROR);

  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

// This routine tests the conversion from FP16 to DLF
// Input: a "bad" value in FP16, which will "trip" the
//        floating point exception trigger on VCNF

void test_ztensor_bad_value_FP16(uint16_t bad_value) {

#define INF_FP16_POS 0X7C00
#define INF_FP16_NEG 0xFC00
#define NAN_FP16_POS 0x7FFF
#define NAN_FP16_NEG 0xFFFF

#define STICK_ENTRIES_FP16 7

  uint32_t stick_entries_to_try[STICK_ENTRIES_FP16] = {0, 1, 7, 8, 9, 62, 63};
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;
  uint16_t *array; // Alternate view on data
  zdnn_status status;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &pre_tfrmd_desc, 1, 1, 1, 64);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);
  array = (uint16_t *)data; /* use data as an INT array */

  for (int i = 0; i < STICK_ENTRIES_FP16; i++) {
    array[stick_entries_to_try[i]] = bad_value;

    ztensor.is_transformed = false; /* set false for next attempt, required
                                       for underflow case */
    status = zdnn_transform_ztensor(&ztensor, data);

    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_CONVERT_FAILURE,
        "zdnn_transform_ztensor() succeeded (status = %08x, expects = "
        "%08x, i = %d, value = %04x)",
        status, ZDNN_CONVERT_FAILURE, i, bad_value);

    TEST_ASSERT_MESSAGE_FORMATTED(
        ztensor.is_transformed == false,
        "zdnn_transform_ztensor() set is_transformed (status = %08x, "
        "expects = %08x, i = %d, value = %08x)",
        status, ZDNN_CONVERT_FAILURE, i, bad_value);

    array[stick_entries_to_try[i]] = 0; // set entry to 0 for next iteration
  }
  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_ztensor_fp16_bad_values() {

#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE("needs NNPA to trigger overflow/invalid-op/etc");
#endif

  test_ztensor_bad_value_FP16(
      INF_FP16_POS); // is not a number, will cause overflow
  test_ztensor_bad_value_FP16(
      INF_FP16_NEG); // is not a number, will cause overflow
  test_ztensor_bad_value_FP16(
      NAN_FP16_POS); // is not a number, will cause invalid op
  test_ztensor_bad_value_FP16(
      NAN_FP16_NEG); // is not a number, will cause Invalid Op
  // Underflow not possible converting FP16 to DLF (VCNF)
}

// This routine tests the conversion from FP32 to DLFloat16
// Input: a "bad" value in FP32, which will "trip" the
//        floating point exception trigger on VCRNF
// NOTE:  Only Not-A-Number values will trip the exception.
void test_ztensor_bad_value_FP32(uint32_t bad_value) {
#define TOO_SMALL_FP32_POS 0x00000FF0
#define TOO_SMALL_FP32_NEG 0x80000FF0
#define TOO_LARGE_INF_FP32_POS 0x7F800000
#define TOO_LARGE_INF_FP32_NEG 0xFF800000
#define NAN_FP32_POS 0x7FFFFFFF
#define NAN_FP32_NEG 0xFFFFFFFF

#define STICK_ENTRIES_FP32 9

  uint32_t stick_entries_to_try[STICK_ENTRIES_FP32] = {0, 1, 3,  4, 7,
                                                       8, 9, 15, 63};
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  unsigned char *data;
  uint32_t *array;
  zdnn_status status;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, 1, 1, 1, 64);
  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  data = create_and_fill_random_fp_data(&ztensor);
  array = (uint32_t *)data; /* use data as an INT array */

  for (int i = 0; i < STICK_ENTRIES_FP32; i++) {
    array[stick_entries_to_try[i]] = bad_value;
    ztensor.is_transformed = false; /* set false for next attempt, required
                                       for underflow case */

    status = zdnn_transform_ztensor(&ztensor, data);

    if (bad_value != TOO_SMALL_FP32_NEG &&
        bad_value != TOO_SMALL_FP32_POS) { // if not underflow case

      TEST_ASSERT_MESSAGE_FORMATTED(
          status == ZDNN_CONVERT_FAILURE,
          "zdnn_transform_ztensor() with overflow succeeded (status = "
          "%08x, expects = "
          "%08x, i = %d, value = %08x)",
          status, ZDNN_CONVERT_FAILURE, i, bad_value);

      TEST_ASSERT_MESSAGE_FORMATTED(
          ztensor.is_transformed == false,
          "zdnn_transform_ztensor() set is_transformed (status = %08x, "
          "expects = %08x, i = %d, value = %08x)",
          status, ZDNN_CONVERT_FAILURE, i, bad_value);
    } else { // Must be underflow case

      TEST_ASSERT_MESSAGE_FORMATTED(
          status != ZDNN_CONVERT_FAILURE,
          "zdnn_transform_ztensor() with underflow did not succeed (status "
          "= %08x, expects = "
          "%08x, i = %04x, value = %08x)",
          status, ZDNN_CONVERT_FAILURE, i, bad_value);

      TEST_ASSERT_MESSAGE_FORMATTED(
          ztensor.is_transformed == true,
          "zdnn_transform_ztensor() set is_transformed (status = %08x, "
          "expects = %08x, i = %d, value = %08x))",
          status, ZDNN_CONVERT_FAILURE, i, bad_value);
    }

    array[stick_entries_to_try[i]] = 0; // set entry to 0 for next iteration
  }
  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

void test_ztensor_fp32_bad_values() {

#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE("needs NNPA to trigger overflow/invalid-op/etc");
#endif

  test_ztensor_bad_value_FP32(
      TOO_SMALL_FP32_POS); // non-zero converts to 0, cause underflow
  test_ztensor_bad_value_FP32(
      TOO_SMALL_FP32_NEG); // non-zero converts to 0, cause underflow

  test_ztensor_bad_value_FP32(
      TOO_LARGE_INF_FP32_POS); // is not a number, will cause overflow
  test_ztensor_bad_value_FP32(
      TOO_LARGE_INF_FP32_NEG); // is not a number, will cause overflow

  test_ztensor_bad_value_FP32(
      NAN_FP32_POS); // is not a number, will cause invalid op
  test_ztensor_bad_value_FP32(
      NAN_FP32_NEG); // is not a number, will cause invalid op
}

/**************************************************************
 * HWCK
 **************************************************************/

#define HWCK_TEST_WITH_FILE(h, w, c, k)                                        \
  void test_hwck_##h##x##w##x##c##x##k() {                                     \
    test_stickify(h, w, c, k, ZDNN_HWCK, FILE_OFFSETS,                         \
                  OFFSET_FILE(hwck, h, w, c, k));                              \
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

  // NHWC tests that use offset_files
  RUN_TEST_ALL_DATATYPES(test_nhwc_1x2x3x4);
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
  RUN_TEST_ALL_DATATYPES(test_nhwc_2x3x33x129);

  RUN_TEST_ALL_DATATYPES(test_3ds_4x4x1);
  RUN_TEST_ALL_DATATYPES(test_3ds_32x32x3);

  RUN_TEST_ALL_DATATYPES(test_2ds_4x2);
  RUN_TEST_ALL_DATATYPES(test_2ds_2x2049);

  RUN_TEST_ALL_DATATYPES(test_lstm_biases_1x4);
  RUN_TEST_ALL_DATATYPES(test_lstm_biases_2x4);
  RUN_TEST_ALL_DATATYPES(test_lstm_biases_2x65);
  RUN_TEST_ALL_DATATYPES(test_lstm_biases_2x2049);

  RUN_TEST_ALL_DATATYPES(test_lstm_no_vconcat_weights_1x3x4);
  RUN_TEST_ALL_DATATYPES(test_lstm_no_vconcat_weights_2x3x4);
  RUN_TEST_ALL_DATATYPES(test_lstm_no_vconcat_weights_2x33x65);

  RUN_TEST_ALL_DATATYPES(test_lstm_prev_bidir_weights_1x6x4);
  RUN_TEST_ALL_DATATYPES(test_lstm_prev_bidir_weights_2x6x4);
  RUN_TEST_ALL_DATATYPES(test_lstm_prev_bidir_weights_2x66x65);

  RUN_TEST_ALL_DATATYPES(test_gru_biases_1x4);
  RUN_TEST_ALL_DATATYPES(test_gru_biases_2x4);
  RUN_TEST_ALL_DATATYPES(test_gru_biases_2x65);
  RUN_TEST_ALL_DATATYPES(test_gru_biases_2x2049);

  RUN_TEST_ALL_DATATYPES(test_gru_no_vconcat_weights_1x3x4);
  RUN_TEST_ALL_DATATYPES(test_gru_no_vconcat_weights_2x3x4);
  RUN_TEST_ALL_DATATYPES(test_gru_no_vconcat_weights_2x33x65);

  RUN_TEST_ALL_DATATYPES(test_gru_prev_bidir_weights_1x6x4);
  RUN_TEST_ALL_DATATYPES(test_gru_prev_bidir_weights_2x6x4);
  RUN_TEST_ALL_DATATYPES(test_gru_prev_bidir_weights_2x66x65);

  RUN_TEST_ALL_DATATYPES(test_lstm_no_vconcat_weights_odd_dim2_pass);
  RUN_TEST_ALL_DATATYPES(test_lstm_prev_bidir_weights_odd_dim2_fail);
  RUN_TEST_ALL_DATATYPES(test_gru_no_vconcat_weights_odd_dim2_pass);
  RUN_TEST_ALL_DATATYPES(test_gru_prev_bidir_weights_odd_dim2_fail);

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

  RUN_TEST_ALL_DATATYPES(test_nhwc_nchw_comp_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_nhwc_nchw_comp_1x32x32x3);
  RUN_TEST_ALL_DATATYPES(test_nhwc_nchw_comp_2x3x33x129);

  RUN_TEST_ALL_DATATYPES(test_hwck_1x4x4x1);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x2x3x4);
  RUN_TEST_ALL_DATATYPES(test_hwck_2x3x33x129);
  RUN_TEST_ALL_DATATYPES(test_hwck_1x32x32x3);
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

  RUN_TEST_ALL_DATATYPES(test_ztensor_reuse_with_reset);
  RUN_TEST_ALL_DATATYPES(test_ztensor_reuse_without_reset);
  RUN_TEST_ALL_DATATYPES(test_format_after_stickify_4dfeature_success);
  RUN_TEST_ALL_DATATYPES(test_format_after_stickify_4dfeature_fail);
  RUN_TEST_ALL_DATATYPES(test_ztensor_null_buffer);
  RUN_TEST_ALL_DATATYPES(test_ztensor_not_enough_buffersize);

  RUN_TEST_ALL_DATATYPES(test_ztensor_fp16_bad_values);
  RUN_TEST_ALL_DATATYPES(test_ztensor_fp32_bad_values);

  return UNITY_END();
}
