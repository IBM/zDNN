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

// Allows struct timeval to work on z/OS. Must be before <sys/time.h> include
#ifdef __MVS__
#define _XOPEN_SOURCE_EXTENDED 1
#undef _ALL_SOURCE
#endif

#include "testsupport.h"
#include "zdnn.h"
#include "zdnn_private.h"
#include <assert.h>
#include <float.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

char error_message[ERROR_MESSAGE_STR_LENGTH];
float ZERO_ARRAY[1] = {0};

// Custom FP tolerance for tests to set and use, if needed
fp_tolerance tol_bfloat = {0, 0}, tol_fp16 = {0, 0}, tol_fp32 = {0, 0};

zdnn_concat_info prev_layers[NUM_PREV_LAYERS] = {PREV_LAYER_UNI,
                                                 PREV_LAYER_BIDIR};
zdnn_concat_info biases_usages[NUM_BIASES_USAGES] = {USAGE_BIASES,
                                                     USAGE_HIDDEN_BIASES};
zdnn_concat_info no_vconcat_infos[NUM_NO_VCONCAT_INFOS] = {
    PREV_LAYER_UNI | USAGE_HIDDEN_WEIGHTS,
    PREV_LAYER_BIDIR | USAGE_HIDDEN_WEIGHTS,
    PREV_LAYER_UNI | USAGE_WEIGHTS,
};

// Generate size_t offset array based on dimensions of ztensor.
//
// NOTE: when transformed dim1 is > 64, dim3 can not be > 1
void quick_generate_offsets(zdnn_ztensor *ztensor, size_t *offsets) {

  uint64_t total_elements;

  // fail the testcase right now if dim1 > 64 && dim3 > 1
  // the e1_offset_template[i] = <...> loop doesn't handle that case

  TEST_ASSERT_MESSAGE_FORMATTED(!(ztensor->transformed_desc->dim3 > 1 &&
                                  ztensor->transformed_desc->dim1 > 64),
                                "incorrect quick_generate_offsets() usage: "
                                "dim3 (%u) > 1 and dim1 (%u) > 64",
                                ztensor->transformed_desc->dim3,
                                ztensor->transformed_desc->dim1);

  if ((ztensor->transformed_desc->layout == ZDNN_ZRH) ||
      (ztensor->transformed_desc->layout == ZDNN_FICO) ||
      (ztensor->transformed_desc->layout == ZDNN_BIDIR_ZRH) ||
      (ztensor->transformed_desc->layout == ZDNN_BIDIR_FICO)) {
    total_elements = get_num_elements(ztensor, ELEMENTS_PRE_ALL_GATES);
  } else {
    total_elements = get_num_elements(ztensor, ELEMENTS_PRE);
  }

  // Concatenated trfmd->dim1/dim2 includes padding so get the pre-padded ones.
  // Non-concatenated this will be equal to pre-trfmd's.
  uint32_t unpadded_dim1 = ztensor->pre_transformed_desc->dim1;
  uint32_t unpadded_dim2;

  // Offset template for e1 elements. These offsets will be added to the e1 loop
  // when determining correct offsets for test cases.
  size_t e1_offset_template[unpadded_dim1];

  if (ztensor->transformed_desc->layout != ZDNN_BIDIR_FICO &&
      ztensor->transformed_desc->layout != ZDNN_BIDIR_ZRH) {

    // transformed_desc->dim2 has the correct value we need
    unpadded_dim2 = ztensor->transformed_desc->dim2;

    for (uint32_t i = 0; i < unpadded_dim1; i++) {
      // build an offset template for the unpadded_dim1 number of elements.  all
      // eventual offsets are going to follow that pattern
      e1_offset_template[i] =
          ((i / AIU_2BYTE_CELLS_PER_STICK) *
           CEIL(unpadded_dim2, AIU_STICKS_PER_PAGE) * AIU_PAGESIZE_IN_BYTES) +
          (i % AIU_2BYTE_CELLS_PER_STICK) * get_data_type_size(ZDNN_DLFLOAT16);
      LOG_TRACE("e1_offset_template[%d] = %d", i, e1_offset_template[i]);
    }

    uint64_t offset_i = 0;
    size_t e1_offset_start = 0;

    // Generate an offset for each element. Note: For concatenated ztensors,
    // padding elements will not be included in the offsets.
    while (unpadded_dim1 && offset_i < total_elements) {
      // Add relative e1 template to current stick start to get target offset.
      for (uint32_t dim1_i = 0; dim1_i < unpadded_dim1; dim1_i++) {
        offsets[offset_i] = e1_offset_start + e1_offset_template[dim1_i];
        LOG_TRACE("offsets[%d] = %+08x", offset_i, offsets[offset_i]);
        offset_i++;
      }

      // Jump e1_offset_start to the start of the next unused page as soon as
      // all dim1 elements for each dim2 are processed
      if (offset_i % (unpadded_dim2 * unpadded_dim1) == 0) {
        // We already incremented offset_i so use previous offset_i to determine
        // current page number.
        uint32_t curr_page_num = offsets[offset_i - 1] / AIU_PAGESIZE_IN_BYTES;
        // Reset the e1 offset start to start of next page.
        e1_offset_start = (curr_page_num + 1) * AIU_PAGESIZE_IN_BYTES;
        LOG_TRACE("Jumped to start of next page location = %+08x",
                  e1_offset_start);
      } else {
        // The e1 templates can skip over whole sticks if the number of elements
        // is larger than a single stick. Once the current dim1 row is fully
        // processed, reset e1_offset_start to jump back to the start of the
        // first empty stick.
        e1_offset_start += AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE;
        LOG_TRACE("Jumped to start of first empty stick = %+08x",
                  e1_offset_start);
      }
    }
  } else {

    // transformed_desc->dim2 is vertically concatenated, so instead grab the
    // actual dim2 from pre_transformed_desc
    unpadded_dim2 = ztensor->pre_transformed_desc->dim2;

    // number of pages needed to store a single c-stick (max:
    // AIU_2BYTE_CELLS_PER_STICK) worth of elements
    uint32_t num_pages_vertical =
        PADDED(unpadded_dim2 / 2) / AIU_STICKS_PER_PAGE * 2;

    for (uint32_t i = 0; i < unpadded_dim1; i++) {
      e1_offset_template[i] =
          ((i / AIU_2BYTE_CELLS_PER_STICK) * num_pages_vertical *
           AIU_PAGESIZE_IN_BYTES) +
          (i % AIU_2BYTE_CELLS_PER_STICK) * get_data_type_size(ZDNN_DLFLOAT16);
      LOG_TRACE("e1_offset_template[%d] = %d", i, e1_offset_template[i]);
    }

    uint64_t offset_i = 0;
    size_t e1_offset_start = 0;
    size_t e1_offset_start_slice = 0;

    while (unpadded_dim1 && offset_i < total_elements) {
      // build an offset template like the other case
      for (uint32_t dim1_i = 0; dim1_i < unpadded_dim1; dim1_i++) {
        offsets[offset_i] = e1_offset_start + e1_offset_template[dim1_i];
        LOG_TRACE("offsets[%d] = %+08x", offset_i, offsets[offset_i]);
        offset_i++;
      }

      uint32_t curr_page_num = offsets[offset_i - 1] / AIU_PAGESIZE_IN_BYTES;
      if (offset_i % (unpadded_dim2 * unpadded_dim1) == 0) {
        // when we're done with this slice, reset the e1 offset start.  the new
        // page number is always in multiples of 2 due to vertical concatenation
        e1_offset_start =
            CEIL(curr_page_num + 1, 2) * 2 * AIU_PAGESIZE_IN_BYTES;
        // save the offset start of this new slice to jump back to later
        e1_offset_start_slice = e1_offset_start;
        LOG_TRACE("Jumped to start of new page location = %+08x",
                  e1_offset_start);
      } else if (offset_i % (unpadded_dim2 / 2 * unpadded_dim1) == 0) {
        // when we're done with 1st half of dim2, reset e1_offset_start to
        // beginning of this slice + half num_pages_vertical worth of bytes
        e1_offset_start = e1_offset_start_slice +
                          num_pages_vertical / 2 * AIU_PAGESIZE_IN_BYTES;
        LOG_TRACE("Jumped back to start of 2nd half = %+08x", e1_offset_start);
      } else {
        // go to the next c-stick
        e1_offset_start += AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE;
        LOG_TRACE("Jumped to start of first empty stick = %+08x",
                  e1_offset_start);
      }
    }
  }
}

// Get integer values from a txt file and put them in the size_t array.
// used by the stickify/unstickify test routines. Returns number of values
// read.
uint64_t get_offsets_from_file(const char *file_name, size_t *array) {
  if (file_name == NULL) {
    TEST_FAIL_MESSAGE("file_name required for get_offsets_from_file");
  }

  FILE *file = fopen(file_name, "r");
  uint64_t i = 0;

  if (file) {
    while (!feof(file)) {
      // Read integers from file, stopping at the first non-integer
      // (ie final newline)
      if (fscanf(file, "%" PRIu64 "", &array[i]) != 1) {
        break;
      }
      i++;
    }
    fclose(file);
  }

  return i;
}

size_t *alloc_offsets(zdnn_ztensor *ztensor, offset_mode mode,
                      const char *path) {

  if (path != NULL && mode != FILE_OFFSETS) {
    TEST_FAIL_MESSAGE("path only valid for file mode");
  }

  uint64_t total_elements;

  if ((ztensor->transformed_desc->layout == ZDNN_ZRH) ||
      (ztensor->transformed_desc->layout == ZDNN_FICO) ||
      (ztensor->transformed_desc->layout == ZDNN_BIDIR_ZRH) ||
      (ztensor->transformed_desc->layout == ZDNN_BIDIR_FICO)) {
    total_elements = get_num_elements(ztensor, ELEMENTS_PRE_ALL_GATES);
  } else {
    total_elements = get_num_elements(ztensor, ELEMENTS_PRE);
  }

  LOG_TRACE("ztensor->transformed_desc->layout = %s, total_elements = %ld",
            get_data_layout_str(ztensor->transformed_desc->layout),
            total_elements);

  size_t *offsets = malloc(total_elements * sizeof(size_t));

  switch (mode) {
  case QUICK_OFFSETS: {
    quick_generate_offsets(ztensor, offsets);
    break;
  }
  case FILE_OFFSETS: {
    uint64_t num_offsets = get_offsets_from_file(path, offsets);
    TEST_ASSERT_MESSAGE_FORMATTED(
        get_offsets_from_file(path, offsets) == total_elements,
        "for %" PRIu64
        " elements get_offsets_from_file() on file \"%s\" returned %" PRIu64
        " offsets",
        total_elements, path, num_offsets);
    break;
  }
  default: {
    TEST_FAIL_MESSAGE_FORMATTED("unknown mode: %d", mode);
    break;
  }
  }

  return offsets;
}

size_t *alloc_rnn_output_offsets(const zdnn_ztensor *ztensor) {

  // basically the result is like (dim4 * dim3) pieces of ZDNN_2D (dim2, dim1)
  // offsets stitched together, and everytime we replicate a piece we add some
  // offset to it
  zdnn_tensor_desc tmp_p_desc, tmp_t_desc;
  zdnn_ztensor tmp_ztensor;

  // create a ZDNN_2D (dim2, dim1) tensor and get the offsets of that via
  // alloc_offsets()
  zdnn_init_pre_transformed_desc(ZDNN_2D, test_datatype, &tmp_p_desc,
                                 ztensor->pre_transformed_desc->dim2,
                                 ztensor->pre_transformed_desc->dim1);
  zdnn_generate_transformed_desc(&tmp_p_desc, &tmp_t_desc);

  zdnn_status status =
      zdnn_init_ztensor_with_malloc(&tmp_p_desc, &tmp_t_desc, &tmp_ztensor);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc() failed status = %08x",
      status);

  size_t *piece_offsets = alloc_offsets(&tmp_ztensor, QUICK_OFFSETS, NULL);

  // each replication is seperated by this many bytes
  uint64_t piece_size = zdnn_getsize_ztensor(&tmp_t_desc);

  size_t *offsets = malloc(get_num_elements(ztensor, ELEMENTS_PRE_SINGLE_GATE) *
                           sizeof(size_t));

  // replicate the offsets dim4*dim3 times
  uint64_t c = 0;
  for (uint32_t i = 0; i < ztensor->pre_transformed_desc->dim4 *
                               ztensor->pre_transformed_desc->dim3;
       i++) {
    for (uint32_t j = 0; j < ztensor->pre_transformed_desc->dim2 *
                                 ztensor->pre_transformed_desc->dim1;
         j++) {
      offsets[c] = piece_size * i + piece_offsets[j];
      c++;
    }
  }

  return offsets;
}
/// Creates a data buffer with the provided float values converted to the
/// specified type
///
/// \note This method does not check that the size of values matches expected
/// number of elements.
///
/// Example usage:
/// Setup input tensor
/// \code
///  void *data = alloc_and_convert_float_values(num_values, type, values);
/// \endcode
///
/// \param[in] type data type to convert the values into
/// \param[in] num_values number of values in the float array
/// \param[in] repeat_first_value if true, data will be poplulated with
///                               values[0]
/// \param[in] values float array of values to convert and store in
///                   data
///
/// \return a pointer with alloced memory containing the converted values
///
void *alloc_and_convert_float_values(zdnn_data_types type, uint64_t num_values,
                                     bool repeat_first_value, float *values) {

  // Malloc the data buffer
  size_t data_size = num_values * get_data_type_size(type);
  void *data = malloc(data_size);
  memset(data, 0, data_size);

  // Convert values into desired type and store in data buffer
  for (uint64_t i = 0; i < num_values; i++) {
    float value;
    if (repeat_first_value) {
      value = values[0];
    } else {
      value = values[i];
    }
    switch (type) {
    case BFLOAT:
      ((uint16_t *)data)[i] = cnvt_1_fp32_to_bfloat(value);
      break;
    case FP16:
      ((uint16_t *)data)[i] = cnvt_1_fp32_to_fp16(value);
      break;
    case FP32:
      ((float *)data)[i] = value;
      break;
    default:
      // NOTE: Along with undefined types, DLFLOAT types will also come down
      // this path. zdnn_transform_ztensor() would fail with them as
      // DLFLOATs are a stickified type and transform() expects unstickified
      // data.
      TEST_FAIL_MESSAGE_FORMATTED("unsupported type: %d", type);
    }
  }

  return data;
}

/// Creates a ztensor with the provided values. Values are converted to the
/// specified type. The resulting ztensor is transformed and ready for use in
/// zDNN operations.
///
/// \note This method does not check that the size of values matches expected
/// number of elements.
///
/// Example usage:
/// Setup input tensor
/// \code
///  ztensor *zt = alloc_ztensor_with_values(shape, pre_tfrmd_layout,
///                                          type, NO_CONCAT, false, values);
/// \endcode
/// Setup Output tensor
/// \code
/// ztensor *zt = alloc_ztensor_with_values(shape, pre_tfrmd_layout,
///                                         type, NO_CONCAT, true,
///                                         ZERO_ARRAY);
/// \endcode
///
/// \param[in] shape array of dimensions
/// \param[in] pre_tfrmd_layout pre-transformed data layout
/// \param[in] type data type
/// \param[in] zdnn_concat_info
///                     indicates the type of concatenation to use
///                     This indirectly sets the transformed ztensor layout
///                     and the number of values arrays to expect.
/// \param[in] repeat_first_value if true, ztensor will be poplulated with
///                               values[0]
/// \param[in] ... float array(s) to tensor data or gates data.
///                1 array for NO_CONCAT, 3 arrays for GRU, 4 arrays for LSTM
///
/// \return zdnn_ztensor* Pointer to a malloc'd ztensor with transformed data
///
zdnn_ztensor *alloc_ztensor_with_values(uint32_t *shape,
                                        zdnn_data_layouts pre_tfrmd_layout,
                                        zdnn_data_types type,
                                        zdnn_concat_info info,
                                        int repeat_first_value, ...) {
  zdnn_status status = GENERAL_TESTCASE_FAILURE;

  // Create the pretransformed description
  zdnn_tensor_desc *pre_tfrmd_desc =
      (zdnn_tensor_desc *)malloc(sizeof(zdnn_tensor_desc));

  switch (pre_tfrmd_layout) {
  case (ZDNN_1D):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0]);
    break;
  case (ZDNN_2D):
  case (ZDNN_2DS):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0], shape[1]);
    break;
  case (ZDNN_3D):
  case (ZDNN_3DS):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0], shape[1], shape[2]);
    break;
  case (ZDNN_4D):
  case (ZDNN_4DS):
  case (ZDNN_NHWC):
  case (ZDNN_NCHW):
  case (ZDNN_HWCK):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0], shape[1], shape[2], shape[3]);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "pre_tfrmd_layout. Could you teach me?",
        get_data_layout_str(pre_tfrmd_layout));
    break;
  }

  // Create the transformed description
  zdnn_tensor_desc *tfrmd_desc =
      (zdnn_tensor_desc *)malloc(sizeof(zdnn_tensor_desc));

  if (info == NO_CONCAT) {
    status = zdnn_generate_transformed_desc(pre_tfrmd_desc, tfrmd_desc);
    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_OK,
        "zdnn_generate_transformed_desc failed (status = %08x)", status);
  } else {
    status = zdnn_generate_transformed_desc_concatenated(pre_tfrmd_desc, info,
                                                         tfrmd_desc);
    TEST_ASSERT_MESSAGE_FORMATTED(status == ZDNN_OK,
                                  "zdnn_generate_transformed_desc_concatenated "
                                  "with info %08x failed (status = %08x)",
                                  info, status);
  }

  // Create the ztensor with malloc'd buffer pointer
  zdnn_ztensor *ztensor = (zdnn_ztensor *)malloc(sizeof(zdnn_ztensor));

  status = zdnn_init_ztensor_with_malloc(pre_tfrmd_desc, tfrmd_desc, ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc failed (status = %08x)",
      status);

  // Prepare to iterate over the passed in values arrays
  va_list values_list;
  va_start(values_list, repeat_first_value);
  uint64_t num_elements = get_num_elements(ztensor, ELEMENTS_PRE_SINGLE_GATE);

  if (pre_tfrmd_layout == ZDNN_4DS) {
    // For testing outputs, we want to be able initialize rnn output ztensors to
    // zeros but we don't need to support setting arbitrary values
    memset(ztensor->buffer, 0, ztensor->buffer_size);
  } else {
    uint32_t num_things;

    // Find out how many things to stickify
    if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
      num_things = get_func_code_num_gates(NNPA_LSTMACT);
    } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
      num_things = get_func_code_num_gates(NNPA_GRUACT);
    } else {
      num_things = 1;
      // the NO_CONCAT case, so we have 1 thing
    }

    void *values_data[num_things];

    // Convert that many things
    for (uint32_t i = 0; i < num_things; i++) {
      values_data[i] = alloc_and_convert_float_values(
          type, num_elements, repeat_first_value, va_arg(values_list, float *));
    }

    // Stickify ztensor using data that we type converted above
    if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
      status = zdnn_transform_ztensor(ztensor, values_data[0], values_data[1],
                                      values_data[2], values_data[3]);
    } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
      status = zdnn_transform_ztensor(ztensor, values_data[0], values_data[1],
                                      values_data[2]);
    } else {
      status = zdnn_transform_ztensor(ztensor, values_data[0]);
    }

    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_OK,
        "zdnn_transform_ztensor failed with status %08x \"%s\"", status,
        zdnn_get_status_message(status));

    for (uint32_t i = 0; i < num_things; i++) {
      free(values_data[i]);
    }
  }

  va_end(values_list);
  return ztensor;
}

// -----------------------------------------------------------------------------
// ULP-based Floating Point Comparsino Functions
// -----------------------------------------------------------------------------

// used to get around "breaking strict-aliasing rules"
typedef union float_int_u {
  // cppcheck-suppress unusedStructMember
  float f;
  int i;
} float_int_u;

int ulps_diff_float(float a, float b) {
  float_int_u au = {a};
  float_int_u bu = {b};

  // Make au.i lexicographically ordered as a twos-complement int
  if (au.i < 0)
    au.i = 0x80000000 - au.i;
  // Make bu.i lexicographically ordered as a twos-complement int
  if (bu.i < 0)
    bu.i = 0x80000000 - bu.i;
  return abs(au.i - bu.i);
}

int ulps_diff_16(uint16_t a, uint16_t b) {
  int16_t a_int = *(int16_t *)&a;
  int16_t b_int = *(int16_t *)&b;
  // Make a_int lexicographically ordered as a twos-complement int
  if (a_int < 0)
    a_int = 0x8000 - a_int;
  // Make b_int lexicographically ordered as a twos-complement int
  if (b_int < 0)
    b_int = 0x8000 - b_int;
  return abs(a_int - b_int);
}

// -----------------------------------------------------------------------------
// Floating Point Verify Functions
//
// - basic version (uses default fp_tolerance defined in testsupport.h)
// - advanced version, suppply custom fp_tolerance
//
// Use ULPs comparsion first, then epsilon as fallback
// -----------------------------------------------------------------------------

// advanced versions
bool almost_equal_bfloat_adv(uint16_t actual, uint16_t expected,
                             fp_tolerance tol) {
  // try ulps verification first, so we don't need to convert things to float
  int ulps_diff = ulps_diff_16(actual, expected);
  if (ulps_diff > tol.ulps) {
    LOG_DEBUG("actual = %f, expected = %f: ulps diff = %d (max = %d)",
              cnvt_1_bfloat_to_fp32(actual), cnvt_1_bfloat_to_fp32(expected),
              ulps_diff, tol.ulps);
    // epsilon verification
    float diff =
        fabs(cnvt_1_bfloat_to_fp32(actual) - cnvt_1_bfloat_to_fp32(expected));
    float max_diff = EPSILON_BFLOAT * tol.epsilon_mult;
    LOG_DEBUG("    diff = %f (max = %f)", diff, max_diff);
    return !(diff > max_diff);
  }
  return true;
}

bool almost_equal_fp16_adv(uint16_t actual, uint16_t expected,
                           fp_tolerance tol) {
  // try ulps verification first, so we don't need to convert things to float
  int ulps_diff = ulps_diff_16(actual, expected);
  if (ulps_diff > tol.ulps) {
    LOG_DEBUG("actual = %f, expected = %f: ulps diff = %d (max = %d)",
              cnvt_1_fp16_to_fp32(actual), cnvt_1_fp16_to_fp32(expected),
              ulps_diff, tol.ulps);
    // epsilon verification
    float diff =
        fabs(cnvt_1_fp16_to_fp32(actual) - cnvt_1_fp16_to_fp32(expected));
    float max_diff = EPSILON_FP16 * tol.epsilon_mult;
    LOG_DEBUG("    diff = %f (max = %f)", diff, max_diff);
    return !(diff > max_diff);
  }
  return true;
}

bool almost_equal_float_adv(float actual, float expected, fp_tolerance tol) {
  // ulps-based verification
  int ulps_diff = ulps_diff_float(actual, expected);
  if (ulps_diff > tol.ulps) {
    LOG_DEBUG("actual = %f, expected = %f: ulps diff = %d (max = %d)", actual,
              expected, ulps_diff, tol.ulps);
    // epsilon verification
    float diff = fabs(actual - expected);
    float max_diff = EPSILON_FLOAT * tol.epsilon_mult;
    LOG_DEBUG("    diff = %f (max = %f)", diff, max_diff);
    return !(diff > max_diff);
  }
  return true;
}

bool almost_equal_dlf16_adv(uint16_t actual, uint16_t expected,
                            fp_tolerance tol) {
  // try ulps verification first, so we don't need to convert things to float
  int ulps_diff = ulps_diff_16(actual, expected);
  if (ulps_diff > tol.ulps) {
    LOG_DEBUG("actual = %f, expected = %f: ulps diff = %d (max = %d)",
              cnvt_1_dlf16_to_fp32(actual), cnvt_1_dlf16_to_fp32(expected),
              ulps_diff, tol.ulps);
    // epsilon verification
    float diff =
        fabs(cnvt_1_dlf16_to_fp32(actual) - cnvt_1_dlf16_to_fp32(expected));
    float max_diff = EPSILON_DLFLOAT16 * tol.epsilon_mult;
    LOG_DEBUG("    diff = %f (max = %f)", diff, max_diff);
    return !(diff > max_diff);
  }
  return true;
}

// basic versions, use default fp_tolerance.
bool almost_equal_bfloat(uint16_t actual, uint16_t expected) {
  fp_tolerance tol = {MAX_ULPS_BFLOAT, MAX_EPSILON_MULT_BFLOAT};
  return almost_equal_bfloat_adv(actual, expected, tol);
}

bool almost_equal_fp16(uint16_t actual, uint16_t expected) {
  fp_tolerance tol = {MAX_ULPS_FP16, MAX_EPSILON_MULT_FP16};
  return almost_equal_fp16_adv(actual, expected, tol);
}

bool almost_equal_float(float actual, float expected) {
  fp_tolerance tol = {MAX_ULPS_FLOAT, MAX_EPSILON_MULT_FLOAT};
  return almost_equal_float_adv(actual, expected, tol);
}

bool almost_equal_dlf16(uint16_t actual, uint16_t expected) {
  fp_tolerance tol = {MAX_ULPS_DLFLOAT16, MAX_EPSILON_MULT_DLFLOAT16};
  return almost_equal_dlf16_adv(actual, expected, tol);
}

/// Asserts each value in the stickified ztensor are within a specified
/// tolerance from the given expected float values.
///
/// \note This method does not check that the size of values array matches the
/// number of elements. If there's not enough expected values, the test will
/// likely fail when garbage data is pulled in as the expected value.
///
/// Example usage:
/// \code
///  assert_ztensor_values_adv(&ztensor, false, values, true, tol);
/// \endcode
///
/// \param[in] ztensor pointer to zdnn_ztensor with actual values
/// \param[in] repeat_first_expected_value if true, all ztensor values will be
///                                        compared to values[0]
/// \param[in] values array of expected values
/// \param[in] tol floating point tolerance information
///
/// \return None (assert fails if any actual value not within expected range)
///
void assert_ztensor_values_adv(zdnn_ztensor *ztensor,
                               bool repeat_first_expected_value, void *values,
                               fp_tolerance tol) {
  zdnn_status status;
  zdnn_tensor_desc *pre_tfrmd_desc = ztensor->pre_transformed_desc;

  uint64_t num_elements = 0;
  switch (ztensor->transformed_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_4DS:
  case ZDNN_NHWC:
    num_elements = get_num_elements(ztensor, ELEMENTS_PRE);
    break;
  case ZDNN_FICO:
  case ZDNN_ZRH:
    TEST_FAIL_MESSAGE_FORMATTED(
        "does not support %s layout as we don't support unstickifying "
        "concatenated ztensors.",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "layout. Could you teach me?",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  }

  // Malloc error_message as it will be large if num_elements is large.
  uint64_t big_error_message_size =
      (uint64_t)sizeof(char) * ERROR_MESSAGE_STR_LENGTH * num_elements;
  char *error_msg = malloc(big_error_message_size);

  void *actual_vals, *expected_vals;

  // Get unstickified data from ztensor to actual_vals[]
  actual_vals = malloc(num_elements * get_data_type_size(pre_tfrmd_desc->type));
  status = zdnn_transform_origtensor(ztensor, actual_vals);
  snprintf(error_msg, big_error_message_size,
           "zdnn_transform_origtensor failed (status = %08x)", status);
  TEST_ASSERT_MESSAGE(status == ZDNN_OK, error_msg);

  // expected_vals[] will contains the expected values (values[]) but in the
  // same data type as actual_vals[], i.e., (pre_tfrmd_desc->type)
  expected_vals =
      malloc(num_elements * get_data_type_size(pre_tfrmd_desc->type));

  // Instead of directly converting from C float to (pre_tfrmd_desc->type), we
  // convert it to DLFLOAT16 first then (pre_tfrmd_desc->type) in order to
  // simulate the precision loss the values have gone through.  The same
  // process applies for FP32.

  for (uint64_t i = 0; i < num_elements; i++) {
    uint16_t tmp_dlf16;

    if (!repeat_first_expected_value) {
      tmp_dlf16 = cnvt_1_fp32_to_dlf16(((float *)values)[i]);
    } else {
      tmp_dlf16 = cnvt_1_fp32_to_dlf16(((float *)values)[0]);
    }

    switch (pre_tfrmd_desc->type) {
    case BFLOAT:
      ((uint16_t *)expected_vals)[i] =
          cnvt_1_fp32_to_bfloat(cnvt_1_dlf16_to_fp32(tmp_dlf16));
      break;
    case FP16:
      ((uint16_t *)expected_vals)[i] =
          cnvt_1_fp32_to_fp16(cnvt_1_dlf16_to_fp32(tmp_dlf16));
      break;
    case FP32:
      ((float *)expected_vals)[i] = cnvt_1_dlf16_to_fp32(tmp_dlf16);
      break;
    default:
      // NOTE: Along with undefined types, DLFLOAT types will also come down
      // this path. DLFLOATS are a stickified types which are not valid types
      // for the pre_tfrmd_desc (ie prestickifed description).
      snprintf(error_msg, big_error_message_size, "unsupported type: %d\n",
               pre_tfrmd_desc->type);
      TEST_FAIL_MESSAGE(error_msg);
      break;
    }
  }

  // Assert ztentor's values (converted back to floats) match does not exceed
  // max ULPs and epsilon
  bool all_pass = true;
  // Loop appends to error_msg so reset it first
  error_msg[0] = '\0';

  char *error_fmt = "Element %" PRIu64 " == %f expecting %f";
  char *error_fmt2 =
      " <==== FAILED (diff beyond ULPs %u, epsilon multiplier %u)";

  // Compared the actual and expected values
  for (uint64_t i = 0; i < num_elements; i++) {

    bool is_almost_equal = false;

    switch (pre_tfrmd_desc->type) {
    case BFLOAT: {
      // Record all actual vs expected values (only printed if one fails)
      // For printf-ing error_msg we'll need to convert "actual" to float
      uint16_t actual = ((uint16_t *)actual_vals)[i];
      uint16_t expected = ((uint16_t *)expected_vals)[i];

      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt, i,
               cnvt_1_bfloat_to_fp32(actual), cnvt_1_bfloat_to_fp32(expected));

      LOG_DEBUG(error_fmt, i, cnvt_1_bfloat_to_fp32(actual),
                cnvt_1_bfloat_to_fp32(expected));

      is_almost_equal = almost_equal_bfloat_adv(actual, expected, tol);
      break;
    }
    case FP16: {
      uint16_t actual = ((uint16_t *)actual_vals)[i];
      uint16_t expected = ((uint16_t *)expected_vals)[i];

      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt, i,
               cnvt_1_fp16_to_fp32(actual), cnvt_1_fp16_to_fp32(expected));

      LOG_DEBUG(error_fmt, i, cnvt_1_fp16_to_fp32(actual),
                cnvt_1_fp16_to_fp32(expected));

      is_almost_equal = almost_equal_fp16_adv(actual, expected, tol);
      break;
    }
    case FP32: {
      float actual = ((float *)actual_vals)[i];
      float expected = ((float *)expected_vals)[i];

      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt, i, actual,
               expected);

      LOG_DEBUG(error_fmt, i, actual, expected);

      is_almost_equal = almost_equal_float_adv(actual, expected, tol);
      break;
    }
    default:
      // would have died earlier
      break;
    }

    if (!is_almost_equal) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt2, tol.ulps,
               tol.epsilon_mult);
      all_pass = false;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");
  }

  // Assert that all passed and clean up temp data
  TEST_ASSERT_MESSAGE(all_pass, error_msg);

  free(expected_vals);
  free(actual_vals);
  free(error_msg);
}

void assert_ztensor_values(zdnn_ztensor *ztensor,
                           bool repeat_first_expected_value, void *values) {

  fp_tolerance tol = {0, 0}; // zero tolerance ==> testcase will likely fail.

  switch (ztensor->pre_transformed_desc->type) {
  case BFLOAT:
    tol.ulps = MAX_ULPS_BFLOAT;
    tol.epsilon_mult = MAX_EPSILON_MULT_BFLOAT;
    break;
  case FP16:
    tol.ulps = MAX_ULPS_FP16;
    tol.epsilon_mult = MAX_EPSILON_MULT_FP16;
    break;
  case FP32:
    tol.ulps = MAX_ULPS_FLOAT;
    tol.epsilon_mult = MAX_EPSILON_MULT_FLOAT;
    break;
  default:
    // let assert_ztensor_values_adv() deal with it
    break;
  }

  assert_ztensor_values_adv(ztensor, repeat_first_expected_value, values, tol);
}

/// Free buffers, descriptions, and ztensors structs for all provided ztensors
///
/// \param[in] num_of_ztensors number of ztensors pointers passed into this
///                            method
/// \param[in] ... variable number of ztensor pointers
///
/// \return None (assert fails if freeing any buffer fails
///
void free_ztensor_buffers(uint32_t num_ztensors, ...) {
  zdnn_status status;

  // Create ztensor_list to handle the multple input ztensors passed in.
  va_list ztensor_list;
  va_start(ztensor_list, num_ztensors);

  // Free data buffer for each provided ztensor
  for (uint32_t i = 0; i < num_ztensors; i++) {
    zdnn_ztensor *ztensor = va_arg(ztensor_list, zdnn_ztensor *);
    if ((status = zdnn_free_ztensor_buffer(ztensor)) != ZDNN_OK) {
      free(ztensor->transformed_desc);
      free(ztensor->pre_transformed_desc);
      free(ztensor);
      TEST_FAIL_MESSAGE_FORMATTED(
          "zdnn_free_ztensor_buffer() failed on tensor %u with status %08x", i,
          status);
    }
  }

  va_end(ztensor_list);
}

/// Allocates a data buffer then fills it with random float values (between
/// SMALLEST_RANDOM_FP to 1)
///
/// \param[out] ztensor A zdnn tensor
///
/// \return pointer to filled data buffer
///
unsigned char *create_and_fill_random_fp_data(zdnn_ztensor *ztensor) {

  // The single concat looks at just the pre_tfrmd shape which matches tfrmd
  // size for everything but concat cases. For concat tests that use this, we
  // want the single concat size specifically because we generate the data for
  // each concat (RNN gate) separately.
  uint64_t num_elements = get_num_elements(ztensor, ELEMENTS_PRE_SINGLE_GATE);
  zdnn_data_types dtype = ztensor->pre_transformed_desc->type;
  void *data = malloc(num_elements * get_data_type_size(dtype));

  struct timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec);

  for (int i = 0; i < num_elements; i++) {

    float filling = 0;

    // https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
    while (filling < SMALLEST_RANDOM_FP) {
      filling = (float)rand() / (float)(RAND_MAX);
    }

    switch (dtype) {
    case BFLOAT:
      ((uint16_t *)data)[i] = cnvt_1_fp32_to_bfloat(filling);
      break;
    case FP16:
      ((uint16_t *)data)[i] = cnvt_1_fp32_to_fp16(filling);
      break;
    case FP32:
      ((float *)data)[i] = filling;
      break;
    case ZDNN_DLFLOAT16:
      ((uint16_t *)data)[i] = cnvt_1_fp32_to_dlf16(filling);
    }
  }

  return data;
}

/**
 * Helper that generates random floats and populate the given array. This will
 * be used for populating tensor buffers in the end-to-end unit tests.
 *
 * https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
 */
void gen_random_float_array(int size, float arr[]) {
  struct timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec);

  // The raw output value will be [0, a]. To make sure we're always at least
  // SMALLEST_RANDOM_FP from zero, add it to the result. Also subtract it
  // from the max so when we add it to the result, we'll still be within max.
  float desired_max = LARGEST_RANDOM_FP - SMALLEST_RANDOM_FP;
  for (int i = 0; i < size; i++) {
    arr[i] =
        ((float)rand() / (float)(RAND_MAX)) * desired_max + SMALLEST_RANDOM_FP;
  }
}

/**
 * Helper that generates random negative floats and populate the given array.
 * This will be used for populating tensor buffers in the end-to-end unit
 * tests.
 */
void gen_random_float_array_neg(int size, float arr[]) {
  struct timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec);

  // The raw output value will be [0, a]. To make sure we're always at least
  // SMALLEST_RANDOM_FP from zero, add it to the result. Also subtract it
  // from the max so when we add it to the result, we'll still be within max.
  float desired_max = LARGEST_RANDOM_FP - SMALLEST_RANDOM_FP;
  for (int i = 0; i < size; i++) {
    arr[i] =
        -((float)rand() / (float)(RAND_MAX)) * desired_max + SMALLEST_RANDOM_FP;
  }
}

/**
 * Helper that generates random negative and positive float values for a given
 * size and for a given array, meant for populating tensor buffers in
 * end-to-end unit tests.
 *
 * Every other array index will be negative:
 *
 * Example: [-1, 2, -3, 4, -5, 6]
 */
void gen_random_float_array_pos_neg(int size, float arr[]) {
  struct timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec);

  float desired_max = LARGEST_RANDOM_FP - SMALLEST_RANDOM_FP;
  for (int i = 0; i < size; i++) {
    arr[i] = (((float)rand() / (float)(RAND_MAX)) * desired_max +
              SMALLEST_RANDOM_FP) *
             ((i % 2 == 0) ? 1 : -1);
  }
}

/**
 * Helper that generates 0 values for a given size
 * and for a given array, meant for populating tensor buffers
 * in end-to-end unit tests.
 */
void gen_float_array_zeros(int size, float arr[]) {
  for (int i = 0; i < size; i++) {
    arr[i] = 0;
  }
}

/**
 * Helper that generates an array copy for a given size and
 * for a given array, meant for populating tensor buffers in
 * end-to-end unit tests.
 */
void copy_to_array(int size, float input[], float output[]) {
  for (int i = 0; i < size; i++) {
    output[i] = input[i];
  }
}

/**
 * Helper that generates an array with every other value equaling zero for a
 * given size and for a given array, meant for populating tensor buffers in
 * end-to-end unit tests.
 *
 * Every other array index will be negative:
 *
 * Example:
 *    input: [1,2,3,4,5,6]
 *   output: [0,2,0,4,0,6]
 */
void fill_everyother_with_zero_float_array(int size, float arr[]) {
  for (int i = 0; i < size; i++) {
    if (i % 2 != 0) {
      arr[i] = 0;
    }
  }
}

/**
 * Helper that generates an array with all values equaling zero for a
 * given size and for a given array, meant for populating tensor buffers in
 * end-to-end unit tests.
 */
void fill_all_with_zero_float_array(int size, float arr[]) {
  for (int i = 0; i < size; i++) {
    arr[i] = 0;
  }
}

int stdout_pipe[2];
int stderr_pipe[2];
int saved_stdout;
int saved_stderr;

void stdout_to_pipe() {
  // save stream for display later
  saved_stdout = dup(STDOUT_FILENO);
  fflush(stdout);

  // make a pipe
  if (pipe(stdout_pipe) != 0) {
    TEST_FAIL_MESSAGE("Can't open pipe()");
  }

  // redirect to pipe
  dup2(stdout_pipe[1], STDOUT_FILENO);
  close(stdout_pipe[1]);

  return;
}

void stderr_to_pipe() {
  // save stream for display later
  saved_stderr = dup(STDERR_FILENO);
  fflush(stderr);

  // make a pipe
  if (pipe(stderr_pipe) != 0) {
    TEST_FAIL_MESSAGE("Can't open pipe()");
  }

  // redirect to pipe
  dup2(stderr_pipe[1], STDERR_FILENO);
  close(stderr_pipe[1]);

  return;
}

void restore_stdout(char *buf, int buf_size) {
  // the read() below blocks if nothing to read, so printf something
  fprintf(stdout, " ");

  fflush(stdout);

  // read from pipe into buffer
  read(stdout_pipe[0], buf, buf_size);
  close(stdout_pipe[0]);

  // restore stream to display
  dup2(saved_stdout, STDOUT_FILENO);
}

void restore_stderr(char *buf, int buf_size) {
  // the read() below blocks if nothing to read, so printf something
  fprintf(stderr, "x");

  fflush(stderr);

  // read from pipe into buffer
  read(stderr_pipe[0], buf, buf_size);
  close(stderr_pipe[0]);

  // restore stream to display
  dup2(saved_stderr, STDERR_FILENO);
}

/**********************************************************
 * Enhanced Unity Functions/Macros
 **********************************************************/

#define NUM_PRE_TFRMD_TYPES 3
zdnn_data_types pre_tfrmd_types[NUM_PRE_TFRMD_TYPES] = {FP16, FP32, BFLOAT};

#define NUM_TFRMD_TYPES 1
zdnn_data_types tfrmd_types[NUM_TFRMD_TYPES] = {ZDNN_DLFLOAT16};

// indicates which data-type UnityDefaultTestRunWith*DataType() is currently
// testing
zdnn_data_types test_datatype = 128; // set initial value to something invalid

// Wrapper of Unity's UnityDefaultTestRun() that runs func() against all
// input data-types.  Uses CamelCase intentionally to align with Unity
void UnityDefaultTestRunWithDataType(UnityTestFunction Func,
                                     const char *FuncName,
                                     const int FuncLineNum) {
  for (int i = 0; i < NUM_PRE_TFRMD_TYPES; i++) {
    test_datatype = pre_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(pre_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

// UnityDefaultTestRunWithDataType() but with transformed data-types
void UnityDefaultTestRunWithTfrmdDataType(UnityTestFunction Func,
                                          const char *FuncName,
                                          const int FuncLineNum) {
  for (int i = 0; i < NUM_TFRMD_TYPES; i++) {
    test_datatype = tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}
