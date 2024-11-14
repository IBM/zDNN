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
          } else if (element_size == 8) {
            ((uint64_t *)nchw_ptr)[nchw_idx] = ((uint64_t *)nhwc_ptr)[nhwc_idx];
          }
        }
      }
    }
  }
}

size_t *alloc_offsets(zdnn_ztensor *ztensor) {
  // create offsets array using the formulas described in the z/Architecture
  // Principles of Operation

  uint64_t total_elements = get_num_elements(ztensor, ELEMENTS_PRE);

  size_t *offsets = malloc(total_elements * sizeof(size_t));

  uint32_t e4 = ztensor->transformed_desc->dim4,
           e3 = ztensor->transformed_desc->dim3,
           e2 = ztensor->transformed_desc->dim2,
           e1 = ztensor->transformed_desc->dim1;

  uint8_t eps = ztensor->transformed_desc->type != ZDNN_BINARY_INT8
                    ? AIU_2BYTE_CELLS_PER_STICK
                    : AIU_1BYTE_CELLS_PER_STICK;

  uint64_t c = 0;

  switch (ztensor->transformed_desc->format) {
  case ZDNN_FORMAT_4DFEATURE: {
    uint32_t e2_limit = CEIL(e2, 32) * 32;
    uint32_t e1_limit = CEIL(e1, eps) * eps;

    for (uint32_t e4x = 0; e4x < e4; e4x++) {
      for (uint32_t e3x = 0; e3x < e3; e3x++) {
        for (uint32_t e2x = 0; e2x < e2; e2x++) {
          for (uint32_t e1x = 0; e1x < e1; e1x++) {
            offsets[c] =
                ( // get to the correct N = e4x
                    (e3 * e2_limit * e1_limit * e4x) +
                    // get to the currect H = e3x, assuming e1x = 0
                    (e2_limit * e3x * eps) +
                    // get to the correct stick (e2x), still assuming e1x = 0
                    (e2x * eps) +
                    // jump to the correct e1x = [0..63] [64..127] of that stick
                    ((uint32_t)(e1x / eps) * e2_limit * e3 * eps) +
                    // jump to correct element within the stick
                    (e1x % eps)) *
                (128 / eps);
            c++;
          }
        }
      }
    }
    if (ztensor->pre_transformed_desc->layout == ZDNN_NCHW) {
      size_t *tmp = malloc(total_elements * sizeof(size_t));
      nhwc_2_nchw(offsets, e4, e3, e2, e1, sizeof(size_t), tmp);
      free(offsets);
      offsets = tmp;
    }
    break;
  }
  case ZDNN_FORMAT_4DKERNEL: {
    uint32_t e2_limit = CEIL(e2, 32) * 32;

    for (uint32_t e4x = 0; e4x < e4; e4x++) {
      for (uint32_t e3x = 0; e3x < e3; e3x++) {
        for (uint32_t e2x = 0; e2x < e2; e2x++) {
          for (uint32_t e1x = 0; e1x < e1; e1x++) {
            offsets[c] =
                ( // jump to the correct e1x = [0..63] [64..127] of that stick
                    ((uint32_t)(e1x / eps) * e4 * e3 * e2_limit * eps) +
                    // get to the correct W = e3x, assuming e1x = 0
                    (e2_limit * e3x * eps) +
                    // get to the correct stick (e2x), still assuming e1x = 0
                    (e2x * eps) +
                    // get to the correct H
                    (e4x * e3 * e2_limit * eps) +
                    // jump to correct element within the stick
                    (e1x % eps)) *
                (128 / eps);
            c++;
          }
        }
      }
    }
    break;
  }
  case ZDNN_FORMAT_4DWEIGHTS: {
    uint32_t e2_limit = CEIL(e2, 64) * 64;
    uint32_t e1_limit = CEIL(e1, 64) * 64;

    for (uint32_t e4x = 0; e4x < e4; e4x++) {
      for (uint32_t e3x = 0; e3x < e3; e3x++) {
        for (uint32_t e2x = 0; e2x < e2; e2x++) {
          for (uint32_t e1x = 0; e1x < e1; e1x++) {
            offsets[c] =
                // get to the correct N = e4x
                (e4x * e3 * e2_limit * e1_limit) +
                // get to the currect H = e3x, assuming e1x = 0
                (e3x * e2_limit * 64) +
                // get to the correct stick
                ((uint32_t)(e2x / 2) * 128) +
                // jump to the correct e1x = [0..63] [64..127] of that stick
                ((uint32_t)(e1x / 64) * e2_limit * e3 * 64) +
                // jump to the correct pair within the stick
                ((e1x * 2) % 128) +
                // jump to correct entry within that pair
                (e2x % 2);
            c++;
          }
        }
      }
    }
    break;
  }
  default:
    TEST_FAIL_MESSAGE_FORMATTED("unknown transformed descriptor format: %d",
                                ztensor->transformed_desc->format);
  }

  return offsets;
}

size_t *alloc_rnn_offsets(const zdnn_ztensor *ztensor) {

  // generate basic offsets based off vanilla ZDNN_2DS/ZDNN_3DS shape

  zdnn_tensor_desc slice_t_desc;
  zdnn_ztensor slice_ztensor;

  size_t *offsets, *slice_offsets = NULL;

  if (ztensor->transformed_desc->layout != ZDNN_BIDIR_FICO &&
      ztensor->transformed_desc->layout != ZDNN_BIDIR_ZRH) {

    // ZDNN_FICO/ZDNN_ZRH is like having a stickified vanilla ZDNN_2DS/ZDNN_3DS
    // stitched together 4 (FICO) or 3 (ZRH) times.
    //
    // so we get the basic stificifed offsets for the ZDNN_2DS/ZDNN_3DS first,
    // then duplicate it 2 or 3 more times while adding some offset to each
    // value

    zdnn_generate_transformed_desc(ztensor->pre_transformed_desc,
                                   &slice_t_desc);
    zdnn_init_ztensor(ztensor->pre_transformed_desc, &slice_t_desc,
                      &slice_ztensor);

    slice_offsets = alloc_offsets(&slice_ztensor);
    uint64_t slice_total_elements =
        get_num_elements(&slice_ztensor, ELEMENTS_PRE);
    uint64_t slice_size = zdnn_getsize_ztensor(slice_ztensor.transformed_desc);

    short num_slices =
        get_data_layout_num_gates(ztensor->transformed_desc->layout);
    offsets = malloc(num_slices * slice_total_elements * sizeof(size_t));

    // make num_slices copies of those offsets, each set is seperated by
    // slice_size bytes
    for (uint64_t i = 0; i < num_slices; i++) {
      for (uint64_t j = 0; j < slice_total_elements; j++) {
        offsets[i * slice_total_elements + j] =
            slice_offsets[j] + i * slice_size;
      }
    }

  } else {

    zdnn_tensor_desc tmp_f_desc, tmp_t_desc;
    zdnn_ztensor tmp_ztensor;

    // get the basic stificifed offsets as if it were a ZDNN_3D of (2,
    // PADDED(dim2 / 2), dim1).
    // set dim3 = 2 to simulate the effect of dividing the entries into 2 halves
    // don't care about num_dirs (dim3) for now
    memcpy(&tmp_f_desc, ztensor->pre_transformed_desc,
           sizeof(zdnn_tensor_desc));
    tmp_f_desc.layout = ZDNN_3D;
    tmp_f_desc.dim3 = 2;
    tmp_f_desc.dim2 = PADDED(tmp_f_desc.dim2 / 2);

    zdnn_generate_transformed_desc(&tmp_f_desc, &tmp_t_desc);
    zdnn_init_ztensor(&tmp_f_desc, &tmp_t_desc, &tmp_ztensor);

    size_t *tmp_offsets = alloc_offsets(&tmp_ztensor);
    uint64_t tmp_ztensor_size =
        zdnn_getsize_ztensor(tmp_ztensor.transformed_desc);

    // we generated (2 * PADDED(dim2 / 2) * dim1) number of offsets, but we
    // actually only care (dim2 / 2 * dim1) of those
    uint64_t slice_total_elements = ztensor->pre_transformed_desc->dim2 *
                                    ztensor->pre_transformed_desc->dim1;

    // in the generated offsets array, only the first (slice_total_elements / 2)
    // entries are valid because the entries follow are simply for the vertical
    // paddings.
    //
    // the 2 halves are actually PADDED(dim2 / 2) * AIU_BYTES_PER_STICK bytes
    // apart
    for (int q = 0; q < slice_total_elements / 2; q++) {
      tmp_offsets[slice_total_elements / 2 + q] =
          tmp_offsets[q] + (PADDED(ztensor->pre_transformed_desc->dim2 / 2) *
                            AIU_BYTES_PER_STICK);
    }

    short num_slices =
        get_data_layout_num_gates(ztensor->transformed_desc->layout);

    offsets = malloc(ztensor->pre_transformed_desc->dim3 * num_slices *
                     slice_total_elements * sizeof(size_t));

    // make num_slices * num_dirs copies of those offsets, each set is seperated
    // by tmp_ztensor_size bytes
    for (uint64_t i = 0; i < ztensor->pre_transformed_desc->dim3; i++) {
      for (uint64_t j = 0; j < num_slices; j++) {
        for (uint64_t k = 0; k < slice_total_elements; k++) {
          offsets[i * num_slices * slice_total_elements +
                  j * slice_total_elements + k] =
              tmp_offsets[k] + tmp_ztensor_size * (i * num_slices + j);
        }
      }
    }
  }

  free(slice_offsets);
  return offsets;
}

size_t *alloc_rnn_output_offsets(const zdnn_ztensor *ztensor) {

  // basically the result is like (dim4 * dim3) pieces of ZDNN_2D (dim2, dim1)
  // offsets stitched together, and everytime we replicate a piece we add some
  // offset to it
  zdnn_tensor_desc tmp_p_desc, tmp_t_desc;
  zdnn_ztensor tmp_ztensor;

  // create a ZDNN_2D (dim2, dim1) tensor and get the offsets of that
  zdnn_init_pre_transformed_desc(ZDNN_2D, test_datatype, &tmp_p_desc,
                                 ztensor->pre_transformed_desc->dim2,
                                 ztensor->pre_transformed_desc->dim1);
  zdnn_generate_transformed_desc(&tmp_p_desc, &tmp_t_desc);

  zdnn_status status =
      zdnn_init_ztensor_with_malloc(&tmp_p_desc, &tmp_t_desc, &tmp_ztensor);

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc() failed status = %08x",
      status);

  size_t *piece_offsets = alloc_offsets(&tmp_ztensor);

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
                                     bool repeat_first_value,
                                     const float *values) {

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
    // For testing outputs, we want to be able initialize rnn output ztensors
    // to zeros but we don't need to support setting arbitrary values
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

/// Creates a ztensor with no value. The resulting ztensor is not transformed
/// and ready for use as an output in zDNN operations.
///
/// Example usage:
/// Setup input tensor
/// \code
///  ztensor *zt = alloc_ztensor_with_values(shape, pre_tfrmd_layout,
///                                          type, NO_CONCAT, false, values);
/// \endcode
/// Setup Output tensor
/// \code
/// ztensor *zt = alloc_output_ztensor(shape, pre_tfrmd_layout, type,
///                                    NO_CONCAT);
/// \endcode
///
/// \param[in] shape array of dimensions
/// \param[in] pre_tfrmd_layout pre-transformed data layout
/// \param[in] type data type
/// \param[in] zdnn_concat_info
///                     indicates the type of concatenation to use
///                     This indirectly sets the transformed ztensor layout
///                     and the number of values arrays to expect.
///
/// \return zdnn_ztensor* Pointer to a malloc'd ztensor without transformed data
///
zdnn_ztensor *alloc_output_ztensor(uint32_t *shape,
                                   zdnn_data_layouts pre_tfrmd_layout,
                                   zdnn_data_types type,
                                   zdnn_concat_info info) {
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

  zdnn_status status;
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

  // Read in ZDNN_TEST_ERROR_ELEMENT_COUNT env var if set
  // Controls the number of errors that get printed to stdout when running tests
  // default to print at most ERROR_ELEMENT_COUNT_MAX_DEFAULT (10) errors
  // per test. If ZDNN_TEST_ERROR_ELEMENT_COUNT=0 all informational output and
  // errors will be printed to stdout.
  uint64_t error_element_count_max = ERROR_ELEMENT_COUNT_MAX_DEFAULT;
  bool always_print = false;
  char *ptr = NULL;
  if ((ptr = getenv(ENVVAR_TEST_ERROR_COUNT))) {
    error_element_count_max = (uint64_t)strtoull(ptr, NULL, 10);
    if (error_element_count_max == 0) {
      always_print = true;
    }
  }

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
    // Handle INT32 case first, since it does not require a conversion.
    if (pre_tfrmd_desc->type == INT32) {
      ((uint32_t *)expected_vals)[i] = ((uint32_t *)values)[i];
      continue;
    }

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
  char *info_fmt =
      "Element %" PRIu64 " == %f expecting %f (within tolerance)\n";
  char *info_fmt_int32 =
      "Element %" PRIu64 " == %u expecting %u (within tolerance)\n";
  char *error_fmt = "Element %" PRIu64 " == %f expecting %f";
  char *error_fmt_int32 = "Element %" PRIu64 " == %u expecting %u";
  char *error_fmt2 =
      " <==== FAILED (diff beyond ULPs %u, epsilon multiplier %u)\n";

  uint64_t error_count = 0;
  // Compared the actual and expected values
  for (uint64_t i = 0; i < num_elements; i++) {
    bool is_almost_equal = false;
    // new line at beginning of each test
    if (i == 0) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), "\n");
    }

    switch (pre_tfrmd_desc->type) {
    case BFLOAT: {
      uint16_t actual = ((uint16_t *)actual_vals)[i];
      uint16_t expected = ((uint16_t *)expected_vals)[i];

      is_almost_equal = almost_equal_bfloat_adv(actual, expected, tol);
      if (!is_almost_equal) {
        // Test not within tolerance
        if (error_count <= error_element_count_max || always_print) {
          // print test failed if error_count < error_element_count_max
          // or always_print=true (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
          // prints message like: Element xxxx == xxxx expecting xxxx
          snprintf(error_msg + strlen(error_msg),
                   big_error_message_size - strlen(error_msg), error_fmt, i,
                   cnvt_1_bfloat_to_fp32(actual),
                   cnvt_1_bfloat_to_fp32(expected));
        }
        error_count++;
      } else if (always_print) {
        // Test within tolerance
        // Output informational message only if always_print=true
        // (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
        // prints message like: Element xxxx == xxxx expecting xxxx (within
        // tolerance)
        snprintf(error_msg + strlen(error_msg),
                 big_error_message_size - strlen(error_msg), info_fmt, i,
                 cnvt_1_bfloat_to_fp32(actual),
                 cnvt_1_bfloat_to_fp32(expected));
      }

      LOG_DEBUG(error_fmt, i, cnvt_1_bfloat_to_fp32(actual),
                cnvt_1_bfloat_to_fp32(expected));
      break;
    }
    case FP16: {
      uint16_t actual = ((uint16_t *)actual_vals)[i];
      uint16_t expected = ((uint16_t *)expected_vals)[i];

      is_almost_equal = almost_equal_fp16_adv(actual, expected, tol);
      if (!is_almost_equal) {
        // Test not within tolerance
        if (error_count < error_element_count_max || always_print) {
          // print test failed if error_count < error_element_count_max
          // or always_print=true (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
          // prints message like: Element xxxx == xxxx expecting xxxx
          snprintf(error_msg + strlen(error_msg),
                   big_error_message_size - strlen(error_msg), error_fmt, i,
                   cnvt_1_fp16_to_fp32(actual), cnvt_1_fp16_to_fp32(expected));
        }
        error_count++;
      } else if (always_print) {
        // Test within tolerance
        // Output informational message only if always_print=true
        // (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
        // prints message like: Element xxxx == xxxx expecting xxxx (within
        // tolerance)
        snprintf(error_msg + strlen(error_msg),
                 big_error_message_size - strlen(error_msg), info_fmt, i,
                 cnvt_1_fp16_to_fp32(actual), cnvt_1_fp16_to_fp32(expected));
      }

      LOG_DEBUG(error_fmt, i, cnvt_1_fp16_to_fp32(actual),
                cnvt_1_fp16_to_fp32(expected));
      break;
    }

    case FP32: {
      float actual = ((float *)actual_vals)[i];
      float expected = ((float *)expected_vals)[i];

      is_almost_equal = almost_equal_float_adv(actual, expected, tol);
      if (!is_almost_equal) {
        // Test not within tolerance
        if (error_count < error_element_count_max || always_print) {
          // print test failed if error_count < error_element_count_max
          // or always_print=true (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
          // prints message like: Element xxxx == xxxx expecting xxxx
          snprintf(error_msg + strlen(error_msg),
                   big_error_message_size - strlen(error_msg), error_fmt, i,
                   actual, expected);
        }
        error_count++;

      } else if (always_print) {
        // Test within tolerance
        // Output informational message only if always_print=true
        // (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
        // prints message like: Element xxxx == xxxx expecting xxxx (within
        // tolerance)
        snprintf(error_msg + strlen(error_msg),
                 big_error_message_size - strlen(error_msg), info_fmt, i,
                 actual, expected);
      }

      LOG_DEBUG(error_fmt, i, actual, expected);
      break;
    }
    case INT32: {
      uint32_t actual = ((uint32_t *)actual_vals)[i];
      uint32_t expected = ((uint32_t *)expected_vals)[i];

      is_almost_equal = (actual == expected);
      if (!is_almost_equal) {
        // Test not within tolerance
        if (error_count <= error_element_count_max || always_print) {
          // print test failed if error_count < error_element_count_max
          // or always_print=true (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
          // prints message like: Element xxxx == xxxx expecting xxxx
          snprintf(error_msg + strlen(error_msg),
                   big_error_message_size - strlen(error_msg), error_fmt_int32,
                   i, actual, expected);
        }
        error_count++;

      } else if (always_print) {
        // Test within tolerance
        // Output informational message only if always_print=true
        // (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
        // prints message like: Element xxxx == xxxx expecting xxxx (within
        // tolerance)
        snprintf(error_msg + strlen(error_msg),
                 big_error_message_size - strlen(error_msg), info_fmt_int32, i,
                 actual, expected);
      }

      LOG_DEBUG(error_fmt_int32, i, actual, expected);
      break;
    }
    default:
      // would have died earlier
      break;
    }
    // Only print when not within tolerance and error_count <=
    // error_element_count_max OR always_print (ZDNN_TEST_ERROR_ELEMENT_COUNT=0)
    // is true. Prints message like:
    // <==== FAILED (diff beyond ULPs X, epsilon multiplier X)
    if ((!is_almost_equal) &&
        (error_count <= error_element_count_max || always_print)) {
      all_pass = false;
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt2, tol.ulps,
               tol.epsilon_mult);
    }
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

  // Create ztensor_list to handle the multple input ztensors passed in.
  va_list ztensor_list;
  va_start(ztensor_list, num_ztensors);

  // Free data buffer for each provided ztensor
  for (uint32_t i = 0; i < num_ztensors; i++) {
    zdnn_status status;
    zdnn_ztensor *ztensor = va_arg(ztensor_list, zdnn_ztensor *);
    if ((status = zdnn_free_ztensor_buffer(ztensor)) != ZDNN_OK) {
      TEST_FAIL_MESSAGE_FORMATTED(
          "zdnn_free_ztensor_buffer() failed on tensor %u with status %08x", i,
          status);
    }
    free(ztensor->transformed_desc);
    free(ztensor->pre_transformed_desc);
    free(ztensor);
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
    default:
      LOG_WARN("Unknown data type: %d", dtype);
    }
  }

  return data;
}

/// Allocates a data buffer then fills it with random INT8 values
///
/// \param[out] ztensor A zdnn tensor
///
/// \return pointer to filled data buffer
///
int8_t *create_and_fill_random_int8_data(zdnn_ztensor *ztensor) {

  uint64_t num_elements = get_num_elements(ztensor, ELEMENTS_PRE_SINGLE_GATE);
  int8_t *data = (int8_t *)malloc(num_elements);

  struct timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec);

  int upper = 127, lower = -128;
  for (int i = 0; i < num_elements; i++) {
    data[i] = (rand() % (upper - lower + 1)) + lower;
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
 * Helper that generates random negative floats and populate the given
 * array. This will be used for populating tensor buffers in the end-to-end
 * unit tests.
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
 * Helper that generates random floats and populate the given array. This will
 * be used for populating tensor buffers in the end-to-end unit tests.
 *
 * https://stackoverflow.com/questions/13408990/how-to-generate-random-float-number-in-c
 */
void gen_random_float_array_range(int size, float arr[], float min, float max) {
  struct timeval t1;
  gettimeofday(&t1, NULL);
  srand(t1.tv_usec * t1.tv_sec);

  // The raw output value will be [min, max].
  for (int i = 0; i < size; i++) {
    arr[i] = min + ((float)rand() / (float)(RAND_MAX)) * (max - min);
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
void copy_to_array(int size, const float input[], float output[]) {
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

/**
 * Helper that recieves a function pointer to some function that estimates a
 * value. For example, this could be the GeLu approximator function. This will
 * calculate the expected results based on the input values passed.
 */
void generate_expected_output(float (*fn)(float), float input_values[],
                              int num_values, float expected_values[]) {
  for (int i = 0; i < num_values; i++) {
    expected_values[i] = fn(input_values[i]);
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

#define NUM_ALL_PRE_TFRMD_TYPES 5
zdnn_data_types all_pre_tfrmd_types[NUM_ALL_PRE_TFRMD_TYPES] = {
    INT8, INT32, FP16, FP32, BFLOAT};

#define NUM_DLFLOAT16_PRE_TFRMD_TYPES 3
zdnn_data_types dlfloat_pre_tfrmd_types[NUM_DLFLOAT16_PRE_TFRMD_TYPES] = {
    FP16, FP32, BFLOAT};

#define NUM_QUANTIZED_PRE_TFRMD_TYPES 1
zdnn_data_types quantized_pre_tfrmd_types[NUM_QUANTIZED_PRE_TFRMD_TYPES] = {
    INT8};

#define NUM_INDEX_PRE_TFRMD_TYPES 1
zdnn_data_types index_pre_tfrmd_types[NUM_INDEX_PRE_TFRMD_TYPES] = {INT32};

#define NUM_ALL_TFRMD_TYPES 4
zdnn_data_types all_tfrmd_types[NUM_ALL_PRE_TFRMD_TYPES] = {
    ZDNN_DLFLOAT16, ZDNN_BINARY_FP32, ZDNN_BINARY_INT8, ZDNN_BINARY_INT32};

#define NUM_DLFLOAT16_TFRMD_TYPES 1
zdnn_data_types dlfloat_tfrmd_types[NUM_DLFLOAT16_TFRMD_TYPES] = {
    ZDNN_DLFLOAT16};

#define NUM_QUANTIZED_TFRMD_TYPES 1
zdnn_data_types quantized_tfrmd_types[NUM_QUANTIZED_TFRMD_TYPES] = {
    ZDNN_BINARY_INT8};

#define NUM_INDEX_TFRMD_TYPES 1
zdnn_data_types index_tfrmd_types[NUM_INDEX_TFRMD_TYPES] = {ZDNN_BINARY_INT32};

// indicates which data-type UnityDefaultTestRunWith*DataType() is currently
// testing
zdnn_data_types test_datatype = 128; // set initial value to something invalid

// Wrapper of Unity's UnityDefaultTestRun() that runs func() against all
// input data-types. Uses CamelCase intentionally to align with Unity.
// Function for All, DLFloat16, Quantized, and Index pre-transformed types.
void UnityDefaultTestRunWithAllPreDataType(UnityTestFunction Func,
                                           const char *FuncName,
                                           const int FuncLineNum) {
  for (int i = 0; i < NUM_ALL_PRE_TFRMD_TYPES; i++) {
    test_datatype = all_pre_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(all_pre_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

void UnityDefaultTestRunWithDLFloat16PreDataType(UnityTestFunction Func,
                                                 const char *FuncName,
                                                 const int FuncLineNum) {
  for (int i = 0; i < NUM_DLFLOAT16_PRE_TFRMD_TYPES; i++) {
    test_datatype = dlfloat_pre_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(dlfloat_pre_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

void UnityDefaultTestRunWithQuantizedPreDataType(UnityTestFunction Func,
                                                 const char *FuncName,
                                                 const int FuncLineNum) {
  for (int i = 0; i < NUM_QUANTIZED_PRE_TFRMD_TYPES; i++) {
    test_datatype = quantized_pre_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(quantized_pre_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

void UnityDefaultTestRunWithIndexPreDataType(UnityTestFunction Func,
                                             const char *FuncName,
                                             const int FuncLineNum) {
  for (int i = 0; i < NUM_INDEX_PRE_TFRMD_TYPES; i++) {
    test_datatype = index_pre_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(index_pre_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

// UnityDefaultTestRunWithAllPreDataType() but with all transformed data-types
void UnityDefaultTestRunWithAllTfrmdDataType(UnityTestFunction Func,
                                             const char *FuncName,
                                             const int FuncLineNum) {
  for (int i = 0; i < NUM_ALL_TFRMD_TYPES; i++) {
    test_datatype = all_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(all_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

// UnityDefaultTestRunWithDLFloat16PreDataType() but with transformed
// data-types
void UnityDefaultTestRunWithDLFloat16TfrmdDataType(UnityTestFunction Func,
                                                   const char *FuncName,
                                                   const int FuncLineNum) {
  for (int i = 0; i < NUM_DLFLOAT16_TFRMD_TYPES; i++) {
    test_datatype = dlfloat_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(dlfloat_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

// UnityDefaultTestRunWithQuantizedPreDataType() but with transformed data-types
// cppcheck-suppress unusedFunction
void UnityDefaultTestRunWithQuantizedTfrmdDataType(UnityTestFunction Func,
                                                   const char *FuncName,
                                                   const int FuncLineNum) {
  for (int i = 0; i < NUM_QUANTIZED_TFRMD_TYPES; i++) {
    test_datatype = quantized_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(quantized_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

// UnityDefaultTestRunWithIndexPreDataType() but with transformed data-types
// cppcheck-suppress unusedFunction
void UnityDefaultTestRunWithIndexTfrmdDataType(UnityTestFunction Func,
                                               const char *FuncName,
                                               const int FuncLineNum) {
  for (int i = 0; i < NUM_INDEX_TFRMD_TYPES; i++) {
    test_datatype = index_tfrmd_types[i];

    // FuncNameWithDataType is FuncName + " (data-type)" for printing
    char FuncNameWithDataType[FUNCNAME_BANNER_LENGTH];
    Unity.CurrentTestName = FuncNameWithDataType;
    snprintf(FuncNameWithDataType, FUNCNAME_BANNER_LENGTH, "%s (%s)", FuncName,
             get_data_type_str(index_tfrmd_types[i]));

    UnityDefaultTestRun(Func, FuncNameWithDataType, FuncLineNum);
  }
}

bool isTelumI() {
  return (zdnn_is_nnpa_installed() && (zdnn_is_nnpa_parmblk_fmt_installed(
                                           1, NNPA_PARMBLKFORMAT_1) == false));
}