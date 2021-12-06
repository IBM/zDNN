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

#include "zdnn.h"
#include "zdnn_private.h"

#include <stdarg.h>
#include <string.h>

#ifdef __MVS__
#pragma export(zdnn_init_pre_transformed_desc)
#pragma export(zdnn_generate_transformed_desc)
#pragma export(zdnn_generate_transformed_desc_concatenated)
#endif

/// Verify if the input zdnn_tensor_desc contains valid pre-transformed type and
/// layout.  dim variables are NOT checked.
///
/// \param[in] input Pointer to the zdnn_tensor_desc being checked
///
/// \return ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_TYPE
///         ZDNN_OK
///
zdnn_status
verify_pre_transformed_descriptor(const zdnn_tensor_desc *pre_tfrmd_desc) {

  // is the layout valid as pre-transformed?
  switch (pre_tfrmd_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_4DS:
  case ZDNN_NHWC:
  case ZDNN_NCHW:
  case ZDNN_HWCK:
    // all of these are good cases
    break;
  default:
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Invalid layout: %d (%s)",
                       pre_tfrmd_desc->layout,
                       get_data_layout_str(pre_tfrmd_desc->layout));
  }

  // is data type valid as pre-transformed?
  switch (pre_tfrmd_desc->type) {
  case BFLOAT:
  case FP16:
  case FP32:
    // all of these are good cases
    break;
  default:
    return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid type: %d (%s)",
                       pre_tfrmd_desc->type,
                       get_data_type_str(pre_tfrmd_desc->type));
  }

  return ZDNN_STATUS_OK;
}

/// Verify if the input zdnn_tensor_desc contains valid transformed information
///
/// \param[in] input Pointer to the zdnn_tensor_desc being checked
///
/// \return ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_TYPE
///
zdnn_status verify_transformed_descriptor(const zdnn_tensor_desc *tfrmd_desc) {

  // First, format must be valid (defined in the enum)
  // Then if format doesn't agree with layout, we declare format is correct and
  // layout is wrong (in reality, either can be wrong, but we have to pick one)
  switch (tfrmd_desc->format) {
  case ZDNN_FORMAT_4DFEATURE:
    switch (tfrmd_desc->layout) {
    case ZDNN_NHWC:
    case ZDNN_FICO:
    case ZDNN_ZRH:
    case ZDNN_BIDIR_FICO:
    case ZDNN_BIDIR_ZRH:
      break;
    default:
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Format is %s but layout is %s",
                         get_data_format_str(tfrmd_desc->format),
                         get_data_layout_str(tfrmd_desc->layout));
    }
    break;
  case ZDNN_FORMAT_4DKERNEL:
    if (tfrmd_desc->layout != ZDNN_HWCK) {
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Format is %s but layout is %s",
                         get_data_format_str(tfrmd_desc->format),
                         get_data_layout_str(tfrmd_desc->layout));
    }
    break;
  default:
    // unrecognized
    return ZDNN_STATUS(ZDNN_INVALID_FORMAT, "Invalid format: %d (%s)",
                       tfrmd_desc->format,
                       get_data_format_str(tfrmd_desc->format));
  }

  // for right now only ZDNN_DLFLOAT16 is valid
  if (tfrmd_desc->type != ZDNN_DLFLOAT16) {
    return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid type: %d (%s)",
                       tfrmd_desc->type, get_data_type_str(tfrmd_desc->type));
  }

  const uint32_t *dims_ptr = &(tfrmd_desc->dim4);

  // is the dimension above the limit or zero?
  // transformed layout uses all dim* entries, so we'll check them all
  for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
    LOG_DEBUG("dim%d: %d", ZDNN_MAX_DIMS - i, dims_ptr[i]);
    if (!dims_ptr[i] || dims_ptr[i] > zdnn_get_nnpa_max_dim_idx_size()) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "Invalid shape: %d (reason: exceeds %d or is 0)",
                         dims_ptr[i], zdnn_get_nnpa_max_dim_idx_size());
    }
  }

  // is stick area size above the limit?
  if (zdnn_getsize_ztensor(tfrmd_desc) > zdnn_get_nnpa_max_tensor_size()) {
    return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                       "Invalid shape (reasons: tensor size: %" PRIu64
                       ", maximum: %" PRIu64 " bytes",
                       zdnn_getsize_ztensor(tfrmd_desc),
                       zdnn_get_nnpa_max_tensor_size());
  }

  return ZDNN_STATUS_OK;
}

/// Convenience function for populating zdnn_tensor_desc with pre-transformed
/// shape information
///
/// \param[in] layout data layout
/// \param[in] type data type
/// \param[out] pre_tfrmd_desc Pointer to zdnn_tensor_desc struct
/// \param[in] ... Number of elements in each dimension, in outermost to
///                innermost order
///
/// \return None
///
void zdnn_init_pre_transformed_desc(zdnn_data_layouts layout,
                                    zdnn_data_types type,
                                    zdnn_tensor_desc *pre_tfrmd_desc, ...) {

  // point to dim4/3/etc via pointer.  they're guaranteed to be in the correct
  // order as written and contiguous and correctly aligned
  uint32_t *dims_ptr = &(pre_tfrmd_desc->dim4);

  va_list v_list;
  va_start(v_list, pre_tfrmd_desc);

  if (pre_tfrmd_desc) {
    // unused dim* vars in pre-transformed descriptor are left alone
    for (int i = ZDNN_MAX_DIMS - get_data_layout_dims(layout);
         i < ZDNN_MAX_DIMS; i++) {
      dims_ptr[i] = va_arg(v_list, uint32_t);
    }
    pre_tfrmd_desc->layout = layout;
    pre_tfrmd_desc->type = type;
  }

  va_end(v_list);
}

/// Convenience function for populating zdnn_tensor_desc with transformed
/// information, for INTERNAL USE only.  .format is NOT set in this routine.
///
/// \param[in] layout data layout
/// \param[in] type data type
/// \param[in] format NNPA data format
/// \param[out] pre_tfrmd_desc Pointer to zdnn_tensor_desc struct
/// \param[in] dim4 number of elements in outermost
/// \param[in] dim3 number of elements
/// \param[in] dim2 number of elements
/// \param[in] dim1 number of elements in innermost
///
/// \return None
///
void init_transformed_desc(zdnn_data_layouts layout, zdnn_data_types type,
                           zdnn_data_formats format,
                           zdnn_tensor_desc *tfrmd_desc, uint32_t dim4,
                           uint32_t dim3, uint32_t dim2, uint32_t dim1) {

  // piggyback on zdnn_init_pre_transformed_desc() for now
  zdnn_init_pre_transformed_desc(layout, type, tfrmd_desc, dim4, dim3, dim2,
                                 dim1);
  tfrmd_desc->format = format;
}

/// Convenience function for slicing a ztensor along dim4. The contents of the
/// input ztensor and its descriptors are copied into the output pointers. Then
/// the output's structs are updated to reflect values for a single slice. The
/// input buffer values are not copied. Instead the output's buffer pointer is
/// adjusted so it points the the correct address of the existing data.
///
/// \param[in] input_ztensor pointer to original unsliced ztensor
/// \param[in] slice_idx dim4 index to use as output slice
/// \param[in] slice_buffer_size size of a sliced buffer. If 0, method will
///                          calculate based on number of elements and data type
/// \param[out] output_pre_tfrmd_desc pointer to pre_tfrmd_desc to edit for
///                                   output (skipped if NULL)
/// \param[out] output_tfrmd_desc pointer to tfrmd_desc to edit for output.
/// \param[out] output_ztensor pointer ztensor to edit for output slice.
///
/// \return ZDNN_INVALID_LAYOUT (where applicable when output_pre_tfrmd_desc is
///                              not NULL)
///         ZDNN_STATUS_OK
///
zdnn_status ztensor_slice_dim4(const zdnn_ztensor *input_ztensor,
                               uint32_t slice_idx, size_t slice_buffer_size,
                               zdnn_tensor_desc *output_pre_tfrmd_desc,
                               zdnn_tensor_desc *output_tfrmd_desc,
                               zdnn_ztensor *output_ztensor) {

  // Copy the input ztensor info into output and set output descriptor pointers
  memcpy(output_ztensor, input_ztensor, sizeof(zdnn_ztensor));
  output_ztensor->pre_transformed_desc = output_pre_tfrmd_desc;
  output_ztensor->transformed_desc = output_tfrmd_desc;

  // Copy the input ztensor descriptors into output descriptors
  memcpy(output_tfrmd_desc, input_ztensor->transformed_desc,
         sizeof(zdnn_tensor_desc));

  // set up pre-transformed desctprors for the sliced output only if caller
  // cares about it and gave us a space for it
  if (output_pre_tfrmd_desc) {

    memcpy(output_pre_tfrmd_desc, input_ztensor->pre_transformed_desc,
           sizeof(zdnn_tensor_desc));

    // Set the output ztensor dim values to reflect the slicing
    switch (input_ztensor->pre_transformed_desc->layout) {
    case ZDNN_2DS:
      output_ztensor->pre_transformed_desc->dim2 = 1;
      break;
    case ZDNN_3DS:
      output_ztensor->pre_transformed_desc->dim3 = 1;
      break;
    case ZDNN_4D:
    case ZDNN_NHWC:
    case ZDNN_NCHW:
      output_ztensor->pre_transformed_desc->dim4 = 1;
      break;
    default:
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Invalid layout for slicing: %d",
                         input_ztensor->transformed_desc->layout);
      break;
    }
  }

  output_ztensor->transformed_desc->dim4 = 1;

  // Check these after we check the layout so we issue better error messages.
  // Otherwise 1D, 2D, 3D, etc would emit ZDNN_INVALID_SHAPE for dim4 == 1
  if (input_ztensor->transformed_desc->dim4 < 2) {
    return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                       "Invalid shape for slicing: transformed_desc->dim4 must "
                       "be greater than one",
                       NO_ARG);
  } else if (slice_idx + 1 > input_ztensor->transformed_desc->dim4) {
    return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                       "Invalid shape for slicing: transformed_desc->dim4 (%d) "
                       "does not support a slice index of %d",
                       input_ztensor->transformed_desc->dim4, slice_idx);
  }

  // We need the exact buffer_size so we slice the buffer correctly. If given,
  // use the specified size, otherwise calculate it now.
  if (slice_buffer_size) {
    output_ztensor->buffer_size = slice_buffer_size;
    LOG_DEBUG("slice buffer_size set to %" PRIu64
              " by specified slice_buffer_size",
              output_ztensor->buffer_size);
  } else {
    output_ztensor->buffer_size =
        zdnn_getsize_ztensor(output_ztensor->transformed_desc);
    LOG_DEBUG("slice buffer_size set to %" PRIu64 " by zdnn_getsize_ztensor()",
              output_ztensor->buffer_size);
  }

  // Set output ztensor buffer address to where the slice starts from input
  output_ztensor->buffer = (void *)((uintptr_t)input_ztensor->buffer +
                                    (slice_idx * output_ztensor->buffer_size));

  return ZDNN_STATUS_OK;
}

/// Generate transformed tensor descriptor based on supplied pre-transformed
/// tensor descriptor
///
/// \param[in] pre_tfrmd_desc Pointer to zdnn_tensor_desc struct with
///                           pre-transformed information
/// \param[out] tfrmd_desc
///                 Pointer to zdnn_tensor_desc struct where transformed
///                 information will be stored
///
/// \return ZDNN_OK
///         ZDNN_INVALID_LAYOUT
///
///
zdnn_status
zdnn_generate_transformed_desc(const zdnn_tensor_desc *pre_tfrmd_desc,
                               zdnn_tensor_desc *tfrmd_desc) {

  zdnn_status status;

  // modify tfrmd_desc only if layout is supported, else leave it untouched

  switch (pre_tfrmd_desc->layout) {
  case (ZDNN_1D):
    // shape (a) -> dims4-1 (1, 1, 1, a)
    tfrmd_desc->dim4 = 1;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = 1;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_2D):
    // shape (a, b) -> dims4-1 (1, 1, a, b)
    tfrmd_desc->dim4 = 1;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_2DS):
    // shape (a, b) -> dims4-1 (a, 1, 1, b)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = 1;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_3D):
    // shape (a, b, c) -> dims4-1 (1, a, b, c)
    tfrmd_desc->dim4 = 1;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_3DS):
    // shape (a, b, c) -> dims4-1 (a, 1, b, c)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_4D):
  case (ZDNN_NHWC):
    // shape (a, b, c, d) -> dims4-1 (a, b, c, d)
    // shape (n, h, w, c) -> dims4-1 (n, h, w, c)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_4DS):
    // ZDNN_4DS is used exclusively as RNN output
    // shape (a, b, c, d)  -> ZDNN_NHWC
    //   when b = 1 (uni-dir)     -> dims4-1 (a, 1, c, d)
    //   otherwise (bi-dir, etc.) -> dims4-1 (a, 1, c, b * PADDED(d))
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = 1;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    if (pre_tfrmd_desc->dim3 == 1) {
      tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    } else {
      // so when dim3 is 0 for whatever reason, tfrmd_desc->dim1 will become 0
      // and will fail transform-desc check later
      tfrmd_desc->dim1 = pre_tfrmd_desc->dim3 * PADDED(pre_tfrmd_desc->dim1);
    }
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case (ZDNN_NCHW):
    // shape (n, c, h, w) -> dims4-1 (n, h, w, c)
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim1;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim3;
    tfrmd_desc->layout = ZDNN_NHWC;
    tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;
    status = ZDNN_OK;
    break;
  case ZDNN_HWCK:
    tfrmd_desc->dim4 = pre_tfrmd_desc->dim4;
    tfrmd_desc->dim3 = pre_tfrmd_desc->dim3;
    tfrmd_desc->dim2 = pre_tfrmd_desc->dim2;
    tfrmd_desc->dim1 = pre_tfrmd_desc->dim1;
    tfrmd_desc->layout = ZDNN_HWCK;
    tfrmd_desc->format = ZDNN_FORMAT_4DKERNEL;
    status = ZDNN_OK;
    break;
  default:
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT, "Invalid layout: %d (%s)",
                       pre_tfrmd_desc->layout,
                       get_data_layout_str(pre_tfrmd_desc->layout));
    break;
  }

  if (status == ZDNN_OK) {
    tfrmd_desc->type = ZDNN_DLFLOAT16;
  }

  return status;
}

/// Generate concatenated transformed tensor descriptor based on supplied
/// pre-transformed tensor descriptor
///
/// \param[in] pre_tfrmd_desc
///                  Pointer to zdnn_tensor_desc struct with pre-transformed
///                  information
/// \param[in] info
///                  Concatenation information
/// \param[out] tfrmd_desc
///                  Pointer to zdnn_tensor_desc struct where transformed
///                  information will be stored
///
/// \return ZDNN_OK
///          ZDNN_INVALID_LAYOUT
///          ZDNN_INVALID_CONCAT_INFO
///          ZDNN_INVALID_SHAPE
///
zdnn_status zdnn_generate_transformed_desc_concatenated(
    const zdnn_tensor_desc *pre_tfrmd_desc, zdnn_concat_info info,
    zdnn_tensor_desc *tfrmd_desc) {

  if ((CONCAT_USAGE(info) == USAGE_WEIGHTS) &&
      (CONCAT_PREV_LAYER(info) == PREV_LAYER_BIDIR)) {
    // dim2 can't be odd number
    if (pre_tfrmd_desc->dim2 & 1) {
      return ZDNN_STATUS(
          ZDNN_INVALID_SHAPE,
          "when PREV_LAYER_BIDIR and USAGE_WEIGHTS, pre-transformed "
          "dim2 must be multiples of 2 (found: %d)",
          pre_tfrmd_desc->dim2);
    }
  }

  // Two kinds of concatenations we need to deal with:
  //
  // - (Hidden-)Weights, (hidden)-biases need to be concatenated horizontally,
  //   new dim1 is calculated via get_rnn_concatenated_dim1()
  //
  // - Weights may need to be concatenated vertically also (when output
  //   from the previous bidir layer is the input), new dim2 is calculated via
  //   get_rnn_concatenated_dim2()

  if ((CONCAT_USAGE(info) == USAGE_BIASES) ||
      (CONCAT_USAGE(info) == USAGE_HIDDEN_BIASES)) {
    if (pre_tfrmd_desc->layout == ZDNN_2DS) {
      tfrmd_desc->dim4 = pre_tfrmd_desc->dim2;
      tfrmd_desc->dim3 = 1;
      tfrmd_desc->dim2 = 1;
      tfrmd_desc->dim1 = get_rnn_concatenated_dim1(pre_tfrmd_desc->dim1, info);
    } else {
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                         "Pre-transformed layout not ZDNN_2DS (found: %s)",
                         get_data_layout_str(pre_tfrmd_desc->layout));
    }
  } else if ((CONCAT_USAGE(info) == USAGE_WEIGHTS) ||
             (CONCAT_USAGE(info) == USAGE_HIDDEN_WEIGHTS)) {
    if (pre_tfrmd_desc->layout == ZDNN_3DS) {
      tfrmd_desc->dim4 = pre_tfrmd_desc->dim3;
      tfrmd_desc->dim3 = 1;
      tfrmd_desc->dim2 = get_rnn_concatenated_dim2(pre_tfrmd_desc->dim2, info);
      tfrmd_desc->dim1 = get_rnn_concatenated_dim1(pre_tfrmd_desc->dim1, info);
    } else {
      return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                         "Pre-transformed layout not ZDNN_3DS (found: %s)",
                         get_data_layout_str(pre_tfrmd_desc->layout));
    }
  } else {
    return ZDNN_STATUS(ZDNN_INVALID_CONCAT_INFO,
                       "Invalid usage in concatenation info: %08x", info);
  }

  // if USAGE is WEIGHTS and PREV_LAYER is BIDIR then
  // ZDNN_BIDIR_FICO/ZDNN_BIDIR_ZRH
  //
  // everything else ZDNN_FICO/ZDNN_ZRH

  if ((CONCAT_USAGE(info) == USAGE_WEIGHTS) &&
      (CONCAT_PREV_LAYER(info) == PREV_LAYER_BIDIR)) {
    if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
      tfrmd_desc->layout = ZDNN_BIDIR_FICO;
    } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
      tfrmd_desc->layout = ZDNN_BIDIR_ZRH;
    } else {
      return ZDNN_STATUS(ZDNN_INVALID_CONCAT_INFO,
                         "Invalid RNN type in concatenation info: %08x", info);
    }
  } else {
    if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
      tfrmd_desc->layout = ZDNN_FICO;
    } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
      tfrmd_desc->layout = ZDNN_ZRH;
    } else {
      return ZDNN_STATUS(ZDNN_INVALID_CONCAT_INFO,
                         "Invalid RNN type in concatenation info: %08x", info);
    }
  }

  tfrmd_desc->type = ZDNN_DLFLOAT16;
  tfrmd_desc->format = ZDNN_FORMAT_4DFEATURE;

  return ZDNN_STATUS_OK;
}
