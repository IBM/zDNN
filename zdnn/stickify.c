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

#include <fenv.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_transform_ztensor)
#pragma export(zdnn_transform_origtensor)
#pragma export(zdnn_transform_quantized_ztensor)
#pragma export(zdnn_transform_ztensor_with_saturation)
#endif

/// Prefetch a memory block for reading
///
/// \param[in] ptr pointer to the base address of the memory block
/// \param[in] offset offset from the base address to the memory block to
/// prefetch
///
static inline void prefetch_read(const void *ptr, uintptr_t offset) {
#if defined(__MVS__)
  __dcbt((void *)((uintptr_t)ptr + offset));
#else
  __builtin_prefetch((void *)((uintptr_t)ptr + offset), 0);
#endif
}
/// Prefetch a memory block for for writing
///
/// \param[in] ptr pointer to the base address of the memory block
/// \param[in] offset offset from the base address to the memory block to
/// prefetch
///
static inline void prefetch_write(const void *ptr, uintptr_t offset) {
#if defined(__MVS__)
  __dcbtst((void *)((uintptr_t)ptr + offset));
#else
  __builtin_prefetch((void *)((uintptr_t)ptr + offset), 1);
#endif
}
/// Flush a memory block from cache
///
/// \param[in] ptr pointer to the base address of the memory block
/// \param[in] offset from the base address to the memory block to flush
///
static inline void cache_flush(const void *ptr, uintptr_t offset) {
#if defined(__MVS__)
  __dcbf((void *)((uintptr_t)ptr + offset));
#else
  // No equivalent for non-MVS, leave empty or add necessary code
#endif
}

/// Calculate stride n size for the tensor. When stride n size >
/// STICK_SW_THRESHOLD use hardware stickification otherwise stay in software
/// stickification as this shows the greatest performance benefit.
///
/// \param[in] ztensor pointer to zdnn_ztensor_desc
///
/// \return true if stride n size > STICK_SW_THRESHOLD else false
static inline bool
n_stride_meets_hardware_limit(const zdnn_tensor_desc *tensor) {
  return ((uint64_t)tensor->dim3 * (uint64_t)tensor->dim2 *
          (uint64_t)tensor->dim1) > STICK_SW_THRESHOLD;
}

// resultant vector from vec_perm() is 16 bytes.  we're merging entries from 2
// sticks so we grab max 8 entries from each when int8, or 4 entries when bfloat
#define VECPERM_MAX_INT8_ENTRIES 8
#define VECPERM_MAX_BFLOAT_ENTRIES 4

#define BEGIN_PRINT_PARMS()                                                    \
  printf("\n%s parameters start "                                              \
         ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",               \
         __func__)

#define PRINT_PARM_ZTENSOR_PTR(ztnsr) print_ztensor(ztnsr, #ztnsr, false)
#define PRINT_PARM_BOOL(val)                                                   \
  printf("\nParameter %s (bool): %s\n", #val, val ? "true" : "false");
#define END_PRINT_PARMS()                                                      \
  printf("\n%s parameters end "                                                \
         "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n",               \
         __func__)

/// Return the byte offset of the field in the stick array, based on the input
/// fields indexes, and the overall dimensions of the input tensor. The use of
/// e4x,e3x, etc is to reflect the four dimensions in the NNPA control block
/// E4,E3,E2,E1
///
/// \param[in] e4x outermost dimension
/// \param[in] e3x e3 dimension
/// \param[in] e2x e2 dimension
/// \param[in] e1x innermost dimension
/// \param[in] desc tensor descriptor that contains the original shape
/// information
///
///
/// \return Byte offset of the field in the stick array, or 0 if error
///
size_t get_stick_offset(uint32_t e4x, uint32_t e3x, uint32_t e2x, uint32_t e1x,
                        const zdnn_tensor_desc *pre_tfrmd_desc) {

  if (pre_tfrmd_desc->layout != ZDNN_HWCK) {
    // Stickified feature tensor elements follow the NHWC layout,
    // so use the n, h, w, c notation for easier read.

    uint32_t h = 1, w = 1, c = 1; // n = 1,
    uint32_t nx = e4x, hx = 1, wx = 1, cx = 1;

    switch (pre_tfrmd_desc->layout) {
    case (ZDNN_1D):
    case (ZDNN_2D):
    case (ZDNN_3D):
    case (ZDNN_4D):
    case (ZDNN_NHWC): {
      h = (get_data_layout_dims(pre_tfrmd_desc->layout) >= 3)
              ? pre_tfrmd_desc->dim3
              : 1;
      w = (get_data_layout_dims(pre_tfrmd_desc->layout) >= 2)
              ? pre_tfrmd_desc->dim2
              : 1;
      c = pre_tfrmd_desc->dim1;
      hx = e3x;
      wx = e2x;
      cx = e1x;
      break;
    }
    case (ZDNN_3DS): {
      w = pre_tfrmd_desc->dim2;
      c = pre_tfrmd_desc->dim1;
      hx = e3x;
      wx = e2x;
      cx = e1x;
      break;
    }
    case (ZDNN_2DS): {
      c = pre_tfrmd_desc->dim1;
      hx = e3x;
      wx = e2x;
      cx = e1x;
      break;
    }
    case (ZDNN_NCHW): {
      h = pre_tfrmd_desc->dim2;
      w = pre_tfrmd_desc->dim1;
      c = pre_tfrmd_desc->dim3;
      cx = e3x;
      hx = e2x;
      wx = e1x;
      break;
    }
    default:
      LOG_DEBUG("get_stick_offset: Unsupported layout (%s)",
                get_data_layout_str(pre_tfrmd_desc->layout));
      return 0;
    }

    uint16_t pages_height_per_h = CEIL(w, AIU_STICKS_PER_PAGE);
    uint32_t pages_height_all_h = pages_height_per_h * h;
    uint64_t pages_per_n =
        pages_height_all_h * CEIL(c, AIU_2BYTE_CELLS_PER_STICK);

    // find out how many pages to traverse: traverse to n = nx section of the
    // stick area, then c = cx, and so forth
    uint64_t page = (pages_per_n * nx) +
                    ((cx / AIU_2BYTE_CELLS_PER_STICK) * pages_height_all_h) +
                    (hx * pages_height_per_h) + (wx / AIU_STICKS_PER_PAGE);

    // find out which stick within the page is the element at
    uint16_t stick = wx % AIU_STICKS_PER_PAGE;

    // traverse this number of cells to get to the element
    uint16_t cell = cx % AIU_2BYTE_CELLS_PER_STICK;

    BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
      uint64_t e =
          (uint64_t)e4x * (pre_tfrmd_desc->dim3 * pre_tfrmd_desc->dim2 *
                           pre_tfrmd_desc->dim1) +
          e3x * (pre_tfrmd_desc->dim2 * pre_tfrmd_desc->dim1) +
          e2x * (pre_tfrmd_desc->dim1) + e1x;

      printf("\ne4x %d e3x %d e2x %d e1x %d -> element #%" PRIu64
             " ----------------------------------------------\n",
             e4x, e3x, e2x, e1x, e);

      printf("nx %u hx %u wx %u cx %u\n", nx, hx, wx, cx);

      printf("pages_height_per_h %u pages_height_all_h %u "
             "pages_per_n %" PRIu64 "\n",
             pages_height_per_h, pages_height_all_h, pages_per_n);

      printf("(nx * pages_per_n) %" PRIu64 "\n", (nx * pages_per_n));
      printf("((cx / AIU_2BYTE_CELLS_PER_STICK) * pages_height_all_h) %u\n",
             ((cx / AIU_2BYTE_CELLS_PER_STICK) * pages_height_all_h));
      printf("(hx * pages_height_per_h) %u\n", (hx * pages_height_per_h));
      printf("(wx / AIU_STICKS_PER_PAGE) %u\n", (wx / AIU_STICKS_PER_PAGE));

      printf("page %" PRIu64 " stick %d cell %d\n", page, stick, cell);
    }

    // quantify those values in number of bytes
    return page * AIU_PAGESIZE_IN_BYTES + stick * AIU_BYTES_PER_STICK +
           cell * AIU_2BYTE_CELL_SIZE;
  } else {
    // Stickified kernel tensor elements follow the HWCK layout,
    // so use the h, w, c, k notation for easier read.

    uint32_t h = pre_tfrmd_desc->dim4, w = pre_tfrmd_desc->dim3,
             c = pre_tfrmd_desc->dim2;
    uint32_t hx = e4x, wx = e3x, cx = e2x, kx = e1x;

    uint16_t pages_height_per_w = CEIL(c, AIU_STICKS_PER_PAGE);
    uint32_t pages_height_per_h = pages_height_per_w * w;
    uint64_t pages_height_all_h = pages_height_per_h * h;

    // traverse to k = kx section of the stick area, then h = hx, then w = wx
    // it's slightly different from NHWC due to the E1/E2 arrangement
    uint64_t page = pages_height_all_h * (kx / AIU_2BYTE_CELLS_PER_STICK) +
                    hx * pages_height_per_h + wx * pages_height_per_w;

    // traverse this number of cells to get to the element
    uint16_t cell = kx % AIU_2BYTE_CELLS_PER_STICK;

    // quantify those values in number of bytes
    return page * AIU_PAGESIZE_IN_BYTES + cx * AIU_BYTES_PER_STICK +
           cell * AIU_2BYTE_CELL_SIZE;
  }
}

/// Main entry point for converting FP16/FP32/BFLOAT <-> ZDNN_DLFLOAT16 when
/// the entries to fetch/set on FP16/FP32/BFLOAT side are not contiguous (e.g.,
/// fetching the c-entries in a NCHW stream).
///
/// \param[in] input_data Pointer to input tensor data stream
/// \param[in] in_data_fmt Input tensor stream data format
/// \param[in] output_data Pointer to output tensor data stream
/// \param[in] out_data_fmt Output tensor stream data format
/// \param[in] num_fields Number of fields to convert
/// \param[in] input_stride How many fields (not bytes) the input entries are
/// apart
///
/// \return Number of fields converted, or 0 if error
///
uint32_t
convert_data_format_in_stride(void *input_data, zdnn_data_types in_data_fmt,
                              void *output_data, zdnn_data_types out_data_fmt,
                              uint32_t num_fields, uint32_t input_stride) {
  uint64_t num_fields_converted = 0;

  // we only care convert to/from ZDNN_DLFLOAT16
  if (out_data_fmt == ZDNN_DLFLOAT16) {
    switch (in_data_fmt) {
    case FP16:
      num_fields_converted = fp16_to_dlf16_in_stride((uint16_t *)input_data,
                                                     (uint16_t *)output_data,
                                                     num_fields, input_stride);
      break;
    case FP32:
      num_fields_converted =
          fp32_to_dlf16_in_stride((float *)input_data, (uint16_t *)output_data,
                                  num_fields, input_stride);
      break;
    case BFLOAT:
      num_fields_converted = bfloat_to_dlf16_in_stride(
          (uint16_t *)input_data, (uint16_t *)output_data, num_fields,
          input_stride);
      break;
    default:
      break; // something really wrong, get out and return 0
    }
  } else if (in_data_fmt == ZDNN_DLFLOAT16) {
    switch (out_data_fmt) {
    case FP16:
      num_fields_converted = dlf16_to_fp16_in_stride((uint16_t *)input_data,
                                                     (uint16_t *)output_data,
                                                     num_fields, input_stride);
      break;
    case FP32:
      num_fields_converted =
          dlf16_to_fp32_in_stride((uint16_t *)input_data, (float *)output_data,
                                  num_fields, input_stride);
      break;
    case BFLOAT:
      num_fields_converted = dlf16_to_bfloat_in_stride(
          (uint16_t *)input_data, (uint16_t *)output_data, num_fields,
          input_stride);
      break;
    default:
      break; // something really wrong, get out and return 0
    }
  } else {
    // something really wrong
    return 0;
  }
  return num_fields_converted;
}

/// Main entry point for converting FP16/FP32/BFLOAT <-> ZDNN_DLFLOAT16 when
/// the entries to fetch/set on FP16/FP32/BFLOAT side are contiguous (e.g.,
/// fetching the c-entries in a NHWC stream).
///
/// \param[in] input_data Pointer to input tensor data stream
/// \param[in] in_data_fmt Input tensor stream data format
/// \param[in] output_data Pointer to output tensor data stream
/// \param[in] out_data_fmt Output tensor stream data format
/// \param[in] num_fields Number of fields to convert
/// \param[in] saturation_func Function to use for (or skip) saturation
///
/// \return Number of fields converted, or 0 if error
///
uint32_t convert_data_format(
    void *input_data, zdnn_data_types in_data_fmt, void *output_data,
    zdnn_data_types out_data_fmt, uint32_t num_fields,
    void (*saturate_func)(const vec_float32 *, const vec_float32 *,
                          vec_float32 *, vec_float32 *)) {

  uint64_t num_fields_converted = 0;

  // we only care convert to/from ZDNN_DLFLOAT16
  if (out_data_fmt == ZDNN_DLFLOAT16) {
    switch (in_data_fmt) {
    case FP16:
      num_fields_converted = fp16_to_dlf16((uint16_t *)input_data,
                                           (uint16_t *)output_data, num_fields);
      break;
    case FP32:
      num_fields_converted =
          fp32_to_dlf16((float *)input_data, (uint16_t *)output_data,
                        num_fields, saturate_func);
      break;
    case BFLOAT:
      num_fields_converted =
          bfloat_to_dlf16((uint16_t *)input_data, (uint16_t *)output_data,
                          num_fields, saturate_func);
      break;
    default:
      // something really wrong, get out and return 0
      return 0;
    }
  } else if (in_data_fmt == ZDNN_DLFLOAT16) {
    switch (out_data_fmt) {
    case FP16:
      num_fields_converted = dlf16_to_fp16((uint16_t *)input_data,
                                           (uint16_t *)output_data, num_fields);
      break;
    case FP32:
      num_fields_converted = dlf16_to_fp32((uint16_t *)input_data,
                                           (float *)output_data, num_fields);
      break;
    case BFLOAT:
      num_fields_converted = dlf16_to_bfloat(
          (uint16_t *)input_data, (uint16_t *)output_data, num_fields);
      break;
    default:
      // something really wrong, get out and return 0
      return 0;
    }
  } else {
    // something really wrong
    return 0;
  }

  return num_fields_converted;
} // End convert_data_format

/// Handle FP Exceptions and map to ZDNN status code if necessary
///
/// \param[in] fe FP Exceptions
///
/// \return mapped ZDNN status code
///
zdnn_status handle_fp_errors(int fe) {
  if (fe & FE_UNDERFLOW) {
    LOG_WARN("Some tensor elements too small and forced to zero in "
             "target.",
             NO_ARG); // underflow (bit 11)
                      // no error externalized
  }
  if ((fe & FE_INVALID) || (fe & FE_OVERFLOW)) {
    return ZDNN_STATUS(ZDNN_CONVERT_FAILURE,
                       "Some tensor elements too large. Consider model "
                       "tuning.",
                       NO_ARG); // invalid op (bit 8) or overflow (bit 10)
  }
  if (fe & FE_INEXACT) {
    return ZDNN_STATUS(ZDNN_CONVERT_FAILURE,
                       "Internal error or Live migration happened"
                       "(Target machine has different characteristics.)",
                       NO_ARG); //  inexact (bit 12)
  }

  return ZDNN_STATUS_OK;
}

zdnn_status handle_fp_errors_saturation(int fe, zdnn_data_types type) {
  if (fe & FE_UNDERFLOW) {
    LOG_WARN("Some tensor elements too small and forced to zero in "
             "target.",
             NO_ARG); // underflow (bit 11)
                      // no error externalized
  }
  if ((fe & FE_INVALID) || (fe & FE_OVERFLOW)) {
    if (type == FP16) {
      return ZDNN_STATUS(ZDNN_CONVERT_FAILURE,
                         "Some tensor elements too large. Consider model "
                         "tuning.",
                         NO_ARG);
    }
    return ZDNN_STATUS(ZDNN_ELEMENT_RANGE_VIOLATION,
                       "Range violation on tensor data", NO_ARG);
  }
  if (fe & FE_INEXACT) {
    return ZDNN_STATUS(ZDNN_CONVERT_FAILURE,
                       "Internal error or Live migration happened"
                       "(Target machine has different characteristics.)",
                       NO_ARG); //  inexact (bit 12)
  }

  return ZDNN_STATUS_OK;
}
/// Call HW to transform from FP32 -> DLFLOAT16 only.
///
/// \param[in] in_buf data buffer to be stickified
/// \param[in] saturation_control saturation control
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
///
/// \return ZDNN_OK
///         TODO
///
zdnn_status hw_transform_ztensor(const void *in_buf, bool saturation_control,
                                 zdnn_ztensor *output) {
  // Create a temporary ztensor to point to user's data buffer
  zdnn_tensor_desc temp_desc;
  init_transformed_desc(
      ZDNN_NHWC, ZDNN_BINARY_FP32, ZDNN_FORMAT_4DGENERIC, &temp_desc,
      output->transformed_desc->dim4, output->transformed_desc->dim3,
      output->transformed_desc->dim2, output->transformed_desc->dim1);
  zdnn_ztensor temp_input;
  zdnn_init_ztensor(&temp_desc, &temp_desc, &temp_input);
  temp_input.buffer = (void *)in_buf;

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS();
    PRINT_PARM_ZTENSOR_PTR(&temp_input);
    PRINT_PARM_BOOL(saturation_control);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS();
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_transform *fsp_transform = (func_sp_parms_transform *)&fsp;
  // Setup Function Specific Parameters:
  fsp_transform->parm1.sc = saturation_control;
  fsp_transform->parm1.toc = NNPA_TOC_STICK_DLFLOAT;

  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_TRANSFORM,
                               &temp_input, NULL, NULL, output, NULL, 0, &fsp);
}

/// The actual routine for stickification, only does the following:
///    NHWC -> NHWC, NCHW -> NHWC, HWCK -> HWCK
/// Does NOT handle concatenated types.
///
/// \param[in] in_buf data buffer to be stickified
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
/// \param[in] saturation_control saturation control
///
/// \return ZDNN_OK
///         ZDNN_CONVERT_FAILURE
///
zdnn_status transform_ztensor(const void *in_buf, zdnn_ztensor *ztensor,
                              bool saturation_control) {
  uint64_t input_offset =
      0; // moving position as the input is processed, in BYTES
  uint64_t output_offset =
      0; // moving position as the output is processed, in BYTES

  short input_cell_size =
      get_data_type_size(ztensor->pre_transformed_desc->type);
  short input_cell_shift = input_cell_size / 2;

  /*
   * Stores the vector operation output directly into the stick_area.  This
   * reduces the number of inefficient loops.
   */
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  feclearexcept(
      FE_ALL_EXCEPT); /* clear exception flags set during conversion */

  // select saturation control function if requested, otherwise pass skip func
  void (*saturate_func)(const vec_float32 *, const vec_float32 *, vec_float32 *,
                        vec_float32 *) = saturation_control
                                             ? &saturate_fp32_to_dlf16
                                             : &skip_saturate_fp32_to_dlf16;
  if (ztensor->transformed_desc->layout == ZDNN_NHWC) {

    // Expected layout is NHWC, stickify normally. Requires a single data
    // buffer.

    // If FP32 and NNPA_TRANSFORM is available, send to hardware
    if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW &&
        ztensor->pre_transformed_desc->type == FP32 &&
        zdnn_is_nnpa_function_installed(1, NNPA_TRANSFORM) &&
        n_stride_meets_hardware_limit(ztensor->transformed_desc)) {
      return hw_transform_ztensor(in_buf, saturation_control, ztensor);
    }

    // loop invariant values
    uint64_t bytes_all_h =
        (uint64_t)ztensor->transformed_desc->dim3 *
        CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
        AIU_PAGESIZE_IN_BYTES;
    uint64_t bytes_per_n = bytes_all_h * CEIL(ztensor->transformed_desc->dim1,
                                              AIU_2BYTE_CELLS_PER_STICK);

    if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW) {

      // N
      for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

        // used for pushing out_offset from n to n+1 (i.e., + bytes_per_n)
        uint64_t out_offset_n = output_offset;

        // H
        for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

          // W
          for (uint32_t e2x = 0; e2x < ztensor->transformed_desc->dim2; e2x++) {
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
            prefetch_read(in_buf, input_offset);

            // used for pushing out_offset from w to w+1 (i.e., +
            // AIU_BYTES_PER_STICK)
            uint64_t out_offset_w = output_offset;

            // process each C-stick (i.e., every 64 elements or whatever
            // left in dim1)
            for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
                 e1x += AIU_2BYTE_CELLS_PER_STICK) {
              // Prefetch to L1 newest offset to write that HW wouldn't
              // know about
              prefetch_write(ztensor->buffer, output_offset);
              fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                      AIU_2BYTE_CELLS_PER_STICK);

              nbr_fields_converted = convert_data_format(
                  (void *)((uintptr_t)in_buf + input_offset),
                  ztensor->pre_transformed_desc->type,
                  (void *)((uintptr_t)ztensor->buffer + output_offset),
                  ztensor->transformed_desc->type, fields_to_convert,
                  saturate_func);

              if (nbr_fields_converted == 0)
                return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

              // Release L1 cacheline for stick. The next "touch" will be
              // from NNPA, and it doesn't need L1 caching.
              cache_flush(ztensor->buffer, output_offset);
              input_offset += (nbr_fields_converted << input_cell_shift);

              // push output_offset to the next c-stick of the same super
              // c-stick, which is bytes_all_h number of bytes away.
              output_offset += bytes_all_h;
            }

            // output_offset was pushed around in dim1 loops, so reset it to
            // the next w
            output_offset = out_offset_w + AIU_BYTES_PER_STICK;
          }

          // after processing all the w-entries, go to the next 4k-boundary
          // location (aka stick padding)
          output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                          (-AIU_PAGESIZE_IN_BYTES);
        }

        // output_offset was pushed around in the dims[2-0] loops, so reset it
        // to the next n
        output_offset = out_offset_n + bytes_per_n;
      }
    } else { // NCHW

      uint8_t sizeof_dlf16 = get_data_type_size(ZDNN_DLFLOAT16);

      // process the entire W number of entries at every pass
      fields_to_convert = ztensor->transformed_desc->dim2;

      // convert_data_format() will dump the converted entries here
      uint16_t temp_buff[fields_to_convert];

      // number of bytes to jump from the beginning of the last C-stick to the
      // next page-boundary
      uint64_t padding =
          (ztensor->transformed_desc->dim2 % AIU_STICKS_PER_PAGE)
              ? (AIU_STICKS_PER_PAGE -
                 (ztensor->transformed_desc->dim2 % AIU_STICKS_PER_PAGE)) *
                    AIU_BYTES_PER_STICK
              : 0;

      for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

        uint64_t out_offset_n = output_offset;

        for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1; e1x++) {

          uint64_t output_offset_c = output_offset;

          for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
            prefetch_read(in_buf, input_offset);
            nbr_fields_converted =
                convert_data_format((void *)((uintptr_t)in_buf + input_offset),
                                    ztensor->pre_transformed_desc->type,
                                    temp_buff, ztensor->transformed_desc->type,
                                    fields_to_convert, saturate_func);

            if (nbr_fields_converted == 0)
              return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

            // read each entry in temp_buff contiguously and scatter write them
            // to stick area locations AIU_BYTES_PER_STICK bytes apart, i.e.,
            // the same C location of the consecutive C-sticks
            for (uint32_t w = 0; w < fields_to_convert; w++) {
              // Prefetch to L1 newest offset to write that HW wouldn't
              // know about
              prefetch_write(ztensor->buffer, output_offset);
              *(uint16_t *)((uintptr_t)ztensor->buffer + output_offset) =
                  temp_buff[w];
              // go to same C location of the next stick
              output_offset += AIU_BYTES_PER_STICK;
            }

            // go to the next 4k-boundary location (aka stick padding)
            output_offset += padding;

            // push input_offset the entire W number of entries
            input_offset += (nbr_fields_converted << input_cell_shift);
          }

          // go to the next C location of H = 0, W = 0
          output_offset = output_offset_c + sizeof_dlf16;
          if (!((e1x + 1) % AIU_2BYTE_CELLS_PER_STICK)) {
            // but if we're at the end of C-stick, roll back 1 stick worth of
            // bytes and jump to the the next c-stick of that super c-stick,
            // which is bytes_all_h number of bytes away.
            output_offset = output_offset - AIU_BYTES_PER_STICK + bytes_all_h;
          }
        }

        // done with all the C/H/W, go to the next n
        output_offset = out_offset_n + bytes_per_n;
      }
    }
  } else if (ztensor->transformed_desc->layout == ZDNN_HWCK) {

    uint64_t bytes_per_h =
        CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
        ztensor->transformed_desc->dim3 * AIU_PAGESIZE_IN_BYTES;

    uint64_t bytes_all_h = bytes_per_h * ztensor->transformed_desc->dim4;

    // H
    for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

      uint64_t out_offset_h = output_offset;

      // W
      for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

        // C
        for (uint32_t e2x = 0; e2x < ztensor->transformed_desc->dim2; e2x++) {

          uint64_t out_offset_c = output_offset;

          // process each K-stick (i.e., every 64 elements or whatever
          // left in dim1)
          for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
               e1x += AIU_2BYTE_CELLS_PER_STICK) {
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
            // Also, Prefetch the new output offset to write that HW wouldn't
            // know about.
            prefetch_read(in_buf, input_offset);
            prefetch_write(ztensor->buffer, output_offset);
            fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                    AIU_2BYTE_CELLS_PER_STICK);

            nbr_fields_converted = convert_data_format(
                (void *)((uintptr_t)in_buf + input_offset),
                ztensor->pre_transformed_desc->type,
                (void *)((uintptr_t)ztensor->buffer + output_offset),
                ztensor->transformed_desc->type, fields_to_convert,
                saturate_func);

            if (nbr_fields_converted == 0)
              return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

            // push input_offset the next c-stick, fake the multiply by
            // bit-shifting
            input_offset += (nbr_fields_converted << input_cell_shift);

            // push output_offset to the next c-stick of the same super
            // c-stick, which is bytes_all_h number of bytes away.
            output_offset += bytes_all_h;
          }

          // output_offset was pushed around in dim1 loops, so reset it to
          // the next c
          output_offset = out_offset_c + AIU_BYTES_PER_STICK;
        }

        // after processing all the c-entries, go to the next 4k-boundary
        // location (aka stick padding)
        output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                        (-AIU_PAGESIZE_IN_BYTES);
      }

      // output_offset was pushed around in the dims[2-0] loops, so reset it
      // to the next h
      output_offset = out_offset_h + bytes_per_h;
    }
  } else {
    // caller messed up if we ever arrive here
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                       "Invalid layout for transformation: %s",
                       get_data_layout_str(ztensor->transformed_desc->layout));
  }

  /* handle any FP errors or return success */
  // Handle saturation vs no saturation differently to match the HW
  zdnn_status fp_error;
  if (saturation_control == true) {
    fp_error = handle_fp_errors_saturation(
        fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW),
        ztensor->pre_transformed_desc->type);
    if (fp_error != ZDNN_OK) {
      return fp_error;
    }

  } else {
    fp_error = handle_fp_errors(
        fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));
    if (fp_error != ZDNN_OK) {
      return fp_error;
    }
  }
  // Update the tensor's format to indicate it has been stickified
  ztensor->is_transformed = true;
  return ZDNN_STATUS_OK;

} // End transform_ztensor

/// Specialized/Simplified version of transform_ztensor() that transforms 2 * a
/// * b elements to (1, 1, 2*PADDED(a), b) shape
///
/// \param[in] in_buf data buffer to be stickified
/// \param[in] real_dim2 actual, non-PADDED dim2 value
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
///
/// \return ZDNN_OK
///          ZDNN_CONVERT_FAILURE
///
zdnn_status transform_bidir_weight_ztensor(const void *in_buf,
                                           uint32_t real_dim2,
                                           zdnn_ztensor *ztensor) {

  // in_buf technically has shape of (2, real_dim2, dim1), meaning there are
  // 2 * real_dim2 * dim1 elements in it. we want to transform it to a ZDNN_2D
  // ztensor of shape (2 * PADDED(real_dim2), dim1)
  //
  // conceptually, this is as if we're inserting (PADDED(real_dim2) - real_dim2)
  // * dim1 of zeros after every dim1 elements, and transform the whole thing as
  // if it's (2, PADDED(real_dim2), dim1) to start with
  //
  // we'll emulate that effect by using mostly the same flow as
  // transform_ztensor()'s, and manipulate the output offset appropriately

  uint64_t input_offset = 0;
  uint64_t output_offset = 0;

  short input_cell_size =
      get_data_type_size(ztensor->pre_transformed_desc->type);
  short input_cell_shift = input_cell_size / 2;

  uint32_t fields_to_convert;
  uint32_t nbr_fields_converted;

  feclearexcept(FE_ALL_EXCEPT);

  // dim2 is always PADDED (i.e., multiples of AIU_2BYTE_CELLS_PER_STICK) and
  // divisible by AIU_STICKS_PER_PAGE
  uint64_t bytes_all_w = ztensor->transformed_desc->dim2 / AIU_STICKS_PER_PAGE *
                         AIU_PAGESIZE_IN_BYTES;

  // use no-op function to request no saturation be used during data conversion
  void (*skip_func)(const vec_float32 *, const vec_float32 *, vec_float32 *,
                    vec_float32 *) = &skip_saturate_fp32_to_dlf16;

  // exactly 2 rounds, each round processes (real_dim2 * dim1) elements
  for (uint32_t i = 0; i < 2; i++) {

    for (uint32_t e2x = 0; e2x < real_dim2; e2x++) {
      prefetch_read(in_buf, input_offset);
      uint64_t out_offset_w = output_offset;

      for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
        prefetch_write(ztensor->buffer, output_offset);
        fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                AIU_2BYTE_CELLS_PER_STICK);

        nbr_fields_converted = convert_data_format(
            (void *)((uintptr_t)in_buf + input_offset),
            ztensor->pre_transformed_desc->type,
            (void *)((uintptr_t)ztensor->buffer + output_offset),
            ztensor->transformed_desc->type, fields_to_convert, skip_func);

        if (nbr_fields_converted == 0)
          return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);
        cache_flush(ztensor->buffer, output_offset);
        input_offset += (nbr_fields_converted << input_cell_shift);
        output_offset += bytes_all_w;
      }

      output_offset = out_offset_w + AIU_BYTES_PER_STICK;
    }

    // start the 2nd (and last) i-loop at offset (bytes_all_w / 2)
    output_offset = bytes_all_w / 2;
  }

  zdnn_status fp_error = handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

  if (fp_error != ZDNN_OK) {
    return fp_error;
  }

  ztensor->is_transformed = true;
  return ZDNN_STATUS_OK;
}

/// Stickification when dim1 is <= 2.  Only handles NHWC -> NHWC.
///
/// \param[in] in_buf data buffer to be stickified
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
///
/// \return ZDNN_OK
///         ZDNN_CONVERT_FAILURE
///         ZDNN_INVALID_TYPE
///
zdnn_status transform_ztensor_smalldim1(const void *in_buf,
                                        zdnn_ztensor *ztensor) {

  uint64_t output_offset =
      0; // moving position as the output is processed, in BYTES

  // input pointer that always moves forward, will be casted as either
  // vec_int16 or vec_float32 depends on input type
  const void *cur_input_data = in_buf;

  uint32_t rows_in_vec = (ztensor->transformed_desc->dim1 == 2) ? 4 : 8;

  // # of remaining fields to convert in the last group (if any)
  uint32_t remaining_el = (ztensor->transformed_desc->dim2 % rows_in_vec) *
                          ztensor->transformed_desc->dim1;

  uint32_t remaining_bytes_to_get =
      remaining_el * get_data_type_size(ztensor->pre_transformed_desc->type);

  vec_int16 in_vector_16 = {0};
  vec_float32 in_vector_32[2] = {{0}, {0}};
  vec_int16 tmp_vector_16[2] = {{0}, {0}};
  vec_int16 tmp_out = {0};
  vec_int16 zero_vector16 = {0};

  // loop invariant values
  uint64_t bytes_all_h =
      (uint64_t)ztensor->transformed_desc->dim3 *
      CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;
  uint64_t bytes_per_n = bytes_all_h * CEIL(ztensor->transformed_desc->dim1,
                                            AIU_2BYTE_CELLS_PER_STICK);

  feclearexcept(
      FE_ALL_EXCEPT); /* clear exception flags set during conversion */

  // N
  for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

    // used for pushing out_offset from n to n+1 (i.e., + bytes_per_n)
    uint64_t out_offset_n = output_offset;

    // H
    for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

      // If there's more than 8 to convert, convert groups of 8
      for (uint32_t e2x = 0;
           e2x < ztensor->transformed_desc->dim2 / rows_in_vec; e2x++) {

        prefetch_read(cur_input_data, 0);
        // convert + put 8 DLFLOAT16s in tmp_out
        switch (ztensor->pre_transformed_desc->type) {
        case FP16:
          tmp_out = aiu_vec_convert_from_fp16(*(vec_int16 *)(cur_input_data));
          // advance 1 vector worth of entries
          cur_input_data = (vec_int16 *)(cur_input_data) + 1;
          break;
        case FP32:
          tmp_out =
              aiu_vec_round_from_fp32(*(vec_float32 *)(cur_input_data),
                                      *((vec_float32 *)cur_input_data + 1));
          // advance 2 vectors worth of entries
          cur_input_data = (vec_float32 *)(cur_input_data) + 2;
          break;
        case BFLOAT:
          tmp_vector_16[0] =
              vec_mergeh(*(vec_int16 *)(cur_input_data), zero_vector16);
          tmp_vector_16[1] =
              vec_mergel(*(vec_int16 *)(cur_input_data), zero_vector16);
          tmp_out = aiu_vec_round_from_fp32((vec_float32)tmp_vector_16[0],
                                            (vec_float32)tmp_vector_16[1]);
          // advance 1 vector worth of entries
          cur_input_data = (vec_int16 *)(cur_input_data) + 1;
          break;
        default:
          // this is for completeness but we should never get here, called
          // should have already checked it before calling this function
          return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                             "unknown/invalid pre-transformed data type: %d",
                             ztensor->pre_transformed_desc->type);

          break;
        }

        // copy the 8 DLFLOAT16s to stick areas, it's either 8x1 or 4x2
        for (uint32_t i = 0; i < 8; i++) {
          *(uint16_t *)((uintptr_t)ztensor->buffer + output_offset) =
              tmp_out[i];
          if (ztensor->transformed_desc->dim1 == 2) {
            i++;
            *((uint16_t *)((uintptr_t)ztensor->buffer + output_offset) + 1) =
                tmp_out[i];
          }
          output_offset += AIU_BYTES_PER_STICK;
        }
      }

      if (remaining_el > 0) { // If none, skip the rest

        // put remaining_el # of DLFLOAT16s in tmp_out
        switch (ztensor->pre_transformed_desc->type) {
        case FP16:
          in_vector_16 = vec_load_len((uint16_t *)cur_input_data,
                                      remaining_bytes_to_get - 1);
          tmp_out = aiu_vec_convert_from_fp16(in_vector_16);
          break;
        case FP32:
          in_vector_32[0] = vec_load_len((uint32_t *)cur_input_data,
                                         remaining_bytes_to_get - 1);

          // grab the other half
          if (remaining_el > 4) {
            in_vector_32[1] =
                vec_load_len((uint32_t *)((vec_float32 *)cur_input_data + 1),
                             (remaining_bytes_to_get - 16) - 1);
          }

          tmp_out = aiu_vec_round_from_fp32(in_vector_32[0], in_vector_32[1]);
          break;
        case BFLOAT:
          in_vector_16 = vec_load_len((uint16_t *)cur_input_data,
                                      remaining_bytes_to_get - 1);
          tmp_vector_16[0] = vec_mergeh(in_vector_16, zero_vector16);
          tmp_vector_16[1] = vec_mergel(in_vector_16, zero_vector16);
          tmp_out = aiu_vec_round_from_fp32((vec_float32)tmp_vector_16[0],
                                            (vec_float32)tmp_vector_16[1]);
          break;
        default:
          // this is for completeness but we should never get here, called
          // should have already checked it before calling this function
          return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                             "unknown/invalid pre-transformed data type: %d",
                             ztensor->pre_transformed_desc->type);
          break;
        }

        cur_input_data =
            (void *)((uintptr_t)(cur_input_data) + remaining_bytes_to_get);

        // copy those DLFLOAT16s to stick areas
        for (uint32_t i = 0; i < remaining_el; i++) {
          *(uint16_t *)((uintptr_t)ztensor->buffer + output_offset) =
              tmp_out[i];
          if (ztensor->transformed_desc->dim1 == 2) {
            i++;
            *((uint16_t *)((uintptr_t)ztensor->buffer + output_offset) + 1) =
                tmp_out[i];
          }
          output_offset += AIU_BYTES_PER_STICK;
        }
      }

      // after processing all the w-entries, go to the next 4k-boundary
      // location (aka stick padding)
      output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                      (-AIU_PAGESIZE_IN_BYTES);
    }

    // output_offset was pushed around in the dims[2-0] loops, so reset it
    // to the next n
    output_offset = out_offset_n + bytes_per_n;
  }

  /* handle any FP errors or return success */
  zdnn_status fp_error = handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

  if (fp_error != ZDNN_OK) {
    return fp_error;
  }

  // Update the tensor's format to indicate it has been stickified
  ztensor->is_transformed = true;
  return ZDNN_STATUS_OK;
}

/// Converts the input tensor to the supported stick format for
/// execution by zDNN operations.
///
/// \param[out] tensor Pointer to zdnn_ztensor to contain stickified
/// data \param[in] ... 1, 3, or 4 data buffers to be stickified. (1 for
/// most, 3 for
///                ZRH, 4 for FICO)
///
/// \return ZDNN_OK
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_BUFFER
///         ZDNN_INVALID_STATE
///         ZDNN_CONVERT_FAILURE
///
zdnn_status zdnn_transform_ztensor(zdnn_ztensor *ztensor, ...) {
  zdnn_status status;

  LOG_DEBUG("zdnn_transform_ztensor layout %s -> %s",
            get_data_layout_str(ztensor->pre_transformed_desc->layout),
            get_data_layout_str(ztensor->transformed_desc->layout));
  LOG_DEBUG("zdnn_transform_ztensor type %s -> %s",
            get_data_type_str(ztensor->pre_transformed_desc->type),
            get_data_type_str(ztensor->transformed_desc->type));

  if ((status = verify_descriptors_transform_ztensor(ztensor)) != ZDNN_OK) {
    return status;
  }

  /*
   * Check for buffer issues. Return an error if:
   *
   * a) buffer is a NULL pointer
   * b) buffer does not start on a 4k boundary
   * c) buffer_size is smaller than what's needed
   */
  if (!ztensor->buffer || (uintptr_t)ztensor->buffer & 0xFFF ||
      ztensor->buffer_size < zdnn_getsize_ztensor(ztensor->transformed_desc)) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // Make sure the buffer doesn't have stickified data
  if (ztensor->is_transformed) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE,
                       "Attempted to transform data into a tensor that is "
                       "already transformed.",
                       NO_ARG);
  }

  va_list argptr;
  va_start(argptr, ztensor);

  if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW &&
      ztensor->transformed_desc->layout == ZDNN_NHWC &&
      ztensor->transformed_desc->dim1 <= 2) {

    const void *data = va_arg(argptr, void *);
    status = transform_ztensor_smalldim1(data, ztensor);
  } else if (ztensor->transformed_desc->layout == ZDNN_NHWC ||
             ztensor->transformed_desc->layout == ZDNN_HWCK) {

    const void *data = va_arg(argptr, void *);
    status = transform_ztensor(data, ztensor, false);
  } else if (ztensor->transformed_desc->layout == ZDNN_FICO ||
             ztensor->transformed_desc->layout == ZDNN_ZRH ||
             ztensor->transformed_desc->layout == ZDNN_BIDIR_FICO ||
             ztensor->transformed_desc->layout == ZDNN_BIDIR_ZRH) {

    do { // do not just return when error, use break instead. we need to
         // va_end() at the end

      uint32_t num_slices = ztensor->transformed_desc->dim4;
      uint64_t gate_num_elements =
          get_num_elements(ztensor, ELEMENTS_PRE_SINGLE_GATE);
      uint64_t gate_data_size =
          gate_num_elements *
          get_data_type_size(ztensor->pre_transformed_desc->type);
      uint64_t sliced_gate_data_size = gate_data_size / num_slices;
      // 4 gates for FICO otherwise 3 gates (ZRH)
      uint8_t num_gates =
          get_data_layout_num_gates(ztensor->transformed_desc->layout);

      zdnn_tensor_desc temp_pre_tfrmd_desc, temp_tfrmd_desc;

      // Copy the real pre_transformed_desc into temp so we can
      // manipulate it without changing the original.
      memcpy(&temp_pre_tfrmd_desc, ztensor->pre_transformed_desc,
             sizeof(zdnn_tensor_desc));

      // Manipulate the temp pre_trfmd_desc.
      //
      // FICO/ZRH are concatenated horizontally.  The BIDIR_* variants
      // are also concatenated vertically.
      //
      // To build such a concatenated ztensor, we process each "slice"
      // (the promoted dim4) of each gate individually. This way the
      // final ztensor can be built to be sliceable along dim4 and each
      // slice will have a complete set of concatenated gates.
      //
      // pre_trfmd      --> slice shape
      // ------------------------------------------------------------------
      // 3DS: (a, b, c) --> 2D: (1, b, c)                (FICO/ZRH)
      //                    2D: (1, 1, 2*PADDED(b/2), c) (BIDIR_FICO/ZRH)
      // 2DS: (b, c)    --> 1D: (c)                      (FICO/ZRH)
      //                --> (no case for BIDIR_*)
      //
      // The slices will be sent to transform_ztensor(), or
      // transform_bidir_weight_ztensor() if ZDNN_BIDIR_* layouts

      uint32_t pre_trfmd_slices;
      if (ztensor->pre_transformed_desc->layout == ZDNN_3DS) {
        pre_trfmd_slices = ztensor->pre_transformed_desc->dim3;
        if (ztensor->transformed_desc->layout == ZDNN_BIDIR_FICO ||
            ztensor->transformed_desc->layout == ZDNN_BIDIR_ZRH) {
          // dim2 has to be some multiple of 2 because of concatenation
          if (ztensor->pre_transformed_desc->dim2 & 1) {
            status = ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                                 "when PREV_LAYER_BIDIR and "
                                 "USAGE_WEIGHTS, pre-transformed dim2 "
                                 "must be multiples of 2 (found: %d)",
                                 ztensor->pre_transformed_desc->dim2);
            break;
          }
          temp_pre_tfrmd_desc.dim4 = 1;
          temp_pre_tfrmd_desc.dim3 = 1;
          temp_pre_tfrmd_desc.dim2 = 2 * PADDED(temp_pre_tfrmd_desc.dim2 / 2);
        }
        temp_pre_tfrmd_desc.layout = ZDNN_2D;
      } else if (ztensor->pre_transformed_desc->layout == ZDNN_2DS) {
        pre_trfmd_slices = ztensor->pre_transformed_desc->dim2;
        temp_pre_tfrmd_desc.layout = ZDNN_1D;
      } else {
        status = ZDNN_STATUS(
            ZDNN_INVALID_LAYOUT, "layout %s is not supported for concatenation",
            get_data_layout_str(ztensor->pre_transformed_desc->layout));
        break;
      }

      // Check that the pre_tfrmd and tfrmd descriptors indicate the same
      // number of expected slices.
      if (pre_trfmd_slices != num_slices) {
        status = ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                             "the pre_transformed_desc's outermost "
                             "dimension (%d) must be "
                             "the same as transformed_desc's dim4 (%d)",
                             pre_trfmd_slices, num_slices);
        break;
      }

      // Create a non-sliced, non-horizontally-concatenated trfmd_desc
      // by using the modified temp_pre_tfrmd_layout.
      if ((status = zdnn_generate_transformed_desc(
               &temp_pre_tfrmd_desc, &temp_tfrmd_desc)) != ZDNN_OK) {
        break;
      }

      uint64_t sliced_gate_buffer_size = zdnn_getsize_ztensor(&temp_tfrmd_desc);

      // Save the gate data for slicing later.
      // (e.g., LSTM) va_arg order: F (FWD,BWD), I (FWD,BWD), C...etc.
      void *gate_data[num_gates];
      for (uint8_t i = 0; i < num_gates; i++) {
        gate_data[i] = va_arg(argptr, void *);
      }

      // Create a temporary ztensor to be used to call
      // transform_ztensor() multiple times with, as if it's not
      // horizontally-concatenated.
      zdnn_ztensor temp_ztensor;

      // Setup the temp ztensor, with a non-sliced,
      // non-horizontally-concatenated buffer_size
      zdnn_init_ztensor(&temp_pre_tfrmd_desc, &temp_tfrmd_desc, &temp_ztensor);

      temp_ztensor.buffer = ztensor->buffer;
      temp_ztensor.buffer_size = sliced_gate_buffer_size;

      // Concatenated tensors require zero padding between the
      // horizontal concatenations, while technically not required for
      // the verticals. However, zero out the entire concatened (not
      // temp) buffer for efficiency.
      size_t total_buffer_size =
          temp_ztensor.buffer_size * num_slices * num_gates;
      memset(ztensor->buffer, 0, total_buffer_size);

      /* Loop sliced_gate_data array to stickify the input data. Because
       * of how sliced_gate_data was built as a 2D array, we can jump
       * around various locations of the original inputs data and read
       * each value only once while building the output ztensor to be in
       * the final desired order.
       *
       * This converts the value order from the input arrays from:
       * slice 0 of gate 0
       * slice 1 of gate 0
       * slice 0 of gate 1
       * slice 1 of gate 1
       * ...
       *
       * to the following order in the final output ztensor:
       * slice 0 of gate 0
       * slice 0 of gate 1
       * ...
       * slice 1 of gate 0
       * slice 1 of gate 1
       * ...
       */

      for (uint32_t slice = 0; slice < num_slices; slice++) {
        for (uint8_t gate = 0; gate < num_gates; gate++) {
          // Points to a single slice of a single gate data.
          const void *gate_data_slice =
              (void *)((uintptr_t)gate_data[gate] +
                       (slice * sliced_gate_data_size));

          // Transform the current slice of the current gate into final
          // ztensor
          if (ztensor->transformed_desc->layout != ZDNN_BIDIR_FICO &&
              ztensor->transformed_desc->layout != ZDNN_BIDIR_ZRH) {
            status = transform_ztensor(gate_data_slice, &temp_ztensor, false);
          } else {
            // transform_bidir_weight_ztensor() wants the actual b/2,
            // not the PADDED one in temp_ztensor->dim2
            status = transform_bidir_weight_ztensor(
                gate_data_slice, ztensor->pre_transformed_desc->dim2 / 2,
                &temp_ztensor);
          }

          if (status != ZDNN_OK) {
            if ((status & WARNING_STATUS_BITMASK) == ZDNN_WARNING) {
              LOG_WARN("transform_ztensor() returned a warning, status = "
                       "%08x (%s)\n",
                       status, zdnn_get_status_message(status));
            } else {
              LOG_ERROR("transform_ztensor() on slice %d of gate data %d "
                        "failed, status = %08x (%s)\n",
                        slice, gate, status, zdnn_get_status_message(status));
              break;
            }
          }

          // Increment the temp_ztensor buffer by one sliced gate size
          // so we write to the correct location in the final output
          // ztensor.
          temp_ztensor.buffer = (void *)((uintptr_t)(temp_ztensor.buffer) +
                                         sliced_gate_buffer_size);

          // Reset temp_ztensor is_transformed so we can recursively
          // call zdnn_transform_ztensor to process each slice of each
          // gate.
          temp_ztensor.is_transformed = false;
        }
        if (status != ZDNN_OK) {
          break;
        }
      }

      if ((status == ZDNN_OK) ||
          ((status & WARNING_STATUS_BITMASK) == ZDNN_WARNING)) {
        // Set that the output ztensor has completed transforma
        ztensor->is_transformed = true;
      }

    } while (false);
  } else {
    status = ZDNN_STATUS(
        ZDNN_INVALID_LAYOUT, "Invalid layout for transformation: %s",
        get_data_layout_str(ztensor->transformed_desc->layout));
  }

  va_end(argptr);
  return status;
}

/// Converts the input tensor to the supported stick format for
/// execution by zDNN operations.
///
/// \param[out] tensor Pointer to zdnn_ztensor to contain stickified
/// data
/// \param[in] data data buffer currently, only supporting 1.
///
/// \return ZDNN_OK
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_BUFFER
///         ZDNN_INVALID_STATE
///         ZDNN_ELEMENT_RANGE_VIOLATION
///
zdnn_status zdnn_transform_ztensor_with_saturation(zdnn_ztensor *ztensor, ...) {
  zdnn_status status;

  LOG_DEBUG("zdnn_transform_ztensor layout %s -> %s",
            get_data_layout_str(ztensor->pre_transformed_desc->layout),
            get_data_layout_str(ztensor->transformed_desc->layout));
  LOG_DEBUG("zdnn_transform_ztensor type %s -> %s",
            get_data_type_str(ztensor->pre_transformed_desc->type),
            get_data_type_str(ztensor->transformed_desc->type));

  if ((status = verify_descriptors_transform_ztensor(ztensor)) != ZDNN_OK) {
    return status;
  }

  const uint32_t *dims_ptr = &(ztensor->transformed_desc->dim4);

  // is the dimension above the limit or zero?
  // transformed layout uses all dim* entries, so we'll check them all
  for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
    LOG_DEBUG("dim%d: %d", ZDNN_MAX_DIMS - i, dims_ptr[i]);
    if (!dims_ptr[i]) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "Invalid shape for dim%d. (reason: dimension is 0)",
                         ZDNN_MAX_DIMS - i);
    }

    if (dims_ptr[i] > zdnn_get_max_for_dim(ZDNN_MAX_DIMS - i)) {

      if (!zdnn_get_max_for_dim(ZDNN_MAX_DIMS - i)) {
        return ZDNN_STATUS(ZDNN_UNSUPPORTED_AIU_EXCEPTION,
                           "Unable to verify shape. (reason: HW environment "
                           "may not be setup properly)",
                           NO_ARG);
      } else {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "Invalid shape for dim%d. (reason: dimension value %d exceeds %d)",
            ZDNN_MAX_DIMS - i, dims_ptr[i],
            zdnn_get_max_for_dim(ZDNN_MAX_DIMS - i));
      }
    }
  }

  // is stick area size above the limit?
  if (zdnn_getsize_ztensor(ztensor->transformed_desc) >
      zdnn_get_nnpa_max_tensor_size()) {
    return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                       "Invalid shape (reasons: tensor size: %" PRIu64
                       ", maximum: %" PRIu64 " bytes",
                       zdnn_getsize_ztensor(ztensor->transformed_desc),
                       zdnn_get_nnpa_max_tensor_size());
  }

  /*
   * Check for buffer issues. Return an error if:
   *
   * a) buffer is a NULL pointer
   * b) buffer does not start on a 4k boundary
   * c) buffer_size is smaller than what's needed
   */
  if (!ztensor->buffer || (uintptr_t)ztensor->buffer & 0xFFF ||
      ztensor->buffer_size < zdnn_getsize_ztensor(ztensor->transformed_desc)) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // Make sure the buffer doesn't have stickified data
  if (ztensor->is_transformed) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE,
                       "Attempted to transform data into a tensor that is "
                       "already transformed.",
                       NO_ARG);
  }

  va_list argptr;
  va_start(argptr, ztensor);
  const void *data = va_arg(argptr, void *);
  status = transform_ztensor(data, ztensor, true);
  if (status == ZDNN_OK || status == ZDNN_ELEMENT_RANGE_VIOLATION) {
    // Set that the ztensor has completed transformation.
    ztensor->is_transformed = true;
  }
  va_end(argptr);
  return status;
}

zdnn_status transform_quantized_ztensor(const void *in_buf, int8_t clip_min,
                                        int8_t clip_max, zdnn_ztensor *output) {
  // Create a temporary ztensor to point to user's data buffer
  zdnn_tensor_desc temp_desc;
  init_transformed_desc(
      ZDNN_NHWC, ZDNN_BINARY_FP32, ZDNN_FORMAT_4DGENERIC, &temp_desc,
      output->transformed_desc->dim4, output->transformed_desc->dim3,
      output->transformed_desc->dim2, output->transformed_desc->dim1);
  zdnn_ztensor temp_input;
  zdnn_init_ztensor(&temp_desc, &temp_desc, &temp_input);
  temp_input.buffer = (void *)in_buf;

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS();
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS();
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_transform *fsp_transform = (func_sp_parms_transform *)&fsp;
  // Setup Function Specific Parameters:
  fsp_transform->parm1.toc = NNPA_TOC_STICK_INT8;
  fsp_transform->parm2.rec_scale = cnvt_1_fp32_to_dlf16(output->rec_scale);
  fsp_transform->parm3.offset = cnvt_1_fp32_to_dlf16(output->offset);
  fsp_transform->parm4.clip_min = clip_min;
  fsp_transform->parm5.clip_max = clip_max;

  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_TRANSFORM,
                               &temp_input, NULL, NULL, output, NULL, 0, &fsp);
}

zdnn_status transform_quantized_int8_ztensor(const int8_t *in_buf,
                                             zdnn_ztensor *output) {
  // moving position as the input is processed, in BYTES
  uint64_t input_offset = 0;
  // moving position as the output is processed, in BYTES
  uint64_t output_offset = 0;

  uint32_t fields_to_convert;

  // loop invariant values
  uint64_t bytes_all_h =
      (uint64_t)output->transformed_desc->dim3 *
      CEIL(output->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  uint64_t bytes_per_n = bytes_all_h * CEIL(output->transformed_desc->dim1,
                                            AIU_1BYTE_CELLS_PER_STICK);

  // N
  for (uint32_t e4x = 0; e4x < output->transformed_desc->dim4; e4x++) {

    // used for pushing out_offset from n to n+1 (i.e., + bytes_per_n)
    uint64_t out_offset_n = output_offset;

    // H
    for (uint32_t e3x = 0; e3x < output->transformed_desc->dim3; e3x++) {

      // W
      for (uint32_t e2x = 0; e2x < output->transformed_desc->dim2; e2x++) {

        // used for pushing out_offset from w to w+1 (i.e., +
        // AIU_BYTES_PER_STICK)
        uint64_t out_offset_w = output_offset;

        for (uint32_t e1x = 0; e1x < output->transformed_desc->dim1;
             e1x += AIU_1BYTE_CELLS_PER_STICK) {
          prefetch_write(output->buffer, output_offset);
          fields_to_convert = MIN((output->transformed_desc->dim1 - e1x),
                                  AIU_1BYTE_CELLS_PER_STICK);
          memcpy((void *)((uintptr_t)output->buffer + output_offset),
                 in_buf + input_offset, fields_to_convert);

          // move on to the next set
          input_offset += fields_to_convert;
          output_offset += bytes_all_h;
        }

        // output_offset was pushed around in dim1 loops, so reset it to
        // the next w
        output_offset = out_offset_w + AIU_BYTES_PER_STICK;
      }

      // after processing all the w-entries, go to the next 4k-boundary
      // location (aka stick padding)
      output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                      (-AIU_PAGESIZE_IN_BYTES);
    }

    // output_offset was pushed around in the dims[2-0] loops, so reset it
    // to the next n
    output_offset = out_offset_n + bytes_per_n;
  }

  // Update the tensor's format to indicate it has been stickified
  output->is_transformed = true;
  return ZDNN_STATUS_OK;
}

zdnn_status transform_quantized_weights_ztensor(const void *in_buf,
                                                zdnn_ztensor *output) {

  // moving position as the input is processed, in BYTES
  uint64_t input_offset = 0;
  // moving position as the output is processed, in BYTES
  uint64_t output_offset = 0;

  // vec_perm(): vector1 bytes are indexed 0 - 15, vector2 16 - 31
  //
  // this pattern selects:
  // 1st byte from vector1, then
  // 1st byte from vector2, then
  // 2nd byte from vector1, then
  // 2nd byte from vector2, and so forth
  vec_char8 sel_vec_full = {0, 16, 1, 17, 2, 18, 3, 19,
                            4, 20, 5, 21, 6, 22, 7, 23};

  vec_char8 sel_vec_partial = {0};

  // selection vector for the leftovers when e1 isn't multiples of
  // VECPERM_MAX_INT8_ENTRIES
  memcpy((void *)&sel_vec_partial, (void *)&sel_vec_full,
         (output->transformed_desc->dim1 % VECPERM_MAX_INT8_ENTRIES) * 2);

  // loop invariant values
  uint64_t bytes_all_h =
      (uint64_t)output->transformed_desc->dim3 *
      CEIL(CEIL(output->transformed_desc->dim2, 2), AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  uint64_t bytes_per_n = bytes_all_h * CEIL(output->transformed_desc->dim1,
                                            (AIU_1BYTE_CELLS_PER_STICK / 2));

  // N
  for (uint32_t e4x = 0; e4x < output->transformed_desc->dim4; e4x++) {

    // used for pushing out_offset from n to n+1 (i.e., + bytes_per_n)
    uint64_t out_offset_n = output_offset;

    // H
    for (uint32_t e3x = 0; e3x < output->transformed_desc->dim3; e3x++) {

      // W, sticks are processed in pairs
      for (uint32_t e2x = 0; e2x < output->transformed_desc->dim2;
           e2x = e2x + 2) {

        // used for pushing out_offset from w to w+1 (i.e., +
        // AIU_BYTES_PER_STICK)
        uint64_t out_offset_w = output_offset;

        // true when dim2 is odd number and we're at the last w
        bool no_stick2 = ((output->transformed_desc->dim2 - e2x) == 1);

        int8_t *stick1 = (int8_t *)in_buf + input_offset;
        int8_t *stick2 = no_stick2 ? stick1
                                   // duplicate stick1 entries if no stick2
                                   : stick1 + output->transformed_desc->dim1;

        prefetch_read(stick1, 0);
        prefetch_read(stick2, 0);

        // this C loop takes care of the full VECPERM_MAX_INT8_ENTRIES-entries
        // groups
        for (uint32_t i = 0;
             i < output->transformed_desc->dim1 / VECPERM_MAX_INT8_ENTRIES;
             i++) {
          prefetch_write(output->buffer, output_offset);
          *(vec_char8 *)((int8_t *)output->buffer + output_offset) = vec_perm(
              *(vec_char8 *)stick1, *(vec_char8 *)stick2, sel_vec_full);

          stick1 += VECPERM_MAX_INT8_ENTRIES;
          stick2 += VECPERM_MAX_INT8_ENTRIES;
          output_offset += VECPERM_MAX_INT8_ENTRIES * 2;

          if ((i + 1) %
                  (AIU_BYTES_PER_STICK / (VECPERM_MAX_INT8_ENTRIES * 2)) ==
              0) {
            // we need to jump to the next c-stick of the same super c-stick
            //
            // roll-back to the beginning and jump to bytes_all_h number of
            // bytes away
            output_offset = output_offset - AIU_BYTES_PER_STICK + bytes_all_h;
          }
        }

        // takes care of the leftever c entries
        if ((output->transformed_desc->dim1 % VECPERM_MAX_INT8_ENTRIES) != 0) {
          *(vec_char8 *)((int8_t *)output->buffer + output_offset) = vec_perm(
              *(vec_char8 *)stick1, *(vec_char8 *)stick2, sel_vec_partial);
        }

        // move on to the next set
        input_offset += output->transformed_desc->dim1 * (no_stick2 ? 1 : 2);
        // output_offset was pushed around in dim1 loops, so reset it to
        // the next w
        output_offset = out_offset_w + AIU_BYTES_PER_STICK;
      }

      // after processing all the w-entries, go to the next 4k-boundary
      // location (aka stick padding)
      output_offset = (output_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                      (-AIU_PAGESIZE_IN_BYTES);
    }

    // output_offset was pushed around in the dims[2-0] loops, so reset it
    // to the next n
    output_offset = out_offset_n + bytes_per_n;
  }

  // Update the tensor's format to indicate it has been stickified
  output->is_transformed = true;
  return ZDNN_STATUS_OK;
}

/// Converts a data buffer of fp16 to fp32 and stores it into a malloc'd space.
float *convert_fp16_fp32_and_store(zdnn_tensor_desc *tfd_desc,
                                   const void *data) {
  uint64_t total_elements = (uint64_t)tfd_desc->dim4 * tfd_desc->dim3 *
                            tfd_desc->dim2 * tfd_desc->dim1;

  // fp16_to_dlf16() / dlf16_to_fp32() work best when there are
  // STICKCVT_MAX_ENTRIES_TO_CONVERT entries to do
  uint64_t size = sizeof(float) *
                  CEIL(total_elements, STICKCVT_MAX_ENTRIES_TO_CONVERT) *
                  STICKCVT_MAX_ENTRIES_TO_CONVERT;

  float *fp32_data = malloc(size);
  if (fp32_data == NULL) {
    return NULL;
  }

  uint16_t tmp_dlf16[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {0};

  uint16_t *data16 = (uint16_t *)(data);
  float *data32 = fp32_data;

  // do the full STICKCVT_MAX_ENTRIES_TO_CONVERT groups 1st
  for (uint64_t i = 0; i < total_elements / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       i++) {

    fp16_to_dlf16(data16, tmp_dlf16, STICKCVT_MAX_ENTRIES_TO_CONVERT);
    dlf16_to_fp32(tmp_dlf16, data32, STICKCVT_MAX_ENTRIES_TO_CONVERT);

    data16 += STICKCVT_MAX_ENTRIES_TO_CONVERT;
    data32 += STICKCVT_MAX_ENTRIES_TO_CONVERT;
  }

  // do the leftovers
  if (total_elements % STICKCVT_MAX_ENTRIES_TO_CONVERT) {
    fp16_to_dlf16(data16, tmp_dlf16,
                  total_elements % STICKCVT_MAX_ENTRIES_TO_CONVERT);
    dlf16_to_fp32(tmp_dlf16, data32, STICKCVT_MAX_ENTRIES_TO_CONVERT);
  }

  return fp32_data;
}

/// Converts a data buffer of bfloat to fp32 and stores it into a malloc'd
/// space.
float *convert_bf_fp32_and_store(zdnn_tensor_desc *tfd_desc, const void *data) {

  // vec_perm(): vector1 bytes are indexed 0 - 15, vector2 16 - 31
  //
  // this pattern selects:
  // 1st 2 bytes from vector1, then
  // 1st 2 bytes from vector2, then
  // 2nd 2 bytes from vector1, then
  // 2nd 2 bytes from vector2, and so forth
  vec_char8 sel_vec_full = {0, 1, 16, 17, 2, 3, 18, 19,
                            4, 5, 20, 21, 6, 7, 22, 23};

  vec_char8 sel_vec_partial = {0};

  vec_int16 zeros = {0};

  // selection vector for the leftovers when e1 isn't multiples of
  // VECPERM_MAX_BFLOAT_ENTRIES
  memcpy((void *)&sel_vec_partial, (void *)&sel_vec_full,
         (tfd_desc->dim1 % VECPERM_MAX_BFLOAT_ENTRIES) * 2);

  uint64_t total_elements = (uint64_t)tfd_desc->dim4 * tfd_desc->dim3 *
                            tfd_desc->dim2 * tfd_desc->dim1;

  float *fp32_data = malloc(sizeof(float) * total_elements);
  if (fp32_data == NULL) {
    return NULL;
  }

  uint16_t *data16 = (uint16_t *)(data);
  float *data32 = fp32_data;

  // do the full VECPERM_MAX_BFLOAT_ENTRIES groups 1st
  for (uint64_t i = 0; i < total_elements / VECPERM_MAX_BFLOAT_ENTRIES; i++) {

    // cppcheck-suppress invalidPointerCast
    *(vec_int16 *)data32 = vec_perm(*(vec_int16 *)data16, zeros, sel_vec_full);

    data16 += VECPERM_MAX_BFLOAT_ENTRIES;
    data32 += VECPERM_MAX_BFLOAT_ENTRIES;
  }

  // do the leftovers
  if (total_elements % VECPERM_MAX_BFLOAT_ENTRIES) {
    // cppcheck-suppress invalidPointerCast
    *(vec_int16 *)data32 =
        vec_perm(*(vec_int16 *)data16, zeros, sel_vec_partial);
  }

  return fp32_data;
}

/// Converts the input tensor to a quantized stick format for
/// execution by zDNN operations.
///
///
/// \return ZDNN_OK
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_BUFFER
///         ZDNN_INVALID_STATE
///         ZDNN_CONVERT_FAILURE
///         ZDNN_ALLOCATION_FAILURE
///
zdnn_status zdnn_transform_quantized_ztensor(zdnn_ztensor *ztensor,
                                             bool saturation_control,
                                             int8_t clip_min, int8_t clip_max,
                                             const void *data) {
  zdnn_status status;

  LOG_DEBUG("zdnn_transform_quantized_ztensor layout %s -> %s",
            get_data_layout_str(ztensor->pre_transformed_desc->layout),
            get_data_layout_str(ztensor->transformed_desc->layout));
  LOG_DEBUG("zdnn_transform_quantized_ztensor type %s -> %s",
            get_data_type_str(ztensor->pre_transformed_desc->type),
            get_data_type_str(ztensor->transformed_desc->type));

  /*
   * Check for buffer issues. Return an error if:
   *
   * a) buffer is a NULL pointer
   * b) buffer does not start on a 4k boundary
   * c) buffer_size is smaller than what's needed
   */
  if (!ztensor->buffer || (uintptr_t)ztensor->buffer & 0xFFF ||
      ztensor->buffer_size < zdnn_getsize_ztensor(ztensor->transformed_desc)) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // Make sure the buffer doesn't have stickified data
  if (ztensor->is_transformed) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE,
                       "Attempted to transform data into a tensor that is "
                       "already transformed.",
                       NO_ARG);
  }

  // Make sure layout is ZDNN_NHWC
  if (ztensor->transformed_desc->layout != ZDNN_NHWC) {
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                       "Invalid transformed layout for transformation: %s",
                       get_data_layout_str(ztensor->transformed_desc->layout));
  }

  // Make sure pre_transformed layout is valid.
  switch (ztensor->pre_transformed_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_NHWC:
    break;
  default:
    return ZDNN_STATUS(
        ZDNN_INVALID_LAYOUT,
        "Invalid pre-transformed layout for transformation: %s",
        get_data_layout_str(ztensor->pre_transformed_desc->layout));
    break;
  }

  if ((status = verify_transformed_dimensions(ztensor->transformed_desc)) !=
      ZDNN_OK) {
    return status;
  }

  if (ztensor->transformed_desc->format == ZDNN_FORMAT_4DFEATURE) {
    if (ztensor->transformed_desc->type == ZDNN_DLFLOAT16) {
      switch (ztensor->pre_transformed_desc->type) {
      case FP32:
        status = hw_transform_ztensor(data, saturation_control, ztensor);
        break;
      case FP16:
      case BFLOAT:
        status = transform_ztensor(data, ztensor, false);
        break;
      default:
        status =
            ZDNN_STATUS(ZDNN_INVALID_TYPE,
                        "Invalid pre-transform type to be transformed to "
                        "quantized type of DLFLOAT16: %d (%s)",
                        ztensor->transformed_desc->type,
                        get_data_type_str(ztensor->transformed_desc->type));
      }
    } else if (ztensor->transformed_desc->type == ZDNN_BINARY_INT8) {
      zdnn_tensor_desc *pt_desc = ztensor->pre_transformed_desc;
      zdnn_tensor_desc *tfd_desc = ztensor->transformed_desc;
      switch (pt_desc->type) {
      case FP16: {
        float *fp32_data = convert_fp16_fp32_and_store(tfd_desc, data);

        if (fp32_data == NULL) {
          return ZDNN_STATUS(
              ZDNN_ALLOCATION_FAILURE,
              "Unable to allocate required bytes for fp16 to fp32 conversion.",
              NO_ARG);
        }

        status =
            transform_quantized_ztensor(fp32_data, clip_min, clip_max, ztensor);
        free(fp32_data);
        break;
      }
      case BFLOAT: {
        float *fp32_data = convert_bf_fp32_and_store(tfd_desc, data);

        if (fp32_data == NULL) {
          return ZDNN_STATUS(
              ZDNN_ALLOCATION_FAILURE,
              "Unable to allocate required bytes for bf to fp32 conversion.",
              NO_ARG);
        }
        status =
            transform_quantized_ztensor(fp32_data, clip_min, clip_max, ztensor);
        free(fp32_data);
        break;
      }
      case FP32:
        status = transform_quantized_ztensor(data, clip_min, clip_max, ztensor);
        break;
      case INT8:
        status = transform_quantized_int8_ztensor(data, ztensor);
        break;
      default:
        return ZDNN_STATUS(
            ZDNN_INVALID_TYPE,
            "Invalid pre-transform type to be transformed to "
            "quantized type of INT8: %d (%s)",
            ztensor->pre_transformed_desc->type,
            get_data_type_str(ztensor->pre_transformed_desc->type));
      }
    } else {
      return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                         "Invalid transform type for transformation: %d (%s)",
                         ztensor->transformed_desc->type,
                         get_data_type_str(ztensor->transformed_desc->type));
    }
  } else if (ztensor->transformed_desc->format == ZDNN_FORMAT_4DWEIGHTS) {
    if (ztensor->transformed_desc->type != ZDNN_BINARY_INT8) {
      return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                         "Invalid transform type for transformation: %d (%s)",
                         ztensor->transformed_desc->type,
                         get_data_type_str(ztensor->transformed_desc->type));
    }
    if (ztensor->pre_transformed_desc->type != INT8) {
      return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                         "Invalid pre-transform type to be transformed to "
                         "quantized type of WEIGHTS_INT8: %d (%s)",
                         ztensor->transformed_desc->type,
                         get_data_type_str(ztensor->transformed_desc->type));
    }
    status = transform_quantized_weights_ztensor(data, ztensor);
  } else {
    return ZDNN_STATUS(ZDNN_INVALID_FORMAT,
                       "Invalid transform format for transformation: %d (%s)",
                       ztensor->transformed_desc->format,
                       get_data_format_str(ztensor->transformed_desc->format));
  }

  return status;
}

/// Call HW to transform from DLFLOAT16 -> FP32 only.
///
/// \param[in] input Pointer to zdnn_ztensor, containing data to be
///                    unstickified
/// \param[out] out_buf data buffer to unstickify to
///
/// \return ZDNN_OK
///         TODO
///
zdnn_status hw_transform_origtensor(const zdnn_ztensor *input, void *out_buf) {
  // Create a temporary ztensor to point to user's data buffer
  zdnn_tensor_desc temp_desc;
  init_transformed_desc(
      ZDNN_NHWC, ZDNN_BINARY_FP32, ZDNN_FORMAT_4DGENERIC, &temp_desc,
      input->transformed_desc->dim4, input->transformed_desc->dim3,
      input->transformed_desc->dim2, input->transformed_desc->dim1);
  zdnn_ztensor temp_output;
  temp_output.transformed_desc = &temp_desc;
  temp_output.buffer = (void *)out_buf;

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS();
    PRINT_PARM_ZTENSOR_PTR(input);
    END_PRINT_PARMS();
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_transform *fsp_transform = (func_sp_parms_transform *)&fsp;
  // Setup Function Specific Parameters:
  fsp_transform->parm1.toc = NNPA_TOC_UNSTICK_DLFLOAT;

  // NNPA_TRANSFORM with TOC=UNSTICK_DLFLOAT requires 8K SaveArea.
  void *savearea_addr;
  if (!(savearea_addr = malloc_aligned_4k(ZDNN_8K_SAVEAREA_SIZE))) {
    return ZDNN_STATUS(ZDNN_ALLOCATION_FAILURE,
                       "Unable to allocate %" PRIu64 " bytes for save area.",
                       ZDNN_8K_SAVEAREA_SIZE);
  }

  zdnn_status status;
  status = aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_TRANSFORM, input,
                                 NULL, NULL, &temp_output, NULL,
                                 (uintptr_t)savearea_addr, &fsp);
  free_aligned_4k(savearea_addr);
  return status;
}

// ------------------------------------------------------------------------------------------------

/// The actual routine for unstickification, only does the following:
///    NHWC -> NHWC, NHWC -> NCHW
/// Does NOT handle concatenated types nor HWCK
///
/// \param[in] ztensor Pointer to zdnn_ztensor, containing data to be
///                    unstickified
/// \param[out] out_buf data buffer to unstickify to
///
/// \return ZDNN_OK
///         ZDNN_CONVERT_FAILURE
///
zdnn_status transform_origtensor(const zdnn_ztensor *ztensor, void *out_buf) {
  uint64_t output_offset =
      0; // moving position as the output is processed, in BYTES
  uint64_t input_offset =
      0; // moving position as the input is processed, in BYTES

  short output_cell_size =
      get_data_type_size(ztensor->pre_transformed_desc->type);
  short output_cell_shift = output_cell_size / 2;

  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  // loop invariant values
  uint64_t bytes_per_h =
      CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  uint64_t bytes_all_h =
      (uint64_t)ztensor->transformed_desc->dim3 * bytes_per_h;

  uint64_t bytes_per_n = bytes_all_h * CEIL(ztensor->transformed_desc->dim1,
                                            AIU_2BYTE_CELLS_PER_STICK);

  feclearexcept(
      FE_ALL_EXCEPT); /* clear exception flags set during conversion */

  // use no-op function to request no saturation be used during data conversion
  void (*skip_func)(const vec_float32 *, const vec_float32 *, vec_float32 *,
                    vec_float32 *) = &skip_saturate_fp32_to_dlf16;

  if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW) {

    // If FP32 and NNPA_TRANSFORM is available, send to hardware
    if (ztensor->pre_transformed_desc->type == FP32 &&
        zdnn_is_nnpa_function_installed(1, NNPA_TRANSFORM) &&
        n_stride_meets_hardware_limit(ztensor->transformed_desc)) {
      return hw_transform_origtensor(ztensor, out_buf);
    }

    for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

      // used for pushing input_offset from n to n+1 (i.e., +
      // bytes_per_n)
      uint64_t in_offset_n = input_offset;

      for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

        for (uint32_t e2x = 0; e2x < ztensor->transformed_desc->dim2; e2x++) {

          // used for pushing input from w to w+1 (i.e., +
          // AIU_BYTES_PER_STICK)
          uint64_t in_offset_w = input_offset;

          // process each c-stick (i.e., every 64 elements or whatever
          // left in dim1)
          for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
               e1x += AIU_2BYTE_CELLS_PER_STICK) {
            // Prefetch (read) the next input buffer to be used. The HW
            // should "notice" our sequential accesses and continue
            // them, so we won't need to aggressively prefetch here.
            // Also, Prefetch the new output offset to write that HW
            // wouldn't know about.

            prefetch_read(ztensor->buffer, input_offset);
            prefetch_write(out_buf, output_offset);
            fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                    AIU_2BYTE_CELLS_PER_STICK);

            nbr_fields_converted = convert_data_format(
                (void *)((uintptr_t)ztensor->buffer + input_offset),
                ztensor->transformed_desc->type,
                (void *)((uintptr_t)out_buf + output_offset),
                ztensor->pre_transformed_desc->type, fields_to_convert,
                skip_func);

            if (nbr_fields_converted == 0)
              return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

            // push output_offset the next c-stick, fake the multiply by
            // bit-shifting
            output_offset += (nbr_fields_converted << output_cell_shift);

            // push input_offset to the next c-stick of the same super
            // c-stick, which is bytes_all_h number of bytes away.
            input_offset += bytes_all_h;
          }

          // input_offset was pushed around in dim1 loops, so reset it
          // to the next w
          input_offset = in_offset_w + AIU_BYTES_PER_STICK;
        }

        // after processing all the w-entries, go to the next
        // 4k-boundary location (aka stick padding)
        input_offset = (input_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                       (-AIU_PAGESIZE_IN_BYTES);
      }

      // input_offset was pushed around in the dims[2-0] loops, so reset
      // it to the next n
      input_offset = in_offset_n + bytes_per_n;
    }
  } else {

    // the loops are in N -> C -> H -> W order in order to write the W
    // entries contiguously

    // N
    for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

      // used for pushing input_offset from n to n+1 (i.e., +
      // bytes_per_n)
      uint64_t in_offset_n = input_offset;

      // C
      for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1; e1x++) {

        // used for pushing input from c to c+1
        uint64_t in_offset_c = input_offset;

        // H
        for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {
          // Prefetch (read) the next input buffer to be used. The HW
          // should "notice" our sequential accesses and continue them,
          // so we won't need to aggressively prefetch here. Also,
          // Prefetch the new output offset to write that HW wouldn't
          // know about.

          prefetch_read(ztensor->buffer, input_offset);
          prefetch_write(out_buf, output_offset);
          // send all the W entries of a given set of N/H/C to
          // convert_data_format_in_stride(), the entries are
          // AIU_BYTES_PER_STICK entries apart

          fields_to_convert = ztensor->transformed_desc->dim2;

          nbr_fields_converted = convert_data_format_in_stride(
              (void *)((uintptr_t)ztensor->buffer + input_offset),
              ztensor->transformed_desc->type,
              (void *)((uintptr_t)out_buf + output_offset),
              ztensor->pre_transformed_desc->type, fields_to_convert,
              AIU_2BYTE_CELLS_PER_STICK);

          if (nbr_fields_converted == 0)
            return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

          // push input_offset to the next H
          input_offset += bytes_per_h;

          // push output_offset to the next H, fake the multiply by
          // bit-shifting
          output_offset += (nbr_fields_converted << output_cell_shift);
        }

        // push in_offset_c to the next C
        in_offset_c += get_data_type_size(ztensor->transformed_desc->type);

        // go to the next C-stick if we're at the end of the current
        // C-stick
        if (!((e1x + 1) % AIU_2BYTE_CELLS_PER_STICK)) {
          in_offset_c = in_offset_c - AIU_BYTES_PER_STICK + bytes_all_h;
        }

        input_offset = in_offset_c;
      }

      // reset input_offset to the next n
      input_offset = in_offset_n + bytes_per_n;
    }
  }

  // handle any FP errors or return success
  return handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

} // End transform_origtensor

/// Unstickification when dim1 is <= 2.  Only handles NHWC -> NHWC.
///
/// \param[in] ztensor Pointer to zdnn_ztensor, containing data to be
///                    unstickified
/// \param[out] out_buf data buffer to unstickify to
///
/// \return ZDNN_OK
///         ZDNN_CONVERT_FAILURE
///         ZDNN_INVALID_TYPE
///
zdnn_status transform_origtensor_smalldim1(const zdnn_ztensor *ztensor,
                                           void *out_buf) {

  uint64_t input_offset =
      0; // moving position as the input is processed, in BYTES

  // Define a input vector.  a Vector Register can fit 8 int16 fields
  vec_int16 input_data = {0};

  // output pointer that always moves forward, will be casted as either
  // vec_int16 or vec_float32 depends on output type
  // ** Note: adding 1 to a vector pointer will move it ahead 16 bytes
  void *output_data = out_buf;

  // loop invariant values
  uint64_t bytes_per_h =
      CEIL(ztensor->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  uint64_t bytes_all_h =
      (uint64_t)ztensor->transformed_desc->dim3 * bytes_per_h;

  uint64_t bytes_per_n = bytes_all_h * CEIL(ztensor->transformed_desc->dim1,
                                            AIU_2BYTE_CELLS_PER_STICK);

  // Set indicies from which values need to be collected for conversion
  vector unsigned int idx, idx_left, idx_right;

  /*

  load one vector worth of DLFLOAT16 entries (8).
  if dim1 == 2 that's 4 rows of entries (rows_in_vec)

  loop e2x = 0                 -->   e2x = 1

  --LEFT-|-RIGHT-|-----------       --LEFT-|-RIGHT-|-----------
  |  0   |   1   | ...      |       |  256 |   257 | ...      |
  |  64  |   65  | ...      |  -->  |  320 |   321 | ...      |
  |  128 |   129 | ...      |       |  384 |   385 | ...      |
  |  192 |   193 | ...      |       |  448 |   449 | ...      |

  if dim1 == 1 then 8 rows

  ---------------------------       ---------------------------
  |  0   | ...              |       |  512 | ...              | LEFT
  |  64  | ...              |       |  576 | ...              | LEFT
  |  128 | ...              |       |  640 | ...              | LEFT
  |  192 | ...              |  -->  |  704 | ...              | LEFT
  |  256 | ...              |       |  768 | ...              | RIGHT
  |  320 | ...              |       |  832 | ...              | RIGHT
  |  384 | ...              |       |  896 | ...              | RIGHT
  |  448 | ...              |       |  960 | ...              | RIGHT

  */

  // ** xlc requires vector shift right operand to be unsigned long

  vector unsigned int idx_left_incr = (ztensor->transformed_desc->dim1 == 2)
                                          ? (vector unsigned int){0, 1, 64, 65}
                                          : (vector unsigned int){0, 1, 2, 3}
                                                << 6ul;
  vector unsigned int idx_right_incr =
      (ztensor->transformed_desc->dim1 == 2)
          ? (vector unsigned int){128, 129, 192, 193}
          : (vector unsigned int){4, 5, 6, 7} << 6ul;

  uint32_t rows_in_vec;
  unsigned long vec_shift;

  if (ztensor->transformed_desc->dim1 == 2) {
    rows_in_vec = 4;
    // when rows_in_vec == 4, groups are 2^8 entries apart
    vec_shift = 8;
  } else {
    rows_in_vec = 8;
    // when rows_in_vec == 8, groups are 2^9 entries apart
    vec_shift = 9;
  }

  // # of remaining fields to convert in the last group (if any)
  uint32_t remaining_el = (ztensor->transformed_desc->dim2 % rows_in_vec) *
                          ztensor->transformed_desc->dim1;

  uint32_t remaining_bytes_to_set =
      remaining_el * get_data_type_size(ztensor->pre_transformed_desc->type);

  vec_int16 tmp_out_16;
  vec_float32 tmp_out_left, tmp_out_right;

  vec_char8 selection_vector = {0,  1,  4,  5,  8,  9,  12, 13,
                                16, 17, 20, 21, 24, 25, 28, 29};

  feclearexcept(
      FE_ALL_EXCEPT); /* clear exception flags set during conversion */

  for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

    // used for pushing input_offset from n to n+1 (i.e., + bytes_per_n)
    uint64_t in_offset_n = input_offset;

    for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

      uint16_t *in_data =
          (uint16_t *)((uintptr_t)ztensor->buffer + input_offset);

      uint32_t e2x;

      // If there's more than 8 to convert, convert groups of 8
      // DLFLOAT16s
      for (e2x = 0; e2x < ztensor->transformed_desc->dim2 / rows_in_vec;
           e2x++) {
        idx = (vector unsigned int){e2x, e2x, e2x, e2x} << vec_shift;
        idx_left = idx + idx_left_incr;
        idx_right = idx + idx_right_incr;

        prefetch_write((void *)output_data, 0);

        input_data = (vec_int16){in_data[idx_left[0]],  in_data[idx_left[1]],
                                 in_data[idx_left[2]],  in_data[idx_left[3]],
                                 in_data[idx_right[0]], in_data[idx_right[1]],
                                 in_data[idx_right[2]], in_data[idx_right[3]]};

        switch (ztensor->pre_transformed_desc->type) {
        case FP16:
          *((vec_int16 *)(output_data)) = aiu_vec_convert_to_fp16(input_data);

          // bump ptr to start of next vector (8 float16s = 16 bytes =
          // +1)
          output_data = (vec_int16 *)output_data + 1;
          break;
        case FP32:
          aiu_vec_lengthen_to_fp32((vec_int16)input_data,
                                   (vec_float32 *)output_data,
                                   (vec_float32 *)output_data + 1);

          // bump ptr to start of next pair of vector (8 float32s = 32
          // bytes = +2)
          output_data = (vec_float32 *)output_data + 2;
          break;
        case BFLOAT:
          aiu_vec_lengthen_to_fp32((vec_int16)input_data, &tmp_out_left,
                                   &tmp_out_right);

          *((vec_int16 *)(output_data)) =
              (vec_int16)vec_perm((vec_char8)(tmp_out_left),
                                  (vec_char8)(tmp_out_right), selection_vector);

          // bump ptr to start of next vector (8 bfloats = 16 bytes =
          // +1)
          output_data = (vec_int16 *)output_data + 1;
          break;
        default:
          // this is for completeness but we should never get here, called
          // should have already checked it before calling this function
          return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                             "unknown/invalid pre-transformed data type: %d",
                             ztensor->pre_transformed_desc->type);
          break;
        }

      } // End of for loop

      // e2x at this point points to the group with remaining fields (if
      // any)
      if (remaining_el > 0) { // If none, skip the rest

        idx = (vector unsigned int){e2x, e2x, e2x, e2x} << vec_shift;
        idx_left = idx + idx_left_incr;
        idx_right = idx + idx_right_incr;

        // input_data[] should contain either 0s or residual values from the
        // previous loop, so no need to fill the entries that we don't need
        switch (remaining_el) {
        // remaining_el will never be 8
        case 7:
          // fill input_data[6-0]
          input_data[6] = in_data[idx_right[2]];
        case 6:
          // fill input_data[5-0]
          input_data[5] = in_data[idx_right[1]];
        case 5:
          input_data[4] = in_data[idx_right[0]];
        case 4:
          input_data[3] = in_data[idx_left[3]];
        case 3:
          input_data[2] = in_data[idx_left[2]];
        case 2:
          input_data[1] = in_data[idx_left[1]];
        default:
          // all scenarios fill at least input_data[0]
          input_data[0] = in_data[idx_left[0]];
        };

        // 1) convert the remaining entries and store to tmp_out_x
        // 2) vec_store_len() from tmp_out_x to output_data
        // 3) advances output_data ptr
        switch (ztensor->pre_transformed_desc->type) {
        case FP16:
          tmp_out_16 = aiu_vec_convert_to_fp16(input_data);

          vec_store_len(tmp_out_16, (uint16_t *)output_data,
                        remaining_bytes_to_set - 1);

          output_data =
              (void *)((uintptr_t)output_data + remaining_bytes_to_set);
          break;
        case FP32:
          aiu_vec_lengthen_to_fp32((vec_int16)input_data, &tmp_out_left,
                                   &tmp_out_right);

          // Store left FP32 to output (1 to 4 values), Length is offset
          // by 1. vec_store_len() stores 16 bytes at the most so it
          // won't matter if remaining_bytes_to_set > 16
          vec_store_len(tmp_out_left, (uint32_t *)output_data,
                        remaining_bytes_to_set - 1);

          // If there's more than 4 to convert (remaining_bytes_to_set >
          // 16), store values 5-8
          if (remaining_el > 4) {
            vec_store_len(tmp_out_right,
                          (uint32_t *)((vec_float32 *)output_data + 1),
                          (remaining_bytes_to_set - 16) - 1);
          }

          output_data =
              (void *)((uintptr_t)output_data + remaining_bytes_to_set);
          break;
        case BFLOAT:
          aiu_vec_lengthen_to_fp32((vec_int16)input_data, &tmp_out_left,
                                   &tmp_out_right);

          tmp_out_16 =
              (vec_int16)vec_perm((vec_char8)tmp_out_left,
                                  (vec_char8)tmp_out_right, selection_vector);

          vec_store_len(tmp_out_16, (uint16_t *)output_data,
                        remaining_bytes_to_set - 1);

          output_data =
              (void *)((uintptr_t)output_data + remaining_bytes_to_set);
          break;
        default:
          // this is for completeness but we should never get here, called
          // should have already checked it before calling this function
          return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                             "unknown/invalid pre-transformed data type: %d",
                             ztensor->pre_transformed_desc->type);
          break;
        }
      }

      // add to input offset all the bytes in h (already aligned to page
      // size)
      input_offset += bytes_per_h;
    }

    // input_offset was pushed around in the dims[2-0] loops, so reset
    // it to the next n
    input_offset = in_offset_n + bytes_per_n;
  }

  // handle any FP errors or return success
  return handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

} // End transform_origtensor_smalldim1

/// Unstickification when type is int32 and when dim1 is == 1, only does the
/// following:
///    NHWC -> NHWC, NHWC -> NCHW
/// Does NOT handle concatenated types nor HWCK
///
/// \param[in] ztensor Pointer to zdnn_ztensor, containing data to be
///                    unstickified
/// \param[out] out_buf unsigned integer data buffer to unstickify to
///
void transform_origtensor_int32(const zdnn_ztensor *ztensor,
                                uint32_t *out_buf) {
  uint64_t output_offset =
      0; // moving position as the output is processed, in BYTES
  uint64_t input_offset =
      0; // moving position as the input is processed, in BYTES

  // Since C == 1, NHWC and NCHW are the same.
  for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

    for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {

      for (uint32_t e2x = 0; e2x < ztensor->transformed_desc->dim2; e2x++) {

        out_buf[output_offset] =
            *(uint32_t *)((void *)((uintptr_t)ztensor->buffer + input_offset));

        // push input_offset to the next stick.
        input_offset += AIU_BYTES_PER_STICK;

        // push output_offset to the next element.
        output_offset++;
      }

      // after processing all the w-entries, go to the next 4k-boundary location
      // (aka stick padding)
      input_offset = (input_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                     (-AIU_PAGESIZE_IN_BYTES);
    }
  }
} // End transform_origtensor_int32

/// Given a ztensor and target data buffer, fill the target data buffer
/// with converted data from the sticks
///
/// \param[out] ztensor Pointer to zdnn_ztensor, containing data to be
///                     unstickified
/// \param[in] out_buf data buffer to store unstickified data
///
/// \return ZDNN_OK
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_BUFFER
///         ZDNN_INVALID_STATE
///         ZDNN_CONVERT_FAILURE
///
zdnn_status zdnn_transform_origtensor(const zdnn_ztensor *ztensor,
                                      void *out_buf) {
  zdnn_status status = ZDNN_OK; // Assume success

  if ((status = verify_descriptors_transform_origtensor(ztensor)) != ZDNN_OK) {
    return status;
  }

  // same check as in stickify except no need to check buffer_size
  if (!ztensor->buffer || (uintptr_t)ztensor->buffer & 0xFFF) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // Make sure the buffer has stickified data
  if (ztensor->is_transformed == false) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE, "Tensor not already transformed.",
                       NO_ARG);
  }

  // We allow the type to be ZDNN_BINARY_INT32
  if (ztensor->transformed_desc->type == ZDNN_BINARY_INT32) {
    transform_origtensor_int32(ztensor, out_buf);
    return ZDNN_OK;
  }

  if (ztensor->transformed_desc->layout == ZDNN_NHWC) {
    if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW &&
        ztensor->transformed_desc->dim1 <= 2) {
      if ((status = transform_origtensor_smalldim1(ztensor, out_buf)) !=
          ZDNN_OK) {
        if ((status & WARNING_STATUS_BITMASK) == ZDNN_WARNING) {
          LOG_WARN(
              "transform_origtensor_smalldim1() returned a warning, status = "
              "%08x (%s)\n",
              status, zdnn_get_status_message(status));
        } else {
          LOG_ERROR("transform_origtensor_smalldim1() (ZDNN_NHWC) failed, "
                    "status = %08x (%s)\n",
                    status, zdnn_get_status_message(status));
        }
      }
    } else if ((ztensor->pre_transformed_desc->layout == ZDNN_4DS) &&
               (ztensor->pre_transformed_desc->dim3 != 1)) {

      /*

      s = hidden state size

      e.g., all-timesteps bidir hn output:
      pre_transformed_desc shape of (ts, 2, b, s) (ZDNN_4DS)
          transformed desc shape of (ts, 1, b, out_pad) (ZDNN_NHWC)

      where out_pad = 2 * PADDED(s) (horizontally concatenated output
      with padding between directions)

      to unstickify, build a temp ztensor with equivalent:

      tensor         | tfrmd (dim4, 3, 2, 1) | equivalent
      ---------------+-------------------------------------
      hn_output      | (ts, 1, b, out_pad)   | (ts * 2, 1, b, s)
                     | (1, 1, b, out_pad)    | (2, 1, b, s)
      cf_output      | (1, 1, b, out_pad)    | (2, 1, b, s)
    */

      zdnn_ztensor temp_ztensor;
      zdnn_tensor_desc temp_trans_desc;

      // only pre-transformed dim3 of 2 (BI-directional) is supported
      if (ztensor->pre_transformed_desc->dim3 != 2) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "found ZDNN_4DS but pre-transformed dim3 is neither "
                           "2 nor 1 (found: %d)",
                           ztensor->pre_transformed_desc->dim3);
      }

      // make sure both pre-transformed and transformed agree with each
      // other wrt all timesteps vs final-only timestep
      if (ztensor->pre_transformed_desc->dim4 !=
          ztensor->transformed_desc->dim4) {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "the pre_transformed_desc's dim4 (%d) not the same as "
            "transformed_desc's dim4 (%d)",
            ztensor->pre_transformed_desc->dim4,
            ztensor->transformed_desc->dim4);
      }

      memcpy(&temp_ztensor, ztensor, sizeof(zdnn_ztensor));
      memcpy(&temp_trans_desc, ztensor->transformed_desc,
             sizeof(zdnn_tensor_desc));
      temp_ztensor.transformed_desc = &temp_trans_desc;

      // old transformed: (ts, 1, b, out_pad)
      // new transformed: (ts * 2, 1, b, s)
      temp_trans_desc.dim4 *= 2;
      // pre_transformed_desc->dim1 is the only place we can obtain the
      // non-padded s value
      temp_trans_desc.dim1 = temp_ztensor.pre_transformed_desc->dim1;
      temp_trans_desc.layout = ZDNN_NHWC;

      if ((status = transform_origtensor(&temp_ztensor, out_buf)) != ZDNN_OK) {
        if ((status & WARNING_STATUS_BITMASK) == ZDNN_WARNING) {
          LOG_WARN("transform_origtensor() returned a warning, status = "
                   "%08x (%s)\n",
                   status, zdnn_get_status_message(status));
        } else {
          LOG_ERROR("transform_origtensor() failed (bidir output), status = "
                    "%08x (%s)\n",
                    status, zdnn_get_status_message(status));
        }
      }
    } else {
      if ((status = transform_origtensor(ztensor, out_buf)) != ZDNN_OK) {
        if ((status & WARNING_STATUS_BITMASK) == ZDNN_WARNING) {
          LOG_WARN("zdnn_transform_origtensor() returned a warning, status = "
                   "%08x (%s)\n",
                   status, zdnn_get_status_message(status));
        } else {
          LOG_ERROR("transform_origtensor() (ZDNN_NHWC) failed, status = "
                    "%08x (%s)\n",
                    status, zdnn_get_status_message(status));
        }
      }
    }
    return status;
  } else {
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                       "Invalid layout for transformation: %s",
                       get_data_layout_str(ztensor->transformed_desc->layout));
  }
}
