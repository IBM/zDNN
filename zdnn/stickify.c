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

#include <fenv.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_transform_ztensor)
#pragma export(zdnn_transform_origtensor)
#endif

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
      printf("((cx / AIU_2BYTE_CELLS_PER_STICK) * pages_height_all_h) %d\n",
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
///
/// \return Number of fields converted, or 0 if error
///
uint32_t convert_data_format(void *input_data, zdnn_data_types in_data_fmt,
                             void *output_data, zdnn_data_types out_data_fmt,
                             uint32_t num_fields) {

  uint64_t num_fields_converted = 0;

  // we only care convert to/from ZDNN_DLFLOAT16
  if (out_data_fmt == ZDNN_DLFLOAT16) {
    switch (in_data_fmt) {
    case FP16:
      num_fields_converted = fp16_to_dlf16((uint16_t *)input_data,
                                           (uint16_t *)output_data, num_fields);
      break;
    case FP32:
      num_fields_converted = fp32_to_dlf16((float *)input_data,
                                           (uint16_t *)output_data, num_fields);
      break;
    case BFLOAT:
      num_fields_converted = bfloat_to_dlf16(
          (uint16_t *)input_data, (uint16_t *)output_data, num_fields);
      break;
    default:
      break; // something really wrong, get out and return 0
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
      break; // something really wrong, get out and return 0
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

/// The actual routine for stickification, only does the following:
///    NHWC -> NHWC, NCHW -> NHWC, HWCK -> HWCK
/// Does NOT handle concatenated types.
///
/// \param[in] in_buf data buffer to be stickified
/// \param[out] ztensor Pointer to zdnn_ztensor to contain stickified data
///
/// \return ZDNN_OK
///         ZDNN_CONVERT_FAILURE
///
zdnn_status transform_ztensor(const void *in_buf, zdnn_ztensor *ztensor) {
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

  if (ztensor->transformed_desc->layout == ZDNN_NHWC) {

    // Expected layout is NHWC, stickify normally. Requires a single data
    // buffer.

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
#if defined(__MVS__)
            __dcbt(in_buf + input_offset);
#else
            __builtin_prefetch(in_buf + input_offset, 0);
#endif
            // used for pushing out_offset from w to w+1 (i.e., +
            // AIU_BYTES_PER_STICK)
            uint64_t out_offset_w = output_offset;

            // process each C-stick (i.e., every 64 elements or whatever
            // left in dim1)
            for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1;
                 e1x += AIU_2BYTE_CELLS_PER_STICK) {
              // Prefetch to L1 newest offset to write that HW wouldn't
              // know about
#if defined(__MVS__)
              __dcbtst((void *)((uintptr_t)ztensor->buffer + output_offset));
#else
              __builtin_prefetch(
                  (void *)((uintptr_t)ztensor->buffer + output_offset), 1);
#endif
              fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                      AIU_2BYTE_CELLS_PER_STICK);

              nbr_fields_converted = convert_data_format(
                  (void *)((uintptr_t)in_buf + input_offset),
                  ztensor->pre_transformed_desc->type,
                  (void *)((uintptr_t)ztensor->buffer + output_offset),
                  ztensor->transformed_desc->type, fields_to_convert);

              if (nbr_fields_converted == 0)
                return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

                // Release L1 cacheline for stick. The next "touch" will be
                // from NNPA, and it doesn't need L1 caching.
#if defined(__MVS__)
              __dcbf((void *)((uintptr_t)ztensor->buffer + output_offset));
#else
// No known equivalent fn without dropping to ASM....
#endif
              // push input_offset the next c-stick, fake the multiply by
              // bit-shifting
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
#if defined(__MVS__)
            __dcbt(in_buf + input_offset);
#else
            __builtin_prefetch(in_buf + input_offset, 0);
#endif

            nbr_fields_converted = convert_data_format(
                (void *)((uintptr_t)in_buf + input_offset),
                ztensor->pre_transformed_desc->type, temp_buff,
                ztensor->transformed_desc->type, fields_to_convert);

            if (nbr_fields_converted == 0)
              return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

            // read each entry in temp_buff contiguously and scatter write them
            // to stick area locations AIU_BYTES_PER_STICK bytes apart, i.e.,
            // the same C location of the consecutive C-sticks
            for (uint32_t w = 0; w < fields_to_convert; w++) {
              // Prefetch to L1 newest offset to write that HW wouldn't
              // know about
#if defined(__MVS__)
              __dcbtst((void *)((uintptr_t)ztensor->buffer + output_offset));
#else
              __builtin_prefetch(
                  (void *)((uintptr_t)ztensor->buffer + output_offset), 1);
#endif

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
#if defined(__MVS__)
            __dcbt(in_buf + input_offset);
            __dcbtst((void *)((uintptr_t)ztensor->buffer + output_offset));
#else
            __builtin_prefetch(in_buf + input_offset, 0);
            __builtin_prefetch(
                (void *)((uintptr_t)ztensor->buffer + output_offset), 1);
#endif
            fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                    AIU_2BYTE_CELLS_PER_STICK);

            nbr_fields_converted = convert_data_format(
                (void *)((uintptr_t)in_buf + input_offset),
                ztensor->pre_transformed_desc->type,
                (void *)((uintptr_t)ztensor->buffer + output_offset),
                ztensor->transformed_desc->type, fields_to_convert);

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
  zdnn_status fp_error = handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW));

  if (fp_error != ZDNN_OK) {
    return fp_error;
  }
  // Update the tensor's format to indicate it has been stickified
  ztensor->is_transformed = true;
  return ZDNN_STATUS_OK;

} // End transform_ztensor

/// Converts the input tensor to the supported stick format for execution by
/// zDNN operations.
///
/// \param[out] tensor Pointer to zdnn_ztensor to contain stickified data
/// \param[in] ... 1, 3, or 4 data buffers to be stickified. (1 for most, 3 for
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

  if ((status = verify_pre_transformed_descriptor(
           ztensor->pre_transformed_desc)) != ZDNN_OK) {
    return status;
  }

  if ((status = verify_transformed_descriptor(ztensor->transformed_desc)) !=
      ZDNN_OK) {
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

  if (ztensor->transformed_desc->layout == ZDNN_NHWC ||
      ztensor->transformed_desc->layout == ZDNN_HWCK) {

    const void *data = va_arg(argptr, void *);
    status = transform_ztensor(data, ztensor);

  } else if (ztensor->transformed_desc->layout == ZDNN_FICO ||
             ztensor->transformed_desc->layout == ZDNN_ZRH) {

    do { // do not just return when error, use break instead. we need to
         // va_end() at the end

      uint32_t num_slices = ztensor->transformed_desc->dim4;
      uint64_t gate_data_size =
          get_num_elements(ztensor, ELEMENTS_CONCAT_SINGLE) *
          get_data_type_size(ztensor->pre_transformed_desc->type);
      uint64_t sliced_gate_data_size = gate_data_size / num_slices;
      // 4 gates for FICO otherwise 3 gates (ZRH)
      uint8_t num_gates =
          get_data_layout_num_gates(ztensor->transformed_desc->layout);

      zdnn_tensor_desc temp_pre_tfrmd_desc, temp_tfrmd_desc;

      // Copy the real pre_transformed_desc into temp so we can manipulate it
      // without changing the original.
      memcpy(&temp_pre_tfrmd_desc, ztensor->pre_transformed_desc,
             sizeof(zdnn_tensor_desc));

      // Manipulate the temp pre_trfmd_desc. To build the concatenated ztensor,
      // we process each "slice" (the promoted dim4) of each gate individually.
      // This way the final ztensor can be built to be sliceable along dim4 and
      // each slice will have a complete set of concatenated gates.
      uint32_t pre_trfmd_slices;
      if (ztensor->pre_transformed_desc->layout == ZDNN_3DS) {
        pre_trfmd_slices = ztensor->pre_transformed_desc->dim3;
        temp_pre_tfrmd_desc.dim3 = 1;
        temp_pre_tfrmd_desc.layout = ZDNN_2D;
      } else if (ztensor->pre_transformed_desc->layout == ZDNN_2DS) {
        pre_trfmd_slices = ztensor->pre_transformed_desc->dim2;
        temp_pre_tfrmd_desc.dim2 = 1;
        temp_pre_tfrmd_desc.layout = ZDNN_1D;
      } else {
        status = ZDNN_STATUS(
            ZDNN_INVALID_LAYOUT, "layout %s is not supported for concatenation",
            get_data_layout_str(ztensor->pre_transformed_desc->layout));
        break;
      }

      // Sanity check that the pre_tfrmd and tfrmd descriptors indicate the same
      // number of expected slices.
      if (pre_trfmd_slices != num_slices) {
        status = ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "the pre_transformed_desc's outermost dimension (%d) must be "
            "the same as transformed_desc's dim4 (%d)",
            pre_trfmd_slices, num_slices);
        break;
      }

      // Create a non-sliced, non-concatenated trfmd_desc by using the modified
      // temp_pre_tfrmd_layout.
      if ((status = zdnn_generate_transformed_desc(
               &temp_pre_tfrmd_desc, &temp_tfrmd_desc)) != ZDNN_OK) {
        break;
      }

      uint64_t sliced_gate_buffer_size = zdnn_getsize_ztensor(&temp_tfrmd_desc);

      // Setup the data arrays so each represents a single slice of a single
      // gate. Later when we read from this 2D array, we'll get the data in the
      // order we need for building a final sliceable ztensor.
      void *sliced_gate_data[num_slices][num_gates];
      for (uint8_t gate = 0; gate < num_gates; gate++) {
        void *gate_data = va_arg(argptr, void *);
        for (uint32_t slice = 0; slice < num_slices; slice++) {
          sliced_gate_data[slice][gate] =
              (void *)((uintptr_t)gate_data + (slice * sliced_gate_data_size));
        }
      }

      /*
      Create a temporary ztensor to be used to call zdnn_transform_ztensor()
      multiple times with, as if it's not concatenated.
      */
      zdnn_ztensor temp_ztensor;

      // Setup the temp ztensor, with a non-sliced, non-concatenated
      // buffer_size
      zdnn_init_ztensor(&temp_pre_tfrmd_desc, &temp_tfrmd_desc, &temp_ztensor);

      temp_ztensor.buffer = ztensor->buffer;
      temp_ztensor.buffer_size = sliced_gate_buffer_size;

      // Concatenated tensors require zero padding between the concatenations.
      // Zero out the entire concatened (not temp) buffer so the addresses not
      // set by input values will have zeros.
      size_t total_buffer_size =
          temp_ztensor.buffer_size * num_slices * num_gates;
      memset(ztensor->buffer, 0, total_buffer_size);

      /* Loop sliced_gate_data array to stickify the input data. Because of
       * how sliced_gate_data was built as a 2D array, we can jump around
       * various locations of the original inputs data and read each value
       * only once while building the output ztensor to be in the final
       * desired order.
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
          const void *gate_data_slice = sliced_gate_data[slice][gate];
          // Transform the current slice of the current gate into final
          // ztensor
          if ((status = transform_ztensor(gate_data_slice, &temp_ztensor)) !=
              ZDNN_OK) {
            LOG_ERROR("transform_ztensor() on slice %d of gate data %d "
                      "failed, status = %08x (%s)\n",
                      slice, gate, status, zdnn_get_status_message(status));
            break;
          }

          // Increment the temp_ztensor buffer by one sliced gate size so we
          // write to the correct location in the final output ztensor.
          temp_ztensor.buffer = (void *)((uintptr_t)(temp_ztensor.buffer) +
                                         sliced_gate_buffer_size);

          // Reset temp_ztensor is_transformed so we can recursively call
          // zdnn_transform_ztensor to process each slice of each gate.
          temp_ztensor.is_transformed = false;
        }
        if (status != ZDNN_OK) {
          break;
        }
      }

      if (status == ZDNN_OK) {
        // Set that the output ztensor has completed transformation.
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

  if (ztensor->pre_transformed_desc->layout != ZDNN_NCHW) {

    for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

      // used for pushing input_offset from n to n+1 (i.e., + bytes_per_n)
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
            // Prefetch (read) the next input buffer to be used. The HW should
            // "notice" our sequential accesses and continue them, so we won't
            // need to aggressively prefetch here.
            // Also, Prefetch the new output offset to write that HW wouldn't
            // know about.
#if defined(__MVS__)
            __dcbt((void *)((uintptr_t)ztensor->buffer + input_offset));
            __dcbtst((void *)((uintptr_t)out_buf + output_offset));
#else
            __builtin_prefetch(
                (void *)((uintptr_t)ztensor->buffer + input_offset), 0);
            __builtin_prefetch((void *)((uintptr_t)out_buf + output_offset), 1);
#endif

            fields_to_convert = MIN((ztensor->transformed_desc->dim1 - e1x),
                                    AIU_2BYTE_CELLS_PER_STICK);

            nbr_fields_converted = convert_data_format(
                (void *)((uintptr_t)ztensor->buffer + input_offset),
                ztensor->transformed_desc->type,
                (void *)((uintptr_t)out_buf + output_offset),
                ztensor->pre_transformed_desc->type, fields_to_convert);

            if (nbr_fields_converted == 0)
              return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

            // push output_offset the next c-stick, fake the multiply by
            // bit-shifting
            output_offset += (nbr_fields_converted << output_cell_shift);

            // push input_offset to the next c-stick of the same super
            // c-stick, which is bytes_all_h number of bytes away.
            input_offset += bytes_all_h;
          }

          // input_offset was pushed around in dim1 loops, so reset it to
          // the next w
          input_offset = in_offset_w + AIU_BYTES_PER_STICK;
        }

        // after processing all the w-entries, go to the next 4k-boundary
        // location (aka stick padding)
        input_offset = (input_offset + (AIU_PAGESIZE_IN_BYTES - 1)) &
                       (-AIU_PAGESIZE_IN_BYTES);
      }

      // input_offset was pushed around in the dims[2-0] loops, so reset it
      // to the next n
      input_offset = in_offset_n + bytes_per_n;
    }

  } else {

    // the loops are in N -> C -> H -> W order in order to write the W entries
    // contiguously

    // N
    for (uint32_t e4x = 0; e4x < ztensor->transformed_desc->dim4; e4x++) {

      // used for pushing input_offset from n to n+1 (i.e., + bytes_per_n)
      uint64_t in_offset_n = input_offset;

      // C
      for (uint32_t e1x = 0; e1x < ztensor->transformed_desc->dim1; e1x++) {

        // used for pushing input from c to c+1
        uint64_t in_offset_c = input_offset;

        // H
        for (uint32_t e3x = 0; e3x < ztensor->transformed_desc->dim3; e3x++) {
          // Prefetch (read) the next input buffer to be used. The HW should
          // "notice" our sequential accesses and continue them, so we won't
          // need to aggressively prefetch here.
          // Also, Prefetch the new output offset to write that HW wouldn't
          // know about.
#if defined(__MVS__)
          __dcbt((void *)((uintptr_t)ztensor->buffer + input_offset));
          __dcbtst((void *)((uintptr_t)out_buf + output_offset));
#else
          __builtin_prefetch(
              (void *)((uintptr_t)ztensor->buffer + input_offset), 0);
          __builtin_prefetch((void *)((uintptr_t)out_buf + output_offset), 1);
#endif
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

          // push output_offset to the next H, fake the multiply by bit-shifting
          output_offset += (nbr_fields_converted << output_cell_shift);
        }

        // push in_offset_c to the next C
        in_offset_c += get_data_type_size(ztensor->transformed_desc->type);

        // go to the next C-stick if we're at the end of the current C-stick
        if (!((e1x + 1) % AIU_2BYTE_CELLS_PER_STICK)) {
          in_offset_c = in_offset_c - AIU_BYTES_PER_STICK + bytes_all_h;
        }

        input_offset = in_offset_c;
      }

      // reset input_offset to the next n
      input_offset = in_offset_n + bytes_per_n;
    }
  }
  return handle_fp_errors(
      fetestexcept(FE_UNDERFLOW | FE_INVALID | FE_INEXACT | FE_OVERFLOW)); /*
                                    handle any FP errors or return success */

} // End transform_origtensor

/// Given a ztensor and target data buffer, fill the target data buffer with
/// converted data from the sticks
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
  uint8_t *stick_area = (uint8_t *)ztensor->buffer;

  if ((status = verify_pre_transformed_descriptor(
           ztensor->pre_transformed_desc)) != ZDNN_OK) {
    return status;
  }

  // same check as in stickify except no need to check buffer_size
  if (!stick_area || (((uintptr_t)stick_area) & 0xFFF)) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // Make sure the buffer has stickified data
  if (ztensor->is_transformed == false) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_STATE);
  }

  // we don't do 4DKERNEL unstickify
  if (ztensor->transformed_desc->format != ZDNN_FORMAT_4DFEATURE) {
    return ZDNN_STATUS(ZDNN_INVALID_FORMAT,
                       "Only transforming feature tensor is supported", NO_ARG);
  }

  // We expect the type to be DLFLOAT16
  if (ztensor->transformed_desc->type != ZDNN_DLFLOAT16) {
    return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                       "Only transforming from ZDNN_DLFLOAT16 type is "
                       "supported",
                       NO_ARG);
  }

  if (ztensor->transformed_desc->layout == ZDNN_NHWC) {
    if ((status = transform_origtensor(ztensor, out_buf)) != ZDNN_OK) {
      LOG_ERROR("transform_origtensor() failed, status = %08x (%s)\n", status,
                zdnn_get_status_message(status));
    }
    return status;
  } else {
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                       "Invalid layout for transformation: %s",
                       get_data_layout_str(ztensor->transformed_desc->layout));
  }
}
