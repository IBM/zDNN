// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2023, 2024
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

#include <stdlib.h>
#include <string.h>

#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_getrange_ztensor)
#endif

typedef vector float vec_fp32;
typedef vector signed int vec_int;
typedef vector signed short vec_short;
typedef vector signed char vec_char;

/// Calculates the min and max values and places them in the passed pointers
///
/// \param[in] ztensor The ztensor to calculate range for
/// \param[in] min minimum value of the ztensor. will not be greater than -0.
/// \param[in] max maximum value of the ztensor. will not be less than 0.
///
zdnn_status zdnn_getrange_ztensor(const zdnn_ztensor *ztensor, float *min,
                                  float *max) {
  if (!ztensor->is_transformed) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE, "tensor is not transformed.",
                       NO_ARG);
  }

  zdnn_tensor_desc *tfrmd_desc = ztensor->transformed_desc;

  if (tfrmd_desc->layout != ZDNN_NHWC) {
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                       "Layout must be either NHWC.  layout: %d.",
                       tfrmd_desc->layout);
  }

  if (tfrmd_desc->format != ZDNN_FORMAT_4DFEATURE) {
    return ZDNN_STATUS(ZDNN_INVALID_FORMAT,
                       "Format must be 4DFEATURE.  format: %d.",
                       tfrmd_desc->format);
  }

  if (tfrmd_desc->type != ZDNN_DLFLOAT16) {
    return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Type must be DLFLOAT16.  type: %d.",
                       tfrmd_desc->type);
  }

  // The number of dim1 pages
  const uint32_t c_pages = CEIL(tfrmd_desc->dim1, AIU_2BYTE_CELLS_PER_STICK);
  // The number of elements per-stick in the last dim1 page if not full (0 if
  // last dim1 page is full)
  const uint32_t c_mod = tfrmd_desc->dim1 % AIU_2BYTE_CELLS_PER_STICK;
  // The number of full dim1 pages
  const uint32_t c_mod_page = c_mod == 0 ? c_pages : c_pages - 1;

  // The number of vectors in the last dim1 page (per-stick). Only used when
  // c_mod != 0 (last dim1 page is not full)
  const uint32_t c_mod_vectors = CEIL(c_mod, 8);
  // The number of elements in the last vector if not full (0 if last vector is
  // full)
  const uint32_t v_mod = c_mod % 8;
  // The number of full vectors in the last dim1 page (per-stick)
  const uint32_t v_mod_vector = v_mod == 0 ? c_mod_vectors : c_mod_vectors - 1;
  // The number of empty vectors in the last dim1 page (per-stick)
  const uint32_t c_padding_vectors = 8 - c_mod_vectors;

  // The number of sticks in dim2 with padding
  const uint32_t w_sticks =
      CEIL(tfrmd_desc->dim2, AIU_STICKS_PER_PAGE) * AIU_STICKS_PER_PAGE;
  // The number of dim2 sticks that are padding sticks
  const uint32_t w_padding_sticks = w_sticks - tfrmd_desc->dim2;
  // The number of vectors for dim2 padding sticks
  const uint32_t w_padding_vectors = w_padding_sticks * 8;

  // Min and max values calculated using element-wise operations.
  // Min is unsigned because negative DLFloat values increase in magnitude.
  uint16_t min_val = 0x8000;
  int16_t max_val = 0;

  // Use no-op function to request no saturation be used during data conversion
  void (*skip_func)(const vec_float32 *, const vec_float32 *, vec_float32 *,
                    vec_float32 *) = &skip_saturate_fp32_to_dlf16;

  // Min and max values calculated using vector operations
  vec_int16 min_vec = vec_splats(min_val);
  vec_short max_vec = vec_splats(max_val);

  const void *buffer = (void *)((uintptr_t)ztensor->buffer);

  vec_int16 *min_input_vec = (vec_int16 *)((uint16_t *)buffer);
  vec_short *max_input_vec = (vec_short *)((int16_t *)buffer);

  const uint32_t c_page_h_iterations = c_mod_page * tfrmd_desc->dim3;
  const uint32_t c_page_w_iterations = tfrmd_desc->dim2 * 8;

  // N
  for (uint32_t e4x = 0; e4x < tfrmd_desc->dim4; e4x++) {

    // C Full Pages and H
    for (uint32_t c_page = 0; c_page < c_page_h_iterations; c_page++) {

      // W
      for (uint32_t e2x = 0; e2x < c_page_w_iterations; e2x++) {
        min_vec = vec_max(min_vec, *min_input_vec++);
        max_vec = vec_max(max_vec, *max_input_vec++);
      }

      min_input_vec += w_padding_vectors;
      max_input_vec += w_padding_vectors;
    }

    // C Non-Full Page
    if (c_mod) {

      // H
      for (uint32_t e3x = 0; e3x < tfrmd_desc->dim3; e3x++) {

        // W
        for (uint32_t e2x = 0; e2x < tfrmd_desc->dim2; e2x++) {

          // Full Vectors
          for (uint32_t e1x = 0; e1x < v_mod_vector; e1x++) {
            min_vec = vec_max(min_vec, *min_input_vec++);
            max_vec = vec_max(max_vec, *max_input_vec++);
          }

          // Padded Vector
          if (v_mod) {
            for (uint32_t i = 0; i < v_mod; i++) {
              min_val = MAX(min_val, (*min_input_vec)[i]);
              max_val = MAX(max_val, (*max_input_vec)[i]);
            }

            min_input_vec++;
            max_input_vec++;
          }

          min_input_vec += c_padding_vectors;
          max_input_vec += c_padding_vectors;
        }

        min_input_vec += w_padding_vectors;
        max_input_vec += w_padding_vectors;
      }
    }
  }

  // Compare element-wise results with vector results to obtain true results
  min_val = MAX(min_val, min_vec[0]);
  min_val = MAX(min_val, min_vec[1]);
  min_val = MAX(min_val, min_vec[2]);
  min_val = MAX(min_val, min_vec[3]);
  min_val = MAX(min_val, min_vec[4]);
  min_val = MAX(min_val, min_vec[5]);
  min_val = MAX(min_val, min_vec[6]);
  min_val = MAX(min_val, min_vec[7]);

  max_val = MAX(max_val, max_vec[0]);
  max_val = MAX(max_val, max_vec[1]);
  max_val = MAX(max_val, max_vec[2]);
  max_val = MAX(max_val, max_vec[3]);
  max_val = MAX(max_val, max_vec[4]);
  max_val = MAX(max_val, max_vec[5]);
  max_val = MAX(max_val, max_vec[6]);
  max_val = MAX(max_val, max_vec[7]);

  float range[2];
  void *dlfloat_range = (void *)range;

  // Store results in the range as DLFloat
  ((uint16_t *)dlfloat_range)[0] = min_val;
  ((int16_t *)dlfloat_range)[1] = max_val;

  // Convert range from DLFloat to FP32 in-place
  uint32_t nbr_fields_converted = convert_data_format(
      dlfloat_range, ZDNN_DLFLOAT16, range, FP32, 2, skip_func);

  // Return if there was a conversion error
  if (nbr_fields_converted == 0)
    return ZDNN_STATUS_NO_MSG(ZDNN_CONVERT_FAILURE);

  *min = range[0];
  *max = range[1];

  return ZDNN_STATUS_OK;
}

#ifndef ZDNN_CONFIG_NO_NNPA
/// Computes the bias to be passed to quantized matmul call when operation is
/// MATMUL_OP_ADDITION.
///
/// The original equation is:
///   M = (Sa * Sb) / Sy
///   qc_tilde = Zy - (Sc / Sy) * Zc + (Sc / Sy) * input_c[j] + M * N * Za * Zb
///
/// Given scales are stored as the reciprocal, the modified equation becomes:
///   M = Sy / (Sa * Sb)
///   qc_tilde = Zy - (Sy / Sc) * Zc + (Sy / Sc) * input_c[j] + M * N * Za * Zb
///
/// We can reorder this to:
///   M = Sy / (Sa * Sb)
///   qc_tilde = input_c[j] * (Sy / Sc) + Zy - (Sy / Sc) * Zc + M * N * Za * Zb
///
/// This allows us to pre-compute a scale and offset to apply to input_c[j]:
///   M = Sy / (Sa * Sb)
///   scale = (Sy / Sc)
///   offset = Zy - scale * Zc + M * N * Za * Zb
///   qc_tilde[j] = input_c[j] * scale + offset
///
/// \param[in] input_c The biases ztensor, quantized.
/// \param[in] scale The result of Sy / Sc.
/// \param[in] offset The result of Zy - scale * Zc + M * N * Za * Zb.
/// \param[out] qc_tilde The computed biases ztensor.
///
static void compute_bias(const zdnn_ztensor *input_c, const float scale,
                         const float offset, zdnn_ztensor *qc_tilde) {

  uint64_t in_c_offset = 0; // moving position as input_c is processed, in BYTES
  uint64_t out_offset = 0;  // moving position as output is processed, in BYTES

  uint32_t remaining_fields;     // number of fields remaining
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  // loop invariant values
  const uint64_t in_c_bytes_per_n =
      CEIL(input_c->transformed_desc->dim1, AIU_1BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t out_bytes_per_n =
      CEIL(qc_tilde->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  vec_fp32 vec_scale = vec_splats(scale);
  vec_fp32 vec_offset = vec_splats(offset);

  for (uint32_t e4x = 0; e4x < input_c->transformed_desc->dim4; e4x++) {

    // Used for pushing in_c_offset from n to n+1 (i.e., + in_c_bytes_per_n)
    const uint64_t in_c_offset_n = in_c_offset;
    // Used for pushing out_offset from n to n+1 (i.e., + out_bytes_per_n)
    const uint64_t out_offset_n = out_offset;

    // input_c has 128 (AIU_1BYTE_CELLS_PER_STICK) int8 elements per-stick but
    // qc_tilde has 64 (AIU_2BYTE_CELLS_PER_STICK) dlfloat16 elements per-stick.
    //
    // This means we iterate over 128 (AIU_1BYTE_CELLS_PER_STICK) input_c
    // elements at a time, but split them into two groups of 64
    // (AIU_2BYTE_CELLS_PER_STICK).
    for (uint32_t e1x = 0; e1x < input_c->transformed_desc->dim1;
         e1x += AIU_1BYTE_CELLS_PER_STICK) {

      vec_char *in_c_vec =
          (vec_char *)((void *)((uintptr_t)input_c->buffer + in_c_offset));

#if defined(__MVS__)
      vec_int16 *qc_tilde_vec =
          (vec_int16 *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#else
      vec_short *qc_tilde_vec =
          (vec_short *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#endif

      remaining_fields = input_c->transformed_desc->dim1 - e1x;
      fields_to_convert = MIN(remaining_fields, AIU_2BYTE_CELLS_PER_STICK);
      nbr_fields_converted = 0;

      // First AIU_2BYTE_CELLS_PER_STICK of AIU_1BYTE_CELLS_PER_STICK
      while (nbr_fields_converted < fields_to_convert) {
        // Load high end of in_c_vec (first 8 elements) into temp_int16
        vec_short temp_int16 = vec_unpackh(*in_c_vec);

#if defined(__MVS__)
        vec_fp32 temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        vec_fp32 temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset),
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset),
            0);
#endif

        nbr_fields_converted += 8;

        if (nbr_fields_converted >= fields_to_convert)
          break;

        // Load low end of in_c_vec (final 8 elements) into temp_int16
        temp_int16 = vec_unpackl(*in_c_vec);

#if defined(__MVS__)
        temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset),
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset),
            0);
#endif

        // Push in_c_vec to the next 16 int8 elements
        in_c_vec++;

        nbr_fields_converted += 8;
      }

      if (nbr_fields_converted >= remaining_fields)
        break;

      // push out_offset to the next c-stick of the same super c-stick, which
      // is AIU_PAGESIZE_IN_BYTES number of bytes away since dim3 and dim2 == 1
      out_offset += AIU_PAGESIZE_IN_BYTES;

#if defined(__MVS__)
      qc_tilde_vec =
          (vec_int16 *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#else
      qc_tilde_vec =
          (vec_short *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#endif

      // Final AIU_2BYTE_CELLS_PER_STICK of AIU_1BYTE_CELLS_PER_STICK
      while (nbr_fields_converted < remaining_fields) {
        // Load high end of in_c_vec (first 8 elements) into temp_int16
        vec_short temp_int16 = vec_unpackh(*in_c_vec);

#if defined(__MVS__)
        vec_fp32 temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        vec_fp32 temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset),
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset),
            0);
#endif

        nbr_fields_converted += 8;

        if (nbr_fields_converted >= remaining_fields)
          break;

        // Load low end of in_c_vec (final 8 elements) into temp_int16
        temp_int16 = vec_unpackl(*in_c_vec);

#if defined(__MVS__)
        temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset),
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset),
            0);
#endif

        // Push in_c_vec to the next 16 int8 elements.
        in_c_vec++;

        nbr_fields_converted += 8;
      }

      in_c_offset += AIU_PAGESIZE_IN_BYTES;
      out_offset += AIU_PAGESIZE_IN_BYTES;
    }

    // reset in_c_offset to the next n
    in_c_offset = in_c_offset_n + in_c_bytes_per_n;
    // reset out_offset to the next n
    out_offset = out_offset_n + out_bytes_per_n;
  }

  qc_tilde->is_transformed = true;
}

/// Computes the folded bias to be passed to quantized matmul call when
/// operation is MATMUL_OP_ADDITION. Zb should be equal to 0, meaning the
/// correction term for input_a is also equal to 0. This allows the correction
/// term for input_b to be folded into qc_tilde, which removes the need for
/// correction being applied after the quantized matmul call.
///
/// The original equation for qc_tilde is:
///   M = (Sa * Sb) / Sy
///   qc_tilde = Zy - (Sc / Sy) * Zc + (Sc / Sy) * input_c[j] + M * N * Za * Zb
///
/// Given scales are stored as the reciprocal, the modified equation becomes:
///   M = Sy / (Sa * Sb)
///   qc_tilde = Zy - (Sy / Sc) * Zc + (Sy / Sc) * input_c[j] + M * N * Za * Zb
///
/// We can reorder this to:
///   M = Sy / (Sa * Sb)
///   qc_tilde = input_c[j] * (Sy / Sc) + Zy - (Sy / Sc) * Zc + M * N * Za * Zb
///
/// This allows us to pre-compute a scale and offset to apply to input_c[j]:
///   M = Sy / (Sa * Sb).
///   scale = (Sy / Sc)
///   offset = Zy - scale * Zc * M * N * Za * Zb.
///   qc_tilde[j] = input_c[j] * scale + offset
///
/// The original equation for the correction term for input_b is:
///   M = (Sa * Sb) / Sy
///   term_b = M * Za * sum(input_b[:,j])
///
/// Given scales are stored as the reciprocal, the modified equation becomes:
///   M = Sy / (Sa * Sb)
///   term_b = M * Za * sum(input_b[:,j])
///
/// This gives us the final equation:
///   M = Sy / (Sa * Sb)
///   MZa = M * Za
///   scale = (Sy / Sc)
///   offset = Zy - scale * Zc + M * N * Za * Zb.
///   qc_tilde[j] = input_c[j] * scale + offset - MZa * sum(input_b[:,j])
///
/// \param[in] input_c The biases ztensor, quantized.
/// \param[in] scale The result of Sy / Sc.
/// \param[in] offset The result of Zy - scale * Zc + M * N * Za * Zb.
/// \param[in] MZa The result of M * Za.
/// \param[out] qc_tilde The computed biases ztensor.
///
static void compute_folded_bias(const zdnn_ztensor *input_b,
                                const zdnn_ztensor *input_c, const float scale,
                                const float offset, const float MZa,
                                zdnn_ztensor *qc_tilde) {

  uint64_t in_b_offset = 0; // moving position as input_b is processed, in BYTES
  uint64_t in_c_offset = 0; // moving position as input_c is processed, in BYTES
  uint64_t out_offset = 0;  // moving position as output is processed, in BYTES

  uint32_t remaining_fields;     // number of fields remaining
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  // loop invariant values
  const uint64_t in_b_bytes_all_w =
      CEIL(input_b->transformed_desc->dim2, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t in_b_bytes_all_w_twice = in_b_bytes_all_w * 2;

  const uint64_t in_b_bytes_per_n =
      CEIL(input_b->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      in_b_bytes_all_w;

  const uint64_t in_c_bytes_per_n =
      CEIL(input_c->transformed_desc->dim1, AIU_1BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t out_bytes_per_n =
      CEIL(qc_tilde->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  vec_fp32 vec_scale = vec_splats(scale);
  vec_fp32 vec_offset = vec_splats(offset);
  vec_fp32 vec_MZa = vec_splats(MZa);

  for (uint32_t e4x = 0; e4x < input_c->transformed_desc->dim4; e4x++) {

    // Used for pushing in_b_offset from n to n+1 (i.e., + in_b_bytes_per_n)
    const uint64_t in_b_offset_n = in_b_offset;
    // Used for pushing in_c_offset from n to n+1 (i.e., + in_c_bytes_per_n)
    const uint64_t in_c_offset_n = in_c_offset;
    // Used for pushing out_offset from n to n+1 (i.e., + out_bytes_per_n)
    const uint64_t out_offset_n = out_offset;

    // input_c has 128 (AIU_1BYTE_CELLS_PER_STICK) int8 elements per-stick but
    // qc_tilde has 64 (AIU_2BYTE_CELLS_PER_STICK) dlfloat16 elements per-stick.
    //
    // input_b has 128 (AIU_1BYTE_CELLS_PER_STICK) int8 elements per-stick but
    // this consists of 64 (AIU_2BYTE_CELLS_PER_STICK) elements from the current
    // dim2 and 64 (AIU_2BYTE_CELLS_PER_STICK) elements from the next dim2.
    //
    // Using w0 and w1 to denote the first and second dim2, respectively, a full
    // stick would look like:
    //
    // [w0_0, w1_0, w0_1, w1_1, ... w0_62, w1_62, w0_63, w1_63]
    //
    // Since we are performing a summation along dim2, we can add each pair of
    // w0_x and w1_x to form a stick with 64 (AIU_2BYTE_CELLS_PER_STICK) int16
    // elements:
    //
    // [w01_0, w01_1, ... w01_62, w01_63]
    //
    // These 64 (AIU_2BYTE_CELLS_PER_STICK) elements are then summed with the
    // remaining n dim2 into int32 elements:
    //
    // [w0n_0, w0n_1, ... w0n_62, w0n_63]
    //
    // This ensures that both:
    //  1 - There is no overflow/underflow within the summation
    //  2 - The summation is 32 bits long, which can be converted to float
    //
    // This means we iterate over 128 (AIU_1BYTE_CELLS_PER_STICK) input_c
    // elements at a time, but split them into two groups of 64
    // (AIU_2BYTE_CELLS_PER_STICK).
    for (uint32_t e1x = 0; e1x < input_c->transformed_desc->dim1;
         e1x += AIU_1BYTE_CELLS_PER_STICK) {

      const uint64_t in_b_w_offset = in_b_offset;

      vec_char *in_c_vec =
          (vec_char *)((void *)((uintptr_t)input_c->buffer + in_c_offset));

#if defined(__MVS__)
      vec_int16 *qc_tilde_vec =
          (vec_int16 *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#else
      vec_short *qc_tilde_vec =
          (vec_short *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#endif

      remaining_fields = input_c->transformed_desc->dim1 - e1x;
      fields_to_convert = MIN(remaining_fields, AIU_2BYTE_CELLS_PER_STICK);
      nbr_fields_converted = 0;

      // First AIU_2BYTE_CELLS_PER_STICK of AIU_1BYTE_CELLS_PER_STICK
      while (nbr_fields_converted < fields_to_convert) {
        // Load high end of in_c_vec (first 8 elements) into temp_int16
        vec_short temp_int16 = vec_unpackh(*in_c_vec);

        vec_fp32 temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        vec_fp32 temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        vec_int summ_vec_hi = vec_splats(0);
        vec_int summ_vec_lo = vec_splats(0);

        vec_char *in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi -= (vec_float(summ_vec_hi) * vec_MZa);
        temp_float_lo -= (vec_float(summ_vec_lo) * vec_MZa);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        nbr_fields_converted += 8;

        if (nbr_fields_converted >= fields_to_convert)
          break;

        // Load low end of in_c_vec (final 8 elements) into temp_int16
        temp_int16 = vec_unpackl(*in_c_vec);

        temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        summ_vec_hi = vec_splats(0);
        summ_vec_lo = vec_splats(0);

        in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi -= (vec_float(summ_vec_hi) * vec_MZa);
        temp_float_lo -= (vec_float(summ_vec_lo) * vec_MZa);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        // Push in_c_vec to the next 16 int8 elements.
        in_c_vec++;

        nbr_fields_converted += 8;
      }

      if (nbr_fields_converted >= remaining_fields)
        break;

      in_b_offset = in_b_w_offset + in_b_bytes_all_w;
      out_offset += AIU_PAGESIZE_IN_BYTES;

#if defined(__MVS__)
      qc_tilde_vec =
          (vec_int16 *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#else
      qc_tilde_vec =
          (vec_short *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#endif

      // Final AIU_2BYTE_CELLS_PER_STICK of AIU_1BYTE_CELLS_PER_STICK
      while (nbr_fields_converted < remaining_fields) {
        // Load high end of in_c_vec (first 8 elements) into temp_int16
        vec_short temp_int16 = vec_unpackh(*in_c_vec);

        vec_fp32 temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        vec_fp32 temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        vec_int summ_vec_hi = vec_splats(0);
        vec_int summ_vec_lo = vec_splats(0);

        vec_char *in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi -= (vec_float(summ_vec_hi) * vec_MZa);
        temp_float_lo -= (vec_float(summ_vec_lo) * vec_MZa);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        nbr_fields_converted += 8;

        if (nbr_fields_converted >= remaining_fields)
          break;

        // Load low end of in_c_vec (final 8 elements) into temp_int16
        temp_int16 = vec_unpackl(*in_c_vec);

        temp_float_hi =
            vec_madd(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        temp_float_lo =
            vec_madd(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        summ_vec_hi = vec_splats(0);
        summ_vec_lo = vec_splats(0);

        in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi -= (vec_float(summ_vec_hi) * vec_MZa);
        temp_float_lo -= (vec_float(summ_vec_lo) * vec_MZa);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        // Push in_c_vec to the next 16 int8 elements.
        in_c_vec++;

        nbr_fields_converted += 8;
      }

      in_b_offset = in_b_w_offset + in_b_bytes_all_w_twice;
      in_c_offset += AIU_PAGESIZE_IN_BYTES;
      out_offset += AIU_PAGESIZE_IN_BYTES;
    }

    // reset in_b_offset to the next n
    in_b_offset = in_b_offset_n + in_b_bytes_per_n;
    // reset in_c_offset to the next n
    in_c_offset = in_c_offset_n + in_c_bytes_per_n;
    // reset out_offset to the next n
    out_offset = out_offset_n + out_bytes_per_n;
  }

  qc_tilde->is_transformed = true;
}

/// Computes the folded bias to be passed to quantized matmul call when
/// operation is not MATMUL_OP_ADDITION. Zb should be equal to 0, meaning the
/// correction term for input_a is also equal to 0. This allows the correction
/// term for input_b to be folded into qc_tilde. This is required for comparison
/// operations since the correction term cannot be applied before the comparison
/// (given comparison happen in hardware).
///
/// The original equation for qc_tilde is:
///   qc_tilde = Sc / (Sa * Sb) * (input_c[j] - Zc) + Za * sum(input_b[:,j])
///
/// Given scales are stored as the reciprocal, the modified equation becomes:
///   qc_tilde = (Sa * Sb) / Sc * (input_c[j] - Zc) + Za * sum(input_b[:,j])
///
/// We can reorder this to:
///   qc_tilde = input_c[j] * (Sa * Sb) / Sc - (Sa * Sb) / Sc * Zc +
///              Za * sum(input_b[:,j])
///
/// This allows us to pre-compute a scale and offset to apply to input_c[j]:
///   scale = (Sa * Sb) / Sc
///   offset = scale * Zc
///   qc_tilde = input_c[j] * scale - offset + Za * sum(input_b[:,j])
///
/// \param[in] input_b The weights ztensor, quantized.
/// \param[in] input_c The biases ztensor, quantized.
/// \param[in] scale The result of Sy / Sc.
/// \param[in] offset The result of Zy - scale * Zc + M * N * Za * Zb.
/// \param[in] Za The scale for input_a.
/// \param[out] qc_tilde The computed biases ztensor.
///
static void compute_comparison_bias(const zdnn_ztensor *input_b,
                                    const zdnn_ztensor *input_c,
                                    const float scale, const float offset,
                                    const float Za, zdnn_ztensor *qc_tilde) {

  uint64_t in_b_offset = 0; // moving position as input_b is processed, in BYTES
  uint64_t in_c_offset = 0; // moving position as input_c is processed, in BYTES
  uint64_t out_offset = 0;  // moving position as output is processed, in BYTES

  uint32_t remaining_fields;     // number of fields remaining
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  // loop invariant values
  const uint64_t in_b_bytes_all_w =
      CEIL(input_b->transformed_desc->dim2, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t in_b_bytes_all_w_twice = in_b_bytes_all_w * 2;

  const uint64_t in_b_bytes_per_n =
      CEIL(input_b->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      in_b_bytes_all_w;

  const uint64_t in_c_bytes_per_n =
      CEIL(input_c->transformed_desc->dim1, AIU_1BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t out_bytes_per_n =
      CEIL(qc_tilde->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  vec_fp32 vec_scale = vec_splats(scale);
  vec_fp32 vec_offset = vec_splats(offset);
  vec_fp32 vec_Za = vec_splats(Za);

  for (uint32_t e4x = 0; e4x < input_c->transformed_desc->dim4; e4x++) {

    // Used for pushing in_b_offset from n to n+1 (i.e., + in_b_bytes_per_n)
    const uint64_t in_b_offset_n = in_b_offset;
    // Used for pushing in_c_offset from n to n+1 (i.e., + in_c_bytes_per_n)
    const uint64_t in_c_offset_n = in_c_offset;
    // Used for pushing out_offset from n to n+1 (i.e., + out_bytes_per_n)
    const uint64_t out_offset_n = out_offset;

    // input_c has 128 (AIU_1BYTE_CELLS_PER_STICK) int8 elements per-stick but
    // qc_tilde has 64 (AIU_2BYTE_CELLS_PER_STICK) dlfloat16 elements per-stick.
    //
    // input_b has 128 (AIU_1BYTE_CELLS_PER_STICK) int8 elements per-stick but
    // this consists of 64 (AIU_2BYTE_CELLS_PER_STICK) elements from the current
    // dim2 and 64 (AIU_2BYTE_CELLS_PER_STICK) elements from the next dim2.
    //
    // Using w0 and w1 to denote the first and second dim2, respectively, a full
    // stick would look like:
    //
    // [w0_0, w1_0, w0_1, w1_1, ... w0_62, w1_62, w0_63, w1_63]
    //
    // Since we are performing a summation along dim2, we can add each pair of
    // w0_x and w1_x to form a stick with 64 (AIU_2BYTE_CELLS_PER_STICK) int16
    // elements:
    //
    // [w01_0, w01_1, ... w01_62, w01_63]
    //
    // These 64 (AIU_2BYTE_CELLS_PER_STICK) elements are then summed with the
    // remaining n dim2 into int32 elements:
    //
    // [w0n_0, w0n_1, ... w0n_62, w0n_63]
    //
    // This ensures that both:
    //  1 - There is no overflow/underflow within the summation
    //  2 - The summation is 32 bits long, which can be converted to float
    //
    // This means we iterate over 128 (AIU_1BYTE_CELLS_PER_STICK) input_c
    // elements at a time, but split them into two groups of 64
    // (AIU_2BYTE_CELLS_PER_STICK).
    for (uint32_t e1x = 0; e1x < input_c->transformed_desc->dim1;
         e1x += AIU_1BYTE_CELLS_PER_STICK) {

      const uint64_t in_b_w_offset = in_b_offset;

      vec_char *in_c_vec =
          (vec_char *)((void *)((uintptr_t)input_c->buffer + in_c_offset));

#if defined(__MVS__)
      vec_int16 *qc_tilde_vec =
          (vec_int16 *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#else
      vec_short *qc_tilde_vec =
          (vec_short *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#endif

      remaining_fields = input_c->transformed_desc->dim1 - e1x;
      fields_to_convert = MIN(remaining_fields, AIU_2BYTE_CELLS_PER_STICK);
      nbr_fields_converted = 0;

      // First AIU_2BYTE_CELLS_PER_STICK of AIU_1BYTE_CELLS_PER_STICK
      while (nbr_fields_converted < fields_to_convert) {
        // Load high end of in_c_vec (first 8 elements) into temp_int16
        vec_short temp_int16 = vec_unpackh(*in_c_vec);

        vec_fp32 temp_float_hi =
            vec_msub(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        vec_fp32 temp_float_lo =
            vec_msub(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        vec_int summ_vec_hi = vec_splats(0);
        vec_int summ_vec_lo = vec_splats(0);

        vec_char *in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi += (vec_float(summ_vec_hi) * vec_Za);
        temp_float_lo += (vec_float(summ_vec_lo) * vec_Za);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        nbr_fields_converted += 8;

        if (nbr_fields_converted >= fields_to_convert)
          break;

        // Load low end of in_c_vec (final 8 elements) into temp_int16
        temp_int16 = vec_unpackl(*in_c_vec);

        temp_float_hi =
            vec_msub(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        temp_float_lo =
            vec_msub(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        summ_vec_hi = vec_splats(0);
        summ_vec_lo = vec_splats(0);

        // Zero out summ_vec, which will hold the summation for W dim.
        summ_vec_hi = vec_splats(0);
        summ_vec_lo = vec_splats(0);

        in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi += (vec_float(summ_vec_hi) * vec_Za);
        temp_float_lo += (vec_float(summ_vec_lo) * vec_Za);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        // Push in_c_vec to the next 16 int8 elements.
        in_c_vec++;

        nbr_fields_converted += 8;
      }

      if (nbr_fields_converted >= remaining_fields)
        break;

      in_b_offset = in_b_w_offset + in_b_bytes_all_w;
      out_offset += AIU_PAGESIZE_IN_BYTES;

#if defined(__MVS__)
      qc_tilde_vec =
          (vec_int16 *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#else
      qc_tilde_vec =
          (vec_short *)((void *)((uintptr_t)qc_tilde->buffer + out_offset));
#endif

      // Final AIU_2BYTE_CELLS_PER_STICK of AIU_1BYTE_CELLS_PER_STICK
      while (nbr_fields_converted < remaining_fields) {
        // Load high end of in_c_vec (first 8 elements) into temp_int16
        vec_short temp_int16 = vec_unpackh(*in_c_vec);

        vec_fp32 temp_float_hi =
            vec_msub(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        vec_fp32 temp_float_lo =
            vec_msub(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        vec_int summ_vec_hi = vec_splats(0);
        vec_int summ_vec_lo = vec_splats(0);

        vec_char *in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi += (vec_float(summ_vec_hi) * vec_Za);
        temp_float_lo += (vec_float(summ_vec_lo) * vec_Za);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        nbr_fields_converted += 8;

        if (nbr_fields_converted >= remaining_fields)
          break;

        // Load low end of in_c_vec (final 8 elements) into temp_int16
        temp_int16 = vec_unpackl(*in_c_vec);

        temp_float_hi =
            vec_msub(vec_float(vec_unpackh(temp_int16)), vec_scale, vec_offset);
        temp_float_lo =
            vec_msub(vec_float(vec_unpackl(temp_int16)), vec_scale, vec_offset);

        // Zero out summ_vec, which will hold the summation for W dim.
        summ_vec_hi = vec_splats(0);
        summ_vec_lo = vec_splats(0);

        in_b_vec =
            (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

        for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
             e2x++) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_hi
          summ_vec_hi += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every 2 int16 elements into summ_vec_lo
          summ_vec_lo += (vec_int){
              temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
              temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

          in_b_vec += 8;
        }

        if (input_b->transformed_desc->dim2 % 2) {
          // Load high end of in_b_vec (first 8 elements) into temp_int16
          temp_int16 = vec_unpackh(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_hi
          summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
          // Load low end of in_b_vec (final 8 elements) into temp_int16
          temp_int16 = vec_unpackl(*in_b_vec);
          // Perform sum operation on every other int16 element into
          // summ_vec_lo
          summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2], temp_int16[4],
                                   temp_int16[6]};
        }

        temp_float_hi += (vec_float(summ_vec_hi) * vec_Za);
        temp_float_lo += (vec_float(summ_vec_lo) * vec_Za);

#if defined(__MVS__)
        *qc_tilde_vec++ =
            aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_float_hi),
                                    *(vec_float32 *)((void *)&temp_float_lo));
#else
        *qc_tilde_vec++ = vec_round_from_fp32(temp_float_hi, temp_float_lo, 0);
#endif

        in_b_offset += 16;

        // Push in_c_vec to the next 16 int8 elements.
        in_c_vec++;

        nbr_fields_converted += 8;
      }

      in_b_offset = in_b_w_offset + in_b_bytes_all_w_twice;
      in_c_offset += AIU_PAGESIZE_IN_BYTES;
      out_offset += AIU_PAGESIZE_IN_BYTES;
    }

    // reset in_b_offset to the next n
    in_b_offset = in_b_offset_n + in_b_bytes_per_n;
    // reset in_c_offset to the next n
    in_c_offset = in_c_offset_n + in_c_bytes_per_n;
    // reset out_offset to the next n
    out_offset = out_offset_n + out_bytes_per_n;
  }

  qc_tilde->is_transformed = true;
}

/// Performs the actual quantized matmul (HW) processing.
///
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, quantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The computed biases ztensor, unquantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
static zdnn_status
quantized_matmul(const uint8_t function_code, const zdnn_ztensor *input_a,
                 const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
                 zdnn_matmul_ops op_type, zdnn_ztensor *output) {

  float sz[3] = {input_a->rec_scale, input_b->rec_scale, output->rec_scale};

  if (op_type != MATMUL_OP_ADDITION) {
    sz[0] = 1.f;
    sz[1] = 1.f;
    sz[2] = 1.f;
  }

  vec_float32 sz_vec = (vec_float32)vec_load_len(sz, 11);
  vec_float32 zero_vec = (vec_float32){0, 0, 0, 0};
  vec_int16 converted_sz = aiu_vec_round_from_fp32(sz_vec, zero_vec);

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_matmul *fsp_matmul = (func_sp_parms_matmul *)&fsp;

  fsp_matmul->parm1.operation = op_type;
  fsp_matmul->parm3.rec_scale = converted_sz[0];
  fsp_matmul->parm5.rec_scale = converted_sz[1];
  fsp_matmul->parm7.rec_scale = converted_sz[2];

  zdnn_status status;

  // Perform matmul against input features, weights, and modified bias
  if ((status = aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, function_code,
                                      input_a, input_b, input_c, output, NULL,
                                      0, &fsp)) != ZDNN_OK) {
    return ZDNN_STATUS(
        status, "Failure within Quantized Matmul call (status = %d)\n", status);
  }

  return ZDNN_STATUS_OK;
}

/// Performs the actual quantized matmul (HW) processing.
///
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, unquantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The computed biases ztensor, unquantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
static zdnn_status quantized_matmul_on_the_fly(
    const uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    zdnn_ztensor *output) {

  float sz[4] = {input_a->rec_scale, input_a->offset, input_b->rec_scale,
                 output->rec_scale};

  if (op_type != MATMUL_OP_ADDITION) {
    sz[2] = 1.f;
    sz[3] = sz[0];
  }

  vec_float32 sz_vec = (vec_float32)vec_load_len(sz, 15);
  vec_float32 zero_vec = (vec_float32){0, 0, 0, 0};
  vec_int16 converted_sz = aiu_vec_round_from_fp32(sz_vec, zero_vec);

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_matmul *fsp_matmul = (func_sp_parms_matmul *)&fsp;

  fsp_matmul->parm1.operation = op_type;
  fsp_matmul->parm3.rec_scale = converted_sz[0];
  fsp_matmul->parm4.offset = converted_sz[1];
  fsp_matmul->parm5.rec_scale = converted_sz[2];
  fsp_matmul->parm7.rec_scale = converted_sz[3];
  fsp_matmul->parm9.clip_min = clip_min;
  fsp_matmul->parm10.clip_max = clip_max;

  zdnn_status status;

  // Perform matmul against input features, weights, and modified bias
  if ((status = aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, function_code,
                                      input_a, input_b, input_c, output, NULL,
                                      0, &fsp)) != ZDNN_OK) {
    return ZDNN_STATUS(
        status, "Failure within Quantized Matmul call (status = %d)\n", status);
  }

  return ZDNN_STATUS_OK;
}

/// Dequantize vec_hi and vec_lo using vec_scale and vec_offset.
///
/// \param[in] vec_hi The first vector to dequantize.
/// \param[in] vec_lo The second vector to dequantize.
/// \param[out] vec_scale The scale factor for quantization.
/// \param[out] vec_offset The offset for quantization.
///
static void apply_dequantization(vec_fp32 *vec_hi, vec_fp32 *vec_lo,
                                 vec_fp32 vec_scale, vec_fp32 vec_offset) {
  *vec_hi = ((*vec_hi - vec_offset) * vec_scale);
  *vec_lo = ((*vec_lo - vec_offset) * vec_scale);
}

/// No op that does not dequantize vec_hi or vec_lo.
///
/// \param[in] vec_hi The first vector to dequantize.
/// \param[in] vec_lo The second vector to dequantize.
/// \param[out] vec_scale The scale factor for quantization.
/// \param[out] vec_offset The offset for quantization.
///
static void skip_dequantization(vec_fp32 *vec_hi, vec_fp32 *vec_lo,
                                vec_fp32 vec_scale, vec_fp32 vec_offset) {}

/// No op that does not clip and round vec_hi or vec_lo.
///
/// \param[in] vec_hi The first vector to clip and round
/// \param[in] vec_clip_min The clip minimum
/// \param[in] vec_lo The second vector to clip and round
/// \param[in] vec_clip_max The clip maximum
///
static void skip_clip_and_round(vec_fp32 *vec_hi, vec_fp32 *vec_clip_min,
                                vec_fp32 *vec_clip_max) {}

/// Clip and round vec_hi using vec_clip_min and vec_clip_max
///
/// \param[in] vec_hi The first vector to clip and round
/// \param[in] vec_clip_min The clip minimum
/// \param[in] vec_lo The second vector to clip and round
/// \param[in] vec_clip_max The clip maximum
///
static void clip_and_round_hi(vec_fp32 *vec_hi, vec_fp32 *vec_clip_min,
                              vec_fp32 *vec_clip_max) {
  *vec_hi = vec_min(vec_max(vec_round(*vec_hi), *vec_clip_min), *vec_clip_max);
}

/// Clip and round vec_lo using vec_clip_min and vec_clip_max
///
/// \param[in] vec_hi The first vector to clip and round
/// \param[in] vec_clip_min The clip minimum
/// \param[in] vec_lo The second vector to clip and round
/// \param[in] vec_clip_max The clip maximum
///
static void clip_and_round_lo(vec_fp32 *vec_lo, vec_fp32 *vec_clip_min,
                              vec_fp32 *vec_clip_max) {
  *vec_lo = vec_min(vec_max(vec_round(*vec_lo), *vec_clip_min), *vec_clip_max);
}

/// Clips the output between clip_min and clip_max.
///
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
static void apply_clipping(const int8_t clip_min, const int8_t clip_max,
                           zdnn_ztensor *output, const bool dequantize,
                           const bool disable_clipping) {

  // return immediately if dequantize=false or disable_clipping=true
  // so we don't perform an unstickify and then immediately a stickify
  // which is unneeded when dequantize=false and disable_clipping=true
  if (!dequantize && disable_clipping) {
    return;
  }
  uint64_t out_offset = 0; // moving position as output is processed, in BYTES

  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  const uint64_t out_bytes_all_w =
      CEIL(output->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t out_bytes_per_n =
      CEIL(output->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      out_bytes_all_w;

  vec_fp32 vec_clip_min = vec_splats((float)clip_min);
  vec_fp32 vec_clip_max = vec_splats((float)clip_max);
  vec_fp32 vec_scale = vec_splats((1.f / output->rec_scale));
  vec_fp32 vec_offset = vec_splats(output->offset);

  void (*clip_round_hi_func)(vec_fp32 *, vec_fp32 *, vec_fp32 *) =
      disable_clipping ? &skip_clip_and_round : &clip_and_round_hi;
  void (*clip_round_lo_func)(vec_fp32 *, vec_fp32 *, vec_fp32 *) =
      disable_clipping ? &skip_clip_and_round : &clip_and_round_lo;
  void (*deq_func)(vec_fp32 *, vec_fp32 *, vec_fp32, vec_fp32) =
      dequantize ? &apply_dequantization : &skip_dequantization;

  for (uint32_t e4x = 0; e4x < output->transformed_desc->dim4; e4x++) {

    // Used for pushing out_offset from n to n+1 (i.e., + out_bytes_per_n)
    const uint64_t out_offset_n = out_offset;

    for (uint32_t e2x = 0; e2x < output->transformed_desc->dim2; e2x++) {

      const uint64_t out_w_offset = out_offset;

      for (uint32_t e1x = 0; e1x < output->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
#if defined(__MVS__)
        vec_int16 *output_vec =
            (vec_int16 *)((void *)((uintptr_t)output->buffer + out_offset));
#else
        vec_short *output_vec =
            (vec_short *)((void *)((uintptr_t)output->buffer + out_offset));
#endif

        fields_to_convert = MIN(output->transformed_desc->dim1 - e1x,
                                AIU_2BYTE_CELLS_PER_STICK);
        nbr_fields_converted = 0;

        while (nbr_fields_converted < fields_to_convert) {
#if defined(__MVS__)
          vec_fp32 temp_vec_hi, temp_vec_lo;
          aiu_vec_lengthen_to_fp32(*output_vec,
                                   (vec_float32 *)((void *)&temp_vec_hi),
                                   (vec_float32 *)((void *)&temp_vec_lo));
#else
          vec_fp32 temp_vec_hi = vec_extend_to_fp32_hi(*output_vec, 0);
          vec_fp32 temp_vec_lo = vec_extend_to_fp32_lo(*output_vec, 0);
#endif
          (*clip_round_hi_func)(&temp_vec_hi, &vec_clip_min, &vec_clip_max);
          (*clip_round_lo_func)(&temp_vec_lo, &vec_clip_min, &vec_clip_max);
          (*deq_func)(&temp_vec_hi, &temp_vec_lo, vec_scale, vec_offset);

#if defined(__MVS__)
          *output_vec++ =
              aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_vec_hi),
                                      *(vec_float32 *)((void *)&temp_vec_lo));
#else
          *output_vec++ = vec_round_from_fp32(temp_vec_hi, temp_vec_lo, 0);
#endif
          nbr_fields_converted += 8;
        }

        out_offset += out_bytes_all_w;
      }

      out_offset = out_w_offset + AIU_BYTES_PER_STICK;
    }

    // reset out_offset to the next n
    out_offset = out_offset_n + out_bytes_per_n;
  }
}

/// Computes the appropriate correction term and adjusts the matmul output. Then
/// clips the output between clip_min and clip_max.
///
/// \param[in] input_a The input ztensor, quantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] M The result of Sy / (Sa * Sb).
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
static void apply_correction_term(const zdnn_ztensor *input_a,
                                  const zdnn_ztensor *input_b, const float M,
                                  const int8_t clip_min, const int8_t clip_max,
                                  zdnn_ztensor *output, const bool dequantize,
                                  const bool disable_clipping) {
  uint64_t in_a_offset = 0; // moving position as input_a is processed, in BYTES
  uint64_t in_b_offset = 0; // moving position as input_b is processed, in BYTES
  uint64_t out_offset = 0;  // moving position as output is processed, in BYTES

  uint32_t remaining_fields;     // number of fields remaining
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  // loop invariant values
  const uint64_t in_a_bytes_all_w =
      CEIL(input_a->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t in_a_bytes_per_n =
      CEIL(input_a->transformed_desc->dim1, AIU_1BYTE_CELLS_PER_STICK) *
      in_a_bytes_all_w;

  const uint64_t in_b_bytes_all_w =
      CEIL(input_b->transformed_desc->dim2, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t in_b_bytes_per_n =
      CEIL(input_b->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      in_b_bytes_all_w;

  const uint64_t out_bytes_all_w =
      CEIL(output->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t out_bytes_per_n =
      CEIL(output->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      out_bytes_all_w;

  const float MZb = M * input_b->offset;
  vec_fp32 vec_MZa = vec_splats(M * input_a->offset);

  float term_a[input_a->transformed_desc->dim2];
  // cppcheck-suppress unassignedVariable
  float term_b[input_b->transformed_desc->dim1 + 7];

  vec_fp32 vec_clip_min = vec_splats((float)clip_min);
  vec_fp32 vec_clip_max = vec_splats((float)clip_max);
  vec_fp32 vec_scale = vec_splats((1.f / output->rec_scale));
  vec_fp32 vec_offset = vec_splats(output->offset);

  void (*clip_round_hi_func)(vec_fp32 *, vec_fp32 *, vec_fp32 *) =
      disable_clipping ? &skip_clip_and_round : &clip_and_round_hi;
  void (*clip_round_lo_func)(vec_fp32 *, vec_fp32 *, vec_fp32 *) =
      disable_clipping ? &skip_clip_and_round : &clip_and_round_lo;
  void (*deq_func)(vec_fp32 *, vec_fp32 *, vec_fp32, vec_fp32) =
      dequantize ? &apply_dequantization : &skip_dequantization;

  // Output dim4, which is equal to MAX(input_a dim4, input_b dim4)
  for (uint32_t e4x = 0; e4x < output->transformed_desc->dim4; e4x++) {

    // Computation of term_a from input_a and MZb. Only compute on first dim4
    // index if input_a needs to be broadcast
    if (e4x < input_a->transformed_desc->dim4) {
      // Used for pushing in_a_offset from n to n+1 (i.e., + in_c_bytes_per_n)
      const uint64_t in_a_offset_n = in_a_offset;

      for (uint32_t e2x = 0; e2x < input_a->transformed_desc->dim2; e2x++) {
        const uint64_t in_a_w_offset = in_a_offset;

        // Zero out summ_vec, which will hold the summation for C dim.
        vec_int summ_vec = vec_splats(0);

        for (uint32_t e1x = 0; e1x < input_a->transformed_desc->dim1;
             e1x += AIU_1BYTE_CELLS_PER_STICK) {
          vec_char *in_a_vec =
              (vec_char *)((void *)((uintptr_t)input_a->buffer + in_a_offset));

          remaining_fields = MIN(input_a->transformed_desc->dim1 - e1x,
                                 AIU_1BYTE_CELLS_PER_STICK);
          fields_to_convert = remaining_fields - (remaining_fields % 16);
          nbr_fields_converted = 0;

          while (nbr_fields_converted < fields_to_convert) {
            // Load high end of in_b_vec (first 8 elements) into temp_int16
            vec_short temp_int16 = vec_unpackh(*in_a_vec);
            // Perform sum operation on every 4 int8 elements into summ_vec
            summ_vec += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
            temp_int16 = vec_unpackl(*in_a_vec);
            // Perform sum operation on every 4 int8 elements into summ_vec
            summ_vec += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
            in_a_vec++;
            nbr_fields_converted += 16;
          }

          if (nbr_fields_converted < remaining_fields) {
            // Load remaining fields into temp_vec
            vec_char temp_vec =
                vec_load_len((signed char *)in_a_vec,
                             (remaining_fields - nbr_fields_converted) - 1);
            // Load high end of in_b_vec (first 8 elements) into temp_int16
            vec_short temp_int16 = vec_unpackh(temp_vec);
            // Perform sum operation on every 4 int8 elements into summ_vec
            summ_vec += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
            temp_int16 = vec_unpackl(temp_vec);
            // Perform sum operation on every 4 int8 elements into summ_vec
            summ_vec += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
          }

          in_a_offset += in_a_bytes_all_w;
        }

        term_a[e2x] =
            (float)(summ_vec[0] + summ_vec[1] + summ_vec[2] + summ_vec[3]) *
            MZb;

        in_a_offset = in_a_w_offset + AIU_BYTES_PER_STICK;
      }

      // reset in_a_offset to the next n
      in_a_offset = in_a_offset_n + in_a_bytes_per_n;
    }

    // Computation of term_b from input_b and MZa. Only compute on first dim4
    // index if input_b needs to be broadcast
    if (e4x < input_b->transformed_desc->dim4) {
      // Used for pushing in_b_offset from n to n+1 (i.e., + in_b_bytes_per_n)
      const uint64_t in_b_offset_n = in_b_offset;

      vec_fp32 *term_b_vec = (vec_fp32 *)term_b;

      for (uint32_t e1x = 0; e1x < input_b->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
        const uint64_t in_b_w_offset = in_b_offset;

        fields_to_convert = MIN(input_b->transformed_desc->dim1 - e1x,
                                AIU_2BYTE_CELLS_PER_STICK);
        nbr_fields_converted = 0;

        while (nbr_fields_converted < fields_to_convert) {
          // Zero out summ_vec, which will hold the summation for W dim.
          vec_int summ_vec_hi = vec_splats(0);
          vec_int summ_vec_lo = vec_splats(0);

          vec_char *in_b_vec =
              (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

          for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
               e2x++) {
            // Load high end of in_b_vec (first 8 elements) into temp_int16
            vec_short temp_int16 = vec_unpackh(*in_b_vec);
            // Perform sum operation on every 2 int16 elements into summ_vec_hi
            summ_vec_hi += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
            // Load low end of in_b_vec (final 8 elements) into temp_int16
            temp_int16 = vec_unpackl(*in_b_vec);
            // Perform sum operation on every 2 int16 elements into summ_vec_lo
            summ_vec_lo += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

            in_b_vec += 8;
          }

          if (input_b->transformed_desc->dim2 % 2) {
            // Load high end of in_b_vec (first 8 elements) into temp_int16
            vec_short temp_int16 = vec_unpackh(*in_b_vec);
            // Perform sum operation on every other int16 element into
            // summ_vec_hi
            summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2],
                                     temp_int16[4], temp_int16[6]};
            // Load low end of in_b_vec (final 8 elements) into temp_int16
            temp_int16 = vec_unpackl(*in_b_vec);
            // Perform sum operation on every other int16 element into
            // summ_vec_lo
            summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2],
                                     temp_int16[4], temp_int16[6]};
          }

          *term_b_vec++ = vec_float(summ_vec_hi) * vec_MZa;
          *term_b_vec++ = vec_float(summ_vec_lo) * vec_MZa;

          in_b_offset += 16;

          nbr_fields_converted += 8;
        }

        in_b_offset = in_b_w_offset + in_b_bytes_all_w;
      }

      // reset in_b_offset to the next n
      in_b_offset = in_b_offset_n + in_b_bytes_per_n;
    }

    // Used for pushing out_offset from n to n+1 (i.e., + out_bytes_per_n)
    const uint64_t out_offset_n = out_offset;

    // Computation of output[e2x][e1x] = term_a[e2x] + term_b[e1x]
    for (uint32_t e2x = 0; e2x < output->transformed_desc->dim2; e2x++) {
      const uint64_t out_w_offset = out_offset;

      vec_fp32 term_a_vec = vec_splats(term_a[e2x]);

      vec_fp32 *term_b_vec = (vec_fp32 *)term_b;

      for (uint32_t e1x = 0; e1x < output->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
#if defined(__MVS__)
        vec_int16 *output_vec =
            (vec_int16 *)((void *)((uintptr_t)output->buffer + out_offset));
#else
        vec_short *output_vec =
            (vec_short *)((void *)((uintptr_t)output->buffer + out_offset));
#endif

        fields_to_convert = MIN(output->transformed_desc->dim1 - e1x,
                                AIU_2BYTE_CELLS_PER_STICK);
        nbr_fields_converted = 0;

        while (nbr_fields_converted < fields_to_convert) {
#if defined(__MVS__)
          vec_fp32 temp_vec_hi, temp_vec_lo;
          aiu_vec_lengthen_to_fp32(*output_vec,
                                   (vec_float32 *)((void *)&temp_vec_hi),
                                   (vec_float32 *)((void *)&temp_vec_lo));
#else
          vec_fp32 temp_vec_hi = vec_extend_to_fp32_hi(*output_vec, 0);
          vec_fp32 temp_vec_lo = vec_extend_to_fp32_lo(*output_vec, 0);
#endif
          temp_vec_hi -= (*term_b_vec + term_a_vec);
          (*clip_round_hi_func)(&temp_vec_hi, &vec_clip_min, &vec_clip_max);
          term_b_vec++;
          temp_vec_lo -= (*term_b_vec + term_a_vec);
          (*clip_round_lo_func)(&temp_vec_lo, &vec_clip_min, &vec_clip_max);
          term_b_vec++;
          (*deq_func)(&temp_vec_hi, &temp_vec_lo, vec_scale, vec_offset);

#if defined(__MVS__)
          *output_vec++ =
              aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_vec_hi),
                                      *(vec_float32 *)((void *)&temp_vec_lo));
#else
          *output_vec++ = vec_round_from_fp32(temp_vec_hi, temp_vec_lo, 0);
#endif
          nbr_fields_converted += 8;
        }

        out_offset += out_bytes_all_w;
      }

      out_offset = out_w_offset + AIU_BYTES_PER_STICK;
    }

    // reset out_offset to the next n
    out_offset = out_offset_n + out_bytes_per_n;
  }
}

/// Computes the appropriate correction term and adjusts the matmul output. Then
/// clips the output between clip_min and clip_max.
///
/// \param[in] input_a The input ztensor, unquantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] M The result of Sy / (Sa * Sb).
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
static void apply_correction_term_on_the_fly(
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b, const float M,
    const int8_t clip_min, const int8_t clip_max, zdnn_ztensor *output,
    const bool dequantize, const bool disable_clipping) {
  uint64_t in_a_offset = 0; // moving position as input_a is processed, in BYTES
  uint64_t in_b_offset = 0; // moving position as input_b is processed, in BYTES
  uint64_t out_offset = 0;  // moving position as output is processed, in BYTES

  uint32_t remaining_fields;     // number of fields remaining
  uint32_t fields_to_convert;    // number of fields to actually convert
  uint32_t nbr_fields_converted; // number of fields converted

  // loop invariant values
  const uint64_t in_a_bytes_all_w =
      CEIL(input_a->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t in_a_bytes_per_n =
      CEIL(input_a->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      in_a_bytes_all_w;

  const uint64_t in_b_bytes_all_w =
      CEIL(input_b->transformed_desc->dim2, AIU_2BYTE_CELLS_PER_STICK) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t in_b_bytes_per_n =
      CEIL(input_b->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      in_b_bytes_all_w;

  const uint64_t out_bytes_all_w =
      CEIL(output->transformed_desc->dim2, AIU_STICKS_PER_PAGE) *
      AIU_PAGESIZE_IN_BYTES;

  const uint64_t out_bytes_per_n =
      CEIL(output->transformed_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
      out_bytes_all_w;

  const float scale = M * input_b->offset * input_a->rec_scale;

  vec_fp32 vec_MZa = vec_splats(M * input_a->offset);

  float term_a[input_a->transformed_desc->dim2];
  // cppcheck-suppress unassignedVariable
  float term_b[input_b->transformed_desc->dim1 + 7];

  vec_fp32 vec_clip_min = vec_splats((float)clip_min);
  vec_fp32 vec_clip_max = vec_splats((float)clip_max);
  vec_fp32 vec_scale = vec_splats((1.f / output->rec_scale));
  vec_fp32 vec_offset = vec_splats(output->offset);

  void (*clip_round_hi_func)(vec_fp32 *, vec_fp32 *, vec_fp32 *) =
      disable_clipping ? &skip_clip_and_round : &clip_and_round_hi;
  void (*clip_round_lo_func)(vec_fp32 *, vec_fp32 *, vec_fp32 *) =
      disable_clipping ? &skip_clip_and_round : &clip_and_round_lo;
  void (*deq_func)(vec_fp32 *, vec_fp32 *, vec_fp32, vec_fp32) =
      dequantize ? &apply_dequantization : &skip_dequantization;

  // Output dim4, which is equal to MAX(input_a dim4, input_b dim4)
  for (uint32_t e4x = 0; e4x < output->transformed_desc->dim4; e4x++) {

    // Computation of term_a from input_a and MZb. Only compute on first dim4
    // index if input_a needs to be broadcast
    if (e4x < input_a->transformed_desc->dim4) {
      // Used for pushing in_a_offset from n to n+1 (i.e., + in_c_bytes_per_n)
      const uint64_t in_a_offset_n = in_a_offset;

      for (uint32_t e2x = 0; e2x < input_a->transformed_desc->dim2; e2x++) {
        const uint64_t in_a_w_offset = in_a_offset;

        // Zero out temp_float, which will hold the summation for C dim.
        vec_fp32 summ_vec_a_hi = vec_splats(0.f);
        vec_fp32 summ_vec_a_lo = vec_splats(0.f);

        for (uint32_t e1x = 0; e1x < input_a->transformed_desc->dim1;
             e1x += AIU_2BYTE_CELLS_PER_STICK) {
#if defined(__MVS__)
          vec_int16 *in_a_vec =
              (vec_int16 *)((void *)((uintptr_t)input_a->buffer + in_a_offset));
#else
          vec_short *in_a_vec =
              (vec_short *)((void *)((uintptr_t)input_a->buffer + in_a_offset));
#endif

          remaining_fields = MIN(input_a->transformed_desc->dim1 - e1x,
                                 AIU_2BYTE_CELLS_PER_STICK);
          fields_to_convert = remaining_fields - (remaining_fields % 8);
          nbr_fields_converted = 0;

          while (nbr_fields_converted < fields_to_convert) {
#if defined(__MVS__)
            vec_float32 temp_float_hi, temp_float_lo;
            aiu_vec_lengthen_to_fp32(*in_a_vec, &temp_float_hi, &temp_float_lo);

            summ_vec_a_hi += *(vec_fp32 *)((void *)&temp_float_hi);
            summ_vec_a_lo += *(vec_fp32 *)((void *)&temp_float_lo);
#else
            summ_vec_a_hi += vec_extend_to_fp32_hi(*in_a_vec, 0);
            summ_vec_a_lo += vec_extend_to_fp32_lo(*in_a_vec, 0);
#endif

            in_a_vec++;
            nbr_fields_converted += 8;
          }

          if (nbr_fields_converted < remaining_fields) {
#if defined(__MVS__)
            // Load remaining fields_to_convert into temp_vec
            vec_int16 temp_vec =
                vec_load_len((uint16_t *)in_a_vec,
                             (remaining_fields - nbr_fields_converted) * 2 - 1);

            vec_float32 temp_float_hi, temp_float_lo;
            aiu_vec_lengthen_to_fp32(temp_vec, &temp_float_hi, &temp_float_lo);

            summ_vec_a_hi += *(vec_fp32 *)((void *)&temp_float_hi);
            summ_vec_a_lo += *(vec_fp32 *)((void *)&temp_float_lo);
#else
            // Load remaining fields into temp_vec
            vec_short temp_vec =
                vec_load_len((short *)in_a_vec,
                             (remaining_fields - nbr_fields_converted) * 2 - 1);

            summ_vec_a_hi += vec_extend_to_fp32_hi(temp_vec, 0);
            summ_vec_a_lo += vec_extend_to_fp32_lo(temp_vec, 0);
#endif
          }

          in_a_offset += in_a_bytes_all_w;
        }

        summ_vec_a_hi += summ_vec_a_lo;

        term_a[e2x] = (summ_vec_a_hi[0] + summ_vec_a_hi[1] + summ_vec_a_hi[2] +
                       summ_vec_a_hi[3]) *
                      scale;

        in_a_offset = in_a_w_offset + AIU_BYTES_PER_STICK;
      }

      // reset in_a_offset to the next n
      in_a_offset = in_a_offset_n + in_a_bytes_per_n;
    }

    // Computation of term_b from input_b and MZa. Only compute on first dim4
    // index if input_b needs to be broadcast
    if (e4x < input_b->transformed_desc->dim4) {
      // Used for pushing in_b_offset from n to n+1 (i.e., + in_b_bytes_per_n)
      const uint64_t in_b_offset_n = in_b_offset;

      vec_fp32 *term_b_vec = (vec_fp32 *)term_b;

      for (uint32_t e1x = 0; e1x < input_b->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
        const uint64_t in_b_w_offset = in_b_offset;

        fields_to_convert = MIN(input_b->transformed_desc->dim1 - e1x,
                                AIU_2BYTE_CELLS_PER_STICK);
        nbr_fields_converted = 0;

        while (nbr_fields_converted < fields_to_convert) {
          // Zero out summ_vec, which will hold the summation for W dim.
          vec_int summ_vec_hi = vec_splats(0);
          vec_int summ_vec_lo = vec_splats(0);

          vec_char *in_b_vec =
              (vec_char *)((void *)((uintptr_t)input_b->buffer + in_b_offset));

          for (uint32_t e2x = 0; e2x < input_b->transformed_desc->dim2 / 2;
               e2x++) {
            // Load high end of in_b_vec (first 8 elements) into temp_int16
            vec_short temp_int16 = vec_unpackh(*in_b_vec);
            // Perform sum operation on every 2 int16 elements into summ_vec_hi
            summ_vec_hi += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};
            // Load low end of in_b_vec (final 8 elements) into temp_int16
            temp_int16 = vec_unpackl(*in_b_vec);
            // Perform sum operation on every 2 int16 elements into summ_vec_lo
            summ_vec_lo += (vec_int){
                temp_int16[0] + temp_int16[1], temp_int16[2] + temp_int16[3],
                temp_int16[4] + temp_int16[5], temp_int16[6] + temp_int16[7]};

            in_b_vec += 8;
          }

          if (input_b->transformed_desc->dim2 % 2) {
            // Load high end of in_b_vec (first 8 elements) into temp_int16
            vec_short temp_int16 = vec_unpackh(*in_b_vec);
            // Perform sum operation on every other int16 element into
            // summ_vec_hi
            summ_vec_hi += (vec_int){temp_int16[0], temp_int16[2],
                                     temp_int16[4], temp_int16[6]};
            // Load low end of in_b_vec (final 8 elements) into temp_int16
            temp_int16 = vec_unpackl(*in_b_vec);
            // Perform sum operation on every other int16 element into
            // summ_vec_lo
            summ_vec_lo += (vec_int){temp_int16[0], temp_int16[2],
                                     temp_int16[4], temp_int16[6]};
          }

          *term_b_vec++ = vec_float(summ_vec_hi) * vec_MZa;
          *term_b_vec++ = vec_float(summ_vec_lo) * vec_MZa;

          in_b_offset += 16;

          nbr_fields_converted += 8;
        }

        in_b_offset = in_b_w_offset + in_b_bytes_all_w;
      }

      // reset in_b_offset to the next n
      in_b_offset = in_b_offset_n + in_b_bytes_per_n;
    }

    // Used for pushing out_offset from n to n+1 (i.e., + out_bytes_per_n)
    const uint64_t out_offset_n = out_offset;

    // Compute output[e2x][e1x] = term_a[e2x] + term_b[e1x]
    for (uint32_t e2x = 0; e2x < output->transformed_desc->dim2; e2x++) {
      const uint64_t out_w_offset = out_offset;

      vec_fp32 term_a_vec = vec_splats(term_a[e2x]);

      vec_fp32 *term_b_vec = (vec_fp32 *)term_b;

      for (uint32_t e1x = 0; e1x < output->transformed_desc->dim1;
           e1x += AIU_2BYTE_CELLS_PER_STICK) {
#if defined(__MVS__)
        vec_int16 *output_vec =
            (vec_int16 *)((void *)((uintptr_t)output->buffer + out_offset));
#else
        vec_short *output_vec =
            (vec_short *)((void *)((uintptr_t)output->buffer + out_offset));
#endif

        fields_to_convert = MIN(output->transformed_desc->dim1 - e1x,
                                AIU_2BYTE_CELLS_PER_STICK);
        nbr_fields_converted = 0;

        while (nbr_fields_converted < fields_to_convert) {
#if defined(__MVS__)
          vec_fp32 temp_vec_hi, temp_vec_lo;
          aiu_vec_lengthen_to_fp32(*output_vec,
                                   (vec_float32 *)((void *)&temp_vec_hi),
                                   (vec_float32 *)((void *)&temp_vec_lo));
#else
          vec_fp32 temp_vec_hi = vec_extend_to_fp32_hi(*output_vec, 0);
          vec_fp32 temp_vec_lo = vec_extend_to_fp32_lo(*output_vec, 0);
#endif
          temp_vec_hi -= (*term_b_vec + term_a_vec);
          (*clip_round_hi_func)(&temp_vec_hi, &vec_clip_min, &vec_clip_max);
          term_b_vec++;
          temp_vec_lo -= (*term_b_vec + term_a_vec);
          (*clip_round_lo_func)(&temp_vec_lo, &vec_clip_min, &vec_clip_max);
          term_b_vec++;
          (*deq_func)(&temp_vec_hi, &temp_vec_lo, vec_scale, vec_offset);

#if defined(__MVS__)
          *output_vec++ =
              aiu_vec_round_from_fp32(*(vec_float32 *)((void *)&temp_vec_hi),
                                      *(vec_float32 *)((void *)&temp_vec_lo));
#else
          *output_vec++ = vec_round_from_fp32(temp_vec_hi, temp_vec_lo, 0);
#endif
          nbr_fields_converted += 8;
        }

        out_offset += out_bytes_all_w;
      }

      out_offset = out_w_offset + AIU_BYTES_PER_STICK;
    }

    // reset out_offset to the next n
    out_offset = out_offset_n + out_bytes_per_n;
  }
}

/// Calls the NNPA operations that makeup quantized matmul. This method preforms
/// "pre and post" work. For "pre" it computes the appropriate bias, storing the
/// result in qc_tilde. It then calls quantized_matmul() to perform the matmul
/// op. For "post" it computes the appropriate correction term (if applicable)
/// and adjusts the matmul output. Method stops and returns on the first error
/// encountered or ZDNN_OK.
///
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, quantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The biases ztensor, quantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] qc_tilde The computed biases ztensor.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
static zdnn_status aiu_quantized_matmul_internal(
    const uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    zdnn_ztensor *qc_tilde, zdnn_ztensor *output, const bool dequantize,
    const bool disable_clipping) {

  const float Sa = input_a->rec_scale;
  const float Za = input_a->offset;

  const float Sb = input_b->rec_scale;
  const float Zb = input_b->offset;

  const float Sc = input_c->rec_scale;
  const float Zc = input_c->offset;

  const float Sy = output->rec_scale;
  const float Zy = output->offset;

  if (op_type == MATMUL_OP_ADDITION) {
    zdnn_status status;

    const float M = Sy / (Sa * Sb);
    const float scale = Sy / Sc;

    if (Zb == 0.f) {
      const float offset = Zy - scale * Zc;
      const float MZa = M * Za;

      compute_folded_bias(input_b, input_c, scale, offset, MZa, qc_tilde);

      status = quantized_matmul(function_code, input_a, input_b, qc_tilde,
                                op_type, output);

      if (status == ZDNN_OK) {
        apply_clipping(clip_min, clip_max, output, dequantize,
                       disable_clipping);
      }
    } else {
      const float N = (float)(input_a->transformed_desc->dim1);
      const float offset = Zy - scale * Zc + M * N * Za * Zb;

      compute_bias(input_c, scale, offset, qc_tilde);

      status = quantized_matmul(function_code, input_a, input_b, qc_tilde,
                                op_type, output);

      // Upon success, compute correction term and subtract it from output
      if (status == ZDNN_OK) {
        apply_correction_term(input_a, input_b, M, clip_min, clip_max, output,
                              dequantize, disable_clipping);
      }
    }

    return status;
  }

  const float scale = (Sa * Sb) / Sc;
  const float offset = scale * Zc;

  compute_comparison_bias(input_b, input_c, scale, offset, Za, qc_tilde);

  zdnn_matmul_ops modified_op = op_type;

  // When scale is negative, certain operations need to be flipped.
  if (scale < 0.f) {
    switch (op_type) {
    case MATMUL_OP_GREATER:
      modified_op = MATMUL_OP_LESSER;
      break;
    case MATMUL_OP_GREATER_EQUAL:
      modified_op = MATMUL_OP_LESSER_EQUAL;
      break;
    case MATMUL_OP_LESSER_EQUAL:
      modified_op = MATMUL_OP_GREATER_EQUAL;
      break;
    case MATMUL_OP_LESSER:
      modified_op = MATMUL_OP_GREATER;
      break;
    default:
      break;
    }
  }

  return quantized_matmul(function_code, input_a, input_b, qc_tilde,
                          modified_op, output);
}

/// Calls the NNPA operations that makeup quantized matmul. This method preforms
/// "pre and post" work. For "pre" it computes the appropriate bias, storing the
/// result in qc_tilde. It then calls quantized_matmul_on_the_fly() to perform
/// the matmul op. For "post" it computes the appropriate correction term (if
/// applicable) and adjusts the matmul output. Method stops and returns on the
/// first error encountered or ZDNN_OK.
///
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, unquantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The biases ztensor, quantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] qc_tilde The computed biases ztensor.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
static zdnn_status aiu_quantized_matmul_on_the_fly_internal(
    const uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    zdnn_ztensor *qc_tilde, zdnn_ztensor *output, const bool dequantize,
    const bool disable_clipping) {

  const float Sa = input_a->rec_scale;
  const float Za = input_a->offset;

  const float Sb = input_b->rec_scale;
  const float Zb = input_b->offset;

  const float Sc = input_c->rec_scale;
  const float Zc = input_c->offset;

  const float Sy = output->rec_scale;
  const float Zy = output->offset;

  if (op_type == MATMUL_OP_ADDITION) {
    zdnn_status status;

    const float M = Sy / (Sa * Sb);
    const float scale = Sy / Sc;
    const float offset = Zy - scale * Zc;

    if (Zb == 0.f) {
      const float MZa = M * Za;

      compute_folded_bias(input_b, input_c, scale, offset, MZa, qc_tilde);

      status =
          quantized_matmul_on_the_fly(function_code, input_a, input_b, qc_tilde,
                                      op_type, clip_min, clip_max, output);

      if (status == ZDNN_OK) {
        apply_clipping(clip_min, clip_max, output, dequantize,
                       disable_clipping);
      }
    } else {
      compute_bias(input_c, scale, offset, qc_tilde);

      status =
          quantized_matmul_on_the_fly(function_code, input_a, input_b, qc_tilde,
                                      op_type, clip_min, clip_max, output);

      // Upon success, compute correction term and subtract it from output
      if (status == ZDNN_OK) {
        apply_correction_term_on_the_fly(input_a, input_b, M, clip_min,
                                         clip_max, output, dequantize,
                                         disable_clipping);
      }
    }

    return status;
  }

  const float scale = (Sa * Sb) / Sc;
  const float offset = scale * Zc;

  compute_comparison_bias(input_b, input_c, scale, offset, Za, qc_tilde);

  zdnn_matmul_ops modified_op = op_type;

  // When scale is negative, certain operations need to be flipped.
  if (scale < 0.f) {
    switch (op_type) {
    case MATMUL_OP_GREATER:
      modified_op = MATMUL_OP_LESSER;
      break;
    case MATMUL_OP_GREATER_EQUAL:
      modified_op = MATMUL_OP_LESSER_EQUAL;
      break;
    case MATMUL_OP_LESSER_EQUAL:
      modified_op = MATMUL_OP_GREATER_EQUAL;
      break;
    case MATMUL_OP_LESSER:
      modified_op = MATMUL_OP_GREATER;
      break;
    default:
      break;
    }
  }

  return quantized_matmul_on_the_fly(function_code, input_a, input_b, qc_tilde,
                                     modified_op, clip_min, clip_max, output);
}

/// Calls the NNPA operations that makeup quantized matmul. This method preforms
/// "post" work. It first calls quantized_matmul() to perform the matmul op. For
/// "post" it computes the appropriate correction term (if applicable) and
/// adjusts the matmul output. Method stops and returns on the first error
/// encountered or ZDNN_OK.
///
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, quantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The computed biases ztensor, unquantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
static zdnn_status aiu_quantized_matmul_pre_computed_internal(
    const uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    zdnn_ztensor *output, const bool dequantize, const bool disable_clipping) {

  if (op_type == MATMUL_OP_ADDITION) {
    zdnn_status status = quantized_matmul(function_code, input_a, input_b,
                                          input_c, op_type, output);

    if (status == ZDNN_OK) {
      apply_clipping(clip_min, clip_max, output, dequantize, disable_clipping);
    }

    return status;
  }

  const float scale =
      (input_a->rec_scale * input_b->rec_scale) / input_c->rec_scale;

  zdnn_matmul_ops modified_op = op_type;

  // When scale is negative, certain operations need to be flipped.
  if (scale < 0.f) {
    switch (op_type) {
    case MATMUL_OP_GREATER:
      modified_op = MATMUL_OP_LESSER;
      break;
    case MATMUL_OP_GREATER_EQUAL:
      modified_op = MATMUL_OP_LESSER_EQUAL;
      break;
    case MATMUL_OP_LESSER_EQUAL:
      modified_op = MATMUL_OP_GREATER_EQUAL;
      break;
    case MATMUL_OP_LESSER:
      modified_op = MATMUL_OP_GREATER;
      break;
    default:
      break;
    }
  }

  return quantized_matmul(function_code, input_a, input_b, input_c, modified_op,
                          output);
}

/// Calls the NNPA operations that makeup quantized matmul. This method preforms
/// "post" work. It first calls quantized_matmul_on_the_fly() to perform the
/// matmul op. For "post" it computes the appropriate correction term (if
/// applicable) and adjusts the matmul output. Method stops and returns on the
/// first error encountered or ZDNN_OK.
///
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, unquantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The computed biases ztensor, unquantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value for input_a or NULL.
/// \param[in] clip_max The maximim quantized value for input_a or NULL.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[out] dequantize Whether to dequantize returned ztensor.
/// \param[out] disable_clipping Whether to disable clipping and rounding.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
static zdnn_status aiu_quantized_matmul_pre_computed_on_the_fly_internal(
    const uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    zdnn_matmul_ops op_type, const int8_t clip_min, const int8_t clip_max,
    zdnn_ztensor *output, const bool dequantize, const bool disable_clipping) {

  if (op_type == MATMUL_OP_ADDITION) {
    zdnn_status status =
        quantized_matmul_on_the_fly(function_code, input_a, input_b, input_c,
                                    op_type, clip_min, clip_max, output);

    if (status == ZDNN_OK) {
      apply_clipping(clip_min, clip_max, output, dequantize, disable_clipping);
    }

    return status;
  }

  const float scale =
      (input_a->rec_scale * input_b->rec_scale) / input_c->rec_scale;

  zdnn_matmul_ops modified_op = op_type;

  // When scale is negative, certain operations need to be flipped.
  if (scale < 0.f) {
    switch (op_type) {
    case MATMUL_OP_GREATER:
      modified_op = MATMUL_OP_LESSER;
      break;
    case MATMUL_OP_GREATER_EQUAL:
      modified_op = MATMUL_OP_LESSER_EQUAL;
      break;
    case MATMUL_OP_LESSER_EQUAL:
      modified_op = MATMUL_OP_GREATER_EQUAL;
      break;
    case MATMUL_OP_LESSER:
      modified_op = MATMUL_OP_GREATER;
      break;
    default:
      break;
    }
  }

  return quantized_matmul_on_the_fly(function_code, input_a, input_b, input_c,
                                     modified_op, clip_min, clip_max, output);
}
#endif

/// Calls the NNPA operations that makeup quantized matmul. It first allocates
/// the work_area (if necessary) and then calls either
/// aiu_quantized_matmul_internal() or
/// aiu_quantized_matmul_on_the_fly_internal(). After output is processed it
/// cleans up the work area (if necessary) and returns the final status. Method
/// stops and returns on the first error encountered or ZDNN_OK.
///
/// \param[in] op_parm_block_version Parmblock Version.
/// \param[in] function_code The matmul operation to be run.
/// \param[in] input_a The input ztensor, quantized or unquantized values.
/// \param[in] input_b The weights ztensor, quantized values.
/// \param[in] input_c The biases ztensor, quantized values.
/// \param[in] op_type The operation to perform against the matmul dot product.
/// \param[in] clip_min The minimum quantized value.
/// \param[in] clip_max The maximim quantized value.
/// \param[in] work_area Pointer to pre-allocated work area for our internal
///                      ztensors or NULL.
/// \param[out] output The returned ztensor from the zAIU.
/// \param[in] dequantize Whether to dequantize returned ztensor.
/// \param[in] disable_clipping Whether to disable clipping and rounding.
/// \param[in] pre_computed Whether bias is already pre-computed.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
zdnn_status
aiu_quantized_matmul(uint16_t op_parm_block_version,
                     const uint8_t function_code, const zdnn_ztensor *input_a,
                     const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
                     zdnn_matmul_ops op_type, const int8_t clip_min,
                     const int8_t clip_max, void *work_area,
                     zdnn_ztensor *output, const bool dequantize,
                     const bool disable_clipping, const bool pre_computed) {
#ifndef ZDNN_CONFIG_NO_NNPA
  if (!is_query_parmblock_installed(op_parm_block_version)) {
    return ZDNN_UNAVAILABLE_FUNCTION;
  }

  // Setup qc_tilde ztensor using same layout, format, and dims as input_c but
  // dlfloat16 type. Using values from input_c transformed_desc means
  // validation of qc_tilde applies to input_c.
  zdnn_tensor_desc qc_tilde_desc;
  zdnn_ztensor qc_tilde;
  // Work area is heap memory allocated for internal bias ztensor buffer
  bool alloced_work_area = false;

  // If not passed in a pointer to pre-allocated space for the work_area,
  // allocate it now and record that we need to free what we allocated.
  void *output_work_area = work_area;

  if (!pre_computed) {
    if (input_c->transformed_desc->type != ZDNN_BINARY_INT8) {
      // input_c is never sent to hardware, it is only used for computing
      // qc_tilde so there will only ever be a software error when input_c has
      // an invalid type.
      return ZDNN_STATUS(
          ZDNN_INVALID_TYPE,
          "input_c tensor type is invalid (found %s (%d), expects "
          "ZDNN_BINARY_INT8 (8))",
          get_data_type_str(input_c->transformed_desc->type),
          input_c->transformed_desc->type);
    }

    init_transformed_desc(
        input_c->transformed_desc->layout, ZDNN_DLFLOAT16,
        input_c->transformed_desc->format, &qc_tilde_desc,
        input_c->transformed_desc->dim4, input_c->transformed_desc->dim3,
        input_c->transformed_desc->dim2, input_c->transformed_desc->dim1);
    zdnn_init_ztensor(&qc_tilde_desc, &qc_tilde_desc, &qc_tilde);
    qc_tilde.buffer_size = zdnn_getsize_ztensor(&qc_tilde_desc);

    if (output_work_area == NULL) {
      if (!(output_work_area = malloc_aligned_4k(qc_tilde.buffer_size))) {
        return ZDNN_STATUS(ZDNN_ALLOCATION_FAILURE,
                           "Unable to allocate %" PRIu64
                           " bytes for output_work_area.",
                           qc_tilde.buffer_size);
      }
      // Used so we only free the alloced_work_area if we allocated it.
      alloced_work_area = true;
    }

    qc_tilde.buffer = output_work_area;
  }

  zdnn_status status;
  if (input_a->transformed_desc->type == ZDNN_BINARY_INT8) {
    if (!pre_computed) {
      status = aiu_quantized_matmul_internal(
          function_code, input_a, input_b, input_c, op_type, clip_min, clip_max,
          &qc_tilde, output, dequantize, disable_clipping);
    } else
      status = aiu_quantized_matmul_pre_computed_internal(
          function_code, input_a, input_b, input_c, op_type, clip_min, clip_max,
          output, dequantize, disable_clipping);
  } else if (!pre_computed) {
    status = aiu_quantized_matmul_on_the_fly_internal(
        function_code, input_a, input_b, input_c, op_type, clip_min, clip_max,
        &qc_tilde, output, dequantize, disable_clipping);
  } else {
    status = aiu_quantized_matmul_pre_computed_on_the_fly_internal(
        function_code, input_a, input_b, input_c, op_type, clip_min, clip_max,
        output, dequantize, disable_clipping);
  }

  // Frees the entire output_work_area for all outputs (if required)
  if (alloced_work_area) {
    free_aligned_4k(output_work_area);
  }

  // Upon success, indicate that the output tensor has a stickified
  // (4DFeature) tensor and return status.
  if (status == ZDNN_OK) {
    output->is_transformed = true;
  }

  return status;
#else
  return ZDNN_STATUS_OK;
#endif
}