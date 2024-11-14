// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2023
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

/**
 * Helper macro that given the indices and sizes of a multidimensional array
 * returns equivalent index to a flat representation of the same array. The
 * result is cast to uint64_t as that's the largest number of total elements a
 * ztensor supports as opposed to the single dimension maximum of unint32_t
 *
 * Note: Default usage is for 3D arrays. For 2D arrays, use 0 for the
 * undefined dimension's index and 1 its size.
 */
#define GET_FLAT_IDX(stack, row, col, row_size, col_size)                      \
  (uint64_t)(stack) * (row_size) * (col_size) + (row) * (col_size) + (col)

/**
 * Helper macro that given a real value, a scale, and an offset, will produce
 * a quantized value clipped between the limits for a signed eight-bit integer.
 */
#define QUANTIZE(r, scale, offset)                                             \
  (MIN(MAX(roundf(r / scale + offset), -128.f), 127.f))

/**
 * Helper macro that given a quantized value, a scale, and an offset, will
 * produce a real value clipped.
 */
#define DEQUANTIZE(q, scale, offset) ((q - offset) * scale)

/**
 * Helper macro that qunatizes and then dequantizes a real value using a scale
 * and an offset.
 */
#define CLEANSE_QUANTIZED(r, scale, offset)                                    \
  (DEQUANTIZE(QUANTIZE(r, scale, offset), scale, offset))

zdnn_ztensor *alloc_quantized_ztensor_with_values(
    uint32_t *shape, zdnn_data_layouts pre_tfrmd_layout, zdnn_data_types type,
    zdnn_quantized_transform_types transform_type, const float *values_data,
    const float scale, const float offset);

void assert_quantized_ztensor_values(zdnn_ztensor *ztensor,
                                     bool repeat_first_expected_value,
                                     const float *expected_vals);

void assert_dequantized_ztensor_values(zdnn_ztensor *ztensor,
                                       bool repeat_first_expected_value,
                                       const float *expected_vals);

void assert_quantized_ztensor_compare_values(zdnn_ztensor *ztensor,
                                             bool repeat_first_expected_value,
                                             const float *expected_vals);
