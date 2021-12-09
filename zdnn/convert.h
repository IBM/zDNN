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

#ifndef ZDNN_CONVERT_H_
#define ZDNN_CONVERT_H_

#include <inttypes.h>
#include <stddef.h>

/* Within convert.h/convert.c, we must treat the floating point values
   to be converted as simple INT values for the conversions to be
   successful, so we typedef those here.    */
typedef uint16_t float_bit16; /* Generic type for any of the 16 bit items
   that are really considered "short floats":  dlfloat, fp16 or bfloat */

typedef uint32_t float_bit32; /* Generic type for the 32 bit items that
   are really considered "floats":  float or fp32 */

// used to get around "breaking strict-aliasing rules" so that we can
// manipulate the bits within a float
typedef union uint32_float_u {
  uint32_t u;
  float f;
} uint32_float_u;

uint64_t fp16_to_dlf16(uint16_t *input_fp16_data, uint16_t *output_dflt16_data,
                       uint64_t nbr_fields_to_convert);
uint64_t fp32_to_dlf16(float *input_data, uint16_t *output_data,
                       uint64_t nbr_fields_to_convert);
uint64_t bfloat_to_dlf16(uint16_t *input_data, uint16_t *output_data,
                         uint64_t nbr_fields_to_convert);

uint64_t dlf16_to_fp16(uint16_t *input_dflt16_data, uint16_t *output_fp16_data,
                       uint64_t nbr_fields_to_convert);
uint64_t dlf16_to_fp32(uint16_t *input_data, float *output_data,
                       uint64_t nbr_fields_to_convert);
uint64_t dlf16_to_bfloat(uint16_t *input_data, uint16_t *output_data,
                         uint64_t nbr_fields_to_convert);

uint64_t fp16_to_dlf16_in_stride(uint16_t *fp16_data, uint16_t *dflt16_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride);
uint64_t fp32_to_dlf16_in_stride(float *fp32_data, uint16_t *dflt16_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride);
uint64_t bfloat_to_dlf16_in_stride(uint16_t *bflt_data, uint16_t *dflt16_data,
                                   uint64_t nbr_fields_to_convert,
                                   uint32_t input_stride);

uint64_t dlf16_to_fp16_in_stride(uint16_t *dflt16_data, uint16_t *fp16_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride);
uint64_t dlf16_to_fp32_in_stride(uint16_t *dflt16_data, float *fp32_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride);
uint64_t dlf16_to_bfloat_in_stride(uint16_t *dflt16_data, uint16_t *bflt_data,
                                   uint64_t nbr_fields_to_convert,
                                   uint32_t input_stride);

/**********************************************************************
 *  cnvt_1 functions - These functions invoke the aiu_vec functions to
 *  convert one value.  Highly inefficient.
 **********************************************************************/
/*  cnvt_1_fp32_to_dlf */
uint16_t cnvt_1_fp32_to_dlf16(float a);

/*  cnvt_1_dlf16_to_fp32 */
float cnvt_1_dlf16_to_fp32(uint16_t a);

/*  cnvt_1_fp16_to_dlf */
uint16_t cnvt_1_fp16_to_dlf16(uint16_t a);

/*  cnvt_1_dlf16_to_fp16 */
uint16_t cnvt_1_dlf16_to_fp16(uint16_t a);

/*  cnvt_1_bfloat_to_dlf */
uint16_t cnvt_1_bfloat_to_dlf16(uint16_t a);

/*  cnvt_1_dlf16_to_bfloat */
uint16_t cnvt_1_dlf16_to_bfloat(uint16_t a);

float cnvt_1_bfloat_to_fp32(uint16_t a);

float cnvt_1_fp16_to_fp32(uint16_t a);

uint16_t cnvt_1_fp32_to_bfloat(float a);

uint16_t cnvt_1_fp32_to_fp16(float a);

// End of cnvt_1 functions
#endif /* ZDNN_CONVERT_H_ */
