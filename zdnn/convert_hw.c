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

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"

/*
 * Implementation note:  This routine receives various floating point
 * data types. But for the purposes of type conversion, we treat
 * these as integers.  C likes to help convert floats to integers,
 * but we need to control all aspects of conversion to ensure
 * proper results for the AIU.  Hence, routines that receive
 * "float"-types will immediately cast to one of the float_bitxx
 * types and use those from then on.
 */

#define STICKCVT_MAX_ENTRIES_TO_CONVERT 8
/* Number of entries to be converted at a time. Conversion to/from
   FP32 to DLFLOAT require 2 vector regs to contain the 8 values,
   all others use 1 VR */

vec_char8 selection_vector = {0,  1,  4,  5,  8,  9,  12, 13,
                              16, 17, 20, 21, 24, 25, 28, 29};
static vec_int16 zero_vector16 = {0, 0, 0, 0, 0, 0, 0, 0};

/***********************************************************************
 * aiu_vec_round_from_fp32 routines
 *
 * Converts 2 vectors (4 elements each) of 32-bit floating point
 * numbers to 1 vector of 16-bit DLFLOAT numbers (8 numbers total)
 *
 * Input: 2 vectors of 4 FP32 data elements to convert
 * Output: vector of 8 DLFloat16 floats
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static vec_int16 inline aiu_vec_round_from_fp32_inline(vec_float32 a,
                                                       vec_float32 b) {
  vec_int16 out;

#ifndef ZDNN_CONFIG_NO_NNPA
#if defined(__MVS__)
  /*
       Invoke the VCRNF
                  "*     VCRNF VReg0,VRegL,VRegR,mask2,0        \n\t"
       Note that registers are hardcoded (vs using %0 notation) to ensure
       that the hardcoded instruction (E60120020075) has expected regs free
    */
  // clang-format off
  __asm volatile("      VL    1,%[in_vector_left]            \n\t"
                 "      VL    2,%[in_vector_right]           \n\t"
                 "      DC    XL6'E60120020075'              \n\t"
                 "      DS    0H                             \n\t"
                 "      VST   0,%[out_vector]                \n\t"
                 : /* Outputs - out_vector */
                 [out_vector] "=m"(out) //(out_vector)
                 :                        /* Inputs                        */
                 [mask2] "i"(2),        /* 2 = Internal NNPA format (DLF)  */
                 [mask0] "i"(0),        /* 0 = FP32                        */
                 [in_vector_left] "m"(a), /* data */
                 [in_vector_right] "m"(b) /* data */
                 :  /* "%v0", "%v1", "%v2"   Clobbered */
  );
// clang-format on
#else
  // clang-format off
  __asm volatile(".insn vrr,0xe60000000075,%[out],%[in_hi],%[in_lo],0,2,0"
                : [ out ] "=v"(out)
                : [ in_hi ] "v"(a), [ in_lo ] "v"(b));
// clang-format on
#endif
#else
  // cast our 32 bit vectors as bytes, then select via vec_perm which
  // bytes to return. (Will be first 2 bytes of every float32)
  out = (vec_int16)vec_perm((vec_char8)a, (vec_char8)b, selection_vector);
#endif // ZDNN_CONFIG_NO_NNPA

  return out;
} // End aiu_vec_round_from_fp32

//  Common version of non-inlined aiu_vec_round_from_fp32
vec_int16 aiu_vec_round_from_fp32(vec_float32 a, vec_float32 b) {
  return aiu_vec_round_from_fp32_inline(a, b);
}

/***********************************************************************
 * aiu_vec_convert_from_fp16
 *
 * Converts 1 vector (8 elements) of 16-bit floating point
 * numbers to 1 vector of 16-bit DLFLOAT numbers (8 numbers total)
 *
 * Input: 1 vector of 8 FP16 data elements to convert
 * Output: vector of 8 DLFloat16 floats
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static vec_int16 inline aiu_vec_convert_from_fp16_inline(vec_int16 a) {
  vec_int16 out;

#ifndef ZDNN_CONFIG_NO_NNPA
#if defined(__MVS__)
  /*
      Invoke the VCNF
                 "*     VCNF VReg0,VReg1,,mask1,0        \n\t"
      Note that registers are hardcoded (vs using %0 notation) to ensure
      that the hardcoded instruction (E60100010055) has expected regs free
   */
  // clang-format off
  __asm volatile("      VL    1,%[in_vector]                 \n\t"
                 "      DC    XL6'E60100010055'              \n\t"
                 "      DS    0H                             \n\t"
                 "      VST   0,%[out_vector]                \n\t"
                 : /* Outputs - out_vector */
                 [out_vector] "=m"(out) //(out_vector)
                 :                      /* Inputs                        */
                 [mask1] "i"(1),        /* 1 = BFP tiny format (FP16)    */
                 [mask0] "i"(0),        /* 0 = NNP format                */
                 [in_vector] "m"(a)     /* data */
                 :  /* "%v0", "%v1"   Clobbered */
                 // clang-format on
  );
#else
  // clang-format off
    __asm volatile(".insn vrr,0xe60000000055,%[out],%[in_vec],0,0,1,0"
                : [ out ] "=v"(out)
                : [ in_vec ] "v"(a));
// clang-format on
#endif
#else
  // scaffolding: just copy the input 16-bit elements as is to output
  memcpy(&out, &a, sizeof(vec_int16));
#endif // ZDNN_CONFIG_NO_NNPA

  return out;
} // End aiu_vec_convert_from_fp16

//  Common wrapper version of non-inlined aiu_vec_convert_from_fp16
vec_int16 aiu_vec_convert_from_fp16(vec_int16 a) {
  return aiu_vec_convert_from_fp16_inline(a);
}

/***********************************************************************
 * aiu_vec_lengthen_to_fp32
 *
 * Converts 1 vector of 16-bit DLFLOAT numbers (8 numbers total) to
 * 2 vectors (4 elements each) of 32-bit floating point
 *
 * Input: 1 vector (input) of 8 DLFloat16 floats to convert
 *        2 vectors (output) of 4 FP32 data elements
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static void inline aiu_vec_lengthen_to_fp32_inline(vec_int16 a,
                                                   vec_float32 *out1,
                                                   vec_float32 *out2) {

#ifndef ZDNN_CONFIG_NO_NNPA
  vec_float32 work_float_1;
  vec_float32 work_float_2;

#if defined(__MVS__)
  /*
   *  Invoke the VCLFNx
   *  "*     VCLFN(H/L) VReg0,VReg2,mask0,mask2      \n\t"
   */
  // clang-format off
    __asm volatile("      VL   2,%[in_vector]            \n\t" // load VR with 8 DLFs
                   "      DC   XL6'E60200002056'         \n\t" //VCLFNH to VR0
                   "      DC   XL6'E6120000205E'         \n\t" //VCLFNL to VR1
                   "      DS   0H                        \n\t"
                   "      VST  0,%[out_vector_left]      \n\t" //store 1-4 FP32s to output
                   "      VST  1,%[out_vector_right]     \n\t" //store 5-8 FP32s to output
                   : /* Outputs - out_vector */
                   [ out_vector_left ] "=m"(work_float_1),  //(out_vector)
                   [ out_vector_right ] "=m"(work_float_2)  //(out_vector)
                   :                 /* Inputs                        */
                   [ in_vector ] "m" (a),        /* data */
                   [ mask2 ] "i"(2), /* 2 = Internal NNPA format (DLF)  */
                   [ mask0 ] "i"(0)  /* 0 = FP32                        */
                 :  /* "%v0", "%v1", "%v2"   Clobbered */
    );
  // clang-format on
  *out1 = work_float_1;
  *out2 = work_float_2;
#else
  // clang-format off
  __asm volatile(".insn vrr,0xe60000000056,%[out1],%[in_vec],0,2,0,0    \n\t"
                ".insn vrr,0xe6000000005E,%[out2],%[in_vec],0,2,0,0     \n\t"
                : [ out1 ] "=&v"(work_float_1), [ out2 ] "=v"(work_float_2)
                : [ in_vec ] "v"(a));
  // clang-format on

  *out1 = work_float_1;
  *out2 = work_float_2;
#endif
#else
  *out1 = (vec_float32)vec_mergeh(a, zero_vector16);
  *out2 = (vec_float32)vec_mergel(a, zero_vector16);
#endif // ZDNN_CONFIG_NO_NNPA

  return;
} // End aiu_vec_lengthen_to_fp32

//  Common wrapper version of non-inlined aiu_vec_lengthen_to_fp32
void aiu_vec_lengthen_to_fp32(vec_int16 a, vec_float32 *out1,
                              vec_float32 *out2) {
  aiu_vec_lengthen_to_fp32_inline(a, out1, out2);
}

/***********************************************************************
 * aiu_vec_convert_to_fp16
 *
 * Converts 1 vector (8 elements) of 16-bit DLFloat numbers
 * to 1 vector of 16-bit FP16 numbers (8 numbers total)
 *
 * Input: 1 vector of 8 DLFloat data elements to convert
 * Output: 1 vector of 8 FP16 elements
 *
 * Note:  There is also a non-inlined wrapper function as well.
 **********************************************************************/

static vec_int16 inline aiu_vec_convert_to_fp16_inline(vec_int16 a) {
  vec_int16 work_short_1;

#ifndef ZDNN_CONFIG_NO_NNPA
#if defined(__MVS__)
  /*
   *  Invoke the VCFN
   *  "*     VCFN VReg0,VReg2,mask0,mask1      \n\t"
   */
  // clang-format off
    __asm volatile("      VL   2,%[in_vector]            \n\t" // load VR with 8 DLFs
                   "      DC   XL6'E6020000105D'         \n\t" //VCFN to VR0
                   "      DS   0H                        \n\t"
                   "      VST  0,%[out_vector]           \n\t" //store 8 FP16s to output
                   : /* Outputs - out_vector */
                   [ out_vector ] "=m"(work_short_1)
                   :                 /* Inputs                        */
                   [ in_vector ] "m" (a),        /* data */
                   [ mask1 ] "i"(1), /* 1 = FP16  */
                   [ mask0 ] "i"(0)  /* 0 = Internal NNPA format (DLF) */
                 :  /* "%v0", "%v2"   Clobbered */
    );
  // clang-format on
#else
  // clang-format off
  __asm volatile(".insn vrr,0xe6000000005D,%[out_vec],%[in_vec],0,1,0,0  \n\t"
                : [out_vec] "=v"(work_short_1)
                : [in_vec] "v"(a));
  // clang-format on
#endif
#else
  // scaffolding: just copy the input 16-bit elements as is to output
  memcpy(&work_short_1, &a, sizeof(vec_int16));
#endif // #ifndef ZDNN_CONFIG_NO_NNPA
  return work_short_1;
}

//  Common wrapper version of non-inlined aiu_vec_convert_to_fp16
vec_int16 aiu_vec_convert_to_fp16(vec_int16 a) {
  return aiu_vec_convert_to_fp16_inline(a);
}
// End of ASM functions

/***********************************************************************
 *  cnvt_1 functions - These functions invoke the aiu_vec functions to
 *  convert one value.  Highly inefficient.
 **********************************************************************/
/*  cnvt_1_fp32_to_dlf16 */
uint16_t cnvt_1_fp32_to_dlf16(float a) {

  vec_int16 aiu_op_output_dfloat; // vector output from aiu_vec_round...
  /* Copy value to work area, use AIU op routine to convert value from fp32
     to dlfloat in pseudo vector (array), then copy the 1 converted entry
     into the expected data area */

  uint32_float_u tempfp32array[8] = {0};
  memcpy(tempfp32array, &a,
         sizeof(float)); /* used as input to aiu_vec_round conversion */

  aiu_op_output_dfloat = aiu_vec_round_from_fp32(
      *((vec_float32 *)&tempfp32array[0]),
      *((vec_float32 *)&tempfp32array[4])); /* Convert from fp32 to
                                               dlfloat with rounding */
  return (uint16_t)aiu_op_output_dfloat[0]; // return first value from vector
}

/*  cnvt_1_dlf16_to_fp32 */
float cnvt_1_dlf16_to_fp32(uint16_t a) {

  vec_float32 aiu_op_output_fp32[2]; // vector output from aiu_vec_lengthen

  /* Copy value to work area, use AIU op routine to convert value from
     dlfloat to fp32 in pseudo vector (array), then copy the 1 converted
     entry into the expected data area */
  float_bit16 tempshortarray[8] = {a}; // used as input to aiu_vec_lengthen...
                                       // conversion
  aiu_vec_lengthen_to_fp32(*((vec_int16 *)&tempshortarray[0]),
                           aiu_op_output_fp32,
                           aiu_op_output_fp32 + 1); /* Convert from dlfloat to
                                                       fp32 with lengthening */
  return (*(uint32_float_u *)aiu_op_output_fp32)
      .f; /* return first value from vector output */
}

/*  cnvt_1_fp16_to_dlf */

uint16_t cnvt_1_fp16_to_dlf16(uint16_t a) {
  vec_int16 aiu_op_output;

  /* Copy value to work area, use AIU op routine to convert value from fp16
     to dlfloat in pseudo vector (array), then copy the 1 converted entry
     into the expected data area */
  float_bit16 tempFP16array[8] = {a}; /* used as input to
                                      aiu_vec_convert... conversion */
  aiu_op_output = aiu_vec_convert_from_fp16(
      *(vec_int16 *)(tempFP16array)); // Convert from fp16 to dlfloat
  return (uint16_t)aiu_op_output[0];  // return first value from vector
}

/*  cnvt_1_dlf16_to_fp16 */

uint16_t cnvt_1_dlf16_to_fp16(uint16_t a) {
  vec_int16 aiu_op_output;

  /* Copy value to work area, use AIU op routine to convert value from dlfloat
     to fp16 in pseudo vector (array), then copy the 1 converted entry
     into the expected data area */
  float_bit16 tempFP16array[8] = {a}; /* input to aiu_vec_lengthen
                              conversion, with input as first (only) entry */
  aiu_op_output = aiu_vec_convert_to_fp16(
      *(vec_int16 *)(&tempFP16array));  // Convert from dlfloat to fp16
  return (float_bit16)aiu_op_output[0]; // return value from vector
}

uint16_t cnvt_1_bfloat_to_dlf16(uint16_t a) {
  /* Copy value to work area adding decimal places to make it into a FP32,
         use AIU op routine to convert value from FP32 to dlfloat  */
  uint32_float_u temp_pseudo_float;

  // copy bfloat value into left side of a float, implementing a pseudo version
  // of the vector merge op
  temp_pseudo_float.u = ((float_bit32)a << 16);

  return cnvt_1_fp32_to_dlf16(temp_pseudo_float.f); /* Convert value (now fp32)
                      to dlfloat with rounding */
}

uint16_t cnvt_1_dlf16_to_bfloat(uint16_t a) {
  uint32_float_u temp_pseudo_float;
  /* Convert (and lengthen) the dlfloat back to fp32 */
  temp_pseudo_float.f = cnvt_1_dlf16_to_fp32(a);

  // Return the left 2 bytes of the returned float as our bfloat
  return (temp_pseudo_float.u >> 16);
}

// convert 1 BFLOAT element to FP32 (C float)
float cnvt_1_bfloat_to_fp32(uint16_t a) {
  // simply appends 16 0-bits as mantissa
  uint32_t u = (uint32_t)a << 16;
  return (*(uint32_float_u *)&u).f;
}

/***********************************************************************
 * There's no direct hardware support for the following conversions,
 * therefore do it via a x -> DLFLOAT16 -> y chain.  Precision loss
 * may occur
 ***********************************************************************/

// convert 1 FP16 element to FP32 (C float)
float cnvt_1_fp16_to_fp32(uint16_t a) {
  uint32_float_u x = {cnvt_1_dlf16_to_fp32(cnvt_1_fp16_to_dlf16(a))};
  return x.f;
}

// convert 1 FP32 element to BFLOAT
uint16_t cnvt_1_fp32_to_bfloat(float a) {
  uint32_float_u x = {cnvt_1_dlf16_to_bfloat(cnvt_1_fp32_to_dlf16(a))};
  return x.u;
}

// convert 1 FP32 element to FP16
uint16_t cnvt_1_fp32_to_fp16(float a) {
  uint32_float_u x = {cnvt_1_dlf16_to_fp16(cnvt_1_fp32_to_dlf16(a))};
  return x.u;
}

// End of cnvt_1 functions

/***********************************************************************
 * fp16_to_dlf16
 *
 * Converts 16-bit floating point elements (BFP tiny)
 * to 16-bit DLFLOAT stick elements
 *
 * Input: Address of N consecutive fp16 data elements to convert
 *        Address to store N converted DLFLOAT16 data elements
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 **********************************************************************/
uint64_t fp16_to_dlf16(uint16_t *input_fp16_data, uint16_t *output_dflt16_data,
                       uint64_t nbr_fields_to_convert) {

  // Set vector pointers from input/output pointers passed in.
  // Note: adding 1 to a vector pointer will move it ahead 16 bytes

  vec_int16 *cur_input_data =
      (vec_int16 *)input_fp16_data; // Point to input vector data
  vec_int16 *cur_output_data =
      (vec_int16 *)output_dflt16_data; // Point to output vector data

  vec_int16 in_vector;  // Define a vector to load eight of the input data
                        // fields into. A vector can fit 8 int16 fields
  vec_int16 out_vector; // Define a output vector for PACK operation.
                        // Vector can fit 8 int16 fields

  /*If there's 8 or more to convert, convert groups of 8 FP16s to 8 DL16s */
  for (int i = 0; i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       ++i) {
    *(cur_output_data) = aiu_vec_convert_from_fp16_inline(
        *(vec_int16 *)(cur_input_data)); // Convert from fp16
    cur_input_data++;                    /* bump ptr to start of next vector (8
                                              uint16s = 16 bytes = 1 vector) */
    cur_output_data++; /* bump ptr to start of next output vector (8 shorts =
        1 vector) */
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left to
                                       // convert

  if (!curnbr_fields_to_convert)  // If none,
    return nbr_fields_to_convert; // Return, indicating all converted

  // If there's still some to convert, it will be 7 or less, and we should
  // tread carefully to avoid touching user data beyond what they said they
  // gave us. Simply load by length the proper amount of data, convert the
  // whole vector, then store with proper length.
  in_vector =
      (vec_int16)vec_load_len((const uint16_t *)cur_input_data,
                              curnbr_fields_to_convert * sizeof(uint16_t) - 1);

  // Invoke the VCNF function
  out_vector = aiu_vec_convert_from_fp16_inline(in_vector);

  // Store results from vector to caller's storage
  vec_store_len(out_vector, (uint16_t *)cur_output_data,
                curnbr_fields_to_convert * 2 - 1);

  return nbr_fields_to_convert;
}
/***********************************************************************
 * dlf16_to_fp16
 *
 * Converts 16-bit DLFloat floating point elements (NNP format)
 * to 16-bit floating point (BFP tiny format)
 *
 * Input: Address of N consecutive DLFloat16 data elements to convert
 *        Address to store N converted FP16 data elements
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 **********************************************************************/
uint64_t dlf16_to_fp16(uint16_t *input_dflt16_data, uint16_t *output_fp16_data,
                       uint64_t nbr_fields_to_convert) {

  // Set vector pointers from input/output pointers passed in.
  // Note: adding 1 to a vector pointer will move it ahead 16 bytes

  vec_int16 *cur_input_data =
      (vec_int16 *)input_dflt16_data; // Point to input vector data
  vec_int16 *cur_output_data =
      (vec_int16 *)output_fp16_data; // Point to output vector data

  vec_int16 in_vector;  // Define a vector to load eight of the input data
                        // fields into. A vector can fit 8 int16 fields
  vec_int16 out_vector; // Define a output vector for PACK operation.
                        // Vector can fit 8 int16 fields

  /* If there's 8 or more to convert, convert groups of 8 FP16s to 8 DL16s */
  for (int i = 0; i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       ++i) {
    *(cur_output_data) = aiu_vec_convert_to_fp16_inline(
        *(vec_int16 *)(cur_input_data)); // Convert from dlfloat to fp16
    cur_input_data++;                    /* bump ptr to start of next vector (8
                                              uint16s = 16 bytes = 1 vector) */
    cur_output_data++; /* bump ptr to start of next output vector (8 shorts =
1 vector) */
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left to
                                       // convert

  if (!curnbr_fields_to_convert)  // If none,
    return nbr_fields_to_convert; // Return, indicating all converted

  // If there's still some to convert, it will be 7 or less, and we should
  // tread carefully to avoid touching user data beyond what they said they
  // gave us. Simply load by length the proper amount of data, convert the
  // whole vector, then store with proper length.
  in_vector =
      (vec_int16)vec_load_len((const uint16_t *)cur_input_data,
                              curnbr_fields_to_convert * sizeof(uint16_t) - 1);

  // Invoke the VCNF function
  out_vector = aiu_vec_convert_to_fp16_inline(in_vector);

  // Store results from vector to caller's storage
  vec_store_len(out_vector, (uint16_t *)cur_output_data,
                curnbr_fields_to_convert * 2 - 1);

  return nbr_fields_to_convert;
}
/***********************************************************************
 * fp32_to_dlf16
 *
 * Converts 32-bit Floating Point elements to 16 bit DLFLOAT stick elements
 *
 * Input: Address of N FP32 data elements to convert
 *        Address to store N converted DLFLOAT16 data elements
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/
uint64_t fp32_to_dlf16(float *input_data, uint16_t *output_data,
                       uint64_t nbr_fields_to_convert) {

  // Set vector pointers from input/output pointers passed in.
  // Note: adding 1 to a vector pointer will move it ahead 16 bytes

  vec_float32 *cur_input_data =
      (vec_float32 *)input_data; // Point to input vector data
  vec_int16 *cur_output_data =
      (vec_int16 *)output_data; // Point to output vector data

  vec_float32
      in_vector_left; // Define a vector to load four of the input data
                      // fields into. This will be the Left half of the
                      // concatenated vector. Vector can fit 4 int32 fields
  vec_float32
      in_vector_right;  // Define a vector to load four more input data fields
                        // into. This will be the Right half of the
                        // concatenated vector. Vector can fit 4 int32 fields
  vec_int16 out_vector; // Define a output vector for PACK operation.
                        // Vector can fit 8 int16 fields

  /* If there's 8 or more to convert, convert groups of 8 FP32s to 8 DL16s */
  for (int i = 0; i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       ++i) {
    *(cur_output_data) = aiu_vec_round_from_fp32_inline(
        *(vec_float32 *)(cur_input_data),
        *(vec_float32 *)(cur_input_data +
                         1)); // Convert from fp32 with rounding
    cur_input_data += 2; /* bump ptr to start of next concatenated vector (8
                       uint32s = 32 bytes = 2 vectors) */
    cur_output_data++;   /* bump ptr to start of next output vector (8 shorts =
                       1 vector) */
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left to
                                       // convert

  if (!curnbr_fields_to_convert)  // If none,
    return nbr_fields_to_convert; // Return, indicating all converted

  // If there's still some to convert, it will be 7 or less, and we should
  // tread carefully to avoid touching user data beyond what they said they
  // gave us.

  // Handle the 1-3 and 4-7 leftover values cases.
  if (curnbr_fields_to_convert >= 4) { /* If there are 4 or more,
                                      load left vector, then decide how
                                      to load partial right */
    in_vector_left =
        *(vec_float32 *)
            cur_input_data++; /* Load vector with left 4 FP32 elements */
    if (curnbr_fields_to_convert > 4) /* if more than 4, partially
                                      load right vector from data */
      in_vector_right =
          (vec_float32)vec_load_len((const float *)cur_input_data,
                                    (curnbr_fields_to_convert - 4) * 4 - 1); /*
                                       Partial load right vector with proper
                                       number of FP32 elements */
    else
      in_vector_right = (vec_float32){0, 0, 0, 0}; // Else clear right vector
  } else { /* Else there are less than 4 values, so partially
                load the left, and clear the right */
    in_vector_left = (vec_float32)vec_load_len(
        (const float *)cur_input_data, curnbr_fields_to_convert * 4 - 1);
    in_vector_right = (vec_float32){0, 0, 0, 0};
  }

  // Invoke the VCRNF
  out_vector = aiu_vec_round_from_fp32_inline(in_vector_left, in_vector_right);
  vec_store_len(out_vector, (uint16_t *)cur_output_data,
                curnbr_fields_to_convert * 2 - 1);

  // Don't need to update the cur_input_data or cur_output_data because
  // we're done.  Also, don't need to update curnbr_fields_to_convert.

  return nbr_fields_to_convert;
} // End fp32_to_dlf16

/***********************************************************************
 * dlf16_to_fp32
 *
 * Converts stick elements from 16 bit DLFLOAT to 32 bit Floating Point
 *
 * Input: Address of N DLFLOAT16 data elements to convert
 *        Address to store N FP32 data elements.
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/

uint64_t dlf16_to_fp32(uint16_t *input_data, float *output_data,
                       uint64_t nbr_fields_to_convert) {

  // Set vector pointers from input/output pointers passed in.
  // Note: adding 1 to a vector pointer will move it ahead 16 bytes
  vec_int16 *cur_input_data =
      (vec_int16 *)input_data; // Point to input vector data
  vec_float32 *cur_output_data =
      (vec_float32 *)output_data; // Point to output data

  vec_int16 in_vector; /* Define a input vector for UNPACK operation.
                          a Vector Register can fit 8 int16 fields */
  vec_float32
      out_vector_left; /* Define a vector to store the left four output
                          data fields from. This will be the Left half of
                          the result vector. Vector can fit 4 int32 fields */
  vec_float32
      out_vector_right; /* Define a vector to store the right four output
                           data fields from. This will be the right half of
                           the result vector. Vector can fit 4 int32 fields */

  // If there's more than 8 to convert, convert groups of 8 FP16s to 8 DL32s
  for (int i = 0; i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       ++i) {
    aiu_vec_lengthen_to_fp32_inline(*(vec_int16 *)(cur_input_data),
                                    (vec_float32 *)(cur_output_data),
                                    (vec_float32 *)(cur_output_data + 1));
    cur_input_data++; /* bump ptr to start of next input vector (8 shorts) */
    cur_output_data += 2; /* bump ptr to start of next pair of vector (8
                          float32s = 32 bytes) */
  }                       // End of for loop

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left to
                                       // convert

  if (!curnbr_fields_to_convert)  // If none,
    return nbr_fields_to_convert; // Return, indicating all converted

  // If there's still some to convert, it will be 7 or less, and we should
  // tread carefully to avoid touching user data beyond what they said they
  // gave us.

  // use functions to load partial vector
  int curnbr_bytes_to_set = curnbr_fields_to_convert * sizeof(uint32_t);

  in_vector =
      vec_load_len((uint16_t *)cur_input_data,
                   curnbr_fields_to_convert * sizeof(uint16_t) - 1); /* Load
                              vector with up to 8 elements,
                              Length is offset by 1. */

  aiu_vec_lengthen_to_fp32_inline(in_vector, (vec_float32 *)&out_vector_left,
                                  (vec_float32 *)&out_vector_right);
  vec_store_len(out_vector_left, (uint32_t *)cur_output_data,
                curnbr_bytes_to_set - 1); /* Store left FP32 to output (1 to 4
                                             values), Length is offset by 1. */
  // If there's more than 4 to convert, store values 5-8
  if (curnbr_fields_to_convert > 4) {
    curnbr_bytes_to_set -= sizeof(uint32_t) * 4; /* Reduce bytes_to_set by how
                                                    many we already stored */
    vec_store_len(out_vector_right, (uint32_t *)(cur_output_data + 1),
                  curnbr_bytes_to_set - 1); /* Store right 4 FP32 to output */
  }

  return nbr_fields_to_convert;
} // End dlf16_to_fp32

/**********************************************************************
 * bfloat_to_dlf16
 *
 * Converts stick elements from bfloat to 16 bit DLFLOAT by:
 *   - extending the 16-bit bfloat to a 32-bit float (FP32) with
 *     vector merge ops, inserting extra decimal into each value
 *   - using aiu_vec_round_from_fp32 (vector DLFloat op) to convert
 *     all values to DLFloat
 *
 * Input: Address of N bfloat data elements to convert
 *        Address to store N dlfloat data elements.
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/
uint64_t bfloat_to_dlf16(uint16_t *input_data, uint16_t *output_data,
                         uint64_t nbr_fields_to_convert) {

  // Set vector pointers from input/output pointers passed in.
  // Note: adding 1 to a vector pointer will move it ahead 16 bytes
  vec_int16 *cur_input_data =
      (vec_int16 *)input_data; // Point to input vector data
  vec_int16 interim_data1;     // Holds interim FP32 data
  vec_int16 interim_data2;     // Holds interim FP32 data
  vec_int16 *cur_output_data = (vec_int16 *)output_data; // Point to output data

  vec_int16 in_vector;  // Define a input vector for UNPACK operation.
                        // a Vector Register can fit 8 int16 fields
  vec_int16 out_vector; // Define a vector to hold a partial output
                        // vector.

  // If there's more than 8 to convert, convert groups of 8 FP16s to 8 DL32s
  for (int i = 0; i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       ++i) {
    // Conversion processing: Use vector merge to insert extra decimal
    // places into the vector, expanding the bfloat16 into an FP32,
    // then use our "convert and round" routine to transform to dlfloat
    interim_data1 = vec_mergeh(*cur_input_data, zero_vector16);
    interim_data2 = vec_mergel(*cur_input_data, zero_vector16);
    *cur_output_data = aiu_vec_round_from_fp32_inline(
        (vec_float32)interim_data1, (vec_float32)interim_data2);

    cur_input_data++;  /* bump ptr to start of next input vector (8 shorts) */
    cur_output_data++; /* bump ptr to start of next vector (8 short) */
  }                    // End of for loop

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left to
                                       // convert

  if (!curnbr_fields_to_convert)  // If none,
    return nbr_fields_to_convert; // Return, indicating all converted

  // If there's still some to convert, it will be 7 or less, and we should
  // tread carefully to avoid touching user data beyond what they said they
  // gave us.

  // use functions to load partial vector
  int curnbr_bytes_to_set = curnbr_fields_to_convert * sizeof(uint16_t);

  in_vector =
      vec_load_len((uint16_t *)cur_input_data,
                   curnbr_fields_to_convert * sizeof(uint16_t) - 1); /* Load
                              vector with up to 8 elements,
                              Length is offset by 1. */

  // Conversion processing: Use vector merge to insert extra decimal
  // places into the vector, expanding the bfloat16 into an FP32,
  // then use our "convert and round" routine to transform to dlfloat
  interim_data1 = vec_mergeh(in_vector, zero_vector16);
  interim_data2 = vec_mergel(in_vector, zero_vector16);
  out_vector = aiu_vec_round_from_fp32_inline((vec_float32)interim_data1,
                                              (vec_float32)interim_data2);

  // Store the vector (with 1-7 elements) using a computed length to
  // ensure an exact fit.
  vec_store_len(out_vector, (uint16_t *)cur_output_data,
                curnbr_bytes_to_set - 1); /* Store left bfloat to output (1 to 4
                                             values), Length is offset by 1. */

  return nbr_fields_to_convert;
} // End bfloat_to_dlf16

/***********************************************************************
 * dlf16_to_bfloat
 *
 * Converts stick elements from 16 bit DLFLOAT to BFLOAT by:
 *   - using aiu_vec_lengthen_to_fp32 (vector DLFloat op) to convert
 *     all values to FP32
 *   - Truncate the 32-bit float (FP32) to a 16-bit BFLOAT with
 *     vector permute ops, selecting which bytes of each 32-bit value
 *     to keep, discarding the rest
 *
 * Input: Address of N DLFLOAT16 data elements to convert
 *        Address to store N bfloat data elements.
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/

uint64_t dlf16_to_bfloat(uint16_t *input_data, uint16_t *output_data,
                         uint64_t nbr_fields_to_convert) {

  // Set vector pointers from input/output pointers passed in.
  // Note: adding 1 to a vector pointer will move it ahead 16 bytes
  vec_int16 *cur_input_data =
      (vec_int16 *)input_data; // Point to input vector data
  vec_int16 *cur_output_data = (vec_int16 *)output_data; // Point to output data

  vec_float32 interim_data1; // Holds interim FP32 data
  vec_float32 interim_data2; // Holds interim FP32 data

  vec_int16 in_vector;  // Define a input vector for UNPACK operation.
                        // a Vector Register can fit 8 int16 fields
  vec_int16 out_vector; // Define a vector to store the partial output
                        // data fields.

  // If there's more than 8 to convert, convert groups of 8 FP16s to 8 DL32s
  for (int i = 0; i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT;
       ++i) {

    // Conversion processing: Use our "convert and lengthen" routine to
    // transform the DLFloat to FP32, then use vector permute to select
    // which bytes to keep (e.g. the two high order bytes of each FP32),
    // further transforming our FP32 elements into BFLOAT.
    aiu_vec_lengthen_to_fp32_inline(*(vec_int16 *)(cur_input_data),
                                    (vec_float32 *)&interim_data1,
                                    (vec_float32 *)&interim_data2);

    *cur_output_data = (vec_int16)vec_perm(
        (vec_char8)interim_data1, (vec_char8)interim_data2, selection_vector);

    cur_input_data++;  /* bump ptr to start of next input vector (8 shorts) */
    cur_output_data++; /* bump ptr to start of next output vector (8
                            shorts) */
  }                    // End of for loop

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %

      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left to
                                       // convert

  if (!curnbr_fields_to_convert)  // If none,
    return nbr_fields_to_convert; // Return, indicating all converted

  // If there's still some to convert, it will be 7 or less, and we should
  // tread carefully to avoid touching user data beyond what they said they
  // gave us.

  // use functions to load partial vector
  in_vector =
      vec_load_len((uint16_t *)cur_input_data,
                   curnbr_fields_to_convert * sizeof(uint16_t) - 1); /* Load
                              vector with up to 8 elements,
                              Length is offset by 1. */
  // Conversion processing: Use our "convert and lengthen" routine to
  // transform the DLFloat to FP32, then use vector permute to select
  // which bytes to keep (e.g. the two high order bytes of each FP32),
  // further transforming our FP32 elements into BFLOAT.

  aiu_vec_lengthen_to_fp32_inline(in_vector, (vec_float32 *)&interim_data1,
                                  (vec_float32 *)&interim_data2);

  out_vector = (vec_int16)vec_perm((vec_char8)interim_data1,
                                   (vec_char8)interim_data2, selection_vector);
  vec_store_len(out_vector, (uint16_t *)cur_output_data,
                (curnbr_fields_to_convert * sizeof(uint16_t)) -
                    1); /* Store left bfloat to output (1 to 4 values), Length
                           is offset by 1. */

  return nbr_fields_to_convert;

} // End dlf16_to_bfloat

/***********************************************************************
 * fp16_to_dlf16_in_stride
 *
 * Converts N 16-bit Floating Point to 16-bit DLFLOAT stick elements,
 * gathering the elements from a "strided" input in the user's buffer,
 * storing into contiguous elements in a stick
 *
 * Input: Address of N FP16 data elements to gather and convert
 *        Address to store N DLFLOAT16 data elements.
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/
uint64_t fp16_to_dlf16_in_stride(uint16_t *fp16_data, uint16_t *dflt16_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride) {
  float_bit16 gathered_data[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {0};
  vec_int16 *gathered_vector = (vec_int16 *)&gathered_data;

  float_bit16 *cur_fp16_data = fp16_data;
  vec_int16 *cur_dflt16_data = (vec_int16 *)dflt16_data;

  for (uint64_t i = 0;
       i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT; ++i) {
    // Gather values from across sticks
    for (int j = 0; j < STICKCVT_MAX_ENTRIES_TO_CONVERT; ++j) {
      gathered_data[j] = *cur_fp16_data;
      cur_fp16_data += input_stride;
    }

    // Convert the values
    *(vec_int16 *)(cur_dflt16_data) = aiu_vec_convert_from_fp16_inline(
        *(vec_int16 *)gathered_vector); /* Convert
            gathered values directly to user
            tensor buffer */
    cur_dflt16_data += 1; // bump ptr to next target area (16 bytes)
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left
                                       // to convert
  if (!curnbr_fields_to_convert)
    return nbr_fields_to_convert;
  else {
    vec_int16 aiu_op_output; // temp area for partial vector output
    // Gather values from across sticks
    for (int j = 0; j < (curnbr_fields_to_convert); ++j) {
      gathered_data[j] = *cur_fp16_data;
      cur_fp16_data += input_stride;
    }

    // Convert the values
    aiu_op_output =
        aiu_vec_convert_from_fp16_inline(*(vec_int16 *)(&gathered_data));
    // Copy remaining values to user tensor area
    memcpy(cur_dflt16_data, &aiu_op_output,
           (sizeof(uint16_t) * (curnbr_fields_to_convert)));
  }
  return nbr_fields_to_convert;
} // End fp16_to_dlf16_in_stride

/***********************************************************************
 * fp32_to_dlf16_in_stride
 *
 * Converts N 32-bit Floating Point to 16-bit DLFLOAT stick elements,
 * gathering the elements from a "strided" input across multiple sticks,
 * storing into contiguous elements in a stick
 *
 * Input: Address of N FP32 data elements to gather and convert
 *        Address to store N DLFLOAT16 data elements.
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/
uint64_t fp32_to_dlf16_in_stride(float *fp32_data, uint16_t *dflt16_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride) {

  float_bit32 gathered_data[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {
      0}; /* define an array */
  vec_float32 *gathered_vector =
      (vec_float32 *)&gathered_data; /* redefine the same
                                        array storage as a vector */

  float_bit32 *cur_fp32_data = &(*(float_bit32 *)fp32_data);

  vec_int16 *cur_dflt16_data = (vec_int16 *)dflt16_data;

  for (uint64_t i = 0;
       i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT; ++i) {
    // Gather values from across sticks
    for (int j = 0; j < STICKCVT_MAX_ENTRIES_TO_CONVERT; ++j) {
      gathered_data[j] = *cur_fp32_data;
      cur_fp32_data += input_stride;
    }

    // Convert the values
    *(vec_int16 *)(cur_dflt16_data) = aiu_vec_round_from_fp32_inline(
        *(vec_float32 *)(gathered_vector),
        *(vec_float32 *)(gathered_vector + 1)); /* Convert
              gathered values directly to stick ztensor buffer */
    cur_dflt16_data += 1; // bump ptr to next target area (16 bytes)
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left
                                       // to convert
  if (!curnbr_fields_to_convert)
    return nbr_fields_to_convert;
  else {
    vec_int16 aiu_op_output; // temp area for partial vector output
    // Gather values from across sticks
    for (int j = 0; j < (curnbr_fields_to_convert); ++j) {
      gathered_data[j] = *cur_fp32_data;
      cur_fp32_data += input_stride;
    }

    // Convert the values
    aiu_op_output = aiu_vec_round_from_fp32_inline(
        *(vec_float32 *)(gathered_vector),
        *(vec_float32 *)(gathered_vector + 1)); /* Convert
              gathered values directly to stick ztensor buffer */

    // Copy remaining values to user tensor area
    memcpy(cur_dflt16_data, &aiu_op_output,
           (sizeof(uint16_t) * (curnbr_fields_to_convert)));
  }
  return nbr_fields_to_convert;
} // End fp32_to_dlf16_in_stride

/***********************************************************************
 * bfloat_to_dlf16_in_stride
 *
 * Converts N 16-bit BFLOAT values to 16-bit DLFLOAT stick elements,
 * gathering the elements from a "strided" input across multiple sticks,
 * storing into contiguous elements in a stick
 *
 * Input: Address of N BFLOAT data elements to gather and convert
 *        Address to store N DLFLOAT16 data elements.
 *        Number of elements to convert (N)
 * Output: Number of elements that were converted
 *
 * Dependency:  the number of elements to convert in this call should
 *              not cross from one stick to another!
 *              (i.e. 'N' must be <= 64)
 **********************************************************************/
uint64_t bfloat_to_dlf16_in_stride(uint16_t *bflt_data, uint16_t *dflt16_data,
                                   uint64_t nbr_fields_to_convert,
                                   uint32_t input_stride) {
  float_bit16 gathered_data[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {0}; /* define
                                         an array */
  vec_int16 *gathered_vector =
      (vec_int16 *)&gathered_data; /* redefine the same array
                                      storage as a vector */

  vec_float32 interim_data1; // Holds interim FP32 data
  vec_float32 interim_data2; // Holds interim FP32 data

  float_bit16 *cur_bflt_data = bflt_data;
  vec_int16 *cur_dflt16_data = (vec_int16 *)dflt16_data;

  for (uint64_t i = 0;
       i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT; ++i) {
    // Gather values from across sticks
    for (int j = 0; j < STICKCVT_MAX_ENTRIES_TO_CONVERT; ++j) {
      gathered_data[j] = *cur_bflt_data;
      cur_bflt_data += input_stride;
    }

    // Conversion processing: Use vector merge to insert extra decimal
    // places into the vector, expanding the bfloat16 into an FP32,
    // then use our "convert and round" routine to transform to dlfloat
    interim_data1 = (vec_float32)vec_mergeh(*gathered_vector, zero_vector16);
    interim_data2 = (vec_float32)vec_mergel(*gathered_vector, zero_vector16);
    *cur_dflt16_data = aiu_vec_round_from_fp32_inline(
        (vec_float32)interim_data1, (vec_float32)interim_data2); /* Convert
                            gathered values directly to user
                            tensor buffer */

    cur_dflt16_data += 1; // bump ptr to next target area (16 bytes)
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left
                                       // to convert
  if (!curnbr_fields_to_convert)
    return nbr_fields_to_convert;
  else {
    vec_int16 aiu_op_output; // temp area for partial vector output
    // Gather values from across sticks
    for (int j = 0; j < (curnbr_fields_to_convert); ++j) {
      gathered_data[j] = *cur_bflt_data;
      cur_bflt_data += input_stride;
    }

    // Conversion processing: Use vector merge to insert extra decimal
    // places into the vector, expanding the bfloat16 into an FP32,
    // then use our "convert and round" routine to transform to dlfloat
    interim_data1 = (vec_float32)vec_mergeh(*gathered_vector, zero_vector16);
    interim_data2 = (vec_float32)vec_mergel(*gathered_vector, zero_vector16);
    aiu_op_output = aiu_vec_round_from_fp32_inline((vec_float32)interim_data1,
                                                   (vec_float32)interim_data2);

    // Copy remaining values to user tensor area
    memcpy(cur_dflt16_data, &aiu_op_output,
           (sizeof(uint16_t) * (curnbr_fields_to_convert)));
  }
  return nbr_fields_to_convert;
} // End bfloat_to_dlf16_in_stride

/***********************************************************************
 * dlf16_to_fp16_in_stride
 *
 * Converts N 16-bit DLFLOAT elements from across multiple sticks
 * to 16 bit Floating Point, storing them contiguously in the
 * user's data area
 *
 * Input: Address of N DLFLOAT16 data elements to convert
 *        Address to store N FP16 data elements.
 *        Number of elements to convert (N)
 *        Length of the element stride arcoss sticks
 * Output: Number of elements that were converted
 *
 * Assumption:  Caller will provide a stride value that allows us
 *              to gather discontiguous values from the sticks and store
 *              in contiguous output values in the user's data area.
 *              The 'N' is not required to be <= 64, because the input
 *              data area to be converted is taken from the same
 *              relative position within each stick, and contiguously
 *              written to the users data area, which is not stickified.
 **********************************************************************/
uint64_t dlf16_to_fp16_in_stride(uint16_t *dflt16_data, uint16_t *fp16_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride) {
  float_bit16 gathered_data[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {0};
  vec_int16 *gathered_vector = (vec_int16 *)&gathered_data;

  float_bit16 *cur_dflt16_data = dflt16_data;
  vec_int16 *cur_fp16_data = (vec_int16 *)fp16_data;

  for (uint64_t i = 0;
       i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT; ++i) {
    // Gather values from across sticks
    for (int j = 0; j < STICKCVT_MAX_ENTRIES_TO_CONVERT; ++j) {
      gathered_data[j] = *cur_dflt16_data;
      cur_dflt16_data += input_stride;
    }

    // Convert the values
    *(vec_int16 *)(cur_fp16_data) = aiu_vec_convert_to_fp16_inline(
        *(vec_int16 *)gathered_vector); /* Convert
            gathered values directly to user
            tensor buffer */
    cur_fp16_data += 1; // bump ptr to next target area (16 bytes)
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left
                                       // to convert
  if (!curnbr_fields_to_convert)
    return nbr_fields_to_convert;
  else {
    vec_int16 aiu_op_output; // temp area for partial vector output
    // Gather values from across sticks
    for (int j = 0; j < (curnbr_fields_to_convert); ++j) {
      gathered_data[j] = *cur_dflt16_data;
      cur_dflt16_data += input_stride;
    }

    // Convert the values
    aiu_op_output =
        aiu_vec_convert_to_fp16_inline(*(vec_int16 *)(&gathered_data));
    // Copy remaining values to user tensor area
    memcpy(cur_fp16_data, &aiu_op_output,
           (sizeof(uint16_t) * (curnbr_fields_to_convert)));
  }
  return nbr_fields_to_convert;
} // End dlf16_to_fp16_in_stride

/***********************************************************************
 * dlf16_to_fp32_in_stride
 *
 * Converts N 16-bit DLFLOAT elements from across multiple sticks
 * to 32 bit Floating Point, storing them contiguously in the
 * user's data area
 *
 * Input: Address of N DLFLOAT16 data elements to convert
 *        Address to store N FP32 data elements.
 *        Number of elements to convert (N)
 *        Length of the element stride arcoss sticks
 * Output: Number of elements that were converted
 *
 * Assumption:  Caller will provide a stride value that allows us
 *              to gather discontiguous values from the sticks and store
 *              in contiguous output values in the user's data area.
 *              The 'N' is not required to be <= 64, because the input
 *              data area to be converted is taken from the same
 *              relative position within each stick, and contiguously
 *              written to the users data area, which is not stickified.
 **********************************************************************/
uint64_t dlf16_to_fp32_in_stride(uint16_t *dflt16_data, float *fp32_data,
                                 uint64_t nbr_fields_to_convert,
                                 uint32_t input_stride) {
  float_bit16 gathered_data[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {0};

  float_bit16 *cur_dflt16_data = dflt16_data;
  float_bit32 *cur_fp32_data = (float_bit32 *)fp32_data;

  for (uint64_t i = 0;
       i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT; ++i) {
    // Gather values from across sticks
    for (int j = 0; j < STICKCVT_MAX_ENTRIES_TO_CONVERT; ++j) {
      gathered_data[j] = *cur_dflt16_data;
      cur_dflt16_data += input_stride;
    }

    // Convert the values
    aiu_vec_lengthen_to_fp32_inline(
        *(vec_int16 *)(&gathered_data), (vec_float32 *)(cur_fp32_data),
        (vec_float32 *)(cur_fp32_data + 4)); /* Convert gathered values directly
                                                into user tensor buffer */
    cur_fp32_data += STICKCVT_MAX_ENTRIES_TO_CONVERT; // bump ptr to next target
                                                      // area (32 bytes)
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left
                                       // to convert
  if (!curnbr_fields_to_convert)
    return nbr_fields_to_convert;
  else {
    vec_float32 aiu_op_output[2]; // temp area for partial vector output
    // Gather values from across sticks
    for (int j = 0; j < (curnbr_fields_to_convert); ++j) {
      gathered_data[j] = *cur_dflt16_data;
      cur_dflt16_data += input_stride;
    }

    // Convert the values
    aiu_vec_lengthen_to_fp32_inline(*(vec_int16 *)(&gathered_data),
                                    (&aiu_op_output[0]), (&aiu_op_output[1]));
    // Copy remaining values to user tensor area
    memcpy(cur_fp32_data, &aiu_op_output[0],
           (sizeof(uint32_t) * curnbr_fields_to_convert));
  }

  return nbr_fields_to_convert;
} // End dlf16_to_fp32_in_stride

/***********************************************************************
 * dlf16_to_bfloat_in_stride
 *
 * Converts N 16-bit DLFLOAT elements from across multiple sticks
 * to 16 bit Floating Point (bfloat), storing them contiguously in the
 * user's data area
 *
 * Input: Address of N DLFLOAT16 data elements to convert
 *        Address to store N bfloat data elements.
 *        Number of elements to convert (N)
 *        Length of the element stride arcoss sticks
 * Output: Number of elements that were converted
 *
 * Assumption:  Caller will provide a stride value that allows us
 *              to gather discontiguous values from the sticks and store
 *              in contiguous output values in the user's data area.
 *              The 'N' is not required to be <= 64, because the input
 *              data area to be converted is taken from the same
 *              relative position within each stick, and contiguously
 *              written to the users data area, which is not stickified.
 **********************************************************************/
uint64_t dlf16_to_bfloat_in_stride(uint16_t *dflt16_data, uint16_t *bflt_data,
                                   uint64_t nbr_fields_to_convert,
                                   uint32_t input_stride) {
  float_bit16 gathered_data[STICKCVT_MAX_ENTRIES_TO_CONVERT] = {0};
  vec_int16 *gathered_vector = (vec_int16 *)&gathered_data;

  vec_float32 interim_data1; // Holds interim FP32 data
  vec_float32 interim_data2; // Holds interim FP32 data

  float_bit16 *cur_dflt16_data = dflt16_data;
  vec_int16 *cur_bflt_data = (vec_int16 *)bflt_data;

  for (uint64_t i = 0;
       i < nbr_fields_to_convert / STICKCVT_MAX_ENTRIES_TO_CONVERT; ++i) {
    // Gather values from across sticks
    for (int j = 0; j < STICKCVT_MAX_ENTRIES_TO_CONVERT; ++j) {
      gathered_data[j] = *cur_dflt16_data;
      cur_dflt16_data += input_stride;
    }

    // Conversion processing: Use our "convert and lengthen" routine to
    // transform the DLFloat to FP32, then use vector permute to select
    // which bytes to keep (e.g. the two high order bytes of each FP32),
    // further transforming our FP32 elements into BFLOAT.
    aiu_vec_lengthen_to_fp32_inline(*(vec_int16 *)(gathered_vector),
                                    (vec_float32 *)&interim_data1,
                                    (vec_float32 *)&interim_data2);

    *cur_bflt_data = (vec_int16)vec_perm(
        (vec_char8)interim_data1, (vec_char8)interim_data2, selection_vector);

    cur_bflt_data += 1; // bump ptr to next target area (16 bytes)
  }

  int curnbr_fields_to_convert =
      nbr_fields_to_convert %
      STICKCVT_MAX_ENTRIES_TO_CONVERT; // Determine # fields left
                                       // to convert
  if (!curnbr_fields_to_convert)       // If none,
    return nbr_fields_to_convert;      // Return, indicating all converted
  else {
    vec_int16 aiu_op_output; // temp area for partial vector output
    // Gather values from across sticks
    for (int j = 0; j < (curnbr_fields_to_convert); ++j) {
      gathered_data[j] = *cur_dflt16_data;
      cur_dflt16_data += input_stride;
    }

    // Conversion processing: Use our "convert and lengthen" routine to
    // transform the DLFloat to FP32, then use vector permute to select
    // which bytes to keep (e.g. the two high order bytes of each FP32),
    // further transforming our FP32 elements into BFLOAT.
    aiu_vec_lengthen_to_fp32_inline(*(vec_int16 *)(gathered_vector),
                                    (vec_float32 *)&interim_data1,
                                    (vec_float32 *)&interim_data2);

    aiu_op_output = (vec_int16)vec_perm(
        (vec_char8)interim_data1, (vec_char8)interim_data2, selection_vector);

    memcpy(cur_bflt_data, &aiu_op_output,
           (sizeof(uint16_t) * curnbr_fields_to_convert));
  }
  return nbr_fields_to_convert;
} // End dlf16_to_bfloat_in_stride
