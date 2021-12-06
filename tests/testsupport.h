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

#ifndef TESTS_TESTSUPPORT_H_
#define TESTS_TESTSUPPORT_H_

#include "convert.h"
#include "unity.h"
#include "zdnn.h"
#include "zdnn_private.h"

// NOTE: ZDNN_CONFIG_NO_NNPA is not defined until after "zdnn_private.h"
#ifdef ZDNN_CONFIG_NO_NNPA
// when ZDNN_CONFIG_NO_NNPA is set we might need the scaffolded conversion
// routines
#include "convert.h"
#endif

#include <float.h>
#include <stddef.h>

#define AIU_METHOD_STR_LENGTH 32

extern float ZERO_ARRAY[1];

#define NO_CONCAT 0xFFFFFFFF

// "default" failure when non of the ZDNN_STATUS's si appropriate,
// likely due to something's wrong with the testcase itself
#define GENERAL_TESTCASE_FAILURE 0xDEADBEEF

// Generate path string to a pregenerated offset file
#define OFFSET_FILE(layout, d4, d3, d2, d1)                                    \
  "resources/offset_files/" #layout "_" #d4 "x" #d3 "x" #d2 "x" #d1 ".txt"

typedef enum offset_mode {
  NO_OFFSETS,    // Don't generate offsets. Not a valid mode.
  QUICK_OFFSETS, // Fast but not always correct. Best for small dims.
                 // see https://jsw.ibm.com/browse/ZAI-206 for details.
  FILE_OFFSETS   // Load pre-generated offsets (see stick_fe.py).
} offset_mode;

uint64_t get_offsets_from_file(const char *file_name, size_t *array);
size_t *alloc_offsets(zdnn_ztensor *ztensor, offset_mode mode,
                      const char *path);
size_t *alloc_rnn_output_offsets(const zdnn_ztensor *ztensor);

void *alloc_and_convert_float_values(zdnn_data_types type, uint64_t num_values,
                                     bool repeat_first_value, float *values);
zdnn_ztensor *alloc_ztensor_with_values(uint32_t *shape,
                                        zdnn_data_layouts pre_tfrmd_layout,
                                        zdnn_data_types type,
                                        zdnn_concat_info info,
                                        int repeat_first_value, ...);
void free_ztensor_buffers(uint32_t num_ztensors, ...);

// Struct for floating point value tolerance information.
typedef struct fp_tolerance {
  uint32_t ulps;         // unit in the last place
  uint32_t epsilon_mult; // epsilon multiplier
} fp_tolerance;

extern fp_tolerance tol_bfloat, tol_fp16, tol_fp32;

void assert_ztensor_values(zdnn_ztensor *ztensor,
                           bool repeat_first_expected_value, void *values);
void assert_ztensor_values_adv(zdnn_ztensor *ztensor,
                               bool repeat_first_expected_value, void *values,
                               fp_tolerance tol);

unsigned char *create_and_fill_fp_data(zdnn_tensor_desc *desc);
unsigned char *create_and_fill_random_fp_data(zdnn_ztensor *ztensor);
void gen_random_float_array(int size, float arr[]);

void gen_random_float_array_neg(int size, float arr[]);
void gen_random_float_array_pos_neg(int size, float arr[]);
void gen_float_array_zeros(int size, float arr[]);
void copy_to_array(int size, float input[], float output[]);

void fill_everyother_with_zero_float_array(int size, float arr[]);
void fill_all_with_zero_float_array(int size, float arr[]);

#define SEQUENTIAL_FILL_INTERVAL 1.0F
#define SEQUENTIAL_FILL_MAX 1024.0F // sacrifice BFLOAT, 256 is too small

// "OK" tolerance values.
//
// As everything gets converted to DLFLOAT16 and back, some data types will fare
// better dealing with precision loss than others, thus the different values
// among the data types.
//
// Some ops may need higher/lower tolerance than these defaults.
#define MAX_ULPS_BFLOAT 8
#define MAX_ULPS_FP16 8
#define MAX_ULPS_FLOAT (16384 * 8)
#define MAX_ULPS_DLFLOAT16 8

#define MAX_EPSILON_MULT_BFLOAT 8
#define MAX_EPSILON_MULT_FP16 8
#define MAX_EPSILON_MULT_FLOAT (5120 * 8)
#define MAX_EPSILON_MULT_DLFLOAT16 8

// epsilon = 2 ^ (num_mantissa_bits - 1)
#define EPSILON_BFLOAT 0.00390625F                // 2 ^ -8
#define EPSILON_FP16 0.00048828125F               // 2 ^ -11
#define EPSILON_FLOAT 0.000000059604644775390625F // 2 ^ -24, FLT_EPSILON
#define EPSILON_DLFLOAT16 0.0009765625F           // 2 ^ -10

bool almost_equal_bfloat_adv(uint16_t actual, uint16_t expected,
                             fp_tolerance tol);
bool almost_equal_fp16_adv(uint16_t actual, uint16_t expected,
                           fp_tolerance tol);
bool almost_equal_float_adv(float actual, float expected, fp_tolerance tol);
bool almost_equal_dlf16_adv(uint16_t actual, uint16_t expected,
                            fp_tolerance tol);

bool almost_equal_bfloat(uint16_t actual, uint16_t expected);
bool almost_equal_fp16(uint16_t actual, uint16_t expected);
bool almost_equal_float(float actual, float expected);
bool almost_equal_dlf16(uint16_t actual, uint16_t expected);

// in some cases we can't use the single-precision float values as-is for
// calculating expected results.  these macros convert a given single-precision
// value to its "representable-by-AIU" value w.r.t. its pre-transformed data
// type
#define CLEANSE_BFLOAT(x)                                                      \
  cnvt_1_dlf16_to_fp32(                                                        \
      cnvt_1_fp32_to_dlf16(cnvt_1_bfloat_to_fp32((cnvt_1_fp32_to_bfloat(x)))))
#define CLEANSE_FP16(x)                                                        \
  cnvt_1_dlf16_to_fp32(                                                        \
      cnvt_1_fp32_to_dlf16(cnvt_1_fp16_to_fp32((cnvt_1_fp32_to_fp16(x)))))
#define CLEANSE_FP32(x) cnvt_1_dlf16_to_fp32(cnvt_1_fp32_to_dlf16(x))

// Max/min absolute values for some of the test random float generators
#define LARGEST_RANDOM_FP 5.0F
#define SMALLEST_RANDOM_FP 0.00006F

// -----------------------------------------------------------------------------
// Max values by type (to create NNPA overflow)
// -----------------------------------------------------------------------------

#define MAX_FP32 FLT_MAX
//#define MAX_FP32 3.4028234663852885981170418348452e+38
#define MAX_FP16 ((float)65504) // 2^15 * (1 + 1023/1024)
#define MAX_BFLOAT FLT_MAX
#define MAX_DLF16 ((float)8581545984) // 2^32 * (1 + 511/512)

#if defined(ZDNN_CONFIG_SIMULATION)
#define NUM_PRE_TFRMD_TYPES 1
#else
#define NUM_PRE_TFRMD_TYPES 3
#endif

#define NUM_TFRMD_TYPES 1
extern zdnn_data_types pre_tfrmd_types[NUM_PRE_TFRMD_TYPES];
extern zdnn_data_types tfrmd_types[NUM_TFRMD_TYPES];

#define NUM_PREV_LAYERS 2
#define NUM_BIASES_USAGES 2
#define NUM_NO_VCONCAT_INFOS 3

extern zdnn_concat_info prev_layers[NUM_PREV_LAYERS];
extern zdnn_concat_info biases_usages[NUM_BIASES_USAGES];
extern zdnn_concat_info no_vconcat_infos[NUM_NO_VCONCAT_INFOS];

void stdout_to_pipe();
void stderr_to_pipe();
void restore_stdout(char *buf, int buf_size);
void restore_stderr(char *buf, int buf_size);

/* The following defines a macro to verify the hardware environment for our
 * tests to successfully run in. Most tests require the proper HW environment
 * to succeed. Even some of the others, like "..._fail" tests, are looking for
 * a specific error, but can't rely on the root cause of that error without
 * the proper HW environment. In the event the proper HW environment is not
 * available, we will ignore or skip those tests.
 *
 * When an NNPA build occurs (ZDNN_CONFIG_NO_NNPA wasn't defined) we require
 * that NNPA hardware be available, otherwise we must skip tests.
 *
 * When we have a non-NNPA build, we require that NNPA hardware is not
 * available, otherwise we must skip tests.
 *
 * Simply invoke it in the Unity "setup" proc or within specific tests.
 */
#ifndef ZDNN_CONFIG_NO_NNPA
#define VERIFY_HW_ENV                                                          \
  if (!zdnn_is_nnpa_installed())                                               \
    TEST_IGNORE_MESSAGE("NNPA build but NNPA hardware not available");
#else
#define VERIFY_HW_ENV                                                          \
  if (zdnn_is_nnpa_installed())                                                \
    TEST_IGNORE_MESSAGE("Non-NNPA build but NNPA Scaffold is not setup");
#endif

/**********************************************************
 * Enhanced Unity Functions/Macros
 **********************************************************/

// standard error message string buffer for all tests to send down to Unity
#define ERROR_MESSAGE_STR_LENGTH 512
extern char error_message[ERROR_MESSAGE_STR_LENGTH];

#define TEST_FAIL_MESSAGE_FORMATTED(f, ...)                                    \
  snprintf(error_message, ERROR_MESSAGE_STR_LENGTH, (f), __VA_ARGS__);         \
  TEST_FAIL_MESSAGE(error_message);

#define TEST_ASSERT_MESSAGE_FORMATTED(cond, f, ...)                            \
  snprintf(error_message, ERROR_MESSAGE_STR_LENGTH, (f), __VA_ARGS__);         \
  TEST_ASSERT_MESSAGE((cond), error_message);

#define FUNCNAME_BANNER_LENGTH 256

extern zdnn_data_types test_datatype;

void UnityDefaultTestRunWithDataType(UnityTestFunction Func,
                                     const char *FuncName,
                                     const int FuncLineNum);

void UnityDefaultTestRunWithTfrmdDataType(UnityTestFunction Func,
                                          const char *FuncName,
                                          const int FuncLineNum);

// Macro to run test func() against all pre-transformed data-types
#define RUN_TEST_ALL_DATATYPES(func)                                           \
  UnityDefaultTestRunWithDataType(func, #func, __LINE__);

// Macro to run test func() against all transformed data-types
#define RUN_TEST_ALL_TFRMD_DATATYPES(func)                                     \
  UnityDefaultTestRunWithTfrmdDataType(func, #func, __LINE__);

#endif /* TESTS_TESTSUPPORT_H_ */
