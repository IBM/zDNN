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

#ifndef TESTS_TESTSUPPORT_H_
#define TESTS_TESTSUPPORT_H_

#include "convert.h"
#include "unity.h"
#include "zdnn.h"
#include "zdnn_private.h"

#include <float.h>
#include <stddef.h>

#define ENVVAR_TEST_ERROR_COUNT "ZDNN_TEST_ERROR_ELEMENT_COUNT"
#define ERROR_ELEMENT_COUNT_MAX_DEFAULT 10

#define AIU_METHOD_STR_LENGTH 32

extern float ZERO_ARRAY[1];

#define NO_CONCAT 0xFFFFFFFF

// "default" failure when non of the ZDNN_STATUS's si appropriate,
// likely due to something's wrong with the testcase itself
#define GENERAL_TESTCASE_FAILURE 0xDEADBEEF

size_t *alloc_offsets(zdnn_ztensor *ztensor);
size_t *alloc_rnn_offsets(const zdnn_ztensor *ztensor);
size_t *alloc_rnn_output_offsets(const zdnn_ztensor *ztensor);

void *alloc_and_convert_float_values(zdnn_data_types type, uint64_t num_values,
                                     bool repeat_first_value,
                                     const float *values);
zdnn_ztensor *alloc_ztensor_with_values(uint32_t *shape,
                                        zdnn_data_layouts pre_tfrmd_layout,
                                        zdnn_data_types type,
                                        zdnn_concat_info info,
                                        int repeat_first_value, ...);
zdnn_ztensor *alloc_output_ztensor(uint32_t *shape,
                                   zdnn_data_layouts pre_tfrmd_layout,
                                   zdnn_data_types type, zdnn_concat_info info);
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
int8_t *create_and_fill_random_int8_data(zdnn_ztensor *ztensor);
void gen_random_float_array(int size, float arr[]);

void gen_random_float_array_neg(int size, float arr[]);
void gen_random_float_array_pos_neg(int size, float arr[]);
void gen_random_float_array_range(int size, float arr[], float min, float max);
void gen_float_array_zeros(int size, float arr[]);
void copy_to_array(int size, const float input[], float output[]);

void fill_everyother_with_zero_float_array(int size, float arr[]);
void fill_all_with_zero_float_array(int size, float arr[]);
void generate_expected_output(float (*fn)(float), float input_values[],
                              int num_values, float expected_values[]);

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
// value to its "representable-by-zAIU" value w.r.t. its pre-transformed data
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
#define SMALLEST_RANDOM_FP 0.00008F
// Changed 0.00006F to 0.00008F due to exceeding upper limit of FP16 in div op

// -----------------------------------------------------------------------------
// Max values by type (to create NNPA overflow)
// -----------------------------------------------------------------------------

#define MAX_FP32 FLT_MAX
#define MAX_FP16 ((float)65504) // 2^15 * (1 + 1023/1024)
#define MAX_BFLOAT FLT_MAX
#define MAX_DLF16 ((float)8581545984) // 2^32 * (1 + 511/512)

#define NUM_ALL_PRE_TFRMD_TYPES 5
#define NUM_DLFLOAT16_PRE_TFRMD_TYPES 3
#define NUM_QUANTIZED_PRE_TFRMD_TYPES 1
#define NUM_INDEX_PRE_TFRMD_TYPES 1
#define NUM_ALL_TFRMD_TYPES 4
#define NUM_DLFLOAT16_TFRMD_TYPES 1
#define NUM_QUANTIZED_TFRMD_TYPES 1
#define NUM_INDEX_TFRMD_TYPES 1

extern zdnn_data_types dlfloat_pre_tfrmd_types[NUM_DLFLOAT16_PRE_TFRMD_TYPES];
extern zdnn_data_types tfrmd_types[NUM_DLFLOAT16_TFRMD_TYPES];

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

bool isTelumI();

/* The following defines a macro to verify the hardware environment for our
 * tests to successfully run in. Most tests require the proper HW environment
 * to succeed. Even some of the others, like "..._fail" tests, are looking for
 * a specific error, but can't rely on the root cause of that error without
 * the proper HW environment. In the event the proper HW environment is not
 * available, we will ignore or skip those tests.
 *
 *
 * Simply invoke it in the Unity "setup" proc or within specific tests.
 */
#define VERIFY_HW_ENV                                                          \
  if (!zdnn_is_nnpa_installed())                                               \
    TEST_IGNORE_MESSAGE("NNPA required for test.");

/* The following defines a macro to verify the hardware version for our tests
 * to successfully run in. Some tests require the proper hardware version to
 * succeed. Even some of the others, like "..._fail" tests, are looking for a
 * specific error, but can't rely on the root cause of that error without the
 * proper hardware version. In the event the proper hardware version is not
 * available, we will ignore or skip those tests.
 *
 * We require both that NNPA hardware and NNPA_PARMBLKFORMAT_1 be available,
 * otherwise we must skip tests.
 *
 * Simply invoke it in the Unity "setup" proc or within specific tests.
 */
#define VERIFY_PARMBLKFORMAT_1                                                 \
  if (!is_query_parmblock_installed(NNPA_PARMBLKFORMAT_1))                     \
    TEST_IGNORE_MESSAGE("NNPA hardware version not available");

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

void UnityDefaultTestRunWithDLFloat16PreDataType(UnityTestFunction Func,
                                                 const char *FuncName,
                                                 const int FuncLineNum);

void UnityDefaultTestRunWithQuantizedPreDataType(UnityTestFunction Func,
                                                 const char *FuncName,
                                                 const int FuncLineNum);

void UnityDefaultTestRunWithIndexPreDataType(UnityTestFunction Func,
                                             const char *FuncName,
                                             const int FuncLineNum);

void UnityDefaultTestRunWithDLFloat16TfrmdDataType(UnityTestFunction Func,
                                                   const char *FuncName,
                                                   const int FuncLineNum);

void UnityDefaultTestRunWithQuantizedTfrmdDataType(UnityTestFunction Func,
                                                   const char *FuncName,
                                                   const int FuncLineNum);

void UnityDefaultTestRunWithIndexTfrmdDataType(UnityTestFunction Func,
                                               const char *FuncName,
                                               const int FuncLineNum);

// Macro to run test func() against all pre-transformed data-types
#define RUN_TEST_ALL_PRE_DATATYPES(func)                                       \
  UnityDefaultTestRunWithAllPreDataType(func, #func, __LINE__);

// Macro to run test func() against all dlfloat16 pre-transformed data-types
#define RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(func)                             \
  UnityDefaultTestRunWithDLFloat16PreDataType(func, #func, __LINE__);

// Macro to run test func() against all quantized pre-transformed data-types
#define RUN_TEST_ALL_QUANTIZED_PRE_DATATYPES(func)                             \
  UnityDefaultTestRunWithQuantizedPreDataType(func, #func, __LINE__);

// Macro to run test func() against all dlfloat16 pre-transformed data-types
#define RUN_TEST_ALL_INDEX_PRE_DATATYPES(func)                                 \
  UnityDefaultTestRunWithIndexPreDataType(func, #func, __LINE__);

// Macro to run test func() against all transformed data-types
#define RUN_TEST_ALL_TFRMD_DATATYPES(func)                                     \
  UnityDefaultTestRunWithAllTfrmdDataType(func, #func, __LINE__);

// Macro to run test func() against all dlfloat16 transformed data-types
#define RUN_TEST_ALL_DLFLOAT16_TFRMD_DATATYPES(func)                           \
  UnityDefaultTestRunWithDLFloat16TfrmdDataType(func, #func, __LINE__);

// Macro to run test func() against all quantized transformed data-types
#define RUN_TEST_ALL_QUANTIZED_TFRMD_DATATYPES(func)                           \
  UnityDefaultTestRunWithQuantizedTfrmdDataType(func, #func, __LINE__);

// Macro to run test func() against all index transformed data-types
#define RUN_TEST_ALL_INDEX_TFRMD_DATATYPES(func)                               \
  UnityDefaultTestRunWithIndexTfrmdDataType(func, #func, __LINE__);

#endif /* TESTS_TESTSUPPORT_H_ */
