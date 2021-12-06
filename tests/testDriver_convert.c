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
/*
 * This test driver tests the data type conversion code upon which
 * the Stickify/Unstickify paths are dependent for conversion AND
 * proper value placement.
 *
 * Each test creates a set of random float values (FP32, FP16 or BFLOAT)
 * and calls a common routine to build its own version of the converted
 * values, invoke the library's convert_data_format, then compare the two
 * areas for expected values and placement.  It then does the opposite:
 * invokes teh library's convert_data_format to convert back to the
 * original format, and compares the input area to the converted/unconverted
 * area for proper placement.
 *
 * Note that the 'no stride' Stickify/unstickify processing will handle sets of
 *values numbering larger than 64, so values up to 64 are tested here.
 *
 * Also note that the stride versions will likely have different validation
 * because it *doesn't* have the aforementioned '64 entry' limitation.
 */

#include "convert.h"
#include "testsupport.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * tests:
 * test FP32->DLFloat, using 1,4,7,8,9,15,63,64 (no stride)
 * test FP16->DLFloat, using 1,7,8,9,63,64 (no stride)
 * test BFLOAT->DLFloat, using 1,7,8,9,63,64 (no stride)
 *
 * test DLFloat->FP16, using 1,7,8,9,63,64 (no stride)
 * test DLFloat->FP32, using 1,4,7,8,9,15,63,64 (no stride)
 * test DLFloat->BFloat, using 1,7,8,9,63,64 (no stride)
 */

/* define some packed structs for holding data  */
// midfloat_str used by FP16 testing, easily grabs
// middle two bytes of a FP32 and treats it as a
// 2 byte float.
typedef
#ifdef __MVS__
    _Packed
#endif
    struct midfloat_str {
  uint8_t filler1;
  float_bit16 shortfloat;
  uint8_t filler2;
}
#ifndef __MVS__
__attribute__((packed))
#endif
midfloat_str;

// expected_data_str structure with a union to
// allow us to convert individual values then compare as
// one data area
typedef
#ifdef __MVS__
    _Packed
#endif
    struct expected_data_str {
  union maps {
    float_bit16 shortfloat[64];
    float expfloat[64];
    // cppcheck-suppress unusedStructMember
    char exp_data_reserved[1024];
  } maps;
}
#ifndef __MVS__
__attribute__((packed))
#endif
expected_data_str;

/*
vec_char8 selection_vector = {0,  1,  4,  5,  8,  9,  12, 13,
16, 17, 20, 21, 24, 25, 28, 29};*/

/*  convert_and_compare

    Accepts an array of up to 64 values, converts the values to DL16
    itself, calls convert_data_format to do its thing,
    and compares the two areas. Then converts back and compares that
    to the original. Return values are multiplied by constants to separate
    different types of errors. */
int convert_and_compare(zdnn_data_types in_type, int numvalues,
                        void *fixeddata) {

  // Define areas for stickify conversion to return results
  char converted_DLF_data[1024];
  char converted_orig_data[1024];
  memset((void *)(converted_DLF_data), 0, 1024);
  memset((void *)(converted_orig_data), 0, 1024);

  // Define an expected data area for comparing our version of converted
  // values (and placement) to The Library's.
  expected_data_str expected_DLF_data;
  memset((void *)(&expected_DLF_data), 0, 1024);

  // Define a lossy data area for comparing the original data (with expected
  // precision loss) and The Library's converted-back-to-original data.
  expected_data_str expected_orig_data;
  memset((void *)(&expected_orig_data), 0, 1024);

  float_bit16 *fixed_float_bit16 = (float_bit16 *)fixeddata;
  float *fixedfloat = (float *)fixeddata;

  // Build the "expected" areas that we will compare to conversion results
  for (int i = 0; i < numvalues; i = i + 1) {

    if (in_type == FP32) {

      expected_DLF_data.maps.shortfloat[i] =
          cnvt_1_fp32_to_dlf16(fixedfloat[i]); /* Convert a value, store in
                                                expected dlfloat entry */
      LOG_DEBUG("++ c_1_fp32_to_dlf for expected DLF %d of %d", i, numvalues);
      LOG_DEBUG("First : %x, Second: %x", fixedfloat[i],
                expected_DLF_data.maps.shortfloat[i]);

      expected_orig_data.maps.expfloat[i] = cnvt_1_dlf16_to_fp32(
          expected_DLF_data.maps.shortfloat[i]); /* Convert a value back to
                                                    original format, store in
                                                    expected original format
                                                    entry */
      LOG_DEBUG("++ c_1_dlf16_to_FP32 for expected Orig %d of %d", i,
                numvalues);
      LOG_DEBUG("First : %x, Second: %x", fixedfloat[i],
                expected_orig_data.maps.shortfloat[i]);
    }

    if (in_type == FP16) {
      expected_DLF_data.maps.shortfloat[i] =
          cnvt_1_fp16_to_dlf16(fixed_float_bit16[i]); /* Convert a value, store
                                                in expected dlfloat entry */

      expected_orig_data.maps.shortfloat[i] = cnvt_1_dlf16_to_fp16(
          expected_DLF_data.maps.shortfloat[i]); /* Convert a value back to
                                                    original format, store in
                                                    expected original format
                                                    entry */
    }

    if (in_type == BFLOAT) {
      expected_DLF_data.maps.shortfloat[i] =
          cnvt_1_bfloat_to_dlf16(fixed_float_bit16[i]); /* Convert a value,
                                         store in expected dlfloat entry */
      expected_orig_data.maps.shortfloat[i] =
          cnvt_1_dlf16_to_bfloat(expected_DLF_data.maps.shortfloat[i]); /*
                                         Convert a value back to original
                                         format, store in expected
                                         original format entry */
    }
  }

  // call convert_data to convert/stickify the original data
  LOG_DEBUG("Calling convert_data_format", NO_ARG);
  int converted_cnt = convert_data_format(
      fixeddata, in_type, converted_DLF_data, ZDNN_DLFLOAT16, numvalues);
  if (converted_cnt != numvalues) {
    LOG_DEBUG("convert_data (to DLF) did not return proper result (%d != %d)",
              converted_cnt, numvalues);
    TEST_FAIL_MESSAGE("convert_data (to DLF) count did not match actual");
  }

  // compare expected to convert_data_format output
  LOG_DEBUG("comparing expected to convert_data output", NO_ARG);

  LOG_DEBUG("expected data - first word / last word %d / %d",
            *(int *)((char *)&expected_DLF_data),
            *(int *)((char *)&expected_DLF_data) +
                (numvalues * get_data_type_size(ZDNN_DLFLOAT16)) - 4);
  LOG_DEBUG("expected data address %" PRIXPTR "",
            (uint64_t)((char *)&expected_DLF_data));

  LOG_DEBUG("converted data - first word / last word %d / %d",
            *(int *)((char *)converted_DLF_data),
            *(int *)((char *)converted_DLF_data) +
                (numvalues * get_data_type_size(ZDNN_DLFLOAT16)) - 4);
  LOG_DEBUG("converted data address %" PRIXPTR "",
            (uint64_t)((char *)converted_DLF_data));

  TEST_ASSERT_MESSAGE(sizeof(expected_DLF_data) == sizeof(converted_DLF_data),
                      "expected data sizes different (test u/t error)");

  int compare_data_size = sizeof(expected_DLF_data);

  /* validate converted area has something in it */
  char zeroes[256];
  memset(zeroes, 0, sizeof(zeroes));
  TEST_ASSERT_MESSAGE(
      memcmp(converted_DLF_data, zeroes, (numvalues * sizeof(short))) != 0,
      "converted-to-dlf area left as zeros");

  /* Compare expected DLFLOAT to converted DLFLOAT, and validate */
  int memcmp_rc =
      memcmp(&expected_DLF_data, converted_DLF_data, (size_t)compare_data_size);
  if (memcmp_rc != 0) {
    BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
      printf("memcmp (post convert to DLF) did not return proper result (%d)",
             memcmp_rc);
      printf("expected DLFloat data\n");
      print_hex((numvalues * sizeof(float)), &expected_DLF_data);
      printf("Converted DLFloat data\n");
      print_hex((numvalues * sizeof(float)), converted_DLF_data);
    }
    TEST_FAIL_MESSAGE(
        "memcmp (post convert to DLF, no stride) did not match expected");
  }

  // call convert_data in stride to convert/stickify the original data
  LOG_DEBUG("call convert_data_in_stride", NO_ARG);
  converted_cnt = convert_data_format_in_stride(
      fixeddata, in_type, converted_DLF_data, ZDNN_DLFLOAT16, numvalues, 1);
  if (converted_cnt != numvalues) {
    LOG_DEBUG("Converted (in_stride) count doesn't match actual, %d / %d",
              converted_cnt, numvalues);
    TEST_FAIL_MESSAGE(
        "Convert_data (to DLF) in stride did not return proper result");
  }
  /* Compare expected DLFLOAT to converted DLFLOAT, and validate */
  memcmp_rc =
      memcmp(&expected_DLF_data, converted_DLF_data, (size_t)compare_data_size);
  if (memcmp_rc != 0) {
    BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
      printf("Expected data doesn't match converted, %d", memcmp_rc);
      printf("expected DLFloat data\n");
      print_hex((numvalues * sizeof(float)), &expected_DLF_data);
      printf("Converted DLFloat data\n");
      print_hex((numvalues * sizeof(float)), converted_DLF_data);
    }
    TEST_FAIL_MESSAGE("Converted DLF data (instride) did not match");
  }

  // Now convert back the other way, and compare to original
  LOG_DEBUG(
      "comparing data converted back to Orig format by convert_data output",
      NO_ARG);
  int orig_data_size = numvalues * get_data_type_size(in_type);

  LOG_DEBUG("call convert_data", NO_ARG);
  int converted_cnt2 =
      convert_data_format(converted_DLF_data, ZDNN_DLFLOAT16,
                          converted_orig_data, in_type, numvalues);
  if (converted_cnt2 != numvalues) {
    LOG_DEBUG("converted count (to_orig) did not match actual (%d != %d)",
              converted_cnt2, numvalues);
    TEST_FAIL_MESSAGE(
        "convert_data (to orig, no stride) count did not match actual");
  }
  TEST_ASSERT_MESSAGE(
      memcmp(converted_orig_data, zeroes, (numvalues * sizeof(short))) != 0,
      "converted-to-original area left as zeros");

  int memcmp_rc2 =
      memcmp(&expected_orig_data, converted_orig_data, (size_t)orig_data_size);
  if (memcmp_rc2 != 0) {
    BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
      printf("memcmp (after convert back to original) did not return "
             "proper result (%d)",
             memcmp_rc2);
      printf("expected orig vs converted orig data\n");
      print_hex(orig_data_size, &expected_orig_data);
      print_hex(orig_data_size, converted_orig_data);
    }
    TEST_FAIL_MESSAGE("convert_data (back to orig) did not match initial");
  }
  return ZDNN_STATUS_OK;
}

// generate a float value between SMALLEST_RANDOM_FP to max
#define GEN_RAND_FLOAT(x, max)                                                 \
  while ((x) < SMALLEST_RANDOM_FP) {                                           \
    (x) = (float)rand() / (float)(RAND_MAX / max);                             \
  }

/*********************/
/* FP32 to DLF tests */
/*********************/

void test_FP32_DLF(int count) {

  float fixeddata[128] = {0};

  // Build a tensor data area of req'd type with random data
  for (int i = 0; i < count; i++) {
    GEN_RAND_FLOAT(fixeddata[i], 3);
  }

  int test_result = convert_and_compare(FP32, count, fixeddata);

  TEST_ASSERT_MESSAGE(0 == test_result,
                      "Converted and expected areas did not match");
}

void test_FP32_DLF_1() { test_FP32_DLF(1); }

void test_FP32_DLF_4() { test_FP32_DLF(4); }

void test_FP32_DLF_7() { test_FP32_DLF(7); }

void test_FP32_DLF_8() { test_FP32_DLF(8); }

void test_FP32_DLF_9() { test_FP32_DLF(9); }

void test_FP32_DLF_15() { test_FP32_DLF(15); }

void test_FP32_DLF_63() { test_FP32_DLF(63); }

void test_FP32_DLF_64() { test_FP32_DLF(64); }

void test_16_DLF(zdnn_data_types type, int count) {

  float_bit16 fixeddata[4096] = {0};

  // Build a tensor data area of req'd type with random data
  for (int i = 0; i < count; i++) {
    float temp_float = 0;
    GEN_RAND_FLOAT(temp_float, 3);

    if (type == FP16) {
      fixeddata[i] = cnvt_1_fp32_to_fp16(temp_float);
    } else if (type == BFLOAT) {
      fixeddata[i] = cnvt_1_fp32_to_bfloat(temp_float);
    }
  }

  int test_result = convert_and_compare(type, count, fixeddata);

  TEST_ASSERT_MESSAGE(0 == test_result,
                      "Converted and expected areas did not match");
}

void test_FP16_DLF_1() {
#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE(
      "when ZDNN_CONFIG_NO_NNPA is set FP16<->DLFLOAT16 is noop");
#endif
  test_16_DLF(FP16, 1);
}

void test_FP16_DLF_7() {
#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE(
      "when ZDNN_CONFIG_NO_NNPA is set FP16<->DLFLOAT16 is noop");
#endif
  test_16_DLF(FP16, 7);
}

void test_FP16_DLF_8() {
#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE(
      "when ZDNN_CONFIG_NO_NNPA is set FP16<->DLFLOAT16 is noop");
#endif
  test_16_DLF(FP16, 8);
}

void test_FP16_DLF_9() {
#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE(
      "when ZDNN_CONFIG_NO_NNPA is set FP16<->DLFLOAT16 is noop");
#endif
  test_16_DLF(FP16, 9);
}

void test_FP16_DLF_63() {
#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE(
      "when ZDNN_CONFIG_NO_NNPA is set FP16<->DLFLOAT16 is noop");
#endif
  test_16_DLF(FP16, 63);
}

void test_FP16_DLF_64() {
#ifdef ZDNN_CONFIG_NO_NNPA
  TEST_IGNORE_MESSAGE(
      "when ZDNN_CONFIG_NO_NNPA is set FP16<->DLFLOAT16 is noop");
#endif
  test_16_DLF(FP16, 64);
}

void test_BFLOAT_DLF_1() { test_16_DLF(BFLOAT, 1); }

void test_BFLOAT_DLF_7() { test_16_DLF(BFLOAT, 7); }

void test_BFLOAT_DLF_8() { test_16_DLF(BFLOAT, 8); }

void test_BFLOAT_DLF_9() { test_16_DLF(BFLOAT, 9); }

void test_BFLOAT_DLF_63() { test_16_DLF(BFLOAT, 63); }

void test_BFLOAT_DLF_64() { test_16_DLF(BFLOAT, 64); }

// cppcheck-suppress 	unusedFunction
void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

// cppcheck-suppress 	unusedFunction
void tearDown(void) {}

int main() {
  UNITY_BEGIN();
  srand(time(0)); /* set up to get random values */

  RUN_TEST(test_FP32_DLF_1);
  RUN_TEST(test_FP32_DLF_4);
  RUN_TEST(test_FP32_DLF_7);
  RUN_TEST(test_FP32_DLF_8);
  RUN_TEST(test_FP32_DLF_9);
  RUN_TEST(test_FP32_DLF_15);
  RUN_TEST(test_FP32_DLF_63);
  RUN_TEST(test_FP32_DLF_64);

  RUN_TEST(test_BFLOAT_DLF_1);
  RUN_TEST(test_BFLOAT_DLF_7);
  RUN_TEST(test_BFLOAT_DLF_8);
  RUN_TEST(test_BFLOAT_DLF_9);
  RUN_TEST(test_BFLOAT_DLF_63);
  RUN_TEST(test_BFLOAT_DLF_64);

  RUN_TEST(test_FP16_DLF_1);
  RUN_TEST(test_FP16_DLF_7);
  RUN_TEST(test_FP16_DLF_8);
  RUN_TEST(test_FP16_DLF_9);
  RUN_TEST(test_FP16_DLF_63);
  RUN_TEST(test_FP16_DLF_64);
  return UNITY_END();
}
