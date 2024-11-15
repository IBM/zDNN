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

#include "testsupport.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef vector signed short vec_short;

void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

void approximate_min_max(const float *values, const size_t num_values,
                         float *expected_min, float *expected_max) {
  *expected_min = FLT_MAX;
  *expected_max = -FLT_MAX;

  for (size_t i = 0; i < num_values; ++i) {
    *expected_min = fmin(*expected_min, values[i]);
    *expected_max = fmax(*expected_max, values[i]);
  }

  *expected_min = fmin(-0.f, CLEANSE_FP32(*expected_min));
  *expected_max = fmax(0.f, CLEANSE_FP32(*expected_max));
}

/**
 * zdnn_getrange_ztensor_test
 *
 * Handles all the logic to run custom tests.
 */
void zdnn_getrange_ztensor_test(uint32_t *dims, zdnn_data_layouts layout,
                                float *values, zdnn_status expected_status,
                                float expected_min, float expected_max) {

  /*
   * Input Tensor
   */
  zdnn_ztensor *input_ztensor =
      alloc_ztensor_with_values(dims, layout, FP32, NO_CONCAT, false, values);

  float min_val, max_val;

  /*
   * Begin Testing!
   */
  zdnn_status status = zdnn_getrange_ztensor(input_ztensor, &min_val, &max_val);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_getrange_ztensor() to returned status %08x but expected "
      "%08x\n",
      status, expected_status);

  if (expected_status == ZDNN_OK) {
    bool all_passed = true;

    uint64_t big_error_message_size =
        (uint64_t)sizeof(char) * ERROR_MESSAGE_STR_LENGTH * 2;
    char *error_msg = malloc(big_error_message_size);

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg),
             "Min == %f expecting %f", min_val, expected_min);

    if (min_val != expected_min) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), " <==== FAILED");
      all_passed = false;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg),
             "Max == %f expecting %f", max_val, expected_max);

    if (max_val != expected_max) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), " <==== FAILED");
      all_passed = false;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");

    TEST_ASSERT_MESSAGE(all_passed, error_msg);
  }

  // All done--clean up the tensor buffer
  zdnn_free_ztensor_buffer(input_ztensor);
}

/**
 * zdnn_getrange_ztensor_positive_basic
 */
void zdnn_getrange_ztensor_positive_basic() {
  // Initialize the dimensions for our input and output tensors ZDNN_NHWC
  uint32_t dims[] = {1, 3, 3, 1};

  int num_values = dims[0] * dims[1] * dims[2] * dims[3];

  float values[num_values];
  gen_random_float_array(num_values, values);

  float expected_min, expected_max;
  approximate_min_max(values, num_values, &expected_min, &expected_max);

  zdnn_getrange_ztensor_test(dims, ZDNN_NHWC, values, ZDNN_OK, expected_min,
                             expected_max);
}

/**
 * zdnn_getrange_ztensor_negative_basic
 */
void zdnn_getrange_ztensor_negative_basic() {
  // Initialize the dimensions for our input and output tensors ZDNN_NHWC
  uint32_t dims[] = {1, 3, 3, 1};

  int num_values = dims[0] * dims[1] * dims[2] * dims[3];

  float values[num_values];
  gen_random_float_array_neg(num_values, values);

  float expected_min, expected_max;
  approximate_min_max(values, num_values, &expected_min, &expected_max);

  zdnn_getrange_ztensor_test(dims, ZDNN_NHWC, values, ZDNN_OK, expected_min,
                             expected_max);
}

/**
 * zdnn_getrange_ztensor_positive_negative_basic
 */
void zdnn_getrange_ztensor_positive_negative_basic() {
  // Initialize the dimensions for our input and output tensors ZDNN_NHWC
  uint32_t dims[] = {1, 3, 3, 1};

  int num_values = dims[0] * dims[1] * dims[2] * dims[3];

  float values[num_values];
  gen_random_float_array_pos_neg(num_values, values);

  float expected_min, expected_max;
  approximate_min_max(values, num_values, &expected_min, &expected_max);

  zdnn_getrange_ztensor_test(dims, ZDNN_NHWC, values, ZDNN_OK, expected_min,
                             expected_max);
}

/**
 * zdnn_getrange_ztensor_positive_large
 */
void zdnn_getrange_ztensor_positive_large() {
  // Initialize the dimensions for our input and output tensors ZDNN_NHWC
  uint32_t dims[] = {2, 3, 33, 65};

  int num_values = dims[0] * dims[1] * dims[2] * dims[3];

  float values[num_values];
  gen_random_float_array(num_values, values);

  float expected_min, expected_max;
  approximate_min_max(values, num_values, &expected_min, &expected_max);

  zdnn_getrange_ztensor_test(dims, ZDNN_NHWC, values, ZDNN_OK, expected_min,
                             expected_max);
}

/**
 * zdnn_getrange_ztensor_negative_large
 */
void zdnn_getrange_ztensor_negative_large() {
  // Initialize the dimensions for our input and output tensors ZDNN_NHWC
  uint32_t dims[] = {2, 3, 33, 65};

  int num_values = dims[0] * dims[1] * dims[2] * dims[3];

  float values[num_values];
  gen_random_float_array_neg(num_values, values);

  float expected_min, expected_max;
  approximate_min_max(values, num_values, &expected_min, &expected_max);

  zdnn_getrange_ztensor_test(dims, ZDNN_NHWC, values, ZDNN_OK, expected_min,
                             expected_max);
}

/**
 * zdnn_getrange_ztensor_positive_negative_large
 */
void zdnn_getrange_ztensor_positive_negative_large() {
  // Initialize the dimensions for our input and output tensors ZDNN_NHWC
  uint32_t dims[] = {2, 3, 33, 65};

  int num_values = dims[0] * dims[1] * dims[2] * dims[3];

  float values[num_values];
  gen_random_float_array_pos_neg(num_values, values);

  float expected_min, expected_max;
  approximate_min_max(values, num_values, &expected_min, &expected_max);

  zdnn_getrange_ztensor_test(dims, ZDNN_NHWC, values, ZDNN_OK, expected_min,
                             expected_max);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(zdnn_getrange_ztensor_positive_basic);
  RUN_TEST(zdnn_getrange_ztensor_negative_basic);
  RUN_TEST(zdnn_getrange_ztensor_positive_negative_basic);
  RUN_TEST(zdnn_getrange_ztensor_positive_large);
  RUN_TEST(zdnn_getrange_ztensor_negative_large);
  RUN_TEST(zdnn_getrange_ztensor_positive_negative_large);
  UNITY_END();
}
