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

#include "common_elwise.h"

/**
 * helper function to compute the natural log without using math.h
 */
float ln(float x) {
  float old_sum = 0.0;
  float xmlxpl = (x - 1) / (x + 1);
  float xmlxpl_2 = xmlxpl * xmlxpl;
  float denom = 1.0;
  float frac = xmlxpl;
  float term = frac; // denom start from 1.0
  float sum = term;

  while (sum != old_sum) {
    old_sum = sum;
    denom += 2.0;
    frac *= xmlxpl_2;
    sum += frac / denom;
  }
  return 2.0 * sum;
}

/**
 * Helper function to compute output tensor values using elementwise
 * natural log
 */
void elwise_log(float input[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0) {
      switch (type) {
      case (BFLOAT):
        output[i] = ln(CLEANSE_BFLOAT(input[i]));
        break;
      case (FP16):
        output[i] = ln(CLEANSE_FP16(input[i]));
        break;
      case (FP32):
        output[i] = ln(CLEANSE_FP32(input[i]));
        break;
      default:
        break;
      }
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise
 * exponential
 */
void elwise_exp(float input[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = exp(CLEANSE_BFLOAT(input[i]));
      break;
    case (FP16):
      output[i] = exp(CLEANSE_FP16(input[i]));
      break;
    case (FP32):
      output[i] = exp(CLEANSE_FP32(input[i]));
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise add
 */
void elwise_add(float input1[], float input2[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = CLEANSE_BFLOAT(input1[i]) + CLEANSE_BFLOAT(input2[i]);
      break;
    case (FP16):
      output[i] = CLEANSE_FP16(input1[i]) + CLEANSE_FP16(input2[i]);
      break;
    case (FP32):
      output[i] = CLEANSE_FP32(input1[i]) + CLEANSE_FP32(input2[i]);
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise sub
 */
void elwise_sub(float input1[], float input2[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = CLEANSE_BFLOAT(input1[i]) - CLEANSE_BFLOAT(input2[i]);
      break;
    case (FP16):
      output[i] = CLEANSE_FP16(input1[i]) - CLEANSE_FP16(input2[i]);
      break;
    case (FP32):
      output[i] = CLEANSE_FP32(input1[i]) - CLEANSE_FP32(input2[i]);
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise
 * division
 */
void elwise_div(float input1[], float input2[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = CLEANSE_BFLOAT(input1[i]) / CLEANSE_BFLOAT(input2[i]);
      break;
    case (FP16):
      output[i] = CLEANSE_FP16(input1[i]) / CLEANSE_FP16(input2[i]);
      break;
    case (FP32):
      output[i] = CLEANSE_FP32(input1[i]) / CLEANSE_FP32(input2[i]);
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise
 * multiplication
 */
void elwise_mul(float input1[], float input2[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = CLEANSE_BFLOAT(input1[i]) * CLEANSE_BFLOAT(input2[i]);
      break;
    case (FP16):
      output[i] = CLEANSE_FP16(input1[i]) * CLEANSE_FP16(input2[i]);
      break;
    case (FP32):
      output[i] = CLEANSE_FP32(input1[i]) * CLEANSE_FP32(input2[i]);
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise
 * minimum
 */
void elwise_min(float input1[], float input2[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = (input1[i] < input2[i]) ? CLEANSE_BFLOAT(input1[i])
                                          : CLEANSE_BFLOAT(input2[i]);
      break;
    case (FP16):
      output[i] = (input1[i] < input2[i]) ? CLEANSE_FP16(input1[i])
                                          : CLEANSE_FP16(input2[i]);
      break;
    case (FP32):
      output[i] = (input1[i] < input2[i]) ? CLEANSE_FP32(input1[i])
                                          : CLEANSE_FP32(input2[i]);
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to compute output tensor values using elementwise
 * maximum
 */
void elwise_max(float input1[], float input2[], float output[], int num_elems,
                zdnn_data_types type) {
  for (int i = 0; i < num_elems; i++) {
    switch (type) {
    case (BFLOAT):
      output[i] = (input1[i] > input2[i]) ? CLEANSE_BFLOAT(input1[i])
                                          : CLEANSE_BFLOAT(input2[i]);
      break;
    case (FP16):
      output[i] = (input1[i] > input2[i]) ? CLEANSE_FP16(input1[i])
                                          : CLEANSE_FP16(input2[i]);
      break;
    case (FP32):
      output[i] = (input1[i] > input2[i]) ? CLEANSE_FP32(input1[i])
                                          : CLEANSE_FP32(input2[i]);
      break;
    default:
      break;
    }
  }
}

/**
 * Helper function to run end to end elementwise tests that only have
 * one input tensor
 */
void test_elwise_api_1_input(uint32_t *shape, zdnn_data_layouts layout,
                             float *input_values,
                             nnpa_function_code function_code,
                             zdnn_status expected_status) {

  // Create ztensor with input_values
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, false, input_values);

  // Create output ztensor initialized to 0's
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

#ifdef TEST_AIU // Test requires AIU

  // calculate number of values in each tensor buffer for helper function
  uint64_t num_elements = get_num_elements(output_ztensor, ELEMENTS_PRE);

  // Values in ZDNN_NHWC order
  float expected_values[num_elements];

  char api_method[AIU_METHOD_STR_LENGTH] = "zdnn_<tbd>";
  zdnn_status status = GENERAL_TESTCASE_FAILURE;

  switch (function_code) {
  case NNPA_LOG:
    strcpy(api_method, "zdnn_log");
    // Use public zDNN method to make NNPA call to AIU
    status = zdnn_log(input_ztensor, output_ztensor);

    // fill expected_values array with calculated expected values using
    // helper function
    elwise_log(input_values, expected_values, num_elements, test_datatype);
    break;
  case NNPA_EXP:
    strcpy(api_method, "zdnn_exp");
    // Use public zDNN method to make NNPA call to AIU
    status = zdnn_exp(input_ztensor, output_ztensor);

    // fill expected_values array with calculated expected values using
    // helper function
    elwise_exp(input_values, expected_values, num_elements, test_datatype);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("unsupported function_code: %d", function_code);
    break;
  }
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to %s() to returned status %08x but expected %08x", api_method,
      status, expected_status);

  // Only check expected values if expected status is ZDNN_OK
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }
#endif

  // Cleanup test tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

/**
 * Helper function to run end to end elementwise tests that only have
 * two input tensors.  This version allows the user to select
 * which type (FP32, Bfloat or FP16) they are testing.
 */
void test_elwise_api_2_inputs_adv(uint32_t *shape, zdnn_data_layouts layout,
                                  zdnn_data_types type, float *input1_values,
                                  float *input2_values,
                                  nnpa_function_code function_code,
                                  zdnn_status expected_status) {

  // Create ztensor with input1_values
  zdnn_ztensor *input1_ztensor = alloc_ztensor_with_values(
      shape, layout, type, NO_CONCAT, false, input1_values);

  // Create ztensor with input2_values
  zdnn_ztensor *input2_ztensor = alloc_ztensor_with_values(
      shape, layout, type, NO_CONCAT, false, input2_values);

  // Create output ztensor initialized to 0's
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      shape, layout, type, NO_CONCAT, true, ZERO_ARRAY);

#ifdef TEST_AIU // Test requires AIU

  // calculate number of values in each tensor buffer for helper function
  uint64_t num_elements = get_num_elements(output_ztensor, ELEMENTS_PRE);

  // Values in ZDNN_NHWC order
  float expected_values[num_elements];

  char api_method[AIU_METHOD_STR_LENGTH];
  zdnn_status status = GENERAL_TESTCASE_FAILURE;

  // Use public zDNN method to make NNPA call to AIU
  // then fill expected_values array with calculated expected values using
  // helper function if we expect to succeed.  Otherwise don't bother.
#define CASE(func_code, func_name)                                             \
  case func_code:                                                              \
    strcpy(api_method, "zdnn_" #func_name);                                    \
    status = zdnn_##func_name(input1_ztensor, input2_ztensor, output_ztensor); \
    elwise_##func_name(input1_values, input2_values, expected_values,          \
                       num_elements, type);                                    \
    break;

  switch (function_code) {
    CASE(NNPA_MAX, max)
    CASE(NNPA_MIN, min)
    CASE(NNPA_ADD, add)
    CASE(NNPA_SUB, sub)
    CASE(NNPA_MUL, mul)
    CASE(NNPA_DIV, div)
  default:
    TEST_FAIL_MESSAGE_FORMATTED("unsupported function_code: %d", function_code);
    break;
  }
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to %s() to returned status %08x but expected %08x", api_method,
      status, expected_status);

  // Only check expected values if expected status is ZDNN_OK
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }
#endif

  // Cleanup test tensor buffers
  free_ztensor_buffers(3, input1_ztensor, input2_ztensor, output_ztensor);
}

/**
 * Helper function to run end to end elementwise tests that only have
 * two input tensors.  This version tests all supported data types by
 * looping over the supported data types (FP32, Bfloat and FP16)
 * calling test_elwise_ap_2_input_adv for each.
 */
void test_elwise_api_2_inputs(uint32_t *shape, zdnn_data_layouts layout,
                              float *input1_values, float *input2_values,
                              nnpa_function_code function_code,
                              zdnn_status expected_status) {
  test_elwise_api_2_inputs_adv(shape, layout, test_datatype, input1_values,
                               input2_values, function_code, expected_status);
}
