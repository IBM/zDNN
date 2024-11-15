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

#include "common_quantization.h"
#include "convert.h"
#include "testsupport.h"

#include <stdlib.h>
#include <string.h>

/// Creates a ztensor with the provided values. Values are converted to the
/// specified type. The resulting ztensor is transformed and ready for use in
/// zDNN operations.
///
/// \note This method does not check that the size of values matches expected
/// number of elements.
///
/// Example usage:
/// Setup input tensor
/// \code
///  ztensor *zt = alloc_quantized_ztensor_with_values(
///      shape, pre_tfrmd_layout, INT8, QUANTIZED_INT8, values, scale, offset);
/// \endcode
/// Setup Output tensor
/// \code
///  ztensor *zt = alloc_quantized_ztensor_with_values(
///      shape, pre_tfrmd_layout, ZDNN_DLFLOAT16, QUANTIZED_DLFLOAT16, NULL,
///      scale, offset);
/// \endcode
///
/// \param[in] shape array of dimensions
/// \param[in] pre_tfrmd_layout pre-transformed data layout
/// \param[in] type data type
/// \param[in] transform_type quantized data type
/// \param[in] values_data float data
/// \param[in] scale quantization scale
/// \param[in] offset quantization offset (zero point)
///
/// \return zdnn_ztensor* Pointer to a malloc'd ztensor with transformed data
///
zdnn_ztensor *alloc_quantized_ztensor_with_values(
    uint32_t *shape, zdnn_data_layouts pre_tfrmd_layout, zdnn_data_types type,
    zdnn_quantized_transform_types transform_type, const float *values_data,
    const float scale, const float offset) {
  // Create the pretransformed description
  zdnn_tensor_desc *pre_tfrmd_desc =
      (zdnn_tensor_desc *)malloc(sizeof(zdnn_tensor_desc));

  switch (pre_tfrmd_layout) {
  case (ZDNN_1D):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0]);
    break;
  case (ZDNN_2D):
  case (ZDNN_2DS):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0], shape[1]);
    break;
  case (ZDNN_3DS):
    zdnn_init_pre_transformed_desc(pre_tfrmd_layout, type, pre_tfrmd_desc,
                                   shape[0], shape[1], shape[2]);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "pre_tfrmd_layout. Could you teach me?",
        get_data_layout_str(pre_tfrmd_layout));
    break;
  }

  // Create the transformed description
  zdnn_tensor_desc *tfrmd_desc =
      (zdnn_tensor_desc *)malloc(sizeof(zdnn_tensor_desc));

  zdnn_status status = zdnn_generate_quantized_transformed_desc(
      pre_tfrmd_desc, transform_type, tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc failed (status = %08x)", status);

  // Create the ztensor with malloc'd buffer pointer
  zdnn_ztensor *ztensor = (zdnn_ztensor *)malloc(sizeof(zdnn_ztensor));

  status = zdnn_init_quantized_ztensor_with_malloc(pre_tfrmd_desc, tfrmd_desc,
                                                   scale, offset, ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_init_ztensor_with_malloc failed (status = %08x)",
      status);

  if (transform_type == QUANTIZED_INT8) {
    status = zdnn_transform_quantized_ztensor(ztensor, false, INT8_MIN,
                                              INT8_MAX, values_data);
  } else if (transform_type == QUANTIZED_WEIGHTS_INT8) {
    size_t num_elements =
        tfrmd_desc->dim4 * tfrmd_desc->dim2 * tfrmd_desc->dim1;

    int8_t quant_data[num_elements];
    for (size_t i = 0; i < num_elements; ++i) {
      quant_data[i] = QUANTIZE(values_data[i], scale, offset);
    }

    status = zdnn_transform_quantized_ztensor(ztensor, false, INT8_MIN,
                                              INT8_MAX, quant_data);
  }

  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_transform_quantized_ztensor failed with status %08x \"%s\"", status,
      zdnn_get_status_message(status));

  return ztensor;
}

/// Asserts each value in the stickified ztensor are within 1.0 of the given
/// expected float values.
///
/// \note This method does not check that the size of values array matches the
/// number of elements. If there's not enough expected values, the test will
/// likely fail when garbage data is pulled in as the expected value.
///
/// Example usage:
/// \code
///  assert_quantized_ztensor_values(&ztensor, false, values);
/// \endcode
///
/// \param[in] ztensor pointer to zdnn_ztensor with actual values
/// \param[in] repeat_first_expected_value if true, all ztensor values will be
///                                        compared to values[0]
/// \param[in] expected_vals array of expected quantized values
///
/// \return None (assert fails if any actual value not within expected range)
///
void assert_quantized_ztensor_values(zdnn_ztensor *ztensor,
                                     bool repeat_first_expected_value,
                                     const float *expected_vals) {
  zdnn_status status;
  zdnn_tensor_desc *pre_tfrmd_desc = ztensor->pre_transformed_desc;

  uint64_t num_elements = 0;
  switch (ztensor->transformed_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_4DS:
  case ZDNN_NHWC:
    num_elements = get_num_elements(ztensor, ELEMENTS_PRE);
    break;
  case ZDNN_FICO:
  case ZDNN_ZRH:
    TEST_FAIL_MESSAGE_FORMATTED(
        "does not support %s layout as we don't support unstickifying "
        "concatenated ztensors.",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "layout. Could you teach me?",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  }

  // Malloc error_message as it will be large if num_elements is large.
  uint64_t big_error_message_size =
      (uint64_t)sizeof(char) * ERROR_MESSAGE_STR_LENGTH * num_elements;
  char *error_msg = malloc(big_error_message_size);

  float *actual_vals;

  // Get unstickified data from ztensor to actual_vals[]
  actual_vals = malloc(num_elements * get_data_type_size(pre_tfrmd_desc->type));
  status = zdnn_transform_origtensor(ztensor, actual_vals);
  snprintf(error_msg, big_error_message_size,
           "zdnn_transform_origtensor failed (status = %08x)", status);
  TEST_ASSERT_MESSAGE(status == ZDNN_OK, error_msg);

  // Assert ztentor's values (converted back to floats) match does not match
  // too many times
  bool all_pass = true;
  // Loop appends to error_msg so reset it first
  error_msg[0] = '\0';

  char *error_fmt = "Element %" PRIu64 " == %f expecting %f";
  char *error_fmt2 = " <==== FAILED (diff beyond 1.0)";

  // Compared the actual and expected values
  for (uint64_t i = 0; i < num_elements; i++) {

    bool is_almost_equal = false;

    switch (pre_tfrmd_desc->type) {
    case FP32: {
      float actual = actual_vals[i];
      float expected = expected_vals[i];

      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt, i, actual,
               expected);

      LOG_DEBUG(error_fmt, i, actual, expected);

      is_almost_equal = fabs(fabs(actual) - fabs(expected)) <= 1.f;
      break;
    }
    default:
      // NOTE: Along with undefined types, DLFLOAT types will also come down
      // this path. DLFLOATS are a stickified types which are not valid types
      // for the pre_tfrmd_desc (ie prestickifed description).
      snprintf(error_msg, big_error_message_size, "unsupported type: %d\n",
               pre_tfrmd_desc->type);
      TEST_FAIL_MESSAGE(error_msg);
      break;
    }

    if (!is_almost_equal) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt2);
      all_pass = false;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");
  }

  // Assert that all passed and clean up temp data
  TEST_ASSERT_MESSAGE(all_pass, error_msg);

  free(actual_vals);
  free(error_msg);
}

/// Asserts each value in the stickified ztensor are within 1.0 of the given
/// expected float values.
///
/// \note This method does not check that the size of values array matches the
/// number of elements. If there's not enough expected values, the test will
/// likely fail when garbage data is pulled in as the expected value.
///
/// Example usage:
/// \code
///  assert_dequantized_ztensor_values(&ztensor, false, values);
/// \endcode
///
/// \param[in] ztensor pointer to zdnn_ztensor with actual values
/// \param[in] repeat_first_expected_value if true, all ztensor values will be
///                                        compared to values[0]
/// \param[in] expected_vals array of expected quantized values
///
/// \return None (assert fails if any actual value not within expected range)
///
void assert_dequantized_ztensor_values(zdnn_ztensor *ztensor,
                                       bool repeat_first_expected_value,
                                       const float *expected_vals) {
  zdnn_status status;
  zdnn_tensor_desc *pre_tfrmd_desc = ztensor->pre_transformed_desc;

  uint64_t num_elements = 0;
  switch (ztensor->transformed_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_4DS:
  case ZDNN_NHWC:
    num_elements = get_num_elements(ztensor, ELEMENTS_PRE);
    break;
  case ZDNN_FICO:
  case ZDNN_ZRH:
    TEST_FAIL_MESSAGE_FORMATTED(
        "does not support %s layout as we don't support unstickifying "
        "concatenated ztensors.",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "layout. Could you teach me?",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  }

  // Malloc error_message as it will be large if num_elements is large.
  uint64_t big_error_message_size =
      (uint64_t)sizeof(char) * ERROR_MESSAGE_STR_LENGTH * num_elements;
  char *error_msg = malloc(big_error_message_size);

  float *actual_vals;

  // Get unstickified data from ztensor to actual_vals[]
  actual_vals = malloc(num_elements * get_data_type_size(pre_tfrmd_desc->type));
  status = zdnn_transform_origtensor(ztensor, actual_vals);
  snprintf(error_msg, big_error_message_size,
           "zdnn_transform_origtensor failed (status = %08x)", status);
  TEST_ASSERT_MESSAGE(status == ZDNN_OK, error_msg);

  // Assert ztentor's values (converted back to floats) match does not match
  // too many times
  bool all_pass = true;
  // Loop appends to error_msg so reset it first
  error_msg[0] = '\0';

  char *error_fmt = "Element %" PRIu64 " == %f expecting %f";
  char *error_fmt2 = " <==== FAILED (diff beyond 1.0)";

  // Compared the actual and expected values
  for (uint64_t i = 0; i < num_elements; i++) {

    bool is_almost_equal = false;

    switch (pre_tfrmd_desc->type) {
    case FP32: {
      // expected values are quantized, so we need to quantized the dequantized
      // actual values before comparison.
      float actual =
          QUANTIZE(actual_vals[i], (1.f / ztensor->rec_scale), ztensor->offset);
      float expected = expected_vals[i];

      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt, i, actual,
               expected);

      LOG_DEBUG(error_fmt, i, actual, expected);

      is_almost_equal = fabs(fabs(actual) - fabs(expected)) <= 1.f;
      break;
    }
    default:
      // NOTE: Along with undefined types, DLFLOAT types will also come down
      // this path. DLFLOATS are a stickified types which are not valid types
      // for the pre_tfrmd_desc (ie prestickifed description).
      snprintf(error_msg, big_error_message_size, "unsupported type: %d\n",
               pre_tfrmd_desc->type);
      TEST_FAIL_MESSAGE(error_msg);
      break;
    }

    if (!is_almost_equal) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt2);
      all_pass = false;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");
  }

  // Assert that all passed and clean up temp data
  TEST_ASSERT_MESSAGE(all_pass, error_msg);

  free(actual_vals);
  free(error_msg);
}

/// Asserts that no more than 3% of the values are not equal to expected values.
///
/// \note This method does not check that the size of values array matches the
/// number of elements. If there's not enough expected values, the test will
/// likely fail when garbage data is pulled in as the expected value.
///
/// Example usage:
/// \code
///  assert_quantized_ztensor_values(&ztensor, false, values);
/// \endcode
///
/// \param[in] ztensor pointer to zdnn_ztensor with actual values
/// \param[in] repeat_first_expected_value if true, all ztensor values will be
///                                        compared to values[0]
/// \param[in] expected_vals array of expected values
///
/// \return None (assert fails if any actual value not within expected range)
///
void assert_quantized_ztensor_compare_values(zdnn_ztensor *ztensor,
                                             bool repeat_first_expected_value,
                                             const float *expected_vals) {
  zdnn_status status;
  zdnn_tensor_desc *pre_tfrmd_desc = ztensor->pre_transformed_desc;

  uint64_t num_elements = 0;
  switch (ztensor->transformed_desc->layout) {
  case ZDNN_1D:
  case ZDNN_2D:
  case ZDNN_2DS:
  case ZDNN_3D:
  case ZDNN_3DS:
  case ZDNN_4D:
  case ZDNN_4DS:
  case ZDNN_NHWC:
    num_elements = get_num_elements(ztensor, ELEMENTS_PRE);
    break;
  case ZDNN_FICO:
  case ZDNN_ZRH:
    TEST_FAIL_MESSAGE_FORMATTED(
        "does not support %s layout as we don't support unstickifying "
        "concatenated ztensors.",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "layout. Could you teach me?",
        get_data_layout_str(ztensor->transformed_desc->layout));
    break;
  }

  // Malloc error_message as it will be large if num_elements is large.
  uint64_t big_error_message_size =
      (uint64_t)sizeof(char) * ERROR_MESSAGE_STR_LENGTH * num_elements;
  char *error_msg = malloc(big_error_message_size);

  float *actual_vals;

  // Get unstickified data from ztensor to actual_vals[]
  actual_vals = malloc(num_elements * get_data_type_size(pre_tfrmd_desc->type));
  status = zdnn_transform_origtensor(ztensor, actual_vals);
  snprintf(error_msg, big_error_message_size,
           "zdnn_transform_origtensor failed (status = %08x)", status);
  TEST_ASSERT_MESSAGE(status == ZDNN_OK, error_msg);

  // Assert ztentor's values (converted back to floats) match does not match
  // too many times
  uint64_t num_mismatch = 0;
  // Loop appends to error_msg so reset it first
  error_msg[0] = '\0';

  char *error_fmt = "Element %" PRIu64 " == %f expecting %f";
  char *error_fmt2 = " <==== FAILED (diff beyond 0.0)";

  // Compared the actual and expected values
  for (uint64_t i = 0; i < num_elements; i++) {

    bool is_equal = false;

    switch (pre_tfrmd_desc->type) {
    case FP32: {
      float actual = actual_vals[i];
      float expected = expected_vals[i];

      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt, i, actual,
               expected);

      LOG_DEBUG(error_fmt, i, actual, expected);

      is_equal = actual == expected;
      break;
    }
    default:
      // NOTE: Along with undefined types, DLFLOAT types will also come down
      // this path. DLFLOATS are a stickified types which are not valid types
      // for the pre_tfrmd_desc (ie prestickifed description).
      snprintf(error_msg, big_error_message_size, "unsupported type: %d\n",
               pre_tfrmd_desc->type);
      TEST_FAIL_MESSAGE(error_msg);
      break;
    }

    if (!is_equal) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg), error_fmt2);
      num_mismatch++;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");
  }

  bool enough_pass = (float)num_mismatch / (float)num_elements < 0.01f;

  // Assert that all passed and clean up temp data
  TEST_ASSERT_MESSAGE(enough_pass, error_msg);

  free(actual_vals);
  free(error_msg);
}
