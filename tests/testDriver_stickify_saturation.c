// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024, 2024
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "testsupport.h"

void setUp(void) {}

void tearDown(void) {}

float fp32_saturation_value(float value) {
  float tmp = (value > DLF16_MAX_AS_FP32) ? DLF16_MAX_AS_FP32 : value;
  return (tmp < DLF16_MIN_AS_FP32) ? DLF16_MIN_AS_FP32 : tmp;
}

uint16_t bfloat_saturation_value(uint16_t value) {

  typedef struct float_as_uint16s {
    uint16_t left;
    uint16_t right;
  } float_as_uint16s;

  union {
    float f;
    float_as_uint16s fau;
  } tmp;

  tmp.fau.left = value;
  tmp.fau.right = 0;
  if (tmp.f > DLF16_MAX_AS_FP32) {
    return DLF16_MAX_AS_BFLOAT;
  } else if (tmp.f < DLF16_MIN_AS_FP32) {
    return DLF16_MIN_AS_BFLOAT;
  } else {
    return value;
  }
}

void test_stickify_with_saturation_dims(zdnn_data_layouts layout,
                                        zdnn_data_types type, void *value,
                                        uint32_t dim4, uint32_t dim3,
                                        uint32_t dim2, uint32_t dim1,
                                        zdnn_status expected_status) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status stick_status, unstick_status;

  switch (layout) {
  case (ZDNN_1D):
    zdnn_init_pre_transformed_desc(layout, type, &pre_tfrmd_desc, dim1);
    break;
  case (ZDNN_2D):
  case (ZDNN_2DS):
    zdnn_init_pre_transformed_desc(layout, type, &pre_tfrmd_desc, dim2, dim1);
    break;
  case (ZDNN_3D):
  case (ZDNN_3DS):
    zdnn_init_pre_transformed_desc(layout, type, &pre_tfrmd_desc, dim3, dim2,
                                   dim1);
    break;
  case (ZDNN_ZRH):
  case (ZDNN_FICO):
  case (ZDNN_BIDIR_ZRH):
  case (ZDNN_BIDIR_FICO):
    zdnn_init_pre_transformed_desc(ZDNN_NHWC, type, &pre_tfrmd_desc, dim4, dim3,
                                   dim2, dim1);
    break;
  default:
    zdnn_init_pre_transformed_desc(layout, type, &pre_tfrmd_desc, dim4, dim3,
                                   dim2, dim1);
  }

  zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);

  zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);

  uint64_t num_elements = get_num_elements(&ztensor, ELEMENTS_AIU);
  uint64_t element_size = (FP32) ? 4 : 2; // FP32 = 4 bytes, BFLOAT = 2 bytes

  void *in_data = malloc(num_elements * element_size);
  void *saturated_data = malloc(num_elements * element_size);
  void *out_data = malloc(num_elements * element_size);

  // Check if any allocations failed.
  if (in_data == NULL || saturated_data == NULL || out_data == NULL) {
    free(in_data);
    free(saturated_data);
    free(out_data);
    TEST_FAIL_MESSAGE("Unable to allocate required data");
  }

  for (uint64_t i = 0; i < num_elements; i++) {
    if (type == FP32) {
      ((float *)(in_data))[i] = *((float *)(value));
      ((float *)(saturated_data))[i] =
          fp32_saturation_value(*((float *)(value)));
    } else {
      ((uint16_t *)(in_data))[i] = *((uint16_t *)(value));
      ((uint16_t *)(saturated_data))[i] =
          (type == BFLOAT) ? bfloat_saturation_value(*((uint16_t *)(value)))
                           : *((uint16_t *)(value));
    }
  }

  stick_status = zdnn_transform_ztensor_with_saturation(&ztensor, in_data);

  // Unable to unstickify HWCK. As only 4 elements are passed. Override format
  // and layouts to satisfy unstickifying.
  if (layout == ZDNN_HWCK) {
    ztensor.transformed_desc->format = ZDNN_FORMAT_4DFEATURE;
    ztensor.transformed_desc->layout = ZDNN_NHWC;
    ztensor.pre_transformed_desc->layout = ZDNN_NHWC;
  }

  unstick_status = zdnn_transform_origtensor(&ztensor, out_data);

  bool values_match = true;

  for (uint64_t i = 0; i < num_elements; i++) {
    if (type == FP32) {
      if ((((float *)(out_data))[i]) != (((float *)(saturated_data))[i])) {
        values_match = false;
        printf("Index: %" PRId64 " fp32 value: %f not saturated properly. "
               "Expected %f, input was: %f\n",
               i, ((float *)(out_data))[i], ((float *)(saturated_data))[i],
               ((float *)(in_data))[i]);
      }
    } else {
      if ((((uint16_t *)(out_data))[i]) !=
          (((uint16_t *)(saturated_data))[i])) {
        values_match = false;
        printf("Index: %" PRId64 " bfloat value: %hu not saturated properly. "
               "Expected %hu, input was: %hu\n",
               i, ((uint16_t *)(out_data))[i],
               ((uint16_t *)(saturated_data))[i], ((uint16_t *)(in_data))[i]);
      }
    }
  }

  free(in_data);
  free(saturated_data);
  free(out_data);
  zdnn_free_ztensor_buffer(&ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(stick_status == expected_status,
                                "zdnn_transform_ztensor_with_saturation() "
                                "failed (status = %08x, expects = %08x)",
                                stick_status, expected_status);
  TEST_ASSERT_MESSAGE_FORMATTED(unstick_status == expected_status,
                                "zdnn_transform_origtensor() "
                                "failed (status = %08x, expects = %08x)",
                                stick_status, expected_status);
  TEST_ASSERT_MESSAGE(values_match == true,
                      "values aren't saturated properly.");
}

void test_stickify_with_saturation_float(zdnn_data_layouts layout, float value,
                                         zdnn_status expected_status) {

  test_stickify_with_saturation_dims(layout, FP32, (void *)&value, 1, 1, 1, 4,
                                     expected_status);
}

void test_stickify_with_saturation_bfloat(zdnn_data_layouts layout, float value,
                                          zdnn_status expected_status) {

  uint16_t bfloat_value = cnvt_1_fp32_to_bfloat(value);

  test_stickify_with_saturation_dims(layout, BFLOAT, (void *)&bfloat_value, 1,
                                     1, 1, 4, expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; drive all acceptable layouts for FP32
// Expect ZDNN_OK
void saturation_basic() {
  zdnn_data_layouts layouts[] = {ZDNN_1D,  ZDNN_2D, ZDNN_2DS, ZDNN_3D,
                                 ZDNN_3DS, ZDNN_4D, ZDNN_NHWC};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_float(layouts[i], 100, ZDNN_OK);
  }
}

void saturation_basic_small() {
  zdnn_data_layouts layouts[] = {ZDNN_1D,  ZDNN_2D, ZDNN_2DS, ZDNN_3D,
                                 ZDNN_3DS, ZDNN_4D, ZDNN_NHWC};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_float(layouts[i], 0.5, ZDNN_OK);
  }
}

void saturation_basic_hwck() {
  zdnn_data_layouts layouts[] = {ZDNN_HWCK};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_float(layouts[i], 100, ZDNN_OK);
  }
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max DLFLOAT value.
// Expect ZDNN_OK
void saturation_basic_match_max() {
  test_stickify_with_saturation_float(ZDNN_NHWC, DLF16_MAX_AS_FP32, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max FP32 value.
// Expect ZDNN_OK
void saturation_basic_exceed_max() {
  test_stickify_with_saturation_float(ZDNN_NHWC, FLT_MAX, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min DLFLOAT value.
// Expect ZDNN_OK
void saturation_basic_match_min() {
  test_stickify_with_saturation_float(ZDNN_NHWC, DLF16_MIN_AS_FP32, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min FP32 value.
// Expect ZDNN_OK
void saturation_basic_exceed_min() {
  test_stickify_with_saturation_float(ZDNN_NHWC, -FLT_MAX, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; drive all acceptable layouts for bfloat
// Expect ZDNN_OK
void saturation_basic_bfloat() {
  zdnn_data_layouts layouts[] = {ZDNN_1D,  ZDNN_2D, ZDNN_2DS, ZDNN_3D,
                                 ZDNN_3DS, ZDNN_4D, ZDNN_NHWC};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_bfloat(layouts[i], 100, ZDNN_OK);
  }
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max DLFLOAT value.
// Expect ZDNN_OK
void saturation_basic_match_max_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, DLF16_MAX_AS_FP32, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max bfloat value.
// Expect ZDNN_OK
void saturation_basic_exceed_max_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, FLT_MAX, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min dlfloat value.
// Expect ZDNN_OK
void saturation_basic_match_min_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, DLF16_MIN_AS_FP32, ZDNN_OK);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min bfloat value.
// Expect ZDNN_OK
void saturation_basic_exceed_min_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, -FLT_MAX, ZDNN_OK);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(saturation_basic);
  RUN_TEST(saturation_basic_small);
  RUN_TEST(saturation_basic_hwck);
  RUN_TEST(saturation_basic_match_max);
  RUN_TEST(saturation_basic_exceed_max);
  RUN_TEST(saturation_basic_match_min);
  RUN_TEST(saturation_basic_exceed_min);
  RUN_TEST(saturation_basic_bfloat);
  RUN_TEST(saturation_basic_match_max_bfloat);
  RUN_TEST(saturation_basic_exceed_max_bfloat);
  RUN_TEST(saturation_basic_match_min_bfloat);
  RUN_TEST(saturation_basic_exceed_min_bfloat);

  return UNITY_END();
}
