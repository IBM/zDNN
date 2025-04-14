// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024, 2025
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "testsupport.h"

#define INF_FP16_POS 0X7C00
#define INF_FP16_NEG 0xFC00
#define NAN_FP16_POS 0x7FFF
#define NAN_FP16_NEG 0xFFFF

#define INF_FP32_POS                                                           \
  (((union {                                                                   \
     int i;                                                                    \
     float f;                                                                  \
   }){0x7F800000})                                                             \
       .f)
#define INF_FP32_NEG                                                           \
  (((union {                                                                   \
     int i;                                                                    \
     float f;                                                                  \
   }){0xFF800000})                                                             \
       .f)
#define NAN_FP32_POS                                                           \
  (((union {                                                                   \
     int i;                                                                    \
     float f;                                                                  \
   }){0x7FFFFFFF})                                                             \
       .f)
#define NAN_FP32_NEG                                                           \
  (((union {                                                                   \
     int i;                                                                    \
     float f;                                                                  \
   }){0xFFFFFFFF})                                                             \
       .f)

zdnn_status default_unstick_expected_status = ZDNN_OK;
zdnn_status default_saturate_expected_status = ZDNN_OK;

void setUp(void) {}

void tearDown(void) {}

float fp32_saturation_value(float value) {
  // Expected Saturation value for -NAN,NAN,-INF,INF should be NAN
  if (isnan(value) || isinf(value))
    return NAN;

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
                                        zdnn_status saturation_expected_status,
                                        zdnn_status unstick_expected_status) {
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

  bool values_match = true;

  unstick_status = zdnn_transform_origtensor(&ztensor, out_data);
  // no need to check output if is_transformed set to false
  if (ztensor.is_transformed == true) {
    for (uint64_t i = 0; i < num_elements; i++) {
      if (type == FP32) {
        // check if out values and saturated data are not equal but only if the
        // values are BOTH not NAN.
        if ((((float *)(out_data))[i]) != (((float *)(saturated_data))[i]) &&
            !isnan((((float *)(out_data))[i])) &&
            !isnan((((float *)(saturated_data))[i]))) {
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
  }

  free(in_data);
  free(saturated_data);
  free(out_data);
  zdnn_free_ztensor_buffer(&ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(stick_status == saturation_expected_status,
                                "zdnn_transform_ztensor_with_saturation() "
                                "failed (status = %08x, expects = %08x)",
                                stick_status, saturation_expected_status);

  TEST_ASSERT_MESSAGE_FORMATTED(unstick_status == unstick_expected_status,
                                "zdnn_transform_origtensor() "
                                "failed (status = %08x, expects = %08x)",
                                unstick_status, unstick_expected_status);

  // When stick status is ZDNN_CONVERT_FAILURE (fp16 nan/inf) need not assert as
  // ztensor.is_transformed is false
  if (stick_status != ZDNN_CONVERT_FAILURE) {
    TEST_ASSERT_MESSAGE(values_match == true,
                        "values aren't saturated properly.");
  }
}

void test_stickify_with_saturation_float(zdnn_data_layouts layout, float value,
                                         zdnn_status saturation_expected_status,
                                         zdnn_status unstick_expected_status) {

  test_stickify_with_saturation_dims(layout, FP32, (void *)&value, 1, 1, 1, 4,
                                     saturation_expected_status,
                                     unstick_expected_status);
}

void test_stickify_with_saturation_fp16(zdnn_data_layouts layout,
                                        uint16_t value,
                                        zdnn_status saturation_expected_status,
                                        zdnn_status unstick_expected_status) {
  test_stickify_with_saturation_dims(layout, FP16, (void *)&value, 1, 1, 1, 4,
                                     saturation_expected_status,
                                     unstick_expected_status);
}

void test_stickify_with_saturation_fp32(zdnn_data_layouts layout, float value,
                                        uint32_t dim4, uint32_t dim3,
                                        uint32_t dim2, uint32_t dim1,
                                        zdnn_status saturation_expected_status,
                                        zdnn_status unstick_expected_status) {

  test_stickify_with_saturation_dims(layout, FP32, (void *)&value, dim4, dim3,
                                     dim2, dim1, saturation_expected_status,
                                     unstick_expected_status);
}

void test_stickify_with_saturation_bfloat(
    zdnn_data_layouts layout, float value,
    zdnn_status saturation_expected_status,
    zdnn_status unstick_expected_status) {

  uint16_t bfloat_value = cnvt_1_fp32_to_bfloat(value);

  test_stickify_with_saturation_dims(layout, BFLOAT, (void *)&bfloat_value, 1,
                                     1, 1, 4, saturation_expected_status,
                                     unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; drive all acceptable layouts for FP32
// Expect ZDNN_OK
void saturation_basic() {
  zdnn_data_layouts layouts[] = {ZDNN_1D,  ZDNN_2D, ZDNN_2DS, ZDNN_3D,
                                 ZDNN_3DS, ZDNN_4D, ZDNN_NHWC};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_float(layouts[i], 100,
                                        default_saturate_expected_status,
                                        default_unstick_expected_status);
  }
}

void saturation_basic_small() {
  zdnn_data_layouts layouts[] = {ZDNN_1D,  ZDNN_2D, ZDNN_2DS, ZDNN_3D,
                                 ZDNN_3DS, ZDNN_4D, ZDNN_NHWC};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_float(layouts[i], 0.5,
                                        default_saturate_expected_status,
                                        default_unstick_expected_status);
  }
}

void saturation_basic_hwck() {
  zdnn_data_layouts layouts[] = {ZDNN_HWCK};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_float(layouts[i], 100,
                                        default_saturate_expected_status,
                                        default_unstick_expected_status);
  }
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max DLFLOAT value.
// Expect ZDNN_OK
void saturation_basic_match_max() {
  test_stickify_with_saturation_float(ZDNN_NHWC, DLF16_MAX_AS_FP32,
                                      default_saturate_expected_status,
                                      default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max FP32 value.
// Expect ZDNN_OK
void saturation_basic_exceed_max() {
  test_stickify_with_saturation_float(ZDNN_NHWC, FLT_MAX,
                                      default_saturate_expected_status,
                                      default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min DLFLOAT value.
// Expect ZDNN_OK
void saturation_basic_match_min() {
  test_stickify_with_saturation_float(ZDNN_NHWC, DLF16_MIN_AS_FP32,
                                      default_saturate_expected_status,
                                      default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min FP32 value.
// Expect ZDNN_OK
void saturation_basic_exceed_min() {
  test_stickify_with_saturation_float(ZDNN_NHWC, -FLT_MAX,
                                      default_saturate_expected_status,
                                      default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; drive all acceptable layouts for bfloat
// Expect ZDNN_OK
void saturation_basic_bfloat() {
  zdnn_data_layouts layouts[] = {ZDNN_1D,  ZDNN_2D, ZDNN_2DS, ZDNN_3D,
                                 ZDNN_3DS, ZDNN_4D, ZDNN_NHWC};

  for (int i = 0; i < (sizeof(layouts) / sizeof(layouts[0])); i++) {
    test_stickify_with_saturation_bfloat(layouts[i], 100,
                                         default_saturate_expected_status,
                                         default_unstick_expected_status);
  }
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max DLFLOAT value.
// Expect ZDNN_OK
void saturation_basic_match_max_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, DLF16_MAX_AS_FP32,
                                       default_saturate_expected_status,
                                       default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive max bfloat value.
// Expect ZDNN_OK
void saturation_basic_exceed_max_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, FLT_MAX,
                                       default_saturate_expected_status,
                                       default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min dlfloat value.
// Expect ZDNN_OK
void saturation_basic_match_min_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, DLF16_MIN_AS_FP32,
                                       default_saturate_expected_status,
                                       default_unstick_expected_status);
}

// Basic zdnn_transform_ztensor_with_saturation test.
// No errors; Drive min bfloat value.
// Expect ZDNN_OK
void saturation_basic_exceed_min_bfloat() {
  test_stickify_with_saturation_bfloat(ZDNN_NHWC, -FLT_MAX,
                                       default_saturate_expected_status,
                                       default_unstick_expected_status);
}

// FP32 NAN
void saturation_basic_fp32_nan() {

  // stickification status is always the same for hw/sw
  zdnn_status saturation_expected_status = ZDNN_ELEMENT_RANGE_VIOLATION;

  // Test set #1
  // Small tensor to stay under STICK_SW_THRESHOLD to exercise correct unstick
  // status
  uint32_t dim4 = 1;
  uint32_t dim3 = 1;
  uint32_t dim2 = 1;
  uint32_t dim1 = 4;

  // These following tests will always stay in SW (e.g., not to the AIU) as the
  // product of pre-transformed dim[1..3] product will be <
  // STICK_SW_THRESHOLD so we expect ZDNN_CONVERT_FAILURE for unstick
  // see: n_stride_meets_hardware_limit
  zdnn_status expected_unstick_status = ZDNN_CONVERT_FAILURE;

  test_stickify_with_saturation_fp32(ZDNN_NHWC, INF_FP32_POS, dim4, dim3, dim2,
                                     dim1, ZDNN_ELEMENT_RANGE_VIOLATION,
                                     expected_unstick_status);

  test_stickify_with_saturation_fp32(ZDNN_NHWC, INF_FP32_NEG, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);

  test_stickify_with_saturation_fp32(ZDNN_NHWC, NAN_FP32_NEG, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);

  test_stickify_with_saturation_fp32(ZDNN_NHWC, NAN_FP32_POS, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);

  // Test set #2
  // Larger tensor to go over the STICK_SW_THRESHOLD to exercise correct unstick
  // status. When NNPA_TRANSFORM == true the (un)stickification is done on HW so
  // expect ZDNN_ELEMENT_RANGE_VIOLATION for unstick. When NNPA_TRANSFORM
  // != true expect ZDNN_CONVERT_FAILURE for unstick as this is done in SW

  dim4 = 1;
  dim3 = 1;
  dim2 = 1;
  dim1 = 4096;

  if (zdnn_is_nnpa_function_installed(1, NNPA_TRANSFORM) == true) {
    expected_unstick_status = ZDNN_ELEMENT_RANGE_VIOLATION;
  } else {
    expected_unstick_status = ZDNN_CONVERT_FAILURE;
  }

  test_stickify_with_saturation_fp32(ZDNN_NHWC, INF_FP32_POS, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);

  test_stickify_with_saturation_fp32(ZDNN_NHWC, INF_FP32_NEG, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);

  test_stickify_with_saturation_fp32(ZDNN_NHWC, NAN_FP32_NEG, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);

  test_stickify_with_saturation_fp32(ZDNN_NHWC, NAN_FP32_POS, dim4, dim3, dim2,
                                     dim1, saturation_expected_status,
                                     expected_unstick_status);
}

// FP16 NAN
// Expect: ZDNN_CONVERT_FAILURE
void saturation_basic_fp16_nan() {
  test_stickify_with_saturation_fp16(ZDNN_NHWC, INF_FP16_NEG,
                                     ZDNN_CONVERT_FAILURE, ZDNN_INVALID_STATE);
  test_stickify_with_saturation_fp16(ZDNN_NHWC, INF_FP16_POS,
                                     ZDNN_CONVERT_FAILURE, ZDNN_INVALID_STATE);
  test_stickify_with_saturation_fp16(ZDNN_NHWC, NAN_FP16_POS,
                                     ZDNN_CONVERT_FAILURE, ZDNN_INVALID_STATE);
  test_stickify_with_saturation_fp16(ZDNN_NHWC, NAN_FP16_NEG,
                                     ZDNN_CONVERT_FAILURE, ZDNN_INVALID_STATE);
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
  RUN_TEST(saturation_basic_fp32_nan);
  RUN_TEST(saturation_basic_fp16_nan);

  return UNITY_END();
}
