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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "testsupport.h"

void setUp(void) { VERIFY_HW_ENV; }

void tearDown(void) {}

/*
 * Non-error scenario general strategy:
 *
 * Create 2 tensors:
 *   tensor A: shape (x, y, z, a)
 *   tensor B: shape (i, j, k, b)
 *   which (x * y * z * a) == (i * j * k * b)
 *
 * Create raw data of (x * y * z * a) elements
 *
 * Stickify raw data to tensor A's buffer
 * zdnn_reshape_ztensor() from tensor A to tensor B
 *
 * Compare tensor B's buffer to the raw data, element by element, using
 * get_stick_offset() with respect to tensor B's shape
 *
 * Compare by values due to precision loss:
 *   A goes from FP16/FP32/BFLOAT -> DLFLOAT16, meaning
 *   B goes from FP16/FP32/BFLOAT -> DLFLOAT16 -> FP32 -> DLFLOAT16
 */

void test(zdnn_data_layouts src_layout, uint32_t src_dim4, uint32_t src_dim3,
          uint32_t src_dim2, uint32_t src_dim1, zdnn_data_layouts dest_layout,
          uint32_t dest_dim4, uint32_t dest_dim3, uint32_t dest_dim2,
          uint32_t dest_dim1, zdnn_status exp_status) {

  zdnn_status status;

  zdnn_tensor_desc src_pre_tfrmd_desc, dest_pre_tfrmd_desc, src_tfrmd_desc,
      dest_tfrmd_desc;
  zdnn_ztensor src_ztensor, dest_ztensor;

  zdnn_init_pre_transformed_desc(src_layout, test_datatype, &src_pre_tfrmd_desc,
                                 src_dim4, src_dim3, src_dim2, src_dim1);
  zdnn_init_pre_transformed_desc(dest_layout, test_datatype,
                                 &dest_pre_tfrmd_desc, dest_dim4, dest_dim3,
                                 dest_dim2, dest_dim1);

  zdnn_generate_transformed_desc(&src_pre_tfrmd_desc, &src_tfrmd_desc);
  zdnn_generate_transformed_desc(&dest_pre_tfrmd_desc, &dest_tfrmd_desc);

  status = zdnn_init_ztensor_with_malloc(&src_pre_tfrmd_desc, &src_tfrmd_desc,
                                         &src_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc() (src) failed, status = %08x", status);

  void *raw_data = create_and_fill_random_fp_data(&src_ztensor);

  status = zdnn_init_ztensor_with_malloc(&dest_pre_tfrmd_desc, &dest_tfrmd_desc,
                                         &dest_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc() (dest) failed, status = %08x", status);

  status = zdnn_transform_ztensor(&src_ztensor, raw_data);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK, "zdnn_transform_ztensor() failed, status = %08x",
      status);

  status = zdnn_reshape_ztensor(&src_ztensor, &dest_ztensor);

  if (exp_status == ZDNN_OK) {

    TEST_ASSERT_MESSAGE_FORMATTED(
        status == ZDNN_OK, "zdnn_reshape_ztensor() failed, status = %08x",
        status);

    TEST_ASSERT_MESSAGE(dest_ztensor.is_transformed == true,
                        "zdnn_reshape_ztensor() was successful but "
                        "did not set is_transformed properly for "
                        "destination ztensor");

    uint64_t raw_offset = 0;
    uint64_t cnt = 0;
    for (uint32_t i = 0; i < dest_dim4; i++) {
      for (uint32_t j = 0; j < dest_dim3; j++) {
        for (uint32_t k = 0; k < dest_dim2; k++) {
          for (uint32_t b = 0; b < dest_dim1; b++) {

            uint64_t dest_offset =
                get_stick_offset(i, j, k, b, &dest_tfrmd_desc);

            uint16_t raw_dlf16_val = 0; // this is the "expected" value
            uint16_t dest_dlf16_val =
                *(uint16_t *)((uintptr_t)dest_ztensor.buffer + dest_offset);

            // these 2 are for printf-ing only
            float raw_float_val = 0;
            float dest_float_val = cnvt_1_dlf16_to_fp32(dest_dlf16_val);

            if (test_datatype == BFLOAT) {
              raw_float_val = cnvt_1_bfloat_to_fp32(
                  *(uint16_t *)((uintptr_t)raw_data + raw_offset));
              raw_dlf16_val = cnvt_1_bfloat_to_dlf16(
                  *(uint16_t *)((uintptr_t)raw_data + raw_offset));
            } else if (test_datatype == FP16) {
              raw_float_val = cnvt_1_fp16_to_fp32(
                  *(uint16_t *)((uintptr_t)raw_data + raw_offset));
              raw_dlf16_val = cnvt_1_fp16_to_dlf16(
                  *(uint16_t *)((uintptr_t)raw_data + raw_offset));
            } else if (test_datatype == FP32) {
              raw_float_val = *(float *)((uintptr_t)raw_data + raw_offset);
              raw_dlf16_val = cnvt_1_fp32_to_dlf16(raw_float_val);
            }

            TEST_ASSERT_MESSAGE_FORMATTED(
                almost_equal_dlf16(dest_dlf16_val, raw_dlf16_val),
                "Incorrect value at element %" PRIu64
                ": Expected: %.6f, Found (offset %" PRIu64 "): %.6f",
                cnt, raw_float_val, dest_offset, dest_float_val);

            raw_offset += get_data_type_size(test_datatype);
            cnt++;
          }
        }
      }
    }
  } else {
    TEST_ASSERT_MESSAGE_FORMATTED(exp_status == status,
                                  "expected status = %08x, got status = %08x",
                                  exp_status, status);

    TEST_ASSERT_MESSAGE(
        dest_ztensor.is_transformed == false,
        "zdnn_reshape_ztensor() failed but set is_transformed improperly for "
        "destination ztensor.");
  }

  free(raw_data);
}

// N/H/W/C all the same (memcpy whole buffer)
void test_4x5x6x7_4x5x6x7() {
  test(ZDNN_NHWC, 4, 5, 6, 7, ZDNN_NHWC, 4, 5, 6, 7, ZDNN_OK);
}

// same C, different N/H/W (sticks memcpy)
void test_1x2x3x4_6x1x1x4() {
  test(ZDNN_NHWC, 1, 2, 3, 4, ZDNN_NHWC, 6, 1, 1, 4, ZDNN_OK);
}

// same C, different N/H/W, more elements (sticks memcpy)
void test_2x3x4x68_4x1x6x68() {
  test(ZDNN_NHWC, 2, 3, 4, 68, ZDNN_NHWC, 4, 1, 6, 68, ZDNN_OK);
}

// same C, different N/H/W, even more elements (sticks memcpy)
void test_4x3x40x70_8x20x3x70() {
  test(ZDNN_NHWC, 2, 3, 4, 68, ZDNN_NHWC, 4, 1, 6, 68, ZDNN_OK);
}

// N/H/W/C all different
void test_4x4x4x4_1x1x16x16() {
  test(ZDNN_NHWC, 4, 4, 4, 4, ZDNN_NHWC, 1, 1, 16, 16, ZDNN_OK);
}

void test_fail_total_elements_mismatch() {
  test(ZDNN_NHWC, 4, 4, 4, 4, ZDNN_NHWC, 1, 1, 16, 15, ZDNN_INVALID_SHAPE);
}

void test_fail_not_nhwc_nor_hwck() {
  zdnn_status status, exp_status = ZDNN_INVALID_LAYOUT;

  zdnn_tensor_desc src_pre_tfrmd_desc, dest_pre_tfrmd_desc, src_tfrmd_desc,
      dest_tfrmd_desc;
  zdnn_ztensor src_ztensor, dest_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &src_pre_tfrmd_desc, 4, 4, 4,
                                 4);
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP16, &dest_pre_tfrmd_desc, 4, 4, 4,
                                 4);

  zdnn_generate_transformed_desc(&src_pre_tfrmd_desc, &src_tfrmd_desc);
  zdnn_generate_transformed_desc(&dest_pre_tfrmd_desc, &dest_tfrmd_desc);

  zdnn_init_ztensor(&src_pre_tfrmd_desc, &src_tfrmd_desc, &src_ztensor);
  zdnn_init_ztensor(&dest_pre_tfrmd_desc, &dest_tfrmd_desc, &dest_ztensor);

  src_ztensor.is_transformed = true;
  // sabotage the layouts
  src_tfrmd_desc.layout = ZDNN_NCHW;
  dest_tfrmd_desc.layout = ZDNN_NCHW;

  status = zdnn_reshape_ztensor(&src_ztensor, &dest_ztensor);

  TEST_ASSERT_MESSAGE_FORMATTED(exp_status == status,
                                "expected status = %08x, got status = %08x",
                                exp_status, status);
}

void test_fail_not_same_layout() {
  test_datatype = FP16;
  test(ZDNN_NHWC, 4, 5, 6, 7, ZDNN_HWCK, 4, 5, 6, 7, ZDNN_INVALID_LAYOUT);
}

void test_fail_src_not_transformed() {
  zdnn_status status, exp_status = ZDNN_INVALID_STATE;
  test_datatype = FP16;

  zdnn_tensor_desc src_pre_tfrmd_desc, dest_pre_tfrmd_desc, src_tfrmd_desc,
      dest_tfrmd_desc;
  zdnn_ztensor src_ztensor, dest_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &src_pre_tfrmd_desc,
                                 4, 4, 4, 4);
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &dest_pre_tfrmd_desc,
                                 4, 4, 4, 4);

  zdnn_generate_transformed_desc(&src_pre_tfrmd_desc, &src_tfrmd_desc);
  zdnn_generate_transformed_desc(&dest_pre_tfrmd_desc, &dest_tfrmd_desc);

  zdnn_init_ztensor(&src_pre_tfrmd_desc, &src_tfrmd_desc, &src_ztensor);
  zdnn_init_ztensor(&dest_pre_tfrmd_desc, &dest_tfrmd_desc, &dest_ztensor);

  // src_ztensor is NOT transformed at this point

  status = zdnn_reshape_ztensor(&src_ztensor, &dest_ztensor);

  TEST_ASSERT_MESSAGE_FORMATTED(exp_status == status,
                                "expected status = %08x, got status = %08x",
                                exp_status, status);
}

void test_fail_dest_already_transformed() {
  zdnn_status status, exp_status = ZDNN_INVALID_STATE;
  test_datatype = FP16;

  zdnn_tensor_desc src_pre_tfrmd_desc, dest_pre_tfrmd_desc, src_tfrmd_desc,
      dest_tfrmd_desc;
  zdnn_ztensor src_ztensor, dest_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &src_pre_tfrmd_desc,
                                 4, 4, 4, 4);
  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype, &dest_pre_tfrmd_desc,
                                 4, 4, 4, 4);

  zdnn_generate_transformed_desc(&src_pre_tfrmd_desc, &src_tfrmd_desc);
  zdnn_generate_transformed_desc(&dest_pre_tfrmd_desc, &dest_tfrmd_desc);

  zdnn_init_ztensor(&src_pre_tfrmd_desc, &src_tfrmd_desc, &src_ztensor);
  zdnn_init_ztensor(&dest_pre_tfrmd_desc, &dest_tfrmd_desc, &dest_ztensor);

  src_ztensor.is_transformed = true;
  // sabotage dest_ztensor
  dest_ztensor.is_transformed = true;

  status = zdnn_reshape_ztensor(&src_ztensor, &dest_ztensor);

  TEST_ASSERT_MESSAGE_FORMATTED(exp_status == status,
                                "expected status = %08x, got status = %08x",
                                exp_status, status);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_4x5x6x7_4x5x6x7);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_1x2x3x4_6x1x1x4);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_2x3x4x68_4x1x6x68);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_4x3x40x70_8x20x3x70);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_4x4x4x4_1x1x16x16);

  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(test_fail_total_elements_mismatch);
  RUN_TEST(test_fail_not_nhwc_nor_hwck);
  RUN_TEST(test_fail_not_same_layout);
  RUN_TEST(test_fail_src_not_transformed);
  RUN_TEST(test_fail_dest_already_transformed);

  return UNITY_END();
}
