// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024
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

#include <string.h>

/*
 * Regression coverage for ztensor_slice_dim4() error reporting.
 *
 * This test ensures the function reports the *pre-transformed* layout (the
 * field the switch statement actually uses) when it rejects unsupported slicing
 * layouts.
 */

void setUp(void) {
  log_level = LOGLEVEL_ERROR;
  log_module[0] = '\0';
}

void tearDown(void) {}

void test_error_message_reports_pre_transformed_layout(void) {
  zdnn_tensor_desc pre = {0};
  zdnn_tensor_desc tfrmd = {0};
  zdnn_ztensor in = {0};

  // Pick an unsupported pre-transformed layout for slicing.
  // (Supported: 2DS, 3DS, 4D/NHWC/NCHW)
  zdnn_init_pre_transformed_desc(ZDNN_1D, FP32, &pre, 2);

  // Transformed descriptor just needs to exist for memcpy in the slicer.
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE, &tfrmd,
                        2, 1, 1, 1);

  zdnn_init_ztensor(&pre, &tfrmd, &in);

  zdnn_tensor_desc out_pre = {0};
  zdnn_tensor_desc out_tfrmd = {0};
  zdnn_ztensor out = {0};

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  zdnn_status st = ztensor_slice_dim4(&in, 0, 0, &out_pre, &out_tfrmd, &out);
  restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ZDNN_INVALID_LAYOUT, st,
                                  "Expected invalid layout status");

  // The message should reference the pre-transformed layout we set (ZDNN_1D).
  TEST_ASSERT_NOT_NULL_MESSAGE(
      strstr(buf_stderr, "Invalid pre-transformed layout for slicing"),
      "Expected pre-transformed layout in error message");
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, "ZDNN_1D"),
                               "Expected ZDNN_1D in error message");
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_error_message_reports_pre_transformed_layout);

  return UNITY_END();
}
