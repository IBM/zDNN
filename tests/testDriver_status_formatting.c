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
 * Regression coverage for a previously unsafe format-string prefixing
 * implementation in set_zdnn_status().
 *
 * What this test protects:
 * - Passing a very long `format` string into set_zdnn_status() must not
 *   overflow internal buffers.
 * - The emitted log should still contain the ZDNN status string prefix.
 */

void setUp(void) {}
void tearDown(void) {}

void test_long_format_is_safe(void) {
  // A format string longer than MAX_STATUS_FMTSTR_SIZE.
  // No '%' specifiers: we want a pure string-copy path through vsnprintf.
  char long_format[4096];
  memset(long_format, 'A', sizeof(long_format) - 1);
  long_format[sizeof(long_format) - 1] = '\0';

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  zdnn_status st = set_zdnn_status(ZDNN_INVALID_SHAPE, __func__, __FILE__,
                                  __LINE__, long_format);
  restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ZDNN_INVALID_SHAPE, st,
                                  "Unexpected status returned");

  // Ensure we did not lose the status prefix.
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, "ZDNN_INVALID_SHAPE"),
                               "Expected status string not found in stderr");
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_long_format_is_safe);

  return UNITY_END();
}
