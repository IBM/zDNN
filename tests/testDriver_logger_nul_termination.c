// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2026
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

#include <stdarg.h>
#include <string.h>

/*
 * Regression coverage for log_message()'s no-formatting path and boundary
 * conditions.
 *
 * Historical behavior (bug):
 * - When arg == NULL, log_message() used strncpy() without guaranteeing
 *   NUL-termination and then called strlen() on the buffer.
 * - If the message filled or exceeded the remaining buffer, strlen() could read
 *   past the end of msg_buf (undefined behavior).
 *
 * Expected behavior (fixed):
 * - The message is always NUL-terminated, even when truncated.
 * - Fits-in-buffer messages are preserved completely.
 * - Messages near buffer boundaries are handled safely.
 * - Formatted messages continue to behave correctly.
 */

void setUp(void) {}
void tearDown(void) {}

static void call_log_message_formatted(log_levels lvl, const char *func,
                                       const char *file, int line,
                                       const char *fmt, ...) {
  va_list arg;
  va_start(arg, fmt);
  log_message(lvl, func, file, line, fmt, arg);
  va_end(arg);
}

// 1 & 4. Calling log_message() with arg == NULL and a long message guarantees
// NUL termination upon truncation.
void test_log_message_arg_null_truncation(void) {
  log_level = LOGLEVEL_ERROR;
  log_module[0] = '\0';

  // Construct a message far larger than the internal 512-byte buffer.
  // Omit trailing newline to exercise the auto-newline check on terminated buffer.
  char long_msg[2048];
  memset(long_msg, 'B', sizeof(long_msg) - 1);
  long_msg[sizeof(long_msg) - 1] = '\0';

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  log_message(LOGLEVEL_ERROR, __func__, __FILE__, __LINE__, long_msg, NULL);
  int len = restore_stderr(buf_stderr, BUFSIZ);

  // Buffer must contain payload, be NUL-terminated, and have auto-newline appended
  TEST_ASSERT_TRUE_MESSAGE(len > 0, "stderr output is empty or read failed");
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, "BBB"),
                               "Expected payload not found in stderr");
  TEST_ASSERT_EQUAL_CHAR('\n', buf_stderr[len - 1]);
}

// 2. A message that fits within the available buffer is preserved correctly.
void test_log_message_arg_null_fits_preserved(void) {
  log_level = LOGLEVEL_ERROR;
  log_module[0] = '\0';

  const char *short_msg = "A message that comfortably fits in the buffer";
  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  log_message(LOGLEVEL_ERROR, __func__, __FILE__, __LINE__, short_msg, NULL);
  int len = restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_TRUE_MESSAGE(len > 0, "stderr output is empty or read failed");
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, short_msg),
                               "Short message not preserved in stderr");
  TEST_ASSERT_EQUAL_CHAR('\n', buf_stderr[len - 1]);
}

// 3. A message near the buffer boundary is handled safely.
void test_log_message_arg_null_near_boundary(void) {
  log_level = LOGLEVEL_ERROR;
  log_module[0] = '\0';

  // Size message to approach boundary (~400 bytes, so header + msg is near 512)
  char boundary_msg[401];
  memset(boundary_msg, 'N', sizeof(boundary_msg) - 1);
  boundary_msg[sizeof(boundary_msg) - 1] = '\0';

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  log_message(LOGLEVEL_ERROR, __func__, __FILE__, __LINE__, boundary_msg, NULL);
  int len = restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_TRUE_MESSAGE(len > 0, "stderr output is empty or read failed");
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, "NNN"),
                               "Near-boundary payload not found in stderr");
  TEST_ASSERT_EQUAL_CHAR('\n', buf_stderr[len - 1]);
}

// 5. Existing formatted logging behaviour remains correct.
void test_log_message_formatted_preserved(void) {
  log_level = LOGLEVEL_ERROR;
  log_module[0] = '\0';

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  call_log_message_formatted(LOGLEVEL_ERROR, __func__, __FILE__, __LINE__,
                             "Formatted value: %d, string: %s", 12345, "hello");
  int len = restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_TRUE_MESSAGE(len > 0, "stderr output is empty or read failed");
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, "Formatted value: 12345, string: hello"),
                               "Formatted message not preserved in stderr");
  TEST_ASSERT_EQUAL_CHAR('\n', buf_stderr[len - 1]);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_log_message_arg_null_truncation);
  RUN_TEST(test_log_message_arg_null_fits_preserved);
  RUN_TEST(test_log_message_arg_null_near_boundary);
  RUN_TEST(test_log_message_formatted_preserved);

  return UNITY_END();
}
