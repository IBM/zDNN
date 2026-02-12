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
 * Regression coverage for log_message()'s no-formatting path.
 *
 * Historical behavior (bug):
 * - When arg == NULL, log_message() used strncpy() without guaranteeing
 *   NUL-termination and then called strlen() on the buffer.
 * - If the message exactly filled the remaining buffer, strlen() could read
 *   past the end of msg_buf (undefined behavior).
 *
 * Expected behavior (fixed):
 * - The message is always NUL-terminated, even when truncated.
 */

void setUp(void) {}
void tearDown(void) {}

void test_log_message_arg_null_is_nul_terminated(void) {
  log_level = LOGLEVEL_ERROR;
  log_module[0] = '\0';

  // Construct a message that is far larger than the logger's internal buffer.
  // We intentionally omit a trailing newline to exercise the auto-newline path.
  char long_msg[2048];
  memset(long_msg, 'B', sizeof(long_msg) - 1);
  long_msg[sizeof(long_msg) - 1] = '\0';

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  log_message(LOGLEVEL_ERROR, __func__, __FILE__, __LINE__, long_msg, NULL);
  restore_stderr(buf_stderr, BUFSIZ);

  // If we got here, we did not crash. Also ensure some of our payload made it.
  TEST_ASSERT_NOT_NULL_MESSAGE(strstr(buf_stderr, "BBB"),
                               "Expected payload not found in stderr");
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_log_message_arg_null_is_nul_terminated);

  return UNITY_END();
}
