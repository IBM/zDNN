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
 * Regression coverage for an alignment edge case in the Linux STFLE path.
 *
 * When ZDNN_ENABLE_PRECHECK is enabled, invoke_stfle() verifies the facility
 * list buffer is 8-byte aligned. Historically, zdnn_is_nnpa_installed() used a
 * local unsigned-char array which is not *guaranteed* to be 8-byte aligned.
 *
 * This test asserts that enabling precheck does not trigger a
 * ZDNN_MISALIGNED_PARMBLOCK failure in the STFLE path.
 *
 * Note: This is a Linux-only behavior; z/OS uses a system copy of STFLE.
 */

void setUp(void) {
#ifdef __MVS__
  TEST_IGNORE_MESSAGE("Linux STFLE alignment regression test; skipping on z/OS");
#endif
}

void tearDown(void) {}

void test_precheck_does_not_trigger_misaligned_parmblock(void) {
  precheck_enabled = true;

  char buf_stderr[BUFSIZ] = {0};

  stderr_to_pipe();
  (void)zdnn_is_nnpa_installed();
  restore_stderr(buf_stderr, BUFSIZ);

  TEST_ASSERT_NULL_MESSAGE(
      strstr(buf_stderr, "ZDNN_MISALIGNED_PARMBLOCK"),
      "Unexpected misaligned parmblock failure in STFLE path");

  precheck_enabled = false;
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_precheck_does_not_trigger_misaligned_parmblock);

  return UNITY_END();
}
