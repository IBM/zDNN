// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "testsupport.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/*********************************************************************
 * This testcase only works for LoZ, as there's no way to easily
 * verify ctrace()'s output under z/OS.  The intent of this testcase
 * is to verify if the status_diag code gets invoked when we want it
 * to, not as much as if it's producing the correct output.
 * *******************************************************************/

#ifndef __MVS__

void setUp(void) { /* This is run before EACH TEST */
}

void tearDown(void) {}

void try_diag(uint32_t status_to_diag, uint32_t status_to_set,
              bool expect_backtrace) {

  status_diag = status_to_diag;

  char buf_stdout[BUFSIZ] = {0};

  stdout_to_pipe();
  set_zdnn_status(status_to_set, __func__, __FILE__, __LINE__,
                  "this is a test");
  restore_stdout(buf_stdout, BUFSIZ);

  /*
    the backtrace should have something like:

    obj/../../aiu/libzdnn.so.1(set_zdnn_status+0x1d4)[0x3ffb750a19c]
    ./obj/testDriver_status_diag.out() [0x1001a2c]
    ./obj/testDriver_status_diag.out() [0x1001ade]
    ./obj/testDriver_status_diag.out() [0x1005012]
    ./obj/testDriver_status_diag.out() [0x1001baa]

    so search for "libzdnn" in the captured STDOUT output
  */

  if (expect_backtrace) {
    TEST_ASSERT_MESSAGE(strstr(buf_stdout, "libzdnn") != NULL,
                        "Can't find backtrace in buf_stdout");
  } else {
    TEST_ASSERT_MESSAGE(strstr(buf_stdout, "libzdnn") == NULL,
                        "Backtrace unexpectedly appears");
  }
}

void test_real_error() {
  try_diag(ZDNN_INVALID_SHAPE, ZDNN_INVALID_SHAPE, true);
}

void test_zdnn_ok() { try_diag(ZDNN_OK, ZDNN_OK, true); }

void test_not_match1() { try_diag(ZDNN_INVALID_SHAPE, ZDNN_OK, false); }

void test_not_match2() {
  try_diag(ZDNN_INVALID_SHAPE, ZDNN_INVALID_FORMAT, false);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(test_not_match1);
  RUN_TEST(test_not_match2);

  RUN_TEST(test_real_error);
  RUN_TEST(test_zdnn_ok);

  return UNITY_END();
}

#else

void setUp(void) {}

void tearDown(void) {}

int main() {
  UNITY_BEGIN();
  return UNITY_END();
}

#endif // __MVS__
