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
#include <unistd.h>

#include "testsupport.h"

char msg_trace[] = "TRACE";
char msg_debug[] = "DEBUG";
char msg_info[] = "INFO";
char msg_warn[] = "WARN";
char msg_error[] = "ERROR";
char msg_fatal[] = "FATAL";

void setUp(void) {
#ifndef ZDNN_CONFIG_DEBUG
  TEST_IGNORE_MESSAGE(
      "ZDNN_CONFIG_DEBUG not set. Unable to test full logger. Skip tests.");
#endif
}

void tearDown(void) {}

void try_log(uint32_t loglvl) {

  // override whatever ZDNN_LOGLEVEL/ZDNN_LOGMODULE are set in env
  log_level = loglvl;
  log_module[0] = '\0';

  char buf_stdout[BUFSIZ] = {0};
  char buf_stderr[BUFSIZ] = {0};

  stdout_to_pipe();
  stderr_to_pipe();

  LOG_TRACE(msg_trace, NO_ARG);
  LOG_DEBUG(msg_debug, NO_ARG);
  LOG_INFO(msg_info, NO_ARG);
  LOG_WARN(msg_warn, NO_ARG);
  LOG_ERROR(msg_error, NO_ARG);
  LOG_FATAL(msg_fatal, NO_ARG);

  restore_stdout(buf_stdout, BUFSIZ);
  restore_stderr(buf_stderr, BUFSIZ);

#define EXPECTS_ONLY_STDOUT(msg)                                               \
  if (strstr(buf_stdout, msg) == NULL) {                                       \
    TEST_FAIL_MESSAGE("can't find " #msg " message in STDOUT");                \
  }                                                                            \
  if (strstr(buf_stderr, msg) != NULL) {                                       \
    TEST_FAIL_MESSAGE("found " #msg " message unexpectedly STDERR");           \
  }

#define EXPECTS_ONLY_STDERR(msg)                                               \
  if (strstr(buf_stderr, msg) == NULL) {                                       \
    TEST_FAIL_MESSAGE("can't find " #msg " message in STDERR");                \
  }                                                                            \
  if (strstr(buf_stdout, msg) != NULL) {                                       \
    TEST_FAIL_MESSAGE("found " #msg " message unexpectedly STDOUT");           \
  }

#define EXPECTS_NEITHER(msg)                                                   \
  if (strstr(buf_stdout, msg) != NULL) {                                       \
    TEST_FAIL_MESSAGE("found " #msg " message unexpectedly STDOUT");           \
  }                                                                            \
  if (strstr(buf_stderr, msg) != NULL) {                                       \
    TEST_FAIL_MESSAGE("found " #msg " message unexpectedly STDERR");           \
  }

  switch (loglvl) {
  case (LOGLEVEL_TRACE):
    EXPECTS_ONLY_STDOUT(msg_trace);
    EXPECTS_ONLY_STDOUT(msg_debug);
    EXPECTS_ONLY_STDOUT(msg_info);
    EXPECTS_ONLY_STDOUT(msg_warn);
    EXPECTS_ONLY_STDERR(msg_error);
    EXPECTS_ONLY_STDERR(msg_fatal);
    break;
  case (LOGLEVEL_DEBUG):
    EXPECTS_NEITHER(msg_trace);
    EXPECTS_ONLY_STDOUT(msg_debug);
    EXPECTS_ONLY_STDOUT(msg_info);
    EXPECTS_ONLY_STDOUT(msg_warn);
    EXPECTS_ONLY_STDERR(msg_error);
    EXPECTS_ONLY_STDERR(msg_fatal);
    break;
  case (LOGLEVEL_INFO):
    EXPECTS_NEITHER(msg_trace);
    EXPECTS_NEITHER(msg_debug);
    EXPECTS_ONLY_STDOUT(msg_info);
    EXPECTS_ONLY_STDOUT(msg_warn);
    EXPECTS_ONLY_STDERR(msg_error);
    EXPECTS_ONLY_STDERR(msg_fatal);
    break;
  case (LOGLEVEL_WARN):
    EXPECTS_NEITHER(msg_trace);
    EXPECTS_NEITHER(msg_debug);
    EXPECTS_NEITHER(msg_info);
    EXPECTS_ONLY_STDOUT(msg_warn);
    EXPECTS_ONLY_STDERR(msg_error);
    EXPECTS_ONLY_STDERR(msg_fatal);
    break;
  case (LOGLEVEL_ERROR):
    EXPECTS_NEITHER(msg_trace);
    EXPECTS_NEITHER(msg_debug);
    EXPECTS_NEITHER(msg_info);
    EXPECTS_NEITHER(msg_warn);
    EXPECTS_ONLY_STDERR(msg_error);
    EXPECTS_ONLY_STDERR(msg_fatal);
    break;
  case (LOGLEVEL_FATAL):
    EXPECTS_NEITHER(msg_trace);
    EXPECTS_NEITHER(msg_debug);
    EXPECTS_NEITHER(msg_info);
    EXPECTS_NEITHER(msg_warn);
    EXPECTS_NEITHER(msg_error);
    EXPECTS_ONLY_STDERR(msg_fatal);
    break;
  case (LOGLEVEL_OFF):
    EXPECTS_NEITHER(msg_trace);
    EXPECTS_NEITHER(msg_debug);
    EXPECTS_NEITHER(msg_info);
    EXPECTS_NEITHER(msg_warn);
    EXPECTS_NEITHER(msg_error);
    EXPECTS_NEITHER(msg_fatal);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("Invalid log level %u", loglvl);
  }
}

void test_off(void) { try_log(LOGLEVEL_OFF); }
void test_fatal(void) { try_log(LOGLEVEL_FATAL); }
void test_err0r(void) { try_log(LOGLEVEL_ERROR); } // "error" confuses jenkins
void test_warn(void) { try_log(LOGLEVEL_WARN); }
void test_info(void) { try_log(LOGLEVEL_INFO); }
void test_debug(void) { try_log(LOGLEVEL_DEBUG); }
void test_trace(void) { try_log(LOGLEVEL_TRACE); }

// log_module with only "testDriver_logger.c" in it
void test_in_logmodule() {
  log_level = LOGLEVEL_INFO;
  strncpy(log_module, __FILE__, LOGMODULE_SIZE);

  char buf_stdout[BUFSIZ] = {0};

  stdout_to_pipe();
  LOG_INFO(msg_info, NO_ARG);
  restore_stdout(buf_stdout, BUFSIZ);

  if (strstr(buf_stdout, msg_info) == NULL) {
    TEST_FAIL_MESSAGE("can't find message message in STDOUT");
  }

  fflush(stdout);
}

// log_module with "testDriver_logger.c" somewhere in the string
void test_in_logmodule2() {
  log_level = LOGLEVEL_INFO;
  strncpy(log_module, "fafafa.c " __FILE__ " lalala.c", LOGMODULE_SIZE);

  char buf_stdout[BUFSIZ] = {0};

  stdout_to_pipe();
  LOG_INFO(msg_info, NO_ARG);
  restore_stdout(buf_stdout, BUFSIZ);

  if (strstr(buf_stdout, msg_info) == NULL) {
    TEST_FAIL_MESSAGE("can't find message message in STDOUT");
  }

  fflush(stdout);
}

// log_module with "testDriver_logger.c" completely not in
void test_not_in_logmodule() {
  log_level = LOGLEVEL_INFO;
  strncpy(log_module, "hahahahaha.c", LOGMODULE_SIZE);

  char buf_stdout[BUFSIZ] = {0};

  stdout_to_pipe();
  LOG_INFO(msg_info, NO_ARG);
  restore_stdout(buf_stdout, BUFSIZ);

  if (strstr(buf_stdout, msg_info) != NULL) {
    TEST_FAIL_MESSAGE("found message unexpectedly STDOUT");
  }

  fflush(stdout);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_trace);
  RUN_TEST(test_debug);
  RUN_TEST(test_info);
  RUN_TEST(test_warn);
  RUN_TEST(test_err0r);
  RUN_TEST(test_fatal);
  RUN_TEST(test_off);

  RUN_TEST(test_in_logmodule);
  RUN_TEST(test_in_logmodule2);
  RUN_TEST(test_not_in_logmodule);

  return UNITY_END();
}
