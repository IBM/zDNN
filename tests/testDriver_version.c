// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
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
#include "version.h"
#include <string.h>

#define MAJOR_NEWER(x) (x + 0x00020000)
#define MAJOR_OLDER(x) (x - 0x00020000)
#define MINOR_NEWER(x) (x + 0x00000200)
#define MINOR_OLDER(x) (x - 0x00000200)
#define PATCH_NEWER(x) (x + 0x00000002)
#define PATCH_OLDER(x) (x - 0x00000002)

void setUp(void) { /* This is run before EACH TEST */
  aiu_lib_vernum = AIU_UNKNOWN;
}

void tearDown(void) {}

// ***************************************************
// Under VERSION_C_TEST library version is always: 5.5.5
// ***************************************************

void test_version_runnable(uint32_t app_vernum, uint32_t new_aiu_lib_vernum,
                           bool exp_result) {
  aiu_lib_vernum = new_aiu_lib_vernum;

  TEST_ASSERT_MESSAGE_FORMATTED(
      zdnn_is_version_runnable(app_vernum) == exp_result,
      "zdnn_is_version_runnable() did not return %d", exp_result);
}

// ************************
// *** MAJOR ver tests
// ************************

// ---------------------------------------------------------
// | app     | hw      | library | runnable?
// ---------------------------------------------------------
// | 5.5.5   | 7.x.x   | 5.5.5   | no
// | 7.x.x   | 5.5.5   | 5.5.5   | no
// | 7.x.x   | 7.x.x   | 5.5.5   | no
// | 5.3.x   | 5.5.x   | 5.5.5   | yes
// ---------------------------------------------------------

void hw_major_newer_fail() {
  test_version_runnable(ZDNN_VERNUM, MAJOR_NEWER(ZDNN_VERNUM), false);
}

void app_major_newer_fail() {
  test_version_runnable(MAJOR_NEWER(ZDNN_VERNUM), ZDNN_VERNUM, false);
}

void lib_major_older_fail() {
  test_version_runnable(MAJOR_NEWER(ZDNN_VERNUM), MAJOR_NEWER(ZDNN_VERNUM),
                        false);
}

void major_all_match_pass() {
  test_version_runnable(MINOR_OLDER(ZDNN_VERNUM), ZDNN_VERNUM, true);
}

// ************************
// *** MINOR ver tests
// ************************

// ---------------------------------------------------------
// | app     | hw      | library | runnable?
// ---------------------------------------------------------
// | 5.7.5   | 5.5.5   | 5.5.5   | no
// | 5.3.5   | 5.5.5   | 5.5.5   | yes
// | 5.5.5   | 5.7.5   | 5.5.5   | yes
// | 5.5.5   | 5.3.5   | 5.5.5   | no
// | 5.3.5   | 5.3.5   | 5.5.5   | yes
// | 5.7.5   | 5.7.5   | 5.5.5   | no
// ---------------------------------------------------------
// | 5.3.5   | 5.7.5   | 5.5.5   | yes
// | 5.1.5   | 5.3.5   | 5.5.5   | yes
// | 5.3.5   | 5.1.5   | 5.5.5   | no
// ---------------------------------------------------------

void app_minor_newer_fail() {
  test_version_runnable(MINOR_NEWER(ZDNN_VERNUM), ZDNN_VERNUM, false);
}

void app_minor_older_pass() {
  test_version_runnable(MINOR_OLDER(ZDNN_VERNUM), ZDNN_VERNUM, true);
}

void hw_minor_newer_pass() {
  test_version_runnable(ZDNN_VERNUM, MINOR_NEWER(ZDNN_VERNUM), true);
}

void hw_minor_older_fail() {
  test_version_runnable(ZDNN_VERNUM, MINOR_OLDER(ZDNN_VERNUM), false);
}

void lib_minor_newer_pass() {
  test_version_runnable(MINOR_OLDER(ZDNN_VERNUM), MINOR_OLDER(ZDNN_VERNUM),
                        true);
}

void lib_minor_older_fail() {
  test_version_runnable(MINOR_NEWER(ZDNN_VERNUM), MINOR_NEWER(ZDNN_VERNUM),
                        false);
}

void app_minor_older_hw_minor_newer_pass() {
  test_version_runnable(MINOR_OLDER(ZDNN_VERNUM), MINOR_NEWER(ZDNN_VERNUM),
                        true);
}

void app_minor_older_hw_minor_even_older_pass() {
  test_version_runnable(MINOR_OLDER(ZDNN_VERNUM),
                        MINOR_OLDER(MINOR_OLDER(ZDNN_VERNUM)), false);
}

// ************************
// *** Mixed MAJOR/MINOR ver tests
// ************************

// all of these are the runnable = yes cases in MINOR ver tests but now with
// different MAJOR ver, so they all become runnable = no
// ---------------------------------------------------------
// | app     | hw      | library | runnable?
// ---------------------------------------------------------
// | 7.3.5   | 5.5.5   | 5.5.5   | no
// | 5.5.5   | 7.7.5   | 5.5.5   | no
// | 3.3.5   | 7.3.5   | 5.5.5   | no
// | 7.3.5   | 3.7.5   | 5.5.5   | no
// | 5.1.5   | 3.3.5   | 5.5.5   | no
// ---------------------------------------------------------

void mixed_app_major_newer_fail() {
  test_version_runnable(MAJOR_NEWER(MINOR_OLDER(ZDNN_VERNUM)), ZDNN_VERNUM,
                        false);
}

void mixed_hw_major_newer_fail() {
  test_version_runnable(ZDNN_VERNUM, MAJOR_NEWER(MINOR_NEWER(ZDNN_VERNUM)),
                        false);
}

void mixed_app_major_older_hw_major_newer_fail() {
  test_version_runnable(MAJOR_OLDER(MINOR_OLDER(ZDNN_VERNUM)),
                        MAJOR_NEWER(MINOR_OLDER(ZDNN_VERNUM)), false);
}

void mixed_app_major_newer_hw_major_older_fail() {
  test_version_runnable(MAJOR_NEWER(MINOR_OLDER(ZDNN_VERNUM)),
                        MAJOR_OLDER(MINOR_NEWER(ZDNN_VERNUM)), false);
}

void mixed_hw_major_older_fail() {
  test_version_runnable(MINOR_OLDER(MINOR_OLDER(ZDNN_VERNUM)),
                        MAJOR_OLDER(MINOR_OLDER(ZDNN_VERNUM)), false);
}

// ************************
// *** PATCH ver tests
// ************************

// Everything passes

void app_patch_newer_pass() {
  test_version_runnable(PATCH_NEWER(ZDNN_VERNUM), ZDNN_VERNUM, true);
}

void app_patch_older_pass() {
  test_version_runnable(PATCH_OLDER(ZDNN_VERNUM), ZDNN_VERNUM, true);
}

void hw_patch_newer_pass() {
  test_version_runnable(ZDNN_VERNUM, PATCH_NEWER(ZDNN_VERNUM), true);
}

void hw_patch_older_pass() {
  test_version_runnable(ZDNN_VERNUM, PATCH_OLDER(ZDNN_VERNUM), true);
}

void lib_patch_newer_pass() {
  test_version_runnable(PATCH_OLDER(ZDNN_VERNUM), PATCH_OLDER(ZDNN_VERNUM),
                        true);
}

void lib_patch_older_pass() {
  test_version_runnable(PATCH_NEWER(ZDNN_VERNUM), PATCH_NEWER(ZDNN_VERNUM),
                        true);
}

// ************************
// *** get_max_runnable tests
// ************************

void test_get_max_runnable(uint32_t exp_vernum) {
  uint32_t vernum = zdnn_get_max_runnable_version();
  TEST_ASSERT_MESSAGE_FORMATTED(
      vernum == exp_vernum,
      "zdnn_get_max_runnable_version() did not return %08x (found: %08x)",
      exp_vernum, vernum);
}

void test_max_ver_hw_major_newer() {
  aiu_lib_vernum = MAJOR_NEWER(ZDNN_VERNUM);
  test_get_max_runnable(AIU_UNKNOWN);
}

void test_max_ver_hw_major_older() {
  aiu_lib_vernum = MAJOR_OLDER(ZDNN_VERNUM);
  test_get_max_runnable(AIU_UNKNOWN);
}

void test_max_ver_hw_minor_newer() {
  aiu_lib_vernum = MINOR_NEWER(ZDNN_VERNUM);
  test_get_max_runnable(ZDNN_VERNUM | 0xFF);
}

void test_max_ver_hw_minor_older() {
  aiu_lib_vernum = MINOR_OLDER(ZDNN_VERNUM);
  test_get_max_runnable(MINOR_OLDER(ZDNN_VERNUM) | 0xFF);
}

void test_max_ver_hw_patch_newer() {
  aiu_lib_vernum = PATCH_OLDER(ZDNN_VERNUM);
  test_get_max_runnable(ZDNN_VERNUM | 0xFF);
}

int main(void) {
  UNITY_BEGIN();
#ifdef VERSION_C_TEST

  RUN_TEST(hw_major_newer_fail);
  RUN_TEST(app_major_newer_fail);
  RUN_TEST(lib_major_older_fail);
  RUN_TEST(major_all_match_pass);

  RUN_TEST(app_minor_newer_fail);
  RUN_TEST(app_minor_older_pass);
  RUN_TEST(hw_minor_newer_pass);
  RUN_TEST(hw_minor_older_fail);
  RUN_TEST(lib_minor_newer_pass);
  RUN_TEST(lib_minor_older_fail);
  RUN_TEST(app_minor_older_hw_minor_newer_pass);
  RUN_TEST(app_minor_older_hw_minor_even_older_pass);

  RUN_TEST(mixed_app_major_newer_fail);
  RUN_TEST(mixed_hw_major_newer_fail);
  RUN_TEST(mixed_app_major_older_hw_major_newer_fail);
  RUN_TEST(mixed_app_major_newer_hw_major_older_fail);
  RUN_TEST(mixed_hw_major_older_fail);

  RUN_TEST(app_patch_newer_pass);
  RUN_TEST(app_patch_older_pass);
  RUN_TEST(hw_patch_newer_pass);
  RUN_TEST(hw_patch_older_pass);
  RUN_TEST(lib_patch_newer_pass);
  RUN_TEST(lib_patch_older_pass);

  RUN_TEST(test_max_ver_hw_major_newer);
  RUN_TEST(test_max_ver_hw_major_older);
  RUN_TEST(test_max_ver_hw_minor_newer);
  RUN_TEST(test_max_ver_hw_minor_older);
  RUN_TEST(test_max_ver_hw_patch_newer);

#endif
  return UNITY_END();
}
