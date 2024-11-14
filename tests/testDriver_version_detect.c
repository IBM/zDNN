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

#include "testsupport.h"
#include "version.h"
#include <string.h>

// magic-numbering these to check against what's in version.h
#define LIB_VERNUM_Z16 0x00010000

#define LIB_VERNUM_NEWER_MAJOR LIB_VERNUM(7, 5, 5)
#define LIB_VERNUM_NEWER_MINOR LIB_VERNUM(5, 7, 5)
#define LIB_VERNUM_BASELINE LIB_VERNUM(5, 5, 5)
#define LIB_VERNUM_OLDER_MINOR LIB_VERNUM(5, 3, 5)
#define LIB_VERNUM_OLDER_MAJOR LIB_VERNUM(3, 5, 5)

// newer major: newer minor + mdis bump
aiu_hwinfo aiu_hwinfo_newer_major = {
    {0x00, 0x11, 0x11, 0x11}, {0x00, 0x01}, 7, 5, {0x00, 0x11}, "newer major",
    LIB_VERNUM_NEWER_MAJOR};

// newer minor: baseline + blk1 2nd byte bit bump + blk2 2nd byte bit bump
aiu_hwinfo aiu_hwinfo_newer_minor = {
    {0x00, 0x11, 0x11, 0x11}, {0x00, 0x01}, 5, 5, {0x00, 0x11}, "newer minor",
    LIB_VERNUM_NEWER_MINOR};

aiu_hwinfo aiu_hwinfo_baseline = {
    {0x00, 0x01, 0x11, 0x11}, {0x00, 0x00}, 5, 5, {0x00, 0x11}, "baseline",
    LIB_VERNUM_BASELINE};

// older minor: baseline - blk3 2nd byte bit nerf
aiu_hwinfo aiu_hwinfo_older_minor = {
    {0x00, 0x01, 0x11, 0x11}, {0x00, 0x00}, 5, 5, {0x00, 0x10}, "older minor",
    LIB_VERNUM_OLDER_MINOR};

// older major: older minor - blk1 3rd byte bit nerf - mts nerf
aiu_hwinfo aiu_hwinfo_older_major = {
    {0x00, 0x01, 0x10, 0x11}, {0x00, 0x00}, 5, 3, {0x00, 0x10}, "older major",
    LIB_VERNUM_OLDER_MAJOR};

void reset_qaf_result() {
  // QAF now has the information of the baseline machine
  memset(&nnpa_query_result, 0, sizeof(nnpa_qaf_parameter_block));
  memcpy(QAF_BLK1_PTR, &aiu_hwinfo_baseline.blk1, HWINFO_BLK1_LEN);
  memcpy(QAF_BLK2_PTR, &aiu_hwinfo_baseline.blk2, HWINFO_BLK2_LEN);
  QAF_VAL1 = aiu_hwinfo_baseline.val1;
  QAF_VAL2 = aiu_hwinfo_baseline.val2;
  memcpy(QAF_BLK3_PTR, &aiu_hwinfo_baseline.blk3, HWINFO_BLK3_LEN);
}

void setUp(void) {}

void tearDown(void) {}

// ************************
// *** LIB_VERNUM tests
// ************************

void test_lib_vernum_nnpa() {
  VERIFY_HW_ENV; // verify required HW env is available.
  refresh_aiu_lib_vernum();
  uint32_t expected_lib_vernum;
  if (zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1) == true) {
    expected_lib_vernum = 0x00010100;
  } else {
    expected_lib_vernum = 0x00010000;
  }
  TEST_ASSERT_MESSAGE_FORMATTED(aiu_lib_vernum == expected_lib_vernum,
                                "aiu_lib_vernum is not detected as %08" PRIx32,
                                expected_lib_vernum);
}

// **************************************************
// *** LIB_VERNUM detection tests - Fake machines
// **************************************************

void test_baseline_exact() {
  reset_qaf_result();
  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_BASELINE,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_BASELINE, aiu_lib_vernum);
}

void test_newer_minor_exact() {
  reset_qaf_result();
  *((char *)QAF_BLK1_PTR + 1) = 0x11;
  *((char *)QAF_BLK2_PTR + 1) = 0x01;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_NEWER_MINOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_NEWER_MINOR, aiu_lib_vernum);
}

void test_newer_major_exact() {
  reset_qaf_result();
  *((char *)QAF_BLK1_PTR + 1) = 0x11;
  *((char *)QAF_BLK2_PTR + 1) = 0x01;
  QAF_VAL1 = 7;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_NEWER_MAJOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_NEWER_MAJOR, aiu_lib_vernum);
}

void test_older_minor_exact() {
  reset_qaf_result();
  *((char *)QAF_BLK3_PTR + 1) = 0x10;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_OLDER_MINOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_OLDER_MINOR, aiu_lib_vernum);
}

void test_older_major_exact() {
  reset_qaf_result();
  *((char *)QAF_BLK1_PTR + 2) = 0x10;
  *((char *)QAF_BLK3_PTR + 1) = 0x10;
  QAF_VAL2 = 3;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_OLDER_MAJOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_OLDER_MAJOR, aiu_lib_vernum);
}

void test_exceeds_newer_minor_but_not_newer_major() {
  // turn on all bits, leave val1 and val2 at 5 and 5
  memset(&nnpa_query_result, 0xff, sizeof(nnpa_qaf_parameter_block));
  QAF_VAL1 = 5;
  QAF_VAL2 = 5;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_NEWER_MINOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_NEWER_MINOR, aiu_lib_vernum);
}

void test_older_minor_enough_but_not_baseline() {
  reset_qaf_result();
  *((char *)QAF_BLK1_PTR) = 0xFF;     // better blk1 than baseline
  *((char *)QAF_BLK3_PTR + 1) = 0x10; // worse blk3 than baseline

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_OLDER_MINOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_OLDER_MINOR, aiu_lib_vernum);
}

void test_all_flags_on_but_older_vals() {
  // turn on all bits, set val1 and val2 at 3 and 3 so they are worse than older
  // major
  memset(&nnpa_query_result, 0xff, sizeof(nnpa_qaf_parameter_block));
  QAF_VAL1 = 3;
  QAF_VAL2 = 3;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == AIU_UNKNOWN,
      "aiu_lib_vernum is not detected as %08x (found: %08x)", AIU_UNKNOWN,
      aiu_lib_vernum);
}

void test_super_mythical() {
  // turn on all bits, set val1 and val2 at 100, 100 so it exceeds newer major
  memset(&nnpa_query_result, 0xff, sizeof(nnpa_qaf_parameter_block));
  QAF_VAL1 = 100;
  QAF_VAL2 = 100;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == LIB_VERNUM_NEWER_MAJOR,
      "aiu_lib_vernum is not detected as %08x (found: %08x)",
      LIB_VERNUM_NEWER_MAJOR, aiu_lib_vernum);
}

void test_super_old1() {
  // even fewer bits on than older major
  memset(&nnpa_query_result, 0x00, sizeof(nnpa_qaf_parameter_block));
  *((char *)QAF_BLK3_PTR + 1) = 18;
  QAF_VAL1 = aiu_hwinfo_baseline.val1;
  QAF_VAL2 = aiu_hwinfo_baseline.val2;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == AIU_UNKNOWN,
      "aiu_lib_vernum is not detected as %08x (found: %08x)", AIU_UNKNOWN,
      aiu_lib_vernum);
}

void test_super_old2() {
  // even lower val1 than older major
  reset_qaf_result();
  QAF_VAL1 = 2;

  refresh_aiu_lib_vernum();
  TEST_ASSERT_MESSAGE_FORMATTED(
      aiu_lib_vernum == AIU_UNKNOWN,
      "aiu_lib_vernum is not detected as %08x (found: %08x)", AIU_UNKNOWN,
      aiu_lib_vernum);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_lib_vernum_nnpa);

  // only tests with fake machines this point forward

  aiu_hwinfo_list[0] = &aiu_hwinfo_newer_major;
  aiu_hwinfo_list[1] = &aiu_hwinfo_newer_minor;
  aiu_hwinfo_list[2] = &aiu_hwinfo_baseline;
  aiu_hwinfo_list[3] = &aiu_hwinfo_older_minor;
  aiu_hwinfo_list[4] = &aiu_hwinfo_older_major;

  RUN_TEST(test_baseline_exact);
  RUN_TEST(test_newer_minor_exact);
  RUN_TEST(test_newer_major_exact);
  RUN_TEST(test_older_minor_exact);
  RUN_TEST(test_older_major_exact);

  RUN_TEST(test_exceeds_newer_minor_but_not_newer_major);
  RUN_TEST(test_older_minor_enough_but_not_baseline);
  RUN_TEST(test_all_flags_on_but_older_vals);
  RUN_TEST(test_super_mythical);
  RUN_TEST(test_super_old1);
  RUN_TEST(test_super_old2);

  return UNITY_END();
}
