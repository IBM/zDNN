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

#include <stdbool.h>
#include <stddef.h>

#define MAX_PAD 64

static bool saved_precheck_enabled;

void setUp(void) { saved_precheck_enabled = precheck_enabled; }

void tearDown(void) { precheck_enabled = saved_precheck_enabled; }

/// Call zdnn_is_nnpa_installed() from a stack frame whose size varies with
/// `pad`, so the facility-list buffer inside it lands at different offsets.
static bool __attribute__((noinline)) call_with_pad(size_t pad) {
  volatile char buf[pad + 1]; // VLA shifts the stack pointer
  buf[0] = 0;
  bool result = zdnn_is_nnpa_installed();
  buf[0] = (char)result; // keep buf live across the call
  return result;
}

/*********************************************************************
 * With precheck enabled, invoke_stfle() rejects a facility list that
 * is not on a doubleword boundary, and zdnn_is_nnpa_installed() would
 * then wrongly report "no NNPA".  Detection must therefore give the
 * same answer with precheck on as with precheck off, for any stack
 * layout.
 * *******************************************************************/
void stfle_alignment_precheck_matches_baseline() {
  precheck_enabled = false;
  bool baseline = zdnn_is_nnpa_installed();

  precheck_enabled = true;
  for (size_t pad = 0; pad < MAX_PAD; pad++) {
    bool actual = call_with_pad(pad);
    TEST_ASSERT_MESSAGE_FORMATTED(
        actual == baseline,
        "pad %zu: zdnn_is_nnpa_installed() returned %d with precheck "
        "enabled but %d with precheck disabled (STFLE list misaligned?)",
        pad, actual, baseline);
  }
}

/// Repeated calls with precheck enabled must be stable.
void stfle_alignment_precheck_repeatable() {
  precheck_enabled = true;
  bool first = zdnn_is_nnpa_installed();
  for (int i = 0; i < 100; i++) {
    TEST_ASSERT_EQUAL_MESSAGE(first, zdnn_is_nnpa_installed(),
                              "STFLE result changed between calls");
  }
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(stfle_alignment_precheck_matches_baseline);
  RUN_TEST(stfle_alignment_precheck_repeatable);
  return UNITY_END();
}
