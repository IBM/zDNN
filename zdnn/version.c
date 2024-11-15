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

#include "version.h"

#include <string.h>

#ifdef __MVS__
#pragma export(zdnn_is_version_runnable)
#pragma export(zdnn_get_max_runnable_version)
#endif

// the latest "zAIU hardware version" that this library can identify.
//
// conceptually, this is latest zDNN library version number that the current hw
// is capable of driving, based on all hw version/revision information this
// version of the library knows about.
uint32_t aiu_lib_vernum;

// -----------------------------------------------------------------------------
// nnpa signature
// -----------------------------------------------------------------------------

aiu_hwinfo aiu_hwinfo_telumii = {
    {0x80, 0x00, 0xfc, 0x00, 0xf0, 0x00, 0x7c, 0x00, 0xf0, 0x00,
     0xc0, 0x00, 0xc0, 0x00, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00,
     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
     0xc0, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x82, 0xa0},
    {0xe0, 0x00, 0x00, 0x01},
    0x00008000,
    0x0000000100000000,
    {0x60, 0x00},
    "telumii",
    LIB_VERNUM(1, 1, 0)};

aiu_hwinfo aiu_hwinfo_telumi = {
    {0x80, 0x00, 0xfc, 0x00, 0xc0, 0x00, 0x78, 0x00, 0x80, 0x00,
     0xc0, 0x00, 0xc0, 0x00, 0xe0, 0x00, 0x00, 0x00, 0x00, 0x00,
     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
     0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00},
    {0xc0, 0x00, 0x00, 0x00},
    0x00008000,
    0x0000000100000000,
    {0x60, 0x00},
    "telumi",
    LIB_VERNUM(1, 0, 0)};

// array of all known hw
// ** put NEWER hw versions first !!! ***
// ** MUST NULL TERMINATE !!! ***
aiu_hwinfo *aiu_hwinfo_list[HWINFO_LIST_MAXSIZE] = {&aiu_hwinfo_telumii,
                                                    &aiu_hwinfo_telumi, NULL};

/// Check if the bits specified in "bitmask" are all 1s in the memory block
/// "memblk" of size "size"
///
/// \param[in] bitmask pointer to the bitmask
/// \param[in] memblk pointer to the memory block
/// \param[in] size size of memory block
///
/// \return true/false
///
bool mem_check_bitmask(void *bitmask, void *memblk, size_t size) {
  bool is_match = true;

  // size of the memory block to check is always multiples of 2
  for (uint64_t i = 0; i < size / 2 && is_match == true; i++) {
    uint16_t mask = ((uint16_t *)bitmask)[i];
    uint16_t content = ((uint16_t *)memblk)[i];

    is_match &= ((mask & content) == mask);
  }

  return is_match;
}

/// Refresh aiu_lib_vernum value by interpreting the NNPA-QAF result
///
/// \param[in] None
///
/// \return None
///
void refresh_aiu_lib_vernum() {

  aiu_hwinfo **info = &aiu_hwinfo_list[0];
  uint32_t c = 0;

  aiu_lib_vernum = AIU_UNKNOWN;

  while (*info && c < HWINFO_LIST_MAXSIZE) {

    // each aiu_hwinfo struct contains NNPA-QAF bitmasks and uint values of a
    // known zAIU hw.  so lets say we have x3 (newest), x2 and x1 (oldest)
    //
    // basically we look at the current NNPA-QAF result, and see if it:
    // - meets the bitmask requirements (so it can, e.g., do all the NNPA ops hw
    //   x3 can do), via mem_check_bitmask()
    // - meets or exceeds the value requirements (e.g., it has equal or higher
    //   MDIS value than hw x3), via >=
    // and if so then we know the hw is at least x3 capable.  if not then try
    // the next older hw in the array.
    //
    // with this we can use an older minor version library on a newer hw (say,
    // x4) that the library doesn't know about and use it as x3, since it meets
    // x3's capabilities requirements

    if (mem_check_bitmask((*info)->blk1, QAF_BLK1_PTR, HWINFO_BLK1_LEN) &&
        mem_check_bitmask((*info)->blk2, QAF_BLK2_PTR, HWINFO_BLK2_LEN) &&
        (QAF_VAL1 >= (*info)->val1) && (QAF_VAL2 >= (*info)->val2) &&
        mem_check_bitmask((*info)->blk3, QAF_BLK3_PTR, HWINFO_BLK3_LEN)) {

      // the latest and greatest that we know of.  we're done
      aiu_lib_vernum = (*info)->lib_vernum;
      return;
    }

    info++;
    c++;
  }
}

/// Check if application built for zDNN version "ver_num" can be run on the
/// current hardware with the installed zDNN library
///
/// \param[in] ver_num zDNN version number from application
///
/// \return true/false
///
bool zdnn_is_version_runnable(uint32_t ver_num) {

  // 3 version numbers to deal with
  // - incoming ver_num
  // - this library's (ZDNN_VER_*)
  // - the hw's (aiu_lib_vernum)

  // major: all 3 must match
  if ((MAJOR(ver_num) != ZDNN_VER_MAJOR) ||
      (MAJOR(ver_num) != MAJOR(aiu_lib_vernum))) {
    return false;
  }

  // minor: incoming ver_num must not be newer than library's
  //        incoming ver_num must not be newer than hw's
  if (MINOR(ver_num) > ZDNN_VER_MINOR ||
      MINOR(ver_num) > MINOR(aiu_lib_vernum)) {
    return false;
  }

  // patch: don't care

  return true;
}

/// Returns the maximum zDNN version number that the current hardware and
/// installed zDNN library can run together
///
/// \param[in] None
///
/// \return version number
///
uint32_t zdnn_get_max_runnable_version() {
  if (MAJOR(aiu_lib_vernum) != ZDNN_VER_MAJOR) {
    return AIU_UNKNOWN;
  } else {
    // return minimum ver_num between the library's and the hw's
    // set the patch byte to 0xFF so that it's at "max"
    return MIN(ZDNN_VERNUM, aiu_lib_vernum) | 0xFF;
  }
}
