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

#ifndef ZDNN_VERSION_H_
#define ZDNN_VERSION_H_

#include <inttypes.h>
#include <stdbool.h>

#include "zdnn.h"
#include "zdnn_private.h"

// custom values for tests.
#ifdef VERSION_C_TEST
#undef ZDNN_VERNUM
#undef ZDNN_VER_MAJOR
#undef ZDNN_VER_MINOR
#undef ZDNN_VER_PATCH
#define ZDNN_VERNUM 0x050505
#define ZDNN_VER_MAJOR 0x05
#define ZDNN_VER_MINOR 0x05
#define ZDNN_VER_PATCH 0x05
#endif

#define AIU_UNKNOWN 0x00000000

#define QAF_SIZEOF(x) sizeof((&nnpa_query_result)->x)

#define HWINFO_BLK1_QAF_MEMBER installed_functions_vector
#define HWINFO_BLK2_QAF_MEMBER installed_data_layout_formats
#define HWINFO_BLK3_QAF_MEMBER installed_dt1_conversions_vector
#define HWINFO_VAL1_QAF_MEMBER maximum_dimension_index_size
#define HWINFO_VAL2_QAF_MEMBER maximum_tensor_size

// 3 fields next to each other in nnpa_qaf_parameter_block
#define HWINFO_BLK1_LEN                                                        \
  (QAF_SIZEOF(HWINFO_BLK1_QAF_MEMBER) +                                        \
   QAF_SIZEOF(installed_parameter_block_formats) +                             \
   QAF_SIZEOF(installed_data_types))

// installed_data_layout_formats
#define HWINFO_BLK2_LEN (QAF_SIZEOF(HWINFO_BLK2_QAF_MEMBER))

// installed_dt1_conversions_vector alone
#define HWINFO_BLK3_LEN (QAF_SIZEOF(HWINFO_BLK3_QAF_MEMBER))

#define QAF_BLK1_PTR (&nnpa_query_result.HWINFO_BLK1_QAF_MEMBER)
#define QAF_BLK2_PTR (&nnpa_query_result.HWINFO_BLK2_QAF_MEMBER)
#define QAF_BLK3_PTR (&nnpa_query_result.HWINFO_BLK3_QAF_MEMBER)

#define QAF_VAL1 (nnpa_query_result.HWINFO_VAL1_QAF_MEMBER)
#define QAF_VAL2 (nnpa_query_result.HWINFO_VAL2_QAF_MEMBER)

#define HWINFO_DESC_STR_MAXSIZE 128

typedef struct aiu_hwinfo {
  // each member represent a contiguous segment or standalone value to check
  // within the nnpa_qaf_parameter_block
  char blk1[HWINFO_BLK1_LEN];
  char blk2[HWINFO_BLK2_LEN];
  uint32_t val1; // mdis
  uint64_t val2; // mts
  char blk3[HWINFO_BLK3_LEN];
  char desc_str[HWINFO_DESC_STR_MAXSIZE]; // descriptive string
  uint32_t lib_vernum;                    // lib vernum to assign
} aiu_hwinfo;

#define LIB_VERNUM(major, minor, patch) ((major << 16) + (minor << 8) + patch)

#define MAJOR(v) ((v >> 16) & 0xFF)
#define MINOR(v) ((v >> 8) & 0xFF)
#define PATCH(v) (v & 0xFF)

#define HWINFO_LIST_MAXSIZE 256
extern aiu_hwinfo *aiu_hwinfo_list[HWINFO_LIST_MAXSIZE];

#endif /* ZDNN_VERSION_H_ */
