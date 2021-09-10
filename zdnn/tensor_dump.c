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

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"

#define FLOAT_DECIMAL_PLACES "3"

/// Dump raw tensor data with N/H/W/C or H/W/C/K seperation
///
/// \param[in] desc Pointer to pre-transformed descriptor
/// \param[in] data Pointer to raw tensor data
/// \param[in] mode AS_HEX or AS_FLOAT
///
/// \return None
///
void dumpdata_origtensor(const zdnn_tensor_desc *pre_tfrmd_desc,
                         void *tensor_data, dump_mode mode) {

  uint32_t dim4 = (get_data_layout_dims(pre_tfrmd_desc->layout) >= 4)
                      ? pre_tfrmd_desc->dim4
                      : 1;
  uint32_t dim3 = (get_data_layout_dims(pre_tfrmd_desc->layout) >= 3)
                      ? pre_tfrmd_desc->dim3
                      : 1;
  uint32_t dim2 = (get_data_layout_dims(pre_tfrmd_desc->layout) >= 2)
                      ? pre_tfrmd_desc->dim2
                      : 1;
  uint32_t dim1 = pre_tfrmd_desc->dim1;

  // ZDNN_*DS layouts promote one dim to dim4. This sets the dump's dim*
  // variables match so the dump's NHWC labels line up with the layout.
  switch (pre_tfrmd_desc->layout) {
  case ZDNN_2DS:
    dim4 = pre_tfrmd_desc->dim2;
    dim2 = 1;
    break;
  case ZDNN_3DS:
    dim4 = pre_tfrmd_desc->dim3;
    dim3 = 1;
    break;
  default:
    // Nothing to do here. This is just to avoid compiler warnings
    break;
  }

  int cell_size;
  if (mode == AS_HEX) {
    cell_size = MAX(get_data_type_size(pre_tfrmd_desc->type) * 2, 5) + 2;
  } else {
    cell_size = 10; // xxxxx.yy + 2 spaces
  }

  uint64_t element_idx = 0;
  char dim3_char, dim2_char, dim1_char, dim0_char;

  switch (pre_tfrmd_desc->layout) {
  case ZDNN_NCHW:
    dim3_char = 'N';
    dim2_char = 'C';
    dim1_char = 'H';
    dim0_char = 'W';
    break;
  case ZDNN_HWCK:
    dim3_char = 'H';
    dim2_char = 'W';
    dim1_char = 'C';
    dim0_char = 'K';
    break;
  default:
    // everything else (1D/3D/etc etc treat as NHWC)
    dim3_char = 'N';
    dim2_char = 'H';
    dim1_char = 'W';
    dim0_char = 'C';
  }

  printf("raw tensor layout = %s -> %c%c%c%c %ux%ux%ux%u\n",
         get_data_layout_str(pre_tfrmd_desc->layout), dim3_char, dim2_char,
         dim1_char, dim0_char, dim4, dim3, dim2, dim1);
  printf("data type = %s\n", get_data_type_str(pre_tfrmd_desc->type));

  for (uint32_t e4 = 0; e4 < dim4; e4++) {
    printf(" %c = %-*u\n", dim3_char, 5, e4);
    for (uint32_t e3 = 0; e3 < dim3; e3++) {
      printf(" |  %c = %-*u\n", dim2_char, 5, e3);
      printf(" |  |      %c ->  ", dim0_char);
      for (uint32_t i = 0; i < dim1; i++) {
        printf("%-*u", cell_size, i);
      }
      printf("\n");
      printf(" |  |      %c  +--", dim1_char);
      for (uint32_t i = 0; i < dim1; i++) {
        printf("%.*s", cell_size, "-----------------------");
      }
      printf("\n");
      for (uint32_t e2 = 0; e2 < dim2; e2++) {
        printf(" |  |  %*u  |  ", 5, e2);

        for (uint32_t e1 = 0; e1 < dim1; e1++) {
          if (mode == AS_HEX) {
            switch (pre_tfrmd_desc->type) {
            case BFLOAT:
            case FP16:
              printf("%04x%*s", ((uint16_t *)tensor_data)[element_idx],
                     cell_size - 4, "");
              break;
            case FP32:
              printf("%08x%*s", ((uint32_t *)tensor_data)[element_idx],
                     cell_size - 8, "");
              break;
            default:
              break;
            }
          } else {
            switch (pre_tfrmd_desc->type) {
            case BFLOAT:
              printf("%-*." FLOAT_DECIMAL_PLACES "f", cell_size,
                     cnvt_1_bfloat_to_fp32(
                         ((uint16_t *)tensor_data)[element_idx]));
              break;
            case FP16:
              printf(
                  "%-*." FLOAT_DECIMAL_PLACES "f", cell_size,
                  cnvt_1_fp16_to_fp32(((uint16_t *)tensor_data)[element_idx]));
              break;
            case FP32:
              printf("%-*." FLOAT_DECIMAL_PLACES "f", cell_size,
                     ((float *)tensor_data)[element_idx]);
              break;
            default:
              break;
            }
          }
          element_idx++;
        }
        printf("\n");
      }
    }
  }
}

/// Dump zTensor buffer data with N/H/W/C or H/W/C/K seperation
///
/// \param[in] ztensor Pointer to zdnn_ztensor struct
/// \param[in] mode AS_HEX or AS_FLOAT
/// \param[in] print_all prints unused page-padding sticks if true
///
/// \return None
///
void dumpdata_ztensor(const zdnn_ztensor *ztensor, dump_mode mode,
                      bool print_all) {

  int cell_size;
  if (mode == AS_HEX) {
    cell_size = 7; // XXXXX + 2 spaces
  } else {
    cell_size = 10; // xxxxx.yy + 2 spaces
  }

  zdnn_tensor_desc *pre_desc = ztensor->pre_transformed_desc;
  zdnn_tensor_desc *tfrmd_desc = ztensor->transformed_desc;

  // Print buffer info
  printf("ztensor->buffer = %" PRIXPTR ", ztensor->buffer_size = %" PRIu64 "\n",
         (uintptr_t)ztensor->buffer, ztensor->buffer_size);

  // Print pre_tfrmd_desc layout and shape
  printf("ztensor->pre_transformed_desc->layout = ");
  if (pre_desc) {
    printf("%s ", get_data_layout_str(pre_desc->layout));
    if (get_data_layout_dims(pre_desc->layout) >= 4) {
      printf("%ux", pre_desc->dim4);
    }
    if (get_data_layout_dims(pre_desc->layout) >= 3) {
      printf("%ux", pre_desc->dim3);
    }
    if (get_data_layout_dims(pre_desc->layout) >= 2) {
      printf("%ux", pre_desc->dim2);
    }
    printf("%u\n", pre_desc->dim1);
  } else {
    printf("NULL\n");
  }

  // Print tfrmd_desc layout and shape
  printf("ztensor->transformed_desc->layout = ");
  if (tfrmd_desc) {
    printf("%s %ux%ux%ux%u\n", get_data_layout_str(tfrmd_desc->layout),
           tfrmd_desc->dim4, tfrmd_desc->dim3, tfrmd_desc->dim2,
           tfrmd_desc->dim1);

    if (tfrmd_desc->layout != ZDNN_HWCK) {

      uint32_t current_n = 0, current_h = 0, current_w = 0, current_c = 0;

      // cumulative number of sticks processed, for printing overall offsets
      uint32_t accum_w = 0;

      // bump up h and etc when current_w reaches this value
      uint32_t stop_w =
          CEIL(tfrmd_desc->dim2, AIU_STICKS_PER_PAGE) * AIU_STICKS_PER_PAGE;

      // each element in the stick area is 2-bytes (assuming there's only
      // DLFLOAT16)
      for (uint64_t i = 0; i < ztensor->buffer_size / 2;
           i += AIU_2BYTE_CELLS_PER_STICK) {

        // print a "page break" at every AIU_PAGESIZE_IN_BYTES/2-th element
        if (i && !(i % (AIU_PAGESIZE_IN_BYTES / 2))) {
          printf("                              +--");
          for (uint32_t j = 0; j < AIU_2BYTE_CELLS_PER_STICK; j++) {
            printf("%.*s", cell_size, "-----------------------");
          }
          printf("\n");
        }

        // print the "N = " and "H = " banner when W = 0 and C = 0 or 64 or 128
        // or etc
        if (current_w == 0 && !(current_c % AIU_2BYTE_CELLS_PER_STICK)) {
          printf("                 N = %-*u\n", 5, current_n);
          printf("                 |  H = %-*u\n", 5, current_h);
        }

        if (current_w == 0) {

          // print the horizontal c indices.  if c is not used then print a
          // blank instead.
          printf("                 |  |      C ->  ");
          for (uint32_t c_idx = current_c;
               c_idx < current_c + AIU_2BYTE_CELLS_PER_STICK; c_idx++) {
            if (c_idx < tfrmd_desc->dim1) {
              printf("%-*u", cell_size, c_idx);
            } else {
              printf("%-*s", cell_size, " ");
            }
          }
          printf("\n");

          // print the "W = " banner
          printf("%016" PRIXPTR " |  |      W  +--",
                 (uintptr_t)ztensor->buffer +
                     accum_w * AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE);
          for (uint32_t j = 0; j < AIU_2BYTE_CELLS_PER_STICK; j++) {
            printf("%.*s", cell_size, "-----------------------");
          }
          printf("\n");
        }

        // print a whole stick if w is within valid range, otherwise print out
        // a bunch of blanks
        if (current_w < tfrmd_desc->dim2 || print_all) {
          printf("     (+%08x) |  |  %*u  |  ",
                 accum_w * AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE, 5,
                 current_w);
          for (int j = 0; j < AIU_2BYTE_CELLS_PER_STICK; j++) {
            if (mode == AS_HEX) {
              printf("%04x%*s", ((uint16_t *)ztensor->buffer)[i + j],
                     cell_size - 4, "");
            } else {
              printf(
                  "%-*." FLOAT_DECIMAL_PLACES "f", cell_size,
                  cnvt_1_dlf16_to_fp32(((uint16_t *)ztensor->buffer)[i + j]));
            }
          }
        } else {
          printf("     (+%08x) |  |         |  ",
                 accum_w * AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE);
        }
        printf("\n");

        // manipulate the indices
        current_w++;
        accum_w++;
        if (current_w == stop_w) {
          current_w = 0;
          current_h++;
          if (current_h == tfrmd_desc->dim3) {
            current_h = 0;
            current_c += AIU_2BYTE_CELLS_PER_STICK;
            if (current_c >= tfrmd_desc->dim1) {
              current_c = 0;
            }
          }
        }
        if (!current_c && !current_w && !current_h) {
          current_n++;
        }
      }
    } else {

      uint32_t current_h = 0, current_w = 0, current_c = 0, current_k = 0;

      // cumulative number of sticks processed, for printing overall offsets
      uint32_t accum_c = 0;

      // bump up c and etc when current_c reaches this value
      uint32_t stop_c =
          CEIL(tfrmd_desc->dim2, AIU_STICKS_PER_PAGE) * AIU_STICKS_PER_PAGE;

      // each element in the stick area is 2-bytes (assuming there's only
      // DLFLOAT16)
      for (uint64_t i = 0; i < ztensor->buffer_size / 2;
           i += AIU_2BYTE_CELLS_PER_STICK) {

        // print a "page break" at every AIU_PAGESIZE_IN_BYTES/2-th element
        if (i && !(i % (AIU_PAGESIZE_IN_BYTES / 2))) {

          printf("                              +--");
          for (uint32_t j = 0; j < AIU_2BYTE_CELLS_PER_STICK; j++) {
            printf("%.*s", cell_size, "-----------------------");
          }
          printf("\n");
        }

        // print the "H = " and "W = " banner when C = 0 and K = 0 or 64 or 128
        // or etc
        if (current_c == 0 && !(current_k % AIU_2BYTE_CELLS_PER_STICK)) {
          printf("                 H = %-*u\n", 5, current_h);
          printf("                 |  W = %-*u\n", 5, current_w);
        }

        if (current_c == 0) {

          // print the horizontal k indices.  if k is not used then print a
          // blank instead.
          printf("                 |  |      K ->  ");
          for (uint32_t k_idx = current_k;
               k_idx < current_k + AIU_2BYTE_CELLS_PER_STICK; k_idx++) {
            if (k_idx < tfrmd_desc->dim1) {
              printf("%-*u", cell_size, k_idx);
            } else {
              printf("%-*s", cell_size, " ");
            }
          }
          printf("\n");

          // print the "C = " banner
          printf("%016" PRIXPTR " |  |      C  +--",
                 (uintptr_t)ztensor->buffer +
                     accum_c * AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE);
          for (uint32_t j = 0; j < AIU_2BYTE_CELLS_PER_STICK; j++) {
            printf("%.*s", cell_size, "-----------------------");
          }
          printf("\n");
        }

        // print a whole stick if k is within valid range, otherwise print out
        // a bunch of blanks
        if (current_c < tfrmd_desc->dim2 || print_all) {
          printf("     (+%08x) |  |  %*u  |  ",
                 accum_c * AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE, 5,
                 current_c);
          for (int j = 0; j < AIU_2BYTE_CELLS_PER_STICK; j++) {
            if (mode == AS_HEX) {
              printf("%04x%*s", ((uint16_t *)ztensor->buffer)[i + j],
                     cell_size - 4, "");
            } else {
              printf(
                  "%-*." FLOAT_DECIMAL_PLACES "f", cell_size,
                  cnvt_1_dlf16_to_fp32(((uint16_t *)ztensor->buffer)[i + j]));
            }
          }
        } else {
          printf("     (+%08x) |  |         |  ",
                 accum_c * AIU_2BYTE_CELLS_PER_STICK * AIU_2BYTE_CELL_SIZE);
        }
        printf("\n");

        // manipulate the indices
        current_c++;
        accum_c++;
        if (current_c == stop_c) {
          current_c = 0;
          current_w++;
          if (current_w == tfrmd_desc->dim3) {
            current_w = 0;
            current_h++;
            if (current_h == tfrmd_desc->dim4) {
              current_h = 0;
              current_k += AIU_2BYTE_CELLS_PER_STICK;
            }
          }
        }
      }
    }
  } else {
    printf("NULL\n");
  }
}
