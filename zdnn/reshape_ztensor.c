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

#include <stdlib.h>
#include <string.h>

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_reshape_ztensor)
#endif

/// Reshape and copy buffer content from source zTensor's buffer to destination
/// zTensor's in accordance to destination zTensor's shape.  The following
/// conditions must be satisfied:
///
/// - Both transformed_desc must be fully initialized
/// - dest->buffer must be pre-allocated
/// - src must be transformed
/// - dest must be not already transformed
/// - Both transformed_desc->layout must be the same and either NHWC or HWCK
/// - Both zTensors must contain equal number of elements
///
/// \param[in] src Pointer to source ztensor to copy from
/// \param[out] dest Pointer to destination ztensor to copy to
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_STATE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_BUFFER
///         ZDNN_CONVERT_FAILURE
///
zdnn_status zdnn_reshape_ztensor(const zdnn_ztensor *src, zdnn_ztensor *dest) {

  // It's caller's responsibility to ensure pre_transformed_desc and
  // transformed_desc agree with each other.  This function does not
  // look at pre_transformed_desc at all.

  zdnn_tensor_desc *src_tfrmd_desc = src->transformed_desc,
                   *dest_tfrmd_desc = dest->transformed_desc;

  LOG_TRACE("(transformed) src: %d %d %d %d -> dest: %d %d %d %d\n",
            src_tfrmd_desc->dim4, src_tfrmd_desc->dim3, src_tfrmd_desc->dim2,
            src_tfrmd_desc->dim1, dest_tfrmd_desc->dim4, dest_tfrmd_desc->dim3,
            dest_tfrmd_desc->dim2, dest_tfrmd_desc->dim1);

  if (get_num_elements(src, ELEMENTS_PRE) !=
      get_num_elements(dest, ELEMENTS_PRE)) {
    return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                       "src (%d * %d * %d * %d) does not have the same number "
                       "of elements as dest (%d * %d * %d * %d)",
                       src_tfrmd_desc->dim4, src_tfrmd_desc->dim3,
                       src_tfrmd_desc->dim2, src_tfrmd_desc->dim1,
                       dest_tfrmd_desc->dim4, dest_tfrmd_desc->dim3,
                       dest_tfrmd_desc->dim2, dest_tfrmd_desc->dim1);
  }

  if (src_tfrmd_desc->layout != dest_tfrmd_desc->layout) {
    return ZDNN_STATUS(
        ZDNN_INVALID_LAYOUT,
        "Layouts not the same.  src layout: %d, dest layout: %d.",
        src_tfrmd_desc->layout, dest_tfrmd_desc->layout);
  }

  // check either src/dest, both layouts are same by now
  if (src_tfrmd_desc->layout != ZDNN_NHWC &&
      src_tfrmd_desc->layout != ZDNN_HWCK) {
    return ZDNN_STATUS(ZDNN_INVALID_LAYOUT,
                       "Layout must be either NHWC or HWCK.  layout: %d.",
                       src_tfrmd_desc->layout);
  }

  if (!src->is_transformed) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE, "src tensor is not transformed.",
                       NO_ARG);
  }

  if (dest->is_transformed) {
    return ZDNN_STATUS(ZDNN_INVALID_STATE,
                       "dest tensor contains transformed tensor data.", NO_ARG);
  }

  /*
  /  Different strategies for different shape combinations
  */

  // Scenario: Both have the exact same shape.  Just memcpy() everything.
  if (src_tfrmd_desc->dim4 == dest_tfrmd_desc->dim4 &&
      src_tfrmd_desc->dim3 == dest_tfrmd_desc->dim3 &&
      src_tfrmd_desc->dim2 == dest_tfrmd_desc->dim2 &&
      src_tfrmd_desc->dim1 == dest_tfrmd_desc->dim1) {

    LOG_TRACE("Strategy: full memcpy()", NO_ARG);
    memcpy(dest->buffer, src->buffer, zdnn_getsize_ztensor(src_tfrmd_desc));

    return ZDNN_STATUS_OK;
  }

  // Scenario: src: (x, y, z, c), dest: (i, j, k, c),
  // Both sides have the exact same # of sticks, so memcpy() each stick
  if (src_tfrmd_desc->dim1 == dest_tfrmd_desc->dim1) {
    LOG_TRACE("Strategy: same C, memcpy() every stick", NO_ARG);

    uint32_t x = 0, y = 0, z = 0;

    for (uint32_t i = 0; i < dest_tfrmd_desc->dim4; i++) {
      for (uint32_t j = 0; j < dest_tfrmd_desc->dim3; j++) {
        for (uint32_t k = 0; k < dest_tfrmd_desc->dim2; k++) {
          for (uint32_t c = 0;
               c < CEIL(dest_tfrmd_desc->dim1, AIU_2BYTE_CELLS_PER_STICK);
               c++) {

            // get_stick_offset() tells us where the sticks are
            // use transformed_desc here so we don't need to transposed shapes
            // (e.g., 3DS)
            size_t offset_src = get_stick_offset(
                x, y, z, c * AIU_2BYTE_CELLS_PER_STICK, src->transformed_desc);
            size_t offset_dest = get_stick_offset(
                i, j, k, c * AIU_2BYTE_CELLS_PER_STICK, dest->transformed_desc);

            LOG_TRACE("%d %d %d %d (%" PRIx64 ") -> %d %d %d %d (%" PRIx64
                      ")\n",
                      x, y, z, c, offset_src, i, j, k, c, offset_dest);

            // memcpy() the entire stick to simplify things
            memcpy((void *)((uintptr_t)dest->buffer + offset_dest),
                   (void *)((uintptr_t)src->buffer + offset_src),
                   AIU_BYTES_PER_STICK);
          }
          // go to the next stick on the src side
          z++;
          if (z == src_tfrmd_desc->dim2) {
            z = 0;
            y++;
            if (y == src_tfrmd_desc->dim3) {
              y = 0;
              x++;
            }
          }
        }
      }
    }
    return ZDNN_STATUS_OK;
  }

  LOG_TRACE("Strategy: last resort", NO_ARG);

#define STACK_TMPBUF_SIZE (1024 * 1024) // 1MB

  // last resort: fully unstick and restick

  // NOTE: this will change when we have "no conversion stick/unstick"
  // for now, unstick to FP32 and restick to preserve precision.

  char stack_tmpbuf[STACK_TMPBUF_SIZE];
  void *malloc_tmpbuf = NULL;

  zdnn_ztensor tmp_tensor_src, tmp_tensor_dest;
  zdnn_tensor_desc tmp_pre_tfrmd_desc_src, tmp_pre_tfrmd_desc_dest;

  memcpy(&tmp_tensor_src, src, sizeof(zdnn_ztensor));
  memcpy(&tmp_tensor_dest, dest, sizeof(zdnn_ztensor));
  memcpy(&tmp_pre_tfrmd_desc_src, src->pre_transformed_desc,
         sizeof(zdnn_tensor_desc));
  memcpy(&tmp_pre_tfrmd_desc_dest, dest->pre_transformed_desc,
         sizeof(zdnn_tensor_desc));

  tmp_pre_tfrmd_desc_src.type = FP32;
  tmp_pre_tfrmd_desc_dest.type = FP32;
  tmp_tensor_src.pre_transformed_desc = &tmp_pre_tfrmd_desc_src;
  tmp_tensor_dest.pre_transformed_desc = &tmp_pre_tfrmd_desc_dest;

  zdnn_status status;

  // if unstickified content is small enough (=< STACK_TMPBUF_SIZE) then use
  // stack_tmpbuf instead of malloc()-ing one on heap
  uint64_t s = get_num_elements(src, ELEMENTS_PRE) * get_data_type_size(FP32);
  if (s > STACK_TMPBUF_SIZE) {
    malloc_tmpbuf = malloc(s);
  }

  // no need to log status, zdnn_transform_origtensor() and
  // zdnn_transform_ztensor() already do
  if ((status = zdnn_transform_origtensor(
           &tmp_tensor_src, malloc_tmpbuf ? malloc_tmpbuf : stack_tmpbuf)) ==
      ZDNN_OK) {
    status = zdnn_transform_ztensor(
        &tmp_tensor_dest, malloc_tmpbuf ? malloc_tmpbuf : stack_tmpbuf);
  }

  if (malloc_tmpbuf) {
    free(malloc_tmpbuf);
  }
  return status;
}
