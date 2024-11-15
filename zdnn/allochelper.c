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

#include "zdnn.h"
#include "zdnn_private.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __MVS__
#pragma export(zdnn_allochelper_ztensor)
#pragma export(zdnn_free_ztensor_buffer)
#pragma export(zdnn_getsize_ztensor)
#endif

/// Allocate a buffer with size required for storing transformed tensor data of
/// shape specified in the transformed descriptor, and assocate the buffer with
/// the incoming zTensor
///
/// \param[out] ztensor Pointer to zdnn_ztensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_LAYOUT
///         ZDNN_INVALID_SHAPE
///         ZDNN_ALLOCATION_FAILURE
///
zdnn_status zdnn_allochelper_ztensor(zdnn_ztensor *ztensor) {
  uint64_t size;
  zdnn_status status;

  // only the information in transformed_desc matters, so make sure it's
  // reasonable
  if ((status = verify_transformed_descriptor(ztensor->transformed_desc)) !=
      ZDNN_OK) {
    return status;
  }

  // get the size and allocate space aligned on a 4k boundary. If the malloc
  // fails, return error.
  size = zdnn_getsize_ztensor(ztensor->transformed_desc);
  if (!(ztensor->buffer = malloc_aligned_4k(size))) {
    return ZDNN_STATUS(ZDNN_ALLOCATION_FAILURE,
                       "Unable to allocate %" PRIu64 " bytes.", size);
  }

  // With a successful malloc, set our ztensor's buffer_size with the allocated
  // size.
  ztensor->buffer_size = size;

  // Successful zdnn_allochelper_ztensor call
  return ZDNN_STATUS_OK;
}

/// Free the stickified tensor data buffer within the incoming zTensor
///
/// \param[in] ztensor Pointer to zdnn_ztensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_BUFFER
///
zdnn_status zdnn_free_ztensor_buffer(const zdnn_ztensor *ztensor) {
  if (!ztensor->buffer) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }
  free_aligned_4k(ztensor->buffer);
  return ZDNN_STATUS_OK;
}

/// Calculates the number of bytes required for storing transformed tensor data
/// of shape specified in the transformed descriptor
///
/// \param[in] tfrmd_desc Pointer to transformed tensor descriptor
///
/// \return Memory size (in bytes)
///
uint64_t zdnn_getsize_ztensor(const zdnn_tensor_desc *tfrmd_desc) {
  uint32_t cells_per_stick;
  uint32_t number_of_sticks;
  switch (tfrmd_desc->type) {
  case ZDNN_BINARY_INT8:
    if (tfrmd_desc->format == ZDNN_FORMAT_4DWEIGHTS) {
      // 4DWEIGHTS has two vectors interleaved, therefore only 64 cells vs 128
      // Due to this interleaving, number_of_sticks is halved, but must be
      // rounded up to stay even for proper interleaving.
      cells_per_stick = AIU_2BYTE_CELLS_PER_STICK;
      number_of_sticks = CEIL(tfrmd_desc->dim2, 2);
    } else {
      cells_per_stick = AIU_1BYTE_CELLS_PER_STICK;
      number_of_sticks = tfrmd_desc->dim2;
    }
    break;
  case ZDNN_BINARY_INT32:
    cells_per_stick = AIU_4BYTE_CELLS_PER_STICK;
    number_of_sticks = tfrmd_desc->dim2;
    break;
  case ZDNN_DLFLOAT16: /* fallthrough */
  default:
    cells_per_stick = AIU_2BYTE_CELLS_PER_STICK;
    number_of_sticks = tfrmd_desc->dim2;
  }
  return (uint64_t)(tfrmd_desc->dim4) * tfrmd_desc->dim3 *
         CEIL(number_of_sticks, AIU_STICKS_PER_PAGE) *
         CEIL(tfrmd_desc->dim1, cells_per_stick) * AIU_PAGESIZE_IN_BYTES;
}
