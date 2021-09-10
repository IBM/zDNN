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
  // same formula for 4DFEATURE and 4DKERNEL tensors
  return (uint64_t)(tfrmd_desc->dim4) * tfrmd_desc->dim3 *
         CEIL(tfrmd_desc->dim2, AIU_STICKS_PER_PAGE) *
         CEIL(tfrmd_desc->dim1, AIU_2BYTE_CELLS_PER_STICK) *
         AIU_PAGESIZE_IN_BYTES;
}
