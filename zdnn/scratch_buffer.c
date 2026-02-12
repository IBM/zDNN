// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024
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

#include <stddef.h>

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_reserve_scratch_buffer)
#pragma export(zdnn_release_scratch_buffer)
#endif

zdnn_status zdnn_reserve_scratch_buffer(zdnn_scratch_buffer *scratch,
                                        uint64_t min_size) {
  if (!scratch) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // No-op requests are treated as success to simplify caller control flow.
  if (min_size == 0) {
    return ZDNN_STATUS_OK;
  }

  // Existing allocation already satisfies the request.
  if (scratch->buffer && scratch->size >= min_size) {
    return ZDNN_STATUS_OK;
  }

  if (min_size > SIZE_MAX) {
    return ZDNN_STATUS_NO_MSG(ZDNN_ALLOCATION_FAILURE);
  }

  void *new_buffer = malloc_aligned_4k((size_t)min_size);
  if (!new_buffer) {
    return ZDNN_STATUS_NO_MSG(ZDNN_ALLOCATION_FAILURE);
  }

  if (scratch->buffer) {
    free_aligned_4k(scratch->buffer);
  }

  scratch->buffer = new_buffer;
  scratch->size = min_size;

  return ZDNN_STATUS_OK;
}

void zdnn_release_scratch_buffer(zdnn_scratch_buffer *scratch) {
  if (!scratch) {
    return;
  }

  if (scratch->buffer) {
    free_aligned_4k(scratch->buffer);
  }

  scratch->buffer = NULL;
  scratch->size = 0;
}
