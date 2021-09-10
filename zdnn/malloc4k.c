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
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/// malloc() that does 4k-alignment
///
/// \param[in] size Size to be malloc'd
///
/// \return Pointer to the malloc'd area if successful, or NULL otherwise
///
void *malloc_aligned_4k(size_t size) {

  // request one more page + size of a pointer from the OS
  unsigned short extra_allocation =
      (AIU_PAGESIZE_IN_BYTES - 1) + sizeof(void *);

  // make sure size is reasonable
  if (!size || size > SIZE_MAX) {
    return NULL;
  }

  void *ptr = malloc(size + extra_allocation);
  if (!ptr) {
    perror("Error during malloc");
    fprintf(stderr, "errno = %d\n", errno);
    return ptr;
  }

  // find the 4k boundary after ptr
  void *aligned_ptr = (void *)(((uintptr_t)ptr + extra_allocation) &
                               ~(AIU_PAGESIZE_IN_BYTES - 1));
  // put the original malloc'd address right before aligned_ptr
  ((void **)aligned_ptr)[-1] = ptr;

  LOG_DEBUG("malloc_aligned_4k() malloc() at %016lx, aligned at %016lx, of "
            "size %zu",
            (uintptr_t)ptr, (uintptr_t)aligned_ptr, size);

  return aligned_ptr;
}

/// free() what was allocated via malloc_aligned_4k()
///
/// \param[in] ptr Pointer returned by malloc_aligned_4k()
///
/// \return None
///
void free_aligned_4k(void *aligned_ptr) {
  if (aligned_ptr) {
    // get the original malloc'd address from where we put it and free it
    void *original_ptr = ((void **)aligned_ptr)[-1];
    LOG_DEBUG("free_aligned_4k() aligned_ptr = %016lx original_ptr = %016lx",
              (uintptr_t)aligned_ptr, (uintptr_t)original_ptr);
    free(original_ptr);
  }
}
