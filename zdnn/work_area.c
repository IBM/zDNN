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

#include <stdint.h>

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_get_lstm_work_area_size)
#pragma export(zdnn_get_gru_work_area_size)
#endif

// Common helper for LSTM/GRU work-area sizing.
//
// The work area is internal scratch space used by the RNN implementations for
// temporary ztensors. The required size is described in README.md; these helper
// functions codify that formula to reduce caller error and avoid repeating the
// descriptor math in every integration.
static uint64_t get_rnn_work_area_size(uint32_t num_timesteps, uint32_t dim4_mul,
                                       uint32_t dim4_add, uint32_t num_batches,
                                       uint32_t num_hidden,
                                       lstm_gru_direction direction) {
  if (!num_timesteps || !num_batches || !num_hidden) {
    return 0;
  }

  uint32_t num_dirs;
  switch (direction) {
  case FWD:
  case BWD:
    num_dirs = 1;
    break;
  case BIDIR:
    num_dirs = 2;
    break;
  default:
    return 0;
  }

  // Compute dim4 using a wider intermediate to avoid overflow.
  uint64_t dim4_64 = ((uint64_t)dim4_mul * (uint64_t)num_timesteps) + dim4_add;
  if (dim4_64 > UINT32_MAX) {
    return 0;
  }

  zdnn_tensor_desc desc;
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE, &desc,
                        (uint32_t)dim4_64, 1, num_batches, num_hidden);

  uint64_t size = zdnn_getsize_ztensor(&desc);

  // For bidirectional calls, twice the amount of contiguous storage is
  // required.
  if (num_dirs == 2) {
    if (size > (UINT64_MAX / 2)) {
      return 0;
    }
    size *= 2;
  }

  return size;
}

uint64_t zdnn_get_lstm_work_area_size(uint32_t num_timesteps, uint32_t num_batches,
                                      uint32_t num_hidden,
                                      lstm_gru_direction direction) {
  // LSTM sizing (README): dim4 = (4 * num_timesteps) + 6
  return get_rnn_work_area_size(num_timesteps, 4, 6, num_batches, num_hidden,
                                direction);
}

uint64_t zdnn_get_gru_work_area_size(uint32_t num_timesteps, uint32_t num_batches,
                                     uint32_t num_hidden,
                                     lstm_gru_direction direction) {
  // GRU sizing (README): dim4 = (3 * num_timesteps) + 5
  return get_rnn_work_area_size(num_timesteps, 3, 5, num_batches, num_hidden,
                                direction);
}
