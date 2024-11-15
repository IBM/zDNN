// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2023, 2024
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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// ***************************************************************************
// Sample:
//
// Create a quantized zTensors
// ***************************************************************************
int main(int argc, char *argv[]) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  uint32_t dim_n = 1, dim_h = 32, dim_w = 32, dim_c = 3;
  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes
  uint64_t num_elements = dim_n * dim_h * dim_w * dim_c;

  // allocate tensor data storage
  void *data1 = malloc(num_elements * element_size);

  // read input_data

  // check status for zAIU availability, supported ops, etc. here
  // status = zdnn_query();

  // set input tensor data to 0 to 127 sequentially and repeat
  for (uint64_t i = 0; i < num_elements; i++) {
    ((float *)data1)[i] = (float)(i & 0x7f);
  }

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, type, &pre_tfrmd_desc, dim_n, dim_h,
                                 dim_w, dim_c);
  float scale = 3;
  float offset = 2;

  // generate transformed shape information
  status = zdnn_generate_quantized_transformed_desc(
      &pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &tfrmd_desc);
  assert(status == ZDNN_OK);

  // initialize zTensors and allocate 4k-aligned storage via helper function
  status = zdnn_init_quantized_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc,
                                                   scale, offset, &ztensor);
  assert(status == ZDNN_OK);

  // transform the feature tensor
  status = zdnn_transform_ztensor(&ztensor, data1);
  assert(status == ZDNN_OK);

  // Free zTensors
  status = zdnn_free_ztensor_buffer(&ztensor);
  assert(status == ZDNN_OK);

  free(data1);
}