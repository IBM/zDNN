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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// ***************************************************************************
// Sample:
//
// Create 2 zTensors a and b, and add them together via zdnn_add()
// ***************************************************************************
int main(int argc, char *argv[]) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor_a;
  zdnn_ztensor ztensor_b;
  zdnn_ztensor ztensor_out;
  zdnn_status status;

  uint32_t dim_n = 1, dim_h = 32, dim_w = 32, dim_c = 3;
  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes
  uint64_t num_elements = dim_n * dim_h * dim_w * dim_c;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  // allocate tensor data storage
  void *data1 = malloc(num_elements * element_size);
  void *data2 = malloc(num_elements * element_size);
  void *data_out = malloc(num_elements * element_size);

  // read input_data

  // check status for AIU availability, supported ops, etc. here
  // status = zdnn_query(…);

  // set input tensor data to 0 to 127 sequentially and repeat
  for (uint64_t i = 0; i < num_elements; i++) {
    ((float *)data1)[i] = (float)(i & 0x7f);
    ((float *)data2)[i] = (float)(i & 0x7f);
  }

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, type, &pre_tfrmd_desc, dim_n, dim_h,
                                 dim_w, dim_c);
  // generate transformed shape information
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  assert(status == ZDNN_OK);

  // initialize zTensors and allocate 4k-aligned storage via helper function
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor_a);
  assert(status == ZDNN_OK);
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor_b);
  assert(status == ZDNN_OK);
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor_out);
  assert(status == ZDNN_OK);

  // transform the feature tensor
  status = zdnn_transform_ztensor(&ztensor_a, data1);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&ztensor_b, data2);
  assert(status == ZDNN_OK);

  // perform element-wise add between the two input tensors
  status = zdnn_add(&ztensor_a, &ztensor_b, &ztensor_out);
  assert(status == ZDNN_OK);

  // transform resultant zTensor back to original data format
  status = zdnn_transform_origtensor(&ztensor_out, data_out);
  assert(status == ZDNN_OK);

  for (uint64_t i = 0; i < num_elements; i++) {
    printf("out element %" PRIu64 " %f\n", i, ((float *)data_out)[i]);
  }

  // Free zTensors
  status = zdnn_free_ztensor_buffer(&ztensor_a);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&ztensor_b);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&ztensor_out);
  assert(status == ZDNN_OK);

  free(data1);
  free(data2);
  free(data_out);
}
