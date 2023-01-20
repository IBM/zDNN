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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// ***************************************************************************
// Sample:
//
// Descriptor allocation on heap
// ***************************************************************************
int main(int argc, char *argv[]) {
  zdnn_tensor_desc *pre_tfrmd_desc, *tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  uint32_t dim_n = 1, dim_h = 32, dim_w = 32, dim_c = 3;
  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes
  uint64_t num_elements = dim_n * dim_h * dim_w * dim_c;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  void *data = malloc(num_elements * element_size);

  pre_tfrmd_desc = malloc(sizeof(zdnn_tensor_desc));
  tfrmd_desc = malloc(sizeof(zdnn_tensor_desc));

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, type, pre_tfrmd_desc, dim_n, dim_h,
                                 dim_w, dim_c);
  status = zdnn_generate_transformed_desc(pre_tfrmd_desc, tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(pre_tfrmd_desc, tfrmd_desc, &ztensor);
  assert(status == ZDNN_OK);

  status = zdnn_transform_ztensor(&ztensor, data);
  assert(status == ZDNN_OK);

  free(pre_tfrmd_desc);
  free(tfrmd_desc);
  free(data);
}
