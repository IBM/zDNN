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
// CONCAT_LSTM usage
// ***************************************************************************
int main(int argc, char *argv[]) {
  zdnn_tensor_desc *pre_tfrmd_desc, *tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  uint32_t dim2 = 32, dim1 = 3;
  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes
  uint64_t num_elements = dim2 * dim1;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  void *data_forget = malloc(num_elements * element_size),
       *data_input = malloc(num_elements * element_size),
       *data_cell = malloc(num_elements * element_size),
       *data_output = malloc(num_elements * element_size);

  pre_tfrmd_desc = malloc(sizeof(zdnn_tensor_desc));
  tfrmd_desc = malloc(sizeof(zdnn_tensor_desc));

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, pre_tfrmd_desc, dim2, dim1);
  status = zdnn_generate_transformed_desc_concatenated(pre_tfrmd_desc,
                                                       CONCAT_LSTM, tfrmd_desc);
  assert(status == ZDNN_OK);

  ztensor.pre_transformed_desc = pre_tfrmd_desc;
  ztensor.transformed_desc = tfrmd_desc;

  status = zdnn_allochelper_ztensor(&ztensor);
  assert(status == ZDNN_OK);

  // gate buffers must be supplied in Forget, Input, Cell, Output (FICO) order
  status = zdnn_transform_ztensor(&ztensor, data_forget, data_input, data_cell,
                                  data_output);
  assert(status == ZDNN_OK);

  free(pre_tfrmd_desc);
  free(tfrmd_desc);
  free(data_forget);
  free(data_input);
  free(data_cell);
  free(data_output);
}
