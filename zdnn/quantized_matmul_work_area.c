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
#pragma export(zdnn_get_quantized_matmul_work_area_size)
#endif

uint64_t zdnn_get_quantized_matmul_work_area_size(const zdnn_ztensor *input_c,
                                                  bool pre_computed) {
  // pre_computed mode does not create qc_tilde and therefore does not require
  // quantized-matmul work area scratch storage.
  if (pre_computed) {
    return 0;
  }

  if (!input_c || !input_c->transformed_desc) {
    return 0;
  }

  // aiu_quantized_matmul() requires input_c transformed type to be int8 when
  // pre_computed is false.
  if (input_c->transformed_desc->type != ZDNN_BINARY_INT8) {
    return 0;
  }

  // The internal temporary tensor (qc_tilde) reuses input_c's transformed
  // dimensions/layout/format but promotes the element type to DLFLOAT16.
  zdnn_tensor_desc qc_tilde_desc;
  init_transformed_desc(
      input_c->transformed_desc->layout, ZDNN_DLFLOAT16,
      input_c->transformed_desc->format, &qc_tilde_desc,
      input_c->transformed_desc->dim4, input_c->transformed_desc->dim3,
      input_c->transformed_desc->dim2, input_c->transformed_desc->dim1);

  return zdnn_getsize_ztensor(&qc_tilde_desc);
}
