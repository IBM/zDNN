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
#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"
#include <string.h>

#ifdef __MVS__
#pragma export(zdnn_init_ztensor)
#pragma export(zdnn_init_quantized_ztensor)
#pragma export(zdnn_init_ztensor_with_malloc)
#pragma export(zdnn_init_quantized_ztensor_with_malloc)
#pragma export(zdnn_is_quantized_ztensor)
#pragma export(zdnn_reset_ztensor)
#endif

/// Initialize a zTensor with the pre-transformed and transformed shape
/// informations
///
/// \param[in] pre_tfrmd_desc pre-transformed shape information
/// \param[in] tfrmd_desc transformed shape information
/// \param[out] output the zdnn_ztensor struct being initialized.
///
/// \return None
///
void zdnn_init_ztensor(zdnn_tensor_desc *pre_tfrmd_desc,
                       zdnn_tensor_desc *tfrmd_desc, zdnn_ztensor *output) {

  output->pre_transformed_desc = pre_tfrmd_desc;
  output->transformed_desc = tfrmd_desc;
  output->is_transformed = false;
  memset(&output->reserved, 0, sizeof(output->reserved));
  output->rec_scale = 0;
  output->offset = 0;
  memset(&output->reserved2, 0, sizeof(output->reserved2));
}

/// Initialize a quantized zTensor with the pre-transformed and transformed
/// shape informations. Update tfrmd_desc information to match requested
/// quantized transform type
///
/// \param[in] pre_tfrmd_desc pre-transformed shape information
/// \param[in] tfrmd_desc transformed shape information
/// \param[in] scale scale for quantized ztensor
/// \param[in] offset offset for quantized ztensor
/// \param[out] output the zdnn_ztensor struct being initialized.
///
/// \return None
///
void zdnn_init_quantized_ztensor(zdnn_tensor_desc *pre_tfrmd_desc,
                                 zdnn_tensor_desc *tfrmd_desc, float scale,
                                 float offset, zdnn_ztensor *output) {

  output->pre_transformed_desc = pre_tfrmd_desc;
  output->transformed_desc = tfrmd_desc;
  output->is_transformed = false;
  memset(&output->reserved, 0, sizeof(output->reserved));
  output->rec_scale = (scale != 0) ? (1 / scale) : scale;
  output->offset = offset;
  memset(&output->reserved2, 0, sizeof(output->reserved2));
}

/// @brief  Check if a given ztensor represents a quantized ztensor or not
/// @param[in] ztensor ztensor to check
///
/// @return true if ztensor is quantized, otherwise false
bool zdnn_is_quantized_ztensor(zdnn_ztensor *ztensor) {
  return (ztensor->rec_scale != 0);
}

/// Reset a zTensor for reuse
///
/// \param[out] ztensor the zdnn_ztensor struct being reset.
///
/// \return None
///
void zdnn_reset_ztensor(zdnn_ztensor *ztensor) {
  ztensor->is_transformed = false;
}

/// Convenience function for initializing a zTensor and allocating a buffer for
/// storing transformed tensor data
///
/// \param[in] pre_tfrmd_desc pre-transformed shape information
/// \param[in] tfrmd_desc transformed shape information
/// \param[out] output the zdnn_ztensor struct being initialized.
///
/// \return ZDNN_OK
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_SHAPE
///         ZDNN_ALLOCATION_FAILURE
///
zdnn_status zdnn_init_ztensor_with_malloc(zdnn_tensor_desc *pre_tfrmd_desc,
                                          zdnn_tensor_desc *tfrmd_desc,
                                          zdnn_ztensor *output) {
  zdnn_init_ztensor(pre_tfrmd_desc, tfrmd_desc, output);
  return zdnn_allochelper_ztensor(output);
}

/// Convenience function for initializing a quantized zTensor and allocating a
/// buffer for storing transformed tensor data
///
/// \param[in] pre_tfrmd_desc pre-transformed shape information
/// \param[in] tfrmd_desc transformed shape information
/// \param[in] scale scale for quantized ztensor
/// \param[in] offset offset for quantized ztensor
/// \param[in] transform_type type of quantized transformation
/// \param[out] output the zdnn_ztensor struct being initialized.
///
/// \return ZDNN_OK
///         ZDNN_INVALID_TRANSFORM_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_SHAPE
///         ZDNN_ALLOCATION_FAILURE
///
zdnn_status zdnn_init_quantized_ztensor_with_malloc(
    zdnn_tensor_desc *pre_tfrmd_desc, zdnn_tensor_desc *tfrmd_desc, float scale,
    float offset, zdnn_ztensor *output) {
  zdnn_init_quantized_ztensor(pre_tfrmd_desc, tfrmd_desc, scale, offset,
                              output);
  return zdnn_allochelper_ztensor(output);
}
