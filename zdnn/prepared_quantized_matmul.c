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

#include <string.h>

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_prepare_quantized_matmul_context)
#pragma export(zdnn_quantized_matmul_op_prepared)
#pragma export(zdnn_release_quantized_matmul_context)
#endif

#define ZDNN_PREPARED_QMATMUL_CONTEXT_MAGIC UINT64_C(0x514D415450524550)

static bool has_valid_tfrmd_desc(const zdnn_ztensor *ztensor) {
  return ztensor && ztensor->transformed_desc;
}

static zdnn_status verify_prepared_desc_match(
    const char *tensor_name, const zdnn_tensor_desc *expected,
    const zdnn_ztensor *actual_tensor) {
  if (!has_valid_tfrmd_desc(actual_tensor)) {
    return ZDNN_STATUS(ZDNN_INVALID_BUFFER,
                       "%s ztensor or transformed descriptor is NULL",
                       tensor_name);
  }

  const zdnn_tensor_desc *actual = actual_tensor->transformed_desc;

  if (actual->layout != expected->layout) {
    return ZDNN_STATUS(
        ZDNN_INVALID_LAYOUT,
        "%s transformed layout mismatch (found %s, expected %s)", tensor_name,
        get_data_layout_str(actual->layout), get_data_layout_str(expected->layout));
  }

  if (actual->format != expected->format) {
    return ZDNN_STATUS(
        ZDNN_INVALID_FORMAT,
        "%s transformed format mismatch (found %s (%d), expected %s (%d))",
        tensor_name, get_data_format_str(actual->format), actual->format,
        get_data_format_str(expected->format), expected->format);
  }

  if (actual->type != expected->type) {
    return ZDNN_STATUS(
        ZDNN_INVALID_TYPE, "%s transformed type mismatch (found %s (%d), "
                           "expected %s (%d))",
        tensor_name, get_data_type_str(actual->type), actual->type,
        get_data_type_str(expected->type), expected->type);
  }

  if (actual->dim4 != expected->dim4 || actual->dim3 != expected->dim3 ||
      actual->dim2 != expected->dim2 || actual->dim1 != expected->dim1) {
    return ZDNN_STATUS(
        ZDNN_INVALID_SHAPE,
        "%s transformed dims mismatch (found [%u,%u,%u,%u], expected "
        "[%u,%u,%u,%u])",
        tensor_name, actual->dim4, actual->dim3, actual->dim2, actual->dim1,
        expected->dim4, expected->dim3, expected->dim2, expected->dim1);
  }

  return ZDNN_STATUS_OK;
}

zdnn_status zdnn_prepare_quantized_matmul_context(
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
    const zdnn_ztensor *input_c, bool pre_computed,
    zdnn_quantized_matmul_prepared_context *context) {
  if (!context || !has_valid_tfrmd_desc(input_a) ||
      !has_valid_tfrmd_desc(input_b) || !has_valid_tfrmd_desc(input_c)) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  // Only attempt to reuse/release previous scratch if this context was
  // initialized by this API previously.
  if (context->reserved_internal != ZDNN_PREPARED_QMATMUL_CONTEXT_MAGIC) {
    memset(context, 0, sizeof(*context));
  }

  context->function_code = get_matmul_function(input_a->transformed_desc->dim4,
                                               input_b->transformed_desc->dim4);
  context->pre_computed = pre_computed;
  context->input_a_desc = *input_a->transformed_desc;
  context->input_b_desc = *input_b->transformed_desc;
  context->input_c_desc = *input_c->transformed_desc;
  context->required_work_area_size =
      zdnn_get_quantized_matmul_work_area_size(input_c, pre_computed);

  // In non-precomputed mode input_c must be transformed int8 for qc_tilde
  // scratch sizing and later execution.
  if (!pre_computed &&
      input_c->transformed_desc->type != ZDNN_BINARY_INT8) {
    return ZDNN_STATUS(
        ZDNN_INVALID_TYPE,
        "input_c tensor type is invalid for prepared quantized matmul "
        "(found %s (%d), expects ZDNN_BINARY_INT8 (8))",
        get_data_type_str(input_c->transformed_desc->type),
        input_c->transformed_desc->type);
  }

  if (context->required_work_area_size == 0) {
    // Precomputed path does not require internal work area.
    zdnn_release_scratch_buffer(&context->work_area_scratch);
  } else {
    zdnn_status reserve_status = zdnn_reserve_scratch_buffer(
        &context->work_area_scratch, context->required_work_area_size);
    if (reserve_status != ZDNN_OK) {
      return reserve_status;
    }
  }

  context->reserved_internal = ZDNN_PREPARED_QMATMUL_CONTEXT_MAGIC;
  return ZDNN_STATUS_OK;
}

zdnn_status zdnn_quantized_matmul_op_prepared(
    const zdnn_quantized_matmul_prepared_context *context,
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
    const zdnn_ztensor *input_c, zdnn_matmul_ops op_type, const int8_t clip_min,
    const int8_t clip_max, const bool disable_clipping, const bool dequantize,
    zdnn_ztensor *output) {
  if (!context || context->reserved_internal !=
                      ZDNN_PREPARED_QMATMUL_CONTEXT_MAGIC ||
      !output) {
    return ZDNN_STATUS_NO_MSG(ZDNN_INVALID_BUFFER);
  }

  zdnn_status status = verify_prepared_desc_match("input_a", &context->input_a_desc,
                                                  input_a);
  if (status != ZDNN_OK) {
    return status;
  }

  status = verify_prepared_desc_match("input_b", &context->input_b_desc, input_b);
  if (status != ZDNN_OK) {
    return status;
  }

  status = verify_prepared_desc_match("input_c", &context->input_c_desc, input_c);
  if (status != ZDNN_OK) {
    return status;
  }

  if (context->pre_computed && input_b->offset != 0.f) {
    return ZDNN_STATUS(ZDNN_INVALID_OFFSET,
                       "input_b offset (Zb) is invalid when pre_computed=true "
                       "(found %f, expects %f)",
                       input_b->offset, 0.f);
  }

  void *work_area = NULL;
  if (!context->pre_computed) {
    work_area = context->work_area_scratch.buffer;
    if (!work_area ||
        context->work_area_scratch.size < context->required_work_area_size) {
      return ZDNN_STATUS(
          ZDNN_INVALID_BUFFER,
          "prepared quantized matmul context work_area is invalid "
          "(buffer=%p, size=%" PRIu64 ", required=%" PRIu64 ")",
          work_area, context->work_area_scratch.size,
          context->required_work_area_size);
    }
  }

  return aiu_quantized_matmul(
      NNPA_PARMBLKFORMAT_1, context->function_code, input_a, input_b, input_c,
      op_type, clip_min, clip_max, work_area, output, dequantize,
      disable_clipping, context->pre_computed);
}

void zdnn_release_quantized_matmul_context(
    zdnn_quantized_matmul_prepared_context *context) {
  if (!context) {
    return;
  }

  if (context->reserved_internal == ZDNN_PREPARED_QMATMUL_CONTEXT_MAGIC) {
    zdnn_release_scratch_buffer(&context->work_area_scratch);
  }

  memset(context, 0, sizeof(*context));
}
