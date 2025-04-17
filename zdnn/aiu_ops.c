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

#include "zdnn.h"
#include "zdnn_private.h"
#include <string.h>

/// Convenience wrapper for zAIU ops that don't need function specific
/// parameters
zdnn_status aiu_ops(uint16_t op_parm_block_version, uint8_t function_code,
                    const zdnn_ztensor *input1, const zdnn_ztensor *input2,
                    const zdnn_ztensor *input3, zdnn_ztensor *output1,
                    zdnn_ztensor *output2) {
  function_specific_parameters fsp = {0};
  return aiu_ops_func_specific(op_parm_block_version, function_code, input1,
                               input2, input3, output1, output2, 0, &fsp);
}

/// Common routine for invoking zAIU operations with function specific
/// parameters
///
/// \note Caller MUST set the unused input parameters to NULL (for pointers) or
///       0 (for ints)
///
/// \param[in] op_parm_block_version  Parmblock version
/// \param[in] function_code          NNPA function code
/// \param[in] input1
/// \param[in] input2
/// \param[in] input3
/// \param[in] output1
/// \param[in] output2
/// \param[in] func_sp_savearea_addr  Function-specific-save-area-address
/// \param[in] fsp                    Functions specific parameters struct
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status
aiu_ops_func_specific(uint16_t op_parm_block_version, uint8_t function_code,
                      const zdnn_ztensor *input1, const zdnn_ztensor *input2,
                      const zdnn_ztensor *input3, zdnn_ztensor *output1,
                      zdnn_ztensor *output2, uint64_t func_sp_savearea_addr,
                      function_specific_parameters *fsp) {
  zdnn_status status;
  uint8_t ef = 0;
#define EF_RANGE_VIOLATION_MASK 0x80

  if (!is_query_parmblock_installed(op_parm_block_version)) {
    return ZDNN_UNAVAILABLE_FUNCTION;
  }

  if (precheck_enabled) {
    // some ops use their own verifier.  for everything else use the simple one.
    switch (function_code) {
    case NNPA_BATCHNORMALIZATION:
      if ((status = verify_batchnorm_tensors(input1, input2, input3,
                                             output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_NORM:
      if ((status = verify_norm_tensors(input1, input2, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_MOMENTS:
      if ((status =
               verify_moments_tensors(input1, &fsp->function_specific_parm1,
                                      output1, output2)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_LAYERNORM:
      if ((status = verify_layernorm_tensors(
               input1, input2, input3, &fsp->function_specific_parm1,
               &fsp->function_specific_parm2, &fsp->function_specific_parm3,
               output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_LSTMACT:
    case NNPA_GRUACT:
      if ((status = verify_lstm_or_gru_act_tensors(function_code, input1,
                                                   input2, input3, output1,
                                                   output2)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_MATMUL_OP:
    case NNPA_MATMUL_OP_BCAST23:
    case NNPA_MATMUL_OP_BCAST1:
      if ((status = verify_matmul_op_common(
               function_code, input1, input2, input3,
               &fsp->function_specific_parm2, &fsp->function_specific_parm3,
               &fsp->function_specific_parm4, &fsp->function_specific_parm9,
               &fsp->function_specific_parm10, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_AVGPOOL2D:
    case NNPA_MAXPOOL2D:
      if ((status = verify_pool_avg_max_tensors(
               input1, &fsp->function_specific_parm1,
               &fsp->function_specific_parm2, &fsp->function_specific_parm3,
               &fsp->function_specific_parm4, &fsp->function_specific_parm5,
               output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_CONVOLUTION:
      if ((status = verify_conv2d_tensors(
               input1, input2, input3, &fsp->function_specific_parm1,
               &fsp->function_specific_parm2, &fsp->function_specific_parm3,
               &fsp->function_specific_parm4, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_RELU:
      if ((status = verify_relu_tensors(input1, &fsp->function_specific_parm1,
                                        &fsp->function_specific_parm2,
                                        output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_TRANSFORM:
      if ((status = verify_transform_tensors(
               input1, output1, &fsp->function_specific_parm1,
               &fsp->function_specific_parm4, &fsp->function_specific_parm5)) !=
          ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_INVSQRT:
      if ((status = verify_invsqrt_tensors(
               input1, &fsp->function_specific_parm1, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_REDUCE:
      if ((status = verify_reduce_tensors(input1, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    default:
      if ((status = verify_tensors(input1, input2, input3, output1)) !=
          ZDNN_OK) {
        return status;
      }
      break;
    }
  }

  // SOFTMAX requires a 4k-aligned save area.  Either use caller's or allocate
  // our own.
  void *savearea_addr = (void *)func_sp_savearea_addr;

  if ((function_code == NNPA_SOFTMAX || function_code == NNPA_REDUCE) &&
      func_sp_savearea_addr == 0) {
    if (!(savearea_addr = malloc_aligned_4k(ZDNN_8K_SAVEAREA_SIZE))) {
      return ZDNN_STATUS(ZDNN_ALLOCATION_FAILURE,
                         "Unable to allocate %" PRIu64 " bytes for save area.",
                         ZDNN_8K_SAVEAREA_SIZE);
    }
  }

  nnpa_parameter_block parm_block;

  populate_nnpa_parm_block(&parm_block, op_parm_block_version, input1, input2,
                           input3, output1, output2, savearea_addr, fsp);
  status = invoke_nnpa(function_code, (char *)&parm_block, &ef);

  // free the savearea if using our own, regardless of op
  if (savearea_addr && !func_sp_savearea_addr) {
    free_aligned_4k(savearea_addr);
  }

  // Indicate output tensor is stickified only if invoke_nnpa() was OK
  if (status == ZDNN_OK) {
    if (ef & EF_RANGE_VIOLATION_MASK) {
      status =
          ZDNN_STATUS(ZDNN_ELEMENT_RANGE_VIOLATION,
                      "Range violation on tensor data", NO_ARG); /*
                               zAIU operation returned a RANGE VIOLATION, set
                               as a warning code and continue processing */
    } else if (ef & ~EF_RANGE_VIOLATION_MASK) {
      return status = ZDNN_STATUS(ZDNN_UNSUPPORTED_AIU_EXCEPTION,
                                  "Unsupported exception on ZDNN operation",
                                  NO_ARG); /* zAIU operation returned an
                               unexpected exception, return as a failure */
    }
    output1->is_transformed = true;
    if (function_code == NNPA_LSTMACT) {
      output2->is_transformed = true;
    }
  }

  return status;
}
