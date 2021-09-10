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

#include "zdnn.h"
#include "zdnn_private.h"
#include <string.h>

/// Convenience wrapper for AIU ops that don't need function specific parameters
zdnn_status aiu_ops(uint8_t function_code, const zdnn_ztensor *input1,
                    const zdnn_ztensor *input2, const zdnn_ztensor *input3,
                    zdnn_ztensor *output1, zdnn_ztensor *output2) {
  return aiu_ops_func_specific(function_code, input1, input2, input3, output1,
                               output2, 0, 0, 0, 0, 0, 0);
}

/// Common routine for invoking AIU operations with function specific
/// parameters
///
/// \note Caller MUST set the unused input parameters to NULL (for pointers) or
///       0 (for ints)
///
/// \param[in] function_code          NNPA function code
/// \param[in] input1
/// \param[in] input2
/// \param[in] input3
/// \param[in] output1
/// \param[in] output2
/// \param[in] func_sp_savearea_addr  Function-specific-save-area-address
/// \param[in] func_sp_parm1          Function-specific-parameter-1
/// \param[in] func_sp_parm2          Function-specific-parameter-2
/// \param[in] func_sp_parm3          Function-specific-parameter-3
/// \param[in] func_sp_parm4          Function-specific-parameter-4
/// \param[in] func_sp_parm5          Function-specific-parameter-5
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status
aiu_ops_func_specific(uint8_t function_code, const zdnn_ztensor *input1,
                      const zdnn_ztensor *input2, const zdnn_ztensor *input3,
                      zdnn_ztensor *output1, zdnn_ztensor *output2,
                      uint64_t func_sp_savearea_addr, uint32_t func_sp_parm1,
                      uint32_t func_sp_parm2, uint32_t func_sp_parm3,
                      uint32_t func_sp_parm4, uint32_t func_sp_parm5) {
  zdnn_status status;
  uint8_t ef = 0;
#define EF_RANGE_VIOLATION_MASK 0x80

  if (precheck_enabled) {
    // some ops use their own verifier.  for everything else use the simple one.
    switch (function_code) {
    case NNPA_BATCHNORMALIZATION:
      if ((status = verify_batchnorm_tensors(input1, input2, input3,
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
      if ((status = verify_matmul_op_tensors(input1, input2, input3,
                                             output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_MATMUL_OP_BCAST23:
      if ((status = verify_matmul_bcast_op_tensors(input1, input2, input3,
                                                   output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_AVGPOOL2D:
    case NNPA_MAXPOOL2D:
      // The switch in arg order below is intentional. NNPA func_sp_params2-5
      // are ordered stride_w, stride_h, kernel_w, kernel_h but our function
      // expects a different order.
      if ((status = verify_pool_avg_max_tensors(
               input1, func_sp_parm1, func_sp_parm5, func_sp_parm4,
               func_sp_parm3, func_sp_parm2, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_CONVOLUTION:
      // verify_conv2d_tensors() expects (height, width) order, thus
      // (func_sp_parm3, func_sp_parm2)
      if ((status = verify_conv2d_tensors(input1, input2, input3, func_sp_parm1,
                                          func_sp_parm3, func_sp_parm2,
                                          func_sp_parm4, output1)) != ZDNN_OK) {
        return status;
      }
      break;
    case NNPA_RELU:
      if ((status = verify_relu_tensors(input1, func_sp_parm1, output1)) !=
          ZDNN_OK) {
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

  if (function_code == NNPA_SOFTMAX && func_sp_savearea_addr == 0) {
    if (!(savearea_addr = malloc_aligned_4k(ZDNN_SOFTMAX_SAVEAREA_SIZE))) {
      return ZDNN_STATUS(ZDNN_ALLOCATION_FAILURE,
                         "Unable to allocate %" PRIu64 " bytes for save area.",
                         ZDNN_SOFTMAX_SAVEAREA_SIZE);
    }
  }

  nnpa_parameter_block parm_block;

  populate_nnpa_parm_block(&parm_block, input1, input2, input3, output1,
                           output2, savearea_addr, func_sp_parm1, func_sp_parm2,
                           func_sp_parm3, func_sp_parm4, func_sp_parm5);
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
                               AIU operation returned a RANGE VIOLATION, set as
                               a warning code and continue processing */
    } else if (ef & !EF_RANGE_VIOLATION_MASK) {
      return status = ZDNN_STATUS(ZDNN_UNSUPPORTED_AIU_EXCEPTION,
                                  "Unsupported exception on ZDNN operation",
                                  NO_ARG); /* AIU operation returned an
                               unexpected exception, return as a failure */
    }
    output1->is_transformed = true;
    if (function_code == NNPA_LSTMACT) {
      output2->is_transformed = true;
    }
  }

  return status;
}
