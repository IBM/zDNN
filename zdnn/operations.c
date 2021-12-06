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

#include "convert.h"
#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_add)
#pragma export(zdnn_sub)
#pragma export(zdnn_mul)
#pragma export(zdnn_div)
#pragma export(zdnn_min)
#pragma export(zdnn_max)
#pragma export(zdnn_log)
#pragma export(zdnn_exp)
#pragma export(zdnn_relu)
#pragma export(zdnn_tanh)
#pragma export(zdnn_sigmoid)
#pragma export(zdnn_softmax)
#pragma export(zdnn_lstm)
#pragma export(zdnn_gru)
#pragma export(zdnn_matmul_op)
#pragma export(zdnn_matmul_bcast_op)
#pragma export(zdnn_batchnorm)
#pragma export(zdnn_meanreduce2d)
#pragma export(zdnn_avgpool2d)
#pragma export(zdnn_maxpool2d)
#pragma export(zdnn_conv2d)
#endif

#define BEGIN_PRINT_PARMS                                                      \
  printf("\n%s parameters start "                                              \
         ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",               \
         __func__);

#define PRINT_PARM_ZTENSOR_PTR(ztnsr) print_ztensor(ztnsr, #ztnsr, false);
#define PRINT_PARM_PTR(ptr)                                                    \
  printf("\nParameter %s (pointer): %" PRIxPTR "\n", #ptr, (uintptr_t)ptr);
#define PRINT_PARM_RNN_DIR(dir)                                                \
  printf("\nDirection: %s\n", get_rnn_direction_str(dir));
#define PRINT_PARM_FLOAT_PTR(val)                                              \
  printf("\nParameter %s (float): %f\n", #val, val);
#define PRINT_PARM_UINT32T(val)                                                \
  printf("\nParameter %s (uint32_t): %u\n", #val, val);
#define PRINT_PARM_UINT64T(val)                                                \
  printf("\nParameter %s (uint64_t): %" PRIu64 "\n", #val, val);
#define PRINT_PARM_SOFTMAX_ACT(func)                                           \
  printf("\nSoftmax Activation Function: %s\n", get_softmax_act_str(func));
#define PRINT_PARM_MATMUL_OP(op)                                               \
  printf("\nMatmul Operation: %s\n", get_matmul_op_str(op));
#define PRINT_PARM_MATMUL_BCAST_OP(op)                                         \
  printf("\nMatmul Bcast Operation: %s\n", get_matmul_bcast_op_str(op));
#define PRINT_PARM_POOL_PADDING(pad)                                           \
  printf("\nPool padding: %s\n", get_pool_padding_str(pad));
#define PRINT_PARM_CONV2D_ACT(func)                                            \
  printf("\nConv2D Activation Function: %s\n", get_conv2d_act_str(func));

#define END_PRINT_PARMS                                                        \
  printf("\n%s parameters end "                                                \
         "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n",               \
         __func__);

// -----------------------------------------------------------------------------
// External Activation Operations
// -----------------------------------------------------------------------------

/// External interface for Relu operation
///
/// \param[in] input The input tensor
/// \param[in] clipping_value A pointer to an FP32 clipping value
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_relu(const zdnn_ztensor *input, const void *clipping_value,
                      zdnn_ztensor *output) {

  // Create Function Specific Parm 1 for Relu first to optimize conditional
  // checks.
  func_sp_parm1_relu parm1;
  parm1.val = 0;

  // Create variable for parameter output. Check if value is NULL, followed by a
  // check if it is not 0. If it is 0 it is unnecessary to convert 0 to DLFloat
  // or setting clipping_value (as it is already set by val)
  float clip_val = 0;
  if (clipping_value) {
    clip_val = *(float *)clipping_value;
    if (clip_val != 0) {
      parm1.bits.clipping_value = cnvt_1_fp32_to_dlf16(clip_val);
    }
  }

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_FLOAT_PTR(clip_val);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-1: clipping value
  return aiu_ops_func_specific(NNPA_RELU, input, NULL, NULL, output, NULL, 0,
                               parm1.val, 0, 0, 0, 0);
}

/// External interface for Tanh operation
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_tanh(const zdnn_ztensor *input, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_TANH, input, NULL, NULL, output, NULL);
}

/// External interface for Sigmoid operation
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_sigmoid(const zdnn_ztensor *input, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_SIGMOID, input, NULL, NULL, output, NULL);
}

/// External interface for Softmax operation
///
/// \param[in] input The input tensor
/// \param[in] save_area Pointer to the save area required by NNPA_SOFTMAX
/// \param[in] act_func activation function as specified in the zdnn_softmax_act
/// enum
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_softmax(const zdnn_ztensor *input, void *save_area,
                         zdnn_softmax_act act_func, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_PTR(save_area);
    PRINT_PARM_SOFTMAX_ACT(act_func);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  func_sp_parm1_softmax parm1;
  parm1.val = 0;
  parm1.bits.act = act_func;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: ACTIVATION function
  return aiu_ops_func_specific(NNPA_SOFTMAX, input, NULL, NULL, output, NULL,
                               (uintptr_t)save_area, parm1.val, 0, 0, 0, 0);
}

// -----------------------------------------------------------------------------
// External RNN Operations
// -----------------------------------------------------------------------------

/// External interface for LSTM operation
///
/// \param[in] input The input tensor
/// \param[in] h0 The initial hidden state tensor
/// \param[in] c0 The initial cell state tensor
/// \param[in] weights The concatenated weights tensor
/// \param[in] biases The concatenated biases tensor
/// \param[in] hidden_weights The concatenated hidden weights tensor
/// \param[in] hidden_biases The concatenated hidden biases tensor
/// \param[in] direction Direction (FWD, BWD, BIDIR)
/// \param[in] work_area Pointer to pre-allocated work area, or NULL
/// \param[out] hn_output The output hidden_state tensor
/// \param[out] cf_output The output cell_state tensor
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
zdnn_status zdnn_lstm(const zdnn_ztensor *input, const zdnn_ztensor *h0,
                      const zdnn_ztensor *c0, const zdnn_ztensor *weights,
                      const zdnn_ztensor *biases,
                      const zdnn_ztensor *hidden_weights,
                      const zdnn_ztensor *hidden_biases,
                      lstm_gru_direction direction, void *work_area,
                      zdnn_ztensor *hn_output, zdnn_ztensor *cf_output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(h0);
    PRINT_PARM_ZTENSOR_PTR(c0);
    PRINT_PARM_ZTENSOR_PTR(weights);
    PRINT_PARM_ZTENSOR_PTR(biases);
    PRINT_PARM_ZTENSOR_PTR(hidden_weights);
    PRINT_PARM_ZTENSOR_PTR(hidden_biases);
    PRINT_PARM_RNN_DIR(direction);
    PRINT_PARM_PTR(work_area);
    PRINT_PARM_ZTENSOR_PTR(hn_output);
    PRINT_PARM_ZTENSOR_PTR(cf_output);
    END_PRINT_PARMS;

    // aiu_lstm_gru() dissects the input tensors and makes multiple calls to the
    // AIU.  check the overall input tensors here and precheck will check the
    // dissected tensors later before each and every AIU call
    zdnn_status precheck_status;
    if ((precheck_status = verify_zdnn_lstm_or_gru_tensors(
             NNPA_LSTMACT, input, h0, c0, weights, biases, hidden_weights,
             hidden_biases, direction, hn_output, cf_output)) != ZDNN_OK) {
      return precheck_status;
    }
  }

  return aiu_lstm_gru(NNPA_LSTMACT, input, h0, c0, weights, biases,
                      hidden_weights, hidden_biases, direction, work_area,
                      hn_output, cf_output);
}

/// External interface for GRU operation
///
/// \param[in] input The input tensor
/// \param[in] h0 The initial hidden state tensor
/// \param[in] weights The concatenated weights tensor
/// \param[in] biases The concatenated biases tensor
/// \param[in] hidden_weights The concatenated hidden weights tensor
/// \param[in] hidden_biases The concatenated hidden biases tensor
/// \param[in] direction Direction (FWD, BWD, BIDIR)
/// \param[in] work_area Pointer to pre-allocated work area, or NULL
/// \param[out] hn_output The output hidden_state tensor
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
zdnn_status zdnn_gru(const zdnn_ztensor *input, const zdnn_ztensor *h0,
                     const zdnn_ztensor *weights, const zdnn_ztensor *biases,
                     const zdnn_ztensor *hidden_weights,
                     const zdnn_ztensor *hidden_biases,
                     lstm_gru_direction direction, void *work_area,
                     zdnn_ztensor *hn_output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(h0);
    PRINT_PARM_ZTENSOR_PTR(weights);
    PRINT_PARM_ZTENSOR_PTR(biases);
    PRINT_PARM_ZTENSOR_PTR(hidden_weights);
    PRINT_PARM_ZTENSOR_PTR(hidden_biases);
    PRINT_PARM_RNN_DIR(direction);
    PRINT_PARM_PTR(work_area);
    PRINT_PARM_ZTENSOR_PTR(hn_output);
    END_PRINT_PARMS;

    // aiu_lstm_gru() dissects the input tensors and makes multiple calls to the
    // AIU.  check the overall input tensors here and precheck will check the
    // dissected tensors later before the AIU calls
    zdnn_status precheck_status;
    if ((precheck_status = verify_zdnn_lstm_or_gru_tensors(
             NNPA_GRUACT, input, h0, NULL, weights, biases, hidden_weights,
             hidden_biases, direction, hn_output, NULL)) != ZDNN_OK) {
      return precheck_status;
    }
  }

  return aiu_lstm_gru(NNPA_GRUACT, input, h0, NULL, weights, biases,
                      hidden_weights, hidden_biases, direction, work_area,
                      hn_output, NULL);
}

// -----------------------------------------------------------------------------
// External Elementwise Operations
// -----------------------------------------------------------------------------

/// External interface for Add operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_add(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_ADD, input_a, input_b, NULL, output, NULL);
}

/// External interface for Subtract operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_sub(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_SUB, input_a, input_b, NULL, output, NULL);
}

/// External interface for Divide operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_div(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_DIV, input_a, input_b, NULL, output, NULL);
}

/// External interface for Multiply operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_mul(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_MUL, input_a, input_b, NULL, output, NULL);
}

/// External interface for Max operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_max(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_MAX, input_a, input_b, NULL, output, NULL);
}

/// External interface for Min operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_min(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_MIN, input_a, input_b, NULL, output, NULL);
}

/// External interface for Log operation
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_log(const zdnn_ztensor *input, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_LOG, input, NULL, NULL, output, NULL);
}

/// External interface for Exponential operation
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_exp(const zdnn_ztensor *input, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_EXP, input, NULL, NULL, output, NULL);
}

/// External interface for Matmul operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] op_type The operation performed against matmul dot product
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_matmul_op(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c, zdnn_matmul_ops op_type,
                           zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_MATMUL_OP(op_type);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  func_sp_parm1_matmul_op parm1;
  parm1.val = 0;
  parm1.bits.operation = op_type;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: OPERATION field
  return aiu_ops_func_specific(NNPA_MATMUL_OP, input_a, input_b, input_c,
                               output, NULL, 0, parm1.val, 0, 0, 0, 0);
}

/// External interface for Matmul Broadcast operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_matmul_bcast_op(const zdnn_ztensor *input_a,
                                 const zdnn_ztensor *input_b,
                                 const zdnn_ztensor *input_c,
                                 zdnn_matmul_bcast_ops op_type,
                                 zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_MATMUL_BCAST_OP(op_type);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  func_sp_parm1_matmul_bcast_op parm1;
  parm1.val = 0;
  parm1.bits.operation = op_type;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: OPERATION field
  return aiu_ops_func_specific(NNPA_MATMUL_OP_BCAST23, input_a, input_b,
                               input_c, output, NULL, 0, parm1.val, 0, 0, 0, 0);
}

// -----------------------------------------------------------------------------
// External Norm Operations
// -----------------------------------------------------------------------------

/// External interface for Batch Normalization operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_batchnorm(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_BATCHNORMALIZATION, input_a, input_b, input_c, output,
                 NULL);
}

// -----------------------------------------------------------------------------
// External Pool Operations
// -----------------------------------------------------------------------------

/// External interface for Average Pool 2D operation
///
/// \param[in] input The input tensor
/// \param[in] padding_type VALID_PADDING or SAME_PADDING
/// \param[in] kernel_height height of the kernel
/// \param[in] kernel_width width of the kernel
/// \param[in] stride_height height movement per kernel slide
/// \param[in] stride_width width movement per kernel slide
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_avgpool2d(const zdnn_ztensor *input,
                           zdnn_pool_padding padding_type,
                           uint32_t kernel_height, uint32_t kernel_width,
                           uint32_t stride_height, uint32_t stride_width,
                           zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_POOL_PADDING(padding_type);
    PRINT_PARM_UINT32T(kernel_height);
    PRINT_PARM_UINT32T(kernel_width);
    PRINT_PARM_UINT32T(stride_height);
    PRINT_PARM_UINT32T(stride_width);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  // The switch in arg order is intentional. The AIU op expects a different
  // order than our API.
  return aiu_ops_func_specific(NNPA_AVGPOOL2D, input, NULL, NULL, output, NULL,
                               0, padding_type, stride_width, stride_height,
                               kernel_width, kernel_height);
}

/// External interface for Max Pool 2D operation
///
/// \param[in] input The input tensor
/// \param[in] padding_type VALID_PADDING or SAME_PADDING
/// \param[in] kernel_height height of the kernel
/// \param[in] kernel_width width of the kernel
/// \param[in] stride_height height movement per kernel slide
/// \param[in] stride_width width movement per kernel slide
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_maxpool2d(const zdnn_ztensor *input,
                           zdnn_pool_padding padding_type,
                           uint32_t kernel_height, uint32_t kernel_width,
                           uint32_t stride_height, uint32_t stride_width,
                           zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_POOL_PADDING(padding_type);
    PRINT_PARM_UINT32T(kernel_height);
    PRINT_PARM_UINT32T(kernel_width);
    PRINT_PARM_UINT32T(stride_height);
    PRINT_PARM_UINT32T(stride_width);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  // The switch in arg order is intentional. The AIU op expects a different
  // order than our API.
  return aiu_ops_func_specific(NNPA_MAXPOOL2D, input, NULL, NULL, output, NULL,
                               0, padding_type, stride_width, stride_height,
                               kernel_width, kernel_height);
}

/// Reduces both input tensor's H and W dimensions to 1 storing a mean of
/// the original dimensions' values. Issued to NNPA as a NNPA_AVGPOOL2D
/// call with 0 strides.
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_meanreduce2d(const zdnn_ztensor *input, zdnn_ztensor *output) {

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  return aiu_ops_func_specific(
      NNPA_AVGPOOL2D, input, NULL, NULL, output, NULL, 0, VALID_PADDING, 0, 0,
      input->transformed_desc->dim2, input->transformed_desc->dim3);
}

/// Preforms 2D convolution operation using input tensor and a filter kernel
/// tensor, and computes the output.
///
/// \param[in] input The input tensor
/// \param[in] kernel The input kernel tensor
/// \param[in] bias  The input bias tensor
/// \param[in] padding_type VALID_PADDING or SAME_PADDING
/// \param[in] stride_height height movement per kernel slide
/// \param[in] stride_width width movement per kernel slide
/// \param[in] act_func
///                 activation function as specified in the zdnn_conv2d_act enum
/// \param[in] clipping_value A pointer to an FP32 clipping value
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_conv2d(const zdnn_ztensor *input, const zdnn_ztensor *kernel,
                        const zdnn_ztensor *bias,
                        zdnn_pool_padding padding_type, uint32_t stride_height,
                        uint32_t stride_width, zdnn_conv2d_act act_func,
                        const void *clipping_value, zdnn_ztensor *output) {

  // Create Function Specific Parm 4 for Convolution first to optimize
  // conditional checks.
  func_sp_parm4_conv2d conv2d_parm4;
  conv2d_parm4.val = 0;

  // Create variable for parameter output. Check if value is NULL, followed by a
  // check if it is not 0. If it is 0 it is unnecessary to convert 0 to DLFloat
  // or setting clipping_value (as it is already set by val)
  float clip_val = 0;
  if (clipping_value) {
    clip_val = *(float *)clipping_value;
    if (clip_val != 0) {
      conv2d_parm4.bits.clipping_value = cnvt_1_fp32_to_dlf16(clip_val);
    }
  }
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(kernel);
    PRINT_PARM_ZTENSOR_PTR(bias);
    PRINT_PARM_POOL_PADDING(padding_type);
    PRINT_PARM_UINT32T(stride_height);
    PRINT_PARM_UINT32T(stride_width);
    PRINT_PARM_CONV2D_ACT(act_func);
    PRINT_PARM_FLOAT_PTR(clip_val);
    PRINT_PARM_ZTENSOR_PTR(output);
    END_PRINT_PARMS;
  }

  func_sp_parm1_conv2d conv2d_parm1;
  conv2d_parm1.val = 0;
  conv2d_parm1.bits.act = act_func;
  conv2d_parm1.bits.pad = padding_type;

  // NNPA parameter block expects:
  // - function-specific-parameter-2: dimension-2 (W) stride of NHWC
  // - function-specific-parameter-3: dimension-3 (H) stride of NHWC
  // thus in (stride_width, stride_height) order
  return aiu_ops_func_specific(NNPA_CONVOLUTION, input, kernel, bias, output,
                               NULL, 0, conv2d_parm1.val, stride_width,
                               stride_height, conv2d_parm4.val, 0);
}
