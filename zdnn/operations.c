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
#pragma export(zdnn_add)
#pragma export(zdnn_sub)
#pragma export(zdnn_mul)
#pragma export(zdnn_div)
#pragma export(zdnn_min)
#pragma export(zdnn_max)
#pragma export(zdnn_log)
#pragma export(zdnn_exp)
#pragma export(zdnn_sqrt)
#pragma export(zdnn_invsqrt)
#pragma export(zdnn_relu)
#pragma export(zdnn_leaky_relu)
#pragma export(zdnn_tanh)
#pragma export(zdnn_sigmoid)
#pragma export(zdnn_softmax)
#pragma export(zdnn_softmax_mask)
#pragma export(zdnn_gelu)
#pragma export(zdnn_lstm)
#pragma export(zdnn_gru)
#pragma export(zdnn_matmul_op)
#pragma export(zdnn_matmul_bcast_op)
#pragma export(zdnn_matmul_transpose_op)
#pragma export(zdnn_quantized_matmul_op)
#pragma export(zdnn_batchnorm)
#pragma export(zdnn_norm)
#pragma export(zdnn_meanreduce2d)
#pragma export(zdnn_moments)
#pragma export(zdnn_layernorm)
#pragma export(zdnn_reduce)
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
#define PRINT_PARM_BOOL(val)                                                   \
  printf("\nParameter %s (bool): %s\n", #val, val ? "true" : "false");
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
#define PRINT_PARM_REDUCE_OP(op)                                               \
  printf("\nReduce Operation: %s\n", get_reduce_op_str(op));
#define PRINT_PARM_BESSEL_CORRECTION(val)                                      \
  printf("\nBessel Correction: %s\n", get_bessel_correction_str(val));
#define PRINT_API_AVAILABILITY(operation_name, api)                            \
  printf("Operation %s availability: %s\n", operation_name,                    \
         is_operation_available(api) ? "True" : "False");
#define PRINT_MATMUL_OPS_API_AVAILABILITY(operation_name, function_code,       \
                                          parmblock_version)                   \
  printf("Operation %s availability: %s\n", operation_name,                    \
         is_nnpa_fc_and_parmblock_installed(function_code, parmblock_version)  \
             ? "True"                                                          \
             : "False");

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
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_relu *fsp_relu = (func_sp_parms_relu *)&fsp;

  // Create variable for parameter output. Check if value is NULL, followed by a
  // check if it is not 0. If it is 0 it is unnecessary to convert 0 to DLFloat
  // or setting clipping_value (as it is already set by val)
  float clip_val = 0;
  if (clipping_value) {
    clip_val = *(float *)clipping_value;
    if (clip_val != 0) {
      fsp_relu->parm1.clipping_value = cnvt_1_fp32_to_dlf16(clip_val);
    }
  }

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_FLOAT_PTR(clip_val);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_relu", ZDNN_RELU);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-1: clipping value
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_RELU, input, NULL,
                               NULL, output, NULL, 0, &fsp);
}

/// External interface for LeakyRelu operation
///
/// \param[in] input The input tensor
/// \param[in] clipping_value A pointer to an FP32 clipping value
/// \param[in] adjustment_factor A FP32 value multiplied with negative elements
/// from input.
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_leaky_relu(const zdnn_ztensor *input,
                            const void *clipping_value, float adjustment_factor,
                            zdnn_ztensor *output) {
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_relu *fsp_relu = (func_sp_parms_relu *)&fsp;

  // Create variable for parameter output. Check if value is NULL, followed by a
  // check if it is not 0. If it is 0 it is unnecessary to convert 0 to DLFloat
  // or setting clipping_value (as it is already set by val)
  float clip_val = 0;
  if (clipping_value) {
    clip_val = *(const float *)clipping_value;
    if (clip_val != 0) {
      fsp_relu->parm1.clipping_value = cnvt_1_fp32_to_dlf16(clip_val);
    }
  }

  // Create variable for parameter output. If adjustment_factor is 0 it is
  // unnecessary to convert 0 to DLFloat or setting adjustment_factor (as it is
  // already set by val)
  if (adjustment_factor != 0) {
    fsp_relu->parm2.adjustment_factor = cnvt_1_fp32_to_dlf16(adjustment_factor);
  }

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_FLOAT_PTR(clip_val);
    PRINT_PARM_FLOAT_PTR(adjustment_factor);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_leaky_relu", ZDNN_LEAKY_RELU);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-1: clipping value
  // - function-specific-parameter-2: adjustment factor
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_RELU, input, NULL,
                               NULL, output, NULL, 0, &fsp);
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
    PRINT_API_AVAILABILITY("zdnn_tanh", ZDNN_TANH);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_TANH, input, NULL, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_sigmoid", ZDNN_SIGMOID);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_SIGMOID, input, NULL, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_softmax", ZDNN_SOFTMAX);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_softmax *fsp_softmax = (func_sp_parms_softmax *)&fsp;
  fsp_softmax->parm1.act = act_func;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: ACTIVATION function
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_SOFTMAX, input, NULL,
                               NULL, output, NULL, (uintptr_t)save_area, &fsp);
}

/// External interface for Softmax Mask operation
///
/// \param[in] input The input tensor
/// \param[in] save_area Pointer to the save area required by NNPA_SOFTMAX
/// \param[in] act_func activation function as specified in the zdnn_softmax_act
/// enum
/// \param[in] softmax_mask An integer that specifies the count of dimensions 1
/// elements to be processed.
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_softmax_mask(const zdnn_ztensor *input, void *save_area,
                              zdnn_softmax_act act_func, uint32_t softmax_mask,
                              zdnn_ztensor *output) {
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_PTR(save_area);
    PRINT_PARM_SOFTMAX_ACT(act_func);
    PRINT_PARM_UINT32T(softmax_mask);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_softmax_mask", ZDNN_SOFTMAX_MASK);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_softmax *fsp_softmax = (func_sp_parms_softmax *)&fsp;
  fsp_softmax->parm1.act = act_func;
  fsp_softmax->parm2.mask = softmax_mask;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: ACTIVATION function
  // - function-specific-parameter-2: MASK
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_SOFTMAX, input, NULL,
                               NULL, output, NULL, (uintptr_t)save_area, &fsp);
}

/// External interface for GeLu operation
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_gelu(const zdnn_ztensor *input, zdnn_ztensor *output) {
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_gelu", ZDNN_GELU);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_1, NNPA_GELU, input, NULL, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_lstm", ZDNN_LSTM);
    END_PRINT_PARMS;

    // aiu_lstm_gru() dissects the input tensors and makes multiple calls to the
    // zAIU.  check the overall input tensors here and precheck will check the
    // dissected tensors later before each and every zAIU call
    zdnn_status precheck_status;
    if ((precheck_status = verify_zdnn_lstm_or_gru_tensors(
             NNPA_LSTMACT, input, h0, c0, weights, biases, hidden_weights,
             hidden_biases, direction, hn_output, cf_output)) != ZDNN_OK) {
      return precheck_status;
    }
  }

  return aiu_lstm_gru(NNPA_PARMBLKFORMAT_0, NNPA_LSTMACT, input, h0, c0,
                      weights, biases, hidden_weights, hidden_biases, direction,
                      work_area, hn_output, cf_output);
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
    PRINT_API_AVAILABILITY("zdnn_gru", ZDNN_GRU);
    END_PRINT_PARMS;

    // aiu_lstm_gru() dissects the input tensors and makes multiple calls to the
    // zAIU.  check the overall input tensors here and precheck will check the
    // dissected tensors later before the zAIU calls
    zdnn_status precheck_status;
    if ((precheck_status = verify_zdnn_lstm_or_gru_tensors(
             NNPA_GRUACT, input, h0, NULL, weights, biases, hidden_weights,
             hidden_biases, direction, hn_output, NULL)) != ZDNN_OK) {
      return precheck_status;
    }
  }

  return aiu_lstm_gru(NNPA_PARMBLKFORMAT_0, NNPA_GRUACT, input, h0, NULL,
                      weights, biases, hidden_weights, hidden_biases, direction,
                      work_area, hn_output, NULL);
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
    PRINT_API_AVAILABILITY("zdnn_add", ZDNN_ADD);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_ADD, input_a, input_b, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_sub", ZDNN_SUB);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_SUB, input_a, input_b, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_div", ZDNN_DIV);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_DIV, input_a, input_b, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_mul", ZDNN_MUL);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_MUL, input_a, input_b, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_max", ZDNN_MAX);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_MAX, input_a, input_b, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_min", ZDNN_MIN);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_MIN, input_a, input_b, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_log", ZDNN_LOG);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_LOG, input, NULL, NULL, output,
                 NULL);
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
    PRINT_API_AVAILABILITY("zdnn_exp", ZDNN_EXP);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_EXP, input, NULL, NULL, output,
                 NULL);
}

/// External interface for Square Root operation
///
/// \param[in] input The input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_sqrt(const zdnn_ztensor *input, zdnn_ztensor *output) {
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_sqrt", ZDNN_SQRT);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_1, NNPA_SQRT, input, NULL, NULL, output,
                 NULL);
}

/// External interface for Inverse Square Root operation
///
/// \param[in] input The input tensor
/// \param[in] epsilon A FP32 value added to input prior to computation.
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_invsqrt(const zdnn_ztensor *input, float epsilon,
                         zdnn_ztensor *output) {
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parm1_invsqrt *fsp_invsqrt = (func_sp_parm1_invsqrt *)&fsp;

  // Create variable for parameter output. If epsilon is 0 it is unnecessary to
  // convert 0 to DLFloat or setting epsilon (as it is already set by val)
  if (epsilon != 0) {
    fsp_invsqrt->epsilon = cnvt_1_fp32_to_dlf16(epsilon);
  }

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_FLOAT_PTR(epsilon);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_invsqrt", ZDNN_INVSQRT);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-1: epsilon
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_INVSQRT, input, NULL,
                               NULL, output, NULL, 0, &fsp);
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
    PRINT_API_AVAILABILITY("zdnn_matmul_op", ZDNN_MATMUL_OP);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_matmul *fsp_matmul = (func_sp_parms_matmul *)&fsp;

  fsp_matmul->parm1.operation = op_type;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: OPERATION field
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_MATMUL_OP, input_a,
                               input_b, input_c, output, NULL, 0, &fsp);
}

/// External interface for Matmul Broadcast operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] op_type The operation performed against matmul dot product
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_matmul_bcast_op(const zdnn_ztensor *input_a,
                                 const zdnn_ztensor *input_b,
                                 const zdnn_ztensor *input_c,
                                 zdnn_matmul_bcast_ops op_type,
                                 zdnn_ztensor *output) {
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_matmul_bcast *fsp_matmul_bcast =
      (func_sp_parms_matmul_bcast *)&fsp;

  fsp_matmul_bcast->parm1.operation = op_type;

  // Determine function_code using dim4 of input_a and input_b
  nnpa_function_code function_code = get_matmul_function(
      input_a->transformed_desc->dim4, input_b->transformed_desc->dim4);

  // We want to use NNPA_PARMBLKFORMAT_0 where possible to ensure all previous
  // functionality is still available.
  //
  // When using NNPA_PARMBLKFORMAT_0, NNPA_MATMUL_OP_BCAST23 only supports
  // MATMUL_BCAST_OP_ADDITION for the op_type, as the comparison operations were
  // added with NNPA_PARMBLKFORMAT_1. This means:
  //
  // NNPA_PARMBLKFORMAT_0 should be used when:
  //   1 - function_code is NNPA_MATMUL_OP
  //   2 - function_code is NNPA_MATMUL_OP_BCAST23 and op_type is
  //       MATMUL_BCAST_OP_ADDITION
  // NNPA_PARMBLKFORMAT_1 must be used when:
  //   1 - function_code is NNPA_MATMUL_OP_BCAST1
  //   2 - function_code is NNPA_MATMUL_OP_BCAST23 and op_type is not
  //       MATMUL_BCAST_OP_ADDITION
  nnpa_parmblk_format parm_block_format =
      function_code == NNPA_MATMUL_OP ||
              (function_code == NNPA_MATMUL_OP_BCAST23 &&
               op_type == MATMUL_BCAST_OP_ADDITION)
          ? NNPA_PARMBLKFORMAT_0
          : NNPA_PARMBLKFORMAT_1;

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_MATMUL_BCAST_OP(op_type);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_MATMUL_OPS_API_AVAILABILITY("zdnn_matmul_bcast_op", function_code,
                                      parm_block_format);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-1: OPERATION field
  return aiu_ops_func_specific(parm_block_format, function_code, input_a,
                               input_b, input_c, output, NULL, 0, &fsp);
}

/// External interface for Matmul Transpose operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] transpose_a Whether to transpose input_a prior to matmul
/// \param[in] transpose_b Whether to transpose input_b prior to matmul
/// \param[in] op_type The operation performed against matmul dot product
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_matmul_transpose_op(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     bool transpose_a, bool transpose_b,
                                     zdnn_matmul_ops op_type,
                                     zdnn_ztensor *output) {
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_matmul *fsp_matmul = (func_sp_parms_matmul *)&fsp;

  fsp_matmul->parm1.operation = op_type;
  fsp_matmul->parm2.transpose_a = transpose_a;
  fsp_matmul->parm2.transpose_b = transpose_b;

  // Determine function_code using dim4 of input_a and input_b
  nnpa_function_code function_code = get_matmul_function(
      input_a->transformed_desc->dim4, input_b->transformed_desc->dim4);

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_BOOL(transpose_a);
    PRINT_PARM_BOOL(transpose_b);
    PRINT_PARM_MATMUL_BCAST_OP(op_type);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_MATMUL_OPS_API_AVAILABILITY("zdnn_matmul_transpose_op", function_code,
                                      NNPA_PARMBLKFORMAT_1);
    END_PRINT_PARMS;
  }
  // NNPA parameter block expects:
  // - function-specific-parameter-1: OPERATION field
  // - function-specific-parameter-1: transpose control
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, function_code, input_a,
                               input_b, input_c, output, NULL, 0, &fsp);
}

/// External interface for Quantized Matmul operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] op_type The operation performed against matmul dot product
/// \param[in] clip_min The minimum quantized value.
/// \param[in] clip_max The maximim quantized value.
/// \param[in] disable_clipping Whether to disable clipping and rounding.
/// \param[in] dequantize Whether the output should be dequantized
/// after computation.
/// \param[in] pre_computed Whether bias is already pre-computed.
/// \param[in] work_area Pointer to pre-allocated work area,
/// or NULL
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_quantized_matmul_op(
    const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
    const zdnn_ztensor *input_c, zdnn_matmul_ops op_type, const int8_t clip_min,
    const int8_t clip_max, const bool disable_clipping, const bool dequantize,
    const bool pre_computed, void *work_area, zdnn_ztensor *output) {

  // When pre_computed=true input_b->offset (Zb) must be 0.f
  if (pre_computed && input_b->offset != 0.f) {
    return ZDNN_STATUS(ZDNN_INVALID_OFFSET,
                       "input_b offset (Zb) is invalid when pre_computed=true "
                       "(found %f, expects %f)",
                       input_b->offset, 0.f);
  }

  // Determine function_code using dim4 of input_a and input_b
  nnpa_function_code function_code = get_matmul_function(
      input_a->transformed_desc->dim4, input_b->transformed_desc->dim4);

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_MATMUL_OP(op_type);
    PRINT_PARM_PTR(work_area);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_MATMUL_OPS_API_AVAILABILITY("zdnn_quantized_matmul_op", function_code,
                                      NNPA_PARMBLKFORMAT_1);
    END_PRINT_PARMS;
  }

  return aiu_quantized_matmul(NNPA_PARMBLKFORMAT_1, function_code, input_a,
                              input_b, input_c, op_type, clip_min, clip_max,
                              work_area, output, dequantize, disable_clipping,
                              pre_computed);
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
    PRINT_API_AVAILABILITY("zdnn_batchnorm", ZDNN_BATCHNORM);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_0, NNPA_BATCHNORMALIZATION, input_a,
                 input_b, input_c, output, NULL);
}

/// External interface for Norm operation
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_norm(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                      zdnn_ztensor *output) {
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_norm", ZDNN_NORM);
    END_PRINT_PARMS;
  }

  return aiu_ops(NNPA_PARMBLKFORMAT_1, NNPA_NORM, input_a, input_b, NULL,
                 output, NULL);
}

/// External interface for Moments operation
///
/// \param[in] input_a                The first input tensor
/// \param[in] bessel_correction_type The bessel correction
/// \param[out] output_a              The first output tensor
/// \param[out] output_b              The second output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_moments(const zdnn_ztensor *input,
                         zdnn_moments_bessel bessel_correction_type,
                         zdnn_ztensor *output_a, zdnn_ztensor *output_b) {
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_BESSEL_CORRECTION(bessel_correction_type);
    PRINT_PARM_ZTENSOR_PTR(output_a);
    PRINT_PARM_ZTENSOR_PTR(output_b);
    PRINT_API_AVAILABILITY("zdnn_moments", ZDNN_MOMENTS);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_moments *fsp_moments = (func_sp_parms_moments *)&fsp;
  fsp_moments->parm1.bessel_correction = bessel_correction_type;

  // NNPA parameter block expects:
  // - function-specific-parameter-1: bessel_correction
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_MOMENTS, input, NULL,
                               NULL, output_a, output_b, 0, &fsp);
}

/// External interface for LayerNorm operation
///
/// \param[in] input_a The input tensor A
/// \param[in] input_b The input tensor B
/// \param[in] input_c The input tensor C
/// \param[in] beta_value A pointer to an FP32 beta value
/// \param[in] gamma_value A pointer to an FP32 gamma value
/// \param[in] epsilon_value A pointer to an FP32 epsilon value
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
zdnn_status zdnn_layernorm(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c, const float beta_value,
                           const float gamma_value, const float epsilon_value,
                           zdnn_ztensor *output) {
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_layernorm *fsp_layernorm = (func_sp_parms_layernorm *)&fsp;

  if (beta_value != 0) {
    fsp_layernorm->parm1.beta = cnvt_1_fp32_to_dlf16(beta_value);
  }

  if (gamma_value != 0) {
    fsp_layernorm->parm2.gamma = cnvt_1_fp32_to_dlf16(gamma_value);
  }

  if (epsilon_value != 0) {
    fsp_layernorm->parm3.epsilon = cnvt_1_fp32_to_dlf16(epsilon_value);
  }

  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input_a);
    PRINT_PARM_ZTENSOR_PTR(input_b);
    PRINT_PARM_ZTENSOR_PTR(input_c);
    PRINT_PARM_FLOAT_PTR(beta_value);
    PRINT_PARM_FLOAT_PTR(gamma_value);
    PRINT_PARM_FLOAT_PTR(epsilon_value);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_layernorm", ZDNN_LAYERNORM);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-1: beta value
  // - function-specific-parameter-2: gamma value
  // - function-specific-parameter-3: epsilon value
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_LAYERNORM, input_a,
                               input_b, input_c, output, NULL, 0, &fsp);
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
    PRINT_API_AVAILABILITY("zdnn_avgpool2d", ZDNN_AVGPOOL2D);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_pool2d *fsp_pool2d = (func_sp_parms_pool2d *)&fsp;
  fsp_pool2d->parm1.pad = padding_type;
  fsp_pool2d->parm2.stride_width = stride_width;
  fsp_pool2d->parm3.stride_height = stride_height;
  fsp_pool2d->parm4.kernel_width = kernel_width;
  fsp_pool2d->parm5.kernel_height = kernel_height;

  // The switch in arg order is intentional. The zAIU op expects a different
  // order than our API.
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_AVGPOOL2D, input,
                               NULL, NULL, output, NULL, 0, &fsp);
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
    PRINT_API_AVAILABILITY("zdnn_maxpool2d", ZDNN_MAXPOOL2D);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_pool2d *fsp_pool2d = (func_sp_parms_pool2d *)&fsp;
  fsp_pool2d->parm1.pad = padding_type;
  fsp_pool2d->parm2.stride_width = stride_width;
  fsp_pool2d->parm3.stride_height = stride_height;
  fsp_pool2d->parm4.kernel_width = kernel_width;
  fsp_pool2d->parm5.kernel_height = kernel_height;

  // The switch in arg order is intentional. The zAIU op expects a different
  // order than our API.
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_MAXPOOL2D, input,
                               NULL, NULL, output, NULL, 0, &fsp);
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
    PRINT_API_AVAILABILITY("zdnn_meanreduce2d", ZDNN_MEANREDUCE2D);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_pool2d *fsp_pool2d = (func_sp_parms_pool2d *)&fsp;
  fsp_pool2d->parm1.pad = VALID_PADDING;
  fsp_pool2d->parm4.kernel_width = input->transformed_desc->dim2;
  fsp_pool2d->parm5.kernel_height = input->transformed_desc->dim3;

  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_AVGPOOL2D, input,
                               NULL, NULL, output, NULL, 0, &fsp);
}

/// External interface for Reduce operation
///
/// \param[in] input The input tensor
/// \param[in] save_area Pointer to the save area required by NNPA_REDUCE
/// \param[in] op_type The reduction operation to perform on input
/// \param[out] output The output tensor
///
/// \return ZDNN_OK if all checks pass. or a failure based on why it failed
///
zdnn_status zdnn_reduce(const zdnn_ztensor *input, void *save_area,
                        zdnn_reduce_ops op_type, zdnn_ztensor *output) {
  if (precheck_enabled) {
    BEGIN_PRINT_PARMS;
    PRINT_PARM_ZTENSOR_PTR(input);
    PRINT_PARM_PTR(save_area);
    PRINT_PARM_REDUCE_OP(op_type);
    PRINT_PARM_ZTENSOR_PTR(output);
    PRINT_API_AVAILABILITY("zdnn_reduce", ZDNN_REDUCE);
    END_PRINT_PARMS;
  }

  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_reduce *fsp_reduce = (func_sp_parms_reduce *)&fsp;
  fsp_reduce->parm1.operation = op_type;

  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_1, NNPA_REDUCE, input, NULL,
                               NULL, output, NULL, (uintptr_t)save_area, &fsp);
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
  function_specific_parameters fsp;
  memset(&fsp, 0, sizeof(function_specific_parameters));
  func_sp_parms_conv2d *fsp_conv2d = (func_sp_parms_conv2d *)&fsp;
  fsp_conv2d->parm1.act = act_func;
  fsp_conv2d->parm1.pad = padding_type;
  fsp_conv2d->parm2.stride_width = stride_width;
  fsp_conv2d->parm3.stride_height = stride_height;

  // Create variable for parameter output. Check if value is NULL, followed by a
  // check if it is not 0. If it is 0 it is unnecessary to convert 0 to DLFloat
  // or setting clipping_value (as it is already set by val)
  float clip_val = 0;
  if (clipping_value) {
    clip_val = *(float *)clipping_value;
    if (clip_val != 0) {
      fsp_conv2d->parm4.clipping_value = cnvt_1_fp32_to_dlf16(clip_val);
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
    PRINT_API_AVAILABILITY("zdnn_conv2d", ZDNN_CONV2D);
    END_PRINT_PARMS;
  }

  // NNPA parameter block expects:
  // - function-specific-parameter-2: dimension-2 (W) stride of NHWC
  // - function-specific-parameter-3: dimension-3 (H) stride of NHWC
  // thus in (stride_width, stride_height) order
  return aiu_ops_func_specific(NNPA_PARMBLKFORMAT_0, NNPA_CONVOLUTION, input,
                               kernel, bias, output, NULL, 0, &fsp);
}
