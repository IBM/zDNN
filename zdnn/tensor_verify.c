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
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/// Verify multiple zTensors against specific type/format values.
/// Variadic parameter list MUST be NULL-TERMINATED.
///
/// \param[in] type required data type
/// \param[in] format required format
/// \param[in] ... list of (zdnn_ztensor*, char* name) pairs, NULL-TERMINATED at
///                the end
///
/// \return ZDNN_OK
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
static zdnn_status verify_fields(zdnn_data_types type, zdnn_data_formats format,
                                 ...) {
  zdnn_status status = ZDNN_OK;
  va_list v;
  va_start(v, format);

  zdnn_ztensor *tsr_ptr;
  uint8_t i = 0;

  while ((tsr_ptr = va_arg(v, zdnn_ztensor *)) != NULL) {
    char *tsr_name = va_arg(v, char *);
    if (tsr_ptr->transformed_desc->type != type) {
      status = ZDNN_STATUS(
          ZDNN_INVALID_TYPE,
          "%s tensor type is invalid (found %s (%d), expects %s (%d))",
          tsr_name, get_data_type_str(tsr_ptr->transformed_desc->type),
          tsr_ptr->transformed_desc->type, get_data_type_str(type), type);
      break;
    }
    if (tsr_ptr->transformed_desc->format != format) {
      status = ZDNN_STATUS(
          ZDNN_INVALID_FORMAT,
          "%s tensor format is invalid (found %s (%d), expects %s (%d))",
          tsr_name, get_data_format_str(tsr_ptr->transformed_desc->format),
          tsr_ptr->transformed_desc->format, get_data_format_str(format),
          format);
      break;
    }
    i++;
  }

  va_end(v);
  return status;
}

/// Verify multiple zTensors against specific shape value
/// Variadic parameter list MUST be NULL-TERMINATED
///
/// \param[in] dim_idx dimX index
/// \param[in] val required value
/// \param[in] ... list of (zdnn_ztensor*, char* name) pairs, NULL-TERMINATED at
///                the end
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///
static zdnn_status verify_dim(uint8_t dim_idx, uint32_t val, ...) {
  zdnn_status status = ZDNN_OK;
  va_list v;
  va_start(v, val);

  zdnn_ztensor *tsr_ptr;
  uint8_t i = 0;

  while ((tsr_ptr = va_arg(v, zdnn_ztensor *)) != NULL) {
    char *tsr_name = va_arg(v, char *);
    uint32_t *dims_ptr = &(tsr_ptr->transformed_desc->dim4);
    if (dims_ptr[ZDNN_MAX_DIMS - dim_idx] != val) {
      status = ZDNN_STATUS(
          ZDNN_INVALID_SHAPE,
          "%s dim%d tensor shape is invalid (found %d, expects %d)", tsr_name,
          dim_idx, dims_ptr[ZDNN_MAX_DIMS - dim_idx], val);
      break;
    }
    i++;
  }

  va_end(v);
  return status;
}

#define CAT(x, y) x##y
#define TENSOR_PARMS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...)   \
  CAT(TENSOR_PARM_, N)
#define TENSOR_PARMS(...)                                                      \
  TENSOR_PARMS_HELPER(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

// stringify the tensors to x, "x"
#define TENSOR_PARM_1(x) x, #x
#define TENSOR_PARM_2(x, x2) x, #x, x2, #x2
#define TENSOR_PARM_3(x, x2, x3) x, #x, x2, #x2, x3, #x3
#define TENSOR_PARM_4(x, x2, x3, x4) x, #x, x2, #x2, x3, #x3, x4, #x4
#define TENSOR_PARM_5(x, x2, x3, x4, x5)                                       \
  x, #x, x2, #x2, x3, #x3, x4, #x4, x5, #x5
#define TENSOR_PARM_6(x, x2, x3, x4, x5, x6)                                   \
  x, #x, x2, #x2, x3, #x3, x4, #x4, x5, #x5, x6, #x6
#define TENSOR_PARM_7(x, x2, x3, x4, x5, x6, x7)                               \
  x, #x, x2, #x2, x3, #x3, x4, #x4, x5, #x5, x6, #x6, x7, #x7
#define TENSOR_PARM_8(x, x2, x3, x4, x5, x6, x7, x8)                           \
  x, #x, x2, #x2, x3, #x3, x4, #x4, x5, #x5, x6, #x6, x7, #x7, x8, #x8
#define TENSOR_PARM_9(x, x2, x3, x4, x5, x6, x7, x8, x9)                       \
  x, #x, x2, #x2, x3, #x3, x4, #x4, x5, #x5, x6, #x6, x7, #x7, x8, #x8, x9, #x9

// Convenience macros with auto NULL-terminated variadic list

#define VERIFY_FIELDS(type, format, ...)                                       \
  verify_fields(type, format, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG)

#define VERIFY_DIM4(val, ...)                                                  \
  verify_dim(4, val, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG)
#define VERIFY_DIM3(val, ...)                                                  \
  verify_dim(3, val, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG)
#define VERIFY_DIM2(val, ...)                                                  \
  verify_dim(2, val, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG)
#define VERIFY_DIM1(val, ...)                                                  \
  verify_dim(1, val, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG)

#define VERIFY_HEIGHT VERIFY_DIM3
#define VERIFY_WIDTH VERIFY_DIM2

#define VERIFY_ALL_DIMS(val_dim4, val_dim3, val_dim2, val_dim1, ...)           \
  (verify_dim(4, val_dim4, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG) |   \
   verify_dim(3, val_dim3, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG) |   \
   verify_dim(2, val_dim2, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG) |   \
   verify_dim(1, val_dim1, TENSOR_PARMS(__VA_ARGS__)(__VA_ARGS__), NO_ARG))

#define IS_SHAPE_BIAS(tensor)                                                  \
  (verify_dim(4, 1, TENSOR_PARMS(tensor)(tensor), NO_ARG) |                    \
   verify_dim(3, 1, TENSOR_PARMS(tensor)(tensor), NO_ARG) |                    \
   verify_dim(2, 1, TENSOR_PARMS(tensor)(tensor), NO_ARG))

/// Verifies if all tensors have exact same shape and data type and format
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_tensors(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c,
                           const zdnn_ztensor *output) {
  zdnn_status status;
  zdnn_tensor_desc *input_a_tfrmd_desc = input_a->transformed_desc;

  /*
  Parameter patterns:

  input_a | input_b | input_c | output
  --------+---------+---------+-------
    X     |   NULL  |   NULL  |   X
    X     |   X     |   NULL  |   X
    X     |   X     |   X     |   X

  Use input_a's as the "correct" value
  input_b and input_c are NULL when not being used
  */

  // check shapes first

  if ((status =
           VERIFY_ALL_DIMS(input_a_tfrmd_desc->dim4, input_a_tfrmd_desc->dim3,
                           input_a_tfrmd_desc->dim2, input_a_tfrmd_desc->dim1,
                           output, input_b, input_c)) != ZDNN_OK) {
    return status;
  }

  // then check type and format

  if ((status =
           VERIFY_FIELDS(input_a_tfrmd_desc->type, input_a_tfrmd_desc->format,
                         output, input_b, input_c)) != ZDNN_OK) {
    return status;
  }

  return status;
}

/// Verifies the condition of lstm/gru activation tensors, wrt zAIU's
/// LSTM_ACT/GRU_ACT ops
///
/// \param[in] mode             LSTM or GRU
/// \param[in] ts_fused         timestep fused for RNN activation call
/// \param[in] bias_add_rnn_op  bias tensor
/// \param[in] prev_state       previous state tensor (prev_c LSTM, prev_h GRU)
/// \param[in] h_output         h_output tensor
/// \param[in] c_output         c_output tensor (LSTM only, NULL if GRU)
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_lstm_or_gru_act_tensors(uint8_t function_code,
                                           const zdnn_ztensor *ts_fused,
                                           const zdnn_ztensor *bias_add_rnn_op,
                                           const zdnn_ztensor *prev_state,
                                           const zdnn_ztensor *h_output,
                                           const zdnn_ztensor *c_output) {
  /*
   DIMENSION REQUIREMENTS (NHWC, DLFLOAT16)
   Legend:
   g = number of gates (4 LSTM or 3 GRU)
   b = number of batches
   s = hidden state size

                   |   shape (dim4, dim3, dim2, dim1)
   ----------------+-------------------------------------
   ts_fused        |   (g,1,b,s)
   bias_add_rnn_op |   (g,1,b,s)
   prev_state      |   (1,1,b,s) (LSTM prev_c, GRU prev_h)
   h_output        |   (1,1,b,s)
   c_output        |   (1,1,b,s) (LSTM only, GRU ignores)
  */

  zdnn_status status;
  uint32_t num_gates = get_func_code_num_gates(function_code);

  // These should match in for all tensors so set the expected to one of them.
  uint32_t exp_dim2 = ts_fused->transformed_desc->dim2;
  uint32_t exp_dim1 = ts_fused->transformed_desc->dim1;
  zdnn_data_types exp_type = ts_fused->transformed_desc->type;
  zdnn_data_formats exp_format = ts_fused->transformed_desc->format;

  // check shapes
  if ((status = VERIFY_DIM4(1, prev_state, h_output)) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM4(num_gates, ts_fused, bias_add_rnn_op)) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM3(1, ts_fused, bias_add_rnn_op, prev_state,
                            h_output)) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM2(exp_dim2, ts_fused, bias_add_rnn_op, prev_state,
                            h_output)) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM1(exp_dim1, ts_fused, bias_add_rnn_op, prev_state,
                            h_output)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_DIM4(1, c_output)) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM3(1, c_output)) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM2(exp_dim2, c_output)) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM1(exp_dim1, c_output)) != ZDNN_OK) {
      return status;
    }
  }

  // then check type and format
  if ((status = VERIFY_FIELDS(exp_type, exp_format, ts_fused, bias_add_rnn_op,
                              prev_state, h_output)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_FIELDS(exp_type, exp_format, c_output)) != ZDNN_OK) {
      return status;
    }
  }

  // If we reach this, all checks passed. Return OK
  return ZDNN_OK;
}

/// Verifies the condition of lstm/gru activation tensors, wrt ZDNN's
/// zdnn_lstm()/zdnn_gru() functions
///
/// \param[in] input The input tensor
/// \param[in] h0 The initial hidden state tensor
/// \param[in] c0 The initial cell state tensor
/// \param[in] weights The concatenated weights tensor
/// \param[in] biases The concatenated biases tensor
/// \param[in] hidden_weights The concatenated hidden weights tensor
/// \param[in] hidden_biases The concatenated hidden biases tensor
/// \param[in] direction Direction (FWD, BWD, BIDIR)
/// \param[out] hn_output The output hidden_state tensor
/// \param[out] cf_output The output cell_state tensor
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
zdnn_status verify_zdnn_lstm_or_gru_tensors(
    uint8_t function_code, const zdnn_ztensor *input, const zdnn_ztensor *h0,
    const zdnn_ztensor *c0, const zdnn_ztensor *weights,
    const zdnn_ztensor *biases, const zdnn_ztensor *hidden_weights,
    const zdnn_ztensor *hidden_biases, lstm_gru_direction direction,
    const zdnn_ztensor *hn_output, const zdnn_ztensor *cf_output) {

  /*
  DIMENSION REQUIREMENTS (stickified, i.e., NHWC)
  Legend:
    b = number of batches
    d = number of directions (2 if BIDIR or otherwise 1)
    f = number of features
    g = number of gates (4 LSTM or 3 GRU)
    s = hidden state size
    s_pad = ceil(s/64) * 64 (s with padding to nearest multiple of 64)
    in_pad = g * s_pad (horizontally concatenated gate input with padding
             between gates)
    out_pad = d * s_pad (horizontally concatenated output with padding between
              directions)
    ts = number of timesteps

  Note: The *_output expected shape differs based on unidirectional versus
  bidirectional. For hn_output, the user specified shape also controls whether
  all timestep results are returned or just the final result processed.

  tensor         | tfrmd (dim4, 3, 2, 1) | Note
  ---------------+-------------------------------------
  input          | (ts, 1, b, f)         |
  h0             | (d, 1, b, s)          |
  c0             | (d, 1, b, s)          | (LSTM only, GRU NULL)
  weights        | (d, 1, f, in_pad)     |
  biases         | (d, 1, 1, in_pad)     |
  hidden_weights | (d, 1, s, in_pad)     |
  hidden_biases  | (d, 1, 1, in_pad)     |
  ----------------------------+----------+----------------|
  hn_output      | (ts, 1, b, s)         | (uni all timesteps)
                 | (1, 1, b, s)          | (uni final only)
                 | (ts, 1, b, out_pad)   | (bidir all out_pad)
                 | (1, 1, b, out_pad)    | (bidir final only)
  cf_output      | (1, 1, b, s)          | (uni LSTM only, GRU NULL)
                 | (1, 1, b, out_pad)    | (bidir LSTM only, GRU NULL)
  */

  zdnn_status status;
  uint32_t exp_val;

  // consider input and h0 are the "correct" value for comparsions
  zdnn_tensor_desc *input_tfrmd_desc = input->transformed_desc;
  zdnn_tensor_desc *h0_tfrmd_desc = h0->transformed_desc;

  // order of checks:
  // dims:
  //   - entries related to input dim4 (num_timesteps)
  //   - entries related to input dim2 (num_batches)
  //   - entries related to input dim1 (num_features)
  //   - dim3 of all tensors must be 1
  //   - dim2 of biases/hidden_biases must be 1
  //   - entries related to h0 dim4 (num_dirs)
  //   - entries related to h0 dim1 (num_hidden)
  // data-type and format
  //
  // layouts aren't checked as it doesn't impact the actual aiu_lstm_gru()
  // operation

  // input_tfrmd_desc dim4 (ts) must not be 0 as it is used for division and
  // will result in ABEND.
  if (input_tfrmd_desc->dim4 == 0) {
    return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                       "input dim4 tensor shape is invalid (found %d)",
                       input_tfrmd_desc->dim4);
  }

  // hn_output dim4 (ts) must be either 1 or same as input's
  // not using VERIFY_DIM4 macro because we have 2 valid values
  if ((hn_output->transformed_desc->dim4 != input_tfrmd_desc->dim4) &&
      (hn_output->transformed_desc->dim4 != 1)) {
    return ZDNN_STATUS(
        ZDNN_INVALID_SHAPE,
        "hn_output dim4 tensor shape is invalid (found %d, expects %d or 1)",
        hn_output->transformed_desc->dim4, input_tfrmd_desc->dim4);
  }

  // check input dim2 (num_batches)
  exp_val = input_tfrmd_desc->dim2;
  if ((status = VERIFY_DIM2(exp_val, h0, hn_output)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT &&
      ((status = VERIFY_DIM2(exp_val, c0, cf_output)) != ZDNN_OK)) {
    return status;
  }

  // weight's dim2 must be same as input's dim1 (num_features)
  exp_val = input_tfrmd_desc->dim1;
  if ((status = VERIFY_DIM2(exp_val, weights)) != ZDNN_OK) {
    return status;
  }

  // dim3 of all tensors should be 1
  if ((status = VERIFY_DIM3(1, input, h0, weights, biases, hidden_weights,
                            hidden_biases, hn_output)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT &&
      ((status = VERIFY_DIM3(1, c0, cf_output)) != ZDNN_OK)) {
    return status;
  }

  // check biases/hidden_biases dim2 = 1
  if ((status = VERIFY_DIM2(1, biases, hidden_biases)) != ZDNN_OK) {
    return status;
  }

  // all num_dirs must have the same value
  exp_val = h0_tfrmd_desc->dim4;
  if ((status = VERIFY_DIM4(exp_val, weights, biases, hidden_weights,
                            hidden_biases)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT &&
      (status = VERIFY_DIM4(exp_val, c0)) != ZDNN_OK) {
    return status;
  }

  // num_dirs must agree with "direction"
  exp_val = (direction == BIDIR) ? 2 : 1;
  if ((status = VERIFY_DIM4(exp_val, h0)) != ZDNN_OK) {
    return status;
  }

  // hn_output/cf_output dim1 = num_hidden (un-bir)
  //                            2 * PADDED(num_hidden) (bi-dir)
  exp_val = (h0->transformed_desc->dim4 == 2)
                ? 2 * PADDED(h0->transformed_desc->dim1)
                : h0->transformed_desc->dim1;
  if ((status = VERIFY_DIM1(exp_val, hn_output)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_DIM1(exp_val, cf_output)) != ZDNN_OK) {
      return status;
    }
  }

  // weight/biases/etc = num_gates * num_hidden
  exp_val = get_func_code_num_gates(function_code) *
            PADDED(h0->transformed_desc->dim1);
  if ((status = VERIFY_DIM1(exp_val, weights, biases, hidden_weights,
                            hidden_biases)) != ZDNN_OK) {
    return status;
  }

  // h0/c0 dim1 agree with each other
  exp_val = h0->transformed_desc->dim1;
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_DIM1(exp_val, c0)) != ZDNN_OK) {
      return status;
    }
  }

  // hidden_weights dim2 = num_hidden
  if ((status = VERIFY_DIM2(exp_val, hidden_weights)) != ZDNN_OK) {
    return status;
  }

  // check type and format
  if ((status = VERIFY_FIELDS(input_tfrmd_desc->type, input_tfrmd_desc->format,
                              h0, weights, biases, hidden_weights,
                              hidden_biases, hn_output)) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_FIELDS(input_tfrmd_desc->type,
                                input_tfrmd_desc->format, c0, cf_output)) !=
        ZDNN_OK) {
      return status;
    }
  }

  // If we reach this, all checks passed. Return OK
  return ZDNN_OK;
}

/// Verifies the condition of fused matmul bias add (broadcast) tensors.
///
/// The following conditions are checked against dims:
///
///  Matmul Op:
///  - The dimension-4-index-size of all input tensors and the output
///    tensor are not all equal.
///
///  Matmul Bcast1 Op:
///  - The dimension-4-index-size of input tensor 2, input tensor 3, and output
///    tensor are not equal.
///  - The dimension-4-index-size of input tensor 1 is not equal to 1.
///
///  Matmul Bcast23 Op:
///  - The dimension-4-index-size of input tensor 1 and output tensor are not
///    equal.
///  - The dimension-4-index-size of input tensor 2 and input tensor 3 are not
///    equal to 1.
///
/// The following conditions are checked against type and format:
///
///  Matmul Op:
///  - The data type of the input tensors differs from the data type of
///    the output tensor.
///
///  Quantized Matmul Op:
///  - The format of the input tensor 1 is not ZDNN_FORMAT_4DFEATURE.
///  - The data type of the input tensor 1 is not ZDNN_BINARY_INT8.
///  - The format of the input tensor 2 is not ZDNN_FORMAT_4DWEIGHTS.
///  - The data type of the input tensor 2 is not ZDNN_BINARY_INT8.
///  - The data type of the input tensor 3 differs from the data type of
///    the output tensor.
///  - The format of the input tensor 3 differs from the format of the output
///    tensor.
///  - The data type of the input tensor 3 differs from the data type of
///    the output tensor.
///
///  Quantized Matmul Op (on-the-fly):
///  - The format of the input tensor 2 is not ZDNN_FORMAT_4DWEIGHTS.
///  - The data type of the input tensor 2 is not ZDNN_BINARY_INT8.
///  - The data type of the input tensor 3 differs from the data type of
///    the output tensor.
///  - The format of the input tensor 1 differs from the format of input tensor
///    3 or the output tensor.
///  - The data type of the input tensor 1 differs from the data type of input
///    tensor 3 or the output tensor.
///
///  Common:
///  - The dimension-3-index-size of all input tensors and the output
///    tensor are not equal to 1.
///  - The dimension-2-index-size of input tensor 3 is not equal to 1.
///  - The dimension-2-index-size of input tensor 1 and output tensor
///    are not equal.
///  - The dimension-1-index-size of input tensor 1 is not equal to the
///    dimensions-2-index-size of input tensor 2.
///  - The dimension-1-index-size of input tensor 2, input tensor 3, and
///    output tensor are not equal.
///  - The format of the input tensor differs from the format of the output
///    tensor.
///  - The data type of the input tensors differs from the data type of
///    the output tensor.
///
/// \param[in] uint8_t function_code,
///                    NNPA_MATMUL_OP, NNPA_MATMUL_OP_BCAST1, or
///                    NNPA_MATMUL_OP_BCAST23
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] transpose_control Transpose control for first and second inputs
/// \param[in] a_offset Offset for input_a
/// \param[in] clip_min Clip min value
/// \param[in] clip_max Clip max value
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_SCALE
///         ZDNN_INVALID_OFFSET
///         ZDNN_INVALID_CLIPPING_VALUE
///
zdnn_status verify_matmul_op_common(
    uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    const void *transpose_control, const void *a_scale, const void *a_offset,
    const void *clip_min, const void *clip_max, const zdnn_ztensor *output) {

  zdnn_status status;
  zdnn_tensor_desc *input_a_tfrmd_desc = input_a->transformed_desc;
  zdnn_tensor_desc *input_b_tfrmd_desc = input_b->transformed_desc;

  func_sp_parm2_matmul *matmul_parm2 =
      (func_sp_parm2_matmul *)transpose_control;

  // check shapes first
  // For matmul_op, all tensors must have the same number of stacks (dim4)
  if (function_code == NNPA_MATMUL_OP) {
    if ((status = VERIFY_DIM4(input_a_tfrmd_desc->dim4, input_b, input_c,
                              output)) != ZDNN_OK) {
      return status;
    }
    // For matmul_bcast_op, input_a and output tensors must have the same
    // number of stacks (dim4) but input_b and input_c tensors must have a stack
    // dimension of 1 as they are broadcasted over each stack of the input.
  } else if (function_code == NNPA_MATMUL_OP_BCAST23) {
    if ((status = VERIFY_DIM4(input_a_tfrmd_desc->dim4, output)) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM4(1, input_b, input_c)) != ZDNN_OK) {
      return status;
    }
    // For matmul_bcast1_op, input_b, input_c, and output tensors must have the
    // same number of stacks (dim4) but input_a tensor must have a stack
    // dimension of 1 as it is broadcasted over each stack of the input.
  } else {
    if ((status = VERIFY_DIM4(1, input_a)) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM4(input_b->transformed_desc->dim4, input_c,
                              output)) != ZDNN_OK) {
      return status;
    }
  }

  if ((status = VERIFY_DIM3(1, input_a, input_b, input_c, output)) != ZDNN_OK) {
    return status;
  }

  if ((status = VERIFY_DIM2(1, input_c)) != ZDNN_OK) {
    return status;
  }

  if (matmul_parm2->transpose_a) {
    if (matmul_parm2->transpose_b) {
      // transpose_a and transpose_b [n, m] * [p, n] + [p] = [m, p]
      if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim1, output)) != ZDNN_OK) {
        return status;
      }

      if ((status = VERIFY_DIM1(input_a_tfrmd_desc->dim2, input_b)) !=
          ZDNN_OK) {
        return status;
      }

      if ((status = VERIFY_DIM1(input_b_tfrmd_desc->dim2, input_c, output)) !=
          ZDNN_OK) {
        return status;
      }
    } else {
      // transpose_a [n, m] * [n, p] + [p] = [m, p]
      if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim1, output)) != ZDNN_OK) {
        return status;
      }

      if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim2, input_b)) !=
          ZDNN_OK) {
        return status;
      }

      if ((status = VERIFY_DIM1(input_b_tfrmd_desc->dim1, input_c, output)) !=
          ZDNN_OK) {
        return status;
      }
    }
  } else if (matmul_parm2->transpose_b) {
    // transpose_b [m, n] * [p, n] + [p] = [m, p]
    if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim2, output)) != ZDNN_OK) {
      return status;
    }

    if ((status = VERIFY_DIM1(input_a_tfrmd_desc->dim1, input_b)) != ZDNN_OK) {
      return status;
    }

    if ((status = VERIFY_DIM1(input_b_tfrmd_desc->dim2, input_c, output)) !=
        ZDNN_OK) {
      return status;
    }
  } else {
    // no transpose [m, n] * [n, p] + [p] = [m, p]
    if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim2, output)) != ZDNN_OK) {
      return status;
    }

    if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim1, input_b)) != ZDNN_OK) {
      return status;
    }

    if ((status = VERIFY_DIM1(input_b_tfrmd_desc->dim1, input_c, output)) !=
        ZDNN_OK) {
      return status;
    }
  }

  if (input_a_tfrmd_desc->type == ZDNN_DLFLOAT16 &&
      input_b_tfrmd_desc->type == ZDNN_BINARY_INT8) {

    func_sp_parm3_matmul *matmul_parm3 = (func_sp_parm3_matmul *)a_scale;

    if ((matmul_parm3->rec_scale == 0) || (matmul_parm3->rec_scale == 0xFFFF) ||
        (matmul_parm3->rec_scale == 0x7FFF)) {
      return ZDNN_STATUS(ZDNN_INVALID_SCALE,
                         "a_scale value must be a numeric non-zero value.",
                         NO_ARG);
    }

    func_sp_parm4_matmul *matmul_parm4 = (func_sp_parm4_matmul *)a_offset;

    if ((matmul_parm4->offset == 0xFFFF) || (matmul_parm4->offset == 0x7FFF)) {
      return ZDNN_STATUS(ZDNN_INVALID_OFFSET,
                         "a_offset value must be a numeric value.", NO_ARG);
    }

    func_sp_parm9_matmul *matmul_parm9 = (func_sp_parm9_matmul *)clip_min;
    func_sp_parm10_matmul *matmul_parm10 = (func_sp_parm10_matmul *)clip_max;

    if ((int8_t)matmul_parm9->clip_min >= (int8_t)matmul_parm10->clip_max) {
      return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                         "The minimum-clip value (%d) not less than the "
                         "maximum-clip value (%d).",
                         (int8_t)matmul_parm9->clip_min,
                         (int8_t)matmul_parm10->clip_max);
    }
  }

  // then check type and format

  // When input_b type is ZDNN_DLFLOAT16 the operation is a normal matmul,
  // otherwise it is a quantized matmul.
  if (input_b_tfrmd_desc->type == ZDNN_DLFLOAT16) {
    if ((status =
             VERIFY_FIELDS(input_a_tfrmd_desc->type, input_a_tfrmd_desc->format,
                           input_b, input_c, output)) != ZDNN_OK) {
      return status;
    }
  } else {
    if (input_a_tfrmd_desc->type == ZDNN_DLFLOAT16) {
      if ((status = VERIFY_FIELDS(ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                                  input_a)) != ZDNN_OK) {
        return status;
      }
    } else {
      if ((status = VERIFY_FIELDS(ZDNN_BINARY_INT8, ZDNN_FORMAT_4DFEATURE,
                                  input_a)) != ZDNN_OK) {
        return status;
      }
    }

    if ((status = VERIFY_FIELDS(ZDNN_BINARY_INT8, ZDNN_FORMAT_4DWEIGHTS,
                                input_b)) != ZDNN_OK) {
      return status;
    }

    if ((status = VERIFY_FIELDS(ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE, input_c,
                                output)) != ZDNN_OK) {
      return status;
    }
  }

  return status;
}

/// Verifies the condition of input and output tensors for batchnorm operation.
///
/// The following conditions are checked:
///
///  -  The shape of input tensor 1 differs from the shape of the output
///     tensor.
///  -  The format of the input tensor differs from the format of the output
///     tensor.
///  -  The data type of the input tensors differs from the data type of the
///     output tensor.
///  -  Input tensors 1, 2, 3 and the output tensor do not have the same
///     dimension-1-index-size.
///  -  The dimension 2,3 and 4 index sizes of input tensors 2 and 3 are not 1
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_batchnorm_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const zdnn_ztensor *output) {

  zdnn_status status;
  zdnn_tensor_desc *input_tfrmd_desc = input_a->transformed_desc;

  // check shapes first

  if ((status = VERIFY_ALL_DIMS(input_tfrmd_desc->dim4, input_tfrmd_desc->dim3,
                                input_tfrmd_desc->dim2, input_tfrmd_desc->dim1,
                                output)) != ZDNN_OK) {
    return status;
  }

  if ((status = VERIFY_DIM1(input_a->transformed_desc->dim1, input_b, input_c,
                            output)) != ZDNN_OK) {
    return status;
  }

  if ((status = (IS_SHAPE_BIAS(input_b) | IS_SHAPE_BIAS(input_c))) != ZDNN_OK) {
    return status;
  }

  // then check type and format

  if ((status = VERIFY_FIELDS(input_a->transformed_desc->type,
                              input_a->transformed_desc->format, output)) !=
      ZDNN_OK) {
    return status;
  }

  return status;
}

/// Verifies the condition of input and output tensors for norm operation.
///
/// The following conditions are checked:
///
//   -  The 4th dimension is the same for all specified ztensors
///  -  The 3rd dimension is 1 for all input/output ztensors
///  -  The 2nd dimension is the same for all specified ztensors
///  -  The 1st dimension is the same for both input ztensors
//   -  The 1st dimension is 1 for the output ztensor
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_norm_tensors(const zdnn_ztensor *input_a,
                                const zdnn_ztensor *input_b,
                                const zdnn_ztensor *output) {

  zdnn_status status;
  zdnn_tensor_desc *input_tfrmd_desc = input_a->transformed_desc;

  // Ensure the dim-4 index size of all specified tensors are the same.
  if ((status = VERIFY_DIM4(input_tfrmd_desc->dim4, input_b, output)) !=
      ZDNN_OK) {
    return status;
  }

  // Ensure 3rd dimension is 1 for all input/output ztensors
  if ((status = VERIFY_DIM3(1, input_a, input_b, output)) != ZDNN_OK) {
    return status;
  }

  // Ensure the dim-2 index size of all specified tensors are the same.
  if ((status = VERIFY_DIM2(input_tfrmd_desc->dim2, input_b, output)) !=
      ZDNN_OK) {
    return status;
  }

  // Ensure 1st dimension is the same for both input ztensors
  if ((status = VERIFY_DIM1(input_tfrmd_desc->dim1, input_b)) != ZDNN_OK) {
    return status;
  }

  // Ensures 1st dimension is 1 for output ztensor
  if ((status = VERIFY_DIM1(1, output)) != ZDNN_OK) {
    return status;
  }

  // data type/format of inputb and output should match input1's
  if ((status = VERIFY_FIELDS(input_tfrmd_desc->type, input_tfrmd_desc->format,
                              input_b, output)) != ZDNN_OK) {
    return status;
  }

  return status;
}

zdnn_status verify_moments_tensors(const zdnn_ztensor *input_a,
                                   const void *bessel_correction_type,
                                   zdnn_ztensor *output_a,
                                   zdnn_ztensor *output_b) {

  zdnn_status status;

  func_sp_parm1_moments *moments_parm1 =
      (func_sp_parm1_moments *)bessel_correction_type;

  // The data-layout format and data type of all specified tensors
  // must be the same.
  zdnn_tensor_desc *input_a_tfrmd_desc = input_a->transformed_desc;
  if ((status =
           VERIFY_FIELDS(input_a_tfrmd_desc->type, input_a_tfrmd_desc->format,
                         output_a, output_b)) != ZDNN_OK) {
    return status;
  }

  // The value of FSP 1 must be either zero or one
  if ((moments_parm1->bessel_correction > 1)) {
    return ZDNN_INVALID_BESSEL_CORRECTION;
  }

  // If value of FSP 1 is one, then the total number of elements of input 1 must
  // be greater than one.
  if (moments_parm1->bessel_correction == MOMENTS_BESSEL_SAMPLE) {
    if ((input_a_tfrmd_desc->dim3 == 1) && (input_a_tfrmd_desc->dim2 == 1) &&
        (input_a_tfrmd_desc->dim1 == 1)) {
      return ZDNN_INVALID_BESSEL_CORRECTION;
    }
  }

  // Dimension-3 index size of output tensor 1 and 2 must be one
  if ((status = VERIFY_DIM3(1, output_a, output_b)) != ZDNN_OK) {
    return status;
  }

  // Dimension-2 index size of output tensor 1 and 2 must be one
  if ((status = VERIFY_DIM2(1, output_a, output_b)) != ZDNN_OK) {
    return status;
  }

  // Dimension-1 index size of output tensor 1 and 2 must be one
  if ((status = VERIFY_DIM1(1, output_a, output_b)) != ZDNN_OK) {
    return status;
  }

  // Dimension-4 index size of all specified tensors must be
  // the same.
  if ((status = VERIFY_DIM4(input_a_tfrmd_desc->dim4, output_a, output_b)) !=
      ZDNN_OK) {
    return status;
  }

  return status;
}

/// Verifies the condition of input and output tensors for layernorm
/// operation.
///
/// The following conditions are checked:
///
/// - Ensure the dim-4 index size of all specified tensors are the same.
/// - The dimension-1-index size of input tensor 1 must be the same as the
///   corresponding i index size in output tensor 1.
/// - The dimension-2-index size of input tensor 1 must be the same as the
///   corresponding 2 index size in output tensor 1.
/// - The dimension-3-index size of input tensor 1 must be the same as the
///   corresponding 3 index size in output tensor 1.
///
/// - The dimension-1-index size of input tensor 2 must be the same as the
///   corresponding i index size in input tensor 3.
/// - The dimension-2-index size of input tensor 2 must be the same as the
///   corresponding 2 index size in input tensor 3.
/// - The dimension-3-index size of input tensor 2 must be the same as the
///   corresponding 3 index size in input tensor 3.

/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] beta The beta value
/// \param[in] gamma The gamma value
/// \param[in] epsilon The epsilon value
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_layernorm_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const void *beta, const void *gamma,
                                     const void *epsilon,
                                     const zdnn_ztensor *output) {

  func_sp_parm1_layernorm *layernorm_parm1 = (func_sp_parm1_layernorm *)beta;
  func_sp_parm2_layernorm *layernorm_parm2 = (func_sp_parm2_layernorm *)gamma;
  func_sp_parm3_layernorm *layernorm_parm3 = (func_sp_parm3_layernorm *)epsilon;

  if ((layernorm_parm1->beta == 0xFFFF) || (layernorm_parm1->beta == 0x7FFF)) {
    return ZDNN_STATUS(ZDNN_INVALID_BETA, "Beta value must be a numeric value.",
                       NO_ARG);
  }

  if ((layernorm_parm2->gamma == 0xFFFF) ||
      (layernorm_parm2->gamma == 0x7FFF)) {
    return ZDNN_STATUS(ZDNN_INVALID_GAMMA,
                       "Gamma value must be a numeric value.", NO_ARG);
  }

  if ((layernorm_parm3->epsilon == 0xFFFF) ||
      (layernorm_parm3->epsilon == 0x7FFF)) {
    return ZDNN_STATUS(ZDNN_INVALID_EPSILON,
                       "Epsilon value must be a numeric value.", NO_ARG);
  }

  zdnn_status status;
  zdnn_tensor_desc *input_a_tfrmd_desc = input_a->transformed_desc;

  // Ensure the dim-4 index size of all specified tensors are the same.
  if ((status = VERIFY_DIM4(input_a_tfrmd_desc->dim4, input_b, input_c,
                            output)) != ZDNN_OK) {
    return status;
  }

  // The dimension-1-index size of input tensor 1 must be the same as the
  // corresponding i index size in output tensor 1.
  if ((status = VERIFY_DIM1(input_a_tfrmd_desc->dim1, output)) != ZDNN_OK) {
    return status;
  }

  // The dimension-2-index size of input tensor 1 must be the same as the
  // corresponding 2 index size in output tensor 1.
  if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim2, output)) != ZDNN_OK) {
    return status;
  }

  // The dimension-3-index size of input tensor 1 must be the same as the
  // corresponding 3 index size in output tensor 1.
  if ((status = VERIFY_DIM3(input_a_tfrmd_desc->dim3, output)) != ZDNN_OK) {
    return status;
  }

  zdnn_tensor_desc *input_b_tfrmd_desc = input_b->transformed_desc;

  // The dimension-1-index size of input tensor 2 must be the same as the
  // corresponding i index size in input tensor 3.
  if ((status = VERIFY_DIM1(input_b_tfrmd_desc->dim1, input_c)) != ZDNN_OK) {
    return status;
  }

  // The dimension-2-index size of input tensor 2 must be the same as the
  // corresponding 2 index size in input tensor 3.
  if ((status = VERIFY_DIM2(input_b_tfrmd_desc->dim2, input_c)) != ZDNN_OK) {
    return status;
  }

  // The dimension-3-index size of input tensor 2 must be the same as the
  // corresponding 3 index size in input tensor 3.
  if ((status = VERIFY_DIM3(input_b_tfrmd_desc->dim3, input_c)) != ZDNN_OK) {
    return status;
  }

  // data type/format of input3 and output should match input1's
  if ((status =
           VERIFY_FIELDS(input_a_tfrmd_desc->type, input_a_tfrmd_desc->format,
                         input_b, input_c, output)) != ZDNN_OK) {
    return status;
  }

  return status;
}

/// Verifies the condition of input and output tensors for average pool
/// operation.
///
/// \param[in] input input tensor
/// \param[in] padding_type
/// \param[in] stride_width
/// \param[in] stride_height
/// \param[in] kernel_width
/// \param[in] kernel_height
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_STRIDE_PADDING
///         ZDNN_INVALID_STRIDES
///
zdnn_status
verify_pool_avg_max_tensors(const zdnn_ztensor *input, const void *padding_type,
                            const void *stride_width, const void *stride_height,
                            const void *kernel_width, const void *kernel_height,
                            const zdnn_ztensor *output) {

  // Convenience variables used later
  zdnn_status status;

  uint32_t input_c_size = input->transformed_desc->dim1;
  uint32_t input_w_size = input->transformed_desc->dim2;
  uint32_t input_h_size = input->transformed_desc->dim3;
  uint32_t input_n_size = input->transformed_desc->dim4;

  uint32_t output_w_size = output->transformed_desc->dim2;
  uint32_t output_h_size = output->transformed_desc->dim3;

  uint32_t expected_output_w_size, expected_output_h_size;

  func_sp_parm1_pool2d *pool2d_parm1 = (func_sp_parm1_pool2d *)padding_type;
  func_sp_parm2_pool2d *pool2d_parm2 = (func_sp_parm2_pool2d *)stride_width;
  func_sp_parm3_pool2d *pool2d_parm3 = (func_sp_parm3_pool2d *)stride_height;
  func_sp_parm4_pool2d *pool2d_parm4 = (func_sp_parm4_pool2d *)kernel_width;
  func_sp_parm5_pool2d *pool2d_parm5 = (func_sp_parm5_pool2d *)kernel_height;

  LOG_DEBUG(
      "%s() - padding_type: %d, input_ztensor->transformed_desc shape: (%d, "
      "%d, %d, %d) (NHWC order), kernel_height: %d, kernel_width: %d, "
      "stride_height: %d, stride_width %d, output_ztensor->transformed_desc "
      "shape: (%d, %d, %d, %d) (NHWC order)",
      __func__, pool2d_parm1->pad, input->transformed_desc->dim4, input_h_size,
      input_w_size, input->transformed_desc->dim1, pool2d_parm5->kernel_height,
      pool2d_parm4->kernel_width, pool2d_parm3->stride_height,
      pool2d_parm2->stride_width, output->transformed_desc->dim4, output_h_size,
      output_w_size, output->transformed_desc->dim1);

  // check tensor shapes first
  if ((status = (VERIFY_DIM4(input_n_size, output) |
                 VERIFY_DIM1(input_c_size, output))) != ZDNN_OK) {
    return status;
  }

  // Check that input and output have the same type and format
  // Note: If the output data type is invalid, the zAIU may raise a
  // condition code before we'd reach this exception condition.
  if ((status = VERIFY_FIELDS(input->transformed_desc->type,
                              input->transformed_desc->format, output)) !=
      ZDNN_OK) {
    return status;
  }

  // Checks for when strides are 0
  if (pool2d_parm2->stride_width == 0 && pool2d_parm3->stride_height == 0) {
    if (input_w_size != pool2d_parm4->kernel_width) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "When strides are 0, the input tensor's width "
                         "(%d) and kernel_width (%d) must be equal.",
                         input_w_size, pool2d_parm4->kernel_width);
    }
    if (input_h_size != pool2d_parm5->kernel_height) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "When strides are 0, the input tensor's height "
                         "(%d) and kernel_height (%d) must be equal.",
                         input_h_size, pool2d_parm5->kernel_height);
    }
    if (output_w_size != 1 || output_h_size != 1) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "When strides are 0, the output tensor's height "
                         "(%d) and width (%d) must both be 1",
                         output_h_size, output_w_size);
    }
    if (pool2d_parm1->pad != VALID_PADDING) {
      return ZDNN_STATUS(
          ZDNN_INVALID_STRIDE_PADDING,
          "When strides are 0, the padding_type must be VALID_PADDING", NO_ARG);
    }
    // Checks that if one stride is nonzero then both must be nonzero.
    // We're following order as described in doc to make future comparing easier
    // so we can't just make this the final "else" condition. This boolean is an
    // XOR and will only be true if one (and only one) of these are nonzero.
  } else if (!pool2d_parm2->stride_width != !pool2d_parm3->stride_height) {
    return ZDNN_STATUS(ZDNN_INVALID_STRIDES,
                       "When either stride is non-zero, then both strides "
                       "must be non-zero. Stride width (%d), Stride height "
                       "(%d)",
                       output_h_size, output_w_size);
    // Checks for when strides are both nonzero
  } else {
    bool check_output_size = true;
    switch (pool2d_parm1->pad) {

    case VALID_PADDING:
      if (pool2d_parm4->kernel_width > input_w_size) {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "When VALID_PADDING is used, the the kernel_width (%d) "
            "must not be larger than the input tensor's width (%d) ",
            pool2d_parm4->kernel_width, input_w_size);
      }
      if (pool2d_parm5->kernel_height > input_h_size) {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "When VALID_PADDING is used, the the kernel_height (%d) "
            "must not be larger than the input tensor's height (%d) ",
            pool2d_parm5->kernel_height, input_h_size);
      }
      expected_output_w_size =
          CEIL(input_w_size - pool2d_parm4->kernel_width + 1,
               pool2d_parm2->stride_width);
      expected_output_h_size =
          CEIL(input_h_size - pool2d_parm5->kernel_height + 1,
               pool2d_parm3->stride_height);
      break;

    case SAME_PADDING:
      expected_output_w_size = CEIL(input_w_size, pool2d_parm2->stride_width);
      expected_output_h_size = CEIL(input_h_size, pool2d_parm3->stride_height);
      break;

    default:
      // An invalid padding type raises a condition code from the hardware
      // so it isn't something we need to raise an error for here. However
      // without a type we can't know what to expect for the later output size
      // check. Instead we log a warning and will skip that check.
      LOG_WARN("Not valid padding type (%d)", pool2d_parm1->pad);
      check_output_size = false;
      break;
    }

    if (check_output_size) {
      if (output_w_size != expected_output_w_size ||
          output_h_size != expected_output_h_size) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "Expected the output tensor's height (%d) to "
                           "be %d and width (%d) to be %d",
                           output_h_size, expected_output_h_size, output_w_size,
                           expected_output_h_size);
      }
    }
  }

  return ZDNN_STATUS_OK;
}

/// Verifies the condition of input and output tensors for convolution
/// operation.
///
/// \param[in] input input tensor
/// \param[in] kernel input kernel tensor
/// \param[in] bias input bias tensor
/// \param[in] pad_n_act padding type and act function in zAIU's
///                      function-specific-parameter-1 format
/// \param[in] stride_width
/// \param[in] stride_height
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_STRIDE_PADDING
///         ZDNN_INVALID_STRIDES
///
zdnn_status
verify_conv2d_tensors(const zdnn_ztensor *input, const zdnn_ztensor *kernel,
                      const zdnn_ztensor *bias, const void *pad_n_act,
                      const void *stride_width, const void *stride_height,
                      const void *reserved_n_clipping,
                      const zdnn_ztensor *output) {

  zdnn_status status;

  // hw doc calls input => input1, kernel => input2, bias => input3
  //              stride_height => dim3_stride, stride_width => dim2_stride
  zdnn_tensor_desc *input_desc = input->transformed_desc,
                   *input_kernel_desc = kernel->transformed_desc,
                   *output_desc = output->transformed_desc;

  func_sp_parm1_conv2d *conv2d_parm1 = (func_sp_parm1_conv2d *)pad_n_act;
  func_sp_parm2_conv2d *conv2d_parm2 = (func_sp_parm2_conv2d *)stride_width;
  func_sp_parm3_conv2d *conv2d_parm3 = (func_sp_parm3_conv2d *)stride_height;
  func_sp_parm4_conv2d *conv2d_parm4 =
      (func_sp_parm4_conv2d *)reserved_n_clipping;

  // The dimension-2, dimension-3, and dimension-4 index sizes of the input3
  // must be 1.
  if ((status = (IS_SHAPE_BIAS(bias))) != ZDNN_OK) {
    return status;
  }

  // The dimension-4-index-size of the output must be equal to the
  // dimension-4-index-size of the input1.
  if ((status = VERIFY_DIM4(input_desc->dim4, output)) != ZDNN_OK) {
    return status;
  }

  // The dimension-1 index size of the output must be equal to the dimension-1
  // index size of the input2 and the dimension-1-index size of the input3.
  if ((status = VERIFY_DIM1(output_desc->dim1, kernel, bias)) != ZDNN_OK) {
    return status;
  }

  // The dimension-1 index size of the input1 must be equal to the dimension-2
  // index size of the input2.
  if ((status = VERIFY_DIM1(input_kernel_desc->dim2, input)) != ZDNN_OK) {
    return status;
  }

  if (!conv2d_parm3->stride_height &&
      !conv2d_parm2->stride_width) { // both zero

    // The input1 dimension-2-index-size must be equal to the
    // dimension-3-index-size of input2.
    if (input_desc->dim2 != input_kernel_desc->dim3) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "input_desc->dim2"
                         " (%d) must be equal to "
                         "input_kernel_desc->dim3"
                         " (%d)\n",
                         input_desc->dim2, input_kernel_desc->dim3);
    }

    // The input1 dimension-3-index-size must be equal to the
    // dimension-4-index-size of input2.
    if (input_desc->dim3 != input_kernel_desc->dim4) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "input_desc->dim3"
                         " (%d) must be equal to "
                         "input_kernel_desc->dim4"
                         " (%d)\n",
                         input_desc->dim3, input_kernel_desc->dim4);
    }

    // The dimension-2-index-size and the dimension-3-index-size of the output
    // must be one.
    if (output_desc->dim2 != 1) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "output_desc->dim2"
                         " (%d) must be 1\n",
                         output_desc->dim2);
    }

    if (output_desc->dim3 != 1) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "output_desc->dim3"
                         " (%d) must be 1\n",
                         output_desc->dim3);
    }

    // The specified padding must be VALID
    if (conv2d_parm1->pad != VALID_PADDING) {
      return ZDNN_STATUS(ZDNN_INVALID_STRIDE_PADDING,
                         "padding must be VALID_PADDING when both "
                         "stride_height (%d) and stride_width (%d) are zero",
                         conv2d_parm3->stride_height,
                         conv2d_parm2->stride_width);
    }

  } else if (conv2d_parm3->stride_height &&
             conv2d_parm2->stride_width) { // both > 0

    switch (conv2d_parm1->pad) {
    case VALID_PADDING:
      // the dimension-2-index-size of the input1 must be greater than or equal
      // to the dimension-3-index-size of input2.
      if (input_desc->dim2 < input_kernel_desc->dim3) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "input_desc->dim2"
                           " (%d) must be greater than or equal to "
                           "input_kernel_desc->dim3"
                           " (%d)\n",
                           input_desc->dim2, input_kernel_desc->dim3);
      }

      //  the dimension-3-index-size of the input1 must be greater than or equal
      //  to the dimension-4-index-size of the input2
      if (input_desc->dim3 < input_kernel_desc->dim4) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "input_desc->dim3"
                           " (%d) must be greater than or equal to "
                           "input_kernel_desc->dim4"
                           " (%d)\n",
                           input_desc->dim3, input_kernel_desc->dim4);
      }

      if ((status =
               VERIFY_DIM2(CEIL(input_desc->dim2 - input_kernel_desc->dim3 + 1,
                                conv2d_parm2->stride_width),
                           output) |
               VERIFY_DIM3(CEIL(input_desc->dim3 - input_kernel_desc->dim4 + 1,
                                conv2d_parm3->stride_height),
                           output)) != ZDNN_OK) {

        return status;
      }
      break;
    case SAME_PADDING:
      if ((status =
               VERIFY_DIM2(CEIL(input_desc->dim2, conv2d_parm2->stride_width),
                           output) |
               VERIFY_DIM3(CEIL(input_desc->dim3, conv2d_parm3->stride_height),
                           output)) != ZDNN_OK) {
        return status;
      }
      break;
    default:
      // keep going to the next check, the hardware will handle it with function
      // specific RC later
      LOG_WARN("Not valid padding type (%d)", conv2d_parm1->pad);
      break;
    }

  } else { // only either is zero
    return ZDNN_STATUS(ZDNN_INVALID_STRIDES,
                       "either both stride_height (%d) and stride_width (%d) "
                       "must be non-zero or both be must be zero\n",
                       conv2d_parm3->stride_height, conv2d_parm2->stride_width);
  }

  // data type/format of input3 and output should match input1's
  if ((status = VERIFY_FIELDS(input_desc->type, input_desc->format, bias,
                              output)) != ZDNN_OK) {
    return status;
  }

  // data type of input2 should match input1's
  // not checking input2's format (should be ZDNN_FORMAT_4DKERNEL), let hardware
  // handle it with reponse code if not
  if ((status = VERIFY_FIELDS(input_desc->type, input_kernel_desc->format,
                              kernel)) != ZDNN_OK) {
    return status;
  }

  // If activation is set to RELU, check clipping value.
  if (conv2d_parm1->act == CONV2D_ACT_RELU) {
    // Clipping value cannot be negative.
    if (conv2d_parm4->clipping_value & 0x8000) {
      return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                         "Clipping value cannot be negative.", NO_ARG);
    }
    // Clipping value cannot be NINF+
    if (conv2d_parm4->clipping_value == 0x7FFF) {
      return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                         "Conversion of clipping value unsuccessful.", NO_ARG);
    }
  }

  return ZDNN_STATUS_OK;
}

/// Verifies the condition of input and output tensors for relu operation.
///
/// \param[in] input input tensor
/// \param[in] reserved_n_clipping reserved and clipping value in zAIU's
///                                function-specific-parameter-1 format
/// \param[in] reserved_n_adjustment reserved and adjustment factor in zAIU's
///                                  function-specific-parameter-2 format
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_CLIPPING_VALUE
///         ZDNN_INVALID_ADJUSTMENT_FACTOR
///
zdnn_status verify_relu_tensors(const zdnn_ztensor *input,
                                const void *reserved_n_clipping,
                                const void *reserved_n_adjustment,
                                const zdnn_ztensor *output) {

  zdnn_status status;

  if ((status = verify_tensors(input, NULL, NULL, output)) != ZDNN_OK) {
    return status;
  }

  func_sp_parm1_relu *relu_parm1 = (func_sp_parm1_relu *)reserved_n_clipping;
  // Clipping value cannot be negative.
  if (relu_parm1->clipping_value & 0x8000) {
    return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                       "Clipping value cannot be negative.", NO_ARG);
  }
  // Clipping value cannot be NINF+
  if (relu_parm1->clipping_value == 0x7FFF) {
    return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                       "Conversion of clipping value unsuccessful.", NO_ARG);
  }

  func_sp_parm2_relu *relu_parm2 = (func_sp_parm2_relu *)reserved_n_adjustment;
  // Adjustment factor cannot be negative.
  if (relu_parm2->adjustment_factor & 0x8000) {
    return ZDNN_STATUS(ZDNN_INVALID_ADJUSTMENT_FACTOR,
                       "Adjustment factor cannot be negative.", NO_ARG);
  }
  // Adjustment factor cannot be NINF+
  if (relu_parm2->adjustment_factor == 0x7FFF) {
    return ZDNN_STATUS(ZDNN_INVALID_ADJUSTMENT_FACTOR,
                       "Conversion of adjustment factor unsuccessful.", NO_ARG);
  }
  // Adjustment factor cannot be greater than 1.
  if (relu_parm2->adjustment_factor > 0x3E00) {
    return ZDNN_STATUS(ZDNN_INVALID_ADJUSTMENT_FACTOR,
                       "Adjustment factor cannot be greater than 1.", NO_ARG);
  }

  return ZDNN_STATUS_OK;
}

/// Verifies the condition of input and output tensors for invsqrt operation.
///
/// \param[in] input input tensor
/// \param[in] reserved_n_epsilon reserved and epsilon value in zAIU's
///                               function-specific-parameter-1 format
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_EPSILON
///
zdnn_status verify_invsqrt_tensors(const zdnn_ztensor *input,
                                   const void *reserved_n_epsilon,
                                   const zdnn_ztensor *output) {

  zdnn_status status;

  // The data-layout format and data type of all specified tensors
  // must be the same.
  zdnn_tensor_desc *input_tfrmd_desc = input->transformed_desc;
  if ((status = VERIFY_FIELDS(input_tfrmd_desc->type, input_tfrmd_desc->format,
                              output)) != ZDNN_OK) {
    return status;
  }

  // Verify input and output dims are the same.
  if ((status = verify_tensors(input, NULL, NULL, output)) != ZDNN_OK) {
    return status;
  }

  // Epsilon cannot be NINF- or NINF+
  func_sp_parm1_invsqrt *invsqrt_parm1 =
      (func_sp_parm1_invsqrt *)reserved_n_epsilon;
  if ((invsqrt_parm1->epsilon == 0xFFFF) ||
      (invsqrt_parm1->epsilon == 0x7FFF)) {
    return ZDNN_STATUS(ZDNN_INVALID_EPSILON,
                       "Conversion of epsilon unsuccessful.", NO_ARG);
  }

  return ZDNN_STATUS_OK;
}

zdnn_status verify_transform_tensors(const zdnn_ztensor *input,
                                     const zdnn_ztensor *output,
                                     const void *toc, const void *min_clipping,
                                     const void *max_clipping) {
  zdnn_status status;
  if ((status = VERIFY_ALL_DIMS(
           input->transformed_desc->dim4, input->transformed_desc->dim3,
           input->transformed_desc->dim2, input->transformed_desc->dim1,
           output)) != ZDNN_OK) {
    return status;
  }

  func_sp_parm1_transform *transform_parm1 = (func_sp_parm1_transform *)toc;
  func_sp_parm4_transform *transform_parm4 =
      (func_sp_parm4_transform *)min_clipping;
  func_sp_parm5_transform *transform_parm5 =
      (func_sp_parm5_transform *)max_clipping;

  if (transform_parm1->toc == NNPA_TOC_STICK_INT8) {
    if ((int8_t)transform_parm4->clip_min >=
        (int8_t)transform_parm5->clip_max) {
      return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                         "The minimum-clip value (%d) not less than the "
                         "maximum-clip value (%d).",
                         (int8_t)transform_parm4->clip_min,
                         (int8_t)transform_parm5->clip_max);
    }
  }

  return ZDNN_STATUS_OK;
}

zdnn_status verify_reduce_tensors(const zdnn_ztensor *input,
                                  const zdnn_ztensor *output) {
  zdnn_status status;
  if ((status = VERIFY_DIM4(input->transformed_desc->dim4, output)) !=
      ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM3(input->transformed_desc->dim3, output)) !=
      ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM2(input->transformed_desc->dim2, output)) !=
      ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM1(1, output)) != ZDNN_OK) {
    return status;
  }

  if (input->transformed_desc->format != output->transformed_desc->format) {
    return ZDNN_STATUS(
        ZDNN_INVALID_FORMAT,
        "Output tensor format is invalid (found %s (%d), expects %s (%d))",
        get_data_format_str(input->transformed_desc->format),
        input->transformed_desc->format,
        get_data_format_str(ZDNN_FORMAT_4DFEATURE), ZDNN_FORMAT_4DFEATURE);
  }

  if (input->transformed_desc->type != ZDNN_DLFLOAT16) {
    return ZDNN_STATUS(
        ZDNN_INVALID_TYPE,
        "Input tensor type is invalid (found %s (%d), expects %s (%d))",
        get_data_type_str(input->transformed_desc->type),
        input->transformed_desc->type, get_data_type_str(ZDNN_DLFLOAT16),
        ZDNN_DLFLOAT16);
  }

  if (output->transformed_desc->type != ZDNN_DLFLOAT16 &&
      output->transformed_desc->type != ZDNN_BINARY_INT32) {
    return ZDNN_STATUS(ZDNN_INVALID_TYPE,
                       "Output tensor type is invalid (found %s (%d), expects "
                       "%s (%d) or %s (%d))",
                       get_data_type_str(output->transformed_desc->type),
                       output->transformed_desc->type,
                       get_data_type_str(ZDNN_DLFLOAT16), ZDNN_DLFLOAT16,
                       get_data_type_str(ZDNN_BINARY_INT32), ZDNN_BINARY_INT32);
  }

  return ZDNN_STATUS_OK;
}
