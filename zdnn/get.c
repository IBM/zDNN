
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

#ifdef __MVS__
#pragma export(zdnn_get_library_version_str)
#pragma export(zdnn_get_library_version)
#pragma export(zdnn_get_max_limit)
#pragma export(zdnn_get_min_limit)
#endif

#define DECLARE_DATA_LAYOUT_STR(a) static const char *DATA_LAYOUT_STR_##a = #a;

// const char *DATA_LAYOUT_STR_X
DECLARE_DATA_LAYOUT_STR(ZDNN_1D)
DECLARE_DATA_LAYOUT_STR(ZDNN_2D)
DECLARE_DATA_LAYOUT_STR(ZDNN_2DS)
DECLARE_DATA_LAYOUT_STR(ZDNN_3D)
DECLARE_DATA_LAYOUT_STR(ZDNN_3DS)
DECLARE_DATA_LAYOUT_STR(ZDNN_ZRH)
DECLARE_DATA_LAYOUT_STR(ZDNN_4D)
DECLARE_DATA_LAYOUT_STR(ZDNN_4DS)
DECLARE_DATA_LAYOUT_STR(ZDNN_NHWC)
DECLARE_DATA_LAYOUT_STR(ZDNN_NCHW)
DECLARE_DATA_LAYOUT_STR(ZDNN_FICO)
DECLARE_DATA_LAYOUT_STR(ZDNN_HWCK)
DECLARE_DATA_LAYOUT_STR(ZDNN_BIDIR_ZRH)
DECLARE_DATA_LAYOUT_STR(ZDNN_BIDIR_FICO)

#define DECLARE_DATA_FORMAT_STR(a) static const char *DATA_FORMAT_STR_##a = #a;

// const char *DATA_FORMAT_STR_X
DECLARE_DATA_FORMAT_STR(ZDNN_FORMAT_4DFEATURE)
DECLARE_DATA_FORMAT_STR(ZDNN_FORMAT_4DKERNEL)
DECLARE_DATA_FORMAT_STR(ZDNN_FORMAT_4DWEIGHTS)
DECLARE_DATA_FORMAT_STR(ZDNN_FORMAT_4DGENERIC)

#define DECLARE_DATA_TYPE_STR(a) static const char *DATA_TYPE_STR_##a = #a;

// const char *DATA_TYPE_STR_X
DECLARE_DATA_TYPE_STR(ZDNN_BINARY_INT8)
DECLARE_DATA_TYPE_STR(ZDNN_BINARY_INT32)
DECLARE_DATA_TYPE_STR(ZDNN_BINARY_FP32)
DECLARE_DATA_TYPE_STR(INT8)
DECLARE_DATA_TYPE_STR(BFLOAT)
DECLARE_DATA_TYPE_STR(FP16)
DECLARE_DATA_TYPE_STR(FP32)
DECLARE_DATA_TYPE_STR(ZDNN_DLFLOAT16)

#define DECLARE_FUNCTION_CODE_STR(a)                                           \
  static const char *FUNCTION_CODE_STR_##a = #a;

// const char *FUNCTION_CODE_STR_X
DECLARE_FUNCTION_CODE_STR(NNPA_QAF)
DECLARE_FUNCTION_CODE_STR(NNPA_ADD)
DECLARE_FUNCTION_CODE_STR(NNPA_SUB)
DECLARE_FUNCTION_CODE_STR(NNPA_MUL)
DECLARE_FUNCTION_CODE_STR(NNPA_DIV)
DECLARE_FUNCTION_CODE_STR(NNPA_MIN)
DECLARE_FUNCTION_CODE_STR(NNPA_MAX)
DECLARE_FUNCTION_CODE_STR(NNPA_LOG)
DECLARE_FUNCTION_CODE_STR(NNPA_EXP)
DECLARE_FUNCTION_CODE_STR(NNPA_RELU)
DECLARE_FUNCTION_CODE_STR(NNPA_TANH)
DECLARE_FUNCTION_CODE_STR(NNPA_SIGMOID)
DECLARE_FUNCTION_CODE_STR(NNPA_SOFTMAX)
DECLARE_FUNCTION_CODE_STR(NNPA_BATCHNORMALIZATION)
DECLARE_FUNCTION_CODE_STR(NNPA_MAXPOOL2D)
DECLARE_FUNCTION_CODE_STR(NNPA_AVGPOOL2D)
DECLARE_FUNCTION_CODE_STR(NNPA_LSTMACT)
DECLARE_FUNCTION_CODE_STR(NNPA_GRUACT)
DECLARE_FUNCTION_CODE_STR(NNPA_CONVOLUTION)
DECLARE_FUNCTION_CODE_STR(NNPA_MATMUL_OP)
DECLARE_FUNCTION_CODE_STR(NNPA_MATMUL_OP_BCAST23)
DECLARE_FUNCTION_CODE_STR(NNPA_SQRT)
DECLARE_FUNCTION_CODE_STR(NNPA_INVSQRT)
DECLARE_FUNCTION_CODE_STR(NNPA_GELU)
DECLARE_FUNCTION_CODE_STR(NNPA_MOMENTS)
DECLARE_FUNCTION_CODE_STR(NNPA_LAYERNORM)
DECLARE_FUNCTION_CODE_STR(NNPA_NORM)
DECLARE_FUNCTION_CODE_STR(NNPA_MATMUL_OP_BCAST1)
DECLARE_FUNCTION_CODE_STR(NNPA_TRANSFORM)
DECLARE_FUNCTION_CODE_STR(NNPA_REDUCE)

#define DECLARE_RNN_DIR_STR(a) static const char *RNN_DIR_STR_##a = #a;

// const char *RNN_DIR_STR_X
DECLARE_RNN_DIR_STR(FWD)
DECLARE_RNN_DIR_STR(BWD)
DECLARE_RNN_DIR_STR(BIDIR)

#define DECLARE_SOFTMAX_ACT_STR(a) static const char *SOFTMAX_ACT_STR_##a = #a;

// const char *SOFTMAX_ACT_STR_X
DECLARE_SOFTMAX_ACT_STR(SOFTMAX_ACT_NONE)
DECLARE_SOFTMAX_ACT_STR(SOFTMAX_ACT_LOG)

#define DECLARE_MATMUL_OP_STR(a) static const char *MATMUL_OP_STR_##a = #a;

// const char *MATMUL_OP_STR_X
DECLARE_MATMUL_OP_STR(MATMUL_OP_ADDITION);
DECLARE_MATMUL_OP_STR(MATMUL_OP_GREATER);
DECLARE_MATMUL_OP_STR(MATMUL_OP_GREATER_EQUAL);
DECLARE_MATMUL_OP_STR(MATMUL_OP_EQUAL);
DECLARE_MATMUL_OP_STR(MATMUL_OP_NOT_EQUAL);
DECLARE_MATMUL_OP_STR(MATMUL_OP_LESSER_EQUAL);
DECLARE_MATMUL_OP_STR(MATMUL_OP_LESSER);

#define DECLARE_MATMUL_BCAST_OP_STR(a)                                         \
  static const char *MATMUL_BCAST_OP_STR_##a = #a;

// const char *MATMUL_BCAST_OP_STR_X
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_ADDITION);
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_GREATER);
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_GREATER_EQUAL);
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_EQUAL);
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_NOT_EQUAL);
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_LESSER_EQUAL);
DECLARE_MATMUL_BCAST_OP_STR(MATMUL_BCAST_OP_LESSER);

#define DECLARE_POOL_PADDING_STR(a)                                            \
  static const char *POOL_PADDING_STR_##a = #a;

// const char *POOL_PADDING_STR_X
DECLARE_POOL_PADDING_STR(SAME_PADDING)
DECLARE_POOL_PADDING_STR(VALID_PADDING)

#define DECLARE_CONV2D_ACT_STR(a) static const char *CONV2D_ACT_STR_##a = #a;

// const char *CONV2D_ACT_STR_X
DECLARE_CONV2D_ACT_STR(CONV2D_ACT_NONE)
DECLARE_CONV2D_ACT_STR(CONV2D_ACT_RELU)

#define DECLARE_REDUCE_OP_STR(a) static const char *REDUCE_OP_STR_##a = #a;

// const char *REDUCE_OP_STR_X
DECLARE_REDUCE_OP_STR(REDUCE_OP_MINIMUM)
DECLARE_REDUCE_OP_STR(REDUCE_OP_MINIMUM_IDX)
DECLARE_REDUCE_OP_STR(REDUCE_OP_MAXIMUM)
DECLARE_REDUCE_OP_STR(REDUCE_OP_MAXIMUM_IDX)

#define DECLARE_BESSEL_CORRECTION_STR(a)                                       \
  static const char *BESSEL_CORRECTION_STR_##a = #a;

// const char *BESSEL_OP_STR_X
DECLARE_BESSEL_CORRECTION_STR(MOMENTS_BESSEL_POPULATION)
DECLARE_BESSEL_CORRECTION_STR(MOMENTS_BESSEL_SAMPLE)

static const char *UNDEFINED_STR = "UNDEFINED";

/// Returns number of dimension of a layout
///
/// \param[in] layout tensor layout, must be a non-concatenated layout
///
/// \return number of dimensions, or 0 if concatenated or no such layout exists
///
short get_data_layout_dims(zdnn_data_layouts layout) {

#define CASE_RTN_DIM(a, b)                                                     \
  case a:                                                                      \
    return b;

  switch (layout) {
    CASE_RTN_DIM(ZDNN_1D, 1);
    CASE_RTN_DIM(ZDNN_2D, 2);
    CASE_RTN_DIM(ZDNN_2DS, 2);
    CASE_RTN_DIM(ZDNN_3D, 3);
    CASE_RTN_DIM(ZDNN_3DS, 3);
    CASE_RTN_DIM(ZDNN_4D, 4);
    CASE_RTN_DIM(ZDNN_4DS, 4);
    CASE_RTN_DIM(ZDNN_NHWC, 4);
    CASE_RTN_DIM(ZDNN_NCHW, 4);
    CASE_RTN_DIM(ZDNN_HWCK, 4);
  default:
    LOG_WARN("Unknown or concatenated layout: %d", layout);
    return 0;
  }
#undef CASE_RTN_DIM
}

/// Returns number of gates of a concatenated layout
///
/// \param[in] layout data layout, must be a concatenated layout
///
/// \return number of gates, or 0 if not concatenated or no such layout exists
///
short get_data_layout_num_gates(zdnn_data_layouts layout) {

#define CASE_RTN_GATES(a, b)                                                   \
  case a:                                                                      \
    return b;

  switch (layout) {
    CASE_RTN_GATES(ZDNN_ZRH, 3);
    CASE_RTN_GATES(ZDNN_FICO, 4);
    CASE_RTN_GATES(ZDNN_BIDIR_ZRH, 3);
    CASE_RTN_GATES(ZDNN_BIDIR_FICO, 4);
  default:
    LOG_WARN("Unknown or not concatenated layout: %d", layout);
    return 0;
  }
#undef CASE_RTN_GATES
}

/// Returns concatenated dim1 value based on concatenation info
///
/// \param val incoming dim1 value
/// \param info concatenation info
///
/// \returns concatenated dim1 value
///
uint32_t get_rnn_concatenated_dim1(uint32_t val, zdnn_concat_info info) {
  if (CONCAT_RNN_TYPE(info) == RNN_TYPE_LSTM) {
    return PADDED(val) * 4;
  } else if (CONCAT_RNN_TYPE(info) == RNN_TYPE_GRU) {
    return PADDED(val) * 3;
  } else {
    return val;
  }
}

/// Returns concatenated dim2 value based on concatenation info
///
/// \param val incoming dim2 value
/// \param info concatenation info
///
/// \returns concatenated dim2 value
///
uint32_t get_rnn_concatenated_dim2(uint32_t val, zdnn_concat_info info) {
  // the only case we need vertical concatenation is when a weight tensor is
  // used with bidir output from the previous layer.
  if (CONCAT_USAGE(info) == USAGE_WEIGHTS &&
      CONCAT_PREV_LAYER(info) == PREV_LAYER_BIDIR) {
    return PADDED(val / 2) * 2;
  } else {
    return val;
  }
}

/// Returns number of gates, based on RNN function code
///
/// \param[in] func_code NNPA function code, in zdnn_nnpa_function_code enum
///
/// \return number of gates, or 0 if function code is not RNN related
///
short get_func_code_num_gates(nnpa_function_code func_code) {

#define CASE_RTN_GATES(a, b)                                                   \
  case a:                                                                      \
    return get_data_layout_num_gates(b); // piggyback thus no need to hardcode

  switch (func_code) {
    CASE_RTN_GATES(NNPA_LSTMACT, ZDNN_FICO);
    CASE_RTN_GATES(NNPA_GRUACT, ZDNN_ZRH);
  default:
    LOG_WARN("Unknown or not RNN related function code : %d", func_code);
    return 0;
  }
#undef CASE_RTN_GATES
}

/// Returns the matmul function code that should be used given the passed dim4
/// sizes (stacks) for input_a and input_b
///
/// \param[in] input_a_dim4 the size of matmul input_a dim4
/// \param[in] input_b_dim4 the size of matmul input_b dim4
///
/// \return nnpa_function_code representing which matmul op to be used
///
nnpa_function_code get_matmul_function(uint32_t input_a_dim4,
                                       uint32_t input_b_dim4) {
  // NNPA_MATMUL_OP expects the following dims for [dim4, dim3, dim2, dim1]:
  //   Input a - [S, 1, M, N]
  //   Input b - [S, 1, N, P]
  //   Input c - [S, 1, 1, P]
  //
  // NNPA_MATMUL_OP_BCAST1 expects:
  //   Input a - [1, 1, M, N]
  //   Input b - [S, 1, N, P]
  //   Input c - [S, 1, 1, P]
  //
  // NNPA_MATMUL_OP_BCAST23 expects:
  //   Input a - [S, 1, M, N]
  //   Input b - [1, 1, N, P]
  //   Input c - [1, 1, 1, P]
  //
  // This means we can compare dim4 for the inputs 1 and 2 to determing which
  // function code to use.
  //
  // Note that NNPA_MATMUL_OP is used in cases where S == 1.
  if (input_a_dim4 == 1 && input_b_dim4 != 1)
    return NNPA_MATMUL_OP_BCAST1;
  if (input_b_dim4 == 1 && input_a_dim4 != 1)
    return NNPA_MATMUL_OP_BCAST23;

  return NNPA_MATMUL_OP;
}

/// Returns string representation of the layout
///
/// \param[in] layout tensor layout
///
/// \return string representation of the layout, or UNDEFINED_STR if no such
/// layout exists
///
const char *get_data_layout_str(zdnn_data_layouts layout) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return DATA_LAYOUT_STR_##a;

  switch (layout) {
    CASE_RTN_STR(ZDNN_1D);
    CASE_RTN_STR(ZDNN_2D);
    CASE_RTN_STR(ZDNN_2DS);
    CASE_RTN_STR(ZDNN_3D);
    CASE_RTN_STR(ZDNN_3DS);
    CASE_RTN_STR(ZDNN_ZRH);
    CASE_RTN_STR(ZDNN_4D);
    CASE_RTN_STR(ZDNN_4DS);
    CASE_RTN_STR(ZDNN_NHWC);
    CASE_RTN_STR(ZDNN_NCHW);
    CASE_RTN_STR(ZDNN_FICO);
    CASE_RTN_STR(ZDNN_HWCK);
    CASE_RTN_STR(ZDNN_BIDIR_ZRH);
    CASE_RTN_STR(ZDNN_BIDIR_FICO);
  default:
    LOG_WARN("Unknown layout: %d", layout);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the format
///
/// \param[in] layout tensor format
///
/// \return string representation of the format, or UNDEFINED_STR if no such
/// layout exists
///
const char *get_data_format_str(zdnn_data_formats format) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return DATA_FORMAT_STR_##a;

  switch (format) {
    CASE_RTN_STR(ZDNN_FORMAT_4DFEATURE);
    CASE_RTN_STR(ZDNN_FORMAT_4DKERNEL);
    CASE_RTN_STR(ZDNN_FORMAT_4DWEIGHTS);
    CASE_RTN_STR(ZDNN_FORMAT_4DGENERIC);
  default:
    LOG_WARN("Unknown format: %d", format);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns number of bytes of a data type
///
/// \param[in] type data type
///
/// \return size in number of bytes, or 0 if no such data type exists
///
short get_data_type_size(zdnn_data_types type) {

#define CASE_RTN_SIZE(a, b)                                                    \
  case a:                                                                      \
    return b;

  switch (type) {
    CASE_RTN_SIZE(INT8, 1);
    CASE_RTN_SIZE(INT32, 4);
    CASE_RTN_SIZE(BFLOAT, 2);
    CASE_RTN_SIZE(FP16, 2);
    CASE_RTN_SIZE(FP32, 4);
    CASE_RTN_SIZE(ZDNN_DLFLOAT16, 2);
    CASE_RTN_SIZE(ZDNN_BINARY_FP32, 4);
    CASE_RTN_SIZE(ZDNN_BINARY_INT8, 1);
    CASE_RTN_SIZE(ZDNN_BINARY_INT32, 4);
  default:
    LOG_WARN("Unknown data type: %d", type);
    return 0;
  }
#undef CASE_RTN_SIZE
}

/// Returns string representation of the data type
///
/// \param[in] type tensor data type
///
/// \return string representation of the data type, or UNDEFINED_STR if no such
/// data type exists
///
const char *get_data_type_str(zdnn_data_types type) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return DATA_TYPE_STR_##a;

  switch (type) {
    CASE_RTN_STR(ZDNN_BINARY_INT8);
    CASE_RTN_STR(ZDNN_BINARY_INT32);
    CASE_RTN_STR(ZDNN_BINARY_FP32);
    CASE_RTN_STR(INT8);
    CASE_RTN_STR(BFLOAT);
    CASE_RTN_STR(FP16);
    CASE_RTN_STR(FP32);
    CASE_RTN_STR(ZDNN_DLFLOAT16);
  default:
    LOG_WARN("Unknown data type: %d", type);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the RNN direction
///
/// \param[in] dir direction
///
/// \return string representation of the direction, or UNDEFINED_STR if no such
/// direction exists
///
const char *get_rnn_direction_str(lstm_gru_direction dir) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return RNN_DIR_STR_##a;

  switch (dir) {
    CASE_RTN_STR(FWD);
    CASE_RTN_STR(BWD);
    CASE_RTN_STR(BIDIR);
  default:
    LOG_WARN("Unknown direction: %d", dir);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of a FUNCTION_CODE
///
/// \param[in] func function_code
///
/// \return string representation of the function_code, or UNDEFINED_STR if no
/// such function_code exists
///
const char *get_function_code_str(nnpa_function_code func) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return FUNCTION_CODE_STR_##a;

  switch (func) {
    CASE_RTN_STR(NNPA_QAF)
    CASE_RTN_STR(NNPA_ADD)
    CASE_RTN_STR(NNPA_SUB)
    CASE_RTN_STR(NNPA_MUL)
    CASE_RTN_STR(NNPA_DIV)
    CASE_RTN_STR(NNPA_MIN)
    CASE_RTN_STR(NNPA_MAX)
    CASE_RTN_STR(NNPA_LOG)
    CASE_RTN_STR(NNPA_EXP)
    CASE_RTN_STR(NNPA_RELU)
    CASE_RTN_STR(NNPA_TANH)
    CASE_RTN_STR(NNPA_SIGMOID)
    CASE_RTN_STR(NNPA_SOFTMAX)
    CASE_RTN_STR(NNPA_SQRT)
    CASE_RTN_STR(NNPA_INVSQRT)
    CASE_RTN_STR(NNPA_GELU)
    CASE_RTN_STR(NNPA_BATCHNORMALIZATION)
    CASE_RTN_STR(NNPA_MOMENTS)
    CASE_RTN_STR(NNPA_LAYERNORM)
    CASE_RTN_STR(NNPA_NORM)
    CASE_RTN_STR(NNPA_MAXPOOL2D)
    CASE_RTN_STR(NNPA_AVGPOOL2D)
    CASE_RTN_STR(NNPA_LSTMACT)
    CASE_RTN_STR(NNPA_GRUACT)
    CASE_RTN_STR(NNPA_CONVOLUTION)
    CASE_RTN_STR(NNPA_MATMUL_OP)
    CASE_RTN_STR(NNPA_MATMUL_OP_BCAST23)
    CASE_RTN_STR(NNPA_MATMUL_OP_BCAST1)
    CASE_RTN_STR(NNPA_TRANSFORM)
    CASE_RTN_STR(NNPA_REDUCE)
  default:
    LOG_WARN("Unknown function_code: %d", func);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the softmax activation function
///
/// \param[in] func activation function
///
/// \return string representation of the activation function, or UNDEFINED_STR
/// if no such activation function exists
///
const char *get_softmax_act_str(zdnn_softmax_act func) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return SOFTMAX_ACT_STR_##a;

  switch (func) {
    CASE_RTN_STR(SOFTMAX_ACT_NONE);
    CASE_RTN_STR(SOFTMAX_ACT_LOG);
  default:
    LOG_WARN("Unknown activation function: %d", func);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the matmul operation
///
/// \param[in] op operation
///
/// \return string representation of the operation, or UNDEFINED_STR
/// if no such operation exists
///
const char *get_matmul_op_str(zdnn_matmul_ops op) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return MATMUL_OP_STR_##a;

  switch (op) {
    CASE_RTN_STR(MATMUL_OP_ADDITION);
    CASE_RTN_STR(MATMUL_OP_GREATER);
    CASE_RTN_STR(MATMUL_OP_GREATER_EQUAL);
    CASE_RTN_STR(MATMUL_OP_EQUAL);
    CASE_RTN_STR(MATMUL_OP_NOT_EQUAL);
    CASE_RTN_STR(MATMUL_OP_LESSER_EQUAL);
    CASE_RTN_STR(MATMUL_OP_LESSER);
  default:
    LOG_WARN("Unknown operation: %d", op);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the matmul bcast operation
///
/// \param[in] op operation
///
/// \return string representation of the operation, or UNDEFINED_STR
/// if no such operation exists
///
const char *get_matmul_bcast_op_str(zdnn_matmul_bcast_ops op) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return MATMUL_BCAST_OP_STR_##a;

  switch (op) {
    CASE_RTN_STR(MATMUL_BCAST_OP_ADDITION);
    CASE_RTN_STR(MATMUL_BCAST_OP_GREATER);
    CASE_RTN_STR(MATMUL_BCAST_OP_GREATER_EQUAL);
    CASE_RTN_STR(MATMUL_BCAST_OP_EQUAL);
    CASE_RTN_STR(MATMUL_BCAST_OP_NOT_EQUAL);
    CASE_RTN_STR(MATMUL_BCAST_OP_LESSER_EQUAL);
    CASE_RTN_STR(MATMUL_BCAST_OP_LESSER);
  default:
    LOG_WARN("Unknown operation: %d", op);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the pool padding type
///
/// \param[in] pad padding
///
/// \return string representation of the pool padding, or UNDEFINED_STR if no
/// such padding exists
///
const char *get_pool_padding_str(zdnn_pool_padding pad) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return POOL_PADDING_STR_##a;

  switch (pad) {
    CASE_RTN_STR(SAME_PADDING);
    CASE_RTN_STR(VALID_PADDING);
  default:
    LOG_WARN("Unknown pool padding: %d", pad);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the conv2d activation function
///
/// \param[in] func activation function
///
/// \return string representation of the activation function, or UNDEFINED_STR
///         if no such activation function exists
///
const char *get_conv2d_act_str(zdnn_conv2d_act func) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return CONV2D_ACT_STR_##a;

  switch (func) {
    CASE_RTN_STR(CONV2D_ACT_NONE);
    CASE_RTN_STR(CONV2D_ACT_RELU);
  default:
    LOG_WARN("Unknown activation function: %d", func);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the reduce operation
///
/// \param[in] op operation
///
/// \return string representation of the operation, or UNDEFINED_STR
/// if no such operation exists
///
const char *get_reduce_op_str(zdnn_reduce_ops op) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return REDUCE_OP_STR_##a;

  switch (op) {
    CASE_RTN_STR(REDUCE_OP_MINIMUM);
    CASE_RTN_STR(REDUCE_OP_MINIMUM_IDX);
    CASE_RTN_STR(REDUCE_OP_MAXIMUM);
    CASE_RTN_STR(REDUCE_OP_MAXIMUM_IDX);
  default:
    LOG_WARN("Unknown operation: %d", op);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Returns string representation of the bessel correction
///
/// \param[in] bessel correction
///
/// \return string representation of the correction, or UNDEFINED_STR
/// if no such operation exists
///
const char *get_bessel_correction_str(zdnn_moments_bessel correction) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return BESSEL_CORRECTION_STR_##a;

  switch (correction) {
    CASE_RTN_STR(MOMENTS_BESSEL_POPULATION);
    CASE_RTN_STR(MOMENTS_BESSEL_SAMPLE);
  default:
    LOG_WARN("Unknown bessel correction: %d", correction);
    return UNDEFINED_STR;
  }
#undef CASE_RTN_STR
}

/// Retrieve library version number (ZDNN_VERNUM)
///
/// \param[in] None
///
/// \return ZDNN_VERNUM
///
uint32_t zdnn_get_library_version() { return ZDNN_VERNUM; }

/// Retrieve library version string (ZDNN_VERSION)
///
/// \param[in] None
///
/// \return string pointer containing ZDNN_VERSION
///
char *zdnn_get_library_version_str() { return ZDNN_VERSION; }

/// Return the maximum representable value between a transformed and
/// pre-transformed zdnn_data_type
///
/// \param[in] transformed_type restricted values of ZDNN_DLFLOAT16,
/// ZDNN_BINARY_INT8, or ZDNN_BINARY_INT32
/// \param[in] pre_transformed_type restricted values of INT32, INT8, FP32,
/// FP16, or BFLOAT
/// \param[out] limit pointer to max value between transformed_type and
/// pre_transformed_type in data type of pre_transformed_type
///
/// \return a ZDNN_STATUS indicating whether valid types were used
zdnn_status zdnn_get_max_limit(zdnn_data_types transformed_type,
                               zdnn_data_types pre_transformed_type,
                               void *limit) {
  switch (transformed_type) {
  case ZDNN_DLFLOAT16: {
    switch (pre_transformed_type) {
    case FP32: {
      *(float *)limit = DLF16_MAX_AS_FP32;
      break;
    }
    case FP16: {
      *(uint16_t *)limit = FP16_MAX;
      break;
    }
    case BFLOAT: {
      *(uint16_t *)limit = DLF16_MAX_AS_BFLOAT;
      break;
    }
    default:
      return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid pre_transformed_type.",
                         NO_ARG);
    }
    return ZDNN_STATUS_OK;
  }
  case ZDNN_BINARY_INT8: {
    switch (pre_transformed_type) {
    case FP32: {
      *(float *)limit = INT8_MAX_AS_FP32;
      break;
    }
    case FP16: {
      *(uint16_t *)limit = INT8_MAX_AS_FP16;
      break;
    }
    case BFLOAT: {
      *(uint16_t *)limit = INT8_MAX_AS_BFLOAT;
      break;
    }
    case INT8: {
      *(int8_t *)limit = INT8_MAX;
      break;
    }
    default:
      return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid pre_transformed_type.",
                         NO_ARG);
    }
    return ZDNN_STATUS_OK;
  }
  case ZDNN_BINARY_INT32: {
    switch (pre_transformed_type) {
    case INT32: {
      *(int32_t *)limit = INT32_MAX;
      break;
    }
    default:
      return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid pre_transformed_type.",
                         NO_ARG);
    }
    return ZDNN_STATUS_OK;
  }
  default:
    break;
  }
  return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid transformed_type.", NO_ARG);
}

/// Return the minimum representable value between a transformed and
/// pre-transformed zdnn_data_type
///
/// \param[in] transformed_type restricted values of ZDNN_DLFLOAT16,
/// ZDNN_BINARY_INT8, or ZDNN_BINARY_INT32
/// \param[in] pre_transformed_type restricted values of INT32, INT8, FP32,
/// FP16, or BFLOAT
/// \param[out] limit pointer to min value between transformed_type and
/// pre_transformed_type in data type of pre_transformed_type
///
/// \return a ZDNN_STATUS indicating whether valid types were used
zdnn_status zdnn_get_min_limit(zdnn_data_types transformed_type,
                               zdnn_data_types pre_transformed_type,
                               void *limit) {
  switch (transformed_type) {
  case ZDNN_DLFLOAT16: {
    switch (pre_transformed_type) {
    case FP32: {
      *(float *)limit = DLF16_MIN_AS_FP32;
      break;
    }
    case FP16: {
      *(uint16_t *)limit = FP16_MIN;
      break;
    }
    case BFLOAT: {
      *(uint16_t *)limit = DLF16_MIN_AS_BFLOAT;
      break;
    }
    default:
      return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid pre_transformed_type.",
                         NO_ARG);
    }
    return ZDNN_STATUS_OK;
  }
  case ZDNN_BINARY_INT8: {
    switch (pre_transformed_type) {
    case FP32: {
      *(float *)limit = INT8_MIN_AS_FP32;
      break;
    }
    case FP16: {
      *(uint16_t *)limit = INT8_MIN_AS_FP16;
      break;
    }
    case BFLOAT: {
      *(uint16_t *)limit = INT8_MIN_AS_BFLOAT;
      break;
    }
    case INT8: {
      *(int8_t *)limit = INT8_MIN;
      break;
    }
    default:
      return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid pre_transformed_type.",
                         NO_ARG);
    }
    return ZDNN_STATUS_OK;
  }
  case ZDNN_BINARY_INT32: {
    switch (pre_transformed_type) {
    case INT32: {
      *(int32_t *)limit = INT32_MIN;
      break;
    }
    default:
      return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid pre_transformed_type.",
                         NO_ARG);
    }
    return ZDNN_STATUS_OK;
  }
  default:
    break;
  }
  return ZDNN_STATUS(ZDNN_INVALID_TYPE, "Invalid transformed_type.", NO_ARG);
}