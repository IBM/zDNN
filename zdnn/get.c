
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

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_get_library_version_str)
#pragma export(zdnn_get_library_version)
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

#define DECLARE_DATA_TYPE_STR(a) static const char *DATA_TYPE_STR_##a = #a;

// const char *DATA_TYPE_STR_X
DECLARE_DATA_TYPE_STR(BFLOAT)
DECLARE_DATA_TYPE_STR(FP16)
DECLARE_DATA_TYPE_STR(FP32)
DECLARE_DATA_TYPE_STR(ZDNN_DLFLOAT16)

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

#define DECLARE_POOL_PADDING_STR(a)                                            \
  static const char *POOL_PADDING_STR_##a = #a;

// const char *POOL_PADDING_STR_X
DECLARE_POOL_PADDING_STR(SAME_PADDING)
DECLARE_POOL_PADDING_STR(VALID_PADDING)

#define DECLARE_CONV2D_ACT_STR(a) static const char *CONV2D_ACT_STR_##a = #a;

// const char *CONV2D_ACT_STR_X
DECLARE_CONV2D_ACT_STR(CONV2D_ACT_NONE)
DECLARE_CONV2D_ACT_STR(CONV2D_ACT_RELU)

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
    CASE_RTN_SIZE(BFLOAT, 2);
    CASE_RTN_SIZE(FP16, 2);
    CASE_RTN_SIZE(FP32, 4);
    CASE_RTN_SIZE(ZDNN_DLFLOAT16, 2);
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
