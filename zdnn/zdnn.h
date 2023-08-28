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

#ifndef ZDNN_ZDNN_H_
#define ZDNN_ZDNN_H_

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// NOTE:
// Ensure that symbols in zdnn.h and zdnn.map are in sync!
// Please also have a look at zdnn.map how to add, update or remove a symbol.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Initializer and global variables
// -----------------------------------------------------------------------------

void zdnn_init();

// -----------------------------------------------------------------------------
// zDNN Status
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// NOTE:
// Update status.c and zdnn_private.h after any status modification!
// -----------------------------------------------------------------------------

// Status categories
#define ZDNN_WARNING 0x00020000
#define ZDNN_PARAMETER_ERROR 0x00040000
#define ZDNN_DATA_ERROR 0x00100000
#define ZDNN_HW_ERROR 0x000c0000

// clang-format off
typedef enum zdnn_status {
  // ----------------------------------------------------------------
  ZDNN_OK = 0x00000000, // Success.
  // ----------------------------------------------------------------
  ZDNN_ELEMENT_RANGE_VIOLATION = ZDNN_WARNING + 0x0001, // AIU operation resulted in data that was out of the normal range.
  // ----------------------------------------------------------------
  ZDNN_INVALID_SHAPE = ZDNN_PARAMETER_ERROR + 0x0001, // Invalid shape information in one (or more) of the input/output tensor(s).
  ZDNN_INVALID_LAYOUT,                                // Invalid layout information in one (or more) of the input/output tensor(s).
  ZDNN_INVALID_TYPE,                                  // Invalid type information in one (or more) of the input/output tensor(s).
  ZDNN_INVALID_FORMAT,                                // Invalid format information in one (or more) of the input/output tensor(s).
  ZDNN_INVALID_DIRECTION,                             // Invalid RNN direction.
  ZDNN_INVALID_CONCAT_INFO,                           // Invalid concatenation info.
  ZDNN_INVALID_STRIDE_PADDING,                        // Invalid padding type parameter for current strides
  ZDNN_INVALID_STRIDES,                               // Invalid stride height or width parameter.
  ZDNN_MISALIGNED_PARMBLOCK,                          // NNPA parameter block is not on double word boundary.
  ZDNN_INVALID_CLIPPING_VALUE,                        // Invalid clipping for the specified operation.
  // ----------------------------------------------------------------
  ZDNN_ALLOCATION_FAILURE = ZDNN_DATA_ERROR + 0x0001, // Can not allocate storage.
  ZDNN_INVALID_BUFFER,                                // Buffer address is NULL or not on 4K-byte boundary, or insufficient buffer size.
  ZDNN_CONVERT_FAILURE,                               // Floating point data conversion failure.
  ZDNN_INVALID_STATE,                                 // Invalid zTensor state.
  ZDNN_UNSUPPORTED_AIU_EXCEPTION,                     // AIU operation returned an unexpected exception.
  // ----------------------------------------------------------------
  ZDNN_UNSUPPORTED_PARMBLOCK = ZDNN_HW_ERROR + 0x0001, // NNPA parameter block format is not supported by the model.
  ZDNN_UNAVAILABLE_FUNCTION,                           // Specified NNPA function is not defined or installed on the machine.
  ZDNN_UNSUPPORTED_FORMAT = ZDNN_HW_ERROR + 0x0010,    // Specified tensor data layout format is not supported.
  ZDNN_UNSUPPORTED_TYPE,                               // Specified tensor data type is not supported.
  ZDNN_EXCEEDS_MDIS,                                   // Tensor dimension exceeds maximum dimension index size (MDIS).
  ZDNN_EXCEEDS_MTS,                                    // Total number of elements in tensor exceeds maximum tensor size. (MTS).
  ZDNN_MISALIGNED_TENSOR,                              // Tensor address is not on 4K-byte boundary.
  ZDNN_MISALIGNED_SAVEAREA,                            // Function specific save area address is not on 4K-byte boundary.
  // ----------------------------------------------------------------
  // Function specific response code (F00x)
  ZDNN_FUNC_RC_F000 = ZDNN_HW_ERROR + 0xF000,  // Function specific response code (F000).
  ZDNN_FUNC_RC_F001,                           // Function specific response code (F001).
  ZDNN_FUNC_RC_F002,                           // Function specific response code (F002).
  ZDNN_FUNC_RC_F003,                           // Function specific response code (F003).
  ZDNN_FUNC_RC_F004,                           // Function specific response code (F004).
  ZDNN_FUNC_RC_F005,                           // Function specific response code (F005).
  ZDNN_FUNC_RC_F006,                           // Function specific response code (F006).
  ZDNN_FUNC_RC_F007,                           // Function specific response code (F007).
  ZDNN_FUNC_RC_F008,                           // Function specific response code (F008).
  ZDNN_FUNC_RC_F009,                           // Function specific response code (F009).

  // ----------------------------------------------------------------
} zdnn_status;
// clang-format on

// -----------------------------------------------------------------------------
// NNPA hardware defined values as described in
// z/Architecture - Principles of Operation
// -----------------------------------------------------------------------------

typedef enum nnpa_function_code {
  NNPA_QAF = 0,
  NNPA_ADD = 16,
  NNPA_SUB = 17,
  NNPA_MUL = 18,
  NNPA_DIV = 19,
  NNPA_MIN = 20,
  NNPA_MAX = 21,
  NNPA_LOG = 32,
  NNPA_EXP = 33,
  // reserved = 48
  NNPA_RELU = 49,
  NNPA_TANH = 50,
  NNPA_SIGMOID = 51,
  NNPA_SOFTMAX = 52,
  NNPA_BATCHNORMALIZATION = 64,
  NNPA_MAXPOOL2D = 80,
  NNPA_AVGPOOL2D = 81,
  NNPA_LSTMACT = 96,
  NNPA_GRUACT = 97,
  NNPA_CONVOLUTION = 112,
  NNPA_MATMUL_OP = 113,
  NNPA_MATMUL_OP_BCAST23 = 114
} nnpa_function_code;

typedef enum nnpa_parmblk_format {
  NNPA_PARMBLKFORMAT_0 = 0,
} nnpa_parmblk_format;

typedef enum nnpa_data_type { NNPA_DATATYPE_1 = 0 } nnpa_data_type;

typedef enum nnpa_layout_format {
  NNPA_LAYOUTFMT_4DFEATURE = 0,
  NNPA_LAYOUTFMT_4DKERNEL = 1
} nnpa_layout_format;

typedef enum nnpa_bfp_format {
  // 0 is reversed
  NNPA_BFPFMT_TINY = 1,
  NNPA_BFPFMT_SHORT = 2,
} nnpa_bfp_format;

// NNPA_SOFTMAX work area sizes required by the NNPA hardware
#define ZDNN_SOFTMAX_SAVEAREA_SIZE 8 * 1024

// NNPA Hardware defined values for Function Specific Parameters
typedef enum nnpa_matmul_operations {
  NNPA_MATMUL_OP_ADDITION = 0,
  NNPA_MATMUL_OP_COMP_HIGH = 1,
  NNPA_MATMUL_OP_COMP_NOT_LOW = 2,
  NNPA_MATMUL_OP_COMP_EQUAL = 3,
  NNPA_MATMUL_OP_COMP_NOT_EQUAL = 4,
  NNPA_MATMUL_OP_COMP_NOT_HIGH = 5,
  NNPA_MATMUL_OP_COMP_LOW = 6,
} nnpa_matmul_operations;

typedef enum nnpa_matmul_bcast_operations {
  NNPA_MATMUL_BCAST_OP_ADDITION = 0
} nnpa_matmul_bcast_operations;

typedef enum nnpa_softmax_act {
  NNPA_SOFTMAX_NONE = 0,
  NNPA_SOFTMAX_LOG = 1
} nnpa_softmax_act;

// -----------------------------------------------------------------------------
// zdnn_query_*() bit-field enums
// -----------------------------------------------------------------------------

// pos is counting from left to right
#define MSB_BITMASK(field_size, pos) 1u << ((field_size - 1) - pos)

typedef enum zdnn_query_datatypes {
  QUERY_DATATYPE_INTERNAL1 = MSB_BITMASK(16, NNPA_DATATYPE_1)
} zdnn_query_datatypes;

typedef enum zdnn_query_layoutfmts {
  QUERY_LAYOUTFMT_4DFEATURE = MSB_BITMASK(32, NNPA_LAYOUTFMT_4DFEATURE),
  QUERY_LAYOUTFMT_4DKERNEL = MSB_BITMASK(32, NNPA_LAYOUTFMT_4DKERNEL)
} zdnn_query_layoutfmts;

typedef enum zdnn_query_bfpfmts {
  QUERY_BFPFMT_TINY = MSB_BITMASK(16, NNPA_BFPFMT_TINY),
  QUERY_BFPFMT_SHORT = MSB_BITMASK(16, NNPA_BFPFMT_SHORT)
} zdnn_query_bfpfmts;

// -----------------------------------------------------------------------------
// ZDNN enums
// -----------------------------------------------------------------------------

typedef enum zdnn_data_types {
  ZDNN_DLFLOAT16 = NNPA_DATATYPE_1, // 16-bit deep learning format
  BFLOAT = 253,                     // Brain floating point format
  FP16 = 254,                       // 16-bit IEEE-754 floating point format
  FP32 = 255,                       // 32-bit IEEE-754 floating point format
} zdnn_data_types;

typedef enum zdnn_data_layouts {
  ZDNN_1D,        // 1d tensor
  ZDNN_2D,        // 2d tensor
  ZDNN_2DS,       // represents special 2D tensors required by LSTM/GRU
  ZDNN_3D,        // 3d tensor
  ZDNN_3DS,       // represents special 3D tensors required by
                  // LSTM/GRU/Softmax/Matmul
  ZDNN_ZRH,       // represents (update, reset, hidden) used by GRU
  ZDNN_4D,        // 4d tensor
  ZDNN_4DS,       // represents special 4D tensors required by LSTM/GRU output
  ZDNN_NHWC,      // 4d feature tensor in NHWC
  ZDNN_NCHW,      // 4d feature tensor in NCHW
  ZDNN_FICO,      // represents (forget, input, cell, output) used by LSTM
  ZDNN_HWCK,      // 4d kernel CNN tensor
  ZDNN_BIDIR_ZRH, // ZRH variant to work with bidirectional LSTM/GRU output
  ZDNN_BIDIR_FICO // FICO variant to work with bidirectional LSTM/GRU output
} zdnn_data_layouts;

typedef enum zdnn_data_formats {
  ZDNN_FORMAT_4DFEATURE =
      NNPA_LAYOUTFMT_4DFEATURE, // tensor in AIU data layout format 0
  ZDNN_FORMAT_4DKERNEL =
      NNPA_LAYOUTFMT_4DKERNEL, // tensor in AIU data layout format 1
} zdnn_data_formats;

// Supported padding types for use in pooling functions
typedef enum zdnn_pool_padding {
  VALID_PADDING = 0,
  SAME_PADDING = 1
} zdnn_pool_padding;

// Support operations for use in matmul functions
typedef enum zdnn_matmul_ops {
  MATMUL_OP_ADDITION = NNPA_MATMUL_OP_ADDITION,
  MATMUL_OP_GREATER = NNPA_MATMUL_OP_COMP_HIGH,
  MATMUL_OP_GREATER_EQUAL = NNPA_MATMUL_OP_COMP_NOT_LOW,
  MATMUL_OP_EQUAL = NNPA_MATMUL_OP_COMP_EQUAL,
  MATMUL_OP_NOT_EQUAL = NNPA_MATMUL_OP_COMP_NOT_EQUAL,
  MATMUL_OP_LESSER_EQUAL = NNPA_MATMUL_OP_COMP_NOT_HIGH,
  MATMUL_OP_LESSER = NNPA_MATMUL_OP_COMP_LOW
} zdnn_matmul_ops;

// Support operations for use in matmul function
typedef enum zdnn_matmul_bcast_ops {
  MATMUL_BCAST_OP_ADDITION = NNPA_MATMUL_BCAST_OP_ADDITION
} zdnn_matmul_bcast_ops;

typedef enum zdnn_softmax_act {
  SOFTMAX_ACT_NONE = NNPA_SOFTMAX_NONE,
  SOFTMAX_ACT_LOG = NNPA_SOFTMAX_LOG
} zdnn_softmax_act;

typedef enum zdnn_conv2d_act {
  CONV2D_ACT_NONE,
  CONV2D_ACT_RELU
} zdnn_conv2d_act;

// -----------------------------------------------------------------------------
// Structs
// ----------------------------------------------------------------------------

// describes general pre-transformed or transformed information (e.g. shape) of
// a tensor
typedef struct zdnn_tensor_desc {
  zdnn_data_layouts layout; // data layout
  zdnn_data_formats format; // internal use only
  zdnn_data_types type;     // data type
  uint32_t dim4;            // number of elements in outermost dimension
  uint32_t dim3;            // ... outer dimension
  uint32_t dim2;            // ... inner dimension
  uint32_t dim1;            // number of elements in innermost dimension
} zdnn_tensor_desc;

// struct for describing a ztensor
typedef struct zdnn_ztensor {
  zdnn_tensor_desc
      *pre_transformed_desc; // tensor's shape information before transformation
  zdnn_tensor_desc *transformed_desc; // transformed tensor's shape information
  uint64_t buffer_size;               // tensor size in bytes
  void *buffer;                       // pointer to the tensor in memory
  bool is_transformed; // indicator if data in buffer has been transformed
  char reserved[31];   // not currently used, should contain zeros.
} zdnn_ztensor;

#define ZDNN_VERSION "1.0.2"
#define ZDNN_VERNUM 0x010002 // 0x[major][minor][patch]
#define ZDNN_VER_MAJOR 1
#define ZDNN_VER_MINOR 0
#define ZDNN_VER_PATCH 2

// -----------------------------------------------------------------------------
// External Tensor Functions
// -----------------------------------------------------------------------------

// Concatenation information is encoded into a 32-bit word:
// [RNN_TYPE: 8][PREV_LAYER_TYPE: 8][USAGE: 8][8]

typedef uint32_t zdnn_concat_info;

#define BITSHIFT_RNN_TYPE 24
#define BITSHIFT_PREV_LAYER 16
#define BITSHIFT_USAGE 8

#define RNN_TYPE_LSTM (0 << BITSHIFT_RNN_TYPE)
#define RNN_TYPE_GRU (1 << BITSHIFT_RNN_TYPE)

#define PREV_LAYER_UNI (0 << BITSHIFT_PREV_LAYER)
#define PREV_LAYER_NONE PREV_LAYER_UNI
#define PREV_LAYER_BIDIR (1 << BITSHIFT_PREV_LAYER)

#define USAGE_WEIGHTS (0 << BITSHIFT_USAGE)
#define USAGE_HIDDEN_WEIGHTS (1 << BITSHIFT_USAGE)
#define USAGE_BIASES (2 << BITSHIFT_USAGE)
#define USAGE_HIDDEN_BIASES (3 << BITSHIFT_USAGE)

#define CONCAT_RNN_TYPE(info) (info & (0xFFu << BITSHIFT_RNN_TYPE))
#define CONCAT_PREV_LAYER(info) (info & (0xFFu << BITSHIFT_PREV_LAYER))
#define CONCAT_USAGE(info) (info & (0xFFu << BITSHIFT_USAGE))

void zdnn_init_pre_transformed_desc(zdnn_data_layouts layout,
                                    zdnn_data_types type,
                                    zdnn_tensor_desc *pre_tfrmd_desc, ...);

zdnn_status
zdnn_generate_transformed_desc(const zdnn_tensor_desc *pre_tfrmd_desc,
                               zdnn_tensor_desc *tfrmd_desc);

zdnn_status zdnn_generate_transformed_desc_concatenated(
    const zdnn_tensor_desc *pre_tfrmd_desc, zdnn_concat_info info,
    zdnn_tensor_desc *tfrmd_desc);

zdnn_status zdnn_allochelper_ztensor(zdnn_ztensor *ztensor);
zdnn_status zdnn_free_ztensor_buffer(const zdnn_ztensor *ztensor);

void zdnn_init_ztensor(zdnn_tensor_desc *pre_tfrmd_desc,
                       zdnn_tensor_desc *tfrmd_desc, zdnn_ztensor *output);

zdnn_status zdnn_init_ztensor_with_malloc(zdnn_tensor_desc *pre_tfrmd_desc,
                                          zdnn_tensor_desc *tfrmd_desc,
                                          zdnn_ztensor *output);

void zdnn_reset_ztensor(zdnn_ztensor *ztensor);

uint64_t zdnn_getsize_ztensor(const zdnn_tensor_desc *tfrmd_desc);

// -----------------------------------------------------------------------------
// External Query Functions
// -----------------------------------------------------------------------------

bool zdnn_is_nnpa_installed();
bool zdnn_is_nnpa_function_installed(int count, ...);
bool zdnn_is_nnpa_parmblk_fmt_installed(int count, ...);
bool zdnn_is_nnpa_datatype_installed(uint16_t types_bitmask);
bool zdnn_is_nnpa_layout_fmt_installed(uint32_t layout_bitmask);
bool zdnn_is_nnpa_conversion_installed(nnpa_data_type type,
                                       uint16_t format_bitmask);

uint32_t zdnn_get_nnpa_max_dim_idx_size();
uint64_t zdnn_get_nnpa_max_tensor_size();

zdnn_status zdnn_refresh_nnpa_query_result();

// -----------------------------------------------------------------------------
// Versioning Functions
// -----------------------------------------------------------------------------

bool zdnn_is_version_runnable(uint32_t ver_num);
uint32_t zdnn_get_max_runnable_version();

// -----------------------------------------------------------------------------
// External Elementwise Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_add(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
zdnn_status zdnn_sub(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
zdnn_status zdnn_mul(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
zdnn_status zdnn_div(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
zdnn_status zdnn_min(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
zdnn_status zdnn_max(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);

zdnn_status zdnn_log(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_exp(const zdnn_ztensor *input, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// External Activation Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_relu(const zdnn_ztensor *input, const void *clipping_value,
                      zdnn_ztensor *output);
zdnn_status zdnn_tanh(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_sigmoid(const zdnn_ztensor *input, zdnn_ztensor *output);
zdnn_status zdnn_softmax(const zdnn_ztensor *input, void *save_area,
                         zdnn_softmax_act act_func, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// Recurrent Neural Network (RNN) Operations
// -----------------------------------------------------------------------------

typedef enum lstm_gru_direction { FWD, BWD, BIDIR } lstm_gru_direction;

zdnn_status zdnn_lstm(const zdnn_ztensor *input, const zdnn_ztensor *h0,
                      const zdnn_ztensor *c0, const zdnn_ztensor *weights,
                      const zdnn_ztensor *biases,
                      const zdnn_ztensor *hidden_weights,
                      const zdnn_ztensor *hidden_biases,
                      lstm_gru_direction direction, void *work_area,
                      zdnn_ztensor *hn_output, zdnn_ztensor *cf_output);
zdnn_status zdnn_gru(const zdnn_ztensor *input, const zdnn_ztensor *h0,
                     const zdnn_ztensor *weights, const zdnn_ztensor *biases,
                     const zdnn_ztensor *hidden_weights,
                     const zdnn_ztensor *hidden_biases,
                     lstm_gru_direction direction, void *work_area,
                     zdnn_ztensor *hn_output);

// -----------------------------------------------------------------------------
// Matrix Multiplication Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_matmul_op(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c, zdnn_matmul_ops op_type,
                           zdnn_ztensor *output);
zdnn_status zdnn_matmul_bcast_op(const zdnn_ztensor *input_a,
                                 const zdnn_ztensor *input_b,
                                 const zdnn_ztensor *input_c,
                                 zdnn_matmul_bcast_ops op_type,
                                 zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// External Norm Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_batchnorm(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c, zdnn_ztensor *output);

zdnn_status zdnn_meanreduce2d(const zdnn_ztensor *input, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// External Pool Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_avgpool2d(const zdnn_ztensor *input,
                           zdnn_pool_padding padding_type,
                           uint32_t kernel_height, uint32_t kernel_width,
                           uint32_t stride_height, uint32_t stride_width,
                           zdnn_ztensor *output);

zdnn_status zdnn_maxpool2d(const zdnn_ztensor *input,
                           zdnn_pool_padding padding_type,
                           uint32_t kernel_height, uint32_t kernel_width,
                           uint32_t stride_height, uint32_t stride_width,
                           zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// External Convolution Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_conv2d(const zdnn_ztensor *input, const zdnn_ztensor *kernel,
                        const zdnn_ztensor *bias,
                        zdnn_pool_padding padding_type, uint32_t stride_height,
                        uint32_t stride_width, zdnn_conv2d_act act_func,
                        const void *clipping_value, zdnn_ztensor *output);

// -----------------------------------------------------------------------------
// External Tensor Transform Operations
// -----------------------------------------------------------------------------

zdnn_status zdnn_transform_ztensor(zdnn_ztensor *ztensor, ...);

zdnn_status zdnn_transform_origtensor(const zdnn_ztensor *ztensor,
                                      void *out_buf);

zdnn_status zdnn_reshape_ztensor(const zdnn_ztensor *src, zdnn_ztensor *dest);

// -----------------------------------------------------------------------------
// External Version Related Functions
// -----------------------------------------------------------------------------

char *zdnn_get_library_version_str();
uint32_t zdnn_get_library_version();

// -----------------------------------------------------------------------------
// zDNN Status Related Functions
// -----------------------------------------------------------------------------

const char *zdnn_get_status_message(zdnn_status status);

#endif /* ZDNN_ZDNN_H_ */
