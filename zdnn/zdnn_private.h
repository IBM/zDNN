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

#ifndef ZDNN_ZDNN_PRIVATE_H_
#define ZDNN_ZDNN_PRIVATE_H_

#include <stdarg.h>
#include <stdio.h>

// -----------------------------------------------------------------------------
// convert_hw.c includes
// -----------------------------------------------------------------------------
#if defined(__MVS__)    // If z/OS, use XL C include and typedef
#include <builtins.h>   // needed for XL C vector ops
#else                   // If LoZ, use gcc include and typedef
#include <s390intrin.h> // needed for LoZ vector ops
#endif

#include "../config.h"

#define AIU_BYTES_PER_STICK 128
#define AIU_2BYTE_CELLS_PER_STICK 64

#define AIU_2BYTE_CELL_SIZE 2
#define AIU_STICKS_PER_PAGE 32
#define AIU_PAGESIZE_IN_BYTES 4096

#define ZDNN_MAX_DIMS 4 // number of dims in AIU's Tensor Descriptor

typedef enum log_levels {
  LOGLEVEL_OFF,
  LOGLEVEL_FATAL,
  LOGLEVEL_ERROR,
  LOGLEVEL_WARN,
  LOGLEVEL_INFO,
  LOGLEVEL_DEBUG,
  LOGLEVEL_TRACE,
} log_levels;

typedef enum elements_mode {
  ELEMENTS_ALL,
  ELEMENTS_CONCAT_SINGLE,
  ELEMENTS_CONCAT_WO_PAD
} elements_mode;

#define LOGMODULE_SIZE 1024

extern log_levels log_level;
extern bool precheck_enabled;
extern uint32_t status_diag;
extern char log_module[LOGMODULE_SIZE];

#define ENVVAR_LOGLEVEL "ZDNN_LOGLEVEL"
#define ENVVAR_ENABLE_PRECHECK "ZDNN_ENABLE_PRECHECK"
#define ENVVAR_STATUS_DIAG "ZDNN_STATUS_DIAG"
#define ENVVAR_LOGMODULE "ZDNN_LOGMODULE"

#ifdef __MVS__
#define INIT_FUNCTION_ATTRS
#else
/* zDNN needs to be compiled for IBM z14. In order to allow the
   facility check to be reached also when running on older machines
   the CPU level is lowered to IBM z196 for just the init function and
   the facility check itself.  ALL ZDNN FUNCTIONS INVOKED BEFORE THE
   FACILITY CHECK NEED TO BE FLAGGED THAT WAY TO MAKE IT WORK.  */
#define INIT_FUNCTION_ATTRS __attribute__((target("arch=z196")))
#endif

#define STATUS_DIAG_NOT_SET -1

#define DCL_EXTERN_STATUS_STR(a) extern const char *STATUS_STR_##a;
DCL_EXTERN_STATUS_STR(ZDNN_OK)
DCL_EXTERN_STATUS_STR(ZDNN_ELEMENT_RANGE_VIOLATION)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_SHAPE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_LAYOUT)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_TYPE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_FORMAT)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_DIRECTION)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_CONCAT_TYPE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_STRIDE_PADDING)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_STRIDES)
DCL_EXTERN_STATUS_STR(ZDNN_MISALIGNED_PARMBLOCK)
DCL_EXTERN_STATUS_STR(ZDNN_ALLOCATION_FAILURE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_BUFFER)
DCL_EXTERN_STATUS_STR(ZDNN_CONVERT_FAILURE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_STATE)
DCL_EXTERN_STATUS_STR(ZDNN_UNSUPPORTED_AIU_EXCEPTION)

DCL_EXTERN_STATUS_STR(ZDNN_UNSUPPORTED_PARMBLOCK)
DCL_EXTERN_STATUS_STR(ZDNN_UNAVAILABLE_FUNCTION)
DCL_EXTERN_STATUS_STR(ZDNN_UNSUPPORTED_FORMAT)
DCL_EXTERN_STATUS_STR(ZDNN_UNSUPPORTED_TYPE)
DCL_EXTERN_STATUS_STR(ZDNN_EXCEEDS_MDIS)
DCL_EXTERN_STATUS_STR(ZDNN_EXCEEDS_MTS)
DCL_EXTERN_STATUS_STR(ZDNN_MISALIGNED_TENSOR)
DCL_EXTERN_STATUS_STR(ZDNN_MISALIGNED_SAVEAREA)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F000)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F001)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F002)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F003)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F004)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F005)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F006)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F007)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F008)
DCL_EXTERN_STATUS_STR(ZDNN_FUNC_RC_F009)

#undef DCL_EXTERN_STATUS_STR

/*
 * NNPA use of register 0
 */

typedef union nnpa_return {
  uint64_t r0; // for reading from and writing to r0
  struct fields {
    uint16_t rc;      // response code, bits [0-15]
    uint8_t rsvd1;    // reserved, bits [16-23]
    uint8_t ef;       // exception flags, bits [24-31]
    uint8_t rsvd2[3]; // reserved, bits [32-55]
    uint8_t fc;       // function code, bits [56-63]
  } fields;
} nnpa_return;

/*
 * To interface to the zAIU through the NNPA instruction requires the following
 * parameter blocks
 */
typedef
#ifdef __MVS__
    _Packed
#endif
    struct nnpa_tensor_descriptor {
  uint8_t data_layout_format;
  uint8_t data_type;
  uint8_t reserve1[6];
  uint32_t dim4_index_size;
  uint32_t dim3_index_size;
  uint32_t dim2_index_size;
  uint32_t dim1_index_size;
  uint64_t *tensor_data_addr;
}
#ifndef __MVS__
__attribute__((packed))
#endif
nnpa_tensor_descriptor;

// BIG ENDIAN 128-bits bits-field, most significant bit is bit 0
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bit128_t {
  uint64_t bits_0to63;
  uint64_t bits_64to127;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bit128_t;

// BIG ENDIAN 256-bits bits-field, most significant bit is bit 0
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bit256_t {
  uint64_t bits_0to63;
  uint64_t bits_64to127;
  uint64_t bits_128to191;
  uint64_t bits_192to255;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bit256_t;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct nnpa_parameter_block {
  uint16_t parm_block_version_number; // first 9 bits must be 0
  uint8_t model_version_number;       // Only set by hardware for continuation.
  uint8_t reserved_for_ibm[3];
  uint16_t reserved1; // 1 bit Continuation Flag at end
  uint8_t reserved2[48];
  uint64_t function_specific_save_area_address;
  nnpa_tensor_descriptor output_tensor1;
  nnpa_tensor_descriptor output_tensor2;
  uint8_t reserved3[64];
  nnpa_tensor_descriptor input_tensor1;
  nnpa_tensor_descriptor input_tensor2;
  nnpa_tensor_descriptor input_tensor3;
  uint8_t reserved4[96];
  uint32_t function_specific_parm1;
  uint32_t function_specific_parm2;
  uint32_t function_specific_parm3;
  uint32_t function_specific_parm4;
  uint32_t function_specific_parm5;
  uint8_t reserved5[108];
  uint8_t continuation_state_buffer[3584];
}
#ifndef __MVS__
__attribute__((packed, aligned(8)))
#endif
nnpa_parameter_block
#ifdef __MVS__
    __attribute__((__aligned__(8)))
#endif
    ;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct nnpa_qaf_parameter_block {
  bit256_t installed_functions_vector; // bit set of installed operations
  bit128_t
      installed_parameter_block_formats; // bit set of installed block formats
  uint16_t installed_data_types;         // bit set of installed data types
  uint8_t reserved1[2];
  uint32_t installed_data_layout_formats; // bit set of supported data layouts
  uint8_t reserved2[4];
  uint32_t maximum_dimension_index_size; // maximum supported number of elements
                                         // for any single tensor dimension
  uint64_t maximum_tensor_size; // maximum supported tensor size (bytes) aka
                                // stick-area size
  uint16_t installed_dt1_conversions_vector; // bit set of installed Data-Type-1
                                             // conversions
  uint8_t reserved3[182];
}
#ifndef __MVS__
__attribute__((packed, aligned(8)))
#endif
nnpa_qaf_parameter_block
#ifdef __MVS__
    __attribute__((__aligned__(8)))
#endif
    ;

extern nnpa_qaf_parameter_block nnpa_query_result;

// -----------------------------------------------------------------------------
// Versioning
// -----------------------------------------------------------------------------

extern uint32_t aiu_lib_vernum;
void refresh_aiu_lib_vernum();

// -----------------------------------------------------------------------------
// Floating Point Format Conversion Functions
// -----------------------------------------------------------------------------

uint32_t convert_data_format(void *input_data, zdnn_data_types in_data_fmt,
                             void *output_data, zdnn_data_types out_data_fmt,
                             uint32_t num_fields);

uint32_t
convert_data_format_in_stride(void *input_data, zdnn_data_types in_data_fmt,
                              void *output_data, zdnn_data_types out_data_fmt,
                              uint32_t num_fields, uint32_t input_stride);

// -----------------------------------------------------------------------------
// Tensor Functions
// -----------------------------------------------------------------------------
void init_transformed_desc(zdnn_data_layouts layout, zdnn_data_types type,
                           zdnn_data_formats format,
                           zdnn_tensor_desc *tfrmd_desc, uint32_t dim4,
                           uint32_t dim3, uint32_t dim2, uint32_t dim1);

zdnn_status ztensor_slice_dim4(const zdnn_ztensor *input_ztensor,
                               uint32_t slice_idx, size_t slice_buffer_size,
                               zdnn_tensor_desc *output_pre_tfrmd_desc,
                               zdnn_tensor_desc *output_tfrmd_desc,
                               zdnn_ztensor *output_ztensor);

// -----------------------------------------------------------------------------
// NNPA Parm Block Functions
// -----------------------------------------------------------------------------

// Define NNPA_PARM_BLOCK_VERSION
// Notes:
//   - The PBVN is not used on a Query NNPA
//   - The PBVN is architected to be 9-15 of the first word of the
//     NNPA parm block
//   - Actual supported values are returned on the aforementioned
//     Query NNPA in field IPBF
#define NNPA_PARM_BLOCK_VERSION 0

void populate_descriptor(nnpa_tensor_descriptor *descriptor,
                         const zdnn_ztensor *ztensor);
void populate_nnpa_parm_block(
    nnpa_parameter_block *parm_block, const zdnn_ztensor *input_ztensor1,
    const zdnn_ztensor *input_ztensor2, const zdnn_ztensor *input_ztensor3,
    zdnn_ztensor *output_ztensor1, zdnn_ztensor *output_ztensor2,
    void *func_sp_savearea_addr, uint32_t func_sp_parm1, uint32_t func_sp_parm2,
    uint32_t func_sp_parm3, uint32_t func_sp_parm4, uint32_t func_sp_parm5);

// -----------------------------------------------------------------------------
// Malloc 4k
// -----------------------------------------------------------------------------

void *malloc_aligned_4k(size_t size);
void free_aligned_4k(void *aligned_ptr);

// -----------------------------------------------------------------------------
// NNPA Invoke Functions
// -----------------------------------------------------------------------------

zdnn_status invoke_nnpa(uint8_t function_code, char *parm_block,
                        uint8_t *exception_flags);
zdnn_status invoke_nnpa_query(nnpa_qaf_parameter_block *qpb);

// -----------------------------------------------------------------------------
// Internal Function for AIU Operations
// -----------------------------------------------------------------------------

zdnn_status aiu_ops(uint8_t function_code, const zdnn_ztensor *input1,
                    const zdnn_ztensor *input2, const zdnn_ztensor *input3,
                    zdnn_ztensor *output1, zdnn_ztensor *output2);

zdnn_status
aiu_ops_func_specific(uint8_t function_code, const zdnn_ztensor *input1,
                      const zdnn_ztensor *input2, const zdnn_ztensor *input3,
                      zdnn_ztensor *output1, zdnn_ztensor *output2,
                      uint64_t func_sp_savearea_addr, uint32_t func_sp_parm1,
                      uint32_t func_sp_parm2, uint32_t func_sp_parm3,
                      uint32_t func_sp_parm4, uint32_t func_sp_parm5);

zdnn_status aiu_lstm_gru(uint8_t function_code, const zdnn_ztensor *input,
                         const zdnn_ztensor *h0, const zdnn_ztensor *c0,
                         const zdnn_ztensor *weights,
                         const zdnn_ztensor *biases,
                         const zdnn_ztensor *hidden_weights,
                         const zdnn_ztensor *hidden_biases,
                         lstm_gru_direction direction, void *work_area,
                         zdnn_ztensor *hn_output, zdnn_ztensor *cf_output);

// -----------------------------------------------------------------------------
// Internal Tensor Compatibility Verification
// -----------------------------------------------------------------------------

zdnn_status verify_tensors(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c,
                           const zdnn_ztensor *output);
zdnn_status verify_lstm_or_gru_act_tensors(uint8_t function_code,
                                           const zdnn_ztensor *ts_fused,
                                           const zdnn_ztensor *bias_add_rnn_op,
                                           const zdnn_ztensor *prev_state,
                                           const zdnn_ztensor *h_output,
                                           const zdnn_ztensor *c_output);
zdnn_status verify_matmul_op_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const zdnn_ztensor *output);
zdnn_status verify_matmul_bcast_op_tensors(const zdnn_ztensor *input_a,
                                           const zdnn_ztensor *input_b,
                                           const zdnn_ztensor *input_c,
                                           const zdnn_ztensor *output);
zdnn_status verify_batchnorm_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const zdnn_ztensor *output);
zdnn_status verify_pool_avg_max_tensors(
    const zdnn_ztensor *input, zdnn_pool_padding padding_type,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, const zdnn_ztensor *output);

zdnn_status verify_conv2d_tensors(const zdnn_ztensor *input,
                                  const zdnn_ztensor *kernel,
                                  const zdnn_ztensor *bias, uint32_t pad_n_act,
                                  uint32_t stride_height, uint32_t stride_width,
                                  uint32_t reserved_n_clipping,
                                  const zdnn_ztensor *output);

zdnn_status verify_relu_tensors(const zdnn_ztensor *input,
                                uint32_t reserved_n_clipping,
                                const zdnn_ztensor *output);

zdnn_status
verify_pre_transformed_descriptor(const zdnn_tensor_desc *pre_tfrmd_desc);

zdnn_status verify_transformed_descriptor(const zdnn_tensor_desc *tfrmd_desc);

// -----------------------------------------------------------------------------
// Stickify Related Functions
// -----------------------------------------------------------------------------

size_t get_stick_offset(uint32_t e4x, uint32_t e3x, uint32_t e2x, uint32_t e1x,
                        const zdnn_tensor_desc *pre_tfrmd_desc);
void setbit_128(bit128_t *field, uint8_t bit_pos);
bool is_bitset_128(bit128_t field, uint8_t bit_pos);
void setbit_256(bit256_t *field, uint16_t bit_pos);
bool is_bitset_256(bit256_t field, uint16_t bit_pos);

zdnn_status transform_fico_ztensor(zdnn_ztensor *fico_ztensor, void *fx_data,
                                   void *ix_data, void *cx_data, void *ox_data);
zdnn_status transform_zrh_ztensor(zdnn_ztensor *zrh_ztensor, void *zx_data,
                                  void *rx_data, void *hx_data);

// -----------------------------------------------------------------------------
// convert_hw.c wrapper Functions and types
// -----------------------------------------------------------------------------
typedef vector unsigned int vec_float32;
typedef vector unsigned short vec_int16;
typedef vector unsigned char vec_char8;

vec_int16 aiu_vec_round_from_fp32(vec_float32 a, vec_float32 b);
void aiu_vec_lengthen_to_fp32(vec_int16 a, vec_float32 *out1,
                              vec_float32 *out2);
vec_int16 aiu_vec_convert_from_fp16(vec_int16 a);
vec_int16 aiu_vec_convert_to_fp16(vec_int16 a);

// -----------------------------------------------------------------------------
// NNPA-MATMUL-OP function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-23: reserved
//  bits 24-31: operation
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bitfield_func_sp_parm1_matmul_op {
  uint32_t reserved1 : 24;
  uint32_t operation : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bitfield_func_sp_parm1_matmul_op;

typedef union func_sp_parm1_matmul_op {
  // for get/setting bitfield individually
  bitfield_func_sp_parm1_matmul_op bits;
  // for as a whole.  you must clear this before setting bits individual
  uint32_t val;
} func_sp_parm1_matmul_op;

// -----------------------------------------------------------------------------
// NNPA-MATMUL-OP-BCAST23 function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-23: reserved
//  bits 24-31: operation
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bitfield_func_sp_parm1_matmul_bcast_op {
  uint32_t reserved1 : 24;
  uint32_t operation : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bitfield_func_sp_parm1_matmul_bcast_op;

typedef union func_sp_parm1_matmul_bcast_op {
  // for get/setting bitfield individually
  bitfield_func_sp_parm1_matmul_bcast_op bits;
  // for as a whole.  you must clear this before setting bits individual
  uint32_t val;
} func_sp_parm1_matmul_bcast_op;

// -----------------------------------------------------------------------------
// NNPA-SOFTMAX function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-28: reserved
//  bits 28-31: activation func
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bitfield_func_sp_parm1_softmax {
  uint32_t reserved1 : 28;
  uint32_t act : 4;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bitfield_func_sp_parm1_softmax;

typedef union func_sp_parm1_softmax {
  // for get/setting bitfield individually
  bitfield_func_sp_parm1_softmax bits;
  // for as a whole.  you must clear this before setting bits individual
  uint32_t val;
} func_sp_parm1_softmax;

// -----------------------------------------------------------------------------
// NNPA-RELU function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-15: reserved
//  bits 16-31: clipping value
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bitfield_func_sp_parm1_relu {
  uint32_t reserved1 : 16;
  uint32_t clipping_value : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bitfield_func_sp_parm1_relu;

typedef union func_sp_parm1_relu {
  // for get/setting bitfield individually
  bitfield_func_sp_parm1_relu bits;
  // for as a whole.  you must clear this before setting bits individual
  uint32_t val;
} func_sp_parm1_relu;

// -----------------------------------------------------------------------------
// NNPA-CONVOLUTION function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-23: reserved
//  bits 24-27: activation func
//  bits 28: reserved
//  bits 29-31: padding type
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bitfield_func_sp_parm1_conv2d {
  uint32_t reserved1 : 24;
  uint32_t act : 4;
  uint32_t reserved2 : 1;
  uint32_t pad : 3;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bitfield_func_sp_parm1_conv2d;

typedef union func_sp_parm1_conv2d {
  // for get/setting bitfield individually
  bitfield_func_sp_parm1_conv2d bits;
  // for as a whole.  you must clear this before setting bits individual
  uint32_t val;
} func_sp_parm1_conv2d;

// -----------------------------------------------------------------------------
// NNPA-CONVOLUTION function-specific-parameter-4 bitfields
// -----------------------------------------------------------------------------

//  bits 0-15: reserved
//  bits 16-31: clipping value
typedef
#ifdef __MVS__
    _Packed
#endif
    struct bitfield_func_sp_parm4_conv2d {
  uint32_t reserved1 : 16;
  uint32_t clipping_value : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
bitfield_func_sp_parm4_conv2d;

typedef union func_sp_parm4_conv2d {
  // for get/setting bitfield individually
  bitfield_func_sp_parm4_conv2d bits;
  // for as a whole.  you must clear this before setting bits individual
  uint32_t val;
} func_sp_parm4_conv2d;

// -----------------------------------------------------------------------------
// zDNN Logger Functions
// -----------------------------------------------------------------------------

bool logmodule_matches(const char *file_name);

void log_fatal(const char *func_name, const char *file_name, int line_no,
               char *format, ...);
void log_error(const char *func_name, const char *file_name, int line_no,
               char *format, ...);
void log_warn(const char *func_name, const char *file_name, int line_no,
              char *format, ...);
void log_info(const char *func_name, const char *file_name, int line_no,
              char *format, ...);
void log_debug(const char *func_name, const char *file_name, int line_no,
               char *format, ...);
void log_trace(const char *func_name, const char *file_name, int line_no,
               char *format, ...);

void log_message(log_levels lvl, const char *func_name, const char *file_name,
                 int line_no, const char *format, va_list arg);

#ifndef ZDNN_CONFIG_DEBUG

// when ZDNN_CONFIG_DEBUG is off (i.e., production code):
//
// - no logmodule filtering
// - FATAL/ERROR: log the message, no loglevel check
// - WARN/INFO/DEBUG/TRACE: no-op

#define LOG_FATAL(format, ...)                                                 \
  log_fatal(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_ERROR(format, ...)                                                 \
  log_error(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_WARN(format, ...)
#define LOG_INFO(format, ...)
#define LOG_DEBUG(format, ...)
#define LOG_TRACE(format, ...)

#define BEGIN_BLOCK_IF_LOGLEVEL_FATAL
#define BEGIN_BLOCK_IF_LOGLEVEL_ERROR
#define BEGIN_BLOCK_IF_LOGLEVEL_WARN if (0)
#define BEGIN_BLOCK_IF_LOGLEVEL_INFO if (0)
#define BEGIN_BLOCK_IF_LOGLEVEL_DEBUG if (0)
#define BEGIN_BLOCK_IF_LOGLEVEL_TRACE if (0)

#else

// when ZDNN_CONFIG_DEBUG is on
//
// - fully utilize loglevel/logmodule functionalities

#define LOG_FATAL(format, ...)                                                 \
  log_fatal(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_ERROR(format, ...)                                                 \
  log_error(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_WARN(format, ...)                                                  \
  log_warn(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_INFO(format, ...)                                                  \
  log_info(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_DEBUG(format, ...)                                                 \
  log_debug(__func__, __FILE__, __LINE__, format, __VA_ARGS__)
#define LOG_TRACE(format, ...)                                                 \
  log_trace(__func__, __FILE__, __LINE__, format, __VA_ARGS__)

#define BEGIN_IF_LOGLEVEL(lvl, file_name)                                      \
  if ((log_level >= lvl) && logmodule_matches(file_name))

#define BEGIN_BLOCK_IF_LOGLEVEL_FATAL                                          \
  BEGIN_IF_LOGLEVEL(LOGLEVEL_FATAL, __FILE__)

#define BEGIN_BLOCK_IF_LOGLEVEL_ERROR                                          \
  BEGIN_IF_LOGLEVEL(LOGLEVEL_ERROR, __FILE__)

#define BEGIN_BLOCK_IF_LOGLEVEL_WARN BEGIN_IF_LOGLEVEL(LOGLEVEL_WARN, __FILE__)

#define BEGIN_BLOCK_IF_LOGLEVEL_INFO BEGIN_IF_LOGLEVEL(LOGLEVEL_INFO, __FILE__)

#define BEGIN_BLOCK_IF_LOGLEVEL_DEBUG                                          \
  BEGIN_IF_LOGLEVEL(LOGLEVEL_DEBUG, __FILE__)

#define BEGIN_BLOCK_IF_LOGLEVEL_TRACE                                          \
  BEGIN_IF_LOGLEVEL(LOGLEVEL_TRACE, __FILE__)

#endif

// -----------------------------------------------------------------------------
// zDNN Status Related Functions
// -----------------------------------------------------------------------------

zdnn_status set_zdnn_status(zdnn_status status, const char *func_name,
                            const char *file_name, int line_no,
                            const char *format, ...);

#define ZDNN_STATUS(status, format, ...)                                       \
  set_zdnn_status(status, __func__, __FILE__, __LINE__, format, __VA_ARGS__)

#define NO_ARG 0

#define ZDNN_STATUS_NO_MSG(status) ZDNN_STATUS(status, NULL, NO_ARG)

#ifndef ZDNN_CONFIG_DEBUG
#define ZDNN_STATUS_OK ZDNN_OK
#else
#define ZDNN_STATUS_OK ZDNN_STATUS_NO_MSG(ZDNN_OK)
#endif

// -----------------------------------------------------------------------------
// Misc get_*() Functions
// -----------------------------------------------------------------------------

short get_func_code_num_gates(nnpa_function_code func_code);
short get_data_layout_num_gates(zdnn_data_layouts layout);
short get_data_layout_dims(zdnn_data_layouts layout);
const char *get_data_layout_str(zdnn_data_layouts layout);
const char *get_data_format_str(zdnn_data_formats format);
short get_data_type_size(zdnn_data_types type);
const char *get_data_type_str(zdnn_data_types type);
const char *get_rnn_direction_str(lstm_gru_direction dir);
const char *get_softmax_act_str(zdnn_softmax_act func);
const char *get_matmul_op_str(zdnn_matmul_ops op);
const char *get_matmul_bcast_op_str(zdnn_matmul_bcast_ops op);
const char *get_pool_padding_str(zdnn_pool_padding pad);
const char *get_conv2d_act_str(zdnn_conv2d_act func);
uint64_t get_num_elements(const zdnn_ztensor *ztensor, elements_mode mode);

// -----------------------------------------------------------------------------
// Print Utilities
// -----------------------------------------------------------------------------

void print_bits(size_t const size, void const *const ptr);
void print_hex(size_t const size, void const *const ptr);
void print_dlf16_buffer(void *buffer, uint64_t buffer_size);
void print_desc(zdnn_tensor_desc *desc);
void print_ztensor(const zdnn_ztensor *ztensor, char *name, bool print_data);

typedef enum dump_mode { AS_HEX, AS_FLOAT } dump_mode;
void dumpdata_origtensor(const zdnn_tensor_desc *pre_tfrmd_desc,
                         void *tensor_data, dump_mode mode);
void dumpdata_ztensor(const zdnn_ztensor *ztensor, dump_mode mode,
                      bool print_all);

// -----------------------------------------------------------------------------
// Misc Macros
// -----------------------------------------------------------------------------

#define CEIL(a, b) (uint64_t)((a + b - 1) / b) // positive numbers only
#define MIN(a, b) ((a > b) ? b : a)
#define MAX(a, b) ((a < b) ? b : a)
#define BIT_SIZEOF(a) (sizeof(a) * 8)

// -----------------------------------------------------------------------------
// Private global variables
// -----------------------------------------------------------------------------

#endif /* ZDNN_ZDNN_PRIVATE_H_ */
