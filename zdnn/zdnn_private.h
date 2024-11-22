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
#define AIU_1BYTE_CELLS_PER_STICK 128
#define AIU_2BYTE_CELLS_PER_STICK 64
#define AIU_4BYTE_CELLS_PER_STICK 32

#define AIU_2BYTE_CELL_SIZE 2
#define AIU_STICKS_PER_PAGE 32
#define AIU_PAGESIZE_IN_BYTES 4096

#define ZDNN_MAX_DIMS 4 // number of dims in zAIU's Tensor Descriptor

/*
 * The following values are ranges for transformed data types
 *
 * - DLFLOAT Range
 *        |------------- 0 ------------|
 *  -8573157376.0f                8573157376.0f
 *
 * - INT8 Range
 *        |------------- 0 ------------|
 *      -128                          127
 *
 * - INT32 Range
 *        |------------- 0 ------------|
 *  -2147483647                   2147483647
 *
 */
#define DLFLOAT16_MAX 8573157376.0f
#define DLFLOAT16_MIN -8573157376.0f
#ifndef INT8_MAX
#define INT8_MAX 127
#endif
#ifndef INT8_MIN
#define INT8_MIN -128
#endif
#ifndef INT32_MAX
#define INT32_MAX 2147483647
#endif
#ifndef INT32_MIN
#define INT32_MIN -2147483648
#endif

/*
 *
 * The following values are hardcoded limits for pre-transformed data types to
 * transformed data types.
 *
 * Note: Hex values are treated as uint16_t in C. Therefore it should only be
 * used for data types represented as uint16_t, otherwise unintended conversions
 * may occur.
 *
 * - FP16
 *   - Represented in zDNN as uint16_t
 *   - Can be converted to:
 *     - DLFLOAT16
 *       - Contains a smaller range than DLFLOAT16.
 *     - INT8
 *       - Contains a larger range than INT8
 *
 * - BFLOAT
 *   - Represented in zDNN as uint16_t
 *   - Can be converted to:
 *     - DLFLOAT16
 *       - Contains a larger range than DLFLOAT16.
 *     - INT8
 *       - Contains a larger range than INT8
 *
 * - FP32
 *   - Represented in zDNN as float
 *     - DLFLOAT16
 *       - Contains a larger range than DLFLOAT16.
 *     - INT8
 *       - Contains a larger range than INT8
 *
 */
#define FP16_MAX 0x7BFF         // 65504.0f
#define FP16_MIN 0xFBFF         // -65504.0f
#define INT8_MAX_AS_FP16 0x57F0 // 127.0f
#define INT8_MIN_AS_FP16 0xD800 // -128.0f

#define DLF16_MAX_AS_BFLOAT 0x4FFF // 8573157376.0f
#define DLF16_MIN_AS_BFLOAT 0xCFFF // -8573157376.0f
#define INT8_MAX_AS_BFLOAT 0x42FE  // 127.0f
#define INT8_MIN_AS_BFLOAT 0xC300  // -128.0f

#define DLF16_MAX_AS_FP32 8573157376.0f  // 0x4FFF80000 represented as float
#define DLF16_MIN_AS_FP32 -8573157376.0f // 0xCFFF80000 represented as float
#define INT8_MAX_AS_FP32 127.0f          // 0x42FE0000 represented as float
#define INT8_MIN_AS_FP32 -128.0f         // 0xC3000000 represented as float

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
  ELEMENTS_AIU,
  ELEMENTS_PRE,
  ELEMENTS_PRE_SINGLE_GATE = ELEMENTS_PRE,
  ELEMENTS_PRE_ALL_GATES
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

#define STATUS_DIAG_NOT_SET -1

#define DCL_EXTERN_STATUS_STR(a) extern const char *STATUS_STR_##a;
DCL_EXTERN_STATUS_STR(ZDNN_OK)
DCL_EXTERN_STATUS_STR(ZDNN_ELEMENT_RANGE_VIOLATION)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_SHAPE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_LAYOUT)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_TYPE)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_FORMAT)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_DIRECTION)
DCL_EXTERN_STATUS_STR(ZDNN_INVALID_CONCAT_INFO)
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
    struct function_specific_parameters {
  uint32_t function_specific_parm1;
  uint32_t function_specific_parm2;
  uint32_t function_specific_parm3;
  uint32_t function_specific_parm4;
  uint32_t function_specific_parm5;
  uint32_t function_specific_parm6;
  uint32_t function_specific_parm7;
  uint32_t function_specific_parm8;
  uint32_t function_specific_parm9;
  uint32_t function_specific_parm10;
  uint32_t function_specific_parm11;
  uint32_t function_specific_parm12;
  uint32_t function_specific_parm13;
  uint32_t function_specific_parm14;
  uint32_t function_specific_parm15;
  uint32_t function_specific_parm16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
function_specific_parameters;

// Macro for checking C-struct size
#define CASSERT(test) CASSERT_impl(test, __LINE__)
#define CASSERT_NAME(line) cassert_line_##line
#define CASSERT_impl(test, line)                                               \
  typedef char CASSERT_NAME(line)[2 * (!!(test)) - 1]

// Standard Parm Block sizes
#define NNPA_PARMBLOCK_SIZE 4096
#define QAF_PARMBLOCK_SIZE 256

typedef
#ifdef __MVS__
    _Packed
#endif
    struct nnpa_parameter_block {
  uint16_t parm_block_version_number; // first 9 bits must be 0
  uint8_t model_version_number;       // Only set by hardware for continuation.
  uint8_t reserved_for_ibm1;
  uint32_t reserved_for_ibm2 : 16;
  uint32_t reserved1 : 14;
  uint32_t lf : 1; // prioritized-latency flag
  uint32_t cf : 1; // continuation flag
  uint32_t reserved2;
  uint32_t reserved_for_ibm3;
  uint32_t reserved3;
  uint32_t reserved_for_ibm4;
  uint32_t reserved4;
  uint32_t reserved_for_ibm5;
  uint8_t reserved5[24];
  uint64_t function_specific_save_area_address;
  nnpa_tensor_descriptor output_tensor1;
  nnpa_tensor_descriptor output_tensor2;
  uint8_t reserved6[64];
  nnpa_tensor_descriptor input_tensor1;
  nnpa_tensor_descriptor input_tensor2;
  nnpa_tensor_descriptor input_tensor3;
  uint8_t reserved7[96];
  function_specific_parameters function_specific_parms;
  uint8_t reserved8[64];
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
  uint8_t reserved3[14];
  uint32_t max_dim4_index_size; // maximum dimensions-4 index size
  uint32_t max_dim3_index_size; // maximum dimensions-3 index size
  uint32_t max_dim2_index_size; // maximum dimensions-2 index size
  uint32_t max_dim1_index_size; // maximum dimensions-1 index size
  uint8_t reserved4[152];
}
#ifndef __MVS__
__attribute__((packed, aligned(8)))
#endif
nnpa_qaf_parameter_block
#ifdef __MVS__
    __attribute__((__aligned__(8)))
#endif
    ;

// compile-time size check of QAF and NNPA parameter block
CASSERT(sizeof(nnpa_parameter_block) == NNPA_PARMBLOCK_SIZE);
CASSERT(sizeof(nnpa_qaf_parameter_block) == QAF_PARMBLOCK_SIZE);

extern nnpa_qaf_parameter_block nnpa_query_result;

// -----------------------------------------------------------------------------
// Versioning
// -----------------------------------------------------------------------------

extern uint32_t aiu_lib_vernum;
void refresh_aiu_lib_vernum();

// -----------------------------------------------------------------------------
// Floating Point Format Conversion Functions
// -----------------------------------------------------------------------------

typedef vector unsigned int vec_float32;

uint32_t convert_data_format(void *input_data, zdnn_data_types in_data_fmt,
                             void *output_data, zdnn_data_types out_data_fmt,
                             uint32_t num_fields,
                             void (*skip_func)(const vec_float32 *,
                                               const vec_float32 *,
                                               vec_float32 *, vec_float32 *));

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
void populate_descriptor(nnpa_tensor_descriptor *descriptor,
                         const zdnn_ztensor *ztensor);
void populate_nnpa_parm_block(
    nnpa_parameter_block *parm_block, uint16_t parm_block_version,
    const zdnn_ztensor *input_ztensor1, const zdnn_ztensor *input_ztensor2,
    const zdnn_ztensor *input_ztensor3, zdnn_ztensor *output_ztensor1,
    zdnn_ztensor *output_ztensor2, void *func_sp_savearea_addr,
    const function_specific_parameters *fsp);

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
// Internal Function for zAIU Operations
// -----------------------------------------------------------------------------

zdnn_status aiu_ops(uint16_t op_parm_block_version, uint8_t function_code,
                    const zdnn_ztensor *input1, const zdnn_ztensor *input2,
                    const zdnn_ztensor *input3, zdnn_ztensor *output1,
                    zdnn_ztensor *output2);

zdnn_status
aiu_ops_func_specific(uint16_t op_parm_block_version, uint8_t function_code,
                      const zdnn_ztensor *input1, const zdnn_ztensor *input2,
                      const zdnn_ztensor *input3, zdnn_ztensor *output1,
                      zdnn_ztensor *output2, uint64_t func_sp_savearea_addr,
                      function_specific_parameters *fsp);

zdnn_status aiu_lstm_gru(uint16_t op_parm_block_version, uint8_t function_code,
                         const zdnn_ztensor *input, const zdnn_ztensor *h0,
                         const zdnn_ztensor *c0, const zdnn_ztensor *weights,
                         const zdnn_ztensor *biases,
                         const zdnn_ztensor *hidden_weights,
                         const zdnn_ztensor *hidden_biases,
                         lstm_gru_direction direction, void *work_area,
                         zdnn_ztensor *hn_output, zdnn_ztensor *cf_output);

zdnn_status
aiu_quantized_matmul(uint16_t op_parm_block_version,
                     const uint8_t function_code, const zdnn_ztensor *input_a,
                     const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
                     zdnn_matmul_ops op_type, const int8_t clip_min,
                     const int8_t clip_max, void *work_area,
                     zdnn_ztensor *output, const bool dequantize,
                     const bool disable_clipping, const bool pre_computed);

bool is_query_parmblock_installed(uint8_t parmblock_version);
bool is_nnpa_fc_and_parmblock_installed(uint8_t function_code,
                                        uint8_t parmblock_version);
// -----------------------------------------------------------------------------
// Internal Tensor Compatibility Verification
// -----------------------------------------------------------------------------

zdnn_status verify_tensors(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c,
                           const zdnn_ztensor *output);
zdnn_status verify_zdnn_lstm_or_gru_tensors(
    uint8_t function_code, const zdnn_ztensor *input, const zdnn_ztensor *h0,
    const zdnn_ztensor *c0, const zdnn_ztensor *weights,
    const zdnn_ztensor *biases, const zdnn_ztensor *hidden_weights,
    const zdnn_ztensor *hidden_biases, lstm_gru_direction direction,
    const zdnn_ztensor *hn_output, const zdnn_ztensor *cf_output);
zdnn_status verify_lstm_or_gru_act_tensors(uint8_t function_code,
                                           const zdnn_ztensor *ts_fused,
                                           const zdnn_ztensor *bias_add_rnn_op,
                                           const zdnn_ztensor *prev_state,
                                           const zdnn_ztensor *h_output,
                                           const zdnn_ztensor *c_output);
zdnn_status verify_matmul_op_common(
    uint8_t function_code, const zdnn_ztensor *input_a,
    const zdnn_ztensor *input_b, const zdnn_ztensor *input_c,
    const void *transpose_control, const void *a_scale, const void *a_offset,
    const void *clip_min, const void *clip_max, const zdnn_ztensor *output);
zdnn_status verify_batchnorm_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const zdnn_ztensor *output);
zdnn_status verify_norm_tensors(const zdnn_ztensor *input_a,
                                const zdnn_ztensor *input_b,
                                const zdnn_ztensor *output);
zdnn_status
verify_pool_avg_max_tensors(const zdnn_ztensor *input, const void *padding_type,
                            const void *stride_width, const void *stride_height,
                            const void *kernel_width, const void *kernel_height,
                            const zdnn_ztensor *output);

zdnn_status
verify_conv2d_tensors(const zdnn_ztensor *input, const zdnn_ztensor *kernel,
                      const zdnn_ztensor *bias, const void *pad_n_act,
                      const void *stride_width, const void *stride_height,
                      const void *reserved_n_clipping,
                      const zdnn_ztensor *output);

zdnn_status verify_moments_tensors(const zdnn_ztensor *input_a,
                                   const void *bessel_correction_type,
                                   zdnn_ztensor *output_a,
                                   zdnn_ztensor *output_b);

zdnn_status verify_layernorm_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const void *beta, const void *gamma,
                                     const void *epsilon,
                                     const zdnn_ztensor *output);

zdnn_status verify_relu_tensors(const zdnn_ztensor *input,
                                const void *reserved_n_clipping,
                                const void *reserved_n_adjustment,
                                const zdnn_ztensor *output);

zdnn_status verify_invsqrt_tensors(const zdnn_ztensor *input,
                                   const void *reserved_n_epsilon,
                                   const zdnn_ztensor *output);

zdnn_status verify_transform_tensors(const zdnn_ztensor *input,
                                     const zdnn_ztensor *output,
                                     const void *toc, const void *min_clipping,
                                     const void *max_clipping);

zdnn_status verify_reduce_tensors(const zdnn_ztensor *input,
                                  const zdnn_ztensor *output);

zdnn_status verify_descriptors_transform_ztensor(const zdnn_ztensor *input);

zdnn_status verify_descriptors_transform_origtensor(const zdnn_ztensor *input);

zdnn_status verify_transformed_descriptor(const zdnn_tensor_desc *tfrmd_desc);

zdnn_status verify_transformed_dimensions(const zdnn_tensor_desc *tfrmd_desc);

// -----------------------------------------------------------------------------
// Stickify Related Functions
// -----------------------------------------------------------------------------

size_t get_stick_offset(uint32_t e4x, uint32_t e3x, uint32_t e2x, uint32_t e1x,
                        const zdnn_tensor_desc *pre_tfrmd_desc);
bool is_bitset_128(bit128_t field, uint8_t bit_pos);
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

#if defined(__MVS__) || (defined(__ARCH__) && __ARCH__ < 14)
#define VEC_ROUND_FROM_FP32(FP_HI, FP_LO)                                      \
  aiu_vec_round_from_fp32((vec_float32)(FP_HI), (vec_float32)(FP_LO));
#define VEC_LENGTHEN_TO_FP32(IN, OUT_HI, OUT_LO)                               \
  aiu_vec_lengthen_to_fp32((IN), (vec_float32 *)&(OUT_HI),                     \
                           (vec_float32 *)&(OUT_LO));
#else
#define VEC_ROUND_FROM_FP32(FP_HI, FP_LO)                                      \
  (vec_int16) vec_round_from_fp32((FP_HI), (FP_LO), 0);
/* These compiler intrinsics changed between GCC 13 and 14 from using
   vector short to vector unsigned short.  */
#if __GNUC__ <= 13
#define VEC_LENGTHEN_TO_FP32(IN, OUT_HI, OUT_LO)                               \
  (OUT_HI) = vec_extend_to_fp32_hi((vector short)(IN), 0);                     \
  (OUT_LO) = vec_extend_to_fp32_lo((vector short)(IN), 0);
#else
#define VEC_LENGTHEN_TO_FP32(IN, OUT_HI, OUT_LO)                               \
  (OUT_HI) = vec_extend_to_fp32_hi((IN), 0);                                   \
  (OUT_LO) = vec_extend_to_fp32_lo((IN), 0);
#endif
#endif

// -----------------------------------------------------------------------------
// NNPA-MATMUL-OP function-specific-parameters and their bitfields
// -----------------------------------------------------------------------------
typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bits 24-31: operation
    struct func_sp_parm1_matmul {
  uint32_t reserved1 : 24;
  uint32_t operation : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-29: reserved
    //  bit 30: transpose_b
    //  bit 31: transpose_a
    struct func_sp_parm2_matmul {
  uint32_t reserved1 : 30;
  bool transpose_b : 1;
  bool transpose_a : 1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bit 16-31: rec_scale
    struct func_sp_parm3_matmul {
  uint32_t reserved1 : 16;
  uint32_t rec_scale : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm3_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bit 16-31: a_offset
    struct func_sp_parm4_matmul {
  uint32_t reserved1 : 16;
  uint32_t offset : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm4_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bit 16-31: rec_scale
    struct func_sp_parm5_matmul {
  uint32_t reserved1 : 16;
  uint32_t rec_scale : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm5_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: reserved
    struct func_sp_parm6_matmul {
  uint32_t reserved1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm6_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bit 16-31: rec_scale
    struct func_sp_parm7_matmul {
  uint32_t reserved1 : 16;
  uint32_t rec_scale : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm7_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: reserved
    struct func_sp_parm8_matmul {
  uint32_t reserved1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm8_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bit 24-31: clip_min
    struct func_sp_parm9_matmul {
  uint32_t reserved1 : 24;
  uint32_t clip_min : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm9_matmul;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bit 24-31: clip_max
    struct func_sp_parm10_matmul {
  uint32_t reserved1 : 24;
  uint32_t clip_max : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm10_matmul;

// -----------------------------------------------------------------------------
// NNPA-MATMUL-OP-BCAST23 function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------
typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bits 24-31: operation
    struct func_sp_parm1_matmul_bcast {
  uint32_t reserved1 : 24;
  uint32_t operation : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_matmul_bcast;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_matmul_bcast {
  func_sp_parm1_matmul_bcast parm1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_matmul_bcast;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_matmul {
  func_sp_parm1_matmul parm1;
  func_sp_parm2_matmul parm2;
  func_sp_parm3_matmul parm3;
  func_sp_parm4_matmul parm4;
  func_sp_parm5_matmul parm5;
  func_sp_parm6_matmul parm6;
  func_sp_parm7_matmul parm7;
  func_sp_parm8_matmul parm8;
  func_sp_parm9_matmul parm9;
  func_sp_parm10_matmul parm10;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_matmul;

// -----------------------------------------------------------------------------
// NNPA-SOFTMAX function-specific-parameters and their bitfields
// -----------------------------------------------------------------------------

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-28: reserved
    //  bits 28-31: activation func
    struct func_sp_parm1_softmax {
  uint32_t reserved1 : 28;
  uint32_t act : 4;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_softmax;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: mask
    struct func_sp_parm2_softmax {
  uint32_t mask;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_softmax;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_softmax {
  func_sp_parm1_softmax parm1;
  func_sp_parm2_softmax parm2;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_softmax;

// -----------------------------------------------------------------------------
// NNPA-RELU function-specific-parameters and their bitfields
// -----------------------------------------------------------------------------

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bits 16-31: clipping value
    struct func_sp_parm1_relu {
  uint32_t reserved1 : 16;
  uint32_t clipping_value : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_relu;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bits 16-31: adjustment factor
    struct func_sp_parm2_relu {
  uint32_t reserved1 : 16;
  uint32_t adjustment_factor : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_relu;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_relu {
  func_sp_parm1_relu parm1;
  func_sp_parm2_relu parm2;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_relu;

// -----------------------------------------------------------------------------
// NNPA-CONVOLUTION function-specific-parameters and their bitfields
// -----------------------------------------------------------------------------

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bits 24-27: activation func
    //  bits 28: reserved
    //  bits 29-31: padding type
    struct func_sp_parm1_conv2d {
  uint32_t reserved1 : 24;
  uint32_t act : 4;
  uint32_t reserved2 : 1;
  uint32_t pad : 3;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_conv2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: stride_width
    struct func_sp_parm2_conv2d {
  uint32_t stride_width;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_conv2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: stride_height
    struct func_sp_parm3_conv2d {
  uint32_t stride_height;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm3_conv2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bits 16-31: clipping value
    struct func_sp_parm4_conv2d {
  uint32_t reserved1 : 16;
  uint32_t clipping_value : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm4_conv2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_conv2d {
  func_sp_parm1_conv2d parm1;
  func_sp_parm2_conv2d parm2;
  func_sp_parm3_conv2d parm3;
  func_sp_parm4_conv2d parm4;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_conv2d;

// -----------------------------------------------------------------------------
// NNPA-TRANSFORM function-specific-parameters and their bitfields
// -----------------------------------------------------------------------------
typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0: saturation control
    //  bits 1-23: reserved
    //  bits 24-31: transformation-operation code (TOC)
    struct func_sp_parm1_transform {
  uint32_t sc : 1;
  uint32_t reserved1 : 23;
  uint32_t toc : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_transform;

typedef enum nnpa_transform_operation_code {
  NNPA_TOC_STICK_DLFLOAT = 2,
  NNPA_TOC_STICK_INT8 = 6,
  NNPA_TOC_UNSTICK_DLFLOAT = 129
} nnpa_transform_operation_code;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bits 16-31: rec_scale
    struct func_sp_parm2_transform {
  uint32_t reserved1 : 16;
  uint32_t rec_scale : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_transform;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-15: reserved
    //  bits 16-31: offset
    struct func_sp_parm3_transform {
  uint32_t reserved1 : 16;
  uint32_t offset : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm3_transform;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bits 24-31: clip_min
    struct func_sp_parm4_transform {
  uint32_t reserved1 : 24;
  uint32_t clip_min : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm4_transform;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-23: reserved
    //  bits 24-31: clip_max
    struct func_sp_parm5_transform {
  uint32_t reserved1 : 24;
  uint32_t clip_max : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm5_transform;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_transform {
  func_sp_parm1_transform parm1;
  func_sp_parm2_transform parm2;
  func_sp_parm3_transform parm3;
  func_sp_parm4_transform parm4;
  func_sp_parm5_transform parm5;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_transform;

// -----------------------------------------------------------------------------
// NNPA-INVSQRT function-specific-parameter and bitfields
// -----------------------------------------------------------------------------

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parm1_invsqrt {
  //  bits 0-15: reserved
  //  bits 16-31: epsilon
  uint32_t reserved1 : 16;
  uint32_t epsilon : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_invsqrt;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_invsqrt {
  func_sp_parm1_invsqrt parm1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_invsqrt;

// -----------------------------------------------------------------------------
// NNPA-MOMENTS function-specific-parameter and bitfields
// -----------------------------------------------------------------------------

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parm1_moments {
  uint32_t bessel_correction;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_moments;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_moments {
  func_sp_parm1_moments parm1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_moments;

// -----------------------------------------------------------------------------
// NNPA-LAYERNORM function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-15: reserved
//  bits 16-31: beta
typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parm1_layernorm {
  uint32_t reserved1 : 16;
  uint32_t beta : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_layernorm;

// -----------------------------------------------------------------------------
// NNPA-LAYERNORM function-specific-parameter-2 bitfields
// -----------------------------------------------------------------------------

//  bits 0-15: reserved
//  bits 16-31: gamma
typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parm2_layernorm {
  uint32_t reserved1 : 16;
  uint32_t gamma : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_layernorm;

// -----------------------------------------------------------------------------
// NNPA-LAYERNORM function-specific-parameter-3 bitfields
// -----------------------------------------------------------------------------

//  bits 0-15: reserved
//  bits 16-31: epsilon value
typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parm3_layernorm {
  uint32_t reserved1 : 16;
  uint32_t epsilon : 16;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm3_layernorm;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_layernorm {
  func_sp_parm1_layernorm parm1;
  func_sp_parm2_layernorm parm2;
  func_sp_parm3_layernorm parm3;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_layernorm;

// -----------------------------------------------------------------------------
// NNPA-REDUCE-OP function-specific-parameter-1 bitfields
// -----------------------------------------------------------------------------

//  bits 0-23: reserved
//  bits 24-31: operation
typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parm1_reduce {
  uint32_t reserved1 : 24;
  uint32_t operation : 8;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_reduce;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_reduce {
  func_sp_parm1_reduce parm1;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_reduce;

// -----------------------------------------------------------------------------
// NNPA-AVGPOOL2D and NNPA-MAXPOOL2D function-specific-parameters and their
// bitfields
// -----------------------------------------------------------------------------

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-28: reserved
    //  bits 29-31: padding type
    struct func_sp_parm1_pool2d {
  uint32_t reserved1 : 29;
  uint32_t pad : 3;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm1_pool2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: stride_width aka dim2_stride
    struct func_sp_parm2_pool2d {
  uint32_t stride_width;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm2_pool2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: stride_height aka dim3_stride
    struct func_sp_parm3_pool2d {
  uint32_t stride_height;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm3_pool2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: kernel_width aka dim2_window
    struct func_sp_parm4_pool2d {
  uint32_t kernel_width;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm4_pool2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    //  bits 0-31: kernel_height aka dim3_window
    struct func_sp_parm5_pool2d {
  uint32_t kernel_height;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parm5_pool2d;

typedef
#ifdef __MVS__
    _Packed
#endif
    struct func_sp_parms_pool2d {
  func_sp_parm1_pool2d parm1;
  func_sp_parm2_pool2d parm2;
  func_sp_parm3_pool2d parm3;
  func_sp_parm4_pool2d parm4;
  func_sp_parm5_pool2d parm5;
}
#ifndef __MVS__
__attribute__((packed))
#endif
func_sp_parms_pool2d;

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

#define WARNING_STATUS_BITMASK 0xFFFF0000

// -----------------------------------------------------------------------------
// Misc get_*() Functions
// -----------------------------------------------------------------------------

short get_func_code_num_gates(nnpa_function_code func_code);
short get_data_layout_num_gates(zdnn_data_layouts layout);
short get_data_layout_dims(zdnn_data_layouts layout);
nnpa_function_code get_matmul_function(uint32_t input_a_dim4,
                                       uint32_t input_b_dim4);
const char *get_data_layout_str(zdnn_data_layouts layout);
const char *get_data_format_str(zdnn_data_formats format);
short get_data_type_size(zdnn_data_types type);
const char *get_data_type_str(zdnn_data_types type);
const char *get_rnn_direction_str(lstm_gru_direction dir);
const char *get_function_code_str(nnpa_function_code func);
const char *get_softmax_act_str(zdnn_softmax_act func);
const char *get_matmul_op_str(zdnn_matmul_ops op);
const char *get_matmul_bcast_op_str(zdnn_matmul_bcast_ops op);
const char *get_pool_padding_str(zdnn_pool_padding pad);
const char *get_conv2d_act_str(zdnn_conv2d_act func);
const char *get_reduce_op_str(zdnn_reduce_ops op);
const char *get_bessel_correction_str(zdnn_moments_bessel correction);
uint64_t get_num_elements(const zdnn_ztensor *ztensor, elements_mode mode);

uint32_t get_rnn_concatenated_dim1(uint32_t val, zdnn_concat_info info);
uint32_t get_rnn_concatenated_dim2(uint32_t val, zdnn_concat_info info);

typedef enum zdnn_operation_apis {
  ZDNN_ADD,
  ZDNN_SUB,
  ZDNN_MUL,
  ZDNN_DIV,
  ZDNN_MIN,
  ZDNN_MAX,
  ZDNN_LOG,
  ZDNN_EXP,
  ZDNN_SQRT,
  ZDNN_INVSQRT,
  ZDNN_RELU,
  ZDNN_LEAKY_RELU,
  ZDNN_TANH,
  ZDNN_SIGMOID,
  ZDNN_SOFTMAX,
  ZDNN_SOFTMAX_MASK,
  ZDNN_GELU,
  ZDNN_LSTM,
  ZDNN_GRU,
  ZDNN_MATMUL_OP,
  ZDNN_BATCHNORM,
  ZDNN_NORM,
  ZDNN_MEANREDUCE2D,
  ZDNN_MOMENTS,
  ZDNN_LAYERNORM,
  ZDNN_REDUCE,
  ZDNN_AVGPOOL2D,
  ZDNN_MAXPOOL2D,
  ZDNN_CONV2D,
  ZDNN_TRANSFORM_ZTENSOR,
  ZDNN_TRANSFORM_ZTENSOR_WITH_SATURATION,
  ZDNN_TRANSFORM_QUANTIZED_ZTENSOR,
  ZDNN_TRANSFORM_ORIGTENSOR,
  ZDNN_RESHAPE_ZTENSOR
} zdnn_operation_apis;

// any zdnn op that invokes nnpa
bool query_nnpa_op(zdnn_operation_apis api);
bool is_operation_available(zdnn_operation_apis api);

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

// padded = next multiple of AIU_2BYTE_CELLS_PER_STICK
#define PADDED(x)                                                              \
  ((uint32_t)CEIL(x, AIU_2BYTE_CELLS_PER_STICK) * AIU_2BYTE_CELLS_PER_STICK)

#if !defined(vec_float) || __ARCH__ < 13
#undef vec_float
#define vec_float(X)                                                           \
  ({                                                                           \
    __vector float out;                                                        \
    /* vcefb\t%[out],%[in],0,0 */                                              \
    __asm__(".insn vrr,0xe700000020c3,%[out],%[in],0,2,0,0"                    \
            : [out] "=v"(out)                                                  \
            : [in] "v"(X));                                                    \
    out;                                                                       \
  })
#endif

#if defined(__GNUC__) && __GNUC__ <= 7
#undef vec_round
#define vec_round(X)                                                           \
  ({                                                                           \
    __vector float out;                                                        \
    /* vfisb %[out],%[in],4,4 */                                               \
    __asm__(".insn vrr,0xe700000020c7,%[out],%[in],0,2,4,4"                    \
            : [out] "=v"(out)                                                  \
            : [in] "v"(X));                                                    \
    out;                                                                       \
  })
#endif

// -----------------------------------------------------------------------------
// Private global variables
// -----------------------------------------------------------------------------

#define PRINT_DIMS(x)                                                          \
  printf(#x " pre: %u %u %u %u\n", (x)->pre_transformed_desc->dim4,            \
         (x)->pre_transformed_desc->dim3, (x)->pre_transformed_desc->dim2,     \
         (x)->pre_transformed_desc->dim1);                                     \
  printf(#x ": %u %u %u %u\n", (x)->transformed_desc->dim4,                    \
         (x)->transformed_desc->dim3, (x)->transformed_desc->dim2,             \
         (x)->transformed_desc->dim1);

#endif /* ZDNN_ZDNN_PRIVATE_H_ */
