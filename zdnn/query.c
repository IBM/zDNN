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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__MVS__)
#include <cvt.h>
#include <ihaecvt.h>
#include <ihafacl.h>
#include <ihapsa.h>
#endif

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_is_nnpa_function_installed)
#pragma export(zdnn_is_nnpa_parmblk_fmt_installed)
#pragma export(zdnn_is_nnpa_datatype_installed)
#pragma export(zdnn_is_nnpa_layout_fmt_installed)
#pragma export(zdnn_get_nnpa_max_dim_idx_size)
#pragma export(zdnn_get_nnpa_max_tensor_size)
#pragma export(zdnn_refresh_nnpa_query_result)
#pragma export(zdnn_is_nnpa_conversion_installed)
#endif

// Cached copy of the NNPA-QAF result.  zdnn_refresh_nnpa_query_result() is
// responsible for setting and modifying this.  For performance reasons, all
// query functions that involve NNPA-QAF result read from this cached copy
nnpa_qaf_parameter_block nnpa_query_result;

/// Query if NNPA functions are installed
///
/// \param[in] count, number of NNPA functions to query
/// \param[in] ... (additional arguments), function numbers defined in
///                nnpa_function_code enum
///
/// \return true if all queried functions are installed, false if any is not
///
bool zdnn_is_nnpa_function_installed(int count, ...) {
  va_list ap;
  va_start(ap, count);
  bool result = true;

  uint16_t max_func = BIT_SIZEOF(nnpa_query_result.installed_functions_vector);

  for (uint16_t i = 0; i < count; i++) {
    uint16_t func_num = va_arg(ap, int);
    if (func_num >= max_func || // protect ourselves from out-of-range input
        !is_bitset_256(nnpa_query_result.installed_functions_vector,
                       func_num)) {
      result = false;
      break;
    }
  }
  va_end(ap);
  return result;
}

/// Query if NNPA parameter block formats are installed
///
/// \param[in] count, number of NNPA parameter block formats to query
/// \param[in] ... (additional arguments), NNPA parameter block formats defined
///                in nnpa_parmblk_format enum
///
/// \return true if all queried formats are installed, false if any is not
///
bool zdnn_is_nnpa_parmblk_fmt_installed(int count, ...) {
  va_list ap;
  va_start(ap, count);
  bool result = true;

  uint8_t max_format =
      BIT_SIZEOF(nnpa_query_result.installed_parameter_block_formats);

  for (uint8_t i = 0; i < count; i++) {
    uint8_t func_num = va_arg(ap, int);
    if (func_num >= max_format || // protect ourselves from out-of-range input
        !is_bitset_128(nnpa_query_result.installed_parameter_block_formats,
                       func_num)) {
      result = false;
      break;
    }
  }
  va_end(ap);
  return result;
}

/// Query if NNPA data types are installed
///
/// \param[in] types_bitmask OR'd type numbers as defined in
/// zdnn_query_datatypes
///                     enum
///
/// \return true if all queried data types are installed, false if any is not
///
bool zdnn_is_nnpa_datatype_installed(uint16_t types_bitmask) {
  return (~nnpa_query_result.installed_data_types & types_bitmask) == 0;
}

/// Query if NNPA data layout formats are installed
///
/// \param[in] layout_bitmask OR'd layout numbers as defined in
///                        zdnn_query_layout_fmts enum
///
/// \return true if all queried data layout formats are installed, false if any
///          is not
///
bool zdnn_is_nnpa_layout_fmt_installed(uint32_t layout_bitmask) {
  return (~nnpa_query_result.installed_data_layout_formats & layout_bitmask) ==
         0;
}

/// Query if NNPA data type to/from BFP format conversions are installed
///
/// \param[in] type NNPA data type as defined in nnpa_data_type enum
/// \param[in] format_bitmask OR'd BFP format numbers as defined in
///                           zdnn_query_bfpfmts enum
///
/// \return true if all queried format conversions are installed, false if any
///         is not
///
bool zdnn_is_nnpa_conversion_installed(nnpa_data_type type,
                                       uint16_t format_bitmask) {

  switch (type) {
  case NNPA_DATATYPE_1:
    return (~nnpa_query_result.installed_dt1_conversions_vector &
            format_bitmask) == 0;
  default:
    // unknown nnp data-type means "not installed" regardless of mask
    return false;
  }
}

/// Query the NNPA maximum supported dimension index size value
///
/// \param[in] None
///
/// \return maximum dimension index size value supported by NNPA
///
uint32_t zdnn_get_nnpa_max_dim_idx_size() {
  return nnpa_query_result.maximum_dimension_index_size;
}

/// Query the NNPA maximum supported tensor size (in bytes)
///
/// \param[in] None
///
/// \return maximum tensor size value supported by NNPA
///
uint64_t zdnn_get_nnpa_max_tensor_size() {
  return nnpa_query_result.maximum_tensor_size;
}

/// Refresh the nnpa_query_result struct from zAIU
///
/// \param[in] result pointer to aiu_parameter_block_nnpa_qaf struct
///
/// \return ZDNN_OK
///         ZDNN_UNAVAILABLE_FUNCTION
///
zdnn_status zdnn_refresh_nnpa_query_result() {

  zdnn_status query_status;

#ifndef ZDNN_CONFIG_NO_NNPA
  query_status = invoke_nnpa_query(&nnpa_query_result);
#else
  query_status = ZDNN_STATUS_OK;

#define MAXIMUM_DIMENSION_INDEX_SIZE ((uint32_t)1 << 15) // 32768
#define MAXIMUM_TENSOR_SIZE ((uint64_t)1 << 32)          // 4294967296

  setbit_128(&nnpa_query_result.installed_parameter_block_formats,
             NNPA_PARMBLKFORMAT_0);

  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_QAF);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_ADD);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_SUB);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_MUL);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_DIV);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_MIN);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_MAX);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_LOG);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_EXP);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_RELU);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_TANH);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_SIGMOID);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_SOFTMAX);
  setbit_256(&nnpa_query_result.installed_functions_vector,
             NNPA_BATCHNORMALIZATION);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_MAXPOOL2D);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_AVGPOOL2D);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_LSTMACT);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_GRUACT);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_CONVOLUTION);
  setbit_256(&nnpa_query_result.installed_functions_vector, NNPA_MATMUL_OP);
  setbit_256(&nnpa_query_result.installed_functions_vector,
             NNPA_MATMUL_OP_BCAST23);

  nnpa_query_result.installed_data_types |= QUERY_DATATYPE_INTERNAL1;
  nnpa_query_result.installed_data_layout_formats |=
      (QUERY_LAYOUTFMT_4DFEATURE | QUERY_LAYOUTFMT_4DKERNEL);
  nnpa_query_result.installed_dt1_conversions_vector |=
      (QUERY_BFPFMT_TINY | QUERY_BFPFMT_SHORT);
  nnpa_query_result.maximum_dimension_index_size = MAXIMUM_DIMENSION_INDEX_SIZE;
  nnpa_query_result.maximum_tensor_size = MAXIMUM_TENSOR_SIZE;
#endif

  refresh_aiu_lib_vernum();

  return query_status;
}
