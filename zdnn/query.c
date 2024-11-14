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
#pragma export(zdnn_get_max_for_dim)
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

/// Query the NNPA maximum supported dimension index size value for a given
/// dimension
///
/// \param[in] dimension dimension to get the maximum index size for.
///
/// \return maximum dimension index size value supported by NNPA for a given
/// dimension
///
uint32_t zdnn_get_max_for_dim(uint8_t dimension) {

  switch (dimension) {
  case 4:
    return nnpa_query_result.max_dim4_index_size
               ? nnpa_query_result.max_dim4_index_size
               : nnpa_query_result.maximum_dimension_index_size;
  case 3:
    return nnpa_query_result.max_dim3_index_size
               ? nnpa_query_result.max_dim3_index_size
               : nnpa_query_result.maximum_dimension_index_size;
  case 2:
    return nnpa_query_result.max_dim2_index_size
               ? nnpa_query_result.max_dim2_index_size
               : nnpa_query_result.maximum_dimension_index_size;
  case 1:
    return nnpa_query_result.max_dim1_index_size
               ? nnpa_query_result.max_dim1_index_size
               : nnpa_query_result.maximum_dimension_index_size;
  default:
    // TODO Printout error
    return 0;
  }
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

  query_status = invoke_nnpa_query(&nnpa_query_result);

  refresh_aiu_lib_vernum();

  return query_status;
}

/// Check if zDNN operation is supported on current hardware
///
/// \param[in] api, zDNN api to query
///
/// \return true if api is supported else false
///
bool is_operation_available(zdnn_operation_apis api) {
  return query_nnpa_op(api);
}
