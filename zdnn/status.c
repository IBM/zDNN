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
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __MVS__
#include <execinfo.h>
#include <unistd.h>
#else
#include <ctest.h>
#endif

#ifdef __MVS__
#pragma export(zdnn_get_status_message)
#endif

/*
this macro declares 2 strings:
1) STATUS_STR_XXX, which is the stringification of the status code itself
2) STATUS_MSG_XXX, which is the default status message of the status code
   in "ZDNN_XXX: message" form
*/
#define DECLARE_STATUS_STR_N_MSG(a, b)                                         \
  const char *STATUS_STR_##a = #a;                                             \
  const char *STATUS_MSG_##a = #a ": " b;

DECLARE_STATUS_STR_N_MSG(ZDNN_OK, "Success.")
DECLARE_STATUS_STR_N_MSG(ZDNN_ELEMENT_RANGE_VIOLATION,
                         "One or more output tensor values were not valid.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_INVALID_SHAPE,
    "Invalid shape in one (or more) of the input/output tensor(s).")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_INVALID_LAYOUT,
    "Invalid layout in one (or more) of the input/output tensor(s).")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_INVALID_TYPE,
    "Invalid type in one (or more) of the input/output tensor(s).")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_INVALID_FORMAT,
    "Invalid format in one (or more) of the input/output tensor(s).")
DECLARE_STATUS_STR_N_MSG(ZDNN_INVALID_DIRECTION, "Invalid RNN direction.")
DECLARE_STATUS_STR_N_MSG(ZDNN_INVALID_CONCAT_TYPE,
                         "Invalid concatenation type.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_INVALID_STRIDE_PADDING,
    "Padding type is not valid for the current stride inputs.")
DECLARE_STATUS_STR_N_MSG(ZDNN_INVALID_STRIDES,
                         "Invalid stride height or width.")
DECLARE_STATUS_STR_N_MSG(ZDNN_MISALIGNED_PARMBLOCK,
                         "NNPA parameter block is not on doubleword boundary.")
DECLARE_STATUS_STR_N_MSG(ZDNN_INVALID_CLIPPING_VALUE,
                         "Invalid clipping for the specified operation.")
DECLARE_STATUS_STR_N_MSG(ZDNN_ALLOCATION_FAILURE, "Can not allocate storage.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_INVALID_BUFFER,
    "Buffer address is NULL or not on 4K-byte boundary, or "
    "insufficient buffer size.")
DECLARE_STATUS_STR_N_MSG(ZDNN_CONVERT_FAILURE,
                         "Floating point data conversion failure.")
DECLARE_STATUS_STR_N_MSG(ZDNN_INVALID_STATE, "Invalid zTensor state.")
DECLARE_STATUS_STR_N_MSG(ZDNN_UNSUPPORTED_AIU_EXCEPTION,
                         "AIU operation returned an unexpected exception.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_UNSUPPORTED_PARMBLOCK,
    "NNPA parameter block format is not supported by the model.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_UNAVAILABLE_FUNCTION,
    "Specified NNPA function is not defined or installed on the machine.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_UNSUPPORTED_FORMAT,
    "Specified tensor data layout format is not supported.")
DECLARE_STATUS_STR_N_MSG(ZDNN_UNSUPPORTED_TYPE,
                         "Specified tensor data type is not supported.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_EXCEEDS_MDIS,
    "Tensor dimension exceeds maximum dimension index size (MDIS).")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_EXCEEDS_MTS,
    "Total number of elements in tensor exceeds maximum tensor size (MTS).")
DECLARE_STATUS_STR_N_MSG(ZDNN_MISALIGNED_TENSOR,
                         "Tensor address is not on 4K-byte boundary.")
DECLARE_STATUS_STR_N_MSG(
    ZDNN_MISALIGNED_SAVEAREA,
    "Function specific save area address is not on 4K-byte boundary.")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F000,
                         "Function specific response code (F000).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F001,
                         "Function specific response code (F001).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F002,
                         "Function specific response code (F002).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F003,
                         "Function specific response code (F003).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F004,
                         "Function specific response code (F004).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F005,
                         "Function specific response code (F005).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F006,
                         "Function specific response code (F006).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F007,
                         "Function specific response code (F007).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F008,
                         "Function specific response code (F008).")
DECLARE_STATUS_STR_N_MSG(ZDNN_FUNC_RC_F009,
                         "Function specific response code (F009).")

const char *STATUS_MSG_UNKNOWN_STATUS = "(Status string is not defined.)";
const char *STATUS_STR_UNKNOWN_STATUS = "(?)";

/// Retrieve default status message of the status code
///
/// \param[in] status status code
///
///
/// \return pointer to the default status message
///
const char *zdnn_get_status_message(zdnn_status status) {

#define CASE_RTN_MSG(a)                                                        \
  case a:                                                                      \
    return STATUS_MSG_##a;

  switch (status) {
    CASE_RTN_MSG(ZDNN_OK);
    CASE_RTN_MSG(ZDNN_ELEMENT_RANGE_VIOLATION);
    CASE_RTN_MSG(ZDNN_INVALID_SHAPE);
    CASE_RTN_MSG(ZDNN_INVALID_LAYOUT);
    CASE_RTN_MSG(ZDNN_INVALID_TYPE);
    CASE_RTN_MSG(ZDNN_INVALID_FORMAT);
    CASE_RTN_MSG(ZDNN_INVALID_DIRECTION);
    CASE_RTN_MSG(ZDNN_INVALID_CONCAT_TYPE);
    CASE_RTN_MSG(ZDNN_INVALID_STRIDE_PADDING);
    CASE_RTN_MSG(ZDNN_INVALID_STRIDES);
    CASE_RTN_MSG(ZDNN_MISALIGNED_PARMBLOCK);
    CASE_RTN_MSG(ZDNN_INVALID_CLIPPING_VALUE);
    CASE_RTN_MSG(ZDNN_ALLOCATION_FAILURE);
    CASE_RTN_MSG(ZDNN_INVALID_BUFFER);
    CASE_RTN_MSG(ZDNN_CONVERT_FAILURE);
    CASE_RTN_MSG(ZDNN_INVALID_STATE);
    CASE_RTN_MSG(ZDNN_UNSUPPORTED_AIU_EXCEPTION);
    CASE_RTN_MSG(ZDNN_UNSUPPORTED_PARMBLOCK);
    CASE_RTN_MSG(ZDNN_UNAVAILABLE_FUNCTION);
    CASE_RTN_MSG(ZDNN_UNSUPPORTED_FORMAT);
    CASE_RTN_MSG(ZDNN_UNSUPPORTED_TYPE);
    CASE_RTN_MSG(ZDNN_EXCEEDS_MDIS);
    CASE_RTN_MSG(ZDNN_EXCEEDS_MTS);
    CASE_RTN_MSG(ZDNN_MISALIGNED_TENSOR);
    CASE_RTN_MSG(ZDNN_MISALIGNED_SAVEAREA);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F000);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F001);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F002);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F003);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F004);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F005);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F006);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F007);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F008);
    CASE_RTN_MSG(ZDNN_FUNC_RC_F009);
  default:
    // can't find the corresponding string
    LOG_WARN("Unknown status code: %08x", status);
    return STATUS_MSG_UNKNOWN_STATUS;
  }
#undef CASE_RTN_MSG
}

/// Retrieve status string of the status code
///
/// \param[in] status status code
///
///
/// \return pointer to the status string
///
static const char *get_status_str(zdnn_status status) {

#define CASE_RTN_STR(a)                                                        \
  case a:                                                                      \
    return STATUS_STR_##a;

  switch (status) {
    CASE_RTN_STR(ZDNN_OK);
    CASE_RTN_STR(ZDNN_ELEMENT_RANGE_VIOLATION);
    CASE_RTN_STR(ZDNN_INVALID_SHAPE);
    CASE_RTN_STR(ZDNN_INVALID_LAYOUT);
    CASE_RTN_STR(ZDNN_INVALID_TYPE);
    CASE_RTN_STR(ZDNN_INVALID_FORMAT);
    CASE_RTN_STR(ZDNN_INVALID_DIRECTION);
    CASE_RTN_STR(ZDNN_INVALID_CONCAT_TYPE);
    CASE_RTN_STR(ZDNN_INVALID_STRIDE_PADDING);
    CASE_RTN_STR(ZDNN_INVALID_STRIDES);
    CASE_RTN_STR(ZDNN_MISALIGNED_PARMBLOCK);
    CASE_RTN_STR(ZDNN_INVALID_CLIPPING_VALUE);
    CASE_RTN_STR(ZDNN_ALLOCATION_FAILURE);
    CASE_RTN_STR(ZDNN_INVALID_BUFFER);
    CASE_RTN_STR(ZDNN_CONVERT_FAILURE);
    CASE_RTN_STR(ZDNN_INVALID_STATE);
    CASE_RTN_STR(ZDNN_UNSUPPORTED_AIU_EXCEPTION);
    CASE_RTN_STR(ZDNN_UNSUPPORTED_PARMBLOCK);
    CASE_RTN_STR(ZDNN_UNAVAILABLE_FUNCTION);
    CASE_RTN_STR(ZDNN_UNSUPPORTED_FORMAT);
    CASE_RTN_STR(ZDNN_UNSUPPORTED_TYPE);
    CASE_RTN_STR(ZDNN_EXCEEDS_MDIS);
    CASE_RTN_STR(ZDNN_EXCEEDS_MTS);
    CASE_RTN_STR(ZDNN_MISALIGNED_TENSOR);
    CASE_RTN_STR(ZDNN_MISALIGNED_SAVEAREA);
    CASE_RTN_STR(ZDNN_FUNC_RC_F000);
    CASE_RTN_STR(ZDNN_FUNC_RC_F001);
    CASE_RTN_STR(ZDNN_FUNC_RC_F002);
    CASE_RTN_STR(ZDNN_FUNC_RC_F003);
    CASE_RTN_STR(ZDNN_FUNC_RC_F004);
    CASE_RTN_STR(ZDNN_FUNC_RC_F005);
    CASE_RTN_STR(ZDNN_FUNC_RC_F006);
    CASE_RTN_STR(ZDNN_FUNC_RC_F007);
    CASE_RTN_STR(ZDNN_FUNC_RC_F008);
    CASE_RTN_STR(ZDNN_FUNC_RC_F009);
  default:
    // can't find the corresponding string
    LOG_WARN("Unknown status code: %08x", status);
    return STATUS_STR_UNKNOWN_STATUS;
  }
#undef CASE_RTN_STR
}

// maximum size for the format string, including the prepended STATUS_STR_XXX
#define MAX_STATUS_FMTSTR_SIZE 1024

zdnn_status set_zdnn_status(zdnn_status status, const char *func_name,
                            const char *file_name, int line_no,
                            const char *format, ...) {

  // when ZDNN_CONFIG_DEBUG is on, incoming status is either OK or not OK:
  // - ZDNN_OK: log as LOGLEVEL_INFO
  // - everything else: log as LOGLEVEL_ERROR
  //
  // when ZDNN_CONFIG_DEBUG is off, incoming status is always some sort of not
  // OK, use LOGLEVEL_ERROR so log_message() will send it to STDERR

  log_levels lvl_to_use =
#ifdef ZDNN_CONFIG_DEBUG
      (status == ZDNN_OK) ? LOGLEVEL_INFO :
#endif
                          LOGLEVEL_ERROR;

  if (format) {
    va_list argptr;
    va_start(argptr, format);

    // prepend status string "ZDNN_XXX: " to the incoming "format" string
    char full_fmtstr[MAX_STATUS_FMTSTR_SIZE];
    snprintf(full_fmtstr, MAX_STATUS_FMTSTR_SIZE,
             "%s: ", get_status_str(status));
    strncat(full_fmtstr, format, MAX_STATUS_FMTSTR_SIZE - 1);

    // "full_fmtstr" is now concatenated
    log_message(lvl_to_use, func_name, file_name, line_no, full_fmtstr, argptr);
    va_end(argptr);
  } else {
    // use the default status string if caller doesn't give us one
    log_message(lvl_to_use, func_name, file_name, line_no,
                zdnn_get_status_message(status), NULL);
  }

  /*
    collect backtrace information if status diag is enabled.
    different approaches for gcc vs xlc

    - xlc: use ctrace() to request traceback (via CEE3DMP), usually to a CEEDUMP
    - gcc: print backtrace

    information messages are sent to STDOUT regardless of log level
  */
  if (status == status_diag) { // assuming incoming status will never be
                               // STATUS_DIAG_NOT_SET

    printf("zDNN Diagnostic\n");
    printf("==================================================================="
           "===\n");
    printf("status = 0x%08x, %s\n", status, zdnn_get_status_message(status));

#ifdef __MVS__

#define DUMP_TITLE_SIZE 64
    printf("Invoking CEE3DMP Language Environment callable service...\n");
    char dump_title[DUMP_TITLE_SIZE];
    snprintf(dump_title, DUMP_TITLE_SIZE, "zDNN ctrace for status %08x",
             status);
    int rtn = ctrace(dump_title);
    if (rtn) {
      printf("Successfully invoked CEE3DMP.\n");
    } else {
      printf("Failed to invoke CEE3DMP.\n");
    }

#else

#define MAX_STACK_ENTRIES 30 // max num of stack entries to print
    printf("Backtrace:\n");

    void *array[MAX_STACK_ENTRIES];
    size_t size;
    char **strings;

    size = backtrace(array, MAX_STACK_ENTRIES);
    strings = backtrace_symbols(array, size);

    if (strings != NULL) {
      for (int i = 0; i < size; i++) {
        printf("%s\n", strings[i]);
      }
      free(strings);
    }

#endif // __MVS__

    printf("==================================================================="
           "===\n");
  }

  return status;
}
