
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// Determine if file_name is within ZDNN_LOGMODULE
///
/// \param[in] file_name Pointer to file name
///
/// \return true/false
///
bool logmodule_matches(const char *file_name) {
  if (log_module[0] == '\0') {
    // ZDNN_LOGMODULE is never set
    return true;
  } else {

    // want only the filename, don't want the path
    const char *basename = strrchr(file_name, '/');
    if (basename) {
      basename++;
    } else {
      basename = file_name;
    }

    return (strstr(log_module, basename) ? true : false);
  }
}

void log_error(const char *func_name, const char *file_name, int line_no,
               char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  log_message(LOGLEVEL_ERROR, func_name, file_name, line_no, format, argptr);
  va_end(argptr);
}

void log_warn(const char *func_name, const char *file_name, int line_no,
              char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  log_message(LOGLEVEL_WARN, func_name, file_name, line_no, format, argptr);
  va_end(argptr);
}
void log_info(const char *func_name, const char *file_name, int line_no,
              char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  log_message(LOGLEVEL_INFO, func_name, file_name, line_no, format, argptr);
  va_end(argptr);
}

void log_debug(const char *func_name, const char *file_name, int line_no,
               char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  log_message(LOGLEVEL_DEBUG, func_name, file_name, line_no, format, argptr);
  va_end(argptr);
}

void log_trace(const char *func_name, const char *file_name, int line_no,
               char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  log_message(LOGLEVEL_TRACE, func_name, file_name, line_no, format, argptr);
  va_end(argptr);
}

void log_fatal(const char *func_name, const char *file_name, int line_no,
               char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  log_message(LOGLEVEL_FATAL, func_name, file_name, line_no, format, argptr);
  va_end(argptr);
}

#define LOG_BUFFER_LEN 512 // max length of entire log message
#define LOG_HEADER_LEN 96  // max length of header within the message
#define LOG_HEADER "%s: %s() (%s:%d): " // level, __func__, __FILE__,  __LINE

/// Log message to STDOUT/STDERR
///
/// \param[in] lvl Message's LOGLEVEL
/// \param[in] func_name Calling module's function name
/// \param[in] file_name Calling module's fila name
/// \param[in] line_no Calling module's line number
/// \param[in] format printf()-style format string
/// \param[in] arg Variadic parameters list in va_list form
///
///
/// \return None
///
void log_message(log_levels lvl, const char *func_name, const char *file_name,
                 int line_no, const char *format, va_list arg) {

  // when ZDNN_CONFIG_DEBUG is off, LOGLEVEL and LOGMODULE are not supported so
  // always execute this block of code
#ifdef ZDNN_CONFIG_DEBUG
  BEGIN_IF_LOGLEVEL(lvl, file_name) {
#endif

    char msg_buf[LOG_BUFFER_LEN];
    FILE *stream;
    log_levels lvl_real = (lvl > LOGLEVEL_TRACE) ? LOGLEVEL_TRACE : lvl;

    char *log_levels_str[] = {"",     "FATAL", "ERROR", "WARN",
                              "INFO", "DEBUG", "TRACE"};

    int header_len =
        snprintf(msg_buf, LOG_BUFFER_LEN, LOG_HEADER, log_levels_str[lvl_real],
                 func_name, file_name, line_no);

    if (arg) {
      vsnprintf(msg_buf + header_len, LOG_BUFFER_LEN - header_len, format, arg);
    } else {
      // nothing to format, copy it as-is
      strncpy(msg_buf + header_len, format, LOG_BUFFER_LEN - header_len);
    }

    if (lvl_real == LOGLEVEL_ERROR || lvl_real == LOGLEVEL_FATAL) {
      stream = stderr;
    } else {
      stream = stdout;
    }

    // auto '\n` the string
    if (msg_buf[strlen(msg_buf) - 1] == '\n') {
      fprintf(stream, "%s", msg_buf);
    } else {
      fprintf(stream, "%s\n", msg_buf);
    }
#ifdef ZDNN_CONFIG_DEBUG
  }
#endif
}
