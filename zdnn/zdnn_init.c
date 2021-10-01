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
#include <strings.h>

#if defined(__MVS__)
#include <cvt.h>
#include <ihaecvt.h>
#include <ihafacl.h>
#include <ihapsa.h>
#endif

#include "zdnn.h"
#include "zdnn_private.h"

#ifdef __MVS__
#pragma export(zdnn_init)
#pragma export(zdnn_is_nnpa_installed)
#endif

// global variables, set by zdnn_init() via environment vars
log_levels log_level = LOGLEVEL_ERROR; // log level (see enum log_levels)
bool precheck_enabled = false; // enables tensor pre-check before invoking NNPA
uint32_t status_diag = STATUS_DIAG_NOT_SET; // diagnostic info when status = X
char log_module[LOGMODULE_SIZE] = "\0";

// Index of the facility bit for the NNPA facility
#define STFLE_NNPA 165

/// Initialize the zDNN library and issue NNPA-QAF to the hardware.  Needs to be
/// invoked at least once during the lifetime of the application, either
/// manually or automatically via DLL-load initializer class.
///
/// \return None
///
void zdnn_init() {

  char *ptr, *endptr;

  if ((ptr = getenv(ENVVAR_LOGLEVEL))) {
    if (!strcasecmp("off", ptr)) {
      log_level = LOGLEVEL_OFF;
    }

    if (!strcasecmp("fatal", ptr)) {
      log_level = LOGLEVEL_FATAL;
    }

    if (!strcasecmp("error", ptr)) {
      log_level = LOGLEVEL_ERROR;
    }

    if (!strcasecmp("warn", ptr)) {
      log_level = LOGLEVEL_WARN;
    }

    if (!strcasecmp("info", ptr)) {
      log_level = LOGLEVEL_INFO;
    }

    if (!strcasecmp("debug", ptr)) {
      log_level = LOGLEVEL_DEBUG;
    }

    if (!strcasecmp("trace", ptr)) {
      log_level = LOGLEVEL_TRACE;
    }
  }

  if ((ptr = getenv(ENVVAR_ENABLE_PRECHECK))) {
    precheck_enabled = !strcasecmp("true", ptr);
  }

  if ((ptr = getenv(ENVVAR_STATUS_DIAG))) {

    uint32_t val;

    // if it's prefixed with "0x"/"0X" then treat it as hex string
    if (strstr(ptr, "0x") == ptr || strstr(ptr, "0X") == ptr) {
      val = (uint32_t)strtol((ptr + 2), &endptr, 16);
    } else {
      val = (uint32_t)strtol(ptr, &endptr, 10);
    }

    if (endptr == ptr + strlen(ptr)) {
      status_diag = val;
    }
  }

  if ((ptr = getenv(ENVVAR_LOGMODULE))) {
    strncpy(log_module, ptr, LOGMODULE_SIZE - 1);
  }

  /* Exit silently if there is no NNPA facility installed.  Explicit
     invocations of functions requiring NNPA will result in an
     error.  */
#ifndef ZDNN_CONFIG_NO_NNPA
  if (zdnn_is_nnpa_installed() == false)
    return;
#endif
  zdnn_refresh_nnpa_query_result();
}

#ifndef __MVS__
#define STFLE_LENGTH 32

static int invoke_stfle(unsigned char *facility_list) {
  register uint64_t r0 __asm__("%r0") = STFLE_LENGTH / 8 - 1;
  int cc;
  struct facility_list_type {
    // cppcheck-suppress unusedStructMember
    unsigned char flist[STFLE_LENGTH];
  };

  if (precheck_enabled) {
    // ensure facility_list is on a doubleword boundary.
    if ((uintptr_t)facility_list & 7)
      return ZDNN_STATUS_NO_MSG(ZDNN_MISALIGNED_PARMBLOCK);
  }

  // clang-format off
  __asm__ __volatile__("stfle   %[flist]"                 "\n\t"
                       "ipm     %[cc]"                    "\n\t"
                       "srl     %[cc],28"                 "\n\t"
                       : [flist] "+Q"(*((struct facility_list_type*)facility_list)),
			 "+d"(r0), [cc] "=d"(cc)
                       :
                       : "memory", "cc");
  // clang-format on
  return cc;
}

static inline int check_bitfield(uint8_t *bitfield, int bitno) {
  uint8_t mask = (1 << 7) >> (bitno & 7);
  return !!(bitfield[bitno / 8] & mask);
}
#endif

/// Determine if NNPA hardware support is available
///
/// The function unconditionally uses the STFLE instruction available
/// since IBM z9-109.
///
/// \param[in] None
///
/// \return true
///         false
///
bool zdnn_is_nnpa_installed() {
#ifndef __MVS__
  int nnpa_supported;
  unsigned char *facilities = alloca(STFLE_LENGTH);
  int cc;
  memset(facilities, 0, STFLE_LENGTH);
  cc = invoke_stfle(facilities);

  if (cc) {
    LOG_ERROR("STFLE failed with %d", cc);
    return false;
  }

  nnpa_supported = check_bitfield(facilities, STFLE_NNPA);

  if (nnpa_supported)
    LOG_INFO("Hardware NNPA support available", NO_ARG);
  else
    LOG_INFO("Hardware NNPA support not available", NO_ARG);

  return nnpa_supported;
#else
  /***********************************************************************
   * On z/OS, use system copy of STFLE output ("faclnnpaf").  (LoZ has to
   * worry about dynamic changes to STFLE.  z/OS does not support that so
   * using the static system copy is fine.)
   ***********************************************************************/
  struct psa *psaptr = (struct psa *)0;
  // cppcheck-suppress nullPointer
  struct cvtmap *cvtptr = (struct cvtmap *)psaptr->flccvt;
  struct ecvt *ecvtptr = (struct ecvt *)cvtptr->cvtecvt;
  struct facl *faclptr = (struct facl *)ecvtptr->ecvtfacl;

  return faclptr->faclnnpaf;

#endif
}
