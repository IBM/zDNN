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

/// Fill in NNPA tensor descriptor
///
/// \param[out] descriptor pointer to an nnpa_parameter_block's descriptor
/// \param[in] ztensor pointer to a zTensor
///
/// \return None
///
void populate_descriptor(nnpa_tensor_descriptor *descriptor,
                         const zdnn_ztensor *ztensor) {
  descriptor->data_layout_format = ztensor->transformed_desc->format;
  descriptor->data_type = ztensor->transformed_desc->type;
  descriptor->dim4_index_size = ztensor->transformed_desc->dim4;
  descriptor->dim3_index_size = ztensor->transformed_desc->dim3;
  descriptor->dim2_index_size = ztensor->transformed_desc->dim2;
  descriptor->dim1_index_size = ztensor->transformed_desc->dim1;
  descriptor->tensor_data_addr = ztensor->buffer;
}

/// Fill in NNPA parameter block
///
/// \param[out] parm_block pointer to a nnpa_parameter_block
/// \param[in] input_ztensor1
/// \param[in] input_ztensor2
/// \param[in] input_ztensor3
/// \param[in] output_ztensor1
/// \param[in] output_ztensor2
/// \param[in] func_sp_savearea_addr  Function-specific-save-area-address
/// \param[in] func_sp_parm1          Function-specific-parameter-1
/// \param[in] func_sp_parm2          Function-specific-parameter-2
/// \param[in] func_sp_parm3          Function-specific-parameter-3
/// \param[in] func_sp_parm4          Function-specific-parameter-4
/// \param[in] func_sp_parm5          Function-specific-parameter-5
///
/// \return None
///
void populate_nnpa_parm_block(
    nnpa_parameter_block *parm_block, uint16_t parm_block_version,
    const zdnn_ztensor *input_ztensor1, const zdnn_ztensor *input_ztensor2,
    const zdnn_ztensor *input_ztensor3, zdnn_ztensor *output_ztensor1,
    zdnn_ztensor *output_ztensor2, void *func_sp_savearea_addr,
    const function_specific_parameters *fsp) {

  // clear the block up to CSB
  memset(parm_block, 0,
         sizeof(nnpa_parameter_block) -
             offsetof(nnpa_parameter_block, continuation_state_buffer));
  parm_block->parm_block_version_number = parm_block_version;

  nnpa_tensor_descriptor *cur_desc_ptr;

  cur_desc_ptr = &(parm_block->input_tensor1);

  // there will be at least 1 input
  populate_descriptor(cur_desc_ptr, input_ztensor1);

  if (input_ztensor2) {
    cur_desc_ptr++;
    populate_descriptor(cur_desc_ptr, input_ztensor2);

    if (input_ztensor3) {
      cur_desc_ptr++;
      populate_descriptor(cur_desc_ptr, input_ztensor3);
    }
  }

  cur_desc_ptr = &(parm_block->output_tensor1);

  // there will be at least 1 output
  populate_descriptor(cur_desc_ptr, output_ztensor1);

  if (output_ztensor2) {
    cur_desc_ptr++;
    populate_descriptor(cur_desc_ptr, output_ztensor2);
  }

  parm_block->function_specific_save_area_address =
      (uintptr_t)func_sp_savearea_addr;
  parm_block->function_specific_parms = *fsp;
}

/// Invoke the NNPA instruction to drive a request to the zAIU
///
/// \param[in] function_code 1 byte zAIU function code
/// \param[in] parm_block pointer to a nnpa_parameter_block
/// \param[out] exception_flags 1 byte output exception flags
///
/// \return ZDNN_OK
///         ZDNN_UNAVAILABLE_FUNCTION
///         ZDNN_MISALIGNED_PARMBLOCK
///         ZDNN_HW_ERROR + hardware response code
///
zdnn_status invoke_nnpa(uint8_t function_code, char *parm_block,
                        uint8_t *exception_flags) {
  uint32_t cc = 0;
  nnpa_return rtn; // nnpa_return size set by NNPA architecture

  rtn.r0 = 0;

  if (precheck_enabled) {
    // When not on performance path add extra check to ensure NNPA parm block is
    // on a doubleword boundary.
    if ((uint64_t)parm_block & 7) {
      return ZDNN_STATUS_NO_MSG(ZDNN_MISALIGNED_PARMBLOCK);
    }
  }

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    if (function_code != NNPA_QAF) {

      printf("invoke_nnpa func_code %d\n", function_code);

      printf("            parm_block_version:\n");
      print_hex(
          sizeof(uint16_t),
          &((nnpa_parameter_block *)parm_block)->parm_block_version_number);
      printf("\n");

      printf("            input_tensor1:\n");
      print_hex(sizeof(nnpa_tensor_descriptor),
                &((nnpa_parameter_block *)parm_block)->input_tensor1);
      printf("\n");

      printf("            input_tensor2:\n");
      print_hex(sizeof(nnpa_tensor_descriptor),
                &((nnpa_parameter_block *)parm_block)->input_tensor2);
      printf("\n");

      printf("            input_tensor3:\n");
      print_hex(sizeof(nnpa_tensor_descriptor),
                &((nnpa_parameter_block *)parm_block)->input_tensor3);
      printf("\n");

      printf("            output_tensor1:\n");
      print_hex(sizeof(nnpa_tensor_descriptor),
                &((nnpa_parameter_block *)parm_block)->output_tensor1);
      printf("\n");

      printf("            output_tensor2:\n");
      print_hex(sizeof(nnpa_tensor_descriptor),
                &((nnpa_parameter_block *)parm_block)->output_tensor2);
      printf("\n");

      printf("            function_specific_parm1:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm1);
      printf("\n");

      printf("            function_specific_parm2:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm2);
      printf("\n");

      printf("            function_specific_parm3:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm3);
      printf("\n");

      printf("            function_specific_parm4:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm4);
      printf("\n");

      printf("            function_specific_parm5:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm5);
      printf("\n");

      printf("            function_specific_parm6:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm6);
      printf("\n");

      printf("            function_specific_parm7:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm7);
      printf("\n");

      printf("            function_specific_parm8:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm8);
      printf("\n");

      printf("            function_specific_parm9:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm9);
      printf("\n");

      printf("            function_specific_parm10:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm10);
      printf("\n");

      printf("            function_specific_parm11:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm11);
      printf("\n");

      printf("            function_specific_parm12:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm12);
      printf("\n");

      printf("            function_specific_parm13:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm13);
      printf("\n");

      printf("            function_specific_parm14:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm14);
      printf("\n");

      printf("            function_specific_parm15:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm15);
      printf("\n");

      printf("            function_specific_parm16:\n");
      print_hex(sizeof(uint32_t),
                &((nnpa_parameter_block *)parm_block)
                     ->function_specific_parms.function_specific_parm16);
      printf("\n");

      printf("            function_specific_save_area_address:\n");
      print_hex(sizeof(void *), &((nnpa_parameter_block *)parm_block)
                                     ->function_specific_save_area_address);
      printf("\n");
    }
  }

  // clang-format off
#if defined(__MVS__)
  struct psa *psaptr =
      (struct psa *)0;
  // set _cvt to x10, the pointer to the CVT on z/OS
  // cppcheck-suppress nullPointer
  struct cvtmap *cvtptr = (struct cvtmap *)psaptr->flccvt;
  struct ecvt *ecvtptr = (struct ecvt *)cvtptr->cvtecvt;
  struct facl *faclptr = (struct facl *)ecvtptr->ecvtfacl;

  if (faclptr->faclnnpaf) {
    __asm volatile(
   "      LLGC      0,%[valfunctionCode]        \n\t" // Insert function in R0
   "      LG        1,%[parm_block]             \n\t" // R1=parm_block, which is a pointer
   "      DC        XL4'B93B0000'               \n\t" // NNPA
   "      JO        -2                          \n\t" // jump back for CC3
   "      IPM       %[cc]                       \n\t" // fetch cc to cc reg
   "      SRL       %[cc],28                    \n\t" // shift
   "      LGR       %[areannpa_return],0        \n\t"
   : [cc] "+d" (cc),                                  // ASM outputs
     [areannpa_return] "=d" (rtn.r0)
   : [valfunctionCode] "m" (function_code),           // ASM inputs
     [parm_block] "m" (parm_block)
   : "r0", "r1", "cc"); /* Clobbers - R0, R1, cond code */
  }
  else
  {
        return ZDNN_STATUS(ZDNN_UNAVAILABLE_FUNCTION,
                       "NNPA facility unavailable", NO_ARG);
  }

#else
    register uint64_t r0 __asm__("%r0") = function_code;
    register uint64_t r1 __asm__("%r1") = (uint64_t)parm_block;

    __asm__ __volatile__ (
          "1: .long   0xb93b0000"     "\n\t"   // NNPA
          "   jo      1b"             "\n\t"   // on CC=3, jump to label '1'
          "   ipm     %[cc]"          "\n\t"   // fetch cc to cc reg
          "   srl     %[cc],28"       "\n\t"   // shift
    : [r0] "+d" (r0), [cc] "+d" (cc)          // ASM outputs
    : "d" (r1)                           // ASM inputs
    : "memory", "cc");                   // ASM clobbers
    rtn.r0 = r0;

#endif   // defined(__MVS__)
  // clang-format on

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    printf("invoke_nnpa CC %u:\n", cc);
    printf("            nnpa_return:\n");
    print_hex(8, &rtn.r0);
    printf("\n");
  }

  if (exception_flags)
    *exception_flags = rtn.fields.ef;

  if (cc == 0) {
    return ZDNN_STATUS_OK;
  } else {
    return ZDNN_STATUS_NO_MSG(ZDNN_HW_ERROR + rtn.fields.rc);
  }
}

/// Invoke the NNPA routine to drive a query request to the zAIU
///
/// \param[in] parm_block pointer to a nnpa_parameter_block
///
/// \return ZDNN_OK
///         ZDNN_UNAVAILABLE_FUNCTION
///         ZDNN_MISALIGNED_PARMBLOCK
///
/// \note invoke_nnpa could normally also send a condition code which would
/// lead to a ZDNN_HW_ERROR however documentation states that only CC 0 is
/// possible on NNPA_QAF.
///
zdnn_status invoke_nnpa_query(nnpa_qaf_parameter_block *qpb) {
#if defined(__MVS__)
  /***********************************************************************
   * On z/OS, use system copy of STFLE output ("faclnnpaf").  (LoZ has to
   * worry about dynamic changes to STFLE.  z/OS does not support that so
   * using the static system copy is fine.)
   ***********************************************************************/
  struct psa *psaptr =
      (struct psa *)0; // set _cvt to x10, the pointer to the CVT on z/OS
  // cppcheck-suppress nullPointer
  struct cvtmap *cvtptr = (struct cvtmap *)psaptr->flccvt;
  struct ecvt *ecvtptr = (struct ecvt *)cvtptr->cvtecvt;
  struct facl *faclptr = (struct facl *)ecvtptr->ecvtfacl;

  if (faclptr->faclnnpaf) {
    return invoke_nnpa(NNPA_QAF, (char *)qpb, NULL);
  } else {
    return ZDNN_STATUS(ZDNN_UNAVAILABLE_FUNCTION, "NNPA_QAF unavailable",
                       NO_ARG);
  }
#else
  /***********************************************************************
   * On Linux, invoke the function that uses STFLE to interrogate the
   * hardware.
   ***********************************************************************/
  if (zdnn_is_nnpa_installed() == true) {
    return invoke_nnpa(NNPA_QAF, (char *)qpb, NULL);
  } else {
    return ZDNN_STATUS(ZDNN_UNAVAILABLE_FUNCTION, "NNPA_QAF unavailable",
                       NO_ARG);
  }
#endif
}
