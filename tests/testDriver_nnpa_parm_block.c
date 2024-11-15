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

#include "testsupport.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Query block offsets
#define INSTALLED_FUNCTIONS_VECTOR_OFFSET 0
#define INSTALLED_PARAMETER_BLOCK_FORMATS_OFFSET 32
#define INSTALLED_DATA_TYPES_OFFSET 48
#define INSTALLED_DATA_LAYOUT_FORMATS_OFFSET 52
#define MAXIMUM_DIMENSION_INDEX_SIZE_OFFSET 60
#define MAXIMUM_TENSOR_SIZE_OFFSET 64
#define INSTALLED_DT1_CONVERSIONS_VECTOR_OFFSET 72
#define MAX_DIM4_INDEX_SIZE_OFFSET 88
#define MAX_DIM3_INDEX_SIZE_OFFSET 92
#define MAX_DIM2_INDEX_SIZE_OFFSET 96
#define MAX_DIM1_INDEX_SIZE_OFFSET 100

// Standard NNPA block offsets
#define PARM_BLOCK_VERSION_NUMBER_OFFSET 0
#define MODEL_VERSION_NUMBER_OFFSET 2
#define NNPA_RESERVED_FOR_IBM1 3
#define NNPA_RESERVED_FOR_IBM2 4
#define NNPA_RESERVED_FOR_IBM3 12
#define NNPA_RESERVED_FOR_IBM4 20
#define NNPA_RESERVED_FOR_IBM5 28
#define FUNC_SPECIFIC_SAVE_AREA_ADDR_OFFSET 56
#define OUTPUT_TENSOR_DESC_1_OFFSET 64
#define OUTPUT_TENSOR_DESC_2_OFFSET 96
#define INPUT_TENSOR_DESC_1_OFFSET 192
#define INPUT_TENSOR_DESC_2_OFFSET 224
#define INPUT_TENSOR_DESC_3_OFFSET 256
#define FUNCTION_SPECIFIC_PARMS 384
#define FUNCTION_SPECIFIC_PARM_1 384
#define FUNCTION_SPECIFIC_PARM_2 388
#define FUNCTION_SPECIFIC_PARM_3 392
#define FUNCTION_SPECIFIC_PARM_4 396
#define FUNCTION_SPECIFIC_PARM_5 400
#define FUNCTION_SPECIFIC_PARM_6 404
#define FUNCTION_SPECIFIC_PARM_7 408
#define FUNCTION_SPECIFIC_PARM_8 412
#define FUNCTION_SPECIFIC_PARM_9 416
#define FUNCTION_SPECIFIC_PARM_10 420
#define FUNCTION_SPECIFIC_PARM_11 424
#define FUNCTION_SPECIFIC_PARM_12 428
#define FUNCTION_SPECIFIC_PARM_13 432
#define FUNCTION_SPECIFIC_PARM_14 436
#define FUNCTION_SPECIFIC_PARM_15 440
#define FUNCTION_SPECIFIC_PARM_16 444
#define CSB_OFFSET 512

void setUp(void) {}

void tearDown(void) {}

/*
 * Verify that the tensor descriptor was updated with the correct
 * information from the ztensor. invalid_type set when testing for invalid
 * data_type.
 */
void verify_populate_descriptor(nnpa_tensor_descriptor *descriptor,
                                zdnn_ztensor *ztensor) {
  LOG_DEBUG("Verifying descriptor", NO_ARG);
  TEST_ASSERT_EQUAL_UINT8_MESSAGE(ztensor->transformed_desc->format,
                                  descriptor->data_layout_format,
                                  "Incorrect data layout format.");
  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ztensor->transformed_desc->dim4,
                                   descriptor->dim4_index_size,
                                   "Incorrect dim4 index size");
  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ztensor->transformed_desc->dim3,
                                   descriptor->dim3_index_size,
                                   "Incorrect dim3 index size");
  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ztensor->transformed_desc->dim2,
                                   descriptor->dim2_index_size,
                                   "Incorrect dim2 index size");
  TEST_ASSERT_EQUAL_UINT32_MESSAGE(ztensor->transformed_desc->dim1,
                                   descriptor->dim1_index_size,
                                   "Incorrect dim1 index size");
  TEST_ASSERT_EQUAL_UINT64_MESSAGE(ztensor->buffer,
                                   descriptor->tensor_data_addr,
                                   "Incorrect tensor pointer");
}

/*
 * Common routine for driving all x-inputs y-outputs testcases
 * variadic parameters are input dims followed by output dims, which the
 * dims are in {outermost, ..., innermost} order
 */
void populate_x_inputs_y_outputs(uint8_t num_inputs, uint8_t num_outputs,
                                 zdnn_data_types type, ...) {

  // Allocate and initialize our nnpa_parm_blocks
  nnpa_parameter_block parm_block;
  nnpa_parameter_block parm_block_all;

  zdnn_ztensor input_ztensor[num_inputs], output_ztensor[num_outputs];
  int dummy; // something for ztensor.buffer to point to

  va_list ap;
  va_start(ap, type);

  // variadic: input dim arrays then output dim arrays
  for (int i = 0; i < num_inputs; i++) {
    uint32_t *dims = va_arg(ap, uint32_t *);
    input_ztensor[i].transformed_desc = malloc(sizeof(zdnn_tensor_desc));
    // dims[0] is the outermost dimension
    init_transformed_desc(ZDNN_NHWC, type, ZDNN_FORMAT_4DFEATURE,
                          input_ztensor[i].transformed_desc, dims[0], dims[1],
                          dims[2], dims[3]);
    input_ztensor[i].buffer = &dummy;
  }
  for (int i = 0; i < num_outputs; i++) {
    uint32_t *dims = va_arg(ap, uint32_t *);
    output_ztensor[i].transformed_desc = malloc(sizeof(zdnn_tensor_desc));
    init_transformed_desc(ZDNN_NHWC, type, ZDNN_FORMAT_4DFEATURE,
                          output_ztensor[i].transformed_desc, dims[0], dims[1],
                          dims[2], dims[3]);
    output_ztensor[i].buffer = &dummy;
  }

  va_end(ap);

  function_specific_parameters fsp = {0};

  populate_nnpa_parm_block(
      &parm_block_all, NNPA_PARMBLKFORMAT_0, &input_ztensor[0],
      (num_inputs > 1) ? &input_ztensor[1] : NULL,
      (num_inputs > 2) ? &input_ztensor[2] : NULL, &output_ztensor[0],
      (num_outputs > 1) ? &output_ztensor[1] : NULL, 0, &fsp);

  // treat parm_block->input_tensor1/2/3 as if an array so we can loop them
  nnpa_tensor_descriptor *block_input_ptr = &(parm_block.input_tensor1);
  nnpa_tensor_descriptor *block_all_input_ptr = &(parm_block_all.input_tensor1);
  for (int i = 0; i < num_inputs; i++) {
    populate_descriptor(block_input_ptr + i, &input_ztensor[i]);
    verify_populate_descriptor(block_all_input_ptr + i, &input_ztensor[i]);
    verify_populate_descriptor(block_input_ptr + i, &input_ztensor[i]);
  }

  nnpa_tensor_descriptor *block_output_ptr = &(parm_block.output_tensor1);
  nnpa_tensor_descriptor *block_all_output_ptr =
      &(parm_block_all.output_tensor1);
  for (int i = 0; i < num_outputs; i++) {
    populate_descriptor(block_output_ptr + i, &output_ztensor[i]);
    verify_populate_descriptor(block_all_output_ptr + i, &output_ztensor[i]);
    verify_populate_descriptor(block_output_ptr + i, &output_ztensor[i]);
  }

  for (int i = 0; i < num_inputs; i++) {
    free(input_ztensor[i].transformed_desc);
  }
  for (int i = 0; i < num_outputs; i++) {
    free(output_ztensor[i].transformed_desc);
  }
}

/*
 * Test to ensure using either populate_descriptor or populate_all_descriptor
 * updates the nnpa parm block appropriately for 1 input tensor
 */
void populate_single_input() {
  uint32_t shape[ZDNN_MAX_DIMS] = {1, 1, 1, 3};

  populate_x_inputs_y_outputs(1, 1, ZDNN_DLFLOAT16, shape, shape);
}

/*
 * Test to ensure using either populate_descriptor or populate_all_descriptor
 * updates the nnpa parm block appropriately for 1 input tensor and 2 output
 * tensors
 */
void populate_single_input_double_output() {
  uint32_t shape[ZDNN_MAX_DIMS] = {1, 1, 1, 3};

  populate_x_inputs_y_outputs(1, 2, ZDNN_DLFLOAT16, shape, shape, shape);
}

/*
 * Test to ensure using either populate_descriptor or populate_all_descriptor
 * updates the nnpa parm block appropriately for 2 input tensors
 */
void populate_double_input() {
  unsigned int input_dims[ZDNN_MAX_DIMS] = {4, 2, 1, 3};
  unsigned int output_dims[ZDNN_MAX_DIMS] = {2, 1, 5, 2};

  populate_x_inputs_y_outputs(2, 1, ZDNN_DLFLOAT16, input_dims, input_dims,
                              output_dims);
}

/*
 * Test to ensure using either populate_descriptor or populate_all_descriptor
 * updates the nnpa parm block appropriately for 3 input tensors
 */
void populate_triple_input() {
  unsigned int input_dims[ZDNN_MAX_DIMS] = {5, 3, 1, 1};
  unsigned int output_dims[ZDNN_MAX_DIMS] = {8, 1, 2, 4};

  populate_x_inputs_y_outputs(3, 1, ZDNN_DLFLOAT16, input_dims, input_dims,
                              input_dims, output_dims);
}

/**
 * Function to verify the offsets of each element in a nnpa_parameter_block
 * struct.
 *
 * Parameter block offsets:
 *
  Bytes:        Name:
  0-1           PBVN
  2             MVN
  3             RIBM1
  4-5           RIBM2
  6-7           Reserved (1 bit latency flag and 1-bit Continuation Flag at end)
  8-11          Reserved
  12-15         RIBM3
  16-19         Reserved
  20-23         RIBM4
  24-27         Reserved
  28-31         RIBM5
  32-55         Reserved
  56-63         Function-specific-save-area-address
  64-95         Output Tensor Descriptor 1
  96-127        Output Tensor Descriptor 2
  128-191       Reserved
  192-223       Input Tensor Descriptor 1
  224-255       Input Tensor Descriptor 2
  256-287       Input Tensor Descriptor 3
  288-383       Reserved
  384-387       Function-specific-parameter-1
  388-391       Function-specific-parameter-2
  392-295       Function-specific-parameter-3
  396-399       Function-specific-parameter-4
  400-403       Function-specific-parameter-5
  404-407       Function-specific-parameter-6
  408-411       Function-specific-parameter-7
  412-415       Function-specific-parameter-8
  416-419       Function-specific-parameter-9
  420-423       Function-specific-parameter-10
  424-427       Function-specific-parameter-11
  428-431       Function-specific-parameter-12
  432-435       Function-specific-parameter-13
  436-439       Function-specific-parameter-14
  440-443       Function-specific-parameter-15
  444-447       Function-specific-parameter-16
  448-511       Reserved
  512-4088      CSB
 */
void verify_parm_block_offsets() {
  TEST_ASSERT_EQUAL_MESSAGE(
      PARM_BLOCK_VERSION_NUMBER_OFFSET,
      offsetof(nnpa_parameter_block, parm_block_version_number),
      "parm_block_version in nnpa_parameter_block has incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MODEL_VERSION_NUMBER_OFFSET,
      offsetof(nnpa_parameter_block, model_version_number),
      "model_version in nnpa_parameter_block has incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      NNPA_RESERVED_FOR_IBM1, offsetof(nnpa_parameter_block, reserved_for_ibm1),
      "reserved for IBM in nnpa_parameter_block has incorrect offset");
  /* Skipping as this member has been moved into a bitfield which an offset
   * can't be determined.
  TEST_ASSERT_EQUAL_MESSAGE(
      NNPA_RESERVED_FOR_IBM2, offsetof(nnpa_parameter_block, reserved_for_ibm2),
      "reserved for IBM 2 in nnpa_parameter_block has incorrect offset");
   */
  TEST_ASSERT_EQUAL_MESSAGE(
      NNPA_RESERVED_FOR_IBM3, offsetof(nnpa_parameter_block, reserved_for_ibm3),
      "reserved for IBM 3 in nnpa_parameter_block has incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      NNPA_RESERVED_FOR_IBM4, offsetof(nnpa_parameter_block, reserved_for_ibm4),
      "reserved for IBM 4 in nnpa_parameter_block has incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      NNPA_RESERVED_FOR_IBM5, offsetof(nnpa_parameter_block, reserved_for_ibm5),
      "reserved for IBM 5 in nnpa_parameter_block has incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNC_SPECIFIC_SAVE_AREA_ADDR_OFFSET,
      offsetof(nnpa_parameter_block, function_specific_save_area_address),
      "function_specific_save_area_address in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(OUTPUT_TENSOR_DESC_1_OFFSET,
                            offsetof(nnpa_parameter_block, output_tensor1),
                            "output_tensor1 in nnpa_parameter_block has "
                            "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(OUTPUT_TENSOR_DESC_2_OFFSET,
                            offsetof(nnpa_parameter_block, output_tensor2),
                            "output_tensor2 in nnpa_parameter_block has "
                            "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(INPUT_TENSOR_DESC_1_OFFSET,
                            offsetof(nnpa_parameter_block, input_tensor1),
                            "input_tensor1 in nnpa_parameter_block has "
                            "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(INPUT_TENSOR_DESC_2_OFFSET,
                            offsetof(nnpa_parameter_block, input_tensor2),
                            "input_tensor2 in nnpa_parameter_block has "
                            "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(INPUT_TENSOR_DESC_3_OFFSET,
                            offsetof(nnpa_parameter_block, input_tensor3),
                            "input_tensor3 in nnpa_parameter_block has "
                            "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARMS,
      offsetof(nnpa_parameter_block, function_specific_parms),
      "function_specific_parms in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_1,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm1)),
      "function_specific_parm1 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_2,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm2)),
      "function_specific_parm2 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_3,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm3)),
      "function_specific_parm3 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_4,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm4)),
      "function_specific_parm4 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_5,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm5)),
      "function_specific_parm5 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_6,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm6)),
      "function_specific_parm6 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_7,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm7)),
      "function_specific_parm7 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_8,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm8)),
      "function_specific_parm8 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_9,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm9)),
      "function_specific_parm9 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_10,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm10)),
      "function_specific_parm10 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_11,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm11)),
      "function_specific_parm11 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_12,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm12)),
      "function_specific_parm12 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_13,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm13)),
      "function_specific_parm13 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_14,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm14)),
      "function_specific_parm14 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_15,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm15)),
      "function_specific_parm15 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      FUNCTION_SPECIFIC_PARM_16,
      (offsetof(nnpa_parameter_block, function_specific_parms) +
       offsetof(function_specific_parameters, function_specific_parm16)),
      "function_specific_parm16 in nnpa_parameter_block has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      CSB_OFFSET, offsetof(nnpa_parameter_block, continuation_state_buffer),
      "continuation_state_buffer in nnpa_parameter_block has "
      "incorrect offset");
}

/**
 * Function to verify the offsets of each element in a
 * aiu_parameter_block_nnpa_qaf struct.
 *
 * Parameter block offsets:
 *
  Bytes:        Name:
  0-31          installed_functions_vector;
  32-47         installed_parameter_block_formats;
  48-49         installed_data_types;
  50-51         reserved1[2]
  52-55         installed_data_layout_formats;
  56-59         reserved2[4];
  60-63         maximum_dimension_index_size;
  64-71         maximum_tensor_size;
  72-73         installed_dt1_conversions_vector
  74-87         reserved3[14];
  88-91         max_dim4_index_size
  92-95         max_dim3_index_size
  96-99         max_dim2_index_size
  100-103       max_dim1_index_size
  104-256       reserved4[152]
 */
void verify_qaf_parm_block_offsets() {
  TEST_ASSERT_EQUAL_MESSAGE(
      INSTALLED_FUNCTIONS_VECTOR_OFFSET,
      offsetof(nnpa_qaf_parameter_block, installed_functions_vector),
      "installed_functions_vector in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      INSTALLED_PARAMETER_BLOCK_FORMATS_OFFSET,
      offsetof(nnpa_qaf_parameter_block, installed_parameter_block_formats),
      "reserved1 in aiu_parameter_block_nnpa_qaf has incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      INSTALLED_DATA_TYPES_OFFSET,
      offsetof(nnpa_qaf_parameter_block, installed_data_types),
      "installed_data_type in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      INSTALLED_DATA_LAYOUT_FORMATS_OFFSET,
      offsetof(nnpa_qaf_parameter_block, installed_data_layout_formats),
      "installed_data_layout_formats in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MAXIMUM_DIMENSION_INDEX_SIZE_OFFSET,
      offsetof(nnpa_qaf_parameter_block, maximum_dimension_index_size),
      "maximum_dimension_index_size in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MAXIMUM_TENSOR_SIZE_OFFSET,
      offsetof(nnpa_qaf_parameter_block, maximum_tensor_size),
      "maximum_tensor_size in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      INSTALLED_DT1_CONVERSIONS_VECTOR_OFFSET,
      offsetof(nnpa_qaf_parameter_block, installed_dt1_conversions_vector),
      "installed_dt1_conversions_vector in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MAX_DIM4_INDEX_SIZE_OFFSET,
      offsetof(nnpa_qaf_parameter_block, max_dim4_index_size),
      "max_dim4_index_size in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MAX_DIM3_INDEX_SIZE_OFFSET,
      offsetof(nnpa_qaf_parameter_block, max_dim3_index_size),
      "max_dim3_index_size in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MAX_DIM2_INDEX_SIZE_OFFSET,
      offsetof(nnpa_qaf_parameter_block, max_dim2_index_size),
      "max_dim2_index_size in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
  TEST_ASSERT_EQUAL_MESSAGE(
      MAX_DIM1_INDEX_SIZE_OFFSET,
      offsetof(nnpa_qaf_parameter_block, max_dim1_index_size),
      "max_dim1_index_size in aiu_parameter_block_nnpa_qaf has "
      "incorrect offset");
}

void verify_parm_block_size() {
  TEST_ASSERT_EQUAL_MESSAGE(NNPA_PARMBLOCK_SIZE, sizeof(nnpa_parameter_block),
                            "size of nnpa_paramater_block is incorrect");
}

void verify_qaf_parm_block_size() {
  TEST_ASSERT_EQUAL_MESSAGE(QAF_PARMBLOCK_SIZE,
                            sizeof(nnpa_qaf_parameter_block),
                            "size of nnpa_qaf_paramater_block is incorrect");
}

int main() {
  UNITY_BEGIN();
  RUN_TEST(populate_single_input);
  RUN_TEST(populate_single_input_double_output);
  RUN_TEST(populate_double_input);
  RUN_TEST(populate_triple_input);
  RUN_TEST(verify_parm_block_offsets);
  RUN_TEST(verify_qaf_parm_block_offsets);
  RUN_TEST(verify_parm_block_size);
  RUN_TEST(verify_qaf_parm_block_size);
  return UNITY_END();
}
