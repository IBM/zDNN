// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "testsupport.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) { /* This is run before EACH TEST */
}

void tearDown(void) {}

/*
 * Test ztensor format when created and updated.
 */
void verify_ztensor_format() {

  VERIFY_HW_ENV; // verify required HW env is available.

  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;
  void *data;
  uint32_t dim4 = 1, dim3 = 4, dim2 = 4, dim1 = 1;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, FP32, &pre_tfrmd_desc, dim4, dim3,
                                 dim2, dim1);

  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_generate_transformed_desc() failed (status = %08x)", status);

  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_OK,
      "zdnn_init_ztensor_with_malloc() failed (status = %08x)", status);

  // verify proper state of is_transformed field after ztensor created
  TEST_ASSERT_MESSAGE(
      false == ztensor.is_transformed,
      "Expected ztensor to indicate transform not completed yet.");

  data = create_and_fill_random_fp_data(&ztensor);

  // transform the app tensor's data into stickified data
  LOG_DEBUG("about to transform ztensor", NO_ARG);
  status = zdnn_transform_ztensor(&ztensor, data);
  TEST_ASSERT_MESSAGE(ZDNN_OK == status,
                      "zdnn_transform_ztensor did not return OK as expected");

  // verify proper state of is_transformed field after ztensor has stickified
  // data
  TEST_ASSERT_MESSAGE(true == ztensor.is_transformed,
                      "Expected ztensor to indicate transform was completed.");

  // Free allocated storage
  free(data);
  zdnn_free_ztensor_buffer(&ztensor);
}

/// Common test routine for normal tensors
///
/// \param[in] num_inputs         Number of input tensors
/// \param[in] input_shape_lst    Pointer to array of pointers to input dim
///                               arrays
/// \param[in] input_format_lst   Pointer to array of input formats
/// \param[in] input_type_lst     Pointer to array of input types
/// \param[in] num_outputs        Number of output tensors
/// \param[in] output_shape_lst   Pointer to array of pointers to output dim
///                               arrays
/// \param[in] output_format_lst  Pointer to array of output formats
/// \param[in] output_type_lst    Pointer to array of output types
/// \param[in] exp_status         Expected status
/// \param[in] error_msg          Error message to prepend to the standard error
///                               message
///
void test_normal(uint8_t num_inputs, uint32_t **input_shape_lst,
                 zdnn_data_formats *input_format_lst,
                 zdnn_data_types *input_type_lst, uint8_t num_outputs,
                 uint32_t **output_shape_lst,
                 zdnn_data_formats *output_format_lst,
                 zdnn_data_types *output_type_lst, zdnn_status exp_status,
                 char *error_msg) {
  zdnn_ztensor input_ztensor[num_inputs];
  zdnn_ztensor output_ztensor[num_outputs];
  zdnn_status status = ZDNN_OK;

  // allocate a transformed descriptor with input_shape_lst[i],
  // input_format_lst[i] and input_type_lst[i]
  for (int i = 0; i < num_inputs; i++) {
    uint32_t *shape = input_shape_lst[i];
    input_ztensor[i].transformed_desc = malloc(sizeof(zdnn_tensor_desc));

    init_transformed_desc(
        input_format_lst[i] == ZDNN_FORMAT_4DFEATURE ? ZDNN_NHWC : ZDNN_HWCK,
        input_type_lst[i], input_format_lst[i],
        input_ztensor[i].transformed_desc, shape[0], shape[1], shape[2],
        shape[3]);
  }

  // same idea with the outputs
  for (int i = 0; i < num_outputs; i++) {
    uint32_t *shape = output_shape_lst[i];
    output_ztensor[i].transformed_desc = malloc(sizeof(zdnn_tensor_desc));

    init_transformed_desc(
        output_format_lst[i] == ZDNN_FORMAT_4DFEATURE ? ZDNN_NHWC : ZDNN_HWCK,
        output_type_lst[i], output_format_lst[i],
        output_ztensor[i].transformed_desc, shape[0], shape[1], shape[2],
        shape[3]);
  }

  // number of inputs to send to verify_tensors() depends on num_inputs
  status = verify_tensors(
      &input_ztensor[0], (num_inputs > 1) ? &input_ztensor[1] : NULL,
      (num_inputs > 2) ? &input_ztensor[2] : NULL, &output_ztensor[0]);

  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_status == status, "%s  Expected status = %08x, actual status = %08x",
      error_msg, exp_status, status);

  for (int i = 0; i < num_inputs; i++) {
    free(input_ztensor[i].transformed_desc);
  }

  for (int i = 0; i < num_outputs; i++) {
    free(output_ztensor[i].transformed_desc);
  }
}

/*
 * Test verification of valid output tensor along with an input tensor.
 * All tensors will be built with same properties.
 */
void verify_1input_pass() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(1, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst, ZDNN_OK,
              "The output and the input tensor is different.");
}

/*
 * Test verification of valid output tensor along with 2 input tensors.
 * All tensors will be built with same properties.
 */
void verify_2input_pass() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE,
                                          ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(2, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst, ZDNN_OK,
              "The output and the input tensors are different.");
}

/*
 * Test verification of valid output tensor along with 3 input tensors.
 * All tensors will be built with same properties.
 */
void verify_3input_pass() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16,
                                      ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(3, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst, ZDNN_OK,
              "The output and the input tensors are different.");
}

/*
 * Test verification of different shapes between 2 input tensors.
 * Input tensors will have different shapes.
 * Output tensor will have same properties as Input tensor 1.
 */
void verify_input2_fail_shape() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};
  uint32_t different_shape[ZDNN_MAX_DIMS] = {1, 2, 3, 4};

  uint32_t *input_shape_lst[] = {io_shape, different_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE,
                                          ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(2, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_SHAPE,
              "Failed to fail on different input tensor shapes.");
}

/*
 * Test verification of different shapes between 3 input tensors.
 * Input tensor 3 will have different shapes.
 * Output tensor will have same properties as Input tensor 1 and 2.
 */
void verify_input3_fail_shape() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};
  uint32_t different_shape[ZDNN_MAX_DIMS] = {1, 2, 3, 4};

  uint32_t *input_shape_lst[] = {io_shape, io_shape, different_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16,
                                      ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(3, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_SHAPE,
              "Failed to fail on different input tensor shapes.");
}

/*
 * Test verification of different data formats between 2 input tensors.
 * Input tensors will have different data formats.
 * Output tensor will have same properties as Input tensor 1.
 */
void verify_input2_fail_format() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE,
                                          ZDNN_FORMAT_4DKERNEL};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(2, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_FORMAT,
              "Failed to fail on different input tensor data formats.");
}

/*
 * Test verification of different data formats between 3 input tensors.
 * Input tensor 3 will have different data formats.
 * Output tensor will have same properties as Input tensor 1 and 2.
 */
void verify_input3_fail_format() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16,
                                      ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DKERNEL};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(3, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_FORMAT,
              "Failed to fail on different input tensor data formats.");
}

/*
 * Test verification of different data types between 2 input tensors.
 * Input tensors will have different data types.
 * Output tensor will have same properties as Input tensor 1.
 */
void verify_input2_fail_dtype() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, FP32};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE,
                                          ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(2, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_TYPE,
              "Failed to fail on different input tensor data types.");
}

/*
 * Test verification of different data types between 3 input tensors.
 * Input tensor 3 will have different data type.
 * Output tensor will have same properties as Input tensor 1 and 2.
 */
void verify_input3_fail_dtype() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, FP32};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(3, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_TYPE,
              "Failed to fail on different input tensor data types.");
}

/*
 * Test verification of different shapes between output and input tensor.
 * Input and Output tensor will have a different shape.
 */
void verify_output_fail_shape() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};
  uint32_t different_shape[ZDNN_MAX_DIMS] = {1, 2, 3, 4};

  uint32_t *input_shape_lst[] = {io_shape};
  uint32_t *output_shape_lst[] = {different_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(1, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_SHAPE,
              "Failed to fail on different output/input tensor shapes.");
}

/*
 * Test verification of different data format between output and input
 * tensors. Both input tensors will the same properties. Output tensor will
 * have a different data format.
 */
void verify_output_fail_format() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {ZDNN_DLFLOAT16};

  zdnn_data_formats input_format_lst[] = {ZDNN_FORMAT_4DFEATURE,
                                          ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DKERNEL};

  test_normal(2, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_FORMAT,
              "Failed to fail on different output/input tensor data formats.");
}

/*
 * Test verification of different data types between output and input tensors.
 * All three input tensors will have the same properties.
 * Output tensor will have a different data type.
 */
void verify_output_fail_dtype() {
  uint32_t io_shape[ZDNN_MAX_DIMS] = {1, 1, 4, 3};

  uint32_t *input_shape_lst[] = {io_shape, io_shape, io_shape};
  uint32_t *output_shape_lst[] = {io_shape};

  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16,
                                      ZDNN_DLFLOAT16};
  zdnn_data_types output_type_lst[] = {FP32};

  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_formats output_format_lst[] = {ZDNN_FORMAT_4DFEATURE};

  test_normal(3, input_shape_lst, input_format_lst, input_type_lst, 1,
              output_shape_lst, output_format_lst, output_type_lst,
              ZDNN_INVALID_TYPE,
              "Failed to fail on different output/input tensor data types.");
}

#define MATMUL_NUM_INPUTS 3

/// Common test routine for matmul op + mutmul bcast op tensors
///
/// \param[in] uint8_t function_code,
///                   NNPA_MATMUL_OP or NNPA_MATMUL_OP_BCAST23
/// \param[in] input_shape_lst
///                   2D array, MATMUL_NUM_INPUTS x ZDNN_MAX_DIMS number of
///                   dimensions
/// \param[in] input_shape_displace_lst
///                   MATMUL_NUM_INPUTS x ZDNN_MAX_DIMS number of
///                   displacement for each of the entries in input_shape_lst
///                   (e.g., +1, +5, -3, etc)
/// \param[in] input_format_lst
///                   array, MATMUL_NUM_INPUTS number of entries of formats
/// \param[in] input_type_lst
///                   array, MATMUL_NUM_INPUTS number of entries of types
/// \param[in] output_shape
///                   1D array, ZDNN_MAX_DIMS number of dimensions
/// \param[in] output_shape_displace
///                   ZDNN_MAX_DIMS number of displacement for each of the
///                   entries in output_shape
/// \param[in] output_format      output format
/// \param[in] output_type        output type
/// \param[in] exp_status         Expected status
///
void test_matmul(
    uint8_t function_code,
    uint32_t input_shape_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS],
    int32_t input_shape_displace_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS],
    zdnn_data_formats *input_format_lst, zdnn_data_types *input_type_lst,
    uint32_t *output_shape, int32_t *output_shape_displace,
    zdnn_data_formats output_format, zdnn_data_types output_type,
    zdnn_status exp_status) {

  zdnn_ztensor input_ztensor[MATMUL_NUM_INPUTS];
  zdnn_ztensor output_ztensor;
  zdnn_status status = ZDNN_OK;

  /*
    create MATMUL_NUM_INPUTS numbers of transformed descriptors, using:

    input_shape_lst[i] + input_shape_displace_lst[i] as shape

    e.g., input_shape_lst[i] = {1, 2, 3, 4}
          input_shape_displace_lst[i] = {0, 1, -1, 5}
          resultant to init_transformed_desc() = { 1 + 0 = 1,
                                                   2 + 1 = 3,
                                                   3 + -1 = 2,
                                                   4 + 5 = 9 }

    input_format_lst[i] as format
    input_type_lst[i] as type
  */
  for (int i = 0; i < MATMUL_NUM_INPUTS; i++) {
    input_ztensor[i].transformed_desc = malloc(sizeof(zdnn_tensor_desc));

    LOG_DEBUG("input %d -> format %d, type %d\n", i, input_format_lst[i],
              input_type_lst[i]);
    LOG_DEBUG("            dim4 %d, displace %d\n", input_shape_lst[i][0],
              input_shape_displace_lst[i][0]);
    LOG_DEBUG("            dim3 %d, displace %d\n", input_shape_lst[i][1],
              input_shape_displace_lst[i][1]);
    LOG_DEBUG("            dim2 %d, displace %d\n", input_shape_lst[i][2],
              input_shape_displace_lst[i][2]);
    LOG_DEBUG("            dim1 %d, displace %d\n", input_shape_lst[i][3],
              input_shape_displace_lst[i][3]);

    init_transformed_desc(
        input_format_lst[i] == ZDNN_FORMAT_4DFEATURE ? ZDNN_NHWC : ZDNN_HWCK,
        input_type_lst[i], input_format_lst[i],
        input_ztensor[i].transformed_desc,
        input_shape_lst[i][0] + input_shape_displace_lst[i][0],
        input_shape_lst[i][1] + input_shape_displace_lst[i][1],
        input_shape_lst[i][2] + input_shape_displace_lst[i][2],
        input_shape_lst[i][3] + input_shape_displace_lst[i][3]);
  }

  LOG_DEBUG("output -> format %d, type %d\n", output_format, output_type);
  LOG_DEBUG("          dim4 %d, displace %d\n", output_shape[0],
            output_shape_displace[0]);
  LOG_DEBUG("          dim3 %d, displace %d\n", output_shape[1],
            output_shape_displace[1]);
  LOG_DEBUG("          dim2 %d, displace %d\n", output_shape[2],
            output_shape_displace[2]);
  LOG_DEBUG("          dim1 %d, displace %d\n", output_shape[3],
            output_shape_displace[3]);

  output_ztensor.transformed_desc = malloc(sizeof(zdnn_tensor_desc));
  init_transformed_desc(
      output_format == ZDNN_FORMAT_4DFEATURE ? ZDNN_NHWC : ZDNN_HWCK,
      output_type, output_format, output_ztensor.transformed_desc,
      output_shape[0] + output_shape_displace[0],
      output_shape[1] + output_shape_displace[1],
      output_shape[2] + output_shape_displace[2],
      output_shape[3] + output_shape_displace[3]);

  if (function_code == NNPA_MATMUL_OP)
    status = verify_matmul_op_tensors(&input_ztensor[0], &input_ztensor[1],
                                      &input_ztensor[2], &output_ztensor);
  else if (function_code == NNPA_MATMUL_OP_BCAST23) {
    status =
        verify_matmul_bcast_op_tensors(&input_ztensor[0], &input_ztensor[1],
                                       &input_ztensor[2], &output_ztensor);
  } else {
    TEST_FAIL_MESSAGE("unknown mode");
  }

  TEST_ASSERT_MESSAGE_FORMATTED(exp_status == status,
                                "Expected status = %08x, actual status = %08x",
                                exp_status, status);

  for (int i = 0; i < MATMUL_NUM_INPUTS; i++) {
    free(input_ztensor[i].transformed_desc);
  }

  free(output_ztensor.transformed_desc);
}

void test_matmul_third(
    int32_t input_shape_displace_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS],
    zdnn_data_formats *input_format_lst, zdnn_data_types *input_type_lst,
    int32_t *output_shape_displace, zdnn_data_formats output_format,
    zdnn_data_types output_type, zdnn_status exp_status) {

  uint32_t matmul_op_first_shape[ZDNN_MAX_DIMS] = {4, 1, 16, 8};
  uint32_t matmul_op_second_shape[ZDNN_MAX_DIMS] = {4, 1, 8, 4};
  uint32_t matmul_op_third_shape[ZDNN_MAX_DIMS] = {4, 1, 1, 4};

  // concatenate the 1D arrays into 2D input for test_matmul()
  uint32_t input_shape_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS];
  memcpy(input_shape_lst[0], matmul_op_first_shape,
         sizeof(uint32_t) * ZDNN_MAX_DIMS);
  memcpy(input_shape_lst[1], matmul_op_second_shape,
         sizeof(uint32_t) * ZDNN_MAX_DIMS);
  memcpy(input_shape_lst[2], matmul_op_third_shape,
         sizeof(uint32_t) * ZDNN_MAX_DIMS);

  uint32_t matmul_op_result_shape[ZDNN_MAX_DIMS] = {4, 1, 16, 4};

  test_matmul(NNPA_MATMUL_OP, input_shape_lst, input_shape_displace_lst,
              input_format_lst, input_type_lst, matmul_op_result_shape,
              output_shape_displace, output_format, output_type, exp_status);
}

void test_matmul_bcast_op(
    int32_t input_shape_displace_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS],
    zdnn_data_formats *input_format_lst, zdnn_data_types *input_type_lst,
    int32_t *output_shape_displace, zdnn_data_formats output_format,
    zdnn_data_types output_type, zdnn_status exp_status) {

  uint32_t feature = 32, batch = 4, spad_x4 = 256, timestep = 4;

  uint32_t input_shape[ZDNN_MAX_DIMS] = {timestep, 1, batch, feature};
  uint32_t weights_shape[ZDNN_MAX_DIMS] = {1, 1, feature, spad_x4};
  uint32_t bias_shape[ZDNN_MAX_DIMS] = {1, 1, 1, spad_x4};

  // concatenate the 1D arrays into 2D input for test_matmul()
  uint32_t input_shape_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS];
  memcpy(input_shape_lst[0], input_shape, sizeof(uint32_t) * ZDNN_MAX_DIMS);
  memcpy(input_shape_lst[1], weights_shape, sizeof(uint32_t) * ZDNN_MAX_DIMS);
  memcpy(input_shape_lst[2], bias_shape, sizeof(uint32_t) * ZDNN_MAX_DIMS);

  uint32_t fused_shape[ZDNN_MAX_DIMS] = {timestep, 1, batch, spad_x4};

  test_matmul(NNPA_MATMUL_OP_BCAST23, input_shape_lst, input_shape_displace_lst,
              input_format_lst, input_type_lst, fused_shape,
              output_shape_displace, output_format, output_type, exp_status);
}

/*
 * Test verification of valid matmul third tensors.
 * All tensors will be built with acceptable properties.
 */
void verify_matmul_op_pass() {
  int32_t input_shape_displace_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  test_matmul_third(input_shape_displace_lst, input_format_lst, input_type_lst,
                    output_shape_displace, output_format, output_type, ZDNN_OK);
}

/*
 * Test verification of failed matmul op output shape.
 * All input tensors will have acceptable descriptors.
 * Output will have invalid number in i-th dimension.
 */
void verify_matmul_op_fail_output_shape() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
    int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
    output_shape_displace[i] = 1;
    test_matmul_third(input_shape_displace_lst, input_format_lst,
                      input_type_lst, output_shape_displace, output_format,
                      output_type, ZDNN_INVALID_SHAPE);
  }
}

/*
 * Test verification of failed matmul op third input shape.
 * Output will have valid descriptor.
 * Input j will have a bad i-th dimension.
 */
void verify_matmul_op_fail_input_shape() {
  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16,
                                      ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int j = 0; j < MATMUL_NUM_INPUTS; j++) {
    for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
      int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
      input_shape_displace_lst[j][i] = 1;
      test_matmul_third(input_shape_displace_lst, input_format_lst,
                        input_type_lst, output_shape_displace, output_format,
                        output_type, ZDNN_INVALID_SHAPE);
    }
  }
}

/*
 * Test verification of failed matmul op output format.
 * All input tensors will have acceptable descriptors.
 * Output will have mismatched format.
 */
void verify_matmul_op_fail_output_format() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DKERNEL;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  test_matmul_third(input_shape_displace_lst, input_format_lst, input_type_lst,
                    output_shape_displace, output_format, output_type,
                    ZDNN_INVALID_FORMAT);
}

/*
 * Test verification of failed matmul op third input format.
 * Output will have valid descriptor.
 * Input i will have a different format.
 */
void verify_matmul_op_fail_input_format() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int i = 0; i < MATMUL_NUM_INPUTS; i++) {
    zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
        ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
    input_format_lst[i] = ZDNN_FORMAT_4DKERNEL;

    test_matmul_third(input_shape_displace_lst, input_format_lst,
                      input_type_lst, output_shape_displace, output_format,
                      output_type, ZDNN_INVALID_FORMAT);
  }
}

/*
 * Test verification of failed matmul op output type.
 * All input tensors will have acceptable descriptors.
 * Output will have mismatched type.
 */
void verify_matmul_op_fail_output_type() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = FP32;

  test_matmul_third(input_shape_displace_lst, input_format_lst, input_type_lst,
                    output_shape_displace, output_format, output_type,
                    ZDNN_INVALID_TYPE);
}

/*
 * Test verification of failed matmul third input type.
 * Output will have valid descriptor.
 * Input i will have a different type.
 */
void verify_matmul_op_fail_input_type() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int i = 0; i < MATMUL_NUM_INPUTS; i++) {
    zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
        ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};
    input_type_lst[i] = FP32;

    test_matmul_third(input_shape_displace_lst, input_format_lst,
                      input_type_lst, output_shape_displace, output_format,
                      output_type, ZDNN_INVALID_TYPE);
  }
}

/*
 * Test verification of valid matmul bcast op tensors.
 * All tensors will be built with acceptable properties.
 */
void verify_matmul_bcast_op_pass() {
  int32_t input_shape_displace_lst[MATMUL_NUM_INPUTS][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                       input_type_lst, output_shape_displace, output_format,
                       output_type, ZDNN_OK);
}

/*
 * Test verification of failed matmul bcast op output shape.
 * All input tensors will have acceptable descriptors.
 * Output will have invalid number in i-th dimension.
 */
void verify_matmul_bcast_op_fail_output_shape() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
    int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
    output_shape_displace[i] = 1;
    test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                         input_type_lst, output_shape_displace, output_format,
                         output_type, ZDNN_INVALID_SHAPE);
  }
}

/*
 * Test verification of failed matmul bcast op input shape.
 * Output will have valid descriptor.
 * Input j will have a bad i-th dimension.
 */
void verify_matmul_bcast_op_fail_input_shape() {
  zdnn_data_formats input_format_lst[] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[] = {ZDNN_DLFLOAT16, ZDNN_DLFLOAT16,
                                      ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int j = 0; j < MATMUL_NUM_INPUTS; j++) {
    for (int i = 0; i < ZDNN_MAX_DIMS; i++) {
      int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
          {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
      input_shape_displace_lst[j][i] = 1;
      test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                           input_type_lst, output_shape_displace, output_format,
                           output_type, ZDNN_INVALID_SHAPE);
    }
  }
}

/*
 * Test verification of failed matmul bcast op input format.
 * All input/output tensors will have acceptable descriptors, except
 * input2 will have mismatched format.
 */
void verify_matmul_bcast_op_fail_input_format() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DKERNEL, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                       input_type_lst, output_shape_displace, output_format,
                       output_type, ZDNN_INVALID_FORMAT);
}

/*
 * Test verification of failed matmul bcast op output format.
 * All input tensors will have acceptable descriptors.
 * Output will have mismatched format.
 */
void verify_matmul_bcast_op_fail_output_format() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DKERNEL;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                       input_type_lst, output_shape_displace, output_format,
                       output_type, ZDNN_INVALID_FORMAT);
}

/*
 * Test verification of failed matmul bcast op output type.
 * All input tensors will have acceptable descriptors.
 * Output will have mismatched type.
 */
void verify_matmul_bcast_op_fail_output_type() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};
  zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = FP32;

  test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                       input_type_lst, output_shape_displace, output_format,
                       output_type, ZDNN_INVALID_TYPE);
}

/*
 * Test verification of failed matmul bcast op input type.
 * Output will have valid descriptor.
 * Input i will have a different type.
 */
void verify_matmul_bcast_op_fail_input_type() {
  int32_t input_shape_displace_lst[][ZDNN_MAX_DIMS] = {
      {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  zdnn_data_formats input_format_lst[MATMUL_NUM_INPUTS] = {
      ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE, ZDNN_FORMAT_4DFEATURE};

  int32_t output_shape_displace[ZDNN_MAX_DIMS] = {0, 0, 0, 0};
  zdnn_data_formats output_format = ZDNN_FORMAT_4DFEATURE;
  zdnn_data_types output_type = ZDNN_DLFLOAT16;

  for (int i = 0; i < MATMUL_NUM_INPUTS; i++) {
    zdnn_data_types input_type_lst[MATMUL_NUM_INPUTS] = {
        ZDNN_DLFLOAT16, ZDNN_DLFLOAT16, ZDNN_DLFLOAT16};
    input_type_lst[i] = FP32;

    test_matmul_bcast_op(input_shape_displace_lst, input_format_lst,
                         input_type_lst, output_shape_displace, output_format,
                         output_type, ZDNN_INVALID_TYPE);
  }
}

/// Common test routine for batchnorm tensors
///
/// \param[in] sbtg_input_b_dim_idx
///                   which dimension (4, 3, 2 or 1) of scale tensor shape to
///                   sabotage, set to 0 if nothing to sabotage
/// \param[in] sbtg_input_b_val   scale tensor sabotage value
/// \param[in] sbtg_input_c_dim_idx
///                   which dimension (4, 3, 2 or 1) of bias tensor shape to
///                   sabotage, set to 0 if nothing to sabotage
/// \param[in] sbtg_input_c_val   bias tensor sabotage value
/// \param[in] exp_status     Expected status
///
void test_batchnorm(uint8_t sbtg_input_b_dim_idx, uint32_t sbtg_input_b_val,
                    int8_t sbtg_input_c_dim_idx, uint32_t sbtg_input_c_val,
                    zdnn_status exp_status) {

  zdnn_tensor_desc tfrmd_desc_input_a, tfrmd_desc_input_b, tfrmd_desc_input_c,
      tfrmd_desc_output;

  zdnn_ztensor input_a, input_b, input_c, output;

  input_a.transformed_desc = &tfrmd_desc_input_a;
  input_b.transformed_desc = &tfrmd_desc_input_b;
  input_c.transformed_desc = &tfrmd_desc_input_c;
  output.transformed_desc = &tfrmd_desc_output;

  uint32_t input_a_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t input_b_shape[ZDNN_MAX_DIMS] = {1, 1, 1, 4};
  uint32_t input_c_shape[ZDNN_MAX_DIMS] = {1, 1, 1, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  zdnn_status status;

  // e.g., sabotage dim_idx = 4 -> modify shape[0]
  //       sabotage dim_idx = 1 -> modify shape[3]

  if (sbtg_input_b_dim_idx != 0) {
    input_b_shape[ZDNN_MAX_DIMS - sbtg_input_b_dim_idx] = sbtg_input_b_val;
  }

  if (sbtg_input_c_dim_idx != 0) {
    input_c_shape[ZDNN_MAX_DIMS - sbtg_input_c_dim_idx] = sbtg_input_c_val;
  }

  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &tfrmd_desc_input_a, input_a_shape[0], input_a_shape[1],
                        input_a_shape[2], input_a_shape[3]);

  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &tfrmd_desc_input_b, input_b_shape[0], input_b_shape[1],
                        input_b_shape[2], input_b_shape[3]);

  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &tfrmd_desc_input_c, input_c_shape[0], input_c_shape[1],
                        input_c_shape[2], input_c_shape[3]);

  // The output is a 4D tensor of same shape, format, and data type as the
  // input
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &tfrmd_desc_output, output_shape[0], output_shape[1],
                        output_shape[2], output_shape[3]);

  status = verify_batchnorm_tensors(&input_a, &input_b, &input_c, &output);

  TEST_ASSERT_MESSAGE_FORMATTED(exp_status == status,
                                "Expected status = %08x, actual status = %08x",
                                exp_status, status);
}

/*
 * Simple test of verifying default inputs and output.
 */
void batchnorm_verify_pass() { test_batchnorm(0, 0, 0, 0, ZDNN_OK); }

/*
 * Test that expects error due to dimension-2 of scale tensor is not 1
 */
void batchnorm_verify_input_b_bad_dim2_fail() {
  test_batchnorm(2, 2, 0, 0, ZDNN_INVALID_SHAPE);
}

/*
 * Test that expects error due to dimension-1 of scale tensor is not the same as
 * the other tensors
 */
void batchnorm_verify_input_b_bad_dim1_fail() {
  test_batchnorm(1, 3, 0, 0, ZDNN_INVALID_SHAPE);
}

/*
 * Test that expects error due to dimension-2 of bias tensor is not 1
 */
void batchnorm_verify_input_c_bad_dim2_fail() {
  test_batchnorm(0, 0, 2, 2, ZDNN_INVALID_SHAPE);
}

/*
 * Test that expects error due to dimension-1 of bias tensor is not the same as
 * the other tensors
 */
void batchnorm_verify_input_c_bad_dim1_fail() {
  test_batchnorm(0, 0, 1, 3, ZDNN_INVALID_SHAPE);
}

/// Common test routine for relu tensors
///
/// \param[in] input_shape    Pointer to input dim array
/// \param[in] input_format   Input format
/// \param[in] input_type     Input type
/// \param[in] output_shape   Pointer to output dim array
/// \param[in] output_format  Output format
/// \param[in] output_type    Output type
/// \param[in] exp_status     Expected status
/// \param[in] error_msg      Error message to prepend to the standard error
///                           message
///
void test_relu(uint32_t input_shape[], zdnn_data_formats input_format,
               zdnn_data_types input_type, uint32_t output_shape[],
               zdnn_data_formats output_format, zdnn_data_types output_type,
               zdnn_status exp_status, char *error_msg) {
  zdnn_status status = ZDNN_OK;

  zdnn_ztensor input, output;

  zdnn_tensor_desc tfrmd_desc_input, tfrmd_desc_output;

  input.transformed_desc = &tfrmd_desc_input;
  output.transformed_desc = &tfrmd_desc_output;

  init_transformed_desc(ZDNN_NHWC, input_type, input_format,
                        input.transformed_desc, input_shape[0], input_shape[1],
                        input_shape[2], input_shape[3]);

  uint32_t clipping_value = 0;

  init_transformed_desc(ZDNN_NHWC, output_type, output_format,
                        output.transformed_desc, output_shape[0],
                        output_shape[1], output_shape[2], output_shape[3]);

  status = verify_relu_tensors(&input, clipping_value, &output);

  TEST_ASSERT_MESSAGE_FORMATTED(
      exp_status == status, "%s  Expected status = %08x, actual status = %08x",
      error_msg, exp_status, status);
}

void relu_verify_pass() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_relu(input_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16, output_shape,
            ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16, ZDNN_OK,
            "The output and the input tensor is different.");
}

void relu_verify_fail_shape() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 3};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_relu(input_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16, output_shape,
            ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16, ZDNN_INVALID_SHAPE,
            "Failed to fail on different shapes.");
}

void relu_verify_fail_format() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_relu(input_shape, ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16, output_shape,
            ZDNN_FORMAT_4DKERNEL, ZDNN_DLFLOAT16, ZDNN_INVALID_FORMAT,
            "Failed to fail on different formats.");
}

void relu_verify_fail_dtype() {
  uint32_t input_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  uint32_t output_shape[ZDNN_MAX_DIMS] = {1, 1, 2, 4};
  test_relu(input_shape, ZDNN_FORMAT_4DFEATURE, FP32, output_shape,
            ZDNN_FORMAT_4DFEATURE, ZDNN_DLFLOAT16, ZDNN_INVALID_TYPE,
            "Failed to fail on different types.");
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(verify_ztensor_format);

  RUN_TEST(verify_1input_pass);
  RUN_TEST(verify_2input_pass);
  RUN_TEST(verify_3input_pass);
  RUN_TEST(verify_input2_fail_shape);
  RUN_TEST(verify_input3_fail_shape);
  RUN_TEST(verify_input2_fail_format);
  RUN_TEST(verify_input3_fail_format);
  RUN_TEST(verify_input2_fail_dtype);
  RUN_TEST(verify_input3_fail_dtype);
  RUN_TEST(verify_output_fail_shape);
  RUN_TEST(verify_output_fail_format);
  RUN_TEST(verify_output_fail_dtype);

  RUN_TEST(verify_matmul_op_pass);
  RUN_TEST(verify_matmul_op_fail_output_shape);
  RUN_TEST(verify_matmul_op_fail_input_shape);

  RUN_TEST(verify_matmul_op_fail_output_format);
  RUN_TEST(verify_matmul_op_fail_input_format);
  RUN_TEST(verify_matmul_op_fail_output_type);
  RUN_TEST(verify_matmul_op_fail_input_type);

  RUN_TEST(verify_matmul_bcast_op_pass);
  RUN_TEST(verify_matmul_bcast_op_fail_output_shape);
  RUN_TEST(verify_matmul_bcast_op_fail_input_shape);
  RUN_TEST(verify_matmul_bcast_op_fail_output_format);
  RUN_TEST(verify_matmul_bcast_op_fail_input_format);
  RUN_TEST(verify_matmul_bcast_op_fail_output_type);
  RUN_TEST(verify_matmul_bcast_op_fail_input_type);

  RUN_TEST(batchnorm_verify_pass);
  RUN_TEST(batchnorm_verify_input_b_bad_dim2_fail);
  RUN_TEST(batchnorm_verify_input_b_bad_dim1_fail);
  RUN_TEST(batchnorm_verify_input_c_bad_dim2_fail);
  RUN_TEST(batchnorm_verify_input_c_bad_dim1_fail);

  RUN_TEST(relu_verify_pass);
  RUN_TEST(relu_verify_fail_shape);
  RUN_TEST(relu_verify_fail_format);
  RUN_TEST(relu_verify_fail_dtype);

  return UNITY_END();
}
