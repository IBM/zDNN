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
  VERIFY_HW_ENV;
}

void tearDown(void) {}

#define BAD_FORMAT 255
#define BAD_TYPE 255

typedef enum which_tensor {
  NONE,
  FUSED,
  BIAS,
  CELLSTATE,
  OUTPUT,
  OUTPUT2
} which_tensor;

typedef struct rnn_ztensors_struct {
  zdnn_ztensor *timestep_fused;
  zdnn_ztensor *bias_add;
  zdnn_ztensor *timestep_c;
  zdnn_ztensor *timestep_output1;
  zdnn_ztensor *timestep_output2;
} rnn_ztensors_struct;

void create_ztensors(uint8_t function_code, uint32_t batch,
                     uint32_t hidden_size, rnn_ztensors_struct *rnn_ztens) {

  zdnn_data_layouts layout = ZDNN_NHWC;
  zdnn_data_types dtype = FP32;
  uint8_t num_gates = get_func_code_num_gates(function_code);

  if (num_gates == 0) {
    TEST_FAIL_MESSAGE_FORMATTED(
        "%d is not a valid NNPA function code for testing.", function_code);
  }

  // Declare as CORRECT dimension requirements for NNPA_LSTMACT
  uint32_t timestep_fused_shape[] = {num_gates, 1, batch, hidden_size},
           bias_add_shape[] = {num_gates, 1, batch, hidden_size},
           timestep_c_shape[] = {1, 1, batch, hidden_size},
           timestep_output_shape[] = {1, 1, batch, hidden_size};

  // The first input is the fuzed_ztensor split as a timestep
  rnn_ztens->timestep_fused = alloc_ztensor_with_values(
      timestep_fused_shape, layout, dtype, NO_CONCAT, true, ZERO_ARRAY);

  // The second input is the bias_add_ztensor that would be the result of
  // the bias_add call within NNPA_LSTMACT function.
  rnn_ztens->bias_add = alloc_ztensor_with_values(bias_add_shape, layout, dtype,
                                                  NO_CONCAT, true, ZERO_ARRAY);

  // The third input is the cell state ztensor (only used in NNPA_LSTMACT)
  rnn_ztens->timestep_c = alloc_ztensor_with_values(
      timestep_c_shape, layout, dtype, NO_CONCAT, true, ZERO_ARRAY);

  // The output is the result as output_ztensor1
  rnn_ztens->timestep_output1 = alloc_ztensor_with_values(
      timestep_output_shape, layout, dtype, NO_CONCAT, true, ZERO_ARRAY);

  // The output is the result as output_ztensor2
  rnn_ztens->timestep_output2 = alloc_ztensor_with_values(
      timestep_output_shape, layout, dtype, NO_CONCAT, true, ZERO_ARRAY);
}

void set_dim(zdnn_tensor_desc *desc, uint8_t dim_name_int, uint32_t value) {
  switch (dim_name_int) {
  case (1):
    desc->dim1 = value;
    break;
  case (2):
    desc->dim2 = value;
    break;
  case (3):
    desc->dim3 = value;
    break;
  case (4):
    desc->dim4 = value;
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("%d is not a valid dim_name_int to set.",
                                dim_name_int);
    break;
  }
}

// Verify return status by sabotaging the ztensor dimension
void verify_lstm_gru_act_shape(uint8_t function_code, uint32_t batch,
                               uint32_t hidden_size, which_tensor tensor,
                               uint8_t dim_name_int, uint32_t dim_val,
                               zdnn_status exp_status, char *description) {

  // Create the test tensors
  rnn_ztensors_struct rnn_ztens;
  create_ztensors(function_code, batch, hidden_size, &rnn_ztens);

  // Sabotage the specified ztensor dimension
  switch (tensor) {
  case (NONE):
    // Don't want to sabotage any dimension (to test positive case)
    break;
  case (FUSED):
    set_dim(rnn_ztens.timestep_fused->transformed_desc, dim_name_int, dim_val);
    break;
  case (BIAS):
    set_dim(rnn_ztens.bias_add->transformed_desc, dim_name_int, dim_val);
    break;
  case (CELLSTATE):
    set_dim(rnn_ztens.timestep_c->transformed_desc, dim_name_int, dim_val);
    break;
  case (OUTPUT):
    set_dim(rnn_ztens.timestep_output1->transformed_desc, dim_name_int,
            dim_val);
    break;
  case (OUTPUT2):
    set_dim(rnn_ztens.timestep_output2->transformed_desc, dim_name_int,
            dim_val);
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("%d is not a valid which_tensor for testing.",
                                tensor);
    break;
  }

  zdnn_status actual_status = verify_lstm_or_gru_act_tensors(
      function_code, rnn_ztens.timestep_fused, rnn_ztens.bias_add,
      rnn_ztens.timestep_c, rnn_ztens.timestep_output1,
      rnn_ztens.timestep_output2);

  if (actual_status != exp_status) {
    TEST_FAIL_MESSAGE_FORMATTED(
        "%s: Actual status return (%08x) does not match expected (%08x).",
        description, actual_status, exp_status);
  }

  // Cleanup
  free_ztensor_buffers(5, rnn_ztens.timestep_fused, rnn_ztens.bias_add,
                       rnn_ztens.timestep_c, rnn_ztens.timestep_output1,
                       rnn_ztens.timestep_output2);
}

// Verify return status by sabotaging the ztensor format
void verify_lstm_gru_act_format(uint8_t function_code, uint32_t batch,
                                uint32_t hidden_size, which_tensor tensor,
                                zdnn_data_formats format,
                                zdnn_status exp_status, char *description) {

  // Create the test tensors
  rnn_ztensors_struct rnn_ztens;
  create_ztensors(function_code, batch, hidden_size, &rnn_ztens);

  // Sabotage the specified ztensor format
  switch (tensor) {
  case (NONE):
    // Don't want to sabotage any format (to test positive case)
    break;
  case (FUSED):
    rnn_ztens.timestep_fused->transformed_desc->format = format;
    break;
  case (BIAS):
    rnn_ztens.bias_add->transformed_desc->format = format;
    break;
  case (CELLSTATE):
    rnn_ztens.timestep_c->transformed_desc->format = format;
    break;
  case (OUTPUT):
    rnn_ztens.timestep_output1->transformed_desc->format = format;
    break;
  case (OUTPUT2):
    rnn_ztens.timestep_output2->transformed_desc->format = format;
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("%d is not a valid which_tensor for testing.",
                                tensor);
    break;
  }

  zdnn_status actual_status = verify_lstm_or_gru_act_tensors(
      function_code, rnn_ztens.timestep_fused, rnn_ztens.bias_add,
      rnn_ztens.timestep_c, rnn_ztens.timestep_output1,
      rnn_ztens.timestep_output2);

  if (actual_status != exp_status) {
    TEST_FAIL_MESSAGE_FORMATTED(
        "%s: Actual status return (%08x) does not match expected (%08x).",
        description, actual_status, exp_status);
  }

  // Cleanup
  free_ztensor_buffers(5, rnn_ztens.timestep_fused, rnn_ztens.bias_add,
                       rnn_ztens.timestep_c, rnn_ztens.timestep_output1,
                       rnn_ztens.timestep_output2);
}

// Verify return status by sabotaging the ztensor data type
void verify_lstm_gru_act_type(uint8_t function_code, uint32_t batch,
                              uint32_t hidden_size, which_tensor tensor,
                              zdnn_data_types type, zdnn_status exp_status,
                              char *description) {

  // Create the test tensors
  rnn_ztensors_struct rnn_ztens;
  create_ztensors(function_code, batch, hidden_size, &rnn_ztens);

  // Sabotage the specified ztensor data type
  switch (tensor) {
  case (NONE):
    // Don't want to sabotage any format (to test positive case)
    break;
  case (FUSED):
    rnn_ztens.timestep_fused->transformed_desc->type = type;
    break;
  case (BIAS):
    rnn_ztens.bias_add->transformed_desc->type = type;
    break;
  case (CELLSTATE):
    rnn_ztens.timestep_c->transformed_desc->type = type;
    break;
  case (OUTPUT):
    rnn_ztens.timestep_output1->transformed_desc->type = type;
    break;
  case (OUTPUT2):
    rnn_ztens.timestep_output2->transformed_desc->type = type;
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED("%d is not a valid which_tensor for testing.",
                                tensor);
    break;
  }

  zdnn_status actual_status = verify_lstm_or_gru_act_tensors(
      function_code, rnn_ztens.timestep_fused, rnn_ztens.bias_add,
      rnn_ztens.timestep_c, rnn_ztens.timestep_output1,
      rnn_ztens.timestep_output2);

  if (actual_status != exp_status) {
    TEST_FAIL_MESSAGE_FORMATTED(
        "%s: Actual status return (%08x) does not match expected (%08x).",
        description, actual_status, exp_status);
  }

  // Cleanup
  free_ztensor_buffers(5, rnn_ztens.timestep_fused, rnn_ztens.bias_add,
                       rnn_ztens.timestep_c, rnn_ztens.timestep_output1,
                       rnn_ztens.timestep_output2);
}

/*
 * Test verification of valid lstm activation tensors.
 * All tensors will be built with acceptable properties.
 */
void verify_lstm_act_pass() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect no known error, no bad dims will be set
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, NONE, 0, 0,
                            ZDNN_OK, "verify_lstm_act_pass");
}

/*
 * Test verification of failed lstm output shape.
 * Correct shape is (1, 1, batch, hidden_size)
 * All input tensors will have acceptable descriptors.
 */
void verify_lstm_act_fail_output_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output_ztensor dimension 4 (timestep) is not 2
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT, 4, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim4");

  // Expect failure when output_ztensor dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim3");

  // Expect failure when output_ztensor dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim2");

  // Expect failure when output_ztensor dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim1");
}

/*
 * Test verification of failed lstm output 2 shape.
 * Correct shape is (1, 1, batch, hidden_size)
 * All input tensors will have acceptable descriptors.
 */
void verify_lstm_act_fail_output2_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output_ztensor dimension 4 (timestep) is not 2
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT2, 4, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim4");

  // Expect failure when output_ztensor dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT2, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim3");

  // Expect failure when output_ztensor dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT2, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim2");

  // Expect failure when output_ztensor dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, OUTPUT2, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_output_shape_dim1");
}

/*
 * Test verification of failed lstm input shape.
 * Correct shape is (4, 1, batch, hidden_size)
 * All input tensors except fused will have acceptable descriptors.
 */
void verify_lstm_act_fail_fused_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when fused dimension 4 (timestep) is not 4
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, FUSED, 4, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_fused_shape_dim4");

  // Expect failure when fused dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, FUSED, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_fused_shape_dim3");

  // Expect failure when fused dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, FUSED, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_fused_shape_dim2");

  // Expect failure when fused dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, FUSED, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_fused_shape_dim1");
}

/*
 * Test verification of failed lstm input shape.
 * Correct shape is (4, 1, batch, hidden_size)
 * All input tensors except bias will have acceptable descriptors.
 */
void verify_lstm_act_fail_bias_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when bias dimension 4 is not 4
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, BIAS, 4, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_bias_shape_dim4");

  // Expect failure when bias dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, BIAS, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_bias_shape_dim3");

  // Expect failure when bias dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, BIAS, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_bias_shape_dim2");

  // Expect failure when bias dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, BIAS, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_bias_shape_dim1");
}

/*
 * Test verification of failed lstm input shape.
 * Correct shape is (1, 1, batch, hidden_size)
 * All input tensors except cell-state will have acceptable descriptors.
 */
void verify_lstm_act_fail_cellstate_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when cellstate dimension 4 is not 1
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, CELLSTATE, 4, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_cellstate_shape_dim4");

  // Expect failure when cellstate dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, CELLSTATE, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_cellstate_shape_dim3");

  // Expect failure when cellstate dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, CELLSTATE, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_cellstate_shape_dim2");

  // Expect failure when cellstate dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_LSTMACT, batch, hidden_size, CELLSTATE, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_lstm_act_fail_cellstate_shape_dim1");
}

/*
 * Test verification of failed lstm output format.
 */
void verify_lstm_act_fail_format() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output format does not match fused's
  verify_lstm_gru_act_format(NNPA_LSTMACT, batch, hidden_size, OUTPUT,
                             BAD_FORMAT, ZDNN_INVALID_FORMAT,
                             "verify_lstm_act_fail_output_format");

  // Expect failure when output2 format does not match fused's
  verify_lstm_gru_act_format(NNPA_LSTMACT, batch, hidden_size, OUTPUT2,
                             BAD_FORMAT, ZDNN_INVALID_FORMAT,
                             "verify_lstm_act_fail_output2_format");

  // Expect failure when fused/bias/cells-state format does not match output's
  verify_lstm_gru_act_format(NNPA_LSTMACT, batch, hidden_size, FUSED,
                             BAD_FORMAT, ZDNN_INVALID_FORMAT,
                             "verify_lstm_act_fail_fused_format");
  verify_lstm_gru_act_format(NNPA_LSTMACT, batch, hidden_size, BIAS, BAD_FORMAT,
                             ZDNN_INVALID_FORMAT,
                             "verify_lstm_act_fail_bias_format");
  verify_lstm_gru_act_format(NNPA_LSTMACT, batch, hidden_size, CELLSTATE,
                             BAD_FORMAT, ZDNN_INVALID_FORMAT,
                             "verify_lstm_act_fail_cellstate_format");
} /*
   * Test verification of failed lstm output type.
   */
void verify_lstm_act_fail_type() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output dtype does not match fused's
  verify_lstm_gru_act_type(NNPA_LSTMACT, batch, hidden_size, OUTPUT, BAD_TYPE,
                           ZDNN_INVALID_TYPE,
                           "verify_lstm_act_fail_output_type");

  // Expect failure when output2 dtype does not match fused's
  verify_lstm_gru_act_type(NNPA_LSTMACT, batch, hidden_size, OUTPUT2, BAD_TYPE,
                           ZDNN_INVALID_TYPE,
                           "verify_lstm_act_fail_output2_type");

  // Expect failure when fused/bias/cells-state dtype does not match
  // output's
  verify_lstm_gru_act_type(NNPA_LSTMACT, batch, hidden_size, FUSED, BAD_TYPE,
                           ZDNN_INVALID_TYPE,
                           "verify_lstm_act_fail_fused_type");
  verify_lstm_gru_act_type(NNPA_LSTMACT, batch, hidden_size, BIAS, BAD_TYPE,
                           ZDNN_INVALID_TYPE, "verify_lstm_act_fail_bias_type");
  verify_lstm_gru_act_type(NNPA_LSTMACT, batch, hidden_size, CELLSTATE,
                           BAD_TYPE, ZDNN_INVALID_TYPE,
                           "verify_lstm_act_fail_cellstate_type");
}

/*
 * Test verification of valid gru activation tensors.
 * All tensors will be built with acceptable properties.
 */
void verify_gru_act_pass() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect no error, no dim will be set by verify_lstm_gru_act_shape()
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, NONE, 0, 0,
                            ZDNN_OK, "verify_gru_act_pass");
}

/*
 * Test verification of failed gru output shape.
 * Correct shape is (1, 1, batch, hidden_size)
 * All input tensors will have acceptable descriptors.
 */
void verify_gru_act_fail_output_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output_ztensor dimension 4 is not 1
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT, 4, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_output_shape_dim4");

  // Expect failure when output_ztensor dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_output_shape_dim3");

  // Expect failure when output_ztensor dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_output_shape_dim2");

  // Expect failure when output_ztensor dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_output_shape_dim1");
}

/*
 * Test verification of passing gru output2 shape, as it is ignored.
 * Correct shape is (1, 1, batch, hidden_size)
 * All input tensors will have acceptable descriptors.
 */
void verify_gru_act_pass_output2_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect success when output_ztensor dimension 4 (timestep) is not 1
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT2, 4, 3,
                            ZDNN_OK, "verify_gru_act_pass_output2_shape_dim4");

  // Expect success when output_ztensor dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT2, 3, 3,
                            ZDNN_OK, "verify_gru_act_pass_output2_shape_dim3");

  // Expect success when output_ztensor dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT2, 2,
                            batch + 1, ZDNN_OK,
                            "verify_gru_act_pass_output2_shape_dim2");

  // Expect success when output_ztensor dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, OUTPUT2, 1,
                            hidden_size + 1, ZDNN_OK,
                            "verify_gru_act_pass_output2_shape_dim_1");
}

/*
 * Test verification of failed gru input shape.
 * Correct shape is (3, 1, batch, hidden_size)
 * All input tensors except fused will have acceptable descriptors.
 */
void verify_gru_act_fail_fused_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when fused dimension 4 is not 3
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, FUSED, 4, 2,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_fused_shape_dim4");

  // Expect failure when fused dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, FUSED, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_fused_shape_dim3");

  // Expect failure when fused dimension 2 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, FUSED, 2,
                            batch + 1, ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_fused_shape_dim2");

  // Expect failure when fused dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, FUSED, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_fused_shape_dim1");
}

/*
 * Test verification of failed gru input shape.
 * Correct shape is (3, 1, batch, hidden_size)
 * All input tensors except bias will have acceptable descriptors.
 */
void verify_gru_act_fail_bias_shape() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when bias dimension 4 is not 3
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, BIAS, 4, 2,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_bias_shape_dim4");

  // Expect failure when bias dimension 3 is not 1
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, BIAS, 3, 3,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_bias_shape_dim3");

  // Expect failure when bias dimension 2 does not match others
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, BIAS, 2, batch + 1,
                            ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_bias_shape_dim2");

  // Expect failure when bias dimension 1 does not match input
  verify_lstm_gru_act_shape(NNPA_GRUACT, batch, hidden_size, BIAS, 1,
                            hidden_size + 1, ZDNN_INVALID_SHAPE,
                            "verify_gru_act_fail_bias_shape_dim1");
}

/*
 * Test verification of failed gru output format.
 */
void verify_gru_act_fail_format() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output format does not match fused's
  verify_lstm_gru_act_format(NNPA_GRUACT, batch, hidden_size, OUTPUT,
                             BAD_FORMAT, ZDNN_INVALID_FORMAT,
                             "verify_gru_act_fail_output_format");

  // Expect success when output2 format does not match fused's (since it is
  // ignored)
  verify_lstm_gru_act_format(NNPA_GRUACT, batch, hidden_size, OUTPUT2,
                             BAD_FORMAT, ZDNN_OK,
                             "verify_gru_act_fail_output2_format");

  // Expect failure when fused/bias/cells-state format does not match output's
  verify_lstm_gru_act_format(NNPA_GRUACT, batch, hidden_size, FUSED, BAD_FORMAT,
                             ZDNN_INVALID_FORMAT,
                             "verify_gru_act_fail_fused_format");
  verify_lstm_gru_act_format(NNPA_GRUACT, batch, hidden_size, BIAS, BAD_FORMAT,
                             ZDNN_INVALID_FORMAT,
                             "verify_gru_act_fail_bias_format");
}

/*
 * Test verification of failed gru output type.
 */
void verify_gru_act_fail_type() {
  uint32_t batch = 4;
  uint32_t hidden_size = 16;

  // Expect failure when output dtype does not match fused's
  verify_lstm_gru_act_type(NNPA_GRUACT, batch, hidden_size, OUTPUT, BAD_TYPE,
                           ZDNN_INVALID_TYPE,
                           "verify_gru_act_fail_output_type");

  // Expect success when output2 dtype does not match fused's (since it is
  // ignored)
  verify_lstm_gru_act_type(NNPA_GRUACT, batch, hidden_size, OUTPUT2, BAD_TYPE,
                           ZDNN_OK, "verify_gru_act_fail_output2_type");

  // Expect failure when fused/bias/cells-state dtype does not match
  // output's
  verify_lstm_gru_act_type(NNPA_GRUACT, batch, hidden_size, FUSED, BAD_TYPE,
                           ZDNN_INVALID_TYPE, "verify_gru_act_fail_fused_type");
  verify_lstm_gru_act_type(NNPA_GRUACT, batch, hidden_size, BIAS, BAD_TYPE,
                           ZDNN_INVALID_TYPE, "verify_gru_act_fail_bias_type");
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(verify_lstm_act_pass);
  RUN_TEST(verify_lstm_act_fail_output_shape);
  RUN_TEST(verify_lstm_act_fail_output2_shape);
  RUN_TEST(verify_lstm_act_fail_fused_shape);
  RUN_TEST(verify_lstm_act_fail_bias_shape);
  RUN_TEST(verify_lstm_act_fail_cellstate_shape);
  RUN_TEST(verify_lstm_act_fail_format);
  RUN_TEST(verify_lstm_act_fail_type);
  RUN_TEST(verify_gru_act_pass);
  RUN_TEST(verify_gru_act_fail_output_shape);
  RUN_TEST(verify_gru_act_pass_output2_shape);
  RUN_TEST(verify_gru_act_fail_fused_shape);
  RUN_TEST(verify_gru_act_fail_bias_shape);
  RUN_TEST(verify_gru_act_fail_format);
  RUN_TEST(verify_gru_act_fail_type);

  return UNITY_END();
}
