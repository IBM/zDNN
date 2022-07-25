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

#define DEFAULT_NUM_TIMESTEPS 3
#define DEFAULT_NUM_BATCHES 4
#define DEFAULT_NUM_FEATURES 7
#define DEFAULT_NUM_HIDDEN 16

const uint32_t num_batches = DEFAULT_NUM_BATCHES;
const uint32_t num_hidden = DEFAULT_NUM_HIDDEN;

#define MAX_DESC_LEN 256
char msg[MAX_DESC_LEN];
typedef enum tensor_idx {
  FUSED,
  BIAS,
  CELLSTATE,
  OUTPUT,
  OUTPUT2,
  MAX_TENSOR_IDX,
  NONE = MAX_TENSOR_IDX
} tensor_idx;

// roll our own instead of using get_func_code_num_gates() in case that one
// breaks
#define NUM_GATES(f) ((f == NNPA_LSTMACT) ? 4 : 3)

void create_ztensors(uint8_t function_code, zdnn_ztensor **rnn_ztens) {

  zdnn_data_layouts layout = ZDNN_NHWC;
  zdnn_data_types dtype = FP32;
  uint8_t num_gates = NUM_GATES(function_code);

  // baseline dimensions with correct requirements
  uint32_t *shape[MAX_TENSOR_IDX];

  // create ztensors using transformed shape + ZDNN_NHWC to make the code
  // simplier, so that we can loop through them all rather than dealing with
  // different pre-transformed layouts etc.
  shape[FUSED] = (uint32_t[]){num_gates, 1, num_batches, num_hidden};
  shape[BIAS] = (uint32_t[]){num_gates, 1, num_batches, num_hidden};
  shape[CELLSTATE] = (uint32_t[]){1, 1, num_batches, num_hidden};
  shape[OUTPUT] = (uint32_t[]){1, 1, num_batches, num_hidden};
  shape[OUTPUT2] =
      shape[OUTPUT]; // they share the same shape, final timestep only

  // rnn_ztens[FUSED] is the fuzed_ztensor split as a timestep

  // rnn_ztens[BIAS] is the bias_add_ztensor that would be the result of the
  // bias_add call within NNPA_LSTMACT function.

  // rnn_ztens[CELLSTATE] is the cell state ztensor (only used in NNPA_LSTMACT)

  // rnn_ztens[OUTPUT] is the result as output_ztensor1

  // rnn_ztens[OUTPUT2] is the result as output_ztensor2
  for (int i = 0; i < MAX_TENSOR_IDX; i++) {
    rnn_ztens[i] = alloc_ztensor_with_values(shape[i], layout, dtype, NO_CONCAT,
                                             true, ZERO_ARRAY);
  }
}

void set_dim(zdnn_tensor_desc *desc, uint8_t dim_idx, uint32_t value) {
  switch (dim_idx) {
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
    TEST_FAIL_MESSAGE_FORMATTED("%d is not a valid dim_idx to set.", dim_idx);
    break;
  }
}

// Verify return status by sabotaging a ztensor
void verify(uint8_t function_code, tensor_idx idx, bool sabotage_dim,
            uint8_t dim_idx, uint32_t dim_val, bool sabotage_type,
            zdnn_data_types type, bool sabotage_format,
            zdnn_data_formats format, zdnn_status exp_status,
            char *description) {

  // Create the test tensors
  zdnn_ztensor *rnn_ztens[MAX_TENSOR_IDX];
  create_ztensors(function_code, rnn_ztens);

  // Sabotage the dim/format/type of the ztensor specified in idx
  if (idx != NONE) {
    if (sabotage_dim) {
      set_dim(rnn_ztens[idx]->transformed_desc, dim_idx, dim_val);
    }

    if (sabotage_type) {
      rnn_ztens[idx]->transformed_desc->type = type;
    }

    if (sabotage_format) {
      rnn_ztens[idx]->transformed_desc->format = format;
    }
  }

  zdnn_status actual_status = verify_lstm_or_gru_act_tensors(
      function_code, rnn_ztens[FUSED], rnn_ztens[BIAS], rnn_ztens[CELLSTATE],
      rnn_ztens[OUTPUT], rnn_ztens[OUTPUT2]);

  if (actual_status != exp_status) {
    TEST_FAIL_MESSAGE_FORMATTED(
        "%s: Actual status return (%08x) does not match expected (%08x).",
        description, actual_status, exp_status);
  }

  // Cleanup
  for (int i = 0; i < MAX_TENSOR_IDX; i++) {
    free_ztensor_buffers(1, rnn_ztens[i]);
  }
}

// Verify return status by sabotaging the ztensor dimension
void verify_shape(uint8_t function_code, tensor_idx idx, uint8_t dim_idx,
                  uint32_t dim_val, zdnn_status exp_status, char *description) {
  verify(function_code, idx, true, dim_idx, dim_val, false, 0, false, 0,
         exp_status, description);
}

// Verify return status by sabotaging the ztensor data type
void verify_type(uint8_t function_code, tensor_idx idx, zdnn_data_types type,
                 zdnn_status exp_status, char *description) {
  verify(function_code, idx, false, 0, 0, true, type, false, 0, exp_status,
         description);
}

// Verify return status by sabotaging the ztensor format
void verify_format(uint8_t function_code, tensor_idx idx,
                   zdnn_data_formats format, zdnn_status exp_status,
                   char *description) {
  verify(function_code, idx, false, 0, 0, false, 0, true, format, exp_status,
         description);
}

// this macro assume values of NNPA_LSTMACT and NNPA_GRUACT are next to each
// other
#define LOOP_LSTM_AND_GRU(lg)                                                  \
  for (int lg = NNPA_LSTMACT; lg < NNPA_GRUACT; lg++)

#define TEST_DIM_VAL(tensor_idx, dim_idx, val, exp_status)                     \
  snprintf(msg, MAX_DESC_LEN, "%s %s dim%s", __func__,                         \
           act == NNPA_LSTMACT ? "LSTM" : "GRU", #dim_idx);                    \
  verify_shape(act, tensor_idx, dim_idx, val, exp_status, msg);

/*
 * Test verification of valid activation tensors.
 * All tensors will be built with acceptable properties.
 */
void verify_pass() {
  // Expect no known error, no bad dims will be set
  LOOP_LSTM_AND_GRU(act) { TEST_DIM_VAL(NONE, 0, 0, ZDNN_OK); }
}

/*
 * Test verification of failed output shape.
 * Correct shape is (1, 1, num_batches, num_hidden)
 * All input tensors will have acceptable descriptors.
 */
void verify_fail_output_shape() {
  LOOP_LSTM_AND_GRU(act) {

    // Expect failure when output_ztensor dimension 4 (timestep) is not 1
    TEST_DIM_VAL(OUTPUT, 4, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when output_ztensor dimension 3 is not 1
    TEST_DIM_VAL(OUTPUT, 3, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when output_ztensor dimension 2 does not match num_batches
    TEST_DIM_VAL(OUTPUT, 2, num_batches + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when output_ztensor dimension 1 does not match num_hidden
    TEST_DIM_VAL(OUTPUT, 1, num_hidden + 1, ZDNN_INVALID_SHAPE);
  }
}

/*
 * Test verification of failed output2 shape.
 * Correct shape is (1, 1, num_batches, num_hidden)
 * All input tensors will have acceptable descriptors.
 */
void verify_fail_output2_shape() {
  LOOP_LSTM_AND_GRU(act) {

    // Expect failure when output_ztensor dimension 4 (timestep) is not 1
    TEST_DIM_VAL(OUTPUT2, 4, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when output_ztensor dimension 3 is not 1
    TEST_DIM_VAL(OUTPUT2, 3, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when output_ztensor dimension 2 does not match num_batches
    TEST_DIM_VAL(OUTPUT2, 2, num_batches + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when output_ztensor dimension 1 does not match num_hidden
    TEST_DIM_VAL(OUTPUT2, 1, num_hidden + 1, ZDNN_INVALID_SHAPE);
  }
}

/*
 * Test verification of failed fuzed_ztensor shape.
 * Correct shape is (4, 1, num_batches, num_hidden) for LSTM,
 * (3, 1, num_batches, num_hidden) for GRU
 * All input tensors except fused will have acceptable descriptors.
 */
void verify_fail_fused_shape() {
  LOOP_LSTM_AND_GRU(act) {
    uint32_t num_gates = NUM_GATES(act);

    // Expect failure when bias dimension 4 is not 4 (LSTM) or 3 (GRU)
    TEST_DIM_VAL(FUSED, 4, num_gates + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when fused dimension 3 is not 1
    TEST_DIM_VAL(FUSED, 3, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when fused dimension 2 does not match num_batches
    TEST_DIM_VAL(FUSED, 2, num_batches + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when fused dimension 1 does not match num_hidden
    TEST_DIM_VAL(FUSED, 1, num_hidden + 1, ZDNN_INVALID_SHAPE);
  }
}

/*
 * Test verification of failed bias_add_ztensor shape.
 * Correct shape is (4, 1, num_batches, num_hidden) for LSTM,
 * (3, 1, num_batches, num_hidden) for GRU
 * All input tensors except bias will have acceptable descriptors.
 */
void verify_fail_bias_shape() {
  LOOP_LSTM_AND_GRU(act) {
    uint32_t num_gates = NUM_GATES(act);

    // Expect failure when bias dimension 4 is not 4 (LSTM) or 3 (GRU)
    TEST_DIM_VAL(BIAS, 4, num_gates + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when bias dimension 3 is not 1
    TEST_DIM_VAL(BIAS, 3, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when bias dimension 2 does not match input
    TEST_DIM_VAL(BIAS, 2, num_batches + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when bias dimension 1 does not match input
    TEST_DIM_VAL(BIAS, 1, num_hidden + 1, ZDNN_INVALID_SHAPE);
  }
}

/*
 * Test verification of failed cell state ztensor shape.
 * Correct shape is (1, 1, num_batches, num_hidden)
 * All input tensors except cell-state will have acceptable descriptors.
 */
void verify_fail_cellstate_shape() {
  LOOP_LSTM_AND_GRU(act) {
    // Expect failure when cellstate dimension 4 is not 1
    TEST_DIM_VAL(CELLSTATE, 4, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when cellstate dimension 3 is not 1
    TEST_DIM_VAL(CELLSTATE, 3, 2, ZDNN_INVALID_SHAPE);

    // Expect failure when cellstate dimension 2 does not match num_batches
    TEST_DIM_VAL(CELLSTATE, 2, num_batches + 1, ZDNN_INVALID_SHAPE);

    // Expect failure when cellstate dimension 2 does not matchnum_hidden
    TEST_DIM_VAL(CELLSTATE, 1, num_hidden + 1, ZDNN_INVALID_SHAPE);
  }
}

#define TEST_FORMAT(tensor_idx, format, exp_status)                            \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s", __func__,                            \
           act == NNPA_LSTMACT ? "LSTM" : "GRU", #tensor_idx);                 \
  verify_format(act, tensor_idx, format, exp_status, msg);

/*
 * Test verification of failed format.
 */
void verify_fail_format() {
  LOOP_LSTM_AND_GRU(act) {
    for (int i = 0; i < MAX_TENSOR_IDX; i++) {
      TEST_FORMAT(i, BAD_FORMAT, ZDNN_INVALID_FORMAT);
    }
  }
}

#define TEST_TYPE(tensor_idx, type, exp_status)                                \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s", __func__,                            \
           act == NNPA_LSTMACT ? "LSTM" : "GRU", #tensor_idx);                 \
  verify_type(act, tensor_idx, type, exp_status, msg);

/*
 * Test verification of failed type.
 */
void verify_fail_type() {
  LOOP_LSTM_AND_GRU(act) {
    for (int i = 0; i < MAX_TENSOR_IDX; i++) {
      TEST_TYPE(i, BAD_TYPE, ZDNN_INVALID_TYPE);
    }
  }
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(verify_pass);
  RUN_TEST(verify_fail_output_shape);
  RUN_TEST(verify_fail_output2_shape);
  RUN_TEST(verify_fail_fused_shape);
  RUN_TEST(verify_fail_bias_shape);
  RUN_TEST(verify_fail_cellstate_shape);
  RUN_TEST(verify_fail_format);
  RUN_TEST(verify_fail_type);

  return UNITY_END();
}
