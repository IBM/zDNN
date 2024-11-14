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

#include "common_rnn.h"
#include "testsupport.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) { VERIFY_HW_ENV; }

void tearDown(void) {}

#define BAD_FORMAT 255
#define BAD_TYPE 255

#define DEFAULT_NUM_TIMESTEPS 3
#define DEFAULT_NUM_BATCHES 4
#define DEFAULT_NUM_FEATURES 7
#define DEFAULT_NUM_HIDDEN 16

const uint32_t num_timesteps = DEFAULT_NUM_TIMESTEPS;
const uint32_t num_batches = DEFAULT_NUM_BATCHES;
const uint32_t num_features = DEFAULT_NUM_FEATURES;
const uint32_t num_hidden = DEFAULT_NUM_HIDDEN;

#define MAX_DESC_LEN 256
char msg[MAX_DESC_LEN];

typedef enum tensor_idx {
  INPUT,
  H0,
  C0,
  WEIGHTS,
  BIASES,
  HIDDEN_WEIGHTS,
  HIDDEN_BIASES,
  HN_OUTPUT,
  CF_OUTPUT,
  MAX_TENSOR_IDX,
  NONE = MAX_TENSOR_IDX
} tensor_idx;

// roll our own instead of using get_func_code_num_gates() in case that one
// breaks
#define NUM_GATES(f) ((f == NNPA_LSTMACT) ? 4 : 3)

void create_ztensors(uint8_t function_code, uint32_t num_timesteps,
                     uint32_t num_batches, uint32_t num_features,
                     uint32_t num_hidden, uint32_t num_dirs,
                     bool all_timesteps_out, zdnn_ztensor **rnn_ztens) {

  zdnn_data_layouts layout = ZDNN_NHWC;
  zdnn_data_types dtype = FP32;
  uint8_t num_gates = NUM_GATES(function_code);

  // baseline dimensions with correct requirements: fwd all-timesteps output
  uint32_t *shape[MAX_TENSOR_IDX];

  // create ztensors using transformed shape + ZDNN_NHWC to make the code
  // simplier, so that we can loop through them all rather than dealing with
  // different pre-transformed layouts etc.
  //
  // if the dims transformation logic changes then these shapes need to be
  // changed too.
  shape[INPUT] = (uint32_t[]){num_timesteps, 1, num_batches, num_features};
  shape[H0] = (uint32_t[]){num_dirs, 1, num_batches, num_hidden};
  shape[C0] = (uint32_t[]){num_dirs, 1, num_batches, num_hidden};
  shape[WEIGHTS] =
      (uint32_t[]){num_dirs, 1, num_features, num_gates * PADDED(num_hidden)};
  shape[BIASES] = (uint32_t[]){num_dirs, 1, 1, num_gates * PADDED(num_hidden)};
  shape[HIDDEN_WEIGHTS] =
      (uint32_t[]){num_dirs, 1, num_hidden, num_gates * PADDED(num_hidden)};
  shape[HIDDEN_BIASES] =
      (uint32_t[]){num_dirs, 1, 1, num_gates * PADDED(num_hidden)};
  shape[HN_OUTPUT] =
      (uint32_t[]){all_timesteps_out ? num_timesteps : 1, 1, num_batches,
                   (num_dirs < 2) ? num_hidden : num_dirs * PADDED(num_hidden)};
  shape[CF_OUTPUT] =
      (uint32_t[]){1, 1, num_batches,
                   (num_dirs < 2) ? num_hidden : num_dirs * PADDED(num_hidden)};

  for (int i = 0; i < MAX_TENSOR_IDX; i++) {
    rnn_ztens[i] = alloc_ztensor_with_values(shape[i], layout, dtype, NO_CONCAT,
                                             true, ZERO_ARRAY);
  }

  if (function_code == NNPA_GRUACT) {
    // set these to NULL so the test will blow up if used inappropriately
    shape[C0] = NULL;
    shape[CF_OUTPUT] = NULL;
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
void verify(uint8_t function_code, lstm_gru_direction direction,
            bool all_timesteps_out, tensor_idx idx, bool sabotage_dim,
            uint8_t dim_idx, uint32_t dim_val, bool sabotage_type,
            zdnn_data_types type, bool sabotage_format,
            zdnn_data_formats format, zdnn_status exp_status,
            char *description) {

  // Create the test tensors set
  zdnn_ztensor *rnn_ztens[MAX_TENSOR_IDX];
  create_ztensors(function_code, num_timesteps, num_batches, num_features,
                  num_hidden, (direction == BIDIR) ? 2 : 1, all_timesteps_out,
                  rnn_ztens);

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

  zdnn_status actual_status = verify_zdnn_lstm_or_gru_tensors(
      function_code, rnn_ztens[INPUT], rnn_ztens[H0], rnn_ztens[C0],
      rnn_ztens[WEIGHTS], rnn_ztens[BIASES], rnn_ztens[HIDDEN_WEIGHTS],
      rnn_ztens[HIDDEN_BIASES], (direction == BIDIR) ? 2 : 1,
      rnn_ztens[HN_OUTPUT], rnn_ztens[CF_OUTPUT]);

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
void verify_shape(uint8_t function_code, lstm_gru_direction direction,
                  bool all_timesteps_out, tensor_idx idx, uint8_t dim_idx,
                  uint32_t dim_val, zdnn_status exp_status, char *description) {
  verify(function_code, direction, all_timesteps_out, idx, true, dim_idx,
         dim_val, false, 0, false, 0, exp_status, description);
}

// Verify return status by sabotaging the ztensor data type
void verify_type(uint8_t function_code, lstm_gru_direction direction,
                 bool all_timesteps_out, tensor_idx idx, zdnn_data_types type,
                 zdnn_status exp_status, char *description) {
  verify(function_code, direction, all_timesteps_out, idx, false, 0, 0, true,
         type, false, 0, exp_status, description);
}

// Verify return status by sabotaging the ztensor format
void verify_format(uint8_t function_code, lstm_gru_direction direction,
                   bool all_timesteps_out, tensor_idx idx,
                   zdnn_data_formats format, zdnn_status exp_status,
                   char *description) {
  verify(function_code, direction, all_timesteps_out, idx, false, 0, 0, false,
         0, true, format, exp_status, description);
}

// this macro assume lstm_gru_direction is an 0, 1, 2... enum
#define LOOP_ALL_LSTM_GRU_DIRECTIONS(lgd) for (int lgd = 0; lgd < 3; lgd++)
// this macro assumes false = 0, true = 1
#define LOOP_TRUE_AND_FALSE(tf) for (int tf = 0; tf < 2; tf++)

/*
 * Test verification of valid activation tensors.
 * All tensors will be built with acceptable properties.
 */
void verify_pass() {
  // Expect no known error, no bad dims will be set
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {
        snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,
                 get_function_code_str(act),

                 get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");

        verify_shape(act, direction, all_timesteps_out, NONE, 0, 0, ZDNN_OK,
                     msg);
      }
    }
  }
}

/*
 * Verify num_timesteps is 0 situation
 */
void verify_timestep_zero_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {
        snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,
                 get_function_code_str(act), get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");

        verify_shape(act, direction, all_timesteps_out, INPUT, 3, 0,
                     ZDNN_INVALID_SHAPE, msg);
      }
    }
  }
}

/*
 * Verify num_timesteps mismatch situations
 */
void verify_timestep_mismatch_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {
        snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,
                 get_function_code_str(act), get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");

        verify_shape(act, direction, all_timesteps_out, H0, 3,
                     num_timesteps + 1, ZDNN_INVALID_SHAPE, msg);
      }
    }
  }
}

/*
 * Verify num_batches mismatch situations
 */
void verify_batches_mismatch_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {

        // input, h0, c0 and all outputs require the same dim2 (num_batches)
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s %s all_timesteps_out: %s", __func__,   \
           get_function_code_str(act), #tensor_idx,                            \
           get_rnn_direction_str(direction),                                   \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 2,               \
               num_batches + 1, ZDNN_INVALID_SHAPE, msg);

        TEST(INPUT);
        TEST(H0);
        if (act == NNPA_LSTMACT) {
          TEST(C0);
        }
        TEST(HN_OUTPUT);
        if (act == NNPA_LSTMACT) {
          TEST(CF_OUTPUT);
        }
#undef TEST
      }
    }
  }
}

/*
 * Verify num_features mismatch situations
 */
void verify_features_mismatch_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {
        snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,
                 get_function_code_str(act), get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");

        verify_shape(act, direction, all_timesteps_out, WEIGHTS, 2,
                     num_features + 1, ZDNN_INVALID_SHAPE, msg);
      }
    }
  }
}

/*
 * Verify num_hidden mismatch situations
 */
void verify_hidden_mismatch_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {

        // h0, c0 and all outputs require the same dim1 (num_hidden)
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s%s %s all_timesteps_out: %s", __func__,    \
           get_function_code_str(act), #tensor_idx,                            \
           get_rnn_direction_str(direction),                                   \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 1,               \
               num_hidden + 1, ZDNN_INVALID_SHAPE, msg);

        TEST(H0);
        if (act == NNPA_LSTMACT) {
          TEST(C0);
        }
        TEST(HN_OUTPUT);
        if (act == NNPA_LSTMACT) {
          TEST(CF_OUTPUT);
        }
#undef TEST

        // hidden_weights dim2 is num_hidden
        snprintf(msg, MAX_DESC_LEN, "%s %s %s %s all_timesteps_out: %s",
                 __func__, get_function_code_str(act), "HIDDEN_WEIGHTS",
                 get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");
        verify_shape(act, direction, all_timesteps_out, HIDDEN_WEIGHTS, 2,
                     num_hidden + 1, ZDNN_INVALID_SHAPE, msg);

        // (hidden_) weights and biases should have in_pad value in dim1
        uint32_t in_pad = NUM_GATES(act) * PADDED(num_hidden);
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s %s all_timesteps_out: %s", __func__,   \
           get_function_code_str(act), #tensor_idx,                            \
           get_rnn_direction_str(direction),                                   \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 1, in_pad + 1,   \
               ZDNN_INVALID_SHAPE, msg);

        TEST(WEIGHTS);
        TEST(BIASES);
        TEST(HIDDEN_WEIGHTS);
        TEST(HIDDEN_BIASES);
#undef TEST

        // the outputs should have out_pad value in dim1
        uint32_t out_pad =
            (direction != BIDIR) ? num_hidden : 2 * PADDED(num_hidden);
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,      \
           #tensor_idx, get_rnn_direction_str(direction),                      \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 1, out_pad + 1,  \
               ZDNN_INVALID_SHAPE, msg);

        TEST(HN_OUTPUT);
        if (act == NNPA_LSTMACT) {
          TEST(CF_OUTPUT);
        }
#undef TEST
      }
    }
  }
}

/*
 * Verify num_dirs mismatch situations
 */
void verify_dirs_mismatch_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {

        // h0, c0 and all outputs require the same dim4 (num_dirs)
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s %s all_timesteps_out: %s", __func__,   \
           get_function_code_str(act), #tensor_idx,                            \
           get_rnn_direction_str(direction),                                   \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 4,               \
               ((direction != BIDIR) ? 1 : 2) + 1, ZDNN_INVALID_SHAPE, msg);

        TEST(H0);
        if (act == NNPA_LSTMACT) {
          TEST(C0);
        }
        TEST(WEIGHTS);
        TEST(BIASES);
        TEST(HIDDEN_WEIGHTS);
        TEST(HIDDEN_BIASES);
#undef TEST
      }
    }
  }
}

/*
 * Verify other dims not covered in other tests
 */
void verify_other_dims_fail() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {

        // dim3 of all tensors should be 1
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s %s all_timesteps_out: %s", __func__,   \
           get_function_code_str(act), #tensor_idx,                            \
           get_rnn_direction_str(direction),                                   \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 3, 2,            \
               ZDNN_INVALID_SHAPE, msg);

        TEST(INPUT);
        TEST(H0);
        if (act == NNPA_LSTMACT) {
          TEST(C0);
        }
        TEST(WEIGHTS);
        TEST(BIASES);
        TEST(HIDDEN_WEIGHTS);
        TEST(HIDDEN_BIASES);
        TEST(HN_OUTPUT);
        if (act == NNPA_LSTMACT) {
          TEST(CF_OUTPUT);
        }
#undef TEST

        // dim2 of (hidden_)biases should be 1
#define TEST(tensor_idx)                                                       \
  snprintf(msg, MAX_DESC_LEN, "%s %s %s %s all_timesteps_out: %s", __func__,   \
           get_function_code_str(act), #tensor_idx,                            \
           get_rnn_direction_str(direction),                                   \
           all_timesteps_out ? "true" : "false");                              \
  verify_shape(act, direction, all_timesteps_out, tensor_idx, 2, 2,            \
               ZDNN_INVALID_SHAPE, msg);

        TEST(BIASES);
        TEST(HIDDEN_BIASES);
#undef TEST
      }
    }
  }
}

/*
 * Test verification of failed format
 */
void verify_fail_format() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {
        snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,
                 get_function_code_str(act), get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");

        verify_format(act, direction, all_timesteps_out, HN_OUTPUT, BAD_FORMAT,
                      ZDNN_INVALID_FORMAT, msg);
      }
    }
  }
}

/*
 * Test verification of failed type
 */
void verify_fail_type() {
  LOOP_LSTM_AND_GRU(act) {
    LOOP_ALL_LSTM_GRU_DIRECTIONS(direction) {
      LOOP_TRUE_AND_FALSE(all_timesteps_out) {
        snprintf(msg, MAX_DESC_LEN, "%s %s %s all_timesteps_out: %s", __func__,
                 get_function_code_str(act), get_rnn_direction_str(direction),
                 all_timesteps_out ? "true" : "false");

        verify_type(act, direction, all_timesteps_out, HN_OUTPUT, BAD_TYPE,
                    ZDNN_INVALID_TYPE, msg);
      }
    }
  }
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(verify_pass);
  RUN_TEST(verify_timestep_zero_fail);
  RUN_TEST(verify_timestep_mismatch_fail);
  RUN_TEST(verify_batches_mismatch_fail);
  RUN_TEST(verify_features_mismatch_fail);
  RUN_TEST(verify_hidden_mismatch_fail);
  RUN_TEST(verify_dirs_mismatch_fail);
  RUN_TEST(verify_other_dims_fail);
  RUN_TEST(verify_fail_format);
  RUN_TEST(verify_fail_type);

  return UNITY_END();
}
