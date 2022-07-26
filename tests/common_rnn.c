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

#include "common_rnn.h"
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/// Returns the size in bytes required for a RNN work_area buffer.
///
/// \param[in] rnn_layer RNN layer type (ie LSTM or GRU)
/// \param[in] batch_size batch size for the RNN
/// \param[in] num_timesteps number of timesteps in the RNN
/// \param[in] hidden_state_size number of hidden states in the RNN
///
/// \return number of bytes required for work_area based on RNN values or
/// throws a test failure.
///
size_t calc_rnn_work_area_size(uint8_t function_code, uint32_t batch_size,
                               uint32_t num_timesteps,
                               uint32_t hidden_state_size,
                               lstm_gru_direction direction) {

  uint32_t padded_hidden_state_size = CEIL(hidden_state_size, 64) * 64 * 4;
  uint32_t num_gates = get_func_code_num_gates(function_code);
  zdnn_data_layouts layout = 0;

  if (function_code == NNPA_LSTMACT) {
    layout = ZDNN_4D;
  } else if (function_code == NNPA_GRUACT) {
    layout = ZDNN_3D;
  } else {
    TEST_FAIL_MESSAGE_FORMATTED("NNPA function code %d is not supported.",
                                function_code);
  }

  // Initialize descs for work area
  zdnn_tensor_desc fused_desc, bias_add_desc, c_desc;

  init_transformed_desc(layout, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &fused_desc, num_timesteps, 1, batch_size,
                        padded_hidden_state_size);
  init_transformed_desc(layout, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &bias_add_desc, num_gates, 1, batch_size,
                        hidden_state_size);
  init_transformed_desc(layout, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE, &c_desc,
                        2, 1, batch_size, hidden_state_size);

  size_t work_area_size =
      zdnn_getsize_ztensor(&fused_desc) + zdnn_getsize_ztensor(&bias_add_desc);
  if (function_code == NNPA_LSTMACT) {
    work_area_size += zdnn_getsize_ztensor(&c_desc);
  }

  if (direction == BIDIR) {
    work_area_size *= 2;
  }
  return work_area_size;
}

/// Allocates a 4k aligned work area buffer based on the given size and returns
/// a pointer to the memory.
///
/// \param[in] work_area_size size in bytes required for the work_] area
///
/// \return pointer to the work area buffer or throws test failure
///
void *alloc_rnn_work_area(size_t work_area_size) {

  void *work_area = NULL;
  if (!(work_area = malloc_aligned_4k(work_area_size))) {
    TEST_FAIL_MESSAGE_FORMATTED("malloc_aligned_4k (%zu) failed",
                                work_area_size);
  }
  memset(work_area, 0, work_area_size);
  return work_area;
}

/// Call public API and checks returned status matches expected status. If OK
/// status expected, confirm actual output values match expected values.
///
/// \param[in] rnn_layer Type of RNN layer (ie LSTM or GRU). For LSTM
///            weights and biases will use all four gates values (FICO order)
///            and c0 and cf inputs. For GRU weights and biases use the first
///            three gate values (ZRH order). GRU ignores all g3 values and all
///            c0 and cf related inputs.
/// \param[in] ... shapes, layouts, and values to create required tensors.
/// \param[in] direction RNN layer direction (ie FWD, BWD, BIDIR)
/// \param[in] exp_status Expected status for the public API call
///
/// \return nothing but throws test failure if values don't match
/// expected or an unexpected failure prevents the test from completing.
///
void test_zdnn_api_lstm_gru(
    uint8_t function_code,

    uint32_t *input_shape, zdnn_data_layouts input_layout, float *input_values,

    uint32_t *h0_shape, zdnn_data_layouts h0_layout, float *h0_values,

    uint32_t *c0_shape, zdnn_data_layouts c0_layout, float *c0_values,

    uint32_t *input_weights_shape, zdnn_data_layouts input_weights_layout,
    float *input_weights_g0_values, float *input_weights_g1_values,
    float *input_weights_g2_values, float *input_weights_g3_values,

    uint32_t *input_biases_shape, zdnn_data_layouts input_biases_layout,
    float *input_biases_g0_values, float *input_biases_g1_values,
    float *input_biases_g2_values, float *input_biases_g3_values,

    uint32_t *hidden_weights_shape, zdnn_data_layouts hidden_weights_layout,
    float *hidden_weights_g0_values, float *hidden_weights_g1_values,
    float *hidden_weights_g2_values, float *hidden_weights_g3_values,

    uint32_t *hidden_biases_shape, zdnn_data_layouts hidden_biases_layout,
    float *hidden_biases_g0_values, float *hidden_biases_g1_values,
    float *hidden_biases_g2_values, float *hidden_biases_g3_values,

    uint32_t *hn_out_shape, zdnn_data_layouts hn_out_layout,
    float *exp_hn_out_values,

    uint32_t *cf_out_shape, zdnn_data_layouts cf_out_layout,
    float *exp_cf_out_values,

    lstm_gru_direction direction, zdnn_status exp_status) {

  char api_method[AIU_METHOD_STR_LENGTH] = "zdnn_<tbd>";

  if (function_code != NNPA_LSTMACT && function_code != NNPA_GRUACT) {
    TEST_FAIL_MESSAGE_FORMATTED("NNPA function code %d is not supported.",
                                function_code);
  }

  // Run test for each pretransformed data type
  zdnn_ztensor *input, *h0, *c0, *weights, *biases, *hidden_weights,
      *hidden_biases;
  input = alloc_ztensor_with_values(input_shape, input_layout, test_datatype,
                                    NO_CONCAT, false, input_values);
  h0 = alloc_ztensor_with_values(h0_shape, h0_layout, test_datatype, NO_CONCAT,
                                 false, h0_values);
  if (function_code == NNPA_LSTMACT) {
    // Pass all four gate buffers (FICO) to alloc_ztensor
    weights = alloc_ztensor_with_values(
        input_weights_shape, input_weights_layout, test_datatype,
        RNN_TYPE_LSTM | PREV_LAYER_UNI | USAGE_WEIGHTS, false,
        input_weights_g0_values, input_weights_g1_values,
        input_weights_g2_values, input_weights_g3_values);
    biases = alloc_ztensor_with_values(
        input_biases_shape, input_biases_layout, test_datatype,
        RNN_TYPE_LSTM | USAGE_BIASES, false, input_biases_g0_values,
        input_biases_g1_values, input_biases_g2_values, input_biases_g3_values);
    hidden_weights = alloc_ztensor_with_values(
        hidden_weights_shape, hidden_weights_layout, test_datatype,
        RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS, false, hidden_weights_g0_values,
        hidden_weights_g1_values, hidden_weights_g2_values,
        hidden_weights_g3_values);
    hidden_biases = alloc_ztensor_with_values(
        hidden_biases_shape, hidden_biases_layout, test_datatype,
        RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES, false, hidden_biases_g0_values,
        hidden_biases_g1_values, hidden_biases_g2_values,
        hidden_biases_g3_values);
    // Alloc c0 ztensor
    c0 = alloc_ztensor_with_values(c0_shape, c0_layout, test_datatype,
                                   NO_CONCAT, false, c0_values);
  } else {
    // Pass three gate buffers (ZRH) to alloc_ztensor, the fourth isn't used
    // in GRU.
    weights = alloc_ztensor_with_values(
        input_weights_shape, input_weights_layout, test_datatype,
        RNN_TYPE_GRU | PREV_LAYER_UNI | USAGE_WEIGHTS, false,
        input_weights_g0_values, input_weights_g1_values,
        input_weights_g2_values);
    biases = alloc_ztensor_with_values(
        input_biases_shape, input_biases_layout, test_datatype,
        RNN_TYPE_GRU | USAGE_BIASES, false, input_biases_g0_values,
        input_biases_g1_values, input_biases_g2_values);
    hidden_weights = alloc_ztensor_with_values(
        hidden_weights_shape, hidden_weights_layout, test_datatype,
        RNN_TYPE_GRU | USAGE_HIDDEN_WEIGHTS, false, hidden_weights_g0_values,
        hidden_weights_g1_values, hidden_weights_g2_values);
    hidden_biases = alloc_ztensor_with_values(
        hidden_biases_shape, hidden_biases_layout, test_datatype,
        RNN_TYPE_GRU | USAGE_HIDDEN_BIASES, false, hidden_biases_g0_values,
        hidden_biases_g1_values, hidden_biases_g2_values);
    c0 = NULL; // just so the compiler won't complain about uninitialized c0
  }

  // Get some basic shape info from the shapes of the various inputs
  uint32_t batch_size = input->transformed_desc->dim2;
  uint32_t num_timesteps = input->transformed_desc->dim4;
  uint32_t hidden_state_size = h0->transformed_desc->dim1;

  // Run API once NULL work_area and again with work_area set.
  for (int work_area_pass = 0; work_area_pass < 2; work_area_pass++) {
    zdnn_ztensor *hn_out, *cf_out;

    hn_out =
        alloc_ztensor_with_values(hn_out_shape, hn_out_layout, test_datatype,
                                  NO_CONCAT, true, ZERO_ARRAY);

    size_t work_area_size = 0;
    void *work_area = NULL;
    void *zeroed_work_area = NULL;

    // Set work_area during second pass
    if (work_area_pass == 1) {
      work_area_size =
          calc_rnn_work_area_size(NNPA_LSTMACT, batch_size, num_timesteps,
                                  hidden_state_size, direction);
      work_area = alloc_rnn_work_area(work_area_size);
      zeroed_work_area = alloc_rnn_work_area(work_area_size);
      memset(zeroed_work_area, 0, work_area_size);
    }

    zdnn_status status = GENERAL_TESTCASE_FAILURE;

    // Call to correct API based on layer type
    if (function_code == NNPA_LSTMACT) {
      cf_out =
          alloc_ztensor_with_values(cf_out_shape, cf_out_layout, test_datatype,
                                    NO_CONCAT, true, ZERO_ARRAY);
      // Make API call and confirm status matches expected
      strcpy(api_method, "zdnn_lstm");
      status = zdnn_lstm(input, h0, c0, weights, biases, hidden_weights,
                         hidden_biases, direction, work_area, hn_out, cf_out);
    } else if (function_code == NNPA_GRUACT) {
      strcpy(api_method, "zdnn_gru");
      status = zdnn_gru(input, h0, weights, biases, hidden_weights,
                        hidden_biases, direction, work_area, hn_out);
    }
    TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status,
                                  "work_area_pass %d call to  %s() returned "
                                  "status %08x \"%s\" but expected %08x \"%s\"",
                                  work_area_pass, api_method, status,
                                  zdnn_get_status_message(status), exp_status,
                                  zdnn_get_status_message(exp_status));
    // Check that work_area was written to on second pass
    if (work_area_pass == 1) {
      if (exp_status == ZDNN_OK &&
          !memcmp(work_area, zeroed_work_area, work_area_size)) {
        TEST_FAIL_MESSAGE_FORMATTED(
            "%s() - expected work_area have been written to but it "
            "contains all zeros",
            __func__);
      }
      free_aligned_4k(work_area);
      free_aligned_4k(zeroed_work_area);
    }

    // Confirm per timestep output tensor values match expected values
    if (exp_status == ZDNN_OK) {
      assert_ztensor_values(hn_out, false, exp_hn_out_values);
    }
    free_ztensor_buffers(1, hn_out);

    // (LSTM only) Confirm final cell state tensor values match expected
    if (function_code == NNPA_LSTMACT) {
      if (exp_status == ZDNN_OK) {
        assert_ztensor_values(cf_out, false, exp_cf_out_values);
      }
      free_ztensor_buffers(1, cf_out);
    }
  } // end of work_area_pass loop

  // Free input tensors
  free_ztensor_buffers(6, input, h0, weights, biases, hidden_weights,
                       hidden_biases);
  if (function_code == NNPA_LSTMACT) {
    free_ztensor_buffers(1, c0);
  }
}
