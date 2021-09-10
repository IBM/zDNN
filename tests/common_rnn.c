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

/// Asserts each value in the stickified bidirectional RNN output ztensor are
/// within a specified tolerance from the given expected float values. This
/// method requires the expected values include the zero padding between the FWD
/// and BWD concatenations.
///
/// \note Internally during comparison, the ztensor values are converted from
/// their current type to float.
///
/// \note This method does not check that the size of values array matches the
/// number of elements. If there's not enough expected values, the test will
/// likely fail when garbage data is pulled in as the expected value.
///
/// Example usage:
/// \code
///  assert_bidir_output(&ztensor, false, values, tol);
/// \endcode
///
/// \param[in] ztensor pointer to zdnn_ztensor with actual values
/// \param[in] repeat_first_expected_value if true, all ztensor values will be
///                                    compared to values[0]
/// \param[in] values array of expected values
/// \param[in] tol floating point tolerance information
///
/// \return None (assert fails if any actual value not within expected range)
///
void assert_bidir_output_adv(zdnn_ztensor *concat_output,
                             bool repeat_first_expected_value, float *values,
                             fp_tolerance tol) {

  // Generate a ztensor based on the expected values.
  uint32_t *exp_out_shape = &concat_output->transformed_desc->dim4;
  zdnn_ztensor *exp_hn_out =
      alloc_ztensor_with_values(exp_out_shape, ZDNN_NHWC, FP32, NO_CONCAT,
                                repeat_first_expected_value, values);
  size_t *exp_hn_out_offsets = alloc_offsets(exp_hn_out, QUICK_OFFSETS, NULL);
  BEGIN_BLOCK_IF_LOGLEVEL_TRACE {
    printf("%s(): expected values ztensor dumpdata_ztensor():\n", __func__);
    dumpdata_ztensor(exp_hn_out, AS_FLOAT, false);
  }

  uint64_t num_elements = get_num_elements(concat_output, ELEMENTS_ALL);

  // Setup error message
  uint64_t big_error_message_size =
      (uint64_t)sizeof(char) * ERROR_MESSAGE_STR_LENGTH * num_elements;
  char *error_msg = malloc(big_error_message_size);
  snprintf(error_msg, big_error_message_size,
           "comparing ztensor values to array values\n");

  bool all_pass = true;
  for (uint64_t i = 0; i < num_elements; i++) {
    uint16_t act_dlf16 =
        *((uint16_t *)(concat_output->buffer + exp_hn_out_offsets[i]));
    uint16_t exp_dlf16 =
        *((uint16_t *)(exp_hn_out->buffer + exp_hn_out_offsets[i]));
    float act_fp32 = cnvt_1_dlf16_to_fp32(act_dlf16);
    float exp_fp32 = values[i];

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg),
             "Element %" PRIu64 " at buffer offset %08lx. As %s hex == "
             "%04x expecting %04x. As %s value == %f expecting %f",
             i, exp_hn_out_offsets[i], get_data_type_str(ZDNN_DLFLOAT16),
             act_dlf16, exp_dlf16, get_data_type_str(FP32), act_fp32, exp_fp32);

    if (!almost_equal_dlf16_adv(act_dlf16, exp_dlf16, tol)) {
      snprintf(error_msg + strlen(error_msg),
               big_error_message_size - strlen(error_msg),
               " <==== FAILED (diff beyond ULPs %u, epsilon multiplier %u)",
               tol.ulps, tol.epsilon_mult);
      all_pass = false;
    }

    snprintf(error_msg + strlen(error_msg),
             big_error_message_size - strlen(error_msg), "\n");
  }
  TEST_ASSERT_MESSAGE(all_pass, error_msg);

  free(exp_hn_out_offsets);
  free(exp_hn_out);
}

void assert_bidir_output(zdnn_ztensor *ztensor,
                         bool repeat_first_expected_value, void *values) {
  fp_tolerance tol = {0, 0}; // zero tolerance ==> testcase will likely fail.

  tol.ulps = MAX_ULPS_DLFLOAT16;
  tol.epsilon_mult = MAX_EPSILON_MULT_DLFLOAT16;

  assert_bidir_output_adv(ztensor, repeat_first_expected_value, values, tol);
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
        input_weights_shape, input_weights_layout, test_datatype, CONCAT_LSTM,
        false, input_weights_g0_values, input_weights_g1_values,
        input_weights_g2_values, input_weights_g3_values);
    biases = alloc_ztensor_with_values(
        input_biases_shape, input_biases_layout, test_datatype, CONCAT_LSTM,
        false, input_biases_g0_values, input_biases_g1_values,
        input_biases_g2_values, input_biases_g3_values);
    hidden_weights = alloc_ztensor_with_values(
        hidden_weights_shape, hidden_weights_layout, test_datatype, CONCAT_LSTM,
        false, hidden_weights_g0_values, hidden_weights_g1_values,
        hidden_weights_g2_values, hidden_weights_g3_values);
    hidden_biases = alloc_ztensor_with_values(
        hidden_biases_shape, hidden_biases_layout, test_datatype, CONCAT_LSTM,
        false, hidden_biases_g0_values, hidden_biases_g1_values,
        hidden_biases_g2_values, hidden_biases_g3_values);
    // Alloc c0 ztensor
    c0 = alloc_ztensor_with_values(c0_shape, c0_layout, test_datatype,
                                   NO_CONCAT, false, c0_values);
  } else {
    // Pass three gate buffers (ZRH) to alloc_ztensor, the fourth isn't used
    // in GRU.
    weights = alloc_ztensor_with_values(
        input_weights_shape, input_weights_layout, test_datatype, CONCAT_GRU,
        false, input_weights_g0_values, input_weights_g1_values,
        input_weights_g2_values);

    biases = alloc_ztensor_with_values(
        input_biases_shape, input_biases_layout, test_datatype, CONCAT_GRU,
        false, input_biases_g0_values, input_biases_g1_values,
        input_biases_g2_values);
    hidden_weights = alloc_ztensor_with_values(
        hidden_weights_shape, hidden_weights_layout, test_datatype, CONCAT_GRU,
        false, hidden_weights_g0_values, hidden_weights_g1_values,
        hidden_weights_g2_values);
    hidden_biases = alloc_ztensor_with_values(
        hidden_biases_shape, hidden_biases_layout, test_datatype, CONCAT_GRU,
        false, hidden_biases_g0_values, hidden_biases_g1_values,
        hidden_biases_g2_values);
  }

  // Get some basic shape info from the shapes of the various inputs
  uint32_t batch_size = input->transformed_desc->dim2;
  uint32_t num_timesteps = input->transformed_desc->dim4;
  uint32_t hidden_state_size = h0->transformed_desc->dim1;

  // Run API once NULL work_area and again with work_area set.
  for (int work_area_pass = 0; work_area_pass < 2; work_area_pass++) {
    zdnn_ztensor_concat_types output_concat = NO_CONCAT;
    zdnn_ztensor *hn_out, *cf_out;

    if (direction == BIDIR) {
      output_concat = CONCAT_BIDIR_OUTPUT;
    }
    hn_out =
        alloc_ztensor_with_values(hn_out_shape, hn_out_layout, test_datatype,
                                  output_concat, true, ZERO_ARRAY);

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
                                    output_concat, true, ZERO_ARRAY);
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
        TEST_FAIL_MESSAGE("expected work_area have been written to but it "
                          "contains all zeros");
      }
      free_aligned_4k(work_area);
      free_aligned_4k(zeroed_work_area);
    }

    // Confirm per timestep output tensor values match expected values
    if (exp_status == ZDNN_OK) {
      if (direction == BIDIR) {
        assert_bidir_output(hn_out, false, exp_hn_out_values);
      } else {
        assert_ztensor_values(hn_out, false, exp_hn_out_values);
      }
    }
    free_ztensor_buffers(1, hn_out);

    // (LSTM only) Confirm final cell state tensor values match expected
    if (function_code == NNPA_LSTMACT) {
      if (exp_status == ZDNN_OK) {
        if (direction == BIDIR) {
          assert_bidir_output(cf_out, false, exp_cf_out_values);
        } else {
          assert_ztensor_values(cf_out, false, exp_cf_out_values);
        }
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
