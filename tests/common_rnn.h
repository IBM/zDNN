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

#ifndef TESTS_COMMON_RNN_H_
#define TESTS_COMMON_RNN_H_

#include "testsupport.h"

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

    lstm_gru_direction direction, zdnn_status exp_status);

// this macro assume values of NNPA_LSTMACT and NNPA_GRUACT are next to each
// other
#define LOOP_LSTM_AND_GRU(lg)                                                  \
  for (int lg = NNPA_LSTMACT; lg <= NNPA_GRUACT; lg++)

#endif /* TESTS_COMMON_RNN_H_ */
