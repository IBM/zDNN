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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: LSTM BI-DIR
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * LSTM (BI-DIR):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (2, num_batches, num_hidden)
   * c0              |  ZDNN_3DS  | (2, num_batches, num_hidden)
   * weights         |  ZDNN_3DS  | (2, num_features, num_hidden)
   * biases          |  ZDNN_2DS  | (2, num_hidden)
   * hidden_weights  |  ZDNN_3DS  | (2, num_hidden, num_hidden)
   * hidden_biases   |  ZDNN_2DS  | (2, num_hidden)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_4DS  | (num_timesteps, 2, num_batches, num_hidden)
   *                 |            | or (1, 2, num_batches, num_hidden)
   * cf_output       |  ZDNN_4DS  | (1, 2, num_batches, num_hidden)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hidden = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  lstm_gru_direction dir = BIDIR;
  uint8_t num_dirs = 2;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &input_pre_tfrmd_desc,
                                 num_timesteps, num_batches, num_features);
  status =
      zdnn_generate_transformed_desc(&input_pre_tfrmd_desc, &input_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                         &input_tfrmd_desc, &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size =
      num_timesteps * num_batches * num_features * element_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_ztensor(&input, input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create initial hidden and cell state zTensors
   ***********************************************************************/

  zdnn_tensor_desc h0c0_pre_tfrmd_desc, h0c0_tfrmd_desc;
  zdnn_ztensor h0, c0;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &h0c0_pre_tfrmd_desc, num_dirs,
                                 num_batches, num_hidden);
  status =
      zdnn_generate_transformed_desc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &h0);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &c0);
  assert(status == ZDNN_OK);

  uint64_t h0c0_data_size = num_batches * num_hidden * element_size;
  void *hidden_state_data = malloc(h0c0_data_size);
  void *cell_state_data = malloc(h0c0_data_size);

  status = zdnn_transform_ztensor(&h0, hidden_state_data);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&c0, cell_state_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create input weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &weights_pre_tfrmd_desc,
                                 num_dirs, num_features, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_NONE,
      &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hidden * element_size;
  void *weights_data_f = malloc(weights_data_size);
  void *weights_data_i = malloc(weights_data_size);
  void *weights_data_c = malloc(weights_data_size);
  void *weights_data_o = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_f, weights_data_i,
                                  weights_data_c, weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, RNN_TYPE_LSTM | USAGE_BIASES | PREV_LAYER_NONE,
      &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hidden * element_size;
  void *biases_data_f = malloc(biases_data_size);
  void *biases_data_i = malloc(biases_data_size);
  void *biases_data_c = malloc(biases_data_size);
  void *biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_f, biases_data_i,
                                  biases_data_c, biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hidden, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE,
      &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hidden * num_hidden * element_size;
  void *hidden_weights_data_f = malloc(hidden_weights_data_size);
  void *hidden_weights_data_i = malloc(hidden_weights_data_size);
  void *hidden_weights_data_c = malloc(hidden_weights_data_size);
  void *hidden_weights_data_o = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_f,
                                  hidden_weights_data_i, hidden_weights_data_c,
                                  hidden_weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_biases_pre_tfrmd_desc, hidden_biases_tfrmd_desc;
  zdnn_ztensor hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &hidden_biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_biases_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE,
      &hidden_biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(
      &hidden_biases_pre_tfrmd_desc, &hidden_biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t hidden_biases_data_size = num_hidden * element_size;

  void *hidden_biases_data_f = malloc(hidden_biases_data_size);
  void *hidden_biases_data_i = malloc(hidden_biases_data_size);
  void *hidden_biases_data_c = malloc(hidden_biases_data_size);
  void *hidden_biases_data_o = malloc(hidden_biases_data_size);

  status = zdnn_transform_ztensor(&hidden_biases, hidden_biases_data_f,
                                  hidden_biases_data_i, hidden_biases_data_c,
                                  hidden_biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/

  zdnn_tensor_desc hn_pre_tfrmd_desc, hn_tfrmd_desc, cf_pre_tfrmd_desc,
      cf_tfrmd_desc;

  zdnn_ztensor hn_output_ztensor, cf_output_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_4DS, type, &hn_pre_tfrmd_desc,
                                 num_timesteps, 2, num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc);
  assert(status == ZDNN_OK);

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &cf_pre_tfrmd_desc, 1, 2,
                                 num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc,
                                         &hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc,
                                         &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the AIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_lstm(&input, &h0, &c0, &weights, &biases, &hidden_weights,
                     &hidden_biases, dir, work_area, &hn_output_ztensor,
                     &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  uint64_t hn_data_size =
      num_timesteps * 2 * num_batches * num_hidden * element_size;
  uint64_t cf_data_size = 2 * num_batches * num_hidden * element_size;
  void *hn_output_data = malloc(hn_data_size);
  void *cf_output_data = malloc(cf_data_size);

  status = zdnn_transform_origtensor(&hn_output_ztensor, hn_output_data);
  assert(status == ZDNN_OK);
  status = zdnn_transform_origtensor(&cf_output_ztensor, cf_output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&h0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&c0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&cf_output_ztensor);
  assert(status == ZDNN_OK);

  free(input_data);
  free(hidden_state_data);
  free(cell_state_data);
  free(weights_data_f);
  free(weights_data_i);
  free(weights_data_c);
  free(weights_data_o);
  free(hidden_weights_data_f);
  free(hidden_weights_data_i);
  free(hidden_weights_data_c);
  free(hidden_weights_data_o);
  free(biases_data_f);
  free(biases_data_i);
  free(biases_data_c);
  free(biases_data_o);
  free(hidden_biases_data_f);
  free(hidden_biases_data_i);
  free(hidden_biases_data_c);
  free(hidden_biases_data_o);
  free(hn_output_data);
  free(cf_output_data);
}
