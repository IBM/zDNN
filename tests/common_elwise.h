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

#ifndef TESTS_COMMON_ELWISE_H_
#define TESTS_COMMON_ELWISE_H_

#include "testsupport.h"

#include <string.h>

void test_elwise_api_1_input(uint32_t *shape, zdnn_data_layouts layout,
                             float *input_values,
                             nnpa_function_code function_code,
                             zdnn_status expected_status);
void test_elwise_api_2_inputs(uint32_t *shape, zdnn_data_layouts layout,
                              float *input1_values, float *input2_values,
                              nnpa_function_code function_code,
                              zdnn_status expected_status);

void test_elwise_api_2_inputs_adv(uint32_t *shape, zdnn_data_layouts layout,
                                  zdnn_data_types type, float *input1_values,
                                  float *input2_values,
                                  nnpa_function_code function_code,
                                  zdnn_status expected_status);

#endif /* TESTS_COMMON_ELWISE_H_ */
