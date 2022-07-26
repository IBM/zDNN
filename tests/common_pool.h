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

#ifndef TESTS_COMMON_POOL_H_
#define TESTS_COMMON_POOL_H_

#include "testsupport.h"

#include <string.h>

// Restrictions placed on pooling ops. If they're changed, update the API
// documentation for all pool (avg, max, meanreduce2d) ops!
#define MAXIMUM_POOL_ZERO_STRIDES_KERNEL_SIZE 1024
#define MAXIMUM_POOL_NONZERO_STRIDES_HEIGHT_WIDTH 1024
#define MAXIMUM_POOL_NONZERO_STRIDES_KERNEL_SIZE 64
#define MAXIMUM_POOL_NONZERO_STRIDES_STRIDE_SIZE 30

void test_pool_function(nnpa_function_code function_code, uint32_t *input_shape,
                        zdnn_data_layouts input_layout,
                        bool repeat_first_input_value, float *input_values,
                        zdnn_pool_padding padding_type, uint32_t kernel_height,
                        uint32_t kernel_width, uint32_t stride_height,
                        uint32_t stride_width, uint32_t *output_shape,
                        zdnn_data_layouts output_layout,
                        zdnn_status expected_status,
                        bool repeat_first_expected_value,
                        float *expected_values);

#endif /* TESTS_COMMON_POOL_H_ */
