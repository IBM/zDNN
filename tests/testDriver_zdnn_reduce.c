// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2023, 2024
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

#include "common_elwise.h"

void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

/*
 * Simple test to drive a full reduce api.
 */
void zdnn_reduce_val_test(uint32_t *in_dims, zdnn_data_layouts layout,
                          float *input, uint32_t *out_dims,
                          zdnn_reduce_ops op_type, zdnn_status expected_status,
                          float *expected_values) {
  /*
   * Input Tensor
   */
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      in_dims, layout, test_datatype, NO_CONCAT, false, input);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor =
      alloc_output_ztensor(out_dims, layout, test_datatype, NO_CONCAT);

  /*
   * Begin Testing!
   */
  zdnn_status status =
      zdnn_reduce(input_ztensor, NULL, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(status == expected_status,
                                "call to zdnn_reduce() with op_type %d "
                                "returned status %08x but expected  %08x\n",
                                op_type, status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  zdnn_reset_ztensor(output_ztensor);

  void *self_workarea = malloc_aligned_4k(ZDNN_8K_SAVEAREA_SIZE);
  TEST_ASSERT_MESSAGE_FORMATTED(
      self_workarea, "%s() - can't allocate SOFTMAX workarea\n", __func__);

  status = zdnn_reduce(input_ztensor, self_workarea, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_reduce() with op_type %d and provided "
      "work_area returned status %08x but expected %08x\n",
      op_type, status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }
  free_aligned_4k(self_workarea);

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

void zdnn_reduce_idx_test(uint32_t *in_dims, zdnn_data_layouts layout,
                          float *input, uint32_t *out_dims,
                          zdnn_reduce_ops op_type, zdnn_status expected_status,
                          uint32_t *expected_values) {
  /*
   * Input Tensor
   */
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      in_dims, layout, test_datatype, NO_CONCAT, false, input);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor =
      alloc_output_ztensor(out_dims, layout, INT32, NO_CONCAT);

  /*
   * Begin Testing!
   */
  zdnn_status status =
      zdnn_reduce(input_ztensor, NULL, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(status == expected_status,
                                "call to zdnn_reduce() with op_type %d "
                                "returned status %08x but expected  %08x\n",
                                op_type, status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }

  zdnn_reset_ztensor(output_ztensor);

  void *self_workarea = malloc_aligned_4k(ZDNN_8K_SAVEAREA_SIZE);
  TEST_ASSERT_MESSAGE_FORMATTED(
      self_workarea, "%s() - can't allocate SOFTMAX workarea\n", __func__);

  status = zdnn_reduce(input_ztensor, self_workarea, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "call to zdnn_reduce() with op_type %d and provided "
      "work_area returned status %08x but expected %08x\n",
      op_type, status, expected_status);

  if (expected_status == ZDNN_OK) {
    assert_ztensor_values(output_ztensor, false, expected_values);
  }
  free_aligned_4k(self_workarea);

  // All done--clean up the tensor buffers
  free_ztensor_buffers(2, input_ztensor, output_ztensor);
}

void api_reduce_basic_min() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,1)
  [[
    [[3], [6]],
    [[8], [3]]
  ]]
  */

  uint32_t out_shape[] = {1, 2, 2, 1};
  float expected_values[] = {3, 6, 8, 3};

  zdnn_reduce_val_test(in_shape, ZDNN_NHWC, input_values, out_shape,
                       REDUCE_OP_MINIMUM, ZDNN_OK, expected_values);
}

void api_reduce_nchw_min() {

  /* Input values as NCHW sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 8], [30, 80]],
    [[6, 3], [60, 10]]
  ]]
  */

  // Values in NCHW order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NCHW sized (1,1,2,2)
  [[
    [[3, 30], [3, 10]]
  ]]
  */

  uint32_t out_shape[] = {1, 1, 2, 2};
  float expected_values[] = {3, 30, 3, 10};

  zdnn_reduce_val_test(in_shape, ZDNN_NCHW, input_values, out_shape,
                       REDUCE_OP_MINIMUM, ZDNN_OK, expected_values);
}

void api_reduce_basic_min_idx() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,1)
  [[
    [[0], [0]],
    [[0], [0]]
  ]]
  */

  uint32_t out_shape[] = {1, 2, 2, 1};
  uint32_t expected_values[] = {0, 0, 0, 0};

  zdnn_reduce_idx_test(in_shape, ZDNN_NHWC, input_values, out_shape,
                       REDUCE_OP_MINIMUM_IDX, ZDNN_OK, expected_values);
}

void api_reduce_nchw_min_idx() {

  /* Input values as NCHW sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 8], [30, 80]],
    [[6, 3], [60, 10]]
  ]]
  */

  // Values in ZDNN_NCHW order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NCHW sized (1,1,2,2)
  [[
    [[0, 0]], [[1, 1]]
  ]]
  */

  uint32_t out_shape[] = {1, 1, 2, 2};
  uint32_t expected_values[] = {0, 0, 1, 1};

  zdnn_reduce_idx_test(in_shape, ZDNN_NCHW, input_values, out_shape,
                       REDUCE_OP_MINIMUM_IDX, ZDNN_OK, expected_values);
}

void api_reduce_basic_max() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,1)
  [[
    [[30], [60]],
    [[80], [10]]
  ]]
  */

  uint32_t out_shape[] = {1, 2, 2, 1};
  float expected_values[] = {30, 60, 80, 10};

  zdnn_reduce_val_test(in_shape, ZDNN_NHWC, input_values, out_shape,
                       REDUCE_OP_MAXIMUM, ZDNN_OK, expected_values);
}

void api_reduce_nchw_max() {

  /* Input values as NCHW sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 8], [30, 80]],
    [[6, 3], [60, 10]]
  ]]
  */

  // Values in ZDNN_NCHW order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NCHW sized (1,1,2,2)
  [[
    [[8, 80]], [[6, 60]]
  ]]
  */

  uint32_t out_shape[] = {1, 1, 2, 2};
  float expected_values[] = {8, 80, 6, 60};

  zdnn_reduce_val_test(in_shape, ZDNN_NCHW, input_values, out_shape,
                       REDUCE_OP_MAXIMUM, ZDNN_OK, expected_values);
}

void api_reduce_basic_max_idx() {

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  // Values in ZDNN_NHWC order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NHWC sized (1,2,2,1)
  [[
    [[1], [1]],
    [[1], [1]]
  ]]
  */

  uint32_t out_shape[] = {1, 2, 2, 1};
  uint32_t expected_values[] = {1, 1, 1, 1};

  zdnn_reduce_idx_test(in_shape, ZDNN_NHWC, input_values, out_shape,
                       REDUCE_OP_MAXIMUM_IDX, ZDNN_OK, expected_values);
}

void api_reduce_nchw_max_idx() {

  /* Input values as NCHW sized (1,2,2,2)
  [[
    [[3, 30], [6, 60]],
    [[8, 80], [3, 10]]
  ]]
  */

  /* Input values as true NHWC sized (1,2,2,2)
  [[
    [[3, 8], [30, 80]],
    [[6, 3], [60, 10]]
  ]]
  */

  // Values in ZDNN_NCHW order
  uint32_t in_shape[] = {1, 2, 2, 2};
  float input_values[] = {3, 30, 6, 60, 8, 80, 3, 10};

  /* Expected values as true NCHW sized (1,1,2,2)
  [[
    [[1, 1]], [[0, 0]]
  ]]
  */

  uint32_t out_shape[] = {1, 1, 2, 2};
  uint32_t expected_values[] = {1, 1, 0, 0};

  zdnn_reduce_idx_test(in_shape, ZDNN_NCHW, input_values, out_shape,
                       REDUCE_OP_MAXIMUM_IDX, ZDNN_OK, expected_values);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_basic_min);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_nchw_min);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_basic_min_idx);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_nchw_min_idx);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_basic_max);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_nchw_max);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_basic_max_idx);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(api_reduce_nchw_max_idx);
  return UNITY_END();
}
