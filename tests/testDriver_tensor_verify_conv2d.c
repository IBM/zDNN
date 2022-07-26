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

#include <string.h>

// struct for tensor information
typedef struct tensor_info {
  uint32_t dims[ZDNN_MAX_DIMS];
  zdnn_data_layouts layout;
  zdnn_data_types dtype;
} tensor_info;

// struct for a set of inputs for a testcase (padding + tensors + strides)
typedef struct input_set {
  zdnn_pool_padding padding;
  tensor_info input;
  tensor_info kernel;
  tensor_info bias;
  tensor_info output;
  uint32_t stride_height;
  uint32_t stride_width;
} input_set;

// "good input sets" - initialized during setUp(), shall NOT be modified by
// testcases afterwards
input_set same_padding_nonzero_stride;
input_set valid_padding_nonzero_stride;
input_set valid_padding_zero_stride;

#define DIM4 dims[0]
#define DIM3 dims[1]
#define DIM2 dims[2]
#define DIM1 dims[3]

#define INIT_TENSOR(set, info, dim4, dim3, dim2, dim1, l, t)                   \
  set.info.DIM4 = dim4;                                                        \
  set.info.DIM3 = dim3;                                                        \
  set.info.DIM2 = dim2;                                                        \
  set.info.DIM1 = dim1;                                                        \
  set.info.layout = l;                                                         \
  set.info.dtype = t;

void setUp(void) { /* This is run before EACH TEST */

  VERIFY_HW_ENV;

  same_padding_nonzero_stride.padding = SAME_PADDING;
  INIT_TENSOR(same_padding_nonzero_stride, input, 4, 6, 9, 5, ZDNN_NHWC, FP32);
  INIT_TENSOR(same_padding_nonzero_stride, kernel, 3, 8, 5, 8, ZDNN_HWCK, FP32);
  INIT_TENSOR(same_padding_nonzero_stride, bias, 8, -1, -1, -1, ZDNN_1D,
              FP32); // -1 are ignored since it's 1D
  INIT_TENSOR(same_padding_nonzero_stride, output, 4, 2, 5, 8, ZDNN_NHWC, FP32);
  same_padding_nonzero_stride.stride_height = 3;
  same_padding_nonzero_stride.stride_width = 2;

  valid_padding_nonzero_stride.padding = VALID_PADDING;
  INIT_TENSOR(valid_padding_nonzero_stride, input, 4, 6, 9, 5, ZDNN_NHWC, FP32);
  INIT_TENSOR(valid_padding_nonzero_stride, kernel, 3, 8, 5, 8, ZDNN_HWCK,
              FP32);
  INIT_TENSOR(valid_padding_nonzero_stride, bias, 8, -1, -1, -1, ZDNN_1D, FP32);
  INIT_TENSOR(valid_padding_nonzero_stride, output, 4, 2, 1, 8, ZDNN_NHWC,
              FP32);
  valid_padding_nonzero_stride.stride_height = 3;
  valid_padding_nonzero_stride.stride_width = 2;

  valid_padding_zero_stride.padding = VALID_PADDING;
  INIT_TENSOR(valid_padding_zero_stride, input, 4, 3, 8, 5, ZDNN_NHWC, FP32);
  INIT_TENSOR(valid_padding_zero_stride, kernel, 3, 8, 5, 8, ZDNN_HWCK, FP32);
  INIT_TENSOR(valid_padding_zero_stride, bias, 8, -1, -1, -1, ZDNN_1D, FP32);
  INIT_TENSOR(valid_padding_zero_stride, output, 4, 1, 1, 8, ZDNN_NHWC, FP32);
  valid_padding_zero_stride.stride_height = 0;
  valid_padding_zero_stride.stride_width = 0;
}

void tearDown(void) { /* This is run after EACH TEST */
}

#define NON_EXISTENT_FORMAT -1
#define NON_EXISTENT_DTYPE -1

void run_verify_conv2d_tensors_full(input_set set, zdnn_conv2d_act act_func,
                                    bool use_non_existent_format,
                                    bool use_non_existent_dtype,
                                    zdnn_status expected_status) {

  zdnn_status status = GENERAL_TESTCASE_FAILURE;

  zdnn_ztensor *input_ztensor =
      alloc_ztensor_with_values(set.input.dims, set.input.layout,
                                set.input.dtype, NO_CONCAT, true, ZERO_ARRAY);

  zdnn_ztensor *kernel_ztensor =
      alloc_ztensor_with_values(set.kernel.dims, set.kernel.layout,
                                set.kernel.dtype, NO_CONCAT, true, ZERO_ARRAY);

  zdnn_ztensor *bias_ztensor =
      alloc_ztensor_with_values(set.bias.dims, set.bias.layout, set.bias.dtype,
                                NO_CONCAT, true, ZERO_ARRAY);

  zdnn_ztensor *output_ztensor =
      alloc_ztensor_with_values(set.output.dims, set.output.layout,
                                set.output.dtype, NO_CONCAT, true, ZERO_ARRAY);

  if (use_non_existent_dtype) {
    output_ztensor->transformed_desc->type = NON_EXISTENT_DTYPE;
  }

  if (use_non_existent_format) {
    output_ztensor->transformed_desc->format = NON_EXISTENT_FORMAT;
  }

  func_sp_parm1_conv2d conv2d_parm1;
  conv2d_parm1.val = 0;
  conv2d_parm1.bits.act = act_func;
  conv2d_parm1.bits.pad = set.padding;

  func_sp_parm4_conv2d conv2d_parm4;
  conv2d_parm4.val = 0;
  conv2d_parm4.bits.clipping_value = 0;

  // Make call to verify with our newly created ztensors and other inputs
  TEST_ASSERT_MESSAGE_FORMATTED(
      verify_conv2d_tensors(input_ztensor, kernel_ztensor, bias_ztensor,
                            conv2d_parm1.val, set.stride_height,
                            set.stride_width, conv2d_parm4.val,
                            output_ztensor) == expected_status,
      "Call to verify_conv2d_tensors() returned zdnn_status %08x but we "
      "expected %08x",
      status, expected_status);

  free_ztensor_buffers(4, input_ztensor, kernel_ztensor, bias_ztensor,
                       output_ztensor);
}

void run_verify_conv2d_tensors(input_set set, zdnn_conv2d_act act_func,
                               zdnn_status expected_status) {
  run_verify_conv2d_tensors_full(set, act_func, false, false, expected_status);
}

void same_padding_pass() {
  input_set set;

  memcpy(&set, &same_padding_nonzero_stride, sizeof(input_set));
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_OK);
}

void valid_padding_pass() {
  input_set set;

  memcpy(&set, &valid_padding_nonzero_stride, sizeof(input_set));
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_OK);

  memcpy(&set, &valid_padding_zero_stride, sizeof(input_set));
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_OK);
}

// although actual op would fail, tensor-verify would pass
void unknown_padding_type_pass() {
  input_set set;

  memcpy(&set, &valid_padding_nonzero_stride, sizeof(input_set));
  set.padding = -1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_OK);
}

void output_different_dtype_fail() {
  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    run_verify_conv2d_tensors_full(set[i], CONV2D_ACT_NONE, false, true,
                                   ZDNN_INVALID_TYPE);
  }
}

void output_different_format_fail() {
  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    run_verify_conv2d_tensors_full(set[i], CONV2D_ACT_NONE, true, false,
                                   ZDNN_INVALID_FORMAT);
  }
}

void bias_not_bias_fail() {
  /*
  The dimension-2, dimension-3, and dimension-4
  index sizes of the input 3 tensor must be 1.
  */

  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    set[i].bias.dims[0] = 2;
    set[i].bias.dims[1] = 8;
    set[i].bias.layout = ZDNN_2D;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
  }
}

void different_output_dim4_input_dim4_fail() {
  /*
  The dimension-4-index-size of the output tensor must be equal to the
  dimension-4-index-size of the input 1 tensor.
  */

  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    set[i].output.DIM4 = set[i].input.DIM4 + 1;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
  }
}

void different_output_dim1_input2_dim1_fail() {
  /*
  The dimension-1 index size of the output tensor must be equal to the
  dimension-1 index size of the input 2 tensor and the dimension-1-index size of
  the input 3 tensor.
  */

  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    set[i].output.DIM1 = set[i].kernel.DIM1 + 1;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
  }
}

void different_output_dim1_input3_dim1_fail() {
  /*
  The dimension-1 index size of the output tensor must be equal to the
  dimension-1 index size of the input 2 tensor and the dimension-1-index size of
  the input 3 tensor.
  */

  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    // bias is 1D so dimension-1-index came from dims[0]
    set[i].output.DIM1 = set[i].bias.dims[0] + 1;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
  }
}

void different_input_dim1_input2_dim2_fail() {
  /*
  The dimension-1 index size of the input 1 tensor must be equal to the
  dimension-2 index size of the input 2 tensor.
  */

  input_set set[3];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 2, &valid_padding_zero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    set[i].input.DIM1 = set[i].kernel.DIM2 + 1;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
  }
}

/*****************************************************
If the dimension-2-stride and the dimension-3-
stride are both zero all of the following additional
conditions must be true:
*****************************************************/

void different_input1_dim2_input2_dim3_fail() {
  /*
  The input 1 tensor dimension-2-index-size must be equal to the
  dimension-3-index-size of input 2 tensor.
  */

  input_set set;

  memcpy(&set, &valid_padding_zero_stride, sizeof(input_set));
  set.kernel.DIM3 = set.input.DIM2 + 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void different_input1_dim3_input2_dim4_fail() {
  /*
  The input 1 tensor dimension-3-index-size of the input tensor must be equal to
  the dimension-4-index-size of input 2 tensor.
  */

  input_set set;

  memcpy(&set, &valid_padding_zero_stride, sizeof(input_set));
  set.kernel.DIM4 = set.input.DIM3 + 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void output_dim2_not_one_fail() {
  /*
  The dimension-2-index-size and the dimension-3-index-size of the output tensor
  must be one.
  */

  input_set set;

  memcpy(&set, &valid_padding_zero_stride, sizeof(input_set));
  set.output.DIM2 = 2;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void output_dim3_not_one_fail() {
  /*
  The dimension-2-index-size and the dimension-3-index-size of the output tensor
  must be one.
  */

  input_set set;

  memcpy(&set, &valid_padding_zero_stride, sizeof(input_set));
  set.output.DIM3 = 2;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void zero_height_width_not_validpadding_fail() {
  /*
  The specified padding must be VALID.
  */

  input_set set;

  memcpy(&set, &valid_padding_zero_stride, sizeof(input_set));
  set.padding = SAME_PADDING;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_STRIDE_PADDING);
}

/*********************************************
If the dimension-2-stride and the dimension-3-
stride are both greater than zero all of the
following additional conditions must be true:
*********************************************/

void valid_input_dim2_lessthan_kernel_dim3_fail() {
  /*
  When the specified padding is VALID, the dimension-2-index-size of the input 1
  tensor must be greater than or equal to the dimension-3-index-size of input
  tensor 2.
  */

  input_set set;

  memcpy(&set, &valid_padding_nonzero_stride, sizeof(input_set));
  set.input.DIM2 = set.kernel.DIM3 - 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void valid_input_dim3_lessthan_kernel_dim4_fail() {
  /*
  When the specified padding is VALID, the dimension-3-index-size of the input 1
  tensor must be greater than or equal to the dimension-4-index-size of the
  input 2 tensor.
  */

  input_set set;

  memcpy(&set, &valid_padding_nonzero_stride, sizeof(input_set));
  set.input.DIM3 = set.kernel.DIM4 - 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void same_big_math_equation1_fail() {
  /*
  When the specified padding is SAME, the following relationship between the
  dimension-2-index-size and dimension-3-index-size of the input 1 tensor and
  output tensor must be satisfied:

  Dimension-2-index-size of the output tensor = ceil( Dimension-2-index-size
  of the input 1 tensor / Dimension-2-stride)

  Dimension-3-index-size of the output tensor = ceil( Dimension-3-index-size
  of the input 1 tensor / Dimension-3-stride)
  */

  input_set set;

  memcpy(&set, &same_padding_nonzero_stride, sizeof(input_set));
  set.stride_width = 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void same_big_math_equation2_fail() {
  /*
  When the specified padding is SAME, the following relationship between the
  dimension-2-index-size and dimension-3-index-size of the input 1 tensor and
  output tensor must be satisfied:

  Dimension-2-index-size of the output tensor = ceil( Dimension-2-index-size
  of the input 1 tensor / Dimension-2-stride)

  Dimension-3-index-size of the output tensor = ceil( Dimension-3-index-size
  of the input 1 tensor / Dimension-3-stride)
  */

  input_set set;

  memcpy(&set, &same_padding_nonzero_stride, sizeof(input_set));
  set.stride_height = 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void valid_big_math_equation1_fail() {
  /*
  When the specified padding is VALID, the following relationship between the
  dimension-2-index-size and dimension-3-index-sizes of the input 1 tensor,
  dimension-3-index-size and dimension-4-index-size of the input 2 tensor and
  output tensor must be satisfied:

  Dimension-2-index-size of the output tensor = ceil(
     (Dimension-2-index-size of the input 1 tensor -  Dimension-3-index-size of
      the input 2 tensor + 1 ) / Dimension-2-stride

  Dimension-3-index-size of the output tensor = ceil(
     (Dimension-3-index-size of the input 1 tensor -  Dimension-4-index-size of
      the input 2 tensor + 1 ) / Dimension-3-stride
  */

  input_set set;

  memcpy(&set, &valid_padding_nonzero_stride, sizeof(input_set));
  set.stride_width = 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void valid_big_math_equation2_fail() {
  /*
  When the specified padding is VALID, the following relationship between the
  dimension-2-index-size and dimension-3-index-sizes of the input 1 tensor,
  dimension-3-index-size and dimension-4-index-size of the input 2 tensor and
  output tensor must be satisfied:

  Dimension-2-index-size of the output tensor = ceil(
     (Dimension-2-index-size of the input 1 tensor -  Dimension-3-index-size of
      the input 2 tensor + 1 ) / Dimension-2-stride

  Dimension-3-index-size of the output tensor = ceil(
     (Dimension-3-index-size of the input 1 tensor -  Dimension-4-index-size of
      the input 2 tensor + 1 ) / Dimension-3-stride
  */

  input_set set;

  memcpy(&set, &valid_padding_nonzero_stride, sizeof(input_set));
  set.stride_height = 1;
  run_verify_conv2d_tensors(set, CONV2D_ACT_NONE, ZDNN_INVALID_SHAPE);
}

void height_zero_width_nonzero_fail() {
  /*
  If either the dimension-2-stride or the dimension3-stride is non-zero, then
  both strides must be non-zero.
  */

  input_set set[2];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    set[i].stride_height = 0;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_STRIDES);
  }
}

void height_nonzero_width_zero_fail() {
  /*
  If either the dimension-2-stride or the dimension3-stride is non-zero, then
  both strides must be non-zero.
  */

  input_set set[2];

  memcpy(set, &same_padding_nonzero_stride, sizeof(input_set));
  memcpy(set + 1, &valid_padding_nonzero_stride, sizeof(input_set));

  for (int i = 0; i < sizeof(set) / sizeof(input_set); i++) {
    set[i].stride_width = 0;
    run_verify_conv2d_tensors(set[i], CONV2D_ACT_NONE, ZDNN_INVALID_STRIDES);
  }
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(same_padding_pass);
  RUN_TEST(valid_padding_pass);
  RUN_TEST(unknown_padding_type_pass);

  RUN_TEST(output_different_dtype_fail);
  RUN_TEST(output_different_format_fail);

  RUN_TEST(bias_not_bias_fail);
  RUN_TEST(different_output_dim4_input_dim4_fail);
  RUN_TEST(different_output_dim1_input2_dim1_fail);
  RUN_TEST(different_output_dim1_input3_dim1_fail);
  RUN_TEST(different_input_dim1_input2_dim2_fail);

  RUN_TEST(different_input1_dim2_input2_dim3_fail);
  RUN_TEST(different_input1_dim3_input2_dim4_fail);
  RUN_TEST(different_input1_dim3_input2_dim4_fail);
  RUN_TEST(output_dim2_not_one_fail);
  RUN_TEST(output_dim3_not_one_fail);
  RUN_TEST(zero_height_width_not_validpadding_fail);

  RUN_TEST(valid_input_dim2_lessthan_kernel_dim3_fail);
  RUN_TEST(valid_input_dim3_lessthan_kernel_dim4_fail);

  RUN_TEST(same_big_math_equation1_fail);
  RUN_TEST(same_big_math_equation2_fail);
  RUN_TEST(valid_big_math_equation1_fail);
  RUN_TEST(valid_big_math_equation2_fail);

  RUN_TEST(height_zero_width_nonzero_fail);
  RUN_TEST(height_nonzero_width_zero_fail);

  return UNITY_END();
}
