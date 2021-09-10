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

#include "zdnn.h"
#include "zdnn_private.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

/// Verify multiple zTensors against specific type/format values.
/// Variadic parameter list MUST be NULL-TERMINATED.
///
/// \param[in] type required data type
/// \param[in] format required format
/// \param[in] ... list of (zdnn_ztensor*, char* name) pairs, NULL-TERMINATED at
///                the end
///
/// \return ZDNN_OK
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
static zdnn_status verify_fields(zdnn_data_types type, zdnn_data_formats format,
                                 ...) {
  zdnn_status status = ZDNN_OK;
  va_list v;
  va_start(v, format);

  zdnn_ztensor *tsr_ptr;
  uint8_t i = 0;

  while ((tsr_ptr = va_arg(v, zdnn_ztensor *)) != NULL) {
    char *tsr_name = va_arg(v, char *);
    if (tsr_ptr->transformed_desc->type != type) {
      status = ZDNN_STATUS(
          ZDNN_INVALID_TYPE,
          "%s tensor type is invalid (found %s (%d), expects %s (%d))",
          tsr_name, get_data_type_str(tsr_ptr->transformed_desc->type),
          tsr_ptr->transformed_desc->type, get_data_type_str(type), type);
      break;
    }
    if (tsr_ptr->transformed_desc->format != format) {
      status = ZDNN_STATUS(
          ZDNN_INVALID_FORMAT,
          "%s tensor format is invalid (found %s (%d), expects %s (%d))",
          tsr_name, get_data_format_str(tsr_ptr->transformed_desc->format),
          tsr_ptr->transformed_desc->format, get_data_format_str(format),
          format);
      break;
    }
    i++;
  }

  va_end(v);
  return status;
}

/// Verify multiple zTensors against specific shape value
/// Variadic parameter list MUST be NULL-TERMINATED
///
/// \param[in] dim_idx dimX index
/// \param[in] val required value
/// \param[in] ... list of (zdnn_ztensor*, char* name) pairs, NULL-TERMINATED at
///                the end
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///
static zdnn_status verify_dim(uint8_t dim_idx, uint32_t val, ...) {
  zdnn_status status = ZDNN_OK;
  va_list v;
  va_start(v, val);

  zdnn_ztensor *tsr_ptr;
  uint8_t i = 0;

  while ((tsr_ptr = va_arg(v, zdnn_ztensor *)) != NULL) {
    char *tsr_name = va_arg(v, char *);
    uint32_t *dims_ptr = &(tsr_ptr->transformed_desc->dim4);
    if (dims_ptr[ZDNN_MAX_DIMS - dim_idx] != val) {
      status = ZDNN_STATUS(
          ZDNN_INVALID_SHAPE,
          "%s dim%d tensor shape is invalid (found %d, expects %d)", tsr_name,
          dim_idx, dims_ptr[ZDNN_MAX_DIMS - dim_idx], val);
      break;
    }
    i++;
  }

  va_end(v);
  return status;
}

// Convenience macros with auto NULL-terminated variadic list

#define VERIFY_FIELDS(type, format, ...)                                       \
  verify_fields(type, format, __VA_ARGS__, NO_ARG)

#define TENSOR_PARM(a) a, #a // stringify the tensor name, expands to: x, "x"

#define VERIFY_DIM4(val, ...) verify_dim(4, val, __VA_ARGS__, NO_ARG)
#define VERIFY_DIM3(val, ...) verify_dim(3, val, __VA_ARGS__, NO_ARG)
#define VERIFY_DIM2(val, ...) verify_dim(2, val, __VA_ARGS__, NO_ARG)
#define VERIFY_DIM1(val, ...) verify_dim(1, val, __VA_ARGS__, NO_ARG)

#define VERIFY_HEIGHT VERIFY_DIM3
#define VERIFY_WIDTH VERIFY_DIM2

#define VERIFY_ALL_DIMS(val_dim4, val_dim3, val_dim2, val_dim1, ...)           \
  (verify_dim(4, val_dim4, __VA_ARGS__, NO_ARG) |                              \
   verify_dim(3, val_dim3, __VA_ARGS__, NO_ARG) |                              \
   verify_dim(2, val_dim2, __VA_ARGS__, NO_ARG) |                              \
   verify_dim(1, val_dim1, __VA_ARGS__, NO_ARG))

#define IS_SHAPE_BIAS(tensor)                                                  \
  (verify_dim(4, 1, tensor, NO_ARG) | verify_dim(3, 1, tensor, NO_ARG) |       \
   verify_dim(2, 1, tensor, NO_ARG))

/// Verifies if all tensors have exact same shape and data type and format
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_tensors(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c,
                           const zdnn_ztensor *output) {
  zdnn_status status;
  zdnn_tensor_desc *input_a_tfrmd_desc = input_a->transformed_desc;

  /*
  Parameter patterns:

  input_a | input_b | input_c | output
  --------+---------+---------+-------
    X     |   NULL  |   NULL  |   X
    X     |   X     |   NULL  |   X
    X     |   X     |   X     |   X

  Use input_a's as the "correct" value
  input_b and input_c are NULL when not being used
  */

  // check shapes first

  if ((status =
           VERIFY_ALL_DIMS(input_a_tfrmd_desc->dim4, input_a_tfrmd_desc->dim3,
                           input_a_tfrmd_desc->dim2, input_a_tfrmd_desc->dim1,
                           TENSOR_PARM(output), TENSOR_PARM(input_b),
                           TENSOR_PARM(input_c))) != ZDNN_OK) {
    return status;
  }

  // then check type and format

  if ((status = VERIFY_FIELDS(input_a_tfrmd_desc->type,
                              input_a_tfrmd_desc->format, TENSOR_PARM(output),
                              TENSOR_PARM(input_b), TENSOR_PARM(input_c))) !=
      ZDNN_OK) {
    return status;
  }

  return status;
}

/// Verifies the condition of lstm/gru activation tensors.
///
/// \param[in] mode             LSTM or GRU
/// \param[in] ts_fused         timestep fused for RNN activation call
/// \param[in] bias_add_rnn_op  bias tensor
/// \param[in] prev_state       previous state tensor (prev_c LSTM, prev_h GRU)
/// \param[in] h_output         h_output tensor
/// \param[in] c_output         c_output tensor (LSTM only, NULL if GRU)
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_lstm_or_gru_act_tensors(uint8_t function_code,
                                           const zdnn_ztensor *ts_fused,
                                           const zdnn_ztensor *bias_add_rnn_op,
                                           const zdnn_ztensor *prev_state,
                                           const zdnn_ztensor *h_output,
                                           const zdnn_ztensor *c_output) {
  /*
   DIMENSION REQUIREMENTS (NHWC, DLFLOAT16)
   Legend:
   g = number of gates (4 LSTM or 3 GRU)
   b = number of batches
   s = hidden state size

                   |   shape (dim4, dim3, dim2, dim1)
   ----------------+-------------------------------------
   ts_fused        |   (g,1,b,s)
   bias_add_rnn_op |   (g,1,b,s)
   prev_state      |   (1,1,b,s) (LSTM prev_c, GRU prev_h)
   h_output        |   (1,1,b,s)
   c_output        |   (1,1,b,s) (LSTM only, GRU ignores)
  */

  zdnn_status status;
  uint32_t num_gates = get_func_code_num_gates(function_code);

  // These should match in for all tensors so set the expected to one of them.
  uint32_t exp_dim2 = ts_fused->transformed_desc->dim2;
  uint32_t exp_dim1 = ts_fused->transformed_desc->dim1;
  zdnn_data_types exp_type = ts_fused->transformed_desc->type;
  zdnn_data_formats exp_format = ts_fused->transformed_desc->format;

  // check shapes
  if ((status = VERIFY_DIM4(1, TENSOR_PARM(prev_state),
                            TENSOR_PARM(h_output))) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM4(num_gates, TENSOR_PARM(ts_fused),
                            TENSOR_PARM(bias_add_rnn_op))) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM3(
           1, TENSOR_PARM(ts_fused), TENSOR_PARM(bias_add_rnn_op),
           TENSOR_PARM(prev_state), TENSOR_PARM(h_output))) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM2(
           exp_dim2, TENSOR_PARM(ts_fused), TENSOR_PARM(bias_add_rnn_op),
           TENSOR_PARM(prev_state), TENSOR_PARM(h_output))) != ZDNN_OK) {
    return status;
  }
  if ((status = VERIFY_DIM1(
           exp_dim1, TENSOR_PARM(ts_fused), TENSOR_PARM(bias_add_rnn_op),
           TENSOR_PARM(prev_state), TENSOR_PARM(h_output))) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_DIM4(1, TENSOR_PARM(c_output))) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM3(1, TENSOR_PARM(c_output))) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM2(exp_dim2, TENSOR_PARM(c_output))) != ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM1(exp_dim1, TENSOR_PARM(c_output))) != ZDNN_OK) {
      return status;
    }
  }

  // then check type and format
  if ((status =
           VERIFY_FIELDS(exp_type, exp_format, TENSOR_PARM(ts_fused),
                         TENSOR_PARM(bias_add_rnn_op), TENSOR_PARM(prev_state),
                         TENSOR_PARM(h_output))) != ZDNN_OK) {
    return status;
  }
  if (function_code == NNPA_LSTMACT) {
    if ((status = VERIFY_FIELDS(exp_type, exp_format, TENSOR_PARM(c_output))) !=
        ZDNN_OK) {
      return status;
    }
  }

  // If we reach this, all checks passed. Return OK
  return ZDNN_OK;
}

/// Verifies the condition of fused matmul bias add (broadcast) tensors.
///
/// The following conditions are checked:
///
///  Matmul Op:
///  - The dimension-4-index-size of all input tensors and the output
///    tensor are not all equal.
///
///  Matmul Bcast Op:
///  - The dimension-4-index-size of input tensor 1 and output
///    tensor are not equal.
///  - The dimension-4-index-size of input tensor 2 and input 3 are not
///    equal to 1.
///
///  Common:
///  - The dimension-3-index-size of all input tensors and the output
///    tensor are not equal to 1.
///  - The dimension-2-index-size of input tensor 3 is not equal to 1.
///  - The dimension-2-index-size of input tensor 1 and output tensor
///    are not equal.
///  - The dimension-1-index-size of input tensor 1 is not equal to the
///    dimensions-2-index-size of input tensor 2.
///  - The dimension-1-index-size of input tensor 2, input tensor 3, and
///    output tensor are not equal.
///  - The format of the input tensor differs from the format of the output
///    tensor.
///  - The data type of the input tensors differs from the data type of
///    the output tensor.
///
/// \param[in] uint8_t function_code,
///                   NNPA_MATMUL_OP or NNPA_MATMUL_OP_BCAST23
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
static zdnn_status verify_matmul_op_common(uint8_t function_code,
                                           const zdnn_ztensor *input_a,
                                           const zdnn_ztensor *input_b,
                                           const zdnn_ztensor *input_c,
                                           const zdnn_ztensor *output) {

  zdnn_status status;
  zdnn_tensor_desc *input_a_tfrmd_desc = input_a->transformed_desc;

  // check shapes first
  // For matmul_op, all tensors must have the same number of stacks (dim4)
  if (function_code == NNPA_MATMUL_OP) {
    if ((status = VERIFY_DIM4(input_a_tfrmd_desc->dim4, TENSOR_PARM(input_b),
                              TENSOR_PARM(input_c), TENSOR_PARM(output))) !=
        ZDNN_OK) {
      return status;
    }
    // For matmul_bcast_op, input_a and output tensors must have the same
    // number of stacks (dim4) but input_b and input_c tensors must have a stack
    // dimension of 1 as they are broadcasted over each stack of the input.
  } else {
    if ((status = VERIFY_DIM4(input_a_tfrmd_desc->dim4, TENSOR_PARM(output))) !=
        ZDNN_OK) {
      return status;
    }
    if ((status = VERIFY_DIM4(1, TENSOR_PARM(input_b), TENSOR_PARM(input_c))) !=
        ZDNN_OK) {
      return status;
    }
  }

  if ((status = VERIFY_DIM3(1, TENSOR_PARM(input_a), TENSOR_PARM(input_b),
                            TENSOR_PARM(input_c), TENSOR_PARM(output))) !=
      ZDNN_OK) {
    return status;
  }

  if ((status = VERIFY_DIM2(1, TENSOR_PARM(input_c))) != ZDNN_OK) {
    return status;
  }

  if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim2, TENSOR_PARM(output))) !=
      ZDNN_OK) {
    return status;
  }

  if ((status = VERIFY_DIM2(input_a_tfrmd_desc->dim1, TENSOR_PARM(input_b))) !=
      ZDNN_OK) {
    return status;
  }

  if ((status = VERIFY_DIM1(input_b->transformed_desc->dim1,
                            TENSOR_PARM(input_c), TENSOR_PARM(output))) !=
      ZDNN_OK) {
    return status;
  }

  // then check type and format

  if ((status = VERIFY_FIELDS(input_a_tfrmd_desc->type,
                              input_a_tfrmd_desc->format, TENSOR_PARM(input_b),
                              TENSOR_PARM(input_c), TENSOR_PARM(output))) !=
      ZDNN_OK) {
    return status;
  }

  return status;
}

zdnn_status verify_matmul_op_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const zdnn_ztensor *output) {
  return verify_matmul_op_common(NNPA_MATMUL_OP, input_a, input_b, input_c,
                                 output);
}

zdnn_status verify_matmul_bcast_op_tensors(const zdnn_ztensor *input_a,
                                           const zdnn_ztensor *input_b,
                                           const zdnn_ztensor *input_c,
                                           const zdnn_ztensor *output) {
  return verify_matmul_op_common(NNPA_MATMUL_OP_BCAST23, input_a, input_b,
                                 input_c, output);
}

/// Verifies the condition of input and output tensors for batchnorm operation.
///
/// The following conditions are checked:
///
///  -  The shape of input tensor 1 differs from the shape of the output
///     tensor.
///  -  The format of the input tensor differs from the format of the output
///     tensor.
///  -  The data type of the input tensors differs from the data type of the
///     output tensor.
///  -  Input tensors 1, 2, 3 and the output tensor do not have the same
///     dimension-1-index-size.
///  -  The dimension 2,3 and 4 index sizes of input tensors 2 and 3 are not 1
///
/// \param[in] input_a The first input tensor
/// \param[in] input_b The second input tensor
/// \param[in] input_c The third input tensor
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///
zdnn_status verify_batchnorm_tensors(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     const zdnn_ztensor *output) {

  zdnn_status status;
  zdnn_tensor_desc *input_tfrmd_desc = input_a->transformed_desc;

  // check shapes first

  if ((status = VERIFY_ALL_DIMS(input_tfrmd_desc->dim4, input_tfrmd_desc->dim3,
                                input_tfrmd_desc->dim2, input_tfrmd_desc->dim1,
                                TENSOR_PARM(output))) != ZDNN_OK) {
    return status;
  }

  if ((status =
           VERIFY_DIM1(input_a->transformed_desc->dim1, TENSOR_PARM(input_b),
                       TENSOR_PARM(input_c), TENSOR_PARM(output))) != ZDNN_OK) {
    return status;
  }

  if ((status = (IS_SHAPE_BIAS(TENSOR_PARM(input_b)) |
                 IS_SHAPE_BIAS(TENSOR_PARM(input_c)))) != ZDNN_OK) {
    return status;
  }

  // then check type and format

  if ((status = VERIFY_FIELDS(input_a->transformed_desc->type,
                              input_a->transformed_desc->format,
                              TENSOR_PARM(output))) != ZDNN_OK) {
    return status;
  }

  return status;
}

/// Verifies the condition of input and output tensors for average pool
/// operation.
///
/// \param[in] input input tensor
/// \param[in] padding_type
/// \param[in] kernel_height
/// \param[in] kernel_width
/// \param[in] stride_height
/// \param[in] stride_width
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_STRIDE_PADDING
///         ZDNN_INVALID_STRIDES
///
zdnn_status verify_pool_avg_max_tensors(
    const zdnn_ztensor *input, zdnn_pool_padding padding_type,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_height,
    uint32_t stride_width, const zdnn_ztensor *output) {

  // Convenience variables used later
  zdnn_status status;

  uint32_t input_c_size = input->transformed_desc->dim1;
  uint32_t input_w_size = input->transformed_desc->dim2;
  uint32_t input_h_size = input->transformed_desc->dim3;
  uint32_t input_n_size = input->transformed_desc->dim4;

  uint32_t output_w_size = output->transformed_desc->dim2;
  uint32_t output_h_size = output->transformed_desc->dim3;

  uint32_t expected_output_w_size, expected_output_h_size;

  LOG_DEBUG(
      "%s() - padding_type: %d, input_ztensor->transformed_desc shape: (%d, "
      "%d, %d, %d) (NHWC order), kernel_height: %d, kernel_width: %d, "
      "stride_height: %d, stride_width %d, output_ztensor->transformed_desc "
      "shape: (%d, %d, %d, %d) (NHWC order)",
      __func__, padding_type, input->transformed_desc->dim4, input_h_size,
      input_w_size, input->transformed_desc->dim1, kernel_height, kernel_width,
      stride_height, stride_width, output->transformed_desc->dim4,
      output_h_size, output_w_size, output->transformed_desc->dim1);

  // check tensor shapes first
  if ((status = (VERIFY_DIM4(input_n_size, TENSOR_PARM(output)) |
                 VERIFY_DIM1(input_c_size, TENSOR_PARM(output)))) != ZDNN_OK) {
    return status;
  }

  // Check that input and output have the same type and format
  // Note: If the output data type is invalid, the AIU may raise a
  // condition code before we'd reach this exception condition.
  if ((status = VERIFY_FIELDS(input->transformed_desc->type,
                              input->transformed_desc->format,
                              TENSOR_PARM(output))) != ZDNN_OK) {
    return status;
  }

  // Checks for when strides are 0
  if (stride_width == 0 && stride_height == 0) {
    if (input_w_size != kernel_width) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "When strides are 0, the input tensor's width "
                         "(%d) and kernel_width (%d) must be equal.",
                         input_w_size, kernel_width);
    }
    if (input_h_size != kernel_height) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "When strides are 0, the input tensor's height "
                         "(%d) and kernel_height (%d) must be equal.",
                         input_h_size, kernel_height);
    }
    if (output_w_size != 1 || output_h_size != 1) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "When strides are 0, the output tensor's height "
                         "(%d) and width (%d) must both be 1",
                         output_h_size, output_w_size);
    }
    if (padding_type != VALID_PADDING) {
      return ZDNN_STATUS(
          ZDNN_INVALID_STRIDE_PADDING,
          "When strides are 0, the padding_type must be VALID_PADDING", NULL);
    }
    // Checks that if one stride is nonzero then both must be nonzero.
    // We're following order as described in doc to make future comparing easier
    // so we can't just make this the final "else" condition. This boolean is an
    // XOR and will only be true if one (and only one) of these are nonzero.
  } else if (!stride_width != !stride_height) {
    return ZDNN_STATUS(ZDNN_INVALID_STRIDES,
                       "When either stride is non-zero, then both strides "
                       "must be non-zero. Stride width (%d), Stride height "
                       "(%d)",
                       output_h_size, output_w_size);
    // Checks for when strides are both nonzero
  } else {
    bool check_output_size = true;
    switch (padding_type) {

    case VALID_PADDING:
      if (kernel_width > input_w_size) {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "When VALID_PADDING is used, the the kernel_width (%d) "
            "must not be larger than the input tensor's width (%d) ",
            kernel_width, input_w_size);
      }
      if (kernel_height > input_h_size) {
        return ZDNN_STATUS(
            ZDNN_INVALID_SHAPE,
            "When VALID_PADDING is used, the the kernel_height (%d) "
            "must not be larger than the input tensor's height (%d) ",
            kernel_height, input_h_size);
      }
      expected_output_w_size =
          CEIL(input_w_size - kernel_width + 1, stride_width);
      expected_output_h_size =
          CEIL(input_h_size - kernel_height + 1, stride_height);
      break;

    case SAME_PADDING:
      expected_output_w_size = CEIL(input_w_size, stride_width);
      expected_output_h_size = CEIL(input_h_size, stride_height);
      break;

    default:
      // An invalid padding type raises a condition code from the hardware
      // so it isn't something we need to raise an error for here. However
      // without a type we can't know what to expect for the later output size
      // check. Instead we log a warning and will skip that check.
      LOG_WARN("Not valid padding type (%d)", padding_type);
      check_output_size = false;
      break;
    }

    if (check_output_size) {
      if (output_w_size != expected_output_w_size ||
          output_h_size != expected_output_h_size) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "Expected the output tensor's height (%d) to "
                           "be %d and width (%d) to be %d",
                           output_h_size, expected_output_h_size, output_w_size,
                           expected_output_h_size);
      }
    }
  }

  return ZDNN_STATUS_OK;
}

/// Verifies the condition of input and output tensors for convolution
/// operation.
///
/// \param[in] input input tensor
/// \param[in] kernel input kernel tensor
/// \param[in] bias input bias tensor
/// \param[in] pad_n_act padding type and act function in AIU's
///                      function-specific-parameter-1 format
/// \param[in] stride_height
/// \param[in] stride_width
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_STRIDE_PADDING
///         ZDNN_INVALID_STRIDES
///
zdnn_status verify_conv2d_tensors(const zdnn_ztensor *input,
                                  const zdnn_ztensor *kernel,
                                  const zdnn_ztensor *bias, uint32_t pad_n_act,
                                  uint32_t stride_height, uint32_t stride_width,
                                  uint32_t reserved_n_clipping,
                                  const zdnn_ztensor *output) {

  zdnn_status status;

  // hw doc calls input => input1, kernel => input2, bias => input3
  //              stride_height => dim3_stride, stride_width => dim2_stride
  zdnn_tensor_desc *input_desc = input->transformed_desc,
                   *input_kernel_desc = kernel->transformed_desc,
                   *output_desc = output->transformed_desc;

  func_sp_parm1_conv2d conv2d_parm1;
  conv2d_parm1.val = pad_n_act;

  // The dimension-2, dimension-3, and dimension-4 index sizes of the input3
  // must be 1.
  if ((status = (IS_SHAPE_BIAS(TENSOR_PARM(bias)))) != ZDNN_OK) {
    return status;
  }

  // The dimension-4-index-size of the output must be equal to the
  // dimension-4-index-size of the input1.
  if ((status = VERIFY_DIM4(input_desc->dim4, TENSOR_PARM(output))) !=
      ZDNN_OK) {
    return status;
  }

  // The dimension-1 index size of the output must be equal to the dimension-1
  // index size of the input2 and the dimension-1-index size of the input3.
  if ((status = VERIFY_DIM1(output_desc->dim1, TENSOR_PARM(kernel),
                            TENSOR_PARM(bias))) != ZDNN_OK) {
    return status;
  }

  // The dimension-1 index size of the input1 must be equal to the dimension-2
  // index size of the input2.
  if ((status = VERIFY_DIM1(input_kernel_desc->dim2, TENSOR_PARM(input))) !=
      ZDNN_OK) {
    return status;
  }

  if (!stride_height && !stride_width) { // both zero

    // The input1 dimension-2-index-size must be equal to the
    // dimension-3-index-size of input2.
    if (input_desc->dim2 != input_kernel_desc->dim3) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "input_desc->dim2"
                         " (%d) must be equal to "
                         "input_kernel_desc->dim3"
                         " (%d)\n",
                         input_desc->dim2, input_kernel_desc->dim3);
    }

    // The input1 dimension-3-index-size must be equal to the
    // dimension-4-index-size of input2.
    if (input_desc->dim3 != input_kernel_desc->dim4) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "input_desc->dim3"
                         " (%d) must be equal to "
                         "input_kernel_desc->dim4"
                         " (%d)\n",
                         input_desc->dim3, input_kernel_desc->dim4);
    }

    // The dimension-2-index-size and the dimension-3-index-size of the output
    // must be one.
    if (output_desc->dim2 != 1) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "output_desc->dim2"
                         " (%d) must be 1\n",
                         output_desc->dim2);
    }

    if (output_desc->dim3 != 1) {
      return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                         "output_desc->dim3"
                         " (%d) must be 1\n",
                         output_desc->dim3);
    }

    // The specified padding must be VALID
    if (conv2d_parm1.bits.pad != VALID_PADDING) {
      return ZDNN_STATUS(ZDNN_INVALID_STRIDE_PADDING,
                         "padding must be VALID_PADDING when both "
                         "stride_height (%d) and stride_width (%d) are zero",
                         stride_height, stride_width);
    }

  } else if (stride_height && stride_width) { // both > 0

    switch (conv2d_parm1.bits.pad) {
    case VALID_PADDING:
      // the dimension-2-index-size of the input1 must be greater than or equal
      // to the dimension-3-index-size of input2.
      if (input_desc->dim2 < input_kernel_desc->dim3) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "input_desc->dim2"
                           " (%d) must be greater than "
                           "input_kernel_desc->dim3"
                           " (%d)\n",
                           input_desc->dim2, input_kernel_desc->dim3);
      }

      //  the dimension-3-index-size of the input1 must be greater than or equal
      //  to the dimension-4-index-size of the input2
      if (input_desc->dim3 < input_kernel_desc->dim4) {
        return ZDNN_STATUS(ZDNN_INVALID_SHAPE,
                           "input_desc->dim3"
                           " (%d) must be greater than "
                           "input_kernel_desc->dim4"
                           " (%d)\n",
                           input_desc->dim3, input_kernel_desc->dim4);
      }

      if ((status =
               VERIFY_DIM2(CEIL(input_desc->dim2 - input_kernel_desc->dim3 + 1,
                                stride_width),
                           TENSOR_PARM(output)) |
               VERIFY_DIM3(CEIL(input_desc->dim3 - input_kernel_desc->dim4 + 1,
                                stride_height),
                           TENSOR_PARM(output))) != ZDNN_OK) {

        return status;
      }
      break;
    case SAME_PADDING:
      if ((status = VERIFY_DIM2(CEIL(input_desc->dim2, stride_width),
                                TENSOR_PARM(output)) |
                    VERIFY_DIM3(CEIL(input_desc->dim3, stride_height),
                                TENSOR_PARM(output))) != ZDNN_OK) {
        return status;
      }
      break;
    default:
      // keep going to the next check, the hardware will handle it with function
      // specific RC later
      LOG_WARN("Not valid padding type (%d)", conv2d_parm1.bits.pad);
      break;
    }

  } else { // only either is zero
    return ZDNN_STATUS(ZDNN_INVALID_STRIDES,
                       "either both stride_height (%d) and stride_width (%d) "
                       "must be non-zero or both be must be zero\n",
                       stride_height, stride_width);
  }

  // data type/format of input3 and output should match input1's
  if ((status = VERIFY_FIELDS(input_desc->type, input_desc->format,
                              TENSOR_PARM(bias), TENSOR_PARM(output))) !=
      ZDNN_OK) {
    return status;
  }

  // data type of input2 should match input1's
  // not checking input2's format (should be ZDNN_FORMAT_4DKERNEL), let hardware
  // handle it with reponse code if not
  if ((status = VERIFY_FIELDS(input_desc->type, input_kernel_desc->format,
                              TENSOR_PARM(kernel))) != ZDNN_OK) {
    return status;
  }

  // If activation is set to RELU, check clipping value.
  if (conv2d_parm1.bits.act == CONV2D_ACT_RELU) {
    func_sp_parm4_conv2d conv2d_parm4;
    conv2d_parm4.val = reserved_n_clipping;
    // Clipping value cannot be negative.
    if (conv2d_parm4.bits.clipping_value & 0x8000) {
      return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                         "Clipping value cannot be negative.", NO_ARG);
    }
    // Clipping value cannot be NINF+
    if (conv2d_parm4.bits.clipping_value == 0x7FFF) {
      return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                         "Conversion of clipping value unsuccessful.", NO_ARG);
    }
  }

  return ZDNN_STATUS_OK;
}

/// Verifies the condition of input and output tensors for relu operation.
///
/// \param[in] input input tensor
/// \param[in] reserved_n_clipping reserved and clipping value in AIU's
///                                function-specific-parameter-1 format
/// \param[in] output output tensor
///
/// \return ZDNN_OK
///         ZDNN_INVALID_SHAPE
///         ZDNN_INVALID_TYPE
///         ZDNN_INVALID_FORMAT
///         ZDNN_INVALID_CLIPPING_VALUE
///
zdnn_status verify_relu_tensors(const zdnn_ztensor *input,
                                uint32_t reserved_n_clipping,
                                const zdnn_ztensor *output) {

  zdnn_status status;

  if ((status = verify_tensors(input, NULL, NULL, output)) != ZDNN_OK) {
    return status;
  }

  func_sp_parm1_relu relu_parm1;
  relu_parm1.val = reserved_n_clipping;
  // Clipping value cannot be negative.
  if (relu_parm1.bits.clipping_value & 0x8000) {
    return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                       "Clipping value cannot be negative.", NO_ARG);
  }
  // Clipping value cannot be NINF+
  if (relu_parm1.bits.clipping_value == 0x7FFF) {
    return ZDNN_STATUS(ZDNN_INVALID_CLIPPING_VALUE,
                       "Conversion of clipping value unsuccessful.", NO_ARG);
  }

  return ZDNN_STATUS_OK;
}
