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

#include "testsupport.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {

  tol_bfloat.ulps = 64;
  tol_bfloat.epsilon_mult = (0.1 / EPSILON_BFLOAT) + 1;

  tol_fp16.ulps = 64;
  tol_fp16.epsilon_mult = (0.1 / EPSILON_FP16) + 1;

  tol_fp32.ulps = 64 * 16384;
  tol_fp32.epsilon_mult = (0.1 / EPSILON_FLOAT) + 1;

  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

/**
 * Helper macro that given the indices and sizes of a multidimensional array
 * returns equivalent index to a flat representation of the same array. The
 * result is cast to uint64_t as that's the largest number of total elements a
 * ztensor supports as opposed to the single dimension maximum of unint32_t
 *
 * Note: Default usage is for 3D arrays. For 2D arrays, use 0 for the
 * undefined dimension's index and 1 its size.
 */
#define GET_FLAT_IDX(stack, row, col, row_size, col_size)                      \
  (uint64_t)(stack) * (row_size) * (col_size) + (row) * (col_size) + (col)

/**
 * Helper function to print matmul arrays. 3D arrays are printed as separate
 * stacks of 2D arrays.
 */
void print_matmul_array(uint32_t s, uint32_t r, uint32_t c, char *name,
                        float *arr) {
  printf("Printing \"%s\" as %u stack(s) of array[%u][%u]\n", name, s, r, c);
  for (uint32_t i = 0; i < s; i++) {
    printf("\"%s\" stack %u\n", name, i);
    for (uint32_t j = 0; j < r; j++) {
      for (uint32_t k = 0; k < c; k++) {
        printf("%f ", arr[GET_FLAT_IDX(i, j, k, r, c)]);
      }
      printf("\n");
    }
  }
  printf("end \"%s\"\n\n", name);
}

/**
 * Helper function to compute expected output tensor from randomly generated
 * test input arrays.
 *
 * | first      | second     | bias   | result     |
 * | (s, m, n)  | (s, n, p)  | (s, p) | (s, m, p)  |
 *
 */

void gen_test_expected_fp32_array(uint32_t s, uint32_t m, uint32_t n,
                                  uint32_t p, zdnn_data_types type,
                                  float *first, float *second, float *bias,
                                  float *result) {
  for (uint32_t i = 0; i < s; i++) {     // MATRIX from stack
    for (uint32_t j = 0; j < m; j++) {   // ROW of Mat 1
      for (uint32_t k = 0; k < p; k++) { // COL of Mat 2
        uint64_t result_idx = GET_FLAT_IDX(i, j, k, m, p);
        uint64_t bias_idx = GET_FLAT_IDX(i, 0, k, 1, p);

        float cleansed_bias = 0;

        switch (type) {
        case (BFLOAT):
          cleansed_bias = CLEANSE_BFLOAT(bias[bias_idx]);
          break;
        case (FP16):
          cleansed_bias = CLEANSE_FP16(bias[bias_idx]);
          break;
        case (FP32):
          cleansed_bias = CLEANSE_FP32(bias[bias_idx]);
          break;
        default:
          break;
        }

        result[result_idx] = cleansed_bias; // bias add
        BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
          printf("result[%u][%u][%u] = ", i, j, k);
        }
        for (uint32_t l = 0; l < n; l++) { // COL of Mat 1
          uint64_t first_idx = GET_FLAT_IDX(i, j, l, m, n);
          uint64_t second_idx = GET_FLAT_IDX(i, l, k, n, p);

          float cleansed_first = 0;
          float cleansed_second = 0;

          switch (type) {
          case (BFLOAT):
            cleansed_first = CLEANSE_BFLOAT(first[first_idx]);
            cleansed_second = CLEANSE_BFLOAT(second[second_idx]);
            break;
          case (FP16):
            cleansed_first = CLEANSE_FP16(first[first_idx]);
            cleansed_second = CLEANSE_FP16(second[second_idx]);
            break;
          case (FP32):
            cleansed_first = CLEANSE_FP32(first[first_idx]);
            cleansed_second = CLEANSE_FP32(second[second_idx]);
            break;
          default:
            break;
          }

          result[result_idx] += cnvt_1_dlf16_to_fp32(cnvt_1_fp32_to_dlf16(
              cleansed_first * cleansed_second)); // dot product
          // Prints the math that generates each cell of the output.
          BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
            printf("(%f * %f) + ", cleansed_first, cleansed_second);
          }
        }
        BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
          printf("%f = %f\n", cleansed_bias, result[result_idx]);
        }
      }
    }
  }
}

/**
 * Helper function to transpose randomly generated test input arrays.
 *
 * (s, m, n) -> (s, n, m)
 *
 */

void transpose_array(uint32_t s, uint32_t m, uint32_t n, const float *input,
                     float *result) {
  for (uint32_t i = 0; i < s; i++) {     // MATRIX from stack
    for (uint32_t j = 0; j < m; j++) {   // ROW of Mat 1
      for (uint32_t k = 0; k < n; k++) { // COL of Mat 1
        uint64_t input_idx = GET_FLAT_IDX(i, j, k, m, n);
        uint64_t result_idx = GET_FLAT_IDX(i, k, j, n, m);
        result[result_idx] = input[input_idx];
      }
    }
  }
}

/**
 * zdnn_matmul_op_test
 *
 * Handles all the logic to run custom tests.
 *
 * shapes are in interpreted as:
 * - input_a = s x m x n     ZDNN_3DS
 * - input_b = s x n x p     ZDNN_3DS
 * - bias    = s x p         ZDNN_2DS
 * - output  = s x m x p     ZDNN_3DS
 *
 */
void zdnn_matmul_op_test(uint32_t *input_a_shape, uint32_t *input_b_shape,
                         uint32_t *input_bias_shape, uint32_t *output_shape,
                         bool is_stacked, float *input_a, float *input_b,
                         float *bias, bool transpose_a, bool transpose_b,
                         zdnn_matmul_ops op_type, zdnn_status expected_status,
                         float *expected_values) {

  /*
   * Input A Tensor
   */
  zdnn_ztensor *input_a_ztensor =
      alloc_ztensor_with_values(input_a_shape, is_stacked ? ZDNN_3DS : ZDNN_2D,
                                test_datatype, NO_CONCAT, false, input_a);

  /*
   * Input B Tensor
   */
  zdnn_ztensor *input_b_ztensor =
      alloc_ztensor_with_values(input_b_shape, is_stacked ? ZDNN_3DS : ZDNN_2D,
                                test_datatype, NO_CONCAT, false, input_b);

  /*
   * Bias Tensor
   */

  zdnn_ztensor *input_bias_ztensor = alloc_ztensor_with_values(
      input_bias_shape, is_stacked ? ZDNN_2DS : ZDNN_1D, test_datatype,
      NO_CONCAT, false, bias);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor =
      alloc_ztensor_with_values(output_shape, is_stacked ? ZDNN_3DS : ZDNN_2D,
                                test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Get back zDNN test status
   */
  zdnn_status test_status = GENERAL_TESTCASE_FAILURE;
  test_status = zdnn_matmul_transpose_op(input_a_ztensor, input_b_ztensor,
                                         input_bias_ztensor, transpose_a,
                                         transpose_b, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      expected_status == test_status,
      "Expected status %08x from zdnn_matmul_transpose_op() with %d Op but "
      "%08x was returned.",
      expected_status, op_type, test_status);

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    int s = input_b_ztensor->transformed_desc->dim4;
    int m = input_a_ztensor->transformed_desc->dim2;
    int n = input_a_ztensor->transformed_desc->dim1;
    int p = input_b_ztensor->transformed_desc->dim1;
    print_matmul_array(1, m, n, "input_a", input_a);
    print_matmul_array(s, n, p, "input_b", input_b);
    print_matmul_array(s, 1, p, "bias", bias);
    print_matmul_array(s, m, p, "expected_values", expected_values);
  }

  fp_tolerance *tol = NULL;

  switch (output_ztensor->pre_transformed_desc->type) {
  case BFLOAT:
    tol = &tol_bfloat;
    break;
  case FP16:
    tol = &tol_fp16;
    break;
  case FP32:
    tol = &tol_fp32;
    break;
  default:
    break;
    // should never get here
  }

  // Only check expected values if we expected the NNPA call to be successful
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values_adv(output_ztensor, false, expected_values, *tol);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(4, input_a_ztensor, input_b_ztensor, input_bias_ztensor,
                       output_ztensor);
}

/**
 * - MatMul BiasAdd (non-stacked)
 *
 * - Matrix input_a = 4x3 -- Manually Coded Input
 * - Matrix input_b = 3x2 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void zdnn_matmul_biasadd_4x3_by_3x2() {
  // Setup Input A
  uint32_t input_a_shape[] = {4, 3};
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6};

  // Setup Input bias
  uint32_t input_bias_shape[] = {2};
  float input_bias_values[] = {3, 3};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};
  float expected_values[] = {25, 31, 52, 67, 79, 103, 106, 139};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, false, input_a_values, input_b_values,
                      input_bias_values, false, false, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (non-stacked) (transpose_a)
 *
 * - Matrix input_a = 3x4 -- Manually Coded Input
 * - Matrix input_b = 3x2 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void zdnn_matmul_biasadd_3x4_by_3x2() {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4};
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6};

  // Setup Input bias
  uint32_t input_bias_shape[] = {2};
  float input_bias_values[] = {3, 3};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};
  float expected_values[] = {25, 31, 52, 67, 79, 103, 106, 139};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, false, input_a_values, input_b_values,
                      input_bias_values, true, false, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (non-stacked) (transpose_b)
 *
 * - Matrix input_a = 4x3 -- Manually Coded Input
 * - Matrix input_b = 2x3 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void zdnn_matmul_biasadd_4x3_by_2x3() {
  // Setup Input A
  uint32_t input_a_shape[] = {4, 3};
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6};

  // Setup Input bias
  uint32_t input_bias_shape[] = {2};
  float input_bias_values[] = {3, 3};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};
  float expected_values[] = {25, 31, 52, 67, 79, 103, 106, 139};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, false, input_a_values, input_b_values,
                      input_bias_values, false, true, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (non-stacked) (transpose_a and transpose_b)
 *
 * - Matrix input_a = 3x4 -- Manually Coded Input
 * - Matrix input_b = 2x3 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void zdnn_matmul_biasadd_3x4_by_2x3() {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4};
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6};

  // Setup Input bias
  uint32_t input_bias_shape[] = {2};
  float input_bias_values[] = {3, 3};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};
  float expected_values[] = {25, 31, 52, 67, 79, 103, 106, 139};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, false, input_a_values, input_b_values,
                      input_bias_values, true, true, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (stacked)
 *
 * - Matrix input_a = s x m x n --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x n x p --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_biasadd_smn_by_snp(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, m, n};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, n, p};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_a_values,
                               input_b_values, input_bias_values,
                               expected_values);

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, true, input_a_values, input_b_values,
                      input_bias_values, false, false, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (stacked) (transpose_a)
 *
 * - Matrix input_a = s x n x m --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x n x p --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_biasadd_snm_by_snp(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, n, m};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // transpose input so we know can generate expected values
  float input_at_values[num_values];
  transpose_array(s, n, m, input_a_values, input_at_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, n, p};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_at_values,
                               input_b_values, input_bias_values,
                               expected_values);

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, true, input_a_values, input_b_values,
                      input_bias_values, true, false, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (stacked) (transpose_b)
 *
 * - Matrix input_a = s x m x n --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x p x n --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_biasadd_smn_by_spn(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, m, n};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, p, n};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // transpose input so we know can generate expected values
  float input_bt_values[num_values];
  transpose_array(s, p, n, input_b_values, input_bt_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_a_values,
                               input_bt_values, input_bias_values,
                               expected_values);

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, true, input_a_values, input_b_values,
                      input_bias_values, false, true, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

/**
 * - MatMul BiasAdd (stacked) (transpose_a and transpose_b)
 *
 * - Matrix input_a = s x n x m --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x p x n --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_biasadd_snm_by_spn(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, n, m};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // transpose input so we know can generate expected values
  float input_at_values[num_values];
  transpose_array(s, n, m, input_a_values, input_at_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, p, n};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // transpose input so we know can generate expected values
  float input_bt_values[num_values];
  transpose_array(s, p, n, input_b_values, input_bt_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_at_values,
                               input_bt_values, input_bias_values,
                               expected_values);

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_bias_shape,
                      output_shape, true, input_a_values, input_b_values,
                      input_bias_values, true, true, MATMUL_OP_ADDITION,
                      ZDNN_OK, expected_values);
}

void zdnn_matmul_biasadd_3x10x11_by_3x11x2() {
  zdnn_matmul_biasadd_smn_by_snp(3, 10, 11, 2);
  zdnn_matmul_biasadd_snm_by_snp(3, 10, 11, 2);
  zdnn_matmul_biasadd_smn_by_spn(3, 10, 11, 2);
  zdnn_matmul_biasadd_snm_by_spn(3, 10, 11, 2);
}

/**
 * - MatMul Compare (non-stacked)
 *
 * - Matrix input_a = 4x3 -- Manually Coded Input
 * - Matrix input_b = 3x2 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void test_compare_4x3_by_3x2(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {4, 3};
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_c_shape, output_shape,
                      false, input_a_values, input_b_values, input_c_values,
                      false, false, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Compare (non-stacked) (transpose_a)
 *
 * - Matrix input_a = 3x4 -- Manually Coded Input
 * - Matrix input_b = 3x2 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void test_compare_3x4_by_3x2(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4};
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_c_shape, output_shape,
                      false, input_a_values, input_b_values, input_c_values,
                      true, false, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Compare (non-stacked) (transpose_b)
 *
 * - Matrix input_a = 4x3 -- Manually Coded Input
 * - Matrix input_b = 2x3 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void test_compare_4x3_by_2x3(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {4, 3};
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_c_shape, output_shape,
                      false, input_a_values, input_b_values, input_c_values,
                      false, true, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Compare (non-stacked) (transpose_a and transpose_b)
 *
 * - Matrix input_a = 3x4 -- Manually Coded Input
 * - Matrix input_b = 2x3 -- Manually Coded Input
 * - Matrix    bias = 2   -- Manually Coded Input
 * - Matrix  output = 4x2
 */
void test_compare_3x4_by_2x3(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4};
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {4, 2};

  zdnn_matmul_op_test(input_a_shape, input_b_shape, input_c_shape, output_shape,
                      false, input_a_values, input_b_values, input_c_values,
                      true, true, op, ZDNN_OK, exp_vals);
}

void zdnn_matmul_compare_4x3_by_3x2_greater() {
  float is_greater_exp_vals[] = {0, 0, 0, 0, 1, 0, 1, 1};
  test_compare_4x3_by_3x2(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x4_by_3x2(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_4x3_by_2x3(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x4_by_2x3(MATMUL_OP_GREATER, is_greater_exp_vals);
}

void zdnn_matmul_compare_4x3_by_3x2_greater_equal() {
  float is_greater_equal_exp_vals[] = {0, 0, 0, 0, 1, 1, 1, 1};
  test_compare_4x3_by_3x2(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x4_by_3x2(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_4x3_by_2x3(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x4_by_2x3(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
}

void zdnn_matmul_compare_4x3_by_3x2_equal() {
  float is_equal_exp_vals[] = {0, 0, 0, 0, 0, 1, 0, 0};
  test_compare_4x3_by_3x2(MATMUL_OP_EQUAL, is_equal_exp_vals);
  test_compare_3x4_by_3x2(MATMUL_OP_EQUAL, is_equal_exp_vals);
  test_compare_4x3_by_2x3(MATMUL_OP_EQUAL, is_equal_exp_vals);
  test_compare_3x4_by_2x3(MATMUL_OP_EQUAL, is_equal_exp_vals);
}

void zdnn_matmul_compare_4x3_by_3x2_not_equal() {
  float is_not_equal_exp_vals[] = {1, 1, 1, 1, 1, 0, 1, 1};
  test_compare_4x3_by_3x2(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
  test_compare_3x4_by_3x2(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
  test_compare_4x3_by_2x3(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
  test_compare_3x4_by_2x3(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
}

void zdnn_matmul_compare_4x3_by_3x2_lesser_equal() {
  float is_lesser_equal_exp_vals[] = {1, 1, 1, 1, 0, 1, 0, 0};
  test_compare_4x3_by_3x2(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
  test_compare_3x4_by_3x2(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
  test_compare_4x3_by_2x3(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
  test_compare_3x4_by_2x3(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
}

void zdnn_matmul_compare_4x3_by_3x2_lesser() {
  float is_lesser_exp_vals[] = {1, 1, 1, 1, 0, 0, 0, 0};
  test_compare_4x3_by_3x2(MATMUL_OP_LESSER, is_lesser_exp_vals);
  test_compare_3x4_by_3x2(MATMUL_OP_LESSER, is_lesser_exp_vals);
  test_compare_4x3_by_2x3(MATMUL_OP_LESSER, is_lesser_exp_vals);
  test_compare_3x4_by_2x3(MATMUL_OP_LESSER, is_lesser_exp_vals);
}

/**
 * zdnn_matmul_bcast23_op_test
 *
 * Handles all the logic to run custom tests.
 *
 * shapes are in interpreted as:
 * - input_a = s x m x n ZDNN_3DS
 * - input_b = n x p     ZDNN_2D
 * - bias    = p         ZDNN_1D
 * - output  = s x m x p ZDNN_3DS
 *
 */
void zdnn_matmul_bcast23_op_test(uint32_t *input_a_shape,
                                 uint32_t *input_b_shape,
                                 uint32_t *input_bias_shape,
                                 uint32_t *output_shape, float *input_a,
                                 float *input_b, float *bias, bool transpose_a,
                                 bool transpose_b, zdnn_matmul_ops op_type,
                                 zdnn_status expected_status,
                                 float *expected_values) {

  /*
   * Input A Tensor
   */
  zdnn_ztensor *input_a_ztensor = alloc_ztensor_with_values(
      input_a_shape, ZDNN_3DS, test_datatype, NO_CONCAT, false, input_a);

  /*
   * Input B Tensor
   */
  zdnn_ztensor *input_b_ztensor = alloc_ztensor_with_values(
      input_b_shape, ZDNN_2D, test_datatype, NO_CONCAT, false, input_b);

  /*
   * Bias Tensor
   */

  zdnn_ztensor *input_bias_ztensor = alloc_ztensor_with_values(
      input_bias_shape, ZDNN_1D, test_datatype, NO_CONCAT, false, bias);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      output_shape, ZDNN_3DS, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Get back zDNN test status
   */
  zdnn_status test_status = GENERAL_TESTCASE_FAILURE;
  test_status = zdnn_matmul_transpose_op(input_a_ztensor, input_b_ztensor,
                                         input_bias_ztensor, transpose_a,
                                         transpose_b, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      expected_status == test_status,
      "Expected status %08x from zdnn_matmul_transpose_op() with %d Op but "
      "%08x was returned.",
      expected_status, op_type, test_status);

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    int s = input_b_ztensor->transformed_desc->dim4;
    int m = input_a_ztensor->transformed_desc->dim2;
    int n = input_a_ztensor->transformed_desc->dim1;
    int p = input_b_ztensor->transformed_desc->dim1;
    print_matmul_array(1, m, n, "input_a", input_a);
    print_matmul_array(s, n, p, "input_b", input_b);
    print_matmul_array(s, 1, p, "bias", bias);
    print_matmul_array(s, m, p, "expected_values", expected_values);
  }

  fp_tolerance *tol = NULL;

  switch (output_ztensor->pre_transformed_desc->type) {
  case BFLOAT:
    tol = &tol_bfloat;
    break;
  case FP16:
    tol = &tol_fp16;
    break;
  case FP32:
    tol = &tol_fp32;
    break;
  default:
    break;
    // should never get here
  }

  // Only check expected values if we expected the NNPA call to be successful
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values_adv(output_ztensor, false, expected_values, *tol);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(4, input_a_ztensor, input_b_ztensor, input_bias_ztensor,
                       output_ztensor);
}

/**
 * - MatMul Broadcast 23 BiasAdd
 *
 * - Matrix input_a = s x m x n --Randomly Generated Positive/Negative Array
 * - Matrix input_b = 1 x n x p --Randomly Generated Positive/Negative Array
 * - Matrix    bias = 1 x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_smn_by_np(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, m, n};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {n, p};
  num_values = n * p;
  float input_b_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // manually "broadcast" those n*p entries s times across input_b_values[]
  // because gen_test_expected_fp32_array2() doesn't handle broadcast natively
  uint64_t size = n * p * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_b_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_b_values, size);
    tmp_ptr += size;
  }

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {p};
  num_values = p;
  float input_bias_values[s * num_values];
  gen_random_float_array(num_values, input_bias_values);

  size = p * sizeof(float);
  tmp_ptr = (uint8_t *)((uintptr_t)input_bias_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_bias_values, size);
    tmp_ptr += size;
  }

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_a_values,
                               input_b_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_bias_shape, output_shape,
      input_a_values, input_b_values, input_bias_values, false, false,
      MATMUL_BCAST_OP_ADDITION, ZDNN_OK, expected_values);
}

/**
 * - MatMul Broadcast 23 BiasAdd (transpose_a)
 *
 * - Matrix input_a = s x n x m --Randomly Generated Positive/Negative Array
 * - Matrix input_b = 1 x n x p --Randomly Generated Positive/Negative Array
 * - Matrix    bias = 1 x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_snm_by_np(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, n, m};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // transpose input so we know can generate expected values
  float input_at_values[num_values];
  transpose_array(s, n, m, input_a_values, input_at_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {n, p};
  num_values = n * p;
  float input_b_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // manually "broadcast" those n*p entries s times across input_b_values[]
  // because gen_test_expected_fp32_array2() doesn't handle broadcast natively
  uint64_t size = n * p * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_b_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_b_values, size);
    tmp_ptr += size;
  }

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {p};
  num_values = p;
  float input_bias_values[s * num_values];
  gen_random_float_array(num_values, input_bias_values);

  size = p * sizeof(float);
  tmp_ptr = (uint8_t *)((uintptr_t)input_bias_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_bias_values, size);
    tmp_ptr += size;
  }

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_at_values,
                               input_b_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_bias_shape, output_shape,
      input_a_values, input_b_values, input_bias_values, true, false,
      MATMUL_BCAST_OP_ADDITION, ZDNN_OK, expected_values);
}

/**
 * - MatMul Broadcast 23 BiasAdd (transpose_b)
 *
 * - Matrix input_a = s x m x n --Randomly Generated Positive/Negative Array
 * - Matrix input_b = 1 x p x n --Randomly Generated Positive/Negative Array
 * - Matrix    bias = 1 x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_smn_by_pn(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, m, n};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {p, n};
  num_values = n * p;
  float input_b_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // manually "broadcast" those n*p entries s times across input_b_values[]
  // because gen_test_expected_fp32_array2() doesn't handle broadcast natively
  uint64_t size = n * p * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_b_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_b_values, size);
    tmp_ptr += size;
  }

  // transpose input so we know can generate expected values
  float input_bt_values[s * num_values];
  transpose_array(s, p, n, input_b_values, input_bt_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {p};
  num_values = p;
  float input_bias_values[s * num_values];
  gen_random_float_array(num_values, input_bias_values);

  size = p * sizeof(float);
  tmp_ptr = (uint8_t *)((uintptr_t)input_bias_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_bias_values, size);
    tmp_ptr += size;
  }

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_a_values,
                               input_bt_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_bias_shape, output_shape,
      input_a_values, input_b_values, input_bias_values, false, true,
      MATMUL_BCAST_OP_ADDITION, ZDNN_OK, expected_values);
}

/**
 * - MatMul Broadcast 23 BiasAdd (transpose_a and transpose_b)
 *
 * - Matrix input_a = s x n x m --Randomly Generated Positive/Negative Array
 * - Matrix input_b = 1 x p x n --Randomly Generated Positive/Negative Array
 * - Matrix    bias = 1 x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_snm_by_pn(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, n, m};
  num_values = s * m * n;
  float input_a_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // transpose input so we know can generate expected values
  float input_at_values[num_values];
  transpose_array(s, n, m, input_a_values, input_at_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {p, n};
  num_values = n * p;
  float input_b_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // manually "broadcast" those n*p entries s times across input_b_values[]
  // because gen_test_expected_fp32_array2() doesn't handle broadcast natively
  uint64_t size = n * p * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_b_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_b_values, size);
    tmp_ptr += size;
  }

  // transpose input so we know can generate expected values
  float input_bt_values[s * num_values];
  transpose_array(s, p, n, input_b_values, input_bt_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {p};
  num_values = p;
  float input_bias_values[s * num_values];
  gen_random_float_array(num_values, input_bias_values);

  size = p * sizeof(float);
  tmp_ptr = (uint8_t *)((uintptr_t)input_bias_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_bias_values, size);
    tmp_ptr += size;
  }

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_at_values,
                               input_bt_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_bias_shape, output_shape,
      input_a_values, input_b_values, input_bias_values, true, true,
      MATMUL_BCAST_OP_ADDITION, ZDNN_OK, expected_values);
}

void zdnn_matmul_bcast_bias_add_3x10x11_by_11x2() {
  zdnn_matmul_bcast_op_smn_by_np(3, 10, 11, 2);
  zdnn_matmul_bcast_op_snm_by_np(3, 10, 11, 2);
  zdnn_matmul_bcast_op_smn_by_pn(3, 10, 11, 2);
  zdnn_matmul_bcast_op_snm_by_pn(3, 10, 11, 2);
}

/**
 * - MatMul Broadcast 23 Compare
 *
 * - Matrix input_a = 3x4x3 -- Manually Coded Input
 * - Matrix input_b = 3x2   -- Manually Coded Input
 * - Matrix    bias = 2     -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_3x4x3_by_3x2(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4, 3};
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // manually "broadcast" those 3*2 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_b_shape[] = {3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6, 1, 2, 3,
                            4, 5, 6, 1, 2, 3, 4, 5, 6};

  // manually "broadcast" those 2 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, false, false, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Broadcast 23 Compare (transpose_a)
 *
 * - Matrix input_a = 3x3x4 -- Manually Coded Input
 * - Matrix input_b = 3x2   -- Manually Coded Input
 * - Matrix    bias = 2     -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_3x3x4_by_3x2(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 3, 4};
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // manually "broadcast" those 3*2 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_b_shape[] = {3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6, 1, 2, 3,
                            4, 5, 6, 1, 2, 3, 4, 5, 6};

  // manually "broadcast" those 2 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, true, false, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Broadcast 23 Compare (transpose_b)
 *
 * - Matrix input_a = 3x4x3 -- Manually Coded Input
 * - Matrix input_b = 2x3   -- Manually Coded Input
 * - Matrix    bias = 2     -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_3x4x3_by_2x3(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4, 3};
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // manually "broadcast" those 2*3 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_b_shape[] = {2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6, 1, 3, 5,
                            2, 4, 6, 1, 3, 5, 2, 4, 6};

  // manually "broadcast" those 2 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, false, true, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Broadcast 23 Compare (transpose_a and transpose_b)
 *
 * - Matrix input_a = 3x3x4 -- Manually Coded Input
 * - Matrix input_b = 2x3   -- Manually Coded Input
 * - Matrix    bias = 2     -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_3x3x4_by_2x3(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 3, 4};
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // manually "broadcast" those 2*3 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_b_shape[] = {2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6, 1, 3, 5,
                            2, 4, 6, 1, 3, 5, 2, 4, 6};

  // manually "broadcast" those 2 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint32_t input_c_shape[] = {2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast23_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, true, true, op, ZDNN_OK, exp_vals);
}

void zdnn_matmul_bcast_compare_3x4x3_by_3x2_greater() {
  float is_greater_exp_vals[] = {0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
                                 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1};
  test_compare_3x4x3_by_3x2(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x3x4_by_3x2(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x4x3_by_2x3(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x3x4_by_2x3(MATMUL_OP_GREATER, is_greater_exp_vals);
}

void zdnn_matmul_bcast_compare_3x4x3_by_3x2_greater_equal() {
  float is_greater_equal_exp_vals[] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                                       1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  test_compare_3x4x3_by_3x2(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x3x4_by_3x2(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x4x3_by_2x3(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x3x4_by_2x3(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_3x4x3_by_3x2_equal() {
  float is_equal_exp_vals[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                               0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
  test_compare_3x4x3_by_3x2(MATMUL_OP_EQUAL, is_equal_exp_vals);
  test_compare_3x3x4_by_3x2(MATMUL_OP_EQUAL, is_equal_exp_vals);
  test_compare_3x4x3_by_2x3(MATMUL_OP_EQUAL, is_equal_exp_vals);
  test_compare_3x3x4_by_2x3(MATMUL_OP_EQUAL, is_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_3x4x3_by_3x2_not_equal() {
  float is_not_equal_exp_vals[] = {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                   1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};
  test_compare_3x4x3_by_3x2(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
  test_compare_3x3x4_by_3x2(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
  test_compare_3x4x3_by_2x3(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
  test_compare_3x3x4_by_2x3(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_3x4x3_by_3x2_lesser_equal() {
  float is_lesser_equal_exp_vals[] = {1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
                                      0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0};
  test_compare_3x4x3_by_3x2(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
  test_compare_3x3x4_by_3x2(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
  test_compare_3x4x3_by_2x3(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
  test_compare_3x3x4_by_2x3(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_3x4x3_by_3x2_lesser() {
  float is_lesser_exp_vals[] = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                                0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
  test_compare_3x4x3_by_3x2(MATMUL_OP_LESSER, is_lesser_exp_vals);
  test_compare_3x3x4_by_3x2(MATMUL_OP_LESSER, is_lesser_exp_vals);
  test_compare_3x4x3_by_2x3(MATMUL_OP_LESSER, is_lesser_exp_vals);
  test_compare_3x3x4_by_2x3(MATMUL_OP_LESSER, is_lesser_exp_vals);
}

/**
 * zdnn_matmul_bcast1_op_test
 *
 * Handles all the logic to run custom tests.
 *
 * shapes are in interpreted as:
 * - input_a = m x n     ZDNN_2D
 * - input_b = s x n x p ZDNN_3DS
 * - bias    = s x p     ZDNN_2DS
 * - output  = s x m x p ZDNN_3DS
 *
 */
void zdnn_matmul_bcast1_op_test(uint32_t *input_a_shape,
                                uint32_t *input_b_shape,
                                uint32_t *input_bias_shape,
                                uint32_t *output_shape, float *input_a,
                                float *input_b, float *bias, bool transpose_a,
                                bool transpose_b, zdnn_matmul_ops op_type,
                                zdnn_status expected_status,
                                float *expected_values) {

  /*
   * Input A Tensor
   */
  zdnn_ztensor *input_a_ztensor = alloc_ztensor_with_values(
      input_a_shape, ZDNN_2D, test_datatype, NO_CONCAT, false, input_a);

  /*
   * Input B Tensor
   */
  zdnn_ztensor *input_b_ztensor = alloc_ztensor_with_values(
      input_b_shape, ZDNN_3DS, test_datatype, NO_CONCAT, false, input_b);

  /*
   * Bias Tensor
   */

  zdnn_ztensor *input_bias_ztensor = alloc_ztensor_with_values(
      input_bias_shape, ZDNN_2DS, test_datatype, NO_CONCAT, false, bias);

  /*
   * Output Tensor
   */
  zdnn_ztensor *output_ztensor = alloc_ztensor_with_values(
      output_shape, ZDNN_3DS, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  /*
   * Get back zDNN test status
   */
  zdnn_status test_status = GENERAL_TESTCASE_FAILURE;
  test_status = zdnn_matmul_transpose_op(input_a_ztensor, input_b_ztensor,
                                         input_bias_ztensor, transpose_a,
                                         transpose_b, op_type, output_ztensor);
  TEST_ASSERT_MESSAGE_FORMATTED(
      expected_status == test_status,
      "Expected status %08x from zdnn_matmul_transpose_op() with %d Op but "
      "%08x was returned.",
      expected_status, op_type, test_status);

  BEGIN_BLOCK_IF_LOGLEVEL_DEBUG {
    int s = input_b_ztensor->transformed_desc->dim4;
    int m = input_a_ztensor->transformed_desc->dim2;
    int n = input_a_ztensor->transformed_desc->dim1;
    int p = input_b_ztensor->transformed_desc->dim1;
    print_matmul_array(1, m, n, "input_a", input_a);
    print_matmul_array(s, n, p, "input_b", input_b);
    print_matmul_array(s, 1, p, "bias", bias);
    print_matmul_array(s, m, p, "expected_values", expected_values);
  }

  fp_tolerance *tol = NULL;

  switch (output_ztensor->pre_transformed_desc->type) {
  case BFLOAT:
    tol = &tol_bfloat;
    break;
  case FP16:
    tol = &tol_fp16;
    break;
  case FP32:
    tol = &tol_fp32;
    break;
  default:
    break;
    // should never get here
  }

  // Only check expected values if we expected the NNPA call to be successful
  if (expected_status == ZDNN_OK) {
    assert_ztensor_values_adv(output_ztensor, false, expected_values, *tol);
  }

  // All done--clean up the tensor buffers
  free_ztensor_buffers(4, input_a_ztensor, input_b_ztensor, input_bias_ztensor,
                       output_ztensor);
}

/**
 * - MatMul Broadcast 1 BiasAdd
 *
 * - Matrix input_a = 1 x m x n --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x n x p --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_mn_by_snp(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {m, n};
  num_values = m * n;
  float input_a_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // manually "broadcast" those m*n entries s times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint64_t size = m * n * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_a_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_a_values, size);
    tmp_ptr += size;
  }

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, n, p};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_a_values,
                               input_b_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast1_op_test(input_a_shape, input_b_shape, input_bias_shape,
                             output_shape, input_a_values, input_b_values,
                             input_bias_values, false, false,
                             MATMUL_OP_ADDITION, ZDNN_OK, expected_values);
}

/**
 * - MatMul Broadcast 1 BiasAdd (transpose_a)
 *
 * - Matrix input_a = 1 x n x m --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x n x p --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_nm_by_snp(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {n, m};
  num_values = m * n;
  float input_a_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // manually "broadcast" those m*n entries s times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint64_t size = m * n * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_a_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_a_values, size);
    tmp_ptr += size;
  }

  // transpose input so we know can generate expected values
  float input_at_values[s * num_values];
  transpose_array(s, n, m, input_a_values, input_at_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, n, p};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_at_values,
                               input_b_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast1_op_test(input_a_shape, input_b_shape, input_bias_shape,
                             output_shape, input_a_values, input_b_values,
                             input_bias_values, true, false, MATMUL_OP_ADDITION,
                             ZDNN_OK, expected_values);
}

/**
 * - MatMul Broadcast 1 BiasAdd (transpose_b)
 *
 * - Matrix input_a = 1 x m x n --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x p x n --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_mn_by_spn(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {m, n};
  num_values = m * n;
  float input_a_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // manually "broadcast" those m*n entries s times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint64_t size = m * n * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_a_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_a_values, size);
    tmp_ptr += size;
  }

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, p, n};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // transpose input so we know can generate expected values
  float input_bt_values[num_values];
  transpose_array(s, p, n, input_b_values, input_bt_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_a_values,
                               input_bt_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast1_op_test(input_a_shape, input_b_shape, input_bias_shape,
                             output_shape, input_a_values, input_b_values,
                             input_bias_values, false, true, MATMUL_OP_ADDITION,
                             ZDNN_OK, expected_values);
}

/**
 * - MatMul Broadcast 1 BiasAdd (transpose_a and transpose_b)
 *
 * - Matrix input_a = 1 x n x m --Randomly Generated Positive/Negative Array
 * - Matrix input_b = s x p x n --Randomly Generated Positive/Negative Array
 * - Matrix    bias = s x p     --Randomly Generated Positive Array
 * - Matrix  output = s x m x p
 */
void zdnn_matmul_bcast_op_nm_by_spn(uint64_t s, uint64_t m, uint64_t n,
                                    uint64_t p) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {n, m};
  num_values = m * n;
  float input_a_values[s * num_values];
  gen_random_float_array_pos_neg(num_values, input_a_values);

  // manually "broadcast" those m*n entries s times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint64_t size = m * n * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_a_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_a_values, size);
    tmp_ptr += size;
  }

  // transpose input so we know can generate expected values
  float input_at_values[s * num_values];
  transpose_array(s, n, m, input_a_values, input_at_values);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, p, n};
  num_values = s * n * p;
  float input_b_values[num_values];
  gen_random_float_array_pos_neg(num_values, input_b_values);

  // transpose input so we know can generate expected values
  float input_bt_values[num_values];
  transpose_array(s, p, n, input_b_values, input_bt_values);

  // Setup Input bias using random values
  uint32_t input_bias_shape[] = {s, p};
  num_values = s * p;
  float input_bias_values[num_values];
  gen_random_float_array(num_values, input_bias_values);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};
  num_values = s * m * p;

  float expected_values[num_values];
  gen_test_expected_fp32_array(s, m, n, p, test_datatype, input_at_values,
                               input_bt_values, input_bias_values,
                               expected_values);

  zdnn_matmul_bcast1_op_test(input_a_shape, input_b_shape, input_bias_shape,
                             output_shape, input_a_values, input_b_values,
                             input_bias_values, true, true, MATMUL_OP_ADDITION,
                             ZDNN_OK, expected_values);
}

void zdnn_matmul_bcast_bias_add_10x11_by_3x11x2() {
  zdnn_matmul_bcast_op_mn_by_snp(3, 10, 11, 2);
  zdnn_matmul_bcast_op_nm_by_snp(3, 10, 11, 2);
  zdnn_matmul_bcast_op_mn_by_spn(3, 10, 11, 2);
  zdnn_matmul_bcast_op_nm_by_spn(3, 10, 11, 2);
}

/**
 * - MatMul Broadcast 1 Compare
 *
 * - Matrix input_a = 4x3   -- Manually Coded Input
 * - Matrix input_b = 3x3x2 -- Manually Coded Input
 * - Matrix    bias = 3x2   -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_4x3_by_3x3x2(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {4, 3};

  // manually "broadcast" those 4*3 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6, 1, 2, 3,
                            4, 5, 6, 1, 2, 3, 4, 5, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {3, 2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast1_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, false, false, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Broadcast 1 Compare (transpose_a)
 *
 * - Matrix input_a = 3x4   -- Manually Coded Input
 * - Matrix input_b = 3x3x2 -- Manually Coded Input
 * - Matrix    bias = 3x2   -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_3x4_by_3x3x2(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4};

  // manually "broadcast" those 3*4 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 3, 2};
  float input_b_values[] = {1, 2, 3, 4, 5, 6, 1, 2, 3,
                            4, 5, 6, 1, 2, 3, 4, 5, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {3, 2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast1_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, true, false, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Broadcast 1 Compare (transpose_b)
 *
 * - Matrix input_a = 4x3   -- Manually Coded Input
 * - Matrix input_b = 3x2x3 -- Manually Coded Input
 * - Matrix    bias = 3x2   -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_4x3_by_3x2x3(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {4, 3};

  // manually "broadcast" those 4*3 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  float input_a_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6, 1, 3, 5,
                            2, 4, 6, 1, 3, 5, 2, 4, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {3, 2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast1_op_test(
      input_a_shape, input_b_shape, input_c_shape, output_shape, input_a_values,
      input_b_values, input_c_values, false, true, op, ZDNN_OK, exp_vals);
}

/**
 * - MatMul Broadcast 1 Compare (transpose_a and transpose_b)
 *
 * - Matrix input_a = 3x4   -- Manually Coded Input
 * - Matrix input_b = 3x2x3 -- Manually Coded Input
 * - Matrix    bias = 3x2   -- Manually Coded Input
 * - Matrix  output = 3x4x2
 */
void test_compare_3x4_by_3x2x3(zdnn_matmul_ops op, float *exp_vals) {
  // Setup Input A
  uint32_t input_a_shape[] = {3, 4};

  // manually "broadcast" those 3*4 entries 3 times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  float input_a_values[] = {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12,
                            1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12};

  // Setup Input B
  uint32_t input_b_shape[] = {3, 2, 3};
  float input_b_values[] = {1, 3, 5, 2, 4, 6, 1, 3, 5,
                            2, 4, 6, 1, 3, 5, 2, 4, 6};

  // Setup Input bias
  uint32_t input_c_shape[] = {3, 2};
  float input_c_values[] = {50, 100, 50, 100, 50, 100};

  // Output tensor and expected values
  uint32_t output_shape[] = {3, 4, 2};

  // test without transpose
  zdnn_matmul_bcast1_op_test(input_a_shape, input_b_shape, input_c_shape,
                             output_shape, input_a_values, input_b_values,
                             input_c_values, true, true, op, ZDNN_OK, exp_vals);
}

void zdnn_matmul_bcast_compare_4x3_by_3x3x2_greater() {
  float is_greater_exp_vals[] = {0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
                                 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1};
  test_compare_4x3_by_3x3x2(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x4_by_3x3x2(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_4x3_by_3x2x3(MATMUL_OP_GREATER, is_greater_exp_vals);
  test_compare_3x4_by_3x2x3(MATMUL_OP_GREATER, is_greater_exp_vals);
}

void zdnn_matmul_bcast_compare_4x3_by_3x3x2_greater_equal() {
  float is_greater_equal_exp_vals[] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
                                       1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  test_compare_4x3_by_3x3x2(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x4_by_3x3x2(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_4x3_by_3x2x3(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
  test_compare_3x4_by_3x2x3(MATMUL_OP_GREATER_EQUAL, is_greater_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_4x3_by_3x3x2_equal() {
  float is_equal_exp_vals[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                               0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0};
  test_compare_4x3_by_3x3x2(MATMUL_OP_EQUAL, is_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_4x3_by_3x3x2_not_equal() {
  float is_not_equal_exp_vals[] = {1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
                                   1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1};
  test_compare_4x3_by_3x3x2(MATMUL_OP_NOT_EQUAL, is_not_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_4x3_by_3x3x2_lesser_equal() {
  float is_lesser_equal_exp_vals[] = {1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
                                      0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0};
  test_compare_4x3_by_3x3x2(MATMUL_OP_LESSER_EQUAL, is_lesser_equal_exp_vals);
}

void zdnn_matmul_bcast_compare_4x3_by_3x3x2_lesser() {
  float is_lesser_exp_vals[] = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                                0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
  test_compare_4x3_by_3x3x2(MATMUL_OP_LESSER, is_lesser_exp_vals);
}

int main() {
  UNITY_BEGIN();
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_biasadd_4x3_by_3x2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_biasadd_3x4_by_3x2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_biasadd_4x3_by_2x3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_biasadd_3x4_by_2x3);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_biasadd_3x10x11_by_3x11x2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_compare_4x3_by_3x2_greater);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_compare_4x3_by_3x2_greater_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_compare_4x3_by_3x2_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_compare_4x3_by_3x2_not_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_compare_4x3_by_3x2_lesser_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(zdnn_matmul_compare_4x3_by_3x2_lesser);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_bias_add_3x10x11_by_11x2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_3x4x3_by_3x2_greater);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_3x4x3_by_3x2_greater_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_3x4x3_by_3x2_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_3x4x3_by_3x2_not_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_3x4x3_by_3x2_lesser_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_3x4x3_by_3x2_lesser);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_bias_add_10x11_by_3x11x2);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_4x3_by_3x3x2_greater);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_4x3_by_3x3x2_greater_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_4x3_by_3x3x2_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_4x3_by_3x3x2_not_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_4x3_by_3x3x2_lesser_equal);
  RUN_TEST_ALL_DLFLOAT16_PRE_DATATYPES(
      zdnn_matmul_bcast_compare_4x3_by_3x3x2_lesser);
  return UNITY_END();
}
