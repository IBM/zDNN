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

#include "common_quantization.h"
#include "testsupport.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/******************************************************************************
                           default_input
******************************************************************************/
uint32_t default_input_shape[] = {2, 2, 4};
uint32_t bcast_input_shape[] = {2, 4};

/* Visualization of values in shape (s, m, n) order
[[[-1.2135693  28.734085    8.497408  -1.9210271]
  [-23.742136   16.26094  -21.234303    60.51914]],
 [[-1.2135693  28.734085    8.497408  -1.9210271]
  [-23.742136   16.26094  -21.234303    60.51914]]]
*/
float default_input_values[] = {-1.2135693, 28.734085, 8.497408,   -1.9210271,
                                -23.742136, 16.26094,  -21.234303, 60.51914,
                                -1.2135693, 28.734085, 8.497408,   -1.9210271,
                                -23.742136, 16.26094,  -21.234303, 60.51914};

float default_input_min = -100.f;
float default_input_max = 80.f;
float default_input_scale = 0.70588235294f; // (80.0 - -100.0) / 255.0
float default_input_offset = 14.f;
bool default_disable_clipping = false;

/*
a Quantized:
[[[ 12  55  26  11]
  [-20  37 -16 100]],
 [[ 12  55  26  11]
  [-20  37 -16 100]]]
a Dequantized:
[[[ -1.4117647  28.941177    8.470589   -2.1176472]
  [-24.         16.235294  -21.17647    60.705883 ]],
 [[ -1.4117647  28.941177    8.470589   -2.1176472]
  [-24.         16.235294  -21.17647    60.705883 ]]]
 */

/******************************************************************************
                           default_weights
******************************************************************************/
uint32_t default_weights_shape[] = {2, 4, 3};
uint32_t bcast_weights_shape[] = {4, 3};

/* Visualization of weights values in shape (s, n, p) order
[[[  8.909883   -8.496755   3.7517512]
  [-4.1331525  -2.9586632    7.767899]
  [-17.868917  -17.386122  -19.393448]
  [ 4.9785953   3.3447025   6.1003647]],
 [[  8.909883   -8.496755   3.7517512]
  [-4.1331525  -2.9586632    7.767899]
  [-17.868917  -17.386122  -19.393448]
  [ 4.9785953   3.3447025   6.1003647]]]
*/
float default_weights_values[] = {
    8.909883,   -8.496755,  3.7517512,  -4.1331525, -2.9586632, 7.767899,
    -17.868917, -17.386122, -19.393448, 4.9785953,  3.3447025,  6.1003647,
    8.909883,   -8.496755,  3.7517512,  -4.1331525, -2.9586632, 7.767899,
    -17.868917, -17.386122, -19.393448, 4.9785953,  3.3447025,  6.1003647};

float default_weights_min = -20.f;
float default_weights_max = 10.f;
float default_weights_scale = 0.11764705882f; // (10.0 - -20.0) / 255.0
float default_weights_offset = 42.f;

float symmetric_weights_min = -20.f;
float symmetric_weights_max = 20.f;
float symmetric_weights_scale = 0.15686274509f; // (20.0 - -20.0) / 255.0
float symmetric_weights_offset = 0.f;

/*
b Quantized:
[[[ 118  -30   74]
  [   7   17  108]
  [-110 -106 -123]
  [  84   70   94]],
 [[ 118  -30   74]
  [   7   17  108]
  [-110 -106 -123]
  [  84   70   94]]]
b Dequantized:
[[[  8.941176   -8.470589    3.764706 ]
  [ -4.117647   -2.9411764   7.7647057]
  [-17.882353  -17.411764  -19.411764 ]
  [  4.9411764   3.2941177   6.117647 ]],
 [[  8.941176   -8.470589    3.764706 ]
  [ -4.117647   -2.9411764   7.7647057]
  [-17.882353  -17.411764  -19.411764 ]
  [  4.9411764   3.2941177   6.117647 ]]]
*/

/******************************************************************************
                           default_biases
******************************************************************************/
uint32_t default_biases_shape[] = {2, 3};
uint32_t bcast_biases_shape[] = {3};

/* Visualization of bias values in shape (s, p) order
[[478.61835  299.15857  -38.520638],
 [478.61835  299.15857  -38.520638]]
*/
float default_biases_values[] = {478.61835, 299.15857, -38.520638,
                                 478.61835, 299.15857, -38.520638};

float default_biases_min = -500.f;
float default_biases_max = 500.f;
float default_biases_scale = 3.92156862745f; // (500.0 - -500.0) / 255.0
float default_biases_offset = 0.f;

/*
c Quantized:
[[122  76 -10],
 [122  76 -10]]
c Dequantized:
[[478.43137  298.0392   -39.215687],
 [478.43137  298.0392   -39.215687]]
*/

/******************************************************************************
                           default_output
******************************************************************************/
uint32_t default_output_shape[] = {2, 2, 3};

/*
Expected qc_tilde:
[28.6345098  20.96784314  6.6345098]
Expected qy_hw:
[[28.15803922 15.98784314 23.09568627]
 [57.07803922 55.9972549  55.63686275]]
Expected qy_sw:
[[20.30823529 12.99529412 22.97647059]
 [19.86352941 12.55058824 22.53176471]]
Expected qy:
[[ 7.84980392  2.99254902  0.11921569]
 [37.2145098  43.44666667 33.10509804]]
Expected y Quantized:
[[ 8  3  0]
 [37 44 33]]
Expected y Dequantized:
[[ 188.23529    70.588234    0.      ]
 [ 870.58826  1011.7647    776.4706  ]]
*/

/*
Expected Symmetric qc_tilde:
[20.33333333 12.66666667 -1.66666667]
Expected Symmetric qy_hw:
[[ 7.81568627  2.95163399  0.21568627]
 [37.51503268 43.17647059 33.26666667]]
Expected Symmetric qy_sw:
[[0. 0. 0.]
 [0. 0. 0.]]
Expected Symmetric qy:
[[ 7.81568627  2.95163399  0.21568627]
 [37.51503268 43.17647059 33.26666667]]
Expected Symmetric y Quantized:
[[ 8  3  0]
 [37 44 33]]
Expected Symmetric y Dequantized:
[[ 188.23529    70.588234    0.      ]
 [ 870.58826  1011.7647    776.4706  ]]
*/

/******************************************************************************
                          Unity Methods
******************************************************************************/
void setUp(void) {
  VERIFY_HW_ENV;
  VERIFY_PARMBLKFORMAT_1;
}

void tearDown(void) {}

/******************************************************************************
                          Helper Methods
******************************************************************************/

/// Allocates a 4k aligned work area buffer based on the given size and returns
/// a pointer to the memory.
///
/// \param[in] work_area_size size in bytes required for the work_] area
///
/// \return pointer to the work area buffer or throws test failure
///
void *alloc_quantized_matmul_work_area(size_t work_area_size) {

  void *work_area = NULL;
  if (!(work_area = malloc_aligned_4k(work_area_size))) {
    TEST_FAIL_MESSAGE_FORMATTED("malloc_aligned_4k (%zu) failed",
                                work_area_size);
  }
  memset(work_area, 0, work_area_size);
  return work_area;
}

/// Generates and fills the passed scale and offset for the passed min and max.
///
/// \param[in] min the min float value of the range
/// \param[in] max the max float value of the range
/// \param[in] scale pointer to a float that will store the computed scale
/// \param[in] offset pointer to a float that will store the computed offset
///
void gen_scale_and_offset(float min, float max, float *scale, float *offset) {
  *scale = (max - min) / 255.f;

  int zero_point = (int)((max * -128.f - min * 127.f) / (max - min));
  *offset = (float)(zero_point);
}

/**
 * Helper function to compute expected output tensor from randomly generated
 * test input arrays.
 *
 * | first      | second     | bias   | result     |
 * | (s, m, n)  | (s, n, p)  | (s, p) | (s, m, p)  |
 *
 * The idea is to "cleanse" inputs by quantizing them and then dequantizing them
 * to give us float values representative of the quantized values. We can then
 * perform a standard matrix multiplication and quantize the output. This will
 * match the output of a quantized matrix multiplication call.
 *
 * Note that this method only matches when there is no precision loss. We do
 * however have precision loss since computed bias get converted to DLFloat16.
 * This means results may vary slightly, especially since they are rounded.
 */
void gen_test_expected_fp32_array(uint32_t s, uint32_t m, uint32_t n,
                                  uint32_t p, const float *first,
                                  const float *second, const float *bias,
                                  float Sa, float Za, float Sb, float Zb,
                                  float Sc, float Zc, float *result, float *Sy,
                                  float *Zy, zdnn_matmul_ops op_type) {
  float min_result = FLT_MAX;
  float max_result = -FLT_MAX;

  for (uint32_t i = 0; i < s; i++) {     // MATRIX from stack
    for (uint32_t j = 0; j < m; j++) {   // ROW of Mat 1
      for (uint32_t k = 0; k < p; k++) { // COL of Mat 2
        uint64_t result_idx = GET_FLAT_IDX(i, j, k, m, p);
        uint64_t bias_idx = GET_FLAT_IDX(i, 0, k, 1, p);

        float cleansed_bias = CLEANSE_QUANTIZED(bias[bias_idx], Sc, Zc);

        result[result_idx] = op_type == MATMUL_OP_ADDITION ? cleansed_bias : 0;

        for (uint32_t l = 0; l < n; l++) { // COL of Mat 1
          uint64_t first_idx = GET_FLAT_IDX(i, j, l, m, n);
          uint64_t second_idx = GET_FLAT_IDX(i, l, k, n, p);

          float cleansed_first = CLEANSE_QUANTIZED(first[first_idx], Sa, Za);
          float cleansed_second = CLEANSE_QUANTIZED(second[second_idx], Sb, Zb);

          result[result_idx] += (cleansed_first * cleansed_second);
        }

        min_result = MIN(min_result, result[result_idx]);
        max_result = MAX(max_result, result[result_idx]);

        switch (op_type) {
        case MATMUL_OP_GREATER:
          result[result_idx] = result[result_idx] > cleansed_bias ? 1.f : 0.f;
          break;
        case MATMUL_OP_GREATER_EQUAL:
          result[result_idx] = result[result_idx] >= cleansed_bias ? 1.f : 0.f;
          break;
        case MATMUL_OP_EQUAL:
          result[result_idx] = result[result_idx] == cleansed_bias ? 1.f : 0.f;
          break;
        case MATMUL_OP_NOT_EQUAL:
          result[result_idx] = result[result_idx] != cleansed_bias ? 1.f : 0.f;
          break;
        case MATMUL_OP_LESSER_EQUAL:
          result[result_idx] = result[result_idx] <= cleansed_bias ? 1.f : 0.f;
          break;
        case MATMUL_OP_LESSER:
          result[result_idx] = result[result_idx] < cleansed_bias ? 1.f : 0.f;
          break;
        default:
          break;
        }
      }
    }
  }

  // Generate output scale and offset based on min and max result
  gen_scale_and_offset(min_result, max_result, Sy, Zy);

  // When op_type is MATMUL_OP_ADDITION we quantize the output so it matches the
  // returned output.
  if (op_type == MATMUL_OP_ADDITION) {
    for (uint32_t i = 0; i < s; i++) {     // MATRIX from stack
      for (uint32_t j = 0; j < m; j++) {   // ROW of Mat 1
        for (uint32_t k = 0; k < p; k++) { // COL of Mat 2
          uint64_t result_idx = GET_FLAT_IDX(i, j, k, m, p);
          result[result_idx] = QUANTIZE(result[result_idx], *Sy, *Zy);
        }
      }
    }
  }
}

/// Computes the folded bias to be passed to quantized matmul call when
/// operation is MATMUL_OP_ADDITION. Zb should be equal to 0, meaning the
/// correction term for input_a is also equal to 0. This allows the correction
/// term for input_b to be folded into qc_tilde, which removes the need for
/// correction being applied after the quantized matmul call.
///
/// The original equation is:
///
///   qc_tilde = Zy - (Sc / Sy) * Zc + (Sc / Sy) * q_c[j]
///
/// Since input_c is not quantized, we need to replace q_c with the equation
/// to quantize input_c.
///
///   q_c[j] = QUANTIZE(input_c[j], Sc, Zc)
///   qc_tilde = Zy - (Sc / Sy) * Zc + (Sc / Sy) * q_c[j]
///
/// The original equation for the correction term for input_b is:
///
///   M = (Sa * Sb) / Sy
///   term_b = M * Za * sum(q_b[:,j])
///
/// Since input_b is not quantized, we need to replace q_b with the equation
/// to quantize input_b.
///
///   M = (Sa * Sb) / Sy
///   term_b = M * Za * sum(QUANTIZE(input_b[:,j], Sb, Zb))
///
/// This gives us the final equation:
///
///   q_c[j] = QUANTIZE(input_c[j], Sc, Zc)
///   M = (Sa * Sb) / Sy
///   term_b = M * Za * sum(QUANTIZE(input_b[:,j], Sb, Zb))
///   qc_tilde[j] = Zy - (Sc / Sy) * Zc + (Sc / Sy) * q_c[j] - term_b
void pre_compute_folded_bias(const uint32_t s, const uint32_t n,
                             const uint32_t p, const float *input_b_data,
                             const float *input_c_data, const float Sa,
                             const float Za, const float Sb, const float Sc,
                             const float Zc, const float Sy, const float Zy,
                             float *output_data) {
  const float M = (Sa * Sb) / Sy;

  for (uint32_t i = 0; i < s; i++) {
    for (uint32_t j = 0; j < p; j++) {
      float sum_b = 0;
      for (uint32_t k = 0; k < n; k++) {
        uint64_t second_idx = GET_FLAT_IDX(i, k, j, n, p);
        sum_b += QUANTIZE(input_b_data[second_idx], Sb, 0);
      }
      const float term_b = M * Za * sum_b;
      uint64_t bias_idx = GET_FLAT_IDX(i, 0, j, 1, p);
      const float q_c = QUANTIZE(input_c_data[bias_idx], Sc, Zc);
      output_data[bias_idx] = Zy - (Sc / Sy) * Zc + (Sc / Sy) * q_c - term_b;
    }
  }
}

/// Computes the bias to be passed to quantized matmul call when operation is
/// not MATMUL_OP_ADDITION.
///
/// The original equation for qc_tilde is:
///
///   qc_tilde = Sc / (Sa * Sb) * (q_c[j] - Zc) + Za * sum(q_b[:,j])
///
/// Since input_c is not quantized, we need to replace q_c with the equation
/// to quantize input_c.
///
///   q_c[j] = QUANTIZE(input_c[j], Sc, Zc)
///   qc_tilde = Sc / (Sa * Sb) * (q_c[j] - Zc) + Za * sum(q_b[:,j])
///
/// Since input_b is not quantized, we need to replace q_b with the equation
/// to quantize input_b.
///
///   q_c[j] = QUANTIZE(input_c[j], Sc, Zc)
///   term_b = Za * sum(QUANTIZE(input_b[:,j], Sb, 0))
///   qc_tilde = Sc / (Sa * Sb) * (q_c[j] - Zc) + term_b
void pre_compute_comparison_bias(const uint32_t s, const uint32_t n,
                                 const uint32_t p, const float *input_b_data,
                                 const float *input_c_data, const float Sa,
                                 const float Za, const float Sb, const float Sc,
                                 const float Zc, const float Sy, const float Zy,
                                 float *output_data) {
  const float scale = Sc / (Sa * Sb);

  for (uint64_t i = 0; i < s; i++) {
    for (uint64_t j = 0; j < p; j++) {
      float sum_b = 0;
      for (uint32_t k = 0; k < n; k++) {
        uint64_t second_idx = GET_FLAT_IDX(i, k, j, n, p);
        sum_b += QUANTIZE(input_b_data[second_idx], Sb, 0);
      }
      const float term_b = Za * sum_b;
      uint64_t bias_idx = GET_FLAT_IDX(i, 0, j, 1, p);
      const float q_c = QUANTIZE(input_c_data[bias_idx], Sc, Zc);
      output_data[bias_idx] = scale * (q_c - Zc) + term_b;
    }
  }
}

/// Call public API and checks returned status matches expected status. If OK
/// status expected, confirm actual output values match expected values.
///
/// \param[in] exp_status Expected status for the public API call
///
/// \return nothing but throws test failure if values don't match
/// expected or an unexpected failure prevents the test from completing.
///
void test_zdnn_api_quantized_matmul(
    uint32_t *input_shape, zdnn_data_layouts input_layout, float *input_values,
    float a_scale, float a_offset, int8_t clip_min, int8_t clip_max,

    uint32_t *input_weights_shape, zdnn_data_layouts input_weights_layout,
    float *input_weights_values, float b_scale, float b_offset,

    uint32_t *input_biases_shape, zdnn_data_layouts input_biases_layout,
    float *input_biases_values, float c_scale, float c_offset,

    uint32_t *out_shape, zdnn_data_layouts out_layout,

    zdnn_matmul_ops op_type, bool on_the_fly, zdnn_status exp_status,
    bool disable_clipping) {

  // Run test for each pretransformed data type
  zdnn_ztensor *input, *weights, *biases;

  if (on_the_fly) {
    input = alloc_ztensor_with_values(input_shape, input_layout, FP32,
                                      NO_CONCAT, false, input_values);
    input->rec_scale = 1.f / a_scale;
    input->offset = a_offset;
  } else {
    input = alloc_quantized_ztensor_with_values(input_shape, input_layout, FP32,
                                                QUANTIZED_INT8, input_values,
                                                a_scale, a_offset);
  }

  weights = alloc_quantized_ztensor_with_values(
      input_weights_shape, input_weights_layout, INT8, QUANTIZED_WEIGHTS_INT8,
      input_weights_values, b_scale, b_offset);

  biases = alloc_quantized_ztensor_with_values(
      input_biases_shape, input_biases_layout, FP32, QUANTIZED_INT8,
      input_biases_values, c_scale, c_offset);

  // Generate expected output values
  uint32_t s = out_shape[0];
  uint32_t m = out_shape[1];
  uint32_t n = input->transformed_desc->dim1;
  uint32_t p = out_shape[2];

  float *exp_out_values = malloc(s * m * p * sizeof(float));
  float y_scale, y_offset;
  gen_test_expected_fp32_array(s, m, n, p, input_values, input_weights_values,
                               input_biases_values, a_scale, a_offset, b_scale,
                               b_offset, c_scale, c_offset, exp_out_values,
                               &y_scale, &y_offset, op_type);

  // Run API once NULL work_area and again with work_area set.
  for (int work_area_pass = 0; work_area_pass < 2; work_area_pass++) {
    zdnn_ztensor *out;

    out = alloc_quantized_ztensor_with_values(out_shape, out_layout, FP32,
                                              QUANTIZED_DLFLOAT16, NULL,
                                              y_scale, y_offset);

    void *work_area = NULL;

    // Set work_area during second pass
    if (work_area_pass == 1) {
      work_area = alloc_quantized_matmul_work_area(biases->buffer_size);
    }

    zdnn_status status;
    status = zdnn_quantized_matmul_op(input, weights, biases, op_type, clip_min,
                                      clip_max, disable_clipping, false, false,
                                      work_area, out);
    TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status,
                                  "work_area_pass %d call to %s() returned "
                                  "status %08x \"%s\" but expected %08x \"%s\"",
                                  work_area_pass, "zdnn_quantized_matmul_op",
                                  status, zdnn_get_status_message(status),
                                  exp_status,
                                  zdnn_get_status_message(exp_status));

    // Confirm output tensor values match expected values
    if (exp_status == ZDNN_OK) {
      if (op_type == MATMUL_OP_ADDITION) {
        assert_quantized_ztensor_values(out, false, exp_out_values);
      } else {
        assert_quantized_ztensor_compare_values(out, false, exp_out_values);
      }
    }

    // Reset output buffer
    memset(out->buffer, 0, out->buffer_size);

    // dequantize=true
    status = zdnn_quantized_matmul_op(input, weights, biases, op_type, clip_min,
                                      clip_max, disable_clipping, true, false,
                                      work_area, out);
    TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status,
                                  "work_area_pass %d call to %s() returned "
                                  "status %08x \"%s\" but expected %08x \"%s\"",
                                  work_area_pass, "zdnn_quantized_matmul_op",
                                  status, zdnn_get_status_message(status),
                                  exp_status,
                                  zdnn_get_status_message(exp_status));

    // Confirm output tensor values match expected values
    if (exp_status == ZDNN_OK) {
      if (op_type == MATMUL_OP_ADDITION) {
        assert_dequantized_ztensor_values(out, false, exp_out_values);
      } else {
        assert_quantized_ztensor_compare_values(out, false, exp_out_values);
      }
    }

    // Check that work_area was written to on second pass
    if (work_area_pass == 1) {
      free_aligned_4k(work_area);
    }

    free_ztensor_buffers(1, out);
  } // end of work_area_pass loop

  // Free expected output values
  free(exp_out_values);

  // Free input tensors
  free_ztensor_buffers(3, input, weights, biases);
}

/// Call public API and checks returned status matches expected status. If OK
/// status expected, confirm actual output values match expected values.
///
/// \param[in] exp_status Expected status for the public API call
///
/// \return nothing but throws test failure if values don't match
/// expected or an unexpected failure prevents the test from completing.
///
void test_zdnn_api_quantized_matmul_pre_computed(
    uint32_t *input_shape, zdnn_data_layouts input_layout, float *input_values,
    float a_scale, float a_offset, int8_t clip_min, int8_t clip_max,

    uint32_t *input_weights_shape, zdnn_data_layouts input_weights_layout,
    float *input_weights_values, float b_scale, float b_offset,

    uint32_t *input_biases_shape, zdnn_data_layouts input_biases_layout,
    float *input_biases_values, float c_scale, float c_offset,

    uint32_t *out_shape, zdnn_data_layouts out_layout,

    zdnn_matmul_ops op_type, bool on_the_fly, zdnn_status exp_status) {

  // Run test for each pretransformed data type
  zdnn_ztensor *input, *weights, *biases;

  if (on_the_fly) {
    input = alloc_ztensor_with_values(input_shape, input_layout, FP32,
                                      NO_CONCAT, false, input_values);
    input->rec_scale = 1.f / a_scale;
    input->offset = a_offset;
  } else {
    input = alloc_quantized_ztensor_with_values(input_shape, input_layout, FP32,
                                                QUANTIZED_INT8, input_values,
                                                a_scale, a_offset);
  }

  weights = alloc_quantized_ztensor_with_values(
      input_weights_shape, input_weights_layout, INT8, QUANTIZED_WEIGHTS_INT8,
      input_weights_values, b_scale, b_offset);

  // Generate expected output values
  uint32_t s = out_shape[0];
  uint32_t m = out_shape[1];
  uint32_t n = input->transformed_desc->dim1;
  uint32_t p = out_shape[2];

  float *exp_out_values = malloc(s * m * p * sizeof(float));
  float y_scale, y_offset;
  gen_test_expected_fp32_array(s, m, n, p, input_values, input_weights_values,
                               input_biases_values, a_scale, a_offset, b_scale,
                               b_offset, c_scale, c_offset, exp_out_values,
                               &y_scale, &y_offset, op_type);

  // Pre-compute bias values
  const uint64_t bias_s = input_biases_layout == ZDNN_2DS ? s : 1;
  const uint64_t num_elements = bias_s * p;

  float *computed_biases_values = malloc(num_elements * sizeof(float));
  if (op_type == MATMUL_OP_ADDITION) {
    pre_compute_folded_bias(
        bias_s, n, p, input_weights_values, input_biases_values,
        CLEANSE_FP32(a_scale), CLEANSE_FP32(a_offset), CLEANSE_FP32(b_scale),
        CLEANSE_FP32(c_scale), CLEANSE_FP32(c_offset), CLEANSE_FP32(y_scale),
        CLEANSE_FP32(y_offset), computed_biases_values);
  } else {
    pre_compute_comparison_bias(
        bias_s, n, p, input_weights_values, input_biases_values,
        CLEANSE_FP32(a_scale), CLEANSE_FP32(a_offset), CLEANSE_FP32(b_scale),
        CLEANSE_FP32(c_scale), CLEANSE_FP32(c_offset), CLEANSE_FP32(y_scale),
        CLEANSE_FP32(y_offset), computed_biases_values);
  }

  biases =
      alloc_ztensor_with_values(input_biases_shape, input_biases_layout, FP32,
                                NO_CONCAT, false, computed_biases_values);
  biases->rec_scale = 1.f / c_scale;
  biases->offset = c_offset;

  zdnn_ztensor *out;

  out = alloc_quantized_ztensor_with_values(out_shape, out_layout, FP32,
                                            QUANTIZED_DLFLOAT16, NULL, y_scale,
                                            y_offset);
  // pre_computed=true
  zdnn_status status =
      zdnn_quantized_matmul_op(input, weights, biases, op_type, clip_min,
                               clip_max, false, false, true, NULL, out);
  TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status,
                                "call to %s() returned status %08x \"%s\" but "
                                "expected %08x \"%s\"",
                                "zdnn_quantized_matmul_op", status,
                                zdnn_get_status_message(status), exp_status,
                                zdnn_get_status_message(exp_status));

  // Confirm output tensor values match expected values
  if (exp_status == ZDNN_OK) {
    if (op_type == MATMUL_OP_ADDITION) {
      assert_quantized_ztensor_values(out, false, exp_out_values);
    } else {
      assert_quantized_ztensor_compare_values(out, false, exp_out_values);
    }
  }

  memset(out->buffer, 0, out->buffer_size);

  // dequantize=true
  // pre_computed=true
  status = zdnn_quantized_matmul_op(input, weights, biases, op_type, clip_min,
                                    clip_max, false, true, true, NULL, out);
  TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status,
                                "call to %s() returned status %08x \"%s\" but "
                                "expected %08x \"%s\"",
                                "zdnn_quantized_matmul_op", status,
                                zdnn_get_status_message(status), exp_status,
                                zdnn_get_status_message(exp_status));

  // Confirm output tensor values match expected values
  if (exp_status == ZDNN_OK) {
    if (op_type == MATMUL_OP_ADDITION) {
      assert_dequantized_ztensor_values(out, false, exp_out_values);
    } else {
      assert_quantized_ztensor_compare_values(out, false, exp_out_values);
    }
  }

  // Free expected output and computes bias values
  free(exp_out_values);
  free(computed_biases_values);

  // Free input/ouput tensors
  free_ztensor_buffers(4, input, weights, biases, out);
}

/**
 * - Quantized MatMul BiasAdd (stacked)
 *
 * - Matrix input_a = s x m x n --Randomly Generated Array
 * - Matrix input_b = s x n x p --Randomly Generated Array
 * - Matrix    bias = s x p     --Randomly Generated Array
 * - Matrix  output = s x m x p
 */
void quantized_matmul_smn_by_snp(uint64_t s, uint64_t m, uint64_t n, int64_t p,
                                 zdnn_matmul_ops op_type, bool symmetric,
                                 bool on_the_fly, bool pre_compute) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, m, n};
  num_values = s * m * n;
  float *input_a_values = malloc(num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, n, p};
  num_values = s * n * p;
  float *input_b_values = malloc(num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = symmetric ? 20.f : 10.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {s, p};
  num_values = s * p;
  float *input_c_values = malloc(num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};

  if (pre_compute) {
    test_zdnn_api_quantized_matmul_pre_computed(
        input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
        INT8_MAX,

        input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

        input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

        output_shape, ZDNN_3DS,

        op_type, on_the_fly, ZDNN_OK);
  } else {
    test_zdnn_api_quantized_matmul(
        input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
        INT8_MAX,

        input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

        input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

        output_shape, ZDNN_3DS,

        op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
  }

  free(input_a_values);
  free(input_b_values);
  free(input_c_values);
}

/**
 * - Quantized MatMul BiasAdd (bcast1)
 *
 * - Matrix input_a = m x n     --Randomly Generated Array
 * - Matrix input_b = s x n x p --Randomly Generated Array
 * - Matrix    bias = s x p     --Randomly Generated Array
 * - Matrix  output = s x m x p
 */
void quantized_matmul_mn_by_snp(uint64_t s, uint64_t m, uint64_t n, uint64_t p,
                                zdnn_matmul_ops op_type, bool symmetric,
                                bool on_the_fly, bool pre_compute) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {m, n};
  num_values = m * n;
  float *input_a_values = malloc(s * num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // manually "broadcast" those m*n entries s times across input_a_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint64_t size = num_values * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_a_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_a_values, size);
    tmp_ptr += size;
  }

  // Setup Input B using random values
  uint32_t input_b_shape[] = {s, n, p};
  num_values = s * n * p;
  float *input_b_values = malloc(num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = symmetric ? 20.f : 10.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {s, p};
  num_values = s * p;
  float *input_c_values = malloc(num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};

  if (pre_compute) {
    test_zdnn_api_quantized_matmul_pre_computed(
        input_a_shape, ZDNN_2D, input_a_values, a_scale, a_offset, INT8_MIN,
        INT8_MAX,

        input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

        input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

        output_shape, ZDNN_3DS,

        op_type, on_the_fly, ZDNN_OK);
  } else {
    test_zdnn_api_quantized_matmul(
        input_a_shape, ZDNN_2D, input_a_values, a_scale, a_offset, INT8_MIN,
        INT8_MAX,

        input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

        input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

        output_shape, ZDNN_3DS,

        op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
  }

  free(input_a_values);
  free(input_b_values);
  free(input_c_values);
}

/**
 * - Quantized MatMul BiasAdd (bcast23)
 *
 * - Matrix input_a = s x m x n --Randomly Generated Array
 * - Matrix input_b = n x p     --Randomly Generated Array
 * - Matrix    bias = p         --Randomly Generated Array
 * - Matrix  output = s x m x p
 */
void quantized_matmul_smn_by_np(uint64_t s, uint64_t m, uint64_t n, uint64_t p,
                                zdnn_matmul_ops op_type, bool symmetric,
                                bool on_the_fly, bool pre_compute) {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {s, m, n};
  num_values = s * m * n;
  float *input_a_values = malloc(num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {n, p};
  num_values = n * p;
  float *input_b_values = malloc(s * num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = symmetric ? 20.f : 10.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // manually "broadcast" those n*p entries s times across input_b_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  uint64_t size = num_values * sizeof(float);
  uint8_t *tmp_ptr = (uint8_t *)((uintptr_t)input_b_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_b_values, size);
    tmp_ptr += size;
  }

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {p};
  num_values = p;
  float *input_c_values = malloc(s * num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // manually "broadcast" those p entries s times across input_c_values[]
  // because gen_test_expected_fp32_array() doesn't handle broadcast natively
  size = num_values * sizeof(float);
  tmp_ptr = (uint8_t *)((uintptr_t)input_c_values + size);
  for (uint64_t i = 1; i < s; i++) {
    memcpy((void *)tmp_ptr, (void *)input_c_values, size);
    tmp_ptr += size;
  }

  // Setup Output and expected values
  uint32_t output_shape[] = {s, m, p};

  if (pre_compute) {
    test_zdnn_api_quantized_matmul_pre_computed(
        input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
        INT8_MAX,

        input_b_shape, ZDNN_2D, input_b_values, b_scale, b_offset,

        input_c_shape, ZDNN_1D, input_c_values, c_scale, c_offset,

        output_shape, ZDNN_3DS,

        op_type, on_the_fly, ZDNN_OK);
  } else {
    test_zdnn_api_quantized_matmul(
        input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
        INT8_MAX,

        input_b_shape, ZDNN_2D, input_b_values, b_scale, b_offset,

        input_c_shape, ZDNN_1D, input_c_values, c_scale, c_offset,

        output_shape, ZDNN_3DS,

        op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
  }

  free(input_a_values);
  free(input_b_values);
  free(input_c_values);
}

/******************************************************************************
                              BiasAdd Tests
******************************************************************************/
void quantized_matmul_biasadd_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      default_weights_scale, default_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      default_weights_scale, default_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      default_weights_scale, default_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

// Quantized MatMul with symmetric weights (Zb == 0), which will fold correction
// term for input_a into bias
void quantized_matmul_biasadd_symmetric() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_symmetric_no_clipping() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;
  bool disable_clipping = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, disable_clipping);
}

void quantized_matmul_biasadd_bcast1_symmetric() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_bcast1_symmetric_no_clipping() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;
  bool disable_clipping = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, disable_clipping);
}

void quantized_matmul_biasadd_bcast23_symmetric() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_bcast23_symmetric_no_clipping() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;
  bool disable_clipping = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, disable_clipping);
}

// Quantized MatMul with unquantized input, which will quantize the input on the
// fly
void quantized_matmul_biasadd_on_the_fly() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      default_weights_scale, default_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_bcast1_on_the_fly() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      default_weights_scale, default_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_bcast23_on_the_fly() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      default_weights_scale, default_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_biasadd_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_biasadd_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_biasadd_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_biasadd_symmetric_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_biasadd_symmetric_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_biasadd_symmetric_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_biasadd_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_biasadd_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_biasadd_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

/******************************************************************************
                              Compare Tests
******************************************************************************/
void quantized_matmul_greater_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_greater_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_greater_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_greater_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_greater_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_greater_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_not_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_not_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_not_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_lesser_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_lesser_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_lesser_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_lesser_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_lesser_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_lesser_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_greater_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_greater_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_greater_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_greater_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_greater_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_greater_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_not_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_not_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_not_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_lesser_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_lesser_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_lesser_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_lesser_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_lesser_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_on_the_fly_lesser_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK, default_disable_clipping);
}

void quantized_matmul_greater_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_greater_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_greater_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_not_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_not_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_not_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_lesser_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_lesser_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_greater_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_greater_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_greater_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_not_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_not_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_not_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_lesser_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_lesser_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_lesser_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = false;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

/******************************************************************************
                          Pre-Computed BiasAdd Tests
******************************************************************************/
void quantized_matmul_pre_comp_biasadd() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_biasadd_bcast1() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_biasadd_bcast23() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

// Quantized MatMul with unquantized input, which will quantize the input on the
// fly
void quantized_matmul_pre_comp_biasadd_on_the_fly() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_biasadd_bcast1_on_the_fly() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_biasadd_bcast23_on_the_fly() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_biasadd_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_biasadd_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_biasadd_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_biasadd_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_biasadd_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_biasadd_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

/******************************************************************************
                          Pre-Computed Compare Tests
******************************************************************************/
void quantized_matmul_pre_comp_greater_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_greater_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_greater_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_greater_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_greater_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_greater_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_not_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_not_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_not_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_lesser_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_lesser_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_lesser_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_lesser_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_lesser_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_lesser_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = false;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_greater_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_greater_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_greater_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_greater_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_greater_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_greater_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_not_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_not_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_not_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_lesser_equal_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_lesser_equal_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_lesser_equal_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_lesser_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_lesser_bcast1_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      bcast_input_shape, ZDNN_2D, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      default_weights_shape, ZDNN_3DS, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      default_biases_shape, ZDNN_2DS, default_biases_values,
      default_biases_scale, default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_on_the_fly_lesser_bcast23_basic() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool on_the_fly = true;

  test_zdnn_api_quantized_matmul_pre_computed(
      default_input_shape, ZDNN_3DS, default_input_values, default_input_scale,
      default_input_offset, INT8_MIN, INT8_MAX,

      bcast_weights_shape, ZDNN_2D, default_weights_values,
      symmetric_weights_scale, symmetric_weights_offset,

      bcast_biases_shape, ZDNN_1D, default_biases_values, default_biases_scale,
      default_biases_offset,

      default_output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_OK);
}

void quantized_matmul_pre_comp_greater_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_greater_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_greater_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_not_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_not_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_not_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_equal_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_lesser_equal_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_equal_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_lesser_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = false;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_greater_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_greater_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_greater_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_GREATER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_not_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_not_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_not_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_NOT_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_equal_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_lesser_equal_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_equal_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER_EQUAL;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_on_the_fly_2x20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                              pre_compute);
}

void quantized_matmul_pre_comp_lesser_on_the_fly_20x40_by_2x40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_mn_by_snp(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_pre_comp_lesser_on_the_fly_2x20x40_by_40x30() {
  zdnn_matmul_ops op_type = MATMUL_OP_LESSER;
  bool symmetric = true;
  bool on_the_fly = true;
  bool pre_compute = true;

  quantized_matmul_smn_by_np(2, 20, 40, 30, op_type, symmetric, on_the_fly,
                             pre_compute);
}

void quantized_matmul_invalid_op() {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {2, 20, 40};
  num_values = 2 * 20 * 40;
  float *input_a_values = malloc(num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {2, 40, 30};
  num_values = 2 * 40 * 30;
  float *input_b_values = malloc(num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = 20.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {2, 30};
  num_values = 2 * 30;
  float *input_c_values = malloc(num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // Setup Output and expected values
  uint32_t output_shape[] = {2, 20, 30};

  // Manually set invalid op_type
  zdnn_matmul_ops op_type = 7;

  test_zdnn_api_quantized_matmul(
      input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
      INT8_MAX,

      input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

      input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

      output_shape, ZDNN_3DS,

      op_type, true, ZDNN_FUNC_RC_F000, default_disable_clipping);

  free(input_a_values);
  free(input_b_values);
  free(input_c_values);
}

void quantized_matmul_invalid_format() {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {2, 20, 40};
  num_values = 2 * 20 * 40;
  float *input_a_values = malloc(num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {2, 40, 30};
  num_values = 2 * 40 * 30;
  float *input_b_values = malloc(num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = 20.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {2, 30};
  num_values = 2 * 30;
  float *input_c_values = malloc(num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // Setup Output and expected values
  uint32_t output_shape[] = {2, 20, 30};
  num_values = 2 * 20 * 30;

  float *exp_out_values = malloc(num_values * sizeof(float));
  float y_scale, y_offset;
  gen_test_expected_fp32_array(2, 20, 40, 30, input_a_values, input_b_values,
                               input_c_values, a_scale, a_offset, b_scale,
                               b_offset, c_scale, c_offset, exp_out_values,
                               &y_scale, &y_offset, MATMUL_OP_ADDITION);

  // Setup ztensors
  zdnn_ztensor *input, *weights, *biases, *out;

  // Manually set invalid format for input
  input = alloc_quantized_ztensor_with_values(
      input_a_shape, ZDNN_3DS, INT8, QUANTIZED_WEIGHTS_INT8, input_a_values,
      a_scale, a_offset);

  weights = alloc_quantized_ztensor_with_values(
      input_b_shape, ZDNN_3DS, INT8, QUANTIZED_WEIGHTS_INT8, input_b_values,
      b_scale, b_offset);

  biases = alloc_quantized_ztensor_with_values(input_c_shape, ZDNN_2DS, FP32,
                                               QUANTIZED_INT8, input_c_values,
                                               c_scale, c_offset);

  out = alloc_quantized_ztensor_with_values(output_shape, ZDNN_3DS, FP32,
                                            QUANTIZED_DLFLOAT16, NULL, y_scale,
                                            y_offset);
  // dequantize=true
  zdnn_status status = zdnn_quantized_matmul_op(
      input, weights, biases, MATMUL_OP_ADDITION, INT8_MIN, INT8_MAX,
      default_disable_clipping, true, false, NULL, out);

  TEST_ASSERT_MESSAGE_FORMATTED(status == ZDNN_FUNC_RC_F001,
                                "call to zdnn_quantized_matmul_op() returned "
                                "status %08x \"%s\" but expected %08x \"%s\"",
                                status, zdnn_get_status_message(status),
                                ZDNN_FUNC_RC_F001,
                                zdnn_get_status_message(ZDNN_FUNC_RC_F001));

  // Free ztensors
  free_ztensor_buffers(4, input, weights, biases, out);

  // Free data buffers
  free(input_a_values);
  free(input_b_values);
  free(input_c_values);
  free(exp_out_values);
}

void quantized_matmul_invalid_M() {
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {2, 20, 40};
  num_values = 2 * 20 * 40;
  float *input_a_values = malloc(num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {2, 40, 30};
  num_values = 2 * 40 * 30;
  float *input_b_values = malloc(num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = 20.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {2, 30};
  num_values = 2 * 30;
  float *input_c_values = malloc(num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // Setup Output and expected values
  uint32_t output_shape[] = {2, 20, 30};

  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;

  // Manually set invalid scale, which will cause invalid M value to be
  // computed
  b_scale = 0;

  test_zdnn_api_quantized_matmul(
      input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
      INT8_MAX,

      input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

      input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

      output_shape, ZDNN_3DS,

      op_type, true, ZDNN_FUNC_RC_F002, default_disable_clipping);

  free(input_a_values);
  free(input_b_values);
  free(input_c_values);
}

void quantized_matmul_pre_comp_invalid_Zb() {

  zdnn_matmul_ops op_type = MATMUL_OP_ADDITION;
  bool symmetric = false; // force zB != 0.f
  bool on_the_fly = false;
  uint64_t num_values = 0;

  // Setup Input A using random values
  uint32_t input_a_shape[] = {2, 20, 40};
  num_values = 2 * 20 * 40;
  float *input_a_values = malloc(num_values * sizeof(float));
  float a_min = -100.f;
  float a_max = 80.f;
  gen_random_float_array_range(num_values, input_a_values, a_min, a_max);
  float a_scale, a_offset;
  gen_scale_and_offset(a_min, a_max, &a_scale, &a_offset);

  // Setup Input B using random values
  uint32_t input_b_shape[] = {2, 40, 30};
  num_values = 2 * 40 * 30;
  float *input_b_values = malloc(num_values * sizeof(float));
  float b_min = -20.f;
  float b_max = symmetric ? 20.f : 10.f;
  gen_random_float_array_range(num_values, input_b_values, b_min, b_max);
  float b_scale, b_offset;
  gen_scale_and_offset(b_min, b_max, &b_scale, &b_offset);

  // Setup Input bias using random values
  uint32_t input_c_shape[] = {2, 30};
  num_values = 2 * 30;
  float *input_c_values = malloc(num_values * sizeof(float));
  float c_min = -500.f;
  float c_max = 500.f;
  gen_random_float_array_range(num_values, input_c_values, c_min, c_max);
  float c_scale, c_offset;
  gen_scale_and_offset(c_min, c_max, &c_scale, &c_offset);

  // Setup Output and expected values
  uint32_t output_shape[] = {2, 20, 30};

  test_zdnn_api_quantized_matmul_pre_computed(
      input_a_shape, ZDNN_3DS, input_a_values, a_scale, a_offset, INT8_MIN,
      INT8_MAX,

      input_b_shape, ZDNN_3DS, input_b_values, b_scale, b_offset,

      input_c_shape, ZDNN_2DS, input_c_values, c_scale, c_offset,

      output_shape, ZDNN_3DS,

      op_type, on_the_fly, ZDNN_INVALID_OFFSET);
}

int main() {
  UNITY_BEGIN();

  /*
   * Quantized Bias Tests
   */

  // BiasAdd tests
  RUN_TEST(quantized_matmul_biasadd_basic);
  RUN_TEST(quantized_matmul_biasadd_bcast1_basic);
  RUN_TEST(quantized_matmul_biasadd_bcast23_basic);

  // Symmetric weights test
  RUN_TEST(quantized_matmul_biasadd_symmetric);
  RUN_TEST(quantized_matmul_biasadd_bcast1_symmetric);
  RUN_TEST(quantized_matmul_biasadd_bcast23_symmetric);
  // Symmetric weights test - no clipping
  RUN_TEST(quantized_matmul_biasadd_symmetric_no_clipping);
  RUN_TEST(quantized_matmul_biasadd_bcast1_symmetric_no_clipping);
  RUN_TEST(quantized_matmul_biasadd_bcast23_symmetric_no_clipping);

  // Quantize on the fly tests
  RUN_TEST(quantized_matmul_biasadd_on_the_fly);
  RUN_TEST(quantized_matmul_biasadd_bcast1_on_the_fly);
  RUN_TEST(quantized_matmul_biasadd_bcast23_on_the_fly);

  // BiasAdd tests (random)
  RUN_TEST(quantized_matmul_biasadd_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_biasadd_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_biasadd_2x20x40_by_40x30);

  // Symmetric weights test (random)
  RUN_TEST(quantized_matmul_biasadd_symmetric_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_biasadd_symmetric_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_biasadd_symmetric_2x20x40_by_40x30);

  // Quantize on the fly tests (random)
  RUN_TEST(quantized_matmul_biasadd_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_biasadd_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_biasadd_on_the_fly_2x20x40_by_40x30);

  // Compare tests (always symmetric weights)
  RUN_TEST(quantized_matmul_greater_basic);
  RUN_TEST(quantized_matmul_greater_bcast1_basic);
  RUN_TEST(quantized_matmul_greater_bcast23_basic);
  RUN_TEST(quantized_matmul_greater_equal_basic);
  RUN_TEST(quantized_matmul_greater_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_greater_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_equal_basic);
  RUN_TEST(quantized_matmul_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_not_equal_basic);
  RUN_TEST(quantized_matmul_not_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_not_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_lesser_equal_basic);
  RUN_TEST(quantized_matmul_lesser_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_lesser_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_lesser_basic);
  RUN_TEST(quantized_matmul_lesser_bcast1_basic);
  RUN_TEST(quantized_matmul_lesser_bcast23_basic);

  // Compare quantized on the fly tests (always symmetric weights)
  RUN_TEST(quantized_matmul_on_the_fly_greater_basic);
  RUN_TEST(quantized_matmul_on_the_fly_greater_bcast1_basic);
  RUN_TEST(quantized_matmul_on_the_fly_greater_bcast23_basic);
  RUN_TEST(quantized_matmul_on_the_fly_greater_equal_basic);
  RUN_TEST(quantized_matmul_on_the_fly_greater_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_on_the_fly_greater_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_on_the_fly_equal_basic);
  RUN_TEST(quantized_matmul_on_the_fly_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_on_the_fly_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_on_the_fly_not_equal_basic);
  RUN_TEST(quantized_matmul_on_the_fly_not_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_on_the_fly_not_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_on_the_fly_lesser_equal_basic);
  RUN_TEST(quantized_matmul_on_the_fly_lesser_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_on_the_fly_lesser_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_on_the_fly_lesser_basic);
  RUN_TEST(quantized_matmul_on_the_fly_lesser_bcast1_basic);
  RUN_TEST(quantized_matmul_on_the_fly_lesser_bcast23_basic);

  // Compare tests (random) (always symmetric weights)
  RUN_TEST(quantized_matmul_greater_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_greater_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_not_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_not_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_not_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_lesser_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_lesser_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_2x20x40_by_40x30);

  // Compare  quantized on the fly tests (random) (always symmetric weights)
  RUN_TEST(quantized_matmul_greater_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_greater_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_greater_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_not_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_not_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_not_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_lesser_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_lesser_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_lesser_on_the_fly_2x20x40_by_40x30);

  /*
   * Pre-Computed Bias Tests
   */
  // BiasAdd test (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_biasadd);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_bcast1);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_bcast23);

  // BiasAdd quantized on the fly tests (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_biasadd_on_the_fly);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_bcast1_on_the_fly);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_bcast23_on_the_fly);

  // BiasAdd tests (random) (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_biasadd_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_2x20x40_by_40x30);

  // BiasAdd quantized on the fly tests (random) (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_biasadd_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_biasadd_on_the_fly_2x20x40_by_40x30);

  // Compare tests (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_greater_basic);
  RUN_TEST(quantized_matmul_pre_comp_greater_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_greater_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_lesser_basic);
  RUN_TEST(quantized_matmul_pre_comp_lesser_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_lesser_bcast23_basic);

  // Compare quantized on the fly tests (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_greater_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_greater_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_greater_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_greater_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_greater_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_greater_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_not_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_not_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_not_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_lesser_equal_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_lesser_equal_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_lesser_equal_bcast23_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_lesser_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_lesser_bcast1_basic);
  RUN_TEST(quantized_matmul_pre_comp_on_the_fly_lesser_bcast23_basic);

  // Compare tests (random) (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_greater_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_2x20x40_by_40x30);

  // Compare quantized on the fly tests (random) (always symmetric weights)
  RUN_TEST(quantized_matmul_pre_comp_greater_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(
      quantized_matmul_pre_comp_greater_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_greater_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_not_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(
      quantized_matmul_pre_comp_lesser_equal_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_equal_on_the_fly_2x20x40_by_40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_on_the_fly_2x20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_on_the_fly_20x40_by_2x40x30);
  RUN_TEST(quantized_matmul_pre_comp_lesser_on_the_fly_2x20x40_by_40x30);

  RUN_TEST(quantized_matmul_invalid_op);
  RUN_TEST(quantized_matmul_invalid_format);
  RUN_TEST(quantized_matmul_invalid_M);
  RUN_TEST(quantized_matmul_pre_comp_invalid_Zb);

  return UNITY_END();
}
