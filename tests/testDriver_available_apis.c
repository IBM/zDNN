// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2024
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

#include "testsupport.h"

// cppcheck-suppress 	unusedFunction
void setUp(void) { VERIFY_HW_ENV; }

// cppcheck-suppress 	unusedFunction
void tearDown(void) {}

void test_nnpa_add() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_ADD),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_div() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_DIV),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_min() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_MIN),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_max() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_MAX),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_log() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_LOG),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_sig() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_SIGMOID),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_exp() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_EXP),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_tahn() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_TANH),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_batchnorm() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_BATCHNORM),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_meanreduce2d() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_MEANREDUCE2D),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_avgpool2d() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_AVGPOOL2D),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_maxpool2d() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_MAXPOOL2D),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_sub() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_SUB),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_mul() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_MUL),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_gru() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_GRU),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_gelu() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_GELU);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_relu() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_RELU),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_sqrt() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_SQRT);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_invsqrt() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_INVSQRT);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_norm() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_NORM);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_moments() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_MOMENTS);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_layernorm() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_LAYERNORM);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_reduce() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_REDUCE);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_conv2d() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_CONV2D);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_nnpa_softmax() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_SOFTMAX),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_softmax_mask() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_SOFTMAX_MASK);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}
void test_nnpa_matmul_op() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_MATMUL_OP),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_lstm() {
  TEST_ASSERT_MESSAGE(true == is_operation_available(ZDNN_LSTM),
                      "Expected is_operation_available() to return true.");
}

void test_nnpa_leaky_relu() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_LEAKY_RELU);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_transform_with_saturation() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_TRANSFORM_ZTENSOR_WITH_SATURATION);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

void test_transform_quant_ztensor() {
  bool expected_status = !isTelumI();
  bool status = is_operation_available(ZDNN_TRANSFORM_QUANTIZED_ZTENSOR);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == expected_status,
      "is_operation_available() status is %s but expects %s",
      status ? "true" : "false", expected_status ? "true" : "false");
}

// void test_matmul_bcast_op() {
//   bool expected_status = !isTelumI();
//   bool status = is_operation_available(ZDNN_MATMUL_BCAST_OP);
//   TEST_ASSERT_MESSAGE_FORMATTED(
//       status == expected_status,
//       "is_operation_available() status is %s but expects %s",
//       status ? "true" : "false", expected_status ? "true" : "false");
// }

// void test_matmul_transpose_op() {
//   bool expected_status = !isTelumI();
//   bool status = is_operation_available(ZDNN_MATMUL_TRANSPOSE_OP);
//   TEST_ASSERT_MESSAGE_FORMATTED(
//       status == expected_status,
//       "is_operation_available() status is %s but expects %s",
//       status ? "true" : "false", expected_status ? "true" : "false");
// }

// void test_quant_matmul_op() {
//   bool expected_status = !isTelumI();
//   bool status = is_operation_available(ZDNN_QUANTIZED_MATMUL_OP);
//   TEST_ASSERT_MESSAGE_FORMATTED(
//       status == expected_status,
//       "is_operation_available() status is %s but expects %s",
//       status ? "true" : "false", expected_status ? "true" : "false");
// }

// void test_quant_matmul_pre_computed_op() {
//   bool expected_status = !isTelumI();
//   bool status =
//   is_operation_available(ZDNN_QUANTIZED_MATMUL_PRE_COMPUTED_OP);
//   TEST_ASSERT_MESSAGE_FORMATTED(
//       status == expected_status,
//       "is_operation_available() status is %s but expects %s",
//       status ? "true" : "false", expected_status ? "true" : "false");
// }

int main() {
  UNITY_BEGIN();
  RUN_TEST(test_nnpa_add);
  RUN_TEST(test_nnpa_div);
  RUN_TEST(test_nnpa_min);
  RUN_TEST(test_nnpa_max);
  RUN_TEST(test_nnpa_log);
  RUN_TEST(test_nnpa_sig);
  RUN_TEST(test_nnpa_exp);
  RUN_TEST(test_nnpa_tahn);
  RUN_TEST(test_nnpa_batchnorm);
  RUN_TEST(test_nnpa_avgpool2d);
  RUN_TEST(test_nnpa_meanreduce2d);
  RUN_TEST(test_nnpa_maxpool2d);
  RUN_TEST(test_nnpa_moments);
  RUN_TEST(test_nnpa_layernorm);
  RUN_TEST(test_nnpa_reduce);
  RUN_TEST(test_nnpa_sub);
  RUN_TEST(test_nnpa_mul);
  RUN_TEST(test_nnpa_gru);
  RUN_TEST(test_nnpa_gelu);
  RUN_TEST(test_nnpa_relu);
  RUN_TEST(test_nnpa_sqrt);
  RUN_TEST(test_nnpa_invsqrt);
  RUN_TEST(test_nnpa_norm);
  RUN_TEST(test_nnpa_batchnorm);
  RUN_TEST(test_nnpa_avgpool2d);
  RUN_TEST(test_nnpa_meanreduce2d);
  RUN_TEST(test_nnpa_maxpool2d);
  RUN_TEST(test_nnpa_moments);
  RUN_TEST(test_nnpa_layernorm);
  RUN_TEST(test_nnpa_reduce);
  RUN_TEST(test_nnpa_conv2d);
  RUN_TEST(test_nnpa_softmax);
  RUN_TEST(test_nnpa_softmax_mask);
  RUN_TEST(test_nnpa_matmul_op);
  RUN_TEST(test_nnpa_lstm);
  RUN_TEST(test_nnpa_leaky_relu);
  RUN_TEST(test_transform_with_saturation);
  RUN_TEST(test_transform_quant_ztensor);

  return UNITY_END();
}
