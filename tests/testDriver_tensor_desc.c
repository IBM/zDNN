// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
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

void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

void tearDown(void) {}

// convenience routine for init and verify (pre-transformed)
void set_and_verify_pre_transformed_descriptor(uint32_t dims[],
                                               zdnn_data_layouts layout,
                                               zdnn_data_types type,
                                               zdnn_status exp_status,
                                               char *error_msg) {
  zdnn_status status;
  zdnn_tensor_desc pre_tfrmd_desc;

  zdnn_init_pre_transformed_desc(layout, type, &pre_tfrmd_desc, dims[0],
                                 dims[1], dims[2], dims[3]);
  status = verify_pre_transformed_descriptor(&pre_tfrmd_desc);

  TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status, "%s (%08x)", error_msg,
                                status);
}

// convenience routine for init and verify (transformed)
void set_and_verify_transformed_descriptor(
    uint32_t dims[], zdnn_data_layouts layout, zdnn_data_types type,
    zdnn_data_formats format, zdnn_status exp_status, char *error_msg) {
  zdnn_status status;
  zdnn_tensor_desc tfrmd_desc;

  init_transformed_desc(layout, type, format, &tfrmd_desc, dims[0], dims[1],
                        dims[2], dims[3]);

  status = verify_transformed_descriptor(&tfrmd_desc);

  TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status, "%s (%08x)", error_msg,
                                status);
}

void verify_dims() {

  uint32_t max_dim_size = zdnn_get_nnpa_max_dim_idx_size();

  uint32_t zero_dim[ZDNN_MAX_DIMS] = {0, 1, 1, 1};
  uint32_t limit_minus1[ZDNN_MAX_DIMS] = {1, max_dim_size - 1, 1, 1};
  uint32_t at_limit[ZDNN_MAX_DIMS] = {1, 1, max_dim_size, 1};
  uint32_t limit_plus1[ZDNN_MAX_DIMS] = {1, 1, max_dim_size + 1, 1};

  set_and_verify_transformed_descriptor(
      zero_dim, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE,
      ZDNN_INVALID_SHAPE, "Not returning ZDNN_INVALID_SHAPE for 0 dim tensor");
  set_and_verify_transformed_descriptor(
      limit_minus1, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE, ZDNN_OK,
      "Not returning ZDNN_OK for below dims limit tensor");
  set_and_verify_transformed_descriptor(
      at_limit, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE, ZDNN_OK,
      "Not returning ZDNN_OK for at dims limit tensor");
  set_and_verify_transformed_descriptor(
      limit_plus1, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE,
      ZDNN_INVALID_SHAPE,
      "Not returning ZDNN_INVALID_SHAPE for above dims limit tensor");
}

void verify_layout() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};

  // only pre-transformed descriptor cares about layout
  set_and_verify_pre_transformed_descriptor(
      dims, ZDNN_NHWC, test_datatype, ZDNN_OK,
      "Not returning ZDNN_OK for pre-transformed with ZDNN_NHWC");
}

void verify_max_tensor_size() {

  uint32_t max_dim_size = zdnn_get_nnpa_max_dim_idx_size();

  // try to come up with dim3 so that (1, dim3, max_dim_size, max_dim_size)
  // would sit right at the MAX TENSOR SIZE limit
  uint32_t dim3 =
      zdnn_get_nnpa_max_tensor_size() / (max_dim_size / AIU_STICKS_PER_PAGE) /
      (max_dim_size / AIU_2BYTE_CELLS_PER_STICK) / AIU_PAGESIZE_IN_BYTES;

  unsigned int limit_minus1[ZDNN_MAX_DIMS] = {1, dim3, max_dim_size - 1,
                                              max_dim_size};
  unsigned int at_limit[ZDNN_MAX_DIMS] = {1, dim3, max_dim_size, max_dim_size};
  unsigned int limit_plus1[ZDNN_MAX_DIMS] = {1, dim3, max_dim_size + 1,
                                             max_dim_size};

  set_and_verify_transformed_descriptor(
      limit_minus1, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE, ZDNN_OK,
      "Not returning ZDNN_OK for below tensor size limit tensor");
  set_and_verify_transformed_descriptor(
      at_limit, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE, ZDNN_OK,
      "Not returning ZDNN_OK for at tensor size limit tensor");
  set_and_verify_transformed_descriptor(
      limit_plus1, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE,
      ZDNN_INVALID_SHAPE,
      "Not returning ZDNN_INVALID_SHAPE for above tensor size limit tensor");
}

void verify_datatype_pre_tranformed() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};

  set_and_verify_transformed_descriptor(
      dims, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DFEATURE, ZDNN_INVALID_TYPE,
      "Not returning ZDNN_INVALID_TYPE with ZDNN_NHWC");
}

void verify_datatype_tranformed() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};

  set_and_verify_pre_transformed_descriptor(
      dims, ZDNN_4D, test_datatype, ZDNN_INVALID_TYPE,
      "Not returning ZDNN_INVALID_TYPE with ZDNN_4D");
}

void verify_generated_format() {
  zdnn_tensor_desc pre_tfrmd_feature_desc, tfrmd_feature_desc;
  zdnn_tensor_desc pre_tfrmd_kernel_desc, tfrmd_kernel_desc;

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, test_datatype,
                                 &pre_tfrmd_feature_desc, 1, 1, 1, 1);
  zdnn_init_pre_transformed_desc(ZDNN_HWCK, test_datatype,
                                 &pre_tfrmd_kernel_desc, 1, 1, 1, 1);

  zdnn_generate_transformed_desc(&pre_tfrmd_feature_desc, &tfrmd_feature_desc);
  zdnn_generate_transformed_desc(&pre_tfrmd_kernel_desc, &tfrmd_kernel_desc);

  TEST_ASSERT_MESSAGE(tfrmd_feature_desc.format == ZDNN_FORMAT_4DFEATURE,
                      "tfrmd_feature_desc doesn't have correct format set");
  TEST_ASSERT_MESSAGE(tfrmd_kernel_desc.format == ZDNN_FORMAT_4DKERNEL,
                      "tfrmd_kernel_desc doesn't have correct format set");
}

#define BAD_FORMAT 255
#define BAD_LAYOUT 255

void format_undefined_fail() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};
  set_and_verify_transformed_descriptor(
      dims, ZDNN_NHWC, test_datatype, BAD_FORMAT, ZDNN_INVALID_FORMAT,
      "BAD_FORMAT doesn't yield ZDNN_INVALID_FORMAT");
}

void format_feature_layout_notagree_fail() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};
  set_and_verify_transformed_descriptor(
      dims, ZDNN_HWCK, test_datatype, ZDNN_FORMAT_4DFEATURE,
      ZDNN_INVALID_LAYOUT,
      "ZDNN_FORMAT_4DFEATURE + ZDNN_HWCK doesn't yield ZDNN_INVALID_LAYOUT");
}

void format_kernel_layout_notagree_fail() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};
  set_and_verify_transformed_descriptor(
      dims, ZDNN_NHWC, test_datatype, ZDNN_FORMAT_4DKERNEL, ZDNN_INVALID_LAYOUT,
      "ZDNN_FORMAT_4DKERNEL + ZDNN_NHWC doesn't yield ZDNN_INVALID_LAYOUT");
}

void format_feature_layout_undefined_fail() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};
  set_and_verify_transformed_descriptor(
      dims, BAD_LAYOUT, test_datatype, ZDNN_FORMAT_4DFEATURE,
      ZDNN_INVALID_LAYOUT,
      "ZDNN_FORMAT_4DFEATURE + undefined layout doesn't yield "
      "ZDNN_INVALID_LAYOUT");
}

void format_kernel_layout_undefined_fail() {
  uint32_t dims[ZDNN_MAX_DIMS] = {1, 1, 1, 1};
  set_and_verify_transformed_descriptor(
      dims, BAD_LAYOUT, test_datatype, ZDNN_FORMAT_4DKERNEL,
      ZDNN_INVALID_LAYOUT,
      "ZDNN_FORMAT_4DKERNEL + undefined layout doesn't yield "
      "ZDNN_INVALID_LAYOUT");
}

void verify_ztensor_slicing(uint32_t num_slices, uint32_t *shape,
                            zdnn_data_layouts layout, size_t buffer_size,
                            zdnn_status exp_status) {
  uint64_t num_elements;
  switch (layout) {
  // 1D isn't valid as it has no dim4. Used for negative test case.
  case (ZDNN_1D):
    num_elements = shape[0];
    break;
  case (ZDNN_2DS):
    num_elements = shape[0] * shape[1];
    break;
  case (ZDNN_3DS):
    num_elements = shape[0] * shape[1] * shape[2];
    break;
  case (ZDNN_4D):
  case (ZDNN_NHWC):
  case (ZDNN_NCHW):
    num_elements = shape[0] * shape[1] * shape[2] * shape[3];
    break;
  default:
    TEST_FAIL_MESSAGE_FORMATTED(
        "I'm dreadfully sorry but I don't seem to know how to deal with a %s "
        "layout. Could you teach me?",
        get_data_layout_str(layout));
    break;
  }
  uint64_t num_slice_elements = num_elements / num_slices;

  float values[num_elements];
  gen_random_float_array(num_elements, values);

  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      shape, layout, test_datatype, NO_CONCAT, false, values);
  // Print out the sliced ztensor
  BEGIN_BLOCK_IF_LOGLEVEL_TRACE {
    printf("%s() with type %s: dumpdata_ztensor of unsliced input\n", __func__,
           get_data_type_str(test_datatype));
    dumpdata_ztensor(input_ztensor, AS_FLOAT, false);
  }

  // Make copies of the original input to confirm it isn't altered later.
  zdnn_ztensor copy_input_ztensor;
  zdnn_tensor_desc copy_pre_trfmd_desc;
  zdnn_tensor_desc copy_trfmd_desc;
  memcpy(&copy_input_ztensor, input_ztensor, sizeof(zdnn_ztensor));
  memcpy(&copy_pre_trfmd_desc, input_ztensor->pre_transformed_desc,
         sizeof(zdnn_tensor_desc));
  memcpy(&copy_trfmd_desc, input_ztensor->transformed_desc,
         sizeof(zdnn_tensor_desc));

  // Create output structs
  zdnn_tensor_desc output_pre_tfrmd_desc[num_slices];
  zdnn_tensor_desc output_tfrmd_desc[num_slices];
  zdnn_ztensor output_ztensors[num_slices];

  // Slice the input and if we expect it to succeed, check that values in each
  // slice matches the expected values for that slice.
  for (uint32_t slice = 0; slice < num_slices; slice++) {
    zdnn_status status = ztensor_slice_dim4(
        input_ztensor, slice, buffer_size, &output_pre_tfrmd_desc[slice],
        &output_tfrmd_desc[slice], &output_ztensors[slice]);
    TEST_ASSERT_MESSAGE_FORMATTED(status == exp_status,
                                  "ztensor_slice_dim4() on slice %u failed, "
                                  "status = %08x (%s)",
                                  slice, status,
                                  zdnn_get_status_message(status));

    // Only test that output values are valid in positive test cases
    if (exp_status == ZDNN_OK) {

      // Print out the sliced ztensor
      BEGIN_BLOCK_IF_LOGLEVEL_TRACE {
        printf("%s() with type %s: dumpdata_ztensor of slice %u\n", __func__,
               get_data_type_str(test_datatype), slice);
        dumpdata_ztensor(&output_ztensors[slice], AS_FLOAT, false);
      }

      // Check output buffer_size matches the specified value or calculated
      // value if a size wasn't specified.
      size_t expected_buffer_size;
      if (buffer_size) {
        expected_buffer_size = buffer_size;
      } else {
        expected_buffer_size =
            zdnn_getsize_ztensor(input_ztensor->transformed_desc) / num_slices;
      }
      TEST_ASSERT_MESSAGE_FORMATTED(
          expected_buffer_size == output_ztensors[slice].buffer_size,
          "expected sliced buffer_size to be %" PRIu64 " but found %" PRIu64,
          expected_buffer_size, output_ztensors[slice].buffer_size);

      // Check that slice's values match the expected portion of the input
      assert_ztensor_values(&output_ztensors[slice], false,
                            &values[slice * num_slice_elements]);
    }
  }

  // Confirm input structs weren't altered during slicing
  TEST_ASSERT_MESSAGE(
      memcmp(input_ztensor, &copy_input_ztensor, sizeof(zdnn_ztensor)) == 0,
      "input_ztensor was unexpectedly altered");

  TEST_ASSERT_MESSAGE(
      memcmp(input_ztensor->pre_transformed_desc, &copy_pre_trfmd_desc,
             sizeof(zdnn_tensor_desc)) == 0,
      "input_ztensor->pre_transformed_desc was unexpectedly altered");

  TEST_ASSERT_MESSAGE(
      memcmp(input_ztensor->transformed_desc, &copy_trfmd_desc,
             sizeof(zdnn_tensor_desc)) == 0,
      "input_ztensor->transformed_desc was  unexpectedly altered");

  // Cleanup allocations
  free(input_ztensor);
}

void test_slicing_specified_buffer() {
  uint32_t num_slices = 5;
  uint32_t shape[] = {num_slices, 2049};

  size_t specified_buffer = 135168;
  verify_ztensor_slicing(num_slices, shape, ZDNN_2DS, specified_buffer,
                         ZDNN_OK);
}

void test_slicing_fail_input_has_only_one_dim4() {
  uint32_t num_slices = 1;
  uint32_t shape[] = {num_slices, 2049};

  verify_ztensor_slicing(num_slices, shape, ZDNN_2DS, 0, ZDNN_INVALID_SHAPE);
}

void test_slicing_fail_too_many_slices() {
  uint32_t num_slices = 2;
  uint32_t shape[] = {num_slices, 2049};

  // Create input ztensor
  zdnn_ztensor *input_ztensor = alloc_ztensor_with_values(
      shape, ZDNN_2DS, test_datatype, NO_CONCAT, true, ZERO_ARRAY);

  // Create output structs
  zdnn_tensor_desc output_pre_tfrmd_desc;
  zdnn_tensor_desc output_tfrmd_desc;
  zdnn_ztensor output_ztensors;

  // idx is 0 indexed so this should fail because it's too large
  uint32_t slice_idx = num_slices;

  // Confirm expected failure status
  zdnn_status status =
      ztensor_slice_dim4(input_ztensor, slice_idx, 0, &output_pre_tfrmd_desc,
                         &output_tfrmd_desc, &output_ztensors);
  TEST_ASSERT_MESSAGE_FORMATTED(
      status == ZDNN_INVALID_SHAPE,
      "ztensor_slice_dim4() on slice_idx %u failed, status = %08x (%s)",
      slice_idx, status, zdnn_get_status_message(status));
}

void test_slicing_1D_fail() {
  uint32_t num_slices = 2;
  uint32_t shape[] = {num_slices};

  verify_ztensor_slicing(num_slices, shape, ZDNN_1D, 0, ZDNN_INVALID_LAYOUT);
}

void test_slicing_2DS_5x2049() {
  uint32_t num_slices = 5;
  uint32_t shape[] = {num_slices, 2049};

  verify_ztensor_slicing(num_slices, shape, ZDNN_2DS, 0, ZDNN_OK);
}

void test_slicing_3DS_5x33x65() {
  uint32_t num_slices = 5;
  uint32_t shape[] = {num_slices, 33, 65};

  verify_ztensor_slicing(num_slices, shape, ZDNN_3DS, 0, ZDNN_OK);
}

// ------------------------------------------------------------------------------------------------

int main(void) {
  UNITY_BEGIN();

  RUN_TEST_ALL_TFRMD_DATATYPES(verify_dims);
  RUN_TEST_ALL_DATATYPES(verify_layout);
  RUN_TEST_ALL_TFRMD_DATATYPES(verify_max_tensor_size);

  // test all data-types possible
  RUN_TEST_ALL_DATATYPES(verify_datatype_pre_tranformed);
  RUN_TEST_ALL_TFRMD_DATATYPES(verify_datatype_tranformed);

  RUN_TEST_ALL_TFRMD_DATATYPES(verify_generated_format);

  RUN_TEST_ALL_TFRMD_DATATYPES(format_undefined_fail);
  RUN_TEST_ALL_TFRMD_DATATYPES(format_feature_layout_notagree_fail);
  RUN_TEST_ALL_TFRMD_DATATYPES(format_kernel_layout_notagree_fail);
  RUN_TEST_ALL_TFRMD_DATATYPES(format_feature_layout_undefined_fail);
  RUN_TEST_ALL_TFRMD_DATATYPES(format_kernel_layout_undefined_fail);

  RUN_TEST_ALL_DATATYPES(test_slicing_specified_buffer);
  RUN_TEST_ALL_DATATYPES(test_slicing_fail_input_has_only_one_dim4);
  RUN_TEST_ALL_DATATYPES(test_slicing_fail_too_many_slices);
  RUN_TEST_ALL_DATATYPES(test_slicing_1D_fail);
  RUN_TEST_ALL_DATATYPES(test_slicing_2DS_5x2049);
  RUN_TEST_ALL_DATATYPES(test_slicing_3DS_5x33x65);

  return UNITY_END();
}
