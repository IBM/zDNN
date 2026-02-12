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

#include "testsupport.h"

#include <stdint.h>

/*
 * Tests for reusable scratch-buffer helper APIs.
 */

void setUp(void) {}
void tearDown(void) {}

void test_reserve_scratch_buffer_rejects_null_struct(void) {
  TEST_ASSERT_EQUAL(ZDNN_INVALID_BUFFER,
                    zdnn_reserve_scratch_buffer(NULL, 4096));
}

void test_reserve_scratch_buffer_zero_size_is_noop(void) {
  zdnn_scratch_buffer scratch = {0};

  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_reserve_scratch_buffer(&scratch, 0));
  TEST_ASSERT_NULL(scratch.buffer);
  TEST_ASSERT_EQUAL_UINT64(0, scratch.size);
}

void test_reserve_scratch_buffer_allocates_and_reuses_existing_storage(void) {
  zdnn_scratch_buffer scratch = {0};

  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_reserve_scratch_buffer(&scratch, 1024));
  TEST_ASSERT_NOT_NULL(scratch.buffer);
  TEST_ASSERT_EQUAL_UINT64(1024, scratch.size);
  TEST_ASSERT_EQUAL_UINT64(0,
                           (uintptr_t)scratch.buffer % AIU_PAGESIZE_IN_BYTES);

  void *first_buffer = scratch.buffer;
  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_reserve_scratch_buffer(&scratch, 256));
  TEST_ASSERT_EQUAL_PTR(first_buffer, scratch.buffer);
  TEST_ASSERT_EQUAL_UINT64(1024, scratch.size);

  zdnn_release_scratch_buffer(&scratch);
}

void test_reserve_scratch_buffer_grows_storage_when_needed(void) {
  zdnn_scratch_buffer scratch = {0};

  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_reserve_scratch_buffer(&scratch, 1024));
  TEST_ASSERT_NOT_NULL(scratch.buffer);

  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_reserve_scratch_buffer(&scratch, 8192));
  TEST_ASSERT_NOT_NULL(scratch.buffer);
  TEST_ASSERT_EQUAL_UINT64(8192, scratch.size);
  TEST_ASSERT_EQUAL_UINT64(0,
                           (uintptr_t)scratch.buffer % AIU_PAGESIZE_IN_BYTES);

  zdnn_release_scratch_buffer(&scratch);
}

void test_release_scratch_buffer_resets_state_and_is_idempotent(void) {
  zdnn_scratch_buffer scratch = {0};

  TEST_ASSERT_EQUAL(ZDNN_OK, zdnn_reserve_scratch_buffer(&scratch, 4096));
  TEST_ASSERT_NOT_NULL(scratch.buffer);

  zdnn_release_scratch_buffer(&scratch);
  TEST_ASSERT_NULL(scratch.buffer);
  TEST_ASSERT_EQUAL_UINT64(0, scratch.size);

  zdnn_release_scratch_buffer(&scratch);
  TEST_ASSERT_NULL(scratch.buffer);
  TEST_ASSERT_EQUAL_UINT64(0, scratch.size);

  zdnn_release_scratch_buffer(NULL);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_reserve_scratch_buffer_rejects_null_struct);
  RUN_TEST(test_reserve_scratch_buffer_zero_size_is_noop);
  RUN_TEST(test_reserve_scratch_buffer_allocates_and_reuses_existing_storage);
  RUN_TEST(test_reserve_scratch_buffer_grows_storage_when_needed);
  RUN_TEST(test_release_scratch_buffer_resets_state_and_is_idempotent);

  return UNITY_END();
}
