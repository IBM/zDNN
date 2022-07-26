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
#include <stdlib.h>
#include <string.h>

// External users only need to specify between FWD, BWD, and BIDIR. However our
// directional_rnn() needs more detail. FWD vs BWD controls the order the
// timestep input is processed along with UNI vs BIDIR which affects how we move
// over the hn_output.
typedef enum rnn_internal_direction {
  UNI_FWD,
  UNI_BWD,
  BIDIR_FWD,
  BIDIR_BWD,
} rnn_internal_direction;

// Named indices for numbers passed between the internal methods
typedef enum rnn_integer_indices {
  TS,
  BATCH,
  HID_SIZE,
  IN_PAD,
  GATES,
  SLICEABLE_INPUTS,
  NUM_INTEGER_INDICES // Not an index, used to set size of the array later.
} rnn_integer_indices;

// Must match order in sliceable_inputs[]!
// Named indices for sliceable ztensors passed in by the user.
typedef enum rnn_user_zten_indices {
  H0,
  IN_WEIGHTS,
  IN_BIAS,
  HID_WEIGHTS,
  HID_BIAS,
  NUM_INPUTS_GRU, // Not a tensor, used to set size of the array later.
  C0 = NUM_INPUTS_GRU,
  NUM_INPUTS_LSTM // Not a tensor, used to set size of the array later.
} rnn_user_zten_indices;

// Named indices for ztensors created internally during a RNN call.
typedef enum rnn_internal_zten_indices {
  FUSED,
  TS_FUSED,
  BIAS_ADD,
  PREV_H_OUT,
  TS_H_OUT,
  PREV_C_OUT,
  NUM_INTERNAL_ZTENS_GRU, // Not a ztensor, used to set size of the array later.
  TS_C_OUT = NUM_INTERNAL_ZTENS_GRU,
  NUM_INTERNAL_ZTENS_LSTM // Not a ztensor, used to set size of the array later.
} rnn_internal_zten_indices;

// Named indices for descriptors created internally during a RNN call. These
// descriptors do not affect the work_area size.
typedef enum rnn_internal_desc_indices {
  RNN_IN_TSFUSED_BIASADD_DESC,
  NUM_INTERNAL_DESCS
} rnn_internal_desc_indices; // Not a descriptor, used to set size of the array.

// Named indices for descriptors created internally during a RNN call. These
// descriptors influence the size of the work_area.
typedef enum work_area_desc_indices {
  FUSED_WA_DESC,
  MATMULBIASADD_OUT_WA_DESC,
  TS_HC_OUT_WA_DESC,
  NUM_WA_DESCS
} work_area_desc_indices;

// Struct of work_area descriptors and their calculated sizes. This way we can
// run the calculation before slicing to get the total work_area size and not
// need to recalculate the buffer_sizes after slicing the directional calls.
typedef struct work_area_descriptor {
  zdnn_tensor_desc desc;
  size_t buffer_size;
} work_area_descriptor;

// Helper method that determines the size of the work area for a single
// direction. We create some descriptors to determine that size. Save that
// information in work_area_descriptor structs so we don't need to regenerate it
// later.
static size_t setup_work_area_descs(uint8_t function_code, const uint32_t *nums,
                                    work_area_descriptor *wa_descs) {

  // work_area ------------------------------------
  // |  FUSED <TS 0/TS 1/...>                     |
  // +---------------------------------------------
  // |  BIAS_ADD                                  |
  // +---------------------------------------------
  // |  TS_C_OUT (LSTM) / TS_H_OUT (GRU)          |
  // |  TS_C_OUT (LSTM) / TS_H_OUT (GRU)  <alt>   |
  // ----------------------------------------------

  size_t buff_size = 0;
  size_t work_area_size = 0;

  // Output of NNPA_MATMUL_OP_BCAST23 + ADDITION:
  // (ts, 1, b, in_pad) or (ts * g, 1, b, s)
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &wa_descs[FUSED_WA_DESC].desc, nums[TS], 1, nums[BATCH],
                        nums[IN_PAD]);
  buff_size = zdnn_getsize_ztensor(&wa_descs[FUSED_WA_DESC].desc);
  wa_descs[FUSED_WA_DESC].buffer_size = buff_size;
  work_area_size += buff_size;

  // Output of NNPA_MATMUL_OP + ADDITION: (4, 1, b, s)
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &wa_descs[MATMULBIASADD_OUT_WA_DESC].desc, 1, 1,
                        nums[BATCH], nums[IN_PAD]);
  buff_size = zdnn_getsize_ztensor(&wa_descs[MATMULBIASADD_OUT_WA_DESC].desc);
  wa_descs[MATMULBIASADD_OUT_WA_DESC].buffer_size = buff_size;
  work_area_size += buff_size;

  // Output of NNPA_LSTMACT/NNPA_GRUACT: (1 or 2, 1, b, s)
  // Depending on number of timesteps, we may or may not need to get temporary
  // storage for h/c output.  If nums[TS] == ...
  // 1:  Save the output right into hn/cf_output buffer.
  // 2:  Need space for a single output.
  // 3+: Get 2 times the hn/cf_output buffer_size for TS_H/C_OUT because h/c is
  //     both input and output within the same operation. As we do not support
  //     in-place changes, we must have different input and output pointers.
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &wa_descs[TS_HC_OUT_WA_DESC].desc, 1, 1, nums[BATCH],
                        nums[HID_SIZE]);
  buff_size = zdnn_getsize_ztensor(&wa_descs[TS_HC_OUT_WA_DESC].desc);
  wa_descs[TS_HC_OUT_WA_DESC].buffer_size = buff_size;
  work_area_size += buff_size * MIN(nums[TS] - 1, 2);

  // final math: ((ts * g) + 4 + (1 or 2), 1, b, s)

  return work_area_size;
}

// Setup internal ztensors to store output from our internal NNPA calls.
//
// FUSED will be the output from the matmul_bcast_op with addition call. The
// dimensions will be in the form of (num_ts, 1, batch_size, in_pad)
//
// BIAS_ADD acts as both an output for MATMULBIASADD and an input to RNN NNPA op
// calls. Both require different descriptors. The following are required
// shapes for the recurring ops:
//
// Legend: (see aiu_lstm_gru())
// ----------------------+--------------+----------------------|
//        AIU Op         | Input/Output | Shape                |
// ----------------------+--------------+----------------------|
// NNPA_MATMUL_OP w/ Add |    Output    | (1, 1, b, in_pad)    |
// NNPA_LSTMACT/GRUACT   |    Input     | (num_gates, 1, b, s) |
// ----------------------+--------------+----------------------|
//
// TS_C_OUT will be the output (and reused) for each lstm_act
// over the timesteps. The dimensions will be in the form
// (1, 1, batch_size, hidden_state_size). The same as the cell state output.
//
// For each, setup dims (if required), then initialize descriptor and
// determine buffer_size using helper functions.
static void setup_internal_ztensors(uint8_t function_code, const uint32_t *nums,
                                    const zdnn_ztensor **sliced_inputs,
                                    const zdnn_ztensor *hn_output,
                                    const zdnn_ztensor *cf_output,
                                    work_area_descriptor *wa_descs,
                                    void *work_area,
                                    zdnn_tensor_desc *int_descs,
                                    zdnn_ztensor *internal_ztens) {

  // work_area ------------------------------------------------
  // | <-- [FUSED].buffer                                     |
  // +---------------------------------------------------------
  // | <-- [BIAS_ADD].buffer                                  |
  // +---------------------------------------------------------
  // | <-- [TS_H_OUT].buffer (GRU) / [TS_C_OUT].buffer (LSTM) |
  // +---------------------------------------------------------

  // Setup FUSED ztensor.
  internal_ztens[FUSED].pre_transformed_desc = NULL;
  internal_ztens[FUSED].transformed_desc = &wa_descs[FUSED_WA_DESC].desc;
  internal_ztens[FUSED].buffer = work_area;
  internal_ztens[FUSED].buffer_size = wa_descs[FUSED_WA_DESC].buffer_size;

  // TS_FUSED and the TS based BIAS_ADD both need a (g x 1 x b x s) tfrmd_desc.
  init_transformed_desc(ZDNN_NHWC, ZDNN_DLFLOAT16, ZDNN_FORMAT_4DFEATURE,
                        &int_descs[RNN_IN_TSFUSED_BIASADD_DESC], nums[GATES], 1,
                        nums[BATCH], nums[HID_SIZE]);

  // Setup TS_FUSED which will point to a slice of FUSED which matches the
  // current timestep in the loop.
  internal_ztens[TS_FUSED].pre_transformed_desc = NULL;
  internal_ztens[TS_FUSED].transformed_desc =
      &int_descs[RNN_IN_TSFUSED_BIASADD_DESC];
  internal_ztens[TS_FUSED].buffer = internal_ztens[FUSED].buffer;
  internal_ztens[TS_FUSED].buffer_size =
      internal_ztens[FUSED].buffer_size / nums[TS];

  // Setup BIAS_ADD ztensor. Its buffer starts just after the FUSED buffer. Set
  // its buffer_size to the larger of the two possible descriptors
  internal_ztens[BIAS_ADD].pre_transformed_desc = NULL;
  internal_ztens[BIAS_ADD].buffer =
      (char *)work_area + internal_ztens[FUSED].buffer_size;
  internal_ztens[BIAS_ADD].buffer_size =
      wa_descs[MATMULBIASADD_OUT_WA_DESC].buffer_size;

  // PREV_H_OUT is used to point to the previous loop's h result. The
  // initial H0 is specified by the user. Afterward, during each loop,
  // we update this to be the previous loop's result.
  internal_ztens[PREV_H_OUT].pre_transformed_desc = NULL;
  internal_ztens[PREV_H_OUT].transformed_desc =
      sliced_inputs[H0]->transformed_desc;
  internal_ztens[PREV_H_OUT].buffer = sliced_inputs[H0]->buffer;
  internal_ztens[PREV_H_OUT].buffer_size =
      zdnn_getsize_ztensor(internal_ztens[PREV_H_OUT].transformed_desc);

  // TS_H_OUT is used to track an addr for storing each loop's h output.
  // It's buffer points to an addr inside of hn_output which was passed in by
  // the user for returning output.
  //
  // When returning all timesteps, each loop, this temporary pointer shifts
  // along hn_output's buffer causing all results to be returned.
  //
  // When sized for returning only the final timestep:
  // - LSTM: each iteration will effectively point back to the start of
  //         hn_output through this temp tensor, causing only the last timestep
  //         result to be retained.
  // - GRU:  work_area buffer will be used instead through this temp tensor,
  //         until the last timestep
  internal_ztens[TS_H_OUT].pre_transformed_desc = NULL;
  internal_ztens[TS_H_OUT].transformed_desc = &wa_descs[TS_HC_OUT_WA_DESC].desc;

  // Use work_area buffer if GRU and request only last ts H output, otherwise
  // write directly to the returned hn_output. This also implies LSTM never uses
  // work_area for H output.
  if (function_code == NNPA_GRUACT &&
      (hn_output->transformed_desc->dim4 < nums[TS])) {
    internal_ztens[TS_H_OUT].buffer = (char *)internal_ztens[BIAS_ADD].buffer +
                                      internal_ztens[BIAS_ADD].buffer_size;
  } else {
    internal_ztens[TS_H_OUT].buffer = hn_output->buffer;
  }
  internal_ztens[TS_H_OUT].buffer_size =
      wa_descs[TS_HC_OUT_WA_DESC].buffer_size;

  // Only LSTM has C output
  if (function_code == NNPA_LSTMACT) {
    // PREV_C_OUT is used to point to the pervious loop's c result. The
    // initial "c0" is specified by the user. Afterward, during each loop,
    // we update this to be the previous loop's result.
    internal_ztens[PREV_C_OUT].pre_transformed_desc = NULL;
    internal_ztens[PREV_C_OUT].transformed_desc =
        sliced_inputs[C0]->transformed_desc;
    internal_ztens[PREV_C_OUT].buffer = sliced_inputs[C0]->buffer;
    internal_ztens[PREV_C_OUT].buffer_size =
        zdnn_getsize_ztensor(internal_ztens[PREV_C_OUT].transformed_desc);

    internal_ztens[TS_C_OUT].pre_transformed_desc = NULL;
    internal_ztens[TS_C_OUT].transformed_desc =
        &wa_descs[TS_HC_OUT_WA_DESC].desc;
    // If only 1 TS, write directly to the returned cf_output.
    if (nums[TS] == 1) {
      internal_ztens[TS_C_OUT].buffer = cf_output->buffer;
      // Otherwise use work_area buffer (last TS will write to returned
      // cf_output)
    } else {
      internal_ztens[TS_C_OUT].buffer =
          (char *)internal_ztens[BIAS_ADD].buffer +
          internal_ztens[BIAS_ADD].buffer_size;
    }
    internal_ztens[TS_C_OUT].buffer_size =
        wa_descs[TS_HC_OUT_WA_DESC].buffer_size;
  }
}

// Helper method that performs the bulk of the actual RNN processing. It takes
// the inputs for a single direction and processes each timestep over a loop of
// RNN activation op calls.
static zdnn_status
directional_rnn(uint8_t function_code, const uint32_t *nums,
                const zdnn_ztensor *input, const zdnn_ztensor **sliced_inputs,
                zdnn_ztensor *hn_output, zdnn_ztensor *cf_output,
                rnn_internal_direction direction, void *work_area,
                work_area_descriptor *wa_descs) {
  zdnn_status nnpa_results;

  BEGIN_BLOCK_IF_LOGLEVEL_TRACE {
    printf("%s(): For rnn_internal_direction %d input: dumpdata_ztensor()\n",
           __func__, direction);
    dumpdata_ztensor(input, AS_FLOAT, false);
    for (uint32_t input_idx = 0; input_idx < nums[SLICEABLE_INPUTS];
         input_idx++) {
      printf("%s(): For rnn_internal_direction %d on input_idx %u: "
             "dumpdata_ztensor()\n",
             __func__, direction, input_idx);
      dumpdata_ztensor(sliced_inputs[input_idx], AS_FLOAT, false);
    }
  }

  // Determine type of output based on hn_output's timestep dimension.
  bool all_timesteps;
  if (hn_output->transformed_desc->dim4 == nums[TS]) {
    all_timesteps = true;
  } else {
    all_timesteps = false;
  }

  uint32_t num_internal_ztens = (function_code == NNPA_LSTMACT)
                                    ? NUM_INTERNAL_ZTENS_LSTM
                                    : NUM_INTERNAL_ZTENS_GRU;
  zdnn_ztensor internal_ztens[num_internal_ztens];
  zdnn_tensor_desc int_descs[NUM_INTERNAL_DESCS];

  setup_internal_ztensors(function_code, nums, sliced_inputs, hn_output,
                          cf_output, wa_descs, work_area, int_descs,
                          internal_ztens);

  // Build a func_sp_parm1_matmul_bcast_op as MATMUL_BCAST_OP_ADDITION for
  // NNPA_MATMUL_OP_BCAST23 call
  func_sp_parm1_matmul_bcast_op matmul_bcast_op_parm1;
  matmul_bcast_op_parm1.val = 0;
  matmul_bcast_op_parm1.bits.operation = MATMUL_BCAST_OP_ADDITION;

  // Perform matmul broadcast against input features, weights, and biases
  if ((nnpa_results = aiu_ops_func_specific(
           NNPA_MATMUL_OP_BCAST23, input, sliced_inputs[IN_WEIGHTS],
           sliced_inputs[IN_BIAS], &internal_ztens[FUSED], NULL, 0,
           matmul_bcast_op_parm1.val, 0, 0, 0, 0)) != ZDNN_OK) {
    return ZDNN_STATUS(
        nnpa_results,
        "Failure within Matmul Biasadd Broadcast call (status = %d)\n",
        nnpa_results);
  }

  // We'll be altering the ztensor's pointer each loop for the NNPA call but
  // we need the original address so we can update that pointer each
  // iteration.
  void *org_buffer_start = internal_ztens[TS_FUSED].buffer;

  // Set loop interation variables based on direction
  uint32_t loop_start;
  int64_t loop_end;
  int8_t loop_delta;
  // See where TS_H_OUT/TS_C_OUT is updated each loop for explanation on
  // hn_out_shift
  int8_t hn_out_shift;

  // UNI hn_output (all ts) -----
  // |  TS_H_OUT 0              |
  // |  TS_H_OUT 1              |
  // | ...                      |
  // |  TS_H_OUT N              |
  // ----------------------------
  //
  // UNI hn_output (LSTM 1 ts) --
  // |  TS_H_OUT 0 > 1 .. > N   |
  // ----------------------------
  //
  // UNI hn_output (GRU 1 ts) --
  // |  TS_H_OUT N             |
  // ---------------------------
  //
  // BIDIR hn_output (all ts) --- FWD loop_start
  // |  FWD TS_H_OUT 0          |     |
  // +---------------------------     |              loop_end
  // |  BWD TS_H_OUT 0          | <hn_out_shift>         ^
  // +---------------------------     |                  |
  // |  FWD TS_H_OUT 1          |     |              <hn_out_shift>
  // +---------------------------     |                  |
  // |  BWD TS_H_OUT 1          | <hn_out_shift>         |
  // +---------------------------     |                  |
  // | ...                      |     .                  .
  // +---------------------------     |                  |
  // |  FWD TS_H_OUT N          |     V              <hn_out_shift>
  // +--------------------------- loop_end            BWD loop_start
  // |  BWD TS_H_OUT N          |
  // ----------------------------
  //
  // BIDIR hn_output (LSTM 1 ts) --
  // |  FWD TS_H_OUT 0 > 1 .. > N |
  // +-----------------------------
  // |  BWD TS_H_OUT 0 > 1 .. > N |
  // ------------------------------
  //
  // BIDIR hn_output (GRU 1 ts) --
  // |  FWD TS_H_OUT N           |
  // +----------------------------
  // |  BWD TS_H_OUT N           |
  // -----------------------------
  //
  // UNI cf_output (LSTM) --
  // |  TS_C_OUT N         |
  // -----------------------
  //
  // BIDIR cf_output (LSTM) --
  // |  FWD TS_C_OUT N       |
  // |  BWD TS_C_OUT N       |
  // -------------------------

  switch (direction) {
  case UNI_FWD:
    loop_start = 0;
    loop_end = nums[TS];
    loop_delta = 1;
    hn_out_shift = (all_timesteps) ? 1 : 0;
    break;
  case UNI_BWD:
    loop_start = nums[TS] - 1;
    loop_end = -1;
    loop_delta = -1;
    hn_out_shift = (all_timesteps) ? 1 : 0;
    internal_ztens[TS_H_OUT].buffer =
        (char *)internal_ztens[TS_H_OUT].buffer +
        loop_start * hn_out_shift * internal_ztens[TS_H_OUT].buffer_size;
    break;
  case BIDIR_FWD:
    loop_start = 0;
    loop_end = nums[TS];
    loop_delta = 1;
    hn_out_shift = (all_timesteps) ? 2 : 0;
    break;
  case BIDIR_BWD:
    loop_start = nums[TS] - 1;
    loop_end = -1;
    loop_delta = -1;
    hn_out_shift = (all_timesteps) ? 2 : 0;
    // Start at the last single h output position for BWD. See comment where
    // TS_H_OUT is updated each loop for explanation for why. Since
    // hn_output.buffer_size is set by the user and we only require they set it
    // big enough (but not necessarily exact), we can't use it to find the right
    // output address. So instead we use the TS_H_OUT size that we created
    // which is the exact size of a single FWD or BWD output. Here loop_start
    // gives us the index for the last horizontally concatenated output. So to
    // reach the start of the correct concatenated output address we jump
    // hn_out_shift (ie 2 or 0) * num concatenated outputs plus one more single
    // output to reach the reverse's half.
    internal_ztens[TS_H_OUT].buffer =
        (char *)internal_ztens[TS_H_OUT].buffer +
        loop_start * hn_out_shift * internal_ztens[TS_H_OUT].buffer_size +
        internal_ztens[TS_H_OUT].buffer_size;
    // TS_C_OUT is similar to TS_H_OUT. For BIDIR_BWD must write to the back
    // half of the concatenated cf_output. We only ever return the final C
    // output so there's no shifting like TS_H_OUT. However we only write to
    // cf_output's buffer on the last timestep, any other we write to work_area
    // which is sliced between FWD and BWD BIDIR. So the only case we have to
    // handle right here is if there's only one TS total.
    if (function_code == NNPA_LSTMACT && nums[TS] == 1) {
      internal_ztens[TS_C_OUT].buffer =
          (char *)internal_ztens[TS_C_OUT].buffer +
          internal_ztens[TS_C_OUT].buffer_size;
    }
    break;

  default:
    // Users should never see this as a switch in aiu_lstm_gru() checks their
    // input is valid and our rnn_internal_direction is set afterward.
    return ZDNN_STATUS(ZDNN_INVALID_DIRECTION,
                       "%d is not a valid rnn_internal_direction", direction);
    break;
  }

  // Used for alternating the intermediate TS_C_OUT (LSTM) / TS_H_OUT (GRU)
  // buffer each timestep.
  void *outbuf[2] = {function_code == NNPA_LSTMACT
                         ? internal_ztens[TS_C_OUT].buffer
                         : internal_ztens[TS_H_OUT].buffer,
                     function_code == NNPA_LSTMACT
                         ? (void *)((uintptr_t)internal_ztens[TS_C_OUT].buffer +
                                    internal_ztens[TS_C_OUT].buffer_size)
                         : (void *)((uintptr_t)internal_ztens[TS_H_OUT].buffer +
                                    internal_ztens[TS_H_OUT].buffer_size)};

  // Build a func_sp_parm1_matmul_op as MATMUL_OP_ADDITION for NNPA_MATMUL_OP
  // call
  func_sp_parm1_matmul_op matmul_op_parm1;
  matmul_op_parm1.val = 0;
  matmul_op_parm1.bits.operation = MATMUL_OP_ADDITION;

  // Loop through timesteps based on direction
  for (int64_t i = loop_start, c = 0; i != loop_end; i += loop_delta, c++) {
    // Set iteration's timestep input based on direction.
    internal_ztens[TS_FUSED].buffer =
        (char *)org_buffer_start + (i * internal_ztens[TS_FUSED].buffer_size);

    // Use the BIAS_ADD descriptor setup for MATMULBIASADD output.
    internal_ztens[BIAS_ADD].transformed_desc =
        &wa_descs[MATMULBIASADD_OUT_WA_DESC].desc;

    // Set BIAS_ADD based on previous loop's output (or H0 if first loop).
    if ((nnpa_results = aiu_ops_func_specific(
             NNPA_MATMUL_OP, &internal_ztens[PREV_H_OUT],
             sliced_inputs[HID_WEIGHTS], sliced_inputs[HID_BIAS],
             &internal_ztens[BIAS_ADD], NULL, 0, matmul_op_parm1.val, 0, 0, 0,
             0)) != ZDNN_OK) {
      return ZDNN_STATUS(
          nnpa_results,
          "Failure within Matmul Biasadd for timestep %ld (status = %d)\n", i,
          nnpa_results);
    }

    // Use the BIAS_ADD descriptor setup for the RNN op call.
    internal_ztens[BIAS_ADD].transformed_desc =
        &int_descs[RNN_IN_TSFUSED_BIASADD_DESC];

    // Get results from NNPA
    if ((nnpa_results = aiu_ops(
             function_code, &internal_ztens[TS_FUSED],
             &internal_ztens[BIAS_ADD],
             (function_code == NNPA_LSTMACT) ? &internal_ztens[PREV_C_OUT]
                                             : &internal_ztens[PREV_H_OUT],
             &internal_ztens[TS_H_OUT],
             (function_code == NNPA_LSTMACT) ? &internal_ztens[TS_C_OUT]
                                             : NULL)) != ZDNN_OK) {
      return ZDNN_STATUS(nnpa_results,
                         "Failure within LSTM/GRU Activation call for timestep "
                         "%ld (status = %d)\n",
                         i, nnpa_results);
    }

    // Update PREV_H/C_OUT so next loop uses previous loop's h/c output.
    internal_ztens[PREV_H_OUT].buffer = internal_ztens[TS_H_OUT].buffer;

    if (function_code == NNPA_LSTMACT) {
      internal_ztens[PREV_C_OUT].buffer = internal_ztens[TS_C_OUT].buffer;
    }

    if ((function_code == NNPA_LSTMACT) ||
        (function_code == NNPA_GRUACT && all_timesteps)) {
      // Shift the TS_H_OUT buffer each timestep. TS_H_OUT ultimately points
      // back an address in the returned hn_output.
      //
      // If only returning the final hn result, hn_out_shift will be 0 so the
      // same location is overwritten each time. This causes only the last
      // result to be returned
      //
      // If returning all timesteps, the shift will be 1 for unidirectional
      // output. We write and move one output space each loop.
      //
      // For BIDIR we return a horizontally concatenated output, where the FWDs
      // and BWDs are interleaved:
      //
      //  timestep
      //             -------------
      //     0       | FWD | BWD |
      //     1       | FWD | BWD |
      //    ...      |    ...    |
      //     n       | FWD | BWD |
      //             -------------
      //
      // The BIDIR_FWD and BIDIR_BWD call each starts at different addresses in
      // the same hn_output buffer. Each loop writes one output and shifts 2
      // spaces. This way each direction for each timestep can write to it's
      // half of the concatenated output without overwriting the other's output.
      //
      // For all timesteps FWD (uni or bidir), the pointer starts at the
      // beginning of hn_ouput and each loop shifts forward toward the end of
      // hn_output since loop_delta will be positive.
      //
      // For all timesteps BWD (uni or bidir), we start at the last h output
      // space. Then each loop we shift backward toward the start of hn_output
      // buffer since loop_delta will be negative. This way h output order
      // always matches input timesteps order rather than the order they are
      // processed.

      internal_ztens[TS_H_OUT].buffer =
          (char *)internal_ztens[TS_H_OUT].buffer +
          hn_out_shift * loop_delta * internal_ztens[PREV_H_OUT].buffer_size;
    } else {
      // If GRU and only returning the final hn result:
      // Use the work_area buffer and alternate between the two available spaces
      // for intermediate h output, until about to move to final timestep then
      // we would use hn_output->buffer instead.

      // Do this on the second to last loop so it affects the last
      // iteration...
      if (i + (2 * loop_delta) == loop_end) {
        internal_ztens[TS_H_OUT].buffer = hn_output->buffer;
        if (direction == BIDIR_BWD) {
          internal_ztens[TS_H_OUT].buffer =
              (char *)internal_ztens[TS_H_OUT].buffer +
              internal_ztens[TS_H_OUT].buffer_size;
        }
      } else {
        // c == 0 --> [1], == 1 --> [0], == 2 --> [1] etc etc
        internal_ztens[TS_H_OUT].buffer = outbuf[(~c) & 1];
      }
    }

    // For TS_C_OUT, if we are about to move to final timestep, we
    // will now point to cf_output->buffer so it can be returned to the user.
    // Otherwise, it will use the work_area buffer and alternate between the two
    // available spaces for intermediate c output.
    if (function_code == NNPA_LSTMACT) {
      // Do this on the second to last loop so it affects the last
      // iteration...
      if (i + (2 * loop_delta) == loop_end) {
        internal_ztens[TS_C_OUT].buffer = cf_output->buffer;
        // For BIDIR, cf_output returns a horizontally concatenated FWD and BWD
        // output. For BIDIR_BWD shift one output space size to separate FWD and
        // BWD output.
        if (direction == BIDIR_BWD) {
          internal_ztens[TS_C_OUT].buffer =
              (char *)cf_output->buffer + internal_ztens[TS_C_OUT].buffer_size;
        }
      }
      // Otherwise alternate between intermediate c output buffers
      else {
        internal_ztens[TS_C_OUT].buffer = outbuf[(~c) & 1];
      }
    }
  }
  return ZDNN_STATUS_OK;
}

/// Calls the NNPA operations that makeup LSTM. This method preforms "pre and
/// post" work. For "pre" it allocates the work_area (if necessary) it then
/// calls directional_rnn() to perform the RNN op (slicing the input and calling
/// directional_rnn() twice for BIDIR case). After all directions are processed
/// it cleans up the work area and returns the final status. Method stops and
/// returns on the first error encountered or ZDNN_OK.
///
/// \param[in] function_code NNPA_LSTMACT or NNPA_GRUACT
/// \param[in] input The lstm input ztensor fed into the AIU.
/// \param[in] h0 The hidden state ztensor fed into the AIU.
/// \param[in] c0 The cell state ztensor from the AIU (ignored when mode is GRU)
/// \param[in] weights The input weights ztensor from the AIU.
/// \param[in] biases The input biases ztensor from the AIU.
/// \param[in] hidden_weights The hidden weights ztensor from the AIU.
/// \param[in] hidden_biases The hidden biases ztensor from the AIU.
/// \param[in] direction LSTM/GRU direction (FWD, BWD, BIDIR)
/// \param[in] work_area Pointer to pre-allocated work area for our internal
///                      output ztensors or NULL.
/// \param[out] hn_output The returned hidden_state ztensor from the AIU.
/// \param[out] cf_output The returned cell_state ztensor from the AIU.
///
/// \return ZDNN_OK if all checks pass or a failure based on why it failed.
///
zdnn_status aiu_lstm_gru(uint8_t function_code, const zdnn_ztensor *input,
                         const zdnn_ztensor *h0, const zdnn_ztensor *c0,
                         const zdnn_ztensor *weights,
                         const zdnn_ztensor *biases,
                         const zdnn_ztensor *hidden_weights,
                         const zdnn_ztensor *hidden_biases,
                         lstm_gru_direction direction, void *work_area,
                         zdnn_ztensor *hn_output, zdnn_ztensor *cf_output) {
  /*
  DIMENSION REQUIREMENTS (stickified, i.e., NHWC)
  Legend:
    b = number of batches
    d = number of directions (2 if BIDIR or otherwise 1)
    f = number of features
    g = number of gates (4 LSTM or 3 GRU)
    s = hidden state size
    s_pad = ceil(s/64) * 64 (s with padding to nearest multiple of 64)
    in_pad = g * s_pad (horizontally concatenated gate input with padding
             between gates)
    out_pad = d * s_pad (horizontally concatenated output with padding between
              directions)
    ts = number of timesteps

  Note: The *_output expected shape differs based on unidirectional versus
  bidirectional. For hn_output, the user specified shape also controls whether
  all timestep results are returned or just the final result processed.

  tensor         | tfrmd (dim4, 3, 2, 1) | Note
  ---------------+-------------------------------------
  input          | (ts, 1, b, f)         |
  h0             | (d, 1, b, s)          |
  c0             | (d, 1, b, s)          | (LSTM only, GRU NULL)
  weights        | (d, 1, f, in_pad)     |
  biases         | (d, 1, 1, in_pad)     |
  hidden_weights | (d, 1, s, in_pad)     |
  hidden_biases  | (d, 1, 1, in_pad)     |
  ----------------------------+----------+----------------|
  hn_output      | (ts, 1, b, s)         | (uni all timesteps)
                 | (1, 1, b, s)          | (uni final only)
                 | (ts, 1, b, out_pad)   | (bidir all out_pad)
                 | (1, 1, b, out_pad)    | (bidir final only)
  cf_output      | (1, 1, b, s)          | (uni LSTM only, GRU NULL)
                 | (1, 1, b, out_pad)    | (bidir LSTM only, GRU NULL)

  When bidir output of the previous layer is used as the input of the current
  layer, number of features (f) has the same value as out_pad of the previous
  layer.  In such case, the weights tensor for the current layer needs to be
  vertically concatenated at dim2:

  input: (ts, 1, b, prev_out_pad)
  weights: (d, 1, prev_out_pad, in_pad)
  */

  zdnn_status status;

  // Store special dimensions/values to pass to various internal methods.
  uint32_t nums[NUM_INTEGER_INDICES];
  nums[TS] = input->transformed_desc->dim4;
  nums[BATCH] = input->transformed_desc->dim2;
  nums[HID_SIZE] = h0->transformed_desc->dim1;
  nums[IN_PAD] = weights->transformed_desc->dim1;

  // LSTM and GRU expect different numbers of tensors (ie gates) horizontally
  // concatenated into weights and biases tensors.
  nums[GATES] = get_func_code_num_gates(function_code);

  // Accounts for extra "cell" tensors in LSTM that aren't in GRU.
  nums[SLICEABLE_INPUTS] =
      (function_code == NNPA_LSTMACT) ? NUM_INPUTS_LSTM : NUM_INPUTS_GRU;

  // Work area is heap memory allocated for RNN internal ztensor buffers
  bool alloced_work_area = false;

  // Calculate the work_area size. Save the descriptors used to calculate it.
  work_area_descriptor wa_descs[NUM_WA_DESCS];

  size_t dir_work_area_size =
      setup_work_area_descs(function_code, nums, wa_descs);

  // RNN calls can be unidirectional (FWD or BWD) or bidirectional (BIDIR).
  // Use this to set the number of directions to expect.
  uint32_t num_dirs = (direction == BIDIR) ? 2 : 1;

  // If not passed in a pointer to pre-allocated space for the work_area,
  // allocate it now and record that we need to free what we allocated.
  void *internal_work_area = work_area;
  if (internal_work_area == NULL) {
    size_t total_size = dir_work_area_size * num_dirs;
    if (!(internal_work_area = malloc_aligned_4k(total_size))) {
      return ZDNN_STATUS(ZDNN_ALLOCATION_FAILURE,
                         "Unable to allocate %" PRIu64 " bytes for work_area.",
                         total_size);
    }
    // Used so we only free the work_area if we allocated it.
    alloced_work_area = true;
  }

  // Order must match rnn_user_zten_indices!
  const zdnn_ztensor *sliceable_inputs[] = {
      h0, weights, biases, hidden_weights, hidden_biases, c0};

  switch (direction) {
  // Skip slicing for unidirectional RNN calls
  case FWD:
    status =
        directional_rnn(function_code, nums, input, sliceable_inputs, hn_output,
                        cf_output, UNI_FWD, internal_work_area, wa_descs);
    break;
  case BWD:
    status =
        directional_rnn(function_code, nums, input, sliceable_inputs, hn_output,
                        cf_output, UNI_BWD, internal_work_area, wa_descs);
    break;
  // Slice input along direction dim and make unidirectional calls for each.
  case BIDIR: {
    // A sliced input's buffer size won't change between directions so memoize.
    size_t sliced_zten_buff_sizes[nums[SLICEABLE_INPUTS]];

    // Structs to hold slices of the user's original ztensors.
    zdnn_ztensor sliced_inputs[num_dirs][nums[SLICEABLE_INPUTS]];
    const zdnn_ztensor *sliced_inputs_ptrs[num_dirs][nums[SLICEABLE_INPUTS]];
    zdnn_tensor_desc input_descs[num_dirs * nums[SLICEABLE_INPUTS]];
    uint32_t in_desc_idx = 0;

    // Slice the user's original ztensors based on direction dimension.
    for (uint32_t dir_idx = 0; dir_idx < num_dirs; dir_idx++) {
      // First direction slices are for FWD, the second are BWD.
      rnn_internal_direction rnn_direction =
          (dir_idx == 0) ? BIDIR_FWD : BIDIR_BWD;

      // Slice the inputs over the direction dimension (dim4)
      for (uint32_t input_idx = 0; input_idx < nums[SLICEABLE_INPUTS];
           input_idx++) {
        const zdnn_ztensor *unsliced_input = sliceable_inputs[input_idx];

        // Retrieve or calculate the sliced buffer size for this input.
        if (dir_idx == 0) {
          uint32_t unsliced_dim4 = unsliced_input->transformed_desc->dim4;
          sliced_zten_buff_sizes[input_idx] =
              zdnn_getsize_ztensor(unsliced_input->transformed_desc) /
              unsliced_dim4;
        }
        uint32_t tfrmd_idx = in_desc_idx++;
        status = ztensor_slice_dim4(
            unsliced_input, dir_idx, sliced_zten_buff_sizes[input_idx], NULL,
            &input_descs[tfrmd_idx], &sliced_inputs[dir_idx][input_idx]);
        if (status != ZDNN_OK) {
          break;
        }
        sliced_inputs_ptrs[dir_idx][input_idx] =
            &sliced_inputs[dir_idx][input_idx];
      }
      if (status != ZDNN_OK) {
        break;
      }

      void *dir_work_area =
          (char *)internal_work_area + (dir_idx * dir_work_area_size);
      status = directional_rnn(
          function_code, nums, input, sliced_inputs_ptrs[dir_idx], hn_output,
          cf_output, rnn_direction, dir_work_area, wa_descs);
      if (status != ZDNN_OK) {
        break;
      }
    }
  } break;
  default:
    status = ZDNN_STATUS(ZDNN_INVALID_DIRECTION, "%d is not a valid direction",
                         direction);
    break;
  }

  // Frees the entire work_area for all directions (if required)
  if (alloced_work_area) {
    free_aligned_4k(internal_work_area);
  }

  // Upon success, indicate that the hn_output and cf_output tensors have a
  // stickified (4DFeature) tensor and return status.
  if (status == ZDNN_OK) {
    hn_output->is_transformed = true;
    BEGIN_BLOCK_IF_LOGLEVEL_TRACE {
      printf("%s(): Returning hn_output: dumpdata_ztensor()\n", __func__);
      dumpdata_ztensor(hn_output, AS_FLOAT, false);
    }
    if (function_code == NNPA_LSTMACT) {
      cf_output->is_transformed = true;
      BEGIN_BLOCK_IF_LOGLEVEL_TRACE {
        printf("%s(): Returning cf_output: dumpdata_ztensor()\n", __func__);
        dumpdata_ztensor(cf_output, AS_FLOAT, false);
      }
    }
  }

  // We break from loops if any status was not OK so we'll return either the
  // first failure or OK if everything works.
  return status;
}
