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

#include "common_rnn.h"

/******************************************************************************
                           default_input
******************************************************************************/
uint32_t default_input_shape[] = {5, 2, 4};

/* Visualization of values in shape (timestep, batch, feature) order
  [
    [ # timestep_0
        [.000,    .001,   .002,   .003], # batch_0
        [.010,    .011,   .012,   .013],  # batch_1
        # feat_0  feat_1  feat_2  feat_3
    ],
    [ # timestep_1
        [.100,    .101,   .102,   .103], # batch_0
        [.110,    .111,   .112,   .113], # batch 1
        # feat_0  feat_1  feat_2  feat_3
    ],
    [ # timestep_2
        [.200,    .201,   .202,   .203], # batch_0
        [.210,    .211,   .212,   .213], # batch_1
        # feat_0  feat_1  feat_2  feat_3
    ],
    [ # timestep_3
        [.300,    .301,   .302,   .303], # batch_0
        [.310,    .311,   .312,   .313], # batch_1
        # feat_0  feat_1  feat_2  feat_3
    ],
    [ # timestep_4
        [.400,    .401,   .402,   .403], # batch_0
        [.410,    .411,   .412,   .413], # batch_1
        # feat_0  feat_1  feat_2  feat_3
    ],
  ]
*/
float default_input_values[] = {
    0.0,   0.001, 0.002, 0.003, 0.01,  0.011, 0.012, 0.013, 0.1,   0.101,
    0.102, 0.103, 0.11,  0.111, 0.112, 0.113, 0.2,   0.201, 0.202, 0.203,
    0.21,  0.211, 0.212, 0.213, 0.3,   0.301, 0.302, 0.303, 0.31,  0.311,
    0.312, 0.313, 0.4,   0.401, 0.402, 0.403, 0.41,  0.411, 0.412, 0.413};

/******************************************************************************
                      default_uni_h0_shape
******************************************************************************/
uint32_t default_uni_h0_shape[] = {1, 2, 3};

/* Visualization of values in shape order
[[[0. 0. 0.]
  [0. 0. 0.]]]
*/
float default_uni_h0_values[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/******************************************************************************
                      default_uni_c0_shape
******************************************************************************/
uint32_t default_uni_c0_shape[] = {1, 2, 3};

/* Visualization of values in shape order
[[[0. 0. 0.]
  [0. 0. 0.]]]
*/
float default_uni_c0_values[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/******************************************************************************
                  default_uni_input_weights
******************************************************************************/
uint32_t default_uni_input_weights_shape[] = {1, 4, 3};

/* Visualization of f concatenation values in shape order
[[[-0.4937358  0.5553266  0.1960275]
  [ 0.1839888  0.1733883 -0.2754271]
  [ 0.2482673 -0.5119551 -0.5303364]
  [ 0.0915996  0.4851032  0.329131 ]]]
*/
float default_uni_input_weights_f_values[] = {
    -0.4937358, 0.5553266,  0.1960275,  0.1839888, 0.1733883, -0.2754271,
    0.2482673,  -0.5119551, -0.5303364, 0.0915996, 0.4851032, 0.329131};

/* Visualization of i concatenation values in shape order
[[[ 0.381342   0.4850937 -0.5389395]
  [-0.4317299 -0.44266    0.5706354]
  [ 0.4705055 -0.3875273  0.1228931]
  [ 0.3694199  0.2747256  0.0745605]]]
*/
float default_uni_input_weights_i_values[] = {
    0.381342,  0.4850937,  -0.5389395, -0.4317299, -0.44266,  0.5706354,
    0.4705055, -0.3875273, 0.1228931,  0.3694199,  0.2747256, 0.0745605};

/* Visualization of c concatenation values in shape order
[[[ 0.548669  -0.2726471 -0.5263513]
  [-0.4730297 -0.1263285 -0.0133806]
  [ 0.0315526 -0.385514   0.3423259]
  [ 0.2071373 -0.2729528  0.2808076]]]
*/
float default_uni_input_weights_c_values[] = {
    0.548669,  -0.2726471, -0.5263513, -0.4730297, -0.1263285, -0.0133806,
    0.0315526, -0.385514,  0.3423259,  0.2071373,  -0.2729528, 0.2808076};

/* Visualization of o concatenation values in shape order
[[[ 0.5423677  0.0945408  0.4383084]
  [-0.5070595 -0.1628114  0.4629621]
  [-0.0710383 -0.5199673  0.4833339]
  [ 0.5621256  0.2686667  0.113032 ]]]
*/
float default_uni_input_weights_o_values[] = {
    0.5423677,  0.0945408,  0.4383084, -0.5070595, -0.1628114, 0.4629621,
    -0.0710383, -0.5199673, 0.4833339, 0.5621256,  0.2686667,  0.113032};

/******************************************************************************
                   default_uni_input_biases
******************************************************************************/
uint32_t default_uni_input_biases_shape[] = {1, 3};

/* Visualization of f concatenation values in shape order
[[-0.1775665  0.0771791 -0.2241169]]
*/
float default_uni_input_biases_f_values[] = {-0.1775665, 0.0771791, -0.2241169};

/* Visualization of i concatenation values in shape order
[[ 0.3968375 -0.4157575 -0.3188125]]
*/
float default_uni_input_biases_i_values[] = {0.3968375, -0.4157575, -0.3188125};

/* Visualization of c concatenation values in shape order
[[-0.3590846 -0.1054496 -0.2817501]]
*/
float default_uni_input_biases_c_values[] = {-0.3590846, -0.1054496,
                                             -0.2817501};

/* Visualization of o concatenation values in shape order
[[ 0.0158953 -0.4273889 -0.1443277]]
*/
float default_uni_input_biases_o_values[] = {0.0158953, -0.4273889, -0.1443277};

/******************************************************************************
                default_uni_hidden_weights
******************************************************************************/
uint32_t default_uni_hidden_weights_shape[] = {1, 3, 3};

/* Visualization of f concatenation values in shape order
[[[-0.3689663 -0.3204532 -0.1866051]
  [-0.3069769 -0.3292732 -0.392639 ]
  [ 0.5463605 -0.1544762  0.4665768]]]
*/
float default_uni_hidden_weights_f_values[] = {
    -0.3689663, -0.3204532, -0.1866051, -0.3069769, -0.3292732,
    -0.392639,  0.5463605,  -0.1544762, 0.4665768};

/* Visualization of i concatenation values in shape order
[[[ 0.4114995 -0.049397   0.3073992]
  [-0.1453276 -0.1190602  0.233599 ]
  [ 0.4688771 -0.2869941  0.3672419]]]
*/
float default_uni_hidden_weights_i_values[] = {
    0.4114995, -0.049397, 0.3073992,  -0.1453276, -0.1190602,
    0.233599,  0.4688771, -0.2869941, 0.3672419};

/* Visualization of c concatenation values in shape order
[[[ 0.0643551 -0.3741214 -0.0919193]
  [ 0.2632221  0.4407408  0.4369227]
  [ 0.4282453 -0.2892259  0.5323023]]]
*/
float default_uni_hidden_weights_c_values[] = {
    0.0643551, -0.3741214, -0.0919193, 0.2632221, 0.4407408,
    0.4369227, 0.4282453,  -0.2892259, 0.5323023};

/* Visualization of o concatenation values in shape order
[[[ 0.5068286 -0.2080224 -0.0424343]
  [ 0.3320496 -0.0367477 -0.0702022]
  [ 0.5366269 -0.1974721  0.3084639]]]
*/
float default_uni_hidden_weights_o_values[] = {
    0.5068286,  -0.2080224, -0.0424343, 0.3320496, -0.0367477,
    -0.0702022, 0.5366269,  -0.1974721, 0.3084639};

/******************************************************************************
                   default_uni_hidden_biases
******************************************************************************/
uint32_t default_uni_hidden_biases_shape[] = {1, 3};

/* Visualization of f concatenation values in shape order
[[ 0.3785818 -0.186314  -0.5293279]]
*/
float default_uni_hidden_biases_f_values[] = {0.3785818, -0.186314, -0.5293279};

/* Visualization of i concatenation values in shape order
[[-0.2130262 -0.0797516  0.4536392]]
*/
float default_uni_hidden_biases_i_values[] = {-0.2130262, -0.0797516,
                                              0.4536392};

/* Visualization of c concatenation values in shape order
[[-0.4129714 -0.4429338 -0.0547802]]
*/
float default_uni_hidden_biases_c_values[] = {-0.4129714, -0.4429338,
                                              -0.0547802};

/* Visualization of o concatenation values in shape order
[[-0.2563944 -0.4034805  0.1280097]]
*/
float default_uni_hidden_biases_o_values[] = {-0.2563944, -0.4034805,
                                              0.1280097};

/******************************************************************************
                      default_bidir_h0
******************************************************************************/
uint32_t default_bidir_h0_shape[] = {2, 2, 3};

/* Visualization of values in shape order
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]]
*/
float default_bidir_h0_values[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/******************************************************************************
                      default_bidir_c0
******************************************************************************/
uint32_t default_bidir_c0_shape[] = {2, 2, 3};

/* Visualization of values in shape order
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]]
*/
float default_bidir_c0_values[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/******************************************************************************
                  default_bidir_input_weights
******************************************************************************/
uint32_t default_bidir_input_weights_shape[] = {2, 4, 3};

/* Visualization of f concatenation values in shape order
[[[-0.4937358  0.5553266  0.1960275]
  [ 0.1839888  0.1733883 -0.2754271]
  [ 0.2482673 -0.5119551 -0.5303364]
  [ 0.0915996  0.4851032  0.329131 ]]

 [[-0.4937358  0.5553266  0.1960275]
  [ 0.1839888  0.1733883 -0.2754271]
  [ 0.2482673 -0.5119551 -0.5303364]
  [ 0.0915996  0.4851032  0.329131 ]]]
*/
float default_bidir_input_weights_f_values[] = {
    -0.4937358, 0.5553266,  0.1960275,  0.1839888, 0.1733883, -0.2754271,
    0.2482673,  -0.5119551, -0.5303364, 0.0915996, 0.4851032, 0.329131,
    -0.4937358, 0.5553266,  0.1960275,  0.1839888, 0.1733883, -0.2754271,
    0.2482673,  -0.5119551, -0.5303364, 0.0915996, 0.4851032, 0.329131};

/* Visualization of i concatenation values in shape order
[[[ 0.381342   0.4850937 -0.5389395]
  [-0.4317299 -0.44266    0.5706354]
  [ 0.4705055 -0.3875273  0.1228931]
  [ 0.3694199  0.2747256  0.0745605]]

 [[ 0.381342   0.4850937 -0.5389395]
  [-0.4317299 -0.44266    0.5706354]
  [ 0.4705055 -0.3875273  0.1228931]
  [ 0.3694199  0.2747256  0.0745605]]]
*/
float default_bidir_input_weights_i_values[] = {
    0.381342,  0.4850937,  -0.5389395, -0.4317299, -0.44266,  0.5706354,
    0.4705055, -0.3875273, 0.1228931,  0.3694199,  0.2747256, 0.0745605,
    0.381342,  0.4850937,  -0.5389395, -0.4317299, -0.44266,  0.5706354,
    0.4705055, -0.3875273, 0.1228931,  0.3694199,  0.2747256, 0.0745605};

/* Visualization of c concatenation values in shape order
[[[ 0.548669  -0.2726471 -0.5263513]
  [-0.4730297 -0.1263285 -0.0133806]
  [ 0.0315526 -0.385514   0.3423259]
  [ 0.2071373 -0.2729528  0.2808076]]

 [[ 0.548669  -0.2726471 -0.5263513]
  [-0.4730297 -0.1263285 -0.0133806]
  [ 0.0315526 -0.385514   0.3423259]
  [ 0.2071373 -0.2729528  0.2808076]]]
*/
float default_bidir_input_weights_c_values[] = {
    0.548669,  -0.2726471, -0.5263513, -0.4730297, -0.1263285, -0.0133806,
    0.0315526, -0.385514,  0.3423259,  0.2071373,  -0.2729528, 0.2808076,
    0.548669,  -0.2726471, -0.5263513, -0.4730297, -0.1263285, -0.0133806,
    0.0315526, -0.385514,  0.3423259,  0.2071373,  -0.2729528, 0.2808076};

/* Visualization of o concatenation values in shape order
[[[ 0.5423677  0.0945408  0.4383084]
  [-0.5070595 -0.1628114  0.4629621]
  [-0.0710383 -0.5199673  0.4833339]
  [ 0.5621256  0.2686667  0.113032 ]]

 [[ 0.5423677  0.0945408  0.4383084]
  [-0.5070595 -0.1628114  0.4629621]
  [-0.0710383 -0.5199673  0.4833339]
  [ 0.5621256  0.2686667  0.113032 ]]]
*/
float default_bidir_input_weights_o_values[] = {
    0.5423677,  0.0945408,  0.4383084, -0.5070595, -0.1628114, 0.4629621,
    -0.0710383, -0.5199673, 0.4833339, 0.5621256,  0.2686667,  0.113032,
    0.5423677,  0.0945408,  0.4383084, -0.5070595, -0.1628114, 0.4629621,
    -0.0710383, -0.5199673, 0.4833339, 0.5621256,  0.2686667,  0.113032};

/******************************************************************************
                   default_bidir_input_biases
******************************************************************************/
uint32_t default_bidir_input_biases_shape[] = {2, 3};

/* Visualization of f concatenation values in shape order
[[-0.1775665  0.0771791 -0.2241169]
 [-0.1775665  0.0771791 -0.2241169]]
*/
float default_bidir_input_biases_f_values[] = {
    -0.1775665, 0.0771791, -0.2241169, -0.1775665, 0.0771791, -0.2241169};
;

/* Visualization of i concatenation values in shape order
[[ 0.3968375 -0.4157575 -0.3188125]
 [ 0.3968375 -0.4157575 -0.3188125]]
*/
float default_bidir_input_biases_i_values[] = {
    0.3968375, -0.4157575, -0.3188125, 0.3968375, -0.4157575, -0.3188125};

/* Visualization of c concatenation values in shape order
[[-0.3590846 -0.1054496 -0.2817501]
 [-0.3590846 -0.1054496 -0.2817501]]
*/
float default_bidir_input_biases_c_values[] = {
    -0.3590846, -0.1054496, -0.2817501, -0.3590846, -0.1054496, -0.2817501};

/* Visualization of o concatenation values in shape order
[[ 0.0158953 -0.4273889 -0.1443277]
 [ 0.0158953 -0.4273889 -0.1443277]]
*/
float default_bidir_input_biases_o_values[] = {
    0.0158953, -0.4273889, -0.1443277, 0.0158953, -0.4273889, -0.1443277};

/******************************************************************************
                default_uni_hidden_weights
******************************************************************************/
uint32_t default_bidir_hidden_weights_shape[] = {2, 3, 3};

/* Visualization of f concatenation values in shape order
[[[-0.3689663 -0.3204532 -0.1866051]
  [-0.3069769 -0.3292732 -0.392639 ]
  [ 0.5463605 -0.1544762  0.4665768]]

 [[-0.3689663 -0.3204532 -0.1866051]
  [-0.3069769 -0.3292732 -0.392639 ]
  [ 0.5463605 -0.1544762  0.4665768]]]
*/
float default_bidir_hidden_weights_f_values[] = {
    -0.3689663, -0.3204532, -0.1866051, -0.3069769, -0.3292732, -0.392639,
    0.5463605,  -0.1544762, 0.4665768,  -0.3689663, -0.3204532, -0.1866051,
    -0.3069769, -0.3292732, -0.392639,  0.5463605,  -0.1544762, 0.4665768};

/* Visualization of i concatenation values in shape order
[[[ 0.4114995 -0.049397   0.3073992]
  [-0.1453276 -0.1190602  0.233599 ]
  [ 0.4688771 -0.2869941  0.3672419]]

 [[ 0.4114995 -0.049397   0.3073992]
  [-0.1453276 -0.1190602  0.233599 ]
  [ 0.4688771 -0.2869941  0.3672419]]]
*/
float default_bidir_hidden_weights_i_values[] = {
    0.4114995,  -0.049397,  0.3073992, -0.1453276, -0.1190602, 0.233599,
    0.4688771,  -0.2869941, 0.3672419, 0.4114995,  -0.049397,  0.3073992,
    -0.1453276, -0.1190602, 0.233599,  0.4688771,  -0.2869941, 0.3672419};

/* Visualization of c concatenation values in shape order
[[[ 0.0643551 -0.3741214 -0.0919193]
  [ 0.2632221  0.4407408  0.4369227]
  [ 0.4282453 -0.2892259  0.5323023]]

 [[ 0.0643551 -0.3741214 -0.0919193]
  [ 0.2632221  0.4407408  0.4369227]
  [ 0.4282453 -0.2892259  0.5323023]]]
*/
float default_bidir_hidden_weights_c_values[] = {
    0.0643551, -0.3741214, -0.0919193, 0.2632221, 0.4407408,  0.4369227,
    0.4282453, -0.2892259, 0.5323023,  0.0643551, -0.3741214, -0.0919193,
    0.2632221, 0.4407408,  0.4369227,  0.4282453, -0.2892259, 0.5323023};

/* Visualization of o concatenation values in shape order
[[[ 0.5068286 -0.2080224 -0.0424343]
  [ 0.3320496 -0.0367477 -0.0702022]
  [ 0.5366269 -0.1974721  0.3084639]]

 [[ 0.5068286 -0.2080224 -0.0424343]
  [ 0.3320496 -0.0367477 -0.0702022]
  [ 0.5366269 -0.1974721  0.3084639]]]
*/
float default_bidir_hidden_weights_o_values[] = {
    0.5068286, -0.2080224, -0.0424343, 0.3320496, -0.0367477, -0.0702022,
    0.5366269, -0.1974721, 0.3084639,  0.5068286, -0.2080224, -0.0424343,
    0.3320496, -0.0367477, -0.0702022, 0.5366269, -0.1974721, 0.3084639};

/******************************************************************************
                   default_bidir_hidden_biases
******************************************************************************/
uint32_t default_bidir_hidden_biases_shape[] = {2, 3};

/* Visualization of f concatenation values in shape order
[[ 0.3785818 -0.186314  -0.5293279]
 [ 0.3785818 -0.186314  -0.5293279]]
*/
float default_bidir_hidden_biases_f_values[] = {
    0.3785818, -0.186314, -0.5293279, 0.3785818, -0.186314, -0.5293279};

/* Visualization of i concatenation values in shape order
[[-0.2130262 -0.0797516  0.4536392]
 [-0.2130262 -0.0797516  0.4536392]]
*/
float default_bidir_hidden_biases_i_values[] = {
    -0.2130262, -0.0797516, 0.4536392, -0.2130262, -0.0797516, 0.4536392};

/* Visualization of c concatenation values in shape order
[[-0.4129714 -0.4429338 -0.0547802]
 [-0.4129714 -0.4429338 -0.0547802]]
*/
float default_bidir_hidden_biases_c_values[] = {
    -0.4129714, -0.4429338, -0.0547802, -0.4129714, -0.4429338, -0.0547802};

/* Visualization of o concatenation values in shape order
[[-0.2563944 -0.4034805  0.1280097]
 [-0.2563944 -0.4034805  0.1280097]]
*/
float default_bidir_hidden_biases_o_values[] = {
    -0.2563944, -0.4034805, 0.1280097, -0.2563944, -0.4034805, 0.1280097};

/******************************************************************************
                    default_fwd_exp_hn_out_all_ts
******************************************************************************/
uint32_t default_fwd_hn_out_all_ts_shape[] = {5, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.1496885 -0.0568049 -0.0847668]
  [-0.1502335 -0.057525  -0.0853017]]

 [[-0.212243  -0.0906312 -0.1264551]
  [-0.2129832 -0.0917483 -0.1272719]]

 [[-0.2460073 -0.1145757 -0.1504627]
  [-0.2468257 -0.115835  -0.1514198]]

 [[-0.2677511 -0.1334158 -0.1669724]
  [-0.2686036 -0.1346632 -0.1679834]]

 [[-0.2836966 -0.1488931 -0.180066 ]
  [-0.2845615 -0.1500451 -0.1810745]]]
*/
float default_fwd_exp_hn_out_all_ts_values[] = {
    -0.1496885, -0.0568049, -0.0847668, -0.1502335, -0.057525,  -0.0853017,
    -0.212243,  -0.0906312, -0.1264551, -0.2129832, -0.0917483, -0.1272719,
    -0.2460073, -0.1145757, -0.1504627, -0.2468257, -0.115835,  -0.1514198,
    -0.2677511, -0.1334158, -0.1669724, -0.2686036, -0.1346632, -0.1679834,
    -0.2836966, -0.1488931, -0.180066,  -0.2845615, -0.1500451, -0.1810745};

/******************************************************************************
                    default_fwd_exp_hn_out_final_ts
******************************************************************************/
uint32_t default_fwd_hn_out_final_ts_shape[] = {1, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.2836966 -0.1488931 -0.180066 ]
  [-0.2845615 -0.1500451 -0.1810745]]]
*/
float default_fwd_exp_hn_out_final_ts_values[] = {
    -0.2836966, -0.1488931, -0.180066, -0.2845615, -0.1500451, -0.1810745};

/******************************************************************************
                          default_fwd_cf_exp_out
******************************************************************************/
uint32_t default_fwd_cf_out_shape[] = {1, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.8036579 -0.552912  -0.2915583]
  [-0.8046424 -0.5594633 -0.2916239]]]
*/
float default_fwd_exp_cf_out_values[] = {-0.8036579, -0.552912,  -0.2915583,
                                         -0.8046424, -0.5594633, -0.2916239};

/******************************************************************************
                    default_bwd_exp_hn_out_all_ts
******************************************************************************/
uint32_t default_bwd_hn_out_all_ts_shape[] = {5, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.2486852 -0.1223668 -0.1448121]
  [-0.2495632 -0.1242222 -0.1459369]]

 [[-0.2501265 -0.1314582 -0.1518588]
  [-0.2509633 -0.1329102 -0.1529005]]

 [[-0.2448045 -0.1305399 -0.1532898]
  [-0.2455692 -0.1315801 -0.1541975]]

 [[-0.2248478 -0.1148318 -0.1424497]
  [-0.2254719 -0.1154587 -0.14315  ]]

 [[-0.1676665 -0.0753414 -0.1037449]
  [-0.1679938 -0.0755724 -0.1041366]]]
*/
float default_bwd_exp_hn_out_all_ts_values[] = {
    -0.2486852, -0.1223668, -0.1448121, -0.2495632, -0.1242222, -0.1459369,
    -0.2501265, -0.1314582, -0.1518588, -0.2509633, -0.1329102, -0.1529005,
    -0.2448045, -0.1305399, -0.1532898, -0.2455692, -0.1315801, -0.1541975,
    -0.2248478, -0.1148318, -0.1424497, -0.2254719, -0.1154587, -0.14315,
    -0.1676665, -0.0753414, -0.1037449, -0.1679938, -0.0755724, -0.1041366};

/******************************************************************************
                    default_bwd_exp_hn_out_final_ts
******************************************************************************/
uint32_t default_bwd_hn_out_final_ts_shape[] = {1, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.2486852 -0.1223668 -0.1448121]
  [-0.2495632 -0.1242222 -0.1459369]]]
*/
float default_bwd_exp_hn_out_final_ts_values[] = {
    -0.2486852, -0.1223668, -0.1448121, -0.2495632, -0.1242222, -0.1459369};

/******************************************************************************
                          default_bwd_exp_cf_out
******************************************************************************/
uint32_t default_bwd_cf_out_shape[] = {1, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.7843156 -0.4000301 -0.3048753]
  [-0.7856599 -0.4076315 -0.3049449]]]
*/
float default_bwd_exp_cf_out_values[] = {-0.7843156, -0.4000301, -0.3048753,
                                         -0.7856599, -0.4076315, -0.3049449};

/******************************************************************************
                    default_bidir_exp_hn_out_all_ts
******************************************************************************/
uint32_t default_bidir_hn_out_all_ts_shape[] = {5, 2, 2, 3};

/* Visualization of values in shape order
[[[-0.1496885 -0.0568049 -0.0847668 -0.1502335 -0.057525  -0.0853017]
  [-0.2486852 -0.1223668 -0.1448121 -0.2495632 -0.1242222 -0.1459369]]

 [[-0.212243  -0.0906312 -0.1264551 -0.2129832 -0.0917483 -0.1272719]
  [-0.2501265 -0.1314583 -0.1518588 -0.2509633 -0.1329102 -0.1529005]]

 [[-0.2460073 -0.1145757 -0.1504627 -0.2468257 -0.115835  -0.1514198]
  [-0.2448045 -0.1305399 -0.1532898 -0.2455692 -0.1315801 -0.1541975]]

 [[-0.2677511 -0.1334158 -0.1669723 -0.2686036 -0.1346633 -0.1679834]
  [-0.2248478 -0.1148318 -0.1424497 -0.2254719 -0.1154587 -0.14315  ]]

 [[-0.2836966 -0.1488931 -0.180066  -0.2845615 -0.1500451 -0.1810745]
  [-0.1676665 -0.0753414 -0.1037448 -0.1679938 -0.0755724 -0.1041366]]]
*/

float default_bidir_exp_hn_out_all_ts_values[] = {
    -0.1496885, -0.0568049, -0.0847668, -0.1502335, -0.057525,  -0.0853017,
    -0.2486852, -0.1223668, -0.1448121, -0.2495632, -0.1242222, -0.1459369,
    -0.212243,  -0.0906312, -0.1264551, -0.2129832, -0.0917483, -0.1272719,
    -0.2501265, -0.1314583, -0.1518588, -0.2509633, -0.1329102, -0.1529005,
    -0.2460073, -0.1145757, -0.1504627, -0.2468257, -0.115835,  -0.1514198,
    -0.2448045, -0.1305399, -0.1532898, -0.2455692, -0.1315801, -0.1541975,
    -0.2677511, -0.1334158, -0.1669723, -0.2686036, -0.1346633, -0.1679834,
    -0.2248478, -0.1148318, -0.1424497, -0.2254719, -0.1154587, -0.14315,
    -0.2836966, -0.1488931, -0.180066,  -0.2845615, -0.1500451, -0.1810745,
    -0.1676665, -0.0753414, -0.1037448, -0.1679938, -0.0755724, -0.1041366};

/******************************************************************************
                    default_bidir_exp_hn_out_final_ts
******************************************************************************/
uint32_t default_bidir_hn_out_final_ts_shape[] = {1, 2, 2, 3};

/* Visualization of values in shape order
[[[-0.2836966 -0.1488931 -0.180066  -0.2845615 -0.1500451 -0.1810745]
  [-0.2486852 -0.1223668 -0.1448121 -0.2495632 -0.1242222 -0.1459369]]]
*/

float default_bidir_exp_hn_out_final_ts_values[] = {
    -0.2836966, -0.1488931, -0.180066,  -0.2845615, -0.1500451, -0.1810745,
    -0.2486852, -0.1223668, -0.1448121, -0.2495632, -0.1242222, -0.1459369};

/******************************************************************************
                          default_bidir_cf_exp_out
******************************************************************************/
uint32_t default_bidir_cf_out_shape[] = {1, 2, 2, 3};

/* Visualization of values in shape order
[[[-0.8036579 -0.552912  -0.2915582 -0.8046424 -0.5594633 -0.2916239]
  [-0.7843156 -0.4000301 -0.3048753 -0.7856599 -0.4076315 -0.3049449]]]
*/

float default_bidir_exp_cf_out_values[] = {
    -0.8036579, -0.552912,  -0.2915582, -0.8046424, -0.5594633, -0.2916239,
    -0.7843156, -0.4000301, -0.3048753, -0.7856599, -0.4076315, -0.3049449};

/******************************************************************************
                          Unity Methods
******************************************************************************/
void setUp(void) { /* This is run before EACH TEST */
  VERIFY_HW_ENV;
}

void tearDown(void) { /* This is run after EACH TEST */
}

/******************************************************************************
                              Tests
******************************************************************************/
// Confirm that lstm returns OK and expected values when set to return hn
// results from all timesteps
void lstm_basic_fwd_hn_all() {
  test_zdnn_api_lstm_gru(
      NNPA_LSTMACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      default_uni_c0_shape, ZDNN_3DS, default_uni_c0_values,

      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_f_values, default_uni_input_weights_i_values,
      default_uni_input_weights_c_values, default_uni_input_weights_o_values,

      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_f_values, default_uni_input_biases_i_values,
      default_uni_input_biases_c_values, default_uni_input_biases_o_values,

      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_f_values, default_uni_hidden_weights_i_values,
      default_uni_hidden_weights_c_values, default_uni_hidden_weights_o_values,

      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_f_values, default_uni_hidden_biases_i_values,
      default_uni_hidden_biases_c_values, default_uni_hidden_biases_o_values,

      default_fwd_hn_out_all_ts_shape, ZDNN_4DS,
      default_fwd_exp_hn_out_all_ts_values,

      default_fwd_cf_out_shape, ZDNN_4DS, default_fwd_exp_cf_out_values,

      FWD, ZDNN_OK);
}

// Confirm that lstm returns OK and expected values when set to return only the
// final hn result
void lstm_basic_fwd_hn_final() {
  test_zdnn_api_lstm_gru(
      NNPA_LSTMACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      default_uni_c0_shape, ZDNN_3DS, default_uni_c0_values,

      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_f_values, default_uni_input_weights_i_values,
      default_uni_input_weights_c_values, default_uni_input_weights_o_values,

      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_f_values, default_uni_input_biases_i_values,
      default_uni_input_biases_c_values, default_uni_input_biases_o_values,

      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_f_values, default_uni_hidden_weights_i_values,
      default_uni_hidden_weights_c_values, default_uni_hidden_weights_o_values,

      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_f_values, default_uni_hidden_biases_i_values,
      default_uni_hidden_biases_c_values, default_uni_hidden_biases_o_values,

      default_fwd_hn_out_final_ts_shape, ZDNN_4DS,
      default_fwd_exp_hn_out_final_ts_values,

      default_fwd_cf_out_shape, ZDNN_4DS, default_fwd_exp_cf_out_values,

      FWD, ZDNN_OK);
}

// Confirm that lstm returns OK and expected values when set to return hn
// results from all timesteps
void lstm_basic_bwd_hn_all() {
  test_zdnn_api_lstm_gru(
      NNPA_LSTMACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      default_uni_c0_shape, ZDNN_3DS, default_uni_c0_values,

      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_f_values, default_uni_input_weights_i_values,
      default_uni_input_weights_c_values, default_uni_input_weights_o_values,

      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_f_values, default_uni_input_biases_i_values,
      default_uni_input_biases_c_values, default_uni_input_biases_o_values,

      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_f_values, default_uni_hidden_weights_i_values,
      default_uni_hidden_weights_c_values, default_uni_hidden_weights_o_values,

      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_f_values, default_uni_hidden_biases_i_values,
      default_uni_hidden_biases_c_values, default_uni_hidden_biases_o_values,

      default_bwd_hn_out_all_ts_shape, ZDNN_4DS,
      default_bwd_exp_hn_out_all_ts_values,

      default_bwd_cf_out_shape, ZDNN_4DS, default_bwd_exp_cf_out_values,

      BWD, ZDNN_OK);
}

// Confirm that lstm returns OK and expected values when set to return only the
// final hn result
void lstm_basic_bwd_hn_final() {
  test_zdnn_api_lstm_gru(
      NNPA_LSTMACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      default_uni_c0_shape, ZDNN_3DS, default_uni_c0_values,

      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_f_values, default_uni_input_weights_i_values,
      default_uni_input_weights_c_values, default_uni_input_weights_o_values,

      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_f_values, default_uni_input_biases_i_values,
      default_uni_input_biases_c_values, default_uni_input_biases_o_values,

      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_f_values, default_uni_hidden_weights_i_values,
      default_uni_hidden_weights_c_values, default_uni_hidden_weights_o_values,

      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_f_values, default_uni_hidden_biases_i_values,
      default_uni_hidden_biases_c_values, default_uni_hidden_biases_o_values,

      default_bwd_hn_out_final_ts_shape, ZDNN_4DS,
      default_bwd_exp_hn_out_final_ts_values,

      default_bwd_cf_out_shape, ZDNN_4DS, default_bwd_exp_cf_out_values,

      BWD, ZDNN_OK);
}

// Confirm that lstm returns OK and expected values when set to return hn
// results from all timesteps
void lstm_basic_bidir_hn_all() {
  test_zdnn_api_lstm_gru(
      NNPA_LSTMACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_bidir_h0_shape, ZDNN_3DS, default_bidir_h0_values,

      default_bidir_c0_shape, ZDNN_3DS, default_bidir_c0_values,

      default_bidir_input_weights_shape, ZDNN_3DS,
      default_bidir_input_weights_f_values,
      default_bidir_input_weights_i_values,
      default_bidir_input_weights_c_values,
      default_bidir_input_weights_o_values,

      default_bidir_input_biases_shape, ZDNN_2DS,
      default_bidir_input_biases_f_values, default_bidir_input_biases_i_values,
      default_bidir_input_biases_c_values, default_bidir_input_biases_o_values,

      default_bidir_hidden_weights_shape, ZDNN_3DS,
      default_bidir_hidden_weights_f_values,
      default_bidir_hidden_weights_i_values,
      default_bidir_hidden_weights_c_values,
      default_bidir_hidden_weights_o_values,

      default_bidir_hidden_biases_shape, ZDNN_2DS,
      default_bidir_hidden_biases_f_values,
      default_bidir_hidden_biases_i_values,
      default_bidir_hidden_biases_c_values,
      default_bidir_hidden_biases_o_values,

      default_bidir_hn_out_all_ts_shape, ZDNN_4DS,
      default_bidir_exp_hn_out_all_ts_values,

      default_bidir_cf_out_shape, ZDNN_4DS, default_bidir_exp_cf_out_values,

      BIDIR, ZDNN_OK);
}

// Confirm that lstm returns OK and expected values when set to return only the
// final hn result
void lstm_basic_bidir_hn_final() {
  test_zdnn_api_lstm_gru(
      NNPA_LSTMACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_bidir_h0_shape, ZDNN_3DS, default_bidir_h0_values,

      default_bidir_c0_shape, ZDNN_3DS, default_bidir_c0_values,

      default_bidir_input_weights_shape, ZDNN_3DS,
      default_bidir_input_weights_f_values,
      default_bidir_input_weights_i_values,
      default_bidir_input_weights_c_values,
      default_bidir_input_weights_o_values,

      default_bidir_input_biases_shape, ZDNN_2DS,
      default_bidir_input_biases_f_values, default_bidir_input_biases_i_values,
      default_bidir_input_biases_c_values, default_bidir_input_biases_o_values,

      default_bidir_hidden_weights_shape, ZDNN_3DS,
      default_bidir_hidden_weights_f_values,
      default_bidir_hidden_weights_i_values,
      default_bidir_hidden_weights_c_values,
      default_bidir_hidden_weights_o_values,

      default_bidir_hidden_biases_shape, ZDNN_2DS,
      default_bidir_hidden_biases_f_values,
      default_bidir_hidden_biases_i_values,
      default_bidir_hidden_biases_c_values,
      default_bidir_hidden_biases_o_values,

      default_bidir_hn_out_final_ts_shape, ZDNN_4DS,
      default_bidir_exp_hn_out_final_ts_values,

      default_bidir_cf_out_shape, ZDNN_4DS, default_bidir_exp_cf_out_values,

      BIDIR, ZDNN_OK);
}

int main() {
  UNITY_BEGIN();

// LSTM tests with good input requires AIU to get results and
// validate values.
#ifdef TEST_AIU
  // FWD direction tests
  RUN_TEST_ALL_DATATYPES(lstm_basic_fwd_hn_all);
  RUN_TEST_ALL_DATATYPES(lstm_basic_fwd_hn_final);

  // BWD direction tests
  RUN_TEST_ALL_DATATYPES(lstm_basic_bwd_hn_all);
  RUN_TEST_ALL_DATATYPES(lstm_basic_bwd_hn_final);

  // BIDIR direction tests
  RUN_TEST_ALL_DATATYPES(lstm_basic_bidir_hn_all);
  RUN_TEST_ALL_DATATYPES(lstm_basic_bidir_hn_final);
#endif
  return UNITY_END();
}
