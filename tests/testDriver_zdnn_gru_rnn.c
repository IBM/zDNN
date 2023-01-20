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
                           default_uni_h0
******************************************************************************/
uint32_t default_uni_h0_shape[] = {1, 2, 3};

/* Visualization of values in shape order
[[[0. 0. 0.]
  [0. 0. 0.]]]
*/
float default_uni_h0_values[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

/******************************************************************************
                       default_uni_input_weights
******************************************************************************/
uint32_t default_uni_input_weights_shape[] = {1, 4, 3};

/* Visualization of z concatenation values in shape order
[[[-0.4937358  0.5553266  0.1960275]
  [ 0.1839888  0.1733883 -0.2754271]
  [ 0.2482673 -0.5119551 -0.5303364]
  [ 0.0915996  0.4851032  0.329131 ]]]
*/
float default_uni_input_weights_z_values[] = {
    -0.4937358, 0.5553266,  0.1960275,  0.1839888, 0.1733883, -0.2754271,
    0.2482673,  -0.5119551, -0.5303364, 0.0915996, 0.4851032, 0.329131};

/* Visualization of r concatenation values in shape order
[[[ 0.381342   0.4850937 -0.5389395]
  [-0.4317299 -0.44266    0.5706354]
  [ 0.4705055 -0.3875273  0.1228931]
  [ 0.3694199  0.2747256  0.0745605]]]
*/
float default_uni_input_weights_r_values[] = {
    0.381342,  0.4850937,  -0.5389395, -0.4317299, -0.44266,  0.5706354,
    0.4705055, -0.3875273, 0.1228931,  0.3694199,  0.2747256, 0.0745605};

/* Visualization of h concatenation values in shape order
[[[ 0.548669  -0.2726471 -0.5263513]
  [-0.4730297 -0.1263285 -0.0133806]
  [ 0.0315526 -0.385514   0.3423259]
  [ 0.2071373 -0.2729528  0.2808076]]]
*/
float default_uni_input_weights_h_values[] = {
    0.548669,  -0.2726471, -0.5263513, -0.4730297, -0.1263285, -0.0133806,
    0.0315526, -0.385514,  0.3423259,  0.2071373,  -0.2729528, 0.2808076};

/******************************************************************************
                   default_uni_input_biases
******************************************************************************/
uint32_t default_uni_input_biases_shape[] = {1, 3};

/* Visualization of z concatenation values in shape order
[[0.0643551 0.2632221 0.4282453]]
*/
float default_uni_input_biases_z_values[] = {0.0643551, 0.2632221, 0.4282453};

/* Visualization of r concatenation values in shape order
[[-0.1866051 -0.392639   0.4665768]]
*/
float default_uni_input_biases_r_values[] = {-0.1866051, -0.392639, 0.4665768};

/* Visualization of h concatenation values in shape order
[[-0.3741214  0.4407408 -0.2892259]]
*/
float default_uni_input_biases_h_values[] = {-0.3741214, 0.4407408, -0.2892259};

/******************************************************************************
                   default_uni_hidden_weights
******************************************************************************/
uint32_t default_uni_hidden_weights_shape[] = {1, 3, 3};

/* Visualization of z concatenation values in shape order
[[[ 0.4629621  0.4114995 -0.049397 ]
  [ 0.4833339 -0.1453276 -0.1190602]
  [ 0.113032   0.4688771 -0.2869941]]]
*/
float default_uni_hidden_weights_z_values[] = {
    0.4629621,  0.4114995, -0.049397, 0.4833339, -0.1453276,
    -0.1190602, 0.113032,  0.4688771, -0.2869941};

/* Visualization of r concatenation values in shape order
[[[ 0.5423677  0.5621256 -0.5199673]
  [-0.5070595  0.0945408  0.2686667]
  [-0.0710383 -0.1628114  0.4383084]]]
*/
float default_uni_hidden_weights_r_values[] = {
    0.5423677, 0.5621256,  -0.5199673, -0.5070595, 0.0945408,
    0.2686667, -0.0710383, -0.1628114, 0.4383084};

/* Visualization of h concatenation values in shape order
[[[ 0.3073992 -0.3689663 -0.3204532]
  [ 0.233599  -0.3069769 -0.3292732]
  [ 0.3672419  0.5463605 -0.1544762]]]
*/
float default_uni_hidden_weights_h_values[] = {
    0.3073992,  -0.3689663, -0.3204532, 0.233599,  -0.3069769,
    -0.3292732, 0.3672419,  0.5463605,  -0.1544762};

/******************************************************************************
                   default_uni_hidden_biases
******************************************************************************/
uint32_t default_uni_hidden_biases_shape[] = {1, 3};

/* Visualization of z concatenation values in shape order
[[0.5068286 0.3320496 0.5366269]]
*/
float default_uni_hidden_biases_z_values[] = {0.5068286, 0.3320496, 0.5366269};

/* Visualization of r concatenation values in shape order
[[-0.0919193  0.4369227  0.5323023]]
*/
float default_uni_hidden_biases_r_values[] = {-0.0919193, 0.4369227, 0.5323023};

/* Visualization of h concatenation values in shape order
[[-0.2080224 -0.0367477 -0.1974721]]
*/
float default_uni_hidden_biases_h_values[] = {-0.2080224, -0.0367477,
                                              -0.1974721};

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
                       default_bidir_input_weights
******************************************************************************/
uint32_t default_bidir_input_weights_shape[] = {2, 4, 3};

/* Visualization of z concatenation values in shape order
[[[-0.4937358  0.5553266  0.1960275]
  [ 0.1839888  0.1733883 -0.2754271]
  [ 0.2482673 -0.5119551 -0.5303364]
  [ 0.0915996  0.4851032  0.329131 ]]

 [[-0.4937358  0.5553266  0.1960275]
  [ 0.1839888  0.1733883 -0.2754271]
  [ 0.2482673 -0.5119551 -0.5303364]
  [ 0.0915996  0.4851032  0.329131 ]]]
*/
float default_bidir_input_weights_z_values[] = {
    -0.4937358, 0.5553266,  0.1960275,  0.1839888, 0.1733883, -0.2754271,
    0.2482673,  -0.5119551, -0.5303364, 0.0915996, 0.4851032, 0.329131,
    -0.4937358, 0.5553266,  0.1960275,  0.1839888, 0.1733883, -0.2754271,
    0.2482673,  -0.5119551, -0.5303364, 0.0915996, 0.4851032, 0.329131};

/* Visualization of r concatenation values in shape order
[[[ 0.381342   0.4850937 -0.5389395]
  [-0.4317299 -0.44266    0.5706354]
  [ 0.4705055 -0.3875273  0.1228931]
  [ 0.3694199  0.2747256  0.0745605]]

 [[ 0.381342   0.4850937 -0.5389395]
  [-0.4317299 -0.44266    0.5706354]
  [ 0.4705055 -0.3875273  0.1228931]
  [ 0.3694199  0.2747256  0.0745605]]]
*/
float default_bidir_input_weights_r_values[] = {
    0.381342,  0.4850937,  -0.5389395, -0.4317299, -0.44266,  0.5706354,
    0.4705055, -0.3875273, 0.1228931,  0.3694199,  0.2747256, 0.0745605,
    0.381342,  0.4850937,  -0.5389395, -0.4317299, -0.44266,  0.5706354,
    0.4705055, -0.3875273, 0.1228931,  0.3694199,  0.2747256, 0.0745605};

/* Visualization of h concatenation values in shape order
[[[ 0.548669  -0.2726471 -0.5263513]
  [-0.4730297 -0.1263285 -0.0133806]
  [ 0.0315526 -0.385514   0.3423259]
  [ 0.2071373 -0.2729528  0.2808076]]

 [[ 0.548669  -0.2726471 -0.5263513]
  [-0.4730297 -0.1263285 -0.0133806]
  [ 0.0315526 -0.385514   0.3423259]
  [ 0.2071373 -0.2729528  0.2808076]]]
*/
float default_bidir_input_weights_h_values[] = {
    0.548669,  -0.2726471, -0.5263513, -0.4730297, -0.1263285, -0.0133806,
    0.0315526, -0.385514,  0.3423259,  0.2071373,  -0.2729528, 0.2808076,
    0.548669,  -0.2726471, -0.5263513, -0.4730297, -0.1263285, -0.0133806,
    0.0315526, -0.385514,  0.3423259,  0.2071373,  -0.2729528, 0.2808076};

/******************************************************************************
                   default_bidir_input_biases
******************************************************************************/
uint32_t default_bidir_input_biases_shape[] = {2, 3};

/* Visualization of z concatenation values in shape order
[[0.0643551 0.2632221 0.4282453]
 [0.0643551 0.2632221 0.4282453]]
*/
float default_bidir_input_biases_z_values[] = {0.0643551, 0.2632221, 0.4282453,
                                               0.0643551, 0.2632221, 0.4282453};

/* Visualization of r concatenation values in shape order
[[-0.1866051 -0.392639   0.4665768]
 [-0.1866051 -0.392639   0.4665768]]
*/
float default_bidir_input_biases_r_values[] = {
    -0.1866051, -0.392639, 0.4665768, -0.1866051, -0.392639, 0.4665768};

/* Visualization of h concatenation values in shape order
[[-0.3741214  0.4407408 -0.2892259]
 [-0.3741214  0.4407408 -0.2892259]]
*/
float default_bidir_input_biases_h_values[] = {
    -0.3741214, 0.4407408, -0.2892259, -0.3741214, 0.4407408, -0.2892259};

/******************************************************************************
                   default_bidir_hidden_weights
******************************************************************************/
uint32_t default_bidir_hidden_weights_shape[] = {2, 3, 3};

/* Visualization of z concatenation values in shape order
[[[ 0.4629621  0.4114995 -0.049397 ]
  [ 0.4833339 -0.1453276 -0.1190602]
  [ 0.113032   0.4688771 -0.2869941]]

 [[ 0.4629621  0.4114995 -0.049397 ]
  [ 0.4833339 -0.1453276 -0.1190602]
  [ 0.113032   0.4688771 -0.2869941]]]
*/
float default_bidir_hidden_weights_z_values[] = {
    0.4629621, 0.4114995,  -0.049397,  0.4833339, -0.1453276, -0.1190602,
    0.113032,  0.4688771,  -0.2869941, 0.4629621, 0.4114995,  -0.049397,
    0.4833339, -0.1453276, -0.1190602, 0.113032,  0.4688771,  -0.2869941};

/* Visualization of r concatenation values in shape order
[[[ 0.5423677  0.5621256 -0.5199673]
  [-0.5070595  0.0945408  0.2686667]
  [-0.0710383 -0.1628114  0.4383084]]

 [[ 0.5423677  0.5621256 -0.5199673]
  [-0.5070595  0.0945408  0.2686667]
  [-0.0710383 -0.1628114  0.4383084]]]
*/
float default_bidir_hidden_weights_r_values[] = {
    0.5423677,  0.5621256,  -0.5199673, -0.5070595, 0.0945408,  0.2686667,
    -0.0710383, -0.1628114, 0.4383084,  0.5423677,  0.5621256,  -0.5199673,
    -0.5070595, 0.0945408,  0.2686667,  -0.0710383, -0.1628114, 0.4383084};

/* Visualization of h concatenation values in shape order
[[[ 0.3073992 -0.3689663 -0.3204532]
  [ 0.233599  -0.3069769 -0.3292732]
  [ 0.3672419  0.5463605 -0.1544762]]

 [[ 0.3073992 -0.3689663 -0.3204532]
  [ 0.233599  -0.3069769 -0.3292732]
  [ 0.3672419  0.5463605 -0.1544762]]]
*/
float default_bidir_hidden_weights_h_values[] = {
    0.3073992, -0.3689663, -0.3204532, 0.233599,  -0.3069769, -0.3292732,
    0.3672419, 0.5463605,  -0.1544762, 0.3073992, -0.3689663, -0.3204532,
    0.233599,  -0.3069769, -0.3292732, 0.3672419, 0.5463605,  -0.1544762};

/******************************************************************************
                   default_bidir_hidden_biases
******************************************************************************/
uint32_t default_bidir_hidden_biases_shape[] = {2, 3};

/* Visualization of z concatenation values in shape order
[[0.5068286 0.3320496 0.5366269]
 [0.5068286 0.3320496 0.5366269]]
*/
float default_bidir_hidden_biases_z_values[] = {
    0.5068286, 0.3320496, 0.5366269, 0.5068286, 0.3320496, 0.5366269};

/* Visualization of r concatenation values in shape order
[[-0.0919193  0.4369227  0.5323023]
 [-0.0919193  0.4369227  0.5323023]]
*/
float default_bidir_hidden_biases_r_values[] = {
    -0.0919193, 0.4369227, 0.5323023, -0.0919193, 0.4369227, 0.5323023};

/* Visualization of h concatenation values in shape order
[[-0.2080224 -0.0367477 -0.1974721]
 [-0.2080224 -0.0367477 -0.1974721]]
*/
float default_bidir_hidden_biases_h_values[] = {
    -0.2080224, -0.0367477, -0.1974721, -0.2080224, -0.0367477, -0.1974721};

/******************************************************************************
                    default_fwd_exp_hn_out_all_ts
******************************************************************************/
uint32_t default_fwd_hn_out_all_ts_shape[] = {5, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.1562103  0.1410986 -0.1123356]
  [-0.1553763  0.1372994 -0.1123919]]

 [[-0.253498   0.1940096 -0.1891814]
  [-0.2523776  0.1878957 -0.1889893]]

 [[-0.3126792  0.1866586 -0.2388406]
  [-0.3114854  0.179318  -0.2382826]]

 [[-0.3473134  0.1435677 -0.2676416]
  [-0.3461194  0.1356744 -0.2667077]]

 [[-0.3660706  0.0814286 -0.2807784]
  [-0.3648955  0.0733736 -0.2795098]]]
*/
float default_fwd_exp_hn_out_all_ts_values[] = {
    -0.1562103, 0.1410986, -0.1123356, -0.1553763, 0.1372994, -0.1123919,
    -0.253498,  0.1940096, -0.1891814, -0.2523776, 0.1878957, -0.1889893,
    -0.3126792, 0.1866586, -0.2388406, -0.3114854, 0.179318,  -0.2382826,
    -0.3473134, 0.1435677, -0.2676416, -0.3461194, 0.1356744, -0.2667077,
    -0.3660706, 0.0814286, -0.2807784, -0.3648955, 0.0733736, -0.2795098};

/******************************************************************************
                    default_fwd_exp_hn_out_final_ts
******************************************************************************/
uint32_t default_fwd_hn_out_final_ts_shape[] = {1, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.3660706  0.0814286 -0.2807784]
  [-0.3648955  0.0733736 -0.2795098]]]
*/
float default_fwd_exp_hn_out_final_ts_values[] = {
    -0.3660706, 0.0814286, -0.2807784, -0.3648955, 0.0733736, -0.2795098};

/******************************************************************************
                    default_bwd_exp_hn_out_all_ts
******************************************************************************/
uint32_t default_bwd_hn_out_all_ts_shape[] = {5, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.4037485  0.2564563 -0.2790346]
  [-0.4026485  0.2477951 -0.2778324]]

 [[-0.3612258  0.1689991 -0.2550354]
  [-0.3600727  0.1606691 -0.2541449]]

 [[-0.3028114  0.0906047 -0.224893 ]
  [-0.3015861  0.083261  -0.2243577]]

 [[-0.223746   0.0309375 -0.1819546]
  [-0.2225393  0.025346  -0.1817581]]

 [[-0.1217477 -0.0007261 -0.1141484]
  [-0.1208584 -0.0038126 -0.1141814]]]
*/
float default_bwd_exp_hn_out_all_ts_values[] = {
    -0.4037485, 0.2564563,  -0.2790346, -0.4026485, 0.2477951,  -0.2778324,
    -0.3612258, 0.1689991,  -0.2550354, -0.3600727, 0.1606691,  -0.2541449,
    -0.3028114, 0.0906047,  -0.224893,  -0.3015861, 0.083261,   -0.2243577,
    -0.223746,  0.0309375,  -0.1819546, -0.2225393, 0.025346,   -0.1817581,
    -0.1217477, -0.0007261, -0.1141484, -0.1208584, -0.0038126, -0.1141814};

/******************************************************************************
                    default_bwd_exp_hn_out_final_ts
******************************************************************************/
uint32_t default_bwd_hn_out_final_ts_shape[] = {1, 1, 2, 3};

/* Visualization of values in shape order
[[[-0.4037485  0.2564563 -0.2790346]
  [-0.4026485  0.2477951 -0.2778324]]]
*/
float default_bwd_exp_hn_out_final_ts_values[] = {
    -0.4037485, 0.2564563, -0.2790346, -0.4026485, 0.2477951, -0.2778324};

/******************************************************************************
                    default_bidir_exp_hn_out_all_ts
******************************************************************************/
uint32_t default_bidir_hn_out_all_ts_shape[] = {5, 2, 2, 3};

/* Visualization of values in shape order
[[[-0.1562103  0.1410986 -0.1123356 -0.1553763  0.1372994 -0.1123919]
  [-0.4037485  0.2564563 -0.2790346 -0.4026485  0.2477951 -0.2778324]]

 [[-0.253498   0.1940096 -0.1891814 -0.2523776  0.1878956 -0.1889893]
  [-0.3612258  0.1689991 -0.2550354 -0.3600727  0.1606691 -0.2541449]]

 [[-0.3126791  0.1866586 -0.2388406 -0.3114854  0.179318  -0.2382826]
  [-0.3028114  0.0906047 -0.2248929 -0.3015861  0.083261  -0.2243577]]

 [[-0.3473134  0.1435677 -0.2676416 -0.3461194  0.1356744 -0.2667077]
  [-0.223746   0.0309375 -0.1819546 -0.2225393  0.025346  -0.1817581]]

 [[-0.3660705  0.0814286 -0.2807783 -0.3648955  0.0733736 -0.2795098]
  [-0.1217477 -0.0007261 -0.1141484 -0.1208584 -0.0038126 -0.1141814]]]
*/

float default_bidir_exp_hn_out_all_ts_values[] = {
    -0.1562103, 0.1410986,  -0.1123356, -0.1553763, 0.1372994,  -0.1123919,
    -0.4037485, 0.2564563,  -0.2790346, -0.4026485, 0.2477951,  -0.2778324,
    -0.253498,  0.1940096,  -0.1891814, -0.2523776, 0.1878956,  -0.1889893,
    -0.3612258, 0.1689991,  -0.2550354, -0.3600727, 0.1606691,  -0.2541449,
    -0.3126791, 0.1866586,  -0.2388406, -0.3114854, 0.179318,   -0.2382826,
    -0.3028114, 0.0906047,  -0.2248929, -0.3015861, 0.083261,   -0.2243577,
    -0.3473134, 0.1435677,  -0.2676416, -0.3461194, 0.1356744,  -0.2667077,
    -0.223746,  0.0309375,  -0.1819546, -0.2225393, 0.025346,   -0.1817581,
    -0.3660705, 0.0814286,  -0.2807783, -0.3648955, 0.0733736,  -0.2795098,
    -0.1217477, -0.0007261, -0.1141484, -0.1208584, -0.0038126, -0.1141814};

/******************************************************************************
                    default_bidir_exp_hn_out_final_ts
******************************************************************************/
uint32_t default_bidir_hn_out_final_ts_shape[] = {1, 2, 2, 3};

/* Visualization of values in shape order
[[[-0.3660705  0.0814286 -0.2807783 -0.3648955  0.0733736 -0.2795098]
  [-0.4037485  0.2564563 -0.2790346 -0.4026485  0.2477951 -0.2778324]]]
*/

float default_bidir_exp_hn_out_final_ts_values[] = {
    -0.3660705, 0.0814286, -0.2807783, -0.3648955, 0.0733736, -0.2795098,
    -0.4037485, 0.2564563, -0.2790346, -0.4026485, 0.2477951, -0.2778324};

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
// Confirm that gru returns OK and expected values when set to return hn
// results from all timesteps
void gru_basic_fwd_hn_all() {
  test_zdnn_api_lstm_gru(
      NNPA_GRUACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The test method also supports LSTM which requires c0, pass in h0 again
      // as a stand-in for c0 which the test will ignore for GRU networks.
      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_z_values, default_uni_input_weights_r_values,
      default_uni_input_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_z_values, default_uni_input_biases_r_values,
      default_uni_input_biases_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_z_values, default_uni_hidden_weights_r_values,
      default_uni_hidden_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_z_values, default_uni_hidden_biases_r_values,
      default_uni_hidden_biases_h_values, ZERO_ARRAY,

      default_fwd_hn_out_all_ts_shape, ZDNN_4DS,
      default_fwd_exp_hn_out_all_ts_values,

      // The test method also supports LSTM which requires cf, pass NULL for GRU
      NULL, ZDNN_3DS, NULL,

      FWD, ZDNN_OK);
}

// Confirm that gru returns OK and expected values when set to return only
// the final hn result
void gru_basic_fwd_hn_final() {
  test_zdnn_api_lstm_gru(
      NNPA_GRUACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The test method also supports LSTM which requires c0, pass in h0 again
      // as a stand-in for c0 which the test will ignore for GRU networks.
      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_z_values, default_uni_input_weights_r_values,
      default_uni_input_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_z_values, default_uni_input_biases_r_values,
      default_uni_input_biases_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_z_values, default_uni_hidden_weights_r_values,
      default_uni_hidden_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_z_values, default_uni_hidden_biases_r_values,
      default_uni_hidden_biases_h_values, ZERO_ARRAY,

      default_fwd_hn_out_final_ts_shape, ZDNN_4DS,
      default_fwd_exp_hn_out_final_ts_values,

      // The test method also supports LSTM which requires cf, pass NULL for GRU
      NULL, ZDNN_3DS, NULL,

      FWD, ZDNN_OK);
}

// Confirm that gru returns OK and expected values when set to return hn
// results from all timesteps
void gru_basic_bwd_hn_all() {
  test_zdnn_api_lstm_gru(
      NNPA_GRUACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The test method also supports LSTM which requires c0, pass in h0 again
      // as a stand-in for c0 which the test will ignore for GRU networks.
      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_z_values, default_uni_input_weights_r_values,
      default_uni_input_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_z_values, default_uni_input_biases_r_values,
      default_uni_input_biases_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_z_values, default_uni_hidden_weights_r_values,
      default_uni_hidden_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_z_values, default_uni_hidden_biases_r_values,
      default_uni_hidden_biases_h_values, ZERO_ARRAY,

      default_bwd_hn_out_all_ts_shape, ZDNN_4DS,
      default_bwd_exp_hn_out_all_ts_values,

      // The test method also supports LSTM which requires cf, pass NULL for GRU
      NULL, ZDNN_3DS, NULL,

      BWD, ZDNN_OK);
}

// Confirm that gru returns OK and expected values when set to return only
// the final hn result
void gru_basic_bwd_hn_final() {
  test_zdnn_api_lstm_gru(
      NNPA_GRUACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The test method also supports LSTM which requires c0, pass in h0 again
      // as a stand-in for c0 which the test will ignore for GRU networks.
      default_uni_h0_shape, ZDNN_3DS, default_uni_h0_values,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_weights_shape, ZDNN_3DS,
      default_uni_input_weights_z_values, default_uni_input_weights_r_values,
      default_uni_input_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_input_biases_shape, ZDNN_2DS,
      default_uni_input_biases_z_values, default_uni_input_biases_r_values,
      default_uni_input_biases_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_weights_shape, ZDNN_3DS,
      default_uni_hidden_weights_z_values, default_uni_hidden_weights_r_values,
      default_uni_hidden_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_uni_hidden_biases_shape, ZDNN_2DS,
      default_uni_hidden_biases_z_values, default_uni_hidden_biases_r_values,
      default_uni_hidden_biases_h_values, ZERO_ARRAY,

      default_bwd_hn_out_final_ts_shape, ZDNN_4DS,
      default_bwd_exp_hn_out_final_ts_values,

      // The test method also supports LSTM which requires cf, pass NULL for GRU
      NULL, ZDNN_3DS, NULL,

      BWD, ZDNN_OK);
}

// Confirm that gru returns OK and expected values when set to return hn
// results from all timesteps
void gru_basic_bidir_hn_all() {
  test_zdnn_api_lstm_gru(
      NNPA_GRUACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_bidir_h0_shape, ZDNN_3DS, default_bidir_h0_values,

      // The test method also supports LSTM which requires c0, pass in h0 again
      // as a stand-in for c0 which the test will ignore for GRU networks.
      default_bidir_h0_shape, ZDNN_3DS, default_bidir_h0_values,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_input_weights_shape, ZDNN_3DS,
      default_bidir_input_weights_z_values,
      default_bidir_input_weights_r_values,
      default_bidir_input_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_input_biases_shape, ZDNN_2DS,
      default_bidir_input_biases_z_values, default_bidir_input_biases_r_values,
      default_bidir_input_biases_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_hidden_weights_shape, ZDNN_3DS,
      default_bidir_hidden_weights_z_values,
      default_bidir_hidden_weights_r_values,
      default_bidir_hidden_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_hidden_biases_shape, ZDNN_2DS,
      default_bidir_hidden_biases_z_values,
      default_bidir_hidden_biases_r_values,
      default_bidir_hidden_biases_h_values, ZERO_ARRAY,

      default_bidir_hn_out_all_ts_shape, ZDNN_4DS,
      default_bidir_exp_hn_out_all_ts_values,

      // The test method also supports LSTM which requires cf, pass NULL for GRU
      NULL, ZDNN_3DS, NULL,

      BIDIR, ZDNN_OK);
}

// Confirm that gru returns OK and expected values when set to return only
// the final hn result
void gru_basic_bidir_hn_final() {
  test_zdnn_api_lstm_gru(
      NNPA_GRUACT,

      default_input_shape, ZDNN_3DS, default_input_values,

      default_bidir_h0_shape, ZDNN_3DS, default_bidir_h0_values,

      // The test method also supports LSTM which requires c0, pass in h0 again
      // as a stand-in for c0 which the test will ignore for GRU networks.
      default_bidir_h0_shape, ZDNN_3DS, default_bidir_h0_values,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_input_weights_shape, ZDNN_3DS,
      default_bidir_input_weights_z_values,
      default_bidir_input_weights_r_values,
      default_bidir_input_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_input_biases_shape, ZDNN_2DS,
      default_bidir_input_biases_z_values, default_bidir_input_biases_r_values,
      default_bidir_input_biases_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_hidden_weights_shape, ZDNN_3DS,
      default_bidir_hidden_weights_z_values,
      default_bidir_hidden_weights_r_values,
      default_bidir_hidden_weights_h_values, ZERO_ARRAY,

      // The fourth gate isn't used for GRU so send ZERO_ARRAY
      default_bidir_hidden_biases_shape, ZDNN_2DS,
      default_bidir_hidden_biases_z_values,
      default_bidir_hidden_biases_r_values,
      default_bidir_hidden_biases_h_values, ZERO_ARRAY,

      default_bidir_hn_out_final_ts_shape, ZDNN_4DS,
      default_bidir_exp_hn_out_final_ts_values,

      // The test method also supports LSTM which requires cf, pass NULL for GRU
      NULL, ZDNN_3DS, NULL,

      BIDIR, ZDNN_OK);
}

int main() {
  UNITY_BEGIN();

// GRU tests with good input requires AIU to get results and
// validate values.
#ifdef TEST_AIU
  // FWD direction tests
  RUN_TEST_ALL_DATATYPES(gru_basic_fwd_hn_all);
  RUN_TEST_ALL_DATATYPES(gru_basic_fwd_hn_final);

  // BWD direction tests
  RUN_TEST_ALL_DATATYPES(gru_basic_bwd_hn_all);
  RUN_TEST_ALL_DATATYPES(gru_basic_bwd_hn_final);

  // BIDIR direction tests
  RUN_TEST_ALL_DATATYPES(gru_basic_bidir_hn_all);
  RUN_TEST_ALL_DATATYPES(gru_basic_bidir_hn_final);
#endif
  return UNITY_END();
}
