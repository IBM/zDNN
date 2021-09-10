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
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

/// Print value at ptr, up to size number of bytes, in binary bits
///
/// \param[in] size Number of bytes to be printed
/// \param[in] ptr Pointer to values
///
/// \return None
///
// cppcheck-suppress unusedFunction
void print_bits(size_t const size, void const *const ptr) {
  unsigned char *b = (unsigned char *)ptr;
  unsigned char byte;
  int i, j;

  for (i = 0; i < size; i++) {
    for (j = 7; j >= 0; j--) {
      byte = (b[i] >> j) & 1;
      printf("%u", byte);
    }
    printf(" ");
  }
  printf("\n");
}

/// Print value at ptr, up to size number of bytes, in hex, with seperations
///
/// \param[in] size Number of bytes to be printed
/// \param[in] ptr Pointer to values
///
/// \return None
///
void print_hex(size_t const size, void const *const ptr) {
  unsigned char *b = (unsigned char *)ptr;

  for (int i = 0; i < size; i++) {
    // every 64-bytes: print line-break and offset
    if ((i % 64) == 0) {
      if (i) {
        printf("\n");
      }
      printf("%08x: ", i);
    }
    // every 4-bytes: print a space
    if ((i % 4) == 0) {
      printf(" ");
    }
    printf("%02X", *(b + i));
  }
  printf("\n");
}

/// Set bit at bit_pos to 1 in a bit128_t struct
///  Bit position is determined from left to right of uint64_t field
///
/// \param[in] field Pointer to bit128_t struct
/// \param[in] bit_pos 0-based bit position
///
/// \return None
///
void setbit_128(bit128_t *field, uint8_t bit_pos) {
  if (bit_pos < 64) {
    field->bits_0to63 |= (uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - bit_pos);
  } else if (bit_pos < 128) {
    field->bits_64to127 |= (uint64_t)1
                           << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 64));
  }
}

/// Test if bit at bit_pos is 1 in a bit128_t struct
///  Bit position is assumed to be left to right of uint64_t field
///
/// \param[in] field Pointer to bit128_t struct
/// \param[in] bit_pos 0-based bit position
///
/// \return true or false
///
bool is_bitset_128(bit128_t field, uint8_t bit_pos) {
  if (bit_pos < 64) {
    return field.bits_0to63 &
           ((uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - bit_pos));
  } else if (bit_pos < 128) {
    return field.bits_64to127 &
           ((uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 64)));
  } else {
    return false;
  }
}

/// Set bit at bit_pos to 1 in a bit256_t struct
///  Bit position is determined from left to right of uint64_t field
///
/// \param[in] field Pointer to bit256_t struct
/// \param[in] bit_pos 0-based bit position
///
/// \return None
///
void setbit_256(bit256_t *field, uint16_t bit_pos) {
  if (bit_pos < 64) {
    field->bits_0to63 |= (uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - bit_pos);
  } else if (bit_pos < 128) {
    field->bits_64to127 |= (uint64_t)1
                           << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 64));
  } else if (bit_pos < 192) {
    field->bits_128to191 |= (uint64_t)1
                            << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 128));
  } else if (bit_pos < 256) {
    field->bits_192to255 |= (uint64_t)1
                            << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 192));
  }
}

/// Test if bit at bit_pos is 1 in a bit256_t struct
///  Bit position is assumed to be left to right of uint64_t field
///
/// \param[in] field Pointer to bit256_t struct
/// \param[in] bit_pos 0-based bit position
///
/// \return true or false
///
bool is_bitset_256(bit256_t field, uint16_t bit_pos) {
  if (bit_pos < 64) {
    return field.bits_0to63 &
           ((uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - bit_pos));
  } else if (bit_pos < 128) {
    return field.bits_64to127 &
           ((uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 64)));
  } else if (bit_pos < 192) {
    return field.bits_128to191 &
           ((uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 128)));
  } else if (bit_pos < 256) {
    return field.bits_192to255 &
           ((uint64_t)1 << ((BIT_SIZEOF(uint64_t) - 1) - (bit_pos - 192)));
  } else {
    return false;
  }
}

/// Get the number of elements based on a tensor's dimensions.
///
/// \param[in] ztensor zDNN tensor to get element count from
/// \param[in] elements_mode controls how to count elements.
///     ELEMENTS_ALL - All elements including padding for concatenated tensors
///                    (ie the tfrmd shape)
///     ELEMENTS_CONCAT_SINGLE - The number of elements of a single
///                              concat within the tensor without padding
///                              (ie the pre_tfrmd shape)
///     ELEMENTS_CONCAT_WO_PAD - return total number of elements (all concats)
///                              but do not include the zero padding elements
///                              between concats (ie pre_tfrmd * num_concats)
/// \return number of elements based on desired mode
///
uint64_t get_num_elements(const zdnn_ztensor *ztensor, elements_mode mode) {
  uint64_t num_elements = 1;
  uint32_t *dims_ptr;
  int i;

  // Setup how to loop over the shape based on the mode.
  switch (mode) {
  case ELEMENTS_ALL:
    // tfrmd_desc shape accounts for all elements including concat padding.
    dims_ptr = &(ztensor->transformed_desc->dim4);
    // Loop over all dims since tfrmd_dec sets any "unused" dimensions to 1.
    i = 0;
    break;
  case ELEMENTS_CONCAT_SINGLE:
  case ELEMENTS_CONCAT_WO_PAD:
    // Use pre_tfrmd_desc as we document that should be the shape of a single
    // concat/gate and not the combined shape.
    dims_ptr = &(ztensor->pre_transformed_desc->dim4);
    // Loop will start at outermost dimension we expect for the layout.
    // For example: 2D gets dim2 and dim1. 3D gets dim3, dim2, and dim1.
    i = ZDNN_MAX_DIMS -
        get_data_layout_dims(ztensor->pre_transformed_desc->layout);
    break;
  default:
    LOG_WARN("%d is not a supported elements_mode", mode);
    return 0;
    break;
  }

  // Multiply by the size of each expected dimension
  for (; i < ZDNN_MAX_DIMS; i++) {
    num_elements *= (uint64_t)dims_ptr[i];
  }

  if (mode == ELEMENTS_CONCAT_WO_PAD) {
    short num_gates =
        get_data_layout_num_gates(ztensor->transformed_desc->layout);
    // Non-concatenated layouts return 0 gates. Change to 1 so we don't just
    // return 0 for non-concatenated tensors.
    if (!num_gates) {
      num_gates = 1;
    }
    num_elements *= num_gates;
  }

  return num_elements;
}

/// Prints out DLFLOAT16 buffer data.
///
/// Example Output:
/// Buffer:
/// 	Size: 16384
/// 	Data:
/// 		INDEX		HEX
/// 		      0		0041
/// 		      1		4100
/// 		    128		0042
/// 		    129		4200
/// 		    256		0043
///           .      .
/// 		  12544		004F
/// 		  12545		4F00
/// 		  12672		0050
/// 		  12673		5000
/// =========================================
///
/// \param[in] buffer a pointer to a data buffer
/// \param[in] buffer_size the size of the buffer
///
/// \return None
///
void print_dlf16_buffer(void *buffer, uint64_t buffer_size) {
  printf("Buffer:\n");
  printf("\tSize:%" PRIu64 "\n", buffer_size);
  printf("\tData:\n\t\tINDEX\t\tHEX\n");
  for (uint64_t i = 0; i < (buffer_size / 2); i++) {
    printf("\t\t%" PRIu64 "\t\t%04x\n", i, ((uint16_t *)buffer)[i]);
  }
}

/// Prints out tensor descriptor
///
/// Example Output:
/// Descriptor:
///                     Outermost                               Innermost
///     Dimensions:     1               64              64              1
///     Layout: ZDNN_NHWC       Format: ZDNN_FORMAT_4DFEATURE   Type:   FP16
///
/// \param[in] desc A tensor descriptor
///
/// \return None
///
void print_desc(zdnn_tensor_desc *desc) {
  printf("Descriptor:\n"
         "\t\t\tOutermost\t\t\t\tInnermost\n"
         "\tDimensions:\t%u\t\t%u\t\t%u\t\t%u\n"
         "\tLayout:\t%s\tFormat:\t%s\tType:\t%s\n",
         desc->dim4, desc->dim3, desc->dim2, desc->dim1,
         get_data_layout_str(desc->layout), get_data_format_str(desc->format),
         get_data_type_str(desc->type));
}

/// Prints out ztensor information.
///
/// Example Output:
/// =========================================
/// Contents of zdnn_ztensor: input
/// Pre-transformed Descriptor:
///                         Outermost                               Innermost
///         Dimensions:     1               64              64              1
///         Layout: ZDNN_NHWC       Format: ZDNN_FORMAT_4DFEATURE   Type:   FP16
/// Transformed Descriptor:
///                         Outermost                               Innermost
///         Dimensions:     1               64              64              1
///         Layout: ZDNN_NHWC       Format: ZDNN_FORMAT_4DFEATURE   Type:
///         ZDNN_DLFLOAT16
/// Buffer Addr:    5011007000      Size:   524288
/// Transformed:     True
///
/// Pool padding: SAME_PADDING
///
/// Parameter kernel_height (uint32_t): 64
///
/// Parameter kernel_width (uint32_t): 64
///
/// Parameter stride_height (uint32_t): 1
///
/// Parameter stride_width (uint32_t): 1
/// Buffer:
///   Size:       16384
///   Data:
/// 		INDEX		HEX
/// 		      0		0041
/// 		      1		4100
/// 		    128		0042
/// 		    129		4200
/// 		    256		0043
///           .      .
/// 		  12544		004F
/// 		  12545		4F00
/// 		  12672		0050
/// 		  12673		5000
/// =========================================
///
/// \param[in] ztensor Pointer to zdnn_ztensor to print
/// \param[in] name Name of the zdnn_ztensor (e.g., variable name)
/// \param[in] print_data Print data buffer or not
///
/// \return None
///
void print_ztensor(const zdnn_ztensor *ztensor, char *name, bool print_data) {
  printf("\n=========================================\n"
         "Contents of zdnn_ztensor: %s\n",
         name);

  printf("Pre-transformed ");
  print_desc(ztensor->pre_transformed_desc);

  printf("Transformed ");
  print_desc(ztensor->transformed_desc);

  printf("Buffer Addr:\t%" PRIxPTR "\tSize:\t%" PRIu64 "\n",
         (uintptr_t)ztensor->buffer, ztensor->buffer_size);

  printf("Transformed:\t");
  if (ztensor->is_transformed) {
    printf("True");
  } else {
    printf("False");
  }
  printf("\n");

  if (print_data) {
    print_dlf16_buffer(ztensor->buffer, ztensor->buffer_size);
  }
  printf("=========================================\n");
}
