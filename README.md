# zDNN API Reference

## Contacts

- Nicholas Marion (nmarion@us.ibm.com)
- Andreas Krebbel (krebbel@linux.ibm.com)

## Version

v0.3.0

## Table of Contents <a id="TOC"></a>

1. [Overview](#overview)
2. [Environment](#environment)
3. [Common Data Types and Structs](#common-types-and-structs)

   - [Version Information](#common-version-info)
   - [zDNN zTensor](#common-ztensor)
     - [General zTensor Requirements](#gen-zten-reqs)
     - [Concatenated zTensor Requirements](#concat-zten-reqs)
   - [zDNN Tensor Descriptors](#common-descriptors)
   - [zDNN Data Layouts](#common-layouts)
   - [zDNN Data Formats](#common-formats)
   - [zDNN Data Types](#common-types)
   - [zDNN Statuses](#common-statuses)

4. [Runtime Environment Variables](#env-vars)
5. [API Reference](#api-reference)

   - [Support Functions](#support-functions)
   - [Data Transformation](#data-transformation)
   - [Operations](#operations)

     - [Element-wise](#elwise-ops)
     - [Activation](#act-ops)
     - [Normalization](#norm-ops)
     - [Matmul with Operation](#zdnn_matmul_op)
     - [Matmul Broadcast with Operation](#zdnn_matmul_bcast_op)
     - [LSTM](#zdnn_lstm)
     - [GRU](#zdnn_gru)
     - [Average Pool 2D](#zdnn_avgpool2d)
     - [Max Pool 2D](#zdnn_maxpool2d)
     - [Convolution 2D](#zdnn_conv2d)

   - [Convenience Functions](#convenience-functions)

6. [Usage Examples](#usage-examples)

## Overview

**Deep Learning Library** - the deep learning library support (zDNN) is the SW
enablement technology provided by IBM to meet the following requirements:

- Specialized-function-assist instructions are intended to provide performance
  improvements for specific operations used in software libraries, utilities,
  and operating system (OS) services. The facilities and instructions described
  as specialized-function-assist instructions may be replaced or removed in the
  future. As such, the IBM recommendation for these instructions is that a
  software library or operating system function be used instead of directly
  accessing the instructions. This is the function provided by zDNN.
- zAIU has very complex data layout requirements; these requirements arrange the
  tensor to enhance the performance characteristics of the operations. zDNN will
  format the tensor appropriately on behalf of the caller, and it will do so
  using an optimized approach.
- For deep learning operations, zAIU requires the use of an internal data type
  (DLFLOAT16). This is a 2-byte data type, similar in concept to Brain
  float (BFLOAT); that is, it is an AI optimized format that is used to speed up
  training and inference (from 4-byte formats) while minimizing the loss of
  accuracy at inference time.

The zDNN library will provide a set of APIs that an exploiter will utilize to
drive the desired request. zDNN will be available on both z/OS and Linux on Z;
the inclusion of Linux on Z provides particular benefit, as it will allow us to
enable acceleration in frameworks for z/OS via z/OS Container Extensions (zCX).

---

## Environment

z/OS:

- Problem state
- AMODE64
- XPLINK

### Alignment requirements

#### AIU Op Limits

_This implies a zDNN limitation as well at this point._

- For all ops:

  - Number of elements in any dimension must not exceed the value returned by
    `zdnn_get_nnpa_max_dim_idx_size()`
  - Total number of bytes required for storing a transformed tensor must not
    exceed the value returned by `zdnn_get_nnpa_max_tensor_size()`

### Application interfaces for zAIU Enterprise Neural Network Inference

#### zDNN General

The zDNN deep learning library provides the standard IBM Z software interface to
the zAIU. This IBM-provided C library provides a set of functions that handle
the data transformation requirements of the AIU and provide wrapper functions
for the NNPA instruction primitives.

The zDNN functions use the following criteria to determine if zAIU can be used
to accelerate a deep learning primitive:

- Neural Network Processing Assist (NNPA) facility indicator in the system STFLE
  output.
- Output of the NNPA-QAF (Query Available Functions) request.

#### Using zDNN

To use the IBM-provided zDNN C library for the NNPA instruction, follow these
steps:

1. Link or re-link applications to use the IBM-provided zDNN. The IBM-provided
   zDNN is a library file in the z/OS UNIX System Services file system and can
   be statically or dynamically linked into your applications. The paths for the
   zDNN archive file and the zDNN header files are:

**z/OS (LE required):** Path for 64-bit dynamic library files:

- `/lib/libzdnn.so`
- `/lib/libzdnn.x`

Path for the zDNN header files:

- `/usr/include/`

The XL C/C++ compiler and the z/OS Language Environment provide various
environment variables to control processing, in addition to the variables
provided by the zDNN library itself.

1. Use the environment variable `_CEE_RUNOPTS` to specify invocation Language
   Environment runtime options. For more information about using the environment
   variable `_CEE_RUNOPTS` and other C and LE variables, see z/OS XL C/C++
   Programming Guide.

2. For environment variables accepted by the zDNN library, see
   [Runtime Environment Variables](#env-vars).

**Linux on Z:**

On Linux on Z we expect to ship source as well a package-installable
library and header. The library installation will conform to the standards of
the packaging method chosen.

---

## Common Types and Structs

Include Files: `zdnn.h`

### Version Information <a id="common-version-info"></a>

[Back to Table of Contents](#TOC)

```
#define ZDNN_VERSION "0.3.0"
#define ZDNN_VERNUM 0x000300 // 0x[major][minor][patch]
#define ZDNN_VER_MAJOR 0
#define ZDNN_VER_MINOR 3
#define ZDNN_VER_PATCH 0
```

1. zDNN major version (_ZDNN_VER_MAJOR_) will be incremented if any backwards
   incompatible changes are introduced to the API. It may also include minor and
   patch level changes. Patch and minor version will be reset to 0 when major
   version is incremented.
2. zDNN minor version (_ZDNN_VER_MINOR_) will be incremented if new, backwards
   compatible functionalities are introduced to the API or if any API
   functionalities are marked as deprecated. It may also include patch level
   changes. Patch version will be reset to 0 when minor version is incremented.
3. zDNN patch version (_ZDNN_VER_PATCH_) will be incremented if only backwards
   compatible bug fixes are introduced. A bug fix being defined as an internal
   change that fixes incorrect behavior.

Functions for checking version incompatibility with the zDNN load library are
provided and described in the [Support Functions](#support-functions) section.

### zDNN zTensor <a id="common-ztensor"></a>

[Back to Table of Contents](#TOC)

```
typedef struct zdnn_ztensor {
  zdnn_tensor_desc
      *pre_transformed_desc; // tensor's shape information before transformation
  zdnn_tensor_desc *transformed_desc; // transformed tensor's shape information
  bool is_transformed;  // indicator if data in buffer has been transformed
  uint64_t buffer_size; // tensor size in bytes
  char reserved[32];    // not currently used, exploiter should not touch.
  void *buffer;         // pointer to the tensor in memory
} zdnn_ztensor;
```

#### General zTensor Requirements <a id="gen-zten-reqs"></a>

[Back to Table of Contents](#TOC)

- `buffer` requirements:
  - Calling [zdnn_init_ztensor_with_malloc](#zdnn_init_ztensor_with_malloc)
    automatically allocates and sets a valid `buffer` for a tensor.
  - `buffer` field must point to storage allocated of sufficient size to contain
    the transformed tensor data described by the its `transformed_desc` field.
    - Calling [zdnn_getsize_ztensor](#zdnn_getsize_ztensor) with the tensor's
      `transformed_desc` returns the required size.
  - Start of `buffer` field must be 4k aligned.

#### Concatenated zTensor Requirements <a id="concat-zten-reqs"></a>

[Back to Table of Contents](#TOC)]

- You must use
  [zdnn_generate_transformed_desc_concatenated](#zdnn_generate_transformed_desc_concatenated)
  with the correct concatenation type
  - Do not use `zdnn_generate_transformed_desc` with concatenated tensors
- The pre-transformed shape dimensions should not include the concatenation.
  - For example, the pre-transformed shape should be that of a single gate or
    unidirectional RNN output and not the shape of the combined gates or RNN
    bidirectional output.
- Afterward transform with [zdnn_transform_ztensor](#zdnn_transform_ztensor) as
  normal
- Must follow [general tensor requirements](#gen-zten-reqs)

### zDNN Tensor Descriptors <a id="common-descriptors"></a>

[Back to Table of Contents](#TOC)

```
typedef struct zdnn_tensor_desc {
  zdnn_data_layouts layout; // data layout
  zdnn_data_formats format; // internal use only
  zdnn_data_types type;     // data type
  uint32_t dim4;            // number of elements in outermost dimension
  uint32_t dim3;            // ... outer dimension
  uint32_t dim2;            // ... inner dimension
  uint32_t dim1;            // number of elements in innermost dimension
} zdnn_tensor_desc;
```

#### Programming Notes

- Helper methods
  [zdnn_init_pre_transformed_desc](#zdnn_init_pre_transformed_desc) and
  [zdnn_generate_transformed_desc](#zdnn_generate_transformed_desc) or
  [zdnn_generate_transformed_desc_concatenated](#zdnn_generate_transformed_desc_concatenated)
  will set the correct dims based on the layout and format.
- The [layout](#common-layouts) of the tensor descriptor affects the expected
  order of the dims. For example:
  - For tensors with less than 4 dimensions, unspecified dims:
    - In the [pre_transformed_desc](#common-ztensor) are ignored. For example a
      [ZDNN_3D](#common-layouts) expects values in dim4, dim3, and dim2.
    - In the [transformed_desc](#common-ztensor) "unused" dims must be 1.
  - A [ZDNN_NCHW](#common-layouts) expects dims such that dim4 = N, dim3 = H,
    dim2 = W, dim1 = C
  - A [ZDNN_HWCK](#common-layouts) expects dims such that dim4 = W, dim3 = W,
    dim2 = C, dim1 = K
- The [format](#common-formats) changes the expected dims order for
  [ZDNN_4D](#common-layouts) tensors layouts
  - [ZDNN_FORMAT_4DFEATURE](#common-formats) expects dims such that dim4 = N,
    dim3 = H, dim2 = W, dim1 = C
  - [ZDNN_FORMAT_4DKERNEL](#common-formats) expects dims such that dim4 = H,
    dim3 = W, dim2 = C, dim1 = K

### zDNN Data Layouts <a id="common-layouts"></a>

[Back to Table of Contents](#TOC)

The following are layouts for zDNN ztensor descriptors. These indicate the
number and order of dimensions to expect for the ztensor data.

```
typedef enum zdnn_data_layouts {
  ZDNN_1D,           // 1d tensor
  ZDNN_2D,           // 2d tensor
  ZDNN_2DS,          // represents special 2D tensors required by LSTM/GRU
  ZDNN_BIDIR_OUTPUT, // concatenated output (FWD, BWD) for bidirectional
                     // LSTM/GRU
  ZDNN_3D,           // 3d tensor
  ZDNN_3DS,          // represents special 3D tensors required by
                     // LSTM/GRU/Softmax/Matmul
  ZDNN_ZRH,          // represents (update, reset, hidden) used by GRU
  ZDNN_4D,           // 4d tensor
  ZDNN_NHWC,         // 4d feature tensor in NHWC
  ZDNN_NCHW,         // 4d feature tensor in NCHW
  ZDNN_FICO,         // represents (forget, input, cell, output) used by LSTM
  ZDNN_HWCK          // 4d kernel CNN tensor
} zdnn_data_layouts;
```

Some layouts also indicate special re-arrangement of the data during ztensor
transformation.

- `ZDNN_2DS` - The outermost dimension of the original shape is promoted to dim4
  during transformation. For example, a shape of (a, b) becomes [a, 1, 1, b]
  (dim4, dim3, dim2, dim1) in the `transformed_desc`
- `ZDNN_3DS` - The outermost dimension of the original shape is promoted to dim4
  during transformation. For example, a shape of (a, b, c) becomes [a, 1, b, c]
  (dim4, dim3, dim2, dim1) in the `transformed_desc`
- `ZDNN_BIDIR_OUTPUT` - Set automatically in `transformed_desc` based on
  `concat_type` when calling `zdnn_generate_transformed_desc_concatenated()`.
  This layout supports concatenated FWD and BWD output on the innermost
  dimension for bidirectional RNN results. Supported with
  `pre_transformed_layout` of `ZDNN_3DS`.
- `ZDNN_ZRH` - Set automatically in `transformed_desc` based on `concat_type`
  when calling `zdnn_generate_transformed_desc_concatenated()`. During
  transformation, the input data gates are re-grouped by their outermost
  dimension. For example, if each 2D input data was shaped g1=(a1, b1), g2=(a2,
  b2), and g3=(a3, b2), then the transformed ztensor would look like (a,
  b1+b2+b3). Supported with `pre_transformed_layout` of `ZDNN_2DS` or
  `ZDNN_3DS`.
- `ZDNN_FICO` - Similar to `ZDNN_ZRH` except four gates instead of three.

### zDNN Data Formats <a id="common-formats"></a>

[Back to Table of Contents](#TOC)

```
typedef enum zdnn_data_formats {
  ZDNN_FORMAT_4DFEATURE, // tensor in AIU data layout format 0
  ZDNN_FORMAT_4DKERNEL, // tensor in AIU data layout format 1
} zdnn_data_formats;
```

### zDNN Data Types <a id="common-types"></a>

[Back to Table of Contents](#TOC)

```
typedef enum zdnn_data_types {
  ZDNN_DLFLOAT16, // 16-bit deep learning format
  BFLOAT, // Brain floating point format
  FP16, // 16-bit IEEE-754 floating point format
  FP32, // 32-bit IEEE-754 floating point format
} zdnn_data_types;
```

### zDNN Statuses <a id="common-statuses"></a>

[Back to Table of Contents](#TOC)

<!-- prettier-ignore -->
| Mnemonic Constant                | Value      | Meaning                        |
| -------------------------------- | ---------- | ------------------------------ |
| ZDNN_OK                          | 0x00000000 | Success.                       |

#### Warning Statuses <a id="warning-statuses"></a>

<!-- prettier-ignore -->
| Mnemonic Constant                | Value      | Meaning                        |
| -------------------------------- | ---------- | ------------------------------ |
| ZDNN_ELEMENT_RANGE_VIOLATION     | 0x00020001 | AIU operation resulted in data that was out of the normal range. |

_Note: ZDNN_ELEMENT_RANGE_VIOLATION indicates a **range violation** occurred for
the AIU operation based on the data in the tensors. This usually indicates an
overflow of the NNPA internal data type, but can also be associated with
operation specific errors, such as "divide by zero". See the "z/Architecture
Principles of Operation" for information about range violation on the operation
that encountered the violation._

#### General Failing Statuses <a id="failing-statuses"></a>

<!-- prettier-ignore -->
| Mnemonic Constant                | Value      | Meaning                        |
| -------------------------------- | ---------- | ------------------------------ |
| ZDNN_INVALID_SHAPE\*             | 0x00040001 | Invalid shape information in one (or more) of the input/output tensor(s).      |
| ZDNN_INVALID_LAYOUT              | 0x00040002 | Invalid layout information in one (or more) of the input/output tensor(s).     |
| ZDNN_INVALID_TYPE\*              | 0x00040003 | Invalid type information in one (or more) of the input/output tensor(s).       |
| ZDNN_INVALID_FORMAT\*            | 0x00040004 | Invalid format information in one (or more) of the input/output tensor(s).     |
| ZDNN_INVALID_DIRECTION           | 0x00040005 | Invalid RNN direction.                                                         |
| ZDNN_INVALID_CONCAT_TYPE         | 0x00040006 | Invalid concatenation type.                                                    |
| ZDNN_INVALID_STRIDE_PADDING\*    | 0x00040007 | Invalid padding type parameter for current strides.                            |
| ZDNN_INVALID_STRIDES\*           | 0x00040008 | Invalid stride height or width parameter.                                      |
| ZDNN_MISALIGNED_PARMBLOCK\*      | 0x00040009 | NNPA parameter block is not on double word boundary.                           |
| ZDNN_INVALID_CLIPPING_VALUE      | 0x0004000A | Invalid clipping for the specified operation.                                  |
| ZDNN_ALLOCATION_FAILURE          | 0x00100001 | Can not allocate storage.                                                      |
| ZDNN_INVALID_BUFFER              | 0x00100002 | Buffer address is NULL or not on 4K-byte boundary or insufficient buffer size. |
| ZDNN_CONVERT_FAILURE             | 0x00100003 | Floating point data conversion failure.                                        |
| ZDNN_INVALID_STATE               | 0x00100004 | Invalid zTensor state.                                                         |
| ZDNN_UNSUPPORTED_AIU_EXCEPTION   | 0x00100005 | AIU operation returned an unexpected exception.                                |

_Note: \*In certain scenarios, these statuses are returned only if
[ZDNN_ENABLE_PRECHECK](#env-vars) is enabled. When not enabled, these scenarios
will lead to abnormal program termination._

#### Hardware Statuses <a id="hw-statuses"></a>

The following statuses indicate issues returned from the hardware.

<!-- prettier-ignore -->
| Mnemonic Constant                | Value      | Meaning                        |
| -------------------------------- | ---------- | ------------------------------ |
| ZDNN_UNSUPPORTED_PARMBLOCK       | 0x000C0001 | NNPA parameter block format is not supported by the model.             |
| ZDNN_UNAVAILABLE_FUNCTION        | 0x000C0002 | Specified NNPA function is not defined or installed on the machine.    |
| ZDNN_UNSUPPORTED_FORMAT          | 0x000C0010 | Specified tensor data layout format is not supported.                  |
| ZDNN_UNSUPPORTED_TYPE            | 0x000C0011 | Specified tensor data type is not supported.                           |
| ZDNN_EXCEEDS_MDIS                | 0x000C0012 | Tensor dimension exceeds maximum dimension index size (MDIS).          |
| ZDNN_EXCEEDS_MTS                 | 0x000C0013 | Total number of bytes in tensor exceeds maximum tensor size. (MTS).    |
| ZDNN_MISALIGNED_TENSOR           | 0x000C0014 | Tensor address is not on 4K-byte boundary.                             |
| ZDNN_MISALIGNED_SAVEAREA         | 0x000C0015 | Function specific save area address is not on 4K-byte boundary.        |

The meaning of the following hardware statuses vary based on operation. See the
operation that returned the status for the specific meaning.

<!-- prettier-ignore -->
| Mnemonic Constant                | Value      | Meaning                        |
| -------------------------------- | ---------- | ------------------------------ |
| ZDNN_FUNC_RC_F000                | 0x000CF000 | Function specific response code (F000). |
| ZDNN_FUNC_RC_F001                | 0x000CF001 | Function specific response code (F001). |
| ZDNN_FUNC_RC_F002                | 0x000CF002 | Function specific response code (F002). |
| ZDNN_FUNC_RC_F003                | 0x000CF003 | Function specific response code (F003). |
| ZDNN_FUNC_RC_F004                | 0x000CF004 | Function specific response code (F004). |
| ZDNN_FUNC_RC_F005                | 0x000CF005 | Function specific response code (F005). |
| ZDNN_FUNC_RC_F006                | 0x000CF006 | Function specific response code (F006). |
| ZDNN_FUNC_RC_F007                | 0x000CF007 | Function specific response code (F007). |
| ZDNN_FUNC_RC_F008                | 0x000CF008 | Function specific response code (F008). |
| ZDNN_FUNC_RC_F009                | 0x000CF009 | Function specific response code (F009). |

---

## Runtime Environment Variables <a id="env-vars"></a>

[Back to Table of Contents](#TOC)

- `ZDNN_ENABLE_PRECHECK`: true/false
  - If set to `true`, tensor integrity prechecks are run before issuing NNPA
    operations.
  - Enabling precheck may impact performance.
  - Enable to debug issues which cause hardware exceptions that otherwise would
    result in abnormal program termination.
- `ZDNN_STATUS_DIAG`: nnnnnnnn (decimal) or 0xnnnnnnnn (hexadecimal)
  - Prints or produces diagnostic information whenever zDNN status code is equal
    to the specified value. Only one status value can be specified.

_The following are only available when the zDNN library was built with
`ZDNN_CONFIG_DEBUG` enabled._

- `ZDNN_LOGLEVEL`: off/fatal/error/warn/info/debug/trace
  - Sets logging facility's output level
- `ZDNN_LOGMODULE`: module name(s)
  - Produces log output only when the issuer's module name is in the list. You
    may specify multiple module names by separating them with either commas or
    spaces.

### Programming Notes

- Environment variables settings are checked during initial library load by
  [zdnn_init](#zdnn_init).
- To change environment variable settings afterward, [zdnn_init](#zdnn_init)
  must be called again manually.

---

## API Reference

[Back to Table of Contents](#TOC)

- [Support Functions](#support-functions)
- [Data Transformation](#data-transformation)
- [Operations](#operations)
- [Convenience Functions](#convenience-functions)

---

## Support Functions

[Back to Table of Contents](#TOC)

- [Initialization](#zdnn_init)
- [Query](#zdnn_get_nnpa_max_dim_idx_size)
- [Get Size](#zdnn_getsize_ztensor)
- [Initialize pre-transformed tensor descriptor](#zdnn_init_pre_transformed_desc)
- [Generate transformed tensor descriptor](#zdnn_generate_transformed_desc)
- [Generate concatenated transformed tensor descriptor](#zdnn_generate_transformed_desc_concatenated)
- [Initialize zTensor](#zdnn_init_ztensor)
- [Initialize zTensor with memory allocate](#zdnn_init_ztensor_with_malloc)
- [Reset zTensor](#zdnn_reset_ztensor)
- [Allocate memory for zTensor](#zdnn_allochelper_ztensor)
- [De-allocate memory for zTensor](#zdnn_free_ztensor_buffer)
- [Retrieve status message of the status code](#zdnn_get_status_message)
- [Reshape zTensor](#zdnn_reshape_ztensor)
- [Check if version is runnable](#zdnn_is_version_runnable)
- [Get maximum runnable version](#zdnn_get_max_runnable_version)

---

### zdnn_init

#### Description

Initialize the zDNN library. This sends an NNPA_QAF to query the NNPA and loads
the current environment variable settings.

This needs to be invoked at least once if zDNN library is statically-linked. It
is automatically invoked if zDNN library is dynamically loaded.

#### Format

```
void zdnn_init();
```

#### Parameters

None

#### Returns

None

---

### zdnn_get_nnpa_max_dim_idx_size

#### Description

Retrieve the maximum dimension index size value currently supported by the AIU
from zDNN's internal memory.

#### Format

```
uint32_t zdnn_get_nnpa_max_dim_idx_size();
```

#### Parameters

None

#### Returns

Maximum dimension index size supported by the AIU

---

### zdnn_get_nnpa_max_tensor_size

#### Description

Retrieve the maximum tensor size value (number of bytes required for storing a
transformed tensor) currently supported by the AIU from zDNN's internal memory.

#### Format

```
uint64_t zdnn_get_nnpa_max_tensor_size();
```

#### Parameters

None

#### Returns

Maximum tensor size supported by the AIU

---

### zdnn_is_nnpa_installed

#### Description

Interrogates the hardware to determine if the NNPA and NNP-internal data type
(DLFLOAT16) conversion instructions are installed.

Use this function during application initialization to determine whether the AIU
hardware is available.

#### Format

```
bool zdnn_is_nnpa_installed();
```

#### Parameters

- None.

#### Returns

`true` if NNPA and zdnn conversion instructions are installed, `false`
otherwise.

---

### zdnn_is_nnpa_function_installed

#### Description

Query, from zDNN internal memory, if requested NNPA functions are available.

#### Format

```
bool zdnn_is_nnpa_function_installed(int count, ...);
```

#### Parameters

- `int count`

  - number of NNPA functions to check

- `... (additional arguments)`

  - Function names separated by commas, e.g., _NNPA_MUL, NNPA_MIN_

```
NNPA_QAF
NNPA_ADD
NNPA_SUB
NNPA_MUL
NNPA_DIV
NNPA_MIN
NNPA_MAX
NNPA_LOG
NNPA_EXP
NNPA_RELU
NNPA_TANH
NNPA_SIGMOID
NNPA_SOFTMAX
NNPA_BATCHNORMALIZATION
NNPA_MAXPOOL2D
NNPA_AVGPOOL2D
NNPA_LSTMACT
NNPA_GRUACT
NNPA_CONVOLUTION
NNPA_MATMUL_OP
NNPA_MATMUL_OP_BCAST23
```

#### Returns

`true` if all queried formats are installed or if `count` is zero, `false`
otherwise.

---

### zdnn_is_nnpa_parmblk_fmt_installed

#### Description

Query, from zDNN internal memory, if requested parameter block formats are
installed.

#### Format

```
bool zdnn_is_nnpa_parmblk_fmt_installed(int count, ...);
```

#### Parameters

- `int count`

  - number of NNPA parameter block formats to check

- `... (additional arguments)`

  - NNPA parameter block formats separated by commas

```
NNPA_PARMBLKFORMAT_0
```

#### Returns

`true` if all queried formats are installed or if `count` is zero, `false`
otherwise.

---

### zdnn_is_nnpa_datatype_installed

#### Description

Query, from zDNN internal memory, if requested NNPA data type are installed.

#### Format

```
bool zdnn_is_nnpa_datatype_installed(uint16_t types_bitmask);
```

#### Parameters

- `uint16_t types_bitmask`

  - OR'd type bitmasks as defined in zdnn_query_datatypes enum

```
QUERY_DATATYPE_INTERNAL1
```

#### Returns

`true` if all queried data types are installed, `false` otherwise.

---

### zdnn_is_nnpa_layout_fmt_installed

#### Description

Query, from zDNN internal memory, if requested NNPA data layout format are
installed.

#### Format

```
bool zdnn_is_nnpa_layout_fmt_installed(uint32_t layout_bitmask);
```

#### Parameters

- `uint32_t layout_bitmask`

  - OR'd layout bitmasks as defined in zdnn_query_layoutfmts enum

```
QUERY_LAYOUTFMT_4DFEATURE
QUERY_LAYOUTFMT_4DKERNEL
```

#### Returns

`true` if all queried data layouts are installed, `false` otherwise.

---

### zdnn_is_nnpa_conversion_installed

#### Description

Query, from zDNN internal memory, if requested NNPA data-type to/from BFP format
conversions are installed.

#### Format

```
bool zdnn_is_nnpa_conversion_installed(nnpa_data_type type,
                                       uint16_t format_bitmask);
```

#### Parameters

- `nnpa_data_type type`

  - NNPA data-type number as defined in nnpa_data_type enum

```
NNPA_DATATYPE_1
```

- `uint32_t format_bitmask`

  - OR'd BFP format bitmasks as defined in zdnn_query_bfpfmts enum

```
QUERY_BFPFMT_TINY (FP16)
QUERY_BFPFMT_SHORT (FP32/BFLOAT)
```

#### Returns

`true` if all queried conversions are installed, `false` otherwise.

---

### zdnn_get_library_version

#### Description

Retrieve library version number as a 32-bit hex value
(`0x00[major][minor][patch]`).

#### Format

```
uint32_t zdnn_get_library_version();
```

#### Returns

Library version number in `0x00[major][minor][patch]` format.

---

### zdnn_get_library_version_str

#### Description

Retrieve the library version number and build information as a string.

#### Format

```
char *zdnn_get_library_version_str();
```

#### Returns

Library version number and build information as a string.

---

### zdnn_refresh_nnpa_query_result

#### Description

Refresh zDNN in-memory query result from zAIU.

#### Format

```
zdnn_status zdnn_refresh_nnpa_query_result();
```

#### Parameters

None

##### Programming Notes

This is called automatically as a part of `zdnn_init` and should not need to be
called directly. Manually refreshing query results before making other
`zdnn_query_*` calls may noticeably impact performance.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_UNAVAILABLE_FUNCTION`

---

### zdnn_getsize_ztensor

#### Description

Used to determine the buffer size required for the transformed tensor (including
concatenated) in zDNN transformed format. Requires tensor descriptor
(`zdnn_tensor_desc`) with transformed shape information.

#### Format

```
uint64_t zdnn_getsize_ztensor(const zdnn_tensor_desc *tfrmd_desc);
```

#### Parameters

- `zdnn_tensor_desc *tfrmd_desc`

  - Contains transformed information about the shape, layout and data type.

#### Returns zdnn_status indications

- required buffer size in bytes

---

### zdnn_init_pre_transformed_desc

#### Description

Initialize tensor descriptor (`zdnn_tensor_desc`) struct with pre-transformed
(original) shape information.

#### Format

```
void zdnn_init_pre_transformed_desc(zdnn_data_layouts layout,
                                    zdnn_data_types type,
                                    zdnn_tensor_desc *pre_tfrmd_desc, ...);
```

#### Parameters

- `zdnn_data_layouts layout`

  - data layout

- `zdnn_data_types type`

  - data type

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - output zdnn_tensor_desc struct

- `... (additional arguments)`

  - Variadic: number of elements in each dimension in accordance to the layout,
    in outermost to innermost order

#### Returns

- None

---

### zdnn_generate_transformed_desc

#### Description

Generate transformed tensor descriptor information based on supplied
pre-transformed tensor descriptor.

#### Format

```
zdnn_status zdnn_generate_transformed_desc(
    const zdnn_tensor_desc *pre_tfrmd_desc, zdnn_tensor_desc *tfrmd_desc);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_tensor_desc *tfrmd_desc`

  - output `zdnn_tensor_desc` struct

#### zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_LAYOUT` - `pre_tfrmd_desc->layout` is not recognized or is a
  layout only used for concatenated tensors.

---

### zdnn_generate_transformed_desc_concatenated

#### Description

Generate concatenated transformed tensor descriptor information (for LSTM or GRU
layers) based on a supplied pre-transformed tensor descriptor.

#### Format

```
zdnn_status zdnn_generate_transformed_desc_concatenated(
    const zdnn_tensor_desc *pre_tfrmd_desc,
    zdnn_ztensor_concat_types concat_type, zdnn_tensor_desc *tfrmd_desc);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_ztensor_concat_types concat_type`

  - Valid concatenation types:

    ```
    CONCAT_LSTM
    CONCAT_GRU
    CONCAT_BIDIR_OUTPUT
    ```

- `zdnn_tensor_desc *tfrmd_desc`

  - output `zdnn_tensor_desc` struct

#### zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_LAYOUT` - `pre_tfrmd_desc->layout` is not recognized or is not
  supported for concatenated tensors.
- `ZDNN_INVALID_CONCAT_TYPE` - `concat_type` is not recognized.

---

### zdnn_init_ztensor

#### Description

Initialize a `zdnn_ztensor` struct using the pre-transformed and transformed
tensor shape information

#### Format

```
void zdnn_init_ztensor(zdnn_tensor_desc *pre_tfrmd_desc,
                       zdnn_tensor_desc *tfrmd_desc, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_tensor_desc *tfrmd_desc`

  - input tensor descriptor with transformed shape information

- `zdnn_ztensor *output`

  - The `zdnn_ztensor` struct being initialized.

#### Returns

- None

---

### zdnn_init_ztensor_with_malloc

#### Description

Same functionality as `zdnn_init_ztensor`, and computes the size required for
the tensor in the zDNN transformed format and allocates the storage for it. Sets
`buffer` and `buffer_size` fields within `output`.

#### Format

```
zdnn_status zdnn_init_ztensor_with_malloc(zdnn_tensor_desc *pre_tfrmd_desc,
                                          zdnn_tensor_desc *tfrmd_desc,
                                          zdnn_ztensor *output);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_tensor_desc *tfrmd_desc`

  - input tensor descriptor with transformed shape information

- `zdnn_ztensor *output`

  - The `zdnn_ztensor` struct being initialized.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_FORMAT` - `tfrmd_desc->format` is not recognized.
- `ZDNN_INVALID_TYPE` - `tfrmd_desc->type` is not recognized or is a
  pre_tfrmd_desc type.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `tfrmd_desc->dim*` dimensions is 0.
  - One of `tfrmd_desc->dim*` dimensions is greater than
    `zdnn_get_nnpa_max_dim_idx_size`.
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of tfrmd_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_ALLOCATION_FAILURE` - Unable to allocate required memory on a 4K
  boundary.

---

### zdnn_reset_ztensor

#### Description

Reset a `zdnn_ztensor` struct for reuse.

_Note this operation does not set or reset the `buffer` and `buffer_size` fields
nor free the transformed area storage._

#### Format

```
void zdnn_reset_ztensor(zdnn_ztensor *ztensor);
```

#### Parameters

- `zdnn_ztensor *output`

  - The `zdnn_ztensor` struct being reset.

#### Returns

- None

---

### zdnn_allochelper_ztensor

#### Description

Calculate the size required for the tensor in the zDNN transformed format and
allocate the needed storage, satisfying alignment requirements. Sets `buffer`
and `buffer_size` fields within `ztensor`.

_Note that the calling application assumes ownership of this storage and is
responsible for freeing it._

#### Format

```
zdnn_status zdnn_allochelper_ztensor(zdnn_ztensor *ztensor);
```

#### Parameters

- `zdnn_ztensor *ztensor`

  - A `zdnn_ztensor` struct that contains the transformed shape information in
    the `transformed_desc` field.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_FORMAT` - `ztensor->transformed_desc->format` is not recognized.
- `ZDNN_INVALID_TYPE` - `ztensor->transformed_desc->type` is not recognized or
  is a pre_transformed_desc type.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `ztensor->transformed_desc->dim*` dimensions is 0.
  - One of `ztensor->transformed_desc->dim*` dimensions is greater than
    `zdnn_get_nnpa_max_dim_idx_size`.
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of transformed_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_ALLOCATION_FAILURE` - Unable to allocate required memory on a 4K
  boundary.

---

### zdnn_free_ztensor_buffer

#### Description

Given an input zdnn_ztensor, zdnn_free_ztensor_buffer will free the transformed
area storage associated with it.

_Note that the routine does not free the storage allocated for the zdnn_ztensor
struct itself._

#### Format

```
zdnn_status zdnn_free_ztensor_buffer(const zdnn_ztensor *ztensor);
```

#### Parameters

- `zdnn_ztensor *tensor`

  - A `zdnn_ztensor` struct with field buffer pointing to storage allocated.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_BUFFER` - `tensor->buffer` is `NULL`

---

### zdnn_get_status_message

#### Description

Retrieve status message of the status code

#### Format

```
const char *zdnn_get_status_message(zdnn_status status);
```

#### Parameters

- `zdnn_status status`

  - Status code

#### Returns

Pointer to the description string or "(Status string is not defined.)" if
`status` is not defined.

---

### zdnn_reshape_ztensor

#### Description

Reshape and copy buffer content from source zTensor's buffer to destination
zTensor's in accordance to destination zTensor's shape.

The following conditions must be satisfied:

- Both tensor's transformed_desc must be fully initialized
- `dest->buffer` must be pre-allocated
- `src` must be transformed
- `dest` must be not already transformed
- Both `transformed_desc->layout` must be the same and either NHWC or HWCK
- Both zTensors must contain equal number of elements

#### Format

```
zdnn_status zdnn_reshape_ztensor(const zdnn_ztensor *src, zdnn_ztensor *dest);
```

#### Parameters

- `src`

  - Source zTensor to copy from

- `dest`

  - Destination zTensor to copy to

#### Programming Notes

- If `src` and `dest` have the same `transformed_desc->dim1` dimension size, the
  transformed data is directly copied to the destination without
  untransformation.

- If `src` and `dest` have different `transformed_desc->dim1` dimension sizes,
  reshaping will internally un-transform the source and then re-transform the
  values into the destination.

#### Returns

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - `src`'s and `dest`'s `transformed_desc->dim*` total to different numbers of
    elements.
  - One of `dest->transformed_desc->dim*` dimensions is 0.
  - One of `dest->transformed_desc->dim*` dimensions is greater than
    `zdnn_get_nnpa_max_dim_idx_size`.
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of `dest->transformed_desc-dim*` elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_INVALID_LAYOUT` - (if any of the following are true)
  - `src`'s and `dest`'s `transformed_desc->layout` are not the same.
  - `transformed_desc->layout` is not `ZDNN_NHWC` nor `ZDNN_HWCK`.
  - `src->pre_transformed_desc->layout` is not recognized or is not a valid
    pre_transformed_desc layout.
  - `dest->pre_transformed_desc->layout` is not recognized or is not a valid
    pre_transformed_desc layout.
- `ZDNN_INVALID_STATE` - (if any of the following are true)
  - `src` is not already transformed.
  - `dest` is already transformed.
- `ZDNN_INVALID_FORMAT` - `src->transformed_desc->format` is not
  `ZDNN_FORMAT_4DFEATURE`.
- `ZDNN_INVALID_TYPE` (if any of the following are true)
  - `src->pre_transformed_desc->type` is not recognized or is a transformed_desc
    type.
  - `dest->pre_transformed_desc->type` is not recognized or is a
    transformed_desc type.
  - `dest->transformed_desc->type` is not recognized or is a
    pre_transformed_desc type.
- `ZDNN_INVALID_BUFFER` (if any of the following are true)
  - `src->buffer` is `NULL`.
  - `src->buffer` is not on a 4K boundary.
  - `dest->buffer` is `NULL`.
  - `dest->buffer` is not on a 4K boundary.
  - `dest->buffer_size` is too small to hold transformed values.
- `ZDNN_CONVERT_FAILURE` - Values failed to un-transform or transform.

---

### zdnn_is_version_runnable

#### Description

Check if application built for zDNN version `ver_num` can be run on the current
AIU hardware with the installed zDNN library

#### Format

```
bool zdnn_is_version_runnable(uint32_t ver_num);
```

#### Parameters

- `ver_num`

  - zDNN version number from the application in 0x00[major][minor][patch] form.
    Typically this is ZDNN_VERNUM used to compile the application

#### Returns

- true/false

---

### zdnn_get_max_runnable_version

#### Description

Returns the maximum zDNN version number that the current hardware and installed
zDNN library can run together. The returned value means the current runtime
environment fully supports zDNN APIs set of that `major`.`minor` version and
below.

#### Format

```
uint32_t zdnn_get_max_runnable_version();
```

#### Parameters

- None

#### Returns

- A 32-bit zDNN version number in 0x00[major][minor]FF form.

---

## Data Transformation

[Back to Table of Contents](#TOC)

- [Transform to zTensor](#zdnn_transform_ztensor)
- [Transform to Original](#zdnn_transform_origtensor)

---

zAIU requires the tensor data to be arranged in a format that enhances the
performance characteristics of the operations. In this documentation, it is
referred to as "transformed format". In addition, data conversions are necessary
from the common formats (FP32, FP16, BFLOAT) to the internal format (DLFLOAT16)
supported by the AIU. Two functions are provided:

- '`zdnn_transform_ztensor`

  - zdnn_transform_ztensor will transform the input tensor and convert the input
    data to the format required by the AIU. The resulting transformed ztensor
    can be reused as many times as necessary.

  - See [zdnn_transform_ztensor](#zdnn_transform_ztensor) for details on
    transforming an input tensor to the internal format.

- `zdnn_transform_origtensor`

  - zdnn_transform_origtensor transforms a ztensor (usually output from an
    operation or network) to the format and data types that are usable by the
    application.

  - See [zdnn_transform_origtensor](#zdnn_transform_origtensor) for details on
    transforming an input tensor to the internal format.

---

### zdnn_transform_ztensor

#### Description

Converts the input tensor to the supported transformed format for execution by
zdnn operations. If transformation is successful the `is_transformed` field
within `ztensor` will be set to `true` otherwise it is set to `false`.
Transformation will fail if `is_transformed` was already `true`.

_Note that the tensor layout in memory, once in transformed format, is dependent
on the content of the input tensor's descriptors (`zdnn_tensor_desc` fields).
Once converted, a `zdnn_ztensor` should only be manipulated by zDNN API
functions._

#### Format

```
zdnn_status zdnn_transform_ztensor(zdnn_ztensor *ztensor, ...);
```

#### Parameters

- `zdnn_ztensor *tensor`

  - The input `zdnn_ztensor` struct. `pre_transformed_desc` and
    `transformed_desc` must be set, `is_transformed` must be `false`. A
    4k-aligned tensor storage must be pre-allocated by the caller (directly or
    by calling the zDNN allocation helper function) and field `buffer` must
    point to the storage.

- `... (additional arguments)`

  - Variadic: list of pointers for input data to be transformed:
    - Non-concatenated: 1 data pointer
    - LSTM concatenated: 4 data pointers, one for each input gate in Forget,
      Input, Cell, Output (FICO) order
    - GRU concatenated: 3 data pointers, one for each input gate in (Z)update,
      Reset, Hidden, (ZRH) gate order

#### Programming Notes

- This function clears the pre-thread floating-point exception flags at entry,
  and may set `FE_UNDERFLOW` / `FE_INVALID` / `FE_INEXACT` / `FE_OVERFLOW` when
  it encounters errors during data conversion.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_FORMAT` - `zdnn_ztensor->transformed_desc->format` is not
  recognized.
- `ZDNN_INVALID_LAYOUT` - (if any of the following are true)
  - `zdnn_ztensor->pre_transformed_desc->layout` is not recognized or is not a
    valid pre_transformed_desc layout.
  - `zdnn_ztensor->transformed_desc->layout` is not recognized or is not a valid
    transformed_desc layout.
- `ZDNN_INVALID_TYPE` - (if any of the following are true)
  - `zdnn_ztensor->pre_transformed_desc->type` is not recognized or is a
    transformed_desc type.
  - `zdnn_ztensor->transformed_desc->type` is not recognized or is a
    pre_transformed_desc type.
- `ZDNN_INVALID_BUFFER` (if any of the following are true)
  - `buffer` is `NULL`.
  - `buffer` is not on a 4K boundary.
  - `buffer_size` is too small to hold transformed values.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `zdnn_ztensor->transformed_desc->dim*` dimensions is 0.
  - One of `zdnn_ztensor->transformed_desc->dim*` dimensions is greater than
    `zdnn_get_nnpa_max_dim_idx_size`.
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of transformed_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_INVALID_STATE` - Tensor is already transformed.
- `ZDNN_CONVERT_FAILURE` - Values failed to transform.

---

### zdnn_transform_origtensor

#### Description

Converts the input tensor from the zDNN transformed format back to a standard
non-transformed layout. The `is_transformed` field within `ztensor` must be
`true`.

Only feature tensors that have been converted to transformed format are
supported. Kernel tensors and concatenated tensors are not supported.

#### Format

```
zdnn_status zdnn_transform_origtensor(const zdnn_ztensor *ztensor, void *out_buf);
```

#### Parameters

- `zdnn_ztensor *ztensor`

  - The input `zdnn_ztensor` struct. `pre_transformed_desc`, `transformed_desc`
    and `buffer` must be set, `is_transformed` must be `true`.

- `void *out_buf`

  - The buffer for storing the standard non-transformed tensor data. Must be
    pre-allocated by the caller.

#### Programming Notes

- This function clears the pre-thread floating-point exception flags at entry,
  and may set `FE_UNDERFLOW` / `FE_INVALID` / `FE_INEXACT` / `FE_OVERFLOW` when
  it encounters errors during data conversion.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_FORMAT` - `ztensor->transformed_desc->format` is not
  `ZDNN_FORMAT_4DFEATURE`.
- `ZDNN_INVALID_LAYOUT` - (if any of the following are true)
  - `zdnn_ztensor->pre_transformed_desc->layout` is not recognized or is not a
    valid pre_transformed_desc layout.
  - `zdnn_ztensor->transformed_desc->layout` is not recognized or is not a valid
    transformed_desc layout required by this function.
- `ZDNN_INVALID_TYPE`
  - `ztensor->pre_transformed_desc->type` is not recognized or is a
    transformed_desc type.
  - `ztensor->transformed_desc->type` is not recognized or is a
    pre_transformed_desc type.
- `ZDNN_INVALID_BUFFER` (if any of the following are true)
  - `ztensor->buffer` is `NULL`.
  - `ztensor->buffer` is not on a 4K boundary.
- `ZDNN_INVALID_STATE` - `ztensor` is not transformed.
- `ZDNN_CONVERT_FAILURE` - Values failed to un-transform.

---

## Operations

See [Table of Contents](#TOC) for operations list

---

## Element-wise Operations <a id="elwise-ops"></a>

[Back to Table of Contents](#TOC)

- [Addition](#zdnn_add)
- [Subtraction](#zdnn_sub)
- [Multiplication](#zdnn_mul)
- [Division](#zdnn_div)
- [Minimum](#zdnn_min)
- [Maximum](#zdnn_max)
- [Natural Logarithm](#zdnn_log)
- [Exponential](#zdnn_exp)

---

### zdnn_add

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given two input tensors in zDNN transformed format, performs element-wise
addition and stores the result into the provided output zDNN tensor.

_Note that for zDNN use, broadcasting of the input tensor(s) must be performed
by the caller. As such, the input tensors must be of the same shape._

#### Format

```
zdnn_status zdnn_add(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with addends to add to `input_b` tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with addends to add to `input_a` tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor to hold the result of the addition
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Addition]

[tensorflow addition]: https://www.tensorflow.org/api_docs/python/tf/math/add

[ONNX Addition]

[onnx addition]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add

---

### zdnn_sub

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given two input tensors in zDNN transformed format, performs element-wise
subtraction and stores the result into the provided output zDNN tensor.

_Note that for zDNN use, broadcasting of the input tensor(s) must be performed
by the caller. As such, the input tensors must be of the same shape._

#### Format

```
zdnn_status zdnn_sub(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with minuends that will be subtracted by `input_b` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with subtrahends to subtract from `input_a` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor to hold the result of the subtraction
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Subtraction]

[tensorflow subtraction]:
  https://www.tensorflow.org/api_docs/python/tf/math/subtract

[ONNX Subtraction]

[onnx subtraction]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#sub

---

### zdnn_mul

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given two input tensors in zDNN transformed format, performs element-wise
multiplication and stores the result into the provided output zDNN tensor.

_Note that for zDNN use, broadcasting of the input tensor(s) must be performed
by the caller. As such, the input tensors must be of the same shape._

#### Format

```
zdnn_status zdnn_mul(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with multiplicands that will be multiplied by `input_b` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with multipliers for `input_a` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor to hold the result of the multiplication.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Multiplication]

[tensorflow multiplication]:
  https://www.tensorflow.org/api_docs/python/tf/math/multiply

[ONNX Multiplication]

[onnx multiplication]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul

---

### zdnn_div

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given two input tensors in zDNN transformed format, performs element-wise
division and stores the result into the provided output zDNN tensor.

_Note that for zDNN use, broadcasting of the input tensor(s) must be performed
by the caller. As such, the input tensors must be of the same shape._

#### Format

```
zdnn_status zdnn_div(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with dividends that will be divided by `input_b` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with divisors for `input_a` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor to hold the result of the division.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Division]

[tensorflow division]: https://www.tensorflow.org/api_docs/python/tf/math/divide

[ONNX Division]

[onnx division]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div

---

### zdnn_min

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given two input tensors in zDNN transformed format, computes the element-wise
minimum and stores the result into the provided output zDNN tensor.

_Note that for zDNN use, broadcasting of the input tensor(s) must be performed
by the caller. As such, the input tensors must be of the same shape._

#### Format

```
zdnn_status zdnn_min(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with values that will be compared with `input_b` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with values that will be compared with `input_a` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the smaller value from each comparison of the inputs.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Minimum]

[tensorflow minimum]: https://www.tensorflow.org/api_docs/python/tf/math/minimum

[ONNX Minimum]

[onnx minimum]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#min

---

### zdnn_max

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given two input tensors in zDNN transformed format, computes the element-wise
maximum and stores the result into the provided output zDNN tensor.

_Note that for zDNN use, broadcasting of the input tensor(s) must be performed
by the caller. As such, the input tensors must be of the same shape._

#### Format

```
zdnn_status zdnn_max(const zdnn_ztensor *input_a, const zdnn_ztensor *input_b,
                     zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with values that will be compared with `input_b` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with values that will be compared with `input_a` tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the larger value from each comparison of the inputs.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)s

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Maximum]

[tensorflow maximum]: https://www.tensorflow.org/api_docs/python/tf/math/maximum

[ONNX Maximum]

[onnx maximum]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#max

---

### zdnn_log

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the natural logarithm
element-wise and stores the result into the provided output zDNN tensor.

#### Format

```
zdnn_status zdnn_log(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the calculated natural logarithm of each value from
    `input_a`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Natural Logarithm]

[tensorflow natural logarithm]:
  https://www.tensorflow.org/api_docs/python/tf/math/log

[ONNX Natural Logarithm]

[onnx natural logarithm]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log

---

### zdnn_exp

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the exponential
element-wise and stores the result into the provided output zDNN tensor.

#### Format

```
zdnn_status zdnn_exp(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the calculated exponential of each value from `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Exponential]

[tensorflow exponential]: https://www.tensorflow.org/api_docs/python/tf/math/exp

[ONNX Exponential]

[onnx exponential]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp

---

## Activation Operations <a id="act-ops"></a>

[Back to Table of Contents](#TOC)

- [Rectified Linear](#zdnn_relu)
- [Hyperbolic Tangent](#zdnn_tanh)
- [Sigmoid](#zdnn_sigmoid)
- [Softmax](#zdnn_softmax)

---

### zdnn_relu

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format produce an output tensor where
the rectified linear function, y = max(0, x) is applied to the input
element-wise. If an optional clipping_value is provided, clipping is performed
against the intermediate output where z = min(y, clipping_value).

#### Format

```
zdnn_status zdnn_relu(const zdnn_ztensor *input, const void *clipping_value,
                      zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `void *clipping_value`

  - A pointer to an FP32 value, used to clip input tensor's elements.
  - If set to NULL or 0, no clipping will occur.
  - Must not be a negative value.

- `zdnn_ztensor *output`
  - Tensor that holds the rectified linear function result of each value from
    `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_CLIPPING_VALUE`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Rectified Linear]

[tensorflow rectified linear]:
  https://www.tensorflow.org/api_docs/python/tf/nn/relu

[ONNX Rectified Linear]

[onnx rectified linear]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#relu

---

### zdnn_tanh

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, produces an output tensor
where the hyperbolic tangent is applied to the input element-wise.

#### Format

```
zdnn_status zdnn_tanh(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the hyperbolic tangent result of each value from `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Hyperbolic Tangent]

[tensorflow hyperbolic tangent]:
  https://www.tensorflow.org/api_docs/python/tf/math/tanh

[ONNX Hyperbolic Tangent]

[onnx hyperbolic tangent]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh

---

### zdnn_sigmoid

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, produces an output tensor
where the sigmoid function is applied to the input element-wise.

#### Format

```
zdnn_status zdnn_sigmoid(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the sigmoid result of each value from `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Sigmoid]

[tensorflow sigmoid]: https://www.tensorflow.org/api_docs/python/tf/math/sigmoid

[ONNX Sigmoid]

[onnx sigmoid]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid

---

### zdnn_softmax

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the softmax
(normalized exponential) for each vector formed in dimension-1, then if
`act_func` is not `SOFTMAX_ACT_NONE`, the activation function is applied to the
results. Finally stores the results into the provided output zDNN tensor.

_Note: Other parameters, such as axis, are not supported._

#### Format

```
zdnn_status zdnn_softmax(const zdnn_ztensor *input, void *save_area,
                         zdnn_softmax_act act_func, zdnn_ztensor *output);

```

#### Parameters

- `zdnn_ztensor *input`

  - [ZDNN_3DS](#common-layouts) tensor with pre-transformed shape [batch size,
    batch size, vector dimension size] or output from another operation that is
    of the correct shape.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `void *save_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - The preallocate memory must be at least 8K bytes in size, aligned on a 4k
    boundary.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.

- `zdnn_softmax_act act_func`

  - Activation function to apply to the results.
  - `SOFTMAX_ACT_NONE` or `SOFTMAX_ACT_LOG`

- `zdnn_ztensor *output`
  - [ZDNN_3DS](#common-layouts) tensor with the same shape as `input_a` that
    holds the softmax result of each value from `input_a`.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Programming Notes

- If all elements of a dimension 1 vector are the largest magnitude negative
  number possible for the transformed data type, accuracy may be reduced.

- A `ZDNN_3DS` tensor is expected, where the `transformed_desc` dim1 describes
  the vector, and dim2 and dim4 are used to batch multiple vector requests
  together. Dim3 must always be 1. The `zdnn_softmax` operation is performed
  against the vector in dim1 repeating for each dim1 vector in the dim4 and dim2
  dimensions.

- Tensors that cannot be processed as vectors in dim1 or as batches of dim1
  vectors must be coerced or reshaped by the caller.
  - When the entire tensor is to be processed by softmax, it can be coerced by
    simply creating an alternate descriptor prior to zDNN transformation. For
    example:
    - A 4D tensor with `pre_transformed_desc` dimensions 2x2x2x2 and a data
      array of 16 FP32 entries could have an alternate `ZDNN_3DS` layout
      `pre_transformed_desc` using dimensions 1x1x16 and use the same original
      data array prior to `zdnn_transform_ztensor`. After transformation, such a
      tensor would be valid for `zdnn_softmax`.
    - In another example, the 4D 2x2x2x2 tensor could be processed as 2 batches
      of 8 vectors using a `ZDNN_3DS` layout `pre_transformed_desc` with
      dimensions 1x2x8.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_ALLOCATION_FAILURE` - A preallocated `save_area` was not specified and
  internal allocation for the required memory failed.
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - input tensor `input->transformed_desc->dim3` was
    not 1.
  - `ZDNN_FUNC_RC_F001` - Invalid `act_func`

#### Framework Examples

[TensorFlow Softmax]

[tensorflow softmax]: https://www.tensorflow.org/api_docs/python/tf/nn/softmax

[ONNX Softmax]

[onnx softmax]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#Softmax

---

## Normalization Operations <a id="norm-ops"></a>

[Back to Table of Contents](#TOC)

- [Mean Reduce](#zdnn_meanreduce2d)
- [Batch Norm](#zdnn_batchnorm)

---

### zdnn_meanreduce2d

- [Back to Table of Contents](#TOC)
  - [Back to Normalization Operations](#norm-ops)

#### Description

Given an input tensor in zDNN transformed format, produces a downsampled tensor
reducing the middle dimensions to a size of 1 based on the mean of the original
values and stores the result to the provided output zDNN tensor.

#### Format

```
zdnn_status zdnn_meanreduce2d(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [batch_Num, Height, Width, Channel].
  - Height and Width dimension must be less than or equal to 1024.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - The result tensor which will hold the result of the pooling operation in its
    buffer.
  - Shape:
    - `output` dimensions batch_Num and Channel must be the same as the
      respective input dimensions.
    - `output` dimensions Height and Width must be 1.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE` - Shape of input or output tensor is invalid based on
  given kernel and stride parameters
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F001` - `input` tensor has a Height or Width dimension greater
    than allowed for `zdnn_meanreduce2d`.

#### Framework Examples

[TensorFlow Reduce Mean] with `axis` set for the Height and Width axes and
`keepdims` set to True.

[tensorflow reduce mean]:
  https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean

[ONNX Reduce Mean]

[onnx reduce mean]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean

---

### zdnn_batchnorm

- [Back to Table of Contents](#TOC)
  - [Back to Normalization Operations](#norm-ops)

#### Description

Given three input zDNN tensors `input_a`, `input_b`, and `input_c`, computes the
batch-normalized result for each vector formed in dimension-1 as follows:

output = input_b \* input_a + input_c

where `input_b` is a precomputed elementwise divide of scale and variance
tensors, and `input_c` is a precomputed elementwise multiply of (-1) \* mean and
'input_b' + input bias tensors.

#### Format

```
zdnn_status zdnn_batchnorm(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Must be a 4D [ZDNN_NHWC](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Must be a 1D [ZDNN_1D](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_c`

  - Must be a 1D [ZDNN_1D](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - A zdnn_ztensor of the same size as `input_a` representing the computed value
    of the above formula
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow Batchnorm]

[tensorflow batchnorm]:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

[ONNX Batchnorm]

[onnx batchnorm]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization

---

### zdnn_matmul_op

[Back to Table of Contents](#TOC)

#### Description

Given three input zDNN tensors `input_a`, `input_b`, and `input_c`, determine
the matrix multiplication of `input_a` \* `input_b` then perform one of the
following operations, using `input_c` against the dot product, storing the
result into the specified `output` zDNN tensor:

- Addition
- Compare - If dot product is greater than element.
- Compare - If dot product is greater or equal to element.
- Compare - If dot product is equal to element.
- Compare - If dot product is not equal to element.
- Compare - If dot product is less than or equal to element.
- Compare - If dot product is less than element.

For an operation type of addition, `input_c` is added to the intermediate dot
product. For operation types of comparison, the intermediate dot product is
compared to `input_c` and if the comparison is true, the result is set to a
value of 1; otherwise it is set to a value of 0.

The outermost dimension can optionally indicate that the inputs are stacks of
matrices. The results for each matrix stack is independent of other stacks but
all stacks are calculated in a single call.

#### Format

```
zdnn_status zdnn_matmul_op(const zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c,
                           zdnn_matmul_ops op_type, zdnn_ztensor *output);
```

#### Input / Output matmul tensor requirements <a id="matmul-io-table"></a>

- See table in this section for `pre_transformed_desc` and shape requirements
  for each tensor.
- All tensors must either be stacked or unstacked.
- Must follow [general tensor requirements](#gen-zten-reqs)

| type      | input_a              | input_b              | input_c           | result               |
| --------- | -------------------- | -------------------- | ----------------- | -------------------- |
| unstacked | `ZDNN_2D` (m, n)     | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_2D` (m, p)     |
| stacked   | `ZDNN_3DS` (s, m, n) | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |

#### Parameters

- `zdnn_ztensor *input_a`

  - Input tensor with the first matrix for multiplication
  - pre_transformed shape and layout must match
    [matmul tensor requirements](#matmul-io-table)

- `zdnn_ztensor *input_b`

  - Input tensor with the second matrix for multiplication
  - pre_transformed shape and layout must match
    [matmul tensor requirements](#matmul-io-table)

- `zdnn_ztensor *input_c`

  - Input tensor that will have the requested operation performed against the
    intermediate dot product of `input_a` and `input_b`.
  - pre_transformed shape and layout must match
    [matmul tensor requirements](#matmul-io-table)

- `zdnn_matmul_ops op_type`

  - Operation to perform on dot product.
    - `MATMUL_OP_ADDITION`
    - `MATMUL_OP_GREATER`
    - `MATMUL_OP_GREATER_EQUAL`
    - `MATMUL_OP_EQUAL`
    - `MATMUL_OP_NOT_EQUAL`
    - `MATMUL_OP_LESSER_EQUAL`
    - `MATMUL_OP_LESSER`

- `zdnn_ztensor *output`
  - The output tensor which will hold the result of the operation in its buffer.
  - pre_transformed shape and layout must match
    [matmul tensor requirements](#matmul-io-table)

#### Programming Notes

- Care must be exercised when comparing values for equality or inequality since
  the order of operations and rounding may produce, what appear to be, slightly
  different values when they are essentially the same value.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Invalid `op_type`.

#### Framework Examples

[TensorFlow MatMul]

[tensorflow matmul]:
  https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mat-mul

[ONNX MatMul]

[onnx matmul]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul

---

### zdnn_matmul_bcast_op

[Back to Table of Contents](#TOC)

#### Description

Given three input zDNN tensors `input_a`, `input_b`, and `input_c`, determine
the matrix multiplication of `input_a` \* `input_b`, then perform one of the
following operations, using `input_c` against the dot product, storing the
result into the specified `output` zDNN tensor:

- Addition

The outermost dimension for `input_a` can optionally indicate that the input is
a stack of matrices. Each stack of `input_a` is then multiplied by the same
`input_b` matrix and `input_c` which are broadcast over each stack of `input_a`.
Results for each stack are returned in the corresponding stack index of
`output`.

#### Format

```
zdnn_status zdnn_matmul_bcast_op(const zdnn_ztensor *input_a,
                                 const zdnn_ztensor *input_b,
                                 const zdnn_ztensor *input_c,
                                 zdnn_matmul_bcast_ops op_type, zdnn_ztensor *output);
```

#### Input / Output matmul broadcast tensor requirements <a id="matmul-bcast-io-table"></a>

- See table in this section for `pre_transformed_desc` and shape requirements
  for each tensor.
- Must follow [general tensor requirements](#gen-zten-reqs)

| input_a              | input_b          | input_c       | result               |
| -------------------- | ---------------- | ------------- | -------------------- |
| `ZDNN_3DS` (s, m, n) | `ZDNN_2D` (n, p) | `ZDNN_1D` (p) | `ZDNN_3DS` (s, m, p) |

#### Parameters

- `zdnn_ztensor *input_a`

  - Input tensor with the first matrix for multiplication.
  - pre_transformed shape and layout must match
    [matmul broadcast tensor requirements](#matmul-bcast-io-table)

- `zdnn_ztensor *input_b`

  - Input tensor with the second matrix for multiplication.
  - The same single `input_b` matrix is broadcast and used as the multiplier for
    each stack dimension of `input_a`
  - pre_transformed shape and layout must match
    [matmul broadcast tensor requirements](#matmul-bcast-io-table)

- `zdnn_ztensor *input_c`

  - Input tensor that will have the requested operation performed against the
    intermediate dot product for each "m" dimension in `output`.
  - pre_transformed shape and layout must match
    [matmul broadcast tensor requirements](#matmul-bcast-io-table)

- `zdnn_matmul_bcast_ops op_type`

  - Operation to perform on dot product.

    - `MATMUL_BCAST_OP_ADDITION`

- `zdnn_ztensor *output`
  - The output tensor which will hold the result of the operation in its buffer.
  - pre_transformed shape and layout must match
    [matmul broadcast tensor requirements](#matmul-bcast-io-table)

#### Programming Notes

- `zdnn_matmul_bcast_ops` only supports `MATMUL_BCAST_OP_ADDITION` op_type, any
  other op_types will be ignored and may not operate compatibly in the future.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow MatMul]

[tensorflow matmul]:
  https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mat-mul

[ONNX MatMul]

[onnx matmul]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul

---

### zdnn_lstm

[Back to Table of Contents](#TOC)

#### Description

Implements Long-Short Term Memory layer (LSTM - Hochreiter 1997).

The following formula is computed for the input tensor input(t) for all time
steps:

(Default: f=Sigmoid, g=Tanh, h=Tanh):

```
- it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)

- ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)

- ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

- Ct = ft (.) Ct-1 + it (.) ct

- ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)

- Ht = ot (.) h(Ct)
```

#### Format

```
zdnn_status zdnn_lstm(const zdnn_ztensor *input, const zdnn_ztensor *h0,
                      const zdnn_ztensor *c0, const zdnn_ztensor *weights,
                      const zdnn_ztensor *biases,
                      const zdnn_ztensor *hidden_weights,
                      const zdnn_ztensor *hidden_biases,
                      lstm_gru_direction direction, void *work_area,
                      zdnn_ztensor *hn_output, zdnn_ztensor *cf_output);
```

Also see an [example](#example-of-an-application-calling-the-zdnn_lstm-api) in
the usage example section.

#### LSTM Input / Output requirements

- `hidden_state_size` dimensions: <a id="lstm-hid_sz"></a>
  - Any hidden_state_size dimension must be less than or equal to 8192 elements.

#### Parameters

- `zdnn_ztensor *input`

  - Input must be a tensor with the shape [timestep, batch, feature] prior to
    transformation with the `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *h0`

  - Tensor containing the initial hidden state with shape [direction, batch,
    hidden_state_size] prior to transformation with the `zdnn_transform_ztensor`
    API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

- `zdnn_ztensor *c0`

  - Tensor containing the initial cell state with shape [direction, batch,
    hidden_state_size] prior to transformation with the `zdnn_transform_ztensor`
    API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

- `zdnn_ztensor *weights`

  - Tensor containing the concatenated input connection weights in Forget,
    Input, Cell, Output (FICO) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (direction, features, hidden_state_size) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects a `CONCAT_LSTM` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

- `zdnn_ztensor *biases`

  - Tensor containing the concatenated input connection bias in Forget, Input,
    Cell, Output (FICO) order.
  - Prior to transformation, expects each gate needs to be shape (direction,
    hidden_state_size).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects a `CONCAT_LSTM` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

- `zdnn_ztensor *hidden_weights`

  - Tensor containing the concatenated hidden connection weights in Forget,
    Input, Cell, Output (FICO) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (direction, hidden_state_size, hidden_state_size) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects a `CONCAT_LSTM` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

- `zdnn_ztensor *hidden_biases`

  - Tensor containing the concatenated hidden connection bias in Forget, Input,
    Cell, Output (FICO) order.
  - Prior to transformation, expects each gate needs to be shape (direction,
    hidden_state_size).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects a `CONCAT_LSTM` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

- `lstm_gru_direction direction`

  - Direction indicator of `lstm_gru_direction direction` type. Valid values:
    - `FWD` (forward)
    - `BWD` (backward)
    - `BIDIR` (bi-directional).
  - For input shapes, the direction dimension should be:
    - `1` for unidirectional calls such as FWD or BWD
    - `2` for bidirectional calls such that:
      - direction == 1 contains FWD values.
      - direction == 2 contains BWD values.

- `void *work_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.
  - Amount of required storage can be determined given the LSTM timestep, batch,
    and hidden_state_size values.

    - The sample code below creates a ztensor descriptor that is an equivalent
      size of the required `work_area`. To use this sample code yourself,
      replace the `timestep`, `batch`, and `hidden_state_size` variables with
      your own values.

      ```
        zdnn_tensor_desc desc;
        desc.dim4 = (4 * timestep) + 6;
        desc.dim3 = 1;
        desc.dim2 = batch;
        desc.dim1 = hidden_state_size;
        uint64_t work_area_size = zdnn_getsize_ztensor(&desc);
      ```

  - For bidirectional, twice the amount of contiguous storage is required.
  - The start of the buffer must be 4k aligned.

- `zdnn_ztensor *hn_output`

  - Output results of the hidden states

  - Expects pre_transformed_desc->layout to be `ZDNN_3DS`.

  - Output must be a tensor with either of the following shapes:

    - For output from all timesteps: [timestep, batch, hidden_state_size]
    - For final processed timestep only: [1, batch, hidden_state_size]

  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

  - For bidirectional (`BIDIR`) output:

    - Forward and backward results are concatenated on the innermost dimension.
    - Expects a `CONCAT_BIDIR_OUTPUT` [concatenated tensor](#concat-zten-reqs)
    - Concatenated output is meant for use in subsequent layers. Direct
      untransformation of output is not supported.

  - Note that for `BWD` and the backward component of `BIDIR` directions, the
    output order matches the order of the input, not the processing order. For
    example, the first input timestep is the last to be processed and its result
    is the first timestep of the output.

- `zdnn_ztensor *cf_output`

  - Output results of the cell state for the last processed timestep

  - Expects pre_transformed_desc->layout to be `ZDNN_3DS`.

  - Output must be a tensor with shape [1, batch, hidden_state_size]

  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#lstm-hid_sz)

  - For bidirectional (`BIDIR`):
    - Forward and backward results are concatenated on the innermost dimension.
    - Expects a `CONCAT_BIDIR_OUTPUT` [concatenated tensor](#concat-zten-reqs)
    - Concatenated output is meant for use in subsequent layers. Direct
      untransformation of output is not supported.

#### Summary

|                | pre-transformed layout | pre-transformed shape                                                                                       | create transformed desc via:                                                                                                         |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| input          | `ZDNN_3DS`             | (timestep, batch, feature)                                                                                  | `zdnn_generate_transformed_desc`                                                                                                     |
| h0             | `ZDNN_3DS`             | (direction, batch, hidden_state_size)                                                                       | `zdnn_generate_transformed_desc`                                                                                                     |
| c0             | `ZDNN_3DS`             | (direction, batch, hidden_state_size)                                                                       | `zdnn_generate_transformed_desc`                                                                                                     |
| weights        | `ZDNN_3DS`             | (direction, features, hidden_state_size)                                                                    | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_LSTM`                                                                     |
| bias           | `ZDNN_2DS`             | (direction, hidden_state_size)                                                                              | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_LSTM`                                                                     |
| hidden_weights | `ZDNN_3DS`             | (direction, hidden_state_size, hidden_state_size)                                                           | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_LSTM`                                                                     |
| hidden_biases  | `ZDNN_2DS`             | (direction, hidden_state_size)                                                                              | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_LSTM`                                                                     |
|                |                        |                                                                                                             |                                                                                                                                      |
| hn_output      | `ZDNN_3DS`             | **all timesteps**: (timestep, batch, hidden_state_size)<br>**last timestep**: (1, batch, hidden_state_size) | **FWD/BWD**: `zdnn_generate_transformed_desc`<br>**BIDIR**: `zdnn_generate_transformed_desc_concatenated` with `CONCAT_BIDIR_OUTPUT` |
| cf_output      | `ZDNN_3DS`             | (1, batch, hidden_state_size)                                                                               | **FWD/BWD**: `zdnn_generate_transformed_desc`<br>**BIDIR**: `zdnn_generate_transformed_desc_concatenated` with `CONCAT_BIDIR_OUTPUT` |

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_SHAPE` - (if any of the following are not true)
  - `hn_output` timesteps dimension must be 1 or the same size as `input`
    timestep dimension.
  - All tensors with a direction dimension have the same direction dimension
    size.
  - `input` timestep dimension must be greater than or equal to 1.
  - Other general shape violations (exceeds MDIS, etc.)
- `ZDNN_INVALID_DIRECTION` - `direction` parameter was not a recognized
  `lstm_gru_direction`.
- `ZDNN_ALLOCATION_FAILURE` - A preallocated `work_area` was not specified and
  internal allocation for the required memory failed.
- [hardware statuses](#hw-statuses)

#### Framework Examples

[TensorFlow LSTM]

[tensorflow lstm]:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell

[ONNX LSTM]

[onnx lstm]: https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM

---

### zdnn_gru

[Back to Table of Contents](#TOC)

#### Description

Implements Gated Recurrent Unit (Kyunghyun Cho 2014). Supports only reset after
linear.

The following formula is computed for the input tensor input(t) for all time
steps:

(Default: f=Sigmoid, g=Tanh):

- zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

- rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

- ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)

- Ht = (1 - zt) (.) ht + zt (.) Ht-1

#### Format

```
zdnn_status zdnn_gru(const zdnn_ztensor *input, const zdnn_ztensor *h0,
                     const zdnn_ztensor *weights, const zdnn_ztensor *biases,
                     const zdnn_ztensor *hidden_weights,
                     const zdnn_ztensor *hidden_biases,
                     lstm_gru_direction direction, void *work_area,
                     zdnn_ztensor *hn_output);
```

Also see an [example](#example-of-an-application-calling-the-zdnn_gru-api) in
the usage example section.

#### GRU Input / Output requirements

- `hidden_state_size` dimensions: <a id="gru-hid_sz"></a>
  - Any hidden_state_size dimension must be less than or equal to 10880
    elements.

#### Parameters

- `zdnn_ztensor *input`

  - Input must be a tensor with the shape [timestep, batch, feature] prior to
    transformation with the `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

- `zdnn_ztensor *h0`

  - Tensor containing the initial hidden state with shape [direction, batch,
    hidden_state_size] prior to transformation with the `zdnn_transform_ztensor`
    API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

- `zdnn_ztensor *weights`

  - Tensor containing the concatenated input connection weights in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (direction, features, hidden_state_size) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects a `CONCAT_GRU` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

- `zdnn_ztensor *biases`

  - Tensor containing the concatenated input connection bias in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, expects each gate needs to be shape (direction,
    hidden_state_size).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects a `CONCAT_GRU` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

- `zdnn_ztensor *hidden_weights`

  - Tensor containing the concatenated hidden connection weights in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (direction, hidden_state_size, hidden_state_size) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects a `CONCAT_GRU` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

- `zdnn_ztensor *hidden_biases`

  - Tensor containing the concatenated hidden connection bias in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, expects each gate needs to be shape (direction,
    hidden_state_size).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects a `CONCAT_GRU` [concatenated tensor](#concat-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

- `lstm_gru_direction direction`

  - Direction indicator of `lstm_gru_direction direction` type. Valid values:
    - `FWD` (forward)
    - `BWD` (backward)
    - `BIDIR` (bi-directional).
  - For input shapes, the direction dimension should be:
    - `1` for unidirectional calls such as FWD or BWD
    - `2` for bidirectional calls such that:
      - direction == 1 contains FWD values.
      - direction == 2 contains BWD values.

- `void *work_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.
  - Amount of required storage can be determined given the GRU timestep, batch,
    and hidden_state_size values.

    - The sample code below creates a ztensor descriptor that is an equivalent
      size of the required `work_area`. To use this sample code yourself,
      replace the `timestep`, `batch`, and `hidden_state_size` variables with
      your own values.

      ```
        zdnn_tensor_desc desc;
        desc.dim4 = (3 * timestep) + 5;
        desc.dim3 = 1;
        desc.dim2 = batch;
        desc.dim1 = hidden_state_size;
        uint64_t work_area_size = zdnn_getsize_ztensor(&desc);
      ```

  - For bidirectional, twice the amount of contiguous storage is required.
  - The start of the buffer must be 4k aligned.

- `zdnn_ztensor *hn_output`

  - Output results of the hidden states

  - Expects pre_transformed_desc->layout to be `ZDNN_3DS`.

  - Output must be a tensor with either of the following shapes:

    - For output from all timesteps: [timestep, batch, hidden_state_size]
    - For final processed timestep only: [1, batch, hidden_state_size]

  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [hidden_state_size requirements](#gru-hid_sz)

  - For bidirectional (`BIDIR`):

    - Forward and backward results are concatenated on the innermost dimension.
    - Expects a `CONCAT_BIDIR_OUTPUT` [concatenated tensor](#concat-zten-reqs)
    - Concatenated output is meant for use in subsequent layers. Direct
      untransformation of output is not supported.

  - Note that for `BWD` and the backward component of `BIDIR` directions, the
    output order matches the order of the input, not the processing order. For
    example, the first input timestep is the last to be processed and its result
    is the first timestep of the output.

#### Summary

|                | pre-transformed layout | pre-transformed shape                                                                                       | create transformed desc via:                                                                                                         |
| -------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| input          | `ZDNN_3DS`             | (timestep, batch, feature)                                                                                  | `zdnn_generate_transformed_desc`                                                                                                     |
| h0             | `ZDNN_3DS`             | (direction, batch, hidden_state_size)                                                                       | `zdnn_generate_transformed_desc`                                                                                                     |
| weights        | `ZDNN_3DS`             | (direction, features, hidden_state_size)                                                                    | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_GRU`                                                                      |
| bias           | `ZDNN_2DS`             | (direction, hidden_state_size)                                                                              | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_GRU`                                                                      |
| hidden_weights | `ZDNN_3DS`             | (direction, hidden_state_size, hidden_state_size)                                                           | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_GRU`                                                                      |
| hidden_biases  | `ZDNN_2DS`             | (direction, hidden_state_size)                                                                              | `zdnn_generate_transformed_desc_concatenated` with `CONCAT_GRU`                                                                      |
|                |                        |                                                                                                             |                                                                                                                                      |
| hn_output      | `ZDNN_3DS`             | **all timesteps**: (timestep, batch, hidden_state_size)<br>**last timestep**: (1, batch, hidden_state_size) | **FWD/BWD**: `zdnn_generate_transformed_desc`<br>**BIDIR**: `zdnn_generate_transformed_desc_concatenated` with `CONCAT_BIDIR_OUTPUT` |

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_SHAPE` - (if any of the following are not true)
  - `hn_output` timesteps dimension must be 1 or the same size as `input`
    timestep dimension.
  - All tensors with a direction dimension have the same direction dimension
    size.
  - `input` timestep dimension must be greater than or equal to 1.
  - Other general shape violations (exceeds MDIS, etc.)
- `ZDNN_INVALID_DIRECTION` - `direction` parameter was not a recognized
  `lstm_gru_direction`.
- `ZDNN_ALLOCATION_FAILURE` - A preallocated `work_area` was not specified and
  internal allocation for the required memory failed.
- [hardware statuses](#hw-statuses)

---

### zdnn_avgpool2d

[Back to Table of Contents](#TOC)

#### Description

Given an input tensor in zDNN transformed format, padding type, kernel size and
kernel stride, produces a downsampled tensor reducing the middle dimensions
based on the mean values within the kernel window at each step and stores the
results into the provided output zDNN tensor.

#### Format

```
zdnn_status zdnn_avgpool2d(const zdnn_ztensor *input,
                           zdnn_pool_padding padding_type,
                           uint32_t kernel_height, uint32_t kernel_width,
                           uint32_t stride_height, uint32_t stride_width,
                           zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with original values to be downsampled in the output tensor.
  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [batch_Num, Height, Width, Channel].
  - See [Parameter Restrictions](#avgpool2d-parm-restrictions) below for
    information on the expected shape of the input tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `padding_type`

  - The type of padding to use for the pooling operations.
  - Valid values: are `SAME_PADDING` or `VALID_PADDING`.
  - See [Parameter Restrictions](#avgpool2d-parm-restrictions) below for
    information on the expected value of padding_type.
  - For information on "same" vs "valid" padding see:
    <https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow>.

- `kernel_height`

  - Size of the kernel window that passes over the input's height dimension.
  - See [Parameter Restrictions](#avgpool2d-parm-restrictions) below for
    information on the expected value of kerneL_height.

- `kernel_width`

  - Size of the kernel window that passes over the input's width dimension.
  - See [Parameter Restrictions](#avgpool2d-parm-restrictions) below for
    information on the expected value of kerneL_width.

- `stride_height`

  - Number of positions the kernel moves over input's height dimension at each
    step.
  - If `stride_height` is 0 then `stride_width` must also be 0.
  - If strides are greater than 0 then `stride_height` must be less than or
    equal to 30.

- `stride_width`

  - Number of positions the kernel moves over the input's width dimension at
    each step.
  - If `stride_height` is 0 then `stride_width` must also be 0.
  - If strides are greater than 0 then `stride_width` must be less than or equal
    to 30.

- `zdnn_ztensor *output`
  - The result tensor which will hold the result of the pooling operation its
    buffer.
  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [batch_Num, Height, Width, Channel].
  - See [Parameter Restrictions](#avgpool2d-parm-restrictions) below for
    information on the expected shape of the output tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### AvgPool2D Parameter Restrictions <a id="avgpool2d-parm-restrictions"></a>

Parameter restrictions may vary based on provided strides and padding_type.

- Input tensor batch_Num and Channel dimensions must always match the output
  tensor's respective dimensions.

- If strides are 0:
  - Both input tensor's Height dimension and the kernel_height must match and be
    less than or equal to 1024.
  - Both input tensor's Width dimension and the kernel_width must match and be
    less than or equal to 1024.
  - Output tensor's height and width dimensions must be 1.
  - padding_type must be `VALID_PADDING`.
- If strides are greater than zero:
  - kernel_width and kernel_height must be less than or equal to 64.
  - input tensor's height or weight dimension must not be greater than 1024.
  - If padding_type is `SAME_PADDING`:
    - Output tensor's height dimension must equal
      `ceil((float)input's height / stride_height)`.
    - Output tensor's width dimension must equal
      `ceil((float)input's width / stride_width)`.
  - If padding_type is `VALID_PADDING`:
    - Output tensor's height dimension must equal
      `ceil((float)(input's height - kernel_height + 1) / stride_height)`.
    - Output tensor's width dimension must equal
      `ceil((float)(input's width - kernel_width + 1) / stride_width)`.

#### Programming Notes

- If the magnitude of difference between elements of `input` is large (greater
  than 10), accuracy may be reduced.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
  - Shape of input or output tensor is invalid based on given kernel and stride
    parameters
  - Other general shape violations (exceeds MDIS, etc.)
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_STRIDE_PADDING`
- `ZDNN_INVALID_STRIDES` - One stride was non-zero, but not the other.
- [hardware statuses](#hw-statuses)
  - `ZDNN_EXCEEDS_MDIS` will also occur if any of the following conditions
    occur:
    - stride_height is larger than `zdnn_get_nnpa_max_dim_idx_size`.
    - stride_width is larger than `zdnn_get_nnpa_max_dim_idx_size`.
    - kernel_height is 0 or is larger than `zdnn_get_nnpa_max_dim_idx_size`.
    - kernel_width is 0 or is larger than `zdnn_get_nnpa_max_dim_idx_size`.
  - `ZDNN_FUNC_RC_F000` - Invalid `padding_type`
  - `ZDNN_FUNC_RC_F001` - `stride_height` = 0 and `stride_width` = 0, but a
    kernel parameter is greater than allowed (see `kernel_height` or
    `kernel_width` above)
  - `ZDNN_FUNC_RC_F002` - `stride_height` > 0 and `stride_width` > 0, but a
    kernel parameter is greater than allowed (see `kernel_height` or
    `kernel_width` above)
  - `ZDNN_FUNC_RC_F003` - `stride_height` > 0 and `stride_width` > 0, but a
    stride parameter is greater than allowed (see `stride_height` or
    `stride_width` above)
  - `ZDNN_FUNC_RC_F004` - `stride_height` > 0 and `stride_width` > 0, but either
    input tensor's height or weight dimension is greater than 1024.

#### Framework Examples

[TensorFlow AvgPool]

[tensorflow avgpool]:
  https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/avg-pool

[ONNX AvgPool]

[onnx avgpool]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool

---

### zdnn_maxpool2d

[Back to Table of Contents](#TOC)

#### Description

Given an input tensor in zDNN transformed format, padding type, kernel size and
kernel stride, produces a downsampled tensor reducing the middle dimensions
based on the maximum values within the kernel window at each step and stores the
results into the provided output zDNN tensor.

#### Format

```
zdnn_status zdnn_maxpool2d(const zdnn_ztensor *input,
                           zdnn_pool_padding padding_type,
                           uint32_t kernel_height, uint32_t kernel_width,
                           uint32_t stride_height, uint32_t stride_width,
                           zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with original values to be downsampled in the output tensor.
  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [batch_Num, Height, Width, Channel].
  - See [Parameter Restrictions](#maxpool2d-parm-restrictions) below for
    information on the expected shape of the input tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `padding_type`

  - The type of padding to use for the pooling operations.
  - Valid values: are `SAME_PADDING` or `VALID_PADDING`.
  - See [Parameter Restrictions](#maxpool2d-parm-restrictions) below for
    information on the expected value of padding_type.
  - For information on "same" vs "valid" padding see:
    <https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow>.

- `kernel_height`

  - Size of the kernel window that passes over the input's height dimension.
  - See [Parameter Restrictions](#maxpool2d-parm-restrictions) below for
    information on the expected value of kerneL_height.

- `kernel_width`

  - Size of the kernel window that passes over the input's width dimension.
  - See [Parameter Restrictions](#maxpool2d-parm-restrictions) below for
    information on the expected value of kerneL_width.

- `stride_height`

  - Number of positions the kernel moves over input's height dimension at each
    step.
  - If `stride_height` is 0 then `stride_width` must also be 0.
  - If strides are greater than 0 then `stride_height` must be less than or
    equal to 30.

- `stride_width`

  - Number of positions the kernel moves over the input's width dimension at
    each step.
  - If `stride_height` is 0 then `stride_width` must also be 0.
  - If strides are greater than 0 then `stride_width` must be less than or equal
    to 30.

- `zdnn_ztensor *output`
  - The result tensor which will hold the result of the pooling operation its
    buffer.
  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [batch_Num, Height, Width, Channel].
  - See [Parameter Restrictions](#maxpool2d-parm-restrictions) below for
    information on the expected shape of the output tensor.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### MaxPool2D Parameter Restrictions <a id="maxpool2d-parm-restrictions"></a>

Parameter restrictions may vary based on provided strides and padding_type.

- Input tensor batch_Num and Channel dimensions must always match the output
  tensor's respective dimensions.

- If strides are 0:
  - Both input tensor's Height dimension and the kernel_height must match and be
    less than or equal to 1024.
  - Both input tensor's Width dimension and the kernel_width must match and be
    less than or equal to 1024.
  - Output tensor's height and width dimensions must be 1.
  - padding_type must be `VALID_PADDING`.
- If strides are greater than zero:
  - kernel_width and kernel_height must be less than or equal to 64.
  - input tensor's height or weight dimension must not be greater than 1024.
  - If padding_type is `SAME_PADDING`:
    - Output tensor's height dimension must equal
      `ceil((float)input's height / stride_height)`.
    - Output tensor's width dimension must equal
      `ceil((float)input's width / stride_width)`.
  - If padding_type is `VALID_PADDING`:
    - Output tensor's height dimension must equal
      `ceil((float)(input's height - kernel_height + 1) / stride_height)`.
    - Output tensor's width dimension must equal
      `ceil((float)(input's width - kernel_width + 1) / stride_width)`.

#### Programming Notes

- If the magnitude of difference between elements of `input` is large (greater
  than 10), accuracy may be reduced.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
  - Shape of input or output tensor is invalid based on given kernel and stride
    parameters
  - Other general shape violations (exceeds MDIS, etc.)
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_STRIDE_PADDING`
- `ZDNN_INVALID_STRIDES` - One stride was non-zero, but not the other.
- [hardware statuses](#hw-statuses)
  - `ZDNN_EXCEEDS_MDIS` will also occur if any of the following conditions
    occur:
    - stride_height is larger than `zdnn_get_nnpa_max_dim_idx_size`.
    - stride_width is larger than `zdnn_get_nnpa_max_dim_idx_size`.
    - kernel_height is 0 or is larger than `zdnn_get_nnpa_max_dim_idx_size`.
    - kernel_width is 0 or is larger than `zdnn_get_nnpa_max_dim_idx_size`.
  - `ZDNN_FUNC_RC_F000` - Invalid `padding_type`
  - `ZDNN_FUNC_RC_F001` - `stride_height` = 0 and `stride_width` = 0, but a
    kernel parameter is greater than allowed (see `kernel_height` or
    `kernel_width` above)
  - `ZDNN_FUNC_RC_F002` - `stride_height` > 0 and `stride_width` > 0, but a
    kernel parameter is greater than allowed (see `kernel_height` or
    `kernel_width` above)
  - `ZDNN_FUNC_RC_F003` - `stride_height` > 0 and `stride_width` > 0, but a
    stride parameter is greater than allowed (see `stride_height` or
    `stride_width` above)
  - `ZDNN_FUNC_RC_F004` - `stride_height` > 0 and `stride_width` > 0, but either
    input tensor's height or weight dimension is greater than 1024.

#### Framework Examples

[TensorFlow MaxPool]

[tensorflow maxpool]:
  https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/max-pool

[ONNX MaxPool]

[onnx maxpool]:
  https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool

---

### zdnn_conv2d

[Back to Table of Contents](#TOC)

#### Description

Perform 2D convolution over an input tensor in zDNN transformed format.

First the `input` tensor is convolved with the `kernel` tensor. Then the `bias`
tensor is added to the results. Then if `act_func` is not `CONV2D_ACT_NONE`, the
activation function is applied to the results. Then if `act_func` is set to
`CONV2D_ACT_RELU`, and clipping_value is not `NULL` or `0`, clipping is
performed against the intermediate result where z = min(intermediate_result,
clipping_value). Finally the results are stored into the provided output zDNN
tensor.

#### Format

```
zdnn_status zdnn_conv2d(const zdnn_ztensor *input,
                        const zdnn_ztensor *kernel,
                        const zdnn_ztensor *bias,
                        zdnn_pool_padding padding_type,
                        uint32_t stride_height, uint32_t stride_width,
                        zdnn_conv2d_act act_func,
                        const void *clipping_value, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with original values to be downsampled in the output tensor.
  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [num_batches, height_in, width_in, channels_in].
  - See [Convolution 2D Requirements](#convolution-2d-requirements) for
    requirements.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *kernel`

  - The kernel tensor to convolute with the input tensor.
  - Must be a [ZDNN_HWCK](#common-layouts) tensor with pre_transformed shape
    [kernel_height, kernel_width, channels_in, channels_out].
  - See [Convolution 2D Requirements](#convolution-2d-requirements) for
    requirements.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *bias`

  - The bias tensor to add to the convoluted results.
  - Must be a [ZDNN_1D](#common-layouts) tensor with pre_transformed shape
    [channels_out].
  - See [Convolution 2D Requirements](#convolution-2d-requirements) for
    requirements.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_pool_padding padding_type`

  - The type of padding to use for the pooling operations.
  - Valid values: are `SAME_PADDING` or `VALID_PADDING`.
  - For information on "same" vs "valid" padding see:
    <https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow>.

- `uint32_t stride_height`

  - Number of positions the kernel moves over the input's `dim3` dimension at
    each step.
  - See [Convolution 2D Requirements](#convolution-2d-requirements) for
    requirements.

- `uint32_t stride_width`

  - Number of positions the kernel moves over the input's `dim2` dimension at
    each step.
  - See [Convolution 2D Requirements](#convolution-2d-requirements) for
    requirements.

- `zdnn_conv2d_act act_func`

  - Activation function to apply to the results.
  - `CONV2D_ACT_NONE` or `CONV2D_ACT_RELU`

- `void *clipping_value`

  - A pointer to an FP32 value, used to clip input tensor's elements.
  - If set to NULL or 0, no clipping will occur.
  - Must not be a negative value.
  - Value is ignored if `act_func` is not set to `CONV2D_ACT_RELU`.

- `zdnn_ztensor *output`

  - The result tensor which will hold the results.
  - Must be a [ZDNN_NHWC](#common-layouts) tensor with pre_transformed shape
    [num_batches, height_out, width_out, channels_out].
  - See [Convolution 2D Requirements](#convolution-2d-requirements) for
    requirements.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Convolution 2D Requirements

| strides and padding                       | input (num_batches, height_in, width_in, channels_in)                | kernel (kernel_height, kernel_width, channels_in, channels_out) | bias (channels_out) | output (num_batches, height_out, width_out, channels_out)                                                                        |
| ----------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| both strides > 0 and =< 13, SAME padding  |                                                                      | both kernel_height and kernel_width must be =< 64               |                     | height_out = ceil(kernel_height/stride_height)<br>width_out = ceil(kernel_width/stride_width)                                    |
| both strides > 0 and =< 13, VALID padding | height_in must be > kernel_height<br>width_in must be > kernel_width | both kernel_height and kernel_width must be =< 64               |                     | height_out = ceil((height_in - kernel_height + 1)/stride_height)<br>width_out = ceil((width_in - kernel_width + 1)/stride_width) |
| both strides = 0, VALID padding           | height_in must be = kernel_height<br>width_in must be = kernel_width | both kernel_height and kernel_width must be =< 448              |                     | both height_out and width_out must be 1                                                                                          |

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
  - Shape of input or output tensor is invalid based on given kernel and stride
    parameters
  - Other general shape violations (exceeds MDIS, etc.)
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_STRIDE_PADDING`
- `ZDNN_INVALID_STRIDES`
- `ZDNN_INVALID_CLIPPING_VALUE`
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Invalid `padding_type`
  - `ZDNN_FUNC_RC_F001` - Invalid `act_func`
  - `ZDNN_FUNC_RC_F002` - `stride_height` = 0 and `stride_width` = 0, but either
    `kernel_height` or `kernel_width` > 448
  - `ZDNN_FUNC_RC_F003` - `stride_height` > 0 and `stride_width` > 0, but either
    `kernel_height` or `kernel_width` > 64
  - `ZDNN_FUNC_RC_F004` - Either `stride_height` or `stride_width` > 13

#### Framework Examples

[TensorFlow Conv2D]

[tensorflow conv2d]:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

[ONNX Conv2D]

[onnx conv2d]: https://github.com/onnx/onnx/blob/master/docs/Operators.md

## Convenience Functions

[Back to Table of Contents](#TOC)

- None

---

## Usage Examples

### Example flow of an application calling the zDNN APIs

[Back to Table of Contents](#TOC)

```
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// ***************************************************************************
// Sample:
//
// Create 2 zTensors a and b, and add them together via zdnn_add()
// ***************************************************************************
int main(int argc, char *argv[]) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor_a;
  zdnn_ztensor ztensor_b;
  zdnn_ztensor ztensor_out;
  zdnn_status status;

  uint32_t dim_n = 1, dim_h = 32, dim_w = 32, dim_c = 3;
  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes
  uint64_t num_elements = dim_n * dim_h * dim_w * dim_c;

  // allocate tensor data storage
  void *data1 = malloc(num_elements * element_size);
  void *data2 = malloc(num_elements * element_size);
  void *data_out = malloc(num_elements * element_size);

  // read input_data

  // check status for AIU availability, supported ops, etc. here
  // status = zdnn_query(…);

  // set input tensor data to 0 to 127 sequentially and repeat
  for (uint64_t i = 0; i < num_elements; i++) {
    ((float *)data1)[i] = (float)(i & 0x7f);
    ((float *)data2)[i] = (float)(i & 0x7f);
  }

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, type, &pre_tfrmd_desc, dim_n, dim_h,
                                 dim_w, dim_c);
  // generate transformed shape information
  status = zdnn_generate_transformed_desc(&pre_tfrmd_desc, &tfrmd_desc);
  assert(status == ZDNN_OK);

  // initialize zTensors and allocate 4k-aligned storage via helper function
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor_a);
  assert(status == ZDNN_OK);
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor_b);
  assert(status == ZDNN_OK);
  status =
      zdnn_init_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc, &ztensor_out);
  assert(status == ZDNN_OK);

  // transform the feature tensor
  status = zdnn_transform_ztensor(&ztensor_a, data1);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&ztensor_b, data2);
  assert(status == ZDNN_OK);

  // perform element-wise add between the two input tensors
  status = zdnn_add(&ztensor_a, &ztensor_b, &ztensor_out);
  assert(status == ZDNN_OK);

  // transform resultant zTensor back to original data format
  status = zdnn_transform_origtensor(&ztensor_out, data_out);
  assert(status == ZDNN_OK);

  for (uint64_t i = 0; i < num_elements; i++) {
    printf("out element %" PRIu64 " %f\n", i, ((float *)data_out)[i]);
  }

  free(data1);
  free(data2);
  free(data_out);
}
```

---

### Example of an application calling the zdnn_lstm API (forward)

[Back to Table of Contents](#TOC)

```
// Sample: LSTM
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: LSTM
int main(int argc, char *argv[]) {
  zdnn_status status;

  /***********************************************************************
   *
   * LSTM (FWD/BWD):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (1, num_batches, num_hiddens)
   * c0              |  ZDNN_3DS  | (1, num_batches, num_hiddens)
   * weights         |  ZDNN_3DS  | (1, num_features, num_hiddens)
   * biases          |  ZDNN_2DS  | (1, num_hiddens)
   * hidden_weights  |  ZDNN_3DS  | (1, num_hiddens, num_hiddens)
   * hidden_biases   |  ZDNN_2DS  | (1, num_hiddens)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_3DS  | (num_timesteps, num_batches, num_hiddens)
   *                 |            | or (1, num_batches, num_hiddens)
   * cf_output       |  ZDNN_3DS  | (1, num_batches, num_hiddens)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hiddens = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  zdnn_ztensor_concat_types concat_type = CONCAT_LSTM;
  lstm_gru_direction dir = FWD;
  uint8_t num_dirs = (dir == BIDIR) ? 2 : 1;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &input_pre_tfrmd_desc,
                                 num_timesteps, num_batches, num_features);
  status =
      zdnn_generate_transformed_desc(&input_pre_tfrmd_desc, &input_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                         &input_tfrmd_desc, &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size =
      num_timesteps * num_batches * num_features * element_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_ztensor(&input, input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create initial hidden and cell state zTensors
   ***********************************************************************/

  zdnn_tensor_desc h0c0_pre_tfrmd_desc, h0c0_tfrmd_desc;
  zdnn_ztensor h0, c0;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &h0c0_pre_tfrmd_desc, num_dirs,
                                 num_batches, num_hiddens);
  status =
      zdnn_generate_transformed_desc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &h0);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &c0);
  assert(status == ZDNN_OK);

  uint64_t h0c0_data_size = num_batches * num_hiddens * element_size;
  void *hidden_state_data = malloc(h0c0_data_size);
  void *cell_state_data = malloc(h0c0_data_size);

  status = zdnn_transform_ztensor(&h0, hidden_state_data);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&c0, cell_state_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create input weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &weights_pre_tfrmd_desc,
                                 num_dirs, num_features, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, concat_type, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hiddens * element_size;
  void *weights_data_f = malloc(weights_data_size);
  void *weights_data_i = malloc(weights_data_size);
  void *weights_data_c = malloc(weights_data_size);
  void *weights_data_o = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_f, weights_data_i,
                                  weights_data_c, weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hiddens, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc, concat_type, &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hiddens * num_hiddens * element_size;
  void *hidden_weights_data_f = malloc(hidden_weights_data_size);
  void *hidden_weights_data_i = malloc(hidden_weights_data_size);
  void *hidden_weights_data_c = malloc(hidden_weights_data_size);
  void *hidden_weights_data_o = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_f,
                                  hidden_weights_data_i, hidden_weights_data_c,
                                  hidden_weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases and hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases, hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, concat_type, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hiddens * element_size;
  void *biases_data_f = malloc(biases_data_size);
  void *biases_data_i = malloc(biases_data_size);
  void *biases_data_c = malloc(biases_data_size);
  void *biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_f, biases_data_i,
                                  biases_data_c, biases_data_o);
  assert(status == ZDNN_OK);

  void *hidden_biases_data_f = malloc(biases_data_size);
  void *hidden_biases_data_i = malloc(biases_data_size);
  void *hidden_biases_data_c = malloc(biases_data_size);
  void *hidden_biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&hidden_biases, hidden_biases_data_f,
                                  hidden_biases_data_i, hidden_biases_data_c,
                                  hidden_biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/

  // get only the last timestep, thus hn and cf can share descriptor
  zdnn_tensor_desc hncf_pre_tfrmd_desc, hncf_tfrmd_desc;

  zdnn_ztensor hn_output_ztensor, cf_output_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hncf_pre_tfrmd_desc, 1,
                                 num_batches, num_hiddens);
  status =
      zdnn_generate_transformed_desc(&hncf_pre_tfrmd_desc, &hncf_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&hncf_pre_tfrmd_desc, &hncf_tfrmd_desc,
                                         &hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hncf_pre_tfrmd_desc, &hncf_tfrmd_desc,
                                         &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the AIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_lstm(&input, &h0, &c0, &weights, &biases, &hidden_weights,
                     &hidden_biases, dir, work_area, &hn_output_ztensor,
                     &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  uint64_t hncf_data_size = num_batches * num_hiddens * element_size;
  void *hn_output_data = malloc(hncf_data_size);
  void *cf_output_data = malloc(hncf_data_size);

  status = zdnn_transform_origtensor(&hn_output_ztensor, hn_output_data);
  assert(status == ZDNN_OK);
  status = zdnn_transform_origtensor(&cf_output_ztensor, cf_output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&h0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&c0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&cf_output_ztensor);
  assert(status == ZDNN_OK);

  free(input_data);
  free(hidden_state_data);
  free(cell_state_data);
  free(weights_data_f);
  free(weights_data_i);
  free(weights_data_c);
  free(weights_data_o);
  free(hidden_weights_data_f);
  free(hidden_weights_data_i);
  free(hidden_weights_data_c);
  free(hidden_weights_data_o);
  free(biases_data_f);
  free(biases_data_i);
  free(biases_data_c);
  free(biases_data_o);
  free(hidden_biases_data_f);
  free(hidden_biases_data_i);
  free(hidden_biases_data_c);
  free(hidden_biases_data_o);
  free(hn_output_data);
  free(cf_output_data);
}


```

---

### Example of an application calling the zdnn_lstm API (bi-directional)

[Back to Table of Contents](#TOC)

```
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: LSTM BI-DIR
int main(int argc, char *argv[]) {
  zdnn_status status;

  /***********************************************************************
   *
   * LSTM (BI-DIR):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (2, num_batches, num_hiddens)
   * c0              |  ZDNN_3DS  | (2, num_batches, num_hiddens)
   * weights         |  ZDNN_3DS  | (2, num_features, num_hiddens)
   * biases          |  ZDNN_2DS  | (2, num_hiddens)
   * hidden_weights  |  ZDNN_3DS  | (2, num_hiddens, num_hiddens)
   * hidden_biases   |  ZDNN_2DS  | (2, num_hiddens)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_3DS  | (num_timesteps, num_batches, num_hiddens)
   *                 |            | or (1, num_batches, num_hiddens)
   * cf_output       |  ZDNN_3DS  | (1, num_batches, num_hiddens)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hiddens = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  zdnn_ztensor_concat_types concat_type = CONCAT_LSTM;
  lstm_gru_direction dir = BIDIR;
  uint8_t num_dirs = (dir == BIDIR) ? 2 : 1;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &input_pre_tfrmd_desc,
                                 num_timesteps, num_batches, num_features);
  status =
      zdnn_generate_transformed_desc(&input_pre_tfrmd_desc, &input_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                         &input_tfrmd_desc, &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size =
      num_timesteps * num_batches * num_features * element_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_ztensor(&input, input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create initial hidden and cell state zTensors
   ***********************************************************************/

  zdnn_tensor_desc h0c0_pre_tfrmd_desc, h0c0_tfrmd_desc;
  zdnn_ztensor h0, c0;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &h0c0_pre_tfrmd_desc, num_dirs,
                                 num_batches, num_hiddens);
  status =
      zdnn_generate_transformed_desc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &h0);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &c0);
  assert(status == ZDNN_OK);

  uint64_t h0c0_data_size = num_batches * num_hiddens * element_size;
  void *hidden_state_data = malloc(h0c0_data_size);
  void *cell_state_data = malloc(h0c0_data_size);

  status = zdnn_transform_ztensor(&h0, hidden_state_data);
  assert(status == ZDNN_OK);
  status = zdnn_transform_ztensor(&c0, cell_state_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create input weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &weights_pre_tfrmd_desc,
                                 num_dirs, num_features, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, concat_type, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hiddens * element_size;
  void *weights_data_f = malloc(weights_data_size);
  void *weights_data_i = malloc(weights_data_size);
  void *weights_data_c = malloc(weights_data_size);
  void *weights_data_o = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_f, weights_data_i,
                                  weights_data_c, weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hiddens, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc, concat_type, &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hiddens * num_hiddens * element_size;
  void *hidden_weights_data_f = malloc(hidden_weights_data_size);
  void *hidden_weights_data_i = malloc(hidden_weights_data_size);
  void *hidden_weights_data_c = malloc(hidden_weights_data_size);
  void *hidden_weights_data_o = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_f,
                                  hidden_weights_data_i, hidden_weights_data_c,
                                  hidden_weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases and hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases, hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, concat_type, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hiddens * element_size;
  void *biases_data_f = malloc(biases_data_size);
  void *biases_data_i = malloc(biases_data_size);
  void *biases_data_c = malloc(biases_data_size);
  void *biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_f, biases_data_i,
                                  biases_data_c, biases_data_o);
  assert(status == ZDNN_OK);

  void *hidden_biases_data_f = malloc(biases_data_size);
  void *hidden_biases_data_i = malloc(biases_data_size);
  void *hidden_biases_data_c = malloc(biases_data_size);
  void *hidden_biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&hidden_biases, hidden_biases_data_f,
                                  hidden_biases_data_i, hidden_biases_data_c,
                                  hidden_biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/

  zdnn_tensor_desc hn_pre_tfrmd_desc, hn_tfrmd_desc, cf_pre_tfrmd_desc,
      cf_tfrmd_desc;

  zdnn_ztensor hn_output_ztensor, cf_output_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hn_pre_tfrmd_desc,
                                 num_timesteps, num_batches, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &hn_pre_tfrmd_desc, CONCAT_BIDIR_OUTPUT, &hn_tfrmd_desc);
  assert(status == ZDNN_OK);

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &cf_pre_tfrmd_desc, 1,
                                 num_batches, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &cf_pre_tfrmd_desc, CONCAT_BIDIR_OUTPUT, &cf_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc,
                                         &hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc,
                                         &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the AIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_lstm(&input, &h0, &c0, &weights, &biases, &hidden_weights,
                     &hidden_biases, dir, work_area, &hn_output_ztensor,
                     &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   *
   * NOTE: zdnn_transform_origtensor() bi-directional output is not
   *       supported
   ***********************************************************************/

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&h0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&c0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&cf_output_ztensor);
  assert(status == ZDNN_OK);

  free(input_data);
  free(hidden_state_data);
  free(cell_state_data);
  free(weights_data_f);
  free(weights_data_i);
  free(weights_data_c);
  free(weights_data_o);
  free(hidden_weights_data_f);
  free(hidden_weights_data_i);
  free(hidden_weights_data_c);
  free(hidden_weights_data_o);
  free(biases_data_f);
  free(biases_data_i);
  free(biases_data_c);
  free(biases_data_o);
  free(hidden_biases_data_f);
  free(hidden_biases_data_i);
  free(hidden_biases_data_c);
  free(hidden_biases_data_o);
}



```

---

### Example of an application calling the zdnn_gru API (forward)

[Back to Table of Contents](#TOC)

```
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: GRU
int main(int argc, char *argv[]) {
  zdnn_status status;

  /***********************************************************************
   *
   * GRU (FWD/BWD):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (1, num_batches, num_hiddens)
   * weights         |  ZDNN_3DS  | (1, num_features, num_hiddens)
   * input_biases    |  ZDNN_2DS  | (1, num_hiddens)
   * hidden_weights  |  ZDNN_3DS  | (1, num_hiddens, num_hiddens)
   * hidden_biases   |  ZDNN_2DS  | (1, num_hiddens)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_3DS  | (num_timesteps, num_batches, num_hiddens)
   *                 |            | or (1, num_batches, num_hiddens)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hiddens = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  zdnn_ztensor_concat_types concat_type = CONCAT_GRU;
  lstm_gru_direction dir = FWD;
  uint8_t num_dirs = (dir == BIDIR) ? 2 : 1;
  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &input_pre_tfrmd_desc,
                                 num_timesteps, num_batches, num_features);
  status =
      zdnn_generate_transformed_desc(&input_pre_tfrmd_desc, &input_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                         &input_tfrmd_desc, &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size =
      num_timesteps * num_batches * num_features * element_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_ztensor(&input, input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create initial hidden zTensor
   ***********************************************************************/

  zdnn_tensor_desc h0_pre_tfrmd_desc, h0_tfrmd_desc;
  zdnn_ztensor h0;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &h0_pre_tfrmd_desc, num_dirs,
                                 num_batches, num_hiddens);
  status = zdnn_generate_transformed_desc(&h0_pre_tfrmd_desc, &h0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status =
      zdnn_init_ztensor_with_malloc(&h0_pre_tfrmd_desc, &h0_tfrmd_desc, &h0);
  assert(status == ZDNN_OK);

  uint64_t h0_data_size = num_batches * num_hiddens * element_size;
  void *hidden_state_data = malloc(h0_data_size);

  status = zdnn_transform_ztensor(&h0, hidden_state_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create input weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &weights_pre_tfrmd_desc,
                                 num_dirs, num_features, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, concat_type, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hiddens * element_size;
  void *weights_data_z = malloc(weights_data_size);
  void *weights_data_r = malloc(weights_data_size);
  void *weights_data_h = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_z, weights_data_r,
                                  weights_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hiddens, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc, concat_type, &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hiddens * num_hiddens * element_size;
  void *hidden_weights_data_z = malloc(hidden_weights_data_size);
  void *hidden_weights_data_r = malloc(hidden_weights_data_size);
  void *hidden_weights_data_h = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_z,
                                  hidden_weights_data_r, hidden_weights_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases and hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases, hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hiddens);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, concat_type, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hiddens * element_size;
  void *biases_data_z = malloc(biases_data_size);
  void *biases_data_r = malloc(biases_data_size);
  void *biases_data_h = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_z, biases_data_r,
                                  biases_data_h);
  assert(status == ZDNN_OK);

  void *hidden_biases_data_z = malloc(biases_data_size);
  void *hidden_biases_data_r = malloc(biases_data_size);
  void *hidden_biases_data_h = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&hidden_biases, hidden_biases_data_z,
                                  hidden_biases_data_r, hidden_biases_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/

  // get only the last timestep
  zdnn_tensor_desc hn_pre_tfrmd_desc, hn_tfrmd_desc;

  zdnn_ztensor hn_output_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hn_pre_tfrmd_desc, 1,
                                 num_batches, num_hiddens);
  status = zdnn_generate_transformed_desc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc,
                                         &hn_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the AIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_gru(&input, &h0, &weights, &biases, &hidden_weights,
                    &hidden_biases, dir, work_area, &hn_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  uint64_t hn_data_size = num_batches * num_hiddens * element_size;
  void *hn_output_data = malloc(hn_data_size);

  status = zdnn_transform_origtensor(&hn_output_ztensor, hn_output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&h0);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hidden_biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hn_output_ztensor);
  assert(status == ZDNN_OK);

  free(input_data);
  free(hidden_state_data);
  free(weights_data_z);
  free(weights_data_r);
  free(weights_data_h);
  free(hidden_weights_data_z);
  free(hidden_weights_data_r);
  free(hidden_weights_data_h);
  free(biases_data_z);
  free(biases_data_r);
  free(biases_data_h);
  free(hidden_biases_data_z);
  free(hidden_biases_data_r);
  free(hidden_biases_data_h);
  free(hn_output_data);
}


```
