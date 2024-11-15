# zDNN API Reference

## Contacts

- Nicholas Marion (<nmarion@us.ibm.com>)
- Andreas Krebbel (<krebbel@linux.ibm.com>)
- Steven Jones (<sbj@us.ibm.com>)

## Version

1.2.0

## Table of Contents <a id="TOC"></a>

1. [Overview](#overview)
2. [Environment](#environment)
3. [Building zDNN](#building-and-installing-zdnn)
4. [Common Data Types and Structs](#common-types-and-structs)

   - [Version Information](#common-version-info)
   - [zDNN zTensor](#common-ztensor)
     - [General zTensor Requirements](#gen-zten-reqs)
     - [Concatenated zTensor Requirements](#concat-zten-reqs)
     - [Quantized zTensor Requirements](#quan-zten-reqs)
   - [zDNN Tensor Descriptors](#common-descriptors)
   - [zDNN Data Layouts](#common-layouts)
   - [zDNN Data Formats](#common-formats)
   - [zDNN Data Types](#common-types)
   - [zDNN Quantized Transform Types](#quantized-transform-types)
   - [zDNN Statuses](#common-statuses)

5. [Runtime Environment Variables](#env-vars)
6. [Validating the Runtime Environment](#runtime-val)
7. [API Reference](#api-reference)

   - [Support Functions](#support-functions)
   - [Data Transformation](#data-transformation)
   - [Operations](#operations)

     - [Element-wise](#elwise-ops)
     - [Activation](#act-ops)
     - [Normalization](#norm-ops)
     - [Matmul with Operation](#zdnn_matmul_op)
     - [Matmul Broadcast with Operation](#zdnn_matmul_bcast_op)
     - [Matmul Transpose with Operation](#zdnn_matmul_transpose_op)
     - [Quantized Matmul Operation](#zdnn_quantized_matmul_op)
     - [LSTM](#zdnn_lstm)
     - [GRU](#zdnn_gru)
     - [Average Pool 2D](#zdnn_avgpool2d)
     - [Max Pool 2D](#zdnn_maxpool2d)
     - [Convolution 2D](#zdnn_conv2d)

   - [Convenience Functions](#convenience-functions)

8. [Usage Examples](#usage-examples)

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
- For deep learning operations, zAIU requires the use of internal data types:
  - DLFLOAT16, a 2-byte data type supported in Telum I, which optimizes training
    and inference while minimizing the loss of accuracy at inference time
    (versus standard 4-byte formats),
  - INT8, a 1-byte data type supported with Telum II, which allows tensor
    quantization features.

The zDNN library provides a set of APIs that an exploiter will utilize to drive
the desired request. zDNN will be available on both z/OS and Linux on Z; the
inclusion of Linux on Z provides particular benefit, as it will allow us to
enable acceleration in frameworks for z/OS via z/OS Container Extensions (zCX).

---

## Environment

z/OS:

- Problem state
- AMODE64
- XPLINK

### Alignment requirements

#### zAIU Op Limits

_This implies a zDNN limitation as well at this point._

- For all ops:

  - Number of elements in any dimension must not exceed the value returned by
    `zdnn_get_max_for_dim(uint8_t dimension)`
  - Total number of bytes required for storing a transformed tensor must not
    exceed the value returned by `zdnn_get_nnpa_max_tensor_size()`

### Application interfaces for zAIU Enterprise Neural Network Inference

#### zDNN General

The zDNN deep learning library provides the standard IBM Z software interface to
the zAIU. This IBM-provided C library provides a set of functions that handle
the data transformation requirements of the zAIU and provide wrapper functions
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

On Linux on Z we expect to ship source as well a package-installable library and
header. The library installation will conform to the standards of the packaging
method chosen.

---

## Building and Installing zDNN

### Clone the Repository and Submodules

```
git clone --recurse-submodules git@github.com:IBM/zDNN.git
```

### Create configure script

To create configure script

```
autoreconf .
```

### Configure Build

Prepare the build and install environment and check for necessary dependencies
using `./configure` script.

```
./configure [OPTION]... [VAR=VALUE]...
```

#### Installation Options

- `--prefix=PREFIX`
  - Install architecture-independent files in PREFIX. Default location is
    `/usr/local`
- `--exec-prefix=EPREFIX`
  - Install architecture-independent files in EPREFIX. Default location is
    `PREFIX`

_To explore all available configuration options and features, use `-h`_

### Build Library

Compile zDNN library using:

```
make build
```

### Run Tests

To run tests:

```
make test
```

#### Unity Requirement

_Please note that the Unity test framework source code is required to run unit
tests. If you did not clone submodules along with initial zDNN clone, please
perform the following steps to setup Unity prior to issuing `make tests`:_

1. Clone the source code from the
   [Throw The Switch - Unity](https://github.com/ThrowTheSwitch/Unity)
   repository.
2. Set the `UNITY_ROOT` environment variable to the folder containing the Unity
   source code.

#### Python Package Requirements

_Please note that `junit_xml` and `pyparsing` are required python packages in
order to properly parse and format Unity test results. Follow standard python
package installation practices to meet requirements._

### Install

Install zDNN library:

```
sudo make install
```

### Reference Commands

Configure help:

```
./configure -h
```

Make help:

```
make help
```

### Prerequisite Tools

Compilers:

- `GCC: GNU Compiler Collection (gcc)`

or

- `IBM XL C/C++: (xlc)`

Build Tools and Dependencies:

- `Autoconf`
- `Make`
- `Unity`
- `Python Packages` _For formatting test results_
  - junit_xml
  - pyparsing

---

## Common Types and Structs

Include Files: `zdnn.h`

### Version Information <a id="common-version-info"></a>

[Back to Table of Contents](#TOC)

```C
#define ZDNN_VERSION "1.2.0"
#define ZDNN_VERNUM 0x010200 // 0x[major][minor][patch]
#define ZDNN_VER_MAJOR 1
#define ZDNN_VER_MINOR 2
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

```C
typedef struct zdnn_ztensor {
  zdnn_tensor_desc
      *pre_transformed_desc; // tensor's shape information before transformation
  zdnn_tensor_desc *transformed_desc; // transformed tensor's shape information
  uint64_t buffer_size;               // tensor size in bytes
  void *buffer;                       // pointer to the tensor in memory
  bool is_transformed; // indicator if data in buffer has been transformed
  char reserved[3];    // not currently used, should contain zeros.
  float rec_scale;    // the scale factor for quantization, stored as reciprocal
  float offset;       // the offset for quantization
  char reserved2[20]; // not currently used, should contain zeros.
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
- `reserved` should contain zeros, otherwise the program may not operate
  compatibly in the future.
  - Calling [zdnn_init_ztensor](#zdnn_init_ztensor) or
    [zdnn_init_ztensor_with_malloc](#zdnn_init_ztensor_with_malloc) will set
    `reserved` to zeros.

#### Concatenated zTensor Requirements <a id="concat-zten-reqs"></a>

[Back to Table of Contents](#TOC)

- For use with weights/biases/hidden-weights/hidden-biases RNN-gates tensors.
- You must use
  [zdnn_generate_transformed_desc_concatenated](#zdnn_generate_transformed_desc_concatenated)
  with the appropriate concatenation info
  - Do not use `zdnn_generate_transformed_desc` with concatenated tensors
- The pre-transformed shape dimensions should not include the concatenation.
  - Thus, the pre-transformed shape should be that of a single gate, not the
    shape of the combined gates
- Afterward transform with [zdnn_transform_ztensor](#zdnn_transform_ztensor) as
  normal
- Must follow [general tensor requirements](#gen-zten-reqs)

#### Quantized zTensor Requirements <a id="quan-zten-reqs"></a>

[Back to Table of Contents](#TOC)

- Supported `transform_desc` and `pre_transformed_desc` types for
  [zdnn_transform_quantized_ztensor](#zdnn_transform_quantized_ztensor) and
  [zdnn_generate_quantized_transformed_desc](#zdnn_generate_quantized_transformed_desc):
  - `ZDNN_FORMAT_4DFEATURE` format:
    - ZDNN_DLFLOAT16
      - FP16, FP32, BFLOAT
    - ZDNN_BINARY_INT8
      - INT8, FP16, FP32, BFLOAT
  - `ZDNN_FORMAT_4DWEIGHTS` format:
    - ZDNN_BINARY_INT8
      - INT8

### zDNN Tensor Descriptors <a id="common-descriptors"></a>

[Back to Table of Contents](#TOC)

```C
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
  - A [ZDNN_NHWC](#common-layouts) expects dims such that dim4 = N, dim3 = H,
    dim2 = W, dim1 = C
  - A [ZDNN_NCHW](#common-layouts) expects dims such that dim4 = N, dim3 = C,
    dim2 = H, dim1 = W
  - A [ZDNN_HWCK](#common-layouts) expects dims such that dim4 = H, dim3 = W,
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

```C
typedef enum zdnn_data_layouts {
  ZDNN_1D,          // 1d tensor
  ZDNN_2D,          // 2d tensor
  ZDNN_2DS,         // represents special 2D tensors required by LSTM/GRU
  ZDNN_3D,          // 3d tensor
  ZDNN_3DS,         // represents special 3D tensors required by
                    // LSTM/GRU/Softmax/Matmul
  ZDNN_ZRH,         // represents (update, reset, hidden) used by GRU
  ZDNN_4D,          // 4d tensor
  ZDNN_4DS,         // represents special 4D tensors required by LSTM/GRU output
  ZDNN_NHWC,        // 4d feature tensor in NHWC
  ZDNN_NCHW,        // 4d feature tensor in NCHW
  ZDNN_FICO,        // represents (forget, input, cell, output) used by LSTM
  ZDNN_HWCK,        // 4d kernel CNN tensor
  ZDNN_BIDIR_ZRH,   // ZRH variant to work with bidirectional LSTM/GRU output
  ZDNN_BIDIR_FICO   // FICO variant to work with bidirectional LSTM/GRU output
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
- `ZDNN_4DS` - Arrangement for RNN output tensor

The followings are set automatically in `transformed_desc` based on `info` when
calling `zdnn_generate_transformed_desc_concatenated()`:

- `ZDNN_ZRH/FICO` - During transformation, the RNN input gates data are
  concatenated on the innermost dimension. Supported with
  `pre_transformed_layout` of `ZDNN_2DS` or `ZDNN_3DS`.
- `ZDNN_BIDIR_ZRH/FICO` - Similar to `ZDNN_ZRH/FICO`, used when:
  1. transforming RNN input weight gate data, and
  2. the input tensor for the current RNN layer is a bidirectional RNN output
     from a previous RNN layer

### zDNN Data Formats <a id="common-formats"></a>

[Back to Table of Contents](#TOC)

```C
typedef enum zdnn_data_formats {
  ZDNN_FORMAT_4DFEATURE, // tensor in zAIU data layout format 0
  ZDNN_FORMAT_4DKERNEL,  // tensor in zAIU data layout format 1
  ZDNN_FORMAT_4DWEIGHTS, // tensor in zAIU data layout format 2
  ZDNN_FORMAT_4DGENERIC, // tensor in zAIU data layout format 31
} zdnn_data_formats;
```

### zDNN Data Types <a id="common-types"></a>

[Back to Table of Contents](#TOC)

```C
typedef enum zdnn_data_types {
  ZDNN_DLFLOAT16,    // 16-bit deep learning format
  ZDNN_BINARY_FP32,  // 32-bit binary-floating-point format
  ZDNN_BINARY_INT8,  // 8-bit signed or unsighed binary integer
  ZDNN_BINARY_INT32, // 32-bit signed or unsigned binary integer
  INT8,   // 8-bit signed or unsigned binary integer format
  INT32,  // 32-bit signed or unsigned binary integer format
  BFLOAT, // Brain floating point format
  FP16,   // 16-bit IEEE-754 floating point format
  FP32,   // 32-bit IEEE-754 floating point format
} zdnn_data_types;
```

### zDNN Quantized Transform Types <a id="quantized-transform-types"></a>

[Back to Table of Contents](#TOC)

```C
typedef enum zdnn_quantized_transform_types {
  QUANTIZED_DLFLOAT16 = 0,   // quantized dlfloat16
  QUANTIZED_INT8 = 1,        // quantized int8
  QUANTIZED_WEIGHTS_INT8 = 2 // quantized weights
} zdnn_quantized_transform_types;
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
| ZDNN_ELEMENT_RANGE_VIOLATION     | 0x00020001 | zAIU operation resulted in data that was out of the normal range. |

_Note: ZDNN_ELEMENT_RANGE_VIOLATION indicates a **range violation** occurred for
the zAIU operation based on the data in the tensors. This usually indicates an
overflow of an NNPA internal data type, but can also be associated with
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
| ZDNN_INVALID_CONCAT_INFO         | 0x00040006 | Invalid concatenation info.                                                    |
| ZDNN_INVALID_STRIDE_PADDING\*    | 0x00040007 | Invalid padding type parameter for current strides.                            |
| ZDNN_INVALID_STRIDES\*           | 0x00040008 | Invalid stride height or width parameter.                                      |
| ZDNN_MISALIGNED_PARMBLOCK\*      | 0x00040009 | NNPA parameter block is not on double word boundary.                           |
| ZDNN_INVALID_CLIPPING_VALUE      | 0x0004000A | Invalid clipping for the specified operation.                                  |
| ZDNN_INVALID_ADJUSTMENT_FACTOR   | 0x0004000B | Invalid adjustment for the specified operation.                                |
| ZDNN_INVALID_EPSILON             | 0x0004000C | Invalid epsilon for the specified operation.                                   |
| ZDNN_INVALID_TRANSFORM_TYPE      | 0x0004000D | Invalid transformation type.                                                   |
| ZDNN_INVALID_BETA                | 0x0004000E | Invalid beta value for the specified operation.                                |
| ZDNN_INVALID_GAMMA               | 0x0004000F | Invalid gamma value for the specified operation.                               |
| ZDNN_INVALID_BESSEL_CORRECTION   | 0x00040010 | Invalid bessel correction value for the specified operation.                   |
| ZDNN_INVALID_SCALE               | 0x00040011 | Invalid scale value for the specified operation.                               |
| ZDNN_INVALID_OFFSET              | 0x00040012 | Invalid offset value for the specified operation.                              |
| ZDNN_ALLOCATION_FAILURE          | 0x00100001 | Can not allocate storage.                                                      |
| ZDNN_INVALID_BUFFER              | 0x00100002 | Buffer address is NULL or not on 4K-byte boundary or insufficient buffer size. |
| ZDNN_CONVERT_FAILURE             | 0x00100003 | Floating point data conversion failure.                                        |
| ZDNN_INVALID_STATE               | 0x00100004 | Invalid zTensor state.                                                         |
| ZDNN_UNSUPPORTED_AIU_EXCEPTION   | 0x00100005 | zAIU operation returned an unexpected exception.                               |

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

## Validating the environment at runtime <a id="runtime-val"></a>

### Programming Notes

- Most API calls require a minimum zDNN library and hardware for the API to
  function. There are three zDNN APIs for validation of the zDNN runtime
  environment:
  - Validating the zDNN Library version:
    - This is the version of the libzdnn package installed on the host or
      embedded in the runtime application.
    - The zDNN library version is independent of the hardware available on the
      current system.
    - zDNN APIs introduced in newer versions of the zDNN library will not exist
      in older versions of the library. Attempting to call them will result in
      application crashes.
    - The zDNN library version is returned by
      [zdnn_get_library_version](#zdnn_get_library_version).
  - Validating the zDNN API version:
    - This is the version of zDNN APIs that are compatible on the current system
      and is separate of the zDNN library version.
    - Calling zDNN APIs while running on a system which does not support that
      zDNN API version will return a [hardware status](#hw-statuses) instead of
      [ZDNN_OK](#common-statuses).
    - The zDNN API version available is returned by
      [zdnn_get_max_runnable_version](#zdnn_get_max_runnable_version) and is
      reflected in the return value of
      [zdnn_is_version_runnable](#zdnn_is_version_runnable).
    - zDNN API 1.0.x indicates the API requires Telum I or greater.
    - zDNN API 1.1.x indicates the API requires Telum II or greater.
  - Validating NNPA availability:
    - This indicates if the current system has zAIU hardware present and
      enabled.
    - It is possible to be on a system with zAIU hardware but the feature is
      unavailable, such as z/VM when there is a mix of hardware levels.
    - This is returned by [zdnn_is_nnpa_installed](#zdnn_is_nnpa_installed)
- Examples:
  - Given a Telum I system with zDNN 1.1.0 installed:
    - [zdnn_get_library_version](#zdnn_get_library_version) will return
      `0x00010100` indicating zDNN library 1.1.0 is installed.
    - [zdnn_is_nnpa_installed](#zdnn_is_nnpa_installed) will return `true`
      (unless the zAIU feature is disabled for the system).
    - [zdnn_get_max_runnable_version](#zdnn_get_max_runnable_version) will
      return `0x000100FF` indicating zDNN APIs 1.0.x and below are available for
      use on the system.
    - Checking [zdnn_is_version_runnable(0x00010100)](#zdnn_is_version_runnable)
      (1.1.0) will return `false` as only zDNN APIs 1.0.x and below are
      available for use on the system.
    - Checking [zdnn_is_version_runnable(0x00010100)](#zdnn_is_version_runnable)
      (1.0.0) will return `true` as zDNN APIs 1.0.x and below are available for
      use on the system.
  - Given a Telum II system with zDNN 1.1.0 installed:
    - [zdnn_get_library_version](#zdnn_get_library_version) will return
      `0x00010100` indicating zDNN library 1.1.0 is installed.
    - [zdnn_is_nnpa_installed](#zdnn_is_nnpa_installed) will return `true`
      (unless the zAIU feature is disabled for the system).
    - [zdnn_get_max_runnable_version](#zdnn_get_max_runnable_version) will
      return `0x000101FF` indicating zDNN APIs 1.1.x and below are available for
      use on the system.
    - Checking [zdnn_is_version_runnable(0x00010100)](#zdnn_is_version_runnable)
      (1.1.0) will return `true` as zDNN APIs 1.1.x and below are available for
      use on the system.
    - Checking [zdnn_is_version_runnable(0x00010100)](#zdnn_is_version_runnable)
      (1.0.0) will return `true` as zDNN APIs 1.1.x and below are available for
      use on the system.
  - Given a Telum II system with zDNN 1.0.0 installed:
    - [zdnn_get_library_version](#zdnn_get_library_version) will return
      `0x00010000` indicating zDNN library 1.0.0 is installed.
    - [zdnn_is_nnpa_installed](#zdnn_is_nnpa_installed) will return `true`
      (unless the zAIU feature is disabled for the system).
    - [zdnn_get_max_runnable_version](#zdnn_get_max_runnable_version) will
      return `0x000100FF` indicating zDNN APIs 1.0.x and below are available for
      use on the system.
    - Checking [zdnn_is_version_runnable(0x00010100)](#zdnn_is_version_runnable)
      (1.1.0) will return `false` as only zDNN APIs 1.0.x and below are
      available for use on the system.
    - Checking [zdnn_is_version_runnable(0x00010100)](#zdnn_is_version_runnable)
      (1.0.0) will return `true` as zDNN APIs 1.1.x and below are available for
      use on the system.

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
- [Get smallest of the max index size value from across all dimensions](#zdnn_get_nnpa_max_dim_idx_size)
- [Get max index for a given dimension](#zdnn_get_max_for_dim)
- [Get Size](#zdnn_getsize_ztensor)
- [Get Range](#zdnn_getrange_ztensor)
- [Get maximum limit for a given data type](#zdnn_get_max_limit)
- [Get minimum limit for a given data type](#zdnn_get_min_limit)
- [Initialize pre-transformed tensor descriptor](#zdnn_init_pre_transformed_desc)
- [Generate transformed tensor descriptor](#zdnn_generate_transformed_desc)
- [Generate quantized transformed tensor descriptor](#zdnn_generate_quantized_transformed_desc)
- [Generate concatenated transformed tensor descriptor](#zdnn_generate_transformed_desc_concatenated)
- [Initialize zTensor](#zdnn_init_ztensor)
- [Initialize zTensor with memory allocate](#zdnn_init_ztensor_with_malloc)
- [Initialize quantized zTensor](#zdnn_init_quantized_ztensor)
- [Initialize quantized zTensor with memory allocate](#zdnn_init_quantized_ztensor_with_malloc)
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

```C
void zdnn_init();
```

#### Parameters

None

#### Returns

None

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_nnpa_max_dim_idx_size

#### Description

Retrieve the smallest of the maximum dimension index size values across all
dimensions currently supported by the zAIU from zDNN's internal memory.

#### Format

```C
uint32_t zdnn_get_nnpa_max_dim_idx_size();
```

#### Parameters

None

#### Returns

Maximum dimension index size supported by the zAIU across all dimensions

#### Since

Introduced in zDNN 1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_max_for_dim

#### Description

Retrieve the maximum dimension index size value currently supported by the zAIU
for a given dimension from zDNN's internal memory. These limits relate to
ztensor's transformed descriptor values. Special care is required when using
layouts with special re-arrangements of data. See
[zDNN Data Layouts](#zdnn_data_layouts) for more details.

#### Format

```C
uint32_t zdnn_get_max_for_dim(uint8_t dimension);
```

#### Parameters

- `int dimension`

  - dimension to get maximum index size for

#### Returns

Maximum dimension index size supported by the zAIU for a given dimension

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_nnpa_max_tensor_size

#### Description

Retrieve the maximum tensor size value (number of bytes required for storing a
transformed tensor) currently supported by the zAIU from zDNN's internal memory.

#### Format

```C
uint64_t zdnn_get_nnpa_max_tensor_size();
```

#### Parameters

None

#### Returns

Maximum tensor size supported by the zAIU

---

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

### zdnn_is_nnpa_installed

#### Description

Interrogates the hardware to determine if the NNPA and associated instructions
are installed.

Use this function during application initialization to determine whether the
zAIU hardware is available.

#### Format

```C
bool zdnn_is_nnpa_installed();
```

#### Parameters

- None.

#### Returns

`true` if NNPA and associated instructions are installed, `false` otherwise.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_nnpa_function_installed

#### Description

Query, from zDNN internal memory, if requested NNPA functions are available.

#### Format

```C
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
NNPA_MATMUL_OP_BCAST1
NNPA_TRANSFORM
```

#### Returns

`true` if all queried formats are installed or if `count` is zero, `false`
otherwise.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_nnpa_parmblk_fmt_installed

#### Description

Query, from zDNN internal memory, if requested parameter block formats are
installed.

#### Format

```C
bool zdnn_is_nnpa_parmblk_fmt_installed(int count, ...);
```

#### Parameters

- `int count`

  - number of NNPA parameter block formats to check

- `... (additional arguments)`

  - NNPA parameter block formats separated by commas

```
NNPA_PARMBLKFORMAT_0
NNPA_PARMBLKFORMAT_1
```

#### Returns

`true` if all queried formats are installed or if `count` is zero, `false`
otherwise.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_nnpa_datatype_installed

#### Description

Query, from zDNN internal memory, if requested NNPA data type are installed.

#### Format

```C
bool zdnn_is_nnpa_datatype_installed(uint16_t types_bitmask);
```

#### Parameters

- `uint16_t types_bitmask`

  - OR'd type bitmasks as defined in zdnn_query_datatypes enum

```
QUERY_DATATYPE_INTERNAL1
QUERY_DATATYPE_BINARY_FP32
QUERY_DATATYPE_BINARY_INT8
QUERY_DATATYPE_BINARY_INT32
```

#### Returns

`true` if all queried data types are installed, `false` otherwise.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_nnpa_layout_fmt_installed

#### Description

Query, from zDNN internal memory, if requested NNPA data layout format are
installed.

#### Format

```C
bool zdnn_is_nnpa_layout_fmt_installed(uint32_t layout_bitmask);
```

#### Parameters

- `uint32_t layout_bitmask`

  - OR'd layout bitmasks as defined in zdnn_query_layoutfmts enum

```
QUERY_LAYOUTFMT_4DFEATURE
QUERY_LAYOUTFMT_4DKERNEL
QUERY_LAYOUTFMT_4DWEIGHTS
QUERY_LAYOUTFMT_4DGENERIC
```

#### Returns

`true` if all queried data layouts are installed, `false` otherwise.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_nnpa_conversion_installed

#### Description

Query, from zDNN internal memory, if requested NNPA data-type to/from BFP format
conversions are installed.

#### Format

```C
bool zdnn_is_nnpa_conversion_installed(nnpa_data_type type,
                                       uint16_t format_bitmask);
```

#### Parameters

- `nnpa_data_type type`

  - NNPA data-type number as defined in nnpa_data_type enum

```
NNPA_DATATYPE_1
```

- `uint16_t format_bitmask`

  - OR'd BFP format bitmasks as defined in zdnn_query_bfpfmts enum

```
QUERY_BFPFMT_TINY (FP16)
QUERY_BFPFMT_SHORT (FP32/BFLOAT)
```

#### Returns

`true` if all queried conversions are installed, `false` otherwise.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_library_version

#### Description

Retrieve library version number as a 32-bit hex value in the form
`0x00[major][minor][patch]` where each segment is 1 byte. For example zDNN 1.2.3
would return `0x00010203`.

This is the version of the libzdnn package installed on the system or zDNN
embeded in a runtime application. The zDNN library version is independant of the
system that zDNN is running on.

The library version indicates what zDNN APIs exist in that version of the zDNN
library. It does **NOT** indicate whether those APIs are available for use. To
check API availablity at runtime, see
[Validating the environment at runtime](#runtime-val).

#### Format

```
uint32_t zdnn_get_library_version();
```

#### Returns

Library version number in `0x00[major][minor][patch]` format.

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_library_version_str

#### Description

Retrieve the library version number and build information as a string.

#### Format

```C
char *zdnn_get_library_version_str();
```

#### Returns

Library version number and build information as a string.

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_refresh_nnpa_query_result

#### Description

Refresh zDNN in-memory query result from zAIU.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_getsize_ztensor

#### Description

Used to determine the buffer size required for the transformed tensor (including
concatenated) in zDNN transformed format. Requires tensor descriptor
(`zdnn_tensor_desc`) with transformed shape information.

#### Format

```C
uint64_t zdnn_getsize_ztensor(const zdnn_tensor_desc *tfrmd_desc);
```

#### Parameters

- `zdnn_tensor_desc *tfrmd_desc`

  - Contains transformed information about the shape, layout and data type.

#### Returns zdnn_status indications

- required buffer size in bytes

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_getrange_ztensor

#### Description

Used to determine the minimum negative value and maximum positive value of the
passed zdnn_ztensor, storing the results in min and max.

#### Format

```C
void zdnn_getrange_ztensor(const zdnn_ztensor *ztensor, float *min, float *max);
```

#### Parameters

- `const zdnn_ztensor *ztensor`

  - The zdnn_ztensor to return the min and max value of.

- `float *min`

  - Pointer to a float used to store minimum negative value.
    - If all values are positive, -0.0 will be used instead.

- `float *max`

  - Pointer to a float used to store maximum positive value.
    - If all values are negative, 0.0 will be used instead.

#### Returns

- None

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_max_limit

#### Description

Returns the maximum representable value between a transformed and
pre-transformed zdnn_data_type.

#### Format

```C
zdnn_status zdnn_get_max_limit(zdnn_data_types transformed_type,
                           zdnn_data_types pre_transformed_type, void *limit);
```

#### Parameters

- `zdnn_data_types transformed_type`

  - input zdnn transformed data type.
  - Restricted to the following transformed data types:
    - ZDNN_DLFLOAT16
    - ZDNN_BINARY_INT8
    - ZDNN_BINARY_INT32

- `zdnn_data_types pre_transformed_type`

  - input zdnn pre-transformed data type.
  - Restricted to the following transformed data types:
    - INT32
    - INT8
    - FP32
    - FP16
    - BFLOAT

- `void *limit`

  - pointer to max value between transformed_type and pre_transformed_type in
    data type of pre_transformed_type.

#### Returns

- `ZDNN_OK`
- `ZDNN_INVALID_TYPE` - invalid transformed or pre_transformed `type` used and
  conversion could not be completed.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_min_limit

#### Description

Return the minimum representable value between a transformed and pre-transformed
zdnn_data_type.

#### Format

```C
zdnn_status zdnn_get_min_limit(zdnn_data_types transformed_type,
                           zdnn_data_types pre_transformed_type, void *limit);
```

#### Parameters

- `zdnn_data_types transformed_type`

  - input zdnn transformed data type.
  - Restricted to the following transformed data types:
    - ZDNN_DLFLOAT16
    - ZDNN_BINARY_INT8
    - ZDNN_BINARY_INT32

- `zdnn_data_types pre_transformed_type`

  - input zdnn pre-transformed data type.
  - Restricted to the following transformed data types:
    - INT32
    - INT8
    - FP32
    - FP16
    - BFLOAT

- `void *limit`

  - pointer to min value between transformed_type and pre_transformed_type in
    data type of pre_transformed_type.

#### Returns

- `ZDNN_OK`
- `ZDNN_INVALID_TYPE` - invalid transformed or pre_transformed `type` used and
  conversion could not be completed.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_init_pre_transformed_desc

#### Description

Initialize tensor descriptor (`zdnn_tensor_desc`) struct with pre-transformed
(original) shape information.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_generate_transformed_desc

#### Description

Generate transformed tensor descriptor information based on supplied
pre-transformed tensor descriptor.

#### Format

```C
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
- `ZDNN_INVALID_TYPE` - pre-transformed `type` is not recognized or is a type
  only used for quantized ztensors.
- `ZDNN_INVALID_LAYOUT` - pre-transformed `layout` is not recognized or is a
  layout only used for concatenated tensors.

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_generate_quantized_transformed_desc

#### Description

Generate quantized transformed tensor descriptor information based on supplied
pre-transformed tensor descriptor and quantized transform type.

#### Format

```C
zdnn_status zdnn_generate_quantized_transformed_desc(
    const zdnn_tensor_desc *pre_tfrmd_desc,
    zdnn_quantized_transform_types transform_type,
    zdnn_tensor_desc *tfrmd_desc);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information
  - Has the following additional restrictions:
    - Only the following pre-transformed layouts are supported.
      - ZDNN_1D
      - ZDNN_2D
      - ZDNN_2DS
      - ZDNN_3D
      - ZDNN_3DS
      - ZDNN_4D
      - ZDNN_NHWC

- `zdnn_quantized_transform_types transform_type`

  - Type of quantized transformation
    - QUANTIZED_DLFLOAT16
    - QUANTIZED_INT8
    - QUANTIZED_WEIGHTS_INT8

- `zdnn_tensor_desc *tfrmd_desc`

  - output `zdnn_tensor_desc` struct

#### zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_TYPE` - pre-transformed `type` is not recognized, not supported
  for quantized ztensors: [Quantized zTensor Requirements](#quan-zten-reqs)
- `ZDNN_INVALID_LAYOUT` - pre-transformed `layout` is not recognized, not
  supported for quantized ztensors, or is a layout only used for concatenated
  tensors.
- `ZDNN_INVALID_TRANSFORM_TYPE` - Invalid transformation type:
  [Quantized zTensor Requirements](#quan-zten-reqs)

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_generate_transformed_desc_concatenated

#### Description

Generate concatenated transformed tensor descriptor information for RNN
input-gates tensors based on a supplied pre-transformed tensor descriptor.

#### Format

```C
zdnn_status zdnn_generate_transformed_desc_concatenated(
    const zdnn_tensor_desc *pre_tfrmd_desc,
    zdnn_concat_info info, zdnn_tensor_desc *tfrmd_desc);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_concat_info info`

  - Information about how the tensors will be concatenated, consists of the
    RNN_TYPE, PREV_LAYER and USAGE flags OR'd together:

    RNN_TYPE flags:

    - RNN_TYPE_LSTM - For LSTM
    - RNN_TYPE_GRU - For GRU

    PREV_LAYER flags:

    - PREV_LAYER_UNI - Previous RNN layer is uni-directional
    - PREV_LAYER_NONE - Previous layer is not a RNN layer
    - PREV_LAYER_BIDIR - Previous RNN layer is bi-directional

    USAGE flags:

    - USAGE_WEIGHTS - Concatenate as input weights
    - USAGE_HIDDEN_WEIGHTS - Concatenate as input hidden-weights
    - USAGE_BIASES - Concatenate as input biases
    - USAGE_HIDDEN_BIASES - Concatenate as input hidden-biases

- `zdnn_tensor_desc *tfrmd_desc`

  - output `zdnn_tensor_desc` struct

#### zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_TYPE` - pre-transformed `type` is not recognized or is not
  supported for concatenated tensors.
- `ZDNN_INVALID_LAYOUT` - pre-transformed `layout` is not recognized or is not
  supported for concatenated tensors.
- `ZDNN_INVALID_CONCAT_INFO` - invalid concatenation information.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_init_ztensor

#### Description

Initialize a `zdnn_ztensor` struct using the pre-transformed and transformed
tensor shape information

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_init_ztensor_with_malloc

#### Description

Same functionality as `zdnn_init_ztensor`, and computes the size required for
the tensor in the zDNN transformed format and allocates the storage for it. Sets
`buffer` and `buffer_size` fields within `output`.

#### Format

```C
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
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of tfrmd_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_ALLOCATION_FAILURE` - Unable to allocate required memory on a 4K
  boundary.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_init_quantized_ztensor

#### Description

Initialize a `zdnn_ztensor` struct using the pre-transformed and quantized
transformed tensor shape information along with scale and offset.

#### Format

```C
void zdnn_init_quantized_ztensor(zdnn_tensor_desc *pre_tfrmd_desc,
                                 zdnn_tensor_desc *tfrmd_desc, float scale,
                                 float offset, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_tensor_desc *tfrmd_desc`

  - input tensor descriptor with quantized transformed shape information

- `float scale`

  - scale for quantized ztensor, must not be 0.

- `float offset`

  - offset for quantized ztensor

- `zdnn_ztensor *output`

  - The `zdnn_ztensor` struct being initialized.

#### Programming Notes

- The reciprocal of the `scale` value is stored as `output->rec_scale` and is
  used within subsequent quantized calls with reduced precision. Due to this,
  large `scale` values will lead to a `output->rec_scale` that underflows to 0.0
  and will result in an error in subsequent quantized calls.

- The `offset` value is stored as `output->offset` and is used within subsequent
  quantized calls with reduced precision. Due to this, large `offset` values
  will overflow to infinity and will result in an error in subsequent quantized
  calls.

#### Returns

- None

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_init_quantized_ztensor_with_malloc

#### Description

Same functionality as `zdnn_init_quantized_ztensor`, and computes the size
required for the tensor in the zDNN transformed format and allocates the storage
for it. Sets `buffer` and `buffer_size` fields within `output`.

#### Format

```C
zdnn_status zdnn_init_quantized_ztensor_with_malloc(
    zdnn_tensor_desc *pre_tfrmd_desc, zdnn_tensor_desc *tfrmd_desc, float scale,
    float offset, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_tensor_desc *pre_tfrmd_desc`

  - input tensor descriptor with pre-transformed shape information

- `zdnn_tensor_desc *tfrmd_desc`

  - input tensor descriptor with quantized transformed shape information

- `float scale`

  - scale for quantized ztensor, must not be 0.

- `float offset`

  - offset for quantized ztensor

- `zdnn_ztensor *output`

  - The `zdnn_ztensor` struct being initialized.

#### Programming Notes

- The reciprocal of the `scale` value is stored as `output->rec_scale` and is
  used within subsequent quantized calls with reduced precision. Due to this,
  large `scale` values will lead to a `output->rec_scale` that underflows to 0.0
  and will result in an error in subsequent quantized calls.

- The `offset` value is stored as `output->offset` and is used within subsequent
  quantized calls with reduced precision. Due to this, large `offset` values
  will overflow to infinity and will result in an error in subsequent quantized
  calls.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_FORMAT` - `tfrmd_desc->format` is not recognized.
- `ZDNN_INVALID_TYPE` - `tfrmd_desc->type` is not recognized or is a
  pre_tfrmd_desc type.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `tfrmd_desc->dim*` dimensions is 0.
  - One of `tfrmd_desc->dim*` dimensions is greater than
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
  - The total number of tfrmd_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_ALLOCATION_FAILURE` - Unable to allocate required memory on a 4K
  boundary.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_quantized_ztensor

#### Description

Check if a given `zdnn_ztensor` represents a quantized ztensor or not

#### Format

```C
bool zdnn_is_quantized_ztensor(zdnn_ztensor *ztensor);
```

#### Parameters

- `zdnn_ztensor *ztensor`

  - The `zdnn_ztensor` being checked.

#### Returns

`true` if `zdnn_ztensor` represents a quantized ztensor, `false` if not.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_reset_ztensor

#### Description

Reset a `zdnn_ztensor` struct for reuse.

_Note this operation does not set or reset the `buffer` and `buffer_size` fields
nor free the transformed area storage._

#### Format

```C
void zdnn_reset_ztensor(zdnn_ztensor *ztensor);
```

#### Parameters

- `zdnn_ztensor *output`

  - The `zdnn_ztensor` struct being reset.

#### Returns

- None

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_allochelper_ztensor

#### Description

Calculate the size required for the tensor in the zDNN transformed format and
allocate the needed storage, satisfying alignment requirements. Sets `buffer`
and `buffer_size` fields within `ztensor`.

_Note that the calling application assumes ownership of this storage and is
responsible for freeing it._

#### Format

```C
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
- `ZDNN_INVALID_LAYOUT` - `zdnn_ztensor->transformed_desc->layout` is not
  recognized or is not a valid transformed_desc layout.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `ztensor->transformed_desc->dim*` dimensions is 0.
  - One of `ztensor->transformed_desc->dim*` dimensions is greater than
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of transformed_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_ALLOCATION_FAILURE` - Unable to allocate required memory on a 4K
  boundary.

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_free_ztensor_buffer

#### Description

Given an input zdnn_ztensor, zdnn_free_ztensor_buffer will free the transformed
area storage associated with it.

_Note that the routine does not free the storage allocated for the zdnn_ztensor
struct itself._

#### Format

```C
zdnn_status zdnn_free_ztensor_buffer(const zdnn_ztensor *ztensor);
```

#### Parameters

- `zdnn_ztensor *tensor`

  - A `zdnn_ztensor` struct with field buffer pointing to storage allocated.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_INVALID_BUFFER` - `tensor->buffer` is `NULL`

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_status_message

#### Description

Retrieve status message of the status code

#### Format

```C
const char *zdnn_get_status_message(zdnn_status status);
```

#### Parameters

- `zdnn_status status`

  - Status code

#### Returns

Pointer to the description string or "(Status string is not defined.)" if
`status` is not defined.

#### Since

1.0.0

#### Requirements

- Any System Z hardware level

See [Validating the environment at runtime](#runtime-val).

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

```C
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
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_is_version_runnable

#### Description

Check if application built for zDNN version `ver_num` can be run on the current
zAIU hardware with the installed zDNN library

#### Format

```C
bool zdnn_is_version_runnable(uint32_t ver_num);
```

#### Parameters

- `ver_num`

  - Version number of the zDNN library application itself, in
    0x00\[major\]\[minor\]\[patch\] form. Typically this is the ZDNN_VERNUM used
    to compile the application

#### Returns

- true/false

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_get_max_runnable_version

#### Description

Returns the maximum version number associated with the APIs supported by the
hardware and zDNN software in the current environment. This can be compared with
the version documented in the "REQUIRES" section of each programming interface
to discern whether the interface is supported at run-time.

The returned value is a version number in the `major`.`minor` format. APIs
defined at that level and below will be supported in the current environment.

#### Format

```C
uint32_t zdnn_get_max_runnable_version();
```

#### Parameters

- None

#### Returns

- A 32-bit zDNN version number in `0x00\[major\]\[minor\]FF` form.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

## Data Transformation

[Back to Table of Contents](#TOC)

- [Transform to zTensor](#zdnn_transform_ztensor)
- [Transform to zTensor with saturation](#zdnn_transform_ztensor_with_saturation)
- [Transform to quantized zTensor](#zdnn_transform_quantized_ztensor)
- [Transform to Original](#zdnn_transform_origtensor)

---

zAIU requires the tensor data to be arranged in a format that enhances the
performance characteristics of the operations. In this documentation, it is
referred to as "transformed format". In addition, data conversions are necessary
from the common formats (FP32, FP16, BFLOAT) to formats (DLFLOAT16) supported by
the zAIU (DLFLOAT16, INT8). The following functions are provided:

- '`zdnn_transform_ztensor` and `zdnn_transform_ztensor_with_saturation`

  - These functions will transform the input tensor and convert the input data
    to the format required by the zAIU. The resulting transformed ztensor can be
    reused as many times as necessary.

  - See [zdnn_transform_ztensor](#zdnn_transform_ztensor) and
    [zdnn_transform_ztensor_with_saturation](#zdnn_transform_ztensor_with_saturation)
    for details and restrictions on transforming an input tensor to the internal
    format.

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

```C
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
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
    - Note: concatenation dimensions have a smaller maximum size. See
      [LSTM](#lstm-hid_sz) or [GRU](#gru-hid_sz).
  - The total number of transformed_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_INVALID_STATE` - Tensor is already transformed.
- `ZDNN_CONVERT_FAILURE` - Values failed to transform.
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - unsupported transformation function.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_transform_ztensor_with_saturation

#### Description

Converts the input tensor to the supported transformed format for execution by
zdnn operations. If during transformation, an element results in a value that
exceeds the smallest or largest value that can be represented by DLFLOAT16, the
resulting element will contain the smallest or largest value and no
range-violation status will be triggered. If transformation is successful the
`is_transformed` field within `ztensor` will be set to `true` otherwise it is
set to `false`. Transformation will fail if `is_transformed` was already `true`.

_Note that the tensor layout in memory, once in transformed format, is dependent
on the content of the input tensor's descriptors (`zdnn_tensor_desc` fields).
Once converted, a `zdnn_ztensor` should only be manipulated by zDNN API
functions._

#### Format

```C
zdnn_status zdnn_transform_ztensor_with_saturation(zdnn_ztensor *ztensor, ...);
```

#### Parameters

- `zdnn_ztensor *tensor`

  - The input `zdnn_ztensor` struct. `pre_transformed_desc` and
    `transformed_desc` must be set, `is_transformed` must be `false`. A
    4k-aligned tensor storage must be pre-allocated by the caller (directly or
    by calling the zDNN allocation helper function) and field `buffer` must
    point to the storage.
  - Has the following additional restrictions:
    - Only non-quantized ztensors are supported. Use
      `zdnn_transform_quantized_ztensor` if required.

- `... (additional arguments)`

  - Variadic: list of pointers for input data to be transformed:
    - 1 data pointer supported at this time.

#### Returns zdnn_status indications

- `ZDNN_OK`
- `ZDNN_ELEMENT_RANGE_VIOLATION`
- `ZDNN_INVALID_FORMAT` - `zdnn_ztensor->transformed_desc->format` is not
  ZDNN_FORMAT_4DFEATURE.
- `ZDNN_INVALID_LAYOUT` - (if any of the following are true)
  - `zdnn_ztensor->pre_transformed_desc->layout` is not recognized or is not a
    valid pre_transformed_desc layout.
  - `zdnn_ztensor->transformed_desc->layout` is not recognized or is not a valid
    transformed_desc layout.
- `ZDNN_INVALID_TYPE` - (if any of the following are true)
  - `zdnn_ztensor->pre_transformed_desc->type` is not recognized or is not a
    valid pre_transformed_desc type.
  - `zdnn_ztensor->transformed_desc->type` is not recognized or is not a valid
    transformed_desc type.
- `ZDNN_INVALID_BUFFER` (if any of the following are true)
  - `buffer` is `NULL`.
  - `buffer` is not on a 4K boundary.
  - `buffer_size` is too small to hold transformed values.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `zdnn_ztensor->transformed_desc->dim*` dimensions is 0.
  - One of `zdnn_ztensor->transformed_desc->dim*` dimensions is greater than
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
  - The total number of transformed_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_INVALID_STATE` - Tensor is already transformed.
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - unsupported transformation function.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_transform_quantized_ztensor

#### Description

Converts the input tensor to the supported quantized transformed format for
execution by zdnn operations. If transformation is successful the
`is_transformed` field within `ztensor` will be set to `true` otherwise it is
set to `false`. Transformation will fail if `is_transformed` was already `true`.

_Note that the tensor layout in memory, once in transformed format, is dependent
on the content of the input tensor's descriptors (`zdnn_tensor_desc` fields).
Once converted, a `zdnn_ztensor` should only be manipulated by zDNN API
functions._

#### Format

```C
zdnn_status zdnn_transform_quantized_ztensor(zdnn_ztensor *ztensor,
                                             bool saturation_control,
                                             int8_t clip_min, int8_t clip_max,
                                             const void *data);
```

#### Parameters

- `zdnn_ztensor *tensor`

  - The input `zdnn_ztensor` struct. `pre_transformed_desc` and
    `transformed_desc` must be set, `is_transformed` must be `false`. A
    4k-aligned tensor storage must be pre-allocated by the caller (directly or
    by calling the zDNN allocation helper function) and field `buffer` must
    point to the storage.
  - Has the following additional restrictions:
    - Only the following pre-transformed layouts are supported.
      - ZDNN_1D
      - ZDNN_2D
      - ZDNN_2DS
      - ZDNN_3D
      - ZDNN_3DS
      - ZDNN_4D
      - ZDNN_NHWC
    - Only NHWC transformed layout is supported.
    - See [Quantized zTensor Requirements](#quan-zten-reqs) for supported
      transform types.

- `bool saturation_control`

  - When enabled and an element results in a value that exceeds the smallest or
    largest value that can be represented by DLFLOAT16, the resulting element
    will contain the smallest or largest value and no range-violation status
    will be triggered.
  - Only applicable when all the following are true:

    - `zdnn_ztensor *tensor` is of zdnn_quantized_transform_types
      QUANTIZED_DLFLOAT16.
    - The `pre_transformed_desc` `type` of the `zdnn_ztensor *tensor` is FP32.

- `int8_t clip_min`

  - Minimum clipping value
  - Only applicable when `zdnn_ztensor *tensor` is of
    zdnn_quantized_transform_types QUANTIZED_INT8.
  - Must be less than `clip_max`

- `int8_t clip_max`

  - Maximum clipping value
  - Only applicable when `zdnn_ztensor *tensor` is of
    zdnn_quantized_transform_types QUANTIZED_INT8.
  - Must be greater than `clip_min`

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
    transformed_desc type: [Quantized zTensor Requirements](#quan-zten-reqs)
  - `zdnn_ztensor->transformed_desc->type` is not recognized or is a
    pre_transformed_desc type: [Quantized zTensor Requirements](#quan-zten-reqs)
- `ZDNN_INVALID_BUFFER` (if any of the following are true)
  - `buffer` is `NULL`.
  - `buffer` is not on a 4K boundary.
  - `buffer_size` is too small to hold transformed values.
- `ZDNN_INVALID_SHAPE` - (if any of the following are true)
  - One of `zdnn_ztensor->transformed_desc->dim*` dimensions is 0.
  - One of `zdnn_ztensor->transformed_desc->dim*` dimensions is greater than
    [zdnn_get_max_for_dim](#zdnn_get_max_for_dim).
  - The total number of transformed_desc elements is larger than
    `zdnn_get_nnpa_max_tensor_size`.
- `ZDNN_INVALID_STATE` - Tensor is already transformed.
- `ZDNN_INVALID_CLIPPING_VALUE` - clip_min value is not less than clip_max
  value.
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Unsupported transformation function.
  - `ZDNN_FUNC_RC_F001` - Either scale or offset is non-numeric or scale value
    is zero.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

---

### zdnn_transform_origtensor

#### Description

Converts the input tensor from the zDNN transformed format back to a standard
non-transformed layout. The `is_transformed` field within `ztensor` must be
`true`.

All stick format tensors are supported, except:

- Kernel tensors
- Concatenated RNN input-gates tensors

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

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
- [Square Root](#zdnn_sqrt)
- [Inverse Square Root](#zdnn_invsqrt)

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Addition](https://www.tensorflow.org/api_docs/python/tf/math/add)

[ONNX Addition](https://onnx.ai/onnx/operators/onnx__Add.html#l-onnx-doc-add)

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

```C
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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Subtraction](https://www.tensorflow.org/api_docs/python/tf/math/subtract)

[ONNX Subtraction](https://onnx.ai/onnx/operators/onnx__Sub.html#l-onnx-doc-sub)

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Multiplication](https://www.tensorflow.org/api_docs/python/tf/math/multiply)

[ONNX Multiplication](https://onnx.ai/onnx/operators/onnx__Mul.html#l-onnx-doc-mul)

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Division](https://www.tensorflow.org/api_docs/python/tf/math/divide)

[ONNX Division](https://onnx.ai/onnx/operators/onnx__Div.html#l-onnx-doc-div)

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Minimum](https://www.tensorflow.org/api_docs/python/tf/math/minimum)

[ONNX Minimum](https://onnx.ai/onnx/operators/onnx__Min.html#l-onnx-doc-min)

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Maximum](ttps://www.tensorflow.org/api_docs/python/tf/math/maximum)

[ONNX Maximum](https://onnx.ai/onnx/operators/onnx__Max.html#l-onnx-doc-max)

---

### zdnn_log

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the natural logarithm
element-wise and stores the result into the provided output zDNN tensor.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Natural Logarithm](https://www.tensorflow.org/api_docs/python/tf/math/log)

[ONNX Natural Logarithm](https://onnx.ai/onnx/operators/onnx__Log.html#l-onnx-doc-log)

---

### zdnn_exp

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the exponential
element-wise and stores the result into the provided output zDNN tensor.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Exponential](https://www.tensorflow.org/api_docs/python/tf/math/exp)

[ONNX Exponential](https://onnx.ai/onnx/operators/onnx__Exp.html#l-onnx-doc-exp)

---

### zdnn_sqrt

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the square root
element-wise and stores the result into the provided output zDNN tensor.

#### Format

```C
zdnn_status zdnn_sqrt(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the calculated square root of each value from `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Square Root](https://www.tensorflow.org/api_docs/python/tf/math/sqrt)

[ONNX Square Root](https://onnx.ai/onnx/operators/onnx__Sqrt.html#l-onnx-doc-sqrt)

---

### zdnn_invsqrt

- [Back to Table of Contents](#TOC)
  - [Back to Element-wise Operations](#elwise-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the inverse square
root element-wise and stores the result into the provided output zDNN tensor.

#### Format

```C
zdnn_status zdnn_invsqrt(const zdnn_ztensor *input, float epsilon,
                         zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `float epsilon`

  - A float value added to input prior to computation.

- `zdnn_ztensor *output`
  - Tensor that holds the calculated inverse square root of each value from
    `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_EPSILON`
- [hardware statuses](#hw-statuses)

#### Programming Notes

- On some models, if either or both an element and epsilon are very large, the
  addition of the two may result in a nonnumeric value, the inverse square root
  of which will also be nonnumeric. This may occur even though the inverse
  square root of an unconstrained sum would easily fit in the data type of an
  output-tensor element.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Reciprical Square Root](https://www.tensorflow.org/api_docs/python/tf/math/rsqrt)

---

## Activation Operations <a id="act-ops"></a>

[Back to Table of Contents](#TOC)

- [Rectified Linear](#zdnn_relu)
- [Leaky Rectified Linear](#zdnn_leaky_relu)
- [Hyperbolic Tangent](#zdnn_tanh)
- [Sigmoid](#zdnn_sigmoid)
- [Softmax](#zdnn_softmax)
- [Softmax with Mask](#zdnn_softmax_mask)
- [Gaussian Error Linear Unit](#zdnn_gelu)

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Rectified Linear](https://www.tensorflow.org/api_docs/python/tf/nn/relu)

[ONNX Rectified Linear](https://onnx.ai/onnx/operators/onnx__Relu.html#l-onnx-doc-relu)

---

### zdnn_leaky_relu

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format produce an output tensor where
the leaky rectified linear function is applied to the input element-wise. The
calculation used depends on the input element. When negative, y = a \* x, where
a is the adjustment factor. When 0 or positive, y = x. If an optional
clipping_value is provided, clipping is performed against the intermediate
output where z = min(y, clipping_value).

#### Format

```C
zdnn_status zdnn_leaky_relu(const zdnn_ztensor *input,
                            const void *clipping_value,
                            float adjustment_factor, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `void *clipping_value`

  - A pointer to an FP32 value, used to clip input tensor's elements.
  - If set to NULL or 0, no clipping will occur.
  - Must not be a negative value.

- `float adjustment_factor`

  - A float value multiplied with negative elements from input.
  - Must not be a negative value.
  - Must not be greater than 1.

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
- `ZDNN_INVALID_ADJUSTMENT_FACTOR`
- [hardware statuses](#hw-statuses)

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Leaky Rectified Linear](https://www.tensorflow.org/api_docs/python/tf/nn/leaky_relu)

[ONNX Leaky Rectified Linear](https://onnx.ai/onnx/operators/onnx__LeakyRelu.html#l-onnx-doc-leakyrelu)

---

### zdnn_tanh

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, produces an output tensor
where the hyperbolic tangent is applied to the input element-wise.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Hyperbolic Tangent](https://www.tensorflow.org/api_docs/python/tf/math/tanh)

[ONNX Hyperbolic Tangent](https://onnx.ai/onnx/operators/onnx__Tanh.html#l-onnx-doc-tanh)

---

### zdnn_sigmoid

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, produces an output tensor
where the sigmoid function is applied to the input element-wise.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Sigmoid](https://www.tensorflow.org/api_docs/python/tf/math/sigmoid)

[ONNX Sigmoid](https://onnx.ai/onnx/operators/onnx__Sigmoid.html#l-onnx-doc-sigmoid)

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

```C
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
      `pre_transformed_desc` using dimensions 8x1x2 and use the same original
      data array prior to `zdnn_transform_ztensor`. After transformation, such a
      tensor would be valid for `zdnn_softmax`.
    - In another example, the 4D 2x2x2x2 tensor could be processed as 8 batches
      of 2 vectors using a `ZDNN_3DS` layout `pre_transformed_desc` with
      dimensions 1x8x2.
    - The inner-most dimension must remain the same during this coercion.

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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)

[ONNX Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html#l-onnx-doc-softmax)

---

### zdnn_softmax_mask

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, computes the softmax
(normalized exponential) for each vector formed in dimension-1 (from element
zero to mask - 1), then if `act_func` is not `SOFTMAX_ACT_NONE`, the activation
function is applied to the results. Finally stores the results into the provided
output zDNN tensor.

_Note: Other parameters, such as axis, are not supported._

#### Format

```C
zdnn_status zdnn_softmax_mask(const zdnn_ztensor *input, void *save_area,
                              zdnn_softmax_act act_func, uint32_t softmax_mask,
                              zdnn_ztensor *output);
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

- `uint32_t softmax_mask`

  - 32-bit unsigned binary integer that specifies a count of dimensions 1
    elements to be processed.
  - If 0, behavior matches `zdnn_softmax`
  - Must not exceed dimension 1 of input tensor.

- `zdnn_ztensor *output`
  - [ZDNN_3DS](#common-layouts) tensor with the same shape as `input_a` that
    holds the softmax result of each value from `input_a`.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Programming Notes

- If all elements of a dimension 1 vector are the largest magnitude negative
  number possible for the transformed data type, accuracy may be reduced.

- A `ZDNN_3DS` tensor is expected, where the `transformed_desc` dim1 describes
  the vector, and dim2 and dim4 are used to batch multiple vector requests
  together. Dim3 must always be 1. The `zdnn_softmax_mask` operation is
  performed against the vector in dim1 repeating for each dim1 vector in the
  dim4 and dim2 dimensions.

- Tensors that cannot be processed as vectors in dim1 or as batches of dim1
  vectors must be coerced or reshaped by the caller.
  - When the entire tensor is to be processed by softmax, it can be coerced by
    simply creating an alternate descriptor prior to zDNN transformation. For
    example:
    - A 4D tensor with `pre_transformed_desc` dimensions 2x2x2x2 and a data
      array of 16 FP32 entries could have an alternate `ZDNN_3DS` layout
      `pre_transformed_desc` using dimensions 8x1x2 and use the same original
      data array prior to `zdnn_transform_ztensor`. After transformation, such a
      tensor would be valid for `zdnn_softmax_mask`.
    - In another example, the 4D 2x2x2x2 tensor could be processed as 8 batches
      of 2 vectors using a `ZDNN_3DS` layout `pre_transformed_desc` with
      dimensions 1x8x2.

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
  - `ZDNN_FUNC_RC_F002` - `softmax_mask` exceeds dimension 1 of input tensor.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Softmax](https://www.tensorflow.org/api_docs/python/tf/nn/softmax)

[ONNX Softmax](https://onnx.ai/onnx/operators/onnx__Softmax.html#l-onnx-doc-softmax)

---

### zdnn_gelu

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format produce an output tensor where
the Gaussian Error Linear Unit activation function, y = 0.5 \* x \* (1 +
tanh(x \* 0.7978845608 \* (1 + 0.044715 \* x \* x))), is applied to the input
element-wise.

#### Format

```C
zdnn_status zdnn_gelu(const zdnn_ztensor *input, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor that holds the Gaussian Error Linear Unit results of each value from
    `input`
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Programming Notes

- The range of certain input-element values may result in an error of greater
  than 1% in the output element, however the accuracy of properly conditioned
  models is not significantly degraded.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Gaussian Error Linear Unit](https://www.tensorflow.org/api_docs/python/tf/nn/gelu)

[ONNX Gaussian Error Linear Unit](https://onnx.ai/onnx/operators/onnx__Gelu.html#l-onnx-doc-gelu)

---

## Normalization Operations <a id="norm-ops"></a>

[Back to Table of Contents](#TOC)

- [Mean Reduce](#zdnn_meanreduce2d)
- [Batch Norm](#zdnn_batchnorm)
- [Normalization](#zdnn_norm)
- [Moments](#zdnn_moments)
- [Layer Normalization](#zdnn_layernorm)
- [Reduce](#zdnn_reduce)

---

### zdnn_meanreduce2d

- [Back to Table of Contents](#TOC)
  - [Back to Normalization Operations](#norm-ops)

#### Description

Given an input tensor in zDNN transformed format, produces a downsampled tensor
reducing the middle dimensions to a size of 1 based on the mean of the original
values and stores the result to the provided output zDNN tensor.

#### Format

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Reduce Mean] with `axis` set for the Height and Width axes and
`keepdims` set to True.

[tensorflow reduce mean]:
  https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean

[ONNX Reduce Mean]

[onnx reduce mean]:
  https://onnx.ai/onnx/operators/onnx__ReduceMean.html#l-onnx-doc-reducemean

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

```C
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

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Batchnorm]

[tensorflow batchnorm]:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization

[ONNX Batchnorm]

[onnx batchnorm]:
  https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#l-onnx-doc-batchnormalization

---

### zdnn_norm

- [Back to Table of Contents](#TOC)
  - [Back to Normalization Operations](#norm-ops)

#### Description

Given input_a and input_b tensors in zDNN transformed format, produces the norm
of the difference of vectors. Calculation is performed as follows:

1. Each element in dimension 1 of input_b is subtracted by the corresponding
   element of input_a.
2. The difference is squared.
3. The sum of the squared differences for dimension 1 is computed.
4. The square root of the sum is placed in the first element of dimension 1 of
   output tensor.

#### Format

```C
zdnn_status zdnn_norm(const zdnn_ztensor *input_a, zdnn_ztensor *input_b, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output`
  - Tensor with the result of the normalization operation.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- [hardware statuses](#hw-statuses)

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Normalization]

[tensorflow normalization]:
  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization

[ONNX Normalization]

N / A

---

### zdnn_moments

- [Back to Table of Contents](#TOC)
  - [Back to Normalization Operations](#norm-ops)

#### Description

Given an input tensor in zDNN transformed format and a bessel correction type,
this produces the mean and variance for respective input tensor.

#### Format

```C
zdnn_status zdnn_moments(const zdnn_ztensor *input,
                         zdnn_moments_bessel bessel_correction_type,
                         zdnn_ztensor *output_a, zdnn_ztensor *output_b);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Must be a 4D [ZDNN_NHWC](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_moments_bessel bessel_correction_type`

  - Bessel correction type to perform moments.
    - `MOMENTS_BESSEL_POPULATION`
    - `MOMENTS_BESSEL_SAMPLE`

- `zdnn_ztensor *output_a`

  - The output tensor that will hold the mean.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *output_b`

  - The output tensor that will hold the variance.
  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_BESSEL_CORRECTION`
- [hardware statuses](#hw-statuses)

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Programming Notes

- The `zdnn_moments` operation may be used in combination of the
  `zdnn_layernorm` operation. Please see [zdnn_layernorm](#zdnn_layernorm) for
  more guidance.

- When `MOMENTS_BESSEL_SAMPLE` is provided for the bessel correction type, all
  provided input dimensions of the input tensor must not be equal to 1.

#### Framework Examples

[TensorFlow Moments]

[tensorflow moments]: https://www.tensorflow.org/api_docs/python/tf/nn/moments

[ONNX Moments]

N/A

---

### zdnn_layernorm

- [Back to Table of Contents](#TOC)
  - [Back to Normalization Operations](#norm-ops)

#### Description

Given input_a, input_b, and input_c tensors in zDNN transformed format, produces
the layernorm of the given tensors. Calculation is performed as follows:

1. Each element in dimension 1 of input_b is subtracted by the corresponding
   element of input_a.
2. A corresponding element of input_c is added to epsilon.
3. The square root of the sume from step 2 is computed.
4. The difference from step 1 is divided by the result of step 3.
5. The quotient from step 4 is multiplied by gamma.
6. The product from step 5 is added to beta.
7. Result is stored in the corresponding element of output.

The above calculation could be depicted as follows:

<img src="https://latex.codecogs.com/svg.image?layernorm(a)=(a-b)/sqrt
(c&plus;\epsilon)*\gamma&plus;\beta&space;"
title="layernorm(a)=(a-b)/sqrt(c+\epsilon)*\gamma+\beta "
alt="layernorm(a)=(a-b)/sqrt(c+\epsilon)*\gamma+\beta " />

#### Format

```C
zdnn_status zdnn_layernorm(const
                           zdnn_ztensor *input_a,
                           const zdnn_ztensor *input_b,
                           const zdnn_ztensor *input_c,
                           float beta, float gamma, float epsilon,
                           zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input_a`

  - Must be a 4D [ZDNN_NHWC](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *input_b`

  - Must be a 4D [ZDNN_NHWC](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Contains arithmetic means ([Moments](#zdnn_moments) output_a)

- `zdnn_ztensor *input_c`

  - Must be a 4D [ZDNN_NHWC](#common-layouts) tensor
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Contains arithmetic variances ([Moments](#zdnn_moments) output_b)

- `float beta`

  - Final result adjustment addend.

- `float gamma`

  - Final result adjustment multiplier.

- `float epsilon`

  - Intermediate variance adjustment.

- `zdnn_ztensor *output`

  - Must follow [general tensor requirements](#gen-zten-reqs)

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_BETA`
- `ZDNN_INVALID_GAMMA`
- `ZDNN_INVALID_EPSILON`
- [hardware statuses](#hw-statuses)

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Programming Notes

- `zdnn_layernorm` is intended to be used in combination with the `zdnn_moments`
  normalization operation. The `zdnn_moments` operation produces two output
  tensors containing the means and variances, respectively, of the dimension-
  4-index elements of the input tensor. The original input tensor to
  `zdnn_moments` is intended to be used as the input-tensor 1 to
  `zdnn_layernorm`. The output-tensors 1 and 2 of `zdnn_moments`are intended to
  be used as input as input-tensor 2 and input-tensor 3 of the `zdnn_layernorm`
  operation.

- The beta and gamma values in the 4th and 5th parameters of `zdnn_layernorm`,
  (also reffered to as bias and gain), provide a learned scale and offset. The
  epsilon value in parameter 6 of `zdnn_layernorm` is intended to be a small
  value (for example, 0.001) to provide numerical stability.

#### Framework Examples

[TensorFlow Layernorm](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization)

[ONNX Layernorm](https://onnx.ai/onnx/operators/onnx__LayerNormalization.html#l-onnx-doc-layernormalization)

---

### zdnn_reduce

- [Back to Table of Contents](#TOC)
  - [Back to Activation Operations](#act-ops)

#### Description

Given an input tensor in zDNN transformed format, produces an output tensor
where the given reduction operation is performed.

#### Format

```C
zdnn_status zdnn_reduce(const zdnn_ztensor *input, void *save_area,
                        zdnn_reduce_ops op_type, zdnn_ztensor *output);
```

#### Parameters

- `zdnn_ztensor *input`

  - Tensor with values to evaluate.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `void *save_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - The preallocate memory must be at least 8K bytes in size, aligned on a 4k
    boundary.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.

- `zdnn_reduction_ops op_type`

  - Reduction Operation to perform on input tensor.
    - `REDUCE_OP_MINIMUM`
    - `REDUCE_OP_MINIMUM_IDX`
    - `REDUCE_OP_MAXIMUM`
    - `REDUCE_OP_MINIMUM_IDX`

- `zdnn_ztensor *output`
  - Tensor that holds the reduction operation result of each value from `input`
    - Output dimension 1 must 1
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Data Type must be as follows:
    - (FP32, FP16, BFLOAT) when `op_type` is `REDUCE_OP_MINIMUM` or
      `REDUCE_OP_MAXIMUM`.
    - INT32 when `op_type` is `REDUCE_OP_MINIMUM_IDX` or `REDUCE_OP_MAXIMUM_IDX`

The output when op_type is `REDUCE_OP_MINIMUM` or `REDUCE_OP_MAXIMUM` can be
initialized using:

```C
zdnn_data_layouts input_layout = ZDNN_3DS;
zdnn_data_types input_type = FP32;

uint32_t dim4 = 4;
uint32_t dim2 = 5;
uint32_t dim1 = 6;

zdnn_tensor_desc input_pre_transformed_desc;

zdnn_init_pre_transformed_desc(input_layout, input_type,
                               &input_pre_transformed_desc, dim4, dim2, dim1);

zdnn_tensor_desc output_pre_transformed_desc;

zdnn_init_pre_transformed_desc(input_layout, input_type,
                               &output_pre_transformed_desc, dim4, dim2, 1);
```

The output when op_type is `REDUCE_OP_MINIMUM_IDX` or `REDUCE_OP_MAXIMUM_IDX`
can be initialized using:

```C
zdnn_data_layouts input_layout = ZDNN_3DS;
zdnn_data_types input_type = FP32;

uint32_t dim4 = 4;
uint32_t dim2 = 5;
uint32_t dim1 = 6;

zdnn_tensor_desc input_pre_transformed_desc;

zdnn_init_pre_transformed_desc(input_layout, input_type,
                               &input_pre_transformed_desc, dim4, dim2, dim1);

zdnn_data_types output_type = INT32;

zdnn_tensor_desc output_pre_transformed_desc;

zdnn_init_pre_transformed_desc(input_layout, output_type,
                               &output_pre_transformed_desc, dim4, dim2, 1);
```

#### Programming Notes

- If a nonnumeric element is encountered in a dimension-1 vecotr of input-tenzor
  1, then (a) the resulting element in dimension 1 of output-tensor 1 is
  unpredictable, and the range-violation status will be returned.
- When the reduction operation is `REDUCE_OP_MINIMUM_IDX` the index of the first
  min value, from left-to-right, is returned when there are mulitple elements
  with the same min value.
- When the reduction operation is `REDUCE_OP_MAXIMUM_IDX` the index of the first
  max value, from left-to-right, is returned when there are mulitple elements
  with the same max value.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- [warning statuses](#warning-statuses)
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_UNAVAILABLE_FUNCTION`
- `ZDNN_ALLOCATION_FAILURE` - A preallocated `save_area` was not specified and
  internal allocation for the required memory failed.
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Invalid `op_type`.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Reduce Min and Max]

[tensorflow reduce minimum](https://www.tensorflow.org/api_docs/python/tf/math/reduce_min)
[tensorflow reduce maximum](https://www.tensorflow.org/api_docs/python/tf/math/reduce_max)

[ONNX Reduce Min and Max]

[onnx reduce minimum](https://onnx.ai/onnx/operators/onnx__ReduceMin.html#l-onnx-doc-reducemin)
[onnx reduce maximum](https://onnx.ai/onnx/operators/onnx__ReduceMax.html#l-onnx-doc-reducemax)

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

```C
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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow MatMul](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mat-mul)

[ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul)

---

### zdnn_matmul_bcast_op

[Back to Table of Contents](#TOC)

#### Description

Given three input zDNN tensors `input_a`, `input_b`, and `input_c`, determine
the matrix multiplication of `input_a` \* `input_b`, then perform one of the
following operations, using `input_c` against the dot product, storing the
result into the specified `output` zDNN tensor:

- Addition
- Compare - If dot product is greater than element.
- Compare - If dot product is greater or equal to element.
- Compare - If dot product is equal to element.
- Compare - If dot product is not equal to element.
- Compare - If dot product is less than or equal to element.
- Compare - If dot product is less than element.

When an input is `ZDNN_3DS`, the outermost dimension for that input can
optionally indicate that the input is a stack of matrices. Likewise, when an
input is `ZDNN_2DS`, the outermost dimension for that input can optionally
indicate that the input is a stack of vectors

For exmaple, if `input_a` were `ZDNN_3DS`, each stack of `input_a` is multiplied
by the same `input_b` matrix and `input_c` vector which are broadcast over each
stack of `input_a`. Results for each stack are returned in the corresponding
stack index of `output`.

Likewise, if `input_b` were `ZDNN_3DS` and `input_c` were `ZDNN_2DS`, each stack
of `input_b` is multiplied by the same `input_a` matrix which is broadcast over
each stack of `input_b` and `input_c`. Results for each stack are returned in
the corresponding stack index of `output`.

#### Format

```C
zdnn_status zdnn_matmul_bcast_op(const zdnn_ztensor *input_a,
                                 const zdnn_ztensor *input_b,
                                 const zdnn_ztensor *input_c,
                                 zdnn_matmul_bcast_ops op_type,
                                 zdnn_ztensor *output);
```

#### Input / Output matmul broadcast tensor requirements <a id="matmul-bcast-io-table"></a>

- See table in this section for `pre_transformed_desc` and shape requirements
  for each tensor.
- Must follow [general tensor requirements](#gen-zten-reqs)

| type      | input_a              | input_b              | input_c           | result               |
| --------- | -------------------- | -------------------- | ----------------- | -------------------- |
| unstacked | `ZDNN_2D` (m, n)     | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_2D` (m, p)     |
| stacked   | `ZDNN_3DS` (s, m, n) | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |
| bcast1    | `ZDNN_2D` (m, n)     | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |
| bcast23   | `ZDNN_3DS` (s, m, n) | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_3DS` (s, m, p) |

#### Parameters

- `zdnn_ztensor *input_a`

  - Input tensor with the first matrix for multiplication.
  - pre_transformed shape and layout must match
    [matmul broadcast tensor requirements](#matmul-bcast-io-table)

- `zdnn_ztensor *input_b`

  - Input tensor with the second matrix for multiplication.
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
    - `MATMUL_BCAST_OP_GREATER`
    - `MATMUL_BCAST_OP_GREATER_EQUAL`
    - `MATMUL_BCAST_OP_EQUAL`
    - `MATMUL_BCAST_OP_NOT_EQUAL`
    - `MATMUL_BCAST_OP_LESSER_EQUAL`
    - `MATMUL_BCAST_OP_LESSER`

- `zdnn_ztensor *output`
  - The output tensor which will hold the result of the operation in its buffer.
  - pre_transformed shape and layout must match
    [matmul broadcast tensor requirements](#matmul-bcast-io-table)

#### Programming Notes

- `zdnn_matmul_bcast_ops` only supports `MATMUL_BCAST_OP_ADDITION` op_type when
  `NNPA_PARMBLKFORMAT_1` is not installed. If any other op_types is provided,
  `ZDNN_UNAVAILABLE_FUNCTION` is returned.
- `BCAST1` is not supported when `NNPA_PARMBLKFORMAT_1` is not installed and
  will return `ZDNN_UNAVAILABLE_FUNCTION`.
- Care must be exercised when comparing values for equality or inequality since
  the order of operations and rounding may produce, what appear to be, slightly
  different values when they are essentially the same value.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_UNAVAILABLE_FUNCTION`
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Invalid `op_type`.
  - `ZDNN_FUNC_RC_F001` - Invalid input/output type or format combination.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow MatMul](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mat-mul)

[ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul)

---

### zdnn_matmul_transpose_op

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

```C
zdnn_status zdnn_matmul_transpose_op(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     bool transpose_a, bool transpose_b,
                                     zdnn_matmul_ops op_type,
                                     zdnn_ztensor *output);
```

#### Input / Output matmul transpose tensor requirements <a id="matmul-transpose-io-table"></a>

- See table in this section for `pre_transformed_desc` and shape requirements
  for each tensor.
- All tensors must either be stacked or unstacked.
- Must follow [general tensor requirements](#gen-zten-reqs)

| type      | input_a              | input_b              | input_c           | result               |
| --------- | -------------------- | -------------------- | ----------------- | -------------------- |
| unstacked | `ZDNN_2D` (m, n)     | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_2D` (m, p)     |
| stacked   | `ZDNN_3DS` (s, m, n) | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |
| bcast1    | `ZDNN_2D` (m, n)     | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |
| bcast23   | `ZDNN_3DS` (s, m, n) | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_3DS` (s, m, p) |

#### Parameters

- `zdnn_ztensor *input_a`

  - Input tensor with the first matrix for multiplication
  - pre_transformed shape and layout must match
    [matmul transpose tensor requirements](#matmul-transpose-io-table)

- `zdnn_ztensor *input_b`

  - Input tensor with the second matrix for multiplication
  - pre_transformed shape and layout must match
    [matmul transpose tensor requirements](#matmul-transpose-io-table)

- `zdnn_ztensor *input_c`

  - Input tensor that will have the requested operation performed against the
    intermediate dot product of `input_a` and `input_b`.
  - pre_transformed shape and layout must match
    [matmul transpose tensor requirements](#matmul-transpose-io-table)

- `bool transpose_a`

  - Whether to transpose `input_a` prior to dot product.
  - If `true`, `input_a` should have the unstacked dimensions (n, m) or stacked
    dimensions (s, n, m)

- `bool transpose_b`

  - Whether to transpose `input_b` prior to dot product.
  - If `true`, `input_b` should have the unstacked dimensions (p, n) or stacked
    dimensions (s, p, n)

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
    [matmul transpose tensor requirements](#matmul-transpose-io-table)

#### Programming Notes

- `zdnn_matmul_transpose_op` is not supported when `NNPA_PARMBLKFORMAT_1` is not
  installed and will return `ZDNN_UNAVAILABLE_FUNCTION`.
- Care must be exercised when comparing values for equality or inequality since
  the order of operations and rounding may produce, what appear to be, slightly
  different values when they are essentially the same value.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`
- `ZDNN_INVALID_FORMAT`
- `ZDNN_UNAVAILABLE_FUNCTION`
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Invalid `op_type`.
  - `ZDNN_FUNC_RC_F001` - Invalid input/output type or format combination.

#### Since

1.0.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.0.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow MatMul](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/mat-mul)

[ONNX MatMul](https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul)

---

### zdnn_quantized_matmul_op

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

When dequantize is `true` the output will be dequantized after computation.

When `pre_computed` is `true`. The pre-computed value of `input_c` for Addition
can be achieved using:

```C
Za = input_a->offset;
Sa = 1 / input_a->rec_scale;

Zb = input_b->offset;
Sb = 1 / input_b->rec_scale;

Zc = input_c->offset;
Sc = 1 / input_c->rec_scale;

Zy = output->offset;
Sy = 1 / output->rec_scale;

N = input_b->pre_transformed_desc->dim2;

pre_computed = Zy - (Sc/Sy) * Zc - (Sc/Sy) * input_c + ((Sa * Sb) / Sy) * NZaZb;
```

The pre-computed value of `input_c` for Compare can be achieved using:

```C
Za = input_a->offset;
Sa = 1 / input_a->rec_scale;

Zb = input_b->offset;
Sb = 1 / input_b->rec_scale;

Zc = input_c->offset;
Sc = 1 / input_c->rec_scale;

pre_computed = Sc / (Sa * Sb) * (input_c - Zc) + Za * sum(input_b, axis=-2)
```

#### Format

```C
zdnn_status zdnn_quantized_matmul_op(const zdnn_ztensor *input_a,
                                     const zdnn_ztensor *input_b,
                                     const zdnn_ztensor *input_c,
                                     zdnn_matmul_ops op_type,
                                     const int8_t clip_min,
                                     const int8_t clip_max,
                                     const bool disable_clipping,
                                     const bool dequantize,
                                     const bool pre_computed,
                                     void *work_area,
                                     zdnn_ztensor *output);
```

#### Input / Output quantized matmul tensor requirements <a id="quantized-matmul-io-table"></a>

- See table in this section for `pre_transformed_desc` and shape requirements
  for each tensor.
- All tensors must either be stacked or unstacked.
- Must follow [general tensor requirements](#gen-zten-reqs)
- All tensors should use `zdnn_generate_quantized_transformed_desc` when
  generating transformed descriptors, passing the appropriate
  `zdnn_quantized_transform_types`.
- All quantized tensors should use `zdnn_init_quantized_ztensor` or
  `zdnn_init_quantized_ztensor_with_malloc` when initializing, passing the
  `scale` and `offset` quantization parameters.
  - `scale` must be in range ([-DLFLT_MAX](#zdnn_get_max_limit) <= scale <=
    [DLFLT_MAX](#zdnn_get_max_limit)) and scale != 0.
  - `offset` must be in range ([-DLFLT_MAX](#zdnn_get_max_limit) <= offset <=
    [DLFLT_MAX](#zdnn_get_max_limit)).
- All quantized input tensors should use `zdnn_transform_quantized_ztensor` when
  transforming, passing the `clip_min` and `clip_max` quantization parameters.

##### zdnn_data_layouts

| type      | input_a              | input_b              | input_c           | result               |
| --------- | -------------------- | -------------------- | ----------------- | -------------------- |
| unstacked | `ZDNN_2D` (m, n)     | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_2D` (m, p)     |
| stacked   | `ZDNN_3DS` (s, m, n) | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |
| bcast1    | `ZDNN_2D` (m, n)     | `ZDNN_3DS` (s, n, p) | `ZDNN_2DS` (s, p) | `ZDNN_3DS` (s, m, p) |
| bcast23   | `ZDNN_3DS` (s, m, n) | `ZDNN_2D` (n, p)     | `ZDNN_1D` (p)     | `ZDNN_3DS` (s, m, p) |

##### zdnn_quantized_transform_types

| type       | input_a             | input_b                | input_c        | result              |
| ---------- | ------------------- | ---------------------- | -------------- | ------------------- |
| normal     | QUANTIZED_INT8      | QUANTIZED_WEIGHTS_INT8 | QUANTIZED_INT8 | QUANTIZED_DLFLOAT16 |
| on-the-fly | QUANTIZED_DLFLOAT16 | QUANTIZED_WEIGHTS_INT8 | QUANTIZED_INT8 | QUANTIZED_DLFLOAT16 |

#### Parameters

- `zdnn_ztensor *input_a`

  - Input tensor with the first matrix for multiplication
  - pre_transformed shape and layout must match
    [quantized matmul tensor requirements](#quantized-matmul-io-table)

- `zdnn_ztensor *input_b`

  - Input tensor with the second matrix for multiplication
  - pre_transformed shape and layout must match
    [quantized matmul tensor requirements](#quantized-matmul-io-table)

- `zdnn_ztensor *input_c`

  - Input tensor that will have the requested operation performed against the
    intermediate dot product of `input_a` and `input_b`.
  - pre_transformed shape and layout must match
    [quantized matmul tensor requirements](#quantized-matmul-io-table)

- `int8_t clip_min`

  - Minimum quantized value for `input_a` prior to dot product.
  - Only applicable when performing `on-the-fly` quantization.
  - Must be less than `clip_max`.

- `int8_t clip_max`

  - Maximum quantized value for `input_a` prior to dot product.
  - Only applicable when performing `on-the-fly` quantization.
  - Must be greater than `clip_min`.

- `bool disable_clipping`

  - When `true` disables clipping and rounding.

- `bool dequantize`

  - Whether to dequantize returned ztensor.

- `bool pre_computed`

  - Whether bias is already pre-computed.

- `void *work_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.
  - Amount of required storage is the same as `input_c->buffer_size`.
  - The start of the buffer must be 4k aligned.

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
    [quantized matmul tensor requirements](#quantized-matmul-io-table)

#### Programming Notes

- `zdnn_quantized_matmul_op` is not supported when `NNPA_PARMBLKFORMAT_1` is not
  installed and will return `ZDNN_UNAVAILABLE_FUNCTION`.
- Care must be exercised when comparing values for equality or inequality since
  the order of operations and rounding may produce, what appear to be, slightly
  different values when they are essentially the same value.

#### Returns (see [zDNN Statuses](#common-statuses) for descriptions)

- `ZDNN_OK`
- `ZDNN_INVALID_SHAPE`
- `ZDNN_INVALID_TYPE`: [Quantized zTensor Requirements](#quan-zten-reqs)
- `ZDNN_INVALID_FORMAT`
- `ZDNN_INVALID_SCALE`
- `ZDNN_INVALID_OFFSET`
- `ZDNN_INVALID_CLIPPING_VALUE`
- `ZDNN_UNAVAILABLE_FUNCTION`
- [hardware statuses](#hw-statuses)
  - `ZDNN_FUNC_RC_F000` - Invalid `op_type`.
  - `ZDNN_FUNC_RC_F001` - Invalid input/output type or format combination.
  - `ZDNN_FUNC_RC_F002` - Invalid input/output scale.

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Quantized MatMul](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/quantized-mat-mul)

[ONNX Quantize Linear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html#l-onnx-doc-quantizelinear)

---

### zdnn_lstm

[Back to Table of Contents](#TOC)

#### Description

Implements Long-Short Term Memory layer (LSTM - Hochreiter 1997).

The following formula is computed for the input tensor input(t) for all time
steps:

(Default: f=Sigmoid, g=Tanh, h=Tanh):

```C
- it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)

- ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf)

- ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

- Ct = ft (.) Ct-1 + it (.) ct

- ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo)

- Ht = ot (.) h(Ct)
```

#### Format

```C
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

- `num_hidden` dimensions: <a id="lstm-hid_sz"></a>
  - Any num_hidden dimension must be less than or equal to
    `zdnn_get_max_for_dim(2) / 4` elements.

#### Parameters

- `zdnn_ztensor *input`

  - Input must be a tensor with the shape (num_timesteps, num_batches,
    num_features) prior to transformation with the `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *h0`

  - Tensor containing the initial hidden state with shape (num_dirs,
    num_batches, num_hidden) prior to transformation with the
    `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [num_hidden requirements](#lstm-hid_sz)

- `zdnn_ztensor *c0`

  - Tensor containing the initial cell state with shape (num_dirs, num_batches,
    num_hidden) prior to transformation with the `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [num_hidden requirements](#lstm-hid_sz)

- `zdnn_ztensor *weights`

  - Tensor containing the concatenated input connection weights in Forget,
    Input, Cell, Output (FICO) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (num_dirs, num_features, num_hidden) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_LSTM`
    - `USAGE_WEIGHTS`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#lstm-hid_sz)

- `zdnn_ztensor *biases`

  - Tensor containing the concatenated input connection bias in Forget, Input,
    Cell, Output (FICO) order.
  - Prior to transformation, expects each gate needs to be shape (num_dirs,
    num_hidden).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_LSTM`
    - `USAGE_BIASES`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#lstm-hid_sz)

- `zdnn_ztensor *hidden_weights`

  - Tensor containing the concatenated hidden connection weights in Forget,
    Input, Cell, Output (FICO) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (num_dirs, num_hidden, num_hidden) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_LSTM`
    - `USAGE_HIDDEN_WEIGHTS`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#lstm-hid_sz)

- `zdnn_ztensor *hidden_biases`

  - Tensor containing the concatenated hidden connection bias in Forget, Input,
    Cell, Output (FICO) order.
  - Prior to transformation, expects each gate needs to be shape (num_dirs,
    num_hidden).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_LSTM`
    - `USAGE_HIDDEN_BIASES`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#lstm-hid_sz)

- `lstm_gru_direction direction`

  - Direction indicator of `lstm_gru_direction direction` type. Valid values:
    - `FWD` (forward)
    - `BWD` (backward)
    - `BIDIR` (bi-directional).
  - For input and output shapes, the num_dirs dimension should be:
    - `1` for unidirectional calls such as FWD or BWD
    - `2` for bidirectional calls such that:
      - dimension 0 contains FWD values.
      - dimension 1 contains BWD values.

- `void *work_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.
  - Amount of required storage can be determined given the LSTM timestep, batch,
    and num_hidden values.

    - The sample code below creates a ztensor descriptor that is an equivalent
      size of the required `work_area`. To use this sample code yourself,
      replace the `num_timesteps`, `num_batches`, and `num_hidden` variables
      with your own values.

      ```C
        zdnn_tensor_desc desc;
        desc.dim4 = (4 * num_timesteps) + 6;
        desc.dim3 = 1;
        desc.dim2 = num_batches;
        desc.dim1 = num_hidden;
        uint64_t work_area_size = zdnn_getsize_ztensor(&desc);
      ```

  - For bidirectional, twice the amount of contiguous storage is required.
  - The start of the buffer must be 4k aligned.

- `zdnn_ztensor *hn_output`

  - Output results of the hidden states

  - Expects pre_transformed_desc->layout to be `ZDNN_4DS`.

  - Must follow [general tensor requirements](#gen-zten-reqs)

  - Must follow [num_hidden requirements](#lstm-hid_sz)

  - Output pre-transformed shapes:

    - all timesteps: (num_timesteps, num_dirs, num_batches, num_hidden)
    - final timestep only: (1, num_dirs, num_batches, num_hidden)

  - For bidirectional (`BIDIR`) output:

    - Forward and backward results are concatenated on the innermost dimension.
    - Can be used directly as input for subsequent RNN layers without needing
      untransformation.
      - Can not be used directly as input for other non-RNN zDNN ops.
    - Untransformation is supported.

  - Note that for `BWD` and the backward component of `BIDIR` directions, the
    output order matches the order of the input, not the processing order. For
    example, the first input timestep is the last to be processed and its result
    is the first timestep of the output.

- `zdnn_ztensor *cf_output`

  - Output results of the cell state for the last processed timestep

  - Expects pre_transformed_desc->layout to be `ZDNN_4DS`.

  - Must follow [general tensor requirements](#gen-zten-reqs)

  - Must follow [num_hidden requirements](#lstm-hid_sz)

  - Output pre-transformed shapes:

    - (1, num_dirs, num_batches, num_hidden)

  - For bidirectional (`BIDIR`):
    - Forward and backward results are concatenated on the innermost dimension.
    - Can not be used directly as input for other non-RNN zDNN ops.
    - Untransformation is supported.

#### Summary

|                | pre-transformed layout | pre-transformed shape                                                                               |
| -------------- | ---------------------- | --------------------------------------------------------------------------------------------------- |
| input          | `ZDNN_3DS`             | (num_timesteps, num_batches, num_features)                                                          |
| h0             | `ZDNN_3DS`             | (num_dirs, num_batches, num_hidden)                                                                 |
| c0             | `ZDNN_3DS`             | (num_dirs, num_batches, num_hidden)                                                                 |
| weights        | `ZDNN_3DS`             | (num_dirs, num_features, num_hidden)                                                                |
| bias           | `ZDNN_2DS`             | (num_dirs, num_hidden)                                                                              |
| hidden_weights | `ZDNN_3DS`             | (num_dirs, num_hidden, num_hidden)                                                                  |
| hidden_biases  | `ZDNN_2DS`             | (num_dirs, num_hidden)                                                                              |
| hn_output      | `ZDNN_4DS`             | (num_timesteps, num_dirs, num_batches, num_hidden)<br>(last timestep only when `num_timesteps` = 1) |
| cf_output      | `ZDNN_4DS`             | (1, num_dirs, num_batches, num_hidden)                                                              |

|                | create transformed descriptor via                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input          | `zdnn_generate_transformed_desc`                                                                                                                                          |
| h0             | `zdnn_generate_transformed_desc`                                                                                                                                          |
| c0             | `zdnn_generate_transformed_desc`                                                                                                                                          |
| weights        | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_WEIGHTS` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR`        |
| bias           | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_BIASES` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR`         |
| hidden_weights | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_HIDDEN_WEIGHTS` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR` |
| hidden_biases  | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_HIDDEN_BIASES` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR`  |
| hn_output      | `zdnn_generate_transformed_desc`                                                                                                                                          |
| cf_output      | `zdnn_generate_transformed_desc`                                                                                                                                          |

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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell)

[ONNX LSTM](https://onnx.ai/onnx/operators/onnx__LSTM.html#l-onnx-doc-lstm)

---

### zdnn_gru

[Back to Table of Contents](#TOC)

#### Description

Implements Gated Recurrent Unit (Kyunghyun Cho 2014). Supports only reset after
linear.

The following formula is computed for the input tensor input(t) for all time
steps:

```C
(Default: f=Sigmoid, g=Tanh):

- zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

- rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

- ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)

- Ht = (1 - zt) (.) ht + zt (.) Ht-1
```

#### Format

```C
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

- `num_hidden` dimensions: <a id="gru-hid_sz"></a>
  - Any num_hidden dimension must be less than or equal to
    `zdnn_get_max_for_dim(2) / 3` elements.

#### Parameters

- `zdnn_ztensor *input`

  - Input must be a tensor with the shape (num_timesteps, num_batches,
    num_features) prior to transformation with the `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)

- `zdnn_ztensor *h0`

  - Tensor containing the initial hidden state with shape (num_dirs,
    num_batches, num_hidden) prior to transformation with the
    `zdnn_transform_ztensor` API.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Must follow [general tensor requirements](#gen-zten-reqs)
  - Must follow [num_hidden requirements](#gru-hid_sz)

- `zdnn_ztensor *weights`

  - Tensor containing the concatenated input connection weights in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (num_dirs, num_features, num_hidden) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_GRU`
    - `USAGE_WEIGHTS`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#gru-hid_sz)

- `zdnn_ztensor *biases`

  - Tensor containing the concatenated input connection bias in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, expects each gate needs to be shape (num_dirs,
    num_hidden).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_GRU`
    - `USAGE_BIASES`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#gru-hid_sz)

- `zdnn_ztensor *hidden_weights`

  - Tensor containing the concatenated hidden connection weights in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, each gate needs to be transposed to shape
    (num_dirs, num_hidden, num_hidden) by the caller.
  - Expects `pre_transformed_desc->layout` to be `ZDNN_3DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_GRU`
    - `USAGE_HIDDEN_WEIGHTS`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#gru-hid_sz)

- `zdnn_ztensor *hidden_biases`

  - Tensor containing the concatenated hidden connection bias in (Z)update,
    Reset, Hidden, (ZRH) order.
  - Prior to transformation, expects each gate needs to be shape (num_dirs,
    num_hidden).
  - Expects `pre_transformed_desc->layout` to be `ZDNN_2DS`.
  - Expects `zdnn_concat_info` having the following flags turned on:
    - `RNN_TYPE_GRU`
    - `USAGE_HIDDEN_BIASES`
    - Appropriate `PREV_LAYER` flag:
      - `PREV_LAYER_NONE` if `input` tensor is not from a previous RNN layer
      - `PREV_LAYER_UNI` if `input` tensor is uni-directional output from a
        previous RNN layer
      - `PREV_LAYER_BIDIR` if `input` tensor is bi-directional output from a
        previous RNN layer
  - Must follow [concatenated tensor requirements](#concat-zten-reqs)
  - Must follow [num_hidden requirements](#gru-hid_sz)

- `lstm_gru_direction direction`

  - Direction indicator of `lstm_gru_direction direction` type. Valid values:
    - `FWD` (forward)
    - `BWD` (backward)
    - `BIDIR` (bi-directional).
  - For input shapes, the num_dirs dimension should be:
    - `1` for unidirectional calls such as FWD or BWD
    - `2` for bidirectional calls such that:
      - dimension 0 contains FWD values.
      - dimension 1 contains BWD values.

- `void *work_area`

  - A preallocated memory address to use for temporary storage during internal
    operation processing.
  - If set to NULL, the operation will determine, allocate and free storage
    automatically.
  - Amount of required storage can be determined given the GRU timestep, batch,
    and num_hidden values.

    - The sample code below creates a ztensor descriptor that is an equivalent
      size of the required `work_area`. To use this sample code yourself,
      replace the `num_timesteps`, `num_batches`, and `num_hidden` variables
      with your own values.

      ```C
        zdnn_tensor_desc desc;
        desc.dim4 = (3 * num_timesteps) + 5;
        desc.dim3 = 1;
        desc.dim2 = num_batches;
        desc.dim1 = num_hidden;
        uint64_t work_area_size = zdnn_getsize_ztensor(&desc);
      ```

  - For bidirectional, twice the amount of contiguous storage is required.
  - The start of the buffer must be 4k aligned.

- `zdnn_ztensor *hn_output`

  - Output results of the hidden states

  - Expects pre_transformed_desc->layout to be `ZDNN_4DS`.

  - Must follow [general tensor requirements](#gen-zten-reqs)

  - Must follow [num_hidden requirements](#lstm-hid_sz)

  - Output pre-transformed shapes:

    - all timesteps: (num_timesteps, num_dirs, num_batches, num_hidden)
    - final timestep only: (1, num_dirs, num_batches, num_hidden)

  - For bidirectional (`BIDIR`) output:

    - Forward and backward results are concatenated on the innermost dimension.
    - Can be used directly as input for subsequent RNN layers without needing
      untransformation.
      - Can not be used directly as input for other non-RNN zDNN ops.
    - Untransformation is supported.

  - Note that for `BWD` and the backward component of `BIDIR` directions, the
    output order matches the order of the input, not the processing order. For
    example, the first input timestep is the last to be processed and its result
    is the first timestep of the output.

#### Summary

|                | pre-transformed layout | pre-transformed shape                                                                               |
| -------------- | ---------------------- | --------------------------------------------------------------------------------------------------- |
| input          | `ZDNN_3DS`             | (num_timesteps, num_batches, num_features)                                                          |
| h0             | `ZDNN_3DS`             | (num_dirs, num_batches, num_hidden)                                                                 |
| weights        | `ZDNN_3DS`             | (num_dirs, num_features, num_hidden)                                                                |
| bias           | `ZDNN_2DS`             | (num_dirs, num_hidden)                                                                              |
| hidden_weights | `ZDNN_3DS`             | (num_dirs, num_hidden, num_hidden)                                                                  |
| hidden_biases  | `ZDNN_2DS`             | (num_dirs, num_hidden)                                                                              |
| hn_output      | `ZDNN_4DS`             | (num_timesteps, num_dirs, num_batches, num_hidden)<br>(last timestep only when `num_timesteps` = 1) |

|                | create transformed descriptor via                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| input          | `zdnn_generate_transformed_desc`                                                                                                                                          |
| h0             | `zdnn_generate_transformed_desc`                                                                                                                                          |
| weights        | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_WEIGHTS` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR`        |
| bias           | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_BIASES` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR`         |
| hidden_weights | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_HIDDEN_WEIGHTS` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR` |
| hidden_biases  | `zdnn_generate_transformed_desc_concatenated` - `RNN_TYPE_LSTM` + `USAGE_HIDDEN_BIASES` + one of the following:<br>`PREV_LAYER_NONE`/`PREV_LAYER_UNI`/`PREV_LAYER_BIDIR`  |
| hn_output      | `zdnn_generate_transformed_desc`                                                                                                                                          |

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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell)

[ONNX GRU](https://onnx.ai/onnx/operators/onnx__GRU.html#l-onnx-doc-gru)

---

### zdnn_avgpool2d

[Back to Table of Contents](#TOC)

#### Description

Given an input tensor in zDNN transformed format, padding type, kernel size and
kernel stride, produces a downsampled tensor reducing the middle dimensions
based on the mean values within the kernel window at each step and stores the
results into the provided output zDNN tensor.

#### Format

```C
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
    - stride_height is larger than `zdnn_get_max_for_dim(3)`.
    - stride_width is larger than `zdnn_get_max_for_dim(2)`.
    - kernel_height is 0 or is larger than `zdnn_get_max_for_dim(3)`.
    - kernel_width is 0 or is larger than `zdnn_get_max_for_dim(2)`.
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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow AvgPool](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/avg-pool)

[ONNX AvgPool](https://onnx.ai/onnx/operators/onnx__AveragePool.html#l-onnx-doc-averagepool)

---

### zdnn_maxpool2d

[Back to Table of Contents](#TOC)

#### Description

Given an input tensor in zDNN transformed format, padding type, kernel size and
kernel stride, produces a downsampled tensor reducing the middle dimensions
based on the maximum values within the kernel window at each step and stores the
results into the provided output zDNN tensor.

#### Format

```C
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
    - stride_height is larger than `zdnn_get_max_for_dim(3)`.
    - stride_width is larger than `zdnn_get_max_for_dim(2)`.
    - kernel_height is 0 or is larger than `zdnn_get_max_for_dim(3)`.
    - kernel_width is 0 or is larger than `zdnn_get_max_for_dim(2)`.
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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow MaxPool](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/max-pool)

[ONNX MaxPool](https://onnx.ai/onnx/operators/onnx__MaxPool.html#l-onnx-doc-maxpool)

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

```C
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

| strides and padding                       | input (num_batches, height_in, width_in, channels_in)                  | kernel (kernel_height, kernel_width, channels_in, channels_out) | bias (channels_out) | output (num_batches, height_out, width_out, channels_out)                                                                        |
| ----------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| both strides > 0 and =< 13, SAME padding  |                                                                        | both kernel_height and kernel_width must be =< 64               |                     | height_out = ceil(height_in/stride_height)<br>width_out = ceil(width_in/stride_width)                                            |
| both strides > 0 and =< 13, VALID padding | height_in must be >= kernel_height<br>width_in must be >= kernel_width | both kernel_height and kernel_width must be =< 64               |                     | height_out = ceil((height_in - kernel_height + 1)/stride_height)<br>width_out = ceil((width_in - kernel_width + 1)/stride_width) |
| both strides = 0, VALID padding           | height_in must be = kernel_height<br>width_in must be = kernel_width   | both kernel_height and kernel_width must be =< 448              |                     | both height_out and width_out must be 1                                                                                          |

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

#### Since

1.1.0

#### Requirements

This feature requires that:

- `zdnn_is_nnpa_installed()` returns true
- the underlying hardware supports zDNN APIs 1.1.x or later at runtime

See [Validating the environment at runtime](#runtime-val).

#### Framework Examples

[TensorFlow Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)

[ONNX Conv2D](https://onnx.ai/onnx/operators/onnx__Conv.html#l-onnx-doc-conv)

## Convenience Functions

[Back to Table of Contents](#TOC)

- None

---

## Usage Examples

### Example flow of an application calling the zDNN APIs

[Back to Table of Contents](#TOC)

```C
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

  // check status for zAIU availability, supported ops, etc. here
  // status = zdnn_query();

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

  // Free zTensors
  status = zdnn_free_ztensor_buffer(&ztensor_a);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&ztensor_b);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&ztensor_out);
  assert(status == ZDNN_OK);

  free(data1);
  free(data2);
  free(data_out);
}
```

---

### Example of calling the zdnn_quantized_matmul_op API (normal)

[Back to Table of Contents](#TOC)

```C
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021, 2024
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

#include "zdnn.h"

// Sample: Quantized Matmul
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * Quantized Matmul:
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (s, m, n)
   * weights         |  ZDNN_3DS  | (s, n, p)
   * input_biases    |  ZDNN_2DS  | (s, p)
   *
   * OUTPUTS -------------------------------------------------------------
   * output          |  ZDNN_3DS  | (s, m, p)
   ***********************************************************************/
  uint32_t s = 2;
  uint32_t m = 3;
  uint32_t n = 4;
  uint32_t p = 5;

  short int8_size = 1; // size of each int8 element in bytes
  short float_size = 4; // size of each float element in bytes

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/
  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &input_pre_tfrmd_desc,
                                 s, m, n);

  status = zdnn_generate_quantized_transformed_desc(
      &input_pre_tfrmd_desc, QUANTIZED_INT8, &input_tfrmd_desc);
  assert(status == ZDNN_OK);

  float input_scale = 1.f;
  float input_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                                   &input_tfrmd_desc,
                                                   input_scale, input_offset,
                                                   &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size = s * m * n * float_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_quantized_ztensor(&input, false, INT8_MIN, INT8_MAX,
                                            input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create weights zTensor
   ***********************************************************************/
  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &weights_pre_tfrmd_desc,
                                 s, n, p);

  status = zdnn_generate_quantized_transformed_desc(
      &weights_pre_tfrmd_desc, QUANTIZED_WEIGHTS_INT8, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  float weights_scale = 1.f;
  float weights_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                                   &weights_tfrmd_desc,
                                                   weights_scale,
                                                   weights_offset, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = s * n * p * int8_size;
  void *weights_data = malloc(weights_data_size);

  status = zdnn_transform_quantized_ztensor(&weights, false, INT8_MIN, INT8_MAX,
                                            weights_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensor
   ***********************************************************************/
  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &biases_pre_tfrmd_desc,
                                 s, p);

  status = zdnn_generate_quantized_transformed_desc(
      &biases_pre_tfrmd_desc, QUANTIZED_INT8, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  float biases_scale = 1.f;
  float biases_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                                   &biases_tfrmd_desc,
                                                   biases_scale, biases_offset,
                                                   &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = s * p * float_size;
  void *biases_data = malloc(biases_data_size);

  status = zdnn_transform_quantized_ztensor(&biases, false, INT8_MIN, INT8_MAX,
                                            biases_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/
  zdnn_tensor_desc output_pre_tfrmd_desc, output_tfrmd_desc;
  zdnn_ztensor output;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &output_pre_tfrmd_desc,
                                 s, m, p);

  status = zdnn_generate_quantized_transformed_desc(
      &output_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &output_tfrmd_desc);
  assert(status == ZDNN_OK);

  float output_scale = 1.f;
  float output_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&output_pre_tfrmd_desc,
                                                   &output_tfrmd_desc,
                                                   output_scale, output_offset,
                                                   &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/
  status = zdnn_quantized_matmul_op(&input, &weights, &biases,
                                    MATMUL_OP_ADDITION, INT8_MIN, INT8_MAX,
                                    NULL, &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/
  uint64_t output_data_size = s * m * p * float_size;
  void *output_data = malloc(output_data_size);

  status = zdnn_transform_origtensor(&output, output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&output);
  assert(status == ZDNN_OK);

  free(input_data);
  free(weights_data);
  free(biases_data);
  free(output_data);
}
```

---

### Example of calling the zdnn_quantized_matmul_op API (on-the-fly)

[Back to Table of Contents](#TOC)

```C
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021, 2024
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

#include "zdnn.h"

// Sample: Quantized Matmul on-the-fly
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * Quantized Matmul on-the-fly:
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (s, m, n)
   * weights         |  ZDNN_3DS  | (s, n, p)
   * input_biases    |  ZDNN_2DS  | (s, p)
   *
   * OUTPUTS -------------------------------------------------------------
   * output          |  ZDNN_3DS  | (s, m, p)
   ***********************************************************************/
  uint32_t s = 2;
  uint32_t m = 3;
  uint32_t n = 4;
  uint32_t p = 5;

  short int8_size = 1; // size of each int8 element in bytes
  short float_size = 4; // size of each float element in bytes

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/
  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &input_pre_tfrmd_desc,
                                 s, m, n);

  status = zdnn_generate_quantized_transformed_desc(
      &input_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &input_tfrmd_desc);
  assert(status == ZDNN_OK);

  float input_scale = 1.f;
  float input_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                                   &input_tfrmd_desc,
                                                   input_scale, input_offset,
                                                   &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size = s * m * n * float_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_ztensor(&input, input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create weights zTensor
   ***********************************************************************/
  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &weights_pre_tfrmd_desc,
                                 s, n, p);

  status = zdnn_generate_quantized_transformed_desc(
      &weights_pre_tfrmd_desc, QUANTIZED_WEIGHTS_INT8, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  float weights_scale = 1.f;
  float weights_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                                   &weights_tfrmd_desc,
                                                   weights_scale,
                                                   weights_offset, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = s * n * p * int8_size;
  void *weights_data = malloc(weights_data_size);

  status = zdnn_transform_quantized_ztensor(&weights, false, INT8_MIN, INT8_MAX,
                                            weights_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensor
   ***********************************************************************/
  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &biases_pre_tfrmd_desc,
                                 s, p);

  status = zdnn_generate_quantized_transformed_desc(
      &biases_pre_tfrmd_desc, QUANTIZED_INT8, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  float biases_scale = 1.f;
  float biases_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                                   &biases_tfrmd_desc,
                                                   biases_scale, biases_offset,
                                                   &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = s * p * float_size;
  void *biases_data = malloc(biases_data_size);

  status = zdnn_transform_quantized_ztensor(&biases, false, INT8_MIN, INT8_MAX,
                                            biases_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/
  zdnn_tensor_desc output_pre_tfrmd_desc, output_tfrmd_desc;
  zdnn_ztensor output;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &output_pre_tfrmd_desc,
                                 s, m, p);

  status = zdnn_generate_quantized_transformed_desc(
      &output_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &output_tfrmd_desc);
  assert(status == ZDNN_OK);

  float output_scale = 1.f;
  float output_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&output_pre_tfrmd_desc,
                                                   &output_tfrmd_desc,
                                                   output_scale, output_offset,
                                                   &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/
  status = zdnn_quantized_matmul_op(&input, &weights, &biases,
                                    MATMUL_OP_ADDITION, INT8_MIN, INT8_MAX,
                                    NULL, &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/
  uint64_t output_data_size = s * m * p * float_size;
  void *output_data = malloc(output_data_size);

  status = zdnn_transform_origtensor(&output, output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&output);
  assert(status == ZDNN_OK);

  free(input_data);
  free(weights_data);
  free(biases_data);
  free(output_data);
}
```

---

### Example of calling the zdnn_quantized_matmul with pre_computed=true API (normal)

[Back to Table of Contents](#TOC)

```C
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021, 2024
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

#include "zdnn.h"

// Sample: Quantized Matmul Pre-Computed
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * Quantized Matmul Pre-Computed:
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (s, m, n)
   * weights         |  ZDNN_3DS  | (s, n, p)
   * input_biases    |  ZDNN_2DS  | (s, p)
   *
   * OUTPUTS -------------------------------------------------------------
   * output          |  ZDNN_3DS  | (s, m, p)
   ***********************************************************************/
  uint32_t s = 2;
  uint32_t m = 3;
  uint32_t n = 4;
  uint32_t p = 5;

  short int8_size = 1; // size of each int8 element in bytes
  short float_size = 4; // size of each float element in bytes

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/
  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &input_pre_tfrmd_desc,
                                 s, m, n);

  status = zdnn_generate_quantized_transformed_desc(
      &input_pre_tfrmd_desc, QUANTIZED_INT8, &input_tfrmd_desc);
  assert(status == ZDNN_OK);

  float input_scale = 1.f;
  float input_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                                   &input_tfrmd_desc,
                                                   input_scale, input_offset,
                                                   &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size = s * m * n * float_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_quantized_ztensor(&input, false, INT8_MIN, INT8_MAX,
                                            input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create weights zTensor
   ***********************************************************************/
  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &weights_pre_tfrmd_desc,
                                 s, n, p);

  status = zdnn_generate_quantized_transformed_desc(
      &weights_pre_tfrmd_desc, QUANTIZED_WEIGHTS_INT8, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  float weights_scale = 1.f;
  float weights_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                                   &weights_tfrmd_desc,
                                                   weights_scale,
                                                   weights_offset, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = s * n * p * int8_size;
  void *weights_data = malloc(weights_data_size);

  status = zdnn_transform_quantized_ztensor(&weights, false, INT8_MIN, INT8_MAX,
                                            weights_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create pre-computed biases zTensor
   ***********************************************************************/
  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &biases_pre_tfrmd_desc,
                                 s, p);

  status = zdnn_generate_quantized_transformed_desc(
      &biases_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  float biases_scale = 1.f;
  float biases_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                                   &biases_tfrmd_desc,
                                                   biases_scale, biases_offset,
                                                   &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = s * p * float_size;
  void *biases_data = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/
  zdnn_tensor_desc output_pre_tfrmd_desc, output_tfrmd_desc;
  zdnn_ztensor output;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &output_pre_tfrmd_desc,
                                 s, m, p);

  status = zdnn_generate_quantized_transformed_desc(
      &output_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &output_tfrmd_desc);
  assert(status == ZDNN_OK);

  float output_scale = 1.f;
  float output_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&output_pre_tfrmd_desc,
                                                   &output_tfrmd_desc,
                                                   output_scale, output_offset,
                                                   &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/
  status = zdnn_quantized_matmul_op(&input, &weights, &biases,
                                                 MATMUL_OP_ADDITION, INT8_MIN,
                                                 INT8_MAX, false, true, NULL, &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/
  uint64_t output_data_size = s * m * p * float_size;
  void *output_data = malloc(output_data_size);

  status = zdnn_transform_origtensor(&output, output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&output);
  assert(status == ZDNN_OK);

  free(input_data);
  free(weights_data);
  free(biases_data);
  free(output_data);
}
```

---

### Example of calling the zdnn_quantized_matmul_op with pre_computed=true API (on-the-fly)

[Back to Table of Contents](#TOC)

```C
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2021, 2024
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

#include "zdnn.h"

// Sample: Quantized Matmul Pre-Computed on-the-fly
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * Quantized Matmul Pre-Computed on-the-fly:
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (s, m, n)
   * weights         |  ZDNN_3DS  | (s, n, p)
   * input_biases    |  ZDNN_2DS  | (s, p)
   *
   * OUTPUTS -------------------------------------------------------------
   * output          |  ZDNN_3DS  | (s, m, p)
   ***********************************************************************/
  uint32_t s = 2;
  uint32_t m = 3;
  uint32_t n = 4;
  uint32_t p = 5;

  short int8_size = 1; // size of each int8 element in bytes
  short float_size = 4; // size of each float element in bytes

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/
  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &input_pre_tfrmd_desc,
                                 s, m, n);

  status = zdnn_generate_quantized_transformed_desc(
      &input_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &input_tfrmd_desc);
  assert(status == ZDNN_OK);

  float input_scale = 1.f;
  float input_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&input_pre_tfrmd_desc,
                                                   &input_tfrmd_desc,
                                                   input_scale, input_offset,
                                                   &input);
  assert(status == ZDNN_OK);

  uint64_t input_data_size = s * m * n * float_size;
  void *input_data = malloc(input_data_size);

  status = zdnn_transform_ztensor(&input, input_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create weights zTensor
   ***********************************************************************/
  zdnn_tensor_desc weights_pre_tfrmd_desc, weights_tfrmd_desc;
  zdnn_ztensor weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &weights_pre_tfrmd_desc,
                                 s, n, p);

  status = zdnn_generate_quantized_transformed_desc(
      &weights_pre_tfrmd_desc, QUANTIZED_WEIGHTS_INT8, &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  float weights_scale = 1.f;
  float weights_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                                   &weights_tfrmd_desc,
                                                   weights_scale,
                                                   weights_offset, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = s * n * p * int8_size;
  void *weights_data = malloc(weights_data_size);

  status = zdnn_transform_quantized_ztensor(&weights, false, INT8_MIN, INT8_MAX,
                                            weights_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create pre-computed biases zTensor
   ***********************************************************************/
  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, FP32, &biases_pre_tfrmd_desc,
                                 s, p);

  status = zdnn_generate_quantized_transformed_desc(
      &biases_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  float biases_scale = 1.f;
  float biases_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                                   &biases_tfrmd_desc,
                                                   biases_scale, biases_offset,
                                                   &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = s * p * float_size;
  void *biases_data = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/
  zdnn_tensor_desc output_pre_tfrmd_desc, output_tfrmd_desc;
  zdnn_ztensor output;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &output_pre_tfrmd_desc,
                                 s, m, p);

  status = zdnn_generate_quantized_transformed_desc(
      &output_pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &output_tfrmd_desc);
  assert(status == ZDNN_OK);

  float output_scale = 1.f;
  float output_offset = 0.f;

  status = zdnn_init_quantized_ztensor_with_malloc(&output_pre_tfrmd_desc,
                                                   &output_tfrmd_desc,
                                                   output_scale, output_offset,
                                                   &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/
  status = zdnn_quantized_matmul_op(&input, &weights, &biases,
                                                 MATMUL_OP_ADDITION, INT8_MIN,
                                                 INT8_MAX, false, true, NULL, &output);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/
  uint64_t output_data_size = s * m * p * float_size;
  void *output_data = malloc(output_data_size);

  status = zdnn_transform_origtensor(&output, output_data);
  assert(status == ZDNN_OK);

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&weights);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&biases);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&output);
  assert(status == ZDNN_OK);

  free(input_data);
  free(weights_data);
  free(biases_data);
  free(output_data);
}
```

---

### Example of an application calling the zdnn_lstm API (forward)

[Back to Table of Contents](#TOC)

```C
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: LSTM
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * LSTM (FWD/BWD):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (1, num_batches, num_hidden)
   * c0              |  ZDNN_3DS  | (1, num_batches, num_hidden)
   * weights         |  ZDNN_3DS  | (1, num_features, num_hidden)
   * biases          |  ZDNN_2DS  | (1, num_hidden)
   * hidden_weights  |  ZDNN_3DS  | (1, num_hidden, num_hidden)
   * hidden_biases   |  ZDNN_2DS  | (1, num_hidden)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_4DS  | (num_timesteps, 1, num_batches, num_hidden)
   *                 |            | or (1, 1, num_batches, num_hidden)
   * cf_output       |  ZDNN_4DS  | (1, 1, num_batches, num_hidden)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hidden = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  lstm_gru_direction dir = FWD;
  uint8_t num_dirs = 1;

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
                                 num_batches, num_hidden);
  status =
      zdnn_generate_transformed_desc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &h0);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &c0);
  assert(status == ZDNN_OK);

  uint64_t h0c0_data_size = num_batches * num_hidden * element_size;
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
                                 num_dirs, num_features, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_NONE,
      &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hidden * element_size;
  void *weights_data_f = malloc(weights_data_size);
  void *weights_data_i = malloc(weights_data_size);
  void *weights_data_c = malloc(weights_data_size);
  void *weights_data_o = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_f, weights_data_i,
                                  weights_data_c, weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, RNN_TYPE_LSTM | USAGE_BIASES | PREV_LAYER_NONE,
      &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hidden * element_size;
  void *biases_data_f = malloc(biases_data_size);
  void *biases_data_i = malloc(biases_data_size);
  void *biases_data_c = malloc(biases_data_size);
  void *biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_f, biases_data_i,
                                  biases_data_c, biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hidden, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE,
      &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hidden * num_hidden * element_size;
  void *hidden_weights_data_f = malloc(hidden_weights_data_size);
  void *hidden_weights_data_i = malloc(hidden_weights_data_size);
  void *hidden_weights_data_c = malloc(hidden_weights_data_size);
  void *hidden_weights_data_o = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_f,
                                  hidden_weights_data_i, hidden_weights_data_c,
                                  hidden_weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_biases_pre_tfrmd_desc, hidden_biases_tfrmd_desc;
  zdnn_ztensor hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &hidden_biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_biases_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE,
      &hidden_biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(
      &hidden_biases_pre_tfrmd_desc, &hidden_biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t hidden_biases_data_size = num_hidden * element_size;

  void *hidden_biases_data_f = malloc(hidden_biases_data_size);
  void *hidden_biases_data_i = malloc(hidden_biases_data_size);
  void *hidden_biases_data_c = malloc(hidden_biases_data_size);
  void *hidden_biases_data_o = malloc(hidden_biases_data_size);

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

  zdnn_init_pre_transformed_desc(ZDNN_4DS, type, &hncf_pre_tfrmd_desc, 1, 1,
                                 num_batches, num_hidden);
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
   * Call the zAIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_lstm(&input, &h0, &c0, &weights, &biases, &hidden_weights,
                     &hidden_biases, dir, work_area, &hn_output_ztensor,
                     &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  uint64_t hncf_data_size = num_batches * num_hidden * element_size;
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

#### Example of an application calling the zdnn_lstm API (bi-directional)

[Back to Table of Contents](#TOC)

```C
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: LSTM BI-DIR
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * LSTM (BI-DIR):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (2, num_batches, num_hidden)
   * c0              |  ZDNN_3DS  | (2, num_batches, num_hidden)
   * weights         |  ZDNN_3DS  | (2, num_features, num_hidden)
   * biases          |  ZDNN_2DS  | (2, num_hidden)
   * hidden_weights  |  ZDNN_3DS  | (2, num_hidden, num_hidden)
   * hidden_biases   |  ZDNN_2DS  | (2, num_hidden)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_4DS  | (num_timesteps, 2, num_batches, num_hidden)
   *                 |            | or (1, 2, num_batches, num_hidden)
   * cf_output       |  ZDNN_4DS  | (1, 2, num_batches, num_hidden)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hidden = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  lstm_gru_direction dir = BIDIR;
  uint8_t num_dirs = 2;

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
                                 num_batches, num_hidden);
  status =
      zdnn_generate_transformed_desc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &h0);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &c0);
  assert(status == ZDNN_OK);

  uint64_t h0c0_data_size = num_batches * num_hidden * element_size;
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
                                 num_dirs, num_features, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_NONE,
      &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hidden * element_size;
  void *weights_data_f = malloc(weights_data_size);
  void *weights_data_i = malloc(weights_data_size);
  void *weights_data_c = malloc(weights_data_size);
  void *weights_data_o = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_f, weights_data_i,
                                  weights_data_c, weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, RNN_TYPE_LSTM | USAGE_BIASES | PREV_LAYER_NONE,
      &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hidden * element_size;
  void *biases_data_f = malloc(biases_data_size);
  void *biases_data_i = malloc(biases_data_size);
  void *biases_data_c = malloc(biases_data_size);
  void *biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_f, biases_data_i,
                                  biases_data_c, biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hidden, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE,
      &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hidden * num_hidden * element_size;
  void *hidden_weights_data_f = malloc(hidden_weights_data_size);
  void *hidden_weights_data_i = malloc(hidden_weights_data_size);
  void *hidden_weights_data_c = malloc(hidden_weights_data_size);
  void *hidden_weights_data_o = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_f,
                                  hidden_weights_data_i, hidden_weights_data_c,
                                  hidden_weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_biases_pre_tfrmd_desc, hidden_biases_tfrmd_desc;
  zdnn_ztensor hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &hidden_biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_biases_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE,
      &hidden_biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(
      &hidden_biases_pre_tfrmd_desc, &hidden_biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t hidden_biases_data_size = num_hidden * element_size;

  void *hidden_biases_data_f = malloc(hidden_biases_data_size);
  void *hidden_biases_data_i = malloc(hidden_biases_data_size);
  void *hidden_biases_data_c = malloc(hidden_biases_data_size);
  void *hidden_biases_data_o = malloc(hidden_biases_data_size);

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

  zdnn_init_pre_transformed_desc(ZDNN_4DS, type, &hn_pre_tfrmd_desc,
                                 num_timesteps, 2, num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc);
  assert(status == ZDNN_OK);

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &cf_pre_tfrmd_desc, 1, 2,
                                 num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc,
                                         &hn_output_ztensor);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc,
                                         &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_lstm(&input, &h0, &c0, &weights, &biases, &hidden_weights,
                     &hidden_biases, dir, work_area, &hn_output_ztensor,
                     &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  uint64_t hn_data_size =
      num_timesteps * 2 * num_batches * num_hidden * element_size;
  uint64_t cf_data_size = 2 * num_batches * num_hidden * element_size;
  void *hn_output_data = malloc(hn_data_size);
  void *cf_output_data = malloc(cf_data_size);

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

### Example of an application calling the zdnn_lstm API

#### Example of an application calling the zdnn_lstm API (multi-layer bi-directional)

[Back to Table of Contents](#TOC)

```C
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

void do_bidir_layer(zdnn_ztensor *input, uint32_t num_hidden,
                    zdnn_ztensor *hn_output, bool is_prev_layer_bidir) {

  zdnn_status status;

  uint32_t num_batches = input->pre_transformed_desc->dim2;

  // if input is bidir output from previous layer then number of features for
  // this layer is 2x of hidden-state size (dim1) of the previous layer
  uint32_t num_features =
      input->pre_transformed_desc->dim1 * (is_prev_layer_bidir ? 2 : 1);

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  lstm_gru_direction dir = BIDIR;
  uint8_t num_dirs = 2;

  /***********************************************************************
   * Create initial hidden and cell state zTensors
   ***********************************************************************/

  zdnn_tensor_desc h0c0_pre_tfrmd_desc, h0c0_tfrmd_desc;
  zdnn_ztensor h0, c0;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &h0c0_pre_tfrmd_desc, num_dirs,
                                 num_batches, num_hidden);
  status =
      zdnn_generate_transformed_desc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &h0);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&h0c0_pre_tfrmd_desc, &h0c0_tfrmd_desc,
                                         &c0);
  assert(status == ZDNN_OK);

  uint64_t h0c0_data_size = num_batches * num_hidden * element_size;
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

  // if using previous layer bidir output as input then number of features of
  // this layer is
  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &weights_pre_tfrmd_desc,
                                 num_dirs, num_features, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_WEIGHTS |
          (is_prev_layer_bidir ? PREV_LAYER_BIDIR : PREV_LAYER_UNI),
      &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hidden * element_size;
  void *weights_data_f = malloc(weights_data_size);
  void *weights_data_i = malloc(weights_data_size);
  void *weights_data_c = malloc(weights_data_size);
  void *weights_data_o = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_f, weights_data_i,
                                  weights_data_c, weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_BIASES |
          (is_prev_layer_bidir ? PREV_LAYER_BIDIR : PREV_LAYER_UNI),
      &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hidden * element_size;
  void *biases_data_f = malloc(biases_data_size);
  void *biases_data_i = malloc(biases_data_size);
  void *biases_data_c = malloc(biases_data_size);
  void *biases_data_o = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_f, biases_data_i,
                                  biases_data_c, biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hidden, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS |
          (is_prev_layer_bidir ? PREV_LAYER_BIDIR : PREV_LAYER_UNI),
      &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hidden * num_hidden * element_size;
  void *hidden_weights_data_f = malloc(hidden_weights_data_size);
  void *hidden_weights_data_i = malloc(hidden_weights_data_size);
  void *hidden_weights_data_c = malloc(hidden_weights_data_size);
  void *hidden_weights_data_o = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_f,
                                  hidden_weights_data_i, hidden_weights_data_c,
                                  hidden_weights_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_biases_pre_tfrmd_desc, hidden_biases_tfrmd_desc;
  zdnn_ztensor hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &hidden_biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_biases_pre_tfrmd_desc,
      RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES |
          (is_prev_layer_bidir ? PREV_LAYER_BIDIR : PREV_LAYER_UNI),
      &hidden_biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(
      &hidden_biases_pre_tfrmd_desc, &hidden_biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t hidden_biases_data_size = num_hidden * element_size;

  void *hidden_biases_data_f = malloc(hidden_biases_data_size);
  void *hidden_biases_data_i = malloc(hidden_biases_data_size);
  void *hidden_biases_data_c = malloc(hidden_biases_data_size);
  void *hidden_biases_data_o = malloc(hidden_biases_data_size);

  status = zdnn_transform_ztensor(&hidden_biases, hidden_biases_data_f,
                                  hidden_biases_data_i, hidden_biases_data_c,
                                  hidden_biases_data_o);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create cf output zTensor
   ***********************************************************************/

  zdnn_tensor_desc cf_pre_tfrmd_desc, cf_tfrmd_desc;

  zdnn_ztensor cf_output_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_4DS, type, &cf_pre_tfrmd_desc, 1, 2,
                                 num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&cf_pre_tfrmd_desc, &cf_tfrmd_desc,
                                         &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/

  void *work_area = NULL;

  status =
      zdnn_lstm(input, &h0, &c0, &weights, &biases, &hidden_weights,
                &hidden_biases, dir, work_area, hn_output, &cf_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Cleanup and Return
   ***********************************************************************/

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
  status = zdnn_free_ztensor_buffer(&cf_output_ztensor);
  assert(status == ZDNN_OK);

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

// Sample: LSTM multi-layer BIDIR
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  uint32_t num_hidden[2] = {5, 4};

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

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
   * Create 2 hn output zTensors
   ***********************************************************************/

  zdnn_tensor_desc hn_pre_tfrmd_desc[2], hn_tfrmd_desc[2];
  zdnn_ztensor hn_output[2];

  for (int i = 0; i < 2; i++) {
    zdnn_init_pre_transformed_desc(ZDNN_4DS, type, &hn_pre_tfrmd_desc[i],
                                   num_timesteps, 2, num_batches,
                                   num_hidden[i]);
    status = zdnn_generate_transformed_desc(&hn_pre_tfrmd_desc[i],
                                            &hn_tfrmd_desc[i]);
    assert(status == ZDNN_OK);

    status = zdnn_init_ztensor_with_malloc(&hn_pre_tfrmd_desc[i],
                                           &hn_tfrmd_desc[i], &hn_output[i]);
    assert(status == ZDNN_OK);
  }

  /***********************************************************************
   * Do the layers
   ***********************************************************************/

  // call the first layer with input, previous layer bidir = false, output goes
  // to hn_output[0]
  do_bidir_layer(&input, num_hidden[0], &hn_output[0], false);

  // call the second layer with hn_output[0] from layer 1, previous layer bidir
  // = true, output goes to hn_output[1]
  do_bidir_layer(&hn_output[0], num_hidden[1], &hn_output[1], true);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  void *hn_output_data[2];

  for (int i = 0; i < 2; i++) {
    uint64_t hn_output_data_size = (uint64_t)num_timesteps * num_batches *
                                   num_hidden[i] * 2 * element_size;
    hn_output_data[i] = malloc(hn_output_data_size);

    status = zdnn_transform_origtensor(&hn_output[i], hn_output_data[i]);
    assert(status == ZDNN_OK);
  }

  status = zdnn_free_ztensor_buffer(&input);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hn_output[0]);
  assert(status == ZDNN_OK);
  status = zdnn_free_ztensor_buffer(&hn_output[1]);
  assert(status == ZDNN_OK);

  free(input_data);
  free(hn_output_data[0]);
  free(hn_output_data[1]);
}



```

---

### Example of an application calling the zdnn_gru API

#### Example of an application calling the zdnn_gru API (forward)

[Back to Table of Contents](#TOC)

```C
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// Sample: GRU
int main(int argc, char *argv[]) {
  zdnn_status status;

#ifdef STATIC_LIB
  zdnn_init();
#endif

  /***********************************************************************
   *
   * GRU (FWD/BWD):
   *
   * INPUTS --------------------------------------------------------------
   * input           |  ZDNN_3DS  | (num_timesteps, num_batches, num_features)
   * h0              |  ZDNN_3DS  | (1, num_batches, num_hidden)
   * weights         |  ZDNN_3DS  | (1, num_features, num_hidden)
   * input_biases    |  ZDNN_2DS  | (1, num_hidden)
   * hidden_weights  |  ZDNN_3DS  | (1, num_hidden, num_hidden)
   * hidden_biases   |  ZDNN_2DS  | (1, num_hidden)
   *
   * OUTPUTS -------------------------------------------------------------
   * hn_output       |  ZDNN_4DS  | (num_timesteps, 1, num_batches, num_hidden)
   *                 |            | or (1, 1, num_batches, num_hidden)
   ***********************************************************************/

  /***********************************************************************
   * Create input zTensor
   ***********************************************************************/

  zdnn_tensor_desc input_pre_tfrmd_desc, input_tfrmd_desc;
  zdnn_ztensor input;

  uint32_t num_timesteps = 5;
  uint32_t num_batches = 3;
  uint32_t num_features = 32;
  uint32_t num_hidden = 5;

  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes

  lstm_gru_direction dir = FWD;
  uint8_t num_dirs = 1;

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
                                 num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&h0_pre_tfrmd_desc, &h0_tfrmd_desc);
  assert(status == ZDNN_OK);

  status =
      zdnn_init_ztensor_with_malloc(&h0_pre_tfrmd_desc, &h0_tfrmd_desc, &h0);
  assert(status == ZDNN_OK);

  uint64_t h0_data_size = num_batches * num_hidden * element_size;
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
                                 num_dirs, num_features, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &weights_pre_tfrmd_desc, RNN_TYPE_GRU | USAGE_WEIGHTS | PREV_LAYER_NONE,
      &weights_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&weights_pre_tfrmd_desc,
                                         &weights_tfrmd_desc, &weights);
  assert(status == ZDNN_OK);

  uint64_t weights_data_size = num_features * num_hidden * element_size;
  void *weights_data_z = malloc(weights_data_size);
  void *weights_data_r = malloc(weights_data_size);
  void *weights_data_h = malloc(weights_data_size);

  status = zdnn_transform_ztensor(&weights, weights_data_z, weights_data_r,
                                  weights_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc biases_pre_tfrmd_desc, biases_tfrmd_desc;
  zdnn_ztensor biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &biases_pre_tfrmd_desc, RNN_TYPE_GRU | USAGE_BIASES | PREV_LAYER_NONE,
      &biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&biases_pre_tfrmd_desc,
                                         &biases_tfrmd_desc, &biases);
  assert(status == ZDNN_OK);

  uint64_t biases_data_size = num_hidden * element_size;
  void *biases_data_z = malloc(biases_data_size);
  void *biases_data_r = malloc(biases_data_size);
  void *biases_data_h = malloc(biases_data_size);

  status = zdnn_transform_ztensor(&biases, biases_data_z, biases_data_r,
                                  biases_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden weights zTensor
   * Resultant zTensor is concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_weights_pre_tfrmd_desc, hidden_weights_tfrmd_desc;
  zdnn_ztensor hidden_weights;

  zdnn_init_pre_transformed_desc(ZDNN_3DS, type, &hidden_weights_pre_tfrmd_desc,
                                 num_dirs, num_hidden, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_weights_pre_tfrmd_desc,
      RNN_TYPE_GRU | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE,
      &hidden_weights_tfrmd_desc);
  assert(status == ZDNN_OK);
  status = zdnn_init_ztensor_with_malloc(&hidden_weights_pre_tfrmd_desc,
                                         &hidden_weights_tfrmd_desc,
                                         &hidden_weights);
  assert(status == ZDNN_OK);

  uint64_t hidden_weights_data_size = num_hidden * num_hidden * element_size;
  void *hidden_weights_data_z = malloc(hidden_weights_data_size);
  void *hidden_weights_data_r = malloc(hidden_weights_data_size);
  void *hidden_weights_data_h = malloc(hidden_weights_data_size);

  status = zdnn_transform_ztensor(&hidden_weights, hidden_weights_data_z,
                                  hidden_weights_data_r, hidden_weights_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create hidden biases zTensors
   * Resultant zTensors are concatenated
   ***********************************************************************/

  zdnn_tensor_desc hidden_biases_pre_tfrmd_desc, hidden_biases_tfrmd_desc;
  zdnn_ztensor hidden_biases;

  zdnn_init_pre_transformed_desc(ZDNN_2DS, type, &hidden_biases_pre_tfrmd_desc,
                                 num_dirs, num_hidden);
  status = zdnn_generate_transformed_desc_concatenated(
      &hidden_biases_pre_tfrmd_desc,
      RNN_TYPE_GRU | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE,
      &hidden_biases_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(
      &hidden_biases_pre_tfrmd_desc, &hidden_biases_tfrmd_desc, &hidden_biases);
  assert(status == ZDNN_OK);

  uint64_t hidden_biases_data_size = num_hidden * element_size;
  void *hidden_biases_data_z = malloc(hidden_biases_data_size);
  void *hidden_biases_data_r = malloc(hidden_biases_data_size);
  void *hidden_biases_data_h = malloc(hidden_biases_data_size);

  status = zdnn_transform_ztensor(&hidden_biases, hidden_biases_data_z,
                                  hidden_biases_data_r, hidden_biases_data_h);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Create output zTensor
   ***********************************************************************/

  // get only the last timestep
  zdnn_tensor_desc hn_pre_tfrmd_desc, hn_tfrmd_desc;

  zdnn_ztensor hn_output_ztensor;

  zdnn_init_pre_transformed_desc(ZDNN_4DS, type, &hn_pre_tfrmd_desc, 1, 1,
                                 num_batches, num_hidden);
  status = zdnn_generate_transformed_desc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc);
  assert(status == ZDNN_OK);

  status = zdnn_init_ztensor_with_malloc(&hn_pre_tfrmd_desc, &hn_tfrmd_desc,
                                         &hn_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Call the zAIU
   ***********************************************************************/

  void *work_area = NULL;

  status = zdnn_gru(&input, &h0, &weights, &biases, &hidden_weights,
                    &hidden_biases, dir, work_area, &hn_output_ztensor);
  assert(status == ZDNN_OK);

  /***********************************************************************
   * Output and Cleanup
   ***********************************************************************/

  uint64_t hn_data_size = num_batches * num_hidden * element_size;
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

---

### Example of an application creating a quantized ztensor

[Back to Table of Contents](#TOC)

```C
// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright IBM Corp. 2023
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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zdnn.h"

// ***************************************************************************
// Sample:
//
// Create a quantized zTensors
// ***************************************************************************
int main(int argc, char *argv[]) {
  zdnn_tensor_desc pre_tfrmd_desc, tfrmd_desc;
  zdnn_ztensor ztensor;
  zdnn_status status;

  uint32_t dim_n = 1, dim_h = 32, dim_w = 32, dim_c = 3;
  zdnn_data_types type = FP32;
  short element_size = 4; // size of each element in bytes
  uint64_t num_elements = dim_n * dim_h * dim_w * dim_c;

  // allocate tensor data storage
  void *data1 = malloc(num_elements * element_size);

  // read input_data

  // check status for zAIU availability, supported ops, etc. here
  // status = zdnn_query();

  // set input tensor data to 0 to 127 sequentially and repeat
  for (uint64_t i = 0; i < num_elements; i++) {
    ((float *)data1)[i] = (float)(i & 0x7f);
  }

  zdnn_init_pre_transformed_desc(ZDNN_NHWC, type, &pre_tfrmd_desc, dim_n, dim_h,
                                 dim_w, dim_c);
  float scale = 3;
  float offset = 2;

  // generate transformed shape information
  status = zdnn_generate_quantized_transformed_desc(
      &pre_tfrmd_desc, QUANTIZED_DLFLOAT16, &tfrmd_desc);
  assert(status == ZDNN_OK);

  // initialize zTensors and allocate 4k-aligned storage via helper function
  status = zdnn_init_quantized_ztensor_with_malloc(&pre_tfrmd_desc, &tfrmd_desc,
                                                   scale, offset, &ztensor);
  assert(status == ZDNN_OK);

  // transform the feature tensor
  status = zdnn_transform_ztensor(&ztensor, data1);
  assert(status == ZDNN_OK);

  // Free zTensors
  status = zdnn_free_ztensor_buffer(&ztensor);
  assert(status == ZDNN_OK);

  free(data1);
}
```
