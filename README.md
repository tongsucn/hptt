# HPTT: High-Performance Tensor Transposition

## Introduction

HPTT is a high-performance tensor transposition library written in C++14.
It performs out-of-space tensor transpositiion in following form:

![trans](https://github.com/tongsucn/hptt/blob/master/doc/readme/equ_trans.png)

where A and B are input and output tensor respectively. PI is the dimension
permutation, alpha is used to rescale the elements in input tensor, beta is for
updating the output.

## Features

* Explicit vectorization
* Multi-threaded with OpenMP
* Auto-Tuning
* Four data types supported: single, double, single complex, double complex
* Input and output tensor rescaling

## Installation

### Requirements:

* **Compiler:** Intel C++ compiler (icpc) 15.0, 16.0 or 17.0 (recommend); GNU C++
compiler (g++) 5.4 or newer
* **CMake**: 2.8 or newer
* **Python**: 3.4 or newer
* To build the test, [Google Test](https://github.com/google/googletest) needs
to be installed correctly

```shell
git clone https://github.com/tongsucn/hptt; cd hptt; mkdir build; cmake ..
make -j16; make install
```

By default, only dimensions in range `[2, 6]` are supported. To enable other
dimensions, use following variables in `cmake` command:

```shell
# Specifying minimum support dimension, not less than 2
-DHPTT_CODE_GEN_TRANS_ORDER_MIN=[MIN_DIM_YOU_WANT]
# Specifying maximum support dimension
-DHPTT_CODE_GEN_TRANS_ORDER_MAX=[MAX_DIM_YOU_WANT]
```

## Usage

An example with single floating, dimension permutation `2, 1, 0`
```c++
#include <hptt/hptt.h>

// Initialize data, permutation and sizes
float *in_data = ...;
float *out_data = ...;
std::vector<uint32_t> perm = {...};
std::vector<uint32_t> in_size = {...};

// Create transposition plan with updating coefficients 2.3 and 4.2, 20 threads
auto plan = hptt::create_plan(in_data, out_data, in_size, perm, 2.3f, 4.2f, 20);

// Execute plan
plan->exec();

...

// Reuse plan (sizes and permutations must be identical)
plan->reset_data(in_data_new, out_data_new);
```

The `plan` in above code is actually a `std::shared_ptr`, it does not need to
be released manually..

## Citation

```
@article{2017arXiv170404374S,
  author = {{Springer}, P. and {Su}, T. and {Bientinesi}, P.},
  title = "{HPTT: A High-Performance Tensor Transposition C++ Library}",
  archivePrefix = "arXiv",
  eprint = {1704.04374},
  primaryClass = "cs.MS",
  year = 2017,
  month = apr,
  url = "https://arxiv.org/abs/1704.04374"
}
```
