#include <array>
#include <chrono>
#include <random>
#include <memory>
#include <iostream>
#include <algorithm>
#include <type_traits>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/kernels/kernel_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>
#include <hptc/operations/operation_trans.h>

#include "sTranspose_210_384x2320x64.h"
#include "sTranspose_102_384x2320x64.h"


#define MEASURE_MAX 10


using namespace std;
using namespace hptc;


void reset_data(float *data, TensorIdx len) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<float> dist(-500.0f, 500.0f);
  for (TensorIdx idx = 0; idx < len; ++idx)
    data[idx] = dist(gen);
}


bool verify(const float *input_data, const float *output_data, TensorIdx len) {
  for (TensorIdx idx = 0; idx < len; ++idx) {
    double input_abs = static_cast<double>(input_data[idx]);
    if (input_abs < 0)
      input_abs = -input_abs;
    double output_abs = static_cast<double>(output_data[idx]);
    if (output_abs < 0)
      output_abs = -output_abs;
    double diff = input_abs - output_abs;
    if (diff < 0)
      diff = -diff;
    double max_abs = input_abs > output_abs ? input_abs : output_abs;

    if (diff > 0) {
      double rel_err = diff / max_abs;
      if (rel_err > 4e-5) {
        cout << "ERROR at abs. index: " << idx << ", ref: " << input_data[idx]
            << ", fact: " << output_data[idx] << endl;
        return false;
      }
    }
  }

  return true;
}

#define SIZE_0 384
#define SIZE_1 2320
#define SIZE_2 64

int main(int argc, char *argv[]) {
  // 3-dim
  // Prepare data
  cout << "Preparing data..." << endl;
  constexpr float alpha = 2.3f, beta = 4.2f;
  constexpr TensorIdx dim_0 = SIZE_0, dim_1 = SIZE_1, dim_2 = SIZE_2;
  constexpr TensorIdx len = dim_0 * dim_1 * dim_2;
  array<TensorIdx, 3> size_3 = { dim_0, dim_1, dim_2 };
  array<TensorOrder, 3> perm_3[2] = { { 2, 1, 0 }, { 1, 0, 2 } };

  // Allocate memory and fill in contents
  float *input_data = new float [len];
  float *output_data_0 = new float [len];
  float *output_data_1 = new float [len];

  reset_data(input_data, len);
  reset_data(output_data_0, len);
  reset_data(output_data_1, len);

  // Create reference
  float *ref_data_0 = new float [len];
  float *ref_data_1 = new float [len];

  cout << "Calculating reference time..." << endl;
  chrono::milliseconds ref_duration[2] = { {1000}, {1000} };
  for (TensorIdx idx = 0; idx < MEASURE_MAX; ++idx) {
    // Reset data
    copy(output_data_0, output_data_0 + len, ref_data_0);
    copy(output_data_1, output_data_1 + len, ref_data_1);

    auto start = chrono::high_resolution_clock::now();
    sTranspose_210_384x2320x64<SIZE_0, SIZE_1, SIZE_2>(input_data, ref_data_0,
        alpha, beta, nullptr, nullptr);
    auto diff = chrono::high_resolution_clock::now() - start;
    auto duration = chrono::duration_cast<chrono::milliseconds>(diff);
    if (duration < ref_duration[0])
      ref_duration[0] = duration;

    start = chrono::high_resolution_clock::now();
    sTranspose_102_384x2320x64<SIZE_0, SIZE_1, SIZE_2>(input_data, ref_data_1,
        alpha, beta, nullptr, nullptr);
    diff = chrono::high_resolution_clock::now() - start;
    duration = chrono::duration_cast<chrono::milliseconds>(diff);
    if (duration < ref_duration[1])
      ref_duration[1] = duration;
  }
  cout << "Ref. time: Perm. 2, 1, 0: " << ref_duration[0].count() << "ms."
      << endl;
  cout << "Ref. time: Perm. 1, 0, 2: " << ref_duration[1].count() << "ms."
      << endl;


  // Create actual data
  cout << "Creating actual data..." << endl;
  float *act_data_0 = new float [len];
  float *act_data_1 = new float [len];

  TensorSize<3> input_size(size_3);
  TensorSize<3> output_size_0({ dim_2, dim_1, dim_0 });
  TensorSize<3> output_size_1({ dim_1, dim_0, dim_2 });

  TensorWrapper<float, 3> input_tensor(input_size, input_data);
  TensorWrapper<float, 3> output_tensor_0(output_size_0, act_data_0);
  TensorWrapper<float, 3> output_tensor_1(output_size_1, act_data_1);

  // Create parameters
  cout << "Creating parameter..." << endl;
  auto param_0 = make_shared<ParamTrans<float, 3, CoefUsage::USE_BOTH>>(
      input_tensor, output_tensor_0, perm_3[0], alpha, beta);
  auto param_1 = make_shared<ParamTrans<float, 3, CoefUsage::USE_BOTH>>(
      input_tensor, output_tensor_1, perm_3[1], alpha, beta);

  // Build graph
  cout << "Building graph..." << endl;
  // Select kernels
  MacroTrans<float, KernelTransFull<float, CoefUsage::USE_BOTH>, 2, 2> macro(
      KernelTransFull<float, CoefUsage::USE_BOTH>(), alpha, beta);

  // Create for loop
  OpForTrans<3, ParamTrans<float, 3, CoefUsage::USE_BOTH>, decltype(macro)>
      for_loop_0(param_0), for_loop_1(param_1);
  for (TensorIdx idx = 0; idx < 3; ++idx) {
    for_loop_0.set_end(size_3[idx], idx);
    for_loop_1.set_end(size_3[idx], idx);
  }

  for_loop_0.set_step(macro.get_cont_len(), 0);
  for_loop_0.set_step(macro.get_ncont_len(), param_0->perm[0]);
  for_loop_1.set_step(macro.get_cont_len(), 0);
  for_loop_1.set_step(macro.get_ncont_len(), param_1->perm[0]);

  // Measure time
  cout << "Calculating actual time..." << endl;
  //cout << "From input wrapper: " << input_tensor[arr] << endl;
  chrono::milliseconds act_duration[2] = { {1000}, {1000} };
  for (TensorIdx idx = 0; idx < MEASURE_MAX; ++idx) {
    // Reset data
    copy(output_data_0, output_data_0 + len, act_data_0);
    copy(output_data_1, output_data_1 + len, act_data_1);

    auto start = chrono::high_resolution_clock::now();
    for_loop_0(macro);
    auto diff = chrono::high_resolution_clock::now() - start;
    auto duration = chrono::duration_cast<chrono::milliseconds>(diff);
    if (duration < act_duration[0])
      act_duration[0] = duration;

    start = chrono::high_resolution_clock::now();
    for_loop_1(macro);
    diff = chrono::high_resolution_clock::now() - start;
    duration = chrono::duration_cast<chrono::milliseconds>(diff);
    if (duration < act_duration[1])
      act_duration[1] = duration;
  }
  cout << "Actual time: Perm. 2, 1, 0: " << act_duration[0].count() << "ms."
      << endl;
  cout << "Actual time: Perm. 1, 0, 2: " << act_duration[1].count() << "ms."
      << endl;

  // Verification
  cout << "Verifying results..." << endl;
  if (verify(ref_data_0, act_data_0, len))
    cout << "Perm 2, 1, 0 SUCCEED!" << endl;
  else
    cout << "Perm 2, 1, 0 FAILED!" << endl;
  if (verify(ref_data_1, act_data_1, len))
    cout << "Perm 1, 0, 2 SUCCEED!" << endl;
  else
    cout << "Perm 1, 0, 2 FAILED!" << endl;

  // Release resource
  delete [] input_data;
  delete [] output_data_0;
  delete [] output_data_1;
  delete [] ref_data_0;
  delete [] ref_data_1;
  delete [] act_data_0;
  delete [] act_data_1;

  return 0;
}
