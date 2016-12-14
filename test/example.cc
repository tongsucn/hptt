#include <memory>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iostream>
#include <chrono>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/param/parameter_trans.h>
#include <hptc/operations/operation_trans.h>


using namespace std;
using namespace hptc;


template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH>
class BenchmarkCreator {
public:
  using Deduced = DeducedFloatType<FloatType>;
  using OpBase = Operation<FloatType, ParamTrans>;

  BenchmarkCreator(vector<TensorIdx> input_size,
      const vector<TensorDim> &perm, DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta) {
    // Prepare internal data member
    this->micro_size_ = 32 / sizeof(FloatType);
    this->macro_height_ = HEIGHT * this->micro_size_;
    this->macro_width_ = WIDTH * this->micro_size_;

    // Create tensor size objects
    TensorDim tensor_dim = static_cast<TensorDim>(input_size.size());
    TensorSize input_size_obj(input_size), output_size_obj(tensor_dim);
    for (TensorDim idx = 0; idx < tensor_dim; ++idx)
      output_size_obj[idx] = input_size_obj[perm[idx]];

    // Create raw data and initialize value
    this->input_data_len = this->output_data_len = accumulate(
        input_size.begin(), input_size.end(), 1, multiplies<TensorIdx>());
    this->input_data = new FloatType [this->input_data_len];
    this->output_data = new FloatType [this->output_data_len];
    this->reset_data();

    // Initialize tensor wrapper
    this->input_tensor = TensorWrapper<FloatType>(input_size_obj,
        this->input_data);
    this->output_tensor = TensorWrapper<FloatType>(output_size_obj,
        this->output_data);

    // Initialize transpose parameter
    this->param = make_shared<ParamTrans<FloatType>>(this->input_tensor,
        this->output_tensor, perm, alpha, beta);

    // Initialization computational graph
    this->build_graph();
  }


  ~BenchmarkCreator() {
    delete [] this->input_data;
    delete [] this->output_data;
  }


  void reset_data() {
    constexpr TensorIdx inner_offset = sizeof(FloatType) / sizeof(Deduced);
    // Reset input data
    for (TensorIdx idx = 0; idx < this->input_data_len; ++idx) {
      Deduced *reset_ptr = reinterpret_cast<Deduced *>(&this->input_data[idx]);
      for (TensorIdx inner_idx = 0; inner_idx < inner_offset; ++inner_idx)
        reset_ptr[inner_idx] = static_cast<Deduced>(idx);
    }

    // Reset output data
    Deduced *reset_ptr = reinterpret_cast<Deduced *>(this->output_data);
    fill(reset_ptr, reset_ptr + this->output_data_len * inner_offset,
        static_cast<float>(-1));
  }


  shared_ptr<OpBase> build_graph() {
    using SingleFor = OpLoopFor<FloatType, ParamTrans, 1>;
    if (nullptr == this->input_data or nullptr == output_data)
      return nullptr;

    // Get dimension informaiton
    TensorDim tensor_dim = this->input_tensor.get_size().get_dim();

    // Create for loops
    shared_ptr<SingleFor> curr_op
        = static_pointer_cast<SingleFor>(this->operation);
    for (TensorDim idx = 0; idx < tensor_dim; ++idx) {
      // Compute parameters
      TensorIdx begin = 0, end, step;
      if (0 == idx) {
        end = this->input_tensor.get_size()[0];
        step = this->macro_height_;
      }
      else if (this->param->perm[0] == idx) {
        end = this->input_tensor.get_size()[idx];
        step = this->macro_width_;
      }
      else {
        end = this->input_tensor.get_size()[idx];
        step = 1;
      }

      // Create a new for loop
      shared_ptr<SingleFor> new_op = make_shared<SingleFor>(this->param,
          this->param->macro_loop_idx[idx], begin, end, step);

      // Connect new for loop with previous one
      if (nullptr == curr_op)
        this->operation = new_op;
      else
        curr_op->init_operation(new_op);
      curr_op = new_op;
    }

    // Create macro kernel
    curr_op->init_operation(
        make_shared<OpMacroTrans<FloatType, HEIGHT, WIDTH>>(this->param));

    return this->operation;
  }


  inline void exec() {
    this->operation->exec();
  }


  TensorWrapper<FloatType> &get_input_tensor() {
    return this->input_tensor;
  }


  TensorWrapper<FloatType> &get_output_tensor() {
    return this->output_tensor;
  }

private:
  TensorIdx micro_size_, macro_height_, macro_width_;

  TensorIdx input_data_len, output_data_len;
  FloatType *input_data, *output_data;
  TensorWrapper<FloatType> input_tensor, output_tensor;
  shared_ptr<ParamTrans<FloatType>> param;
  shared_ptr<OpBase> operation;
};


int main(int argc, char *argv[]) {
  // Prepare data
  vector<TensorIdx> size{ 384, 2320, 64 };
  vector<TensorDim> perm{ 2, 1, 0 };

  // Create transpose computational graph
  BenchmarkCreator<float, 2, 2> inst(size, perm, 2.3, 4.2);

  // Execute transpose
  inst.reset_data();
  auto input_tensor = inst.get_input_tensor();
  auto output_tensor = inst.get_output_tensor();
  auto begin_time = chrono::high_resolution_clock::now();
  inst.exec();
  auto time_diff = chrono::high_resolution_clock::now() - begin_time;
  auto ms = chrono::duration_cast<chrono::milliseconds>(time_diff).count();
  cout << "Elapsed time: " << ms << "ms." << endl;

  // Verification
  for (TensorIdx idx_0 = 0; idx_0 < 384; ++idx_0) {
    for (TensorIdx idx_1 = 0; idx_1 < 2320; ++idx_1) {
      for (TensorIdx idx_2 = 0; idx_2 < 64; ++idx_2) {
        if (0 != input_tensor(idx_0, idx_1, idx_2) * 2.3f - 4.2f
            - output_tensor(idx_2, idx_1, idx_0)) {
          cout << "Result does not match at: " << idx_0 << ", " << idx_1
              << ", " << idx_2 << endl;
          cout << "Input: " << input_tensor(idx_0, idx_1, idx_2) << endl;
          cout << "Output: " << output_tensor(idx_2, idx_1, idx_0) << endl;
          return -1;
        }
      }
    }
  }

  cout << "Transpose done!" << endl;
  return 0;
}
