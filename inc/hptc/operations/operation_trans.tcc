#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Implementation for class OpForTransData
 */
template <TensorOrder ORDER,
          typename ParamType,
          typename MacroType>
OpForTransData<ORDER, ParamType, MacroType>::OpForTransData(
    std::shared_ptr<ParamType> param, MacroType macro_kernel)
    : param_(param),
      macro_kernel_(macro_kernel),
      input_stride_(param->input_stride_),
      output_stride_(param->output_stride_) {
  // Initialize loop indices
  std::fill(this->loop_idx_, this->loop_idx_ + ORDER, 0);
  for (TensorOrder idx = 0; idx < ORDER; ++idx)
    this->loop_perm_idx_[idx] = &this->loop_idx_[param->perm[idx]];

  // Initialize loop variables
  std::copy(this->loop_idx_, this->loop_idx_ + ORDER, this->loop_begin_);
  std::fill(this->loop_step_, this->loop_step_ + ORDER, 1);
}


template <TensorOrder ORDER,
          typename ParamType,
          typename MacroType>
INLINE void OpForTransData<ORDER, ParamType, MacroType>::set_begin(
    TensorIdx begin_val, TensorIdx idx) {
  this->loop_begin_[idx] = begin_val;
}


template <TensorOrder ORDER,
          typename ParamType,
          typename MacroType>
INLINE void OpForTransData<ORDER, ParamType, MacroType>::set_end(
    TensorIdx end_val, TensorIdx idx) {
  this->loop_end_[idx] = end_val;
}


template <TensorOrder ORDER,
          typename ParamType,
          typename MacroType>
INLINE void OpForTransData<ORDER, ParamType, MacroType>::set_step(
    TensorIdx step_val, TensorIdx idx) {
  this->loop_step_[idx] = step_val;
}


/*
 * Implementation for class OpForTrans
 */


/*
 * Specialization for class OpForTrans
 */
template <typename ParamType,
          typename MacroType>
class OpForTrans<0, ParamType, MacroType> final
    : public OpForTransData<0, ParamType, MacroType> {
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<1, ParamType, MacroType> final
    : public OpForTransData<1, ParamType, MacroType> {
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<2, ParamType, MacroType> final
    : public OpForTransData<2, ParamType, MacroType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param, MacroType macro_kernel)
      : OpForTransData<2, ParamType, MacroType>(param, macro_kernel) {
  }

  INLINE void operator()() {
    auto &input_tensor = this->param_->input_tensor;
    auto &output_tensor = this->param_->output_tensor;

    for (this->loop_idx_[0] = this->loop_begin_[0];
        this->loop_idx_[0] < this->loop_end_[0];
        this->loop_idx_[0] += this->loop_step_[0]) {
      for (this->loop_idx_[1] = this->loop_begin_[1];
          this->loop_idx_[1] < this->loop_end_[1];
          this->loop_idx_[1] += this->loop_step_[1]) {
        this->macro_kernel_(input_tensor[this->loop_idx_],
            output_tensor[this->loop_perm_idx_],
            this->input_stride_, this->output_stride_);
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<3, ParamType, MacroType> final
    : public OpForTransData<3, ParamType, MacroType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param, MacroType macro_kernel)
      : OpForTransData<3, ParamType, MacroType>(param, macro_kernel) {
  }

  INLINE void operator()() {
    auto &input_tensor = this->param_->input_tensor;
    auto &output_tensor = this->param_->output_tensor;

    for (this->loop_idx_[0] = this->loop_begin_[0];
        this->loop_idx_[0] < this->loop_end_[0];
        this->loop_idx_[0] += this->loop_step_[0]) {
      for (this->loop_idx_[1] = this->loop_begin_[1];
          this->loop_idx_[1] < this->loop_end_[1];
          this->loop_idx_[1] += this->loop_step_[1]) {
        for (this->loop_idx_[2] = this->loop_begin_[2];
            this->loop_idx_[2] < this->loop_end_[2];
            this->loop_idx_[2] += this->loop_step_[2]) {
          this->macro_kernel_(input_tensor[this->loop_idx_],
              output_tensor[this->loop_perm_idx_],
              this->input_stride_, this->output_stride_);
        }
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<4, ParamType, MacroType> final
    : public OpForTransData<4, ParamType, MacroType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param, MacroType macro_kernel)
      : OpForTransData<4, ParamType, MacroType>(param, macro_kernel) {
  }

  INLINE void operator()() {
    auto &input_tensor = this->param_->input_tensor;
    auto &output_tensor = this->param_->output_tensor;

    for (this->loop_idx_[0] = this->loop_begin_[0];
        this->loop_idx_[0] < this->loop_end_[0];
        this->loop_idx_[0] += this->loop_step_[0]) {
      for (this->loop_idx_[1] = this->loop_begin_[1];
          this->loop_idx_[1] < this->loop_end_[1];
          this->loop_idx_[1] += this->loop_step_[1]) {
        for (this->loop_idx_[2] = this->loop_begin_[2];
            this->loop_idx_[2] < this->loop_end_[2];
            this->loop_idx_[2] += this->loop_step_[2]) {
          for (this->loop_idx_[3] = this->loop_begin_[3];
              this->loop_idx_[3] < this->loop_end_[3];
              this->loop_idx_[3] += this->loop_step_[3]) {
            this->macro_kernel_(input_tensor[this->loop_idx_],
                output_tensor[this->loop_perm_idx_],
                this->input_stride_, this->output_stride_);
          }
        }
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<5, ParamType, MacroType> final
    : public OpForTransData<5, ParamType, MacroType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param, MacroType macro_kernel)
      : OpForTransData<5, ParamType, MacroType>(param, macro_kernel) {
  }

  INLINE void operator()() {
    auto &input_tensor = this->param_->input_tensor;
    auto &output_tensor = this->param_->output_tensor;

    for (this->loop_idx_[0] = this->loop_begin_[0];
        this->loop_idx_[0] < this->loop_end_[0];
        this->loop_idx_[0] += this->loop_step_[0]) {
      for (this->loop_idx_[1] = this->loop_begin_[1];
          this->loop_idx_[1] < this->loop_end_[1];
          this->loop_idx_[1] += this->loop_step_[1]) {
        for (this->loop_idx_[2] = this->loop_begin_[2];
            this->loop_idx_[2] < this->loop_end_[2];
            this->loop_idx_[2] += this->loop_step_[2]) {
          for (this->loop_idx_[3] = this->loop_begin_[3];
              this->loop_idx_[3] < this->loop_end_[3];
              this->loop_idx_[3] += this->loop_step_[3]) {
            for (this->loop_idx_[4] = this->loop_begin_[4];
                this->loop_idx_[4] < this->loop_end_[4];
                this->loop_idx_[4] += this->loop_step_[4]) {
              this->macro_kernel_(input_tensor[this->loop_idx_],
                  output_tensor[this->loop_perm_idx_],
                  this->input_stride_, this->output_stride_);
            }
          }
        }
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<6, ParamType, MacroType> final
    : public OpForTransData<6, ParamType, MacroType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param, MacroType macro_kernel)
      : OpForTransData<6, ParamType, MacroType>(param, macro_kernel) {
  }

  INLINE void operator()() {
    auto &input_tensor = this->param_->input_tensor;
    auto &output_tensor = this->param_->output_tensor;

    for (this->loop_idx_[0] = this->loop_begin_[0];
        this->loop_idx_[0] < this->loop_end_[0];
        this->loop_idx_[0] += this->loop_step_[0]) {
      for (this->loop_idx_[1] = this->loop_begin_[1];
          this->loop_idx_[1] < this->loop_end_[1];
          this->loop_idx_[1] += this->loop_step_[1]) {
        for (this->loop_idx_[2] = this->loop_begin_[2];
            this->loop_idx_[2] < this->loop_end_[2];
            this->loop_idx_[2] += this->loop_step_[2]) {
          for (this->loop_idx_[3] = this->loop_begin_[3];
              this->loop_idx_[3] < this->loop_end_[3];
              this->loop_idx_[3] += this->loop_step_[3]) {
            for (this->loop_idx_[4] = this->loop_begin_[4];
                this->loop_idx_[4] < this->loop_end_[4];
                this->loop_idx_[4] += this->loop_step_[4]) {
              for (this->loop_idx_[5] = this->loop_begin_[5];
                  this->loop_idx_[5] < this->loop_end_[5];
                  this->loop_idx_[5] += this->loop_step_[5]) {
                this->macro_kernel_(input_tensor[this->loop_idx_],
                    output_tensor[this->loop_perm_idx_],
                    this->input_stride_, this->output_stride_);
              }
            }
          }
        }
      }
    }
  }
};


#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
