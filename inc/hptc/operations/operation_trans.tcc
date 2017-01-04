#pragma once
#ifndef HPTC_OPERATIONS_OPERATION_TRANS_TCC_
#define HPTC_OPERATIONS_OPERATION_TRANS_TCC_

/*
 * Implementation for class OpForTransData
 */
template <TensorOrder ORDER,
          typename ParamType>
OpForTransData<ORDER, ParamType>::OpForTransData(
    std::shared_ptr<ParamType> param) : param_(param) {
  // Initialize loop variables
  std::fill(this->loop_idx_, this->loop_idx_ + ORDER, 0);
  std::copy(this->loop_idx_, this->loop_idx_ + ORDER, this->loop_begin_);
  std::fill(this->loop_step_, this->loop_step_ + ORDER, 1);

  // Initialize loop indices
  for (TensorOrder idx = ORDER - param->merged_order; idx < ORDER; ++idx)
    this->loop_perm_idx_[idx]
        = &this->loop_idx_[param->perm[idx] + ORDER - param->merged_order];
}


template <TensorOrder ORDER,
          typename ParamType>
INLINE void OpForTransData<ORDER, ParamType>::set_begin(
    TensorIdx begin_val, TensorIdx idx) {
  this->loop_begin_[idx] = begin_val;
}


template <TensorOrder ORDER,
          typename ParamType>
INLINE void OpForTransData<ORDER, ParamType>::set_end(
    TensorIdx end_val, TensorIdx idx) {
  this->loop_end_[idx] = end_val;
}


template <TensorOrder ORDER,
          typename ParamType>
INLINE void OpForTransData<ORDER, ParamType>::set_step(
    TensorIdx step_val, TensorIdx idx) {
  this->loop_step_[idx] = step_val;
}


/*
 * Specialization for class OpForTrans
 */
template <typename ParamType,
          typename MacroType>
class OpForTrans<0, ParamType, MacroType> final
    : public OpForTransData<0, ParamType> {
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<1, ParamType, MacroType> final
    : public OpForTransData<1, ParamType> {
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<2, ParamType, MacroType> final
    : public OpForTransData<2, ParamType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param)
      : OpForTransData<2, ParamType>(param) {
  }

  INLINE void operator()(MacroType &macro_kernel) {
    auto &input_tensor = this->param_->input_tensor;
    auto &output_tensor = this->param_->output_tensor;

    for (this->loop_idx_[0] = this->loop_begin_[0];
        this->loop_idx_[0] < this->loop_end_[0];
        this->loop_idx_[0] += this->loop_step_[0]) {
      for (this->loop_idx_[1] = this->loop_begin_[1];
          this->loop_idx_[1] < this->loop_end_[1];
          this->loop_idx_[1] += this->loop_step_[1]) {
        macro_kernel(&input_tensor[this->loop_idx_],
            &output_tensor[this->loop_perm_idx_],
            this->param_->input_stride, this->param_->output_stride);
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<3, ParamType, MacroType> final
    : public OpForTransData<3, ParamType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param)
      : OpForTransData<3, ParamType>(param) {
  }

  INLINE void operator()(MacroType &macro_kernel) {
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
          macro_kernel(&input_tensor[this->loop_idx_],
              &output_tensor[this->loop_perm_idx_],
              this->param_->input_stride, this->param_->output_stride);
        }
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<4, ParamType, MacroType> final
    : public OpForTransData<4, ParamType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param)
      : OpForTransData<4, ParamType>(param) {
  }

  INLINE void operator()(MacroType &macro_kernel) {
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
            macro_kernel(&input_tensor[this->loop_idx_],
                &output_tensor[this->loop_perm_idx_],
                this->param_->input_stride, this->param_->output_stride);
          }
        }
      }
    }
  }
};


template <typename ParamType,
          typename MacroType>
class OpForTrans<5, ParamType, MacroType> final
    : public OpForTransData<5, ParamType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param)
      : OpForTransData<5, ParamType>(param) {
  }

  INLINE void operator()(MacroType &macro_kernel) {
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
              macro_kernel(&input_tensor[this->loop_idx_],
                  &output_tensor[this->loop_perm_idx_],
                  this->param_->input_stride, this->param_->output_stride);
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
    : public OpForTransData<6, ParamType> {
public:
  OpForTrans(std::shared_ptr<ParamType> param)
      : OpForTransData<6, ParamType>(param) {
  }

  INLINE void operator()(MacroType &macro_kernel) {
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
                macro_kernel(&input_tensor[this->loop_idx_],
                    &output_tensor[this->loop_perm_idx_],
                    this->param_->input_stride, this->param_->output_stride);
              }
            }
          }
        }
      }
    }
  }
};


#endif // HPTC_OPERATIONS_OPERATION_TRANS_TCC_
