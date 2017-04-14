#pragma once
#ifndef HPTT_KERNELS_KERNEL_TRANS_H_
#define HPTT_KERNELS_KERNEL_TRANS_H_

#include <hptt/types.h>
#include <hptt/util/util_trans.h>
#include <hptt/kernels/macro_kernel_trans.h>


namespace hptt {

/*
 * Forward declaration
 */
template <typename TensorType, bool UPDATE_OUT>
struct ParamTrans;


/*
 * Kernel package
 */
template <typename FloatType,
          bool UPDATE_OUT>
struct KernelPackTrans {
  // Friend struct
  template <typename TensorType,
            bool UPDATE>
  friend struct ParamTrans;

  // Kernel number (linear kernel is count for 2)
  static constexpr TensorUInt KERNEL_NUM = 26;

  // Delete move/copy constructors
  KernelPackTrans(KernelPackTrans &&) = delete;
  KernelPackTrans<FloatType, UPDATE_OUT> &operator=(KernelPackTrans &&)
      = delete;
  KernelPackTrans(const KernelPackTrans &) = delete;
  KernelPackTrans<FloatType, UPDATE_OUT> &operator=(const KernelPackTrans &)
      = delete;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  TensorUInt kernel_offset(const KernelTypeTrans kn_type,
      const TensorUInt cont_size, const TensorUInt ncont_size) const;
  TensorUInt kn_cont_len(const KernelTypeTrans kn_type) const;
  TensorUInt kn_ncont_len(const KernelTypeTrans kn_type) const;

  const TensorUInt linear_loop_max;

  // Non-linear full kernels
  MacroTransFull<FloatType, 1, 1, UPDATE_OUT> knf_1x1;
  MacroTransFull<FloatType, 1, 2, UPDATE_OUT> knf_1x2;
  MacroTransFull<FloatType, 1, 3, UPDATE_OUT> knf_1x3;
  MacroTransFull<FloatType, 1, 4, UPDATE_OUT> knf_1x4;
  MacroTransFull<FloatType, 2, 1, UPDATE_OUT> knf_2x1;
  MacroTransFull<FloatType, 2, 2, UPDATE_OUT> knf_2x2;
  MacroTransFull<FloatType, 2, 3, UPDATE_OUT> knf_2x3;
  MacroTransFull<FloatType, 2, 4, UPDATE_OUT> knf_2x4;
  MacroTransFull<FloatType, 3, 1, UPDATE_OUT> knf_3x1;
  MacroTransFull<FloatType, 3, 2, UPDATE_OUT> knf_3x2;
  MacroTransFull<FloatType, 3, 3, UPDATE_OUT> knf_3x3;
  MacroTransFull<FloatType, 3, 4, UPDATE_OUT> knf_3x4;
  MacroTransFull<FloatType, 4, 1, UPDATE_OUT> knf_4x1;
  MacroTransFull<FloatType, 4, 2, UPDATE_OUT> knf_4x2;
  MacroTransFull<FloatType, 4, 3, UPDATE_OUT> knf_4x3;
  MacroTransFull<FloatType, 4, 4, UPDATE_OUT> knf_4x4;

  // Non-linear half kernels
  MacroTransHalf<FloatType, 1, 1, UPDATE_OUT> knh_1x1;
  MacroTransHalf<FloatType, 1, 2, UPDATE_OUT> knh_1x2;
  MacroTransHalf<FloatType, 1, 3, UPDATE_OUT> knh_1x3;
  MacroTransHalf<FloatType, 1, 4, UPDATE_OUT> knh_1x4;
  MacroTransHalf<FloatType, 2, 1, UPDATE_OUT> knh_2x1;
  MacroTransHalf<FloatType, 3, 1, UPDATE_OUT> knh_3x1;
  MacroTransHalf<FloatType, 4, 1, UPDATE_OUT> knh_4x1;

  // Linear kernel
  MacroTransLinear<FloatType, UPDATE_OUT> kn_lin_core;
  MacroTransLinear<FloatType, UPDATE_OUT> kn_lin_right;
  MacroTransLinear<FloatType, UPDATE_OUT> kn_lin_bottom;
  MacroTransLinear<FloatType, UPDATE_OUT> kn_lin_scalar;

  // Scalar kernel
  MacroTransScalar<FloatType, UPDATE_OUT> kn_sca_right;
  MacroTransScalar<FloatType, UPDATE_OUT> kn_sca_bottom;
  MacroTransScalar<FloatType, UPDATE_OUT> kn_sca_scalar;

  // Reference kernels
  const MacroTransFull<FloatType, 4, 4, UPDATE_OUT> &knf_giant;
  const MacroTransFull<FloatType, 1, 1, UPDATE_OUT> &knf_basic;
  const MacroTransHalf<FloatType, 1, 4, UPDATE_OUT> &knh_giant;
  const MacroTransHalf<FloatType, 1, 1, UPDATE_OUT> &knh_basic;

private:
  KernelPackTrans();
};


/*
 * Explicit instantiation declaration for struct KernelPackTrans
 */
extern template struct KernelPackTrans<float, true>;
extern template struct KernelPackTrans<double, true>;
extern template struct KernelPackTrans<FloatComplex, true>;
extern template struct KernelPackTrans<DoubleComplex, true>;

extern template struct KernelPackTrans<float, false>;
extern template struct KernelPackTrans<double, false>;
extern template struct KernelPackTrans<FloatComplex, false>;
extern template struct KernelPackTrans<DoubleComplex, false>;

}

#endif // HPTT_KERNELS_KERNEL_TRANS_H_
