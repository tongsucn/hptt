#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

/*
 * Forward declaration
 */
template <typename TensorType>
struct ParamTrans;


/*
 * Kernel package
 */
template <typename FloatType>
struct KernelPackTrans {
  // Friend struct
  template <typename TensorType>
  friend struct ParamTrans;

  // Kernel number (linear kernel is count for 2)
  static constexpr TensorUInt KERNEL_NUM = 25;

  // Delete move/copy constructors
  KernelPackTrans(KernelPackTrans &&) = delete;
  KernelPackTrans<FloatType> &operator=(KernelPackTrans &&) = delete;
  KernelPackTrans(const KernelPackTrans &) = delete;
  KernelPackTrans<FloatType> &operator=(const KernelPackTrans &)
      = delete;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  TensorUInt kernel_offset(const KernelTypeTrans kn_type,
      const TensorUInt cont_size, const TensorUInt ncont_size,
      const bool is_tail = false) const;
  TensorUInt kn_cont_len(const KernelTypeTrans kn_type) const;
  TensorUInt kn_ncont_len(const KernelTypeTrans kn_type) const;

  const TensorUInt linear_loop_max;

  // Non-linear full kernels
  MacroTransFull<FloatType, 1, 1> knf_1x1;
  MacroTransFull<FloatType, 1, 2> knf_1x2;
  MacroTransFull<FloatType, 1, 3> knf_1x3;
  MacroTransFull<FloatType, 1, 4> knf_1x4;
  MacroTransFull<FloatType, 2, 1> knf_2x1;
  MacroTransFull<FloatType, 2, 2> knf_2x2;
  MacroTransFull<FloatType, 2, 3> knf_2x3;
  MacroTransFull<FloatType, 2, 4> knf_2x4;
  MacroTransFull<FloatType, 3, 1> knf_3x1;
  MacroTransFull<FloatType, 3, 2> knf_3x2;
  MacroTransFull<FloatType, 3, 3> knf_3x3;
  MacroTransFull<FloatType, 3, 4> knf_3x4;
  MacroTransFull<FloatType, 4, 1> knf_4x1;
  MacroTransFull<FloatType, 4, 2> knf_4x2;
  MacroTransFull<FloatType, 4, 3> knf_4x3;
  MacroTransFull<FloatType, 4, 4> knf_4x4;

  // Non-linear half kernels
  MacroTransHalf<FloatType, 1, 1> knh_1x1;
  MacroTransHalf<FloatType, 1, 2> knh_1x2;
  MacroTransHalf<FloatType, 1, 3> knh_1x3;
  MacroTransHalf<FloatType, 1, 4> knh_1x4;
  MacroTransHalf<FloatType, 2, 1> knh_2x1;
  MacroTransHalf<FloatType, 3, 1> knh_3x1;
  MacroTransHalf<FloatType, 4, 1> knh_4x1;

  // Linear kernel
  MacroTransLinear<FloatType> kn_lin_core;
  MacroTransLinear<FloatType> kn_lin_right;
  MacroTransLinear<FloatType> kn_lin_bottom;
  MacroTransLinear<FloatType> kn_lin_scalar;

  // Scalar kernel
  MacroTransScalar<FloatType> kn_scl;

  // Reference kernels
  const MacroTransFull<FloatType, 4, 4> &knf_giant;
  const MacroTransFull<FloatType, 1, 1> &knf_basic;
  const MacroTransHalf<FloatType, 1, 4> &knh_giant;
  const MacroTransHalf<FloatType, 1, 1> &knh_basic;

private:
  KernelPackTrans();
};


/*
 * Explicit instantiation declaration for struct KernelPackTrans
 */
extern template struct KernelPackTrans<float>;
extern template struct KernelPackTrans<double>;
extern template struct KernelPackTrans<FloatComplex>;
extern template struct KernelPackTrans<DoubleComplex>;

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
