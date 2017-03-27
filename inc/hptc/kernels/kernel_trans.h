#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

/*
 * Forward declaration
 */
template <typename TensorType,
          CoefUsageTrans USAGE>
struct ParamTrans;


/*
 * Kernel package
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
struct KernelPackTrans {
  // Friend struct
  template <typename TensorType,
            CoefUsageTrans PARAM_COEF_USAGE>
  friend struct ParamTrans;

  // Kernel number (linear kernel is count for 2)
  static constexpr TensorUInt KERNEL_NUM = 25;

  // Delete move/copy constructors
  KernelPackTrans(KernelPackTrans &&) = delete;
  KernelPackTrans<FloatType, USAGE> &operator=(KernelPackTrans &&) = delete;
  KernelPackTrans(const KernelPackTrans &) = delete;
  KernelPackTrans<FloatType, USAGE> &operator=(const KernelPackTrans &)
      = delete;

  void set_coef(const DeducedFloatType<FloatType> alpha,
      const DeducedFloatType<FloatType> beta);

  TensorUInt kernel_offset(const KernelTypeTrans kn_type,
      const TensorUInt cont_size, const TensorUInt ncont_size,
      const bool is_tail = false) const;
  TensorUInt kn_cont_len(const KernelTypeTrans kn_type,
      const TensorUInt cont_size) const;
  TensorUInt kn_ncont_len(const KernelTypeTrans kn_type,
      const TensorUInt ncont_size) const;


  // Non-linear full kernels
  MacroTransVecFull<FloatType, USAGE, 1, 1> knf_1x1;
  MacroTransVecFull<FloatType, USAGE, 1, 2> knf_1x2;
  MacroTransVecFull<FloatType, USAGE, 1, 3> knf_1x3;
  MacroTransVecFull<FloatType, USAGE, 1, 4> knf_1x4;
  MacroTransVecFull<FloatType, USAGE, 2, 1> knf_2x1;
  MacroTransVecFull<FloatType, USAGE, 2, 2> knf_2x2;
  MacroTransVecFull<FloatType, USAGE, 2, 3> knf_2x3;
  MacroTransVecFull<FloatType, USAGE, 2, 4> knf_2x4;
  MacroTransVecFull<FloatType, USAGE, 3, 1> knf_3x1;
  MacroTransVecFull<FloatType, USAGE, 3, 2> knf_3x2;
  MacroTransVecFull<FloatType, USAGE, 3, 3> knf_3x3;
  MacroTransVecFull<FloatType, USAGE, 3, 4> knf_3x4;
  MacroTransVecFull<FloatType, USAGE, 4, 1> knf_4x1;
  MacroTransVecFull<FloatType, USAGE, 4, 2> knf_4x2;
  MacroTransVecFull<FloatType, USAGE, 4, 3> knf_4x3;
  MacroTransVecFull<FloatType, USAGE, 4, 4> knf_4x4;

  // Non-linear half kernels
  MacroTransVecHalf<FloatType, USAGE, 1, 1> knh_1x1;
  MacroTransVecHalf<FloatType, USAGE, 1, 2> knh_1x2;
  MacroTransVecHalf<FloatType, USAGE, 1, 3> knh_1x3;
  MacroTransVecHalf<FloatType, USAGE, 1, 4> knh_1x4;
  MacroTransVecHalf<FloatType, USAGE, 2, 1> knh_2x1;
  MacroTransVecHalf<FloatType, USAGE, 3, 1> knh_3x1;
  MacroTransVecHalf<FloatType, USAGE, 4, 1> knh_4x1;

  // Linear kernel
  MacroTransLinear<FloatType, USAGE> kn_lin;

  // Reference kernels
  const MacroTransVecFull<FloatType, USAGE, 4, 4> &knf_giant;
  const MacroTransVecFull<FloatType, USAGE, 1, 1> &knf_basic;
  const MacroTransVecHalf<FloatType, USAGE, 1, 4> &knh_giant;
  const MacroTransVecHalf<FloatType, USAGE, 1, 1> &knh_basic;

private:
  KernelPackTrans();
};


/*
 * Import implementation and explicit instantiation declaration
 */
#include "kernel_trans.tcc"

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
