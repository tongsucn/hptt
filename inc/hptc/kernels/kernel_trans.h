#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/util/util_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

/*
 * Kernel package, singleton
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
struct KernelPackTrans {
  template <TensorUInt CONT_LEN,
            TensorUInt NCONT_LEN>
  using KernelFull = MacroTransVecFull<FloatType, USAGE, CONT_LEN, NCONT_LEN>;
  template <TensorUInt CONT_LEN,
            TensorUInt NCONT_LEN>
  using KernelHalf = MacroTransVecHalf<FloatType, USAGE, CONT_LEN, NCONT_LEN>;

  using RegTypeFull = typename KernelFull<1, 1>::RegType;
  using RegTypeHalf = typename KernelHalf<1, 1>::RegType;
  using RegTypeLinear = typename MacroTransLinear<FloatType, USAGE>::RegType;


  KernelPackTrans(const KernelPackTrans<FloatType, USAGE> &) = delete;
  KernelPackTrans<FloatType, USAGE> &operator=(
      const KernelPackTrans<FloatType, USAGE> &) = delete;

  // Static member functions
  static constexpr TensorUInt KERNEL_NUM = 25;
  static KernelPackTrans<FloatType, USAGE> &get_package();

  // Register types
  static RegTypeFull reg_coef_full(const DeducedFloatType<FloatType> coef);
  static RegTypeHalf reg_coef_half(const DeducedFloatType<FloatType> coef);
  static RegTypeLinear reg_coef_linear(const DeducedFloatType<FloatType> coef);

  // Non-static member functions
  TensorUInt kernel_offset(const KernelTypeTrans kn_type,
      const TensorUInt cont_size, const TensorUInt ncont_size,
      const bool is_tail = false) const;

  TensorUInt kn_cont_len(const KernelTypeTrans kn_type,
      const TensorUInt cont_size) const;
  TensorUInt kn_ncont_len(const KernelTypeTrans kn_type,
      const TensorUInt ncont_size) const;


  // Non-linear full kernels
  KernelFull<1, 1> knf_1x1;
  KernelFull<1, 2> knf_1x2;
  KernelFull<1, 3> knf_1x3;
  KernelFull<1, 4> knf_1x4;
  KernelFull<2, 1> knf_2x1;
  KernelFull<2, 2> knf_2x2;
  KernelFull<2, 3> knf_2x3;
  KernelFull<2, 4> knf_2x4;
  KernelFull<3, 1> knf_3x1;
  KernelFull<3, 2> knf_3x2;
  KernelFull<3, 3> knf_3x3;
  KernelFull<3, 4> knf_3x4;
  KernelFull<4, 1> knf_4x1;
  KernelFull<4, 2> knf_4x2;
  KernelFull<4, 3> knf_4x3;
  KernelFull<4, 4> knf_4x4;

  // Non-linear half kernels
  KernelHalf<1, 1> knh_1x1;
  KernelHalf<1, 2> knh_1x2;
  KernelHalf<1, 3> knh_1x3;
  KernelHalf<1, 4> knh_1x4;
  KernelHalf<2, 1> knh_2x1;
  KernelHalf<3, 1> knh_3x1;
  KernelHalf<4, 1> knh_4x1;

  // Linear kernel
  MacroTransLinear<FloatType, USAGE> kn_lin;

  // Reference kernels
  const KernelFull<4, 4> &knf_giant;
  const KernelFull<1, 1> &knf_basic;
  const KernelHalf<1, 4> &knh_giant;
  const KernelHalf<1, 1> &knh_basic;

private:
  KernelPackTrans();
};


/*
 * Import implementation and explicit instantiation declaration
 */
#include "kernel_trans.tcc"

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
