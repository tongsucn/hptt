#pragma once
#ifndef HPTC_KERNELS_KERNEL_TRANS_H_
#define HPTC_KERNELS_KERNEL_TRANS_H_

#include <hptc/types.h>
#include <hptc/config/config_trans.h>
#include <hptc/kernels/macro_kernel_trans.h>


namespace hptc {

/*
 * Kernel package
 */
template <typename FloatType,
          CoefUsageTrans USAGE>
struct KernelPackTrans {
  KernelPackTrans(DeducedFloatType<FloatType> alpha,
      DeducedFloatType<FloatType> beta);

  GenNumType kernel_offset(const KernelTypeTrans kn_type,
      const GenNumType cont_size, const GenNumType ncont_size,
      const bool is_tail = false) const;

  GenNumType kn_cont_len(const KernelTypeTrans kn_type,
      const GenNumType cont_size) const;
  GenNumType kn_ncont_len(const KernelTypeTrans kn_type,
      const GenNumType ncont_size) const;

  static constexpr GenNumType KERNEL_NUM = 25;

  // Non-linear full kernels
  MacroTransVecFull<FloatType, USAGE, 1, 1>         knf_1x1;
  MacroTransVecFull<FloatType, USAGE, 1, 2>         knf_1x2;
  MacroTransVecFull<FloatType, USAGE, 1, 3>         knf_1x3;
  MacroTransVecFull<FloatType, USAGE, 1, 4>         knf_1x4;
  MacroTransVecFull<FloatType, USAGE, 2, 1>         knf_2x1;
  MacroTransVecFull<FloatType, USAGE, 2, 2>         knf_2x2;
  MacroTransVecFull<FloatType, USAGE, 2, 3>         knf_2x3;
  MacroTransVecFull<FloatType, USAGE, 2, 4>         knf_2x4;
  MacroTransVecFull<FloatType, USAGE, 3, 1>         knf_3x1;
  MacroTransVecFull<FloatType, USAGE, 3, 2>         knf_3x2;
  MacroTransVecFull<FloatType, USAGE, 3, 3>         knf_3x3;
  MacroTransVecFull<FloatType, USAGE, 3, 4>         knf_3x4;
  MacroTransVecFull<FloatType, USAGE, 4, 1>         knf_4x1;
  MacroTransVecFull<FloatType, USAGE, 4, 2>         knf_4x2;
  MacroTransVecFull<FloatType, USAGE, 4, 3>         knf_4x3;
  MacroTransVecFull<FloatType, USAGE, 4, 4>         knf_4x4;

  // Non-linear full kernels
  MacroTransVecHalf<FloatType, USAGE, 1, 1>         knh_1x1;
  MacroTransVecHalf<FloatType, USAGE, 1, 2>         knh_1x2;
  MacroTransVecHalf<FloatType, USAGE, 1, 3>         knh_1x3;
  MacroTransVecHalf<FloatType, USAGE, 1, 4>         knh_1x4;
  MacroTransVecHalf<FloatType, USAGE, 2, 1>         knh_2x1;
  MacroTransVecHalf<FloatType, USAGE, 3, 1>         knh_3x1;
  MacroTransVecHalf<FloatType, USAGE, 4, 1>         knh_4x1;

  // Linear kernels
  MacroTransLinear<FloatType, USAGE>                kn_lin;

  // Reference kernels
  const MacroTransVecFull<FloatType, USAGE, 4, 4>   &knf_giant;
  const MacroTransVecFull<FloatType, USAGE, 1, 1>   &knf_basic;
  const MacroTransVecHalf<FloatType, USAGE, 1, 4>   &knh_giant;
  const MacroTransVecHalf<FloatType, USAGE, 1, 1>   &knh_basic;
};


/*
 * Import implementation and explicit instantiation declaration
 */
#include "kernel_trans.tcc"

}

#endif // HPTC_KERNELS_KERNEL_TRANS_H_
