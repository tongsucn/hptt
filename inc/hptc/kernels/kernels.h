#pragma once
#ifndef HPTC_KERNELS_KERNELS_H_
#define HPTC_KERNELS_KERNELS_H_

namespace hptc {

template <typename FloatType,
          GenNumType HEIGHT,
          GenNumType WIDTH,
          MemLayout LAYOUT = MemLayout::COL_MAJOR>
class MacroBase {
public:
  MacroBase() = default;
  MacroBase(const MacroBase &operation) = delete;
  MacroBase &operator=(const MacroBase &operation) = delete;
  virtual ~MacroBase();

  virtual INLINE void exec() = 0;
};

}

#endif // HPTC_KERNELS_KERNELS_H_
