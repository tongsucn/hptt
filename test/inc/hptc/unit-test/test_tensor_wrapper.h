#pragma once
#ifndef HPTC_UNIT_TEST_TEST_TENSOR_WRAPPER_H_
#define HPTC_UNIT_TEST_TEST_TENSOR_WRAPPER_H_

#include <array>
#include <vector>
#include <algorithm>

#include <gtest/gtest.h>

#include <hptc/test_util.h>
#include <hptc/types.h>
#include <hptc/tensor.h>

using namespace std;
using namespace hptc;

#define TEST_ORDER 4


template <typename FloatType>
class TestTensorWrapper : public ::testing::Test {
protected:
  using Deduced = DeducedFloatType<FloatType>;

  TestTensorWrapper()
      : size({ 21, 17, 8, 29 }),
        outer_size({ 23, 17, 12, 34 }),
        size_obj(this->size),
        outer_size_obj(this->outer_size),
        offsets({ 1, 0, 4, 2 }) {
  }

  template <MemLayout LAYOUT>
  class TestCreation : public TensorWrapper<FloatType, TEST_ORDER, LAYOUT> {
  public:
    TestCreation() = default;

    TestCreation(const TensorSize<TEST_ORDER> &size_obj, FloatType *raw_data)
        : TensorWrapper<FloatType, TEST_ORDER, LAYOUT>(size_obj, raw_data) {
    }

    TestCreation(const TensorSize<TEST_ORDER> &size_obj,
        const TensorSize<TEST_ORDER> &outer_size_obj,
        const array<TensorIdx, TEST_ORDER> &order_offset, FloatType *raw_data)
        : TensorWrapper<FloatType, TEST_ORDER, LAYOUT>(size_obj, outer_size_obj,
              order_offset, raw_data) {
    }


    bool check_strides(const array<TensorIdx, TEST_ORDER> &ref_strides) {
      for (TensorIdx idx = 0; idx < TEST_ORDER; ++idx)
        if (ref_strides[idx] != this->strides_[idx])
          return false;
      return true;
    }
  };

  template <MemLayout LAYOUT>
  class TestIndexing {
  public:
    TestIndexing(const vector<TensorOrder> &size)
        : data(size) {
      for (TensorIdx idx = 0; idx < data.data_len; ++idx) {
        auto ptr = reinterpret_cast<Deduced *>(data.act_data);
        for (TensorOrder in_idx = 0; in_idx < inner_offset; ++in_idx)
          ptr[in_idx] = static_cast<Deduced>(idx);
      }
    }

    TensorIdx exec_test(const TensorSize<TEST_ORDER> &size_obj,
        const TensorSize<TEST_ORDER> &outer_size_obj) {
      array<TensorIdx, TEST_ORDER> inner_strides, outer_strides;
      if (MemLayout::COL_MAJOR == LAYOUT) {
        inner_strides[0] = outer_strides[0] = 1;
        for (TensorIdx idx = 1; idx < TEST_ORDER; ++idx) {
          inner_strides[idx] = inner_strides[idx - 1] * size_obj[idx - 1];
          outer_strides[idx] = outer_strides[idx - 1] * outer_size_obj[idx - 1];
        }
      }
      else {
        inner_strides[TEST_ORDER - 1] = outer_strides[TEST_ORDER - 1] = 1;
        for (TensorIdx idx = TEST_ORDER - 2; idx >= 0; --idx) {
          inner_strides[idx] = inner_strides[idx + 1] * size_obj[idx + 1];
          outer_strides[idx] = outer_strides[idx + 1] * outer_size_obj[idx + 1];
        }
      }

      array<TensorIdx, TEST_ORDER> local_offsets;
      if (size_obj == outer_size_obj)
        fill(local_offsets.begin(), local_offsets.end(), 0);
      else
        copy(offsets.begin(), offsets.end(), local_offsets.begin());

      TensorWrapper<FloatType, TEST_ORDER, LAYOUT> tensor(size_obj,
          outer_size_obj, local_offsets, this->data.act_data);

      auto &tensor_size = tensor.get_size();

      for (TensorIdx idx_0 = 0; idx_0 < tensor_size[0]; ++idx_0) {
        TensorIdx offset = idx_0;
        for (TensorIdx idx_1 = 0; idx_1 < tensor_size[1]; ++idx_1) {
          offset += idx_1 * inner_strides[1];
          for (TensorIdx idx_2 = 0; idx_2 < tensor_size[2]; ++idx_2) {
            offset += idx_2 * inner_strides[2];
            for (TensorIdx idx_3 = 0; idx_3 < tensor_size[3]; ++idx_3) {
              offset += idx_3 * inner_strides[3];
              auto in_ptr = reinterpret_cast<Deduced *>(
                  &tensor(idx_0, idx_1, idx_2, idx_3));
              for (TensorOrder in_idx = 0; in_idx < this->inner_offset; ++in_idx)
                in_ptr[in_idx] = static_cast<Deduced>(offset);
            }
          }
        }
      }
    }

  private:
    TestDataWrapper<FloatType> data;
  };

  static const TensorOrder inner_offset = sizeof(FloatType) / sizeof(Deduced);
  vector<TensorOrder> size, outer_size;
  TensorSize<TEST_ORDER> size_obj, outer_size_obj;
  array<TensorIdx, TEST_ORDER> offsets;
};


using TestFloats = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestTensorWrapper, TestFloats);


TYPED_TEST(TestTensorWrapper, CreationColMajor) {
  using CaseGenerator = typename TestTensorWrapper<TypeParam>::
      TestCreation<MemLayout::COL_MAJOR>;

  array<TensorIdx, TEST_ORDER> inner_strides, outer_strides;
  inner_strides[0] = outer_strides[0] = 1;
  for (TensorIdx idx = 1; idx < TEST_ORDER; ++idx) {
    inner_strides[idx] = inner_strides[idx - 1] * this->size_obj[idx - 1];
    outer_strides[idx] = outer_strides[idx - 1] * this->outer_size_obj[idx - 1];
  }

  // Construction from one size object
  CaseGenerator no_outer(this->size_obj, nullptr);
  ASSERT_TRUE(no_outer.check_strides(inner_strides))
      << "Tensor wrapper's indexing strides initialization is not correct when "
      << "using column major layout and no outer size constructor. Expected "
      << "value: " << inner_strides[0] << ", " << inner_strides[1] << ", "
      << inner_strides[2] << ", " << inner_strides[3];

  // Construction from two different size objects
  CaseGenerator with_outer(this->size_obj, this->outer_size, this->offsets,
      nullptr);
  ASSERT_TRUE(with_outer.check_strides(outer_strides))
      << "Tensor wrapper's indexing strides initialization is not correct when "
      << "using column major layout and with outer size constructor. Expected "
      << "value: " << outer_strides[0] << ", " << outer_strides[1] << ", "
      << outer_strides[2] << ", " << outer_strides[3];

  // Construction from two same size objects
  CaseGenerator same_outer(this->size_obj, this->size_obj, this->offsets,
      nullptr);
  ASSERT_TRUE(no_outer.check_strides(inner_strides))
      << "Tensor wrapper's indexing strides initialization is not correct when "
      << "using column major layout and with outer size constructor, the outer "
      << "size object is the same with the size object. Expected value: "
      << inner_strides[0] << ", " << inner_strides[1] << ", "
      << inner_strides[2] << ", " << inner_strides[3];
}


TYPED_TEST(TestTensorWrapper, CreationRowMajor) {
  using CaseGenerator = typename TestTensorWrapper<TypeParam>::
      TestCreation<MemLayout::ROW_MAJOR>;

  array<TensorIdx, TEST_ORDER> inner_strides, outer_strides;
  inner_strides[3] = outer_strides[3] = 1;
  for (TensorIdx idx = 2; idx >= 0; --idx) {
    inner_strides[idx] = inner_strides[idx + 1] * this->size_obj[idx + 1];
    outer_strides[idx] = outer_strides[idx + 1] * this->outer_size_obj[idx + 1];
  }

  // Construction from one size object
  CaseGenerator no_outer(this->size_obj, nullptr);
  ASSERT_TRUE(no_outer.check_strides(inner_strides))
      << "Tensor wrapper's indexing strides initialization is not correct when "
      << "using row major layout and no outer size constructor. Expected "
      << "value: " << inner_strides[0] << ", " << inner_strides[1] << ", "
      << inner_strides[2] << ", " << inner_strides[3];

  // Construction from two different size objects
  CaseGenerator with_outer(this->size_obj, this->outer_size, this->offsets,
      nullptr);
  ASSERT_TRUE(with_outer.check_strides(outer_strides))
      << "Tensor wrapper's indexing strides initialization is not correct when "
      << "using row major layout and with outer size constructor. Expected "
      << "value: " << outer_strides[0] << ", " << outer_strides[1] << ", "
      << outer_strides[2] << ", " << outer_strides[3];

  // Construction from two same size objects
  CaseGenerator same_outer(this->size_obj, this->size_obj, this->offsets,
      nullptr);
  ASSERT_TRUE(no_outer.check_strides(inner_strides))
      << "Tensor wrapper's indexing strides initialization is not correct when "
      << "using row major layout and with outer size constructor, the outer "
      << "size object is the same with the size object. Expected value: "
      << inner_strides[0] << ", " << inner_strides[1] << ", "
      << inner_strides[2] << ", " << inner_strides[3];
}


TYPED_TEST(TestTensorWrapper, IndexingColMajor) {
  using Deduced = DeducedFloatType<TypeParam>;

  TestDataWrapper<TypeParam> data(this->size);
  array<TensorIdx, TEST_ORDER> inner_strides;
  inner_strides[0] = 1;
  for (TensorIdx idx = 1; idx < TEST_ORDER; ++idx)
    inner_strides[idx] = inner_strides[idx - 1] * this->size_obj[idx - 1];

  TensorWrapper<TypeParam, TEST_ORDER, MemLayout::COL_MAJOR>
      tensor(this->size_obj, data.act_data);
}


TYPED_TEST(TestTensorWrapper, IndexingRowMajor) {
}


TYPED_TEST(TestTensorWrapper, SubIndexingColMajor) {
}


TYPED_TEST(TestTensorWrapper, SubIndexingRowMajor) {
}


TYPED_TEST(TestTensorWrapper, Slicing) {
  ;
}

#endif // HPTC_UNIT_TEST_TEST_TENSOR_WRAPPER_H_
