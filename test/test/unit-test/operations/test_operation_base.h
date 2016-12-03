#pragma once
#ifndef TEST_UNIT_TEST_OPERATIONS_TEST_OPERATION_BASE_H_
#define TEST_UNIT_TEST_OPERATIONS_TEST_OPERATION_BASE_H_

#include <vector>
#include <memory>

#include <gtest/gtest.h>

#include <hptc/types.h>
#include <hptc/tensor.h>
#include <hptc/operations/operation_base.h>

using namespace std;
using namespace hptc;


template <typename FloatType>
class TestOperationBase : public ::testing::Test {
protected:
  template <typename ParamFloatType = FloatType>
  struct ParamCube {
    ParamCube(TensorIdx idx_0, TensorIdx idx_1, TensorIdx idx_2)
        : idx_0(-idx_0), idx_1(-idx_1), idx_2(-idx_2),
          raw_data(new ParamFloatType [idx_0 * idx_1 * idx_2]),
          tensor(TensorSize({ idx_0, idx_1, idx_2 }), raw_data) {
      for (TensorIdx idx = 0; idx < idx_0 * idx_1 * idx_2; ++idx)
        this->raw_data[idx] = idx;
    }

    ~ParamCube() {
      delete [] this->raw_data;
    }

    TensorIdx idx_0, idx_1, idx_2;
    ParamFloatType *raw_data;
    TensorWrapper<ParamFloatType> tensor;
  };

  template <typename OpFloatType = FloatType,
            template <typename> typename ParamType = ParamCube>
  class OpInc : public Operation<OpFloatType, ParamType> {
  public:
    OpInc(const shared_ptr<ParamType<OpFloatType>> &param)
        : Operation<OpFloatType, ParamType>(param) {
    }

    OpInc(const OpInc &operation) = default;
    OpInc &operator=(const OpInc &operation) = default;
    virtual ~OpInc() = default;

    virtual void exec() final {
      ++this->param->tensor(this->param->idx_0, this->param->idx_1,
          this->param->idx_2);
    }
  };


  using SingleFor = OpLoopFor<FloatType, ParamCube, 1>;
  using TripleFor = OpLoopFor<FloatType, ParamCube, 3>;

  TestOperationBase()
      : step_1_idx_0(2), step_1_idx_1(23), step_1_idx_2(233),
        step_2_idx_0(30), step_2_idx_1(20), step_2_idx_2(10),
        param_step_1(make_shared<ParamCube<FloatType>>(
            this->step_1_idx_0, this->step_1_idx_1, this->step_1_idx_2)),
        param_step_2(make_shared<ParamCube<FloatType>>(
            this->step_2_idx_0, this->step_2_idx_1, this->step_2_idx_2)),
        len_1(this->step_1_idx_0 * this->step_1_idx_1 * this->step_1_idx_2),
        len_2(this->step_2_idx_0 * this->step_2_idx_1 * this->step_2_idx_2) {
  }

  TensorIdx step_1_idx_0, step_1_idx_1, step_1_idx_2, step_2_idx_0,
      step_2_idx_1, step_2_idx_2;
  shared_ptr<ParamCube<FloatType>> param_step_1, param_step_2;
  TensorIdx len_1, len_2;
};


using FloatTypes = ::testing::Types<float, double, FloatComplex, DoubleComplex>;
TYPED_TEST_CASE(TestOperationBase, FloatTypes);


TYPED_TEST(TestOperationBase, TestOpLoopForSingleOper) {
  using OpInc = typename TestOperationBase<TypeParam>::OpInc<>;
  using SingleFor = typename TestOperationBase<TypeParam>::SingleFor;

  // Single step
  // Initialize operations
  SingleFor outer_for_step_1(this->param_step_1, this->param_step_1->idx_0,
      this->param_step_1->idx_0, -this->param_step_1->idx_0, 1);
  auto mid_for_step_1 = make_shared<SingleFor>(this->param_step_1,
      this->param_step_1->idx_1, this->param_step_1->idx_1,
      -this->param_step_1->idx_1, 1);
  auto inner_for_step_1 = make_shared<SingleFor>(this->param_step_1,
      this->param_step_1->idx_2, this->param_step_1->idx_2,
      -this->param_step_1->idx_2, 1);
  auto inc_obj_step_1 = make_shared<OpInc>(this->param_step_1);

  // Linking operations
  inner_for_step_1->init_operation(inc_obj_step_1);
  mid_for_step_1->init_operation(inner_for_step_1);
  outer_for_step_1.init_operation(mid_for_step_1);

  // Executing operations
  outer_for_step_1.exec();

  // Check results
  for (TensorIdx idx = 0; idx < this->len_1; ++idx) {
    ASSERT_EQ(static_cast<TypeParam>(idx + 8),
        this->param_step_1->raw_data[idx])
        << "Single step increment result does not match expectation, index: "
        << idx;
  }

  // Double step
  // Initialize operations
  SingleFor outer_for_step_2(this->param_step_2, this->param_step_2->idx_0,
      this->param_step_2->idx_0, -this->param_step_2->idx_0, 2);
  auto mid_for_step_2 = make_shared<SingleFor>(this->param_step_2,
      this->param_step_2->idx_1, this->param_step_2->idx_1,
      -this->param_step_2->idx_1, 2);
  auto inner_for_step_2 = make_shared<SingleFor>(this->param_step_2,
      this->param_step_2->idx_2, this->param_step_2->idx_2,
      -this->param_step_2->idx_2, 2);
  auto inc_obj_step_2 = make_shared<OpInc>(this->param_step_2);

  // Linking operations
  inner_for_step_2->init_operation(inc_obj_step_2);
  mid_for_step_2->init_operation(inner_for_step_2);
  outer_for_step_2.init_operation(mid_for_step_2);

  // Executing operations
  outer_for_step_2.exec();

  // Check results
  vector<TensorIdx> dim_offset{ this->step_2_idx_1 * this->step_2_idx_2,
      this->step_2_idx_2, 1 };
  for (TensorIdx idx_0 = 0; idx_0 < this->step_2_idx_0; ++idx_0)
    for (TensorIdx idx_1 = 0; idx_1 < this->step_2_idx_1; ++idx_1)
      for (TensorIdx idx_2 = 0; idx_2 < this->step_2_idx_2; ++idx_2) {
        TensorIdx abs_idx = idx_0 * dim_offset[0] + idx_1 * dim_offset[1]
            + idx_2 * dim_offset[2]
            + (idx_0 % 2 == 0 and idx_1 % 2 == 0 and idx_2 % 2 == 0 ? 8 : 0);

        ASSERT_EQ(static_cast<TypeParam>(abs_idx),
            this->param_step_2->tensor(idx_0, idx_1, idx_2))
            << "Double step increment result does not match expectation, "
            << "index: (" << idx_0 << ", " << idx_1 << ", " << idx_2 << ")";
      }
}


TYPED_TEST(TestOperationBase, TestOpLoopForMultiOper) {
  using OpInc = typename TestOperationBase<TypeParam>::OpInc<>;
  using SingleFor = typename TestOperationBase<TypeParam>::SingleFor;
  using TripleFor = typename TestOperationBase<TypeParam>::TripleFor;

  // Single step
  // Initialize operations
  SingleFor outer_for_step_1(this->param_step_1, this->param_step_1->idx_0,
      this->param_step_1->idx_0, -this->param_step_1->idx_0, 1);
  auto mid_for_step_1 = make_shared<SingleFor>(this->param_step_1,
      this->param_step_1->idx_1, this->param_step_1->idx_1,
      -this->param_step_1->idx_1, 1);
  auto inner_for_step_1 = make_shared<TripleFor>(this->param_step_1,
      this->param_step_1->idx_2, this->param_step_1->idx_2,
      -this->param_step_1->idx_2, 1);
  auto inc_obj_step_1 = make_shared<OpInc>(this->param_step_1);

  // Linking operations
  inner_for_step_1->init_operation(inc_obj_step_1, 0);
  inner_for_step_1->init_operation(inc_obj_step_1, 1);
  inner_for_step_1->init_operation(inc_obj_step_1, 2);
  mid_for_step_1->init_operation(inner_for_step_1);
  outer_for_step_1.init_operation(mid_for_step_1);

  // Executing operations
  outer_for_step_1.exec();

  // Check results
  for (TensorIdx idx = 0; idx < this->len_1; ++idx) {
    ASSERT_EQ(static_cast<TypeParam>(idx + 24),
        this->param_step_1->raw_data[idx])
        << "Single step increment result does not match expectation, index: "
        << idx;
  }

  // Double step
  // Initialize operations
  SingleFor outer_for_step_2(this->param_step_2, this->param_step_2->idx_0,
      this->param_step_2->idx_0, -this->param_step_2->idx_0, 2);
  auto mid_for_step_2 = make_shared<SingleFor>(this->param_step_2,
      this->param_step_2->idx_1, this->param_step_2->idx_1,
      -this->param_step_2->idx_1, 2);
  auto inner_for_step_2 = make_shared<TripleFor>(this->param_step_2,
      this->param_step_2->idx_2, this->param_step_2->idx_2,
      -this->param_step_2->idx_2, 2);
  auto inc_obj_step_2 = make_shared<OpInc>(this->param_step_2);

  // Linking operations
  inner_for_step_2->init_operation(inc_obj_step_2, 0);
  inner_for_step_2->init_operation(inc_obj_step_2, 1);
  inner_for_step_2->init_operation(inc_obj_step_2, 2);
  mid_for_step_2->init_operation(inner_for_step_2);
  outer_for_step_2.init_operation(mid_for_step_2);

  // Executing operations
  outer_for_step_2.exec();

  // Check results
  vector<TensorIdx> dim_offset{ this->step_2_idx_1 * this->step_2_idx_2,
      this->step_2_idx_2, 1 };
  for (TensorIdx idx_0 = 0; idx_0 < this->step_2_idx_0; ++idx_0)
    for (TensorIdx idx_1 = 0; idx_1 < this->step_2_idx_1; ++idx_1)
      for (TensorIdx idx_2 = 0; idx_2 < this->step_2_idx_2; ++idx_2) {
        TensorIdx abs_idx = idx_0 * dim_offset[0] + idx_1 * dim_offset[1]
            + idx_2 * dim_offset[2]
            + (idx_0 % 2 == 0 and idx_1 % 2 == 0 and idx_2 % 2 == 0 ? 24 : 0);

        ASSERT_EQ(static_cast<TypeParam>(abs_idx),
            this->param_step_2->tensor(idx_0, idx_1, idx_2))
            << "Double step increment result does not match expectation, "
            << "index: (" << idx_0 << ", " << idx_1 << ", " << idx_2 << ")";
      }
}

#endif // TEST_UNIT_TEST_OPERATIONS_TEST_OPERATION_BASE_H_
