#pragma once
#ifndef HPTT_HPTT_H_
#define HPTT_HPTT_H_

#include <cstdint>
#include <vector>
#include <memory>

#include <hptt/types.h>
#include <hptt/impl/hptt_trans.h>


namespace hptt {

/**
 * \brief Function for creating a tensor transpose.
 *
 * \details This function is used to create a plan for transposing a tensor. For
 *     the flexibility, the transpose has following form:
 *
 * $B_{perm(i1,i2,..., iN)} = \alpha \cdot A_{i_1,i_2,..., i_N} +
 *     \beta \cdot B_{perm(i_1,i_2,..., i_N)}$
 *
 * where A is an input tensor, B is the output destination. There are two
 * template parameters serving for this function: The transpose strategy is
 * either based on static rule, or from heuristics and auto-tuning. See more
 * from the function parameters.
 *
 * - FloatType is the data type, four floating types are support: single float
 *   (float), double float (double), single complex (FloatComplex), double
 *   complex (DoubleComplex).
 *
 * \param[in] in_data Raw input tensor data, the tensor to be transposed;
 * \param[in,out] out_data Raw output tensor data, the destination for storing
 *     the transpose result.
 * \param[in] in_size Vector for describing size of each order in input tensor.
 * \param[in] perm Vector used for describing permutation (e.g., perm = { 1, 0 }
 *     denotes a matrix transpose). The vector's size is used as the order of
 *     tensor.
 * \param[in] alpha Coefficient for scaling all the elements in input tensor.
 *     When using complex type, its type will be deduced into float (for
 *     FloatComplex) or double (for DoubleComplex).
 * \param[in] beta Coefficient for updating output tensor. When using complex
 *     type, its type will be deduced into float (for FloatComplex) or double
 *     (for DoubleComplex).
 * \param[in] num_threads Thread number for transpose, unsigned integer. When
 *     setting to 0, the OpenMP's OMP_NUM_THREADS will be used.
 * \param[in] tuning_timeout Time limit of auto-tuning (in seconds), this is an
 *     optional parameter (default value is 0.0). Its input value could be:
 *     - tuning_timeout < 0.0: Tune all generated transpose candidate. Caution:
 *       although tuning all possible results could deliver best transpose
 *       performance, it will cost plenty of time before getting the solution as
 *       well, especially when ORDER $n$ is large. The number of candidates
 *       increases with $n!$.
 *     - tuning_timeout == 0.0: No tuning will be performed and a suitable
 *       candidate according to the performance model will be selected.
 *     - tuning_timeout > 0.0: Tuning will last at most tuning_timeout seconds.
 * \param[in] in_outer_size An std::vector for input tensor outer size. It is an
 *     optional parameter, default is an empty std::initializer_list.
 * \param[in] out_outer_size An std::vector for output tensor outer size. It is
 *     an optional parameter, default is an empty std::initializer_list.
 *
 * \return A pointer to the computational plan for this transpose. It can be
 *     used like this:
 * \code{.cpp}
 *   auto plan = create_trans_plan(...);  // Create plan
 *   if (nullptr != plan)
 *     plan->exec();   // call plan's member function exec()
 * \endcode
 *
 * The returned pointer will be null, when:
 * 1. at least one of the two data pointers in function parameter list is null;
 * 2. tensor order is less than 2 (the order value comes from perm's size);
 * 3. at least one of the tensor size vectors is incorrect, i.e. in_size's
 *    size is not tensor's order; in_outer_size's or out_outer_size's size is
 *    neither 0 nor tensor's order.
 * 4. at least one of the tensor size vectors has incorrect value, i.e.
 *    zero-value or the outer size is less than the inner size.
 * 5. the size and content of input permutation array is not valid.
 * 6. shared/static libraries are not correctly installed. They must be either
 *    installed in LD_LIBRARY_PATH or placed in the directory where the
 *    executable locates.
 *
 */
template <typename FloatType>
std::shared_ptr<trans_plan<FloatType>> create_trans_plan(
    const FloatType *in_data, FloatType *out_data,
    const std::vector<uint32_t> &in_size, const std::vector<uint32_t> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta, const uint32_t num_threads,
    const double tuning_timeout = 0.0,
    const std::vector<uint32_t> &in_outer_size = {},
    const std::vector<uint32_t> &out_outer_size = {});


/*
 * Explicit template instantiation declaration for function create_trans_plan
 */
extern template std::shared_ptr<CGraphTransPackBase<float>>
create_trans_plan<float>(const float *, float *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<float>, const DeducedFloatType<float>,
    const TensorUInt, const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);
extern template std::shared_ptr<CGraphTransPackBase<double>>
create_trans_plan<double>(const double *, double *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<double>, const DeducedFloatType<double>,
    const TensorUInt, const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);
extern template std::shared_ptr<CGraphTransPackBase<FloatComplex>>
create_trans_plan<FloatComplex>(const FloatComplex *, FloatComplex *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<FloatComplex>, const DeducedFloatType<FloatComplex>,
    const TensorUInt, const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);
extern template std::shared_ptr<CGraphTransPackBase<DoubleComplex>>
create_trans_plan<DoubleComplex>(const DoubleComplex *, DoubleComplex *,
    const std::vector<TensorUInt> &, const std::vector<TensorUInt> &,
    const DeducedFloatType<DoubleComplex>,
    const DeducedFloatType<DoubleComplex>, const TensorUInt,
    const double, const std::vector<TensorUInt> &,
    const std::vector<TensorUInt> &);

}

#endif // HPTT_HPTT_H_
