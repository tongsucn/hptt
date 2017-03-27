#pragma once
#ifndef HPTC_HPTC_TRANS_H_
#define HPTC_HPTC_TRANS_H_

#include <cstdint>

#include <array>
#include <vector>
#include <memory>
#include <algorithm>

#include <immintrin.h>

#include <hptc/types.h>
#include <hptc/compat.h>
#include <hptc/tensor.h>
#include <hptc/util/util_trans.h>
#include <hptc/plan/plan_trans.h>
#include <hptc/cgraph/cgraph_trans.h>
#include <hptc/param/parameter_trans.h>


namespace hptc {

template <typename FloatType>
class CGraphTransPack;


/**
 * \brief Function for creating a tensor transpose.
 *
 * \details This function is used to create computational a graph for
 *     transposing a tensor. For the flexibility, the transpose has following
 *     form:
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
 * \return A pointer to the computational graph for this transpose. It can be
 *     used like this:
 * \code{.cpp}
 *   auto graph = create_cgraph_trans(...);  // Create graph
 *   if (nullptr != graph)
 *     graph->exec();   // call cgraph's member function exec()
 *   // ...
 *   delete graph;
 *   graph = nullptr;
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
 *
 * The returned graph must be released by user to avoid memory leak.
 *
 * \sa CGraphTrans
 */
template <typename FloatType>
CGraphTransPack<FloatType> *create_cgraph_trans(
    const FloatType *in_data, FloatType *out_data,
    const std::vector<uint32_t> &in_size, const std::vector<uint32_t> &perm,
    const DeducedFloatType<FloatType> alpha,
    const DeducedFloatType<FloatType> beta, const uint32_t num_threads,
    const double tuning_timeout = 0.0,
    const std::vector<uint32_t> &in_outer_size = {},
    const std::vector<uint32_t> &out_outer_size = {});


/*
 * Import implementations
 */
#include "hptc_trans.tcc"

}

#endif // HPTC_HPTC_TRANS_H_
