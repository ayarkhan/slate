// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <sycl/sycl.hpp>
#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.dp.hpp"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Device function implementing element-wise tile scale.
/// Each thread block deals with one tile. gridDim.x == batch_count.
/// Each thread deals with one row.
/// Called by gescale_kernel and gescale_batch_kernel.
///
/// @copydoc gescale
///
template <typename scalar_t, typename scalar_t2>
void gescale_func(
    int64_t m, int64_t n,
    scalar_t2 mul, scalar_t *A, int64_t lda,
    const sycl::nd_item<3> &item_ct1)
{
    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t* rowA = &A[ i ];
        for (int64_t j = 0; j < n; ++j)
            rowA[ j*lda ] = rowA[ j*lda ] * mul;
    }
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile scale.
/// @copydoc gescale
template <typename scalar_t, typename scalar_t2>
void gescale_kernel(
    int64_t m, int64_t n,
    scalar_t2 mul, scalar_t *A, int64_t lda,
    const sycl::nd_item<3> &item_ct1)
{
    gescale_func(m, n, mul, A, lda, item_ct1);
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile scale.
/// @copydoc gescale_batch
template <typename scalar_t, typename scalar_t2>
void gescale_batch_kernel(
    int64_t m, int64_t n,
    scalar_t2 mul, scalar_t **Aarray, int64_t lda,
    const sycl::nd_item<3> &item_ct1)
{
    gescale_func(m, n, mul, Aarray[item_ct1.get_group(2)], lda, item_ct1);
}

//------------------------------------------------------------------------------
/// Kernel implementing element-wise tile scale.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by gescale().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] numer
///     Scale value numerator.
///
/// @param[in] denom
///     Scale value denominator.
///
/// @param[in,out] A
///     An m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
template <typename scalar_t, typename scalar_t2>
void gescale(
    int64_t m, int64_t n,
    scalar_t2 numer, scalar_t2 denom,
    scalar_t* A, int64_t lda,
    blas::Queue& queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;

    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    scalar_t2 mul = numer / denom;

    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           gescale_kernel(m, n, mul, A, lda, item_ct1);
                       });

    /* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:155: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    float* A, int64_t lda,
    blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer, double denom,
    double* A, int64_t lda,
    blas::Queue& queue);

template void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    sycl::float2 *A, int64_t lda,
    blas::Queue& queue);

template void
gescale(
    int64_t m, int64_t n,
    sycl::float2 numer, sycl::float2 denom,
    sycl::float2 *A, int64_t lda,
    blas::Queue& queue);

template void gescale(
    int64_t m, int64_t n,
    double numer, double denom,
    sycl::double2 *A, int64_t lda,
    blas::Queue& queue);

template void
gescale(
    int64_t m, int64_t n,
    sycl::double2 numer, sycl::double2 denom,
    sycl::double2 *A, int64_t lda,
    blas::Queue& queue);

//==============================================================================
namespace batch {

//------------------------------------------------------------------------------
/// Batched routine for element-wise tile scale. Sets
/// \[
///     Aarray[k] *= (numer / denom).
/// \]
/// This does NOT currently take extra care to avoid over/underflow.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] numer
///     Scale value numerator.
///
/// @param[in] denom
///     Scale value denominator.
///
/// @param[in,out] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t, typename scalar_t2>
void gescale(
    int64_t m, int64_t n,
    scalar_t2 numer, scalar_t2 denom,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue)
{
    // quick return
    if (m == 0 || n == 0)
        return;
    // quick return
    if (batch_count == 0)
        return;

    dpct::select_device(queue.device());

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    scalar_t2 mul = numer / denom;

    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           gescale_batch_kernel(m, n, mul, Aarray, lda,
                                                item_ct1);
                       });

    /* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:157: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    float** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template
void gescale(
    int64_t m, int64_t n,
    double numer, double denom,
    double** Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template void gescale(
    int64_t m, int64_t n,
    float numer, float denom,
    sycl::float2 **Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template void
gescale(
    int64_t m, int64_t n,
    sycl::float2 numer, sycl::float2 denom,
    sycl::float2 **Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template void gescale(
    int64_t m, int64_t n,
    double numer, double denom,
    sycl::double2 **Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

template void
gescale(
    int64_t m, int64_t n,
    sycl::double2 numer, sycl::double2 denom,
    sycl::double2 **Aarray, int64_t lda,
    int64_t batch_count, blas::Queue& queue);

} // namespace batch
} // namespace device
} // namespace slate
