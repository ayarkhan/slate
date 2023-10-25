// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <sycl/sycl.hpp>
#include "slate/Exception.hh"
#include "slate/internal/device.hh"

#include "device_util.dp.hpp"

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Kernel implementing copy and precision conversions, copying A to B.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Launched by tzcopy().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile in Aarray. lda >= m.
///
/// @param[out] Barray
///     Array of tiles of dimension gridDim.x,
///     where each Barray[k] is an m-by-n matrix stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of each tile in Barray. ldb >= m.
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy_kernel(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t const *const *Aarray, int64_t lda,
    dst_scalar_t **Barray, int64_t ldb,
    const sycl::nd_item<3> &item_ct1)
{
    src_scalar_t const *tileA = Aarray[item_ct1.get_group(2)];
    dst_scalar_t *tileB = Barray[item_ct1.get_group(2)];

    // thread per row, if more rows than threads, loop by blockDim.x
    for (int64_t i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        src_scalar_t const* rowA = &tileA[ i ];
        dst_scalar_t*       rowB = &tileB[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j <= i && j < n; ++j) { // lower
                copy(rowA[j*lda], rowB[j*ldb]);
            }
        }
        else {
            for (int64_t j = n-1; j >= i; --j) { // upper
                copy(rowA[j*lda], rowB[j*ldb]);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Batched routine for element-wise trapezoidal copy and precision conversion,
/// copying A to B. Sets upper or lower part of
/// \[
///     Barray[k] = Aarray[k].
/// \]
///
/// @param[in] uplo
///     Whether each Aarray[k] is upper or lower trapezoidal.
///
/// @param[in] m
///     Number of rows of each tile. m >= 0.
///
/// @param[in] n
///     Number of columns of each tile. n >= 0.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile in A. lda >= m.
///
/// @param[out] Barray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Barray[k] is an m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] ldb
///     Leading dimension of each tile in B. ldb >= m.
///
/// @param[in] batch_count
///     Size of Aarray and Barray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename src_scalar_t, typename dst_scalar_t>
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    src_scalar_t const* const* Aarray, int64_t lda,
    dst_scalar_t** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue)
{
    // quick return
    if (batch_count == 0)
        return;

    // Max threads/block=1024 for current CUDA compute capability (<= 7.5)
    int64_t nthreads = std::min( int64_t( 1024 ), m );

    dpct::select_device(queue.device());

    ((sycl::queue *)(&queue.stream()))
        ->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1) {
                           tzcopy_kernel(uplo, m, n, Aarray, lda, Barray, ldb,
                                         item_ct1);
                       });

    /* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:171: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template
void tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    float** Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template void
tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    sycl::float2 const *const *Aarray, int64_t lda,
    sycl::float2 **Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template void
tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    sycl::float2 const *const *Aarray, int64_t lda,
    sycl::double2 **Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template void
tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    sycl::double2 const *const *Aarray, int64_t lda,
    sycl::double2 **Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

template void
tzcopy(
    lapack::Uplo uplo,
    int64_t m, int64_t n,
    sycl::double2 const *const *Aarray, int64_t lda,
    sycl::float2 **Barray, int64_t ldb,
    int64_t batch_count, blas::Queue& queue);

} // namespace device
} // namespace slate
