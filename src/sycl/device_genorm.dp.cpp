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
/// Finds the largest absolute value of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * m.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
template <typename scalar_t>
void genorm_max_kernel(
    int64_t m, int64_t n,
    scalar_t const *const *Aarray, int64_t lda,
    blas::real_type<scalar_t> *tiles_maxima,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];

    // Save partial results in shared memory.
    auto dynamic_data = (char *)dpct_local;
    real_t* row_max = (real_t*) dynamic_data;
    int chunk;
    if (item_ct1.get_local_id(2) < item_ct1.get_local_range(2)) {
        row_max[item_ct1.get_local_id(2)] = 0;
    }

    // This does coalesced reads of one column at a time in parallel.
    for (int i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        chunk = i % item_ct1.get_local_range(2);
        scalar_t const* row = &tile[ i ];
        real_t max = 0;

        // Each thread finds max of one row.
        for (int64_t j = 0; j < n; ++j)
            max = max_nan(max, abs(row[j*lda]));

        // Save partial results in shared memory.
        row_max[chunk] = max_nan(max, row_max[chunk]);
    }

    // Reduction to find max of tile.
    item_ct1.barrier();
    max_nan_reduce(item_ct1.get_local_range(2), item_ct1.get_local_id(2),
                   row_max, item_ct1);
    if (item_ct1.get_local_id(2) == 0) {
        tiles_maxima[item_ct1.get_group(2)] = row_max[0];
    }
}

const int ib  = 32;  ///< block size for genorm_one_kernel
const int ib1 = 33;  ///< ib + 1 for stride to avoid GPU bank conflicts

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + j] = max_{i} abs( A^(k)_(i, j) )
///     for row j of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
void genorm_one_kernel(
    int64_t m, int64_t n,
    scalar_t const *const *Aarray, int64_t lda,
    blas::real_type<scalar_t> *tiles_sums, int64_t ldv,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    auto dynamic_data = (char *)dpct_local;
    real_t* shmem_tile = (real_t*)dynamic_data;
    const int k = item_ct1.get_local_id(2);

    for (int64_t jj = 0; jj < n; jj += ib) {
        real_t sum = 0.0;
        for (int64_t ii = 0; ii < m; ii += ib) {
            // Read 32x32 sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < ib; ++j)
                if (jj+j < n && ii+k < m)
                    shmem_tile[ j*ib1 + k ] = abs( tile[ (jj+j)*lda + ii+k ] );
            item_ct1.barrier(); // shmem_tile loaded

            // Each thread sums one column.
            for (int64_t i = 0; i < ib; ++i)
                if (jj+k < n && ii+i < m)
                    sum += shmem_tile[ k*ib1 + i ];
            item_ct1.barrier(); // done with shmem_tile
        }

        if (jj+k < n)
            tiles_sums[item_ct1.get_group(2) * ldv + jj + k] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each row of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///     for row i of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
template <typename scalar_t>
void genorm_inf_kernel(
    int64_t m, int64_t n,
    scalar_t const *const *Aarray, int64_t lda,
    blas::real_type<scalar_t> *tiles_sums, int64_t ldv,
    const sycl::nd_item<3> &item_ct1)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];

    for (int64_t i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t const* row = &tile[ i ];

        // Each thread sums one row.
        // This does coalesced reads of one column at a time in parallel.
        real_t sum = abs(row[0]);
        for (int64_t j = 1; j < n; ++j)
            sum += abs(row[j*lda]);

        tiles_sums[item_ct1.get_group(2) * ldv + i] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (m, n >= 1).
/// Launched by genorm().
///
/// @param[in] m
///     Number of rows of each tile. m >= 1.
///
/// @param[in] n
///     Number of columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension blockDim.x,
///     where each Aarray[k] is an m-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] tiles_values
///     Array of dimension 2 * blockDim.x.
///     On exit,
///         tiles_values[2*k + 0] = scale
///         tiles_values[2*k + 1] = sumsq
///     such that scale^2 * sumsq = sum_{i,j} abs( A^(k)_{i,j} )^2
///     for tile A^(k).
///
template <typename scalar_t>
void genorm_fro_kernel(
    int64_t m, int64_t n,
    scalar_t const *const *Aarray, int64_t lda,
    blas::real_type<scalar_t> *tiles_values,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    int chunk;

    // Save partial results in shared memory.
    auto dynamic_data = (char *)dpct_local;
    real_t* row_scale = (real_t*) &dynamic_data[0];
    real_t *row_sumsq = &row_scale[item_ct1.get_local_range(2)];

    real_t tile_scale = row_scale[0];
    real_t tile_sumsq = row_sumsq[0];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
    for (int i = item_ct1.get_local_id(2); i < m;
         i += item_ct1.get_local_range(2)) {
        scalar_t const* row = &tile[ i ];
        real_t scale = 0;
        real_t sumsq = 1;
        chunk = i % item_ct1.get_local_range(2);

        for (int64_t j = 0; j < n; ++j) {
            add_sumsq(scale, sumsq, abs(row[j*lda]));
        }

        if (i < item_ct1.get_local_range(2)) {
            row_scale[chunk] = 0;
            row_sumsq[chunk] = 1;
        }

        // Save partial results in shared memory.
        combine_sumsq(row_scale[chunk], row_sumsq[chunk], scale, sumsq);
    }

    item_ct1.barrier();

    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
    if (item_ct1.get_local_id(2) == 0) {
        tile_scale = row_scale[0];
        tile_sumsq = row_sumsq[0];
        for (int64_t chunk = 1;
             chunk < item_ct1.get_local_range(2) && chunk < m; ++chunk) {
            combine_sumsq(tile_scale, tile_sumsq, row_scale[chunk], row_sumsq[chunk]);
        }

        tiles_values[item_ct1.get_group(2) * 2 + 0] = tile_scale;
        tiles_values[item_ct1.get_group(2) * 2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
// todo docs
template <typename scalar_t>
void ge_col_norms_max_kernel(
    int64_t m, int64_t n,
    scalar_t const *const *Aarray, int64_t lda,
    blas::real_type<scalar_t> *col_max, int64_t ldv,
    const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    auto dynamic_data = (char *)dpct_local;
    real_t* shmem_tile = (real_t*)dynamic_data;
    const int k = item_ct1.get_local_id(2);

    for (int64_t jj = 0; jj < n; jj += ib) {
        real_t max = 0.0;
        for (int64_t ii = 0; ii < m; ii += ib) {
            // Read 32x32 sub-tile into shared memory.
            // This does coalesced reads of one column at a time in parallel.
            for (int64_t j = 0; j < ib; ++j)
                if (jj+j < n && ii+k < m)
                    shmem_tile[ j*ib1 + k ] = abs( tile[ (jj+j)*lda + ii+k ] );
            item_ct1.barrier(); // shmem_tile loaded

            // Each thread compute max of one column.
            for (int64_t i = 0; i < ib; ++i)
                if (jj+k < n && ii+i < m)
                    max = max_nan( shmem_tile[ k*ib1 + i ], max );
            item_ct1.barrier(); // done with shmem_tile
        }

        if (jj+k < n)
            col_max[item_ct1.get_group(2) * ldv + jj + k] = max;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that computes a partial norm for each tile.
///
/// @param[in] norm
///     Norm to compute. See values for description.
///
/// @param[in] scope
///     Scope of the norm.
///     - NormScope::Matrix  computes partial norm of each tile.
///     - NormScope::Columns computes norm of each column.
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
///     Leading dimension of each tile. lda >= m.
///
/// @param[out] values
///     Array in GPU memory, dimension batch_count * ldv.
///     - Norm::Max: ldv = 1.
///         On exit, values[k] = max_{i, j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count.
///
///     - Norm::One: ldv >= n.
///         On exit, values[k*ldv + j] = sum_{i} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= j < n.
///
///     - Norm::Inf: ldv >= m.
///         On exit, values[k*ldv + i] = sum_{j} abs( A^(k)_(i, j) )
///         for 0 <= k < batch_count, 0 <= i < m.
///
///     - Norm::Max: ldv = 2.
///         On exit,
///             values[k*2 + 0] = scale_k
///             values[k*2 + 1] = sumsq_k
///         where scale_k^2 sumsq_k = sum_{i,j} abs( A^(k)_(i, j) )^2
///         for 0 <= k < batch_count.
///
/// @param[in] ldv
///     Leading dimension of values array.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv,
    int64_t batch_count, blas::Queue& queue)
{
    using real_t = blas::real_type<scalar_t>;
    int64_t nb = 512;

    // quick return
    if (batch_count == 0)
        return;

    dpct::select_device(queue.device());

    if (scope == NormScope::Matrix) {

        //---------
        // max norm
        if (norm == lapack::Norm::Max) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count, queue);
            }
            else {
                assert(ldv == 1);
                size_t shared_mem = sizeof(real_t) * nb;
                ((sycl::queue *)(&queue.stream()))
                    ->submit([&](sycl::handler &cgh) {
                        // accessors to device memory
                        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                            sycl::range<1>(shared_mem), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, batch_count) *
                                    sycl::range<3>(1, 1, nb),
                                sycl::range<3>(1, 1, nb)),
                            [=](sycl::nd_item<3> item_ct1) {
                                genorm_max_kernel(
                                    m, n, Aarray, lda, values, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
                            });
                    });
            }
        }
        //---------
        // one norm
        else if (norm == lapack::Norm::One) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * n, queue);
            }
            else {
                assert(ldv >= n);
                size_t shared_mem = sizeof(real_t) * ib * ib1;
                ((sycl::queue *)(&queue.stream()))
                    ->submit([&](sycl::handler &cgh) {
                        // accessors to device memory
                        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                            sycl::range<1>(shared_mem), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, batch_count) *
                                    sycl::range<3>(1, 1, ib),
                                sycl::range<3>(1, 1, ib)),
                            [=](sycl::nd_item<3> item_ct1) {
                                genorm_one_kernel(
                                    m, n, Aarray, lda, values, ldv, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
                            });
                    });
            }
        }
        //---------
        // inf norm
        else if (norm == lapack::Norm::Inf) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * m, queue);
            }
            else {
                assert(ldv >= m);
                ((sycl::queue *)(&queue.stream()))
                    ->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                              sycl::range<3>(1, 1, nb),
                                          sycl::range<3>(1, 1, nb)),
                        [=](sycl::nd_item<3> item_ct1) {
                            genorm_inf_kernel(m, n, Aarray, lda, values, ldv,
                                              item_ct1);
                        });
            }
        }
        //---------
        // Frobenius norm
        else if (norm == lapack::Norm::Fro) {
            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * 2, queue);
            }
            else {

                assert(ldv == 2);
                size_t shared_mem = sizeof(real_t) * nb * 2;
                ((sycl::queue *)(&queue.stream()))
                    ->submit([&](sycl::handler &cgh) {
                        // accessors to device memory
                        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                            sycl::range<1>(shared_mem), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, batch_count) *
                                    sycl::range<3>(1, 1, nb),
                                sycl::range<3>(1, 1, nb)),
                            [=](sycl::nd_item<3> item_ct1) {
                                genorm_fro_kernel(
                                    m, n, Aarray, lda, values, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
                            });
                    });
            }
        }
    }
    else if (scope == NormScope::Columns) {

        if (norm == Norm::Max) {

            if (m == 0 || n == 0) {
                blas::device_memset(values, 0, batch_count * n, queue);
            }
            else {
                assert(ldv >= n);
                size_t shared_mem = sizeof(real_t) * ib * ib1;
                ((sycl::queue *)(&queue.stream()))
                    ->submit([&](sycl::handler &cgh) {
                        // accessors to device memory
                        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                            sycl::range<1>(shared_mem), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, batch_count) *
                                    sycl::range<3>(1, 1, ib),
                                sycl::range<3>(1, 1, ib)),
                            [=](sycl::nd_item<3> item_ct1) {
                                ge_col_norms_max_kernel(
                                    m, n, Aarray, lda, values, ldv, item_ct1,
                                    dpct_local_acc_ct1.get_pointer());
                            });
                    });
            }
        }
        else {
            slate_not_implemented("The norm isn't yet supported");
        }
    }
    else {
        slate_not_implemented("The norm scope isn't yet supported.");
    }

    /* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:145: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue& queue);

template
void genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue& queue);

template void
genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    sycl::float2 const *const *Aarray, int64_t lda, float *values,
    int64_t ldv, int64_t batch_count, blas::Queue& queue);

template void
genorm(
    lapack::Norm norm, NormScope scope,
    int64_t m, int64_t n,
    sycl::double2 const *const *Aarray, int64_t lda, double *values,
    int64_t ldv, int64_t batch_count, blas::Queue& queue);

} // namespace device
} // namespace slate
