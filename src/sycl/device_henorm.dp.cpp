// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "slate/Exception.hh"
#include "slate/internal/device.hh"

/* DPCT_ORIG #include "device_util.cuh"*/
#include "device_util.dp.hpp"

#include <cstdio>

namespace slate {
namespace device {

//------------------------------------------------------------------------------
/// Finds the largest absolute value of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Uses dynamic shared memory array of length sizeof(real_t) * n.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by henorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_maxima
///     Array of dimension gridDim.x.
///     On exit, tiles_maxima[k] = max_{i, j} abs( A^(k)_(i, j) )
///     for tile A^(k).
///
/* DPCT_ORIG template <typename scalar_t>
__global__ void henorm_max_kernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_maxima)*/
template <typename scalar_t>
void henorm_max_kernel(lapack::Uplo uplo, int64_t n,
                       scalar_t const *const *Aarray, int64_t lda,
                       blas::real_type<scalar_t> *tiles_maxima,
                       const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
/* DPCT_ORIG     scalar_t const* tile = Aarray[ blockIdx.x ]*/
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    int chunk;

    // Save partial results in shared memory.
/* DPCT_ORIG     extern __shared__ char dynamic_data[]*/
    auto dynamic_data = (char *)dpct_local;
    real_t* row_max = (real_t*) dynamic_data;
/* DPCT_ORIG     if (threadIdx.x < blockDim.x) {*/
    if (item_ct1.get_local_id(2) < item_ct1.get_local_range(2)) {
/* DPCT_ORIG         row_max[threadIdx.x] = 0*/
        row_max[item_ct1.get_local_id(2)] = 0;
    }

    // Each thread finds max of one row.
    // This does coalesced reads of one column at a time in parallel.
/* DPCT_ORIG     for (int i = threadIdx.x; i < n; i += blockDim.x) {*/
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
/* DPCT_ORIG         chunk = i % blockDim.x*/
        chunk = i % item_ct1.get_local_range(2);

        scalar_t const* row = &tile[ i ];
/* DPCT_ORIG         if (i < blockDim.x) {*/
        if (i < item_ct1.get_local_range(2)) {
            row_max[chunk] = 0;
        }

        real_t max = 0;
        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                max = max_nan(max, abs(row[j*lda]));
            int64_t j = i;
            max = max_nan(max, abs( real( row[j*lda] )));  // diag (real)
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j > i; --j) // strictly upper
                max = max_nan(max, abs(row[j*lda]));
            int64_t j = i;
            max = max_nan(max, abs( real( row[j*lda] )));  // diag (real)
        }
        row_max[chunk] = max_nan(max, row_max[chunk]);
    }

    // Reduction to find max of tile.
/* DPCT_ORIG     __syncthreads()*/
    /*
    DPCT1065:29: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
/* DPCT_ORIG     max_nan_reduce(blockDim.x, threadIdx.x, row_max)*/
    max_nan_reduce(item_ct1.get_local_range(2), item_ct1.get_local_id(2),
                   row_max, item_ct1);
/* DPCT_ORIG     if (threadIdx.x == 0) {*/
    if (item_ct1.get_local_id(2) == 0) {
/* DPCT_ORIG         tiles_maxima[blockIdx.x] = row_max[0]*/
        tiles_maxima[item_ct1.get_group(2)] = row_max[0];
    }
}

//------------------------------------------------------------------------------
/// Sum of absolute values of each column of elements, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one column.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by henorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block (blockDim.x), hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension gridDim.x,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_sums
///     Array of dimension gridDim.x * ldv.
///     On exit, tiles_sums[k*ldv + j] = max_{i} abs( A^(k)_(i, j) )
///     for row j of tile A^(k).
///
/// @param[in] ldv
///     Leading dimension of tiles_sums (values) array.
///
/* DPCT_ORIG template <typename scalar_t>
__global__ void henorm_one_kernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_sums, int64_t ldv)*/
template <typename scalar_t>
void henorm_one_kernel(lapack::Uplo uplo, int64_t n,
                       scalar_t const *const *Aarray, int64_t lda,
                       blas::real_type<scalar_t> *tiles_sums, int64_t ldv,
                       const sycl::nd_item<3> &item_ct1)
{
    using real_t = blas::real_type<scalar_t>;
/* DPCT_ORIG     scalar_t const* tile = Aarray[ blockIdx.x ]*/
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];

    // Each thread sums one row/column.
    // todo: the row reads are coalesced, but the col reads are not coalesced
/* DPCT_ORIG     for (int k = threadIdx.x; k < n; k += blockDim.x) {*/
    for (int k = item_ct1.get_local_id(2); k < n;
         k += item_ct1.get_local_range(2)) {
        scalar_t const* row    = &tile[ k ];
        scalar_t const* column = &tile[ lda*k ];
        real_t sum = 0;

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j < k; ++j) // strictly lower
                sum += abs(row[j*lda]);
            int64_t j = k;
            sum += abs( real( row[j*lda] )); // diag (real)
            for (int64_t i = k + 1; i < n; ++i) // strictly lower
                sum += abs(column[i]);
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j > k; --j) // strictly upper
                sum += abs(row[j*lda]);
            int64_t j = k;
            sum += abs( real( row[j*lda] )); // diag (real)
            for (int64_t i = 0; i < k && i < n; ++i) // strictly upper
                sum += abs(column[i]);
        }
/* DPCT_ORIG         tiles_sums[ blockIdx.x*ldv + k ] = sum*/
        tiles_sums[item_ct1.get_group(2) * ldv + k] = sum;
    }
}

//------------------------------------------------------------------------------
/// Sum of squares, in scaled representation, for each tile in Aarray.
/// Each thread block deals with one tile.
/// Each thread deals with one row, followed by a reduction.
/// Kernel assumes non-trivial tiles (n >= 1).
/// Launched by henorm().
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 1.
///     Also the number of threads per block, hence,
///
/// @param[in] Aarray
///     Array of tiles of dimension blockDim.x,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[out] tiles_values
///     Array of dimension 2 * blockDim.x.
///     On exit,
///         tiles_values[2*k + 0] = scale
///         tiles_values[2*k + 1] = sumsq
///     such that scale^2 * sumsq = sum_{i,j} abs( A^(k)_{i,j} )^2
///     for tile A^(k).
///
/* DPCT_ORIG template <typename scalar_t>
__global__ void henorm_fro_kernel(
    lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* tiles_values)*/
template <typename scalar_t>
void henorm_fro_kernel(lapack::Uplo uplo, int64_t n,
                       scalar_t const *const *Aarray, int64_t lda,
                       blas::real_type<scalar_t> *tiles_values,
                       const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    using real_t = blas::real_type<scalar_t>;
/* DPCT_ORIG     scalar_t const* tile = Aarray[ blockIdx.x ]*/
    scalar_t const *tile = Aarray[item_ct1.get_group(2)];
    int chunk;

    // Save partial results in shared memory.
/* DPCT_ORIG     extern __shared__ char dynamic_data[]*/
    auto dynamic_data = (char *)dpct_local;
    real_t* row_scale = (real_t*) &dynamic_data[0];
/* DPCT_ORIG     real_t* row_sumsq = &row_scale[blockDim.x]*/
    real_t *row_sumsq = &row_scale[item_ct1.get_local_range(2)];

    // Each thread finds sum-of-squares of one row.
    // This does coalesced reads of one column at a time in parallel.
/* DPCT_ORIG     for (int i = threadIdx.x; i < n; i += blockDim.x) {*/
    for (int i = item_ct1.get_local_id(2); i < n;
         i += item_ct1.get_local_range(2)) {
        real_t scale = 0;
        real_t sumsq = 1;
/* DPCT_ORIG         chunk = i % blockDim.x*/
        chunk = i % item_ct1.get_local_range(2);
        scalar_t const* row = &tile[ i ];

        if (uplo == lapack::Uplo::Lower) {
            for (int64_t j = 0; j < i && j < n; ++j) // strictly lower
                add_sumsq(scale, sumsq, abs(row[j*lda]));
            // double for symmetric entries
            sumsq *= 2;
            // diagonal (real)
            add_sumsq( scale, sumsq, abs( real( row[ i*lda ] ) ) );
        }
        else {
            // Loop backwards (n-1 down to i) to maintain coalesced reads.
            for (int64_t j = n-1; j > i; --j) // strictly upper
                add_sumsq( scale, sumsq, abs( row[ j*lda ] ) );
            // double for symmetric entries
            sumsq *= 2;
            // diagonal (real)
            add_sumsq( scale, sumsq, abs( real( row[ i*lda ] ) ) );
        }

/* DPCT_ORIG         if (i < blockDim.x) {*/
        if (i < item_ct1.get_local_range(2)) {
            row_scale[chunk] = 0;
            row_sumsq[chunk] = 1;
        }
        combine_sumsq(row_scale[chunk], row_sumsq[chunk], scale, sumsq);
/* DPCT_ORIG         __syncthreads()*/
        /*
        DPCT1065:30: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        ///////////////////////////////// item_ct1.barrier();
    }

/////////////////////////////////
    item_ct1.barrier();
/////////////////////////////////


    // Reduction to find sum-of-squares of tile.
    // todo: parallel reduction.
/* DPCT_ORIG     if (threadIdx.x == 0) {*/
    if (item_ct1.get_local_id(2) == 0) {
        real_t tile_scale = row_scale[0];
        real_t tile_sumsq = row_sumsq[0];
/* DPCT_ORIG         for (int64_t chunk = 1; chunk < blockDim.x && chunk < n;
 * ++chunk) {*/
        for (int64_t chunk = 1;
             chunk < item_ct1.get_local_range(2) && chunk < n; ++chunk) {
            combine_sumsq(tile_scale, tile_sumsq, row_scale[chunk], row_sumsq[chunk]);
        }

/* DPCT_ORIG         tiles_values[blockIdx.x*2 + 0] = tile_scale*/
        tiles_values[item_ct1.get_group(2) * 2 + 0] = tile_scale;
/* DPCT_ORIG         tiles_values[blockIdx.x*2 + 1] = tile_sumsq*/
        tiles_values[item_ct1.get_group(2) * 2 + 1] = tile_sumsq;
    }
}

//------------------------------------------------------------------------------
/// Batched routine that computes a partial norm for each tile.
///
/// @param[in] norm
///     Norm to compute. See values for description.
///
/// @param[in] uplo
///     Whether each Aarray[k] is stored in the upper or lower triangle.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///
/// @param[in] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to tiles,
///     where each Aarray[k] is an n-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
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
///     - Norm::Inf: for symmetric, same as Norm::One
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
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    scalar_t const* const* Aarray, int64_t lda,
    blas::real_type<scalar_t>* values, int64_t ldv, int64_t batch_count,
    blas::Queue& queue)
{
    using real_t = blas::real_type<scalar_t>;
    int64_t nb = 512;

    // quick return
    if (batch_count == 0)
        return;

/* DPCT_ORIG     cudaSetDevice( queue.device() )*/
    /*
    DPCT1093:142: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    //---------
    // max norm
    if (norm == lapack::Norm::Max) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count, queue);
        }
        else {
            assert(ldv == 1);
            /*
            DPCT1083:32: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shared_mem = sizeof(real_t) * nb;
/* DPCT_ORIG             henorm_max_kernel<<<batch_count, nb, shared_mem,
   queue.stream()>>> (uplo, n, Aarray, lda, values)*/
            /*
            DPCT1049:31: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
                // accessors to device memory
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        henorm_max_kernel(uplo, n, Aarray, lda, values,
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                    });
            });
        }
    }
    //---------
    // one norm
    else if (norm == lapack::Norm::One || norm == lapack::Norm::Inf) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count * n, queue);
        }
        else {
            assert(ldv >= n);
/* DPCT_ORIG             henorm_one_kernel<<<batch_count, nb, 0,
   queue.stream()>>> (uplo, n, Aarray, lda, values, ldv)*/
            /*
            DPCT1049:33: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))
                ->parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        henorm_one_kernel(uplo, n, Aarray, lda, values, ldv,
                                          item_ct1);
                    });
        }
    }
    //---------
    // Frobenius norm
    else if (norm == lapack::Norm::Fro) {
        if (n == 0) {
            blas::device_memset(values, 0, batch_count * 2, queue);
        }
        else {
            assert(ldv == 2);
            /*
            DPCT1083:35: The size of local memory in the migrated code may be
            different from the original code. Check that the allocated memory
            size in the migrated code is correct.
            */
            size_t shared_mem = sizeof(real_t) * nb * 2;
/* DPCT_ORIG             henorm_fro_kernel<<<batch_count, nb, shared_mem,
   queue.stream()>>> (uplo, n, Aarray, lda, values)*/
            /*
            DPCT1049:34: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
                // accessors to device memory
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shared_mem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, batch_count) *
                                          sycl::range<3>(1, 1, nb),
                                      sycl::range<3>(1, 1, nb)),
                    [=](sycl::nd_item<3> item_ct1) {
                        henorm_fro_kernel(uplo, n, Aarray, lda, values,
                                          item_ct1,
                                          dpct_local_acc_ct1.get_pointer());
                    });
            });
        }
    }

/* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:143: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
/* DPCT_ORIG     slate_assert(error == cudaSuccess)*/
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    float const* const* Aarray, int64_t lda,
    float* values, int64_t ldv, int64_t batch_count,
    blas::Queue& queue);

template
void henorm(
    lapack::Norm norm, lapack::Uplo uplo,
    int64_t n,
    double const* const* Aarray, int64_t lda,
    double* values, int64_t ldv, int64_t batch_count,
    blas::Queue& queue);

template void
henorm(lapack::Norm norm, lapack::Uplo uplo, int64_t n,
       /* DPCT_ORIG     cuFloatComplex const* const* Aarray, int64_t lda,*/
       sycl::float2 const *const *Aarray, int64_t lda, float *values,
       int64_t ldv, int64_t batch_count, blas::Queue &queue);

template void
henorm(lapack::Norm norm, lapack::Uplo uplo, int64_t n,
       /* DPCT_ORIG     cuDoubleComplex const* const* Aarray, int64_t lda,*/
       sycl::double2 const *const *Aarray, int64_t lda, double *values,
       int64_t ldv, int64_t batch_count, blas::Queue &queue);

} // namespace device
} // namespace slate
