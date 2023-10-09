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
#include <complex>

namespace slate {
namespace device {

/// internal blocking
/// 16 x 16 thread block = 256 threads
/// 32 x 32 thread block = 1024 threads
static const int ib = 16;

//------------------------------------------------------------------------------
/// Device routine handles one matrix.
/// Thread block grid:
/// x = batch index (ignored here; see batch kernel),
/// y = block row index,
/// z = block col index.
/// Each thread block is ib-by-ib threads and does one ib-by-ib block of an
/// n-by-n matrix.
///
/// Let nt = ceildiv( n, ib ) be the number of blocks for one n-by-n matrix.
/// An even number of blocks uses an (nt + 1) by (nt/2) grid.
/// Example: for nt = 4 blocks, y by z = 5 by 2 grid:
///   [ A00  A01 ]
///   [----. A11 ]                  [ A10  .  |  .   .  ]
///   [ A10 '----]                  [ A20 A21 |  .   .  ]
///   [ A20  A21 ] covers matrix as [ A30 A31 | A00  .  ]
///   [ A30  A31 ]                  [ A40 A41 | A01 A11 ]
///   [ A40  A41 ]
///
/// An odd number of blocks uses an (nt) by (nt + 1)/2 grid.
/// Example: for nt = 5 blocks, y by z = 5 by 3 grid:
///   [ A00 | A01   A02 ]
///   [     '----.      ]                  [ A00  .   .  |  .   .  ]
///   [ A10   A11 | A12 ]                  [ A10 A11  .  |  .   .  ]
///   [           '-----] covers matrix as [ A20 A21 A22 |  .   .  ]
///   [ A20   A21   A22 ]                  [ A30 A31 A32 | A01  .  ]
///   [ A30   A31   A32 ]                  [ A40 A41 A42 | A02 A12 ]
///   [ A40   A41   A42 ]
///
/* DPCT_ORIG template <typename scalar_t>
__device__ void transpose_func(
    bool is_conj,
    int n,
    scalar_t* A, int64_t lda)*/
template <typename scalar_t>
void transpose_func(bool is_conj, int n, scalar_t *A, int64_t lda,
                    const sycl::nd_item<3> &item_ct1,
                    sycl::local_accessor<scalar_t, 2> sA1,
                    sycl::local_accessor<scalar_t, 2> sA2,
                    sycl::local_accessor<scalar_t, 2> sA)
{
    // +1 to avoid memory bank conflicts.
/* DPCT_ORIG     __shared__ scalar_t sA1[ ib ][ ib+1 ]*/

/* DPCT_ORIG     __shared__ scalar_t sA2[ ib ][ ib+1 ]*/

    // i, j are row & column indices of top-left corner of each block.
    // ii, jj are row & column offsets within each block.
/* DPCT_ORIG     int ii = threadIdx.x*/
    int ii = item_ct1.get_local_id(2);
/* DPCT_ORIG     int jj = threadIdx.y*/
    int jj = item_ct1.get_local_id(1);

    int i, j;
/* DPCT_ORIG     if (gridDim.y - 1 == gridDim.z*2) {*/
    if (item_ct1.get_group_range(1) - 1 == item_ct1.get_group_range(0) * 2) {
        // Even number of blocks.
        //assert( ceildiv(n, ib) % 2 == 0 );
/* DPCT_ORIG         bool lower = (blockIdx.y > blockIdx.z)*/
        bool lower = (item_ct1.get_group(1) > item_ct1.get_group(0));
/* DPCT_ORIG         i = (lower ? (blockIdx.y - 1) : (blockIdx.z +
 * gridDim.z))*/
        i = (lower ? (item_ct1.get_group(1) - 1)
                   : (item_ct1.get_group(0) + item_ct1.get_group_range(0)));
/* DPCT_ORIG         j = (lower ? (blockIdx.z    ) : (blockIdx.y +
 * gridDim.z))*/
        j = (lower ? (item_ct1.get_group(0))
                   : (item_ct1.get_group(1) + item_ct1.get_group_range(0)));
    }
    else {
        // Odd number of blocks.
        //assert( ceildiv(n, ib) % 2 == 1 );
/* DPCT_ORIG         bool lower = (blockIdx.y >= blockIdx.z)*/
        bool lower = (item_ct1.get_group(1) >= item_ct1.get_group(0));
/* DPCT_ORIG         i = (lower ? blockIdx.y : (blockIdx.z + gridDim.z - 1))*/
        i = (lower ? item_ct1.get_group(1)
                   : (item_ct1.get_group(0) + item_ct1.get_group_range(0) - 1));
/* DPCT_ORIG         j = (lower ? blockIdx.z : (blockIdx.y + gridDim.z    ))*/
        j = (lower ? item_ct1.get_group(0)
                   : (item_ct1.get_group(1) + item_ct1.get_group_range(0)));
    }
    i *= ib;
    j *= ib;

    scalar_t* A1 = A + i + ii + (j + jj)*lda;  // A(i, j)
    if (i == j) { // diagonal block
        // Load block A(i, j) into shared memory sA1.
        if (i + ii < n  &&  j + jj < n) {
            sA1[jj][ii] = *A1;
        }
/* DPCT_ORIG         __syncthreads()*/
        /*
        DPCT1065:62: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Save transposed block, A(i, j) = trans(sA1).
        if (i + ii < n  &&  j + jj < n) {
            if (is_conj)
                *A1 = conj(sA1[ii][jj]);
            else
                *A1 = sA1[ii][jj];
        }
    }
    else { // off-diagonal block
        scalar_t* A2 = A + j + ii + (i + jj)*lda;  // A(j, i)
        // Load blocks A(i, j) and A(j, i) into shared memory sA1 and sA2.
        if (i + ii < n  &&  j + jj < n) {
            sA1[jj][ii] = *A1;
        }
        if (j + ii < n  &&  i + jj < n) {
            sA2[jj][ii] = *A2;
        }
/* DPCT_ORIG         __syncthreads()*/
        /*
        DPCT1065:63: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Save transposed blocks, A(i, j) = trans(sA2), A(j, i) = trans(sA1).
        if (i + ii < n && j + jj < n) {
            if (is_conj)
                *A1 = conj(sA2[ii][jj]);
            else
                *A1 = sA2[ii][jj];
        }
        if (j + ii < n && i + jj < n) {
            if (is_conj)
                *A2 = conj(sA1[ii][jj]);
            else
                *A2 = sA1[ii][jj];
        }
    }
}

//------------------------------------------------------------------------------
static const int NB = 32;  ///< block size for transpose_func
static const int NY = 8;   ///< y dim of thread block size for transpose_func
// static const int NX = 32; handled as template parameter, look below


/// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
/// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
/// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
/// for each subtile
///     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
///     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
///     A  += NX
///     AT += NX*ldat
///
/// e.g., with NB=32, NX=32, NY=8 ([sdc] precisions)
///     load 32x32 subtile as 4   blocks of 32x8 columns: (A11  A12  A13  A14 )
///     save 32x32 subtile as 1*4 blocks of 32x8 columns: (AT11 AT12 AT13 AT14)
///
/// e.g., with NB=32, NX=16, NY=8 (z precision)
///     load 16x32 subtile as 4   blocks of 16x8 columns: (A11  A12  A13  A14)
///     save 32x16 subtile as 2*2 blocks of 16x8 columns: (AT11 AT12)
///                                                       (AT21 AT22)
///
/* DPCT_ORIG template <typename scalar_t, int NX>
__device__ void transpose_func(
    bool is_conj,
    int m, int n,
    const scalar_t *A,  int64_t lda,
          scalar_t *AT, int64_t ldat)*/
template <typename scalar_t, int NX>
void transpose_func(bool is_conj, int m, int n, const scalar_t *A, int64_t lda,
                    scalar_t *AT, int64_t ldat,
                    const sycl::nd_item<3> &item_ct1,
                    sycl::local_accessor<scalar_t, 2> sA1,
                    sycl::local_accessor<scalar_t, 2> sA2,
                    sycl::local_accessor<scalar_t, 2> sA)
{
/* DPCT_ORIG     __shared__ scalar_t sA[NB][NX+1]*/

/* DPCT_ORIG     int tx  = threadIdx.x*/
    int tx = item_ct1.get_local_id(2);
/* DPCT_ORIG     int ty  = threadIdx.y*/
    int ty = item_ct1.get_local_id(1);
/* DPCT_ORIG     int iby = blockIdx.y*NB*/
    int iby = item_ct1.get_group(1) * NB;
/* DPCT_ORIG     int ibz = blockIdx.z*NB*/
    int ibz = item_ct1.get_group(0) * NB;
    int i, j;

    A  += iby + tx + (ibz + ty)*lda;
    AT += ibz + tx + (iby + ty)*ldat;

    #pragma unroll
    for (int tile=0; tile < NB/NX; ++tile) {
        // load NX-by-NB subtile transposed from A into sA
        i = iby + tx + tile*NX;
        j = ibz + ty;
        if (i < m) {
            if (is_conj) {
                #pragma unroll
                for (int j2=0; j2 < NB; j2 += NY) {
                    if (j + j2 < n) {
                        sA[ty + j2][tx] = conj(A[j2*lda]);
                    }
                }
            }
            else {
                #pragma unroll
                for (int j2=0; j2 < NB; j2 += NY) {
                    if (j + j2 < n) {
                        sA[ty + j2][tx] = A[j2*lda];
                    }
                }
            }
        }
/* DPCT_ORIG         __syncthreads()*/
        /*
        DPCT1065:64: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // save NB-by-NX subtile from sA into AT
        i = ibz + tx;
        j = iby + ty + tile*NX;
        #pragma unroll
        for (int i2=0; i2 < NB; i2 += NX) {
            if (i + i2 < n) {
                #pragma unroll
                for (int j2=0; j2 < NX; j2 += NY) {
                    if (j + j2 < m) {
                        AT[i2 + j2*ldat] = sA[tx + i2][ty + j2];
                    }
                }
            }
        }
/* DPCT_ORIG         __syncthreads()*/
        /*
        DPCT1065:65: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // move to next subtile
        A  += NX;
        AT += NX*ldat;
    }
}

//------------------------------------------------------------------------------
/// in-place transpose of a square buffer
/* DPCT_ORIG template <typename scalar_t>
__global__ void transpose_kernel(
    bool is_conj,
    int n,
    scalar_t* A, int64_t lda)*/
template <typename scalar_t>
void transpose_kernel(bool is_conj, int n, scalar_t *A, int64_t lda,
                      const sycl::nd_item<3> &item_ct1,
                      sycl::local_accessor<scalar_t, 2> sA1,
                      sycl::local_accessor<scalar_t, 2> sA2,
                      sycl::local_accessor<scalar_t, 2> sA)
{
    transpose_func(is_conj, n, A, lda, item_ct1, sA1, sA2, sA);
}

//------------------------------------------------------------------------------
/// in-place transpose of array of square buffers
/* DPCT_ORIG template <typename scalar_t>
__global__ void transpose_batch_kernel(
    bool is_conj,
    int n,
    scalar_t** Aarray, int64_t lda)*/
template <typename scalar_t>
void transpose_batch_kernel(bool is_conj, int n, scalar_t **Aarray, int64_t lda,
                            const sycl::nd_item<3> &item_ct1,
                            sycl::local_accessor<scalar_t, 2> sA1,
                            sycl::local_accessor<scalar_t, 2> sA2,
                            sycl::local_accessor<scalar_t, 2> sA)
{
/* DPCT_ORIG     transpose_func(is_conj, n, Aarray[blockIdx.x], lda)*/
    transpose_func(is_conj, n, Aarray[item_ct1.get_group(2)], lda, item_ct1, sA1, sA2, sA);
}

//------------------------------------------------------------------------------
/// out-of-place transpose of a rectangular buffer
/// transopses A onto AT
///
/* DPCT_ORIG template <typename scalar_t, int NX>
__global__ void transpose_kernel(
    bool is_conj,
    int m, int n,
    const scalar_t *A,  int64_t lda,
          scalar_t *AT, int64_t ldat)*/
template <typename scalar_t, int NX>
void transpose_kernel(bool is_conj, int m, int n, const scalar_t *A,
                      int64_t lda, scalar_t *AT, int64_t ldat,
                      const sycl::nd_item<3> &item_ct1,
                      sycl::local_accessor<scalar_t, 2> sA1,
                      sycl::local_accessor<scalar_t, 2> sA2,
                      sycl::local_accessor<scalar_t, 2> sA)
{
    transpose_func<scalar_t, NX>(is_conj, m, n, A, lda, AT, ldat, item_ct1, sA1,
                                 sA2, sA);
}

//------------------------------------------------------------------------------
/// out-of-place transpose of an array of rectangular buffers
/// transopses dA_array onto dAT_array
///
/* DPCT_ORIG template <typename scalar_t, int NX>
__global__ void transpose_batch_kernel(
    bool is_conj,
    int m, int n,
    scalar_t **dA_array,  int64_t lda,
    scalar_t **dAT_array, int64_t ldat)*/
template <typename scalar_t, int NX>
void transpose_batch_kernel(bool is_conj, int m, int n, scalar_t **dA_array,
                            int64_t lda, scalar_t **dAT_array, int64_t ldat,
                            const sycl::nd_item<3> &item_ct1,
                            sycl::local_accessor<scalar_t, 2> sA1,
                            sycl::local_accessor<scalar_t, 2> sA2,
                            sycl::local_accessor<scalar_t, 2> sA)
{
    transpose_func<scalar_t, NX>(
        /* DPCT_ORIG         is_conj, m, n, dA_array[blockIdx.x], lda,
           dAT_array[blockIdx.x], ldat )*/
        is_conj,
        m, n, dA_array[item_ct1.get_group(2)], lda,
        dAT_array[item_ct1.get_group(2)], ldat, item_ct1, sA1, sA2, sA);
}

//------------------------------------------------------------------------------
/// Physically transpose a square matrix in place.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///
/// @param[in,out] A
///     A square n-by-n matrix stored in an lda-by-n array in GPU memory.
///     On output, A is transposed.
///
/// @param[in] lda
///     Leading dimension of A. lda >= n.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void transpose(
    bool is_conj,
    int64_t n,
    scalar_t* A, int64_t lda,
    blas::Queue& queue)
{
    if (n <= 1)
        return;
    assert(lda >= n);

/* DPCT_ORIG     cudaSetDevice( queue.device() )*/
    /*
    DPCT1093:158: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    int nt = ceildiv( n, int64_t(ib) );
    assert(nt <= 65535);                // CUDA limitation

    // Need 1/2 * (nt + 1) * nt to cover lower triangle and diagonal of matrix.
    // Block assignment differs depending on whether nt is odd or even.
/* DPCT_ORIG     dim3 blocks*/
    sycl::range<3> blocks(1, 1, 1);
    if (nt % 2 == 0) {
        // even blocks
/* DPCT_ORIG         blocks = { 1, uint(nt + 1), uint(nt/2) }*/
        blocks = sycl::range<3>(uint(nt / 2), uint(nt + 1), 1);
    }
    else {
        // odd blocks
/* DPCT_ORIG         blocks = { 1, uint(nt), uint((nt + 1)/2) }*/
        blocks = sycl::range<3>(uint((nt + 1) / 2), uint(nt), 1);
    }
/* DPCT_ORIG     dim3 threads( ib, ib )*/
    sycl::range<3> threads(1, ib, ib);

/* DPCT_ORIG     transpose_kernel<<< blocks, threads, 0, queue.stream() >>>
        (is_conj, n, A, lda)*/
    /*
    DPCT1049:66: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
        // accessors to device memory
        /*
        DPCT1101:176: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:177: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA1_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:178: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:179: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA2_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:180: 'NB' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<scalar_t, 2> sA_acc_ct1(
            sycl::range<2>(
                32 /*NB*/,
                // dpct_placeholder /*Fix the type mannually*/ + 1),
                16 /*Fix the type mannually*/ + 1),
            cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             transpose_kernel(is_conj, n, A, lda, item_ct1, sA1_acc_ct1, sA2_acc_ct1, sA_acc_ct1);
                         });
    });

/* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:159: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
/* DPCT_ORIG     slate_assert(error == cudaSuccess)*/
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
/// Physically transpose a batch of square matrices in place.
///
/// @param[in] n
///     Number of rows and columns of each tile. n >= 0.
///
/// @param[in,out] Aarray
///     Array in GPU memory of dimension batch_count, containing pointers to
///     matrices, where each Aarray[k] is a square n-by-n matrix stored in an
///     lda-by-n array in GPU memory.
///     On output, each Aarray[k] is transposed.
///
/// @param[in] lda
///     Leading dimension of each tile. lda >= n.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void transpose_batch(
    bool is_conj,
    int64_t n,
    scalar_t** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue)
{
    if (batch_count < 0 || n <= 1)
        return;
    assert(lda >= n);

/* DPCT_ORIG     cudaSetDevice( queue.device() )*/
    /*
    DPCT1093:160: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    int nt = ceildiv( n, int64_t(ib) );
    assert(nt <= 65535);                // CUDA limitation
    assert(batch_count <= 2147483647);  // CUDA limitation, 2^31 - 1

    // Need 1/2 * (nt + 1) * nt to cover lower triangle and diagonal of matrix.
    // Block assignment differs depending on whether nt is odd or even.
/* DPCT_ORIG     dim3 blocks*/
    sycl::range<3> blocks(1, 1, 1);
    if (nt % 2 == 0) {
        // even blocks
/* DPCT_ORIG         blocks = { uint(batch_count), uint(nt + 1), uint(nt/2) }*/
        blocks = sycl::range<3>(uint(nt / 2), uint(nt + 1), uint(batch_count));
    }
    else {
        // odd blocks
/* DPCT_ORIG         blocks = { uint(batch_count), uint(nt), uint((nt + 1)/2)
 * }*/
        blocks = sycl::range<3>(uint((nt + 1) / 2), uint(nt), uint(batch_count));
    }
/* DPCT_ORIG     dim3 threads( ib, ib )*/
    sycl::range<3> threads(1, ib, ib);

/* DPCT_ORIG     transpose_batch_kernel<<< blocks, threads, 0, queue.stream()
   >>> ( is_conj, n, Aarray, lda )*/
    /*
    DPCT1049:67: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
        // accessors to device memory
        /*
        DPCT1101:181: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:182: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA1_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:183: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:184: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA2_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:185: 'NB' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<scalar_t, 2> sA_acc_ct1(
            sycl::range<2>(
                32 /*NB*/,
                // dpct_placeholder /*Fix the type mannually*/ + 1),
                16 /*Fix the type mannually*/ + 1),
            cgh);

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             transpose_batch_kernel(is_conj, n, Aarray, lda, item_ct1, sA1_acc_ct1, sA2_acc_ct1, sA_acc_ct1);
                         });
    });

/* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:161: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
/* DPCT_ORIG     slate_assert(error == cudaSuccess)*/
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
/// Physically transpose a rectangular matrix out-of-place.
///
/// @param[in] m
///     Number of columns of tile. m >= 0.
///
/// @param[in] n
///     Number of rows of tile. n >= 0.
///
/// @param[in] dA
///     A rectangular m-by-n matrix stored in an lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of dA. lda >= m.
///
/// @param[out] dAT
///     A rectangular m-by-n matrix stored in an ldat-by-m array in GPU memory.
///     On output, dAT is the transpose of dA.
///
/// @param[in] ldat
///     Leading dimension of dAT. ldat >= n.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t, int NX>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t* dA,  int64_t lda,
    scalar_t* dAT, int64_t ldat,
    blas::Queue& queue)
{
    if ((m <= 0) || (n <= 0))
        return;
    assert(lda >= m);
    assert(ldat >= n);

/* DPCT_ORIG     cudaSetDevice( queue.device() )*/
    /*
    DPCT1093:162: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    int mt = ceildiv( m, int64_t(NB) );
    assert(mt <= 65535);                // CUDA limitation
    int nt = ceildiv( n, int64_t(NB) );
    assert(nt <= 65535);                // CUDA limitation

/* DPCT_ORIG     dim3 grid( 1, mt, nt )*/
    sycl::range<3> grid(nt, mt, 1);
/* DPCT_ORIG     dim3 threads( NX, NY )*/
    sycl::range<3> threads(1, NY, NX);
/* DPCT_ORIG     transpose_kernel<scalar_t, NX><<< grid, threads, 0,
   queue.stream() >>> ( is_conj, m, n, dA, lda, dAT, ldat )*/
    /*
    DPCT1049:68: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
        // accessors to device memory
        /*
        DPCT1101:186: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:187: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA1_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:188: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:189: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA2_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:190: 'NB' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<scalar_t, 2> sA_acc_ct1(
            sycl::range<2>(
                32 /*NB*/,
                // dpct_placeholder /*Fix the type mannually*/ + 1),
                16 /*Fix the type mannually*/ + 1),
            cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             transpose_kernel<scalar_t, NX>(
                                 is_conj, m, n, dA, lda, dAT, ldat, item_ct1,
                                 sA1_acc_ct1, sA2_acc_ct1, sA_acc_ct1);
                         });
    });

/* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:163: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
/* DPCT_ORIG     slate_assert(error == cudaSuccess)*/
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
/// Physically transpose a batch of rectangular matrices out-of-place.
///
/// @param[in] m
///     Number of columns of each tile. m >= 0.
///
/// @param[in] n
///     Number of rows of each tile. n >= 0.
///
/// @param[in] dA_array
///     Array in GPU memory of dimension batch_count, containing pointers to
///     matrices, where each dA_array[k] is a rectangular m-by-n matrix stored in an
///     lda-by-n array in GPU memory.
///
/// @param[in] lda
///     Leading dimension of each dA_array[k] tile. lda >= m.
///
/// @param[out] dAT_array
///     Array in GPU memory of dimension batch_count, containing pointers to
///     matrices, where each dAT_array[k] is a rectangular m-by-n matrix stored in an
///     ldat-by-m array in GPU memory.
///     On output, each dAT_array[k] is the transpose of dA_array[k].
///
/// @param[in] ldat
///     Leading dimension of each dAT_array[k] tile. ldat >= n.
///
/// @param[in] batch_count
///     Size of Aarray. batch_count >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t, int NX>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    scalar_t **dA_array,  int64_t lda,
    scalar_t **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    if ((m <= 0) || (n <= 0))
        return;
    assert(lda >= m);
    assert(ldat >= n);

/* DPCT_ORIG     cudaSetDevice( queue.device() )*/
    /*
    DPCT1093:164: The "queue.device()" device may be not the one intended for
    use. Adjust the selected device if needed.
    */
    dpct::select_device(queue.device());

    int mt = ceildiv( m, int64_t(NB) );
    assert(mt <= 65535);                // CUDA limitation
    int nt = ceildiv( n, int64_t(NB) );
    assert(nt <= 65535);                // CUDA limitation
    assert(batch_count <= 2147483647);  // CUDA limitation, 2^31 - 1

/* DPCT_ORIG     dim3 grid( uint(batch_count), mt, nt )*/
    sycl::range<3> grid(nt, mt, uint(batch_count));
/* DPCT_ORIG     dim3 threads( NX, NY, 1 )*/
    sycl::range<3> threads(1, NY, NX);
/* DPCT_ORIG     transpose_batch_kernel<scalar_t, NX><<< grid, threads, 0,
   queue.stream() >>> ( is_conj, m, n, dA_array, lda, dAT_array, ldat )*/
    /*
    DPCT1049:69: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(&queue.stream()))->submit([&](sycl::handler &cgh) {
        // accessors to device memory
        /*
        DPCT1101:191: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:192: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA1_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:193: 'ib' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        /*
        DPCT1101:194: 'ib+1' expression was replaced with a value. Modify the
        code to use the original expression, provided in comments, if it is
        correct.
        */
        sycl::local_accessor<scalar_t, 2> sA2_acc_ct1(
            sycl::range<2>(16 /*ib*/, 17 /*ib+1*/), cgh);
        /*
        DPCT1101:195: 'NB' expression was replaced with a value. Modify the code
        to use the original expression, provided in comments, if it is correct.
        */
        sycl::local_accessor<scalar_t, 2> sA_acc_ct1(
            sycl::range<2>(
                32 /*NB*/,
                // dpct_placeholder /*Fix the type mannually*/ + 1),
                16 /*Fix the type mannually*/ + 1),
            cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             transpose_batch_kernel<scalar_t, NX>(
                                 is_conj, m, n, dA_array, lda, dAT_array, ldat,
                                 item_ct1, sA1_acc_ct1, sA2_acc_ct1,
                                 sA_acc_ct1);
                         });
    });

/* DPCT_ORIG     cudaError_t error = cudaGetLastError()*/
    /*
    DPCT1010:165: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    dpct::err0 error = 0;
/* DPCT_ORIG     slate_assert(error == cudaSuccess)*/
    slate_assert(error == 0);
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void transpose(
    bool is_conj,
    int64_t n,
    float* A, int64_t lda,
    blas::Queue& queue);

template
void transpose(
    bool is_conj,
    int64_t n,
    double* A, int64_t lda,
    blas::Queue& queue);

template void transpose(bool is_conj, int64_t n,
                        /* DPCT_ORIG     cuFloatComplex* A, int64_t lda,*/
                        sycl::float2 *A, int64_t lda, blas::Queue &queue);

template void transpose(bool is_conj, int64_t n,
                        /* DPCT_ORIG     cuDoubleComplex* A, int64_t lda,*/
                        sycl::double2 *A, int64_t lda, blas::Queue &queue);

// ----------------------------------------
template
void transpose_batch(
    bool is_conj,
    int64_t n,
    float** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue);

template
void transpose_batch(
    bool is_conj,
    int64_t n,
    double** Aarray, int64_t lda,
    int64_t batch_count,
    blas::Queue& queue);

template void
transpose_batch(bool is_conj, int64_t n,
                /* DPCT_ORIG     cuFloatComplex** Aarray, int64_t lda,*/
                sycl::float2 **Aarray, int64_t lda, int64_t batch_count,
                blas::Queue &queue);

template void
transpose_batch(bool is_conj, int64_t n,
                /* DPCT_ORIG     cuDoubleComplex** Aarray, int64_t lda,*/
                sycl::double2 **Aarray, int64_t lda, int64_t batch_count,
                blas::Queue &queue);

// ----------------------------------------
template<>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    float* dA,  int64_t lda,
    float* dAT, int64_t ldat,
    blas::Queue& queue)
{
    transpose<float,32>(
        is_conj,
        m, n,
        dA,  lda,
        dAT, ldat,
        queue);
}

template<>
void transpose(
    bool is_conj,
    int64_t m, int64_t n,
    double* dA,  int64_t lda,
    double* dAT, int64_t ldat,
    blas::Queue& queue)
{
    transpose<double,32>(
        is_conj,
        m, n,
        dA,  lda,
        dAT, ldat,
        queue);
}

template <>
void transpose(bool is_conj, int64_t m, int64_t n,
               /* DPCT_ORIG     cuFloatComplex* dA,  int64_t lda,*/
               sycl::float2 *dA, int64_t lda,
               /* DPCT_ORIG     cuFloatComplex* dAT, int64_t ldat,*/
               sycl::float2 *dAT, int64_t ldat, blas::Queue &queue)
{
/* DPCT_ORIG     transpose<cuFloatComplex,32>(*/
    transpose<sycl::float2, 32>(is_conj, m, n, dA, lda, dAT, ldat, queue);
}

template <>
void transpose(bool is_conj, int64_t m, int64_t n,
               /* DPCT_ORIG     cuDoubleComplex* dA,  int64_t lda,*/
               sycl::double2 *dA, int64_t lda,
               /* DPCT_ORIG     cuDoubleComplex* dAT, int64_t ldat,*/
               sycl::double2 *dAT, int64_t ldat, blas::Queue &queue)
{
/* DPCT_ORIG     transpose<cuDoubleComplex,16>(*/
    transpose<sycl::double2, 16>(is_conj, m, n, dA, lda, dAT, ldat, queue);
}

// ----------------------------------------
template<>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    float **dA_array,  int64_t lda,
    float **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch<float,32>(
        is_conj,
        m, n,
        dA_array,  lda,
        dAT_array, ldat,
        batch_count,
        queue);
}

template<>
void transpose_batch(
    bool is_conj,
    int64_t m, int64_t n,
    double **dA_array,  int64_t lda,
    double **dAT_array, int64_t ldat,
    int64_t batch_count,
    blas::Queue& queue)
{
    transpose_batch<double,32>(
        is_conj,
        m, n,
        dA_array,  lda,
        dAT_array, ldat,
        batch_count,
        queue);
}

template <>
void transpose_batch(
    bool is_conj, int64_t m, int64_t n,
    /* DPCT_ORIG     cuFloatComplex **dA_array,  int64_t lda,*/
    sycl::float2 **dA_array, int64_t lda,
    /* DPCT_ORIG     cuFloatComplex **dAT_array, int64_t ldat,*/
    sycl::float2 **dAT_array, int64_t ldat, int64_t batch_count,
    blas::Queue &queue)
{
/* DPCT_ORIG     transpose_batch<cuFloatComplex,32>(*/
    transpose_batch<sycl::float2, 32>(
        is_conj, m, n, dA_array, lda, dAT_array, ldat, batch_count, queue );
}

template <>
void transpose_batch(
    bool is_conj, int64_t m, int64_t n,
    /* DPCT_ORIG     cuDoubleComplex **dA_array,  int64_t lda,*/
    sycl::double2 **dA_array, int64_t lda,
    /* DPCT_ORIG     cuDoubleComplex **dAT_array, int64_t ldat,*/
    sycl::double2 **dAT_array, int64_t ldat, int64_t batch_count,
    blas::Queue &queue)
{
/* DPCT_ORIG     transpose_batch<cuDoubleComplex,16>(*/
    transpose_batch<sycl::double2, 16>(
        is_conj,
        m, n,
        dA_array, lda,
        dAT_array, ldat,
        batch_count,
        queue );
}

} // namespace device
} // namespace slate
