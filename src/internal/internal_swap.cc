// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "slate/Matrix.hh"
#include "slate/types.hh"
#include "slate/Tile_blas.hh"
#include "internal/internal.hh"

#include <map>
#include <vector>

namespace slate {
namespace internal {

//------------------------------------------------------------------------------
/// Converts serial pivot vector to parallel pivot map.
///
/// @param[in] direction
///     Direction of pivoting:
///     - Direction::Forward,
///     - Direction::Backward.
///
/// @param[in] in_pivot
///     Serial (LAPACK-style) pivot vector.
///
/// @param[in,out] pivot
///     Parallel pivot for out-of-place pivoting.
///
/// @ingroup permute_internal
///
void makeParallelPivot(
    Direction direction,
    std::vector<Pivot> const& pivot,
    std::map<Pivot, Pivot>& pivot_map)
{
    int64_t begin, end, inc;
    if (direction == Direction::Forward) {
        begin = 0;
        end = pivot.size();
        inc = 1;
    }
    else {
        begin = pivot.size()-1;
        end = -1;
        inc = -1;
    }

    // Put the participating rows in the map.
    for (int64_t i = begin; i != end; i += inc) {
        if (pivot[i] != Pivot(0, i)) {
            pivot_map[ Pivot(0, i) ] = Pivot(0, i);
            pivot_map[ pivot[i]    ] = pivot[i];
        }
    }

    // Perform pivoting in the map.
    for (int64_t i = begin; i != end; i += inc)
        if (pivot[i] != Pivot(0, i))
            std::swap( pivot_map[ pivot[i] ], pivot_map[ Pivot(0, i) ] );
/*
    std::cout << std::endl;
    for (int64_t i = begin; i != end; i += inc)
        std::cout << pivot[i].tileIndex() << "\t"
                  << pivot[i].elementOffset() << std::endl;

    std::cout << std::endl;
    for (auto it : pivot_map)
        std::cout << it.first.tileIndex() << "\t"
                  << it.first.elementOffset() << "\t\t"
                  << it.second.tileIndex() << "\t"
                  << it.second.elementOffset() << std::endl;

    std::cout << "---------------------------" << std::endl;
*/
}

/*
//------------------------------------------------------------------------------
template <Target target, typename scalar_t>
void permuteRows(
    Direction direction,
    Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag)
{
    permuteRows(internal::TargetType<target>(), direction, A, pivot,
                layout, priority, tag);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostTask>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot_vec,
    Layout layout, int priority, int tag)
{
    // CPU uses ColMajor
    assert(layout == Layout::ColMajor);

    std::map<Pivot, Pivot> pivot_map;
    makeParallelPivot(direction, pivot_vec, pivot_map);

    // todo: for performance optimization, merge with the loops below,
    // at least with lookahead, probably selectively
    A.tileGetAllForWriting(A.hostNum(), LayoutConvert(layout));

    {
        trace::Block trace_block("internal::permuteRows");

        for (int64_t j = 0; j < A.nt(); ++j) {
            int64_t nb = A.tileNb(j);

            std::vector<MPI_Request> requests;
            std::vector<MPI_Status> statuses;

            // Make copies of src rows.
            // Make room for dst rows.
            std::map<Pivot, std::vector<scalar_t> > src_rows;
            std::map<Pivot, std::vector<scalar_t> > dst_rows;
            for (auto const& pivot : pivot_map) {

                bool src_local = A.tileIsLocal(pivot.second.tileIndex(), j);
                if (src_local) {
                    src_rows[pivot.second].resize(nb);
                    copyRow(nb, A(pivot.second.tileIndex(), j),
                            pivot.second.elementOffset(), 0,
                            src_rows[pivot.second].data());
                }
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);
                if (dst_local)
                    dst_rows[pivot.first].resize(nb);
            }

            // Local swap.
            for (auto const& pivot : pivot_map) {

                bool src_local = A.tileIsLocal(pivot.second.tileIndex(), j);
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);

                if (src_local && dst_local) {
                    memcpy(dst_rows[pivot.first].data(),
                           src_rows[pivot.second].data(),
                           sizeof(scalar_t)*nb);
                }
            }

            // Launch all MPI.
            for (auto const& pivot : pivot_map) {

                bool src_local = A.tileIsLocal(pivot.second.tileIndex(), j);
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);

                if (src_local && ! dst_local) {

                    requests.resize(requests.size()+1);
                    int dest = A.tileRank(pivot.first.tileIndex(), j);
                    MPI_Isend(src_rows[pivot.second].data(), nb,
                              mpi_type<scalar_t>::value, dest, tag, A.mpiComm(),
                              &requests[requests.size()-1]);
                }
                if (! src_local && dst_local) {

                    requests.resize(requests.size()+1);
                    int source = A.tileRank(pivot.second.tileIndex(), j);
                    MPI_Irecv(dst_rows[pivot.first].data(), nb,
                              mpi_type<scalar_t>::value, source, tag,
                              A.mpiComm(), &requests[requests.size()-1]);
                }
            }

            // Waitall.
            statuses.resize(requests.size());
            MPI_Waitall(requests.size(), requests.data(), statuses.data());

            for (auto const& pivot : pivot_map) {
                bool dst_local = A.tileIsLocal(pivot.first.tileIndex(), j);
                if (dst_local) {
                    copyRow(nb, dst_rows[pivot.first].data(),
                            A(pivot.first.tileIndex(), j),
                            pivot.first.elementOffset(), 0);
                }
            }
        }
    }
}
*/

//------------------------------------------------------------------------------
/// Permutes rows of a general matrix according to the pivot vector.
/// Host implementation.
/// todo: Restructure similarly to Hermitian permuteRowsCols
///       (use the auxiliary swap functions).
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostTask>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index)
{

    // todo: for performance optimization, merge with the loops below,
    // at least with lookahead, probably selectively
    A.tileGetAllForWriting(A.hostNum(), LayoutConvert(layout));

    {
        trace::Block trace_block("internal::permuteRows");


        MPI_Datatype mpi_scalar = mpi_type<scalar_t>::value;

        // todo: what about parallelizing this? MPI blocking?
        for (int64_t j = 0; j < A.nt(); ++j) {
            int root_rank = A.tileRank(0, j);
            bool root = A.mpiRank() == A.tileRank(0, j);

            // Apply pivots forward (0, ..., k-1) or reverse (k-1, ..., 0)
            int64_t begin, end, inc;
            if (direction == Direction::Forward) {
                begin = 0;
                end   = pivot.size();
                inc   = 1;
            }
            else {
                begin = pivot.size() - 1;
                end   = -1;
                inc   = -1;
            }

            // process pivots
            int64_t nb = A.tileNb(j);
            int comm_size;
            MPI_Comm_size(A.mpiComm(), &comm_size);

            MPI_Datatype row_type;
            MPI_Type_contiguous(nb, mpi_scalar, &row_type);
            MPI_Type_commit(&row_type);

            // row indices are stored in int b/c gather/scatter can't use int64_t's
            std::vector<int> remote_lengths(comm_size + 1);
            for (int64_t i = begin; i != end; i += inc) {
                auto piv = pivot[i];
                auto swap_rank = A.tileRank(piv.tileIndex(), j);
                if (root_rank != swap_rank) {
                    ++remote_lengths[swap_rank];
                }
            }
            std::vector<int> remote_index(comm_size);
            std::vector<int> remote_offsets(comm_size+1);
            for (int i = 0; i < comm_size; ++i) {
                remote_offsets[i+1] += remote_offsets[i] + remote_lengths[i];
                remote_index[i] = remote_offsets[i];
            }
            std::map<Pivot, int> remote_pivot_table;
            for (int64_t i = begin; i != end; i += inc) {
                auto piv = pivot[i];
                auto swap_rank = A.tileRank(piv.tileIndex(), j);
                if (root_rank != swap_rank
                    && remote_pivot_table.find(piv) == remote_pivot_table.end()) {
                    int index = remote_index[swap_rank];
                    ++remote_index[swap_rank];
                    remote_pivot_table.insert({piv, index});
                }
            }
            for (int64_t i = 0; i < comm_size; ++i) {
                // trim the lengths to their actual values
                remote_lengths[i] = remote_index[i] - remote_offsets[i];
            }

            if (root) {
                std::vector<scalar_t> remote_rows_vect (remote_offsets[comm_size]*nb);
                scalar_t* remote_rows = remote_rows_vect.data();

                tagged_gatherv(nullptr, 0, row_type,
                               remote_rows, remote_lengths.data(), remote_offsets.data(), row_type,
                               root_rank, tag, A.mpiComm());

                int64_t stride_0j = A(0, j).rowIncrement();

                for (int64_t i = begin; i != end; i += inc) {
                    int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);

                    if (pivot_rank == root_rank) {
                        // If pivot not on the diagonal.
                        if (pivot[i].tileIndex() > 0 ||
                            pivot[i].elementOffset() > i)
                        {

                            // todo: assumes 1-D block cyclic
                            int64_t i1 = i;
                            int64_t i2 = pivot[i].elementOffset();
                            int64_t idx2 = pivot[i].tileIndex();

                            int64_t stride_idx2j = A(idx2, j).rowIncrement();

                            blas::swap(
                                A.tileNb(j),
                                &A(0,    j).at(i1, 0), stride_0j,
                                &A(idx2, j).at(i2, 0), stride_idx2j);
                        }
                    } else {
                        auto remote_index = remote_pivot_table[pivot[i]];
                        blas::swap(
                            A.tileNb(j),
                            &A(0, j).at(i, 0), stride_0j,
                            remote_rows + nb*remote_index, 1);
                    }
                }

                tagged_scatterv(remote_rows, remote_lengths.data(), remote_offsets.data(), row_type,
                                nullptr, 0, row_type,
                                root_rank, tag, A.mpiComm());
            } else {
                std::vector<scalar_t> remote_rows_vect (remote_lengths[A.mpiRank()]*nb);
                scalar_t* remote_rows = remote_rows_vect.data();

                int64_t remote_start  = remote_offsets[A.mpiRank()];
                int64_t remote_length = remote_lengths[A.mpiRank()];

                int64_t count = 0;
                for (int64_t i = begin; i != end; i += inc) {
                    int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);
                    if (pivot_rank == A.mpiRank()) {
                        auto remote_index = remote_pivot_table[pivot[i]] - remote_start;
                        auto tile_index = pivot[i].tileIndex();
                        auto tile_offset = pivot[i].elementOffset();

                        if (remote_index >= count) {
                            int64_t stride_idxj = A(tile_index, j).rowIncrement();
                            blas::copy(
                                A.tileNb(j),
                                &A(tile_index, j).at(tile_offset, 0), stride_idxj,
                                remote_rows + nb*remote_index, 1);
                            ++count;
                        }
                    }
                }

                tagged_gatherv(remote_rows, remote_length, row_type,
                               nullptr, nullptr, nullptr, row_type,
                               root_rank, tag, A.mpiComm());

                tagged_scatterv(nullptr, nullptr, nullptr, row_type,
                                remote_rows, remote_length, row_type,
                                root_rank, tag, A.mpiComm());

                count = 0;
                for (int64_t i = begin; i != end; i += inc) {
                    int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);
                    if (pivot_rank == A.mpiRank()) {
                        auto remote_index = remote_pivot_table[pivot[i]];
                        auto tile_index = pivot[i].tileIndex();
                        auto tile_offset = pivot[i].elementOffset();

                        if (remote_index >= count) {
                            int64_t stride_idxj = A(tile_index, j).rowIncrement();
                            blas::copy(
                                A.tileNb(j),
                                remote_rows + nb*remote_index, 1,
                                &A(tile_index, j).at(tile_offset, 0), stride_idxj);
                            ++count;
                        }
                    }
                }
            }

            MPI_Type_free(&row_type);
        }
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostNest>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index)
{
    // forward to HostTask
    permuteRows(internal::TargetType<Target::HostTask>(),
                direction, A, pivot, layout, priority, tag, queue_index);
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::HostBatch>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index)
{
    // forward to HostTask
    permuteRows(internal::TargetType<Target::HostTask>(),
                direction, A, pivot, layout, priority, tag, queue_index);
}

//------------------------------------------------------------------------------
/// Permutes rows according to the pivot vector.
/// Dispatches to target implementations.
///
/// @param[in] layout
///     Indicates the Layout (ColMajor/RowMajor) to operate with.
///     Local tiles of matrix on target devices will be converted to layout.
///
/// @ingroup permute_internal
///
template <Target target, typename scalar_t>
void permuteRows(
    Direction direction,
    Matrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index)
{
    permuteRows(internal::TargetType<target>(), direction, A, pivot,
                layout, priority, tag, queue_index);
}

//------------------------------------------------------------------------------
/// Permutes rows of a general matrix according to the pivot vector.
/// GPU device implementation.
/// todo: Restructure similarly to Hermitian permute
///       (use the auxiliary swap functions).
/// todo: Just one function forwarding target.
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void permuteRows(
    internal::TargetType<Target::Devices>,
    Direction direction,
    Matrix<scalar_t>& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index)
{
    // GPU uses RowMajor
    assert(layout == Layout::RowMajor);

    // todo: for performance optimization, merge with the loops below,
    // at least with lookahead, probably selectively
    A.tileGetAllForWritingOnDevices(LayoutConvert(layout));

    {
        trace::Block trace_block("internal::permuteRows");

        MPI_Datatype mpi_scalar = mpi_type<scalar_t>::value;

        std::set<int> dev_set;


        for (int64_t j = 0; j < A.nt(); ++j) {
            int root_rank = A.tileRank(0, j);
            bool root = A.mpiRank() == A.tileRank(0, j);

            // todo: relax the assumption of 1-D block cyclic distribution on devices
            int device = A.tileDevice(0, j);
            dev_set.insert(device);
            blas::set_device(device);

            blas::Queue* compute_queue = A.compute_queue(device, queue_index);

            // Apply pivots forward (0, ..., k-1) or reverse (k-1, ..., 0)
            int64_t begin, end, inc;
            if (direction == Direction::Forward) {
                begin = 0;
                end   = pivot.size();
                inc   = 1;
            }
            else {
                begin = pivot.size() - 1;
                end   = -1;
                inc   = -1;
            }

            // process pivots
            int64_t nb = A.tileNb(j);
            int comm_size;
            MPI_Comm_size(A.mpiComm(), &comm_size);

            MPI_Datatype row_type;
            MPI_Type_contiguous(nb, mpi_scalar, &row_type);
            MPI_Type_commit(&row_type);

            // row indices are stored in int b/c gather/scatter can't use int64_t's
            std::vector<int> remote_lengths(comm_size + 1);
            for (int64_t i = begin; i != end; i += inc) {
                auto piv = pivot[i];
                auto swap_rank = A.tileRank(piv.tileIndex(), j);
                if (root_rank != swap_rank) {
                    ++remote_lengths[swap_rank];
                }
            }
            std::vector<int> remote_index(comm_size);
            std::vector<int> remote_offsets(comm_size+1);
            for (int i = 0; i < comm_size; ++i) {
                remote_offsets[i+1] += remote_offsets[i] + remote_lengths[i];
                remote_index[i] = remote_offsets[i];
            }
            std::map<Pivot, int> remote_pivot_table;
            for (int64_t i = begin; i != end; i += inc) {
                auto piv = pivot[i];
                auto swap_rank = A.tileRank(piv.tileIndex(), j);
                if (root_rank != swap_rank
                    && remote_pivot_table.find(piv) == remote_pivot_table.end()) {
                    int index = remote_index[swap_rank];
                    ++remote_index[swap_rank];
                    remote_pivot_table.insert({piv, index});
                }
            }
            for (int64_t i = 0; i < comm_size; ++i) {
                // trim the lengths to their actual values
                remote_lengths[i] = remote_index[i] - remote_offsets[i];
            }

            if (root) {
                int64_t remote_rows_size = remote_offsets[comm_size]*nb;
                std::vector<scalar_t> remote_rows_vect (remote_rows_size);
                scalar_t* remote_rows = remote_rows_vect.data();

                tagged_gatherv(nullptr, 0, row_type,
                               remote_rows, remote_lengths.data(), remote_offsets.data(), row_type,
                               root_rank, tag, A.mpiComm());

                scalar_t* remote_rows_dev = blas::device_malloc<scalar_t>(remote_rows_size);
                blas::device_memcpy<scalar_t>(remote_rows_dev, remote_rows,
                                              remote_rows_size, *compute_queue);

                for (int64_t i = begin; i != end; i += inc) {
                    int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);

                    if (pivot_rank == root_rank) {
                        // If pivot not on the diagonal.
                        if (pivot[i].tileIndex() > 0 ||
                            pivot[i].elementOffset() > i)
                        {
                            // todo: assumes 1-D block cyclic
                            assert(A(0, j, device).layout() == Layout::RowMajor);
                            int64_t i1 = i;
                            int64_t i2 = pivot[i].elementOffset();
                            int64_t idx2 = pivot[i].tileIndex();
                            blas::swap(
                                A.tileNb(j),
                                &A(0,    j, device).at(i1, 0), 1,
                                &A(idx2, j, device).at(i2, 0), 1,
                                *compute_queue);
                        }
                    } else {
                        auto remote_index = remote_pivot_table[pivot[i]];
                        blas::swap(
                            A.tileNb(j),
                            &A(0, j, device).at(i, 0), 1,
                            remote_rows_dev + nb*remote_index, 1,
                            *compute_queue);
                    }
                }
                compute_queue->sync();

                blas::device_memcpy<scalar_t>(remote_rows, remote_rows_dev,
                                              remote_rows_size, *compute_queue);
                blas::device_free(remote_rows_dev);

                tagged_scatterv(remote_rows, remote_lengths.data(), remote_offsets.data(), row_type,
                                nullptr, 0, row_type,
                                root_rank, tag, A.mpiComm());
            } else if (remote_lengths[A.mpiRank()] > 0) {
                int64_t remote_start  = remote_offsets[A.mpiRank()];
                int64_t remote_length = remote_lengths[A.mpiRank()];
                int64_t remote_rows_size = remote_length*nb;
                scalar_t* remote_rows_dev = blas::device_malloc<scalar_t>(remote_rows_size);

                int64_t count = 0;
                for (int64_t i = begin; i != end; i += inc) {
                    int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);
                    if (pivot_rank == A.mpiRank()) {
                        auto remote_index = remote_pivot_table[pivot[i]] - remote_start;
                        auto tile_index = pivot[i].tileIndex();
                        auto tile_offset = pivot[i].elementOffset();

                        if (remote_index >= count) {
                            // blas::copy( // no device copy in blaspp
                            blas::swap(
                                nb,
                                &A(tile_index, j, device).at(tile_offset, 0), 1,
                                remote_rows_dev + nb*remote_index, 1,
                                *compute_queue);
                            ++count; // only swap the first time
                        }
                    }
                }
                compute_queue->sync();

                std::vector<scalar_t> remote_rows_vect (remote_rows_size);
                scalar_t* remote_rows = remote_rows_vect.data();
                blas::device_memcpy<scalar_t>(remote_rows, remote_rows_dev,
                                              remote_rows_size, *compute_queue);


                tagged_gatherv(remote_rows, remote_length, row_type,
                               nullptr, nullptr, nullptr, row_type,
                               root_rank, tag, A.mpiComm());

                tagged_scatterv(nullptr, nullptr, nullptr, row_type,
                                remote_rows, remote_length, row_type,
                                root_rank, tag, A.mpiComm());

                blas::device_memcpy<scalar_t>(remote_rows_dev, remote_rows,
                                              remote_rows_size, *compute_queue);

                count = 0;
                for (int64_t i = begin; i != end; i += inc) {
                    int pivot_rank = A.tileRank(pivot[i].tileIndex(), j);
                    if (pivot_rank == A.mpiRank()) {
                        auto remote_index = remote_pivot_table[pivot[i]] - remote_start;
                        auto tile_index = pivot[i].tileIndex();
                        auto tile_offset = pivot[i].elementOffset();

                        if (remote_index >= count) {
                            // blas::copy( // no device copy in blaspp
                            blas::swap(
                                A.tileNb(j),
                                remote_rows_dev + nb*remote_index, 1,
                                &A(tile_index, j, device).at(tile_offset, 0), 1,
                                *compute_queue);
                            ++count; // only swap the first time
                        }
                    }
                }
                compute_queue->sync();
                blas::device_free(remote_rows_dev);
            } else {
                // no rows to process but must participate in the collectives
                tagged_gatherv(nullptr, 0, row_type,
                               nullptr, nullptr, nullptr, row_type,
                               root_rank, tag, A.mpiComm());

                tagged_scatterv(nullptr, nullptr, nullptr, row_type,
                                nullptr, 0, row_type,
                                root_rank, tag, A.mpiComm());
            }

            MPI_Type_free(&row_type);
        }

        for (int device : dev_set) {
            A.compute_queue(device, queue_index)->sync();
        }
    }
}

//------------------------------------------------------------------------------
/// Swap a partial row of two tiles, either locally or remotely. Swaps
///     op1( A( ij_tuple_1 ) )[ offset_i1, j_offset : j_offset+n-1 ] and
///     op2( A( ij_tuple_2 ) )[ offset_i2, j_offset : j_offset+n-1 ].
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void swapRow(
    int64_t j_offset, int64_t n,
    HermitianMatrix<scalar_t>& A,
    Op op1, std::tuple<int64_t, int64_t>&& ij_tuple_1, int64_t offset_i1,
    Op op2, std::tuple<int64_t, int64_t>&& ij_tuple_2, int64_t offset_i2,
    int tag)
{
    int64_t i1 = std::get<0>(ij_tuple_1);
    int64_t j1 = std::get<1>(ij_tuple_1);

    int64_t i2 = std::get<0>(ij_tuple_2);
    int64_t j2 = std::get<1>(ij_tuple_2);

    if (A.tileRank(i1, j1) == A.mpiRank()) {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // local swap
            swapLocalRow(
                j_offset, n,
                op1 == Op::NoTrans ? A(i1, j1) : transpose(A(i1, j1)), offset_i1,
                op2 == Op::NoTrans ? A(i2, j2) : transpose(A(i2, j2)), offset_i2);
        }
        else {
            // sending tile 1
            swapRemoteRow(
                j_offset, n,
                op1 == Op::NoTrans ? A(i1, j1) : transpose(A(i1, j1)), offset_i1,
                A.tileRank(i2, j2), A.mpiComm(), tag);
        }
    }
    else if (A.tileRank(i2, j2) == A.mpiRank()) {
        // sending tile 2
        swapRemoteRow(
            j_offset, n,
            op2 == Op::NoTrans ? A(i2, j2) : transpose(A(i2, j2)), offset_i2,
            A.tileRank(i1, j1), A.mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Swap a single element of two tiles, either locally or remotely. Swaps
///     A( ij_tuple_1 )[ offset_i1, offset_j1 ] and
///     A( ij_tuple_2 )[ offset_i2, offset_j2 ].
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void swapElement(
    HermitianMatrix<scalar_t>& A,
    std::tuple<int64_t, int64_t>&& ij_tuple_1,
    int64_t offset_i1, int64_t offset_j1,
    std::tuple<int64_t, int64_t>&& ij_tuple_2,
    int64_t offset_i2, int64_t offset_j2,
    int tag)
{
    int64_t i1 = std::get<0>(ij_tuple_1);
    int64_t j1 = std::get<1>(ij_tuple_1);

    int64_t i2 = std::get<0>(ij_tuple_2);
    int64_t j2 = std::get<1>(ij_tuple_2);

    if (A.tileRank(i1, j1) == A.mpiRank()) {
        if (A.tileRank(i2, j2) == A.mpiRank()) {
            // local swap
            std::swap(A(i1, j1).at(offset_i1, offset_j1),
                      A(i2, j2).at(offset_i2, offset_j2));
        }
        else {
            // sending tile 1
            swapRemoteElement(A(i1, j1), offset_i1, offset_j1,
                              A.tileRank(i2, j2), A.mpiComm(), tag);
        }
    }
    else if (A.tileRank(i2, j2) == A.mpiRank()) {
        // sending tile 2
        swapRemoteElement(A(i2, j2), offset_i2, offset_j2,
                          A.tileRank(i1, j1), A.mpiComm(), tag);
    }
}

//------------------------------------------------------------------------------
/// Permutes rows and cols, symmetrically, of a Hermitian matrix according to
/// the pivot vector.
/// Host implementation.
///
/// @ingroup permute_internal
///
template <typename scalar_t>
void permuteRowsCols(
    internal::TargetType<Target::HostTask>,
    Direction direction,
    HermitianMatrix<scalar_t>& A, std::vector<Pivot>& pivot,
    int priority, int tag)
{
    using blas::conj;

    assert(A.uplo() == Uplo::Lower);

    for (int64_t i = 0; i < A.mt(); ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            if (A.tileIsLocal(i, j)) {
                #pragma omp task shared(A) priority(priority)
                {
                    A.tileGetForWriting(i, j, LayoutConvert::ColMajor);
                }
            }
        }
    }
    #pragma omp taskwait

    {
        trace::Block trace_block("internal::permuteRowsCols");

        // Apply pivots forward (0, ..., k-1) or reverse (k-1, ..., 0)
        int64_t begin, end, inc;
        if (direction == Direction::Forward) {
            begin = 0;
            end   = pivot.size();
            inc   = 1;
        }
        else {
            begin = pivot.size() - 1;
            end   = -1;
            inc   = -1;
        }
        for (int64_t i1 = begin; i1 != end; i1 += inc) {
            int64_t i2 = pivot[i1].elementOffset();
            int64_t j2 = pivot[i1].tileIndex();

            // If pivot not on the diagonal.
            if (j2 > 0 || i2 > i1) {

                // in the upper band
                swapRow(0, i1, A,
                        Op::NoTrans, {0,  0}, i1,
                        Op::NoTrans, {j2, 0}, i2, tag);
                if (j2 == 0) {
                    swapRow(i1+1, i2-i1, A,
                            Op::Trans,   {0, 0}, i1,
                            Op::NoTrans, {0, 0}, i2, tag);

                    swapRow(i2, A.tileNb(0)-i2, A,
                            Op::Trans, {0, 0}, i1,
                            Op::Trans, {0, 0}, i2, tag);
                }
                else {
                    swapRow(i1+1, A.tileNb(0)-i1-1, A,
                            Op::Trans,   {0,  0}, i1,
                            Op::NoTrans, {j2, 0}, i2, tag);

                    // in the lower band
                    swapRow(0, i2, A,
                            Op::Trans,   {j2,  0}, i1,
                            Op::NoTrans, {j2, j2}, i2, tag+1);

                    swapRow(i2+1, A.tileNb(j2)-i2-1, A,
                            Op::Trans, {j2,  0}, i1,
                            Op::Trans, {j2, j2}, i2, tag+1);
                }

                // Conjugate the crossing point.
                if (A.tileRank(j2, 0) == A.mpiRank())
                    A(j2, 0).at(i2, i1) = conj(A(j2, 0).at(i2, i1));

                // Swap the corners.
                swapElement(A,
                            {0, 0}, i1, i1,
                            {j2, j2}, i2, i2, tag);

                // before the lower band
                for (int64_t j1=1; j1 < j2; ++j1) {
                    swapRow(0, A.tileNb(j1), A,
                            Op::Trans,   {j1,  0}, i1,
                            Op::NoTrans, {j2, j1}, i2, tag+1+j1);
                }

                // after the lower band
                for (int64_t j1=j2+1; j1 < A.nt(); ++j1) {
                    swapRow(0, A.tileNb(j1), A,
                            Op::Trans, {j1,  0}, i1,
                            Op::Trans, {j1, j2}, i2, tag+1+j1);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
/// Permutes rows and columns symmetrically according to the pivot vector.
/// Dispatches to target implementations.
/// @ingroup permute_internal
///
template <Target target, typename scalar_t>
void permuteRowsCols(
    Direction direction,
    HermitianMatrix<scalar_t>&& A, std::vector<Pivot>& pivot,
    int priority, int tag)
{
    permuteRowsCols(internal::TargetType<target>(), direction, A, pivot,
                    priority, tag);
}

//------------------------------------------------------------------------------
// Explicit instantiations for (general) Matrix.
// ----------------------------------------
template
void permuteRows<Target::HostTask, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows<Target::HostNest, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows<Target::HostBatch, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows<Target::Devices, float>(
    Direction direction,
    Matrix<float>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

// ----------------------------------------
template
void permuteRows<Target::HostTask, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows<Target::HostNest, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows<Target::HostBatch, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows<Target::Devices, double>(
    Direction direction,
    Matrix<double>&& A, std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

// ----------------------------------------
template
void permuteRows< Target::HostTask, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows< Target::HostNest, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows< Target::HostBatch, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows< Target::Devices, std::complex<float> >(
    Direction direction,
    Matrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

// ----------------------------------------
template
void permuteRows< Target::HostTask, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows< Target::HostNest, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows< Target::HostBatch, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

template
void permuteRows< Target::Devices, std::complex<double> >(
    Direction direction,
    Matrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    Layout layout, int priority, int tag, int queue_index);

//------------------------------------------------------------------------------
// Explicit instantiations for HermitianMatrix.
// ----------------------------------------
template
void permuteRowsCols<Target::HostTask, float>(
    Direction direction,
    HermitianMatrix<float>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void permuteRowsCols<Target::HostTask, double>(
    Direction direction,
    HermitianMatrix<double>&& A, std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void permuteRowsCols< Target::HostTask, std::complex<float> >(
    Direction direction,
    HermitianMatrix< std::complex<float> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

// ----------------------------------------
template
void permuteRowsCols< Target::HostTask, std::complex<double> >(
    Direction direction,
    HermitianMatrix< std::complex<double> >&& A,
    std::vector<Pivot>& pivot,
    int priority, int tag);

} // namespace internal
} // namespace slate
