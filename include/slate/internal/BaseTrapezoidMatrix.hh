//------------------------------------------------------------------------------
// Copyright (c) 2017, University of Tennessee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the University of Tennessee nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL UNIVERSITY OF TENNESSEE BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//------------------------------------------------------------------------------
// This research was supported by the Exascale Computing Project (17-SC-20-SC),
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//------------------------------------------------------------------------------
// For assistance with SLATE, email <slate-user@icl.utk.edu>.
// You can also join the "SLATE User" Google group by going to
// https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user,
// signing in with your Google credentials, and then clicking "Join group".
//------------------------------------------------------------------------------

#ifndef SLATE_BASE_TRAPEZOID_MATRIX_HH
#define SLATE_BASE_TRAPEZOID_MATRIX_HH

#include "slate/internal/BaseMatrix.hh"
#include "slate/Matrix.hh"
#include "slate/Tile.hh"
#include "slate/types.hh"
#include "slate/Exception.hh"

#include "lapack.hh"

#include <algorithm>
#include <utility>

#include "slate/internal/cuda.hh"
#include "slate/internal/cublas.hh"
#include "slate/internal/mpi.hh"
#include "slate/internal/openmp.hh"

namespace slate {

//==============================================================================
/// Base class for all SLATE distributed, tiled trapezoidal storage matrices.
/// Either the upper or lower trapezoid is stored, with the opposite triangle
/// assumed by symmetry (SymmetricMatrix, HermitianMatrix),
/// or assumed to be zero (TrapezoidMatrix, TriangularMatrix).
///
template <typename scalar_t>
class BaseTrapezoidMatrix: public BaseMatrix<scalar_t> {
public:
    using ij_tuple = std::tuple<int64_t, int64_t>;

protected:
    // constructors
    BaseTrapezoidMatrix();

    BaseTrapezoidMatrix(Uplo uplo, int64_t m, int64_t n, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm);

    // conversion
    BaseTrapezoidMatrix(Uplo uplo, Matrix<scalar_t>& orig);

    // off-diagonal sub-matrix
    Matrix<scalar_t> sub(int64_t i1, int64_t i2, int64_t j1, int64_t j2);

    // used by sub-classes' fromLAPACK and fromScaLAPACK
    BaseTrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                        scalar_t* A, int64_t lda, int64_t nb,
                        int p, int q, MPI_Comm mpi_comm, bool is_scalapack);

    // used by sub-classes' fromDevices
    BaseTrapezoidMatrix(Uplo uplo, int64_t m, int64_t n,
                        scalar_t** Aarray, int num_devices, int64_t lda,
                        int64_t nb, int p, int q, MPI_Comm mpi_comm);

    // used by sub-classes' off-diagonal sub
    BaseTrapezoidMatrix(Uplo uplo, Matrix<scalar_t>& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2);

    // used by sub
    BaseTrapezoidMatrix(BaseTrapezoidMatrix& orig,
                        int64_t i1, int64_t i2,
                        int64_t j1, int64_t j2);

public:
    template <typename T>
    friend void swap(BaseTrapezoidMatrix<T>& A, BaseTrapezoidMatrix<T>& B);

    int64_t getMaxHostTiles();
    int64_t getMaxDeviceTiles(int device);
    void allocateBatchArrays(int64_t batch_size=0);
    void reserveHostWorkspace();
    void reserveDeviceWorkspace();
    void gather(scalar_t* A, int64_t lda);
    Uplo uplo_logical() const { return this->uploLogical(); }  ///< @deprecated
    void insertLocalTiles(Target origin=Target::Host);
    void insertLocalTiles(bool on_devices);

    void tileGetAllForReading(int device, LayoutConvert layout);
    void tileGetAllForReadingOnDevices(LayoutConvert layout);
    void tileGetAllForWriting(int device, LayoutConvert layout);
    void tileGetAllForWritingOnDevices(LayoutConvert layout);
    void tileGetAndHoldAll(int device, LayoutConvert layout);
    void tileGetAndHoldAllOnDevices(LayoutConvert layout);
    void tileUnsetHoldAll(int device=hostNum());
    void tileUnsetHoldAllOnDevices();
    void tileUpdateAllOrigin();
    void tileLayoutReset();
    int  hostNum()  const { return this->host_num_; }
};

//--------------------------------------------------------------------------
/// Default constructor creates an empty matrix.
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix()
    : BaseMatrix<scalar_t>()
{
    this->uplo_ = Uplo::Lower;
}

//------------------------------------------------------------------------------
/// Constructor creates an m-by-n matrix, with no tiles allocated.
/// Tiles can be added with tileInsert().
//
// todo: have allocate flag? If true, allocate data; else user will insert tiles?
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, int64_t m, int64_t n, int64_t nb, int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;
}

//------------------------------------------------------------------------------
/// Used by subclasses' fromLAPACK and fromScaLAPACK.
/// Construct matrix by wrapping existing memory of an m-by-n lower
/// or upper trapezoidal storage (Sca)LAPACK matrix. Triangular, symmetric, and
/// Hermitian matrices all use this storage scheme (with m = n).
/// The caller must ensure that the memory remains valid for the lifetime
/// of the Matrix object and any shallow copies of it.
/// Input format is an LAPACK-style column-major matrix or
/// ScaLAPACK-style 2D block-cyclic column-major matrix
/// with local leading dimension (column stride) lda,
/// p block rows and q block columns.
/// Matrix gets tiled with square nb-by-nb tiles.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] A
///     The local portion of the 2D block cyclic distribution of
///     the m-by-n matrix A, with local leading dimension lda.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
/// @param[in] is_scalapack
///     If true,  A is a ScaLAPACK matrix.
///     If false, A is an LAPACK matrix.
///
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, int64_t m, int64_t n,
    scalar_t* A, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm, bool is_scalapack)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;

    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    if (this->uplo() == Uplo::Lower) {
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);
            int64_t jj_local = jj;
            if (is_scalapack) {
                jj_local = indexGlobal2Local(jj, nb, q);
            }

            int64_t ii = j*nb;
            for (int64_t i = j; i < this->mt(); ++i) {  // lower
                int64_t ib = this->tileMb(i);
                int64_t ii_local = ii;
                if (is_scalapack) {
                    ii_local = indexGlobal2Local(ii, nb, p);
                }

                if (this->tileIsLocal(i, j)) {
                    this->tileInsert(i, j, this->host_num_,
                                     &A[ii_local + jj_local*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
    else {  // Upper
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);
            int64_t jj_local = jj;
            if (is_scalapack) {
                jj_local = indexGlobal2Local(jj, nb, q);
            }

            int64_t ii = 0;
            for (int64_t i = 0; i <= j && i < this->mt(); ++i) {  // upper
                int64_t ib = this->tileMb(i);
                int64_t ii_local = ii;
                if (is_scalapack) {
                    ii_local = indexGlobal2Local(ii, nb, p);
                }

                if (this->tileIsLocal(i, j)) {
                    this->tileInsert(i, j, this->host_num_,
                                     &A[ii_local + jj_local*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
}

//------------------------------------------------------------------------------
/// Used by subclasses' fromDevices.
/// Construct matrix by wrapping existing memory of an m-by-n lower
/// or upper trapezoidal storage that is 2D block-cyclic across nodes
/// and 1D block-cyclic across GPU devicse within a node.
///
/// @param[in] m
///     Number of rows of the matrix. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix. n >= 0.
///
/// @param[in,out] Aarray
///     TODO
///     The local portion of the 2D block cyclic distribution of
///     the m-by-n matrix A, with local leading dimension lda.
///
/// @param[in] num_devices
///     Dimension of Aarray.
///
/// @param[in] lda
///     Local leading dimension of the array A. lda >= local number of rows.
///
/// @param[in] nb
///     Block size in 2D block-cyclic distribution. nb > 0.
///
/// @param[in] p
///     Number of block rows in 2D block-cyclic distribution. p > 0.
///
/// @param[in] q
///     Number of block columns of 2D block-cyclic distribution. q > 0.
///
/// @param[in] mpi_comm
///     MPI communicator to distribute matrix across.
///     p*q == MPI_Comm_size(mpi_comm).
///
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, int64_t m, int64_t n,
    scalar_t** Aarray, int num_devices, int64_t lda, int64_t nb,
    int p, int q, MPI_Comm mpi_comm)
    : BaseMatrix<scalar_t>(m, n, nb, p, q, mpi_comm)
{
    slate_error_if(this->num_devices() != num_devices);
    slate_error_if(uplo == Uplo::General);
    this->uplo_ = uplo;

    // ii, jj are row, col indices
    // ii_local and jj_local are the local array indices in A
    // 2D block-cyclic layout.
    // jj_dev is the local array index for the current device in A
    // 1D block-cyclic layout within a node.
    // i, j are tile (block row, block col) indices
    if (this->uplo() == Uplo::Lower) {
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);
            int64_t jj_local = indexGlobal2Local(jj, nb, q);

            int64_t ii = j*nb;
            for (int64_t i = j; i < this->mt(); ++i) {  // lower
                int64_t ib = this->tileMb(i);
                int64_t ii_local = indexGlobal2Local(ii, nb, p);

                if (this->tileIsLocal(i, j)) {
                    int dev = this->tileDevice(i, j);
                    int64_t jj_dev
                        = indexGlobal2Local(jj_local, nb, num_devices);
                    this->tileInsert(
                        i, j, dev, &Aarray[dev][ii_local + jj_dev*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
    else {  // Upper
        int64_t jj = 0;
        for (int64_t j = 0; j < this->nt(); ++j) {
            int64_t jb = this->tileNb(j);
            int64_t jj_local = indexGlobal2Local(jj, nb, q);

            int64_t ii = 0;
            for (int64_t i = 0; i <= j && i < this->mt(); ++i) {  // upper
                int64_t ib = this->tileMb(i);
                int64_t ii_local = indexGlobal2Local(ii, nb, p);

                if (this->tileIsLocal(i, j)) {
                    int dev = this->tileDevice(i, j);
                    int64_t jj_dev
                        = indexGlobal2Local(jj_local, nb, num_devices);
                    this->tileInsert(
                        i, j, dev, &Aarray[dev][ii_local + jj_dev*lda], lda);
                }
                ii += ib;
            }
            jj += jb;
        }
    }
}

//------------------------------------------------------------------------------
/// Conversion from general matrix
/// creates shallow copy view of original matrix.
///
/// Requires the original matrix to have mb == nb, so that diagonal tiles are
/// square (except perhaps the last one). Currently this is enforced for
/// only the (0, 0) tile when there are more than one block rows and columns.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] orig
///     Original matrix.
///
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, Matrix<scalar_t>& orig)
    : BaseMatrix<scalar_t>(orig)
{
    slate_error_if(uplo == Uplo::General);
    slate_assert(orig.mt() <= 1 || orig.nt() <= 1 ||
                 orig.tileMb(0) == orig.tileNb(0));
    this->uplo_ = uplo;
}

//------------------------------------------------------------------------------
/// Conversion from general matrix, sub-matrix constructor
/// creates shallow copy view of original matrix, A[ i1:i2, j1:j2 ].
///
/// Requires the original matrix to have mb == nb, so that diagonal tiles are
/// square (except perhaps the last one). Currently this is enforced for
/// only the (0, 0) tile when there are more than one block rows and columns.
///
/// @param[in] uplo
///     - Upper: upper triangle of A is stored.
///     - Lower: lower triangle of A is stored.
///
/// @param[in] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    Uplo uplo, Matrix<scalar_t>& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{
    slate_error_if(uplo == Uplo::General);
    slate_assert(orig.mt() <= 1 || orig.nt() <= 1 ||
                 orig.tileMb(0) == orig.tileNb(0));
    this->uplo_ = uplo;
}

//------------------------------------------------------------------------------
/// Sub-matrix constructor creates shallow copy view of parent matrix,
/// A[ i1:i2, j1:j2 ]. The new view is still a trapezoid matrix.
/// If lower, requires i1 >= j1 and (i2 - i1) >= (j2 - j1).
/// If upper, requires i1 <= j1 and (i2 - i1) <= (j2 - j1).
/// If i1 == j1, it has the same diagonal as the parent matrix.
///
/// @param[in] orig
///     Original matrix.
///
/// @param[in] i1
///     Starting block row index. 0 <= i1 < mt.
///
/// @param[in] i2
///     Ending block row index (inclusive). i2 < mt.
///
/// @param[in] j1
///     Starting block column index. 0 <= j1 < nt.
///
/// @param[in] j2
///     Ending block column index (inclusive). j2 < nt.
///
template <typename scalar_t>
BaseTrapezoidMatrix<scalar_t>::BaseTrapezoidMatrix(
    BaseTrapezoidMatrix& orig,
    int64_t i1, int64_t i2,
    int64_t j1, int64_t j2)
    : BaseMatrix<scalar_t>(orig, i1, i2, j1, j2)
{
    this->uplo_ = orig.uplo_;
    if (this->uplo_ == Uplo::Lower) {
        if (i1 < j1 || (i2 - i1) < (j2 - j1))
            throw std::exception();
    }
    else {
        // Upper
        if (i1 > j1 || (i2 - i1) > (j2 - j1))
            throw std::exception();
    }
}

//------------------------------------------------------------------------------
/// Swap contents of matrices A and B.
template <typename scalar_t>
void swap(BaseTrapezoidMatrix<scalar_t>& A, BaseTrapezoidMatrix<scalar_t>& B)
{
    using std::swap;
    swap(static_cast< BaseMatrix<scalar_t>& >(A),
         static_cast< BaseMatrix<scalar_t>& >(B));
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank.
// todo: numLocalTiles? use for life as well?
template <typename scalar_t>
int64_t BaseTrapezoidMatrix<scalar_t>::getMaxHostTiles()
{
    int64_t num_tiles = 0;
    if (this->uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = j; i < this->mt(); ++i)  // lower
                if (this->tileIsLocal(i, j))
                    ++num_tiles;
    }
    else {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = 0; i <= j && i < this->mt(); ++i)  // upper
                if (this->tileIsLocal(i, j))
                    ++num_tiles;
    }

    return num_tiles;
}

//------------------------------------------------------------------------------
/// Returns number of local tiles of the matrix on this rank and given device.
// todo: numLocalDeviceTiles
template <typename scalar_t>
int64_t BaseTrapezoidMatrix<scalar_t>::getMaxDeviceTiles(int device)
{
    int64_t num_tiles = 0;
    if (this->uplo() == Uplo::Lower) {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = j; i < this->mt(); ++i)  // lower
                if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                    ++num_tiles;
    }
    else {
        for (int64_t j = 0; j < this->nt(); ++j)
            for (int64_t i = 0; i <= j && i < this->mt(); ++i)  // upper
                if (this->tileIsLocal(i, j) && this->tileDevice(i, j) == device)
                    ++num_tiles;
    }
    return num_tiles;
}

//------------------------------------------------------------------------------
/// Allocates batch arrays for all devices.
/// This overrides BaseMatrix::allocateBatchArrays.
///
/// @param[in] batch_size
///     On exit, size of batch arrays >= batch_size >= 0.
///     If batch_size = 0 (default), uses batch_size = getMaxDeviceTiles.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::allocateBatchArrays(int64_t batch_size)
{
    if (batch_size == 0) {
        for (int device = 0; device < this->num_devices_; ++device)
            batch_size = std::max(batch_size, getMaxDeviceTiles(device));
    }
    this->storage_->allocateBatchArrays(batch_size);
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on host.
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::reserveHostWorkspace()
{
    this->storage_->reserveHostWorkspace(getMaxHostTiles());
}

//------------------------------------------------------------------------------
/// Reserve space for temporary workspace tiles on all GPU devices.
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::reserveDeviceWorkspace()
{
    int64_t num_tiles = 0;
    for (int device = 0; device < this->num_devices_; ++device)
        num_tiles = std::max(num_tiles, getMaxDeviceTiles(device));
    this->storage_->reserveDeviceWorkspace(num_tiles);
}

//------------------------------------------------------------------------------
/// Gathers the entire matrix to the LAPACK-style matrix A on MPI rank 0.
/// Primarily for debugging purposes.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::gather(scalar_t* A, int64_t lda)
{
    // this code assumes the matrix is not transposed
    Op op_save = this->op();
    this->op_ = Op::NoTrans;

    // ii, jj are row, col indices
    // i, j are tile (block row, block col) indices
    int64_t jj = 0;
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t jb = this->tileNb(j);

        int64_t ii = 0;
        for (int64_t i = 0; i < this->mt(); ++i) {
            int64_t ib = this->tileMb(i);

            if ((this->uplo() == Uplo::Lower && i >= j) ||
                (this->uplo() == Uplo::Upper && i <= j)) {
                if (this->mpi_rank_ == 0) {
                    if (! this->tileIsLocal(i, j)) {
                        // erase any existing non-local tile and insert new one
                        this->tileErase(i, j, this->host_num_);
                        this->tileInsert(i, j, this->host_num_,
                                         &A[(size_t)lda*jj + ii], lda);
                        auto Aij = this->at(i, j);
                        Aij.recv(this->tileRank(i, j), this->mpi_comm_, this->layout());
                        this->tileLayout(i, j, this->layout_);
                    }
                    else {
                        this->tileGetForReading(i, j, LayoutConvert(this->layout()));
                        // copy local tiles if needed.
                        auto Aij = this->at(i, j);
                        if (Aij.data() != &A[(size_t)lda*jj + ii]) {
                            lapack::lacpy(lapack::MatrixType::General, ib, jb,
                                          Aij.data(), Aij.stride(),
                                          &A[(size_t)lda*jj + ii], lda);
                        }
                    }
                }
                else if (this->tileIsLocal(i, j)) {
                    this->tileGetForReading(i, j, LayoutConvert(this->layout()));
                    auto Aij = this->at(i, j);
                    Aij.send(0, this->mpi_comm_);
                }
            }
            ii += ib;
        }
        jj += jb;
    }

    this->op_ = op_save;
}

//------------------------------------------------------------------------------
/// Returns off-diagonal sub-matrix that is a shallow copy view of the
/// parent matrix, A[ i1:i2, j1:j2 ].
/// This version returns a general Matrix, which:
/// - if uplo = Lower, is strictly below the diagonal, or
/// - if uplo = Upper, is strictly above the diagonal.
///
template <typename scalar_t>
Matrix<scalar_t> BaseTrapezoidMatrix<scalar_t>::sub(
    int64_t i1, int64_t i2, int64_t j1, int64_t j2)
{
    if (this->uplo() == Uplo::Lower) {
        // require top-right corner (i1, j2) to be at or below diagonal
        if (i1 < j2)
            slate_error("submatrix outside lower triangle; requires i1 >= j2");
    }
    else {
        // require bottom-left corner (i2, j1) to be at or above diagonal
        if (i2 > j1)
            slate_error("submatrix outside upper triangle; requires i2 <= j1");
    }
    return Matrix<scalar_t>(*this, i1, i2, j1, j2);
}

//------------------------------------------------------------------------------
/// Move all tiles back to their origin.
//
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileUpdateAllOrigin()
{
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j))
                this->tileUpdateOrigin(i, j);
        }
    }
}

//------------------------------------------------------------------------------
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] target
///     - if target = Devices, inserts tiles on appropriate GPU devices, or
///     - if target = Host, inserts on tiles on CPU host.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::insertLocalTiles(Target origin)
{
    bool on_devices = (origin == Target::Devices);
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                int dev = (on_devices ? this->tileDevice(i, j)
                                      : this->host_num_);
                this->tileInsert(i, j, dev);
            }
        }
    }
}

//------------------------------------------------------------------------------
/// @deprecated
///
/// Inserts all local tiles into an empty matrix.
///
/// @param[in] on_devices
///     If on_devices, inserts tiles on appropriate GPU devices,
///     otherwise inserts tiles on CPU host.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::insertLocalTiles(bool on_devices)
{
    insertLocalTiles(on_devices ? Target::Devices : Target::Host);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on device.
/// @see tileGetForReading.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileGetAllForReading(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }
        }
    }

    this->tileGetForReading(tiles_set, device, layout);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for writing on device.
/// @see tileGetForWriting.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileGetAllForWriting(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }
        }
    }

    this->tileGetForWriting(tiles_set, device, layout);
}

//------------------------------------------------------------------------------
/// Gets all local tiles on device and marks them as MOSI::OnHold.
/// @see tileGetAndHold.
///
/// @param[in] device
///     Tile's destination: host or device ID, defaults to host.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileGetAndHoldAll(int device, LayoutConvert layout)
{
    std::set<ij_tuple> tiles_set;
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                tiles_set.insert({i, j});
            }
        }
    }

    this->tileGetAndHold(tiles_set, layout, device);
}

//------------------------------------------------------------------------------
/// Gets all local tiles for reading on corresponding devices.
/// @see tileGetForReading.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileGetAllForReadingOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(this->num_devices());
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                tiles_set[this->tileDevice(i, j)].insert({i, j});
            }
        }
    }

    // todo: omp parallel for?
    for (int d = 0; d < this->num_devices(); ++d) {
        this->tileGetForReading(tiles_set[d], d, layout);
    }
}

//------------------------------------------------------------------------------
/// Gets all local tiles for writing on corresponding devices.
/// @see tileGetForWriting.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileGetAllForWritingOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(this->num_devices());
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                tiles_set[this->tileDevice(i, j)].insert({i, j});
            }
        }
    }

    // todo: omp parallel for?
    for (int d = 0; d < this->num_devices(); ++d) {
        this->tileGetForWriting(tiles_set[d], d, layout);
    }
}

//------------------------------------------------------------------------------
/// Gets all local tiles on corresponding devices and marks them as MOSI::OnHold.
/// @see tileGetAndHold.
///
/// @param[in] layout
///     Indicates whether to convert the Layout of the received data:
///     - ColMajor: convert layout to column major.
///     - RowMajor: convert layout to row major.
///     - None: do not convert layout.
//
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileGetAndHoldAllOnDevices(LayoutConvert layout)
{
    std::vector< std::set<ij_tuple> > tiles_set(this->num_devices());
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {
                tiles_set[this->tileDevice(i, j)].insert({i, j});
            }
        }
    }

    // todo: omp parallel for?
    for (int d = 0; d < this->num_devices(); ++d) {
        this->tileGetAndHold(tiles_set[d], d, layout);
    }
}

//------------------------------------------------------------------------------
/// Unsets all local tiles' hold on device.
///
/// @param[in] device
///     Tile's device ID.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileUnsetHoldAll(int device)
{
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j))
                this->tileUnsetHold(i, j, device);
        }
    }
}

//------------------------------------------------------------------------------
/// Unsets all local tiles' hold on all devices.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileUnsetHoldAllOnDevices()
{
    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j))
                this->tileUnsetHold(i, j, this->tileDevice(i, j));
        }
    }
}

//------------------------------------------------------------------------------
/// Converts all origin tiles into current matrix-layout.
/// Operates in batch mode.
///
template <typename scalar_t>
void BaseTrapezoidMatrix<scalar_t>::tileLayoutReset()
{
    std::set<ij_tuple> tiles_set_host;
    std::vector< std::set<ij_tuple> > tiles_set_dev(this->num_devices());

    int64_t mt = this->mt();
    for (int64_t j = 0; j < this->nt(); ++j) {
        int64_t istart = (this->uplo() == Uplo::Lower ? j : 0);
        int64_t iend   = (this->uplo() == Uplo::Lower ? mt : std::min( j+1, mt ));
        for (int64_t i = istart; i < iend; ++i) {
            if (this->tileIsLocal(i, j)) {

                auto tile = this->tileUpdateOrigin(i, j);
                if (tile->layout() != this->layout() ) {
                    assert(tile->isTransposable());
                }

                if (tile->device() == hostNum()) {
                    tiles_set_host.insert({i, j});
                }
                else{
                    tiles_set_dev[tile->device()].insert({i, j});
                }
            }
        }
    }

    if (! tiles_set_host.empty()) {
        this->tileLayoutReset(tiles_set_host, hostNum(), this->layout());
    }
    // todo: omp tasks?
    for (int d = 0; d < this->num_devices(); ++d) {
        if (! tiles_set_dev[d].empty()) {
            this->tileLayoutReset(tiles_set_dev[d], d, this->layout());
        }
    }
}

} // namespace slate

#endif // SLATE_BASE_TRAPEZOID_MATRIX_HH