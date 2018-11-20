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
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUT E GOODS OR SERVICES;
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

#ifndef UTIL_MATRIX_HH
#define UTIL_MATRIX_HH

#include "slate_Matrix.hh"
#include "slate_HermitianMatrix.hh"
#include "slate_SymmetricMatrix.hh"
#include "slate_TrapezoidMatrix.hh"
#include "slate_TriangularMatrix.hh"
#include "slate_util.hh"

#include "unit_test.hh"

using slate::ceildiv;
using slate::roundup;

//------------------------------------------------------------------------------
// global variables
extern int p, q;
extern int mpi_rank;
extern int mpi_size;
extern MPI_Comm mpi_comm;
extern int host_num, num_devices;

//==============================================================================
// fromLAPACK

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original LAPACK-style matrix that A is created from.
void verify_tile_lapack(
    slate::BaseMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ i*nb + j*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        test_assert(tile.uplo()   == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original LAPACK-style matrix that A is created from.
void verify_tile_lapack(
    slate::BaseTrapezoidMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ i*nb + j*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        if (i == j)
            test_assert(tile.uplo() == A.uplo());
        else
            test_assert(tile.uplo() == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//==============================================================================
// fromScaLAPACK

//------------------------------------------------------------------------------
/// Computes local dimensions for a block-cyclic distribution.
void get_cyclic_dimensions(
    int num_ranks, int rank, int n, int nb,
    int& ntiles, int& ntiles_local, int& n_local )
{
    assert(num_ranks > 0);

    // full tiles
    ntiles = int64_t(n / nb);
    ntiles_local = int64_t(ntiles / num_ranks);
    if (rank < ntiles % num_ranks) {
        ntiles_local += 1;
    }
    n_local = ntiles_local * nb;

    // partial tile
    if (n % nb > 0) {
        if (rank == ntiles % num_ranks) {
            ntiles_local += 1;
            n_local += n % nb;
        }
        ntiles += 1;
    }
}

//------------------------------------------------------------------------------
/// Computes local dimensions:
/// mtiles, mtiles_local, m_local,
/// ntiles, ntiles_local, n_local,
/// lda.
void get_2d_cyclic_dimensions(
    int m, int n, int nb,
    int& mtiles, int& mtiles_local, int& m_local,
    int& ntiles, int& ntiles_local, int& n_local,
    int& lda )
{
    int err;
    int mpi_rank;
    err = MPI_Comm_rank(mpi_comm, &mpi_rank);
    assert(err == 0);

    assert(p > 0 && q > 0);
    bool columnwise = true;
    int p_rank = (columnwise ? mpi_rank % p : mpi_rank / q);
    int q_rank = (columnwise ? mpi_rank / p : mpi_rank % q);

    get_cyclic_dimensions(p, p_rank, m, nb, mtiles, mtiles_local, m_local);
    get_cyclic_dimensions(q, q_rank, n, nb, ntiles, ntiles_local, n_local);
    lda = roundup(m_local, nb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original ScaLAPACK-style matrix that A is created from.
/// Similar to verify_tile_lapack, with different formula for Ad[ i, j ].
void verify_tile_scalapack(
    slate::BaseMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ int(i/p)*nb + int(j/q)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        test_assert(tile.uplo()   == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Ad is the original ScaLAPACK-style matrix that A is created from.
void verify_tile_scalapack(
    slate::BaseTrapezoidMatrix<double>& A, int i, int j, int nb,
    int m, int n, double* Ad, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   == &Ad[ int(i/p)*nb + int(j/q)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        if (i == j)
            test_assert(tile.uplo() == A.uplo());
        else
            test_assert(tile.uplo() == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == host_num);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//==============================================================================
// fromDevices

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Aarray is the original array of device matrices that A is created from.
/// Similar to verify_tile_lapack, but with Aarray[ dev ][ i, j ]
/// and device().
void verify_tile_device(
    slate::BaseMatrix<double>& A, int i, int j, int nb,
    int m, int n, double** Aarray, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int dev = (j / p) % num_devices;
    test_assert(A.tileDevice(i, j) == dev);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j, dev);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   ==
            &Aarray[ dev ][ int(i/p)*nb + int(int(j/q)/num_devices)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        test_assert(tile.uplo()   == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == dev);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j, dev);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
/// Verifies that tile A(i, j) has the expected properties.
/// Aarray is the original array of device matrices that A is created from.
/// Similar to verify_tile_lapack, but with Aarray[ dev ][ i, j ].
void verify_tile_device(
    slate::BaseTrapezoidMatrix<double>& A, int i, int j, int nb,
    int m, int n, double** Aarray, int lda )
{
    int rank = (i % p) + (j % q) * p;
    test_assert(A.tileRank(i, j) == rank);

    int dev = (j / p) % num_devices;
    test_assert(A.tileDevice(i, j) == dev);

    int jb = (j == A.nt()-1 ? n - j*nb : nb);
    int ib = (i == A.mt()-1 ? m - i*nb : nb);

    if (mpi_rank == rank) {
        test_assert(A.tileIsLocal(i, j));

        // check tile values
        slate::Tile<double> tile = A(i, j, dev);
        test_assert(tile.mb()     == ib);
        test_assert(tile.nb()     == jb);
        test_assert(tile.stride() == lda);
        test_assert(tile.data()   ==
            &Aarray[ dev ][ int(i/p)*nb + int(int(j/q)/num_devices)*nb*lda ]);
        test_assert(tile.op()     == blas::Op::NoTrans);
        if (i == j)
            test_assert(tile.uplo() == A.uplo());
        else
            test_assert(tile.uplo() == blas::Uplo::General);
        test_assert(tile.origin() == true);
        test_assert(tile.valid()  == true);
        test_assert(tile.device() == dev);
        test_assert(tile.size()   == size_t(ib * jb));
        test_assert(tile.bytes()  == sizeof(double) * ib * jb);

        // A(i, j) and A.at(i, j) should return identical tiles
        slate::Tile<double> tile2 = A.at(i, j, dev);
        test_assert(tile.mb()     == tile2.mb()    );
        test_assert(tile.nb()     == tile2.nb()    );
        test_assert(tile.stride() == tile2.stride());
        test_assert(tile.data()   == tile2.data()  );
        test_assert(tile.op()     == tile2.op()    );
        test_assert(tile.uplo()   == tile2.uplo()  );
        test_assert(tile.origin() == tile2.origin());
        test_assert(tile.valid()  == tile2.valid() );
        test_assert(tile.device() == tile2.device());
        test_assert(tile.size()   == tile2.size()  );
        test_assert(tile.bytes()  == tile2.bytes() );
    }
    else {
        test_assert(! A.tileIsLocal(i, j));
    }

    test_assert(A.tileMb(i) == ib);
    test_assert(A.tileNb(j) == jb);
}

//------------------------------------------------------------------------------
void init_process_grid(int mpi_size, int* p, int* q)
{
    // Make default p x q grid as square as possible.
    // Worst case is p=1, q=mpi_size.
    int p_ = 1, q_ = 1;
    for (p_ = int( sqrt( mpi_size )); p_ > 0; --p_) {
        q_ = int( mpi_size / p_ );
        if (p_*q_ == mpi_size)
            break;
    }
    *p = p_;
    *q = q_;
}

#endif        //  #ifndef UTIL_MATRIX_HH