#include <mpi.h>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cstring>
#include "shared_var.h"
#include "CSRdouble.hpp"

void printdense ( int m, int n, double *mat, char *filename ) {
    FILE *fd;
    fd = fopen ( filename,"w" );
    if ( fd==NULL )
        printf ( "error creating file" );
    int i,j;
    for ( i=0; i<m; ++i ) {
        //fprintf ( fd,"[\t" );
        for ( j=0; j<n; ++j ) {
            fprintf ( fd,"%12.8g\t",*(mat+i*n +j));
        }
        fprintf ( fd,"\n" );
    }
    fclose ( fd );
}

//////////////////////////////////////////////////////////////////////////////
//
// process_mem_usage(double &, double &) - takes two doubles by reference,
// attempts to read the system-dependent data for a process' virtual memory
// size and resident set size, and return the results in KB.
//
// On failure, returns 0.0, 0.0
void process_mem_usage(double& vm_usage, double& resident_set, double& cpu_user, double& cpu_sys)
{
    using std::ios_base;
    using std::ifstream;
    using std::string;

    vm_usage     = 0.0;
    resident_set = 0.0;
    cpu_user     = 0.0;
    cpu_sys      = 0.0;

    // 'file' stat seems to give the most reliable results
    //
    ifstream stat_stream("/proc/self/stat",ios_base::in);

    // dummy vars for leading entries in stat that we don't care about
    //
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string cutime, cstime, priority, nice;
    string O, itrealvalue, starttime;

    // the two fields we want
    //
    unsigned long vsize, utime, stime;
    long rss;

    stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                >> utime >> stime >> cutime >> cstime >> priority >> nice
                >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

    stat_stream.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    vm_usage     = vsize / 1024.0;
    resident_set = rss * page_size_kb;
    cpu_user     = utime / (float) sysconf(_SC_CLK_TCK);
    cpu_sys      = stime / (float) sysconf(_SC_CLK_TCK);
}

/**
 * @brief Computes some columns of sparse A with dense B and stores it into sparse C starting at a given column. A and C must have the same number of rows
 *
 * @param A Sparse matrix of which some columns are selected to be multiplied with B
 * @param B Dense matrix to be multiplied with the columns of B
 * @param lld_B local leading dimension of B (should always be larger than Cncols)
 * @param Acolstart First column of the submatrix of A to be multiplied with B (if Acolstart > A.ncols no multiplication is performed, but no error is returned)
 * @param Ancols Number of columns in the submatrix of A to be multiplied with B (if Acolstart + Ancols > A.ncols no multiplication is performed for columns > a.ncols, but no error is returned
 * @param Ccolstart First column of the submatrix of C where the result of A * B is stored.
 * @param Cncols Number of columns of the submatrix of C which are calculated in A * B.
 * @param C Sparse matrix containing the result of A * B (output)
 * @param trans Can be set to 1 of we want to store the transposed result B' * A'.
 * @return void
 **/
void mult_colsA_colsC ( CSRdouble& A, double *B, int lld_B, int Acolstart, int Ancols, int Ccolstart, int Cncols, //input
                        CSRdouble& C, bool trans ) {
    int i, j,row, col, C_nnz,C_ncols, *prows;
    double cij;

    /*assert(Cncols < lld_B);
    assert(Ccolstart+Cncols <= C.ncols);*/

    C_ncols=C.ncols;

    vector<int> Ccols;
    vector<double> Cdata;

    prows = new int[A.nrows + 1];
    C_nnz = 0;
    prows[0]=0;

    for ( row=0; row<A.nrows; ++row ) {
        for ( col=Ccolstart; col<Ccolstart+Cncols; ++col ) {
            cij=0;
            for ( i=A.pRows[row]; i<A.pRows[row+1]; ++i ) {
                j = A.pCols[i];
                if ( j>=Acolstart && j<Acolstart+Ancols )
                    cij += A.pData[i] * * ( B + col-Ccolstart + lld_B * ( j-Acolstart ) ) ;
            }
            if ( fabs ( cij ) >1e-10 ) {
                C_nnz++;
                Ccols.push_back ( col );
                Cdata.push_back ( cij );
            }
        }
        prows[row+1]=C_nnz;
    }
    int*    pcols = new int[C_nnz];
    memcpy ( pcols, &Ccols[0], C_nnz*sizeof ( int ) );
    Ccols.clear();
    double* pdata = new double[C_nnz];
    memcpy ( pdata, &Cdata[0], C_nnz*sizeof ( double ) );
    Cdata.clear();

    C.clear();
    C.make ( A.nrows,C_ncols,C_nnz,prows,pcols,pdata );

    if ( trans )
        C.transposeIt ( 1 );
}

void mult_colsA_colsC_denseC ( CSRdouble& A,double *B, int lld_B, int Acolstart, int Ancols, int Ccolstart, int Cncols,
                               double *C, int lld_C, bool sum, double alpha ) {
    int index, j,row, col;

    /*assert(Cncols < lld_B);
    assert(Ccolstart+Cncols <= C.ncols);*/

    for ( row=0; row<A.nrows; ++row ) {
        for ( col=Ccolstart; col<Ccolstart+Cncols; ++col ) {
            if (!sum)
                *(C + row + col * lld_C) = 0;
            for ( index=A.pRows[row]; index<A.pRows[row+1]; ++index ) {
                j = A.pCols[index];
                if ( j>=Acolstart && j<Acolstart+Ancols )
                    *(C + row + col * lld_C) += alpha * A.pData[index] * * ( B + lld_B * ( col-Ccolstart ) + j-Acolstart ) ;
            }
        }
    }
}

void mult_colsAtrans_colsC_denseC ( CSRdouble& A,double *B, int lld_B, int Acolstart, int Ancols, int Ccolstart, int Cncols, double *C, int lld_C, double alpha ) {
    int index, Atcol,row, col;

    /*assert(Cncols < lld_B);
    assert(Ccolstart+Cncols <= C.ncols);*/


    for ( col=Ccolstart; col<Ccolstart+Cncols; ++col ) {
        for ( Atcol=Acolstart; Atcol<Acolstart+Ancols; ++Atcol ) {
            for ( index=A.pRows[Atcol]; index<A.pRows[Atcol+1]; ++index ) {
                row = A.pCols[index];
                *(C + row + col * lld_C) += alpha * A.pData[index] * * ( B + lld_B * ( col-Ccolstart ) + Atcol-Acolstart ) ;
            }
        }
    }
}

void dense2CSR ( double *mat, int m, int n, CSRdouble& A ) {
    int i,j, nnz;
    double *pdata;
    int  *prows,*pcols;

    nnz=0;

    for ( i=0; i<m; ++i ) {
        for ( j=0; j<n; ++j ) {
            if ( fabs ( * ( mat+i*n+j ) ) >1e-10 ) {
                nnz++;
            }
        }
    }

    prows= ( int * ) calloc ( m+1,sizeof ( int ) );
    pcols= ( int * ) calloc ( nnz,sizeof ( int ) );
    pdata= ( double * ) calloc ( nnz,sizeof ( double ) );

    *prows=0;
    nnz=0;
    for ( i=0; i<m; ++i ) {
        for ( j=0; j<n; ++j ) {
            if ( fabs ( * ( mat+j*m+i ) ) >1e-10 ) { //If stored column-wise (BLAS), then moving through a row is going up by m (number of rows).
                * ( pdata+nnz ) =* ( mat+j*m+i );
                * ( pcols+nnz ) =j;
                nnz++;
            }
        }
        * ( prows+i+1 ) =nnz;
    }

    A.make ( m,n,nnz,prows,pcols,pdata );
}


