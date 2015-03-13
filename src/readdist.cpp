#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "shared_var.h"
#include <shared_var.h>
#include <limits.h>
#include "CSRdouble.hpp"
#define MPI_SUCCESS          0      /* Successful return code */
#define DLEN_ 		     9	    /* length of descriptor array*/
typedef int MPI_Comm;


extern "C" {
    /*int MPI_Ssend(void*, int, MPI_Datatype, int, int, MPI_Comm);
    int MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
    int MPI_Get_count(MPI_Status *, MPI_Datatype, int *);*/
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    int blacs_pnum_ ( int *ConTxt, int *prow, int *pcol );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );

    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_ ( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_ ( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_ ( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    double dnrm2_ ( int *n, double *x, int *incx );
    double ddot_ ( const int *n, const double *x, const int *incx, const double *y, const int *incy );
    void dgemm_ ( const char *transa, const char *transb, const int *m, const int *n, const int *k,
                  const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
                  const double *beta, double *c, const int *ldc );
}

int set_up_BDY_ori ( int * DESCD, double * Dmat, CSRdouble& BT_i, CSRdouble& B_j, int * DESCYTOT, double * ytot, double *respnrm, CSRdouble& Btsparse ) {

    // Read-in of matrices X, Z and T from file (filename[X,Z,T])
    // X and Z are read in entrely by every process
    // T is read in strip by strip (number of rows in each process is at maximum = blocksize)
    // D is constructed directly in a distributed way
    // B is first assembled sparse in root process and afterwards the necessary parts
    // for constructing the distributed Schur complement are sent to each process

    FILE *fT, *fY;
    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCZtY, *DESCXtY;
    double *Tblock, *temp, *Y, * ZtY, *XtY;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;
    int nYblocks, pYblocks, lld_Y, Ystart;
    timing secs;
    double totalTime, interTime;

    MPI_Status status;

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;

    Ztsparse.loadFromFile ( filenameZ );
    Ztsparse.transposeIt ( 1 );
    Xtsparse.loadFromFile ( filenameX );
    Xtsparse.transposeIt ( 1 );

    /*XtT_sparse.allocate ( m,k,0 );
    ZtT_sparse.allocate ( l,k,0 );*/

    pcol= * ( position+1 );

    // Matrix T is read in by strips of size (blocksize * *(dims+1), k)
    // Strips of T are read in row-wise and thus it is as if we store strips of T' (transpose) column-wise with dimensions (k, blocksize * *(dims+1))
    // However we must then also transpose the process grid to distribute T' correctly

    // number of strips in which we divide matrix T'
    nstrips= n % ( blocksize * * ( dims+1 ) ) ==0 ?  n / ( blocksize * * ( dims+1 ) ) : ( n / ( blocksize * * ( dims+1 ) ) ) +1;

    //the number of columns of T' included in each strip
    stripcols= blocksize * * ( dims+1 );

    //number of blocks necessary to store complete column of T'
    nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;

    //number of blocks necessary in this process to store complete column of T'
    pTblocks= ( nTblocks - *position ) % *dims == 0 ? ( nTblocks- *position ) / *dims : ( nTblocks- *position ) / *dims +1;
    pTblocks= pTblocks <1? 1:pTblocks;

    //local leading dimension of the strip of T' (different from process to process)
    lld_T=pTblocks*blocksize;
    lld_Y= * ( dims+1 ) * nstrips * blocksize;

    // Initialisation of descriptor of strips of matrix T'
    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    DESCY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCY==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    DESCXtY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCXtY==NULL ) {
        printf ( "unable to allocate memory for descriptor for XtY\n" );
        return -1;
    }
    DESCZtY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCZtY==NULL ) {
        printf ( "unable to allocate memory for descriptor for ZtY\n" );
        return -1;
    }

    // strip of T (k,stripcols) is distributed across ICTXT2D starting in process (0,0) in blocks of size (blocksize,blocksize)
    // the local leading dimension in this process is lld_T
    descinit_ ( DESCT, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix T returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCY, &lld_Y, &i_one, &lld_Y, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_Y, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCXtY, &m, &i_one, &m, &blocksize, &i_zero, &i_zero, &ICTXT2D, &m, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCZtY, &l, &i_one, &l, &blocksize, &i_zero, &i_zero, &ICTXT2D, &l, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }



    // Allocation of memory for the strip of T' in all processes

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }
    if ( iam==0 ) {
        Y= ( double* ) calloc ( lld_Y, sizeof ( double ) );
        if ( Y==NULL ) {
            printf ( "Error in allocating memory for Y in root process\n" );
            return -1;
        }
    }


    // Initialisation of matrix D (all diagonal elements of D equal to lambda)
    temp=Dmat;
    for ( i=0,rowcur=0,colcur=0; i<Dblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims ) {
            rowcur=0;
            temp += blocksize;
        }
        if ( colcur==* ( dims+1 ) ) {
            colcur=0;
            temp += blocksize*lld_D;
        }
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            for ( j=0; j<blocksize; ++j ) {
                * ( temp + j  * lld_D +j ) = 1/gamma_var;
            }
            if ( i==Dblocks-1 && Ddim % blocksize != 0 ) {
                for ( j=blocksize-1; j>= Ddim % blocksize; --j ) {
                    * ( temp + j * lld_D + j ) =0.0;
                }
            }
        }

    }
    if ( iam==0 ) {
        fY=fopen ( filenameY,"rb" );
        if ( fY==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        info=fread ( Y,sizeof ( double ),n,fY );
        if ( info<n ) {
            printf ( "Only %d values were read from %s",info,filenameY );
        }
        printf ( "Responses were read in correctly.\n" );
        info=fclose ( fY );
        if ( info!=0 ) {
            printf ( "Error in closing open streams" );
            return -1;
        }
        for ( i=n; i<lld_Y; ++i ) {
            * ( Y+i ) =0;
        }
    }

    fT=fopen ( filenameT,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    blacs_barrier_ ( &ICTXT2D,"A" );

    secs.tick ( totalTime );

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {

            if ( Tblock != NULL )
                free ( Tblock );
            Tblock=NULL;

            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
        }

        //Each process only reads in a part of the strip of T'
        //When k is not a multiple of blocksize, read-in of the last elements of the rows of T is tricky
        if ( ( nTblocks-1 ) % *dims == *position && k%blocksize !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( pcol * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),k%blocksize,fT );
            }
            //Normal read-in of the strips of T from a binary file (each time blocksize elements are read in)
        } else {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( pcol * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                info=fseek ( fT, ( long ) ( ( k - blocksize * ( ( pTblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D,"A" );

        if ( ni==0 ) {
            secs.tack ( totalTime );
            cout << "Read-in of Tblock: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }

        // End of read-in

        // Matrix D is the sum of the multiplications of all strips of T' by their transpose
        // Up unitl now, the entire matrix is stored, not only upper/lower triangular, which is possible since D is symmetric
        // Be aware, that you akways have to allocate memory for the enitre matrix, even when only dealing with the upper/lower triangular part

        pdgemm_ ( "N","T",&k,&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &i_one, &i_one, DESCD ); //T'T
        //pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &t_plus, &t_plus, DESCD );
        Ystart=ni * * ( dims+1 ) * blocksize + 1;
        /*if (iam==0)
        printf("Ystart= %d\n", Ystart);*/
        pdgemm_ ( "N","N",&k,&i_one,&stripcols,&d_one,Tblock,&i_one, &i_one, DESCT,Y,&Ystart,&i_one,DESCY,&d_one,ytot,&ml_plus,&i_one,DESCYTOT ); //T'y

        if ( ni==0 ) {
            secs.tack ( totalTime );
            cout << "dense multiplications of Tblock: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }

        // Matrix B consists of X'T and Z'T, since each process only has some parts of T at its disposal,
        // we need to make sure that the correct columns of Z and X are multiplied with the correct columns of T.
        for ( i=0; i<pTblocks; ++i ) {
            XtT_temp.ncols=k;

            //This function multiplies the correct columns of X' with the blocks of T at the disposal of the process
            // The result is also stored immediately at the correct positions of X'T. (see src/tools.cpp)
            mult_colsA_colsC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                               ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );
            /*mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                                   ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );*/
            if ( XtT_temp.nonzeros>0 ) {
                if ( XtT_sparse.nonzeros==0 )
                    XtT_sparse.make2 ( XtT_temp.nrows,XtT_temp.ncols,XtT_temp.nonzeros,XtT_temp.pRows,XtT_temp.pCols,XtT_temp.pData );
                else {
                    XtT_sparse.addBCSR ( XtT_temp );
                }
            }
            XtT_temp.clear();
        }

        if ( ni==0 ) {
            secs.tack ( totalTime );
            cout << "Creation of XtT: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }
        //Same as above for calculating Z'T

        for ( i=0; i<pTblocks; ++i ) {
            ZtT_temp.ncols=k;
            if ( ni==0 && i==1 ) {
                secs.tack ( totalTime );
                secs.tick ( totalTime );
            }
            mult_colsA_colsC ( Ztsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                               blocksize * ( *dims * i + *position ), blocksize, ZtT_temp, 0 );
            /*mult_colsA_colsC_denseC ( Ztsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                                   blocksize * ( *dims * i + *position ), blocksize, ZtT_dense,l, 1, 1 );
            dense2CSR(ZtT_dense,Ztsparse.nrows,Dblocks * blocksize,ZtT_temp);*/
            if ( ni==0 && i==1 ) {
                secs.tack ( totalTime );
                cout << "Multiplication Zt and Tblock: " << totalTime * 0.001 << " secs" << endl;
                secs.tick ( totalTime );
            }
            //free(ZtT_dense);
            if ( ZtT_temp.nonzeros>0 ) {
                if ( ZtT_sparse.nonzeros==0 )
                    ZtT_sparse.make2 ( ZtT_temp.nrows,ZtT_temp.ncols,ZtT_temp.nonzeros,ZtT_temp.pRows,ZtT_temp.pCols,ZtT_temp.pData );
                else
                    ZtT_sparse.addBCSR ( ZtT_temp );
            }
            if ( ni==0 && i==1 ) {
                secs.tack ( totalTime );
                cout << "Adding new piece of ZtT: " << totalTime * 0.001 << " secs" << endl;
                secs.tick ( totalTime );
            }
            ZtT_temp.clear();
        }
        if ( ni==0 ) {
            secs.tack ( totalTime );
            cout << "Creation of ZtT: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    secs.tack ( totalTime );
    if ( Tblock != NULL )
        free ( Tblock );
    Tblock=NULL;
    if ( iam==0 ) {
        cout << "Assignments with Tblock finished in: " << totalTime * 0.001 << " secs" << endl;

        secs.tick ( totalTime );

        *respnrm = dnrm2_ ( &n,Y,&i_one );
        *respnrm = *respnrm * *respnrm;
        XtY= ( double * ) calloc ( m,sizeof ( double ) );
        if ( XtY==NULL ) {
            printf ( "Unable to allocate memory for XtY in root process.\n" );
            return -1;
        }
        //printf("Xtsparse.nrows = %d \nm = %d\n",Xtsparse.nrows,m);
        mult_colsA_colsC_denseC ( Xtsparse, Y, n, 0, n, 0, 1, XtY, m, false, 1.0 );
        ZtY= ( double * ) calloc ( l,sizeof ( double ) );
        if ( ZtY==NULL ) {
            printf ( "Unable to allocate memory for ZtY in root process.\n" );
            return -1;
        }
        //printf("Ztsparse.nrows = %d \nl = %d\n",Ztsparse.nrows,l);
        mult_colsA_colsC_denseC ( Ztsparse, Y, n, 0, n, 0, 1, ZtY, l, false, 1.0 );
        if ( Y!= NULL )
            free ( Y );
        Y=NULL;

        secs.tack ( totalTime );
        cout << "Creation of XtY and ZtY: " << totalTime * 0.001 << " secs" << endl;
    }
    blacs_barrier_ ( &ICTXT2D,"A" );

    Xtsparse.clear();
    Ztsparse.clear();

    pdcopy_ ( &m,XtY,&i_one,&i_one,DESCXtY,&i_one, ytot,&i_one,&i_one,DESCYTOT,&i_one );
    pdcopy_ ( &l,ZtY,&i_one,&i_one,DESCZtY,&i_one, ytot,&m_plus,&i_one,DESCYTOT,&i_one );

    if ( iam==0 ) {
        if ( XtY != NULL )
            free ( XtY );
        XtY=NULL;
        if ( ZtY != NULL )
            free ( ZtY );
        ZtY = NULL;
    }

    info=fclose ( fT );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    //Each process only has calculated some parts of B
    //All parts are collected by the root process (iam==0), which assembles B
    //Each process then receives BT_i and B_j corresponding to the D_ij available to the process
    if ( iam!=0 ) {
        //Each process other than root sends its X' * T and Z' * T to the root process.
        MPI_Ssend ( & ( XtT_sparse.nonzeros ),1, MPI_INT,0,iam,MPI_COMM_WORLD );
        MPI_Ssend ( & ( XtT_sparse.pRows[0] ),XtT_sparse.nrows + 1, MPI_INT,0,iam+size,MPI_COMM_WORLD );
        MPI_Ssend ( & ( XtT_sparse.pCols[0] ),XtT_sparse.nonzeros, MPI_INT,0,iam+2*size,MPI_COMM_WORLD );
        MPI_Ssend ( & ( XtT_sparse.pData[0] ),XtT_sparse.nonzeros, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD );
        XtT_sparse.clear();
        MPI_Ssend ( & ( ZtT_sparse.nonzeros ),1, MPI_INT,0,iam,MPI_COMM_WORLD );
        MPI_Ssend ( & ( ZtT_sparse.pRows[0] ),ZtT_sparse.nrows + 1, MPI_INT,0,4*size + iam,MPI_COMM_WORLD );
        MPI_Ssend ( & ( ZtT_sparse.pCols[0] ),ZtT_sparse.nonzeros, MPI_INT,0,iam+ 5*size,MPI_COMM_WORLD );
        MPI_Ssend ( & ( ZtT_sparse.pData[0] ),ZtT_sparse.nonzeros, MPI_DOUBLE,0,iam+6*size,MPI_COMM_WORLD );
        ZtT_sparse.clear();

        // And eventually receives the necessary BT_i and B_j
        // Blocking sends are used, which is why the order of the receives is critical depending on the coordinates of the process
        int nonzeroes;
        if ( *position >= pcol ) {
            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            BT_i.clear();
            BT_i.allocate ( blocksize*Drows,m+l,nonzeroes );
            MPI_Recv ( & ( BT_i.pRows[0] ),blocksize*Drows + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            int count;
            MPI_Get_count ( &status,MPI_INT,&count );
            BT_i.nrows=count-1;
            MPI_Recv ( & ( BT_i.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( BT_i.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );

            MPI_Recv ( &nonzeroes,1, MPI_INT,0,iam+4*size,MPI_COMM_WORLD,&status );

            B_j.clear();
            B_j.allocate ( blocksize*Dcols,m+l,nonzeroes );

            MPI_Recv ( & ( B_j.pRows[0] ),blocksize*Dcols + 1, MPI_INT,0,iam + 5*size,MPI_COMM_WORLD,&status );
            MPI_Get_count ( &status,MPI_INT,&count );
            B_j.nrows=count-1;
            MPI_Recv ( & ( B_j.pCols[0] ),nonzeroes, MPI_INT,0,iam+6*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( B_j.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+7*size,MPI_COMM_WORLD,&status );

            //Actually BT_j is sent, so it still needs to be transposed
            B_j.transposeIt ( 1 );
        } else {
            MPI_Recv ( &nonzeroes,1, MPI_INT,0,iam+4*size,MPI_COMM_WORLD,&status );

            B_j.clear();

            B_j.allocate ( blocksize*Dcols,m+l,nonzeroes );

            MPI_Recv ( & ( B_j.pRows[0] ),blocksize*Dcols + 1, MPI_INT,0,iam + 5*size,MPI_COMM_WORLD,&status );
            int count;
            MPI_Get_count ( &status,MPI_INT,&count );
            B_j.nrows=count-1;

            MPI_Recv ( & ( B_j.pCols[0] ),nonzeroes, MPI_INT,0,iam+6*size,MPI_COMM_WORLD,&status );

            MPI_Recv ( & ( B_j.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+7*size,MPI_COMM_WORLD,&status );

            B_j.transposeIt ( 1 );

            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            BT_i.clear();
            BT_i.allocate ( blocksize*Drows,m+l,nonzeroes );
            MPI_Recv ( & ( BT_i.pRows[0] ),blocksize*Drows + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            MPI_Get_count ( &status,MPI_INT,&count );
            BT_i.nrows=count-1;
            MPI_Recv ( & ( BT_i.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( BT_i.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );
        }
    } else {
        secs.tick ( totalTime );
        for ( i=1; i<size; ++i ) {
            // The root process receives parts of X' * T and Z' * T sequentially from all processes and directly adds them together.
            int nonzeroes;
            MPI_Recv ( &nonzeroes,1,MPI_INT,i,i,MPI_COMM_WORLD,&status );
            if ( nonzeroes>0 ) {
                XtT_temp.clear();
                XtT_temp.allocate ( m,k,nonzeroes );
                MPI_Recv ( & ( XtT_temp.pRows[0] ),m + 1, MPI_INT,i,i+size,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( XtT_temp.pCols[0] ),nonzeroes, MPI_INT,i,i+2*size,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( XtT_temp.pData[0] ),nonzeroes, MPI_DOUBLE,i,i+3*size,MPI_COMM_WORLD,&status );

                XtT_sparse.addBCSR ( XtT_temp );
            }

            MPI_Recv ( &nonzeroes,1, MPI_INT,i,i,MPI_COMM_WORLD,&status );

            if ( nonzeroes>0 ) {
                ZtT_temp.clear();
                ZtT_temp.allocate ( l,k,nonzeroes );

                MPI_Recv ( & ( ZtT_temp.pRows[0] ),l + 1, MPI_INT,i,4*size + i,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( ZtT_temp.pCols[0] ),nonzeroes, MPI_INT,i,i+ 5*size,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( ZtT_temp.pData[0] ),nonzeroes, MPI_DOUBLE,i,i+6*size,MPI_COMM_WORLD,&status );

                ZtT_sparse.addBCSR ( ZtT_temp );
            }
        }
        secs.tack ( totalTime );
        cout << "Receiving XtT and ZtT: " << totalTime * 0.001 << " secs" << endl;
        XtT_temp.clear();
        ZtT_temp.clear();

        XtT_sparse.transposeIt ( 1 );
        ZtT_sparse.transposeIt ( 1 );

        // B' is created by concatening blocks X'T and Z'T
        create1x2BlockMatrix ( XtT_sparse, ZtT_sparse,Btsparse );
        //Btsparse.writeToFile("BT_sparse.csr");

        XtT_sparse.clear();
        ZtT_sparse.clear();

        secs.tick ( totalTime );
        // For each process row i BT_i is created which is also sent to processes in column i to become B_j.
        for ( int rowproc= *dims - 1; rowproc>= 0; --rowproc ) {
            BT_i.clear();
            BT_i.allocate ( 0, Btsparse.ncols,0 );
            int Drows_rowproc;
            if ( rowproc!=0 ) {
                Drows_rowproc= ( Dblocks - rowproc ) % *dims == 0 ? ( Dblocks- rowproc ) / *dims : ( Dblocks- rowproc ) / *dims +1;
                Drows_rowproc= Drows_rowproc<1? 1 : Drows_rowproc;
            } else
                Drows_rowproc=Drows;
            for ( i=0; i<Drows_rowproc; ++i ) {
                //Each process in row i can hold several blocks of contiguous rows of D for which we need the corresponding rows of B_T
                // Therefore we use the function extendrows to create BT_i (see src/tools.cpp)
                BT_i.extendrows ( Btsparse, ( i * *dims + rowproc ) * blocksize,blocksize );
            }
            for ( int colproc= ( rowproc==0 ? 1 : 0 ); colproc < * ( dims+1 ); ++colproc ) {
                int *curpos, rankproc;
                rankproc= blacs_pnum_ ( &ICTXT2D, &rowproc,&colproc );

                MPI_Ssend ( & ( BT_i.nonzeros ),1, MPI_INT,rankproc,rankproc,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pRows[0] ),BT_i.nrows + 1, MPI_INT,rankproc,rankproc+size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pCols[0] ),BT_i.nonzeros, MPI_INT,rankproc,rankproc+2*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pData[0] ),BT_i.nonzeros, MPI_DOUBLE,rankproc,rankproc+3*size,MPI_COMM_WORLD );

                //printf("BT_i's sent to processor %d\n",rankproc);

                rankproc= blacs_pnum_ ( &ICTXT2D, &colproc,&rowproc );
                MPI_Ssend ( & ( BT_i.nonzeros ),1, MPI_INT,rankproc,rankproc+4*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pRows[0] ),BT_i.nrows + 1, MPI_INT,rankproc,rankproc+5*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pCols[0] ),BT_i.nonzeros, MPI_INT,rankproc,rankproc+6*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pData[0] ),BT_i.nonzeros, MPI_DOUBLE,rankproc,rankproc+7*size,MPI_COMM_WORLD );

                //printf("B_j's sent to processor %d\n",rankproc);
            }
        }
        B_j.make2 ( BT_i.nrows,BT_i.ncols,BT_i.nonzeros,BT_i.pRows,BT_i.pCols,BT_i.pData );
        B_j.transposeIt ( 1 );
        secs.tack ( totalTime );
        cout << "Sending Bt_i and B_j: " << totalTime * 0.001 << " secs" << endl;
    }
    if ( DESCT!=NULL )
        free ( DESCT );
    DESCT=NULL;
    if ( DESCXtY!=NULL )
        free ( DESCXtY );
    DESCXtY=NULL;
    if ( DESCY!=NULL )
        free ( DESCY );
    DESCY=NULL;
    if ( DESCZtY!=NULL )
        free ( DESCZtY );
    DESCZtY=NULL;

    return 0;
}


int set_up_BDY ( int * DESCD, double * Dmat, int * DESCB, double * Bmat, int * DESCYTOT, double * ytot, double *respnrm ) {

    // Read-in of matrices X, Z and T from file (filename[X,Z,T])
    // X and Z are read in entrely by every process
    // T is read in strip by strip (number of rows in each process is at maximum = blocksize)
    // D is constructed directly in a distributed way
    // B is first assembled sparse in root process and afterwards the necessary parts
    // for constructing the distributed Schur complement are sent to each process

    FILE *fT, *fY;
    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCZtY, *DESCXtY;
    double *Tblock, *temp, *Y, * ZtY, *XtY;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;
    int nYblocks, pYblocks, lld_Y, Ystart;
    timing secs;
    double totalTime, interTime;

    MPI_Status status;

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;

    Ztsparse.loadFromFile ( filenameZ );
    Ztsparse.transposeIt ( 1 );
    Xtsparse.loadFromFile ( filenameX );
    Xtsparse.transposeIt ( 1 );

    /*XtT_sparse.allocate ( m,k,0 );
    ZtT_sparse.allocate ( l,k,0 );*/

    pcol= * ( position+1 );

    // Matrix T is read in by strips of size (blocksize * *(dims+1), k)
    // Strips of T are read in row-wise and thus it is as if we store strips of T' (transpose) column-wise with dimensions (k, blocksize * *(dims+1))
    // However we must then also transpose the process grid to distribute T' correctly

    // number of strips in which we divide matrix T
    nstrips= n % ( blocksize * * ( dims ) ) ==0 ?  n / ( blocksize * * ( dims ) ) : ( n / ( blocksize * * ( dims ) ) ) +1;

    //the number of columns of T included in each strip
    stripcols= blocksize * * ( dims );

    //number of blocks necessary to store complete column of T
    nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;

    //number of blocks necessary in this process to store complete column of T
    pTblocks= ( nTblocks - pcol ) % * ( dims+1 ) == 0 ? ( nTblocks- pcol ) / * ( dims+1 ) : ( nTblocks- pcol ) / * ( dims+1 ) +1;
    pTblocks= pTblocks <1? 1:pTblocks;

    //local leading dimension of the strip of T (different from process to process)
    lld_T=blocksize;
    lld_Y= * ( dims+1 ) * nstrips * blocksize;

    // Initialisation of descriptor of strips of matrix T'
    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    DESCY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCY==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    DESCXtY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCXtY==NULL ) {
        printf ( "unable to allocate memory for descriptor for XtY\n" );
        return -1;
    }
    DESCZtY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCZtY==NULL ) {
        printf ( "unable to allocate memory for descriptor for ZtY\n" );
        return -1;
    }
    /*if(*(position+1)==0)
        cout << "Descriptors declared" << endl;*/

    // strip of T (k,stripcols) is distributed across ICTXT2D starting in process (0,0) in blocks of size (blocksize,blocksize)
    // the local leading dimension in this process is lld_T
    descinit_ ( DESCT, &stripcols, &k, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix T returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCY, &lld_Y, &i_one, &lld_Y, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_Y, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCXtY, &m, &i_one, &m, &blocksize, &i_zero, &i_zero, &ICTXT2D, &m, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCZtY, &l, &i_one, &l, &blocksize, &i_zero, &i_zero, &ICTXT2D, &l, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }

    /*if(*(position+1)==0)
        cout << "Descriptors initialised" << endl;*/


    // Allocation of memory for the strip of T' in all processes

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }
    if ( * ( position + 1 ) == 0 ) {
        Y= ( double* ) calloc ( lld_Y, sizeof ( double ) );
        if ( Y==NULL ) {
            printf ( "Error in allocating memory for Y in root process\n" );
            return -1;
        }
    }

    /*if(*(position+1)==0)
        cout << "Tblock and Y initialised" << endl;*/

    // Initialisation of matrix D (all diagonal elements of D equal to lambda)
    temp=Dmat;
    for ( i=0,colcur=0; i<Dblocks; ++i, ++colcur, temp += blocksize ) {

        if ( colcur==* ( dims+1 ) ) {
            colcur=0;
            temp += blocksize*lld_D;
        }
        if ( * ( position+1 ) == colcur ) {
            if ( i==Dblocks-1 && Ddim % blocksize != 0 ) {
                for ( j=0; j< Ddim % blocksize; ++j ) {
                    * ( temp + j * lld_D + j ) =1/gamma_var;
                }
            }
            else {
                for ( j=0; j<blocksize; ++j ) {
                    * ( temp + j  * lld_D +j ) = 1/gamma_var;
                }
            }
        }

    }
    /*if(*(position+1)==0)
        cout << "Diagonal of Dmat initialised" << endl;*/

    if ( * ( position + 1 ) ==0 ) {
        fY=fopen ( filenameY,"rb" );
        if ( fY==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        info=fread ( Y,sizeof ( double ),n,fY );
        if ( info<n ) {
            printf ( "Only %d values were read from %s",info,filenameY );
        }
        printf ( "Responses were read in correctly.\n" );
        info=fclose ( fY );
        if ( info!=0 ) {
            printf ( "Error in closing open streams" );
            return -1;
        }
        for ( i=n; i<lld_Y; ++i ) {
            * ( Y+i ) =0;
        }
    }

    fT=fopen ( filenameT,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    blacs_barrier_ ( &ICTXT2D,"A" );

    secs.tick ( totalTime );

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {

            if ( Tblock != NULL )
                free ( Tblock );
            Tblock=NULL;

            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
        }

        //Each process only reads in a part of the strip of T'
        //When k is not a multiple of blocksize, read-in of the last elements of the rows of T is tricky
        if ( ( nTblocks-1 ) % * ( dims +1 ) == pcol && k%blocksize !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( *position * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * pcol * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    for ( int el=0; el<blocksize; ++el ) {
                        fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                    }
                    info=fseek ( fT, ( long ) ( ( ( * ( dims+1 ) ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                for ( int el=0; el< k % blocksize; ++el ) {
                    fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                }
            }
            //Normal read-in of the strips of T from a binary file (each time blocksize elements are read in)
        } else {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( *position * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * pcol * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    for ( int el=0; el< blocksize; ++el ) {
                        fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                    }
                    info=fseek ( fT, ( long ) ( ( * ( dims + 1 )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                for ( int el=0; el< blocksize; ++el ) {
                    fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                }
                info=fseek ( fT, ( long ) ( ( k - blocksize * ( ( pTblocks-1 ) * * ( dims+1 ) + pcol +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D,"A" );

        if ( ni==0 && * ( position+1 ) ==0 ) {
            secs.tack ( totalTime );
            cout << "Read-in of Tblock: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }
        /*char *Tfile;
            Tfile=(char *) calloc(100,sizeof(char));
            *Tfile='\0';
            sprintf(Tfile,"Tmat_(%d,%d)_%d.txt",*position,pcol,ni);
            printdense(blocksize,pTblocks * blocksize,Tblock,Tfile);*/

        // End of read-in

        // Matrix D is the sum of the multiplications of all strips of T' by their transpose
        // Up until now, the entire matrix is stored, not only upper/lower triangular, which is possible since D is symmetric
        // Be aware, that you always have to allocate memory for the enitre matrix, even when only dealing with the upper/lower triangular part

        pdgemm_ ( "T","N",&k,&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &i_one, &i_one, DESCD ); //T'T
        //pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &t_plus, &t_plus, DESCD );
        Ystart=ni * * ( dims ) * blocksize + 1;
        /*if (*(position +1)==0)
            printf("Ystart= %d\n", Ystart);*/
        pdgemm_ ( "T","N",&k,&i_one,&stripcols,&d_one,Tblock,&i_one, &i_one, DESCT,Y,&Ystart,&i_one,DESCY,&d_one,ytot,&ml_plus,&i_one,DESCYTOT ); //T'y

        if ( ni==0 && * ( position+1 ) ==0 ) {
            secs.tack ( totalTime );
            cout << "dense multiplications of Tblock: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }

        // Matrix B consists of X'T and Z'T, since each process only has some parts of T at its disposal,
        // we need to make sure that the correct columns of Z and X are multiplied with the correct columns of T.
        for ( i=0; i<pTblocks; ++i ) {

            //This function multiplies the correct columns of X' with the blocks of T at the disposal of the process
            // The result is also stored immediately at the correct positions of X'T. (see src/tools.cpp)
            mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize*blocksize, lld_T, ( ni * *dims + *position ) *blocksize, blocksize,
                                      i*blocksize, blocksize, Bmat, Adim, true, 1 );
            /*mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                                   ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );*/
        }

        if ( ni==0 && * ( position+1 ) ==0 ) {
            secs.tack ( totalTime );
            cout << "Creation of XtT: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }
        //Same as above for calculating Z'T

        for ( i=0; i<pTblocks; ++i ) {
            for ( i=0; i<pTblocks; ++i ) {

                //This function multiplies the correct columns of X' with the blocks of T at the disposal of the process
                // The result is also stored immediately at the correct positions of X'T. (see src/tools.cpp)
                mult_colsA_colsC_denseC ( Ztsparse, Tblock+i*blocksize*blocksize, lld_T, ( ni * *dims + *position ) *blocksize, blocksize,
                                          i*blocksize, blocksize, Bmat+Xtsparse.nrows, Adim, true, 1 );
                /*mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                                       ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );*/
            }
            if ( ni==0 && i==1 && * ( position+1 ) ==0 ) {
                secs.tack ( totalTime );
                cout << "Multiplication Zt and Tblock: " << totalTime * 0.001 << " secs" << endl;
                secs.tick ( totalTime );
            }
            //free(ZtT_dense);

        }
        if ( ni==0 && * ( position+1 ) ==0 ) {
            secs.tack ( totalTime );
            cout << "Creation of ZtT: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    /*char *Bfile;
    Bfile=(char *) calloc(100,sizeof(char));
    *Bfile='\0';
    sprintf(Bfile,"Bmat_(%d,%d).txt",*position,pcol);
    printdense(Dcols * blocksize,Adim,Bmat,Bfile);*/

    secs.tack ( totalTime );
    if ( Tblock != NULL )
        free ( Tblock );
    Tblock=NULL;
    if ( * ( position +1 ) == 0 ) {
        cout << "Assignments with Tblock finished in: " << totalTime * 0.001 << " secs" << endl;

        secs.tick ( totalTime );

        *respnrm = dnrm2_ ( &n,Y,&i_one );
        *respnrm = *respnrm * *respnrm;
        XtY= ( double * ) calloc ( m,sizeof ( double ) );
        if ( XtY==NULL ) {
            printf ( "Unable to allocate memory for XtY in root process.\n" );
            return -1;
        }
        //printf("Xtsparse.nrows = %d \nm = %d\n",Xtsparse.nrows,m);
        mult_colsA_colsC_denseC ( Xtsparse, Y, n, 0, n, 0, 1, XtY, m, false, 1.0 );
        ZtY= ( double * ) calloc ( l,sizeof ( double ) );
        if ( ZtY==NULL ) {
            printf ( "Unable to allocate memory for ZtY in root process.\n" );
            return -1;
        }
        //printf("Ztsparse.nrows = %d \nl = %d\n",Ztsparse.nrows,l);
        mult_colsA_colsC_denseC ( Ztsparse, Y, n, 0, n, 0, 1, ZtY, l, false, 1.0 );
        if ( Y!= NULL )
            free ( Y );
        Y=NULL;

        secs.tack ( totalTime );
        cout << "Creation of XtY and ZtY: " << totalTime * 0.001 << " secs" << endl;
    }
    blacs_barrier_ ( &ICTXT2D,"A" );

    //cout << "process " << iam << " got here" << endl;

    Xtsparse.clear();
    Ztsparse.clear();

    pdcopy_ ( &m,XtY,&i_one,&i_one,DESCXtY,&i_one, ytot,&i_one,&i_one,DESCYTOT,&i_one );
    pdcopy_ ( &l,ZtY,&i_one,&i_one,DESCZtY,&i_one, ytot,&m_plus,&i_one,DESCYTOT,&i_one );

    if ( * ( position + 1 ) ==0 ) {
        if ( XtY != NULL )
            free ( XtY );
        XtY=NULL;
        if ( ZtY != NULL )
            free ( ZtY );
        ZtY = NULL;
    }

    info=fclose ( fT );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    if ( DESCT!=NULL )
        free ( DESCT );
    DESCT=NULL;
    if ( DESCXtY!=NULL )
        free ( DESCXtY );
    DESCXtY=NULL;
    if ( DESCY!=NULL )
        free ( DESCY );
    DESCY=NULL;
    if ( DESCZtY!=NULL )
        free ( DESCZtY );
    DESCZtY=NULL;

    return 0;
}

int set_up_D ( int * DESCD, double * Dmat ) {

    // Read-in of matrices X, Z and T from file (filename[X,Z,T])
    // X and Z are read in entrely by every process
    // T is read in strip by strip (number of rows in each process is at maximum = blocksize)
    // D is constructed directly in a distributed way
    // B is first assembled sparse in root process and afterwards the necessary parts
    // for constructing the distributed Schur complement are sent to each process

    FILE *fT;
    int ni, i,j, info;
    int *DESCT;
    double *Tblock, *temp;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;
    timing secs;
    double totalTime, interTime;

    MPI_Status status;

    /*XtT_sparse.allocate ( m,k,0 );
    ZtT_sparse.allocate ( l,k,0 );*/

    pcol= * ( position+1 );

    // Matrix T is read in by strips of size (blocksize * *(dims+1), k)
    // Strips of T are read in row-wise and thus it is as if we store strips of T' (transpose) column-wise with dimensions (k, blocksize * *(dims+1))
    // However we must then also transpose the process grid to distribute T' correctly

    // number of strips in which we divide matrix T'
    nstrips= n % ( blocksize * * ( dims ) ) ==0 ?  n / ( blocksize * * ( dims ) ) : ( n / ( blocksize * * ( dims ) ) ) +1;

    //the number of columns of T' included in each strip
    stripcols= blocksize * * ( dims );

    //number of blocks necessary to store complete column of T'
    nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;

    //number of blocks necessary in this process to store complete column of T'
    pTblocks= ( nTblocks - pcol ) % *(dims+1) == 0 ? ( nTblocks- pcol ) / *(dims+1) : ( nTblocks- pcol ) / *(dims+1) +1;
    pTblocks= pTblocks <1? 1:pTblocks;

    //local leading dimension of the strip of T' (different from process to process)
    lld_T=blocksize;

    // Initialisation of descriptor of strips of matrix T'
    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }


    // strip of T (k,stripcols) is distributed across ICTXT2D starting in process (0,0) in blocks of size (blocksize,blocksize)
    // the local leading dimension in this process is lld_T
    descinit_ ( DESCT, &stripcols, &k, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix T returns info: %d\n",info );
        return info;
    }

    // Allocation of memory for the strip of T' in all processes

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }

    // Initialisation of matrix D (all diagonal elements of D equal to lambda)
    temp=Dmat;
    for ( i=0,colcur=0; i<Dblocks; ++i, ++colcur, temp += blocksize ) {
        if ( colcur==* ( dims+1 ) ) {
            colcur=0;
            temp += blocksize*lld_D;
        }
        if ( * ( position+1 ) == colcur ) {
            if ( i==Dblocks-1 && Ddim % blocksize != 0 ) {
                for ( j=0; j< Ddim % blocksize; ++j ) {
                    * ( temp + j * lld_D + j ) =1/gamma_var;
                }
            }
            else {
                for ( j=0; j<blocksize; ++j ) {
                    * ( temp + j  * lld_D +j ) = 1/gamma_var;
                }
            }
        }

    }

    fT=fopen ( filenameT,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    blacs_barrier_ ( &ICTXT2D,"A" );

    secs.tick ( totalTime );

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {

            if ( Tblock != NULL )
                free ( Tblock );
            Tblock=NULL;

            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
        }

        //Each process only reads in a part of the strip of T'
        //When k is not a multiple of blocksize, read-in of the last elements of the rows of T is tricky
        if ( ( nTblocks-1 ) % *(dims+1) == pcol && k%blocksize !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( *position * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * pcol * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    for ( int el=0; el<blocksize; ++el ) {
                        fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                    }
                    info=fseek ( fT, ( long ) ( ( ( *(dims+1) ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                for ( int el=0; el< k % blocksize; ++el ) {
                    fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                }
            }
            //Normal read-in of the strips of T from a binary file (each time blocksize elements are read in)
        } else {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( *position * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * pcol * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    for ( int el=0; el< blocksize; ++el ) {
                        fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                    }
                    info=fseek ( fT, ( long ) ( ( * ( dims +1)-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                for ( int el=0; el< blocksize; ++el ) {
                    fread ( Tblock + ( j*blocksize + el ) *blocksize + i,sizeof ( double ),1,fT );
                }
                info=fseek ( fT, ( long ) ( ( k - blocksize * ( ( pTblocks-1 ) * *(dims+1) + pcol +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D,"A" );

        // End of read-in

        // Matrix D is the sum of the multiplications of all strips of T' by their transpose
        // Up unitl now, the entire matrix is stored, not only upper/lower triangular, which is possible since D is symmetric
        // Be aware, that you akways have to allocate memory for the enitre matrix, even when only dealing with the upper/lower triangular part

        pdgemm_ ( "T","N",&k,&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &i_one, &i_one, DESCD ); //T'T
        //pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &t_plus, &t_plus, DESCD );

        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    secs.tack ( totalTime );
    if ( Tblock != NULL )
        free ( Tblock );
    Tblock=NULL;
    blacs_barrier_ ( &ICTXT2D,"A" );

    info=fclose ( fT );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    if ( DESCT!=NULL )
        free ( DESCT );
    DESCT=NULL;
    return 0;
}


int set_up_C ( int * DESCC, double * Cmat, int * DESCB, double * Bmat,int * DESCS, double * Smat, int * DESCYTOT, double * ytot, double *respnrm ) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    FILE *fZ, *fX;
    int ni, i,j, info;
    int *DESCZ, *DESCY, *DESCX;
    double *Zblock, *Xblock, *yblock, *nrmblock, *temp;
    int nZblocks, nXblocks, nstrips, pZblocks, pXblocks, stripcols, lld_Z, lld_X, pcol, colcur,rowcur;

    // Allocation of descriptors for different matrices

    DESCZ= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCZ==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    DESCY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCY==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    DESCX= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCX==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    //Matrices Z & X are read in from the binary files, however, since PBLAS only accepts column-major matrices, we store the transposed Z and X matrices (column major)

    pcol= * ( position+1 );
    nstrips= n % ( blocksize * * ( dims+1 ) ) ==0 ?  n / ( blocksize * * ( dims+1 ) ) : ( n / ( blocksize * * ( dims+1 ) ) ) +1; 	// number of strips in which we divide matrix Z' and X'
    stripcols= blocksize * * ( dims+1 ); 												//the number of columns taken into the strip of Z' and X'
    nZblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;										//number of blocks necessary to store complete column of Z'
    pZblocks= ( nZblocks - *position ) % *dims == 0 ? ( nZblocks- *position ) / *dims : ( nZblocks- *position ) / *dims +1;		//number of blocks necessary per processor
    pZblocks= pZblocks <1? 1:pZblocks;
    lld_Z=pZblocks*blocksize;													//local leading dimension of the strip of Z (different from processor to processor)
    nXblocks= m%blocksize==0 ? m/blocksize : m/blocksize +1;									//number of blocks necessary to store complete column of X'
    pXblocks= ( nXblocks - *position ) % *dims == 0 ? ( nXblocks- *position ) / *dims : ( nXblocks- *position ) / *dims +1;		//number of blocks necessary per processor
    pXblocks= pXblocks <1? 1:pXblocks;
    lld_X=pXblocks*blocksize;													//local leading dimension of the strip of Z (different from processor to processor)


    // Initialisation of descriptors of different matrices

    descinit_ ( DESCZ, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_Z, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    descinit_ ( DESCY, &i_one, &stripcols, &i_one, &blocksize, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCX, &m, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_X, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix X returns info: %d\n",info );
        return info;
    }

    // Allocation of memory for the different matrices in all processes

    Zblock= ( double* ) calloc ( pZblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Zblock==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }

    yblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
    if ( yblock==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    Xblock= ( double* ) calloc ( pXblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Xblock==NULL ) {
        printf ( "Error in allocating memory for a strip of X in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    nrmblock = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nrmblock==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    // Initialisation of matrix C (all diagonal elements in lower right block (m x m) equal to lambda)
    temp=Cmat;
    for ( i=0,rowcur=0,colcur=0; i<Dblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims ) {
            rowcur=0;
            temp += blocksize;
        }
        if ( colcur==* ( dims+1 ) ) {
            colcur=0;
            temp += blocksize*lld_D;
        }
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            for ( j=0; j<blocksize; ++j ) {
                * ( temp + j  * lld_D +j ) =1/gamma_var;
            }
            if ( i==Dblocks-1 && Ddim % blocksize != 0 ) {
                for ( j=blocksize-1; j>= Ddim % blocksize; --j ) {
                    * ( temp + j * lld_D + j ) =0.0;
                }
            }
        }

    }
    temp=Cmat;
    for ( i=0,rowcur=0,colcur=0; i<nXblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims ) {
            rowcur=0;
            temp += blocksize;
        }
        if ( colcur==* ( dims+1 ) ) {
            colcur=0;
            temp += blocksize*lld_D;
        }
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            if ( i<nXblocks-1 ) {
                for ( j=0; j<blocksize; ++j ) {
                    * ( temp + j * lld_D + j ) =0.0;
                }
            } else {
                for ( j=0; j<= ( m-1 ) %blocksize; ++j ) {
                    * ( temp + j * lld_D + j ) =0.0;
                }
            }
        }
    }


    fZ=fopen ( filenameT,"rb" );
    if ( fZ==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    fX=fopen ( filenameX,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }
    *respnrm=0.0;
    *nrmblock=0.0;

    // Set up of matrix C per strip of Z and X (every strip contains $blocksize complete rows)

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {

            free ( Zblock );
            free ( yblock );
            free ( Xblock );
            Zblock= ( double* ) calloc ( pZblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Zblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
            yblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
            if ( yblock==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
            }
            Xblock= ( double* ) calloc ( pXblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Xblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }

            /*Other possibility:
            Zcols=n%blocksize==0 ? blocksize * * ( dims+1 ) : blocksize * ( * ( dims+1 )-1 ) + ( n%blocksize );
            descinit_ ( DESCZ, &m, &Zcols, &blocksize, &blocksize, &RSRC, &CSRC, &ICTXT, &lld_Z, &INFO );
            if ( INFO!=0 )
            {
               printf ( "Descriptor of matrix Z returns info: %d\n",INFO );
               return INFO;
            }
            descinit_ ( DESCY, &i_one, &Zcols, &i_one, &blocksize, &RSRC, &CSRC, &ICTXT, &i_one, &INFO );
            if ( INFO!=0 )
            {
               printf ( "Descriptor of matrix Y returns info: %d\n",INFO );
               return INFO;
            }
            */
        }

        //Creation of matrix Z in every process

        if ( ( nZblocks-1 ) % *dims == *position && k%blocksize !=0 ) { //last block of row that needs to be read in will be treated seperately
            if ( ni==0 ) {
                info=fseek ( fZ, ( long ) ( pcol * blocksize * ( k+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fZ, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k+1 ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fZ, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                if ( *position==0 )
                    fread ( yblock + i,sizeof ( double ),1,fZ );
                else
                    info=fseek ( fZ,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pZblocks-1; ++j ) {
                    fread ( Zblock + i*pZblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fZ );
                    info=fseek ( fZ, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Zblock + i*pZblocks*blocksize + j*blocksize,sizeof ( double ),k%blocksize,fZ );
            }
        } else {									//Normal read-in of the matrix from a binary file
            if ( ni==0 ) {
                info=fseek ( fZ, ( long ) ( pcol * blocksize * ( k+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fZ, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k+1 ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fZ, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                if ( *position==0 )
                    fread ( yblock + i,sizeof ( double ),1,fZ );
                else
                    info=fseek ( fZ,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pZblocks-1; ++j ) {
                    fread ( Zblock + i*pZblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fZ );
                    info=fseek ( fZ, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Zblock + i*pZblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fZ );
                info=fseek ( fZ, ( long ) ( ( k - blocksize * ( ( pZblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }
        /*char *Zfile;
        Zfile=(char *) calloc(100,sizeof(char));
        *Zfile='\0';
        sprintf(Zfile,"Zmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(blocksize,blocksize * pZblocks, Zblock,Zfile);*/

        //Creation of matrix X in every process

        if ( ( nXblocks-1 ) % *dims == *position && m%blocksize !=0 ) {									//last block of row that needs to be read in will be treated seperately
            if ( ni==0 ) {
                info=fseek ( fX, ( long ) ( pcol * blocksize *  m * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fX, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * m * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fX, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pXblocks-1; ++j ) {
                    fread ( Xblock + i*pXblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fX );
                    info=fseek ( fX, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Xblock + i*pXblocks*blocksize + j*blocksize,sizeof ( double ),m%blocksize,fX );
            }
        } else {													//Normal read-in of the matrix from a binary file
            if ( ni==0 ) {
                info=fseek ( fX, ( long ) ( pcol * blocksize *  m * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fX, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * m * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fX, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pXblocks-1; ++j ) {
                    fread ( Xblock + i*pXblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fX );
                    info=fseek ( fX, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Xblock + i*pXblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fX );
                info=fseek ( fX, ( long ) ( ( m - blocksize * ( ( pXblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }
        /*char *Xfile;
        Xfile=(char *) calloc(100,sizeof(char));
        *Xfile='\0';
        sprintf(Xfile,"Xmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(blocksize,blocksize * pXblocks, Xblock,Xfile);*/
        blacs_barrier_ ( &ICTXT2D,"A" );

        // End of read-in

        // Creation of symmetric matrix C in every process per block of Z, X and y
        // Only the upper triangular part is stored as a full matrix distributed over all processes (column major)

        pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Zblock,&i_one, &i_one,DESCZ, &d_one, Cmat, &i_one, &i_one, DESCC ); //Z'Z

        pdgemm_ ( "N","T",&k,&i_one,&stripcols,&d_one,Zblock,&i_one, &i_one, DESCZ,yblock,&i_one,&i_one,DESCY,&d_one,ytot,&m_plus,&i_one,DESCYTOT ); //Z'y

        pdsyrk_ ( "U","N",&m,&stripcols,&d_one, Xblock,&i_one, &i_one,DESCX, &d_one, Smat, &i_one, &i_one, DESCS ); //X'X

        pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,yblock,&i_one,&i_one,DESCY,&d_one,ytot,&i_one,&i_one,DESCYTOT ); //X'y

        pdgemm_ ( "N","T",&m,&k,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,Zblock,&i_one,&i_one,DESCZ,&d_one,Bmat,&i_one,&m_plus,DESCB ); //X'Z

        //y'y is square of 2-norm (used for calculation of sigma)

        pdnrm2_ ( &stripcols,nrmblock,yblock,&i_one,&i_one,DESCY,&i_one );
        *respnrm += *nrmblock * *nrmblock;

        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    info=fclose ( fZ );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    free ( DESCX );
    free ( DESCY );
    free ( DESCZ );
    free ( Zblock );
    free ( Xblock );
    free ( yblock );
    free ( nrmblock );
    return 0;
}

/**
 * @brief This function can be used to update the parameter lambda in coefficient matrix C
 *
 * When a copy of coefficient matrix C is stored in memory, this can easily be updated every iteration by calling this function.
 * The parameter lambda is updated by update in the lower right m x m block of C
 *
 * @param DESCC Pointer to descriptor of coefficient matrix C
 * @param Cmat Pointer to coeffisicent matrix C stored distributed in every proces
 * @param update value with which parameter lambda should be updated
 * @return int (info)
 **/
int update_C ( int * DESCC, double * Cmat, double update ) {

    int i,j, rowcur,colcur,nXblocks;

    //nXblocks= m%blocksize==0 ? m/blocksize : m/blocksize +1;

    for ( i=0,rowcur=0,colcur=0; i<Dblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims )
            rowcur=0;
        if ( colcur==* ( dims+1 ) )
            colcur=0;
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            if ( i< ( Dblocks -1 ) ) {
                for ( j=0; j<blocksize; ++j ) {
                    * ( Cmat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ) +=update;
                }
            } else {
                for ( j=0; j< Ddim % blocksize; ++j ) {
                    * ( Cmat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ) +=update;
                }
            }
        }
    }
    /*
        for ( i=0,rowcur=0,colcur=0; i<nXblocks; ++i, ++colcur, ++rowcur ) {
            if ( rowcur==*dims )
                rowcur=0;
            if ( colcur==* ( dims+1 ) )
                colcur=0;
            if ( *position==rowcur && * ( position+1 ) == colcur ) {
                if ( i<nXblocks-1 ) {
                    for ( j=0; j<blocksize; ++j ) {
                        * ( Cmat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ) -=update;
                    }
                } else {
                    for ( j=0; j< m%blocksize; ++j ) {
                        * ( Cmat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ) -=update;
                    }
                }
            }
        }*/
    return 0;
}


int set_up_AI_ori ( double * AImat, int * DESCAI,int * DESCSOL, double * solution, int * DESCD, double * Dmat, CSRdouble &Asparse, CSRdouble &Btsparse, double sigma ) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    FILE* fT, * fY;
    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCTD, *DESCZU, *DESCQRHS, *DESCQSOL, *DESCQDENSE;
    double *Tblock, *yblock, *Tdblock, *QRHS, *Qsol,*nrmblock, sigma_rec, phi_rec, gamma_rec, *Zu, *Qdense;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur, lld_Y;

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;


    // Initialisation of descriptors of different matrices

    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    DESCY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCY==NULL ) {
        printf ( "unable to allocate memory for descriptor for Y\n" );
        return -1;
    }
    DESCTD= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCTD==NULL ) {
        printf ( "unable to allocate memory for descriptor for Td\n" );
        return -1;
    }
    DESCZU= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCTD==NULL ) {
        printf ( "unable to allocate memory for descriptor for Zu\n" );
        return -1;
    }
    DESCQRHS= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCQRHS==NULL ) {
        printf ( "unable to allocate memory for descriptor for QRHS\n" );
        return -1;
    }
    DESCQSOL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCQSOL==NULL ) {
        printf ( "unable to allocate memory for descriptor for QSOL\n" );
        return -1;
    }
    DESCQDENSE= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCQDENSE==NULL ) {
        printf ( "unable to allocate memory for descriptor for ZtY\n" );
        return -1;
    }

    //Matrices Z & X are read in from the binary files, however, since PBLAS only accepts column-major matrices, we store the transposed Z and X matrices (column major)

    pcol= * ( position+1 );
    nstrips= n % ( blocksize * * ( dims+1 ) ) ==0 ?  n / ( blocksize * * ( dims+1 ) ) : ( n / ( blocksize * * ( dims+1 ) ) ) +1; 	// number of strips in which we divide matrix Z' and X'
    stripcols= blocksize * * ( dims+1 ); 												//the number of columns taken into the strip of Z' and X'
    nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;										//number of blocks necessary to store complete column of Z'
    pTblocks= ( nTblocks - *position ) % *dims == 0 ? ( nTblocks- *position ) / *dims : ( nTblocks- *position ) / *dims +1;		//number of blocks necessary per processor
    pTblocks= pTblocks <1? 1:pTblocks;
    lld_T=pTblocks*blocksize;													//local leading dimension of the strip of Z (different from processor to processor)														//local leading dimension of the strip of Z (different from processor to processor)

    lld_Y= * ( dims+1 ) * nstrips * blocksize;
    sigma_rec=1/sigma;
    phi_rec=1/phi;
    gamma_rec=1/gamma_var;

    // Initialisation of descriptors of different matrices

    descinit_ ( DESCT, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCY, &lld_Y, &i_one, &lld_Y, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_Y, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCTD, &i_one, &stripcols, &i_one, &stripcols, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCZU, &lld_Y, &i_one, &lld_Y, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_Y, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQRHS, &ydim, &i_three, &ydim, &i_three, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQSOL, &ydim, &i_three, &ydim, &i_three, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQDENSE, &Ddim, &i_two, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }

    // Allocation of memory for the different matrices in all processes

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    nrmblock = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nrmblock==NULL ) {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }
    Qdense= ( double* ) calloc ( Drows*blocksize * 2, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }


    if ( iam==0 ) {
        Zu= ( double * ) calloc ( lld_Y,sizeof ( double ) );
        if ( Zu==NULL ) {
            printf ( "Error in allocating memory for Zu in root process\n" );
            return -1;
        }
        yblock= ( double * ) calloc ( lld_Y, sizeof ( double ) );
        if ( yblock==NULL ) {
            printf ( "Error in allocating memory for yblock in root process\n" );
            return -1;
        }
        Xtsparse.loadFromFile ( filenameX );
        Ztsparse.loadFromFile ( filenameZ );
        mult_colsA_colsC_denseC ( Ztsparse,solution+m,ydim,0,Ztsparse.ncols,0,1,Zu,n,false,1.0 ); //Zu
        //printdense(nstrips * stripcols, 1, Zu, "Zu.txt");
        Xtsparse.transposeIt ( 1 );
        Ztsparse.transposeIt ( 1 );
        fY=fopen ( filenameY,"rb" );
        if ( fY==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        fread ( yblock,sizeof ( double ),n,fY );
        info=fclose ( fY );
        if ( info!=0 ) {
            printf ( "Error in closing open streams\n" );
            return -1;
        }
        QRHS= ( double * ) calloc ( ydim * 3,sizeof ( double ) );
        if ( QRHS==NULL ) {
            printf ( "Error in allocating memory for QRHS in root process\n" );
            return -1;
        }
        Qsol= ( double * ) calloc ( ydim * 3,sizeof ( double ) );
        if ( Qsol==NULL ) {
            printf ( "Error in allocating memory for Qsol in root process\n" );
            return -1;
        }
        Tdblock = ( double* ) calloc ( stripcols,sizeof ( double ) );
        if ( Tdblock==NULL ) {
            printf ( "unable to allocate memory for Matrix Td\n" );
            return EXIT_FAILURE;
        }
        mult_colsA_colsC_denseC ( Xtsparse,yblock,lld_Y,0,n,0,1,QRHS,ydim,false,sigma_rec );		//X'y/sigma
        mult_colsA_colsC_denseC ( Ztsparse,yblock,n,0,n,0,1,QRHS+m,ydim,false,sigma_rec );		//Z'y/sigma
        mult_colsA_colsC_denseC ( Xtsparse,Zu,lld_Y,0,n,0,1,QRHS+ydim,ydim,false,phi_rec );		//X'Zu/phi
        mult_colsA_colsC_denseC ( Ztsparse,Zu,n,0,n,0,1,QRHS+ydim+m,ydim,false,phi_rec );		//Z'Zu/phi
        *nrmblock = dnrm2_ ( &n,yblock,&i_one );
        *AImat = *nrmblock * *nrmblock/sigma/sigma; 								//y'y/sigma²
        * ( AImat+1 ) = ddot_ ( &n,yblock,&i_one,Zu,&i_one ) /sigma/phi;
        * ( AImat+3 ) = ddot_ ( &n,yblock,&i_one,Zu,&i_one ) /sigma/phi;
        * ( AImat+4 ) = dnrm2_ ( &n,Zu,&i_one ) /phi/phi * dnrm2_ ( &n,Zu,&i_one );
        //printdense(n,1,Zu,"Zu.txt");
        //printf("Fourth element of AImat is: %g\n", *(AImat+4));
    }

    blacs_barrier_ ( &ICTXT2D,"A" );

    fT=fopen ( filenameT,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    *nrmblock=0.0;

    // Set up of matrices used for Average information matrix calculation per strip of Z and X (one strip consists of $blocksize complete rows)

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {
            // The last strip may consist of less rows than $blocksize so previous values should be erased
            if ( Tblock!=NULL )
                free ( Tblock );
            Tblock=NULL;

            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }

            if ( iam==0 ) {
                if ( Tdblock != NULL )
                    free ( Tdblock );
                Tdblock=NULL;
                Tdblock = ( double* ) calloc ( stripcols,sizeof ( double ) );
                if ( Tdblock==NULL ) {
                    printf ( "unable to allocate memory for Matrix Y\n" );
                    return EXIT_FAILURE;
                }
            }

            /*Other possibility:
            Zcols=n%blocksize==0 ? blocksize * * ( dims+1 ) : blocksize * ( * ( dims+1 )-1 ) + ( n%blocksize );
            descinit_ ( DESCZ, &m, &Zcols, &blocksize, &blocksize, &RSRC, &CSRC, &ICTXT, &lld_Z, &INFO );
            if ( INFO!=0 )
            {
               printf ( "Descriptor of matrix Z returns info: %d\n",INFO );
               return INFO;
            }
            descinit_ ( DESCY, &i_one, &Zcols, &i_one, &blocksize, &RSRC, &CSRC, &ICTXT, &i_one, &INFO );
            if ( INFO!=0 )
            {
               printf ( "Descriptor of matrix Y returns info: %d\n",INFO );
               return INFO;
            }
            */
        }

        //Creation of matrix T in every process

        if ( ( nTblocks-1 ) % *dims == *position && k%blocksize !=0 ) {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( pcol * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),k%blocksize,fT );
            }
            //Normal read-in of the strips of T from a binary file (each time blocksize elements are read in)
        } else {
            if ( ni==0 ) {
                info=fseek ( fT, ( long ) ( pcol * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            } else {
                info=fseek ( fT, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
                for ( j=0; j < pTblocks-1; ++j ) {
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                info=fseek ( fT, ( long ) ( ( k - blocksize * ( ( pTblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D,"A" );

        // End of read-in

        // Creation of matrix needed for calculation of AI matrix distributed over every process per block of Z, X and y

        int ystart=ni * * ( dims+1 ) * blocksize + 1;

        pdgemm_ ( "T","N", &i_one, &stripcols,&k,&gamma_rec, solution, &ml_plus,&i_one,DESCSOL,Tblock,&i_one,&i_one,DESCT,&d_zero,Tdblock,&i_one,&i_one,DESCTD ); //Td/gamma (in blocks)

        /*if(iam==0) {
            //printf("Tdblock calculated\n");
            for (i=0; i<stripcols; ++i) {
                *(Tdblock+i) += *(Zu+ ni*stripcols+i) * lambda;  //(Td + Zu)/gamma (in blocks)
            }
            /*char *Tdfile;
            Tdfile=(char *) calloc(100,sizeof(char));
            *Tdfile='\0';
            sprintf(Tdfile,"Tdmat_%d.txt",ni);
            printdense(1,stripcols, Tdblock,Tdfile);*/
        //}

        pdgemm_ ( "N","N",&k,&i_one,&stripcols,&sigma_rec,Tblock,&i_one, &i_one, DESCT,yblock,&ystart,&i_one,DESCY,&d_one,QRHS,&ml_plus,&i_one,DESCQRHS ); //T'y/sigma

        //pdgemm_ ( "N","T",&m,&i_one,&stripcols,&sigma_rec,Xblock,&i_one, &i_one, DESCX,yblock,&i_one,&i_one,DESCY,&d_one,QRHS,&i_one,&i_one,DESCQRHS ); //X'y/sigma

        pdgemm_ ( "N","T",&k,&i_one,&stripcols,&d_one,Tblock,&i_one, &i_one, DESCT,Tdblock,&i_one,&i_one,DESCTD,&d_one,Qdense,&i_one,&i_one,DESCQDENSE ); //T'Td/gamma
        pdgemm_ ( "N","N",&k,&i_one,&stripcols,&phi_rec,Tblock,&i_one, &i_one, DESCT,Zu,&ystart,&i_one,DESCZU,&d_one,QRHS,&ml_plus,&i_two,DESCQRHS ); //T'Zu/phi

        //pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,Tdblock,&i_one,&i_one,DESCTD,&d_one,QRHS,&i_one,&i_two,DESCQRHS ); //X'Zu/gamma

        blacs_barrier_ ( &ICTXT2D,"A" );
        if ( iam==0 ) {
            //printf("Dense multiplications with strip %d of T done\n",ni);
            mult_colsA_colsC_denseC ( Xtsparse, Tdblock, stripcols, ni*stripcols, stripcols,0, 1, QRHS+2*ydim, ydim, true, 1.0 ); 	//X'Td/gamma

            mult_colsA_colsC_denseC ( Ztsparse, Tdblock, stripcols, ni*stripcols, stripcols,0, 1, QRHS+2*ydim+m, ydim, true, 1.0 );	//Z'Td/gamma

            *nrmblock = dnrm2_ ( &stripcols,Tdblock,&i_one );							// norm (Td/gamma)
            * ( AImat + 8 ) += *nrmblock * *nrmblock;										//(Td)'(Td)/gamma^2
            //printf("Sparse multiplications with strip %d of T done\n",ni);
        }

        blacs_barrier_ ( &ICTXT2D,"A" );


        // Q'Q is calculated and stored directly in AI matrix (complete in every process)
        pddot_ ( &stripcols,nrmblock,Tdblock,&i_one,&i_one,DESCTD,&i_one,yblock,&ystart,&i_one,DESCY,&i_one );
        * ( AImat + 6 ) += *nrmblock /sigma;							//y'Td/gamma/sigma
        * ( AImat + 2 ) += *nrmblock /sigma;							//y'Td/gamma/sigma
        pddot_ ( &stripcols,nrmblock,Tdblock,&i_one,&i_one,DESCTD,&i_one,Zu,&ystart,&i_one,DESCZU,&i_one );
        * ( AImat + 5 ) += *nrmblock /phi;							//y'Td/gamma/sigma
        * ( AImat + 7 ) += *nrmblock /phi;							//y'Td/gamma/sigma
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    if ( Tblock != NULL )
        free ( Tblock );
    Tblock=NULL;

    if ( iam==0 ) {
        //printf("All strips of T processed\n");
        if ( Zu != NULL )
            free ( Zu );
        Zu=NULL;
        if ( yblock != NULL )
            free ( yblock );
        yblock=NULL;
        /*if(Tdblock != NULL)
            free ( Tdblock );
        Tdblock=NULL;*/
        Xtsparse.clear();
        Ztsparse.clear();
    }

    info=fclose ( fT );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    /*char *TdZufile;
    TdZufile=(char *) calloc(100,sizeof(char));
    *TdZufile='\0';
    sprintf(TdZufile,"TTdZumat_(%d,%d).txt",*position,pcol);
    printdense(1,Drows*blocksize, Qdense,TdZufile);*/

    //T'(Td)/gamma is in Qdense and is copied to QRHS
    pdcopy_ ( &k,Qdense,&i_one, &i_one, DESCQDENSE, &i_one, QRHS,&ml_plus,&i_three,DESCQRHS, &i_one );

    /*if(iam==0)
        printdense(3,ydim,QRHS,"QRHS.txt");*/

    // In Qsol we calculate the solution of M * Qsol = QRHS, but we still need QRHS a bit further


    pdcopy_ ( &k,QRHS,&ml_plus,&i_two,DESCQRHS,&i_one,Qsol,&ml_plus,&i_two,DESCQSOL, &i_one );
    pdcopy_ ( &k,QRHS,&ml_plus,&i_three,DESCQRHS,&i_one,Qsol,&ml_plus,&i_three,DESCQSOL, &i_one );
    blacs_barrier_ ( &ICTXT2D,"A" );
    if ( iam==0 ) {
        printf ( "Solving system AQsol_1,2 = Q_1,2 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol,QRHS+ydim, 2, 1 );
        printf ( "Solving system AQsol_1,3 = Q_1,3 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol+ydim,QRHS+2*ydim, 2, 1 );
        //printf("AQsol=QRHS_2 is solved\n");
        mult_colsA_colsC_denseC ( Btsparse,Qsol,ydim,0,Btsparse.ncols,0,2,Qsol+ydim+m+l,ydim, true,-1.0 );
        //printf("B^T * Qsol is calculated\n");
    }
    blacs_barrier_ ( &ICTXT2D,"A" );
    pdcopy_ ( &k,Qsol,&ml_plus,&i_two,DESCQSOL,&i_one,Qdense,&i_one,&i_one,DESCQDENSE,&i_one );
    pdcopy_ ( &k,Qsol,&ml_plus,&i_three,DESCQSOL,&i_one,Qdense,&i_one,&i_two,DESCQDENSE,&i_one );

    pdpotrs_ ( "U",&Ddim,&i_two,Dmat,&i_one,&i_one,DESCD,Qdense,&i_one,&i_one,DESCQDENSE,&info );
    if ( info!=0 ) {
        printf ( "Parallel Cholesky solution for Q was unsuccesful, error returned: %d\n",info );
        return -1;
    }
    blacs_barrier_ ( &ICTXT2D,"A" );

    pdcopy_ ( &k,Qdense,&i_one,&i_one,DESCQDENSE,&i_one,Qsol,&ml_plus,&i_two,DESCQSOL,&i_one );
    pdcopy_ ( &k,Qdense,&i_one,&i_two,DESCQDENSE,&i_one,Qsol,&ml_plus,&i_three,DESCQSOL,&i_one );

    if ( Qdense != NULL )
        free ( Qdense );
    Qdense=NULL;

    blacs_barrier_ ( &ICTXT2D,"A" );



    pdcopy_ ( &Adim,QRHS,&i_one,&i_two,DESCQRHS,&i_one,Qsol,&i_one,&i_two,DESCQSOL, &i_one );
    pdcopy_ ( &Adim,QRHS,&i_one,&i_three,DESCQRHS,&i_one,Qsol,&i_one,&i_three,DESCQSOL, &i_one );

    if ( iam==0 ) {
        //printf("Solution of DX=Q OK\n");
        //Btsparse.transposeIt ( 1 );
        mult_colsAtrans_colsC_denseC ( Btsparse,Qsol+ydim+m+l,ydim,0,Btsparse.nrows,0,2,Qsol+ydim,ydim, -1.0 );
        //Btsparse.transposeIt(1);
        double * sparse_sol= ( double * ) calloc ( Asparse.nrows, sizeof ( double ) );
        printf ( "Solving system AQsol_1,2 = Q_1,2 - B Qsol_2,2 on process 0\n" );
        solveSystemwoFact ( Asparse, sparse_sol,Qsol+ydim, 2, 1 );
        memcpy ( Qsol+ydim,sparse_sol, ( m+l ) * sizeof ( double ) );
        printf ( "Solving system AQsol_1,3 = Q_1,3 - B Qsol_2,3 on process 0\n" );
        solveSystemwoFact ( Asparse, sparse_sol,Qsol+2*ydim, 2, 1 );
        memcpy ( Qsol+2*ydim,sparse_sol, ( m+l ) * sizeof ( double ) );
        //printf("sparse_sol copied to Qsol\n");
    }
    blacs_barrier_ ( &ICTXT2D,"A" );

    pdcopy_ ( &ydim,solution,&i_one,&i_one,DESCSOL,&i_one,Qsol,&i_one,&i_one,DESCQSOL,&i_one );
    blacs_barrier_ ( &ICTXT2D,"A" );
    /*if(iam==0)
        printf("solution copied to Qsol\n");*/
    pdscal_ ( &ydim,&sigma_rec,Qsol,&i_one,&i_one,DESCQSOL,&i_one );
    blacs_barrier_ ( &ICTXT2D,"A" );
    /*if(iam==0)
        printf("Qsol scaled with sigma_rec\n");*/

    //printdense(2,ydim,QRHS,"QRHS.txt");
    //printdense(2,ydim,Qsol,"Qsol.txt");

    // AImat = (Q'Q - QRHS' * Qsol) / 2 / sigma

    //printdense(2,2,AImat,"QQ.txt");

    pdgemm_ ( "T","N",&i_three,&i_three,&ydim,&d_negone,QRHS,&i_one,&i_one,DESCQRHS,Qsol,&i_one,&i_one,DESCQSOL,&d_one, AImat,&i_one,&i_one,DESCAI );

    /*if(iam==0) {
        if(QRHS != NULL)
            free(QRHS);
        QRHS=NULL;
        if(Qsol != NULL)
            free(Qsol);
        Qsol=NULL;
    }*/

    //printdense(2,2,AImat,"AI_nonorm.txt");

    for ( i=0; i<9; ++i )
        * ( AImat + i ) = * ( AImat + i ) / 2 / sigma;

    blacs_barrier_ ( &ICTXT2D,"A" );

    if ( DESCQRHS != NULL )
        free ( DESCQRHS );
    DESCQRHS=NULL;
    if ( DESCQSOL != NULL )
        free ( DESCQSOL );
    DESCQSOL=NULL;
    if ( DESCQDENSE != NULL )
        free ( DESCQDENSE );
    DESCQDENSE=NULL;
    if ( DESCY != NULL )
        free ( DESCY );
    DESCY=NULL;
    if ( DESCT != NULL )
        free ( DESCT );
    DESCT=NULL;
    if ( DESCTD != NULL )
        free ( DESCTD );
    DESCTD=NULL;
    if ( nrmblock != NULL )
        free ( nrmblock );
    nrmblock=NULL;
    //free ( QRHS );
    //free ( Qsol );
    /**/

    return 0;
}

int set_up_AI ( double * AImat, int * DESCDENSESOL, double * densesol, int * DESCD, double * Dmat, CSRdouble &Asparse, int *DESCB, double *Bmat, double sigma ) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    FILE* fT, * fY;
    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCTD, *DESCZU, *DESCQRHS, *DESCQSOL, *DESCQDENSE;
    double *Tblock, *yblock, *Tdblock, *QRHS, *Qsol,*nrmblock, sigma_rec, phi_rec, gamma_rec, *Zu, *Qdense;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur, lld_Y;
    double vm_usage, resident_set, cpu_sys, cpu_user;

    MPI_Status status;

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;

    nstrips= n % ( blocksize * * ( dims+1 ) ) ==0 ?  n / ( blocksize * * ( dims+1 ) ) : ( n / ( blocksize * * ( dims+1 ) ) ) +1; 	// number of strips in which we divide matrix Z' and X'
    stripcols= blocksize * * ( dims+1 ); 												//the number of columns taken into the strip of Z' and X'

    sigma_rec=1/sigma;
    phi_rec=1/phi;
    gamma_rec=1/gamma_var;

    lld_Y= * ( dims+1 ) * nstrips * blocksize;


    if ( iam !=0 ) {

        // Initialisation of descriptors of different matrices

        DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCT==NULL ) {
            printf ( "unable to allocate memory for descriptor for Z\n" );
            return -1;
        }
        DESCY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCY==NULL ) {
            printf ( "unable to allocate memory for descriptor for Y\n" );
            return -1;
        }
        DESCTD= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCTD==NULL ) {
            printf ( "unable to allocate memory for descriptor for Td\n" );
            return -1;
        }
        DESCZU= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCTD==NULL ) {
            printf ( "unable to allocate memory for descriptor for Zu\n" );
            return -1;
        }
        DESCQRHS= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCQRHS==NULL ) {
            printf ( "unable to allocate memory for descriptor for QRHS\n" );
            return -1;
        }
        DESCQSOL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCQSOL==NULL ) {
            printf ( "unable to allocate memory for descriptor for QSOL\n" );
            return -1;
        }
        DESCQDENSE= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCQDENSE==NULL ) {
            printf ( "unable to allocate memory for descriptor for ZtY\n" );
            return -1;
        }

        //Matrices Z & X are read in from the binary files, however, since PBLAS only accepts column-major matrices, we store the transposed Z and X matrices (column major)

        pcol= * ( position+1 );
        nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;										//number of blocks necessary to store complete column of Z'
        pTblocks= ( nTblocks - *position ) % *dims == 0 ? ( nTblocks- *position ) / *dims : ( nTblocks- *position ) / *dims +1;		//number of blocks necessary per processor
        pTblocks= pTblocks <1? 1:pTblocks;
        lld_T=pTblocks*blocksize;													//local leading dimension of the strip of Z (different from processor to processor)														//local leading dimension of the strip of Z (different from processor to processor)


        // Initialisation of descriptors of different matrices

        descinit_ ( DESCT, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Z returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCY, &lld_Y, &i_one, &lld_Y, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_Y, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Y returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCTD, &i_one, &stripcols, &i_one, &stripcols, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Z returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCZU, &lld_Y, &i_one, &lld_Y, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_Y, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Z returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCQRHS, &ydim, &i_three, &ydim, &i_three, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Z returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCQSOL, &ydim, &i_three, &ydim, &i_three, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Z returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCQDENSE, &Ddim, &i_two, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix Y returns info: %d\n",info );
            return info;
        }

        // Allocation of memory for the different matrices in all processes

        Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
        if ( Tblock==NULL ) {
            printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
            return -1;
        }
        Qdense= ( double* ) calloc ( Drows*blocksize * 2, sizeof ( double ) );
        if ( Tblock==NULL ) {
            printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
            return -1;
        }

        fT=fopen ( filenameT,"rb" );
        if ( fT==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        if ( * ( position+1 ) ==0 ) {
            yblock= ( double * ) calloc ( lld_Y, sizeof ( double ) );
            if ( yblock==NULL ) {
                printf ( "Error in allocating memory for yblock in root process\n" );
                return -1;
            }
            Zu= ( double * ) calloc ( lld_Y, sizeof ( double ) );
            if ( Zu==NULL ) {
                printf ( "Error in allocating memory for Zu in root process\n" );
                return -1;
            }
            Tdblock = ( double* ) calloc ( stripcols,sizeof ( double ) );
            if ( Tdblock==NULL ) {
                printf ( "unable to allocate memory for Matrix Td\n" );
                return EXIT_FAILURE;
            }
            QRHS= ( double * ) calloc ( ydim * 3,sizeof ( double ) );
            if ( QRHS==NULL ) {
                printf ( "Error in allocating memory for QRHS in root process\n" );
                return -1;
            }
            Qsol= ( double * ) calloc ( ydim * 3,sizeof ( double ) );
            if ( Qsol==NULL ) {
                printf ( "Error in allocating memory for Qsol in root process\n" );
                return -1;
            }
            fY=fopen ( filenameY,"rb" );
            if ( fY==NULL ) {
                printf ( "Error opening file\n" );
                return -1;
            }
            fread ( yblock,sizeof ( double ),n,fY );
            info=fclose ( fY );
            if ( info!=0 ) {
                printf ( "Error in closing open streams\n" );
                return -1;
            }
            MPI_Recv ( Zu,n, MPI_DOUBLE,0,n,MPI_COMM_WORLD,&status );
            //cout << "Zu received" << endl;
            process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            clustout << "At end of allocations in cluster processes (set_up_AI)" << endl;
            clustout << "======================================================" << endl;
            clustout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            clustout << "Resident set size:    " << resident_set << " kb" << endl;
            clustout << "CPU time (user):      " << cpu_user << " s"<< endl;
            clustout << "CPU time (system):    " << cpu_sys << " s" << endl;
        }

        // Set up of matrices used for Average information matrix calculation per strip of Z and X (one strip consists of $blocksize complete rows)

        for ( ni=0; ni<nstrips; ++ni ) {
            if ( ni==nstrips-1 ) {
                // The last strip may consist of less rows than $blocksize so previous values should be erased
                if ( Tblock!=NULL )
                    free ( Tblock );
                Tblock=NULL;

                Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
                if ( Tblock==NULL ) {
                    printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                    return -1;
                }

                if ( * ( position + 1 ) ==0 ) {
                    if ( Tdblock != NULL )
                        free ( Tdblock );
                    Tdblock=NULL;
                    Tdblock = ( double* ) calloc ( stripcols,sizeof ( double ) );
                    if ( Tdblock==NULL ) {
                        printf ( "unable to allocate memory for Matrix Y\n" );
                        return EXIT_FAILURE;
                    }
                }
            }

            //Creation of matrix T in every process

            if ( ( nTblocks-1 ) % *dims == *position && k%blocksize !=0 ) {
                if ( ni==0 ) {
                    info=fseek ( fT, ( long ) ( pcol * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                } else {
                    info=fseek ( fT, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                for ( i=0; i<blocksize; ++i ) {
                    info=fseek ( fT, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                    for ( j=0; j < pTblocks-1; ++j ) {
                        fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                        info=fseek ( fT, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                        if ( info!=0 ) {
                            printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                            return -1;
                        }
                    }
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),k%blocksize,fT );
                }
                //Normal read-in of the strips of T from a binary file (each time blocksize elements are read in)
            } else {
                if ( ni==0 ) {
                    info=fseek ( fT, ( long ) ( pcol * blocksize * ( k ) * sizeof ( double ) ),SEEK_SET );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                } else {
                    info=fseek ( fT, ( long ) ( blocksize * ( * ( dims+1 )-1 ) * ( k ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
                for ( i=0; i<blocksize; ++i ) {
                    info=fseek ( fT, ( long ) ( blocksize * *position * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                    for ( j=0; j < pTblocks-1; ++j ) {
                        fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                        info=fseek ( fT, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                        if ( info!=0 ) {
                            printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                            return -1;
                        }
                    }
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( k - blocksize * ( ( pTblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
            }

            blacs_barrier_ ( &ICTXT2D,"A" );

            // End of read-in

            // Creation of matrix needed for calculation of AI matrix distributed over every process per block of Z, X and y

            int ystart=ni * * ( dims+1 ) * blocksize + 1;

            /*if(*(position +1)==0)
              cout << "ystart = " << ystart << endl;*/

            pdgemm_ ( "T","N", &i_one, &stripcols,&k,&gamma_rec, densesol, &i_one,&i_one,DESCDENSESOL,Tblock,&i_one,&i_one,DESCT,&d_zero,Tdblock,&i_one,&i_one,DESCTD ); //Td/gamma (in blocks)
            //pdgemm_ ( "T","N", &i_one, &stripcols,&k,&gamma_rec, solution, &ml_plus,&i_one,DESCSOL,Tblock,&i_one,&i_one,DESCT,&d_zero,Tdblock,&i_one,&i_one,DESCTD ); //Td/gamma (in blocks)


            /*if(iam==0) {
                //printf("Tdblock calculated\n");
                for (i=0; i<stripcols; ++i) {
                    *(Tdblock+i) += *(Zu+ ni*stripcols+i) * lambda;  //(Td + Zu)/gamma (in blocks)
                }*/
            if ( * ( position +1 ) ==0 ) {
                //cout << "Td calculated" << endl;
                MPI_Send ( Tdblock,stripcols, MPI_DOUBLE,0,ni, MPI_COMM_WORLD );
                /*cout << "Td sent" << endl;
                        char *Tdfile;
                        Tdfile=(char *) calloc(100,sizeof(char));
                        *Tdfile='\0';
                        sprintf(Tdfile,"Tdmat_%d.txt",ni);
                        printdense(1,stripcols, Tdblock,Tdfile);*/
            }

            pdgemm_ ( "N","N",&k,&i_one,&stripcols,&sigma_rec,Tblock,&i_one, &i_one, DESCT,yblock,&ystart,&i_one,DESCY,&d_one,QRHS,&ml_plus,&i_one,DESCQRHS ); //T'y/sigma

            pdgemm_ ( "N","T",&k,&i_one,&stripcols,&d_one,Tblock,&i_one, &i_one, DESCT,Tdblock,&i_one,&i_one,DESCTD,&d_one,Qdense,&i_one,&i_two,DESCQDENSE ); //T'Td/gamma
            pdgemm_ ( "N","N",&k,&i_one,&stripcols,&phi_rec,Tblock,&i_one, &i_one, DESCT,Zu,&ystart,&i_one,DESCZU,&d_one,QRHS,&ml_plus,&i_two,DESCQRHS ); //T'Zu/phi

            blacs_barrier_ ( &ICTXT2D,"A" );

        }
        pdcopy_ ( &k,Qdense,&i_one, &i_two, DESCQDENSE, &i_one, QRHS,&ml_plus,&i_three,DESCQRHS, &i_one );
        pdcopy_ ( &k,QRHS,&ml_plus, &i_two, DESCQRHS, &i_one, Qdense,&i_one,&i_one,DESCQDENSE, &i_one );
        if ( Tblock != NULL )
            free ( Tblock );
        Tblock=NULL;
        if ( * ( position+1 ) ==0 ) {
            if ( yblock!=NULL ) {
                free ( yblock );
            }
            yblock==NULL;
            if ( Zu!=NULL ) {
                free ( Zu );
            }
            Zu==NULL;

            MPI_Send ( QRHS + Adim,k, MPI_DOUBLE,0, k, MPI_COMM_WORLD );
            MPI_Send ( QRHS + Adim+ ydim,k, MPI_DOUBLE, 0, 2 * k, MPI_COMM_WORLD );
            MPI_Send ( QRHS + Adim+ ydim *2,k, MPI_DOUBLE,0,3 * k, MPI_COMM_WORLD );

            MPI_Recv ( QRHS + ydim,Adim, MPI_DOUBLE,0,ydim,MPI_COMM_WORLD,&status ); // here we get A^{-1} x QRHS (sparse part)
            MPI_Recv ( QRHS + 2*ydim,Adim, MPI_DOUBLE,0,2*ydim,MPI_COMM_WORLD,&status );



        }
        info=fclose ( fT );
        if ( info!=0 ) {
            printf ( "Error in closing open streams" );
            return -1;
        }

        pdgemm_ ( "T","N",&k,&i_two,&Adim,&d_negone,Bmat,&i_one, &i_one, DESCB,QRHS,&i_one,&i_two,DESCQRHS,&d_one,Qdense,&i_one,&i_one,DESCQDENSE ); //[T'Zu/phi T'Td/gamma] - B'A^{-1} [X'Q, Z'Q]

        pdpotrs_ ( "U",&Ddim,&i_two,Dmat,&i_one,&i_one,DESCD,Qdense,&i_one,&i_one,DESCQDENSE,&info );
        if ( info!=0 ) {
            printf ( "Parallel Cholesky solution for Q was unsuccesful, error returned: %d\n",info );
            return -1;
        }
        pdgemm_ ( "N","N",&Adim,&i_two,&Ddim,&d_one,Bmat,&i_one, &i_one, DESCB,Qdense,&i_one,&i_one,DESCQDENSE,&d_zero,QRHS,&i_one,&i_one,DESCQRHS ); //B multiplied with dense part of Qsol

        pdcopy_ ( &k,Qdense,&i_one, &i_one, DESCQDENSE, &i_one, QRHS,&ml_plus,&i_one,DESCQRHS, &i_one );
        pdcopy_ ( &k,Qdense,&i_one, &i_two, DESCQDENSE, &i_one, QRHS,&ml_plus,&i_two,DESCQRHS, &i_one );

        if ( * ( position+1 ) ==0 ) {
            MPI_Send ( QRHS,ydim, MPI_DOUBLE,0, ydim, MPI_COMM_WORLD );
            MPI_Send ( QRHS + ydim,ydim, MPI_DOUBLE, 0, 2 * ydim, MPI_COMM_WORLD );
            if ( QRHS!=NULL ) {
                free ( QRHS );
            }
            QRHS==NULL;
            if ( Qsol!=NULL ) {
                free ( Qsol );
            }
            Qsol==NULL;
        }


        if ( Qdense != NULL )
            free ( Qdense );
        Qdense=NULL;

        if ( DESCQRHS != NULL )
            free ( DESCQRHS );
        DESCQRHS=NULL;
        if ( DESCQSOL != NULL )
            free ( DESCQSOL );
        DESCQSOL=NULL;
        if ( DESCQDENSE != NULL )
            free ( DESCQDENSE );
        DESCQDENSE=NULL;
        if ( DESCY != NULL )
            free ( DESCY );
        DESCY=NULL;
        if ( DESCT != NULL )
            free ( DESCT );
        DESCT=NULL;
        if ( DESCTD != NULL )
            free ( DESCTD );
        DESCTD=NULL;
        if ( DESCZU != NULL )
            free ( DESCZU );
        DESCZU=NULL;
        if ( nrmblock != NULL )
            free ( nrmblock );
        nrmblock=NULL;
    }
    else {
        Zu= ( double * ) calloc ( lld_Y,sizeof ( double ) );
        if ( Zu==NULL ) {
            printf ( "Error in allocating memory for Zu in root process\n" );
            return -1;
        }
        yblock= ( double * ) calloc ( lld_Y, sizeof ( double ) );
        if ( yblock==NULL ) {
            printf ( "Error in allocating memory for yblock in root process\n" );
            return -1;
        }
        Tdblock = ( double* ) calloc ( stripcols,sizeof ( double ) );
        if ( Tdblock==NULL ) {
            printf ( "unable to allocate memory for Matrix Td\n" );
            return EXIT_FAILURE;
        }
        nrmblock = ( double* ) calloc ( 1,sizeof ( double ) );
        if ( nrmblock==NULL ) {
            printf ( "unable to allocate memory for norm\n" );
            return EXIT_FAILURE;
        }
        *nrmblock=0.0;
        Xtsparse.loadFromFile ( filenameX );
        Ztsparse.loadFromFile ( filenameZ );
        mult_colsA_colsC_denseC ( Ztsparse,densesol+m,ydim,0,Ztsparse.ncols,0,1,Zu,n,false,1.0 ); //Zu
        //printdense(n, 1, Zu, "Zu.txt");
        Xtsparse.transposeIt ( 1 );
        Ztsparse.transposeIt ( 1 );
        //cout << "X and Z transposed" << endl;
        MPI_Send ( Zu,n, MPI_DOUBLE,1,n, MPI_COMM_WORLD );

        //cout << "Zu sent" << endl;
        fY=fopen ( filenameY,"rb" );
        if ( fY==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        fread ( yblock,sizeof ( double ),n,fY );
        info=fclose ( fY );
        if ( info!=0 ) {
            printf ( "Error in closing open streams\n" );
            return -1;
        }
        //cout << "Y read from file" << endl;
        QRHS= ( double * ) calloc ( ydim * 3,sizeof ( double ) );
        if ( QRHS==NULL ) {
            printf ( "Error in allocating memory for QRHS in root process\n" );
            return -1;
        }
        Qsol= ( double * ) calloc ( ydim * 3,sizeof ( double ) );
        if ( Qsol==NULL ) {
            printf ( "Error in allocating memory for Qsol in root process\n" );
            return -1;
        }
        process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
        rootout << "At end of allocations in root proces (set_up_AI)" << endl;
        rootout <<  "================================================" << endl;
        rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
        rootout << "Resident set size:    " << resident_set << " kb" << endl;
        rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
        rootout << "CPU time (system):    " << cpu_sys << " s" << endl;

        //cout << "Memory allocated" << endl;
        mult_colsA_colsC_denseC ( Xtsparse,yblock,n,0,n,0,1,QRHS,ydim,false,sigma_rec );		//X'y/sigma
        mult_colsA_colsC_denseC ( Ztsparse,yblock,n,0,n,0,1,QRHS+m,ydim,false,sigma_rec );		//Z'y/sigma
        mult_colsA_colsC_denseC ( Xtsparse,Zu,n,0,n,0,1,QRHS+ydim,ydim,false,phi_rec );		//X'Zu/phi
        mult_colsA_colsC_denseC ( Ztsparse,Zu,n,0,n,0,1,QRHS+ydim+m,ydim,false,phi_rec );		//Z'Zu/phi
        //cout << "Multiplications performed" << endl;
        *nrmblock = dnrm2_ ( &n,yblock,&i_one );
        //cout << "Norm of Y calculated: " << *nrmblock << endl;
        *AImat = *nrmblock * *nrmblock/sigma/sigma; 								//y'y/sigma²
        * ( AImat+1 ) = ddot_ ( &n,yblock,&i_one,Zu,&i_one ) /sigma/phi;
        * ( AImat+3 ) = ddot_ ( &n,yblock,&i_one,Zu,&i_one ) /sigma/phi;
        * ( AImat+4 ) = dnrm2_ ( &n,Zu,&i_one ) /phi/phi * dnrm2_ ( &n,Zu,&i_one );
        //printdense(n,1,Zu,"Zu.txt");
        //printf("Fourth element of AImat is: %g\n", *(AImat+4));


        for ( ni=0; ni<nstrips; ++ni ) {

            MPI_Recv ( Tdblock,stripcols, MPI_DOUBLE,1,ni,MPI_COMM_WORLD,&status );
            //cout << "Received Tdblock " << ni +1 << endl;
            //printf("Dense multiplications with strip %d of T done\n",ni);
            mult_colsA_colsC_denseC ( Xtsparse, Tdblock, stripcols, ni*stripcols, stripcols,0, 1, QRHS+2*ydim, ydim, true, 1.0 ); 	//X'Td/gamma

            mult_colsA_colsC_denseC ( Ztsparse, Tdblock, stripcols, ni*stripcols, stripcols,0, 1, QRHS+2*ydim+m, ydim, true, 1.0 );	//Z'Td/gamma

            *nrmblock = dnrm2_ ( &stripcols,Tdblock,&i_one );							// norm (Td/gamma)
            * ( AImat + 8 ) += *nrmblock * *nrmblock;										//(Td)'(Td)/gamma^2
            //printf("Sparse multiplications with strip %d of T done\n",ni);

            // Q'Q is calculated and stored directly in AI matrix (complete in every process)
            * nrmblock = ddot_ ( &stripcols,Tdblock,&i_one,yblock + ni * * ( dims+1 ) * blocksize ,&i_one );
            * ( AImat + 6 ) += *nrmblock /sigma;							//y'Td/gamma/sigma
            * ( AImat + 2 ) += *nrmblock /sigma;							//y'Td/gamma/sigma
            * nrmblock = ddot_ ( &stripcols,Tdblock,&i_one,Zu + ni * * ( dims+1 ) * blocksize,&i_one );
            * ( AImat + 5 ) += *nrmblock /phi;							//(Zu)'Td/gamma/phi
            * ( AImat + 7 ) += *nrmblock /phi;							//(Zu)'Td/gamma/phi
        }
        //printdense(3,3,AImat,"QQ.txt");
        if ( Zu != NULL )
            free ( Zu );
        Zu=NULL;
        if ( yblock != NULL )
            free ( yblock );
        yblock=NULL;
        if ( Tdblock != NULL )
            free ( Tdblock );
        Tdblock=NULL;
        Xtsparse.clear();
        Ztsparse.clear();

        // In Qsol we calculate the solution of M * Qsol = QRHS, but we still need QRHS a bit further
        // We first get the missing parts of QRHS from process 1
        MPI_Recv ( QRHS + Adim,k, MPI_DOUBLE,1,k,MPI_COMM_WORLD,&status );
        MPI_Recv ( QRHS + Adim +ydim,k, MPI_DOUBLE,1,2*k,MPI_COMM_WORLD,&status );
        MPI_Recv ( QRHS + Adim+2*ydim,k, MPI_DOUBLE,1,3*k,MPI_COMM_WORLD,&status );

        printf ( "Solving system AQsol_1,2 = Q_1,2 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol,QRHS+ydim, 2, 1 );
        printf ( "Solving system AQsol_1,3 = Q_1,3 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol+ydim,QRHS+2*ydim, 2, 1 );
        //printf("AQsol=QRHS_2 is solved\n");
        MPI_Send ( Qsol,Adim, MPI_DOUBLE,1,ydim, MPI_COMM_WORLD );
        MPI_Send ( Qsol+ydim,Adim, MPI_DOUBLE,1,ydim*2, MPI_COMM_WORLD );

        double * Qtemp = new double [2 * Adim];

        MPI_Recv ( Qsol+ydim,ydim, MPI_DOUBLE,1,ydim,MPI_COMM_WORLD,&status );
        MPI_Recv ( Qsol + 2*ydim,ydim, MPI_DOUBLE,1,2*ydim,MPI_COMM_WORLD,&status );

        for ( i=0; i<Adim; ++i ) {
            Qtemp[i] = * ( QRHS + ydim + i ) - * ( Qsol + ydim + i );
            Qtemp[i + Adim] = * ( QRHS + 2*ydim + i ) - * ( Qsol + 2*ydim + i );
        }

        printf ( "Solving system AQsol_1,2 = Q_1,2 - B Qsol_2,2 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol+ydim,Qtemp, 2, 1 );
        printf ( "Solving system AQsol_1,3 = Q_1,3 - B Qsol_2,3 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol+2*ydim,Qtemp+Adim, 2, 1 );

        if ( Qtemp != NULL )
            delete [] Qtemp ;
        Qtemp=NULL;

        for ( i=0; i<ydim; ++i ) {
            * ( Qsol+i ) = * ( densesol+i ) / sigma;
        }

        /*printdense(3,ydim,QRHS,"QRHS.txt");
        printdense(3,ydim,Qsol,"Qsol.txt");*/

        // AImat = (Q'Q - QRHS' * Qsol) / 2 / sigma
        dgemm_ ( "T","N",&i_three,&i_three,&ydim,&d_negone,QRHS,&ydim,Qsol,&ydim,&d_one, AImat,&i_three );

        if ( QRHS != NULL )
            free ( QRHS );
        QRHS=NULL;
        if ( Qsol != NULL )
            free ( Qsol );
        Qsol=NULL;

        for ( i=0; i<9; ++i )
            * ( AImat + i ) = * ( AImat + i ) / 2 / sigma;

        //printf("B^T * Qsol is calculated\n");

    }

    MPI_Barrier ( MPI_COMM_WORLD );

    return 0;
}

double trace_CZZ ( double *mat, int * DESCMAT ) {

    // Every proces calculates the sum of the diagonal elements of mat which are stored in the proces

    double trace_proc;
    int i, j, rowcur,colcur, nXblocks;

    trace_proc=0.0;

    nXblocks= m%blocksize==0 ? m/blocksize : m/blocksize +1;									//number of blocks necessary to store complete column of X'

    for ( i=0,rowcur=0,colcur=0; i<Dblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims )
            rowcur=0;
        if ( colcur==* ( dims+1 ) )
            colcur=0;
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            if ( i< ( Dblocks -1 ) ) {
                for ( j=0; j<blocksize; ++j ) {
                    trace_proc += * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j );
                }
            } else {
                for ( j=0; j< Ddim % blocksize; ++j ) {
                    trace_proc += * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j );
                }
            }
        }
    }

    for ( i=0,rowcur=0,colcur=0; i<nXblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims )
            rowcur=0;
        if ( colcur==* ( dims+1 ) )
            colcur=0;
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            if ( i<nXblocks-1 ) {
                for ( j=0; j<blocksize; ++j ) {
                    trace_proc -= * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j );
                }
            } else {
                for ( j=0; j<= ( m-1 ) %blocksize; ++j ) {
                    trace_proc -= * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j );
                }
            }
        }
    }
    return trace_proc;
}

double log_determinant_C ( double *mat, int * DESCMAT ) {

    // Every proces calculates the sum of the logarithm of the diagonal elements of mat, which are stored in the proces, multiplied by lld_C

    double logdet_proc;
    int i, j, rowcur,colcur, nXblocks;

    logdet_proc=0.0;

    for ( i=0,rowcur=0,colcur=0; i<Dblocks; ++i, ++colcur, ++rowcur ) {
        if ( rowcur==*dims )
            rowcur=0;
        if ( colcur==* ( dims+1 ) )
            colcur=0;
        if ( *position==rowcur && * ( position+1 ) == colcur ) {
            if ( i==Dblocks-1 && Ddim%blocksize !=0 ) {
                for ( j=0; j< Ddim % blocksize; ++j ) {
                    logdet_proc += log ( * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ) );
                }
            } else {
                for ( j=0; j<blocksize; ++j ) {
                    logdet_proc += log ( * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ) );
                }
            }
        }
    }
    return logdet_proc;
}

