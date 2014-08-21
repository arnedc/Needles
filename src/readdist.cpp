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
    int MPI_Ssend(void*, int, MPI_Datatype, int, int, MPI_Comm);
    int MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
    int MPI_Get_count(MPI_Status *, MPI_Datatype, int *);
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    int blacs_pnum_ ( int *ConTxt, int *prow, int *pcol );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib,
		   int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdtran_ ( int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdlacpy_(char *uplo, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb);
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    double dnrm2_ ( int *n, double *x, int *incx );
    //void dgebs2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda );
    //void dgebr2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
}

int set_up_BDY ( int * DESCD, double * Dmat, CSRdouble& BT_i, CSRdouble& B_j, int * DESCYTOT, double * ytot, double *respnrm, CSRdouble& Btsparse  ) {

    // Read-in of matrices X, Z and T from file (filename[X,Z,T])
    // X and Z are read in entrely by every process
    // T is read in strip by strip (number of rows in each process is at maximum = blocksize)
    // D is constructed directly in a distributed way
    // B is first assembled sparse in root process and afterwards the necessary parts
    // for constructing the distributed Schur complement are sent to each process

    FILE *fT, *fY;
    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCZtY, *DESCXtY;
    double *Tblock, *temp, *Y;
    double * ZtY, *XtY;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;
    int nYblocks, pYblocks, lld_Y, Ystart;

    MPI_Status status;

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;

    Xtsparse.loadFromFile ( filenameX );
    Ztsparse.loadFromFile ( filenameZ );

    Xtsparse.transposeIt ( 1 );
    Ztsparse.transposeIt ( 1 );

    XtT_sparse.allocate ( m,k,0 );
    ZtT_sparse.allocate ( l,k,0 );

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
    descinit_ ( DESCY, &n, &i_one, &n, &blocksize, &i_zero, &i_zero, &ICTXT2D, &n, &info );
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
    if(iam==0) {
        Y= ( double* ) calloc ( n, sizeof ( double ) );
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
                * ( temp + j  * lld_D +j ) =lambda;
            }
            if ( i==Dblocks-1 && Ddim % blocksize != 0 ) {
                for ( j=blocksize-1; j>= Ddim % blocksize; --j ) {
                    * ( temp + j * lld_D + j ) =0.0;
                }
            }
        }

    }
    if (iam==0) {
        fY=fopen ( filenameY,"rb" );
        if ( fY==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        info=fread ( Y,sizeof ( double ),n,fY );
	if(info<n){
	  printf("Only %d values were read from %s",info,filenameY);
	}
    }

    fT=fopen ( filenameT,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {

            free ( Tblock );

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

        // End of read-in

        // Matrix D is the sum of the multiplications of all strips of T' by their transpose
        // Up unitl now, the entire matrix is stored, not only upper/lower triangular, which is possible since D is symmetric
        // Be aware, that you akways have to allocate memory for the enitre matrix, even when only dealing with the upper/lower triangular part

        pdgemm_ ( "N","T",&k,&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &i_one, &i_one, DESCD ); //T'T
	
        //pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &t_plus, &t_plus, DESCD );
        Ystart=ni * *(dims+1) * blocksize + 1;
        pdgemm_ ( "N","N",&k,&i_one,&stripcols,&d_one,Tblock,&i_one, &i_one, DESCT,Y,&Ystart,&i_one,DESCY,&d_one,ytot,&ml_plus,&i_one,DESCYTOT ); //T'y
	

        // Matrix B consists of X'T and Z'T, since each process only has some parts of T at its disposal,
        // we need to make sure that the correct columns of Z and X are multiplied with the correct columns of T.
        for ( i=0; i<pTblocks; ++i ) {
            XtT_temp.ncols=k;

            //This function multiplies the correct columns of X' with the blocks of T at the disposal of the process
            // The result is also stored immediately at the correct positions of X'T. (see src/tools.cpp)
            mult_colsA_colsC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                               ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );
            if ( XtT_temp.nonzeros>0 ) {
                if ( XtT_sparse.nonzeros==0 )
                    XtT_sparse.make ( XtT_temp.nrows,XtT_temp.ncols,XtT_temp.nonzeros,XtT_temp.pRows,XtT_temp.pCols,XtT_temp.pData );
                else {
                    XtT_sparse.addBCSR ( XtT_temp );
                }
            }
        }
        //Same as above for calculating Z'T
        for ( i=0; i<pTblocks; ++i ) {
            ZtT_temp.ncols=k;
            mult_colsA_colsC ( Ztsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                               blocksize * ( *dims * i + *position ), blocksize, ZtT_temp, 0 );
            if ( ZtT_temp.nonzeros>0 ) {
                if ( ZtT_sparse.nonzeros==0 )
                    ZtT_sparse.make ( ZtT_temp.nrows,ZtT_temp.ncols,ZtT_temp.nonzeros,ZtT_temp.pRows,ZtT_temp.pCols,ZtT_temp.pData );
                else
                    ZtT_sparse.addBCSR ( ZtT_temp );
            }
        }
        blacs_barrier_ ( &ICTXT2D,"A" );
    }
    free ( Tblock );
    if (iam==0) {
        *respnrm = dnrm2_ (&n,Y,&i_one);
        *respnrm = *respnrm * *respnrm;
        XtY=(double * ) calloc(m,sizeof(double));
        if(XtY==NULL) {
            printf("Unable to allocate memory for XtY in root process.\n");
            return -1;
        }
        mult_colsA_colsC_denseC ( Xtsparse, Y, n, 0, n, 0, 1, XtY, m, false, 1.0);
        ZtY=(double * ) calloc(l,sizeof(double));
        if(ZtY==NULL) {
            printf("Unable to allocate memory for ZtY in root process.\n");
            return -1;
        }
        mult_colsA_colsC_denseC ( Ztsparse, Y, n, 0, n, 0, 1, ZtY, l, false, 1.0);
        free(Y);
    }
    pdlacpy_("A",&m,&i_one,XtY,&i_one,&i_one,DESCXtY, ytot,&i_one,&i_one,DESCYTOT);
    pdlacpy_("A",&l,&i_one,ZtY,&i_one,&i_one,DESCZtY, ytot,&m_plus,&i_one,DESCYTOT);

    if(iam==0) {
        free(XtY);
        free(ZtY);
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
        MPI_Ssend ( & ( ZtT_sparse.nonzeros ),1, MPI_INT,0,iam,MPI_COMM_WORLD );
        MPI_Ssend ( & ( ZtT_sparse.pRows[0] ),ZtT_sparse.nrows + 1, MPI_INT,0,4*size + iam,MPI_COMM_WORLD );
        MPI_Ssend ( & ( ZtT_sparse.pCols[0] ),ZtT_sparse.nonzeros, MPI_INT,0,iam+ 5*size,MPI_COMM_WORLD );
        MPI_Ssend ( & ( ZtT_sparse.pData[0] ),ZtT_sparse.nonzeros, MPI_DOUBLE,0,iam+6*size,MPI_COMM_WORLD );

        // And eventually receives the necessary BT_i and B_j
        // Blocking sends are used, which is why the order of the receives is critical depending on the coordinates of the process
        int nonzeroes;
        if (*position >= pcol) {
            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            BT_i.allocate ( blocksize*Drows,m+l,nonzeroes );
            MPI_Recv ( & ( BT_i.pRows[0] ),blocksize*Drows + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            int count;
            MPI_Get_count(&status,MPI_INT,&count);
            BT_i.nrows=count-1;
            MPI_Recv ( & ( BT_i.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( BT_i.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );

            MPI_Recv ( &nonzeroes,1, MPI_INT,0,iam+4*size,MPI_COMM_WORLD,&status );

            B_j.allocate ( blocksize*Dcols,m+l,nonzeroes );

            MPI_Recv ( & ( B_j.pRows[0] ),blocksize*Dcols + 1, MPI_INT,0,iam + 5*size,MPI_COMM_WORLD,&status );
            MPI_Get_count(&status,MPI_INT,&count);
            B_j.nrows=count-1;
            MPI_Recv ( & ( B_j.pCols[0] ),nonzeroes, MPI_INT,0,iam+6*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( B_j.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+7*size,MPI_COMM_WORLD,&status );

            //Actually BT_j is sent, so it still needs to be transposed
            B_j.transposeIt ( 1 );
        }
        else {
            MPI_Recv ( &nonzeroes,1, MPI_INT,0,iam+4*size,MPI_COMM_WORLD,&status );

            B_j.allocate ( blocksize*Dcols,m+l,nonzeroes );

            MPI_Recv ( & ( B_j.pRows[0] ),blocksize*Dcols + 1, MPI_INT,0,iam + 5*size,MPI_COMM_WORLD,&status );
            int count;
            MPI_Get_count(&status,MPI_INT,&count);
            B_j.nrows=count-1;

            MPI_Recv ( & ( B_j.pCols[0] ),nonzeroes, MPI_INT,0,iam+6*size,MPI_COMM_WORLD,&status );

            MPI_Recv ( & ( B_j.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+7*size,MPI_COMM_WORLD,&status );

            B_j.transposeIt ( 1 );

            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            BT_i.allocate ( blocksize*Drows,m+l,nonzeroes );
            MPI_Recv ( & ( BT_i.pRows[0] ),blocksize*Drows + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            MPI_Get_count(&status,MPI_INT,&count);
            BT_i.nrows=count-1;
            MPI_Recv ( & ( BT_i.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( BT_i.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );
        }
    }
    else {
        for ( i=1; i<size; ++i ) {
            // The root process receives parts of X' * T and Z' * T sequentially from all processes and directly adds them together.
            int nonzeroes;
            MPI_Recv ( &nonzeroes,1,MPI_INT,i,i,MPI_COMM_WORLD,&status );
            if(nonzeroes>0) {
                XtT_temp.allocate ( m,k,nonzeroes );
                MPI_Recv ( & ( XtT_temp.pRows[0] ),m + 1, MPI_INT,i,i+size,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( XtT_temp.pCols[0] ),nonzeroes, MPI_INT,i,i+2*size,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( XtT_temp.pData[0] ),nonzeroes, MPI_DOUBLE,i,i+3*size,MPI_COMM_WORLD,&status );

                XtT_sparse.addBCSR ( XtT_temp );
            }

            MPI_Recv ( &nonzeroes,1, MPI_INT,i,i,MPI_COMM_WORLD,&status );

            if(nonzeroes>0) {
                ZtT_temp.allocate ( l,k,nonzeroes );

                MPI_Recv ( & ( ZtT_temp.pRows[0] ),l + 1, MPI_INT,i,4*size + i,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( ZtT_temp.pCols[0] ),nonzeroes, MPI_INT,i,i+ 5*size,MPI_COMM_WORLD,&status );
                MPI_Recv ( & ( ZtT_temp.pData[0] ),nonzeroes, MPI_DOUBLE,i,i+6*size,MPI_COMM_WORLD,&status );

                ZtT_sparse.addBCSR ( ZtT_temp );
            }
        }
        XtT_sparse.transposeIt ( 1 );
        ZtT_sparse.transposeIt ( 1 );

        // B' is created by concatening blocks X'T and Z'T
        create1x2BlockMatrix ( XtT_sparse, ZtT_sparse,Btsparse );
        //Btsparse.writeToFile("BT_sparse.csr");

        // For each process row i BT_i is created which is also sent to processes in column i to become B_j.
        for ( int rowproc= *dims - 1; rowproc>= 0; --rowproc ) {
            BT_i.ncols=Btsparse.ncols;
            BT_i.nrows=0;
            BT_i.nonzeros=0;
            int Drows_rowproc;
            if (rowproc!=0) {
                Drows_rowproc= ( Dblocks - rowproc ) % *dims == 0 ? ( Dblocks- rowproc ) / *dims : ( Dblocks- rowproc ) / *dims +1;
                Drows_rowproc= Drows_rowproc<1? 1 : Drows_rowproc;
            }
            else
                Drows_rowproc=Drows;
            for ( i=0; i<Drows_rowproc; ++i ) {
                //Each process in row i can hold several blocks of contiguous rows of D for which we need the corresponding rows of B_T
                // Therefore we use the function extendrows to create BT_i (see src/tools.cpp)
                BT_i.extendrows ( Btsparse, ( i * *dims + rowproc ) * blocksize,blocksize );
            }
            for ( int colproc= ( rowproc==0 ? 1 : 0 ); colproc < * ( dims+1 ); ++colproc ) {
                int *curpos, rankproc;
                rankproc= blacs_pnum_ (&ICTXT2D, &rowproc,&colproc);

                MPI_Ssend ( & ( BT_i.nonzeros ),1, MPI_INT,rankproc,rankproc,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pRows[0] ),BT_i.nrows + 1, MPI_INT,rankproc,rankproc+size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pCols[0] ),BT_i.nonzeros, MPI_INT,rankproc,rankproc+2*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pData[0] ),BT_i.nonzeros, MPI_DOUBLE,rankproc,rankproc+3*size,MPI_COMM_WORLD );

                //printf("BT_i's sent to processor %d\n",rankproc);

                rankproc= blacs_pnum_ (&ICTXT2D, &colproc,&rowproc);
                MPI_Ssend ( & ( BT_i.nonzeros ),1, MPI_INT,rankproc,rankproc+4*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pRows[0] ),BT_i.nrows + 1, MPI_INT,rankproc,rankproc+5*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pCols[0] ),BT_i.nonzeros, MPI_INT,rankproc,rankproc+6*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( BT_i.pData[0] ),BT_i.nonzeros, MPI_DOUBLE,rankproc,rankproc+7*size,MPI_COMM_WORLD );

                //printf("B_j's sent to processor %d\n",rankproc);
            }
        }
        B_j.make ( BT_i.nrows,BT_i.ncols,BT_i.nonzeros,BT_i.pRows,BT_i.pCols,BT_i.pData );
        B_j.transposeIt ( 1 );
    }
    free ( DESCT ), free(DESCXtY), free(DESCY), free(DESCZtY);
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
                * ( temp + j  * lld_D +j ) =lambda;
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
            if (ni==0) {
                info=fseek ( fZ, ( long ) ( pcol * blocksize * ( k+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fZ, ( long ) ( blocksize * (*( dims+1 )-1) * ( k+1 ) * sizeof ( double ) ),SEEK_CUR );
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
        }
        else {									//Normal read-in of the matrix from a binary file
            if (ni==0) {
                info=fseek ( fZ, ( long ) ( pcol * blocksize * ( k+1 ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading Z file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fZ, ( long ) ( blocksize * (*( dims+1 )-1) * ( k+1 ) * sizeof ( double ) ),SEEK_CUR );
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
                info=fseek ( fZ, ( long ) ( (k - blocksize * ((pZblocks-1) * *dims + *position +1 )) * sizeof ( double ) ),SEEK_CUR );
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
            if (ni==0) {
                info=fseek ( fX, ( long ) ( pcol * blocksize *  m * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fX, ( long ) ( blocksize * (*( dims+1 )-1) * m * sizeof ( double ) ),SEEK_CUR );
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
            if (ni==0) {
                info=fseek ( fX, ( long ) ( pcol * blocksize *  m * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                    return -1;
                }
            }
            else {
                info=fseek ( fX, ( long ) ( blocksize * (*( dims+1 )-1) * m * sizeof ( double ) ),SEEK_CUR );
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
                fread ( Xblock + i*pXblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fX  );
                info=fseek ( fX, ( long ) ( (m - blocksize * ((pXblocks-1) * *dims + *position +1 )) * sizeof ( double ) ),SEEK_CUR );
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
int update_C ( int * DESCC, double * Cmat, double update) {

    int i,j, rowcur,colcur,nXblocks;

    nXblocks= m%blocksize==0 ? m/blocksize : m/blocksize +1;

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
    }
}


int set_up_AI ( double * AImat, int * DESCAI,int * DESCSOL, double * solution, int * DESCD, double * Dmat, CSRdouble &Asparse, CSRdouble &Btsparse, double sigma ) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    FILE* fT, * fY;
    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCTD, *DESCQRHS, *DESCQSOL, *DESCQDENSE;
    double *Tblock, *yblock, *Tdblock, *QRHS, *Qsol,*nrmblock, sigma_rec, *Zu, *Qdense;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;

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

    sigma_rec=1/sigma;

    // Initialisation of descriptors of different matrices

    descinit_ ( DESCT, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCY, &n, &i_one, &n, &i_one, &i_zero, &i_zero, &ICTXT2D, &n, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCTD, &i_one, &stripcols, &i_one, &stripcols, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQRHS, &ydim, &i_two, &ydim, &i_two, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQSOL, &ydim, &i_two, &ydim, &i_two, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQDENSE, &Ddim, &i_one, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
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
    yblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
    if ( yblock==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    Tdblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
    if ( Tdblock==NULL ) {
        printf ( "unable to allocate memory for Matrix Zu\n" );
        return EXIT_FAILURE;
    }
    nrmblock = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nrmblock==NULL ) {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }
    Qdense= ( double* ) calloc ( Drows*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }

    if (iam==0) {
        Zu=(double *) calloc(nstrips * stripcols,sizeof(double));
        yblock=(double *) calloc(n, sizeof(double));
        Xtsparse.loadFromFile(filenameX);
        Ztsparse.loadFromFile(filenameZ);
        mult_colsA_colsC_denseC(Ztsparse,solution+m,ydim,0,Ztsparse.ncols,0,1,Zu,n,false,1.0); //Zu
	printdense(nstrips * stripcols, 1, Zu, "Zu.txt");
        Xtsparse.transposeIt ( 1 );
        Ztsparse.transposeIt ( 1 );
        fY=fopen(filenameY,"rb");
        if ( fY==NULL ) {
            printf ( "Error opening file\n" );
            return -1;
        }
        fread ( yblock,sizeof ( double ),n,fY );
        QRHS= ( double * ) calloc ( ydim * 2,sizeof ( double ) );
        if ( QRHS==NULL ) {
            printf ( "Error in allocating memory for QRHS in root process");
            return -1;
        }
        Qsol= ( double * ) calloc ( ydim * 2,sizeof ( double ) );
        if ( Qsol==NULL ) {
            printf ( "Error in allocating memory for QRHS in root process");
            return -1;
        }
        mult_colsA_colsC_denseC(Xtsparse,yblock,n,0,n,0,1,QRHS,ydim,false,sigma_rec);		//X'y/sigma
        mult_colsA_colsC_denseC(Ztsparse,yblock,n,0,n,0,1,QRHS+m,ydim,false,sigma_rec);		//Z'y/sigma
        *nrmblock = dnrm2_ ( &n,yblock,&i_one);
        *AImat = *nrmblock * *nrmblock/sigma/sigma; 								//y'y/sigma²
        //printf("First element of AImat is: %g\n", *AImat);
    }

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
            free ( Tblock );
            free ( Tdblock );
            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
            Tdblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
            if ( Tdblock==NULL ) {
                printf ( "unable to allocate memory for Matrix Y\n" );
                return EXIT_FAILURE;
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

        pdgemm_ ( "T","N", &i_one, &stripcols,&k,&lambda, solution, &ml_plus,&i_one,DESCSOL,Tblock,&i_one,&i_one,DESCT,&d_zero,Tdblock,&i_one,&i_one,DESCTD ); //Td/gamma (in blocks)

        if(iam==0) {
            for (i=0; i<stripcols; ++i) {
                *(Tdblock+i) += *(Zu+ ni*stripcols+i) * lambda;  //(Td + Zu)/gamma (in blocks)
            }
        }
        /*char *Tdfile;
        Tdfile=(char *) calloc(100,sizeof(char));
        *Tdfile='\0';
        sprintf(Tdfile,"Tdmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(1,stripcols, Tdblock,Tdfile);*/
	
        int ystart=ni * *(dims+1) * blocksize + 1;

        pdgemm_ ( "N","N",&k,&i_one,&stripcols,&sigma_rec,Tblock,&i_one, &i_one, DESCT,yblock,&ystart,&i_one,DESCY,&d_one,QRHS,&ml_plus,&i_one,DESCQRHS ); //T'y/sigma
	
        //pdgemm_ ( "N","T",&m,&i_one,&stripcols,&sigma_rec,Xblock,&i_one, &i_one, DESCX,yblock,&i_one,&i_one,DESCY,&d_one,QRHS,&i_one,&i_one,DESCQRHS ); //X'y/sigma

        pdgemm_ ( "N","T",&k,&i_one,&stripcols,&d_one,Tblock,&i_one, &i_one, DESCT,Tdblock,&i_one,&i_one,DESCTD,&d_one,Qdense,&i_one,&i_one,DESCQDENSE ); //T'(Td+Zu)/gamma

        //pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,Tdblock,&i_one,&i_one,DESCTD,&d_one,QRHS,&i_one,&i_two,DESCQRHS ); //X'Zu/gamma

        if(iam==0) {
            mult_colsA_colsC_denseC ( Xtsparse, Tdblock, stripcols, ni*stripcols, stripcols,0, 1, QRHS+ydim, ydim, true, 1.0 ); 	//X'(Td+Zu)/gamma
            mult_colsA_colsC_denseC ( Ztsparse, Tdblock, stripcols, ni*stripcols, stripcols,0, 1, QRHS+ydim+m, ydim, true, 1.0 );	//Z'(Td+Zu)/gamma
            *nrmblock = dnrm2_ ( &stripcols,Tdblock,&i_one );							// norm (Td+Zu)/gamma
            * ( AImat + 3 ) += *nrmblock * *nrmblock;										//(Td+Zu)' * (Td +Zu)/gamma^2
        }

        blacs_barrier_(&ICTXT2D,"A");


        // Q'Q is calculated and stored directly in AI matrix (complete in every process)
        pddot_ ( &stripcols,nrmblock,Tdblock,&i_one,&i_one,DESCTD,&i_one,yblock,&ystart,&i_one,DESCY,&i_one );
        * ( AImat + 1 ) += *nrmblock /sigma;							//y'(Zu+Td)/gamma/sigma
        * ( AImat + 2 ) += *nrmblock /sigma;							//y'(Zu+Td)/gamma/sigma
        blacs_barrier_ ( &ICTXT2D,"A" );
    }
    
    pdlacpy_("A",&k,&i_one,Qdense,&i_one, &i_one, DESCQDENSE, QRHS,&ml_plus,&i_two,DESCQRHS);

    // In Qsol we calculate the solution of M * Qsol = QRHS, but we still need QRHS a bit further

    pdcopy_ ( &ydim,QRHS,&i_one,&i_two,DESCQRHS,&i_one,Qsol,&i_one,&i_two,DESCQSOL,&i_one );

    if (iam==0) {
        solveSystem(Asparse, Qsol,QRHS+ydim, 2, 1);
        mult_colsA_colsC_denseC(Btsparse,Qsol,ydim,0,Btsparse.ncols,0,1,Qsol+ydim+m+l,ydim, true,-1.0);
    }
    pdcopy_ ( &k,Qsol,&ml_plus,&i_two,DESCQSOL,&i_one,Qdense,&i_one,&i_one,DESCQSOL,&i_one );
    pdpotrs_ ( "U",&Ddim,&i_one,Dmat,&i_one,&i_one,DESCD,Qdense,&i_one,&i_one,DESCQDENSE,&info );
    if ( info!=0 ){
        printf ( "Parallel Cholesky solution for Q was unsuccesful, error returned: %d\n",info );
	return -1;
    }
    pdcopy_ ( &k,Qdense,&i_one,&i_one,DESCQDENSE,&i_one,Qsol,&ml_plus,&i_two,DESCQSOL,&i_one );

    if (iam==0) {
        Btsparse.transposeIt(1);
        mult_colsA_colsC_denseC(Btsparse,Qsol+ydim+m+l,ydim,0,Btsparse.ncols,0,1,Qsol+ydim,ydim, true,-1.0);
	double * sparse_sol=(double *) calloc(Asparse.ncols, sizeof(double));
        solveSystem(Asparse, sparse_sol,Qsol+ydim, 2, 1);
	memcpy(Qsol+ydim,sparse_sol,(m+l) * sizeof(double));
    }

    pdcopy_ ( &ydim,solution,&i_one,&i_one,DESCSOL,&i_one,Qsol,&i_one,&i_one,DESCQSOL,&i_one );
    pdscal_ ( &ydim,&sigma_rec,Qsol,&i_one,&i_one,DESCQSOL,&i_one );
    
    printdense(2,ydim,QRHS,"QRHS.txt");
    printdense(2,ydim,Qsol,"Qsol.txt");

    // AImat = (Q'Q - QRHS' * Qsol) / 2 / sigma
    
    printdense(2,2,AImat,"QQ.txt");

    pdgemm_ ( "T","N",&i_two,&i_two,&ydim,&d_negone,QRHS,&i_one,&i_one,DESCQRHS,Qsol,&i_one,&i_one,DESCQSOL,&d_one, AImat,&i_one,&i_one,DESCAI );
    
    printdense(2,2,AImat,"AI_nonorm.txt");

    for ( i=0; i<4; ++i )
        * ( AImat + i ) = * ( AImat + i ) / 2 / sigma;

    info=fclose ( fY );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    info=fclose ( fT );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    free ( DESCQRHS );
    free ( DESCQSOL );
    free ( DESCY );
    free ( DESCT );
    free ( DESCTD );
    free ( Tblock );
    free ( yblock );
    free ( nrmblock );
    free ( QRHS );
    free ( Qsol );
    free ( Tdblock );

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
                for ( j=0; j<= (m-1)%blocksize; ++j ) {
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
            if ( i< ( Dblocks -1 ) ) {
                for ( j=0; j<blocksize; ++j ) {
                    logdet_proc += log( * ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ));
                }
            } else {
                for ( j=0; j< Ddim % blocksize; ++j ) {
                    logdet_proc += log(* ( mat+ ( j + i / * ( dims+1 ) * blocksize ) * lld_D + i / *dims *blocksize +j ));
                }
            }
        }
    }
    return logdet_proc;
}
