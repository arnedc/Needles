#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "src/shared_var.h"
#include <shared_var.h>
#include <mkl_types.h>
#include <smat.h>
#include "CSRdouble.hpp"
#define MPI_SUCCESS          0      /* Successful return code */
#define MPI_CHAR           ((MPI_Datatype)0x4c000101)


typedef int MPI_Comm;
typedef int MPI_Datatype;

double d_one=1.0, d_zero=0.0, d_negone=-1.0;
int DLEN_=9, i_negone=-1, i_zero=0, i_one=1, i_two=2, i_four=4; // some many used constants
int k,n,m, l, blocksize; //dimensions of different matrices
int lld_D, Dblocks, Ddim, m_plus, lld_y, yblocks, ydim, Adim;
int Drows,Dcols,yrows,ycols;
int size, *dims, * position, ICTXT2D, ICTXT1D, ICTXT1PROC, iam;
int ntests, maxiterations,datahdf5, copyC;
char *SNPdata, *phenodata;
char *filenameT, *filenameX, *filenameZ, *filenameY, *TestSet;
double lambda, epsilon;

extern "C" {
    void blacs_pinfo_ ( int *mypnum, int *nprocs );
    void blacs_setup_ ( int *mypnum, int *nprocs );
    void blacs_get_ ( int *ConTxt, int *what, int *val );
    void blacs_gridinit_ ( int *ConTxt, char *order, int *nprow, int *npcol );
    void blacs_gridexit_ ( int *ConTxt );
    void blacs_pcoord_ ( int *ConTxt, int *nodenum, int *prow, int *pcol );
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void igebs2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, int *A, int *lda );
    void igebr2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, int *A, int *lda, int *rsrc, int *csrc );
    void dgebs2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda );
    void dgebr2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
    void dgsum2d_ ( int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rdest, int *cdest );
    void dgesd2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rdest, int *cdest );
    void dgerv2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
    void pdcopy_ ( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pddot_ ( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    double pdlansy_ ( char *norm, char *uplo, int *n, double *a, int *ia, int *ja, int *desca, double *work );
    void pdlacpy_ ( char *uplo, int *m, int *n, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb );
    void pdpotrf_ ( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pdpotri_ ( char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void dsyrk ( const char *uplo, const char *trans, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *beta, double *c, const int *ldc );
    void dgemm ( const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc );
    double  dnrm2 ( const int *n, const double *x, const int *incx );
    void    dcopy ( const int *n, const double *x, const int *incx, double *y, const int *incy );
    double  ddot ( const int *n, const double *x, const int *incx, const double *y, const int *incy );
    void dpotrf_ ( const char* uplo, const int* n, double* a, const int* lda, int* info );
    void dpotrs_ ( const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info );
    void dpotri_ ( const char* uplo, const int* n, double* a, const int* lda, int* info );
    int MPI_Init ( int *, char *** );
    int MPI_Dims_create ( int, int, int * );
    int MPI_Finalize ( void );
    int MPI_Bcast ( void*, int, MPI_Datatype, int, MPI_Comm );
    int MPI_Barrier( MPI_Comm comm );
}

int main ( int argc, char **argv ) {
    int info;
    info = MPI_Init ( &argc, &argv );
    if ( info != MPI_SUCCESS ) {
        printf ( "Error in MPI initialisation: %d",info );
        return info;
    }
    int i,j,pcol, counter, breakvar;
    double *Dmat, *Smat, *Bmat, *Btransmat,*ytot, *RHS, *respnrm, *randnrm, *AImat, *solution, *Cmatcopy;
    double sigma, dot, trace_proc, convergence_criterium, loglikelihood,prevloglike,update_loglikelihood;
    int *DESCD, *DESCYTOT, *DESCRHS, *DESCAI, *DESCCCOPY;
    double c0, c1, c2, c3, c4;
    struct timeval tz0,tz1, tz2,tz3;
    double vm_usage, resident_set, cpu_sys, cpu_user;
    double *work, normC,norm1C, norminv, norm1inv, Cmax, colmax;
    CSRdouble BT_i, B_j, Xsparse, Zsparse;

    // declaration of descriptors of different matrices

    DESCD= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCD==NULL ) {
        printf ( "unable to allocate memory for descriptor for C\n" );
        return -1;
    }
    DESCYTOT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCYTOT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Ytot\n" );
        return -1;
    }
    DESCRHS= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCRHS==NULL ) {
        printf ( "unable to allocate memory for descriptor for RHS\n" );
        return -1;
    }
    DESCAI= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCAI==NULL ) {
        printf ( "unable to allocate memory for descriptor for AI\n" );
        return -1;
    }

    //Some important parameters for the creation of the proces grid

    position= ( int* ) calloc ( 2,sizeof ( int ) );
    if ( position==NULL ) {
        printf ( "unable to allocate memory for processor position coordinate\n" );
        return EXIT_FAILURE;
    }
    dims= ( int* ) calloc ( 2,sizeof ( int ) );
    if ( dims==NULL ) {
        printf ( "unable to allocate memory for grid dimensions coordinate\n" );
        return EXIT_FAILURE;
    }

    blacs_pinfo_ ( &iam,&size ); 			//determine the number of processes involved
    blacs_setup_ ( &iam,&size );
    if ( iam ==-1 ) {
        printf ( "Error in initialisation of proces grid" );
        return -1;
    }
    info=MPI_Dims_create ( size, 2, dims );			//determine the best 2D cartesian grid with the number of processes
    if ( info != MPI_SUCCESS ) {
        printf ( "Error in MPI creation of dimensions: %d",info );
        return info;
    }
    //Until now the code can only work with square process grids
    //So we try to get the biggest square grid possible with the number of processes involved
    if (*dims != *(dims+1)) {
        while (*dims * *dims > size)
            *dims -=1;
        *(dims+1)= *dims;
        if (iam==0)
            printf("WARNING: %d processor(s) unused due to reformatting to a square process grid\n", size - (*dims * *dims));
        size = *dims * *dims;
        //cout << "New size of process grid: " << size << endl;
    }

    blacs_get_ ( &i_negone,&i_zero,&ICTXT2D );

    //Initialisation of the BLACS process grid, which is referenced as ICTXT2D
    blacs_gridinit_ ( &ICTXT2D,"R",dims, dims+1 );

    if (iam < size) {

        blacs_pcoord_ ( &ICTXT2D,&iam,position, position+1 );
        if ( *position ==-1 ) {
            printf ( "Error in proces grid" );
            return -1;
        }

        // Check of correct usage of parreml and reading of the input file by processor 0

        if ( argc !=2 ) {
            if ( * ( position+1 ) ==0 && *position==0 ) {

                printf ( "The correct use of pardiso_blup is:\n ./pardiso_blup <input_file>\n" );
                return -1;
            } else
                return -1;
        }
        info=read_input ( *++argv );
        //    igebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&info,&i_one );
        if ( info!=0 ) {
            printf ( "Something went wrong when reading input file for processor %d\n",iam );
            return -1;
        }
        blacs_barrier_ ( &ICTXT2D,"ALL" );
        if ( * ( position+1 ) ==0 && *position==0 )
            printf ( "Reading of input-file succesful\n" );

        m_plus=m+1;
	Adim=m+l;

        Ddim=k;
        pcol= * ( position+1 );
        Dblocks= Ddim%blocksize==0 ? Ddim/blocksize : Ddim/blocksize +1;	 //define number of blocks needed to store a complete column/row of C
        Drows= ( Dblocks - *position ) % *dims == 0 ? ( Dblocks- *position ) / *dims : ( Dblocks- *position ) / *dims +1;
        Drows= Drows<1? 1 : Drows;
        Dcols= ( Dblocks - pcol ) % * ( dims+1 ) == 0 ? ( Dblocks- pcol ) / * ( dims+1 ) : ( Dblocks- pcol ) / * ( dims+1 ) +1;
        Dcols=Dcols<1? 1 : Dcols;
        lld_D=Drows*blocksize;
	
	ydim=k+l+m;
        yblocks= ydim%blocksize==0 ? ydim/blocksize : ydim/blocksize +1;	 //define number of blocks needed to store a complete column/row of C
        yrows= ( yblocks - *position ) % *dims == 0 ? ( yblocks- *position ) / *dims : ( yblocks- *position ) / *dims +1;
        yrows= yrows<1? 1 : yrows;
        lld_y=yrows*blocksize;

        // Initialisation of different descriptors

        if ( copyC ) {
            DESCCCOPY= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
            if ( DESCCCOPY==NULL ) {
                printf ( "unable to allocate memory for descriptor for copy of C\n" );
                return -1;
            }
            descinit_ ( DESCCCOPY, &Ddim, &Ddim, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
            if ( info!=0 ) {
                printf ( "Descriptor of copy of matrix C returns info: %d\n",info );
                return info;
            }
            Cmatcopy= ( double* ) calloc ( Drows * blocksize * Dcols * blocksize,sizeof ( double ) );
            if ( Cmatcopy==NULL ) {
                printf ( "unable to allocate memory for copy of Matrix C (required: %dl bytes)\n", Drows * ( long ) blocksize * Dcols * ( long ) blocksize );
                return EXIT_FAILURE;
            }
        }

        /*printf("size lld_C is %d\n", sizeof(lld_C));
        printf("size Cdim is %d\n", sizeof(Cdim));
        printf("blocksize is %d\n", blocksize);
        printf("i_zero is %d\n", i_zero);*/

        descinit_ ( DESCD, &Ddim, &Ddim, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCYTOT, &ydim, &i_one, &blocksize, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_y, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of response matrix returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCRHS, &ydim, &i_one, &blocksize, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_y, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of RHS matrix returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCAI, &i_two, &i_two, &i_two, &i_two, &i_zero, &i_zero, &ICTXT2D, &i_two, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of AI matrix returns info: %d\n",info );
            return info;
        }

        convergence_criterium=0;
        counter=0;
        /*strcat ( filenameZ, ".bin" );
        strcat ( filenameX, ".bin" );
        strcat ( filenameTest, ".bin" );*/

        if ( * ( position+1 ) ==0 && *position==0 ) {
            printf ( "\nA linear mixed model with %d observations, %d random effects and %d fixed effects\n", n,k,m );
            printf ( "was analyzed using %d (%d x %d) processors\n",size,*dims,* ( dims+1 ) );
            gettimeofday ( &tz2,NULL );
            c2= tz2.tv_sec*1000000 + ( tz2.tv_usec );
            solution= ( double * ) calloc ( Dblocks * blocksize, sizeof ( double ) );
            if ( solution==NULL ) {
                printf ( "unable to allocate memory for solution matrix\n" );
                return EXIT_FAILURE;
            }
        }

        Dmat= ( double* ) calloc ( Drows * blocksize * Dcols * blocksize,sizeof ( double ) );
        if ( Dmat==NULL ) {
            printf ( "unable to allocate memory for Matrix D  (required: %ld bytes)\n", Drows * blocksize * Dcols * blocksize * sizeof ( double ) );
            return EXIT_FAILURE;
        }
        ytot = ( double* ) calloc ( yrows * blocksize,sizeof ( double ) );
        if ( ytot==NULL ) {
            printf ( "unable to allocate memory for Matrix Y (required: %d bytes)\n", yrows * blocksize*sizeof ( double ) );
            return EXIT_FAILURE;
        }
        RHS = ( double* ) calloc ( yrows * blocksize,sizeof ( double ) );
        if ( RHS==NULL ) {
            printf ( "unable to allocate memory for RHS (required: %d bytes)\n", yrows * blocksize * sizeof ( double ) );
            return EXIT_FAILURE;
        }
        AImat = ( double* ) calloc ( 2*2,sizeof ( double ) );
        if ( RHS==NULL ) {
            printf ( "unable to allocate memory for AI matrix (required: %d bytes)\n",2*2*sizeof ( double ) );
            return EXIT_FAILURE;
        }
        respnrm= ( double * ) calloc ( 1,sizeof ( double ) );
        if ( respnrm==NULL ) {
            printf ( "unable to allocate memory for norm\n" );
            return EXIT_FAILURE;
        }
        randnrm= ( double * ) calloc ( 1,sizeof ( double ) );
        if ( randnrm==NULL ) {
            printf ( "unable to allocate memory for norm\n" );
            return EXIT_FAILURE;
        }

// Start of AI-REML iteration cycle, stops when relative update of gamma < epsilon

        while ( fabs ( convergence_criterium ) >epsilon || fabs ( update_loglikelihood/loglikelihood ) > epsilon || counter<2 ) {
            ++counter;

            // Number of iterations is limited to $maxiterations to avoid divergence of algorithm

            if ( counter > maxiterations ) {
                if ( * ( position+1 ) ==0 && *position==0 ) {
                    printf ( "maximum number of iterations reached, AI-REML has not converged\n" );
                }
                break;
            }

            if ( * ( position+1 ) ==0 && *position==0 ) {
                printf ( "\nParallel results: loop %d\n",counter );
                printf ( "=========================\n" );
                gettimeofday ( &tz3,NULL );
                c3= tz3.tv_sec*1000000 + ( tz3.tv_usec );
            }

            // Since after every iteration the inverse of C is stored in Cmat and C must be updated with the new lambda,
            // every cycle C has to be set up again from scratch
            // Maybe it would be better not to free and allocate every time but just set every element to zero (for loop)??

            if ( counter > 1 ) {
                free ( ytot );
                //free ( Cmat );
                free ( AImat );

                Dmat= ( double* ) calloc ( Drows*blocksize*Dcols*blocksize,sizeof ( double ) );
                if ( Dmat==NULL ) {
                    printf ( "unable to allocate memory for Matrix C (required: %d bytes)\n", Drows*blocksize*Dcols*blocksize*sizeof ( double ) );
                    return EXIT_FAILURE;
                }
                ytot = ( double* ) calloc ( yrows * blocksize,sizeof ( double ) );
                if ( ytot==NULL ) {
                    printf ( "unable to allocate memory for Matrix Y\n" );
                    return EXIT_FAILURE;
                }
                AImat = ( double* ) calloc ( 2*2,sizeof ( double ) );
                if ( RHS==NULL ) {
                    printf ( "unable to allocate memory for AI matrix\n" );
                    return EXIT_FAILURE;
                }
                blacs_barrier_ ( &ICTXT2D,"A" );

                if ( * ( position+1 ) ==0 && *position==0 ) {
                    gettimeofday ( &tz1,NULL );
                    c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                    printf ( "\t elapsed wall time allocation of memory:		%10.3f s\n", ( c1 - c3 ) /1000000.0 );
                }

                if ( copyC ) {
                    update_C ( DESCCCOPY,Cmatcopy,lambda/ ( 1+convergence_criterium ) - lambda );
                    pdlacpy_ ( "U", &Ddim, &Ddim, Cmatcopy, &i_one, &i_one, DESCCCOPY, Dmat, &i_one, &i_one, DESCD );
                    pdcopy_ ( &Ddim, RHS,&i_one,&i_one,DESCRHS,&i_one,ytot,&i_one,&i_one,DESCYTOT,&i_one );
                    if ( * ( position+1 ) ==0 && *position==0 ) {
                        gettimeofday ( &tz1,NULL );
                        c0= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                        printf ( "\t elapsed wall time copy of Y and C:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                    }
                    lambda=lambda/ ( 1+convergence_criterium ); // Update for lambda (which is 1/gamma)
                } else {
                    free ( respnrm );
                    respnrm= ( double * ) calloc ( 1,sizeof ( double ) );
                    if ( respnrm==NULL ) {
                        printf ( "unable to allocate memory for norm\n" );
                        return EXIT_FAILURE;
                    }
                    lambda=lambda/ ( 1+convergence_criterium ); // Update for lambda (which is 1/gamma)
                    if ( datahdf5 )
                        info = set_up_C_hdf5 ( DESCD, Dmat, DESCYTOT, ytot, respnrm );
                    else
			info = set_up_BDY ( DESCD, Dmat, BT_i, B_j, DESCYTOT, ytot, respnrm );
                    if ( info!=0 ) {
                        printf ( "Something went wrong with set-up of matrix D, error nr: %d\n",info );
                        return info;
                    }
                    if ( * ( position+1 ) ==0 && *position==0 ) {
                        gettimeofday ( &tz0,NULL );
                        c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                        printf ( "\t elapsed wall time set-up of C, B, S and Y:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                    }

                }
            } else {
                if ( * ( position+1 ) ==0 && *position==0 ) {
                    gettimeofday ( &tz1,NULL );
                    c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                    printf ( "\t elapsed wall time allocation of memory:		%10.3f s\n", ( c1 - c3 ) /1000000.0 );
                }
                if ( datahdf5 )
                    info = set_up_C_hdf5 ( DESCD, Dmat, DESCYTOT, ytot, respnrm );
                else
                    info = set_up_BDY ( DESCD, Dmat, BT_i, B_j, DESCYTOT, ytot, respnrm );
                if ( info!=0 ) {
                    printf ( "Something went wrong with set-up of matrix D, error nr: %d\n",info );
                    return info;
                }
                if ( * ( position+1 ) ==0 && *position==0 ) {
                    gettimeofday ( &tz0,NULL );
                    c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                    printf ( "\t elapsed wall time set-up of D, B and Y:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                }

                // RHS is copied for use afterwards (in ytot we will get the estimates for the effects)
                // This only needs to be done the first time
                pdcopy_ ( &Ddim, ytot,&i_one,&i_one,DESCYTOT,&i_one,RHS,&i_one,&i_one,DESCRHS,&i_one );
                if ( copyC )
                    pdlacpy_ ( "U", &Ddim, &Ddim, Dmat, &i_one, &i_one, DESCD, Cmatcopy, &i_one, &i_one, DESCCCOPY );

                if ( * ( position+1 ) ==0 && *position==0 ) {
                    gettimeofday ( &tz1,NULL );
                    c4= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                    printf ( "\t elapsed wall time copy of Y (and C):			%10.3f s\n", ( c4 - c0 ) /1000000.0 );
                }
            }

            // Calculation of Frobenius norm of C

            work= ( double * ) calloc ( 2* ( long ) blocksize* ( Dcols+Drows ),sizeof ( double ) );
            if ( work==NULL ) {
                printf ( "unable to allocate memory for work (norm)\n" );
                return EXIT_FAILURE;
            }
            normC=pdlansy_ ( "F","U",&Ddim,Dmat,&i_one,&i_one,DESCD,work );
            norm1C=pdlansy_ ( "1","U",&Ddim,Dmat,&i_one,&i_one,DESCD,work );
            Cmax=pdlansy_ ( "M","U",&Ddim,Dmat,&i_one, &i_one, DESCD,work );
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c1= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time norm and max of C:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
                printf ( "The new parallel lambda is: %15.10g\n",lambda );
                printf ( "norm of y-vector is: %g\n",*respnrm );
            }

            //Now every matrix has to set up the sparse matrix A, consisting of X'X, X'Z, Z'X and Z'Z + lambda*I
        Xsparse.loadFromFile ( filenameX );
        Zsparse.loadFromFile ( filenameZ );

        smat_t *X_smat, *Z_smat;

        X_smat = smat_new_from ( Xsparse.nrows,Xsparse.ncols,Xsparse.pRows,Xsparse.pCols,Xsparse.pData,0,0 );
        Z_smat = smat_new_from ( Zsparse.nrows,Zsparse.ncols,Zsparse.pRows,Zsparse.pCols,Zsparse.pData,0,0 );

        smat_t *Xt_smat, *Zt_smat;
        Xt_smat = smat_copy_trans ( X_smat );
        Zt_smat = smat_copy_trans ( Z_smat );

        CSRdouble Asparse;
        smat_t *XtX_smat, *XtZ_smat, *ZtZ_smat, *lambda_smat, *ZtZlambda_smat;

        XtX_smat = smat_matmul ( Xt_smat, X_smat );
        XtZ_smat = smat_matmul ( Xt_smat, Z_smat );
        ZtZ_smat = smat_matmul ( Zt_smat,Z_smat );

        /*smat_free(X_smat);
        smat_free(Z_smat);
        smat_free(Xt_smat);
        smat_free(Zt_smat);
        Xsparse.clear();
        Zsparse.clear();*/

        CSRdouble Imat;

        makeIdentity ( l, Imat );

        lambda_smat = smat_new_from ( Imat.nrows,Imat.ncols,Imat.pRows,Imat.pCols,Imat.pData,0,0 );

        smat_scale_diag ( lambda_smat, -lambda );

        ZtZlambda_smat = smat_add ( lambda_smat, ZtZ_smat );

        /*smat_free(ZtZ_smat);
        smat_free(lambda_smat);
        Imat.clear();*/

        smat_to_symmetric_structure ( XtX_smat );
        smat_to_symmetric_structure ( ZtZlambda_smat );

        CSRdouble XtX_sparse, XtZ_sparse, ZtZ_sparse;

        XtX_sparse.make ( XtX_smat->m,XtX_smat->n,XtX_smat->nnz,XtX_smat->ia,XtX_smat->ja,XtX_smat->a );
        XtZ_sparse.make ( XtZ_smat->m,XtZ_smat->n,XtZ_smat->nnz,XtZ_smat->ia,XtZ_smat->ja,XtZ_smat->a );
        ZtZ_sparse.make ( ZtZlambda_smat->m,ZtZlambda_smat->n,ZtZlambda_smat->nnz,ZtZlambda_smat->ia,ZtZlambda_smat->ja,ZtZlambda_smat->a );

        if (iam==0) {
            cout << "***                                           [  t     t  ] *** " << endl;
            cout << "***                                           [ X X   X Z ] *** " << endl;
            cout << "***                                           [           ] *** " << endl;
            cout << "*** G e n e r a t i n g    m a t r i x    A = [           ] *** " << endl;
            cout << "***                                           [  t     t  ] *** " << endl;
            cout << "***                                           [ Z X   Z Z ] *** " << endl;
        }

        //Sparse matrix A only contains the upper triangular part of A
        create2x2SymBlockMatrix ( XtX_sparse, XtZ_sparse, ZtZ_sparse, Asparse );
	if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time for creating sparse matrix A:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            }

        /*smat_free(XtX_smat);
        smat_free(XtZ_smat);
        smat_free(ZtZlambda_smat);
        XtX_sparse.clear();
        XtZ_sparse.clear();
        ZtZ_sparse.clear();*/

        blacs_barrier_ ( &ICTXT2D,"ALL" );

        //AB_sol will contain the solution of A*X=B, distributed across the process rows. Processes in the same process row possess the same part of AB_sol
        double * AB_sol;
        int * DESCAB_sol;
        DESCAB_sol= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCAB_sol==NULL ) {
            printf ( "unable to allocate memory for descriptor for AB_sol\n" );
            return -1;
        }
        //AB_sol (Adim, Ddim) is distributed across all processes in ICTXT2D starting from process (0,0) into blocks of size (Adim, blocksize)
        descinit_ ( DESCAB_sol, &Adim, &Ddim, &Adim, &blocksize, &i_zero, &i_zero, &ICTXT2D, &Adim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }

        AB_sol=(double *) calloc(Adim * Dcols*blocksize,sizeof(double));

        // Each process calculates the Schur complement of the part of D at its disposal. (see src/schur.cpp)
        // The solution of A * Y = B_j is stored in AB_sol (= A^-1 * B_j)
        make_Sij_parallel_denseB ( Asparse, BT_i, B_j, D, lld_D, AB_sol );
	if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time for creating Schur complement of D:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            }
	
	//From here on the Schur complement S of D is stored in D

        blacs_barrier_ ( &ICTXT2D,"ALL" );

            pdpotrf_ ( "U",&Ddim,Dmat,&i_one, &i_one,DESCD,&info );
            if ( info!=0 ) {
                printf ( "Parallel Cholesky decomposition of D was unsuccesful, error returned: %d\n",info );
                return -1;
            }
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time Cholesky decomposition of C:		%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            }

            //Estimation of fixed and random effects, stored in ytot distributed among all processors

            pdpotrs_ ( "U",&Ddim,&i_one,Dmat,&i_one,&i_one,DESCD,ytot,&i_one,&i_one,DESCYTOT,&info );
            if ( info!=0 )
                printf ( "Parallel Cholesky solution was unsuccesful, error returned: %d\n",info );
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time estimation of effects:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            }

            // sigma is updated based on estimations of effects and square of norm of y (y'y)

            pddot_ ( &Ddim,&dot,RHS,&i_one,&i_one,DESCRHS,&i_one,ytot,&i_one,&i_one,DESCYTOT,&i_one );

            sigma= ( *respnrm - dot ) / ( n-m );
            if ( * ( position+1 ) ==0 && *position==0 ) {
                dgebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&sigma,&i_one );
            } else
                dgebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &sigma,&i_one,&i_zero,&i_zero );

            loglikelihood=log_determinant_C ( Dmat,DESCD );
            dgsum2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&loglikelihood,&i_one,&i_negone,&i_negone );

            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time calculation and sending of sigma and log(det(C)):	%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            }

            // Average information matrix is set up using sigma, lambda and estimation of variable effects

            if ( datahdf5 )
                info = set_up_AI_hdf5 ( AImat, DESCAI,DESCYTOT, ytot, DESCD, Dmat, sigma ) ;
            else
                info = set_up_AI ( AImat, DESCAI,DESCYTOT, ytot, DESCD, Dmat, sigma ) ;

            if ( info!=0 ) {
                printf ( "Something went wrong with set-up of AI-matrix, error nr: %d\n",info );
                return EXIT_FAILURE;
            }

            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time set up of AI matrix:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            }

            // Inverse of C is calculated for use in scoring function

            pdpotri_ ( "U",&Ddim,Dmat,&i_one,&i_one,DESCD,&info );
            if ( info!=0 )
                printf ( "Parallel Cholesky inverse was unsuccesful, error returned: %d\n",info );
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time inverse of C:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            }


            norminv=pdlansy_ ( "F","U",&Ddim,Dmat,&i_one,&i_one,DESCD,work );
            norm1inv=pdlansy_ ( "1","U",&Ddim,Dmat,&i_one,&i_one,DESCD,work );
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time set norm of inverse of C:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            }

            free ( work );

            // The trace of the inverse of C is calculated per diagonal block and then summed over all processes and stored in proces (0,0)

            trace_proc=trace_CZZ ( Dmat,DESCD );

            free ( Dmat );

            dgsum2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&trace_proc,&i_one,&i_negone,&i_negone );

            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time trace of inverse of C:		%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            }

            // The norm of the estimation of the random effects is calculated for use in the score function

            pdnrm2_ ( &k,randnrm,ytot,&m_plus,&i_one,DESCYTOT,&i_one );

            // The score function (first derivative of log likelihood) and the update for lambda are only calculated in proces (0,0)
            // Afterwards the update is sent to every proces.

            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time set norm of estimation of u:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                double *score;
                printf ( "dot product = %15.10g \n",dot );
                printf ( "parallel sigma = %15.10g\n",sigma );
                printf ( "The trace of CZZ is: %15.10g \n",trace_proc );
                printf ( "The norm of the estimation of u is: %g \n",*randnrm );
                loglikelihood += ( k * log ( 1/lambda ) + ( n-m ) * log ( sigma ) + n-m ) /2;
                loglikelihood *= -1.0;


                score= ( double * ) calloc ( 2,sizeof ( double ) );
                if ( score==NULL ) {
                    printf ( "unable to allocate memory for score function\n" );
                    return EXIT_FAILURE;
                }
                * ( score+1 ) = - ( k-trace_proc*lambda- *randnrm * *randnrm * lambda / sigma ) * lambda / 2;
                printf ( "The score function is: %g\n",* ( score+1 ) );
                printdense ( 2,2, AImat, "AI_par.txt" );
                breakvar=0;
                if ( fabs ( * ( score+1 ) ) < epsilon * epsilon ) {
                    printf ( "Score function too close to zero to go further, solution may not have converged\n " );
                    breakvar=1;
                    igebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&breakvar,&i_one );
                    break;
                }
                igebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&breakvar,&i_one );

                //Use of damping factor

                if ( counter==1 ) {
                    trace_proc= *AImat + * ( AImat+3 );
                    //damping=trace_proc/m;
                    prevloglike=loglikelihood;
                    printf ( "The loglikelihood is: %g\n",loglikelihood );

                } else {
                    /*if (loglikelihood<prevloglike)
                        damping /=2;
                    else
                        damping *=10;*/
                    update_loglikelihood=loglikelihood-prevloglike;
                    prevloglike=loglikelihood;
                    printf ( "The update for the loglikelihood is: %g\n",update_loglikelihood );
                    printf ( "The new loglikelihood is: %g\n",loglikelihood );
                }
                dgebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&update_loglikelihood,&i_one );
                /*
                            *AImat += damping;
                            *(AImat+1) += damping;

                            printf("Used damping factor is %g\n",damping);
                */
                dpotrf_ ( "U", &i_two, AImat, &i_two, &info );
                if ( info!=0 ) {
                    printf ( "Cholesky decomposition of AI matrix was unsuccesful, error returned: %d\n",info );
                    return -1;
                }
                dpotrs_ ( "U",&i_two,&i_one,AImat,&i_two,score,&i_two,&info );
                if ( info!=0 ) {
                    printf ( "Parallel solution for AI matrix was unsuccesful, error returned: %d\n",info );
                    return -1;
                }
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time update for lambda:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
                printf ( "The update for gamma is: %g \n", * ( score+1 ) );
                while ( * ( score+1 ) +1/lambda <0 ) {
                    * ( score+1 ) =* ( score+1 ) /2;
                    printf ( "Half a step is used to avoid negative gamma\n" );
                }
                convergence_criterium=lambda * * ( score+1 );
                dgebs2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&convergence_criterium,&i_one );
                free ( score );
                printf ( "The eventual relative update for gamma is: %g \n", convergence_criterium );

            } else {
                igebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &breakvar,&i_one,&i_zero,&i_zero );
                if ( breakvar >0 ) {
                    break;
                }
                dgebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &update_loglikelihood,&i_one,&i_zero,&i_zero );
                dgebr2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one, &convergence_criterium,&i_one,&i_zero,&i_zero );
            }
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time sending and receiving update lambda:	%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                printf ( "\t elapsed wall time iteration loop %d:			%10.3f s\n", counter, ( c0 - c3 ) /1000000.0 );
            }

        }
        blacs_barrier_ ( &ICTXT2D,"A" );


        free ( AImat );
        free ( RHS );
        free ( respnrm );
        free ( randnrm );
        free ( DESCD );
        free ( DESCAI );
        free ( DESCRHS );

        if ( copyC ) {
            free ( Cmatcopy );
            free ( DESCCCOPY );
        }

        if ( ntests>0 ) {
            if ( datahdf5 )
                info=crossvalidate_hdf5 ( ytot,DESCYTOT );
            else
                info=crossvalidate ( ytot, DESCYTOT );
            if ( info!=0 ) {
                printf ( "Cross-validation was unsuccesful, error returned: %d\n",info );
                return -1;
            }
        }


        //Set of equations is solved. Solution matrix is distributed across processes and needs to be sent back to proces 0

        if ( * ( position+1 ) ==0 ) {
            for ( i=0,j=0; i<Dblocks; ++i,++j ) {
                if ( j==*dims )
                    j=0;
                if ( *position==j ) {
                    dgesd2d_ ( &ICTXT2D,&blocksize,&i_one,ytot+ i / *dims *blocksize,&blocksize,&i_zero,&i_zero );
                }
                if ( *position==0 ) {
                    dgerv2d_ ( &ICTXT2D,&blocksize,&i_one,solution+blocksize*i,&blocksize,&j,&i_zero );
                }
            }
        }

        blacs_barrier_ ( &ICTXT2D, "A" );

        if ( * ( position+1 ) ==0 && *position==0 ) {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\n\tOverall results:\n" );
            printf ( "\t================\n" );
            printf ( "\tThe maximum element in C is:          %10.5f\n", Cmax );
            printf ( "\tThe Frobenius norm of C is:           %15.10e\n", normC );
            printf ( "\tThe 1-norm of C is:                   %15.10e\n", norm1C );
            printf ( "\tThe Frobenius norm of Cinv is:        %15.10e\n", norminv );
            printf ( "\tThe 1-norm of Cinv is:                %15.10e\n", norm1inv );
            printf ( "\tThe Frobenius condition number is:    %15.10e\n", norminv*normC );
            printf ( "\tThe condition number (1-norm) is:     %15.10e\n", norm1inv*norm1C );
            printf ( "\tThe accuracy is:                      %15.10e\n", norminv*normC*Cmax/pow ( 2,53 ) );
            printf ( "\tThe ultimate lambda is:               %15.10g\n",lambda );
            printf ( "\tThe ultimate sigma is:                %15.10g\n", sigma );

            printf ( "\telapsed total wall time:              %10.3f s\n", ( c0 - c2 ) /1000000.0 );

            printf ( "\tProcessor: %d \n\t ========================\n", iam );
            printf ( "\tVirtual memory used:                  %10.0f kb\n", vm_usage );
            printf ( "\tResident set size:                    %10.0f kb\n", resident_set );
            printf ( "\tCPU time (user):                      %10.3f s\n", cpu_user );
            printf ( "\tCPU time (system):                    %10.3f s\n", cpu_sys );
            printdense ( Ddim,1,solution,"solution.txt" );
            free ( solution );
        }

        /*char *srank, filen[50];
        srank= ( char* ) calloc ( 10,sizeof ( char ) );
        filen[0]='\0';
        sprintf ( srank,"%d.txt",iam );
        strcat ( filen,"matrix" );
        strcat ( filen,srank );
        free ( srank );

        printdense ( Crows * blocksize,1, ytot, filen );*/
//printdense ( 2,2, AImat, filen );
//printdense (blocksize,pblocks * blocksize, Zblock, filen );

        free ( DESCYTOT );
        free ( position ),free ( dims );
        free ( filenameT );
        free ( filenameX );
        free ( TestSet );
        free ( ytot );
        free ( SNPdata );
        free ( phenodata );

        blacs_barrier_ ( &ICTXT2D, "A" );
        blacs_gridexit_ ( &ICTXT2D );

    }
    //cout << iam << " reached end before MPI_Barrier" << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}

