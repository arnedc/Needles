#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "src/shared_var.h"

#include "CSRdouble.hpp"
#include "ParDiSO.hpp"
#define MPI_SUCCESS          0      /* Successful return code */
#define MPI_CHAR           ((MPI_Datatype)0x4c000101)


typedef int MPI_Comm;
typedef int MPI_Datatype;

double d_one=1.0, d_zero=0.0, d_negone=-1.0;
int DLEN_=9, i_negone=-1, i_zero=0, i_one=1, i_two=2, i_three=3; // some many used constants
int k,l,m, n, blocksize; //dimensions of different matrices
int lld_D, Dblocks, Ddim, m_plus, ml_plus, ydim, Adim;
int Drows,Dcols;
int size, *dims, * position, ICTXT2D, iam;
int ntests, maxiterations,datahdf5, copyC;
char *SNPdata, *phenodata;
char *filenameX, *filenameT, *filenameZ, *filenameY, *TestSet;
double gamma_var, phi, epsilon;
int Bassparse_bool;
ParDiSO pardiso_var ( -2,0 );
ofstream rootout, clustout;



extern "C" {
    void blacs_pinfo_ ( int *mypnum, int *nprocs );
    void blacs_setup_ ( int *mypnum, int *nprocs );
    void blacs_get_ ( int *ConTxt, int *what, int *val );
    void blacs_gridinit_ ( int *ConTxt, char *order, int *nprow, int *npcol );
    void blacs_gridinfo_ ( int *ConTxt, int *nprow, int *npcol, int *myrow, int *mycol );
    void blacs_gridexit_ ( int *ConTxt );
    void blacs_pcoord_ ( int *ConTxt, int *nodenum, int *prow, int *pcol );
    void blacs_gridmap_ ( int *ConTxt, int *usermap, int *ldup, int *nprow0, int *npcol0 );
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
    void pdsymm_ ( char *side, char *uplo, int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb,
                   int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void dpotrf_ ( const char* uplo, const int* n, double* a, const int* lda, int* info );
    void dpotrs_ ( const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info );
    double  ddot_ ( const int *n, const double *x, const int *incx, const double *y, const int *incy );
    double dnrm2_ ( int *n, double *x, int *incx );
    int MPI_Init ( int *, char *** );
    int MPI_Dims_create ( int, int, int * );
    int MPI_Finalize ( void );
    int MPI_Barrier ( MPI_Comm comm );
}

int main ( int argc, char **argv ) {
    int info;
    info = MPI_Init ( &argc, &argv );
    if ( info != MPI_SUCCESS ) {
        printf ( "Error in MPI initialisation: %d",info );
        return info;
    }

    int i,j,pcol, counter, breakvar;
    double *Dmat, *ytot, *respnrm, *randnrm, *AImat, *solution, *Cmatcopy, *densesol, *AB_sol, *YSrow, *Bmat;
    double sigma, dot, trace_ZZ, trace_TT, *convergence_criterium, loglikelihood,prevloglike,update_loglikelihood;
    int *DESCD, *DESCYTOT, *DESCCCOPY, *DESCSOL, *DESCDENSESOL, *DESCAB_sol, *DESCYSROW, *DESCB, *DESCSPARSESOL;
    double vm_usage, resident_set, cpu_sys, cpu_user;
    double c0, c1, c2, c3, c4;
    struct timeval tz0,tz1, tz2,tz3;
    double *work, normC,norm1C, norminv, norm1inv, Cmax, colmax;
    CSRdouble Xsparse, Zsparse, Asparse;
    CSRdouble XtX_sparse, XtZ_sparse, ZtZ_sparse, Diagmat;
    timing secs;
    double totalTime, interTime;
    int * gridmap;
    size_t D_elements, B_elements;
    MPI_Status status;

    rootout.open ( "root_output.txt" );
    clustout.open ( "cluster_output.txt" );

    // declaration of descriptors of different matrices
    secs.tick ( totalTime );

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
    convergence_criterium= ( double * ) calloc ( 2,sizeof ( double ) );
    if ( convergence_criterium==NULL ) {
        printf ( "unable to allocate memory for convergence criterium\n" );
        return EXIT_FAILURE;
    }

    blacs_pinfo_ ( &iam,&size ); 			//determine the number of processes involved
    blacs_setup_ ( &iam,&size );
    if ( iam ==-1 ) {
        printf ( "Error in initialisation of proces grid" );
        return -1;
    }
    /*    info=MPI_Dims_create ( size, 2, dims );			//determine the best 2D cartesian grid with the number of processes
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
    */
    blacs_get_ ( &i_negone,&i_zero,&ICTXT2D );

    //Initialisation of the BLACS process grid, which is referenced as ICTXT2D
    //blacs_gridinit_ ( &ICTXT2D,"R",dims, dims+1  );

    gridmap=new int [size-1];
    for ( i=1; i < size; ++i ) {
        gridmap[i-1]=i;
    }
    *dims=1;
    * ( dims+1 ) =size -1;
    blacs_gridmap_ ( &ICTXT2D,gridmap,&i_one,dims,dims+1 );

    delete [] gridmap;

    if ( iam==0 ) {
        cout << "size of size_t: " << sizeof ( size_t ) << endl;
        cout << "size of int: " << sizeof ( int ) << endl;
    }

    int nprow, npcol;

    blacs_gridinfo_ ( &ICTXT2D, &nprow, &npcol, position, position+1 );
    //blacs_pcoord_ ( &ICTXT2D,&iam,position, position+1 );
    //cout << "Process " << iam << ": ( " << *position << ", " << * ( position+1 ) << ") in process grid." << endl;
    //blacs_barrier_(&ICTXT2D,"A");
    MPI_Barrier ( MPI_COMM_WORLD );
    /*if ( *position == -1 ) {
        printf ( "Error in process grid" );
        return -1;
    }*/

    // Check of correct usage of parreml and reading of the input file by processor 0

    if ( argc !=2 ) {
        if ( iam==0 ) {

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
    MPI_Barrier ( MPI_COMM_WORLD );
    if ( * ( position+1 ) ==0 && *position==0 )
        printf ( "Reading of input-file succesful\n" );

    m_plus=m+1;
    Adim=m+l;
    ml_plus=m+l+1;

    Ddim=k;
    pcol= * ( position+1 );
    Dblocks= Ddim%blocksize==0 ? Ddim/blocksize : Ddim/blocksize +1;	 //define number of blocks needed to store a complete column/row of C
    Drows= Ddim;
    Dcols= ( Dblocks - pcol ) % * ( dims+1 ) == 0 ? ( Dblocks- pcol ) / * ( dims+1 ) : ( Dblocks- pcol ) / * ( dims+1 ) +1;
    Dcols=Dcols<1? 1 : Dcols;
    lld_D=Ddim;

    ydim=k+l+m;
    /*yblocks= ydim%blocksize==0 ? ydim/blocksize : ydim/blocksize +1;	 //define number of blocks needed to store a complete column/row of C
    yrows= ( yblocks - *position ) % *dims == 0 ? ( yblocks- *position ) / *dims : ( yblocks- *position ) / *dims +1;
    yrows= yrows<1? 1 : yrows;*/

    // Initialisation of different descriptors

    /*if ( copyC ) {
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
    }*/

    /*printf("size lld_C is %d\n", sizeof(lld_C));
    printf("size Cdim is %d\n", sizeof(Cdim));
    printf("blocksize is %d\n", blocksize);
    printf("i_zero is %d\n", i_zero);*/

    if ( iam!=0 ) {

        DESCD= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCD==NULL ) {
            printf ( "unable to allocate memory for descriptor for C\n" );
            return -1;
        }
        DESCB= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCD==NULL ) {
            printf ( "unable to allocate memory for descriptor for C\n" );
            return -1;
        }
        DESCYTOT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCYTOT==NULL ) {
            printf ( "unable to allocate memory for descriptor for Ytot\n" );
            return -1;
        }
        DESCSOL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCSOL==NULL ) {
            printf ( "unable to allocate memory for descriptor for AI\n" );
            return -1;
        }
        DESCDENSESOL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCDENSESOL==NULL ) {
            printf ( "unable to allocate memory for descriptor for AI\n" );
            return -1;
        }
        DESCSPARSESOL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCSPARSESOL==NULL ) {
            printf ( "unable to allocate memory for descriptor for AI\n" );
            return -1;
        }
        //AB_sol will contain the solution of A*X=B, distributed across the process rows. Processes in the same process row possess the same part of AB_sol
        DESCAB_sol= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCAB_sol==NULL ) {
            printf ( "unable to allocate memory for descriptor for AB_sol\n" );
            return -1;
        }
        DESCYSROW= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
        if ( DESCYSROW==NULL ) {
            printf ( "unable to allocate memory for descriptor for AB_sol\n" );
            return -1;
        }

        descinit_ ( DESCD, &Ddim, &Ddim, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCB, &Adim, &Ddim, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &Adim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCDENSESOL, &k, &i_one, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of Dense solution returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCSPARSESOL, &Adim, &i_one, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &Adim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of Dense solution returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCYTOT, &ydim, &i_one, &ydim, &i_one, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of response matrix returns info: %d\n",info );
            return info;
        }
        descinit_ ( DESCSOL, &ydim, &i_one, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &ydim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of solution returns info: %d\n",info );
            return info;
        }
        //AB_sol (Adim, Ddim) is distributed across all processes in ICTXT2D starting from process (0,0) into blocks of size (Adim, blocksize)
        descinit_ ( DESCAB_sol, &Adim, &Ddim, &Adim, &blocksize, &i_zero, &i_zero, &ICTXT2D, &Adim, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }

        //YSrow (1,Ddim) is distributed across processes of ICTXT2D starting from process (0,0) into blocks of size (1,blocksize)
        descinit_ ( DESCYSROW, &i_one, &Ddim, &i_one,&blocksize, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
        if ( info!=0 ) {
            printf ( "Descriptor of matrix C returns info: %d\n",info );
            return info;
        }

        /*strcat ( filenameZ, ".bin" );
        strcat ( filenameX, ".bin" );
        strcat ( filenameTest, ".bin" );*/

        //solution= ( double * ) calloc ( (k+l+m), sizeof ( double ) );

        if ( * ( position+1 ) ==0 && *position==0 ) {
            printf ( "\nA linear mixed model with %d observations, %d random effects, %d SNP effects and %d fixed effects\n", n,l,k,m );
            printf ( "was analyzed using %d (%d x %d) processors\n",size,1, size );
            gettimeofday ( &tz2,NULL );
            c2= tz2.tv_sec*1000000 + ( tz2.tv_usec );
        }

        D_elements= ( size_t ) Drows * ( size_t ) Dcols * ( size_t ) blocksize;

        Dmat= ( double* ) calloc ( D_elements,sizeof ( double ) );
        if ( Dmat==NULL ) {
            printf ( "unable to allocate memory for Matrix D  (required: %lld bytes)\n",D_elements * sizeof ( double ) );
            return EXIT_FAILURE;
        }
        B_elements= ( size_t ) Adim * ( size_t ) Dcols * ( size_t ) blocksize;

        Bmat= ( double* ) calloc ( B_elements,sizeof ( double ) );
        if ( Bmat==NULL ) {
            printf ( "unable to allocate memory for Matrix B  (required: %lld bytes)\n",B_elements * sizeof ( double ) );
            return EXIT_FAILURE;
        }
        size_t densesol_elements= ( size_t ) Drows ;
        densesol = ( double * ) calloc ( densesol_elements,sizeof ( double ) );
        if ( densesol==NULL ) {
            printf ( "unable to allocate memory for distributed solution matrix (required: %lld bytes)\n", densesol_elements*sizeof ( double ) );
            return EXIT_FAILURE;
        }
        if ( * ( position+1 ) ==0 ) {
            ytot = ( double* ) calloc ( ydim,sizeof ( double ) );
            if ( ytot==NULL ) {
                printf ( "unable to allocate memory for Matrix Y (required: %d bytes)\n", ydim*sizeof ( double ) );
                return EXIT_FAILURE;
            }
            solution= ( double * ) calloc ( ( k+l+m ), sizeof ( double ) );
            if ( solution==NULL ) {
                printf ( "unable to allocate memory for solution matrix\n" );
                return EXIT_FAILURE;
            }
        }

        respnrm= ( double * ) calloc ( 1,sizeof ( double ) );
        if ( respnrm==NULL ) {
            printf ( "unable to allocate memory for norm\n" );
            return EXIT_FAILURE;
        }

        if ( * ( position +1 ) ==0 ) {
            process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            clustout << "Before allocation of AB_sol" << endl;
            clustout << "===========================" << endl ;
            clustout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            clustout << "Resident set size:    " << resident_set << " kb" << endl;
            clustout << "CPU time (user):      " << cpu_user << " s"<< endl;
            clustout << "CPU time (system):    " << cpu_sys << " s" << endl;
        }
        AB_sol= ( double * ) calloc ( B_elements,sizeof ( double ) );
        if ( AB_sol==NULL ) {
            printf ( "unable to allocate memory for AB_sol (required %lld bytes)\n",B_elements*sizeof ( double ) );
            return EXIT_FAILURE;
        }
        // To minimize memory usage, and because only the diagonal elements of the inverse are needed, Y' * S is calculated row by rowblocks
        // the diagonal element is calculated as the dot product of this row and the corresponding column of Y. (Y is solution of AY=B)
        size_t YS_elements= ( size_t ) Dcols * ( size_t ) blocksize;
        YSrow= ( double* ) calloc ( YS_elements,sizeof ( double ) );
        if ( YSrow==NULL ) {
            printf ( "unable to allocate memory for YSrow (required %lld bytes)\n",YS_elements*sizeof ( double ) );
            return EXIT_FAILURE;
        }
        if ( * ( position+1 ) ==0 && *position==0 ) {
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            //printf ( "\t elapsed wall time allocation of memory:		%10.3f s\n", ( c1 - c3 ) /1000000.0 );
        }
        if ( datahdf5 )
            info = set_up_C_hdf5 ( DESCD, Dmat, DESCYTOT, ytot, respnrm );
        else
            info = set_up_BDY ( DESCD, Dmat, DESCB, Bmat, DESCYTOT, ytot, respnrm );
        if ( info!=0 ) {
            printf ( "Something went wrong with set-up of matrix D, error nr: %d\n",info );
            return info;
        }
        if ( * ( position+1 ) ==0 ) {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time set-up of D, B and Y:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            clustout << "At end of allocations in cluster processes" << endl;
            clustout << "==========================================" << endl;
            clustout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            clustout << "Resident set size:    " << resident_set << " kb" << endl;
            clustout << "CPU time (user):      " << cpu_user << " s"<< endl;
            clustout << "CPU time (system):    " << cpu_sys << " s" << endl;
            //printdense(Drows*blocksize,Dcols*blocksize,Dmat,"Dmat.txt");
            /*Btsparse.transposeIt(1);
            Btsparse.writeToFile("Bsparse.csr");
            Btsparse.transposeIt(1);*/
        }
    }

    else {
        ytot = ( double* ) calloc ( ydim,sizeof ( double ) );
        if ( ytot==NULL ) {
            printf ( "unable to allocate memory for Matrix Y (required: %d bytes)\n", ydim*sizeof ( double ) );
            return EXIT_FAILURE;
        }
        solution= ( double * ) calloc ( ( k+l+m ), sizeof ( double ) );
        if ( solution==NULL ) {
            printf ( "unable to allocate memory for solution matrix\n" );
            return EXIT_FAILURE;
        }
        randnrm= ( double * ) calloc ( 2,sizeof ( double ) );
        if ( randnrm==NULL ) {
            printf ( "unable to allocate memory for norm\n" );
            return EXIT_FAILURE;
        }
        respnrm= ( double * ) calloc ( 1,sizeof ( double ) );
        if ( respnrm==NULL ) {
            printf ( "unable to allocate memory for norm\n" );
            return EXIT_FAILURE;
        }
        AImat = ( double* ) calloc ( 3*3,sizeof ( double ) );
        if ( AImat==NULL ) {
            printf ( "unable to allocate memory for AI matrix (required: %d bytes)\n",3*3*sizeof ( double ) );
            return EXIT_FAILURE;
        }

        Xsparse.loadFromFile ( filenameX );
        Zsparse.loadFromFile ( filenameZ );

        XtX_sparse.matmul ( Xsparse,1,Xsparse );
        XtX_sparse.reduceSymmetric();
        XtZ_sparse.matmul ( Xsparse,1,Zsparse );
        Xsparse.clear();
        ZtZ_sparse.matmul ( Zsparse,1,Zsparse );
        Zsparse.clear();
        ZtZ_sparse.reduceSymmetric();
        process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
        rootout << "At end of allocations in root process"  << endl;
        rootout << "====================================="  << endl;
        rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
        rootout << "Resident set size:    " << resident_set << " kb" << endl;
        rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
        rootout << "CPU time (system):    " << cpu_sys << " s" << endl;

    }
    MPI_Barrier ( MPI_COMM_WORLD );

    counter=0;


// Start of AI-REML iteration cycle, stops when relative update of gamma < epsilon

    while ( fabs ( *convergence_criterium ) >epsilon || fabs ( update_loglikelihood/loglikelihood ) > epsilon || counter<2 ||
            fabs ( * ( convergence_criterium+1 ) ) >epsilon ) {
        ++counter;

        // Number of iterations is limited to $maxiterations to avoid divergence of algorithm

        if ( counter > maxiterations ) {
            if ( iam==0 ) {
                printf ( "maximum number of iterations reached, AI-REML has not converged\n" );
            }
            break;
        }

        if ( iam==0 ) {
            printf ( "\nParallel results: loop %d\n",counter );
            printf ( "=========================\n" );
            gettimeofday ( &tz3,NULL );
            c3= tz3.tv_sec*1000000 + ( tz3.tv_usec );
        }

        // Since after every iteration the inverse of C is stored in Cmat and C must be updated with the new lambda,
        // every cycle C has to be set up again from scratch
        // Maybe it would be better not to free and allocate every time but just set every element to zero (for loop)??

        if ( iam !=0 ) {
            if ( counter > 1 ) {
                Dmat= ( double* ) calloc ( D_elements,sizeof ( double ) );
                if ( Dmat==NULL ) {
                    printf ( "unable to allocate memory for Matrix C (required: %lld bytes)\n", D_elements*sizeof ( double ) );
                    return EXIT_FAILURE;
                }
                blacs_barrier_ ( &ICTXT2D,"A" );

                if ( copyC ) {
                    update_C ( DESCCCOPY,Cmatcopy,gamma_var * * ( convergence_criterium+1 ) );
                    pdlacpy_ ( "U", &Ddim, &Ddim, Cmatcopy, &i_one, &i_one, DESCCCOPY, Dmat, &i_one, &i_one, DESCD );
                    //pdcopy_ ( &Ddim, RHS,&i_one,&i_one,DESCRHS,&i_one,ytot,&i_one,&i_one,DESCYTOT,&i_one );
                    if ( * ( position+1 ) ==0 && *position==0 ) {
                        gettimeofday ( &tz1,NULL );
                        c0= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                        printf ( "\t elapsed wall time copy of Y and C:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                    }
                    gamma_var=gamma_var * ( 1+ * ( convergence_criterium+1 ) ); // Update for gamma
                } else {
                    gamma_var=gamma_var * ( 1 + * ( convergence_criterium+1 ) ); // Update for lambda (which is 1/gamma)
                    phi=phi* ( 1 + *convergence_criterium );
                    if ( datahdf5 )
                        info = set_up_C_hdf5 ( DESCD, Dmat, DESCYTOT, ytot, respnrm );
                    else
                        info = set_up_D ( DESCD, Dmat );
                    if ( info!=0 ) {
                        printf ( "Something went wrong with set-up of matrix D, error nr: %d\n",info );
                        return info;
                    }
                    if ( * ( position+1 ) ==0 && *position==0 ) {
                        gettimeofday ( &tz0,NULL );
                        c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                        printf ( "\t elapsed wall time set-up of D:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                    }

                }
            } else {


                // RHS is copied for use afterwards (in ytot we will get the estimates for the effects)
                // This only needs to be done the first time
                //pdcopy_ ( &Ddim, ytot,&i_one,&i_one,DESCYTOT,&i_one,RHS,&i_one,&i_one,DESCRHS,&i_one );
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
                printf ( "The new variance components gamma and phi are: %15.10g \t %15.10g\n",gamma_var, phi );
                printf ( "norm of y-vector is: %g\n",*respnrm );
            }
        }

        //Since lambda changes every iteration we need to construct A every iteration
        else {
            gamma_var=gamma_var * ( 1 + * ( convergence_criterium+1 ) ); // Update for lambda (which is 1/gamma)
            phi=phi* ( 1 + *convergence_criterium );
            if ( counter >1 ) {
                if ( AImat != NULL )
                    free ( AImat );
                AImat=NULL;
                AImat = ( double* ) calloc ( 3*3,sizeof ( double ) );
                if ( AImat==NULL ) {
                    printf ( "unable to allocate memory for AI matrix\n" );
                    return EXIT_FAILURE;
                }
            }
            makeDiag ( ZtZ_sparse.nrows,1/phi,Diagmat );

            ZtZ_sparse.addBCSR ( Diagmat );

            Diagmat.clear();

            process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            rootout << "At end of calculation of ZtZ" << endl;
            rootout << "===========================" << endl;
            rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            rootout << "Resident set size:    " << resident_set << " kb" << endl;
            rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
            rootout << "CPU time (system):    " << cpu_sys << " s" << endl;



            rootout << "***                                           [  t     t  ] *** " << endl;
            rootout << "***                                           [ X X   X Z ] *** " << endl;
            rootout << "***                                           [           ] *** " << endl;
            rootout << "*** G e n e r a t i n g    m a t r i x    A = [           ] *** " << endl;
            rootout << "***                                           [  t     t  ] *** " << endl;
            rootout << "***                                           [ Z X   Z Z ] *** " << endl;

            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            //Sparse matrix A only contains the upper triangular part of A
            create2x2SymBlockMatrix ( XtX_sparse, XtZ_sparse, ZtZ_sparse, Asparse );
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time for creating sparse matrix A:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            /*Asparse.writeToFile ( "Asparse.csr" );
            double * Adense = new double[Asparse.nrows * Asparse.ncols];
            CSR2dense ( Asparse,Adense );
            printdense ( Asparse.nrows,Asparse.ncols,Adense,"Adense.txt" );*/
            process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            rootout << "After creation of Asparse" << endl;
            rootout << "=========================" << endl;
            rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            rootout << "Resident set size:    " << resident_set << " kb" << endl;
            rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
            rootout << "CPU time (system):    " << cpu_sys << " s" << endl;

            makeDiag ( ZtZ_sparse.nrows,-1/phi,Diagmat );

            ZtZ_sparse.addBCSR ( Diagmat );

            Diagmat.clear();
        }

        MPI_Barrier ( MPI_COMM_WORLD );

        if ( iam==0 ) {
            for ( i=1; i<size; ++i ) {
                MPI_Ssend ( & ( Asparse.nonzeros ),1, MPI_INT,i,i,MPI_COMM_WORLD );
                MPI_Ssend ( & ( Asparse.pRows[0] ),Asparse.nrows + 1, MPI_INT,i,i+size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( Asparse.pCols[0] ),Asparse.nonzeros, MPI_INT,i,i+2*size,MPI_COMM_WORLD );
                MPI_Ssend ( & ( Asparse.pData[0] ),Asparse.nonzeros, MPI_DOUBLE,i,i+3*size,MPI_COMM_WORLD );
            }
            MPI_Recv ( ytot,ydim, MPI_DOUBLE,1,ydim,MPI_COMM_WORLD,&status );
            //printdense ( k+l+m,1,ytot,"ytot.txt" );
            printf ( "Solving system Ax_u = y_u on process 0\n" );
            solveSystem ( Asparse,solution,ytot,2,1 );
            //printdense ( ydim,1,solution,"wA.txt" );
            MPI_Ssend ( solution,Adim, MPI_DOUBLE,1,1,MPI_COMM_WORLD );
            MPI_Recv ( solution+Adim,k, MPI_DOUBLE,1,k,MPI_COMM_WORLD,&status );
            MPI_Recv ( solution,Adim, MPI_DOUBLE,1,Adim,MPI_COMM_WORLD,&status );
            MPI_Recv ( respnrm,1, MPI_DOUBLE,1,1,MPI_COMM_WORLD,&status );
	    
	    process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            rootout << "After factorisation of Asparse" << endl;
            rootout << "==============================" << endl;
            rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            rootout << "Resident set size:    " << resident_set << " kb" << endl;
            rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
            rootout << "CPU time (system):    " << cpu_sys << " s" << endl;

            double * sparse_sol = new double[Adim];

            printf ( "Solving system Au=y_u - Bd on process 0\n" );
            loglikelihood=solveSystemWithDet ( Asparse, sparse_sol,solution, -2, 1 ) /2;
            memcpy ( solution,sparse_sol, ( m+l ) * sizeof ( double ) );
            if ( sparse_sol != NULL )
                free ( sparse_sol );
            sparse_sol=NULL;
            //printdense ( ydim,1,solution,"solution.txt" );
            printf ( "Half of the log of determinant of A is: %g\n",loglikelihood );

            dot = ddot_ ( &ydim,ytot,&i_one,solution,&i_one );

            // sigma is updated based on estimations of effects and square of norm of y (y'y)
            sigma= ( *respnrm - dot ) / ( n-m );
            MPI_Bcast ( &sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
            printf ( "dot product : %g \n sigma: %g\n", dot,sigma );
	    double log_det_D;
	    MPI_Recv ( &log_det_D,1, MPI_DOUBLE,1,1,MPI_COMM_WORLD,&status );
	    loglikelihood += log_det_D;
	    printf ( "Half of the log of determinant of entire matrix C is: %g\n",loglikelihood );
	    
        } else {
            int nonzeroes, count;

            MPI_Recv ( &nonzeroes,1,MPI_INT,0,iam,MPI_COMM_WORLD,&status );
            Asparse.allocate ( Adim,Adim,nonzeroes );
            MPI_Recv ( & ( Asparse.pRows[0] ),Adim + 1, MPI_INT,0,iam + size,MPI_COMM_WORLD,&status );
            MPI_Get_count ( &status,MPI_INT,&count );
            MPI_Recv ( & ( Asparse.pCols[0] ),nonzeroes, MPI_INT,0,iam+2*size,MPI_COMM_WORLD,&status );
            MPI_Recv ( & ( Asparse.pData[0] ),nonzeroes, MPI_DOUBLE,0,iam+3*size,MPI_COMM_WORLD,&status );
            if ( * ( position+1 ) ==0 ) {
                MPI_Ssend ( ytot,ydim, MPI_DOUBLE,0,ydim,MPI_COMM_WORLD );
                process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
                clustout << "Before calculation of Schur complement in cluster processes" << endl;
                clustout << "===========================================================" << endl;
                clustout << "Virtual memory used:  " << vm_usage << " kb" << endl;
                clustout << "Resident set size:    " << resident_set << " kb" << endl;
                clustout << "CPU time (user):      " << cpu_user << " s"<< endl;
                clustout << "CPU time (system):    " << cpu_sys << " s" << endl;
            }

            make_Si_distributed_denseB ( Asparse, Bmat, DESCB, Dmat, DESCD, AB_sol, DESCAB_sol );
            if ( * ( position+1 ) ==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time for creating Schur complement of D:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                //printdense(Drows*blocksize, Drows*blocksize,Dmat,"Dmat.txt");
            }
            Asparse.clear();
            /*char *Dfile;
            Dfile= ( char * ) calloc ( 100,sizeof ( char ) );
            *Dfile='\0';
            sprintf ( Dfile,"Smat_(%d,%d).txt",*position,pcol );
            printdense ( Dcols * blocksize,Drows,Dmat,Dfile );
            char *Bfile;
            Bfile= ( char * ) calloc ( 100,sizeof ( char ) );
            *Bfile='\0';
            sprintf ( Bfile,"ABsol_(%d,%d).txt",*position,pcol );
            printdense ( Dcols * blocksize,Adim,AB_sol,Bfile );*/

            //BT_i.clear();
            //B_j.clear();

            //From here on the Schur complement S of D is stored in D
            if ( * ( position+1 ) ==0 ) {
                MPI_Recv ( solution,Adim, MPI_DOUBLE,0,1,MPI_COMM_WORLD,&status );
            }
            pdcopy_ ( &k,ytot,&ml_plus,&i_one,DESCYTOT,&i_one,densesol,&i_one,&i_one, DESCDENSESOL, &i_one );

            pdgemm_ ( "T","N",&Ddim,&i_one,&Adim,&d_negone, Bmat,&i_one, &i_one,DESCB, solution,&i_one, &i_one,DESCSPARSESOL, &d_one, densesol, &i_one, &i_one, DESCDENSESOL ); //T'T

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

            //Estimation of genetic effects, stored in solution on root process.
            //pdcopy_(&k,solution,&ml_plus,&i_one,DESCSOL,&i_one,densesol,&i_one,&i_one, DESCDENSESOL, &i_one);
            pdpotrs_ ( "U",&Ddim,&i_one,Dmat,&i_one,&i_one,DESCD,densesol,&i_one,&i_one,DESCDENSESOL,&info );
            if ( info!=0 ) {
                printf ( "Parallel Cholesky solution was unsuccesful, error returned: %d\n",info );
                return -1;
            }
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time estimation of dense effects:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                //printdense ( k, 1,densesol,"densesol.txt" );
            }

            pdcopy_ ( &ml_plus,ytot,&i_one,&i_one,DESCYTOT,&i_one,solution,&i_one,&i_one, DESCSOL, &i_one );
            pdgemm_ ( "N","N",&Adim,&i_one,&Ddim,&d_negone, Bmat,&i_one, &i_one,DESCB, densesol,&i_one, &i_one,DESCDENSESOL, &d_one, solution, &i_one, &i_one, DESCSOL );

            if ( * ( position+1 ) == 0 ) {
                MPI_Ssend ( densesol,k, MPI_DOUBLE,0,k,MPI_COMM_WORLD );
                MPI_Ssend ( solution,Adim, MPI_DOUBLE,0,Adim,MPI_COMM_WORLD );
                MPI_Ssend ( respnrm,1, MPI_DOUBLE,0,1,MPI_COMM_WORLD );
            }
            loglikelihood=0;
            MPI_Bcast ( &sigma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
            loglikelihood += log_determinant_C ( Dmat,DESCD );
            dgsum2d_ ( &ICTXT2D,"ALL","1-tree",&i_one,&i_one,&loglikelihood,&i_one,&i_negone,&i_negone );

            if ( * ( position+1 ) ==0 ) {
                printf ( "Half of the log of determinant of D is: %g\n",loglikelihood );
		MPI_Ssend ( &loglikelihood,1, MPI_DOUBLE,0,1,MPI_COMM_WORLD );
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time calculation and sending of sigma and log(det(M)):	%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            }
        }

        MPI_Barrier ( MPI_COMM_WORLD );

        if ( iam==0 ) {
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );

            info = set_up_AI ( AImat,DESCDENSESOL, solution, DESCD, Dmat, Asparse, DESCB, Bmat,sigma ) ;

            if ( info!=0 ) {
                printf ( "Something went wrong with set-up of AI-matrix, error nr: %d\n",info );
                return EXIT_FAILURE;
            }

            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time set up of AI matrix:			%10.3f s\n", ( c0 - c1 ) /1000000.0 );

	    process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            rootout << "Before inversion of Asparse" << endl;
            rootout << "===========================" << endl;
            rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            rootout << "Resident set size:    " << resident_set << " kb" << endl;
            rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
            rootout << "CPU time (system):    " << cpu_sys << " s" << endl;

            int number_of_processors = 1;
            char* var = getenv ( "OMP_NUM_THREADS" );
            if ( var != NULL )
                sscanf ( var, "%d", &number_of_processors );
            else {
                printf ( "Set environment OMP_NUM_THREADS to 1" );
                exit ( 1 );
            }

            pardiso_var.iparm[2]  = 2;
            pardiso_var.iparm[3]  = number_of_processors;
            pardiso_var.iparm[8]  = 0;
            pardiso_var.iparm[11] = 1;
            pardiso_var.iparm[13]  = 0;
            pardiso_var.iparm[28]  = 0;
            pardiso_var.iparm[36]  = 0;

            //This function calculates the factorisation of A once again so this might be optimized.
            pardiso_var.findInverseOfA ( Asparse );
            rootout << "memory allocated by PARDISO: " << pardiso_var.memoryAllocated() << endl;

            pardiso_var.clear();

            printf ( "Processor %d inverted matrix A\n",iam );

            double* Diag_inv_rand_block = ( double* ) calloc ( Dblocks * blocksize + Adim ,sizeof ( double ) );

            MPI_Recv ( Diag_inv_rand_block,Dblocks * blocksize + Adim, MPI_DOUBLE,1,Adim,MPI_COMM_WORLD,&status );

            for ( i=0; i<Adim; ++i ) {
                j=Asparse.pRows[i];
                * ( Diag_inv_rand_block+i ) += Asparse.pData[j];
            }
            Asparse.clear();
	    
	    process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            rootout << "After deleting Asparse" << endl;
            rootout << "======================" << endl;
            rootout << "Virtual memory used:  " << vm_usage << " kb" << endl;
            rootout << "Resident set size:    " << resident_set << " kb" << endl;
            rootout << "CPU time (user):      " << cpu_user << " s"<< endl;
            rootout << "CPU time (system):    " << cpu_sys << " s" << endl;
	    
            trace_ZZ=0;
            for ( i=m; i<m+l; ++i ) {
                trace_ZZ +=* ( Diag_inv_rand_block+i );
            }
            trace_TT=0;
            for ( i=m+l; i<ydim; ++i ) {
                trace_TT +=* ( Diag_inv_rand_block+i );
            }
            //printdense ( Adim+k,1,Diag_inv_rand_block,"diag_inverse_C_parallel.txt" );
            if ( Diag_inv_rand_block != NULL )
                free ( Diag_inv_rand_block );
            Diag_inv_rand_block=NULL;

            // The norm of the estimation of the random effects is calculated for use in the score function

            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );

            *randnrm = dnrm2_ ( &l,solution+m,&i_one );
            * ( randnrm+1 ) = dnrm2_ ( &k,solution+m+l,&i_one );

            // The score function (first derivative of log likelihood) and the update for lambda are only calculated in proces (0,0)
            // Afterwards the update is sent to every proces.

            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time set norm of estimation of u and d:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            double *score;
            printf ( "dot product = %15.10g \n",dot );
            printf ( "parallel sigma = %15.10g\n",sigma );
            printf ( "The trace of the (1,1) block of the inverse of M is: %15.10g \n",trace_ZZ );
            printf ( "The trace of the (2,2) block of the inverse of M is: %15.10g \n",trace_TT );
            printf ( "The norm of the estimation of u and d is: %g and %g \n",*randnrm, * ( randnrm+1 ) );
            loglikelihood += ( k * log ( gamma_var ) + l * log ( phi ) + ( n-m ) * log ( sigma ) + n - m ) /2;
            loglikelihood *= -1.0;


            score= ( double * ) calloc ( 3,sizeof ( double ) );
            if ( score==NULL ) {
                printf ( "unable to allocate memory for score function\n" );
                return EXIT_FAILURE;
            }
            * ( score+1 ) = - ( l - trace_ZZ / phi - *randnrm * *randnrm / phi / sigma ) / phi / 2;
            * ( score+2 ) = - ( k - trace_TT / gamma_var - * ( randnrm+1 ) * * ( randnrm+1 ) / gamma_var / sigma ) / gamma_var / 2;
            printf ( "The score function is: [%g, %g, %g]\n",*score,* ( score+1 ), * ( score+2 ) );
            printdense ( 2,2, AImat, "AI_par.txt" );
            breakvar=0;
            if ( fabs ( * ( score+1 ) ) < epsilon * epsilon ) {
                printf ( "Score function too close to zero to go further, solution may not have converged\n " );
                breakvar=1;
                MPI_Bcast ( &breakvar,1,MPI_INT,0,MPI_COMM_WORLD );
                break;
            }
            MPI_Bcast ( &breakvar,1,MPI_INT,0,MPI_COMM_WORLD );

            //Use of damping factor

            if ( counter==1 ) {
                //trace_proc= *AImat + * ( AImat+3 );
                //damping=trace_proc/m;
                prevloglike=loglikelihood;
                printf ( "The loglikelihood is: %g\n",loglikelihood );
                update_loglikelihood=0;

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
            MPI_Bcast ( &update_loglikelihood,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
            //cout << "process " << iam << " sent update_loglikelihood" << endl;
            MPI_Bcast ( &loglikelihood,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
            //cout << "process " << iam << " sent loglikelihood" << endl;
            /*
                        *AImat += damping;
                        *(AImat+1) += damping;

                        printf("Used damping factor is %g\n",damping);
            */
            //printdense ( 3,3,AImat,"AImat.txt" );
            dpotrf_ ( "U", &i_three, AImat, &i_three, &info );
            if ( info!=0 ) {
                printf ( "Cholesky decomposition of AI matrix was unsuccesful, error returned: %d\n",info );
                return -1;
            }
            //printdense ( 3,3,AImat,"AImat_chol.txt" );

            dpotrs_ ( "U",&i_three,&i_one,AImat,&i_three,score,&i_three,&info );
            if ( info!=0 ) {
                printf ( "Parallel solution for AI matrix was unsuccesful, error returned: %d\n",info );
                return -1;
            }
            cout << "score function solved" << endl;
            gettimeofday ( &tz1,NULL );
            c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
            printf ( "\t elapsed wall time update for lambda:			%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            printf ( "The update for sigma is: %g \n", *score );
            printf ( "The update for phi is: %g \n", * ( score+1 ) );
            printf ( "The update for gamma is: %g \n", * ( score+2 ) );
            while ( * ( score+2 ) + gamma_var <0 || * ( score+1 ) + phi < 0 ) {
                * ( score+1 ) =* ( score+1 ) /2;
                * ( score+2 ) = * ( score+2 ) /2;
                printf ( "Half a step is used to avoid negative gamma or phi\n" );
            }
            *convergence_criterium= * ( score+1 ) /phi;
            * ( convergence_criterium+1 ) = * ( score+2 ) /gamma_var;
            MPI_Bcast ( convergence_criterium,2,MPI_DOUBLE,0,MPI_COMM_WORLD );
            if ( score != NULL )
                free ( score );
            score=NULL;
            printf ( "The relative update for phi is: %g \n", *convergence_criterium );
            printf ( "The relative update for gamma is: %g \n", * ( convergence_criterium+1 ) );
        } else {
            info = set_up_AI ( AImat,DESCDENSESOL, densesol, DESCD, Dmat, Asparse, DESCB, Bmat,sigma ) ;

            if ( info!=0 ) {
                printf ( "Something went wrong with set-up of AI-matrix, error nr: %d\n",info );
                return EXIT_FAILURE;
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

            //From here on the inverse of the Schur complement S is stored in D

            blacs_barrier_ ( &ICTXT2D,"A" );

            double* Diag_inv_rand_block = ( double* ) calloc ( Dblocks * blocksize + Adim ,sizeof ( double ) );

            //Diagonal elements of the (1,1) block of C^-1 are still distributed and here they are gathered in InvD_T_Block in the root process.
            for ( i=0; i<Ddim; ++i ) {
                if ( pcol == ( i/blocksize ) % * ( dims+1 ) ) {
                    int Dpos = i%blocksize + ( ( i/blocksize ) / * ( dims+1 ) ) * blocksize ;
                    * ( Diag_inv_rand_block + Adim +i ) = * ( Dmat + i + lld_D * Dpos );
                }
            }
            for ( i=0,j=0; i<Dblocks; ++i,++j ) {
                if ( j==* ( dims+1 ) )
                    j=0;
                if ( pcol==j ) {
                    dgesd2d_ ( &ICTXT2D,&blocksize,&i_one,Diag_inv_rand_block + Adim + i * blocksize,&blocksize,&i_zero,&i_zero );
                }
                if ( pcol==0 ) {
                    dgerv2d_ ( &ICTXT2D,&blocksize,&i_one,Diag_inv_rand_block + Adim + blocksize*i,&blocksize,&i_zero,&j );
                }
            }

            //Calculating diagonal elements 1 by 1 of the (0,0)-block of C^-1.
            for ( i=1; i<=Adim; ++i ) {
                pdsymm_ ( "R","U",&i_one,&Ddim,&d_one,Dmat,&i_one,&i_one,DESCD,AB_sol,&i,&i_one,DESCAB_sol,&d_zero,YSrow,&i_one,&i_one,DESCYSROW );
                pddot_ ( &Ddim,Diag_inv_rand_block+i-1,AB_sol,&i,&i_one,DESCAB_sol,&Adim,YSrow,&i_one,&i_one,DESCYSROW,&i_one );
            }
            blacs_barrier_ ( &ICTXT2D,"A" );

            if ( * ( position+1 ) ==0 )
                MPI_Ssend ( Diag_inv_rand_block,Dblocks * blocksize + Adim, MPI_DOUBLE,0,Adim,MPI_COMM_WORLD );

            if ( Diag_inv_rand_block != NULL )
                free ( Diag_inv_rand_block );
            Diag_inv_rand_block=NULL;

            norminv=pdlansy_ ( "F","U",&Ddim,Dmat,&i_one,&i_one,DESCD,work );
            norm1inv=pdlansy_ ( "1","U",&Ddim,Dmat,&i_one,&i_one,DESCD,work );
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time set norm of inverse of C:		%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                process_mem_usage ( vm_usage, resident_set, cpu_user, cpu_sys );
            }
            if ( work != NULL )
                free ( work );
            work=NULL;
            if ( Dmat != NULL )
                free ( Dmat );
            Dmat=NULL;

            MPI_Bcast ( &breakvar,1,MPI_INT,0,MPI_COMM_WORLD );
            if ( breakvar >0 ) {
                break;
            }
            //cout << "process " << iam << " received breakvar" << endl;
            MPI_Bcast ( &update_loglikelihood,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
            //cout << "process " << iam << " received update_loglikelihood" << endl;
            MPI_Bcast ( &loglikelihood,1,MPI_DOUBLE,0,MPI_COMM_WORLD );
            //cout << "process " << iam << " received loglikelihood" << endl;
            MPI_Bcast ( convergence_criterium,2,MPI_DOUBLE,0,MPI_COMM_WORLD );
            //cout << "process " << iam << " received convergence_criterium" << endl;
            blacs_barrier_ ( &ICTXT2D,"A" );
        }

        MPI_Barrier ( MPI_COMM_WORLD );
        if ( iam==0 ) {
            gettimeofday ( &tz0,NULL );
            c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
            printf ( "\t elapsed wall time sending and receiving update lambda:	%10.3f s\n", ( c0 - c1 ) /1000000.0 );
            printf ( "\t elapsed wall time iteration loop %d:			%10.3f s\n", counter, ( c0 - c3 ) /1000000.0 );
        }
        //cout << "process " << iam << " at end of iteration" << endl;
    }




    // The trace of the inverse of C is calculated per diagonal block and then summed over all processes and stored in proces (0,0)


    /*
            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz1,NULL );
                c1= tz1.tv_sec*1000000 + ( tz1.tv_usec );
                printf ( "\t elapsed wall time trace of inverse of C:		%10.3f s\n", ( c1 - c0 ) /1000000.0 );
            }



            if ( * ( position+1 ) ==0 && *position==0 ) {
                gettimeofday ( &tz0,NULL );
                c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
                printf ( "\t elapsed wall time sending and receiving update lambda:	%10.3f s\n", ( c0 - c1 ) /1000000.0 );
                printf ( "\t elapsed wall time iteration loop %d:			%10.3f s\n", counter, ( c0 - c3 ) /1000000.0 );
            }

        }
    */


    //cout << "process " << iam << ": ntests=" << ntests << endl;

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

    //cout << "process " << iam << " Before output" << endl;

    MPI_Barrier ( MPI_COMM_WORLD );
    secs.tack ( totalTime );
    if ( iam==0 ) {
        gettimeofday ( &tz0,NULL );
        c0= tz0.tv_sec*1000000 + ( tz0.tv_usec );
        printf ( "\n\tOverall results:\n" );
        printf ( "\t================\n" );
        /*printf ( "\tThe maximum element in C is:          %10.5f\n", Cmax );
        printf ( "\tThe Frobenius norm of C is:           %15.10e\n", normC );
        printf ( "\tThe 1-norm of C is:                   %15.10e\n", norm1C );
        printf ( "\tThe Frobenius norm of Cinv is:        %15.10e\n", norminv );
        printf ( "\tThe 1-norm of Cinv is:                %15.10e\n", norm1inv );
        printf ( "\tThe Frobenius condition number is:    %15.10e\n", norminv*normC );
        printf ( "\tThe condition number (1-norm) is:     %15.10e\n", norm1inv*norm1C );
        printf ( "\tThe accuracy is:                      %15.10e\n", norminv*normC*Cmax/pow ( 2,53 ) );*/
        printf ( "\tThe ultimate gamma is:               %15.10g\n", gamma_var );
        printf ( "\tThe ultimate phi is:               %15.10g\n", phi );
        printf ( "\tThe ultimate sigma is:                %15.10g\n", sigma );

        printf ( "\telapsed total wall time:              %10.3f s\n", totalTime * 0.001 );

        printf ( "\tProcessor: %d \n\t ========================\n", iam );
        printf ( "\tVirtual memory used:                  %10.0f kb\n", vm_usage );
        printf ( "\tResident set size:                    %10.0f kb\n", resident_set );
        printf ( "\tCPU time (user):                      %10.3f s\n", cpu_user );
        printf ( "\tCPU time (system):                    %10.3f s\n", cpu_sys );
        printdense ( m,1,solution,"estimates_fixed_effects.txt" );
        printdense ( l,1,solution+m,"estimates_random_sparse_effects.txt" );
        printdense ( k,1,solution+m+l,"estimates_random_genetic_effects.txt" );

        if ( solution != NULL )
            free ( solution );
        solution=NULL;
        if ( ytot != NULL )
            free ( ytot );
        ytot=NULL;
        if ( randnrm != NULL )
            free ( randnrm );
        randnrm=NULL;
        if ( respnrm != NULL )
            free ( respnrm );
        respnrm=NULL;
        if ( AImat != NULL )
            free ( AImat );
        AImat=NULL;
        XtX_sparse.clear();
        XtZ_sparse.clear();
        ZtZ_sparse.clear();
        Asparse.clear();

    } else {
        if ( * ( position+1 ) ==0 ) {
            if ( ytot != NULL )
                free ( ytot );
            ytot=NULL;
            if ( solution != NULL )
                free ( solution );
            solution=NULL;
        }

        if ( DESCAB_sol != NULL )
            free ( DESCAB_sol );
        DESCAB_sol=NULL;
        if ( DESCD != NULL )
            free ( DESCD );
        DESCD=NULL;
        if ( DESCSOL != NULL )
            free ( DESCSOL );
        DESCSOL=NULL;
        if ( DESCDENSESOL != NULL )
            free ( DESCDENSESOL );
        DESCDENSESOL=NULL;
        if ( DESCSPARSESOL != NULL )
            free ( DESCSPARSESOL );
        DESCSPARSESOL=NULL;
        if ( DESCYSROW != NULL )
            free ( DESCYSROW );
        DESCYSROW=NULL;
        if ( DESCYTOT != NULL )
            free ( DESCYTOT );
        DESCYTOT=NULL;
        if ( DESCB != NULL )
            free ( DESCB );
        DESCB=NULL;
        if ( copyC ) {
            if ( Cmatcopy != NULL )
                free ( Cmatcopy );
            Cmatcopy=NULL;
            if ( DESCCCOPY != NULL )
                free ( DESCCCOPY );
            DESCCCOPY=NULL;
        }

        if ( Dmat != NULL )
            free ( Dmat );
        Dmat=NULL;
        if ( Bmat != NULL )
            free ( Bmat );
        Bmat=NULL;
        if ( densesol != NULL )
            free ( densesol );
        densesol=NULL;
        if ( respnrm != NULL )
            free ( respnrm );
        respnrm=NULL;
        if ( AB_sol != NULL )
            free ( AB_sol );
        AB_sol=NULL;
        if ( YSrow != NULL )
            free ( YSrow );
        YSrow=NULL;
    }

    MPI_Barrier ( MPI_COMM_WORLD );

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


    if ( position != NULL )
        free ( position );
    position=NULL;
    if ( dims != NULL )
        free ( dims );
    dims=NULL;
    if ( filenameT != NULL )
        free ( filenameT );
    filenameT=NULL;
    if ( filenameX != NULL )
        free ( filenameX );
    filenameX=NULL;
    if ( filenameY != NULL )
        free ( filenameY );
    filenameY=NULL;
    if ( filenameZ != NULL )
        free ( filenameZ );
    filenameZ=NULL;
    if ( TestSet != NULL )
        free ( TestSet );
    TestSet=NULL;
    if ( SNPdata != NULL )
        free ( SNPdata );
    SNPdata=NULL;
    if ( phenodata != NULL )
        free ( phenodata );
    phenodata=NULL;

    if ( convergence_criterium != NULL )
        free ( convergence_criterium );
    convergence_criterium=NULL;


    if ( iam !=0 )
        blacs_gridexit_ ( &ICTXT2D );

    //cout << iam << " reached end before MPI_Barrier" << endl;
    MPI_Barrier ( MPI_COMM_WORLD );
    MPI_Finalize();

    return 0;

}

