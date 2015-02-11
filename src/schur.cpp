#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
//#include <mkl_lapack.h>
#include "shared_var.h"
//#include <mkl_blas.h>
#include "CSRdouble.hpp"
#include "ParDiSO.hpp"
#include <cassert>

extern "C" {
    void dgemm_ ( const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc );
    void dpotrf_( const char* uplo, const int* n, double* a, const int* lda, int* info );
    void dpotrs_( const char* uplo, const int* n, const int* nrhs,const double* a, const int* lda, double* b, const int* ldb, int* info );
}

/**
 * @brief I tried to create the Schur complement only using sparse B's and the Schur complement modus of PARDISO, but it did not work due to the fact that C can get rectangular
 *
 * @param A Full sparse (0,0)-block of which we want the Schur complement in matrix C
 * @param BT_i Sparse (1,0)-block of C corresponding to T_ij
 * @param B_j Sparse (0,1)-block of C corresponding to T_ij
 * @param T_ij Dense (1,1)-block of C
 * @param lld_T local leading dimension of T_ij
 * @return int
 **/
/*int make_Sij_sparse_parallel(CSRdouble& A, CSRdouble& BT_i, CSRdouble& B_j, double * T_ij, int lld_T) {
    CSRdouble C,S;
    if (iam==0) {
        cout << "***                                           [ A      B_j ] *** " << endl;
        cout << "***                                           [            ] *** " << endl;
        cout << "*** G e n e r a t i n g    m a t r i x    C = [            ] *** " << endl;
        cout << "***                                           [    t       ] *** " << endl;
        cout << "***                                           [ B_i    T_ij] *** " << endl;
    }
    create2x2BlockMatrix_denseT_lldT(A, BT_i, B_j, T_ij, lld_T,  C);
    blacs_barrier_ ( &ICTXT2D, "A" );

    S.nrows=BT_i.nrows;
    S.ncols=B_j.ncols;

    calculateSchurComplement(C, 11, S);
    blacs_barrier_ ( &ICTXT2D, "A" );

    CSR2dense_lld ( S, T_ij, lld_T ) ;
    return 0;
}
*/
/**
 * @brief BT_i and B_j are converted to dense matrices in each process to solve the sparse system AX=B_j and afterwards do BT_i * X. X is stored as a dense matrix in AB_sol
 *
 * @param A Sparse (0,0)-block of which we want to compute the Schur complement in matrix C
 * @param BT_i Sparse (1,0)-block of C corresponding to T_ij.
 * @param B_j Sparse (0,1)-block of C corresponding to T_ij
 * @param T_ij Dense (1,1)-block of C
 * @param lld_T local leading dimension of T_ij
 * @param AB_sol_out Dense solution of AX=B_j (output)
 * @return int
 **/
int make_Sij_parallel_denseB(CSRdouble& A, CSRdouble& BT_i, CSRdouble& B_j, double * T_ij, int lld_T, double * AB_sol_out) {

    double *BT_i_dense;

    timing secs;
    double MultTime       = 0.0;

    assert(A.nrows == BT_i.ncols);

    if(Bassparse_bool) {
        CSRdouble AB_sol, W, unit, zero;
        int ori_cols;

        ori_cols=B_j.ncols;

        if(B_j.ncols < B_j.nrows) {
            B_j.ncols=B_j.nrows;
        }
        zero.nrows=B_j.ncols;
        zero.ncols=B_j.ncols;
        zero.nonzeros=0;
        zero.pCols=new int[1];
        zero.pData=new double[1];
        zero.pRows=new int[zero.nrows+1];
        unit.nrows=B_j.ncols;
        unit.ncols=A.ncols;
        unit.nonzeros=A.ncols;
        unit.pCols=new int[unit.ncols];
        unit.pData=new double[unit.ncols];
        unit.pRows=new int[unit.nrows +1];
        for (int i =0; i<unit.ncols; ++i) {
            unit.pData[i]=-1.0;
            unit.pCols[i]=i;
            unit.pRows[i]=i;
            zero.pRows[i]=0;
        }
        for (int i=unit.ncols; i<=unit.nrows; ++i) {
            unit.pRows[i]=unit.ncols;
            zero.pRows[i]=0;
        }
        zero.pData[0]=0;
        zero.pCols[0]=0;

        A.fillSymmetric();

        create2x2BlockMatrix(A, B_j, unit, zero, W);

        A.reduceSymmetric();

        //W.writeToFile("W.csr");

        unit.clear();
        zero.clear();

        AB_sol.nrows=B_j.ncols;
        AB_sol.ncols=B_j.ncols;

        assert(A.nrows==A.ncols);
        assert(W.nrows==W.ncols);

        //printf("Dimension of W: %d \nDimension of A: %d \n Dimension of AB_sol: %d \n", W.nrows, A.nrows, AB_sol.nrows );

        calculateSchurComplement( W, 11, AB_sol);

        //AB_sol.writeToFile("AB_sol.csr");
        AB_sol.transposeIt(1);

        W.clear();

        AB_sol.nrows=ori_cols;
        AB_sol.nonzeros=AB_sol.pRows[ori_cols];
        AB_sol.pRows= (int *) realloc(AB_sol.pRows, (ori_cols +1) * sizeof(int) );

        B_j.ncols=ori_cols;

        //AB_sol.writeToFile("AB_sol_trans.csr");

        AB_sol.transposeIt(1);

        //AB_sol.writeToFile("AB_sol_trans2.csr");
        CSR2dense(AB_sol,AB_sol_out);

        /*if(iam==0)
          printdense(B_j.ncols,A.nrows,AB_sol_out,"AB_sol_sparse.txt");*/

        AB_sol.clear();

    }
    else {
        double *B_j_dense;

        B_j_dense=(double *) calloc(B_j.nrows * B_j.ncols,sizeof(double));

        CSR2dense(B_j,B_j_dense);
        if(iam==0)
            printf("Solving systems AX_j = B_j on all processes\n");
        solveSystem(A, AB_sol_out,B_j_dense, -2, B_j.ncols);

        /*if(iam==0)
          printdense(B_j.ncols,A.nrows,AB_sol_out,"AB_sol_dense.txt");*/

        if(B_j_dense!=NULL) {
            free(B_j_dense);
            B_j_dense=NULL;
        }

        //printf("Processor %d finished solving system AX=B\n",iam);

    }
    int Arows = A.nrows;
    if(iam !=0 ) {
        A.clear();
        pardiso_var.clear();
    }

    BT_i_dense=(double *) calloc(BT_i.nrows * BT_i.ncols,sizeof(double));

    CSR2dense(BT_i,BT_i_dense);

    secs.tick(MultTime);
    dgemm_("N","N",&(BT_i.nrows),&(B_j.ncols),&(BT_i.ncols),&d_negone,BT_i_dense,&(BT_i.nrows),
           AB_sol_out,&(Arows),&d_one,T_ij,&lld_T);
    secs.tack(MultTime);

    /*if(iam==0)
      cout << "Time for multiplying BT_i and Y_j: " << MultTime * 0.001 << " sec" << endl;*/

    if(BT_i_dense!=NULL) {
        free(BT_i_dense);
        BT_i_dense=NULL;
    }

    return 0;
}

int make_Si_distributed_denseB(CSRdouble& A, double * B, int * DESCB, double * S, int *DESCS, double * AB_sol_out, int *DESCABSOL) {


    timing secs;
    double MultTime       = 0.0;

    if(*(position+1)==0)
        printf("Solving systems AX_j = B_j on all processes\n");
    solveSystem(A, AB_sol_out,B, -2, Dcols * blocksize);


    int Arows = A.nrows;
    if(iam !=0 ) {
        A.clear();
        pardiso_var.clear();
    }

    secs.tick(MultTime);
    pdgemm_("T","N",&Ddim, &Ddim, &Adim, &d_negone,B, &i_one, &i_one, DESCB, AB_sol_out, &i_one, &i_one, DESCABSOL, &d_one, S, &i_one, &i_one, DESCS);
    secs.tack(MultTime);


    return 0;
}
