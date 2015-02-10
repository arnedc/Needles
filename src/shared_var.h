#ifndef Shared_Var_h
#define Shared_Var_h

#ifndef config_hpp
#define config_hpp

#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <vector>
#include <mpi.h>


using std::complex;
using std::endl;
using std::setw;
using std::cout;
using std::fstream;
using std::ios;
using std::string;
using std::vector;

enum MatrixStorage
{
    NOT_SET   = 0,
    NORMAL    = 1,
    TRANSPOSE = 2,
    SYMMETRIC = 3,
    HERMITIAN = 4,
};

#endif
/*
#define MPI_COMM_WORLD ((MPI_Comm)0x44000000)

typedef int MPI_Comm;

typedef int MPI_Datatype;
#define MPI_CHAR           ((MPI_Datatype)0x4c000101)
#define MPI_SIGNED_CHAR    ((MPI_Datatype)0x4c000118)
#define MPI_UNSIGNED_CHAR  ((MPI_Datatype)0x4c000102)
#define MPI_BYTE           ((MPI_Datatype)0x4c00010d)
#define MPI_WCHAR          ((MPI_Datatype)0x4c00040e)
#define MPI_SHORT          ((MPI_Datatype)0x4c000203)
#define MPI_UNSIGNED_SHORT ((MPI_Datatype)0x4c000204)
#define MPI_INT            ((MPI_Datatype)0x4c000405)
#define MPI_UNSIGNED       ((MPI_Datatype)0x4c000406)
#define MPI_LONG           ((MPI_Datatype)0x4c000807)
#define MPI_UNSIGNED_LONG  ((MPI_Datatype)0x4c000808)
#define MPI_FLOAT          ((MPI_Datatype)0x4c00040a)
#define MPI_DOUBLE         ((MPI_Datatype)0x4c00080b)
#define MPI_LONG_DOUBLE    ((MPI_Datatype)0x4c00100c)
#define MPI_LONG_LONG_INT  ((MPI_Datatype)0x4c000809)
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)0x4c000819)
#define MPI_LONG_LONG      MPI_LONG_LONG_INT

typedef struct MPI_Status {
    int count;
    int cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;

} MPI_Status;
*/

#ifndef timing_hpp
#define timing_hpp

#ifndef WIN32
#include "sys/time.h"
#include <iostream>
using std::cout;

class timing
{
private:
    struct timeval timer_;
    struct timezone timezone_;

public:
    timing()
    {}

    inline void tick(double& time_counter)
    {
        gettimeofday(&timer_, &timezone_);
        time_counter -= 1000 * timer_.tv_sec + timer_.tv_usec / 1000;
    }

    inline void tack(double& time_counter)
    {
        gettimeofday(&timer_, &timezone_);
        time_counter += 1000 * timer_.tv_sec + timer_.tv_usec / 1000;
    }

    void reportTimeNeeded(const char* message, double time_counter)
    {
        cout << message << " | " << time_counter << " milliseconds |\n" ;
    }
};

#endif

#ifdef WIN32

#include <time.h>
#include <iostream>
using std::cout;

class timing
{
private:
    double timePassed_;
    long begin_;
    long end_;

public:
    timing()
    {}

    inline void tick(double& time_counter)
    {
        time_counter -= clock();
    }

    inline void tack(double& time_counter)
    {
        time_counter += clock();
    }

    void reportTimeNeeded(const char* message, double time_counter)
    {
        cout << message << " | " << time_counter << " milliseconds |\n" ;
    }
};

#endif
#endif

class CSRdouble;
class ParDiSO;


double* create_matrix_binary ( long int seed, int m, int n, double max, char* filename );
void process_mem_usage(double& vm_usage, double& resident_set, double& cpu_user, double& cpu_sys);
void printdense ( int m, int n, double *mat, char *filename );
int set_up_T (int* DESCC, double* Cmat, int* DESCYTOT, double* ytot, double* respnrm) ;
int set_up_C_hdf5 ( int* DESCC, double* Cmat, int* DESCYTOT, double* ytot, double* respnrm ) ;
int update_C ( int * DESCC, double * Cmat, double update) ;
int set_up_AI (double* AImat, int* DESCAI, int* DESCSOL, double* solution, int* DESCD, double* Dmat, CSRdouble& Asparse, CSRdouble& Btsparse, double sigma);
int set_up_AI_hdf5 ( double* AImat, int* DESCAI, int* DESCYTOT, double* ytot, int* DESCC, double* Cmat, double sigma );
double trace_CZZ(double *mat, int * DESCMAT);
double log_determinant_C ( double *mat, int * DESCMAT ) ;
int read_input(char* filename) ;
int crossvalidate(double* estimates, int* DESCEST);
int crossvalidate_hdf5(double * estimates, int *DESCEST) ;
int make_Sij(int i, int j, int Adim, int Tdim, double * A, double * B_i, double * B_j, double * T_ij);
int make_Sij_parallel_denseB(CSRdouble& A, CSRdouble& BT_i, CSRdouble& B_j, double * T_ij, int lld_T, double * AB_sol_out) ;

void mult_colsA_colsC ( CSRdouble& A, double *B, int lld_B, int Acolstart, int Ancols, int Ccolstart, int Cncols, //input
                        CSRdouble& C, bool trans ) ;
void mult_colsA_colsC_denseC ( CSRdouble& A,  double *B, int lld_B, int Acolstart, int Ancols, int Ccolstart, int Cncols,
                               double *C, int lld_C,  bool sum, double alpha );
void mult_colsAtrans_colsC_denseC ( CSRdouble& A,double *B, int lld_B, int Acolstart, int Ancols, int Ccolstart, int Cncols,
                                    double *C, int lld_C, double alpha ) ;
int set_up_BDY ( int* DESCD, double* Dmat, int* DESCB, double* Bmat, int* DESCYTOT, double* ytot, double* respnrm ) ;
int set_up_D ( int * DESCD, double * Dmat ) ;
void CSR2dense ( CSRdouble& matrix,double *dense ) ;

void create1x2BlockMatrix(CSRdouble& A, CSRdouble& B, // input
                          CSRdouble& C);  // output
void create2x2SymBlockMatrix(CSRdouble& A, CSRdouble& B, CSRdouble& T, // input
                             CSRdouble& C);  // output
void create2x2BlockMatrix(CSRdouble& A, CSRdouble& B, CSRdouble& C, CSRdouble& D, // input
                          CSRdouble& W);  // output
void makeIdentity(int n, CSRdouble& I);
void makeDiag(int n, double lambda, CSRdouble& I);
void errorReport(int number_of_rhs, CSRdouble& A, double* x, double* b);
void solveSystem(CSRdouble& A, double* X, double* B, int pardiso_mtype, int number_of_rhs);
double solveSystemWithDet(CSRdouble& A, double* X, double* B, int pardiso_mtype, int number_of_rhs);
void solveSystemwoFact(CSRdouble& A, double* X, double* B, int pardiso_mtype, int number_of_rhs);
void calculateSchurComplement(CSRdouble& A, int pardiso_mtype, CSRdouble& S);

void dense2CSR ( double *mat, int m, int n, CSRdouble& A ) ;

extern double d_one, d_zero, d_negone;
extern int DLEN_, i_negone, i_zero, i_one, i_two, i_three; // some many used constants
extern int k,l,m,n, blocksize; //dimensions of different matrices
extern int lld_D, Dblocks, Ddim, m_plus,ml_plus,Adim, ydim;
extern int Drows,Dcols;
extern int size, *dims, * position, ICTXT2D, iam;
extern int ntests, maxiterations,datahdf5, copyC;
extern char *SNPdata, *phenodata;
extern char *filenameX, *filenameT, *filenameZ, *filenameY, *TestSet;
extern double gamma_var, phi,epsilon;
extern ParDiSO pardiso_var;
extern int Bassparse_bool;

#endif

