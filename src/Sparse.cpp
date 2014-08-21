#include <cassert>
#include <cstring>
#include "shared_var.h"
#include "ParDiSo.hpp"
#include "CSRdouble.hpp"

void create1x2BlockMatrix(CSRdouble& A, CSRdouble& B, // input
                          CSRdouble& C)  // output
{
    cout << "***  G e n e r a t i n g    m a t r i x   C = [ A      B ] *** " << endl;

    int nrows    = A.nrows;
    int ncols    = A.ncols + B.ncols;
    int nonzeros = A.nonzeros + B.nonzeros;

    int* ic   = new int[nrows + 1];
    int* jc   = new int[nonzeros];
    double* c = new double[nonzeros];

    int nonzero_counter = 0;
    ic[0] = nonzero_counter;
    for (int i = 0; i < A.nrows; i++)
    {
        // push ith row of A
        for (int index = A.pRows[i]; index < A.pRows[i+1]; index++)
        {
            int& j              = A.pCols[index];
            double& a_ij        = A.pData[index];

            c[nonzero_counter] = a_ij;
            jc[nonzero_counter] = j;

            nonzero_counter++;

        }

        // push ith row of B
        for (int index = B.pRows[i]; index < B.pRows[i+1]; index++)
        {
            int& j              = B.pCols[index];
            double& b_ij        = B.pData[index];

            c[nonzero_counter] = b_ij;
            jc[nonzero_counter] = A.ncols + j;

            nonzero_counter++;
        }

        ic[i+1] = nonzero_counter;
    }


    if (nonzero_counter != nonzeros)
        cout << "Nonzeroes do not match, nonzero_counter= " << nonzero_counter << "; nonzeros= " << nonzeros <<endl;


    C.make(nrows, ncols, nonzeros, ic, jc, c);
    C.sortColumns();
    // C.writeToFile("C.csr");
}

void create2x2SymBlockMatrix(CSRdouble& A, CSRdouble& B, CSRdouble& T, // input
                             CSRdouble& C)  // output
{
    assert(A.nrows==B.nrows);
    assert(A.ncols==B.nrows);
    assert(T.ncols==B.ncols);
    assert(T.nrows==B.ncols);

    int nrows    = A.nrows + T.nrows;
    int ncols    = A.ncols + T.ncols;
    int nonzeros = A.nonzeros + B.nonzeros + T.nonzeros;

    int* ic   = new int[nrows + 1];
    int* jc   = new int[nonzeros];
    double* c = new double[nonzeros];

    int nonzero_counter = 0;
    ic[0] = nonzero_counter;
    for (int i = 0; i < A.nrows; i++)
    {
        // push ith row of A
        for (int index = A.pRows[i]; index < A.pRows[i+1]; index++)
        {
            int& j              = A.pCols[index];
            if (j>=i)
            {
                double& a_ij        = A.pData[index];

                c[nonzero_counter] = a_ij;
                jc[nonzero_counter] = j;

                nonzero_counter++;
            }
        }

        // push ith row of B
        for (int index = B.pRows[i]; index < B.pRows[i+1]; index++)
        {
            int& j              = B.pCols[index];
            double& b_ij        = B.pData[index];

            c[nonzero_counter] = b_ij;
            jc[nonzero_counter] = A.ncols + j;

            nonzero_counter++;
        }

        ic[i+1] = nonzero_counter;
    }

    for (int i = 0; i < T.nrows; i++)
    {
        // push ith row of T
        for (int index = T.pRows[i]; index < T.pRows[i+1]; index++)
        {
            int& j              = T.pCols[index];
            double& t_ij        = T.pData[index];

            c[nonzero_counter] = t_ij;
            jc[nonzero_counter] = A.ncols + j;

            nonzero_counter++;
        }

        ic[A.nrows+i+1] = nonzero_counter;
    }
    if (nonzero_counter != nonzeros)
        cout << "Nonzeroes do not match, nonzero_counter= " << nonzero_counter << "; nonzeros= " << nonzeros <<endl;


    C.make(nrows, ncols, nonzeros, ic, jc, c);
    C.sortColumns();
    // C.writeToFile("C.csr");
}


void makeIdentity(int n, CSRdouble& I)
{
    int*    prows = new int[n+1];
    int*    pcols = new int[n];
    double* pdata = new double[n];

    prows[0]      = 0;
    for (int i = 0; i < n; i++)
    {
        prows[i+1]  = prows[i] + 1;
        pcols[i]    = i;
        pdata[i]    = -1.0;
    }

    I.make(n, n, n, prows, pcols, pdata);
}

void solveSystem(CSRdouble& A, double* X, double* B, int pardiso_mtype, int number_of_rhs)
{
    cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
    cout << "@@@ S O L V I N G     A    L I N E A R    S Y S T E M  @@@" << endl;
    cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;


    cout << "*** G e n e r a t i n g    # " << number_of_rhs << "   r h s *** " << endl;

    // initialize pardiso and forward to it minimum number of necessary parameters
    int pardiso_message_level = 0;

    ParDiSO pardiso(pardiso_mtype, pardiso_message_level);

    // Numbers of processors, value of OMP_NUM_THREADS
    int number_of_processors = 1;
    char* var = getenv("OMP_NUM_THREADS");

    if(var != NULL)
        sscanf( var, "%d", &number_of_processors );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }

    pardiso.iparm[3]  = number_of_processors;
    pardiso.iparm[8]  = 0;


    timing secs;
    double initializationTime = 0.0;
    double factorizationTime  = 0.0;
    double solutionTime       = 0.0;



    cout << "S Y M B O L I C     V O O D O O" << endl;

    secs.tick(initializationTime);
    pardiso.init(A, number_of_rhs);
    secs.tack(initializationTime);



    cout << "L U                 F A C T O R I Z A T I O N" << endl;

    secs.tick(factorizationTime);
    pardiso.factorize(A);
    secs.tack(factorizationTime);



    cout << "L U                 B A C K - S U B S T I T U T I O N" << endl;

    secs.tick(solutionTime);
    pardiso.solve(A, X, B);
    secs.tack(solutionTime);


    errorReport(number_of_rhs, A, B, X);
    // writeSolution(number_of_rhs, A.nrows, X);


    cout << "-------------------------------" << endl;
    cout << "T I M I N G         R E P O R T" << endl;
    cout << "-------------------------------" << endl;
    cout.setf(ios::floatfield, ios::scientific);
    cout.precision(2);
    cout << "Initialization phase: " << initializationTime*0.001 << " sec" << endl;
    cout << "Factorization  phase: " << factorizationTime*0.001 << " sec" << endl;
    cout << "Solution       phase: " << solutionTime*0.001 << " sec" << endl;
}

double solveSystemWithDet(CSRdouble& A, double* X, double* B, int pardiso_mtype, int number_of_rhs)
{
    cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
    cout << "@@@ S O L V I N G     A    L I N E A R    S Y S T E M  @@@" << endl;
    cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;


    cout << "*** G e n e r a t i n g    # " << number_of_rhs << "   r h s *** " << endl;

    // initialize pardiso and forward to it minimum number of necessary parameters
    int pardiso_message_level = 0;

    ParDiSO pardiso(pardiso_mtype, pardiso_message_level);

    // Numbers of processors, value of OMP_NUM_THREADS
    int number_of_processors = 1;
    char* var = getenv("OMP_NUM_THREADS");

    if(var != NULL)
        sscanf( var, "%d", &number_of_processors );
    else {
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    }

    pardiso.iparm[3]  = number_of_processors;
    pardiso.iparm[8]  = 0;
    pardiso.iparm[33] = 1;


    timing secs;
    double initializationTime = 0.0;
    double factorizationTime  = 0.0;
    double solutionTime       = 0.0;



    cout << "S Y M B O L I C     V O O D O O" << endl;

    secs.tick(initializationTime);
    pardiso.init(A, number_of_rhs);
    secs.tack(initializationTime);



    cout << "L U                 F A C T O R I Z A T I O N" << endl;

    secs.tick(factorizationTime);
    pardiso.factorize(A);
    secs.tack(factorizationTime);



    cout << "L U                 B A C K - S U B S T I T U T I O N" << endl;

    secs.tick(solutionTime);
    pardiso.solve(A, X, B);
    secs.tack(solutionTime);

    errorReport(number_of_rhs, A, B, X);
    // writeSolution(number_of_rhs, A.nrows, X);


    cout << "-------------------------------" << endl;
    cout << "T I M I N G         R E P O R T" << endl;
    cout << "-------------------------------" << endl;
    cout.setf(ios::floatfield, ios::scientific);
    cout.precision(2);
    cout << "Initialization phase: " << initializationTime*0.001 << " sec" << endl;
    cout << "Factorization  phase: " << factorizationTime*0.001 << " sec" << endl;
    cout << "Solution       phase: " << solutionTime*0.001 << " sec" << endl;
    
    return pardiso.dparm[33];
}

void errorReport(int number_of_rhs, CSRdouble& A, double* b, double* x)
{
    double* r = new double[number_of_rhs * A.nrows];


    for (int i = 0; i < number_of_rhs; i++)
    {
        double* b_i = b + i*A.nrows;
        double* x_i = x + i*A.nrows;
        double* r_i = r + i*A.nrows;

        A.residual(r_i, x_i, b_i);
    }


    double rnorm = 0.0;
    double normb = 0.0;

    for (int i = 0; i < number_of_rhs*A.nrows; i++)
    {
        rnorm += r[i]*r[i];
        normb += b[i]*b[i];
    }


    cout.setf(ios::scientific, ios::floatfield);
    cout.precision(16);

    cout << endl << endl;
    cout << "-----------------------------" << endl;
    cout << "R E S I D U A L     N O R M S" << endl;
    cout << "-----------------------------" << endl;
    cout << "||r_k||_2                = " << sqrt(rnorm) / number_of_rhs << endl;
    cout << "|| b ||_2                = " << sqrt(normb) << endl;
    cout << "||r_k||_2 / || b ||_2    = " << sqrt(rnorm) / sqrt(normb) << endl;
    cout << endl << endl;

    delete[] r;
}

//Convert CSR to column-wise stored dense matrix
void CSR2dense ( CSRdouble& matrix,double *dense ) {
    int i, row;
    row=0;
    for ( i=0; i<matrix.nonzeros; ++i ) {
        while ( i==matrix.pRows[row+1] )
            row++;
        * ( dense + row + matrix.nrows * matrix.pCols[i] ) =matrix.pData[i] ;
    }
}


