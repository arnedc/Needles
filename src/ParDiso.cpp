#include <cassert>
#include <cstring>
#include "shared_var.h"

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
