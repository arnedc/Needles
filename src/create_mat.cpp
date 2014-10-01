#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdlib>
#include "shared_var.h"
#include <shared_var.h>

double *create_matrix ( long seed, int m, int n, double max ) {
    int i,j;
    srand48 ( seed );
    double * matrix;
    matrix = ( double * ) malloc ( m * n * sizeof ( double ) );
    if ( matrix==NULL ) {
        printf ( "Unable to create dense matrix (function: create_matrix)\n" );
        return NULL;
    }
    for ( i=0; i<m; ++i ) {
        for ( j=0; j<n; ++j ) {
            * ( matrix + j +i*n ) = drand48() *2.0*max-max;
        }
    }
    return matrix;
}

double *read_matrix_binary ( int m, int n, char *filename, double *y ) {
    FILE *f;
    int i, info;
    double *mat;

    // mat = new double[m*n];
    mat= ( double* ) calloc ( m* ( n-1 ),sizeof ( double ) );
    if ( mat==NULL ) {
        printf ( "Error in allocating memory for matrix" );
        return NULL;
    }
    strcat ( filename,".bin" );
    f=fopen ( filename,"rb" );
    if ( f==NULL ) {
        printf ( "Error opening file\n" );
        return NULL;
    }
    for ( i=0; i<m; ++i ) {
        info = fread ( y+i,sizeof ( double ),1,f );
        if ( info !=1 ) {
            printf ( "Error reading in file, only %d characters read\n",info );
            return NULL;
        }
        info = fread ( mat+i* ( n-1 ),sizeof ( double ),n-1,f );
        if ( info !=n-1 ) {
            printf ( "Error reading in file, only %d characters read\n",info );
            return NULL;
        }
    }
    return mat;
}

double * create_matrix_binary ( long seed, int m, int n, double max, char * filename ) {
    FILE *f;
    double *mat;
    char binfile[50], txtfile[50];
    int i,info;
    binfile[0]='\0';
    txtfile[0]='\0';
    strcat ( binfile,filename );
    strcat ( txtfile,filename );
    strcat ( binfile,".bin" );
    strcat ( txtfile,".txt" );
    mat=create_matrix ( seed,m,n,max );
    if ( mat==NULL )
        return NULL;

    f=fopen ( binfile,"wb" );			//creation of binary file
    if ( f==NULL ) {
        printf ( "The file could not be opened \n" );
        return NULL;
    }
    for ( i=0; i<m; ++i ) {
        info=fwrite ( mat+i*n,sizeof ( double ),n,f );
        if ( info!=n ) {
            printf ( "Error in writing to file, only %d elements written: \n",info );
            return NULL;
        }
    }
    info=fclose ( f );
    if ( info!=0 ) {
        printf ( "Error in closing of file: %d \n",info );
        return NULL;
    }
    printdense ( m,n,mat,txtfile );		//creation of text-file
    return mat;
}



