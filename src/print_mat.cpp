#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void printdense ( int m, int n, double *mat, char *filename ) {
    FILE *fd;
    fd = fopen ( filename,"w" );
    if ( fd==NULL )
        printf ( "error creating file" );
    int i,j;
    for ( i=0; i<m; ++i ) {
        fprintf ( fd,"[\t" );
        for ( j=0; j<n; ++j ) {
            fprintf ( fd,"%12.8g\t",*(mat+i*n +j));
        }
        fprintf ( fd,"]\n" );
    }
    fclose ( fd );
}
