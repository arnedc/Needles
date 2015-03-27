#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string.h>
#include <math.h>
#include <shared_var.h>
#include <H5Cpp.h>
#include "CSRdouble.hpp"

extern "C" {
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdtran_ ( int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_ ( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_ ( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_ ( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
    double dnrm2_ ( int *n, double *x, int *incx );
    double ddot_ ( const int *n, const double *x, const int *incx, const double *y, const int *incy );
    void dgemm_ ( const char *transa, const char *transb, const int *m, const int *n, const int *k,
                  const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
                  const double *beta, double *c, const int *ldc );
    H5_DLL hid_t H5Pcreate ( hid_t cls_id );
}

#define MPI_INFO_NULL         ((MPI_Info)0x1c000000)

int set_up_BDY_hdf5 ( int * DESCD, double * Dmat, int * DESCB, double * Bmat, int * DESCYTOT, double * ytot, double *respnrm ) {

    // Read-in of matrices X, Z and T from file (filename[X,Z,T])
    // X and Z are read in entrely by every process
    // T is read in strip by strip (number of rows in each process is at maximum = blocksize)
    // D is constructed directly in a distributed way
    // B is first assembled sparse in root process and afterwards the necessary parts
    // for constructing the distributed Schur complement are sent to each process

    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCZtY, *DESCXtY;
    double *Tblock, *temp, *Y, * ZtY, *XtY;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;
    int nYblocks, pYblocks, lld_Y, Ystart;
    timing secs;
    double totalTime, interTime;

    pcol= * ( position+1 );

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;

    if( pcol==0)
        cout << "Before opening of files of X and Z" << endl;

    Ztsparse.loadFromFile ( filenameZ );
    Ztsparse.transposeIt ( 1 );
    Xtsparse.loadFromFile ( filenameX );
    Xtsparse.transposeIt ( 1 );

    /*XtT_sparse.allocate ( m,k,0 );
    ZtT_sparse.allocate ( l,k,0 );*/

    hid_t       file_id, dataset_geno_id, dataset_pheno_id, space_geno_id;
    hid_t	plist_id, memspace_geno, space_pheno_id, memspace_pheno;
    herr_t	status;
    hsize_t	dimsm[2], offset[2],count[2], stride[2],block[2];

    int mpinfo  = MPI_INFO_NULL;

    // Creation of important id's to open datasets in HDF5

    if( pcol==0)
        cout << "Before setting plist" << endl;

    plist_id = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( plist_id, COMM_DENSE, mpinfo );
    if (mpinfo<0) {
        printf("Something went wrong with setting IO options for HDF5-file, error: %d \n",mpinfo);
        return mpinfo;
    }
    file_id = H5Fopen ( filenameT, H5F_ACC_RDONLY, plist_id );
    if (file_id <0) {
        printf("Something went wrong with opening HDF5-file, error: %d \n",file_id);
        return file_id;
    }
    dataset_geno_id = H5Dopen ( file_id, SNPdata, H5P_DEFAULT );
    if (dataset_geno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_geno_id);
        return dataset_geno_id;
    }
    space_geno_id=H5Dget_space ( dataset_geno_id );
    if (space_geno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",space_geno_id);
        return space_geno_id;
    }

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

    if(*(position+1)==0)
        cout << "Descriptors initialised" << endl;


    // Allocation of memory for the strip of T' in all processes

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }
    dimsm[0]=blocksize * pTblocks*blocksize;
    memspace_geno = H5Screate_simple ( 1,dimsm,NULL );
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

    plist_id = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );  // Data access in one process is independent of other processes

    if ( * ( position + 1 ) ==0 ) {
        cout << "before read-in of Y" << endl;
        dataset_pheno_id = H5Dopen ( file_id, phenodata, H5P_DEFAULT );
        if (dataset_pheno_id <0) {
            printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_pheno_id);
            return dataset_pheno_id;
        }

        status = H5Dread(dataset_pheno_id,H5T_NATIVE_DOUBLE_g,H5S_ALL,H5S_ALL,plist_id,Y);
        if ( status<0 ) {
            printf ( "reading of geno hyperslab was unsuccesful\n" );
            return status;
        }
        else {
            printf ( "Responses were read in correctly.\n" );
        }
        for ( i=n; i<lld_Y; ++i ) {
            * ( Y+i ) =0;
        }

        H5Dclose ( dataset_pheno_id );

    }

    blacs_barrier_ ( &ICTXT2D,"A" );

    hsize_t ccoord[blocksize * blocksize * (pTblocks-1)];

    secs.tick ( totalTime );

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( *(position+1) >= nTblocks )
            goto CALC;
        if ( ni==nstrips-1 ) {

            if ( Tblock != NULL )
                free ( Tblock );
            Tblock=NULL;

            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
            if ( ( nstrips ) *blocksize == n )
                block[0]=blocksize;
            else
                block[0]=n%blocksize;
        }
        else {
            block[0]=blocksize;
        }

        if ( ( nTblocks-1 ) % *(dims+1) == pcol && k%blocksize !=0 ) {
            offset[0] = ni * * ( dims ) * blocksize + *position * blocksize;
            offset[1] = pcol * blocksize;
            count[0] = 1;
            count[1] = pTblocks-1;
            stride[0] = blocksize * * ( dims );
            stride[1] = blocksize * *(dims+1);
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            offset[0] = ni * * ( dims ) * blocksize + *position * blocksize;
            offset[1] = ( nTblocks-1 ) * blocksize;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * * ( dims );
            stride[1] = blocksize * *(dims+1);
            block[1] = k%blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_OR, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            for (int i = 0; i<block[0]; ++i) {
                for(int j = 0; j< (pTblocks-1) * blocksize + block[1]; ++j) {
                    ccoord[i * ((pTblocks-1) * blocksize + block[1]) + j] = j * ((pTblocks-1) * blocksize + block[1]) + i;
                }
            }
            H5Sselect_elements(memspace_geno, H5S_SELECT_SET, block[0] * ((pTblocks-1) * blocksize + block[1]), (const hsize_t *) ccoord);
            status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Tblock );
            if ( status<0 ) {
                printf ( "reading of geno hyperslab was unsuccesful\n" );
                return status;
            }

        } else {
            offset[0] = ni * * ( dims ) * blocksize + *position * blocksize;
            offset[1] = pcol * blocksize;
            count[0] = 1;
            count[1] = pTblocks;
            stride[0] = blocksize * * ( dims );
            stride[1] = blocksize * *(dims+1);
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                return status;
            }

            for (int i = 0; i<block[0]; ++i) {
                for(int j = 0; j< pTblocks * blocksize ; ++j) {
                    ccoord[i * pTblocks * blocksize + j] = j * pTblocks * blocksize + i;
                }
            }
            H5Sselect_elements(memspace_geno, H5S_SELECT_SET, block[0] * pTblocks * blocksize, (const hsize_t *) ccoord);
            status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Tblock );
            if ( status<0 ) {
                printf ( "reading of geno hyperslab was unsuccesful\n" );
                return status;
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

CALC:
        blacs_barrier_ ( &ICTXT2D,"A" );
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
        #pragma omp parallel shared(Ztsparse, Tblock, blocksize, lld_T, ni, dims, position, Bmat, Xtsparse, Adim) private(i, totalTime, secs)
        {
            #pragma omp for schedule (static)
            for ( i=0; i<pTblocks; ++i ) {

                //This function multiplies the correct columns of X' with the blocks of T at the disposal of the process
                // The result is also stored immediately at the correct positions of X'T. (see src/tools.cpp)
                mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize*blocksize, lld_T, ( ni * *dims + *position ) *blocksize, blocksize,
                                          i*blocksize, blocksize, Bmat, Adim, true, 1 );
                /*mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                                       ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );*/
            }

            /*if ( ni==0 && * ( position+1 ) ==0 ) {
                secs.tack ( totalTime );
                cout << "Creation of XtT: " << totalTime * 0.001 << " secs" << endl;
                secs.tick ( totalTime );
            }*/
            //Same as above for calculating Z'T

            #pragma omp for schedule (static)
            for ( i=0; i<pTblocks; ++i ) {

                //This function multiplies the correct columns of X' with the blocks of T at the disposal of the process
                // The result is also stored immediately at the correct positions of X'T. (see src/tools.cpp)
                mult_colsA_colsC_denseC ( Ztsparse, Tblock+i*blocksize*blocksize, lld_T, ( ni * *dims + *position ) *blocksize, blocksize,
                                          i*blocksize, blocksize, Bmat+Xtsparse.nrows, Adim, true, 1 );
                /*mult_colsA_colsC_denseC ( Xtsparse, Tblock+i*blocksize, lld_T, ( * ( dims+1 ) * ni + pcol ) *blocksize, blocksize,
                                       ( *dims * i + *position ) *blocksize, blocksize, XtT_temp, 0 );*/
                /*if ( ni==0 && i==1 && * ( position+1 ) ==0 ) {
                    secs.tack ( totalTime );
                    cout << "Multiplication Zt and Tblock: " << totalTime * 0.001 << " secs" << endl;
                    secs.tick ( totalTime );
                }*/
                //free(ZtT_dense);

            }
        }
        if ( ni==0 && * ( position+1 ) ==0 ) {
            secs.tack ( totalTime );
            cout << "Creation of ZtT: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    H5Dclose ( dataset_geno_id );

    H5Sclose ( memspace_geno );

    H5Sclose ( space_geno_id );

    /*
     * Close property list.
     */
    H5Pclose ( plist_id );

    /*
     * Close the file.
     */
    H5Fclose ( file_id );

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

int set_up_D_hdf5 ( int * DESCD, double * Dmat ) {

    // Read-in of matrices X, Z and T from file (filename[X,Z,T])
    // X and Z are read in entrely by every process
    // T is read in strip by strip (number of rows in each process is at maximum = blocksize)
    // D is constructed directly in a distributed way
    // B is first assembled sparse in root process and afterwards the necessary parts
    // for constructing the distributed Schur complement are sent to each process

    int ni, i,j, info;
    int *DESCT;
    double *Tblock, *temp;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur;
    timing secs;
    double totalTime, interTime;

    pcol= * ( position+1 );

    hid_t       file_id, dataset_geno_id, space_geno_id;
    hid_t	plist_id, memspace_geno;
    herr_t	status;
    hsize_t	dimsm[2], offset[2],count[2], stride[2],block[2];

    int mpinfo  = MPI_INFO_NULL;

    // Creation of important id's to open datasets in HDF5

    plist_id = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( plist_id, COMM_DENSE, mpinfo );
    if (mpinfo<0) {
        printf("Something went wrong with setting IO options for HDF5-file, error: %d \n",mpinfo);
        return mpinfo;
    }

    file_id = H5Fopen ( filenameT, H5F_ACC_RDONLY, plist_id );
    if (file_id <0) {
        printf("Something went wrong with opening HDF5-file, error: %d \n",file_id);
        return file_id;
    }
    dataset_geno_id = H5Dopen ( file_id, SNPdata, H5P_DEFAULT );
    if (dataset_geno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_geno_id);
        return dataset_geno_id;
    }
    space_geno_id=H5Dget_space ( dataset_geno_id );
    if (space_geno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",space_geno_id);
        return space_geno_id;
    }

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

    // Initialisation of descriptor of strips of matrix T'
    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
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

    // Allocation of memory for the strip of T' in all processes

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of T in processor (%d,%d)\n",*position,* ( position+1 ) );
        return -1;
    }
    dimsm[0]=blocksize * pTblocks*blocksize;
    memspace_geno = H5Screate_simple ( 1,dimsm,NULL );


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


    blacs_barrier_ ( &ICTXT2D,"A" );

    hsize_t ccoord[blocksize * blocksize * (pTblocks-1)];

    secs.tick ( totalTime );
    plist_id = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );  // Data access in one process is independent of other processes

    // Set up of matrix D and B per strip of T'

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( *(position+1) >= nTblocks )
            goto CALC;
        if ( ni==nstrips-1 ) {

            if ( Tblock != NULL )
                free ( Tblock );
            Tblock=NULL;

            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)\n",*position,* ( position+1 ) );
                return -1;
            }
            if ( ( nstrips ) *blocksize == n )
                block[0]=blocksize;
            else
                block[0]=n%blocksize;
        }
        else {
            block[0]=blocksize;
        }

        if ( ( nTblocks-1 ) % *(dims+1) == pcol && k%blocksize !=0 ) {
            offset[0] = ni * * ( dims ) * blocksize + *position * blocksize;
            offset[1] = pcol * blocksize;
            count[0] = 1;
            count[1] = pTblocks-1;
            stride[0] = blocksize * * ( dims );
            stride[1] = blocksize * *(dims+1);
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            offset[0] = ni * * ( dims ) * blocksize + *position * blocksize;
            offset[1] = ( nTblocks-1 ) * blocksize;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * * ( dims );
            stride[1] = blocksize * *(dims+1);
            block[1] = k%blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_OR, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            for (int i = 0; i<block[0]; ++i) {
                for(int j = 0; j< (pTblocks-1) * blocksize + block[1]; ++j) {
                    ccoord[i * ((pTblocks-1) * blocksize + block[1]) + j] = j * ((pTblocks-1) * blocksize + block[1]) + i;
                }
            }
            H5Sselect_elements(memspace_geno, H5S_SELECT_SET, block[0] * ((pTblocks-1) * blocksize + block[1]), (const hsize_t *) ccoord);
            status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Tblock );
            if ( status<0 ) {
                printf ( "reading of geno hyperslab was unsuccesful\n" );
                return status;
            }

        } else {
            offset[0] = ni * * ( dims ) * blocksize + *position * blocksize;
            offset[1] = pcol * blocksize;
            count[0] = 1;
            count[1] = pTblocks;
            stride[0] = blocksize * * ( dims );
            stride[1] = blocksize * *(dims+1);
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                return status;
            }

            for (int i = 0; i<block[0]; ++i) {
                for(int j = 0; j< pTblocks * blocksize ; ++j) {
                    ccoord[i * pTblocks * blocksize + j] = j * pTblocks * blocksize + i;
                }
            }
            H5Sselect_elements(memspace_geno, H5S_SELECT_SET, block[0] * pTblocks * blocksize, (const hsize_t *) ccoord);
            status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Tblock );
            if ( status<0 ) {
                printf ( "reading of geno hyperslab was unsuccesful\n" );
                return status;
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

CALC:
        blacs_barrier_ ( &ICTXT2D,"A" );
        pdgemm_ ( "T","N",&k,&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &i_one, &i_one, DESCD ); //T'T
        //pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Tblock,&i_one, &i_one,DESCT, &d_one, Dmat, &t_plus, &t_plus, DESCD );

        if ( ni==0 && * ( position+1 ) ==0 ) {
            secs.tack ( totalTime );
            cout << "dense multiplications of Tblock: " << totalTime * 0.001 << " secs" << endl;
            secs.tick ( totalTime );
        }


    }

    H5Dclose ( dataset_geno_id );

    H5Sclose ( memspace_geno );

    H5Sclose ( space_geno_id );

    /*
     * Close property list.
     */
    H5Pclose ( plist_id );

    /*
     * Close the file.
     */
    H5Fclose ( file_id );

    /*char *Bfile;
    Bfile=(char *) calloc(100,sizeof(char));
    *Bfile='\0';
    sprintf(Bfile,"Bmat_(%d,%d).txt",*position,pcol);
    printdense(Dcols * blocksize,Adim,Bmat,Bfile);*/

    secs.tack ( totalTime );
    if ( Tblock != NULL )
        free ( Tblock );
    Tblock=NULL;

    if ( DESCT!=NULL )
        free ( DESCT );
    DESCT=NULL;

    return 0;
}


int set_up_C_hdf5 ( int * DESCC, double * Cmat, int * DESCYTOT, double * ytot, double *respnrm) {
    FILE *fX;
    int ni, i,j, info;
    int *DESCZ, *DESCY, *DESCX;
    double *Zblock, *Xblock, *yblock, *nrmblock, *temp;
    int nZblocks, nXblocks, nstrips, pZblocks, pXblocks, stripcols, lld_Z, lld_X, pcol, colcur,rowcur;

    hid_t       file_id, dataset_geno_id, dataset_pheno_id, space_geno_id;
    hid_t	plist_id, memspace_geno, space_pheno_id, memspace_pheno;
    herr_t	status;
    hsize_t	dimsm[2], offset[2],count[2], stride[2],block[2];

    int mpinfo  = MPI_INFO_NULL;

    // Creation of important id's to open datasets in HDF5

    plist_id = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( plist_id, MPI_COMM_WORLD, mpinfo );
    if (mpinfo<0) {
        printf("Something went wrong with setting IO options for HDF5-file, error: %d \n",mpinfo);
        return mpinfo;
    }

    file_id = H5Fopen ( filenameT, H5F_ACC_RDONLY, plist_id );
    if (file_id <0) {
        printf("Something went wrong with opening HDF5-file, error: %d \n",file_id);
        return file_id;
    }
    dataset_geno_id = H5Dopen ( file_id, SNPdata, H5P_DEFAULT );
    if (dataset_geno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_geno_id);
        return dataset_geno_id;
    }
    dataset_pheno_id = H5Dopen ( file_id, phenodata, H5P_DEFAULT );
    if (dataset_pheno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_pheno_id);
        return dataset_pheno_id;
    }
    space_geno_id=H5Dget_space ( dataset_geno_id );
    if (space_geno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",space_geno_id);
        return space_geno_id;
    }
    space_pheno_id=H5Dget_space ( dataset_pheno_id );
    if (space_pheno_id <0) {
        printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",space_pheno_id);
        return space_pheno_id;
    }

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

    //The transposed Z and X matrices (column major) are read from their files (Z from hdf5, X from binary file)

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

    // Allocation of memory for the different matrices in all processes and for the memory spaces (hdf5 data transfer)

    Zblock= ( double* ) calloc ( pZblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Zblock==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    dimsm[0]=blocksize;
    dimsm[1]=pZblocks*blocksize;
    memspace_geno = H5Screate_simple ( 2,dimsm,NULL );

    yblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
    if ( yblock==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    dimsm[0]=blocksize;
    dimsm[1]=1;
    memspace_pheno = H5Screate_simple ( 1,dimsm,NULL );

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

    /*char *Cfile;
    Cfile= ( char * ) calloc ( 100,sizeof ( char ) );
    *Cfile='\0';
    sprintf ( Cfile,"Cmat_(%d,%d)_%d.txt",*position,pcol,ni );
    printdense ( Ccols * blocksize,Crows * blocksize, Cmat,Cfile );*/

    fX=fopen ( filenameX,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }
    *respnrm=0.0;
    *nrmblock=0.0;

    plist_id = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );  // Data access in one process is independent of other processes

    // Set up of matrix C per strip of Z and X (every strip contains $blocksize complete rows per processor)

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( *position >= nZblocks )
            goto CALC;
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

            if ( ( pcol + 1 + ( nstrips-1 ) * * ( dims+1 ) ) *blocksize <= n )
                block[0]=blocksize;
            else if ( ( pcol + ( nstrips-1 ) * * ( dims+1 ) ) *blocksize >= n )
                block[0]=0;
            else
                block[0]=n%blocksize;
        } else {
            block[0]=blocksize;
        }
        if ( ( nZblocks-1 ) % *dims == *position && k%blocksize !=0 ) {
            offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
            offset[1] = *position * blocksize;
            count[0] = 1;
            count[1] = pZblocks-1;
            stride[0] = blocksize * * ( dims+1 );
            stride[1] = blocksize * *dims;
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }
            offset[0] = 0;
            offset[1] = 0;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }

            offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
            offset[1] = ( nZblocks-1 ) * blocksize;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * * ( dims+1 );
            stride[1] = blocksize * *dims;
            block[1] = k%blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_OR, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            offset[0] = 0;
            offset[1] = ( pZblocks-1 ) * blocksize;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_OR, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }
        } else {
            offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
            offset[1] = *position * blocksize;
            count[0] = 1;
            count[1] = pZblocks;
            stride[0] = blocksize * * ( dims+1 );
            stride[1] = blocksize * *dims;
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                return status;
            }

            offset[0] = 0;
            offset[1] = 0;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful\n" );
                return status;
            }
        }
        status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Zblock );
        if ( status<0 ) {
            printf ( "reading of geno hyperslab was unsuccesful\n" );
            return status;
        }
        if ( *position==0 ) {

            offset[0] = ni * blocksize * * ( dims+1 ) + pcol * blocksize;
            offset[1] = 0;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * *dims;
            stride[1] = 1;
            block[1] = 1;

            status = H5Sselect_hyperslab ( space_pheno_id, H5S_SELECT_SET, offset, stride, count,block );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }
            offset[0] = 0;
            offset[1] = 0;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * *dims;
            stride[1] = 1;
            block[1] = 1;

            status = H5Sselect_hyperslab ( memspace_pheno, H5S_SELECT_SET, offset, stride, count,block );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }

            status=H5Dread ( dataset_pheno_id,H5T_NATIVE_DOUBLE_g,memspace_pheno,space_pheno_id,plist_id,yblock );
            if ( status<0 ) {
                printf ( "reading of pheno hyperslab was unsuccesful\n" );
                return -1;
            }

        }
        /*char *Zfile;
        Zfile=(char *) calloc(100,sizeof(char));
        *Zfile='\0';
        sprintf(Zfile,"Zmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(blocksize,blocksize * pZblocks, Zblock,Zfile);*/

        /*char *FileY;
            FileY=(char *) calloc(100,sizeof(char));
            *FileY='\0';
            sprintf(FileY,"Ymat_(%d,%d)_%d.txt",*position,pcol,ni);
            printdense(blocksize,1, yblock,FileY);*/

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
                if ( m>*position * blocksize ) {
                    info=fseek ( fX, ( long ) ( ( m - blocksize * ( ( pXblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
            }
        }
        /*char *fileX;
        fileX=(char *) calloc(100,sizeof(char));
        *fileX='\0';
        sprintf(fileX,"Xmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(blocksize,blocksize * pXblocks, Xblock,fileX);*/

CALC:
        blacs_barrier_ ( &ICTXT2D,"A" );

        // End of read-in

        // Creation of symmetric matrix C in every process per block of Z, X and y
        // Only the upper triangular part is stored as a full matrix distributed over all processes (column major)

        pdsyrk_ ( "U","N",&k,&stripcols,&d_one, Zblock,&i_one, &i_one,DESCZ, &d_one, Cmat, &m_plus, &m_plus, DESCC ); //Z'Z

        pdgemm_ ( "N","T",&k,&i_one,&stripcols,&d_one,Zblock,&i_one, &i_one, DESCZ,yblock,&i_one,&i_one,DESCY,&d_one,ytot,&m_plus,&i_one,DESCYTOT ); //Z'y

        pdsyrk_ ( "U","N",&m,&stripcols,&d_one, Xblock,&i_one, &i_one,DESCX, &d_one, Cmat, &i_one, &i_one, DESCC ); //X'X

        pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,yblock,&i_one,&i_one,DESCY,&d_one,ytot,&i_one,&i_one,DESCYTOT ); //X'y

        pdgemm_ ( "N","T",&m,&k,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,Zblock,&i_one,&i_one,DESCZ,&d_one,Cmat,&i_one,&m_plus,DESCC ); //X'Z

        //y'y is square of 2-norm (used for calculation of sigma)

        pdnrm2_ ( &stripcols,nrmblock,yblock,&i_one,&i_one,DESCY,&i_one );
        *respnrm += *nrmblock * *nrmblock;

        /*char *Cfile;
        Cfile=(char *) calloc(100,sizeof(char));
        *Cfile='\0';
        sprintf(Cfile,"Cmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(Ccols * blocksize,Crows * blocksize, Cmat,Cfile);*/


        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }
    /*info=fclose ( fZ );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }*/

    H5Dclose ( dataset_geno_id );
    H5Dclose ( dataset_pheno_id );
    H5Sclose ( memspace_geno );
    H5Sclose ( memspace_pheno );
    H5Sclose ( space_geno_id );
    H5Sclose ( space_pheno_id );
    /*
     * Close property list.
     */
    H5Pclose ( plist_id );

    /*
     * Close the file.
     */
    H5Fclose ( file_id );
    free ( DESCX );
    free ( DESCY );
    free ( DESCZ );
    free ( Zblock );
    free ( Xblock );
    free ( yblock );
    free ( nrmblock );
    return 0;

}


int set_up_AI_hdf5 ( double * AImat, int * DESCDENSESOL, double * densesol, int * DESCD, double * Dmat, CSRdouble &Asparse, int *DESCB, double *Bmat, double sigma ) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    int ni, i,j, info;
    int *DESCT, *DESCY, *DESCTD, *DESCZU, *DESCQRHS, *DESCQSOL, *DESCQDENSE;
    double *Tblock, *yblock, *Tdblock, *QRHS, *Qsol,*nrmblock, sigma_rec, phi_rec, gamma_rec, *Zu, *Qdense;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T, pcol, colcur,rowcur, lld_Y;
    double vm_usage, resident_set, cpu_sys, cpu_user;

    MPI_Status MPI_status;

    CSRdouble Xtsparse, Ztsparse,XtT_sparse,ZtT_sparse,XtT_temp, ZtT_temp;

    hid_t       file_id, dataset_geno_id, dataset_pheno_id, space_geno_id;         /* file and dataset identifiers */
    hid_t	plist_id, memspace_geno, space_pheno_id, memspace_pheno;        /* property list identifier( access template) */
    herr_t	status;
    hsize_t	dimsm[2], offset[2],count[2], stride[2],block[2];

    MPI_Info mpinfo  = MPI_INFO_NULL;

    nstrips= n % ( blocksize * * ( dims+1 ) ) ==0 ?  n / ( blocksize * * ( dims+1 ) ) : ( n / ( blocksize * * ( dims+1 ) ) ) +1; 	// number of strips in which we divide matrix Z' and X'
    stripcols= blocksize * * ( dims+1 ); 												//the number of columns taken into the strip of Z' and X'

    sigma_rec=1/sigma;
    phi_rec=1/phi;
    gamma_rec=1/gamma_var;

    lld_Y= * ( dims+1 ) * nstrips * blocksize;

    plist_id = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( plist_id, MPI_COMM_WORLD, mpinfo );

    file_id = H5Fopen ( filenameT, H5F_ACC_RDWR, plist_id );


    if ( iam !=0 ) {

        dataset_geno_id = H5Dopen ( file_id, SNPdata, H5P_DEFAULT );
        space_geno_id=H5Dget_space ( dataset_geno_id );

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

        dimsm[0]=blocksize;
        dimsm[1]=pTblocks*blocksize;
        memspace_geno = H5Screate_simple ( 2,dimsm,NULL );

        plist_id = H5Pcreate ( H5P_DATASET_XFER );
        H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );
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
            dataset_pheno_id = H5Dopen ( file_id, phenodata, H5P_DEFAULT );
            if (dataset_pheno_id <0) {
                printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_pheno_id);
                return dataset_pheno_id;
            }


            status = H5Dread(dataset_pheno_id,H5T_NATIVE_DOUBLE_g,H5S_ALL,H5S_ALL,plist_id,yblock);
            if ( status<0 ) {
                printf ( "reading of pheno hyperslab was unsuccesful\n" );
                return status;
            }

            H5Dclose ( dataset_pheno_id );

            MPI_Recv ( Zu,n, MPI_DOUBLE,0,n,MPI_COMM_WORLD,&MPI_status );
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
            if ( *position >= nTblocks )
                goto CALC;
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
                if ( ( nstrips ) *blocksize == n )
                    block[0]=blocksize;
                else
                    block[0]=n%blocksize;
            }
            else {
                block[0]=blocksize;
            }

            //Creation of matrix T in every process

            if ( ( nTblocks-1 ) % *dims == *position && k%blocksize !=0 ) {
                offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
                offset[1] = *position * blocksize;
                count[0] = 1;
                count[1] = pTblocks-1;
                stride[0] = blocksize * * ( dims+1 );
                stride[1] = blocksize * *dims;
                block[1] = blocksize;

                status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
                if ( status<0 ) {
                    printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                    return status;
                }
                offset[0] = 0;
                offset[1] = 0;
                stride[0] = blocksize;
                stride[1] = blocksize;

                status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
                if ( status<0 ) {
                    printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                    return status;
                }

                offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
                offset[1] = ( nTblocks-1 ) * blocksize;
                count[0] = 1;
                count[1] = 1;
                stride[0] = blocksize * * ( dims+1 );
                stride[1] = blocksize * *dims;
                block[1] = k%blocksize;

                status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_OR, offset, stride, count, block );
                if ( status<0 ) {
                    printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                    return status;
                }

                offset[0] = 0;
                offset[1] = ( pTblocks-1 ) * blocksize;
                stride[0] = blocksize;
                stride[1] = blocksize;

                status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_OR, offset, stride, count, block );
                if ( status<0 ) {
                    printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                    return status;
                }
            } else {
                offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
                offset[1] = *position * blocksize;
                count[0] = 1;
                count[1] = pTblocks;
                stride[0] = blocksize * * ( dims+1 );
                stride[1] = blocksize * *dims;
                block[1] = blocksize;

                status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
                if ( status<0 ) {
                    printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                    return status;
                }

                offset[0] = 0;
                offset[1] = 0;
                stride[0] = blocksize;
                stride[1] = blocksize;

                status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
                if ( status<0 ) {
                    printf ( "selection of hyperslab in memory was unsuccesful\n" );
                    return status;
                }
            }
            status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Tblock );
            if ( status<0 ) {
                printf ( "reading of geno hyperslab was unsuccesful\n" );
                return status;
            }

CALC:
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

        H5Dclose ( dataset_geno_id );
        H5Sclose ( memspace_geno );
        H5Sclose ( space_geno_id );
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

            MPI_Recv ( QRHS + ydim,Adim, MPI_DOUBLE,0,ydim,MPI_COMM_WORLD,&MPI_status ); // here we get A^{-1} x QRHS (sparse part)
            MPI_Recv ( QRHS + 2*ydim,Adim, MPI_DOUBLE,0,2*ydim,MPI_COMM_WORLD,&MPI_status );



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
        dataset_pheno_id = H5Dopen ( file_id, phenodata, H5P_DEFAULT );
        if (dataset_pheno_id <0) {
            printf("Something went wrong with opening dataset in HDF5-file, error: %d \n",dataset_pheno_id);
            return dataset_pheno_id;
        }
        plist_id = H5Pcreate ( H5P_DATASET_XFER );
        H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );
        status = H5Dread(dataset_pheno_id,H5T_NATIVE_DOUBLE_g,H5S_ALL,H5S_ALL,plist_id,yblock);
        if ( status<0 ) {
            printf ( "reading of geno hyperslab was unsuccesful\n" );
            return status;
        }

        H5Dclose ( dataset_pheno_id );
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

            MPI_Recv ( Tdblock,stripcols, MPI_DOUBLE,1,ni,MPI_COMM_WORLD,&MPI_status );
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
        MPI_Recv ( QRHS + Adim,k, MPI_DOUBLE,1,k,MPI_COMM_WORLD,&MPI_status );
        MPI_Recv ( QRHS + Adim +ydim,k, MPI_DOUBLE,1,2*k,MPI_COMM_WORLD,&MPI_status );
        MPI_Recv ( QRHS + Adim+2*ydim,k, MPI_DOUBLE,1,3*k,MPI_COMM_WORLD,&MPI_status );

        printf ( "Solving system AQsol_1,2 = Q_1,2 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol,QRHS+ydim, 2, 1 );
        printf ( "Solving system AQsol_1,3 = Q_1,3 on process 0\n" );
        solveSystemwoFact ( Asparse, Qsol+ydim,QRHS+2*ydim, 2, 1 );
        //printf("AQsol=QRHS_2 is solved\n");
        MPI_Send ( Qsol,Adim, MPI_DOUBLE,1,ydim, MPI_COMM_WORLD );
        MPI_Send ( Qsol+ydim,Adim, MPI_DOUBLE,1,ydim*2, MPI_COMM_WORLD );

        double * Qtemp = new double [2 * Adim];

        MPI_Recv ( Qsol+ydim,ydim, MPI_DOUBLE,1,ydim,MPI_COMM_WORLD,&MPI_status );
        MPI_Recv ( Qsol + 2*ydim,ydim, MPI_DOUBLE,1,2*ydim,MPI_COMM_WORLD,&MPI_status );

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

    /*
     * Close property list.
     */
    H5Pclose ( plist_id );

    /*
     * Close the file.
     */
    H5Fclose ( file_id );

    MPI_Barrier ( MPI_COMM_WORLD );

    return 0;
}

int set_up_AI_hdf5_ori ( double * AImat, int * DESCAI,int * DESCYTOT, double * ytot, int * DESCC, double * Cmat, double sigma) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    FILE *fX;
    int ni, i,j, info;
    int *DESCZ, *DESCY, *DESCX, *DESCZU, *DESCQRHS, *DESCQSOL;
    double *Zblock, *Xblock, *yblock, *Zublock, *QRHS, *Qsol,*nrmblock, sigma_rec, gamma_rec;
    int nZblocks, nXblocks, nstrips, pZblocks, pXblocks, stripcols, lld_Z, lld_X, pcol, colcur,rowcur;

    hid_t       file_id, dataset_geno_id, dataset_pheno_id, space_geno_id;         /* file and dataset identifiers */
    hid_t	plist_id, memspace_geno, space_pheno_id, memspace_pheno;        /* property list identifier( access template) */
    herr_t	status;
    hsize_t	dimsm[2], offset[2],count[2], stride[2],block[2];

    MPI_Info mpinfo  = MPI_INFO_NULL;

    // Creation of important id's to open datasets in HDF5

    plist_id = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( plist_id, MPI_COMM_WORLD, mpinfo );

    file_id = H5Fopen ( filenameT, H5F_ACC_RDWR, plist_id );
    dataset_geno_id = H5Dopen ( file_id, SNPdata, H5P_DEFAULT );
    dataset_pheno_id = H5Dopen ( file_id, phenodata, H5P_DEFAULT );
    space_geno_id=H5Dget_space ( dataset_geno_id );
    space_pheno_id=H5Dget_space ( dataset_pheno_id );

    // Initialisation of descriptors of different matrices

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
        printf ( "unable to allocate memory for descriptor for X\n" );
        return -1;
    }
    DESCZU= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCZU==NULL ) {
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

    //The transposed Z and X matrices (column major) are read from their files (Z from hdf5, X from binary file)

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
    lld_X=pXblocks*blocksize;														//local leading dimension of the strip of Z (different from processor to processor)
    sigma_rec=1/sigma;
    gamma_rec=1/gamma_var;

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
    descinit_ ( DESCZU, &i_one, &stripcols, &i_one, &blocksize, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQRHS, &Ddim, &i_two, &blocksize, &i_two, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }
    descinit_ ( DESCQSOL, &Ddim, &i_two, &blocksize, &i_two, &i_zero, &i_zero, &ICTXT2D, &lld_D, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    // Allocation of memory for the different matrices in all processes

    Zblock= ( double* ) calloc ( pZblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Zblock==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    dimsm[0]=blocksize;
    dimsm[1]=pZblocks*blocksize;
    memspace_geno = H5Screate_simple ( 2,dimsm,NULL );

    yblock = ( double* ) calloc ( blocksize,sizeof ( double ) );
    if ( yblock==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }
    dimsm[0]=blocksize;
    dimsm[1]=1;
    memspace_pheno = H5Screate_simple ( 1,dimsm,NULL );

    Zublock = ( double* ) calloc ( blocksize,sizeof ( double ) );
    if ( Zublock==NULL ) {
        printf ( "unable to allocate memory for Matrix Zu\n" );
        return EXIT_FAILURE;
    }
    Xblock= ( double* ) calloc ( pXblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Xblock==NULL ) {
        printf ( "Error in allocating memory for a strip of X in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    QRHS= ( double * ) calloc ( Drows * blocksize * 2,sizeof ( double ) );
    if ( QRHS==NULL ) {
        printf ( "Error in allocating memory for QRHS in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    Qsol= ( double * ) calloc ( Drows * blocksize * 2,sizeof ( double ) );
    if ( Qsol==NULL ) {
        printf ( "Error in allocating memory for QRHS in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    nrmblock = ( double* ) calloc ( 1,sizeof ( double ) );
    if ( nrmblock==NULL ) {
        printf ( "unable to allocate memory for norm\n" );
        return EXIT_FAILURE;
    }

    fX=fopen ( filenameX,"rb" );
    if ( fX==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }
    *nrmblock=0.0;

    plist_id = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );

    // Set up of matrices used for Average information matrix calculation per strip of Z and X (one strip consists of $blocksize complete rows)

    for ( ni=0; ni<nstrips; ++ni ) {

        //Creation of matrix Z in every process
        if ( *position >= nZblocks )
            goto CALC;
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

            if ( ( pcol + 1 + ( nstrips-1 ) * * ( dims+1 ) ) *blocksize <= n )
                block[0]=blocksize;
            else if ( ( pcol + ( nstrips-1 ) * * ( dims+1 ) ) *blocksize >= n )
                block[0]=0;
            else
                block[0]=n%blocksize;
        } else {
            block[0]=blocksize;
        }
        if ( ( nZblocks-1 ) % *dims == *position && k%blocksize !=0 ) {
            offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
            offset[1] = *position * blocksize;
            count[0] = 1;
            count[1] = pZblocks-1;
            stride[0] = blocksize * * ( dims+1 );
            stride[1] = blocksize * *dims;
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }
            offset[0] = 0;
            offset[1] = 0;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }

            offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
            offset[1] = ( nZblocks-1 ) * blocksize;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * * ( dims+1 );
            stride[1] = blocksize * *dims;
            block[1] = k%blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_OR, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni );
                return status;
            }

            offset[0] = 0;
            offset[1] = ( pZblocks-1 ) * blocksize;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_OR, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful, strip: %d\n",ni );
                return status;
            }
        } else {
            offset[0] = ni * * ( dims+1 ) * blocksize + pcol * blocksize;
            offset[1] = *position * blocksize;
            count[0] = 1;
            count[1] = pZblocks;
            stride[0] = blocksize * * ( dims+1 );
            stride[1] = blocksize * *dims;
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of geno hyperslab in file was unsuccesful\n" );
                return status;
            }

            offset[0] = 0;
            offset[1] = 0;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
            if ( status<0 ) {
                printf ( "selection of hyperslab in memory was unsuccesful\n" );
                return status;
            }
        }
        status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Zblock );
        if ( status<0 ) {
            printf ( "reading of geno hyperslab was unsuccesful\n" );
            return status;
        }
        if ( *position==0 ) {

            offset[0] = ni * blocksize * * ( dims+1 ) + pcol * blocksize;
            offset[1] = 0;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * *dims;
            stride[1] = 1;
            block[1] = 1;

            status = H5Sselect_hyperslab ( space_pheno_id, H5S_SELECT_SET, offset, stride, count,block );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }
            offset[0] = 0;
            offset[1] = 0;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * *dims;
            stride[1] = 1;
            block[1] = 1;

            status = H5Sselect_hyperslab ( memspace_pheno, H5S_SELECT_SET, offset, stride, count,block );
            if ( status<0 ) {
                printf ( "selection of pheno hyperslab in file was unsuccesful\n" );
                return -1;
            }

            status=H5Dread ( dataset_pheno_id,H5T_NATIVE_DOUBLE_g,memspace_pheno,space_pheno_id,plist_id,yblock );
            if ( status<0 ) {
                printf ( "reading of pheno hyperslab was unsuccesful\n" );
                return -1;
            }

        }
        /*char *FileZ;
        FileZ=(char *) calloc(100,sizeof(char));
        *FileZ='\0';
        sprintf(FileZ,"Zmat_(%d,%d)_%d.txt",*position,pcol,ni);
        printdense(blocksize,blocksize * pZblocks, Zblock,FileZ);*/
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
                if ( m>*position * blocksize ) {
                    info=fseek ( fX, ( long ) ( ( m - blocksize * ( ( pXblocks-1 ) * *dims + *position +1 ) ) * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading X file\nprocessor (%d,%d), error: %d \n", *position,pcol,info );
                        return -1;
                    }
                }
            }
        }
CALC:
        blacs_barrier_ ( &ICTXT2D,"A" );

        // End of read-in

        // Creation of matrix needed for calculation of AI matrix distributed over every process per block of Z, X and y

        pdgemm_ ( "T","N", &i_one, &stripcols,&k,&gamma_rec, ytot, &m_plus,&i_one,DESCYTOT,Zblock,&i_one,&i_one,DESCZ,&d_zero,Zublock,&i_one,&i_one,DESCZU ); //Zu/gamma (in blocks)

        pdgemm_ ( "N","T",&k,&i_one,&stripcols,&sigma_rec,Zblock,&i_one, &i_one, DESCZ,yblock,&i_one,&i_one,DESCY,&d_one,QRHS,&m_plus,&i_one,DESCQRHS ); //Z'y/sigma

        pdgemm_ ( "N","T",&m,&i_one,&stripcols,&sigma_rec,Xblock,&i_one, &i_one, DESCX,yblock,&i_one,&i_one,DESCY,&d_one,QRHS,&i_one,&i_one,DESCQRHS ); //X'y/sigma

        pdgemm_ ( "N","T",&k,&i_one,&stripcols,&d_one,Zblock,&i_one, &i_one, DESCZ,Zublock,&i_one,&i_one,DESCZU,&d_one,QRHS,&m_plus,&i_two,DESCQRHS ); //Z'Zu/gamma

        pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Xblock,&i_one, &i_one, DESCX,Zublock,&i_one,&i_one,DESCZU,&d_one,QRHS,&i_one,&i_two,DESCQRHS ); //X'Zu/gamma


        // Q'Q is calculated and stored directly in AI matrix (complete in every proces)

        pdnrm2_ ( &stripcols,nrmblock,yblock,&i_one,&i_one,DESCY,&i_one );
        *AImat += *nrmblock * *nrmblock/sigma/sigma; 								//y'y/sigma²
        pdnrm2_ ( &stripcols,nrmblock,Zublock,&i_one,&i_one,DESCZU,&i_one );
        * ( AImat + 3 ) += *nrmblock * *nrmblock;								//u'Z'Zu/gamma²
        pddot_ ( &stripcols,nrmblock,Zublock,&i_one,&i_one,DESCZU,&i_one,yblock,&i_one,&i_one,DESCY,&i_one );
        * ( AImat + 1 ) += *nrmblock /sigma;							//y'Zu/gamma/sigma
        * ( AImat + 2 ) += *nrmblock /sigma;							//y'Zu/gamma/sigma
        blacs_barrier_ ( &ICTXT2D,"A" );
    }

    // In Qsol we calculate the solution of C * Qsol = QRHS, but we still need QRHS a bit further

    pdcopy_ ( &Ddim,QRHS,&i_one,&i_two,DESCQRHS,&i_one,Qsol,&i_one,&i_two,DESCQSOL,&i_one );
    pdcopy_ ( &Ddim,ytot,&i_one,&i_one,DESCYTOT,&i_one,Qsol,&i_one,&i_one,DESCQSOL,&i_one );
    pdscal_ ( &Ddim,&sigma_rec,Qsol,&i_one,&i_one,DESCQSOL,&i_one );
    pdpotrs_ ( "U",&Ddim,&i_one,Cmat,&i_one,&i_one,DESCC,Qsol,&i_one,&i_two,DESCQSOL,&info );
    if ( info!=0 )
        printf ( "Parallel Cholesky solution for Q was unsuccesful, error returned: %d\n",info );

    // AImat = (Q'Q - QRHS' * Qsol) / 2 / sigma

    pdgemm_ ( "T","N",&i_two,&i_two,&Ddim,&d_negone,QRHS,&i_one,&i_one,DESCQRHS,Qsol,&i_one,&i_one,DESCQSOL,&d_one, AImat,&i_one,&i_one,DESCAI );

    for ( i=0; i<4; ++i )
        * ( AImat + i ) = * ( AImat + i ) / 2 / sigma;

    info=fclose ( fX );
    if ( info!=0 ) {
        printf ( "Error in closing open streams" );
        return -1;
    }

    H5Dclose ( dataset_geno_id );
    H5Dclose ( dataset_pheno_id );
    H5Sclose ( memspace_geno );
    H5Sclose ( memspace_pheno );
    H5Sclose ( space_geno_id );
    H5Sclose ( space_pheno_id );
    /*
     * Close property list.
     */
    H5Pclose ( plist_id );

    /*
     * Close the file.
     */
    H5Fclose ( file_id );

    free ( DESCQRHS );
    free ( DESCQSOL );
    free ( DESCX );
    free ( DESCY );
    free ( DESCZ );
    free ( DESCZU );
    free ( Zblock );
    free ( Xblock );
    free ( yblock );
    free ( nrmblock );
    free ( QRHS );
    free ( Qsol );
    free ( Zublock );

    return 0;
}
