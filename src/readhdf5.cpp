#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string.h>
#include <math.h>
#include <shared_var.h>

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
    H5_DLL hid_t H5Pcreate ( hid_t cls_id );
}

#define MPI_INFO_NULL         ((MPI_Info)0x1c000000)

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

int set_up_AI_hdf5 ( double * AImat, int * DESCAI,int * DESCYTOT, double * ytot, int * DESCC, double * Cmat, double sigma) {

    // Read-in of matrices Z,X and y from file (filename) directly into correct processes and calculation of matrix C
    // Is done strip per strip

    FILE *fX;
    int ni, i,j, info;
    int *DESCZ, *DESCY, *DESCX, *DESCZU, *DESCQRHS, *DESCQSOL;
    double *Zblock, *Xblock, *yblock, *Zublock, *QRHS, *Qsol,*nrmblock, sigma_rec;
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

        pdgemm_ ( "T","N", &i_one, &stripcols,&k,&lambda, ytot, &m_plus,&i_one,DESCYTOT,Zblock,&i_one,&i_one,DESCZ,&d_zero,Zublock,&i_one,&i_one,DESCZU ); //Zu/gamma (in blocks)

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
