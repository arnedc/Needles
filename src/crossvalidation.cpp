#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "shared_var.h"
#include <shared_var.h>
#include <hdf5.h>

extern "C" {
    void descinit_ ( int*, int*, int*, int*, int*, int*, int*, int*, int*, int* );
    void blacs_barrier_ ( int*, char* );
    void pdsyrk_ ( char*, char*, int*, int*, double*, double*, int*, int*, int*, double*, double*, int*, int*, int* );
    void pdgemm_ ( char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdtran_ ( int *m, int *n, double *alpha, double *a, int *ia, int *ja, int *desca, double *beta, double *c, int *ic, int *jc, int *descc );
    void pdnrm2_ ( int *n, double *norm2, double *x, int *ix, int *jx, int *descx, int *incx );
    void pdpotrs_ ( char *uplo, int *n, int *nrhs, double *a, int *ia, int *ja, int *desca, double *b, int *ib, int *jb, int *descb, int *info );
    void pddot_( int *n, double *dot, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdcopy_( int *n, double *x, int *ix, int *jx, int *descx, int *incx, double *y, int *iy, int *jy, int *descy, int *incy );
    void pdscal_( int *n, double *a, double *x, int *ix, int *jx, int *descx, int *incx );
    void dgsum2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rdest, int *cdest);
    void dgebs2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda);
    void dgebr2d_(int *ConTxt, char *scope, char *top, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc);
    void dgesd2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rdest, int *cdest );
    void dgerv2d_ ( int *ConTxt, int *m, int *n, double *A, int *lda, int *rsrc, int *csrc );
}

int crossvalidate(double * estimates, int *DESCEST) {
    FILE *fT;
    int ni, i,j, info;
    int *DESCT, *DESCEBV;
    double *Tblock, value, *EBV;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T,pcol,curtest, lld_E;

    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    /*DESCVAL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCVAL==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }*/
    DESCEBV= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCEBV==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }

    pcol= * ( position+1 );
    nstrips= ntests % ( blocksize * * ( dims+1 ) ) ==0 ?  ntests / ( blocksize * * ( dims+1 ) ) : ( ntests / ( blocksize * * ( dims+1 ) ) ) +1; 	// number of strips in which we divide matrix Z' and X'
    stripcols= blocksize * * ( dims+1 ); 												//the number of columns taken into the strip of Z' and X'
    nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;										//number of blocks necessary to store complete column of Z'
    pTblocks= ( nTblocks - *position ) % *dims == 0 ? ( nTblocks- *position ) / *dims : ( nTblocks- *position ) / *dims +1;		//number of blocks necessary per processor
    pTblocks= pTblocks <1? 1:pTblocks;
    lld_T=pTblocks*blocksize;
    lld_E=nstrips*blocksize* *(dims+1);

    descinit_ ( DESCT, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    /*descinit_ ( DESCVAL, &i_one, &ntests, &i_one, &blocksize, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }*/
    descinit_ ( DESCEBV, &lld_E, &i_one, &lld_E, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_E, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of EBV returns info: %d\n",info );
        return info;
    }

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }

    /*values = ( double* ) calloc ( nstrips * blocksize,sizeof ( double ) );
    if ( values==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }*/
    EBV = ( double* ) calloc ( lld_E,sizeof ( double ) );
    if ( EBV==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    fT=fopen ( TestSet,"rb" );
    if ( fT==NULL ) {
        printf ( "Error opening file\n" );
        return -1;
    }

    for ( ni=0; ni<nstrips; ++ni ) {
        if ( ni==nstrips-1 ) {

            free ( Tblock );
            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
                return -1;
            }
        }
        if ( ( nTblocks-1 ) % *dims == *position && k%blocksize !=0 ) {									//last block of row that needs to be read in will be treated seperately
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( ( ( ni * * ( dims+1 ) * blocksize + pcol * blocksize + i ) * ( k+1 ) + blocksize * *position ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading file" );
                    return -1;
                }
                if ( *position==0 )
                    fread ( &value,sizeof ( double ),1,fT );
                else
                    info=fseek ( fT,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pTblocks-1; ++j ) {
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( ( *dims ) -1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading file" );
                        return -1;
                    }
                }
                fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),k%blocksize,fT );
            }
        } else {													//Normal read-in of the matrix from a binary file
            for ( i=0; i<blocksize; ++i ) {
                info=fseek ( fT, ( long ) ( ( ( ni * * ( dims+1 ) * blocksize + pcol * blocksize + i ) * ( k+1 ) + blocksize * *position ) * sizeof ( double ) ),SEEK_SET );
                if ( info!=0 ) {
                    printf ( "Error in setting correct begin position for reading file" );
                    return -1;
                }
                if ( *position==0 )
                    fread ( &value,sizeof ( double ),1,fT );
                else
                    info=fseek ( fT,1L * sizeof ( double ), SEEK_CUR );
                for ( j=0; j < pTblocks; ++j ) {
                    fread ( Tblock + i*pTblocks*blocksize + j*blocksize,sizeof ( double ),blocksize,fT );
                    info=fseek ( fT, ( long ) ( ( * ( dims )-1 ) * blocksize * sizeof ( double ) ),SEEK_CUR );
                    if ( info!=0 ) {
                        printf ( "Error in setting correct begin position for reading file" );
                        return -1;
                    }
                }
            }
        }
        blacs_barrier_ ( &ICTXT2D,"A" );

        curtest=1 + ni * stripcols;

        pdgemm_ ( "T","N",&stripcols,&i_one,&k,&d_one,Tblock,&i_one,&i_one,DESCT,estimates,&m_plus,&i_one,DESCEST,&d_one,EBV,&curtest,&i_one,DESCEBV);
        //pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Zblock,&i_one, &i_one, DESCZ,yblock,&i_one,&i_one,DESCY,&d_one,ytot,&t_plus,&i_one,DESCYTOT ); //Z'y

    }
    /*    for(i=0;i<blocksize*nstrips;++i){
    	  valmean += *(values+i);
    	  EBVmean += *(EBV+i);
    	  MSE += (*values+i - *EBV+i) * (*values+i - *EBV+i);
    	}
        dgsum2d_(ICTXT2D,"ALL","1-tree",&i_one,&i_one,&MSE,&i_one,&i_zero,&i_zero);
        dgsum2d_(ICTXT2D,"ALL","1-tree",&i_one,&i_one,&valmean,&i_one,&i_zero,&i_zero);
        MSE=MSE/ntests;
        valmean=valmean/ntests;
        if (*position==0 && *(position+1)==0){
          dgebs2d_ (&ICTXT2D,"ALL","1-tree",&i_one,&i_one,&valmean,&i_one);
        }
        else{
          dgebr2d_(&ICTXT2D,"ALL","1-tree",&i_one,&i_one,&valmean,&i_one,&i_zero,&i_zero);
        }
        for(i=0;i<blocksize*nstrips;++i){
          TSS += (*values+i - valmean) * (*values+i - valmean);
        }
        dgsum2d_(ICTXT2D,"ALL","1-tree",&i_one,&i_one,&TSS,&i_one,&i_zero,&i_zero);
        R_squared=1-MSE*ntests/TSS;
        */
    fclose(fT);
    if(*position==0 && *(position+1)==0) {
        printdense(ntests,1,EBV,"EBV.txt" );
    }
    blacs_barrier_(&ICTXT2D, "A" );
    free(EBV);
    free(Tblock);
    free(DESCEBV);
    free(DESCT);



    return info;

}

int crossvalidate_hdf5(double * estimates, int *DESCEST) {
    int ni, i,j, info;
    int *DESCT, *DESCEBV;
    double *Tblock, value, *EBV;
    int nTblocks, nstrips, pTblocks, stripcols, lld_T,pcol,curtest, lld_E;

    hid_t       file_id, dataset_geno_id, space_geno_id;
    hid_t	plist_id, memspace_geno;
    herr_t	status;
    hsize_t	dimsm[2], offset[2],count[2], stride[2],block[2];

    int mpinfo  = 0;

    // Creation of important id's to open datasets in HDF5

    plist_id = H5Pcreate ( H5P_FILE_ACCESS );
    H5Pset_fapl_mpio ( plist_id, MPI_COMM_WORLD, mpinfo );

    file_id = H5Fopen ( filenameT, H5F_ACC_RDWR, plist_id );
    dataset_geno_id = H5Dopen ( file_id, TestSet, H5P_DEFAULT );
    space_geno_id=H5Dget_space(dataset_geno_id);

    DESCT= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCT==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }
    /*DESCVAL= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCVAL==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }*/
    DESCEBV= ( int* ) malloc ( DLEN_ * sizeof ( int ) );
    if ( DESCEBV==NULL ) {
        printf ( "unable to allocate memory for descriptor for Z\n" );
        return -1;
    }

    pcol= * ( position+1 );
    nstrips= ntests % ( blocksize * * ( dims+1 ) ) ==0 ?  ntests / ( blocksize * * ( dims+1 ) ) : ( ntests / ( blocksize * * ( dims+1 ) ) ) +1; 	// number of strips in which we divide matrix Z' and X'
    stripcols= blocksize * * ( dims+1 ); 												//the number of columns taken into the strip of Z' and X'
    nTblocks= k%blocksize==0 ? k/blocksize : k/blocksize +1;										//number of blocks necessary to store complete column of Z'
    pTblocks= ( nTblocks - *position ) % *dims == 0 ? ( nTblocks- *position ) / *dims : ( nTblocks- *position ) / *dims +1;		//number of blocks necessary per processor
    pTblocks= pTblocks <1? 1:pTblocks;
    lld_T=pTblocks*blocksize;
    lld_E=nstrips*blocksize* *(dims+1);

    descinit_ ( DESCT, &k, &stripcols, &blocksize, &blocksize, &i_zero, &i_zero, &ICTXT2D, &lld_T, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Z returns info: %d\n",info );
        return info;
    }

    /*descinit_ ( DESCVAL, &i_one, &ntests, &i_one, &blocksize, &i_zero, &i_zero, &ICTXT2D, &i_one, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of matrix Y returns info: %d\n",info );
        return info;
    }*/
    descinit_ ( DESCEBV, &lld_E, &i_one, &lld_E, &i_one, &i_zero, &i_zero, &ICTXT2D, &lld_E, &info );
    if ( info!=0 ) {
        printf ( "Descriptor of EBV returns info: %d\n",info );
        return info;
    }

    Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
    if ( Tblock==NULL ) {
        printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
        return -1;
    }
    dimsm[0]=blocksize;
    dimsm[1]=pTblocks*blocksize;
    memspace_geno = H5Screate_simple(2,dimsm,NULL);

    /*values = ( double* ) calloc ( nstrips * blocksize,sizeof ( double ) );
    if ( values==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }*/
    EBV = ( double* ) calloc ( lld_E,sizeof ( double ) );
    if ( EBV==NULL ) {
        printf ( "unable to allocate memory for Matrix Y\n" );
        return EXIT_FAILURE;
    }

    plist_id = H5Pcreate ( H5P_DATASET_XFER );
    H5Pset_dxpl_mpio ( plist_id, H5FD_MPIO_INDEPENDENT );  // Data access in one process is independent of other processes

    for ( ni=0; ni<nstrips; ++ni ) {
        if(*position >= nTblocks)
            goto CALC;
        if ( ni==nstrips-1 ) {

            free ( Tblock );
            Tblock= ( double* ) calloc ( pTblocks*blocksize*blocksize, sizeof ( double ) );
            if ( Tblock==NULL ) {
                printf ( "Error in allocating memory for a strip of Z in processor (%d,%d)",*position,* ( position+1 ) );
                return -1;
            }
            if((pcol + 1 + (nstrips-1) * *(dims+1))*blocksize <= ntests)
                block[0]=blocksize;
            else if ((pcol + (nstrips-1) * *(dims+1))*blocksize >= ntests)
                block[0]=0;
            else
                block[0]=ntests%blocksize;
        }
        else {
            block[0]=blocksize;
        }
        if ( ( nTblocks-1 ) % *dims == *position && k%blocksize !=0 ) {									//last block of row that needs to be read in will be treated seperately
            offset[0] = ni * *(dims+1) * blocksize + pcol * blocksize;
            offset[1] = *position * blocksize;
            count[0] = 1;
            count[1] = pTblocks-1;
            stride[0] = blocksize * *(dims+1);
            stride[1] = blocksize * *dims;
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if (status<0) {
                printf("selection of geno hyperslab in file was unsuccesful, strip: %d\n",ni);
                return status;
            }
            offset[0] = 0;
            offset[1] = 0;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
            if (status<0) {
                printf("selection of hyperslab in memory was unsuccesful, strip: %d\n",ni);
                return status;
            }

            offset[0] = ni * *(dims+1) * blocksize + pcol * blocksize;
            offset[1] = (nTblocks-1) * blocksize;
            count[0] = 1;
            count[1] = 1;
            stride[0] = blocksize * *(dims+1);
            stride[1] = blocksize * *dims;
            block[1] = k%blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_OR, offset, stride, count, block );
            if (status<0) {
                printf("selection of geno extended hyperslab in file was unsuccesful, strip: %d\n",ni);
                return status;
            }

            offset[0] = 0;
            offset[1] = (pTblocks-1) * blocksize;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_OR, offset, stride, count, block );
            if (status<0) {
                printf("selection of hyperslab in memory was unsuccesful, strip: %d\n",ni);
                return status;
            }
        }
        else {
            offset[0] = ni * *(dims+1) * blocksize + pcol * blocksize;
            offset[1] = *position * blocksize;
            count[0] = 1;
            count[1] = pTblocks;
            stride[0] = blocksize * *(dims+1);
            stride[1] = blocksize * *dims;
            block[1] = blocksize;

            status = H5Sselect_hyperslab ( space_geno_id, H5S_SELECT_SET, offset, stride, count, block );
            if (status<0) {
                printf("selection of geno hyperslab in file was unsuccesful\n");
                return status;
            }

            offset[0] = 0;
            offset[1] = 0;
            stride[0] = blocksize;
            stride[1] = blocksize;

            status = H5Sselect_hyperslab ( memspace_geno, H5S_SELECT_SET, offset, stride, count, block );
            if (status<0) {
                printf("selection of hyperslab in memory was unsuccesful\n");
                return status;
            }
        }
        status= H5Dread ( dataset_geno_id,H5T_NATIVE_DOUBLE_g,memspace_geno,space_geno_id,plist_id,Tblock );
        if (status<0) {
            printf("reading of geno hyperslab was unsuccesful\n");
            return status;
        }
CALC:
        blacs_barrier_ ( &ICTXT2D,"A" );

        curtest=1 + ni * stripcols;

        pdgemm_ ( "T","N",&stripcols,&i_one,&k,&d_one,Tblock,&i_one,&i_one,DESCT,estimates,&m_plus,&i_one,DESCEST,&d_one,EBV,&curtest,&i_one,DESCEBV);
        //pdgemm_ ( "N","T",&m,&i_one,&stripcols,&d_one,Zblock,&i_one, &i_one, DESCZ,yblock,&i_one,&i_one,DESCY,&d_one,ytot,&t_plus,&i_one,DESCYTOT ); //Z'y

    }
    /*    for(i=0;i<blocksize*nstrips;++i){
    	  valmean += *(values+i);
    	  EBVmean += *(EBV+i);
    	  MSE += (*values+i - *EBV+i) * (*values+i - *EBV+i);
    	}
        dgsum2d_(ICTXT2D,"ALL","1-tree",&i_one,&i_one,&MSE,&i_one,&i_zero,&i_zero);
        dgsum2d_(ICTXT2D,"ALL","1-tree",&i_one,&i_one,&valmean,&i_one,&i_zero,&i_zero);
        MSE=MSE/ntests;
        valmean=valmean/ntests;
        if (*position==0 && *(position+1)==0){
          dgebs2d_ (&ICTXT2D,"ALL","1-tree",&i_one,&i_one,&valmean,&i_one);
        }
        else{
          dgebr2d_(&ICTXT2D,"ALL","1-tree",&i_one,&i_one,&valmean,&i_one,&i_zero,&i_zero);
        }
        for(i=0;i<blocksize*nstrips;++i){
          TSS += (*values+i - valmean) * (*values+i - valmean);
        }
        dgsum2d_(ICTXT2D,"ALL","1-tree",&i_one,&i_one,&TSS,&i_one,&i_zero,&i_zero);
        R_squared=1-MSE*ntests/TSS;
        */
    if(*position==0 && *(position+1)==0) {
        printdense(ntests,1,EBV,"EBV.txt" );
    }
    blacs_barrier_(&ICTXT2D, "A" );
    free(EBV);
    free(Tblock);
    free(DESCEBV);
    free(DESCT);
    
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



    return info;

}
