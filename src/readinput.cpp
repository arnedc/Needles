#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <iosfwd>
#include <string>
#include "shared_var.h"
#include <shared_var.h>
#include <iostream>
#include <fstream>
using namespace std;

int read_input ( char * filename ) {
    std::ifstream inputfile ( filename );
    string line;
    bool fixedfile_bool=false, randfile_bool=false, SNPdatafile_bool=false, lam_bool=false, phenopath_bool=false, testpath_bool=false;
    bool eps_bool=false, maxit_bool=false, blocksize_bool=false, testfile_bool=false, ntest_bool=false, h5_bool=false, genopath_bool=false, rand_bool=false;
    bool Phenodatafile_bool=false,phi_bool=false;

    filenameT= ( char* ) calloc ( 100,sizeof ( char ) );
    filenameX= ( char* ) calloc ( 100,sizeof ( char ) );
    filenameZ= ( char* ) calloc ( 100,sizeof ( char ) );
    filenameY= ( char * ) calloc ( 100,sizeof ( char ) );
    TestSet= ( char* ) calloc ( 100,sizeof ( char ) );
    SNPdata = ( char* ) calloc ( 100,sizeof ( char ) );
    phenodata = ( char* ) calloc ( 100,sizeof ( char ) );

    gamma_var=0.01;
    phi=0.01;
    blocksize=64;
    epsilon=0.01;
    maxiterations=20;
    ntests=0;
    copyC=0;
    k=-1, l=-1,m=-1,n=-1;
    Bassparse_bool=0;
    datahdf5=0;


    while ( std::getline ( inputfile,line ) ) {
        if ( line=="#Observations" ) {
            std::getline ( inputfile,line );
            n=atoi ( line.c_str() );
        } else if ( line=="#SNPs" ) {
            std::getline ( inputfile,line );
            k=atoi ( line.c_str() );
        } else if ( line=="#FixedEffects" ) {
            std::getline ( inputfile,line );
            m=atoi ( line.c_str() );
        } else if ( line=="#RandomEffects" ) {
            std::getline ( inputfile,line );
            l=atoi ( line.c_str() );
        } else if ( line=="#FileFixedEffects" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameX,100 );
            fixedfile_bool=true;
        } else if ( line=="#FileRandomEffects" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameZ,100 );
            randfile_bool=true;
        } else if ( line=="#DataFileHDF5" ) {
            std::getline ( inputfile,line );
            datahdf5=atoi ( line.c_str() );
            h5_bool=true;
        } else if ( line=="#SNPDataFile" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameT,100 );
            SNPdatafile_bool=true;
        } else if ( line=="#PhenoDataFile" ) {
            std::getline ( inputfile,line );
            line.copy ( filenameY,100 );
            Phenodatafile_bool=true;
        } else if ( line=="#PathGeno" ) {
            std::getline ( inputfile,line );
            line.copy ( SNPdata,100 );
            genopath_bool=true;
        } else if ( line=="#PathPheno" ) {
            std::getline ( inputfile,line );
            line.copy ( phenodata,100 );
            phenopath_bool=true;
        } else if ( line=="#PathTest" ) {
            std::getline ( inputfile,line );
            line.copy ( TestSet,100 );
            testpath_bool=true;
        } else if ( line=="#TestFile" ) {
            std::getline ( inputfile,line );
            line.copy ( TestSet,100 );
            testfile_bool=true;
        } else if ( line=="#TestSamples" ) {
            std::getline ( inputfile,line );
            ntests=atoi ( line.c_str() );
            ntest_bool=true;
        } else if ( line=="#KeepCopyOfCMatrix" ) {
            std::getline ( inputfile,line );
            copyC=atoi ( line.c_str() );
        } else if ( line=="#BlockSize" ) {
            std::getline ( inputfile,line );
            blocksize=atoi ( line.c_str() );
            blocksize_bool=true;
        } else if ( line=="#Gamma" ) {
            std::getline ( inputfile,line );
            gamma_var=atof ( line.c_str() );
            lam_bool=true;
        } else if ( line=="#Phi" ) {
            std::getline ( inputfile,line );
            phi=atof ( line.c_str() );
            phi_bool=true;
        } else if ( line=="#Epsilon" ) {
            std::getline ( inputfile,line );
            epsilon=atof ( line.c_str() );
            eps_bool=true;
        } else if ( line=="#MaximumIterations" ) {
            std::getline ( inputfile,line );
            maxiterations=atoi ( line.c_str() );
            maxit_bool=true;
        } else if ( line[0]=='/' || line.size() ==0 ) {}
        else {
            printf ( "Unknown parameter in inputfile, the following line was ignored: \n" );
            printf ( "%s\n",line.c_str() );
        }
    }
    if ( *position==0 && * ( position+1 ) ==0 ) {
        if ( n>=0 ) {
            if ( k>=0 ) {
                if ( m>=0 ) {
                    if ( l>=0 ) {
                        if ( *position==0 && * ( position+1 ) ==0 ) {
                            printf ( "number of observations:   \t %d\n", n );
                            printf ( "number of SNP effects:    \t %d\n", k );
                            printf ( "number of random effects: \t %d\n", l );
                            printf ( "number of fixed effects:  \t %d\n", m );
                        }


                    } else {
                        printf ( "ERROR: number of random effects was not in input file or not read correctly\n" );
                        return -1;
                    }
                } else {
                    printf ( "ERROR: number of fixed effects was not in input file or not read correctly\n" );
                    return -1;
                }
            } else {
                printf ( "ERROR: number of SNP effects was not in input file or not read correctly\n" );
                return -1;
            }
        } else {
            printf ( "ERROR: number of observations was not in input file or not read correctly\n" );
            return -1;
        }
        if ( fixedfile_bool ) {
            if ( randfile_bool ) {
                printf ( "filename of fixed effects (sparse):           \t %s\n", filenameX );
                printf ( "filename of random (non-SNP) effects (sparse):\t %s\n", filenameZ );
            } else {
                printf ( "ERROR: filename of fixed effects was not in input file or not read correctly\n" );
                return -1;
            }
        } else {
            printf ( "ERROR: filename of random effects was not in input file or not read correctly\n" );
            return -1;
        }

        if ( datahdf5 ) {
            if ( SNPdatafile_bool ) {
                if ( genopath_bool ) {
                    if ( phenopath_bool ) {
                        printf ( "Dataset file is an HDF5-file:       \t %s\n", filenameT );
                        printf ( "path for genotypes in dataset:      \t %s\n", SNPdata );
                        printf ( "path for phenotypes in dataset:     \t %s\n", phenodata );
                    } else {
                        printf ( "ERROR: path for phenotypes of dataset was not in input file or not read correctly\n" );
                        return -1;
                    }
                } else {
                    printf ( "ERROR: path for genotypes of dataset was not in input file or not read correctly\n" );
                    return -1;
                }
            } else {
                printf ( "ERROR: Name of file with dataset was not in input file or not read correctly\n" );
                return -1;
            }
        } else {
            printf ( "Dataset files are binary files\n" );
            if ( SNPdatafile_bool ) {
                if ( Phenodatafile_bool ) {
                    printf ( "filename of SNP dataset:                      \t %s\n", filenameT );
                    printf ( "filename of phenotype dataset:                \t %s\n", filenameY );
                } else {
                    printf ( "ERROR: filename of phenotypes was not in input file or not read correctly\n" );
                    return -1;
                }
            } else {
                printf ( "ERROR: filename of SNP effects was not in input file or not read correctly\n" );
                return -1;
            }
        }
        if ( copyC )
            printf ( "A copy of the coefficient matrix will be stored throughout the computations\n" );
        else
            printf ( "The coefficient matrix will be read in at the beginning of every iteration to save memory\n" );
        if ( Bassparse_bool )
            printf ( "B will be treated as a sparse matrix \n" );
        else
            printf ( "B will be treated as a dense matrix\n" );
        if ( blocksize_bool ) {
            printf ( "Blocksize of %d was used to distribute matrices across processes\n", blocksize );
        } else {
            printf ( "Default blocksize of %d was used to distribute matrices across processes\n", blocksize );
        }
        if ( lam_bool )
            printf ( "Start value of %g was used to estimate variance component lambda\n", gamma_var );
        else
            printf ( "Default start value of %g was used to estimate variance component lambda\n", gamma_var );
        if ( phi_bool )
            printf ( "Start value of %g was used to estimate variance component phi\n", phi );
        else
            printf ( "Default start value of %g was used to estimate variance component phi\n", phi );
        if ( eps_bool )
            printf ( "Convergence criterium of %g was used to estimate variance component lambda\n", epsilon );
        else
            printf ( "Default convergence criterium of %g was used to estimate variance component lambda\n", epsilon );
        if ( maxit_bool )
            printf ( "Maximum number of REML iterations : %d\n", maxiterations );
        else
            printf ( "Default maximum number of REML iterations : %d\n", maxiterations );
        if ( datahdf5==0 && testfile_bool ) {
            printf ( "Cross-validation will be performed with test set in file: \t%s\n", TestSet );
            if ( ntest_bool )
                printf ( "Cross-validation will be performed on sample with size: \t%d\n", ntests );
            else {
                printf ( "ERROR: Number of test samples is required when cross-validation is performed\n" );
                return -1;
            }
        } else if ( datahdf5 && testpath_bool ) {
            printf ( "Cross-validation will be performed with test set in path: \t%s\n", TestSet );
            if ( ntest_bool )
                printf ( "Cross-validation will be performed on sample with size: \t%d\n", ntests );
            else {
                printf ( "ERROR: Number of test samples is required when cross-validation is performed\n" );
                return -1;
            }
        }

        else
            printf ( "No cross-validation is performed\n" );
    } else {
        if ( k<0 || l<0 || m<0 || n<0 ) {
            //printf("Problem with reading input in process %d", iam);
            return -1;
        }
        if ( !fixedfile_bool || !randfile_bool )
            return -1;

        if ( testfile_bool && !ntest_bool )
            return -1;
        if ( datahdf5 && ( !genopath_bool || !phenopath_bool ) )
            return -1;
        if ( !datahdf5 && ( !SNPdatafile_bool || !Phenodatafile_bool ) )
            return -1;
    }
    return 0;
}
