Needles
=======

A Distributed AI-REML Best Linear Unbiased Prediction framework for genomic prediction including marker-by-environment interaction. This software has been described and validated in the manuscript `Needles: towards large-scale genomic prediction with marker-by-environment interaction`. (De Coninck et al., 2015, submitted to GENETICS)


This software was developed by Arne De Coninck and can only be used for research purposes.

Genomic datasets used for genomic prediction are constantly growing due to the decreasing costs of genotyping and increasing interest in improving agronomic performance of animals and plants. To be able to deal with those large-scale datasets, a distributed-memory framework was developed based on a message passing interface the ScaLAPACK library and the PARDISO library for efficiently dealing with the sparse information introduced by the marker-by-environemnt interaction effects. The complexity of the algorithm is defined by the number of genetic markers and environments included in the genomic prediction setting; the number of individuals only has a linear effect on the read-in time. To enhance performance it is advised to compile and execute Needles on an MPI-optimized machine.

#Installation

## Dependencies

Needles relies heavily on the following software packages, which have to be installed prior to installation of Needles. These software packages are all open source, except for the vendor-optimized implementations and PARDISO, but an academic license of PARDISO is free of charge.

1. MPI ([OpenMPI](http://www.open-mpi.org/), [MPICH](http://www.mpich.org/), [IntelMPI](http://software.intel.com/en-us/intel-mpi-library))
2. [ScaLAPACK](http://www.netlib.org/scalapack/) and all its dependencies BLAS, BLACS, LAPACK, PBLAS (It is recommended to install a [vendor optimized implementation](http://www.netlib.org/scalapack/faq.html#1.3) )
3. [PARDISO] (http://www.pardiso-project.org/)
4. CMake (http://www.cmake.org/)

Currently, compilation will only work with the Intel MKL libraries installed. When MKL libraries are not available, one must change the MKL libraries in the CMakelists.txt file to the ones which are installed. 

## Step-by-step

1. Unpack zip-file or clone git-repository
2. go into the directory `Needles`
3. make a new directory `build`
4. go into the directory `build`
5. type `cmake ..`
6. type `make`

# Usage

Needles only needs an input file to start. A default input file is provided: `defaultinput.txt`, more information on the arguments in the input-file can be found on the wiki.

To launch Needles with the default input file using for example 4 processes, the following command should be entered in the `build` directory:
`mpirun -np 4 ./DAIRRy-BLUP defaultinput.txt`
At least 2 MPI processes should be initialised, because all sparse operations are performed by a single MPI process, while the other MPI processes are used to handle the dense operations.

# Output

Needles creates 3 output-files with the estimates/predictors for the different effects.
* `estimates_fixed_effects.txt`: Lists the estimates for the fixed effects. Usually these are the fixed environmental effects, but users are free to choose the included fixed effects.
* `estimates_random_genetic_effects.txt`: Lists the predictions for the random genetic effects. These are the predictions for the global genetic effects, independent of the environment.
* `estimates_random_sparse_effects.txt`: Lists the predictions for the random marker-by-environment interaction effects. 

Both random effects can be chosen by the user to model something else than genetic effects and their environmental interaction, but up until now one of the random effects should result in a sparse part of the coefficient matrix and the other shoudl result in a dense part. Also, both random effects can have a different variance, but the variandce is homoscedastic for each of the random effects, meaning that the variance of each random effect so modeled as a constant diagonal matrix. 

Next to the result files, two files are given as output that monitor memory usage in the root node, which performs all the operations on the sparse part of the system (`root_output.txt`), and in the other nodes, performing all operations on the dense part of the system (`cluster_output.txt`).

# Version history

* Version 0.1 (12/2013):
  1. First public release of DAIRRy-BLUP

# Contact

Please feel free to contact arne.deconinck[at]ugent.be for any questions or suggestions. 
