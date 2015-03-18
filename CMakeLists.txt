project(pardiso_blup)

cmake_minimum_required(VERSION 2.6) 

enable_language(Fortran)

find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIR} ${MPI_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/src)
include(CMakeForceCompiler)
  CMAKE_FORCE_C_COMPILER(mpicc ${CMAKE_C_COMPILER_ID})
  CMAKE_FORCE_CXX_COMPILER(mpic++ ${CMAKE_CXX_COMPILER_ID})
  
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fopenmp -m64" )

aux_source_directory(src helpfunctions)
add_executable(pardiso_blup ${helpfunctions} main.cpp)

if(MPI_COMPILE_FLAGS)
  set_target_properties(pardiso_blup PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

#if(MPI_LINK_FLAGS)
  set_target_properties(pardiso_blup PROPERTIES
    LINK_FLAGS "-L/user/home/gent/vsc405/vsc40508/Libraries -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64 -L/home/arnedc/Libraries ${MPI_LINK_FLAGS}")
#endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

target_link_libraries(pardiso_blup
                      #${LAPACK_LIBRARIES}
                      #${BLAS_LIBRARIES}
		      #-mt_mpi
		      -lpardiso500-GNU461-X86-64
                      ${MPI_LIBRARY}
		      ${HDF5_LIBRARIES}
		      -Wl,--no-as-needed		      
		      -lmkl_scalapack_lp64          # dit is de MKL variant van ScaLapack
		      -lmkl_blas95_lp64
		      -lmkl_lapack95_lp64
		      -lmkl_intel_lp64
		      -lmkl_intel_thread
		      -lmkl_gnu_thread 
		      -lmkl_core
                      -lmkl_blacs_intelmpi_lp64     # dit is de MKL variant van BLACS
		      -ldl
		      -liomp5
		      -lmkl_gf_lp64
		      -lgfortran
		      #-lifcore
		      -fopenmp
		      -lm -lpthread)

