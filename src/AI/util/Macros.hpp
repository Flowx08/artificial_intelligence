#ifndef MACROS_HPP
#define MACROS_HPP

//Uncomment to use nvdia gpu with cuda
//#define CUDA_BACKEND

//We most use cuda if we compile the source code using the nvcc compiler
#ifdef __NVCC__
#define CUDA_BACKEND
#endif

#define CUDA_MAX_THREADS 1024
#define CUDA_MAX_CORES 65535

#endif /* end of include guard: MACROS_HPP */

