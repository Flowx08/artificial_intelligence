sudo nvcc -std=c++11 -Wall -O3 -D_FORCE_INLINES charRNN_cuda.cpp ../src/AI/*/*.cpp -o charRNN
