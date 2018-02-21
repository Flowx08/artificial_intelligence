CPP_FILES := $(wildcard ../../src/AI/deeplearning/*.cpp) $(wildcard ../../src/AI/util/*.cpp) $(wildcard ../../src/AI/datasets/*.cpp) $(wildcard ../../src/AI/visualization/*.cpp) data_augmentation_cuda.cpp
CU_FILES := $(wildcard ../../src/AI/deeplearning/*.cu) $(wildcard ../../src/AI/util/*.cu)
OBJ_FILES = $(addprefix obj/, $(notdir $(CPP_FILES:.cpp=.o))) $(addprefix obj/, $(notdir $(CU_FILES:.cu=.o)))
CC = nvcc
FLAGS = -std=c++11 -O3 -D_FORCE_INLINES
LIBS = -lcudnn -lcublas
EXE_NAME = data_augmentation_cuda

all: makedirs main

clear:
	rm -r obj/

obj/%.o : ../../src/AI/*/%.cpp
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

obj/%.o : ../../src/AI/*/%.cu
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

obj/%.o : %.cpp
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

makedirs:
	mkdir -p obj
	
main: $(OBJ_FILES)
	$(CC) $(FLAGS) $^ -o $(EXE_NAME) $(LIBS)
