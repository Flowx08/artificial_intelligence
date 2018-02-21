CPP_FILES := $(wildcard ../../src/AI/*/*.cpp) CIFAR10.cpp
OBJ_FILES = $(addprefix obj/, $(notdir $(CPP_FILES:.cpp=.o)))
CC = g++
FLAGS = -std=c++11 -Wall -O3
LIBS = 
EXE_NAME = CIFAR10

all: makedirs main

clear:
	rm -r obj/

obj/%.o : ../../src/AI/*/%.cpp
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

obj/%.o : %.cpp
	$(CC) $(FLAGS) -c $< -o $@ $(LIBS)

makedirs:
	mkdir -p obj
	
main: $(OBJ_FILES)
	$(CC) $(FLAGS) $^ -o $(EXE_NAME) $(LIBS)
