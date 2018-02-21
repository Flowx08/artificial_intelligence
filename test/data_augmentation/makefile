CPP_FILES := $(wildcard ../../src/AI/*/*.cpp) data_augmentation.cpp
OBJ_FILES = $(addprefix obj/, $(notdir $(CPP_FILES:.cpp=.o)))
CC = g++
FLAGS = -std=c++11 -Wall -O3
LIBS = 
EXE_NAME = data_augmentation

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
