#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/deeplearning/ResidualBlock.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"
#include "../../src/AI/datasets/CIFAR10_loader.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"
#include "../../src/AI/util/Util.hpp"
#include "../../src/AI/deeplearning/DataAugmentation.hpp"
#include <random>
#include <math.h>
#include <iostream>
#include <chrono>

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

float brachistochrone(float x)
{
	static const float pi = 3.14159265;
	static const float r = 0.5;
	return (-r * acos(1-x/r) - sqrt(x*(2 * r - x)) + r * pi) / (pi * 0.5);
}

int main(int argc, const char *argv[])
{
	//srand((int)time(NULL));
	
	//Create network
	ai::NeuralNetwork gg;
	gg.push("INPUT1",	"",				ai::Variable::make(32, 32, 3));
	
	/*
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(3, 24, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	gg.push("CONV2",	"REL1",			ai::Convolution::make(3, 24, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());
	
	gg.push("MAX1",		"REL2",			ai::Maxpooling::make(2, 2));
	
	gg.push("CONV3",	"MAX1",			ai::Convolution::make(3, 32, 1, 1));
	gg.push("NORM3",	"CONV3",		ai::Normalization::make(0.5));
	gg.push("REL3",		"NORM3",		ai::Relu::make());
	
	gg.push("CONV4",	"REL3",			ai::Convolution::make(3, 32, 1, 1));
	gg.push("NORM4",	"CONV4",		ai::Normalization::make(0.5));
	gg.push("REL4",		"NORM4",		ai::Relu::make());
	
	gg.push("CONV5",	"REL4",			ai::Convolution::make(3, 32, 1, 1));
	gg.push("NORM5",	"CONV5",		ai::Normalization::make(0.5));
	gg.push("REL5",		"NORM5",		ai::Relu::make());
	
	gg.push("CONV6",	"REL5",			ai::Convolution::make(3, 32, 1, 1));
	gg.push("NORM6",	"CONV6",		ai::Normalization::make(0.5));
	gg.push("REL6",		"NORM6",		ai::Relu::make());
	
	gg.push("CONV7",	"REL6",			ai::Convolution::make(3, 32, 1, 1));
	gg.push("NORM7",	"CONV7",		ai::Normalization::make(0.5));
	gg.push("REL7",		"NORM7",		ai::Relu::make());
	
	gg.push("CONV8",	"REL7",			ai::Convolution::make(3, 32, 1, 1));
	gg.push("NORM8",	"CONV8",		ai::Normalization::make(0.5));
	gg.push("REL8",		"NORM8",		ai::Relu::make());
	
	gg.push("MAX3",		"REL8",			ai::Maxpooling::make(2, 2));
	
	gg.push("CONV5",	"MAX3",			ai::Convolution::make(3, 96, 1, 1));
	gg.push("NORM5",	"CONV5",		ai::Normalization::make(0.5));
	gg.push("REL5",		"NORM5",		ai::Relu::make());
	
	gg.push("CONV6",	"REL5",			ai::Convolution::make(3, 256, 1, 1));
	gg.push("NORM6",	"CONV6",		ai::Normalization::make(0.5));
	gg.push("REL6",		"NORM6",		ai::Relu::make());
	
	gg.push("MAX4",		"REL6",			ai::Averagepooling::make(2, 2));
	
	gg.push("L1",		"MAX4",			ai::Linear::make(128));
	gg.push("NORM7",	"L1",			ai::Normalization::make(0.5));
	gg.push("L2",		"NORM7",		ai::Relu::make());
	
	gg.push("L6",		"L2",			ai::Linear::make(10));
	gg.push("OUTPUT",	"L6",			ai::Softmax::make(1.f));
	*/
	
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 32, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	
	gg.push("MAX1",		"REL1",			ai::Maxpooling::make(2, 2));
	
	/*
	gg.push("CONV3",	"MAX1",			ai::Convolution::make(3, 24, 1, 1));
	gg.push("NORM3",	"CONV3",		ai::Normalization::make(0.5));
	gg.push("REL3",		"NORM3",		ai::Relu::make());
	*/
	gg.push("RES2", "MAX1",				ai::ResidualBlock::make(3, 16, 1, 1, 1));

	gg.push("L1",		"RES2",			ai::Linear::make(128));
	gg.push("NORMA",	"L1",			ai::Normalization::make(0.5));
	gg.push("L2",		"NORMA",		ai::Relu::make());
	
	gg.push("L6",		"L2",			ai::Linear::make(10));
	gg.push("OUTPUT",	"L6",			ai::Softmax::make(1.f));
	
	/*
	gg.push("CONV1", "INPUT1", ai::Convolution::make(3, 16, 1, 1));
	gg.push("MAX1", "CONV1", ai::Maxpooling::make(2, 2));
	gg.push("RES2", "MAX1", ai::ResidualBlock::make(3, 32, 1, 1, 4));
	//gg.push("MAX2", "RES2", ai::Maxpooling::make(2, 2));
	//gg.push("RES3", "MAX2", ai::ResidualBlock::make(3, 64, 1, 1, 4));
	gg.push("MAX4", "RES2", ai::Maxpooling::make(2, 2));
	gg.push("L1", "MAX4",	ai::Linear::make(10));
	gg.push("OUTPUT", "L1", ai::Softmax::make(1.f));
	*/

	//Show network structure
	gg.printstack();
	//gg.load("CIFAR10.net");

	//Load trainingset
	ai::Tensor_float trainingset, training_targets;
	ai::Tensor_float testingset, testing_targets;
	ai::Tensor_float extra_trainingset, extra_training_targets;
	ai::loadCIFAR10("/home/flowx08/cifar10_png", trainingset, training_targets, testingset, testing_targets);
	ai::TensorCUDA_float dev_trainingset(trainingset.width(), trainingset.height(), trainingset.depth());
	ai::TensorCUDA_float dev_training_targets(training_targets.width(), training_targets.height(), training_targets.depth());
	ai::TensorCUDA_float dev_testingset(testingset.width(), testingset.height(), testingset.depth());
	ai::TensorCUDA_float dev_testing_targets(testing_targets.width(), testing_targets.height(), testing_targets.depth());
	dev_trainingset.copyToDevice(trainingset.pointer(), dev_trainingset.size());
	dev_training_targets.copyToDevice(training_targets.pointer(), dev_training_targets.size());
	dev_testingset.copyToDevice(testingset.pointer(), dev_testingset.size());
	dev_testing_targets.copyToDevice(testing_targets.pointer(), dev_testing_targets.size());
	
	//Create optimizer
	ai::OptimizerSDG opt(10, 0.001, 0.5, ai::Cost::CrossEntropy);
	gg.printstack();
    
	//TRAINING
	ai::TensorCUDA_float input(32 * 32, 3);
	double error = 0;
	const int cicles = 200000;
	const int restarts = 20;
	const double top_lr = 0.001;
	const double bottom_lr = 0.0002;
	const unsigned int total_cicles = restarts * cicles;
	int best = 4000;
	double errorarea = 0;
	Timer tmr;
	for (int j = 0; j < restarts; j++) {
	
		error = 0;
		for (int i = 0; i < cicles; i++) {	
				
			float baselr = top_lr * exp(-(double)(j * cicles + i) * ( -(log(bottom_lr/top_lr)) / (double)total_cicles));
			if (j == 0 && i < 5000) baselr = 0.0003;
			opt.setLearningrate(baselr);
			//opt.setLearningrate(baselr * (1.f - (j * cicles + i) / (double)(restarts * cicles)));
			//opt.setLearningrate(baselr * brachistochrone(i / (double)cicles));
			
			//Select random sample
			int type = rand() % (dev_trainingset.height());
			
			//Data augmentation
			input.copy(dev_trainingset.ptr(0, type));
			//ai::augmentation::hflip(input, 32, 32, 3);
			//ai::augmentation::rotate(input, 32, 32, 3, -20 + ai::util::randf() * 40);
			//ai::augmentation::scaling(input, 32, 32, 3, 0.8 + 0.3 * ai::util::randf());
			//ai::augmentation::translate(input, 32, 32, 3, -4 + ai::util::randf() * 8, -4 + ai::util::randf() * 8);
			
			//Optimize neural network with random sample
			error += gg.optimize(input, dev_training_targets.ptr(0, type), &opt);
			//error += gg.optimize(dev_trainingset.ptr(0, type), dev_training_targets.ptr(0, type), &opt);
			
			if (i % 1000 == 0 && i != 0) {
				//Save log
				FILE* lossfile = fopen("loss.txt", "a");
				fprintf(lossfile, "%d %f\n", i + j * cicles, error / 1000.f);
				fclose(lossfile);
				
				double elapsed = tmr.elapsed();
				printf("Cicle: %d Error: %f lr: %f CicleTime: %f ms TimeToFinish: %f sec  ErrorArea: %f\n", i,
					error / 1000.f, opt.getLearningrate(), (elapsed * 1000.f) / 1000.f,
					elapsed * ((restarts - j) * (cicles / 1000.f) - i / 1000.f), errorarea);
				tmr.reset();
				errorarea += error;
				error = 0;
			}
			
			if (i % 50000 == 0 && i != 0) {
				//TESTING
				printf("Tesing...\n");
				int errors = 0;
				for (int k = 0; k < (int)dev_testingset.height(); k++) {
					if (gg.test(dev_testingset.ptr(0, k), dev_testing_targets.ptr(0, k)) == 0) errors++;
				}
				printf("Testing errors: %d\n", errors);
				
				//Save log
				FILE* testfile = fopen("test.txt", "a");
				fprintf(testfile, "%d %d\n", i + j * cicles, errors);
				fclose(testfile);
				
				if (errors < best) {
					best = errors;
					gg.save("CIFAR10.net");
					printf("Network saved!\n");
				}
			}
		}
	}
	
	printf("Saving network...\n");
	
	printf("Done\n");
	return 0;
}
