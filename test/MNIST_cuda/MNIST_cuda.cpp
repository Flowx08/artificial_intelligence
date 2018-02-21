#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/deeplearning/ResidualBlock.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"
#include "../../src/AI/deeplearning/DataAugmentation.hpp"
#include "../../src/AI/util/Util.hpp"
#include "../../src/AI/datasets/MNIST_loader.hpp"
#include <random>
#include "../clock.h"

int main(int argc, const char *argv[])
{
	//srand((int)time(NULL));
		
	//Create network
	ai::NeuralNetwork gg;
	gg.push("INPUT1",	"",				ai::Variable::make(28, 28));
	
	/*
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 32, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	
	gg.push("MAX",		"REL1",			ai::Maxpooling::make(2, 2));
	gg.push("DROP1",	"MAX",			ai::Dropout::make(0.2));
	
	gg.push("CONV4",	"DROP1",		ai::Convolution::make(2, 24, 1));
	gg.push("NORM4",	"CONV4",		ai::Normalization::make(0.5));
	gg.push("REL4",		"NORM4",		ai::Relu::make());
	
	gg.push("CONV2",	"REL4",			ai::Convolution::make(3, 16, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());
	
	gg.push("MAX2",		"REL2",			ai::Averagepooling::make(2, 2));
	gg.push("DROP2",	"MAX2",			ai::Dropout::make(0.15));
	
	gg.push("CONV3",	"DROP2",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM3",	"CONV3",		ai::Normalization::make(0.5));
	gg.push("REL3",		"NORM3",		ai::Relu::make());
	
	gg.push("L5",	"REL3",		ai::Linear::make(100, true));
	gg.push("L6",	"L5",		ai::Relu::make());
	gg.push("L7",	"L6",		ai::Linear::make(10));
	gg.push("OUTPUT", "L7",		ai::Softmax::make(1.f));
	*/
	
	/*
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 32, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	
	gg.push("MAX",		"REL1",			ai::Maxpooling::make(2, 2));
	gg.push("DROP1",	"MAX",			ai::Dropout::make(0.1));
	
	gg.push("CONV4",	"DROP1",		ai::Convolution::make(2, 24, 1));
	gg.push("NORM4",	"CONV4",		ai::Normalization::make(0.5));
	gg.push("REL4",		"NORM4",		ai::Relu::make());
	
	gg.push("CONV2",	"REL4",			ai::Convolution::make(3, 24, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());
	
	gg.push("MAX2",		"REL2",			ai::Maxpooling::make(2, 2));
	gg.push("DROP2",	"MAX2",			ai::Dropout::make(0.1));
	
	gg.push("CONV3",	"DROP2",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM3",	"CONV3",		ai::Normalization::make(0.5));
	gg.push("REL3",		"NORM3",		ai::Relu::make());
	gg.push("DROP3",	"REL3",			ai::Dropout::make(0.1));
	
	gg.push("L5",	"DROP3",	ai::Linear::make(100, true));
	gg.push("L6",	"L5",		ai::Relu::make());
	gg.push("L7",	"L6",		ai::Linear::make(10));
	gg.push("OUTPUT", "L7",		ai::Softmax::make(1.f));
	*/

	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 10, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	
	gg.push("MAX",		"REL1",			ai::Maxpooling::make(2, 2));
	
	gg.push("RES2",		"MAX",			ai::ResidualBlock::make(3, 8, 1, 1, 5));

	/*
	gg.push("CONV2",	"MAX",			ai::Convolution::make(3, 24, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());
	*/

	gg.push("MAX2",		"RES2",			ai::Averagepooling::make(2, 2));
	gg.push("DROP2",	"MAX2",			ai::Dropout::make(0.1));
	
	gg.push("L5",	"DROP2",	ai::Linear::make(100, true));
	gg.push("L6",	"L5",		ai::Relu::make());
	gg.push("L7",	"L6",		ai::Linear::make(10));
	gg.push("OUTPUT", "L7",		ai::Softmax::make(1.f));

	//gg.load("MNIST.net");
	
	//Show network structure
	gg.printstack();
	
	//Load trainingset on host
    ai::Tensor_float trainingset, training_targets, testingset, testing_targets;
	//ai::loadMNIST("/home/flowx08/mnist_png", trainingset, training_targets, testingset, testing_targets);
	ai::loadMNIST_from_binary(
			"/home/flowx08/mnist_binary/train-images-idx3-ubyte",
			"/home/flowx08/mnist_binary/t10k-images-idx3-ubyte",
			"/home/flowx08/mnist_binary/train-labels-idx1-ubyte",
			"/home/flowx08/mnist_binary/t10k-labels-idx1-ubyte",
			trainingset, training_targets, testingset, testing_targets);
	
	//Upload trainingset on device
	printf("Uploading trainingset and testingset on device...\n");
    ai::TensorCUDA_float dev_trainingset, dev_training_targets, dev_testingset, dev_testing_targets;
	dev_trainingset.setshape(trainingset.width(), trainingset.height());
	dev_trainingset.copyToDevice(trainingset.pointer(), trainingset.size());
	dev_testingset.setshape(testingset.width(), testingset.height());
	dev_testingset.copyToDevice(testingset.pointer(), testingset.size());
	dev_training_targets.setshape(training_targets.width(), training_targets.height());
	dev_training_targets.copyToDevice(training_targets.pointer(), training_targets.size());
	dev_testing_targets.setshape(testing_targets.width(), testing_targets.height());
	dev_testing_targets.copyToDevice(testing_targets.pointer(), testing_targets.size());
	printf("Done\n");

	ai::TensorCUDA_float input(28 * 28);
	
	//Create optimizer
    ai::OptimizerSDG opt(10, 0.001, 0.4, ai::Cost::CrossEntropy);

	//TRAINING
	double error = 0;
	double error_integral = 0;
	const int cicles = 200000;
	const int restarts = 40;
	int best = 200;
	measure_start();
	for (int j = 0; j < restarts; j++) {

		//Rest learning rate
		float baselr;
		baselr = 0.0025;
		//else baselr = 0.005;
		opt.setLearningrate(baselr);
		
		for (int i = 0; i < cicles; i++) {
			
			//reset timer
			double comptime = measure_stop();
			measure_start();
			
			//Update learningrate
			opt.setLearningrate(opt.getLearningrate() - baselr / (double)cicles);
			
			//Select random sample
			int random_sample = rand() % dev_trainingset.height();
			
			//Data augmentation
			input.copy(dev_trainingset.ptr(0, random_sample));
			ai::augmentation::rotate(input, 28, 28, 1, -20 + ai::util::randf() * 40);
			ai::augmentation::translate(input, 28, 28, 1, -4 + ai::util::randf() * 8, -4 + ai::util::randf() * 8);

			//Optimize neural network 
			error += gg.optimize(input, dev_training_targets.ptr(0, random_sample), &opt);

			if (i % 1000 == 0 && i != 0) {
				printf("Cicle: %d Error: %f ErrorIntegral: %f LR: %f ExecTime: %f\n", i, error / 1000.f, error_integral, opt.getLearningrate(), comptime);
				error_integral += error;
				error = 0;
			}
			
			if (i % 25000 == 0 && i != 0) {
				//TESTING
				printf("Testing...\n");
				int errors = 0;
				for (int j = 0; j < (int)dev_testingset.height(); j++) {
					if (gg.test(dev_testingset.ptr(0, j), dev_testing_targets.ptr(0, j)) == false) errors++;
				}
				printf("Testing errors: %d\n", errors);
				
				if (errors < best) {
					best = errors;
					gg.save("MNIST.net");
					printf("Network saved!\n");
				}
			}
		}
	}
	
	printf("Saving network...\n");
	
	printf("Done\n");
	return 0;
}
