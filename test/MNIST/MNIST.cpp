#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/deeplearning/ResidualBlock.hpp"
#include "../../src/AI/deeplearning/Autoencoder.hpp"
#include "../../src/AI/deeplearning/LadderNetwork_MLP.hpp"
#include "../../src/AI/deeplearning/CPU_backend.hpp"
#include "../../src/AI/deeplearning/DataAugmentation.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"
#include "../../src/AI/deeplearning/OptimizerDFA.hpp"
#include "../../src/AI/deeplearning/MLP.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"
#include "../../src/AI/visualization/visualization.hpp"
#include "../../src/AI/util/Util.hpp"
#include "../../src/AI/datasets/MNIST_loader.hpp"
#include <random>

int main(int argc, const char *argv[])
{
	//srand((int)time(NULL));
    
	//Create network
	ai::NeuralNetwork gg;
	gg.push("INPUT1",	"",				ai::Variable::make(28, 28));
	
	/*
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 10, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	
	gg.push("MAX",		"REL1",			ai::Maxpooling::make(2, 2));
	gg.push("DROP1",	"MAX",			ai::Dropout::make(0.1));
	
	gg.push("CONV2",	"DROP1",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());
	
	gg.push("MAX2",		"REL2",			ai::Averagepooling::make(2, 2));
	gg.push("DROP2",	"MAX2",			ai::Dropout::make(0.1));
	
	gg.push("CONV3",	"DROP2",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM3",	"CONV3",		ai::Normalization::make(0.5));
	gg.push("REL3",		"NORM3",		ai::Relu::make());
	
	gg.push("L5",	"REL3",		ai::Linear::make(100, true));
	gg.push("L6",	"L5",		ai::Relu::make());
	gg.push("L7",	"L6",		ai::Linear::make(10));
	gg.push("OUTPUT", "L7",		ai::Softmax::make(1.f));
	*/
	/*
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 10, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	
	gg.push("MAX",		"REL1",			ai::Maxpooling::make(2, 2));
	
	gg.push("RES2",	"MAX",			ai::ResidualBlock::make(3, 8, 1, 1, 4));
	
	gg.push("MAX2",		"RES2",			ai::Averagepooling::make(2, 2));
	gg.push("DROP2",	"MAX2",			ai::Dropout::make(0.1));
	
	gg.push("L5",	"DROP2",	ai::Linear::make(100, true));
	gg.push("L6",	"L5",		ai::Relu::make());
	gg.push("L7",	"L6",		ai::Linear::make(10));
	gg.push("OUTPUT", "L7",		ai::Softmax::make(1.f));
	*/
	
	/*
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(4, 10, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());

	gg.push("MAX",		"REL1",			ai::Maxpooling::make(2, 2));
	//gg.push("DROP1",	"MAX",			ai::Dropout::make(0.1));

	gg.push("CONV2",	"MAX",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());

	gg.push("MAX2",		"REL2",			ai::Averagepooling::make(2, 2));
	//gg.push("DROP2",	"MAX2",			ai::Dropout::make(0.1));

	gg.push("CONV3",	"MAX2",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM3",	"CONV3",		ai::Normalization::make(0.5));
	gg.push("REL3",		"NORM3",		ai::Relu::make());

	gg.push("K1",	"REL3",		ai::Linear::make(128, true));
	gg.push("NORM5",	"K1",		ai::Normalization::make(0.5));
	gg.push("REL",	"NORM5",		ai::Relu::make());
	//gg.push("DROP",	"REL",		ai::Dropout::make(0.4));
	gg.push("L9",	"REL",		ai::Linear::make(10));
	gg.push("OUTPUT", "L9",		ai::Softmax::make(1.f));
	*/
	
	gg.push("LIN1",		"INPUT1",		ai::Linear::make(200));
	gg.push("REL1",		"LIN1",			ai::Relu::make());
	gg.push("LIN2",		"REL1",		ai::Linear::make(100));
	gg.push("REL2",		"LIN2",			ai::Relu::make());
	gg.push("LIN3",		"REL2",			ai::Linear::make(10));
	gg.push("OUTPUT",		"LIN3",			ai::Sigmoid::make());
	//gg.push("OUTPUT",		"ACT",			ai::Softmax::make());


	/*
	gg.push("T1",	"INPUT1",		ai::Linear::make(256));
	gg.push("N1",	"T1",		ai::Normalization::make(0.5));
	gg.push("R1",	"N1",		ai::Relu::make());
	gg.push("T2",	"R1",		ai::Linear::make(256));
	gg.push("N2",	"T2",		ai::Normalization::make(0.5));
	gg.push("R2",	"N2",		ai::Relu::make());
	//gg.push("L9",	"R2",		ai::Linear::make(10));
	gg.push("T3",	"R2",		ai::Linear::make(10));
	gg.push("OUTPUT", "T3",		ai::Softmax::make(1.f));
	*/

	//gg.load("MNIST.net");
	
	//Show network structure
	gg.printstack();
	
	//Load trainingset
  ai::Tensor_float trainingset, training_targets, testingset, testing_targets;
	//ai::loadMNIST("/home/flowx08/mnist_png", trainingset, training_targets, testingset, testing_targets);
  ai::loadMNIST_from_binary(
		"/home/flowx08/mnist_binary/train-images-idx3-ubyte",
		"/home/flowx08/mnist_binary/t10k-images-idx3-ubyte",
		"/home/flowx08/mnist_binary/train-labels-idx1-ubyte",
		"/home/flowx08/mnist_binary/t10k-labels-idx1-ubyte",
		trainingset, training_targets, testingset, testing_targets);

	//Create optimizer
	ai::OptimizerSDG opt(1, 0.004, 0.5, ai::Cost::SquaredError);
	
	//ai::LadderNetwork_MLP ladder(28 * 28, (std::vector<unsigned int>){256, 32});
	//ladder.print_structure();

	//TRAINING
	double error = 0;
	const int cicles = 100000;
	const int restarts = 1;
	int best = 200;
	ai::Tensor_float boosting(100);
	boosting.fill(1);
	for (int j = 0; j < restarts; j++) {
		//Rest learning rate
		float baselr;
		baselr = 0.01;
		//else baselr = 0.005;
		opt.setLearningrate(baselr);
		
		ai::Tensor_float input;
		ai::Tensor_float out_t;
		ai::Tensor_float out_tmp(10);

		for (int i = 0; i < cicles; i++) {

			//Update learningrate
			opt.setLearningrate(opt.getLearningrate() - baselr / (double)cicles);

			//Select random sample
			int random_sample_id = rand() % trainingset.height();
			input.copy(trainingset.ptr(0, random_sample_id));
			
			//Data augmentation
			//ai::augmentation::rotate(input, 28, 28, 1, -25 + rand() % (25 * 2));
			//ai::augmentation::translate(input, 28, 28, 1, -5 + rand() % 10, -5 + rand() % 10);
			//ai::augmentation::noise(input, 28, 28, 1, ai::util::randf() * 0.10);
			
			//Optimize deep neural network
			//error += gg.optimize(input, training_targets.ptr(0, random_sample_id), &opt);
			//ladder.learn_supervised(input, training_targets.ptr(0, random_sample_id), 0.007);
			//ladder.learn_unsupervised(input, opt.getLearningrate());
				
			//int random_sample_id = (rand() % (trainingset.height() / 120)) * 120;
			//input.copy(trainingset.ptr(0, random_sample_id));
			error += gg.optimize(input, training_targets.ptr(0, random_sample_id), &opt);

			//Log
			if (i % 1000 == 0 && i != 0) {
				printf("Cicle: %d Error: %f Learningrate: %f\n", i, error / 1000.f, opt.getLearningrate());
				error = 0;
			}
			
			if (i % 25000 == 0 && i != 0) {
				
				printf("Testing...\n");
				
				int errors = 0;
				for (int i = 0; i < (int)testingset.height(); i++) {
					gg.run(testingset.ptr(0, i), false);
					//ladder.predict(testingset.ptr(0, i));
					int maxid;
					//ladder.get_output().max(NULL, &maxid);
					gg.get_output("OUTPUT").max(NULL, &maxid);
					int target_id;
					testing_targets.ptr(0, i).max(NULL, &target_id);
					if (maxid != target_id) errors++;
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
	
	//printf("Saving network...\n");
	//gg.save("MNIST.net");
	
	printf("Done\n");
	return 0;
}
