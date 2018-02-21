#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"
#include "../../src/AI/datasets/CIFAR10_loader.hpp"

int main(int argc, const char *argv[])
{
	//srand((int)time(NULL));

	//Create network
	ai::NeuralNetwork gg;
	gg.push("INPUT1",	"",				ai::Variable::make(32, 32, 3));
	
	gg.push("CONV1",	"INPUT1",		ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM1",	"CONV1",		ai::Normalization::make(0.5));
	gg.push("REL1",		"NORM1",		ai::Relu::make());
	gg.push("CONV2",	"REL1",			ai::Convolution::make(3, 8, 1, 1));
	gg.push("NORM2",	"CONV2",		ai::Normalization::make(0.5));
	gg.push("REL2",		"NORM2",		ai::Relu::make());
	
	//gg.push("MAX1",		"REL2",			new ai::Averagepooling(2, 2));
	gg.push("MAX1",		"REL2",			ai::Maxpooling::make(2, 2));
	
	gg.push("L1",		"MAX1",			ai::Linear::make(128));
	gg.push("NORM5",	"L1",			ai::Normalization::make(0.5));
	gg.push("L2",		"NORM5",		ai::Relu::make());
	
	gg.push("L6",		"L2",			ai::Linear::make(10));
	gg.push("OUTPUT",	"L6",			ai::Softmax::make(1.f));

	//Ligheweight
	/*
	gg.push("CONV1",	"INPUT1",		new ai::Convolution(3, 8, 1, 1));
	gg.push("NORM1",	"CONV1",		new ai::Normalization(0.5));
	gg.push("REL1",		"NORM1",		new ai::Relu());
	gg.push("CONV2",	"REL1",			new ai::Convolution(3, 8, 1, 1));
	gg.push("NORM2",	"CONV2",		new ai::Normalization(0.5));
	gg.push("REL2",		"NORM2",		new ai::Relu());
	
	gg.push("MAX1",		"REL2",			new ai::Maxpooling(2, 2));
	
	gg.push("CONV3",	"MAX1",			new ai::Convolution(3, 16, 1, 1));
	gg.push("NORM3",	"CONV3",		new ai::Normalization(0.5));
	gg.push("REL3",		"NORM3",		new ai::Relu());
	
	gg.push("CONV4",	"REL3",			new ai::Convolution(3, 16, 1, 1));
	gg.push("NORM4",	"CONV4",		new ai::Normalization(0.5));
	gg.push("REL4",		"NORM4",		new ai::Relu());
	
	gg.push("CONV5",	"REL4",			new ai::Convolution(3, 32, 1, 1));
	gg.push("NORM5",	"CONV5",		new ai::Normalization(0.5));
	gg.push("REL5",		"NORM5",		new ai::Relu());
	
	gg.push("MAX3",		"REL5",			new ai::Maxpooling(2, 2));
	
	gg.push("L1",		"MAX3",			new ai::Linear(128));
	gg.push("NORM5",	"L1",			new ai::Normalization(0.5));
	gg.push("L2",		"NORM5",		new ai::Relu());
	
	gg.push("L6",		"L2",			new ai::Linear(10));
	gg.push("OUTPUT",	"L6",			new ai::Softmax(1.f));
	*/

	//Show network structure
	gg.printstack();
	
	//gg.load("CIFAR10.net");

	//Load trainingset
	ai::Tensor_float trainingset, training_targets, testingset, testing_targets;
	ai::loadCIFAR10("/home/flowx08/cifar10_png", trainingset, training_targets, testingset, testing_targets);
	ai::Tensor_float input, target;
	
	ensure_print(trainingset.height() == 50000, "%d\n", trainingset.height());
	ensure_print(training_targets.height() == 50000, "%d\n", training_targets.height());
	ensure(training_targets.width() == 10);
	ensure(trainingset.width() == 32 * 32 * 3);
	ensure(testingset.height() == 10000);
	ensure(testing_targets.height() == 10000);
	ensure(testing_targets.width() == 10);
	ensure(testingset.width() == 32 * 32 * 3);

	//Create optimizer
	ai::OptimizerSDG opt(10, 0.001, 0.5, ai::Cost::CrossEntropy);
	//gg.load("../../CIFAR10.net");
	gg.printstack();
    
	//TRAINING
	double error = 0;
	const int cicles = 200000;
	const int restarts = 8;
	int best = 10000;
	for (int j = 0; j < restarts; j++) {

		//Rest learning rate
		float baselr;
		baselr = 0.001;
		opt.setLearningrate(baselr);
		
		for (int i = 0; i < cicles; i++) {
			//Update learningrate
			//opt.setLearningrate(opt.getLearningrate() - baselr / (double)cicles);

			//Optimize neural network with random sample
			int type = rand() % trainingset.height();
			input.point(trainingset, 0, type);
			target.point(training_targets, 0, type);
			error += gg.optimize(input, target, &opt);

			if (i % 1000 == 0 && i != 0) {
				printf("Cicle: %d Error: %f\n", i, error / 1000.f);
				error = 0;
			}
			
			if (i % 10000 == 0 && i != 0) {
				//TESTING
				printf("Tesing...\n");
				int errors = 0;
				for (int i = 0; i < (int)testingset.height(); i++) {
					input.point(testingset, 0, i);
					gg.run(input);
					int p_maxid;
					float p_max;
					gg.get_output("OUTPUT").max(&p_max, &p_maxid);
					int t_maxid;
					float t_max;
					testing_targets.max_at(&t_max, &t_maxid, 0, i);
					if (p_maxid != t_maxid) errors++;
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
	gg.save("net.dat");
	
	printf("Done\n");
	return 0;
}
