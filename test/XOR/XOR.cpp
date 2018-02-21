#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"

int main(int argc, const char *argv[])
{
	//Create network
	ai::NeuralNetwork gg;
	gg.push("INPUT1",	"",				ai::Variable::make(2));
	gg.push("L1",		"INPUT1",		ai::Linear::make(3));
	gg.push("L2",		"L1",			ai::Tanh::make());
	gg.push("L3",		"L2",			ai::Linear::make(1));
	gg.push("L4",		"L3",			ai::Sigmoid::make());
	
	//Show network structure
	gg.printstack();
	
	//Create XOR trainingset
	ai::Tensor_float dataset(2, 4);
	ai::Tensor_float dataset_target(1, 4);
	dataset.at(0, 0) = 1.0; dataset.at(0, 1) = 0.0; dataset_target.at(0, 0) = 1;
	dataset.at(1, 0) = 0.0; dataset.at(1, 1) = 1.0; dataset_target.at(1, 0) = 1;
	dataset.at(2, 0) = 0.0; dataset.at(2, 1) = 0.0; dataset_target.at(2, 0) = 0;
	dataset.at(3, 0) = 1.0; dataset.at(3, 1) = 1.0; dataset_target.at(3, 0) = 0;
	ai::Tensor_float input;
	ai::Tensor_float target;

	printf("%s\n", dataset.tostring().c_str());
	printf("%s\n", dataset_target.tostring().c_str());

	//Create optimizer
	ai::OptimizerSDG opt(4, 0.05, 0.8, ai::Cost::SquaredError);

	//Train
	double error = 0;
	const int cicles = 300000;
	for (int i = 0; i < cicles; i++) {
		
		input.point(dataset, 0, i % 4);
		target.point(dataset_target, 0, i % 4);
		error += gg.optimize(input, target, &opt);

		if (i % 15000 == 0 && i != 0) {
			printf("E: %f\n", error / 15000.f);
			error = 0;
		}
	}
	
	//Test
	for (int i = 0; i < 4; i++) {
		input.point(dataset, 0, i);
		gg.run(input);
		printf("%s -> %s\n", input.tostring().c_str(),
			gg.get_output("L4").tostring().c_str());
	}

	printf("Done\n");
	return 0;
}
