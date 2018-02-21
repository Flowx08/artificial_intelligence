#include <stdio.h>
#include <fstream>
#include <stdio.h>
#include <string>
#include "../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../src/AI/deeplearning/Linear.hpp"
#include "../src/AI/deeplearning/Sigmoid.hpp"
#include "../src/AI/deeplearning/Tanh.hpp"
#include "../src/AI/deeplearning/Softmax.hpp"
#include "../src/AI/deeplearning/Variable.hpp"
#include "../src/AI/deeplearning/Recurrent.hpp"
#include "../src/AI/deeplearning/Relu.hpp"
#include "../src/AI/deeplearning/OptimizerSDG.hpp"
#include <math.h>
#include <stdlib.h>

std::vector<int> TensorOrder(const ai::Tensor_float t)
{
	std::vector<int> indicies(t.size());
	std::vector<float> vals(t.size());

	for (int i = 0; i < t.size(); i++)
		vals[i] = t[i];
	
	int indexpos = 0;
	for (int i = 0; i < t.size(); i++) {
		float maxval = -0xFFFF;
		int maxid = 0;
		for (int k = 0; k < t.size(); k++) {
			if (maxval < vals[k])
			{
				maxval = vals[k];
				maxid = k;
			}
		}
		indicies[indexpos++] = maxid;
		vals[maxid] = -0xFFFF;
	}
	return indicies;
}

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));
	int inputsize = 94;
	int unusedchars = 32;
	ai::NeuralNetwork net;
	
	if (false)
	{
		printf("Loading neural network\n");
		net.load("brain_filosofia");
	}
	else
	{
		printf("Creating neural network\n");
		
		net.push("INPUT1",	"",				ai::Variable::make(inputsize));
		net.push("L1",		"INPUT1",		ai::Recurrent::make(200, 5));
		net.push("L2",		"L1",			ai::Normalization::make(0.5));
		net.push("L3",		"L2",			ai::Relu::make());
		net.push("L5",		"L3",			ai::Linear::make(inputsize));
		net.push("OUTPUT",	"L5",			ai::Softmax::make());
		net.printstack();
	}
	printf("Network ready\n");
	
	ai::OptimizerSDG opt(1, 0.001, 0.1, ai::Cost::CrossEntropy);

	ai::Tensor_float input(inputsize);
	for (int i = 0; i < inputsize; i++) input[i] = 0;
	ai::Tensor_float output(inputsize);
	for (int i = 0; i < inputsize; i++) output[i] = 0;
	

	std::ifstream file("dataset.txt");
	if (!file) {
		printf("File not found!\n");
		return -1;
	}
	std::string data((std::istreambuf_iterator<char>(file)),
                	(std::istreambuf_iterator<char>()));
	file.close();
	
	double error = 0;
	const int epocs = 30;
	double learningrate = 0.003;
	for (int i = 0; i < epocs; i++) {
		printf("Cicle %d lr: %f\n", i, opt.getLearningrate());
		
		/*
		printf("Testing...\n");
		float sum = 0;
		std::string test;
		test += data[rand() % data.size()];
		for (int j = 0; j < 200; j++) {
			int charid;
			if (rand() % 15 == 0) charid = rand() % inputsize;
			else charid = test[test.size()-1];
			input[charid - unusedchars] = 1.0;
			net.run(input);
			output = net.get_output("OUTPUT");
			input[charid - unusedchars] = 0.0;

			std::vector<int> order = TensorOrder(output);
			float sum = output[order[0]] + output[order[1]] + output[order[2]];
			float p1 = output[order[0]] / sum;
			float p2 = output[order[1]] / sum;
			float p3 = output[order[2]] / sum;
			int maxid;
			float r = (rand() % 10000) / (double)10000.f;
			if (r < p1) maxid = order[0];
			else if (r < p1 + p2) maxid = order[1];
			else maxid = order[2];

			test += (char)(maxid + unusedchars);
		}
		printf("%s\n", test.c_str());
		printf("\nDone\n");
		*/

		for (int c = 0; c < (int)data.size()-1; c++) {
			
			opt.setLearningrate(learningrate * (1.f - ( (i * data.size() + c) / (double)((epocs + 1) * data.size()))));
			
			if (data[c] < unusedchars || data[c] > unusedchars + inputsize) continue;
			if (data[c+1] < unusedchars || data[c+1] > unusedchars + inputsize) continue;
			input[data[c]-unusedchars] = 1.0;	
			output[data[c+1]-unusedchars] = 1.0;
			error += net.optimize(input, output, &opt);
			input[data[c]-unusedchars] = 0.0;	
			output[data[c+1]-unusedchars] = 0.0;
			if (c % 10000 == 0) {
				printf("Medium Error %f\n", error / 10000.f);
				error = 0;
				
			}	
		}
	}
	
	net.save("brain_filosofia");
	
	std::string test = "s";
	
	printf("Testing...\n");
	float sum = 0;
	for (int j = 0; j < 200; j++) {
		int charid;
		if (rand() % 15 == 0) charid = rand() % inputsize;
		else charid = test[test.size()-1];
		input[charid - unusedchars] = 1.0;
		net.run(input);
		output = net.get_output("OUTPUT");
		input[charid - unusedchars] = 0.0;
		
		std::vector<int> order = TensorOrder(output);
		float sum = output[order[0]] + output[order[1]] + output[order[2]];
		float p1 = output[order[0]] / sum;
		float p2 = output[order[1]] / sum;
		float p3 = output[order[2]] / sum;
		int maxid;
		float r = (rand() % 10000) / (double)10000.f;
		if (r < p1) maxid = order[0];
		else if (r < p1 + p2) maxid = order[1];
		else maxid = order[2];
		
		test += (char)(maxid + unusedchars);
	}
	printf("\nDone\n");

	printf("%s\n", test.c_str());

	return 0;
}

