#include "../../src/AI/optimization/GA.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <time.h>

Bitmap *bm;

std::vector<unsigned char> generate(std::vector<unsigned char>& t1, std::vector<unsigned char>& t2)
{
	std::vector<unsigned char> son;
	if (rand() % 2) son = t1;
	else son = t2;
	son[rand() % son.size()] = rand();
	return son;
}

double evaluate(std::vector<unsigned char>& t)
{
	double score = 0.0;
	for (int k = 0; k < (int)t.size(); k++)
		score += 255 - abs(bm->m_data[k] - t[k]);
	return score;
}

void destroy(std::vector<unsigned char>& t) {}

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));
	
	bm = new Bitmap("data.png", Bitmap::RGB);

	std::vector< std::vector<unsigned char> > pop(10);
	for (int g = 0; g < 10; g++) {
		pop[g] = std::vector<unsigned char>(bm->m_width * bm->m_height * bm->m_channels);
		for (int i = 0; i < (int)pop[g].size(); i++) 
			pop[g][i] = rand();
	}
	
	ai::genetic_optimizer< std::vector<unsigned char> > order(pop, 5, evaluate, generate, destroy); 
	for (int i = 0; i < 30; i++) {
		order.run(10000, 4);
		printf("Cicle: %d Score:%f MediumError:%f\n", i, order.getBestGenomeFitness(), 255 - (order.getBestGenomeFitness()/(double)pop[0].size()));
	}
	
	printf("score: %f\n", order.getBestGenomeFitness());
	printf("Medium distance: %f\n", 255 - (order.getBestGenomeFitness()/(double)pop[0].size()));
	
	Bitmap result(bm->m_width, bm->m_height, Bitmap::RGB, 0x000000);
	for (int i = 0; i < (int)order.getBestGenomeData().size(); i++) {
		result.m_data[i] = order.getBestGenomeData()[i];
	}
	result.save("result.png");
	return 0;
}
