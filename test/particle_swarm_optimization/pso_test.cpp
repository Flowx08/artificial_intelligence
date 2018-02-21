#include "../../src/AI/optimization/pso.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <time.h>

Bitmap *bm;

double evaluate(std::vector<double>& t)
{
	double score = 0.0;
	for (int k = 0; k < (int)t.size(); k++)
		score += 255 - fabs(bm->m_data[k] - t[k]);
	return score;
}

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));
	
	bm = new Bitmap("data.png", Bitmap::RGB);

	printf("Starting...\n");
	ai::pso order(30, 64*64*3, evaluate, 0.1, 0.1); 
	for (int i = 0; i < 30; i++) {
		order.run(100);
		printf("Cicle: %d Score:%f MediumError:%f\n", i, order.getBestScore(), 255 - (order.getBestScore()/(64*64*3.0)));
	}
	
	Bitmap result(bm->m_width, bm->m_height, Bitmap::RGB, 0x000000);
	for (int i = 0; i < (int)order.getBestParams().size(); i++) {
		result.m_data[i] = order.getBestParams()[i];
	}
	result.save("result.png");
	return 0;
}
