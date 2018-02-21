#include "../../src/AI/classical/genetic_programming.hpp"
#include "../../src/AI/optimization/GA.hpp"
#include <iostream>
#include <math.h>

using namespace std;

struct data_point {
	float x, y;
} /* optional variable list */;


std::vector< data_point> points;

double evaluate(ai::gp::ExpressionNode*& prog)
{
	double score = 0.0;
	for (int i = 0; i < 20; i++) {
		float px = -5.f + ((double)i / 20.f) * 10.f;
		float py = 1.f / (1.f + exp(-px));
		float y = ai::gp::program_evaluate(prog, {px});
		score -= fabs(py - y);
	}
	return score;
}

ai::gp::ExpressionNode* mutate(ai::gp::ExpressionNode*& p1, ai::gp::ExpressionNode*& p2)
{
	ai::gp::ExpressionNode* child;
	if (rand() % 2 == 0) child = ai::gp::program_crossover_initialization(p1, p2);
	else child = ai::gp::program_copy_initialization(p1);
	ai::gp::program_mutate(child, 0.1, 1, -10, 10);
	return child;
}

void destroy(ai::gp::ExpressionNode*& node)
{
	ai::gp::program_free(node);
	node = nullptr;
}

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));
	
	//Generate dataset
	points.clear();
	for (int k = -2; k < 10; k++) {
		data_point pt;
		pt.x = k;
		pt.y = sin(k); //k * k + 3;
		std::cout << pt.x << " " << pt.y << std::endl;
		points.push_back(pt);
	}
	std::cout << "Dataset generated" << std::endl;
	
	//Generate initial population
	std::vector< ai::gp::ExpressionNode* > population(1000);
	for (int i = 0; i < population.size(); i++)
		population[i] = ai::gp::program_random_initialization(4, 1, -10, 10);	
	std::cout << "Inizial population generated" << std::endl;

	//Initialize genetic algoritm framework
	ai::genetic_optimizer< ai::gp::ExpressionNode* > optimizer(population, 4, evaluate, mutate, destroy);
	std::cout << "Optimizer initialized" << std::endl;
	for (int i = 0; i < 5000; i++) {
		optimizer.run(400, 1);
		std::cout << "Cicle: " << i << " Score: " << optimizer.getBestGenomeFitness() << std::endl;
	}
	
	std::cout << ai::gp::program_parse(optimizer.getBestGenomeData()) << std::endl;
	float score = 0.0;
	for (int i = 0; i < points.size(); i++) {
		float y = ai::gp::program_evaluate(optimizer.getBestGenomeData(), {points[i].x});
		score -= fabs(points[i].y - y);
		std::cout << points[i].x << " " << y << ":" << points[i].y << std::endl;
	}
	std::cout << score << std::endl;

	return 0;
}
