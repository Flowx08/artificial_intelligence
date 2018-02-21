#include "../src/AI/classical/linear_regression.hpp"
#include "../src/AI/classical/logistic_regression.hpp"
#include "../src/AI/classical/ensemble_logistic_regression.hpp"
#include "../src/AI/visualization/Bitmap.hpp"
#include "../src/AI/util/Util.hpp"
#include "datasetloader.hpp"
#include <random>

int main(int argc, const char *argv[])
{
	//srand((int)time(NULL));
    
	//Load trainingset
    std::vector<ai::Tensor_float> trainingset, testingset;
	ai::Tensor_float training_targets, testing_targets;
	datasetloader::loadMNIST(trainingset, training_targets, testingset, testing_targets);
	
	ai::Tensor_float trains, tests, train_t, test_t;
	int trains_size = 0;
	int tests_size = 0;
	for (int i = 0; i < (int)trainingset.size(); i++) {
		trains_size += trainingset[i].height();
		tests_size += testingset[i].height();
	}
	trains.setshape(28 * 28, trains_size);
	tests.setshape(28 * 28, tests_size);
	train_t.setshape(10, trains_size);
	test_t.setshape(10, tests_size);
	int k = 0, m = 0;
	for (int i = 0; i < (int)trainingset.size(); i++) {
		for (int l = 0; l < trainingset[i].height(); l++) {
			for (int p = 0; p < trainingset[i].width(); p++) trains.at(k, p) = trainingset[i].at(l, p);	
			for (int p = 0; p < training_targets.width(); p++) train_t.at(k, p) = training_targets.at(i, p);	
			k++;
		}
		
		for (int l = 0; l < testingset[i].height(); l++) {
			for (int p = 0; p < testingset[i].width(); p++) tests.at(m, p) = testingset[i].at(l, p);	
			for (int p = 0; p < testing_targets.width(); p++) test_t.at(m, p) = testing_targets.at(i, p);	
			m++;
		}
	}
	
	//ai::linear_regression lr(28 * 28, 10);
	ai::logistic_regression lr(28 * 28, 10);
	//ai::linear_regression lr("test.txt");
	ai::ensemble_logistic_regression elr(28 * 28, 10, 20);
	elr.fit(trains, train_t, 0.01, 5);
	//lr.test(tests, test_t);
	//elr.save("test.txt");
	elr.test(tests, test_t);
	
	return 0;
}
