#include "../../src/AI/deeplearning/DataAugmentation.hpp"
#include "../../src/AI/deeplearning/CUDA_backend.hpp"
#include "../../src/AI/util/TensorCUDA.hpp"
#include "../../src/AI/datasets/MNIST_loader.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"

void print(const ai::TensorCUDA_float& t)
{
	ai::Tensor_float host(28 * 28);
	t.copyToHost(host.pointer(), host.size()); 
	for (int y = 0; y < 28; y++) {
		for (int x = 0; x < 28; x++) {
			if (host[y * 28 + x] > 0.5) printf("#");
			else printf(" ");
		}
		printf("\n");
	}
}

int main(int argc, const char *argv[])
{
	ai::Tensor_float trainingset, training_targets, testingset, testing_targets;
	ai::loadMNIST("/home/flowx08/mnist_png", trainingset, training_targets, testingset, testing_targets);
	
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
	
	int digit_type = 6400;
	ai::TensorCUDA_float tmp(28 * 28);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	printf("Normal image:\n");
	print(tmp);
	ai::augmentation::hflip(tmp, 28, 28, 1);
	printf("Flipped horizontally:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::vflip(tmp, 28, 28, 1);
	printf("Flipped vertically:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::rotate(tmp, 28, 28, 1, 25);
	printf("Rotated 25 degree:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::rotate(tmp, 28, 28, 1, 90);
	printf("Rotated 90 degree:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::translate(tmp, 28, 28, 1, 8, 8);
	printf("Shifted by [+8, +8]:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::scaling(tmp, 28, 28, 1, 0.5);
	printf("Scaled by 0.5:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::scaling(tmp, 28, 28, 1, 2);
	printf("Scaled by 2:\n");
	print(tmp);
	tmp.copy(dev_trainingset.ptr(0, 6400));
	ai::augmentation::noise(tmp, 28, 28, 1, 0.10);
	printf("Noise with 0.10 probability:\n");
	print(tmp);

	return 0;
}
