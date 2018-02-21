#include "../../src/AI/deeplearning/DataAugmentation.hpp"
#include "../../src/AI/datasets/MNIST_loader.hpp"
#include "../../src/AI/visualization/Bitmap.hpp"

void print(const ai::Tensor_float& t)
{
	for (int y = 0; y < 28; y++) {
		for (int x = 0; x < 28; x++) {
			if (t[y * 28 + x] > 0.5) printf("#");
			else printf(" ");
		}
		printf("\n");
	}
}

int main(int argc, const char *argv[])
{
	ai::Tensor_float trainingset, training_labels, testingset, testing_labels;
	ai::loadMNIST("/home/flowx08/mnist_png", trainingset, training_labels, testingset, testing_labels);
	
	ai::Tensor_float tmp;
	tmp.setshape(28 * 28);
	tmp.copy(trainingset.ptr(0, 32000));
	printf("Normal image:\n");
	print(tmp);
	ai::augmentation::rotate(tmp, 28, 28, 1, 20);
	printf("Rotated by 20 degree:\n");
	print(tmp);
	ai::augmentation::rotate(tmp, 28, 28, 1, 90);
	printf("Rotated by 90 degree:\n");
	print(tmp);
	tmp.copy(trainingset.ptr(0, 32000));
	ai::augmentation::translate(tmp, 28, 28, 1, 5, 5);
	printf("Traslated by [5, 5]:\n");
	print(tmp);
	tmp.copy(trainingset.ptr(0, 32000));
	ai::augmentation::scaling(tmp, 28, 28, 1, 0.5);
	printf("Scaled by 0.5:\n");
	print(tmp);
	tmp.copy(trainingset.ptr(0, 32000));
	ai::augmentation::scaling(tmp, 28, 28, 1, 2);
	printf("Scaled by 2:\n");
	print(tmp);
	tmp.copy(trainingset.ptr(0, 32000));
	ai::augmentation::hflip(tmp, 28, 28, 1);
	printf("Flipped horizzontally:\n");
	print(tmp);
	tmp.copy(trainingset.ptr(0, 32000));
	ai::augmentation::vflip(tmp, 28, 28, 1);
	printf("Flipped vertically:\n");
	print(tmp);
	tmp.copy(trainingset.ptr(0, 32000));
	ai::augmentation::noise(tmp, 28, 28, 1, 0.1);
	printf("Random noise with probability 0.1:\n");
	print(tmp);
	
	return 0;
}
