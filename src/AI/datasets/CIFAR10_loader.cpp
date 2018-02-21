////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "MNIST_loader.hpp"
#include "../util/Files.hpp"
#include "../visualization/Bitmap.hpp"
#include "CSV.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	void loadCIFAR10(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels)
	{
		const int image_area = 32 * 32;
		const int image_channels = 3;
		const int sample_size = image_area * image_channels;
		const int target_size = 10;

		//Load labels from csv files
		int trainingset_size = 0, testingset_size = 0;
		std::vector< std::vector< std::string > > training_csv = ai::loadCSV(folder_path + "/training_labels.csv");
		std::vector< std::vector< std::string > > testing_csv = ai::loadCSV(folder_path + "/testing_labels.csv");
		trainingset_size = training_csv.size();
		testingset_size = testing_csv.size();

		//Allocate tensors
		trainingset.setshape(sample_size, trainingset_size);
		training_labels.setshape(target_size, trainingset_size);
		training_labels.fill(0);
		testingset.setshape(sample_size, testingset_size);
		testing_labels.setshape(target_size, testingset_size);
		testing_labels.fill(0);

		//Log
		printf("Loading trainingset...\n");

		//Load and normalize all the training images and compute the labels
		for (int i = 0; i < trainingset_size; i++) {
			Bitmap bm(folder_path + "/training/" + std::to_string(i) + ".png", Bitmap::RGB);
			for (int c = 0; c < image_channels; c++)
				for (int x = 0; x < image_area; x++)
					trainingset.at(i, c * image_area + x) = bm.getData()[x * image_channels + c] / 255.f;
			training_labels.at(i, std::stoi(training_csv[i][1])) = 1.f;
			if (i % (trainingset_size / 10) == 0) printf("Progress: %f\n", (double)i / (double)trainingset_size);
		}
		
		//Log
		printf("Loading testingset...\n");
		
		//Load and normalize all the testing images and compute the labels
		for (int i = 0; i < testingset_size; i++) {
			Bitmap bm(folder_path + "/testing/" + std::to_string(i) + ".png", Bitmap::RGB);
			for (int c = 0; c < image_channels; c++)
				for (int x = 0; x < image_area; x++)
					testingset.at(i, c * image_area + x) = bm.getData()[x * image_channels + c] / 255.f;
			testing_labels.at(i, std::stoi(testing_csv[i][1])) = 1.f;
			if (i % (testingset_size / 10) == 0) printf("Progress: %f\n", (double)i / (double)testingset_size);
		}
	}

} /* namespace ai */
