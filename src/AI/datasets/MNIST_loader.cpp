////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "MNIST_loader.hpp"
#include "mnist_binary_loader.hpp"
#include "../util/Files.hpp"
#include "../visualization/Bitmap.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	void loadMNIST(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels)
	{
		const int digits_count = 10;
		const int sample_size = 28 * 28;
		const int target_size = 10;
		
		//Get all files names and calculate trainingset and testingset size
		int trainingset_size = 0, testingset_size = 0;
		std::vector< std::vector<std::string> > training_files(digits_count);
		for (int i = 0; i < digits_count; i++) {
			training_files[i] = ai::files::listdir(folder_path + "/training/" + std::to_string(i) + "/");
			trainingset_size += (int)training_files[i].size();
		}
		std::vector< std::vector<std::string> > testing_files(digits_count);
		for (int i = 0; i < digits_count; i++) {
			testing_files[i] = ai::files::listdir(folder_path + "/testing/" + std::to_string(i) + "/");
			testingset_size += (int)testing_files[i].size();
		}
		
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
		int offset = 0;
		for (int i = 0; i < digits_count; i++) {
			for (int k = 0; k < (int)training_files[i].size(); k++) {
				Bitmap bm(folder_path + "/training/" + std::to_string(i) + "/" + training_files[i][k], Bitmap::MONO);
				bm.convertToMono();
				for (int j = 0; j < sample_size; j++)
					trainingset.at(offset + k, j) = bm.getData()[j] / 255.f;
				training_labels.at(offset + k, i) = 1.f;
			}
			offset += (int)training_files[i].size();
			printf("Progress: %f\n", (double)offset / (double)trainingset_size);
		}
		
		//Log
		printf("Loading testingset...\n");
		
		//Load and normalize all the testing images and compute the labels
		offset = 0;
		for (int i = 0; i < digits_count; i++) {
			for (int k = 0; k < (int)testing_files[i].size(); k++) {
				Bitmap bm(folder_path + "/testing/" + std::to_string(i) + "/" + testing_files[i][k], Bitmap::MONO);
				bm.convertToMono();
				for (int j = 0; j < sample_size; j++)
					testingset.at(offset + k, j) = bm.getData()[j] / 255.f;
				testing_labels.at(offset + k, i) = 1.f;
			}
			offset += (int)testing_files[i].size();
			printf("Progress: %f\n", (double)offset / (double)testingset_size);
		}
	}
	
	void loadMNIST_from_binary(const std::string train_images_path, const std::string test_images_path,
		const std::string train_labels_path, const std::string test_labels_path, 
		Tensor_float& trainingset, Tensor_float& training_labels, Tensor_float& testingset, Tensor_float& testing_labels)
	{
		printf("Loading MNIST dataset...\n");
		
		//Load all the raw data
		mnist_binary_loader mnist(train_images_path, test_images_path, train_labels_path, test_labels_path);
		
		//Allocate trainingset tensor and training_labels tensor
		trainingset.setshape(28 * 28, (int)mnist.get_train_images().size());
		training_labels.setshape(10, (int)mnist.get_train_images().size());
		training_labels.fill(0);

		//Fill trainingset tensor
		for (int i = 0; i < (int)mnist.get_train_images().size(); i++)
			for (int k = 0; k < 28 * 28; k++)
				trainingset.at(i, k) = mnist.get_train_images()[i][k] / 255.f;
		
		//Fill training_labels tensor
		for (int i = 0; i < (int)mnist.get_train_images().size(); i++)
			training_labels.at(i, (int)mnist.get_train_labels()[i]) = 1.f;
		
		//Allocate testingset tensor and testing_labels tensor
		testingset.setshape(28 * 28, (int)mnist.get_test_images().size());
		testing_labels.setshape(10, (int)mnist.get_test_images().size());
		testing_labels.fill(0);
		
		//Fill testingset tensor
		for (int i = 0; i < (int)mnist.get_test_images().size(); i++)
			for (int k = 0; k < 28 * 28; k++)
				testingset.at(i, k) = mnist.get_test_images()[i][k] / 255.f;
		
		//Fill testing_labels tensor
		for (int i = 0; i < (int)mnist.get_test_images().size(); i++)
			testing_labels.at(i, (int)mnist.get_test_labels()[i]) = 1.f;
	}

} /* namespace ai */
