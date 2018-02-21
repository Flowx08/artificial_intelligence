#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "../util/Tensor.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	void loadMNIST(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels);
	
	void loadMNIST_from_binary(const std::string train_images_path, const std::string test_images_path,
		const std::string train_labels_path, const std::string test_labels_path, 
		Tensor_float& trainingset, Tensor_float& training_labels, Tensor_float& testingset, Tensor_float& testing_labels);

} /* namespace ai */

#endif /* end of include guard: MNIST_LOADER_HPP */

