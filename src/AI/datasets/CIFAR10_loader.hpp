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
	void loadCIFAR10(std::string folder_path, Tensor_float& trainingset, Tensor_float& training_labels,
		Tensor_float& testingset, Tensor_float& testing_labels);

} /* namespace ai */

#endif /* end of include guard: MNIST_LOADER_HPP */

