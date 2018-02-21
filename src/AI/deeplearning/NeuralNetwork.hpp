#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include "Operation.hpp"
#include "NetworkNode.hpp"
#include "Optimizer.hpp"
#include "../util/Macros.hpp"
#include <memory>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class NeuralNetwork
	{
		public:
			NeuralNetwork();
			NeuralNetwork(std::string filepath);
			~NeuralNetwork();
			void save(std::string filepath);
			void load(std::string filepath);
			void push(std::string node_name, std::string inputs_names, std::shared_ptr<Operation> operation);
			void clear();
			NetworkNode* get_byname(std::string node_name);
			#ifdef CUDA_BACKEND
			void run(TensorCUDA_float input, const bool training = false);
			float optimize(TensorCUDA_float input, TensorCUDA_float target, Optimizer* opt);
			bool test(TensorCUDA_float input, TensorCUDA_float target);
			TensorCUDA_float& get_output(std::string node_name);
			#else
			void run(Tensor_float input, const bool training = false);
			float optimize(Tensor_float input, Tensor_float target, Optimizer* opt);
			Tensor_float& get_output(std::string node_name);
			#endif
			std::vector<NetworkNode>& getNodes();
			void printstack();
			
		private:
			void resetOperationsErrors();
			std::vector<std::string> splitString(std::string s, char delimiter);

			std::vector<NetworkNode> _nodes;
	};

} /* namespace ai */

#endif /* end of include guard: NEURALNETWORK_HPP */

