#ifndef NETWORKNODE_HPP
#define NETWORKNODE_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include "Operation.hpp"
#include "../util/IOData.hpp"

//Nodes
#include "Variable.hpp"
#include "Linear.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "Recurrent.hpp"
#include "Partial.hpp"
#include "Dropout.hpp"
#include "Convolution.hpp"
#include "Normalization.hpp"
#include "Addition.hpp"
#include "Concatenate.hpp"
#include "Maxpooling.hpp"
#include "Averagepooling.hpp"
#include "Selu.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	class NeuralNetwork;

	class NetworkNode
	{
		public:
			NetworkNode(std::string node_name, std::vector<std::string> input_names,
				NeuralNetwork* network, std::shared_ptr<Operation> operation);
			NetworkNode(ai::IOData& data, NeuralNetwork* network);
			
			void run(bool training = false);
			void backprop();
			void accumulate_deltas();
			void update_parameters(const float learningrate);
			void reset_errors();
			void reset_deltas(const float momentum);
			void save(ai::IOData& data);
			const std::string getName() const;
			const std::vector<std::string> getInputsNames();
			const std::vector<int> getInputsIndicies();
			void print();
			#ifdef CUDA_BACKEND
			Tensor_float getOperationOutput();
			const TensorCUDA_float& getOperationOutputDevice();
			void setOperationOutput(TensorCUDA_float& output);
			#else
			Tensor_float& getOperationOutput();
			void setOperationOutput(Tensor_float& output);
			#endif
			Operation* getOperation();

		private:
			void load(ai::IOData& data);
			void checkInvalidInputNames();	
			void initInputsIndiciesVector();
			void initInputsOperationsVector();

			NeuralNetwork* _network;
			std::shared_ptr<Operation> _operation;
			std::string _name;
			std::vector<std::string> _input_names;
			std::vector<int> _input_indicies;
			std::vector<Operation*> _input_operations;
	};

} /* namespace ai */

#endif /* end of include guard: NETWORKNODE_HPP */

