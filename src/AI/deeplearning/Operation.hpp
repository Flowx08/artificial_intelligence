#ifndef OPERATION_HPP
#define OPERATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <fstream>
#include <vector>
#include <memory>
#include "../util/Tensor.hpp"
#include "../util/IOData.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "../util/TensorCUDA.hpp"
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class Operation
	{
		public:

			////////////////////////////////////////////////////////////
			///	OPERATIONS TYPES AVAILABLE
			////////////////////////////////////////////////////////////
			enum Type
			{
				Unknown,
				Variable,
				Linear,
				Sigmoid,
				Tanh,
				Relu,
				Softmax,
				Recurrent,
				Partial,
				Dropout,
				Convolution,
				Normalization,
				Addition,
				Concatenate,
				Maxpooling,
				Averagepooling,
				Selu,
				Autoencoder,
				ResidualBlock,
				CapsulesDense,
				Types_Count
			};

			virtual ~Operation();
			virtual void initialize(std::vector<Operation*> &inputs);
			virtual void run(std::vector<Operation*>& inputs, const bool training);
			virtual void backprop(std::vector<Operation*>& inputs);
			virtual void accumulate_deltas(std::vector<Operation*>& inputs);
			virtual void update_parameters(const float learningrate);
			virtual void print();
			virtual const Type get_type() const;
			virtual void reset_deltas(const double momentum);
			virtual void reset_errors();

			static std::shared_ptr<Operation> loadFromFile(ai::IOData& data);
			static void saveToFile(std::shared_ptr<Operation>& operation, ai::IOData& data);
			
			int _size;
			#ifdef CUDA_BACKEND
			TensorCUDA_float _outputs;
			TensorCUDA_float _errors;
			#else
			Tensor_float _outputs;
			Tensor_float _errors;
			#endif

		private:
			virtual void save(ai::IOData& data);
	};

} /* namespace ai */

#endif /* end of include guard: OPERATION_HPP */

