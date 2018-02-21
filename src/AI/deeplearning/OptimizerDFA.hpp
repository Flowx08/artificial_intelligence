#ifndef OPTIMIZERDFA_HPP
#define OPTIMIZERDFA_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Optimizer.hpp"
#include "../util/Macros.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class OptimizerDFA : public Optimizer
	{
		public:
			OptimizerDFA();
			OptimizerDFA(const int batch_size, const double learningrate, const double momentum,
					const Cost::CostType cost_function = Cost::SquaredError);
			#ifdef CUDA_BACKEND
			void fit(NeuralNetwork& net, TensorCUDA_float &inputs, TensorCUDA_float &targets);
			#else
			void fit(NeuralNetwork& net, Tensor_float &inputs, Tensor_float &targets);
			#endif

		private:
			int _batch_size;
			int _current_sample;
			#ifdef CUDA_BACKEND
			TensorCUDA_float _targets;
			TensorCUDA_float _feedback_weights;
			TensorCUDA_float _feedback_errors;
			#else
			Tensor_float _feedback_weights;
			Tensor_float _feedback_errors;
			#endif
	};
} /* namespace ai */

#endif /* end of include guard: OPTIMIZERDFA_HPP */

