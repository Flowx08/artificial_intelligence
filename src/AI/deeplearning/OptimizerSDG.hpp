#ifndef OPTIMIZERSDG_HPP
#define OPTIMIZERSDG_HPP

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

	class OptimizerSDG : public Optimizer
	{
		public:
			OptimizerSDG();
			OptimizerSDG(const int batch_size, const double learningrate, const double momentum,
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
			#endif
	};
} /* namespace ai */

#endif /* end of include guard: OPTIMIZERSDG_HPP */

