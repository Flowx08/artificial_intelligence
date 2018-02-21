#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Cost.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "../util/TensorCUDA.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{	
	
	class NeuralNetwork;

	class Optimizer
	{
		public:
			virtual void fit(NeuralNetwork& net, Tensor_float &inputs, Tensor_float &targets);
			#ifdef CUDA_BACKEND
			virtual void fit(NeuralNetwork& net, TensorCUDA_float &inputs, TensorCUDA_float &targets);
			#endif
			void setLearningrate(const float learningrate);
			void setMomentum(const float momentum);
			const float getLearningrate() const;
			const float getMomentum() const;
			const float getError() const;

		protected:
			float _learningrate; 
			float _momentum;
			float _error;
			Cost _costfunction;
	};

} /* namespace ai */

#endif /* end of include guard: OPTIMIZER_HPP */

