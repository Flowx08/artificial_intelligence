////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "WeightRegularization.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	///	NAMESPACE WEIGHTREGULARIZATION
	////////////////////////////////////////////////////////////
	namespace weightreg
	{
		
		#ifdef CUDA_BACKEND
			
		////////////////////////////////////////////////////////////		
		void l1_regularization(TensorCUDA_float& weights, const float l1_factor, const float learningrate)
		{
			cuda::l1_regularization(weights.pointer(), l1_factor, learningrate, weights.size());
		}

		////////////////////////////////////////////////////////////		
		void l2_regularization(TensorCUDA_float& weights, const float l2_factor, const float learningrate)
		{
			cuda::l2_regularization(weights.pointer(), l2_factor, learningrate, weights.size());
		}

		#else
		
		////////////////////////////////////////////////////////////
		void l1_regularization(Tensor_float& weights, const float l1_factor, const float learningrate)
		{
			for (int i = 0; i < weights.size(); i++)
				weights[i] += (weights[i] > 0 ? -1.f : 1.f) * l1_factor * learningrate;
		}
				
		////////////////////////////////////////////////////////////
		void l2_regularization(Tensor_float& weights, const float l2_factor, const float learningrate)
		{
			for (int i = 0; i < weights.size(); i++)
				weights[i] += (0 - weights[i]) * l2_factor * learningrate;
		}
		
		#endif
		
	} /* namespace weightreg */
	
} /* namespace ai */
