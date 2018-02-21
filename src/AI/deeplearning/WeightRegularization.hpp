#ifndef WEIGHTREGULARIZATION_HPP
#define WEIGHTREGULARIZATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../util/Macros.hpp"
#include "../util/Tensor.hpp"
#ifdef CUDA_BACKEND
#include "../util/TensorCUDA.hpp"
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
		
		void l1_regularization(TensorCUDA_float& weights, const float l1_factor, const float learningrate);
		void l2_regularization(TensorCUDA_float& weights, const float l2_factor, const float learningrate);

		#else

		void l1_regularization(Tensor_float& weights, const float l1_factor, const float learningrate);
		void l2_regularization(Tensor_float& weights, const float l2_factor, const float learningrate);
		
		#endif

	} /* namespace weightregularization */

} /* namespace ai */

#endif /* end of include guard: WEIGHTREGULARIZATION_HPP */

