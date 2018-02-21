#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <vector>
#include <string>
#include "Operation.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	class Normalization : public Operation
	{
		public:
			Normalization(float momentum = 0.1);
			Normalization(ai::IOData& data);
			void save(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(float momentum = 0.1);

		private:
			int _width, _height, _depth;
			float _gamma;
			float _beta;
			float _epsilon;
			float _momentum;
			
			//Foreward informations
			double _mean;
			double _variance;
			#ifdef CUDA_BACKEND
			TensorCUDA_float _deviation;
			TensorCUDA_float _normalized;
			TensorCUDA_float _params;
			#else
			Tensor_float _deviation;
			Tensor_float _normalized;
			#endif

			//Backward informations
			float _d_beta;
			float _d_gamma;
	};

} /* namespace ai */

#endif /* end of include guard: NORMALIZATION_HPP */

