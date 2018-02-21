#ifndef PARTIAL_HPP
#define PARTIAL_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "Operation.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	class Partial : public Operation
	{
		public:
			Partial(const int size, const double connectivity);
			Partial(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(ai::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			void reset_deltas(const double momentum);
			void pruning(float alpha);
			float pruned_percent();
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(const int size, const double connectivity);
			
			#ifdef CUDA_BACKEND
			TensorCUDA_float _weights;
			TensorCUDA_float _bias;
			TensorCUDA_float _deltas;
			#else
			Tensor_float _weights;
			Tensor_float _bias;
			Tensor_float _deltas;
			#endif
			std::vector< std::vector< int > > _foreward_map;
			int _input_size;
			double _connectivity;
	};

} /* namespace ai */

#endif /* end of include guard: PARTIAL_HPP */

