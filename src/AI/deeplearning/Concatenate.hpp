#ifndef CONCATENATE_HPP
#define CONCATENATE_HPP

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
	class Concatenate : public Operation
	{
		public:
			Concatenate();
			Concatenate(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(ai::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make();
			
		private:
			int _width, _height, _depth;
			#ifdef CUDA_BACKEND
			TensorCUDA_float_ptr _inputs_pointers;
			TensorCUDA_float_ptr _outerrors_pointers;
			TensorCUDA_int _pointers_sizes;
			#endif
	};

} /* namespace ai */

#endif /* end of include guard: CONCATENATE_HPP */

