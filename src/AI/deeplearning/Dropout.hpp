#ifndef DROPOUT_HPP
#define DROPOUT_HPP

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
	class Dropout : public Operation
	{
		public:
			Dropout(const double drop_probability);
			Dropout(ai::IOData& data);
			void save(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(const double drop_probability);

			float _drop_probability;
		private:
			#ifdef CUDA_BACKEND
			ai::cudnn::Dropout _cuda_dropout;
			ai::TensorCUDA_float _state_buffer, _reserve_space_buffer;
			#endif
			int _width, _height, _depth;
	};

} /* namespace ai */

#endif /* end of include guard: DROPOUT_HPP */

