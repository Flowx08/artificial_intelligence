#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

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
	class Softmax : public Operation
	{
		public:
			Softmax(double input_scale = 1.f);
			Softmax(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void save(ai::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();
			static std::shared_ptr<Operation> make(double input_scale = 1.f);

			float _input_scale;

		private:
			float _epsilon;
	};

} /* namespace ai */

#endif /* end of include guard: SOFTMAX_HPP */

