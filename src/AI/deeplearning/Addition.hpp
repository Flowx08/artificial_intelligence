#ifndef ADDITION_HPP
#define ADDITION_HPP

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
	class Addition : public Operation
	{
		public:
			Addition();
			Addition(ai::IOData& data);
			void save(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			const Operation::Type get_type() const;
			void print();

			static std::shared_ptr<Operation> make();
		
		private:
			int _width, _height, _depth;
	};

} /* namespace ai */

#endif /* end of include guard: ADDITION_HPP */

