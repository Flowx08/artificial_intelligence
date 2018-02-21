#ifndef ENVIRONMENT_HPP
#define ENVIRONMENT_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../util/Tensor.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class Environment
	{
		public:
			virtual ~Environment();
			virtual void observe(ai::Tensor_float& experience, float& previous_reward);
			virtual void act(ai::Tensor_float action);
			const unsigned int input_width() const;
			const unsigned int input_height() const;
			const unsigned int input_depth() const;
			const unsigned int input_size() const;
			const unsigned int actions_count() const;

		protected:
			unsigned int _input_width, _input_height, _input_depth;
			unsigned int _actions_count;
	};

} /* namespace ai */

#endif /* end of include guard: ENVIRONMENT_HPP */

