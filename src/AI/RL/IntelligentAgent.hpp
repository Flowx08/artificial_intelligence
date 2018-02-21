#ifndef INTELLIGENTAGENT_HPP
#define INTELLIGENTAGENT_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../util/Tensor.hpp"
#include <vector>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	class IntelligentAgent
	{
		public:
			virtual int act(ai::Tensor_float& state);
			virtual int act(ai::Tensor_float& state, std::vector<bool>& allowed_actions);
			virtual void teach(ai::Tensor_float& state, const int actionid);
			virtual void observe(ai::Tensor_float& newstate, const float oldreward);
	};

} /* namespace ai */

#endif /* end of include guard: INTELLIGENTAGENT_HPP */

