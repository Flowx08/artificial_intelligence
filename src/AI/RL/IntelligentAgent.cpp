////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "IntelligentAgent.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	 int IntelligentAgent::act(ai::Tensor_float& state) { return 0; }
	 int IntelligentAgent::act(ai::Tensor_float& state, std::vector<bool>& allowed_actions) { return 0; }
	 void IntelligentAgent::teach(ai::Tensor_float& state, const int actionid) {}
	 void IntelligentAgent::observe(ai::Tensor_float& newstate, const float oldreward) {}

} /* namespace ai */
