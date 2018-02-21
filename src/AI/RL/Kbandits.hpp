#ifndef KBANDITS_HPP
#define KBANDITS_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Environment.hpp"
#include <vector>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class Kbandits : public Environment
	{
		public:
			Kbandits(int bandits_count, const float min_reward = 0.3, const float max_reward = 0.8);
			void observe(ai::Tensor_float& experience, float& previous_reward);
			void act(int action_id);
			const std::vector<float> get_reward_prob_vector();
		private:
			void fill_reward_vector(float min_reward, float max_reward);
			void shuffle_reward_vector();
			std::vector<float> _reward_probability;
			float _reward;
	};

} /* namespace ai */

#endif /* end of include guard: KBANDITS_HPP */

