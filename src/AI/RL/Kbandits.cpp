////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Kbandits.hpp"
#include "../util/Util.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	Kbandits::Kbandits(int bandits_count, const float min_reward, const float max_reward)
	{
		ensure(min_reward < max_reward);
		ensure(min_reward >= -1.f && min_reward <= 1.f);
		ensure(max_reward >= -1.f && max_reward <= 1.f);
		ensure(bandits_count > 0 && bandits_count < 1000);
		_input_width = 1;
		_input_height = 1;
		_input_depth = 1;
		_actions_count = bandits_count;
		_reward = 0;
		_reward_probability = std::vector<float>(bandits_count);
		fill_reward_vector(min_reward, max_reward);
		shuffle_reward_vector();
	}
	
	////////////////////////////////////////////////////////////
	void Kbandits::fill_reward_vector(float min_reward, float max_reward)
	{
		for (unsigned int i = 0; i < _reward_probability.size(); i++)
			_reward_probability[i] = min_reward + (max_reward - min_reward) * ((float)i / (float)_reward_probability.size());
	}
	
	////////////////////////////////////////////////////////////
	void Kbandits::shuffle_reward_vector()
	{
		if (_reward_probability.size() <= 1) return;

		float tmp_value = 0;
		unsigned int tmp_index = 0;
		for (unsigned int i = 0; i < _reward_probability.size(); i++) {
			tmp_index = ai::util::randint() % _reward_probability.size();
			if (tmp_index == i) continue;
			tmp_value = _reward_probability[i];
			_reward_probability[i] = _reward_probability[tmp_index];
			_reward_probability[tmp_index] = tmp_value;
		}
	}

	////////////////////////////////////////////////////////////
	void Kbandits::observe(ai::Tensor_float& experience, float& previous_reward)
	{
		experience.setshape(_input_width, _input_height, _input_depth);
		experience.fill(1);
		previous_reward = _reward;
	}

	////////////////////////////////////////////////////////////
	void Kbandits::act(int action_id)
	{
		ensure(action_id >= 0 && action_id < (int)_actions_count);
		if (ai::util::randf() < _reward_probability[action_id]) _reward = 1;
		else _reward = 0;
	}

	////////////////////////////////////////////////////////////
	const std::vector<float> Kbandits::get_reward_prob_vector()
	{
		return _reward_probability;
	}

} /* namespace ai */
