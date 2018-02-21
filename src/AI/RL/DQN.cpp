////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "DQN.hpp"
#include "../util/Util.hpp"
#include <float.h>
#include <algorithm>

////////////////////////////////////////////////////////////
///	AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	DQN::DQN(ai::NeuralNetwork& net, int short_term_mem_size, float learningrate, float curiosity, float longtermexploitation)
	{
		_experience_current = Experience();
		_internalmodel = &net;
		_short_term_mem_size = short_term_mem_size;
		_short_term_mem_pos = 0;
		_short_term_mem_sampling_size = 10;
		_short_term_mem = std::vector<Experience>();
		_actionid = 0;
		_curiosity = curiosity;
		_learningrate = learningrate;
		_longtermexploitation = longtermexploitation;
		_optimizer = OptimizerSDG(_short_term_mem_sampling_size, learningrate, 0.0, Cost::SquaredError);
	}

	////////////////////////////////////////////////////////////
	int DQN::act(Tensor_float& state) 
	{
		//Shortcut output node
		Operation* output_node = _internalmodel->getNodes().back().getOperation();

		//Create new experience
		_experience_current = Experience();
		_experience_current.state = state;
		_internalmodel->run(state);
		_experience_current.action = output_node->_outputs;

		//Exploration / Exploitation
		if (ai::util::randf() < _curiosity)
		{
			//Explore the state-action space randomly
			_actionid = rand() % output_node->_outputs.size();
			_experience_current.actionid = _actionid;
		}
		else
		{
			//Exploit the state-action space intelligently
			_actionid = getmaxpos(_experience_current.action);
			_experience_current.actionid = _actionid;
		}

		//return the best action id
		return _actionid;
	}
	
	////////////////////////////////////////////////////////////
	int DQN::act(Tensor_float& state, const std::vector<bool>& allowed_actions)
	{
		//Shortcut output node
		Operation* output_node = _internalmodel->getNodes().back().getOperation();

		//Create new experience
		_experience_current = Experience();
		_experience_current.state = state;
		_internalmodel->run(state);
		_experience_current.action = output_node->_outputs;

		//Exploration / Exploitation
		if (ai::util::randf() < _curiosity)
		{
			//Explore the state-action space randomly
			std::vector<int> allowed_actions_ids;
			for (int i = 0; i < (int)allowed_actions.size(); i++)
				if (allowed_actions[i] == true) allowed_actions_ids.push_back(i);
			_actionid = allowed_actions_ids[rand() % allowed_actions_ids.size()];
			_experience_current.actionid = _actionid;
			ensure(allowed_actions[_actionid]);
		}
		else
		{
			//Exploit the state-action space intelligently
			_actionid = getmaxpos(_experience_current.action, allowed_actions);
			_experience_current.actionid = _actionid;
		}
		
		//Convergence problem...
		ensure_print(allowed_actions[_actionid], "%s\n", _experience_current.action.tostring().c_str());

		//return the best action id
		return _actionid;
		
	}
	
	////////////////////////////////////////////////////////////
	void DQN::teach(Tensor_float& state, int actionid)
	{
		//Create new experience
		_experience_current = Experience();
		_experience_current.state = state;
		_experience_current.actionid = actionid;
		_actionid = actionid;
	}
	
	////////////////////////////////////////////////////////////
	void DQN::observe(Tensor_float &newstate, float oldreward)
	{
		if (_experience_current.state.size() == 0) return;
		
		//Shortcut output node
		Operation* output_node = _internalmodel->getNodes().back().getOperation();

		//Update experience informations
		_experience_current.state_next = newstate;
		_experience_current.reward = oldreward;

		//Store experience in the short term memory
		if ((int)_short_term_mem.size() < _short_term_mem_size)
		{
			_short_term_mem.push_back(_experience_current);
		}
		else
		{
			_short_term_mem[_short_term_mem_pos] = _experience_current;
			if (++_short_term_mem_pos >= _short_term_mem_size) _short_term_mem_pos = 0;
		}

		//Update the internal agent world model with one or more random experiences 
		//from the short-term memory buffer
		if ((int)_short_term_mem.size() < _short_term_mem_sampling_size) return;
		for (int i = 0; i < _short_term_mem_sampling_size; i++) {

			//sample random experience
			int exp_pos = rand() % _short_term_mem.size();
			
			//Calculate reward for this experience
			float reward;
			if (_short_term_mem[exp_pos].state_next.size() == 0) //it's a final state
			{
				//r = r_a
				reward = _short_term_mem[exp_pos].reward;
			}
			else
			{
				//r = r_a + y * max(Q(s'))
				_internalmodel->run(_short_term_mem[exp_pos].state_next);
				float next_q_values;
				output_node->_outputs.max(&next_q_values, NULL);
				reward = _short_term_mem[exp_pos].reward + _longtermexploitation * next_q_values;
			}
			
			//Update the internal agent model with the new action-reward combination
			_internalmodel->run(_short_term_mem[exp_pos].state);
			Tensor_float actions;
			actions.copy(output_node->_outputs);
			actions[_short_term_mem[exp_pos].actionid] = reward;
			_internalmodel->optimize(_short_term_mem[exp_pos].state, actions, &_optimizer);
		}
	}
	
	////////////////////////////////////////////////////////////
	void DQN::setcuriosity(double curiosity)
	{
		_curiosity = curiosity;
	}
	
	////////////////////////////////////////////////////////////
	void DQN::setlongtermexploitation(double longtermexploitation)
	{
		_longtermexploitation = longtermexploitation;
	}

	////////////////////////////////////////////////////////////
	void DQN::setlearningrate(double learningrate)
	{
		_learningrate = learningrate;
	}
	
	////////////////////////////////////////////////////////////
	const Experience DQN::getcurrentexperience()
	{
		return _experience_current;
	}
	
	////////////////////////////////////////////////////////////
	int DQN::getlastaction()
	{
		return _actionid;	
	}
	
	////////////////////////////////////////////////////////////
	double DQN::getcuriosity()
	{
		return _curiosity;
	}
	
	////////////////////////////////////////////////////////////
	double DQN::getlongtermexploitation()
	{
		return _longtermexploitation;
	}
		
	////////////////////////////////////////////////////////////
	double DQN::getlearningrate()
	{
		return _learningrate;
	}
	
	////////////////////////////////////////////////////////////
	const NeuralNetwork& DQN::getneuralnet()
	{
		return *_internalmodel;
	}
	
	////////////////////////////////////////////////////////////
	int DQN::getmaxpos(Tensor_float &data)
	{
		double max = -DBL_MAX;
		int id = 0;
		for (int i = 0; i < (int)data.size(); i++) {
			if (data[i] > max) {
				max = data[i];
				id = i;
			}
		}
		return id;
	}
	
	////////////////////////////////////////////////////////////
	int DQN::getmaxpos(Tensor_float &data, const std::vector<bool>& mask)
	{
		ensure((int)mask.size() == data.size());
		double max = -DBL_MAX;
		int id = 0;
		for (int i = 0; i < (int)data.size(); i++) {
			if (mask[i] == false) continue;
			if (data[i] > max) {
				max = data[i];
				id = i;
			}
		}
		return id;
	}

} //namespace AI
