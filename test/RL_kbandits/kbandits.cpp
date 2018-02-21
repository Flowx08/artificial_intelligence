#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/RL/Kbandits.hpp"
#include "../../src/AI/RL/DQN.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"
#include <random>
#include <stdlib.h>

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));

	//Create environment
	ai::Kbandits environment(10, 0.1, 0.9);
	ai::Tensor_float experience(environment.input_width(),
		environment.input_height(), environment.input_depth());
	experience.fill(0);
	float previous_reward;
	
	//Create agent
	int action_id;
	ai::NeuralNetwork ai_model;
	ai_model.push("INPUT", "", ai::Variable::make(environment.input_width()));
	ai_model.push("L1", "INPUT", ai::Linear::make(5));
	ai_model.push("TANH", "L1", ai::Tanh::make());
	ai_model.push("OUTPUT", "TANH", ai::Linear::make(environment.actions_count()));
	
	ai::DQN agent(30, 0.005, 0.25, 0.0);
	agent.set_neuralnetwork(ai_model);
	
	//World steps
	const int max_steps = 200000;

	//Statistics
	float medium_reward = 0;
	float total_area_reward = 0;
	std::vector<int> action_performed_count(environment.actions_count(), 0);

	for (int i = 0; i < max_steps; i++) {

		//Agent-World interaction
		action_id = agent.act(experience);
		environment.act(action_id);
		environment.observe(experience, previous_reward);
		agent.observe(experience, previous_reward);
		
		//Update statistics
		medium_reward += previous_reward;
		total_area_reward += previous_reward;
		action_performed_count[action_id]++;

		if (i % 1000 == 0 && i != 0) {
			medium_reward /= (float)1000.f;
			printf("WorldStep: %d Medium reward: %f\n", i, medium_reward);
		}

	}

	for (int i = 0; i < (int)environment.get_reward_prob_vector().size(); i++) {
		printf("Action: %d RewardProability: %f PerformedCount: %d\n", i,
			environment.get_reward_prob_vector()[i], action_performed_count[i]);
	}
	printf("TotalReward: %f\n", total_area_reward);


	return 0;
}
