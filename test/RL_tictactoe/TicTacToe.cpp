#include "../../src/AI/deeplearning/NeuralNetwork.hpp"
#include "../../src/AI/RL/TicTacToe.hpp"
#include "../../src/AI/RL/DQN.hpp"
#include "../../src/AI/deeplearning/OptimizerSDG.hpp"
#include <random>
#include <stdlib.h>

int main(int argc, const char *argv[])
{
	srand((int)time(NULL));

	//Create environment
	ai::TicTacToe environment;
	ai::Tensor_float experience(environment.input_width(),
		environment.input_height(), environment.input_depth());
	experience.fill(0);
	float previous_reward;

	environment._unit_testing();

	//Create agent
	int action_id;
	
	ai::NeuralNetwork net;
	net.push("INPUT", "", ai::Variable::make(environment.input_width(), environment.input_height()));
	net.push("L1", "INPUT", ai::Linear::make(32));
	net.push("REL1", "L1", ai::Relu::make());
	net.push("NORM", "REL1", ai::Normalization::make(0.5));
	net.push("L2", "NORM", ai::Linear::make(32));
	net.push("NORM2", "L2", ai::Normalization::make(0.5));
	net.push("REL2", "NORM2", ai::Relu::make());
	net.push("OUTPUT", "REL2", ai::Linear::make(environment.actions_count()));
	
	ai::DQN ai(net, 50000, 0.001, 0.6, 0.9);
	
	//World steps
	const int max_steps = 30000;

	//Statistics
	float medium_reward = 0;
	float total_area_reward = 0;
	std::vector<int> action_performed_count(environment.actions_count(), 0);
	int won_counter = 0, lost_counter = 0, draw_counter = 0;

	for (int i = 0; i < max_steps; i++) {

		if (i < max_steps * 0.95) ai.setcuriosity( 1.f - ((float)i / (max_steps * 0.95)) );
		else ai.setcuriosity(0);

		//Agent-World interaction
		action_id = ai.act(experience, environment.act_mask());
		environment.act(action_id);
		environment.observe(experience, previous_reward);
		ai.observe(experience, previous_reward);
		if (environment.getWinner() == ai::TicTacToe::Agent) won_counter++;
		else if (environment.getWinner() == ai::TicTacToe::Opponent) lost_counter++;
		else if (environment.getWinner() == ai::TicTacToe::Draw) draw_counter++;
		
		if (i > max_steps - 30) {
			environment.print_table();
		}

		//Update statistics
		medium_reward += previous_reward;
		total_area_reward += previous_reward;
		action_performed_count[action_id]++;

		if (i % 1000 == 0 && i != 0) {
			medium_reward /= (float)1000.f;
			printf("WorldStep: %d Medium reward: %f Curiosity: %f Won: %d Lost: %d Draw: %d\n",
				i, medium_reward, ai.getcuriosity(), won_counter, lost_counter, draw_counter);
			won_counter = 0;
			lost_counter = 0;
			draw_counter = 0;
		}

	}

	printf("TotalReward: %f\n", total_area_reward);

	for (int i = 0; i < 20; i++) {
		action_id = ai.act(experience, environment.act_mask());
		environment.act(action_id);
		environment.print_table();
		int useraction = 0;
		scanf("action: %d\n", &useraction);
		printf("my actiond: %d\n", useraction);
		environment.user_play(useraction, experience);
	}

	return 0;
}
