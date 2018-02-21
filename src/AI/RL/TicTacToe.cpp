////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "TicTacToe.hpp"
#include "../util/ensure.hpp"
#include "../util/Util.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	TicTacToe::TicTacToe() 
	{
		_input_width = 3;
		_input_height = 3;
		_input_depth = 1;
		_actions_count = board_size;
		_reward = 0;
		_board_states = std::vector<BoardState>(board_size);
		_winner = Unknown;
		clear_table();
		recalculate_act_mask();
	}

	////////////////////////////////////////////////////////////
	void TicTacToe::observe(ai::Tensor_float& experience, float& previous_reward) 
	{
		opponent_take_action();
		compute_winner();
		_reward = reward_function();
		if (getWinner() != Unknown) {
			clear_table();
			//Opponent act first?
			if (util::randint() % 2 == 0)
				opponent_take_action();
		}
		recalculate_act_mask();
		if (experience.size() != board_size)
			experience.setshape(board_width, board_height);

		for (int i = 0; i < board_size; i++) {
			if (_board_states[i] == AgentMark) experience[i] = 1.f;
			else if (_board_states[i] == OpponentMark) experience[i] = -1.f;
			else experience[i] = 0.f;
		}

		previous_reward = _reward;
	}

	////////////////////////////////////////////////////////////
	void TicTacToe::act(int action_id) 
	{
		ensure(_board_states[action_id] == Empty);
		_board_states[action_id] = AgentMark;
	}
	
	////////////////////////////////////////////////////////////
	float TicTacToe::reward_function()
	{
		if (getWinner() == Unknown) return 0;
		else if (getWinner() == Agent) return 1;
		else if (getWinner() == Opponent) return -1;
		else if (getWinner() == Draw) return 0.2;
		return 0;
	}
	
	////////////////////////////////////////////////////////////
	const std::vector<bool>& TicTacToe::act_mask()
	{
		return _act_mask;
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::recalculate_act_mask()
	{
		if ((int)_act_mask.size() != board_size)
			_act_mask = std::vector<bool>(_board_states.size());
		for (unsigned int i = 0; i < _board_states.size(); i++) {
			if (_board_states[i] == Empty) _act_mask[i] = true;
			else _act_mask[i] = false;
		}
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::clear_table()
	{
		for (unsigned int i = 0; i < _board_states.size(); i++)
			_board_states[i] = Empty;
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::flip_table()
	{
		for (unsigned int i = 0; i < _board_states.size(); i++) { 
			if (_board_states[i] == AgentMark) _board_states[i] = OpponentMark;
			else if (_board_states[i] == OpponentMark) _board_states[i] = AgentMark;
		}
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::compute_winner()
	{
		int match = 0;
		_winner = Unknown;

		//CHECK AGENT

		//Check columns
		for (int x = 0; x < board_width; x++) {
			match = 0; 
			for (int y = 0; y < board_height; y++) {
				if (_board_states[y * board_width + x] == AgentMark) match++;
			}
			if (match == board_height) {
				_winner = Agent;
				return;
			}
		}
		
		//Check rows
		for (int y = 0; y < board_height; y++) {
			match = 0; 
			for (int x = 0; x < board_width; x++) {
				if (_board_states[y * board_width + x] == AgentMark) match++;
			}
			if (match == board_width) {
				_winner = Agent;
				return;
			}
		}
		
		//Check obliques 1
		match = 0; 
		for (int x = 0; x < board_width; x++)
			if (_board_states[x * board_width + x] == AgentMark) match++;
		if (match == board_width) {
			_winner = Agent;
			return;
		}
		
		//Check obliques 2
		match = 0; 
		for (int x = 0; x < board_width; x++)
			if (_board_states[(board_height - 1 - x) * board_width + x] == AgentMark) match++;
		if (match == board_width) {
			_winner = Agent;
			return;
		}
		
		//CHECK OPPONENT

		//Check columns
		for (int x = 0; x < board_width; x++) {
			match = 0; 
			for (int y = 0; y < board_height; y++) {
				if (_board_states[y * board_width + x] == OpponentMark) match++;
			}
			if (match == board_height) {
				_winner = Opponent;
				return;
			}
		}
		
		//Check rows
		for (int y = 0; y < board_height; y++) {
			match = 0; 
			for (int x = 0; x < board_width; x++) {
				if (_board_states[y * board_width + x] == OpponentMark) match++;
			}
			if (match == board_width) {
				_winner = Opponent;
				return;
			}
		}
		
		//Check obliques 1
		match = 0; 
		for (int x = 0; x < board_width; x++)
			if (_board_states[x * board_width + x] == OpponentMark) match++;
		if (match == board_width) {
			_winner = Opponent;
			return;
		}
		
		//Check obliques 2
		match = 0; 
		for (int x = 0; x < board_width; x++) {
			if (_board_states[(board_height - 1 - x) * board_width + x] == OpponentMark) match++;
		}
		if (match == board_width) {
			_winner = Opponent;
			return;
		}
		
		//Check for draw
		for (int x = 0; x < board_size; x++)
			if (_board_states[x] == Empty) return;

		_winner = Draw;
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::_unit_testing()
	{
		clear_table();
		compute_winner();
		ensure(_winner == Unknown);

		clear_table();
		_board_states[0] = AgentMark;
		_board_states[3] = AgentMark;
		_board_states[6] = AgentMark;
		compute_winner();
		ensure(_winner == Agent); 
		
		clear_table();
		_board_states[0] = AgentMark;
		_board_states[4] = AgentMark;
		_board_states[8] = AgentMark;
		compute_winner();
		ensure(_winner == Agent);
		
		clear_table();
		_board_states[2] = AgentMark;
		_board_states[4] = AgentMark;
		_board_states[6] = AgentMark;
		compute_winner();
		ensure(_winner == Agent);

		flip_table();
		compute_winner();
		ensure(_winner == Opponent);

		clear_table();
		_board_states[0] = OpponentMark;
		_board_states[1] = AgentMark;
		_board_states[2] = OpponentMark;
		_board_states[3] = AgentMark;
		_board_states[4] = AgentMark;
		_board_states[5] = OpponentMark;
		_board_states[6] = AgentMark;
		_board_states[7] = OpponentMark;
		_board_states[8] = AgentMark;
		compute_winner();
		ensure_print(_winner == Draw, "Winner: %d\n", (int)_winner);

		clear_table();
		_winner = Unknown;
	}

	////////////////////////////////////////////////////////////
	void TicTacToe::print_table()
	{
		printf("======\n");
		for (int y = 0; y < board_height; y++) {
			for (int x = 0; x < board_width; x++) {
				if (_board_states[y * board_width + x] == Empty) printf("_ ");	
				else if (_board_states[y * board_width + x] == AgentMark) printf("X ");	
				else if (_board_states[y * board_width + x] == OpponentMark) printf("O ");	
			}
			printf("\n");
		}
		printf("======\n");
	}
	
	////////////////////////////////////////////////////////////
	const TicTacToe::Winner TicTacToe::getWinner()
	{
		return _winner;
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::opponent_take_action()
	{
		//Random take action
		std::vector<int> available_actions;
		for (int i = 0; i < board_size; i++) {
			if (_board_states[i] == Empty)
				available_actions.push_back(i);
		}
		
		if (available_actions.size() != 0) {
			_board_states[available_actions[util::randint() % available_actions.size()]] = OpponentMark;
		}
	}
	
	////////////////////////////////////////////////////////////
	void TicTacToe::user_play(int actionid, Tensor_float& experience)
	{
		ensure(_board_states[actionid] == Empty);
		_board_states[actionid] = OpponentMark;
		compute_winner();
		if (getWinner() != Unknown) {
			clear_table();
		}
		recalculate_act_mask();
		if (experience.size() != board_size)
			experience.setshape(board_width, board_height);
		
		for (int i = 0; i < board_size; i++) {
			if (_board_states[i] == AgentMark) experience[i] = 1.f;
			else if (_board_states[i] == OpponentMark) experience[i] = -1.f;
			else experience[i] = 0.f;
		}
	}

} /* namespace ai */
