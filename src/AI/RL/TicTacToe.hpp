#ifndef TICTACTOE_HPP
#define TICTACTOE_HPP

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

	class TicTacToe : public Environment
	{
		public:
			
			enum BoardState 
			{
				Empty,
				AgentMark,
				OpponentMark
			};

			enum Winner
			{
				Unknown,
				Agent,
				Opponent,
				Draw
			};

			TicTacToe();
			void observe(ai::Tensor_float& experience, float& previous_reward);
			void act(int action_id);
			void user_play(int actionid, Tensor_float& experience);
			const std::vector<bool>& act_mask();
			void print_table();
			const Winner getWinner();
			
			void _unit_testing();
			

		private:
			void recalculate_act_mask();
			void clear_table();
			void flip_table();
			void compute_winner();
			float reward_function();
			void opponent_take_action();

			std::vector<BoardState> _board_states;
			std::vector<bool> _act_mask;
			Winner _winner;
			float _reward;

			static const int board_size = 9;
			static const int board_width = 3;
			static const int board_height = 3;
	};

} /* namespace ai */

#endif /* end of include guard: TICTACTOE_HPP */

