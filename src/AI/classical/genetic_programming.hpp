#ifndef GENETIC_PROGRAMMING_HPP
#define GENETIC_PROGRAMMING_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include <vector>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	///	NAMESPACE GENETIC PROGRAMMING
	////////////////////////////////////////////////////////////
	namespace gp
	{

		enum NodeType
		{
			Operation,
			Variable,
			Constant
		};

		enum OperationType
		{
			Summation,
			Subtraction,
			Multiplication,
			Division,
			OperationsCount
		};

		struct ExpressionNode {
			int node_type;
			int operation_type;
			int variable_id;
			float value;
			ExpressionNode* child_left;
			ExpressionNode* child_right;
		};

		std::string program_parse(ExpressionNode* root);
		float program_evaluate(ExpressionNode* root, const std::vector<float>& variables);
		ExpressionNode* program_random_initialization(int depth, int variables, float min_constant, float max_constant);
		ExpressionNode* program_crossover_initialization(ExpressionNode* first_root, ExpressionNode* second_root);
		ExpressionNode* program_copy_initialization(ExpressionNode* root);
		void program_mutate(ExpressionNode*& root, float mutation_probability, int variables_count, float min_constant, float max_constant);
		void program_free(ExpressionNode* node);

	} /* namespace gp */

} /* namespace ai */

#endif /* end of include guard: GENETIC_PROGRAMMING_HPP */

