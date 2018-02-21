////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "genetic_programming.hpp"
#include <random>
#include <assert.h>
#include <iostream>

////////////////////////////////////////////////////////////
/// NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	///	NAMESPACE GENETIC PROGRAMMING
	////////////////////////////////////////////////////////////
	namespace gp
	{
		//generate random numbers between 0 and 1
		float randf() { return (double)(rand() % RAND_MAX) / (double)RAND_MAX; }
		
		////////////////////////////////////////////////////////////
		std::string node_to_string(ExpressionNode* node)
		{
			std::string s;

			if (node->node_type == gp::Operation)
			{
				if (node->operation_type == ai::gp::Summation) s = " + ";
				else if (node->operation_type == ai::gp::Subtraction) s = " - ";
				else if (node->operation_type == ai::gp::Multiplication) s = " * ";
				else if (node->operation_type == ai::gp::Division) s = " / ";
			}
			else if (node->node_type == gp::Variable)
			{
				s = " X" + std::to_string(node->variable_id) + " ";
			}
			else
			{
				s = std::to_string(node->value);
			}

			return s;
		}
		
		////////////////////////////////////////////////////////////
		std::string program_parse(ExpressionNode* root)
		{
			if (root == nullptr) return "";
			if (root->child_left != nullptr)
			{
				return "(" + program_parse(root->child_left) + node_to_string(root) +
					program_parse(root->child_right) + ")";
			}
			else
			{
				return node_to_string(root);
			}	
		}
			
		////////////////////////////////////////////////////////////
		float program_evaluate(ExpressionNode* root, const std::vector<float>& variables)
		{
			if (root == nullptr) return 0;
			if (root->node_type == gp::Constant) return root->value;
			if (root->node_type == gp::Variable) {
				assert(root->variable_id < variables.size());
				return variables[root->variable_id];
			}
			const float val_left = program_evaluate(root->child_left, variables);
			const float val_right = program_evaluate(root->child_right, variables);
			if (root->operation_type == ai::gp::Summation) return val_left + val_right;
			else if (root->operation_type == ai::gp::Subtraction) return val_left - val_right;
			else if (root->operation_type == ai::gp::Multiplication) return val_left * val_right;
			else return val_left / val_right;
		}
		
		////////////////////////////////////////////////////////////
		ExpressionNode* program_random_initialization(int current_depth, int max_depth, std::vector<ExpressionNode*>& terminal_nodes)
		{
			if (current_depth != max_depth)
			{
				//Create operation
				ExpressionNode* new_node = new ExpressionNode;
				new_node->node_type = gp::Operation;
				new_node->operation_type = rand() % (int)(gp::OperationsCount);
				new_node->child_left = program_random_initialization(current_depth + 1, max_depth, terminal_nodes);
				new_node->child_right = program_random_initialization(current_depth + 1, max_depth, terminal_nodes);
				return new_node;
			}
			else
			{
				//Place terminal node
				ExpressionNode* new_node = terminal_nodes.back();
				terminal_nodes.pop_back();
				return new_node;
			}
		}
		
		////////////////////////////////////////////////////////////
		ExpressionNode* program_random_initialization(int depth, int variables, float min_constant, float max_constant)
		{
			assert(depth > 0);
			assert(min_constant < max_constant);
			assert(variables >= 0);
			const int terminal_nodes_count = pow(2, depth - 1);
			assert(terminal_nodes_count >= variables);

			//Create terminal nodes
			std::vector<ExpressionNode*> terminal_nodes(terminal_nodes_count);
			for (int i = 0; i < terminal_nodes.size(); i++) {
				terminal_nodes[i] = new ExpressionNode;
				if (i < variables)
				{
					terminal_nodes[i]->node_type = gp::Variable;
					terminal_nodes[i]->variable_id = i;
				}
				else
				{
					terminal_nodes[i]->node_type = gp::Constant;
					terminal_nodes[i]->value = min_constant + gp::randf() * (max_constant - min_constant);
				}
				terminal_nodes[i]->child_left = nullptr;
				terminal_nodes[i]->child_right = nullptr;
			}

			//randomly shuffle terminal nodes
			ExpressionNode* tmp;
			for (int i = 0; i < terminal_nodes.size(); i++) {
				const int first = rand() % terminal_nodes.size();	
				const int second = rand() % terminal_nodes.size();
				tmp = terminal_nodes[first];
				terminal_nodes[first] = terminal_nodes[second];
				terminal_nodes[second] = tmp;
			}

			//Create tree using terminal nodes
			return program_random_initialization(1, depth, terminal_nodes);
		}
		
		////////////////////////////////////////////////////////////
		int program_nodes_count(ExpressionNode* root)
		{
			if (root == nullptr) return 0;
			return 1 + program_nodes_count(root->child_left) + program_nodes_count(root->child_right);
		}
		
		////////////////////////////////////////////////////////////
		ExpressionNode* program_crossover(ExpressionNode* first_root, ExpressionNode* second_root, int& node_id, int crossover_position)
		{
			if (first_root == nullptr || second_root == nullptr) return nullptr;
			if (node_id == 0)
			{
				ExpressionNode* new_node = new ExpressionNode;
				new_node->node_type = first_root->node_type;
				new_node->operation_type = first_root->operation_type;
				new_node->variable_id = first_root->variable_id;
				new_node->value = first_root->value;

				node_id++;
				new_node->child_left = program_crossover(first_root->child_left, second_root->child_left, node_id, crossover_position); 
				new_node->child_right = program_crossover(first_root->child_right, second_root->child_right, node_id, crossover_position); 
				return new_node;
			}
			else
			{
				ExpressionNode* new_node = new ExpressionNode;
				if (node_id < crossover_position)
				{
					new_node->node_type = first_root->node_type;
					new_node->operation_type = first_root->operation_type;
					new_node->variable_id = first_root->variable_id;
					new_node->value = first_root->value;
				}
				else
				{
					new_node->node_type = second_root->node_type;
					new_node->operation_type = second_root->operation_type;
					new_node->variable_id = second_root->variable_id;
					new_node->value = second_root->value;
				}

				node_id++;
				new_node->child_left = program_crossover(first_root->child_left, second_root->child_left, node_id, crossover_position); 
				new_node->child_right = program_crossover(first_root->child_right, second_root->child_right, node_id, crossover_position); 
				return new_node;		
			}
		}
		
		////////////////////////////////////////////////////////////
		ExpressionNode* program_crossover_initialization(ExpressionNode* first_root, ExpressionNode* second_root)
		{
			//get number of nodes in the tree
			int nodes_count = program_nodes_count(first_root);
			assert(nodes_count != 0);

			//crossover position
			int cross_position = rand() % nodes_count;

			//create new tree from crossover
			ExpressionNode* root = nullptr;
			int node_id = 0;
			root = program_crossover(first_root, second_root, node_id, cross_position);

			return root;	
		}
		
		////////////////////////////////////////////////////////////
		ExpressionNode* program_copy_initialization(ExpressionNode* root)
		{
			if (root == nullptr) return nullptr;
			ExpressionNode* new_node = new ExpressionNode;
			new_node->node_type = root->node_type;
			new_node->operation_type = root->operation_type;
			new_node->variable_id = root->variable_id;
			new_node->value = root->value;
			new_node->child_left = program_copy_initialization(root->child_left);
			new_node->child_right = program_copy_initialization(root->child_right);
			return new_node;
		}
		
		////////////////////////////////////////////////////////////
		void program_mutate(ExpressionNode*& root, float mutation_probability, int variables_count,
				float min_constant, float max_constant)
		{
			if (root == nullptr) return;
			if (randf() < mutation_probability) {
				if (root->node_type == gp::Operation)
				{
					root->operation_type = rand() % (int)(gp::OperationsCount);
				}
				else
				{
					if (rand() % 2 == 0)
					{
						root->node_type = gp::Variable;
						root->variable_id = rand() % variables_count;
					}
					else
					{
						root->node_type = gp::Constant;
						root->value = min_constant + randf() * (max_constant - min_constant); 
					}
				}
			}
			program_mutate(root->child_left, mutation_probability, variables_count, min_constant, max_constant);
			program_mutate(root->child_right, mutation_probability, variables_count, min_constant, max_constant);
		}
		
		////////////////////////////////////////////////////////////
		void program_free(ExpressionNode* node)
		{
			if (node == nullptr) return;
			program_free(node->child_left);
			program_free(node->child_right);
			delete node;
		}

	} /* namespace gp */

} /* namespace ai */
