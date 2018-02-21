#ifndef RESIDUALBLOCK_HPP
#define RESIDUALBLOCK_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include "Operation.hpp"
#include "Convolution.hpp"
#include "Normalization.hpp"
#include "Relu.hpp"
#include "Addition.hpp"
#include <vector>
#include <memory>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	class ResidualBlock : public Operation
	{
		public:
			ResidualBlock(const int filter_size, const int filer_count, const int stride, const int padding, const unsigned int blocks_count);
			ResidualBlock(ai::IOData& data);
			void initialize(std::vector<Operation*> &inputs);
			void initialize(const int input_width, const int input_height, const int input_count);
			void save(ai::IOData& data);
			void run(std::vector<Operation*> &inputs, const bool training);
			void backprop(std::vector<Operation*> &inputs);
			void accumulate_deltas(std::vector<Operation*> &inputs);
			void update_parameters(const float learningrate);
			void reset_deltas(const double momentum);
			void reset_errors();
			void reset_outputs();
			const Operation::Type get_type() const;
			void print();
			
			static std::shared_ptr<Operation> make(const int filter_size, const int filter_count, const int stride, const int padding, const unsigned int blocks_count);
			
			std::vector< std::unique_ptr< ai::Operation > > _layers;
			std::vector< std::vector< Operation* > > _layers_connections;

			std::vector< Operation* > link_layers(Operation* op);
			std::vector< Operation* > link_layers(Operation* op1, Operation* op2);

			unsigned int _input_width;
			unsigned int _input_height;
			unsigned int _input_count;
			unsigned int _filter_width;
			unsigned int _filter_height;
			unsigned int _filter_count;
			unsigned int _stride;
			unsigned int _input_size;
			unsigned int _padding;
			unsigned int _blocks_count;
	};

} /* namespace ai */

#endif /* end of include guard: RESIDUALBLOCK_HPP */

