////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "ResidualBlock.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#include "WeightRegularization.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif
#include "Cost.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai 
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> ResidualBlock::make(const int filter_size, const int filter_count, const int stride, const int padding, const unsigned int blocks_count)
	{
		return std::shared_ptr<Operation>(new ResidualBlock(filter_size, filter_count, stride, padding, blocks_count));
	}
	
	////////////////////////////////////////////////////////////
	ResidualBlock::ResidualBlock(const int filter_size, const int filter_count, const int stride, const int padding, const unsigned int blocks_count)
	{
		_filter_width = filter_size;
		_filter_height = filter_size;
		_filter_count = filter_count;
		_stride = stride;
		_padding = padding;
		_blocks_count = blocks_count;
	}
	
	////////////////////////////////////////////////////////////
	ResidualBlock::ResidualBlock(ai::IOData& data)
	{
		//TODO
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::save(ai::IOData& data)
	{
		//TODO
	}
	
	////////////////////////////////////////////////////////////
	std::vector< Operation* > ResidualBlock::link_layers(Operation* op)
	{
		std::vector< Operation* > graph(1);
		graph[0] = op;
		return graph;
	}

	////////////////////////////////////////////////////////////
	std::vector< Operation* > ResidualBlock::link_layers(Operation* op1, Operation* op2)
	{
		std::vector< Operation* > graph(2);
		graph[0] = op1;
		graph[1] = op2;
		return graph;
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::initialize(std::vector<Operation*> &inputs)
	{
		//Only one input allowed
		ensure(inputs.size() == 1); 
		
		//Store input informations
		_input_width = inputs[0]->_outputs.width();
		_input_height = inputs[0]->_outputs.height();
		_input_count = inputs[0]->_outputs.depth();
			
		int last_block_output_id;
		int mid_output_id;
		for (int i = 0; i < (int)_blocks_count; i++) {
			//printf("Block: %d\n", i);

			//printf("Conv 1\n");
			if (i == 0) _layers_connections.push_back(inputs);
			else _layers_connections.push_back(link_layers(_layers.back().get()));
			_layers.push_back(std::unique_ptr<ai::Convolution>(new ai::Convolution(_filter_width, _filter_count, _stride, _padding, 1)));
			_layers.back().get()->initialize(_layers_connections.back());
			
			//printf("Norm 1\n");
			//printf("%d %d %d\n", _layers.back().get()->_outputs.width(), _layers.back().get()->_outputs.height(), _layers.back().get()->_outputs.depth());
			_layers_connections.push_back(link_layers(_layers.back().get()));
			_layers.push_back(std::unique_ptr<ai::Normalization>(new ai::Normalization()));
			_layers.back().get()->initialize(_layers_connections.back());
			
			//printf("Relu 1\n");
			_layers_connections.push_back(link_layers(_layers.back().get()));
			_layers.push_back(std::unique_ptr<ai::Relu>(new ai::Relu()));
			_layers.back().get()->initialize(_layers_connections.back());
			
			//printf("Conv 2\n");
			_layers_connections.push_back(link_layers(_layers.back().get()));
			_layers.push_back(std::unique_ptr<ai::Convolution>(new ai::Convolution(_filter_width, _filter_count, _stride, _padding, 1)));
			_layers.back().get()->initialize(_layers_connections.back());
			
			//printf("Norm 2\n");
			_layers_connections.push_back(link_layers(_layers.back().get()));
			_layers.push_back(std::unique_ptr<ai::Normalization>(new ai::Normalization()));
			_layers.back().get()->initialize(_layers_connections.back());
			mid_output_id = _layers.size() -1;
			
			if (i == 0)
			{
				//printf("Conv 3\n");
				//We must have the same number of filters, we use 1x1 convolution to do this
				_layers_connections.push_back(inputs);
				_layers.push_back(std::unique_ptr<ai::Convolution>(new ai::Convolution(1, _filter_count, 1, 0, 1)));
				_layers.back().get()->initialize(_layers_connections.back());

				//printf("Norm 3\n");
				_layers_connections.push_back(link_layers(_layers.back().get()));
				_layers.push_back(std::unique_ptr<ai::Normalization>(new ai::Normalization()));
				_layers.back().get()->initialize(_layers_connections.back());
				
				//printf("Add 1\n");
				//Add results together
				_layers_connections.push_back(link_layers(_layers.back().get(), _layers[mid_output_id].get()));
				_layers.push_back(std::unique_ptr<ai::Addition>(new ai::Addition()));
				_layers.back().get()->initialize(_layers_connections.back());
			}
			else
			{
				//printf("Add 1\n");
				_layers_connections.push_back(link_layers(_layers.back().get(), _layers[last_block_output_id].get()));
				_layers.push_back(std::unique_ptr<ai::Addition>(new ai::Addition()));
				_layers.back().get()->initialize(_layers_connections.back());
			}
			
			//printf("Relu 2\n");
			_layers_connections.push_back(link_layers(_layers.back().get()));
			_layers.push_back(std::unique_ptr<ai::Relu>(new ai::Relu()));
			_layers.back().get()->initialize(_layers_connections.back());
			last_block_output_id = _layers.size() - 1;
		}
		
		ensure(_layers.size() == _layers_connections.size());

		//Wire output and error to last layer
		_outputs.point(_layers.back().get()->_outputs);
		_errors.point(_layers.back().get()->_errors);
		
		_size = _outputs.size();
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::initialize(const int input_width, const int input_height, const int input_count) {}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::run(std::vector<Operation*> &inputs, const bool training) 
	{
		for (int i = 0; i < (int)_layers.size(); i++)
			_layers[i].get()->run(_layers_connections[i], true);
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::backprop(std::vector<Operation*> &inputs) 
	{
		for (int i = (int)_layers.size()-1; i >= 0; i--)
			_layers[i].get()->backprop(_layers_connections[i]);
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		for (int i = 0; i < (int)_layers.size(); i++)
			_layers[i].get()->accumulate_deltas(_layers_connections[i]);
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::update_parameters(const float learningrate)
	{
		for (int i = 0; i < (int)_layers.size(); i++) _layers[i].get()->update_parameters(learningrate);
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::reset_deltas(const double momentum)
	{
		for (int i = 0; i < (int)_layers.size(); i++) _layers[i].get()->reset_deltas(momentum);
	}

	////////////////////////////////////////////////////////////
	void ResidualBlock::reset_errors()
	{
		for (int i = 0; i < (int)_layers.size(); i++) _layers[i].get()->reset_errors();
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::reset_outputs()
	{
		_outputs.fill(0);	
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type ResidualBlock::get_type() const
	{
		return Operation::ResidualBlock;
	}
	
	////////////////////////////////////////////////////////////
	void ResidualBlock::print()
	{
		printf("Type: ResidualBlock, Blocks: %d, Block_Outputs: %d, Input: (%dx%dx%d), Filter_Size: (%dx%d), Stride: %d, Padding: %d",
			_blocks_count, _size, _input_width, _input_height, _input_count, _filter_width, _filter_height, _stride, _padding);
	}

} /* namespace ai */
