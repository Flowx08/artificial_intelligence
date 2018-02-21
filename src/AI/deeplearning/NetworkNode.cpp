////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "NetworkNode.hpp"
#include "NeuralNetwork.hpp"
#include "../util/ensure.hpp"
#include <stdlib.h>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	NetworkNode::NetworkNode(std::string node_name, std::vector<std::string> input_names,
		NeuralNetwork* network, std::shared_ptr<Operation> operation)
	{
		ensure_print(node_name != "", "Error, empty node name not allowed\n");

		_name = node_name;
		_input_names = input_names;
		_network = network;
		_operation = operation;
		
		checkInvalidInputNames();

		initInputsIndiciesVector();
		initInputsOperationsVector();
		
		ensure(_input_indicies.size() == _input_names.size());

		_operation->initialize(_input_operations);
	}
	
	////////////////////////////////////////////////////////////
	NetworkNode::NetworkNode(ai::IOData& data, NeuralNetwork* network)
	{
		_network = network;
		load(data);
		initInputsOperationsVector();
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::checkInvalidInputNames()
	{
		for (int k  = 0; k < (int)_input_names.size(); k++) {
			bool found = false;
			for (int i = 0; i < (int)_network->getNodes().size(); i++) {
				if (_network->getNodes()[i].getName() == _input_names[k]) {
					found = true;
					break;
				}
			}
			
			//Invalid name
			if (found == false) {
				printf("Error while creating node '%s', invalid input name %s\n", _name.c_str(), _input_names[k].c_str());
				exit(-1);
			}
		}
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::initInputsIndiciesVector()
	{
		_input_indicies = std::vector<int>();
		int index = 0;
		for (int i = 0; i < (int)_network->getNodes().size(); i++)
			for (int k  = 0; k < (int)_input_names.size(); k++)
				if (_network->getNodes()[i].getName() == _input_names[k]) {
					_input_indicies.push_back(i);
					index++;
				}
	}

	////////////////////////////////////////////////////////////
	void NetworkNode::initInputsOperationsVector()
	{
		_input_operations = std::vector<Operation*>(_input_indicies.size());
		for (int i = 0; i < (int)_input_operations.size(); i++)
			_input_operations[i] = _network->getNodes()[_input_indicies[i]].getOperation();
	}

	////////////////////////////////////////////////////////////
	void NetworkNode::load(ai::IOData& data)
	{
		ai::IOData* node_name = data.findNode("node_name");
		ensure(node_name != NULL);
		ai::IOData* inputs = data.findNode("inputs");
		ensure(inputs != NULL);
		ai::IOData* operation = data.findNode("operation");
		ensure(operation != NULL);
		node_name->get(_name);
		_input_names.clear();
		_input_indicies.clear();
		for (int i = 0; i < (int)inputs->getSubNodes().size(); i++) {		
			_input_names.push_back(inputs->getSubNodes()[i].getName());
			_input_indicies.push_back(0);
			inputs->getSubNodes()[i].get(_input_indicies.back());
		}
		_operation = Operation::loadFromFile(*operation);
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::save(ai::IOData& data)
	{
		data.pushNode("node_name", _name);
		data.pushNode("inputs");
		ai::IOData& inputs = *data.findNode("inputs");
		for (int i = 0; i < (int)_input_names.size(); i++)
			inputs.pushNode(_input_names[i], _input_indicies[i]);
		data.pushNode("operation");
		ai::IOData& operation_data = *data.findNode("operation");
		Operation::saveToFile(_operation, operation_data);
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::run(bool training)
	{
		_operation->run(_input_operations, training);
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::backprop()
	{
		_operation->backprop(_input_operations);
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::accumulate_deltas()
	{
		_operation->accumulate_deltas(_input_operations);
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::update_parameters(const float learningrate)
	{
		_operation->update_parameters(learningrate);
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::reset_errors()
	{
		_operation->reset_errors();
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::reset_deltas(const float momentum)
	{
		_operation->reset_deltas(momentum);
	}

	////////////////////////////////////////////////////////////
	const std::string NetworkNode::getName() const
	{
		return _name;
	}

	////////////////////////////////////////////////////////////
	const std::vector<std::string> NetworkNode::getInputsNames()
	{
		return _input_names;
	}
	
	////////////////////////////////////////////////////////////
	const std::vector<int> NetworkNode::getInputsIndicies()
	{
		return _input_indicies;
	}
	
	////////////////////////////////////////////////////////////
	Operation* NetworkNode::getOperation()
	{
		return _operation.get();
	}
			
	#ifdef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	Tensor_float NetworkNode::getOperationOutput()
	{
		Tensor_float tmp(_operation->_outputs.width(), _operation->_outputs.height(), _operation->_outputs.depth());
		_operation->_outputs.copyToHost(&tmp[0], tmp.size());
		return tmp;
	}
	
	////////////////////////////////////////////////////////////
	const TensorCUDA_float& NetworkNode::getOperationOutputDevice()
	{
		return _operation->_outputs;
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::setOperationOutput(TensorCUDA_float& output)
	{
		_operation->_outputs.point(output);	
	}

	#else
	
	////////////////////////////////////////////////////////////
	Tensor_float& NetworkNode::getOperationOutput()
	{
		return _operation->_outputs;
	}
	
	////////////////////////////////////////////////////////////
	void NetworkNode::setOperationOutput(Tensor_float& output)
	{
		_operation->_outputs.point(output);	
	}

	#endif
	
	////////////////////////////////////////////////////////////
	void NetworkNode::print()
	{
		printf("%s\t ", _name.c_str());
		_operation->print();
		printf("\n\t Inputs: [");
		if (_input_names.size() == 0) {
			printf("]\n");
			return;
		}
		for (int i = 0; i < (int)_input_names.size(); i++) {
			if (i != (int)_input_names.size() - 1) printf(" %s,", _input_names[i].c_str());
			else printf(" %s ]\n", _input_names[i].c_str());
		}
	}
	
} /* namespace ai */
