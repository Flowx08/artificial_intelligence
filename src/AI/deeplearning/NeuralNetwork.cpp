////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include "NeuralNetwork.hpp"
#include "../util/IOData.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NEMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	NeuralNetwork::NeuralNetwork() {}

	////////////////////////////////////////////////////////////
	NeuralNetwork::NeuralNetwork(std::string filepath)
	{
		load(filepath);
	}

	////////////////////////////////////////////////////////////
	NeuralNetwork::~NeuralNetwork() { clear(); }

	////////////////////////////////////////////////////////////
	void NeuralNetwork::save(std::string filepath)
	{
		ai::IOData data("network");
		data.pushNode("node_count", (int)_nodes.size());
		data.pushNode("nodes");
		ai::IOData* node_data = data.findNode("nodes");
		ensure(node_data != NULL);
		for (int i = 0; i < (int)_nodes.size(); i++) {
			node_data->pushNode("node_" + std::to_string(i));
			ai::IOData* node = node_data->findNode("node_" + std::to_string(i));
			_nodes[i].save(*node);
		}
		if (!data.writeToFile(filepath)) {
			printf("Error in NeuralNework.cpp: can't save network to filepath %s\n", filepath.c_str());
		}
	}

	////////////////////////////////////////////////////////////
	void NeuralNetwork::load(std::string filepath)
	{
		ai::IOData data("");
		if (!data.loadFromFile(filepath)) {
			printf("Error in NeuralNework.cpp: can't load network from filepath %s\n", filepath.c_str());
			return;
		}

		clear();
		
		ai::IOData* node_count = data.findNode("node_count");
		ensure(node_count != NULL);
		ai::IOData* nodes = data.findNode("nodes");
		ensure(nodes != NULL);
		int node_count_val;
		node_count->get(node_count_val);
		for (int i = 0; i < (int)node_count_val; i++) {
			ai::IOData* node = nodes->findNode("node_" + std::to_string(i));
			_nodes.push_back(NetworkNode(*node, this));
		}
	}

	////////////////////////////////////////////////////////////
	void NeuralNetwork::push(std::string node_name, std::string inputs_names, std::shared_ptr<Operation> operation)
	{
		std::vector<std::string> splitted_inputs_names = splitString(inputs_names, ',');
		_nodes.push_back(NetworkNode(node_name, splitted_inputs_names, this, operation));
	}
	
	////////////////////////////////////////////////////////////
	NetworkNode* NeuralNetwork::get_byname(std::string node_name)
	{
		for (int i = 0; i < (int)_nodes.size(); i++)
			if (_nodes[i].getName() == node_name)
				return &_nodes[i];
		return NULL;
	}
	
	////////////////////////////////////////////////////////////
	std::vector<std::string> NeuralNetwork::splitString(std::string s, char delimiter)
	{
		std::vector<std::string> substrings;

		if (s == "")
		{
			//Nothing to do
		}
		else
		{
			std::string block;
			std::stringstream stream(s);
			while(std::getline(stream, block, delimiter))
				substrings.push_back(block);
		}
		
		return substrings;
	}
	
	#ifdef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	void NeuralNetwork::run(TensorCUDA_float input, const bool training)
	{
		_nodes.front().setOperationOutput(input);

        //Feedforeward
        for (int i = 0; i < (int)_nodes.size(); i++)
			_nodes[i].run(training);
	}
	
	////////////////////////////////////////////////////////////
	float NeuralNetwork::optimize(TensorCUDA_float input, TensorCUDA_float target, Optimizer* opt)
	{
		opt->fit(*this, input, target);
		return opt->getError();
	}
	
	////////////////////////////////////////////////////////////
	bool NeuralNetwork::test(TensorCUDA_float input, TensorCUDA_float target)
	{
		//Feedforeward
		run(input, false);

		Tensor_float outputs_host = _nodes.back().getOperationOutput();
		Tensor_float target_host(target.size());
		target.copyToHost(target_host.pointer(), target_host.size());
		
		int target_id;
		target_host.max(NULL, &target_id);

		int output_id;
		outputs_host.max(NULL, &output_id);
		
		if (target_id == output_id) return true;
		else return false;
	}
	
	////////////////////////////////////////////////////////////
	TensorCUDA_float& NeuralNetwork::get_output(std::string node_name)
	{
		for (int i = 0; i < (int)_nodes.size(); i++)
			if (_nodes[i].getName() == node_name)
				return _nodes[i].getOperationOutputDevice();
		printf("Node with name  %s not found\n", node_name.c_str());
		exit(-1);
	}

	#else
	
	////////////////////////////////////////////////////////////
	void NeuralNetwork::run(Tensor_float input, const bool training)
	{
		_nodes.front().setOperationOutput(input);

		//Feedforeward
		for (int i = 0; i < (int)_nodes.size(); i++)
			_nodes[i].run(training);
	}
	
	////////////////////////////////////////////////////////////
	float NeuralNetwork::optimize(Tensor_float input, Tensor_float target, Optimizer* opt)
	{
		opt->fit(*this, input, target);
		return opt->getError();
	}
	
	////////////////////////////////////////////////////////////
	Tensor_float& NeuralNetwork::get_output(std::string node_name)
	{
		for (int i = 0; i < (int)_nodes.size(); i++)
			if (_nodes[i].getName() == node_name)
				return _nodes[i].getOperationOutput();
		printf("Node with name  %s not found\n", node_name.c_str());
		exit(-1);
	}
	
	#endif
	
	////////////////////////////////////////////////////////////
	void NeuralNetwork::clear()
	{
		_nodes.clear();
	}

	////////////////////////////////////////////////////////////
	std::vector<NetworkNode>& NeuralNetwork::getNodes()
	{
		return _nodes;
	}

	////////////////////////////////////////////////////////////
	void NeuralNetwork::printstack()
	{
		for (int i = 0; i < (int)_nodes.size(); i++)
			_nodes[i].print();
	}

} /* namespace ai */
