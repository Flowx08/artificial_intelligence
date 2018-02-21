////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "OptimizerDFA.hpp"
#include "NeuralNetwork.hpp"
#include "../util/ensure.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	OptimizerDFA::OptimizerDFA()
	{
		_current_sample = 0;
		_learningrate = 0.1;
		_momentum = 0;
		_batch_size = 1;
		_costfunction = Cost::SquaredError;
	}
	
	////////////////////////////////////////////////////////////
	OptimizerDFA::OptimizerDFA(const int batch_size, const double learningrate, const double momentum,
					const Cost::CostType cost_function)
	{
		_current_sample = 0;
		_learningrate = learningrate;
		_momentum = momentum;
		_batch_size = batch_size;
		_costfunction = cost_function;
		_feedback_weights.setshape(200 * 10);
		_feedback_weights.fill(0, 0.5);
		for (int i = 0; i < _feedback_weights.size(); i++) {
			//if (rand() % 1000 < 700) _feedback_weights[i] = 0;
			//if (_feedback_weights[i] < 0) _feedback_weights[i] = -1;
			//else _feedback_weights[i] = 1; 
		}
		_feedback_errors.setshape(200);
		_feedback_errors.fill(0);
	}
	
	#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void OptimizerDFA::fit(NeuralNetwork& net, TensorCUDA_float &inputs, TensorCUDA_float &targets)
	{
		/*
		ensure(targets.depth() == 1 && targets.height() == 1);
		
		//Feedforeward
		net.run(inputs, true);
		
		//Shortcut
		std::vector< NetworkNode > &nodes = net.getNodes();

		//Reset errors
		for (int i = 0; i < (int)nodes.size(); i++)
			nodes[i].reset_errors();
		
		//Calculate cost on GPU
		_costfunction.getDeltaCUDA(nodes.back().getOperation()->_outputs, targets, nodes.back().getOperation()->_errors);
		_error = _costfunction.getErrorCUDA(nodes.back().getOperation()->_outputs, targets);

		//Accumulate deltas
		for (int i = (int)nodes.size()-1; i >= 0; i--) {
			nodes[i].backprop();
			nodes[i].accumulate_deltas();
		}
		
		//Update weights if we reach the batch size
		if (++_current_sample >= _batch_size) {
			
			//Update weights and reset deltas
			for (int i = 0; i < (int)nodes.size(); i++) {
				nodes[i].update_parameters(_learningrate);
				nodes[i].reset_deltas(_momentum);
			}

			_current_sample = 0;
		}
		*/
	}
	
	#else

	////////////////////////////////////////////////////////////
	void OptimizerDFA::fit(NeuralNetwork& net, Tensor_float &inputs, Tensor_float &targets)
	{
		ensure(targets.depth() == 1 && targets.height() == 1);
		
		//Feedforeward
		net.run(inputs, true);
			
		//Shortcuts
		std::vector< NetworkNode >& nodes = net.getNodes();

		//Reset errors
		for (int i = 0; i < (int)nodes.size(); i++)
			nodes[i].reset_errors();
		
		//Calculate cost on host
		_costfunction.getDelta(nodes.back().getOperation()->_outputs, targets, nodes.back().getOperation()->_errors);
		_error = _costfunction.getError(nodes.back().getOperation()->_outputs, targets);
		
		const Tensor_float& errs = nodes.back().getOperation()->_errors;
				
		const float scale = 0.09;
		int l = 0;
		for (int k = 0; k < _feedback_errors.size(); k++) {
			_feedback_errors[k] = 0;
			for (int j = 0; j < errs.size(); j++) {
				_feedback_errors[k] += errs[j] * _feedback_weights[l++] * scale;
			}
		}
		
		nodes.back().backprop();

		//Accumulate deltas
		l = 0;
		for (int i = (int)nodes.size()-1; i >= 0 ; i--) {
			Operation* op = nodes[i].getOperation();
			if (op->get_type() == Operation::Softmax ||
				op->get_type() == Operation::Sigmoid ||
				op->get_type() == Operation::Relu ||
				op->get_type() == Operation::Tanh ||
				op->get_type() == Operation::Averagepooling ||
				op->get_type() == Operation::Maxpooling)
			{

				Tensor_float& node_errors = nodes[i].getOperation()->_errors;
				for (int k = 0; k < node_errors.size(); k++) {
					node_errors[k] += _feedback_errors[l++];
					if (l >= _feedback_errors.size()) l = 0;
				}

				nodes[i].backprop();
			}
			else if (op->get_type() == Operation::Normalization)
			{
				nodes[i].backprop();
			}

			//Calculate deltas
			nodes[i].accumulate_deltas();
		}
		
		/*
		nodes.back().backprop();

		//Accumulate deltas
		int l = 1;
		for (int i = (int)nodes.size()-1; i >= 0 ; i--) {
			Operation* op = nodes[i].getOperation();
			if (op->get_type() == Operation::Softmax ||
				op->get_type() == Operation::Sigmoid ||
				op->get_type() == Operation::Relu ||
				op->get_type() == Operation::Tanh ||
				op->get_type() == Operation::Averagepooling ||
				op->get_type() == Operation::Maxpooling)
			{

				for (int j = 0; j < errs.size(); j++) {
					Tensor_float& node_errors = nodes[i].getOperation()->_errors;
					float scale = 0.01;
					for (int k = 0; k < node_errors.size(); k++) {
						node_errors[k] += errs[j] * _feedback_weights[l++] * scale;
						if (l >= _feedback_weights.size()) l = 0;
					}
				}

				nodes[i].backprop();
			}
			else if (op->get_type() == Operation::Normalization)
			{
				nodes[i].backprop();
			}

			//Calculate deltas
			nodes[i].accumulate_deltas();
		}
		*/
		
		//Update weights if we reach the batch size
		if (++_current_sample >= _batch_size) {
			
			//Update weights and reset deltas
			for (int i = 0; i < (int)nodes.size(); i++) {
				nodes[i].update_parameters(_learningrate);
				nodes[i].reset_deltas(_momentum);
			}

			_current_sample = 0;
		}
	}

	#endif

} /* namespace ai */
