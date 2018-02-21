////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "OptimizerSDG.hpp"
#include "NeuralNetwork.hpp"
#include "../util/ensure.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	OptimizerSDG::OptimizerSDG()
	{
		_current_sample = 0;
		_learningrate = 0.1;
		_momentum = 0;
		_batch_size = 1;
		_costfunction = Cost(Cost::SquaredError);
	}
	
	////////////////////////////////////////////////////////////
	OptimizerSDG::OptimizerSDG(const int batch_size, const double learningrate, const double momentum,
					const Cost::CostType cost_function)
	{
		_current_sample = 0;
		_learningrate = learningrate;
		_momentum = momentum;
		_batch_size = batch_size;
		_costfunction = Cost(cost_function);
	}
	
	#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void OptimizerSDG::fit(NeuralNetwork& net, TensorCUDA_float &inputs, TensorCUDA_float &targets)
	{
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
	}
	
	#else

	////////////////////////////////////////////////////////////
	void OptimizerSDG::fit(NeuralNetwork& net, Tensor_float &inputs, Tensor_float &targets)
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

		//Accumulate deltas
		for (int i = (int)nodes.size()-1; i >= 0; i--) {
			
			//Backpropagate errors
			nodes[i].backprop();

			//Calculate deltas
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
		
		/*
		bool nan_f = false;
		for (int i = 0; i < nodes.size(); i++) {
			if (nodes[i].getOperation()->_outputs.isNaN()) {
				printf("Node %d Output NaN\n", i);
				nan_f = true;
			}
			float maxval;
			nodes[i].getOperation()->_outputs.max(&maxval, NULL);
			if (maxval > 250.f) {
				printf("Node %d Max %f\n", i, maxval);
				nan_f = true;
			}
		}
		for (int i = (int)nodes.size()-1; i >= 0; i--) {
			if (nodes[i].getOperation()->_errors.isNaN()) {
				printf("Node %d errors NaN\n", i);
				nan_f = true;
			}
			float maxval;
			nodes[i].getOperation()->_errors.max(&maxval, NULL);
			float minval;
			nodes[i].getOperation()->_errors.min(&minval, NULL);
			maxval = maxval > -minval ? maxval : -minval;
			if (maxval > 10.f) {
				printf("Node %d MaxErr %f\n", i, maxval);
				printf("%s\n", nodes.back().getOperation()->_outputs.tostring().c_str());
				printf("%s\n", targets.tostring().c_str());
				printf("%s\n", nodes.back().getOperation()->_errors.tostring().c_str());
				nan_f = true;
			}
		}
		if (nan_f == true) {
			for (int i = 0; i < nodes.front().getOperation()->_outputs.size(); i++) {
				printf("%f\n", nodes.front().getOperation()->_outputs[i]);	
			}

			//printf("%s\n", nodes.front().getOperation()->_outputs.tostring().c_str())
		}
		ensure(nan_f == false);
		*/
	}

	#endif

} /* namespace ai */
