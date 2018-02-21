////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Linear.hpp"
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
	std::shared_ptr<Operation> Linear::make(const int size, bool use_bias,
			const float gradient_clipping, float l1_regularization, float l2_regularization)
	{
		return std::shared_ptr<Operation>(new Linear(size, use_bias, gradient_clipping, l1_regularization, l2_regularization));
	}

	////////////////////////////////////////////////////////////
	Linear::Linear()
	{
		_size = 0;
		_use_bias = true;
		_l1_regularization = 0;
		_l2_regularization = 0;
		_fixed_parameters = false;
	}

	////////////////////////////////////////////////////////////
	Linear::Linear(const int size, bool use_bias, const float gradient_clipping, float l1_regularization, float l2_regularization)
	{
		_size = size;
		_use_bias = use_bias;
		ensure(l1_regularization >= 0 && l1_regularization < 1);
		ensure(l2_regularization >= 0 && l2_regularization < 1);
		_gradient_clipping = gradient_clipping;
		_l1_regularization = l1_regularization;
		_l2_regularization = l2_regularization;
		_fixed_parameters = false;
	}

	////////////////////////////////////////////////////////////
	Linear::Linear(ai::IOData& data)
	{
		ai::IOData* size = data.findNode("size");
		ensure(size != NULL);
		ai::IOData* input_size = data.findNode("input_size");
		ensure(input_size != NULL);
		ai::IOData* use_bias = data.findNode("use_bias");
		ensure(use_bias != NULL);
		size->get(_size);
		input_size->get(_input_size);
		use_bias->get(_use_bias);

		ai::IOData* l1 = data.findNode("l1_regularization");
		ai::IOData* l2 = data.findNode("l2_regularization");
		if (l1 != NULL) l1->get(_l1_regularization);
		else _l1_regularization = 0; //default value
		if (l2 != NULL) l1->get(_l2_regularization);
		else _l2_regularization = 0; //default value

		ai::IOData* grad_clippning = data.findNode("gradient_clipping");
		if (grad_clippning != NULL) grad_clippning->get(_gradient_clipping);

		_weights.load(data, "weights");
		_bias.load(data, "bias");

		_outputs.setshape(_size);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
		_deltas.setshape(_size * (_input_size + 1));
		_deltas.fill(0);
	}

	////////////////////////////////////////////////////////////
	void Linear::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("input_size", _input_size);
		data.pushNode("use_bias", _use_bias);
		data.pushNode("l1_regularization", _l1_regularization);
		data.pushNode("l2_regularization", _l2_regularization);
		data.pushNode("gradient_clipping", _gradient_clipping);
		_weights.save(data, "weights");
		_bias.save(data, "bias");
	}

	////////////////////////////////////////////////////////////
	void Linear::initialize(std::vector<Operation*> &inputs)
	{
		//Calculate input size
		int input_size = 0;
		for (int i = 0; i < (int)inputs.size(); i++)
			input_size += inputs[i]->_outputs.size();

		initialize(input_size);
	}

	////////////////////////////////////////////////////////////
	void Linear::initialize(int input_size)
	{
		_input_size = input_size;

		//Initialize variables and buffers
		_outputs.setshape(_size);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
		_deltas.setshape(_size * (_input_size + 1));
		_deltas.fill(0);

		//Initialize weights
		_weights.setshape(_size, _input_size);
		_weights.fill(0.0, sqrt(2.0f / (_input_size + _size)));
		_bias.setshape(_size);
		_bias.fill(0.0, sqrt(2.0f / (_input_size + _size)));
	}

	////////////////////////////////////////////////////////////
	void Linear::run(std::vector<Operation*> &inputs, const bool training) 
	{
		run(inputs[0]->_outputs, false);
	}

	////////////////////////////////////////////////////////////
	void Linear::backprop(std::vector<Operation*> &inputs) 
	{
		backprop(inputs[0]->_errors);	
	}

	////////////////////////////////////////////////////////////
	void Linear::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		accumulate_deltas(inputs[0]->_outputs);
	}

	////////////////////////////////////////////////////////////
	void Linear::update_parameters(const float learningrate)
	{
		if (_fixed_parameters == true) return;

#ifdef CUDA_BACKEND
		
		//Gradient clipping
		if (_gradient_clipping != 0)
			cuda::gradient_clipping(_deltas.pointer(), _deltas.size(), _gradient_clipping);

		//Update weights using gradients
		cuda::linear_update_parameters(_weights.pointer(), _bias.pointer(), _deltas.pointer(),
				learningrate, _input_size, _size);
		
		//Regularization penality
		if (_l1_regularization != 0) ai::weightreg::l1_regularization(_weights, _l1_regularization, learningrate);
		else if (_l2_regularization != 0) ai::weightreg::l2_regularization(_weights, _l2_regularization, learningrate);

#else
		
		//Gradient clipping
		if (_gradient_clipping != 0) {
			for (int i = 0; i < (int)_deltas.size(); i++) {
				if (_deltas[i] > _gradient_clipping) _deltas[i] = _gradient_clipping;
				else if(_deltas[i] < -_gradient_clipping) _deltas[i] = -_gradient_clipping;
			}
		}

		//Update weights using gradients
		int d = 0;
		for (int i = 0; i < _weights.width(); i++) {
			for (int k = 0; k <	_weights.height(); k++)
				_weights.at(k, i) += _deltas[d++] * learningrate;
			_bias[i] += _deltas[d++] * learningrate;
		}

		//Regularization penality
		if (_l1_regularization != 0) ai::weightreg::l1_regularization(_weights, _l1_regularization, learningrate);
		else if (_l2_regularization != 0) ai::weightreg::l2_regularization(_weights, _l2_regularization, learningrate);

#endif
	}

	////////////////////////////////////////////////////////////
	void Linear::reset_deltas(const double momentum)
	{
#ifdef CUDA_BACKEND

		if (_deltas.size() > 0) TensorCUDA_float_scale(_deltas, momentum);

#else

		for (int i = 0; i < _deltas.size(); i++)
			_deltas[i] *= momentum;

#endif
	}

#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Linear::run(const TensorCUDA_float& input, bool accumulate)
	{
		cuda::linear_foreward(_weights.pointer(), _bias.pointer(), input.pointer(),
				_outputs.pointer(), input.size(), _outputs.size(), accumulate, _use_bias);
	}

	////////////////////////////////////////////////////////////
	void Linear::backprop(TensorCUDA_float& out_errors)
	{
		cuda::linear_backward(_weights.pointer(), out_errors.pointer(), _errors.pointer(),
				out_errors.size(), _errors.size());
	}

	////////////////////////////////////////////////////////////
	void Linear::accumulate_deltas(const TensorCUDA_float& input)
	{
		cuda::linear_accumulate_deltas(_deltas.pointer(), input.pointer(), _errors.pointer(),
				input.size(), _errors.size(), _use_bias);
	}

#else

	////////////////////////////////////////////////////////////
	void Linear::run(const Tensor_float input, bool accumulate)
	{
		if (_use_bias)
		{
			if (!accumulate)
			{
				//Reset outputs with bias
				for (int i = 0; i < _outputs.size(); i++)
					_outputs[i] = _bias[i];
			}
			else
			{
				//Add bias to output
				for (int i = 0; i < _outputs.size(); i++)
					_outputs[i] += _bias[i];
			}
		}
		else
		{
			if (!accumulate)
			{
				for (int i = 0; i < _outputs.size(); i++)
					_outputs[i] = 0;
			}
		}

		//Compute all inputs
		int weight_index = 0;
		for (int i = 0; i < input.size(); i++) {
			if (input[i] == 0) continue;
			weight_index = i * _outputs.size();
			for (int k = 0; k < _outputs.size(); k++)
				_outputs[k] += _weights[weight_index++] * input[i];
		}
	}

	////////////////////////////////////////////////////////////
	void Linear::backprop(Tensor_float out_errors)
	{
		//Check we must have only one input
		if (out_errors.size() == 0) return;

		//Back-propagate errors
		for (int i = 0; i < _errors.size(); i++) {
			if (_errors[i] == 0) continue;
			for (int k = 0; k < _weights.height(); k++)
				out_errors[k] += _weights.at(k, i) * _errors[i];
		}
	}

	////////////////////////////////////////////////////////////
	void Linear::setFixedParameters(const bool fixedparameters)
	{
		_fixed_parameters = fixedparameters;
	}

	////////////////////////////////////////////////////////////
	void Linear::accumulate_deltas(const Tensor_float input)
	{
		int d = 0;
		for (int i = 0; i < _errors.size(); i++) {
			for (int k = 0; k <	input.size(); k++)
				_deltas[d++] += input[k] * _errors[i];
			if (_use_bias) _deltas[d++] += _errors[i];
			else d++;
		}
	}

#endif

	////////////////////////////////////////////////////////////
	void Linear::reset_outputs()
	{
		_outputs.fill(0);	
	}

#ifndef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Linear::gradient_check()
	{
		//params
		const int size = 100;
		const int input_size = 100;
		const float epsilon = 10e-4;

		//test node
		Linear node(size);
		node.initialize(input_size);

		//random input
		Tensor_float input(size);
		input.fill(0.5, 0.5);

		//random target
		Tensor_float target(size);
		target.fill(0.5, 0.5);

		//Cost function
		Cost costfun(Cost::SquaredError);

		//computed numerical gradients
		Tensor_float numgrad(node._weights.width(), node._weights.height());

		//For each parameter
		for (int i = 0; i < node._weights.size(); i++) {
			float init_param = node._weights[i];
			node._weights[i] = init_param + epsilon;
			node.run(input, false);
			float lossPlus = costfun.getError(node._outputs, target);

			node._weights[i] = init_param - epsilon;
			node.run(input, false);
			float lossMinus = costfun.getError(node._outputs, target);

			numgrad[i] = (lossPlus - lossMinus) / (2.f * epsilon);

			node._weights[i] = init_param;
		}

		//compute gradients with backprop code
		node.reset_deltas(0);
		node.run(input, false);
		costfun.getDelta(node._outputs, target, node._errors);
		node.accumulate_deltas(input);

		int d = 0;
		float max = 0;
		double medium_delta_size = 0;
		Tensor_float distances(numgrad.width(), numgrad.height());
		for (int i = 0; i < node._weights.width(); i++) {
			for (int k = 0; k <	node._weights.height(); k++) {
				medium_delta_size += fabs(node._deltas[d]);
				distances.at(k, i) = fabs(numgrad.at(k, i) + node._deltas[d]);
				if (distances.at(k, i) > max)
					max = distances.at(k, i); 
				d++;
			}
			d++; //bias
		}
		medium_delta_size /= node._deltas.size();

		const float tollerance = medium_delta_size * 0.05;
		if (max > tollerance) printf("Gradient looks bad, differs by %f with medium_delta_size %f\n", max, medium_delta_size);
		else printf("Gradient looks good, differs max by %f with medium_delta_size %f\n", max, medium_delta_size);

		//printf("%s\n", distances.tostring().c_str());
	}

#endif

	////////////////////////////////////////////////////////////
	const Operation::Type Linear::get_type() const
	{
		return Operation::Linear;
	}

	////////////////////////////////////////////////////////////
	void Linear::print()
	{
		printf("Type: Linear, Size: %d, Input_Size: %d, Weights: %d, Bias: %d, l1: %f, l2: %f",
				_size, _input_size, _size * (_input_size + 1), (int)_use_bias, _l1_regularization, _l2_regularization);
	}

} /* namespace ai */
