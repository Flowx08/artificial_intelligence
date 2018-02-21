////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Autoencoder.hpp"
#include "Relu.hpp"
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
	std::shared_ptr<Operation> Autoencoder::make(const int size, const float noise)
	{
		return std::shared_ptr<Operation>(new Autoencoder(size, noise));
	}

	////////////////////////////////////////////////////////////
	Autoencoder::Autoencoder()
	{
		_size = 0;
		_fixed_parameters = false;
		_learningrate = 0.0005;
		_noise = 0;
	}

	////////////////////////////////////////////////////////////
	Autoencoder::Autoencoder(const int size, const float noise)
	{
		_size = size;
		_fixed_parameters = false;
		_learningrate = 0.0005;
		_noise = noise;
	}

	////////////////////////////////////////////////////////////
	Autoencoder::Autoencoder(ai::IOData& data)
	{
		ai::IOData* size = data.findNode("size");
		ensure(size != NULL);
		ai::IOData* input_size = data.findNode("input_size");
		ensure(input_size != NULL);
		size->get(_size);
		input_size->get(_input_size);
		
		_weights.load(data, "weights");
		_bias.load(data, "bias");
		
		//Initialize variables and buffers
		_w_deltas.setshape(_size, _input_size);
		_w_deltas.fill(0);
		_b_deltas.setshape(_size);
		_b_deltas.fill(0);
		_outputs.setshape(_size);
		_outputs.fill(0);
		_activations.setshape(_size);
		_activations.fill(0);
		_prediction.setshape(_input_size);
		_prediction.fill(0);
		_prediction_error.setshape(_input_size);
		_prediction_error.fill(0);
		_noise_mask.setshape(_input_size);
		_noise_mask.fill(0);
		_hidden_errors.setshape(_size);
		_hidden_errors.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("input_size", _input_size);
		_weights.save(data, "weights");
		_bias.save(data, "bias");
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::initialize(std::vector<Operation*> &inputs)
	{
		//Calculate input size
		int input_size = 0;
		for (int i = 0; i < (int)inputs.size(); i++)
			input_size += inputs[i]->_outputs.size();
		initialize(input_size);
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::initialize(int input_size)
	{
		_input_size = input_size;
		_weights.setshape(_size, _input_size);
		_weights.fill(0.0, sqrt(2.f / (_size + _input_size)));
		_bias.setshape(_size);
		_bias.fill(0.0, sqrt(2.f / (_size + _input_size)));

		//Initialize variables and buffers
		_w_deltas.setshape(_size, _input_size);
		_w_deltas.fill(0);
		_b_deltas.setshape(_size);
		_b_deltas.fill(0);
		_outputs.setshape(_size);
		_outputs.fill(0);
		_activations.setshape(_size);
		_activations.fill(0);
		_prediction.setshape(_input_size);
		_prediction.fill(0);
		_prediction_error.setshape(_input_size);
		_prediction_error.fill(0);
		_noise_mask.setshape(_input_size);
		_noise_mask.fill(0);
		_hidden_errors.setshape(_size);
		_hidden_errors.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::run(std::vector<Operation*> &inputs, const bool training) 
	{
		run(inputs[0]->_outputs, false, training);
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::backprop(std::vector<Operation*> &inputs) 
	{
		backprop(inputs[0]->_errors);	
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		accumulate_deltas(inputs[0]->_outputs);
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::update_parameters(const float learningrate)
	{
		if (_fixed_parameters) return;

#ifdef CUDA_BACKEND

#else

		//Compute deltas
		int out_index = 0;
		for (int i = 0; i < (int)_size; i++) {
			out_index = i;
			for (int k = 0; k < _input_size; k++)
				_weights.at(k, out_index) += _w_deltas.at(k,  out_index) * _learningrate;
			_bias[out_index] += _b_deltas[out_index] * _learningrate;
		}

#endif
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::reset_deltas(const double momentum)
	{
#ifdef CUDA_BACKEND


#else
	
		for (int i = 0; i < _w_deltas.size(); i++)
			_w_deltas[i] *= momentum;	
		for (int i = 0; i < _b_deltas.size(); i++)
			_b_deltas[i] *= momentum;	

#endif
	}

#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Autoencoder::run(const TensorCUDA_float& input, bool accumulate, bool training)
	{
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::backprop(TensorCUDA_float& out_errors)
	{
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::accumulate_deltas(const TensorCUDA_float& input)
	{
	}

#else

	////////////////////////////////////////////////////////////
	void Autoencoder::run(const Tensor_float input, bool accumulate, bool training)
	{
	

		//Apply bias
		for (int i = 0; i < _outputs.size(); i++)
			_outputs[i] = _bias[i];

		if (training && _noise)
		{
			_noise_mask.fill(0, _noise);

			//Linear output calculation
			int weight_index = 0;
			for (int i = 0; i < input.size(); i++) {
				if (input[i] == 0) continue;
				weight_index = i * _outputs.size();
				for (int k = 0; k < _outputs.size(); k++)
					_outputs[k] += _weights[weight_index++] * (input[i] + _noise_mask[i]);
			}
		}
		else
		{
			//Linear output calculation
			int weight_index = 0;
			for (int i = 0; i < input.size(); i++) {
				if (input[i] == 0) continue;
				weight_index = i * _outputs.size();
				for (int k = 0; k < _outputs.size(); k++)
					_outputs[k] += _weights[weight_index++] * input[i];
			}
		}

		//Apply activation function
		ai::Relu::foreward(_outputs, _outputs);
		
		//Store activations
		int p = 0;
		for (int i = 0; i < _size; i++)
			if (_outputs[i] > 0) _activations[p++] = i;
		_activations[p] = -1;
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::backprop(Tensor_float out_errors)
	{
		
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::accumulate_deltas(const Tensor_float input)
	{
		if (_fixed_parameters) return;

		_prediction.fill(0);
		int out_index = 0;
		for (int i = 0; i < (int)_activations.size(); i++) {
			if (_activations[i] == -1) break;
			out_index = _activations[i];
			for (int k = 0; k < _input_size; k++)
				_prediction[k] += _weights.at(k, out_index) * _outputs[out_index];
		}

		//Compute prediction error
		_error = 0;
		for (int i = 0; i < _input_size; i++) {
			_prediction_error[i] = input[i] - _prediction[i];
			_error += fabs(_prediction_error[i]);
		}
	
		//Calculate hidden errors
		out_index = 0;
		for (int i = 0; i < (int)_activations.size(); i++) {
			if (_activations[i] == -1) break;
			out_index = _activations[i];
			for (int k = 0; k < _input_size; k++) {
				_hidden_errors[out_index] += _weights.at(k, out_index) * _prediction_error[k]; //bias
			}
		}
		ai::Relu::backward(_hidden_errors, _outputs, _hidden_errors);

		//Compute deltas
		out_index = 0;
		for (int i = 0; i < (int)_activations.size(); i++) {
			if (_activations[i] == -1) break;
			out_index = _activations[i];
			for (int k = 0; k < _input_size; k++) {
				_w_deltas.at(k,  out_index) += _outputs[out_index] * _prediction_error[k]; //weights decoder
				_w_deltas.at(k,  out_index) += input[k] * _hidden_errors[out_index]; //weights encoder
			}
			_b_deltas[out_index] += _hidden_errors[out_index];
		}
		
		//Reset errors
		_hidden_errors.fill(0);
	}

#endif
	
	////////////////////////////////////////////////////////////
	const float Autoencoder::getPredictionError()
	{
		return _error;
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::reset_outputs()
	{
		_outputs.fill(0);	
	}
	
	////////////////////////////////////////////////////////////
	void Autoencoder::setFixedParameters(const bool fixedparameters)
	{
		_fixed_parameters = fixedparameters;
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Autoencoder::get_type() const
	{
		return Operation::Autoencoder;
	}

	////////////////////////////////////////////////////////////
	void Autoencoder::print()
	{
		printf("Type: Autoencoder, Size: %d, Input_Size: %d, Weights: %d", _size, _input_size, _size * (_input_size + 1) * 2);
	}

} /* namespace ai */
