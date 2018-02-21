////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "linear_regression.hpp"
#include "../util/ensure.hpp"
#include <algorithm>
#include <math.h>
#include <fstream>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	linear_regression::linear_regression(const unsigned int input_size, const unsigned int output_size)
	{
		_input_size = input_size;
		_output_size = output_size;
		_weights.setshape(output_size, input_size);
		_weights.fill(0.0, 1.f / sqrt(input_size));
		_bias.setshape(output_size);
		_bias.fill(0.0, 1.f / sqrt(input_size));
		_outputs.setshape(output_size);
		_outputs.fill(0);
		_errors.setshape(output_size);
		_errors.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	linear_regression::linear_regression(const std::string filepath)
	{
		std::ifstream file(filepath, std::ios::binary);
		ensure(file && "can't open the file for reading!");
		file.read(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.read(reinterpret_cast<char*>(&_output_size), sizeof(_output_size));
		_weights.setshape(_output_size, _input_size);
		_bias.setshape(_output_size);
		_outputs.setshape(_output_size);
		_outputs.fill(0);
		_errors.setshape(_output_size);
		_errors.fill(0);
		file.read(reinterpret_cast<char*>(&_weights[0]), sizeof(float) * _input_size * _output_size);
		file.read(reinterpret_cast<char*>(&_bias[0]), sizeof(float) * _output_size);
	}
	
	////////////////////////////////////////////////////////////
	void linear_regression::save(const std::string filepath)
	{
		std::ofstream file(filepath, std::ios::binary);
		ensure(file && "can't open the file for writing!");
		file.write(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.write(reinterpret_cast<char*>(&_output_size), sizeof(_output_size));
		file.write(reinterpret_cast<char*>(&_weights[0]), sizeof(float) * _input_size * _output_size);
		file.write(reinterpret_cast<char*>(&_bias[0]), sizeof(float) * _output_size);
	}

	////////////////////////////////////////////////////////////
	const Tensor_float& linear_regression::predict(const Tensor_float input)
	{
		ensure(input.width() == (int)_input_size);
		ensure(input.height() == 1 && input.depth() == 1);
		
		for (int k = 0; k < (int)_output_size; k++)
			_outputs[k] = _bias[k];	
		
		for (unsigned int i = 0; i < _input_size; i++) {
			if (input[i] == 0) continue;
			for (unsigned int k = 0; k < _output_size; k++) {
				_outputs[k] += _weights.at(i, k) * input[i];
			}
		}

		return _outputs;
	}
	
	////////////////////////////////////////////////////////////
	void linear_regression::fit(Tensor_float inputs, const Tensor_float targets, const float starting_learningrate,
		const unsigned int epochs, const bool verbose)
	{
		ensure(inputs.width() == (int)_input_size);
		ensure(inputs.height() > 1 && inputs.depth() == 1);
		ensure(targets.width() == (int)_output_size);
		ensure(targets.height() == inputs.height());


		float learning_rate = starting_learningrate;
		double medium_error = 0;
		for (unsigned int e = 0; e < epochs; e++) {	
			
			medium_error = 0;
		
			std::vector<int> shuffle_idx(inputs.height());
			for (int i = 0; i < (int)shuffle_idx.size(); i++) shuffle_idx[i] = i;
			std::random_shuffle(shuffle_idx.begin(), shuffle_idx.end());

			for (int i = 0; i < (int)inputs.height(); i++) {
				
				//compute learningrate
				learning_rate = starting_learningrate * (1.f - (double)(e * inputs.height() + i) / (double)(epochs * inputs.height()));
				
				int indx = shuffle_idx[i];
				
				//calculate output
				predict(inputs.ptr(0, indx));

				//calculate prediction errors
				for (unsigned int j = 0; j < _output_size; j++) {
					_errors[j] = targets.at(indx, j) - _outputs[j];
					medium_error += fabs(_errors[j]);
				}

				//update weights
				for (unsigned int j = 0; j < _output_size; j++) {
					_bias[j] += _errors[j] * learning_rate;
					for (unsigned int k = 0; k < _input_size; k++)
						_weights.at(k, j) += _errors[j] * inputs.at(indx, k) * learning_rate; 
				}
			}
			if (verbose) printf("Epoch: %d MediumError: %f\n", e, medium_error / (double)inputs.height());
			learning_rate *= 0.7;
		}
	}

	////////////////////////////////////////////////////////////
	const Tensor_float& linear_regression::get_output()
	{
		return _outputs;
	}

} /* namespace ai */
