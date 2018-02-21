////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "logistic_regression.hpp"
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
	logistic_regression::logistic_regression()
	{
		_input_size = 0;
		_output_size = 0;
	}

	////////////////////////////////////////////////////////////
	logistic_regression::logistic_regression(const unsigned int input_size, const unsigned int output_size)
	{
		_input_size = input_size;
		_output_size = output_size;
		_weights.setshape(output_size, input_size);
		_weights.fill(0.0, 2.f / sqrt(input_size + output_size));
		_bias.setshape(output_size);
		_bias.fill(0.0, 2.f / sqrt(input_size + output_size));
		_outputs.setshape(output_size);
		_outputs.fill(0);
		_errors.setshape(output_size);
		_errors.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	logistic_regression::logistic_regression(const std::string filepath)
	{
		std::ifstream file(filepath, std::ios::binary);
		ensure(file && "can't open the file for reading!");
		load(file);
	}
	
	////////////////////////////////////////////////////////////
	void logistic_regression::save(const std::string filepath)
	{
		std::ofstream file(filepath, std::ios::binary);
		ensure(file && "can't open the file for writing!");
		save(file);
	}
	
	////////////////////////////////////////////////////////////
	void logistic_regression::load(std::ifstream& filestream)
	{
		filestream.read(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		filestream.read(reinterpret_cast<char*>(&_output_size), sizeof(_output_size));
		_weights.setshape(_output_size, _input_size);
		_bias.setshape(_output_size);
		_outputs.setshape(_output_size);
		_outputs.fill(0);
		_errors.setshape(_output_size);
		_errors.fill(0);
		filestream.read(reinterpret_cast<char*>(&_weights[0]), sizeof(float) * _input_size * _output_size);
		filestream.read(reinterpret_cast<char*>(&_bias[0]), sizeof(float) * _output_size);
	}

	////////////////////////////////////////////////////////////
	void logistic_regression::save(std::ofstream& filestream)
	{
		filestream.write(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		filestream.write(reinterpret_cast<char*>(&_output_size), sizeof(_output_size));
		filestream.write(reinterpret_cast<char*>(&_weights[0]), sizeof(float) * _input_size * _output_size);
		filestream.write(reinterpret_cast<char*>(&_bias[0]), sizeof(float) * _output_size);
	}

	////////////////////////////////////////////////////////////
	const Tensor_float& logistic_regression::predict(const Tensor_float input)
	{
		ensure(input.width() == (int)_input_size);
		ensure(input.height() == 1 && input.depth() == 1);
		
		//fill outputs with bias
		for (int k = 0; k < (int)_output_size; k++)
			_outputs[k] = _bias[k];	
		
		//Compute weighted sum
		for (int i = 0; i < (int)_input_size; i++) {
			if (input[i] == 0) continue; //exploit sparsity
			for (int k = 0; k < (int)_output_size; k++) {
				_outputs[k] += _weights.at(i, k) * input[i];
			}
		}
		
		//Apply sigmoid actication function
		for (int k = 0; k < (int)_output_size; k++)
			_outputs[k] = sigmoid(_outputs[k]);	

		return _outputs;
	}
	
	////////////////////////////////////////////////////////////
	void logistic_regression::fit(Tensor_float inputs, Tensor_float targets, const float starting_learningrate,
		const unsigned int epochs, const bool verbose)
	{
		ensure(inputs.width() == (int)_input_size); //check for correct input and target shape
		ensure(inputs.height() > 1 && inputs.depth() == 1);
		ensure(targets.width() == (int)_output_size);
		ensure(targets.height() == inputs.height());
			
		std::vector<int> shuffle_idx(inputs.height());
		for (int i = 0; i < (int)shuffle_idx.size(); i++) shuffle_idx[i] = i;
		
		float learning_rate = starting_learningrate;
		double medium_error = 0;
		for (int e = 0; e < (int)epochs; e++) {	
			
			medium_error = 0;
			
			//Shuffle dataset using indicies
			std::random_shuffle(shuffle_idx.begin(), shuffle_idx.end());

			for (int i = 0; i < (int)inputs.height(); i++) {
				
				//compute learningrate
				learning_rate = starting_learningrate * (1.f - (double)(e * inputs.height() + i) / (double)(epochs * inputs.height()));
				
				//Fit out sample and update medium_error for log infos
				int indx = shuffle_idx[i];
				medium_error += fit_single_sample(inputs.ptr(0, indx), targets.ptr(0, indx), learning_rate);
			}

			if (verbose) printf("Epoch: %d MediumError: %f lr: %f\n", e, medium_error / (double)inputs.height(), learning_rate);
		}
	}
	
	////////////////////////////////////////////////////////////
	const float logistic_regression::fit_single_sample(Tensor_float input, Tensor_float target, const float learningrate)
	{
		//calculate output
		predict(input);

		//calculate prediction errors
		float error = 0;
		for (int j = 0; j < (int)_output_size; j++) {
			_errors[j] = target[j] - _outputs[j];
			error += fabs(_errors[j]);
		}

		//update weights
		for (int j = 0; j < (int)_output_size; j++) {
			_bias[j] += _errors[j] * learningrate;
			for (int k = 0; k < (int)_input_size; k++)
				_weights.at(k, j) += _errors[j] * sigmoid_deriv(_outputs[j]) * input[k] * learningrate; 
		}

		return error;
	}
	
	////////////////////////////////////////////////////////////
	void logistic_regression::test(Tensor_float inputs, Tensor_float targets)
	{
		ensure(inputs.width() == (int)_input_size); //check for correct input and target shape
		ensure(inputs.height() > 1 && inputs.depth() == 1);
		ensure(targets.width() == (int)_output_size);
		ensure(targets.height() == inputs.height());
		
		int errors = 0;
		int samples = inputs.height();
		int output_class;
		int target_id;
		for (int i = 0; i < inputs.height(); i++) {
			predict(inputs.ptr(0, i));
			_outputs.max(NULL, &output_class);
			targets.ptr(0, i).max(NULL, &target_id); //get correct output id
			if (target_id != output_class) errors++;
		}
		printf("Testing -> samples: %d errors: %d accuracy: %f\n", samples, errors, 1.f - (double)errors / (double)samples);
	}
	
	////////////////////////////////////////////////////////////
	const float logistic_regression::sigmoid(const float x)
	{
		return 1.f / (1.f + exp(-x));
	}
	
	////////////////////////////////////////////////////////////
	const float logistic_regression::sigmoid_deriv(const float x)
	{
		return x * (1 - x);
	}

	////////////////////////////////////////////////////////////
	const Tensor_float& logistic_regression::get_output()
	{
		return _outputs;
	}

} /* namespace ai */
