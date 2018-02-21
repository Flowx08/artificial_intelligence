////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CPU_backend.hpp"
#include <math.h>
#include <stdlib.h>

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	///	UTIL
	////////////////////////////////////////////////////////////
	const float randf() { return (double)rand() / (double)RAND_MAX; }

	////////////////////////////////////////////////////////////
	///	IMPLEMENTATION
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	void conv_foreward(float* weights, float* bias, float* inputs, float* outputs, int* out_in_map, int input_width, int input_height, int input_count, int stride, int output_width, int output_height, int filters_count, int filter_area)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void conv_backward(float* weights, float* out_errors, float* errors, int* in_weight_map, int* in_out_map, int input_count, int output_size, int input_width, int input_height, int filter_area, int filters_count)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void conv_accumulate_deltas(float* weights_deltas, float* bias_deltas, float* errors, float* inputs, float* outputs, int* out_in_map, int input_count, int input_width, int input_height, int output_size, int filter_area, int filters_count)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void conv_update_parameters(float* weights, float* bias, float* weights_deltas, float* bias_deltas, int filter_area, int input_count, int filter_count, float learningrate)
	{
		int deltas_index = 0;
		int weights_index = 0;

		for (int f = 0; f < filter_count; f++) {

			//Update weights
			for (int k = 0; k < input_count; k++) {

				//Update filter weights for this input
				for (int w = 0; w < filter_area; w++)
					weights[weights_index++] += weights_deltas[deltas_index++] * learningrate;
			}

			//Update bias
			bias[f] += bias_deltas[f] * learningrate;
		}
	}

	////////////////////////////////////////////////////////////
	void linear_foreward(float* weights, float* bias, float* inputs, float* outputs, int input_size, int output_size, bool use_bias, bool accumulate)
	{
		if (use_bias)
		{
			if (!accumulate)
			{
				//Reset outputs with bias
				for (int i = 0; i < output_size; i++)
					outputs[i] = bias[i];
			}
			else
			{
				//Add bias to output
				for (int i = 0; i < output_size; i++)
					outputs[i] += bias[i];
			}
		}
		else
		{
			if (!accumulate)
			{
				for (int i = 0; i < output_size; i++)
					outputs[i] = 0;
			}
		}

		//Compute all inputs
		int weight_index = 0;
		for (int i = 0; i < input_size; i++) {
			if (inputs[i] == 0) continue;
			weight_index = i * output_size;
			for (int k = 0; k < output_size; k++)
				outputs[k] += weights[weight_index++] * inputs[i];
		}
	}

	////////////////////////////////////////////////////////////
	void linear_backward(float* weights, float* out_errors, float* errors, int input_size, int output_size)
	{
		for (int i = 0; i < output_size; i++) {
			if (errors[i] == 0) continue;
			for (int k = 0; k < input_size; k++)
				out_errors[k] += weights[k * output_size + i] * errors[i];
		}
	}

	////////////////////////////////////////////////////////////
	void linear_accumulate_deltas(float* deltas, float* inputs, float* errors, int input_size, int output_size, bool use_bias)
	{
		int d = 0;
		for (int i = 0; i < output_size; i++) {
			for (int k = 0; k <	input_size; k++)
				deltas[d++] += inputs[k] * errors[i];
			if (use_bias) deltas[d++] += errors[i];
			else d++;
		}
	}

	////////////////////////////////////////////////////////////
	void linear_update_parameters(float* weights, float* bias, float* deltas, float learningrate, int input_size, int output_size)
	{
		int d = 0;
		for (int i = 0; i < output_size; i++) {
			for (int k = 0; k <	input_size; k++)
				weights[k * output_size + i] += deltas[d++] * learningrate;
			bias[i] += deltas[d++] * learningrate;
		}
	}
	
	////////////////////////////////////////////////////////////
	void capsules_dense_foreward(float* weights, float* bias, float* inputs, float* outputs, float* input_coupling_coeff,
		int input_size, int input_capsule_size, int output_size, int output_capsule_size, int capsule_size, bool use_bias)
	{
		if (use_bias)
		{
				//Reset outputs with bias
				for (int i = 0; i < output_size; i++)
					outputs[i] = bias[i];
		}
		else
		{
				for (int i = 0; i < output_size; i++)
					outputs[i] = 0;
		}

		//Compute all inputs
		int weight_index = 0;
		int coef_id;
		const int coef_stride = input_size / input_capsule_size;
		for (int i = 0; i < input_size; i++) {
			if (inputs[i] == 0) continue;
			weight_index = i * output_size;
			coef_id = i / input_capsule_size;
			for (int k = 0; k < output_size; k++)
				outputs[k] += input_coupling_coeff[k * coef_stride + coef_id] * weights[weight_index++] * inputs[i];
		}
	}

	////////////////////////////////////////////////////////////
	void normalization_foreward(float* inputs, float* deviation, float* normalized, float* outputs, float* variance, float* gamma, float* beta, float epsilon, int size)
	{
		//Calculate mean
		double mean = 0;
		for (int i = 0; i < size; i++)
			mean += inputs[i];
		mean /= (double)size;

		//Subtract mean vector to all inputs and calculate variance
		*variance = 0;
		for (int i = 0; i < size; i++) {
			deviation[i] = inputs[i] - mean;
			*variance += deviation[i] * deviation[i];
		}
		*variance /= (double)size;

		//Calculate normalized vector
		for (int i = 0; i < size; i++) {
			normalized[i] = deviation[i] / sqrt(*variance + epsilon);
			outputs[i] = normalized[i] * *gamma + *beta;
		}
	}

	////////////////////////////////////////////////////////////
	void normalization_backward(float* errors, float* out_errors, float* deviation, float* variance, float* gamma, float* beta, float epsilon, int size)
	{
		//Pre compute some expressions
		float sum_errors = 0.f;
		float sum_errors_dev = 0.f;
		for (int i = 0; i < size; i++) {
			sum_errors += errors[i];
			sum_errors_dev += errors[i] * deviation[i];
		}

		//Calculate output errors
		for (int i = 0; i < size; i++) {
			out_errors[i] = 1.0 / (float)size * *gamma / sqrt(*variance + epsilon) * ((float)size *
					errors[i] - sum_errors - deviation[i] / (*variance + epsilon) * sum_errors_dev);
		}
	}

	////////////////////////////////////////////////////////////
	void normalization_accumulate_deltas(float* errors, float* deviation, float* variance, float* d_gamma, float* d_beta, float epsilon, int size)
	{
		//calculate beta delta
		for (int i = 0; i < size; i++)
			*d_beta += errors[i];

		//calculate gamma delta
		for (int i = 0; i < size; i++)
			*d_gamma += deviation[i] * sqrt(*variance + epsilon) * errors[i];
	}

	////////////////////////////////////////////////////////////
	void normalization_update_parameters(float* gamma, float* beta, float* d_gamma, float* d_beta, float momentum, int size, float learningrate)
	{
		*beta += ((double)*d_beta / (double)size) * learningrate;
		*gamma += ((double)*d_gamma / (double)size) * learningrate;
		*d_beta *= momentum;
		*d_gamma *= momentum;
	}

	////////////////////////////////////////////////////////////
	void dropout_foreward(const float* inputs, float* outputs, const unsigned int size, const float drop_probability, const bool training)
	{
		if (training == true)
		{
			for (int j = 0; j < size; j++)
				outputs[j] = (randf() < drop_probability) ? 0 : inputs[j];
		}
		else
		{
			for (int j = 0; j < size; j++)
				outputs[j] = inputs[j];
		}
	}

	////////////////////////////////////////////////////////////
	void dropout_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size, const float drop_probability)
	{
		for (int j = 0; j < size; j++)
			out_errors[j] = (outputs[j] == 0) ? 0 :  errors[j] * (1 - drop_probability);
	}

	////////////////////////////////////////////////////////////
	void maxpooling_foreward(float* inputs, float* outputs, int* maxbuffer, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier
		int stopX, stopY;
		double max;
		int sx, sy;
		int inputx, inputy;
		float tmp;

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Get input X e Y
					inputx = x * stride;
					inputy = y * stride;

					//Check for bad borders
					stopX = (inputx + filter_size > input_width) ? (inputx + filter_size) - inputx : filter_size;
					stopY = (inputy + filter_size > input_height) ? (inputy + filter_size) - inputy : filter_size;

					//Get max of kernel region
					max = -0x6FFFFFF;
					for (sx = 0; sx < stopX; sx++) {
						for (sy = 0; sy < stopY; sy++) {

							//Store value and check if it's greater, we must do all the calculus
							//because 'data' my be a tensor pointer and have wrong dimensions
							tmp = inputs[f * input_width * input_height + input_width * (inputy + sy) + inputx + sx];
							if (tmp > max) {
								max = tmp;
								maxbuffer[outid] = f * input_width * input_height + input_width * (inputy + sy) + inputx + sx;
							}
						}
					}

					//Update output
					outputs[outid] = max;

					//Next output
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void maxpooling_backward(float* out_errors, float* errors, int* maxbuffer, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Backproapgate errors to each sub region
					out_errors[maxbuffer[outid]] += errors[outid];

					//Update output index
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void averagepooling_foreward(float* inputs, float* outputs, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier
		int stopX, stopY;
		int sx, sy;
		int inputx, inputy;

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Get input X e Y
					inputx = x * stride;
					inputy = y * stride;

					//Check for bad borders
					stopX = (inputx + filter_size > input_width) ? (inputx + filter_size) - inputx : filter_size;
					stopY = (inputy + filter_size > input_height) ? (inputy + filter_size) - inputy : filter_size;

					//Get max of kernel region
					outputs[outid] = 0;
					for (sx = 0; sx < stopX; sx++)
						for (sy = 0; sy < stopY; sy++)
							outputs[outid] += inputs[f * input_width * input_height + input_width * (inputy + sy) + inputx + sx];
					outputs[outid] /= (float)stopX * stopY;

					//Next output
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void averagepooling_backward(float* out_errors, float* errors, int input_width, int input_height, int input_count, int stride, int filter_size, int output_width, int output_height)
	{
		int outid = 0; //Output identifier
		int stopX, stopY;
		int sx, sy;
		int inputx, inputy;

		//Let's do all the filters
		for (int f = 0; f < input_count; f++) {
			for (int y = 0; y < output_height; y++) {
				for (int x = 0; x < output_width; x++) {

					//Get input X e Y
					inputx = x * stride;
					inputy = y * stride;

					//Check for bad borders
					stopX = (inputx + filter_size > input_width) ? (inputx + filter_size) - inputx : filter_size;
					stopY = (inputy + filter_size > input_height) ? (inputy + filter_size) - inputy : filter_size;

					//Get max of kernel region
					for (sx = 0; sx < stopX; sx++)
						for (sy = 0; sy < stopY; sy++)
							out_errors[f * input_width * input_height + input_width * (inputy + sy) + inputx + sx] += errors[outid];

					//Update output index
					outid++;
				}
			}
		}
	}

	////////////////////////////////////////////////////////////
	void relu_foreward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++)
			outputs[i] = inputs[i] * (inputs[i] > 0);
	}

	////////////////////////////////////////////////////////////
	void relu_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++)
			out_errors[i] = errors[i] * (outputs[i] > 0);
	}

	////////////////////////////////////////////////////////////
	void tanh_foreward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++)
			outputs[i] = tanh(inputs[i]);
	}

	////////////////////////////////////////////////////////////
	void tanh_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++)
			out_errors[i] = (1.f - outputs[i] * outputs[i]) * errors[i];
	}

	////////////////////////////////////////////////////////////
	void sigmoid_foreward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++)
			outputs[i] = 1.0 / (1.0 + exp(-inputs[i]));
	}

	////////////////////////////////////////////////////////////
	void sigmoid_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++)
			out_errors[i] = outputs[i] * (1.f - outputs[i]) * errors[i];
	}

	static const double _alpha = 1.6732632423543772;
	static const double _scale = 1.0507009873554804;

	////////////////////////////////////////////////////////////
	void selu_foreward(const float* inputs, float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++) {
			if (inputs[i] >= 0.0) outputs[i] = _scale * inputs[i];
			else outputs[i] = _scale * (_alpha * exp(inputs[i]) - _alpha);
		}
	}

	////////////////////////////////////////////////////////////
	void selu_backward(const float* errors, float* out_errors, const float* outputs, const unsigned int size)
	{
		for (unsigned int i = 0; i < size; i++) {
			if (outputs[i] > 0.0) out_errors[i] = _scale * errors[i];
			else out_errors[i] = errors[i] * (outputs[i] + _scale * _alpha);
		}
	}

	////////////////////////////////////////////////////////////
	void softmax_foreward(float* inputs, float* outputs, float scale, int size, float epsilon)
	{
		//Calculate sum of all the inp
		double sum = 0;
		for (int i = 0; i < size; i++) {
			outputs[i] = exp(inputs[i] * scale);
			sum += outputs[i];
		}

		//Calculate outputs
		for (int i = 0; i < size; i++)
			outputs[i] = outputs[i] / (sum + epsilon);
	}

	////////////////////////////////////////////////////////////
	void softmax_backward(float* errors, float* out_errors, float* outputs, int size)
	{
		for (int i = 0; i < size; i++)
			out_errors[i] = outputs[i] * (1.f - outputs[i]) * errors[i];
	}
	
	////////////////////////////////////////////////////////////
	void capsule_squashing_foreward(float* inputs, float* outputs, int size)
	{
		double lenght = 0.f;
		for (int i = 0; i < size; i++)
			lenght += inputs[i] * inputs[i];
		lenght = sqrt(lenght);
		
		const float multiplier = (lenght * lenght) / (1.f + lenght * lenght);
		for (int i = 0; i < size; i++)
			outputs[i] = multiplier * (inputs[i] / lenght);
	}

	////////////////////////////////////////////////////////////
	void capsule_squashing_backward(float* errors, float* out_errors, float* outputs, int size)
	{

	}

	////////////////////////////////////////////////////////////
	void gradient_clipping(float* deltas, int size, const float clipping_deviation)
	{
		for (int i = 0; i < size; i++) {
			if (deltas[i] > clipping_deviation) deltas[i] = clipping_deviation;
			else if(deltas[i] < -clipping_deviation) deltas[i] = -clipping_deviation;
		}
	}

	////////////////////////////////////////////////////////////
	void l1_regularization(float* weights, const float l1_factor, const float learningrate, int size)
	{
		for (int i = 0; i < size; i++)
			weights[i] += (weights[i] > 0 ? -1.f : 1.f) * l1_factor * learningrate;
	}

	////////////////////////////////////////////////////////////
	void l2_regularization(float* weights, const float l2_factor, const float learningrate, int size)
	{
		for (int i = 0; i < size; i++)
			weights[i] += (0 - weights[i]) * l2_factor * learningrate;
	}

} /* namespace ai */
