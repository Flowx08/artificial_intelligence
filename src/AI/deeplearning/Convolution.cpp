////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Convolution.hpp"
#include "../util/Util.hpp"
#include "WeightRegularization.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif
#include "Cost.hpp"
#include <stdlib.h> //REM

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai 
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Convolution::make(const int filter_size, const int filter_count,
			const int stride, const int padding, const float gradient_clipping, const float l1_regularization,
			const float l2_regularization)
	{
		return std::shared_ptr<Operation>(new Convolution(filter_size, filter_count, stride,
					padding, gradient_clipping, l1_regularization, l2_regularization));
	}

	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Convolution::make(const ai::Point filter_size, const int filter_count,
			const int stride, const int padding, const float gradient_clipping, const float l1_regularization,
			const float l2_regularization)
	{
		return std::shared_ptr<Operation>(new Convolution(filter_size, filter_count, stride,
					padding, gradient_clipping, l1_regularization, l2_regularization));
	}

	////////////////////////////////////////////////////////////
	Convolution::Convolution(const int filter_size, const int filter_count, const int stride,
			const int padding, const float gradient_clipping, const float l1_regularization, const float l2_regularization)
	{
		_input_width = 0;
		_input_height = 0;
		_input_count = 0;
		_filter_width = filter_size;
		_filter_height = filter_size;
		_filter_count = filter_count;
		_stride = stride;
		_padding = padding;
		_gradient_clipping = gradient_clipping;
		_l1_regularization = l1_regularization;
		_l2_regularization = l2_regularization;
		_fixed_parameters = false;
	}

	////////////////////////////////////////////////////////////
	Convolution::Convolution(const ai::Point filter_size, const int filter_count,
			const int stride, const int padding, const float gradient_clipping, const float l1_regularization,
			const float l2_regularization)
	{
		_input_width = 0;
		_input_height = 0;
		_input_count = 0;
		_filter_width = filter_size.x;
		_filter_height = filter_size.y;
		_filter_count = filter_count;
		_stride = stride;
		_padding = padding;
		_gradient_clipping = gradient_clipping;
		_l1_regularization = l1_regularization;
		_l2_regularization = l2_regularization;
		_fixed_parameters = false;
	}

	////////////////////////////////////////////////////////////
	Convolution::Convolution(ai::IOData& data)
	{
		ai::IOData* input_width = data.findNode("input_width");
		ensure(input_width != NULL);
		ai::IOData* input_height = data.findNode("input_height");
		ensure(input_height != NULL);
		ai::IOData* input_count = data.findNode("input_count");
		ensure(input_count != NULL);
		ai::IOData* output_width = data.findNode("output_width");
		ensure(output_width != NULL);
		ai::IOData* output_height = data.findNode("output_height");
		ensure(output_height != NULL);
		ai::IOData* stride = data.findNode("stride");
		ensure(stride != NULL);
		ai::IOData* filter_width = data.findNode("filter_width");
		ensure(filter_width != NULL);
		ai::IOData* filter_height = data.findNode("filter_height");
		ensure(filter_height != NULL);
		ai::IOData* filter_count = data.findNode("filter_count");
		ensure(filter_count != NULL);
		ai::IOData* padding = data.findNode("padding");
		ensure(padding != NULL);

		ai::IOData* l1 = data.findNode("l1_regularization");
		ai::IOData* l2 = data.findNode("l2_regularization");
		if (l1 != NULL) l1->get(_l1_regularization);
		else _l1_regularization = 0; //default value
		if (l2 != NULL) l1->get(_l2_regularization);
		else _l2_regularization = 0; //default value
		
		ai::IOData* grad_clippning = data.findNode("gradient_clipping");
		if (grad_clippning != NULL) grad_clippning->get(_gradient_clipping);

		input_width->get(_input_width); 
		input_height->get(_input_height);
		input_count->get(_input_count);
		output_width->get(_output_width);
		output_height->get(_output_height);
		stride->get(_stride); 
		filter_width->get(_filter_width);
		filter_height->get(_filter_height);
		filter_count->get(_filter_count);
		padding->get(_padding);

		_output_size = _output_width * _output_height; //output size of one filter
		_input_size = _input_width * _input_height * _input_count; 
		_size = _output_size * _filter_count;

		//Load weights
		_weights.load(data, "weights");

		//Load bias
		_bias.load(data, "bias");

		//Create output-input map
		_convmap = std::vector< std::vector<int> >(_output_width * _output_height);
		for (int x = 0; x < _output_width; x++) {
			for (int y = 0; y < _output_height; y++) {
				_convmap[y * _output_width + x] = std::vector<int>(_filter_width * _filter_height, 0);

				const int input_x = x * _stride - _padding;
				const int input_y = y * _stride - _padding;

				for (int kx = 0; kx < _filter_width; kx++) {
					for (int ky = 0; ky < _filter_height; ky++) {
						if (input_y + ky < 0 || input_y + ky >= _input_height ||
								input_x + kx < 0 || input_x + kx >= _input_width)
							_convmap[y * _output_width + x][ky * _filter_width + kx] = -1;
						else 
							_convmap[y * _output_width + x][ky * _filter_width + kx] = (input_y + ky) * _input_width + input_x + kx; 
					}
				}
			}
		}

#ifdef CUDA_BACKEND

		cudaconv.create(_input_width, _input_height, _input_count, 1, _filter_width, _filter_height,
				_filter_count, _padding, _padding, _stride, _stride, true);

		_workspace.setshape(cudaconv.getWorkspaceSize() / sizeof(float) + 1);
		_workspace.fill(0);

		_out_in_map.setshape(_output_width * _output_height * _filter_width * _filter_height);
		int* tmp_out_in_map = (int*)malloc(sizeof(int) * _out_in_map.size());
		for (int i = 0; i < _out_in_map.size(); i++)
			tmp_out_in_map[i] = _convmap[i / (_filter_width * _filter_height)][i % (_filter_width * _filter_height)];
		_out_in_map.copyToDevice(&tmp_out_in_map[0], _out_in_map.size());
		free(tmp_out_in_map);

		_in_weights_map.setshape(_filter_width * _filter_height, _input_width, _input_height);
		_in_out_map.setshape(_filter_width * _filter_height, _input_width, _input_height);

		//Allocate cpu temporary buffers and reset them
		int* tmp_in_weights_map = (int*)malloc(sizeof(int) * _in_weights_map.size());
		int* tmp_in_out_map = (int*)malloc(sizeof(int) * _in_out_map.size());
		for (int i = 0; i < _in_out_map.size(); i++) {
			tmp_in_weights_map[i] = -1;
			tmp_in_out_map[i] = -1;
		}

		//Fill temporary buffers
		for (int x = 0; x < _output_width; x++) {
			for (int y = 0; y < _output_height; y++) {
				for (int w = 0; w < _filter_width * _filter_height; w++) {
					if (_convmap[y * _output_width + x][w] == -1) continue;
					ensure_print(_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w <
							_filter_width * _filter_height * _input_width * _input_height, "%d %d\n",
							_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w,
							_convmap[y * _output_width + x][w]);
					tmp_in_out_map[_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w] = y * _output_width + x; 
					tmp_in_weights_map[_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w] = w; 
				}
			}
		}

		//Copy temporary buffers to GPU
		_in_weights_map.copyToDevice(&tmp_in_weights_map[0], _in_weights_map.size());
		_in_out_map.copyToDevice(&tmp_in_out_map[0], _in_out_map.size());

		//Deallocate
		free(tmp_in_weights_map);
		free(tmp_in_out_map);
#endif

		//Outputs and deltas
		_outputs.setshape(_output_width, _output_height, _filter_count);
		_outputs.fill(0);
		_errors.setshape(_output_width, _output_height, _filter_count);
		_errors.fill(0);
		_weights_deltas.setshape(_filter_width * _filter_height  * _input_count * _filter_count);
		_weights_deltas.fill(0);
		_bias_deltas.setshape(_filter_count);
		_bias_deltas.fill(0);
	}

	////////////////////////////////////////////////////////////
	void Convolution::initialize(std::vector<Operation*> &inputs)
	{
		//Only one input allowed
		ensure(inputs.size() == 1); 

		initialize(inputs[0]->_outputs.width(), inputs[0]->_outputs.height(), inputs[0]->_outputs.depth());
	}

	////////////////////////////////////////////////////////////
	void Convolution::initialize(const int input_width, const int input_height, const int input_count)
	{
		ensure(input_count >= 1);
		ensure(input_height >= 1 && input_width >= 1);
		_input_count = input_count;
		_input_height = input_height;
		_input_width = input_width;

		//Compute caratteristics
		_input_size = _input_width * _input_height * _input_count;
		_output_width = (_input_width - _filter_width + 2 * _padding) / _stride + 1.0;
		_output_height = (_input_height - _filter_height + 2 * _padding) / _stride + 1.0;
		_output_size = _output_width * _output_height;
		_size = _output_size * _filter_count;

		//Check for errors
		ensure_print(_output_width > 0 && _output_height > 0 && _filter_count > 0,
				"%d %d %d\n", _output_width, _output_height, _filter_count);

		//Initialize variables and buffers
		_outputs.setshape(_output_width, _output_height, _filter_count);
		_outputs.fill(0);
		_errors.setshape(_output_width, _output_height, _filter_count);
		_errors.fill(0);
		_weights_deltas.setshape(_filter_width * _filter_height  * _input_count * _filter_count);
		_weights_deltas.fill(0);
		_bias_deltas.setshape(_filter_count);
		_bias_deltas.fill(0);

		//Initialize weights
		_weights.setshape(_filter_width * _filter_height, _input_count, _filter_count);
		_weights.fill(0.0, sqrt(2.0 / (_filter_width * _filter_height * _input_count + 1)));
		_bias.setshape(_filter_count);
		_bias.fill(0.0, sqrt(2.0 / (_filter_width * _filter_height * _input_count + 1)));

		//Create output-input map
		_convmap = std::vector< std::vector<int> >(_output_width * _output_height);
		for (int x = 0; x < _output_width; x++) {
			for (int y = 0; y < _output_height; y++) {
				_convmap[y * _output_width + x] = std::vector<int>(_filter_width * _filter_height, 0);

				const int input_x = x * _stride - _padding;
				const int input_y = y * _stride - _padding;

				for (int kx = 0; kx < _filter_width; kx++) {
					for (int ky = 0; ky < _filter_height; ky++) {
						if (input_y + ky < 0 || input_y + ky >= _input_height ||
								input_x + kx < 0 || input_x + kx >= _input_width)
							_convmap[y * _output_width + x][ky * _filter_width + kx] = -1;
						else 
							_convmap[y * _output_width + x][ky * _filter_width + kx] = (input_y + ky) * _input_width + input_x + kx; 
					}
				}
			}
		}

#ifdef CUDA_BACKEND

		cudaconv.create(_input_width, _input_height, _input_count, 1, _filter_width, _filter_height,
				_filter_count, _padding, _padding, _stride, _stride, true);

		_workspace.setshape(cudaconv.getWorkspaceSize() / sizeof(float) + 1);
		_workspace.fill(0);

		_out_in_map.setshape(_output_width * _output_height * _filter_width * _filter_height);
		int* tmp_out_in_map = (int*)malloc(sizeof(int) * _out_in_map.size());
		for (int i = 0; i < _out_in_map.size(); i++)
			tmp_out_in_map[i] = _convmap[i / (_filter_width * _filter_height)][i % (_filter_width * _filter_height)];
		_out_in_map.copyToDevice(&tmp_out_in_map[0], _out_in_map.size());
		free(tmp_out_in_map);

		_in_weights_map.setshape(_filter_width * _filter_height, _input_width, _input_height);
		_in_out_map.setshape(_filter_width * _filter_height, _input_width, _input_height);

		//Allocate cpu temporary buffers and reset them
		int* tmp_in_weights_map = (int*)malloc(sizeof(int) * _in_weights_map.size());
		int* tmp_in_out_map = (int*)malloc(sizeof(int) * _in_out_map.size());
		for (int i = 0; i < _in_out_map.size(); i++) {
			tmp_in_weights_map[i] = -1;
			tmp_in_out_map[i] = -1;
		}

		//Fill temporary buffers
		for (int x = 0; x < _output_width; x++) {
			for (int y = 0; y < _output_height; y++) {
				for (int w = 0; w < _filter_width * _filter_height; w++) {
					if (_convmap[y * _output_width + x][w] == -1) continue;
					ensure_print(_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w <
							_filter_width * _filter_height * _input_width * _input_height, "%d %d\n",
							_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w,
							_convmap[y * _output_width + x][w]);
					tmp_in_out_map[_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w] = y * _output_width + x; 
					tmp_in_weights_map[_convmap[y * _output_width + x][w] * _filter_width * _filter_height + w] = w; 
				}
			}
		}

		//Copy temporary buffers to GPU
		_in_weights_map.copyToDevice(&tmp_in_weights_map[0], _in_weights_map.size());
		_in_out_map.copyToDevice(&tmp_in_out_map[0], _in_out_map.size());

		//Deallocate
		free(tmp_in_weights_map);
		free(tmp_in_out_map);
#endif

	}

	////////////////////////////////////////////////////////////
	void Convolution::save(ai::IOData& data)
	{
		data.pushNode("input_width", _input_width);
		data.pushNode("input_height", _input_height);
		data.pushNode("input_count", _input_count);
		data.pushNode("output_width", _output_width);
		data.pushNode("output_height", _output_height);
		data.pushNode("stride", _stride);
		data.pushNode("filter_width", _filter_width);
		data.pushNode("filter_height", _filter_height);
		data.pushNode("filter_count", _filter_count);
		data.pushNode("padding", _padding);
		data.pushNode("l1_regularization", _l1_regularization);
		data.pushNode("l2_regularization", _l2_regularization);
		data.pushNode("gradient_clipping", _gradient_clipping);

		//Load weights
		_weights.save(data, "weights");

		//Load bias
		_bias.save(data, "bias");
	}

	////////////////////////////////////////////////////////////
	void Convolution::run(std::vector<Operation*> &inputs, const bool training) 
	{
		ensure(inputs.size() == 1);
		run(inputs[0]->_outputs, training);
	}

	////////////////////////////////////////////////////////////
	void Convolution::backprop(std::vector<Operation*> &inputs) 
	{
		ensure(inputs.size() == 1);
		backprop(inputs[0]->_errors);
	}

	////////////////////////////////////////////////////////////
	void Convolution::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		ensure(inputs.size() == 1);
		accumulate_deltas(inputs[0]->_outputs);
	}

	////////////////////////////////////////////////////////////
	void Convolution::update_parameters(const float learningrate)
	{
		if (_fixed_parameters) return;

#ifdef CUDA_BACKEND
		
		//Gradient clipping
		if (_gradient_clipping != 0)
			cuda::gradient_clipping(_weights_delas.pointer(), _weights_delas.size(), _gradient_clipping);

		//Update weights using gradients
		cudaconv.update_weights(_weights.pointer(), _weights_deltas.pointer(), _bias.pointer(),
				_bias_deltas.pointer(), learningrate);
		
		//Regularization penality
		if (_l1_regularization != 0) ai::weightreg::l1_regularization(_weights, _l1_regularization, learningrate);
		else if (_l2_regularization != 0) ai::weightreg::l2_regularization(_weights, _l2_regularization, learningrate);

#else

		//Gradient clipping
		if (_gradient_clipping != 0) {
			for (int i = 0; i < (int)_weights_deltas.size(); i++) {
				if (_weights_deltas[i] > _gradient_clipping) _weights_deltas[i] = _gradient_clipping;
				else if(_weights_deltas[i] < -_gradient_clipping) _weights_deltas[i] = -_gradient_clipping;
			}
		}

		int deltas_index = 0;
		int weights_index = 0;

		for (int f = 0; f < _filter_count; f++) {

			//Update weights
			for (int k = 0; k < _input_count; k++) {

				//Update filter weights for this input
				for (int w = 0; w < _filter_width * _filter_height; w++)
					_weights[weights_index++] += _weights_deltas[deltas_index++] * learningrate;
			}

			//Update bias
			_bias[f] += _bias_deltas[f] * learningrate;
		}

		//Regularization penality
		if (_l1_regularization != 0) ai::weightreg::l1_regularization(_weights, _l1_regularization, learningrate);
		else if (_l2_regularization != 0) ai::weightreg::l2_regularization(_weights, _l2_regularization, learningrate);

#endif
	}

	////////////////////////////////////////////////////////////
	void Convolution::reset_deltas(const double momentum)
	{
#ifdef CUDA_BACKEND

		if (_weights_deltas.size() > 0) TensorCUDA_float_scale(_weights_deltas, momentum);
		if (_bias_deltas.size() > 0) TensorCUDA_float_scale(_bias_deltas, momentum);

#else

		for (int i = 0; i < _weights_deltas.size(); i++) _weights_deltas[i] *= momentum;
		for (int i = 0; i < _bias_deltas.size(); i++) _bias_deltas[i] *= momentum;

#endif
	}

	////////////////////////////////////////////////////////////
	void Convolution::setFixedParameters(const bool fixedparameters)
	{
		_fixed_parameters = fixedparameters;
	}

	////////////////////////////////////////////////////////////
	void Convolution::saveParameters(std::string filepath)
	{
		std::ofstream file(filepath, std::ios::binary | std::ios::out);
		if (!file) {
			printf("Error, can't open file at filepath %s\n", filepath.c_str());
			return;
		}
		_weights.save(file);
	}

	////////////////////////////////////////////////////////////
	void Convolution::loadParameters(std::string filepath)
	{
		std::ifstream file(filepath, std::ios::binary | std::ios::in);
		if (!file) {
			printf("Error, can't open file at filepath %s\n", filepath.c_str());
			return;
		}
		_weights.load(file);
		ensure(_weights.width() == _filter_width * _filter_height);
		ensure_print(_weights.height() == _input_count, "%d %d\n", _weights.height(), _input_count);
		ensure_print(_weights.depth() == _filter_count, "%d %d\n", _weights.depth(), _filter_count);
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Convolution::get_type() const
	{
		return ai::Operation::Convolution;
	}

	////////////////////////////////////////////////////////////
	void Convolution::print()
	{
		printf("Type: Convolution, Size: %d, Input: (%dx%dx%d), Output: (%dx%dx%d) Filter_Size: (%dx%d), Stride: %d, Padding: %d, l1: %f, l2: %f, Weights: %d",
				_size, _input_width, _input_height, _input_count, _output_width, _output_height, _filter_count,
				_filter_width, _filter_height, _stride, _padding, _l1_regularization, _l2_regularization,
				(_filter_width * _filter_height * _input_count + 1) * _filter_count);
	}

#ifdef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Convolution::run(const TensorCUDA_float &input, const bool training)
	{
		cudaconv.foreward(input.pointer(), _outputs.pointer(), _weights.pointer(), _bias.pointer(),
				_workspace.pointer());
	}

	////////////////////////////////////////////////////////////
	void Convolution::backprop(TensorCUDA_float &out_errors)
	{	
		if (out_errors.size() == 0) return;

		cudaconv.backward(_errors.pointer(), out_errors.pointer(), _weights.pointer(), _workspace.pointer());
	}

	////////////////////////////////////////////////////////////
	void Convolution::accumulate_deltas(const TensorCUDA_float &input)
	{
		cudaconv.accumulate_deltas(input.pointer(), _outputs.pointer(), _errors.pointer(), _weights_deltas.pointer(),
				_bias_deltas.pointer(), _workspace.pointer());
	}

#else

	////////////////////////////////////////////////////////////
	void Convolution::gradient_check()
	{
		//params
		const int input_width = 8;
		const int input_height = 8;
		const int input_depth = 3;
		const double epsilon = 10e-4;

		//test node
		const int filter_size = 4;
		const int filter_count = 10;
		const int stride = 1;
		const int padding = 1;
		Convolution node(filter_size, filter_count, stride, padding);
		node.initialize(input_width, input_height, input_depth);

		//random input
		Tensor_float input(input_width, input_height, input_depth);
		input.fill(0.5, 0.5);

		//random target
		Tensor_float target(node._outputs.size());
		target.fill(0.5, 0.5);

		//Cost function
		Cost costfun(Cost::SquaredError);

		//computed numerical gradients
		Tensor_float numgrad(node._weights.width(), node._weights.height(), node._weights.depth());

		//For each parameter
		for (int i = 0; i < node._weights.size(); i++) {
			float init_param = node._weights[i];
			node._weights[i] = init_param + epsilon;
			node.run(input, false);
			double lossPlus = costfun.getError(node._outputs, target);

			node._weights[i] = init_param - epsilon;
			node.run(input, false);
			double lossMinus = costfun.getError(node._outputs, target);

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
		Tensor_float distances(numgrad.width(), numgrad.height(), numgrad.depth());
		for (int f = 0; f < node._filter_count; f++) {

			//Update weights
			for (int k = 0; k < node._input_count; k++) {

				//Update filter weights for this input
				for (int w = 0; w < node._filter_width * node._filter_height; w++) {
					medium_delta_size += fabs(node._weights_deltas[d]);
					distances.at(f, k, w) = fabs(numgrad.at(f, k, w) + node._weights_deltas[d]);
					if (distances.at(f, k, w) > max) {
						max = distances.at(f, k, w); 
					}
					d++;
				}
			}

			d++; //bias
		}
		medium_delta_size /= node._weights_deltas.size();

		const float tollerance = medium_delta_size * 0.05;
		if (max > tollerance) printf("Gradient looks bad, differs max by %f with medium_delta_size: %f\n", max, medium_delta_size);
		else printf("Gradient looks good, differs max by %f with medium_delta_size: %f\n", max, medium_delta_size);

		//printf("%s\n", distances.tostring().c_str());
	}

	////////////////////////////////////////////////////////////
	void Convolution::run(const Tensor_float &input, const bool training)
	{
		const Tensor_float& data = input;

		int out_index = 0;	
		for (int f = 0; f < _filter_count; f++) { //Each filter

			//Shortcut for this filter outt
			out_index = f * _output_size;

			//For each output
			for (int o = 0; o < _output_size; o++) {

				//For each weight
				_outputs[out_index] = _bias[f];

				//Compute each input group
				for (int k = 0; k < _input_count; k++) {

					//Shortcut for this input group
					const float *in = &data[_input_width * _input_height * k];

					for (int w = 0; w < _filter_width * _filter_height; w++) {
						if (_convmap[o][w] == -1) continue;
						_outputs[out_index] += in[_convmap[o][w]] * _weights.at(f, k, w);
					}

				} // for each output

				out_index++;
			} // for each input group
		} // for each filter
	}

	////////////////////////////////////////////////////////////
	void Convolution::backprop(Tensor_float &out_errors)
	{
		Tensor_float& input_errors = out_errors;
		if (out_errors.size() == 0) return;

		int upcomming_errors_index = 0;

		for (int f = 0; f < _filter_count; f++) { //Each filter

			//Shortcut for this filter output
			upcomming_errors_index = f * _output_size;

			//For each output
			for (int o = 0; o < _output_size; o++) {

				//Jump computation
				if (_errors[upcomming_errors_index] == 0) continue;

				//Compute each input group
				for (int k = 0; k < _input_count; k++) {

					//Shortcut for this input group
					float *leaving_errors = &input_errors[_input_width * _input_height * k];

					//For each weight
					for (int w = 0; w < _filter_width * _filter_height; w++) {
						if (_convmap[o][w] == -1) continue;
						leaving_errors[_convmap[o][w]] += _weights.at(f, k, w) * _errors[upcomming_errors_index];
					}

				} // for each output

				upcomming_errors_index++;
			} // for each input group
		} // for each filter
	}

	////////////////////////////////////////////////////////////
	void Convolution::accumulate_deltas(const Tensor_float &input)
	{
		//Shortcut
		const Tensor_float& data = input;

		for (int f = 0; f < _filter_count; f++) { //Each filter

			//Shortcut for this filter output
			const float *upcomming_errors = &_errors[f * _output_size];
			float *t_filter_deltas = &_weights_deltas[f * _input_count * _filter_width * _filter_height];

			//For each output
			for (int o = 0; o < _output_size; o++) {

				//Jump computation
				if (upcomming_errors[o] == 0) continue;

				//Compute each input group
				for (int k = 0; k < _input_count; k++) {

					float *filter_deltas = &t_filter_deltas[k * _filter_width * _filter_height];

					//Shortcut for this input group
					const float *in = &data[_input_width * _input_height * k];

					//For each weight
					for (int w = 0; w < _filter_width * _filter_height; w++) {
						if (_convmap[o][w] == -1) continue;
						filter_deltas[w] += in[_convmap[o][w]] * upcomming_errors[o];
					}

				} // for each output

				//Bias
				_bias_deltas[f] += upcomming_errors[o];

			} // for each input group
		} // for each filter
	}

#endif

} /* namespace ai */
