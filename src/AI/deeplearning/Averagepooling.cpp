////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Averagepooling.hpp"
#include "../util/Util.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai 
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Averagepooling::make(const int filter_size, const int stride)
	{
		return std::shared_ptr<Operation>(new Averagepooling(filter_size, stride));
	}
	
	////////////////////////////////////////////////////////////
	Averagepooling::Averagepooling(const int filter_size, const int stride)
	{
		_input_width = 0;
		_input_height = 0;
		_input_count = 0;
		_filter_size = filter_size;
		_stride = stride;
	}
	
	////////////////////////////////////////////////////////////
	Averagepooling::Averagepooling(ai::IOData& data)
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
		ai::IOData* filter_size = data.findNode("filter_size");
		ensure(filter_size != NULL);
		
		input_width->get(_input_width); 
		input_height->get(_input_height);
		input_count->get(_input_count);
		output_width->get(_output_width);
		output_height->get(_output_height);
		stride->get(_stride); 
		filter_size->get(_filter_size);
		
		_output_size = _output_width * _output_height; //output size of one filter
		_input_size = _input_width * _input_height * _input_count; 
		_size = _output_size * _input_count;

		//Outputs and deltas
        _outputs.setshape(_output_width, _output_height, _input_count);
		_outputs.fill(0);
        _errors.setshape(_output_width, _output_height, _input_count);
		_errors.fill(0);
		
		#ifdef CUDA_BACKEND
		_cuda_pooling.create(_input_width, _input_height, _input_count, 1,
			_filter_size, _filter_size, ai::cudnn::POOLING_AVERAGE);
		#else
		//Average buffer
		_average_in.setshape(_size);
		_average_in.fill(0);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Averagepooling::initialize(std::vector<Operation*> &inputs)
	{
		//Only one input allowed
		ensure(inputs.size() == 1); 
		
		//Automatic infer input shape
		if (_input_count == 0 && _input_width == 0 && _input_height == 0)
		{
			ensure(inputs[0]->_outputs.depth() >= 1);
			ensure(inputs[0]->_outputs.height() >= 1 && inputs[0]->_outputs.width() >= 1);
			_input_count = inputs[0]->_outputs.depth();
			_input_height = inputs[0]->_outputs.height();
			_input_width = inputs[0]->_outputs.width();
		}

		//Check that the inputs have the correct size
		ensure(inputs[0]->_outputs.size() == _input_width * _input_height * _input_count);
		
		//Compute caratteristics
		_input_size = _input_width * _input_height * _input_count;
		_output_width = (_input_width - _filter_size) / _stride + 1.0;
		_output_height = (_input_height - _filter_size) / _stride + 1.0;
		_output_size = _output_width * _output_height;
		_size = _output_size * _input_count;

		//Initialize variables and buffers
        _outputs.setshape(_output_width, _output_height, _input_count);
        _outputs.fill(0);
        _errors.setshape(_output_width, _output_height, _input_count);
        _errors.fill(0);
		
		#ifdef CUDA_BACKEND
		_cuda_pooling.create(_input_width, _input_height, _input_count, 1,
			_filter_size, _filter_size, ai::cudnn::POOLING_AVERAGE);
		#else
		//Average buffer
		_average_in.setshape(_size);
		_average_in.fill(0);
		#endif
    }
	
	////////////////////////////////////////////////////////////
	void Averagepooling::save(ai::IOData& data)
	{
		data.pushNode("input_width", _input_width);
		data.pushNode("input_height", _input_height);
		data.pushNode("input_count", _input_count);
		data.pushNode("output_width", _output_width);
		data.pushNode("output_height", _output_height);
		data.pushNode("stride", _stride);
		data.pushNode("filter_size", _filter_size);
	}
	
	////////////////////////////////////////////////////////////
	void Averagepooling::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Errors checking
		ensure(inputs.size() == 1);

		#ifdef CUDA_BACKEND

		_cuda_pooling.foreward(inputs[0]->_outputs.pointer(), _outputs.pointer());

		//==== TESTING CUDA =====
		/*
		ai::Tensor_float t_input(inputs[0]->_outputs.size());
		inputs[0]->_outputs.copyToHost(&t_input[0], t_input.size());
		ai::Tensor_float t_out(_outputs.size());
		_outputs.copyToHost(&t_out[0], t_out.size());
		std::vector<int> t_maxids(_maxin.size());
		_maxin.copyToHost(&t_maxids[0], t_maxids.size());
		for (int i = 0; i < t_out.size(); i++) {
			ensure (t_out[i] == t_input[t_maxids[i]]);
			int z = i / (_output_width * _output_height);
			int x = i % _output_width;
			int y = (i / _output_width) % _output_height;
			const int stopX = (x * _stride + _filter_size > _input_width) ? (x * _stride + _filter_size) - x : _filter_size;
			const int stopY = (y * _stride + _filter_size > _input_height) ? (y * _stride + _filter_size) - y : _filter_size;
			int index, sx, sy;
			float max = -0x8FFF;
			for (sx = 0; sx < stopX; sx++)
				for (sy = 0; sy < stopY; sy++) {
					index = z * _input_width * _input_height + _input_width * (y * _stride + sy) + x * _stride + sx;
					if (t_input[index] > max) max = t_input[index];
				}
			ensure (t_out[i] == max);
		}
		
		printf("%f %f %f %f\n", t_input[0], t_input[1], t_input[_input_width], t_input[_input_width + 1]);
		printf("%f\n", t_out[0]);
		printf("%d\n", t_maxids[0]);
		*/

		#else
		
		//Shortcut
		const Tensor_float& data = inputs[0]->_outputs;

		int outid = 0; //Output identifier
		int stopX, stopY;
		int sx, sy;
		int inputx, inputy;

		//Let's do all the filters
		for (int f = 0; f < _input_count; f++) {
			for (int y = 0; y < _output_height; y++) {
				for (int x = 0; x < _output_width; x++) {

					//Get input X e Y
					inputx = x * _stride;
					inputy = y * _stride;

					//Check for bad borders
					stopX = (inputx + _filter_size > _input_width) ? (inputx + _filter_size) - inputx : _filter_size;
					stopY = (inputy + _filter_size > _input_height) ? (inputy + _filter_size) - inputy : _filter_size;

					//Get max of kernel region
					_outputs[outid] = 0;
					for (sx = 0; sx < stopX; sx++)
						for (sy = 0; sy < stopY; sy++)
							_outputs[outid] += data[f * _input_width * _input_height + _input_width * (inputy + sy) + inputx + sx];
					_outputs[outid] /= (float)stopX * stopY;

					//Next output
					outid++;
				}
			}
		}

		//==== TESTING CPU ====
		/*
		for (int i = 0; i < _outputs.size(); i++) {
			ensure (_outputs[i] == data[_maxin[i]]);
			int z = i / (_output_width * _output_height);
			int x = i % _output_width;
			int y = (i / _output_width) % _output_height;
			const int stopX = (x * _stride + _filter_size > _input_width) ? (x * _stride + _filter_size) - x : _filter_size;
			const int stopY = (y * _stride + _filter_size > _input_height) ? (y * _stride + _filter_size) - y : _filter_size;
			int index, sx, sy;
			float max = -0x8FFF;
			for (sx = 0; sx < stopX; sx++)
				for (sy = 0; sy < stopY; sy++) {
					index = z * _input_width * _input_height + _input_width * (y * _stride + sy) + x * _stride + sx;
					if (data[index] > max) max = data[index];
				}
			ensure (_outputs[i] == max);
		}
		*/

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Averagepooling::backprop(std::vector<Operation*> &inputs) 
	{
		//Errors checking
		ensure(inputs.size() == 1);
		if (inputs[0]->_errors.size() == 0) return;
		
		#ifdef CUDA_BACKEND

		_cuda_pooling.backward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _errors.pointer(),
			inputs[0]->_errors.pointer());

		#else
		
		Tensor_float& out_errors = inputs[0]->_errors;
		
		int outid = 0; //Output identifier
		int stopX, stopY;
		int sx, sy;
		int inputx, inputy;

		//Let's do all the filters
		for (int f = 0; f < _input_count; f++) {
			for (int y = 0; y < _output_height; y++) {
				for (int x = 0; x < _output_width; x++) {

					//Get input X e Y
					inputx = x * _stride;
					inputy = y * _stride;

					//Check for bad borders
					stopX = (inputx + _filter_size > _input_width) ? (inputx + _filter_size) - inputx : _filter_size;
					stopY = (inputy + _filter_size > _input_height) ? (inputy + _filter_size) - inputy : _filter_size;

					//Get max of kernel region
					for (sx = 0; sx < stopX; sx++)
						for (sy = 0; sy < stopY; sy++)
							out_errors[f * _input_width * _input_height + _input_width * (inputy + sy) + inputx + sx] += _errors[outid];

					//Update output index
					outid++;
				}
			}
		}

		#endif
	}
	
	
	////////////////////////////////////////////////////////////
	const Operation::Type Averagepooling::get_type() const
	{
		return ai::Operation::Averagepooling;
	}
	
	////////////////////////////////////////////////////////////
	void Averagepooling::print()
	{
		printf("Type: Averagepooling, Size: %d, Input: (%dx%dx%d), Output: (%dx%dx%d) Filter_Size: %d, Stride: %d",
			_size, _input_width, _input_height, _input_count, _output_width, _output_height, _input_count,
			_filter_size, _stride);
	}

} /* namespace ai */
