////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Relu.hpp"
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
	std::shared_ptr<Operation> Relu::make()
	{
		return std::shared_ptr<Operation>(new Relu());
	}

	////////////////////////////////////////////////////////////
	Relu::Relu() {}
	
	////////////////////////////////////////////////////////////
	Relu::Relu(ai::IOData& data)
	{
		ai::IOData* size = data.findNode("size");
		ensure(size != NULL);
		ai::IOData* width = data.findNode("width");
		ensure(width != NULL);
		ai::IOData* height = data.findNode("height");
		ensure(height != NULL);
		ai::IOData* depth = data.findNode("depth");
		ensure(depth != NULL);
		size->get(_size);
		width->get(_width);
		height->get(_height);
		depth->get(_depth);
        _outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_outputs.fill(0);
		
		#ifdef CUDA_BACKEND
		_cudnnactivation.create(_size, 1, ai::cudnn::ACTIVATION_RELU);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Relu::initialize(std::vector<Operation*> &inputs)
	{
		//Check for errors
		ensure(inputs.size() == 1);

		//Calculate size
		_size = inputs[0]->_outputs.size();
		_width = inputs[0]->_outputs.width();
		_height = inputs[0]->_outputs.height();
		_depth = inputs[0]->_outputs.depth();

		//Initialize vectors
        _outputs.setshape(_width, _height, _depth);
        _outputs.fill(0);
        _errors.setshape(_size);
        _outputs.fill(0);
		
		#ifdef CUDA_BACKEND
		_cudnnactivation.create(_size, 1, ai::cudnn::ACTIVATION_RELU);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Relu::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Relu::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		ensure(inputs[0]->_outputs.size() == _outputs.size());
		
		#ifdef CUDA_BACKEND
	
		_cudnnactivation.foreward(inputs[0]->_outputs.pointer(), _outputs.pointer());

		#else
		//Shortcuts
		const Tensor_float& in = inputs[0]->_outputs;

		//Feedforeward
		for (int i = 0; i < (int)_outputs.size(); i++)
			_outputs[i] = in[i] * (in[i] > 0);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Relu::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);

		#ifdef CUDA_BACKEND

		_cudnnactivation.backward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _errors.pointer(), inputs[0]->_errors.pointer());

		#else

		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;
		
		//Feedforeward
		for (int i = 0; i < (int)out_errors.size(); i++)
			out_errors[i] = _errors[i] * (_outputs[i] > 0);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Relu::get_type() const
	{
		return Operation::Relu;
	}
	
	////////////////////////////////////////////////////////////
	void Relu::print()
	{
		printf("Type: Relu, Size: %d", _size);
	}
	
	#ifndef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	void Relu::foreward(const Tensor_float input, Tensor_float output)
	{
		//Check for errors
		ensure(output.size() == input.size());

		//Feedforeward
		for (int i = 0; i < (int)output.size(); i++)
			output[i] = input[i] * (input[i] > 0);
	}

	////////////////////////////////////////////////////////////
	void Relu::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Check for errors
		ensure(errors.size() == out_errors.size());

		//Backward
		for (int i = 0; i < (int)out_errors.size(); i++)
			out_errors[i] = errors[i] * (outputs[i] > 0);
	}
	
	#endif
	
} /* namespace ai */
