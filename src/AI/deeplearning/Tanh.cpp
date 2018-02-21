////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Tanh.hpp"
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
	std::shared_ptr<Operation> Tanh::make()
	{
		return std::shared_ptr<Operation>(new Tanh());
	}

	////////////////////////////////////////////////////////////
	Tanh::Tanh() {} 
	
	////////////////////////////////////////////////////////////
	Tanh::Tanh(ai::IOData& data)
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
		_cudnnactivation.create(_size, 1, ai::cudnn::ACTIVATION_TANH);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::initialize(std::vector<Operation*> &inputs)
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
		_cudnnactivation.create(_size, 1, ai::cudnn::ACTIVATION_TANH);
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Tanh::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		
		//Check for errors
		ensure(_outputs.size() == inputs[0]->_outputs.size());

		#ifdef CUDA_BACKEND
	
		_cudnnactivation.foreward(inputs[0]->_outputs.pointer(), _outputs.pointer());

		#else

		foreward(inputs[0]->_outputs, _outputs);

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		
		//Check for errors
		ensure(_errors.size() == inputs[0]->_errors.size());

		#ifdef CUDA_BACKEND
	
		_cudnnactivation.backward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _errors.pointer(), inputs[0]->_errors.pointer());

		#else
		
		backward(_errors, _outputs, inputs[0]->_errors);

		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Tanh::get_type() const
	{
		return Operation::Tanh; 
	}
	
	////////////////////////////////////////////////////////////
	void Tanh::print()
	{
		printf("Type: Tanh, Size: %d", _size);
	}
	
	////////////////////////////////////////////////////////////
	///	RAW OPERATIONS
	////////////////////////////////////////////////////////////
	
	#ifndef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Tanh::foreward(const Tensor_float input, Tensor_float output)
	{
		//Feedforeward
		for (int i = 0; i < (int)output.size(); i++)
			output[i] = tanh(input[i]);
	}

	////////////////////////////////////////////////////////////
	void Tanh::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Backward
		for (int i = 0; i < (int)out_errors.size(); i++)
			out_errors[i] = (1.f - outputs[i] * outputs[i]) * errors[i];
	}
	
	#endif
	
} /* namespace ai */
