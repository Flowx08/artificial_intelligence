////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Selu.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

static const double _alpha = 1.6732632423543772;
static const double _scale = 1.0507009873554804;

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Selu::make()
	{
		return std::shared_ptr<Operation>(new Selu());
	}

	////////////////////////////////////////////////////////////
	Selu::Selu() {}

	////////////////////////////////////////////////////////////
	Selu::Selu(ai::IOData& data)
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
	void Selu::initialize(std::vector<Operation*> &inputs)
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
	void Selu::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}

	////////////////////////////////////////////////////////////
	void Selu::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		ensure(inputs[0]->_outputs.size() == _outputs.size());

#ifdef CUDA_BACKEND

		cuda::selu_foreward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _size);

#else
		//Shortcuts
		const Tensor_float& in = inputs[0]->_outputs;

		//Feedforeward
		for (int i = 0; i < (int)_outputs.size(); i++) {
			if (in[i] >= 0.0) _outputs[i] = _scale * in[i];
			else _outputs[i] = _scale * (_alpha * exp(in[i]) - _alpha);
		}
#endif
	}

	////////////////////////////////////////////////////////////
	void Selu::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);

#ifdef CUDA_BACKEND

		cuda::selu_backward(_errors.pointer(), inputs[0]->_errors.pointer(), _outputs.pointer(), _size);

#else

		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;

		//Feedforeward
		for (int i = 0; i < (int)out_errors.size(); i++) {
			if (_outputs[i] >= 0.0) out_errors[i] = _scale * _errors[i];
			else out_errors[i] = _errors[i] * (_outputs[i] + _scale * _alpha);
		}
#endif
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Selu::get_type() const
	{
		return Operation::Selu;
	}

	////////////////////////////////////////////////////////////
	void Selu::print()
	{
		printf("Type: Selu, Size: %d", _size);
	}

#ifndef CUDA_BACKEND

	////////////////////////////////////////////////////////////
	void Selu::foreward(const Tensor_float input, Tensor_float output)
	{
		//Check for errors
		ensure(output.size() == input.size());

		//Feedforeward
		for (int i = 0; i < (int)output.size(); i++) {
			if (input[i] >= 0.0) output[i] = _scale * input[i];
			else output[i] = _scale * (_alpha * exp(input[i]) - _alpha);
		}
	}

	////////////////////////////////////////////////////////////
	void Selu::backward(const Tensor_float errors, const Tensor_float outputs, Tensor_float out_errors)
	{
		//Check for errors
		ensure(errors.size() == out_errors.size());

		//Backward
		for (int i = 0; i < (int)out_errors.size(); i++) {
			if (outputs[i] > 0.0) out_errors[i] = _scale * errors[i];
			else out_errors[i] = errors[i] * (outputs[i] + _scale * _alpha);
		}
	}

#endif

} /* namespace ai */
