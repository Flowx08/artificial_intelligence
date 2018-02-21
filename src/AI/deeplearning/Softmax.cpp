////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Softmax.hpp"
#include <cmath>
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
	std::shared_ptr<Operation> Softmax::make(double input_scale)
	{
		return std::shared_ptr<Operation>(new Softmax(input_scale));
	}

	////////////////////////////////////////////////////////////
	Softmax::Softmax(double input_scale) 
	{
		_input_scale = input_scale;
		_epsilon = 1e-4;
	}

	////////////////////////////////////////////////////////////
	Softmax::Softmax(ai::IOData& data)
	{
		ai::IOData* size = data.findNode("size");
		ensure(size != NULL);
		ai::IOData* input_scale = data.findNode("input_scale");
		ensure(input_scale != NULL);
		ai::IOData* epsilon = data.findNode("epsilon");
		ensure(epsilon != NULL);
		size->get(_size);
		input_scale->get(_input_scale);
		epsilon->get(_epsilon);
		_outputs.setshape(_size);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
	}

	////////////////////////////////////////////////////////////
	void Softmax::initialize(std::vector<Operation*> &inputs)
	{
		//Check for errors
		ensure(inputs.size() == 1);

		//Calculate size
		_size = inputs[0]->_size;

		//Initialize vectors
		_outputs.setshape(_size);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
	}

	////////////////////////////////////////////////////////////
	void Softmax::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("input_scale", _input_scale);
		data.pushNode("epsilon", _epsilon);
	}

	////////////////////////////////////////////////////////////
	void Softmax::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		ensure(inputs[0]->_outputs.size() == _outputs.size());

#ifdef CUDA_BACKEND

		cuda::softmax_foreward(inputs[0]->_outputs.pointer(), _outputs.pointer(), _input_scale, _size, _epsilon);

#else

		//Shortcuts
		Tensor_float& in = inputs[0]->_outputs;
		double sum = 0;
		
		//Calculate sum of all the inp
		for (int i = 0; i < _size; i++) {
			_outputs[i] = std::exp(in[i] * _input_scale);
			sum += _outputs[i];
		}
		
		//Calculate outputs
		for (int i = 0; i < _size; i++)
			_outputs[i] = _outputs[i] / (sum + _epsilon);

#endif
	}

	////////////////////////////////////////////////////////////
	void Softmax::backprop(std::vector<Operation*>& inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);

#ifdef CUDA_BACKEND

		cuda::softmax_backward(_errors.pointer(), inputs[0]->_errors.pointer(), _outputs.pointer(), _size);

#else
		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;

		//Feedforeward
		for (int i = 0; i < (int)out_errors.size(); i++)
			out_errors[i] = _outputs[i] * (1.f - _outputs[i]) * _errors[i];
#endif
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Softmax::get_type() const
	{
		return Operation::Softmax;
	}

	////////////////////////////////////////////////////////////
	void Softmax::print()
	{
		printf("Type: Softmax, Size: %d, Input_Scale: %f", _size, _input_scale);
	}

} /* namespace ai */
