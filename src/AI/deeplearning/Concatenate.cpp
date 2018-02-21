////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Concatenate.hpp"
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
	std::shared_ptr<Operation> Concatenate::make()
	{
		return std::shared_ptr<Operation>(new Concatenate());
	}

	////////////////////////////////////////////////////////////
	Concatenate::Concatenate() {}
	
	////////////////////////////////////////////////////////////
	Concatenate::Concatenate(ai::IOData& data)
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
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::initialize(std::vector<Operation*> &inputs)
	{
		//We have to decide the final shape of the tensor
		//and the number of dimensions used
		#define SAME_WIDTH_AND_HEIGHT 0
		#define SAME_WIDTH  1
		#define DIFFERENT_SHAPES 2
		int concatenation_type = SAME_WIDTH_AND_HEIGHT;

		_width = inputs[0]->_outputs.width();
		for (int i = 0; i < (int)inputs.size(); i++) {
			if (_width != inputs[i]->_outputs.width()) {
				concatenation_type = DIFFERENT_SHAPES;
				break;
			}
		}
		
		//All the inputs have the same width
		//Check if they have the same height
		if (concatenation_type == SAME_WIDTH_AND_HEIGHT) {
			_height = inputs[0]->_outputs.height();
			for (int i = 0; i < (int)inputs.size(); i++) {
				if (_height != inputs[i]->_outputs.height()) {
					concatenation_type = SAME_WIDTH;
					break;
				}
			}
		}

		switch (concatenation_type)
		{
			case SAME_WIDTH_AND_HEIGHT:
				_depth = 0;
				for (int i = 0; i < (int)inputs.size(); i++)
					_depth += inputs[i]->_outputs.depth();
				break;
			
			case SAME_WIDTH:
				_height = 0;
				_depth = 1;
				for (int i = 0; i < (int)inputs.size(); i++)
					_height += inputs[i]->_outputs.height();		
				break;

			case DIFFERENT_SHAPES:
				_width = 0;
				_height = 1;
				_depth = 1;
				for (int i = 0; i < (int)inputs.size(); i++)
					_width += inputs[i]->_outputs.width();		
				break;
		}
		
		//Calculate total output size
		_size = _width * _height * _depth;

		//Initialize vectors
        _outputs.setshape(_width, _height, _depth);
        _outputs.fill(0);
        _errors.setshape(_size);
        _outputs.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Concatenate::run(std::vector<Operation*> &inputs, const bool training) 
	{
		#ifdef CUDA_BACKEND
		
		//Make sure we have the inputs pointers
		if (_inputs_pointers.size() != inputs.size()) {
			_inputs_pointers.setshape(inputs.size());
			std::vector<float*> host_pointers_inputs(inputs.size());
			 for (int i = 0; i < inputs.size(); i++)
			 	host_pointers_inputs[i] = inputs[i]->_outputs.pointer();
			_inputs_pointers.copyToDevice(&host_pointers_inputs[0], (int)host_pointers_inputs.size());
		}
		
		//Store inputs sizes
		if (_pointers_sizes.size() != inputs.size()) {
			std::vector<int> host_pointers_sizes(inputs.size());
			for (int i = 0; i < (int)host_pointers_sizes.size(); i++) 
				host_pointers_sizes[i] = inputs[i]->_outputs.size();
			_pointers_sizes.setshape(inputs.size());
			_pointers_sizes.copyToDevice(&host_pointers_sizes[0], (int)host_pointers_sizes.size());
		}
		
		cuda::concatenate_foreward(_inputs_pointers.pointer(), _outputs.pointer(), _pointers_sizes.pointer(), (int)inputs.size(), _size);

		#else
		
		int offset  = 0;
		for (int j = 0; j < (int)inputs.size(); j++) {
			const Tensor_float& in = inputs[j]->_outputs;
			for (int i = 0; i < (int)in.size(); i++)
				_outputs[offset + i] = in[i];
			offset += in.size();
		}
		
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::backprop(std::vector<Operation*> &inputs) 
	{
		#ifdef CUDA_BACKEND
		
		//Make sure we have the outerrors pointers
		if (_outerrors_pointers.size() != inputs.size()) {
			_outerrors_pointers.setshape(inputs.size());
			std::vector<float*> host_pointers_outerrors(inputs.size());
			 for (int i = 0; i < inputs.size(); i++)
				host_pointers_outerrors[i] = inputs[i]->_errors.pointer();
			_outerrors_pointers.copyToDevice(&host_pointers_outerrors[0], (int)host_pointers_outerrors.size());
		}
	
		cuda::concatenate_backward(_errors.pointer(), _outerrors_pointers.pointer(), _pointers_sizes.pointer(), (int)inputs.size(), _size);

		#else

		int offset  = 0;
		for (int j = 0; j < (int)inputs.size(); j++) {
			Tensor_float& oe = inputs[j]->_errors;
			for (int i = 0; i < (int)oe.size(); i++)
				oe[i] = _errors[offset + i];
			offset += oe.size();
		}

		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Concatenate::get_type() const
	{
		return Operation::Concatenate;
	}
	
	////////////////////////////////////////////////////////////
	void Concatenate::print()
	{
		printf("Type: Concatenate, Output: (%dx%dx%d)", _width, _height, _depth);
	}
	
} /* namespace ai */
