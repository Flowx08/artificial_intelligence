////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Addition.hpp"
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
	std::shared_ptr<Operation> Addition::make()
	{
		return std::shared_ptr<Operation>(new Addition());
	}
	
	////////////////////////////////////////////////////////////
	Addition::Addition() {}
	
	////////////////////////////////////////////////////////////
	Addition::Addition(ai::IOData& data)
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
	void Addition::initialize(std::vector<Operation*> &inputs)
	{
		//Check for errors
		ensure(inputs.size() > 0);
		_size = inputs[0]->_outputs.size();
		for (int i = 0; i < (int)inputs.size(); i++)
			ensure_print(_size == inputs[i]->_outputs.size(), "%d %d\n", _size, inputs[i]->_outputs.size());
		
		//Calculate size
		_width = inputs[0]->_outputs.width();
		_height = inputs[0]->_outputs.height();
		_depth = inputs[0]->_outputs.depth();

		//Initialize vectors
        _outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_outputs.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	void Addition::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
	}
		
	////////////////////////////////////////////////////////////
	void Addition::run(std::vector<Operation*> &inputs, const bool training) 
	{
		#ifdef CUDA_BACKEND
	
		for (int i = 0; i < (int)inputs.size(); i++) {
			if (i == 0) TensorCUDA_float_copy(inputs[i]->_outputs, _outputs);
			else TensorCUDA_float_sum(inputs[i]->_outputs, _outputs);
		}

		#else
		for (int i = 0; i < (int)inputs.size(); i++) {
			
			//Shortcut
			const Tensor_float& in = inputs[i]->_outputs;
			
			if (i == 0)
			{
				for (int j = 0; j < _size; j++)
					_outputs[j] = in[j];
			}
			else
			{
				for (int j = 0; j < _size; j++)
					_outputs[j] += in[j];
			}
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Addition::backprop(std::vector<Operation*> &inputs) 
	{
		#ifdef CUDA_BACKEND

		for (int i = 0; i < (int)inputs.size(); i++)
			TensorCUDA_float_sum(_errors, inputs[i]->_errors);

		#else
		for (int i = 0; i < (int)inputs.size(); i++) {
			
			//Shortcuts
			Tensor_float &out_errors = inputs[i]->_errors;
			
			//Feedforeward
			for (int i = 0; i < _size; i++)
				out_errors[i] += _errors[i];
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Addition::get_type() const
	{
		return Operation::Addition;
	}
	
	////////////////////////////////////////////////////////////
	void Addition::print()
	{
		printf("Type: Addition, Size: %d", _size);
	}
	
} /* namespace ai */
