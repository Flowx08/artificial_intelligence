////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Partial.hpp"
#include "../util/Util.hpp"
#include <math.h>
#include "../util/ensure.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai 
{
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Partial::make(const int size, const double connectivity)
	{
		return std::shared_ptr<Operation>(new Partial(size, connectivity)); 
	}
	
	////////////////////////////////////////////////////////////
	Partial::Partial(const int size, const double connectivity)
	{
		_size = size;
		_connectivity = connectivity;
	}
	
	////////////////////////////////////////////////////////////
	Partial::Partial(ai::IOData& data)
	{
		/*
		TODO
		file.read(reinterpret_cast<char*>(&_size), sizeof(_size));
		file.read(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.read(reinterpret_cast<char*>(&_connectivity), sizeof(_connectivity));
		
		int _foreward_map_size;
		file.read(reinterpret_cast<char*>(&_foreward_map_size), sizeof(_foreward_map_size));
		_foreward_map = std::vector< std::vector< int > >(_foreward_map_size);
		for (int i = 0; i < _foreward_map_size; i++) {
			int _map_size; 
			file.read(reinterpret_cast<char*>(&_map_size), sizeof(_map_size));
			_foreward_map[i] = std::vector<int>(_map_size);
			file.read(reinterpret_cast<char*>(&_foreward_map[i][0]), sizeof(int) * _map_size);
		}
		
		_outputs.setshape(_size);
        _outputs.fill(0);
		_errors.setshape(_size);
        _errors.fill(0);
		_deltas.setshape(_outputs.size() * (_input_size + 1));
        _deltas.fill(0);
		_weights.load(file);
		_bias.load(file);
		*/
	}
	
	////////////////////////////////////////////////////////////
	void Partial::save(ai::IOData& data)
	{
		/*
		TODO
		file.write(reinterpret_cast<char*>(&_size), sizeof(_size));
		file.write(reinterpret_cast<char*>(&_input_size), sizeof(_input_size));
		file.write(reinterpret_cast<char*>(&_connectivity), sizeof(_connectivity));
		int _foreward_map_size = _foreward_map.size();
		file.write(reinterpret_cast<char*>(&_foreward_map_size), sizeof(_foreward_map_size));
		for (int i = 0; i < _foreward_map_size; i++) {
			int _map_size = _foreward_map[i].size(); 
			file.write(reinterpret_cast<char*>(&_map_size), sizeof(_map_size));
			file.write(reinterpret_cast<char*>(&_foreward_map[i][0]), sizeof(int) * _map_size);
		}
		_weights.save(file);
		_bias.save(file);
		*/
	}
	
	////////////////////////////////////////////////////////////
	void Partial::initialize(std::vector<Operation*> &inputs)
	{
		//We can have only one input
		ensure(inputs.size() == 1);

		//Calculate input size
		_input_size = 0;
		for (int i = 0; i < (int)inputs.size(); i++)
			_input_size += inputs[i]->_outputs.size();
		
		//Initialize variables and buffers
        _outputs.setshape(_size);
        _outputs.fill(0);
        _errors.setshape(_size);
        _errors.fill(0);
        _deltas.setshape(_outputs.size() * (_input_size + 1));
        _deltas.fill(0);
		_foreward_map = std::vector< std::vector< int > >(_input_size);
		/*
		for (int i = 0; i < _input_size; i++) {
			_foreward_map[i] = std::vector<int>();
			for (int k = 0; k < _size; k++)
				if (ai::util::randf() < _connectivity)
					_foreward_map[i].push_back(k);
		}
		*/
		for (int i = 0; i < _input_size; i++) {
			_foreward_map[i] = std::vector<int>();
			for (int k = 0; k < _size; k++)
				_foreward_map[i].push_back(k);
		}

		//Initialize weights
        _weights.setshape(_size, _input_size);
        _weights.fill(0.0, 6.0 / sqrt(_input_size + _size));
        _bias.setshape(_size);
		_bias.fill(0.0, 6.0 / sqrt(_input_size + _size));
	}
	
	////////////////////////////////////////////////////////////
	void Partial::run(std::vector<Operation*> &inputs, const bool training) 
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		//Shortcuts
		const Tensor_float &in = inputs[0]->_outputs;

		//Reset outputs with bias
		for (int i = 0; i < _size; i++)
			_outputs[i] = _bias[i];
		
		//Compute all inputs
		for (int i = 0; i < in.size(); i++) {
			if (in[i] == 0) continue;
			for (int k = 0; k < (int)_foreward_map[i].size(); k++) {
				_outputs[_foreward_map[i][k]] += _weights.at(i, _foreward_map[i][k]) * in[i];
			}
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::backprop(std::vector<Operation*> &inputs) 
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		//Check we must have only one input
		Tensor_float &out_errors = inputs[0]->_errors;
		if (out_errors.size() == 0) return;

		//Back-propagate errors
		for (int i = 0; i < out_errors.size(); i++) {
			for (int k = 0; k < (int)_foreward_map[i].size(); k++) {
				out_errors[i] += _weights.at(i, _foreward_map[i][k]) * _errors[_foreward_map[i][k]];
			}
		}
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		const Tensor_float &in = inputs[0]->_outputs;
		
		/*
		int d = 0;
		for (int i = 0; i < _weights.width(); i++) {
			for (int k = 0; k <	_weights.height(); k++)
				_deltas[d++] += in[k] * _errors[i];
			_deltas[d++] += _errors[i];
		}*/
		
		for (int i = 0; i < _weights.height(); i++)
			for (int k = 0; k < _foreward_map[i].size(); k++)
				_deltas[i * _size + _foreward_map[i][k]] += in[i] * _errors[_foreward_map[i][k]];

		for (int k = 0; k < _size; k++)
			_deltas[_size * _input_size + k] += _errors[k];

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::update_parameters(const float learningrate)
	{
		#ifdef CUDA_BACKEND

		//TODO

		#else
		/*
		int d = 0;
		for (int i = 0; i < _weights.width(); i++) {
			for (int k = 0; k <	_weights.height(); k++)
				_weights.at(k, i) += _deltas[d++] * learningrate;
			_bias[i] += _deltas[d++] * learningrate;
		}*/

		for (int i = 0; i < _weights.height(); i++)
			for (unsigned int k = 0; k < _foreward_map[i].size(); k++)
				_weights.at(i, _foreward_map[i][k]) += _deltas[i * _size + _foreward_map[i][k]] * learningrate;

		for (int k = 0; k < _size; k++)
			_bias[k] += _deltas[_size * _input_size + k] * learningrate;
		#endif
	}

	////////////////////////////////////////////////////////////
	void Partial::pruning(float alpha)
	{
		#ifdef CUDA_BACKEND
		
		//TODO

		#else

		for (int i = 0; i < _input_size; i++) {
			float medium = 0;
			for (int k = 0; k < (int)_foreward_map[i].size(); k++)
				medium += fabs(_weights.at(i, _foreward_map[i][k]));
			medium /= (float)_size;
			for (int k = 0; k < (int)_foreward_map[i].size(); k++) {
				if (fabs(_weights.at(i, _foreward_map[i][k])) < medium * alpha)
					_foreward_map[i].erase(_foreward_map[i].begin() + k);
			}
		}

		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Partial::reset_deltas(const double momentum)
	{
		#ifdef CUDA_BACKEND
		
		if (_deltas.size() > 0) TensorCUDA_float_scale(_deltas, momentum);

		#else
		
		for (int i = 0; i < _deltas.size(); i++)
			_deltas[i] *= momentum;

		#endif
	}
	
	////////////////////////////////////////////////////////////
	float Partial::pruned_percent()
	{
		int c = 0;
		for (int i = 0; i < _input_size; i++)
			c += _foreward_map[i].size();
		return (float)c / (_input_size * _size);
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Partial::get_type() const
	{
		return Operation::Partial;
	}
	
	////////////////////////////////////////////////////////////
	void Partial::print()
	{
		printf("Type: Partial, Size: %d, Input_Size: %d, Connectivity: %f, Weights: %d", _size, _input_size, _connectivity, _size * (_input_size + 1));
	}

} /* namespace ai */
