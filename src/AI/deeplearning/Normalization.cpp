////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Normalization.hpp"
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
	std::shared_ptr<Operation> Normalization::make(float momentum)
	{
		return std::shared_ptr<Operation>(new Normalization(momentum));
	}

	////////////////////////////////////////////////////////////
	Normalization::Normalization(float momentum)
	{
		ensure(momentum >= 0 && momentum < 1);
		_gamma = 1.f;
		_beta = 0.f;
		_epsilon = 1e-4;
		_momentum = momentum;
		_d_beta = 0;
		_d_gamma = 0;
	}

	////////////////////////////////////////////////////////////
	Normalization::Normalization(ai::IOData& data)
	{
		ai::IOData* size = data.findNode("size");
		ensure(size != NULL);
		ai::IOData* width = data.findNode("width");
		ensure(width != NULL);
		ai::IOData* height = data.findNode("height");
		ensure(height != NULL);
		ai::IOData* depth = data.findNode("depth");
		ensure(depth != NULL);
		ai::IOData* gamma = data.findNode("gamma");
		ensure(gamma != NULL);
		ai::IOData* beta = data.findNode("beta");
		ensure(beta != NULL);
		ai::IOData* epsilon = data.findNode("epsilon");
		ensure(epsilon != NULL);
		ai::IOData* momentum = data.findNode("momentum");
		ensure(momentum != NULL);
		size->get(_size);
		width->get(_width);
		height->get(_height);
		depth->get(_depth);
		gamma->get(_gamma);
		beta->get(_beta);
		epsilon->get(_epsilon);
		momentum->get(_momentum);

		_outputs.setshape(_width, _height, _depth);
		_outputs.fill(0);
		_errors.setshape(_size);
		_errors.fill(0);
		_deviation.setshape(_size);
		_deviation.fill(0);
		_normalized.setshape(_size);
		_normalized.fill(0);
		_d_beta = 0;
		_d_gamma = 0;

#ifdef CUDA_BACKEND
		float* tmp = new float[5];
		tmp[0] = 0; //variance
		tmp[1] = _gamma; //gamma
		tmp[2] = _beta; //beta
		tmp[3] = 0; //gamma delta
		tmp[4] = 0; //beta delta
		_params.setshape(5);
		_params.copyToDevice(tmp, 5);
		delete[] tmp;
#endif
	}

	////////////////////////////////////////////////////////////
	void Normalization::initialize(std::vector<Operation*> &inputs)
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
		_errors.fill(0);
		_deviation.setshape(_size);
		_deviation.fill(0);
		_normalized.setshape(_size);
		_normalized.fill(0);
		_d_beta = 0;
		_d_gamma = 0;

#ifdef CUDA_BACKEND
		float* tmp = new float[5];
		tmp[0] = 0; //variance
		tmp[1] = _gamma; //gamma
		tmp[2] = _beta; //beta
		tmp[3] = 0; //gamma delta
		tmp[4] = 0; //beta delta
		_params.setshape(5);
		_params.copyToDevice(tmp, 5);
		delete[] tmp;
#endif
	}

	////////////////////////////////////////////////////////////
	void Normalization::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("width", _width);
		data.pushNode("height", _height);
		data.pushNode("depth", _depth);
		data.pushNode("gamma", _gamma);
		data.pushNode("beta", _beta);
		data.pushNode("epsilon", _epsilon);
		data.pushNode("momentum", _momentum);
	}

	////////////////////////////////////////////////////////////
	void Normalization::run(std::vector<Operation*> &inputs, const bool training) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);
		ensure(inputs[0]->_outputs.size() == _outputs.size());

#ifdef CUDA_BACKEND

		ai::cuda::normalization_foreward(inputs[0]->_outputs.pointer(), _deviation.pointer(), _normalized.pointer(),
				_outputs.pointer(), &_params.pointer()[0], &_params.pointer()[1], &_params.pointer()[2], _epsilon, _size);


		//===== TESTING NORMALIZATION ======
		/*
			 Tensor_float t_out(_outputs.size());
			 _outputs.copyToHost(t_out.pointer(), t_out.size());
			 printf("%s\n", t_out.tostring().c_str());
			 */

#else

		//Shortcuts
		Tensor_float& in = inputs[0]->_outputs;
		
		//Calculate mean
		_mean = 0;
		for (int i = 0; i < in.size(); i++)
			_mean += in[i];
		_mean /= (double)in.size();
		
		//Subtract mean vector to all inputs and calculate variance
		_variance = 0;
		for (int i = 0; i < in.size(); i++) {
			_deviation[i] = in[i] - _mean;
			_variance += _deviation[i] * _deviation[i];
		}
		_variance /= (double)in.size();
		
		//Calculate normalized vector
		for (int i = 0; i < in.size(); i++) {
			_normalized[i] = _deviation[i] / std::sqrt(_variance + _epsilon);
			_outputs[i] = _normalized[i] * _gamma + _beta;
		}
	
		//===== TESTING NORMALIZATION ======
		//printf("%s\n", _outputs.tostring().c_str());

#endif
	}

	////////////////////////////////////////////////////////////
	void Normalization::backprop(std::vector<Operation*> &inputs) 
	{
		//Check for correct input size
		ensure(inputs.size() == 1);

#ifdef CUDA_BACKEND

		ai::cuda::normalization_backward(_errors.pointer(), inputs[0]->_errors.pointer(), _deviation.pointer(),
				&_params.pointer()[0], &_params.pointer()[1], &_params.pointer()[2], _epsilon, _size);

#else
		//Shortcuts
		Tensor_float &out_errors = inputs[0]->_errors;

		//Pre compute some expressions
		float sum_errors = 0.f;
		float sum_errors_dev = 0.f;
		for (int i = 0; i < _errors.size(); i++) {
			sum_errors += _errors[i];
			sum_errors_dev += _errors[i] * _deviation[i];
		}

		//Calculate output errors
		for (int i = 0; i < out_errors.size(); i++) {
			out_errors[i] = 1.0 / (float)_size * _gamma / sqrt(_variance + _epsilon) * ((float)_size *
					_errors[i] - sum_errors - _deviation[i] / (_variance + _epsilon) * sum_errors_dev);
		}

		/*
			 dh = (1. / N) * gamma * (var + eps)**(-1. / 2.) * (N * dy - np.sum(dy, axis=0)
			 - (h - mu) * (var + eps)**(-1.0) * np.sum(dy * (h - mu), axis=0))
			 */
#endif
	}

	////////////////////////////////////////////////////////////
	void Normalization::accumulate_deltas(std::vector<Operation*> &inputs)
	{
#ifdef CUDA_BACKEND

		ai::cuda::normalization_accumulate_deltas(_errors.pointer(), _deviation.pointer(), &_params.pointer()[0],
				&_params.pointer()[3], &_params.pointer()[4], _epsilon, _size);

#else
		//calculate beta delta
		for (int i = 0; i < _errors.size(); i++)
			_d_beta += _errors[i];

		//calculate gamma delta
		for (int i = 0; i < _errors.size(); i++)
			_d_gamma += _deviation[i] * sqrt(_variance + _epsilon) * _errors[i];
#endif
	}

	////////////////////////////////////////////////////////////
	void Normalization::update_parameters(const float learningrate)
	{
#ifdef CUDA_BACKEND

		ai::cuda::normalization_update_parameters(&_params.pointer()[1], &_params.pointer()[2], &_params.pointer()[3],
				&_params.pointer()[4], _momentum, _size, learningrate);

#else
		_beta += ((double)_d_beta / (double)_size) * learningrate;
		_gamma += ((double)_d_gamma / (double)_size) * learningrate;
		_d_beta *= _momentum;
		_d_gamma *= _momentum;
#endif
	}

	////////////////////////////////////////////////////////////
	const Operation::Type Normalization::get_type() const
	{
		return Operation::Normalization;
	}

	////////////////////////////////////////////////////////////
	void Normalization::print()
	{
		printf("Type: Normalization, Size: %d, Momentum: %f", _size, _momentum);
	}

} /* namespace ai */
