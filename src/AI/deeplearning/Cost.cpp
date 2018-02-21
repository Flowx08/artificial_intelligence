////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Cost.hpp"
#include <math.h>
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif

////////////////////////////////////////////////////////////
///	NAMESPACcE 
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	Cost::Cost()
	{
		_type = Cost::SquaredError;
	}

	////////////////////////////////////////////////////////////
	Cost::Cost(const CostType type)
	{
		_type = type;
	}

	#ifdef CUDA_BACKEND
	////////////////////////////////////////////////////////////
	float Cost::getErrorCUDA(TensorCUDA_float& prediction, TensorCUDA_float& target)
	{
		if (_gpu_errors.size() == 0) {
			_gpu_errors.setshape(target.size());
			_host_errors.setshape(target.size());
		}
		TensorCUDA_float_diff(target, prediction, _gpu_errors);
		_gpu_errors.copyToHost(_host_errors.pointer(), _host_errors.size());
		float error = 0;
		for (int i = 0; i < _host_errors.size(); i++)
			error += fabs(_host_errors[i]);
		return error;
		//TODO
	}

	////////////////////////////////////////////////////////////
	void Cost::getDeltaCUDA(TensorCUDA_float& prediction, TensorCUDA_float& target, TensorCUDA_float& errors)
	{
		switch (_type) {
			case Cost::SquaredError:
				TensorCUDA_float_diff(target, prediction, errors);
				break;
			
			case Cost::CrossEntropy:
				cuda::cost_crossentropy(prediction.pointer(), target.pointer(), errors.pointer(), errors.size());	
				break;
		}
	}

	#else
	
	////////////////////////////////////////////////////////////
	float Cost::getError(Tensor_float& prediction, Tensor_float& target)
	{
		float error = 0;
		switch (_type) {
			case Cost::SquaredError:
				for (int i = 0; i < (int)prediction.size(); i++)
					error += pow(target[i] - prediction[i], 2) / 2.f;
				break;
			
			case Cost::CrossEntropy:
				float epsilon = 1e-04;
				for (int i = 0; i < (int)prediction.size(); i++)
					error += target[i] * log(prediction[i] + epsilon) + (1 - target[i]) * log(1 - prediction[i] + epsilon);
				error = -error;
				break;
		}
		
		return error;
	}

	////////////////////////////////////////////////////////////
	void Cost::getDelta(Tensor_float& prediction, Tensor_float& target, Tensor_float& errors)
	{
		switch (_type) {
			case Cost::SquaredError:
				for (int i = 0; i < (int)errors.size(); i++)
					errors[i] = target[i] - prediction[i];
				break;
			
			case Cost::CrossEntropy:
				double denominator;
				double epsilon = 1e-4;
				for (int i = 0; i < (int)errors.size(); i++) {
					denominator = prediction[i] - prediction[i]*prediction[i];
					if (denominator < epsilon) denominator = epsilon;
					errors[i] = (target[i] - prediction[i]) / denominator;
				}
				break;
		}
	}

	#endif

} //namespace ai
