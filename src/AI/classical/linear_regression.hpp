#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "../util/Tensor.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class linear_regression
	{
		public:
			linear_regression(const unsigned int input_size, const unsigned int output_size);
			linear_regression(const std::string filepath);
			const Tensor_float& predict(const Tensor_float input);
			void fit(Tensor_float inputs, const Tensor_float targets, const float starting_learningrate = 0.01,
				const unsigned int epochs = 20, const bool verbose=true);
			const Tensor_float& get_output();
			void save(const std::string filepath);

		private:
			unsigned int _input_size, _output_size;
			Tensor_float _weights, _bias;
			Tensor_float _outputs;
			Tensor_float _errors;
	};

} /* namespace ai */

#endif /* end of include guard: LINEAR_REGRESSION_HPP */

