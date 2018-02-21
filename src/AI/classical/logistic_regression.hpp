#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "../util/Tensor.hpp"
#include <fstream>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	class logistic_regression
	{
		public:
			logistic_regression();
			logistic_regression(const unsigned int input_size, const unsigned int output_size);
			logistic_regression(const std::string filepath);
			const Tensor_float& predict(const Tensor_float input);
			void fit(Tensor_float inputs, Tensor_float targets, const float starting_learningrate = 0.01, const unsigned int epochs = 20, const bool verbose=true);
			void test(Tensor_float inputs, Tensor_float targets);
			const float fit_single_sample(Tensor_float input, Tensor_float target, const float learningrate);
			const Tensor_float& get_output();
			void save(const std::string filepath);
			void load(std::ifstream& filestream);
			void save(std::ofstream& filestream);

		private:
			const float sigmoid(const float x);
			const float sigmoid_deriv(const float x);
			unsigned int _input_size, _output_size;
			Tensor_float _weights, _bias;
			Tensor_float _outputs;
			Tensor_float _errors;
	};

} /* namespace ai */

#endif /* end of include guard: LOGISTIC_REGRESSION_HPP */

