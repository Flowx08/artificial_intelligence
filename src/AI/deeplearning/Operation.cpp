///////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Operation.hpp"
#include "Variable.hpp"
#include "Linear.hpp"
#include "Sigmoid.hpp"
#include "Tanh.hpp"
#include "Relu.hpp"
#include "Softmax.hpp"
#include "Recurrent.hpp"
#include "Partial.hpp"
#include "Dropout.hpp"
#include "Convolution.hpp"
#include "Normalization.hpp"
#include "Addition.hpp"
#include "Concatenate.hpp"
#include "Maxpooling.hpp"
#include "Averagepooling.hpp"
#include "Selu.hpp"
#include "Autoencoder.hpp"
#include "ResidualBlock.hpp"
#include "CapsulesDense.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	Operation::~Operation() {}

	////////////////////////////////////////////////////////////
	void Operation::initialize(std::vector<Operation*> &inputs) {}
	
	////////////////////////////////////////////////////////////
	void Operation::save(ai::IOData& data) {}
	
	////////////////////////////////////////////////////////////
	void Operation::run(std::vector<Operation*>& inputs, const bool training) {}
	
	////////////////////////////////////////////////////////////
	void Operation::backprop(std::vector<Operation*>& input_errors) {}
	
	////////////////////////////////////////////////////////////
	void Operation::accumulate_deltas(std::vector<Operation*>& inputs) {}
	
	////////////////////////////////////////////////////////////
	void Operation::update_parameters(const float learningrate) {}
	
	////////////////////////////////////////////////////////////
	void Operation::print() {}
	
	////////////////////////////////////////////////////////////
	void Operation::reset_errors()
	{
		#ifdef CUDA_BACKEND
		if (_errors.size() > 0) TensorCUDA_float_fill(_errors, 0);
		#else
		for (int i = 0; i < _errors.size(); i++)
			_errors[i] = 0;
		#endif
	}
	
	////////////////////////////////////////////////////////////
	void Operation::reset_deltas(const double momentum)
	{
		/*
		#ifdef CUDA_BACKEND
		if (_deltas.size() > 0) TensorCUDA_float_scale(_deltas, momentum);
		#else
		for (int i = 0; i < _deltas.size(); i++)
			_deltas[i] *= momentum;
		#endif
		*/
	}
	
	////////////////////////////////////////////////////////////
	const Operation::Type Operation::get_type() const { return Operation::Unknown; }
	
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Operation::loadFromFile(ai::IOData& data)
	{
		ai::IOData* op_type = data.findNode("operation_type");
		ensure(op_type != NULL);
		ai::IOData* op = data.findNode("operation_data");
		ensure(op != NULL);
		int operation_type;
		op_type->get(operation_type);
		switch (operation_type) {
			case Unknown: return std::shared_ptr<Operation>(NULL);
			case Variable: return std::shared_ptr<Operation>(new ai::Variable(*op));
			case Linear: return std::shared_ptr<Operation>(new ai::Linear(*op));
			case Sigmoid: return std::shared_ptr<Operation>(new ai::Sigmoid(*op));
			case Relu: return std::shared_ptr<Operation>(new ai::Relu(*op));
			case Tanh: return std::shared_ptr<Operation>(new ai::Tanh(*op));
			case Softmax: return std::shared_ptr<Operation>(new ai::Softmax(*op));
			case Recurrent: return std::shared_ptr<Operation>(new ai::Recurrent(*op));
			case Partial: return std::shared_ptr<Operation>(new ai::Partial(*op));
			case Dropout: return std::shared_ptr<Operation>(new ai::Dropout(*op));
			case Convolution: return std::shared_ptr<Operation>(new ai::Convolution(*op));
			case Normalization: return std::shared_ptr<Operation>(new ai::Normalization(*op));
			case Addition: return std::shared_ptr<Operation>(new ai::Addition(*op));
			case Concatenate: return std::shared_ptr<Operation>(new ai::Concatenate(*op));
			case Maxpooling: return std::shared_ptr<Operation>(new ai::Maxpooling(*op));
			case Averagepooling: return std::shared_ptr<Operation>(new ai::Averagepooling(*op));
			case Selu: return std::shared_ptr<Operation>(new ai::Selu(*op));
			case Autoencoder: return std::shared_ptr<Operation>(new ai::Autoencoder(*op));
			case ResidualBlock: return std::shared_ptr<Operation>(new ai::ResidualBlock(*op));
			default: return std::shared_ptr<Operation>(NULL);
		}
	}
	
	////////////////////////////////////////////////////////////
	void Operation::saveToFile(std::shared_ptr<Operation>& operation, ai::IOData& data)
	{
		data.pushNode("operation_type", (int)operation.get()->get_type());
		data.pushNode("operation_data");
		ai::IOData* operation_data = data.findNode("operation_data");
		operation.get()->save(*operation_data);		
	}

} /* namespace ai */
