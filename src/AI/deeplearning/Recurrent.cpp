////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Recurrent.hpp"
#include <math.h>
#include "../util/ensure.hpp"
#include "Linear.hpp"
#include "Tanh.hpp"
#include "Cost.hpp"
#ifdef CUDA_BACKEND
#include "CUDA_backend.hpp"
#endif



////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai 
{

	////////////////////////////////////////////////////////////
	///	UTIL
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	///	RECURRENT
	////////////////////////////////////////////////////////////
	
	////////////////////////////////////////////////////////////
	std::shared_ptr<Operation> Recurrent::make(const int size, int btt_steps)
	{
		return std::shared_ptr<Operation>(new Recurrent(size, btt_steps)); 
	}

	////////////////////////////////////////////////////////////
	Recurrent::Recurrent(const int size, int btt_steps)
	{
		_size = size;
		_btt_steps = btt_steps;
	}
	
	////////////////////////////////////////////////////////////
	Recurrent::Recurrent(ai::IOData& data)
	{
		ai::IOData* size = data.findNode("size");
		ensure(size != NULL);
		ai::IOData* input_size = data.findNode("input_size");
		ensure(input_size != NULL);
		ai::IOData* btt_steps = data.findNode("btt_steps");
		ensure(btt_steps != NULL);
		size->get(_size);
		input_size->get(_input_size);
		btt_steps->get(_btt_steps);
		
		//Dense connections
		ai::IOData* node_x = data.findNode("node_x");
		ensure(node_x != NULL);
		ai::IOData* node_rec = data.findNode("node_rec");
		ensure(node_rec != NULL);
		ai::IOData* node_out = data.findNode("node_out");
		ensure(node_out != NULL);
		_x = ai::Linear(*node_x);
		_rec = ai::Linear(*node_rec);
		_out = ai::Linear(*node_out);
		
		#ifdef CUDA_BACKEND

		//Memory parameters
		_mem_outputs.setshape(_size, _btt_steps);
		_mem_outputs.fill(0);
		_mem_tmp_errors.setshape(_size, _btt_steps);
		_mem_tmp_errors.fill(0);
		_mem_inputs.setshape(_input_size, _btt_steps);
		_mem_inputs.fill(0);
		
		#else
		
		//Memory parameters
		_mem_outputs.setshape(_size, _btt_steps);
		_mem_outputs.fill(0);
		_mem_tmp_errors.setshape(_size, _btt_steps);
		_mem_tmp_errors.fill(0);
		_mem_inputs.setshape(_input_size, _btt_steps);
		_mem_inputs.fill(0);
		
		#endif
		
		_errors.setshape(_size);
		_errors.fill(0);
		_outputs.setshape(_size);
		_outputs.fill(0);
		
		_btt_pos = 0;
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::save(ai::IOData& data)
	{
		data.pushNode("size", _size);
		data.pushNode("input_size", _input_size);
		data.pushNode("btt_steps", _btt_steps);
		data.pushNode("node_x");
		data.pushNode("node_rec");
		data.pushNode("node_out");
		_x.save(*data.findNode("node_x"));
		_rec.save(*data.findNode("node_rec"));
		_out.save(*data.findNode("node_out"));
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::initialize(std::vector<Operation*> &inputs)
	{
		//Calculate input size
		_input_size = 0;
		for (int i = 0; i < (int)inputs.size(); i++)
			_input_size += inputs[i]->_outputs.size();

		initialize(_input_size);
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::initialize(const int input_size)
	{
		_input_size = input_size;	

		//Dense connections
		_x = ai::Linear(_size);
		_x.initialize(_input_size);
		_rec = ai::Linear(_size);
		_rec.initialize(_size);
		_out = ai::Linear(_size);
		_out.initialize(_size);
		
		#ifdef CUDA_BACKEND

		//Memory parameters
		_mem_outputs.setshape(_size, _btt_steps);
		_mem_outputs.fill(0);
		_mem_tmp_errors.setshape(_size, _btt_steps);
		_mem_tmp_errors.fill(0);
		_mem_inputs.setshape(_input_size, _btt_steps);
		_mem_inputs.fill(0);
		
		#else

		//Memory parameters
		_mem_outputs.setshape(_size, _btt_steps);
		_mem_outputs.fill(0);
		_mem_tmp_errors.setshape(_size, _btt_steps);
		_mem_tmp_errors.fill(0);
		_mem_inputs.setshape(_input_size, _btt_steps);
		_mem_inputs.fill(0);
		
		#endif
		
		_errors.setshape(_size);
		_errors.fill(0);
		_outputs.setshape(_size);
		_outputs.fill(0);

		_btt_pos = 0;

	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::run(std::vector<Operation*> &inputs, const bool training) 
	{
		ensure(inputs.size() == 1);
		run(inputs[0]->_outputs, training);
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::backprop(std::vector<Operation*> &inputs) 
	{
		ensure(inputs.size() == 1);
		backprop(inputs[0]->_errors);
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::accumulate_deltas(std::vector<Operation*> &inputs)
	{
		ensure(inputs.size() == 1);
		accumulate_deltas(inputs[0]->_outputs);
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::update_parameters(const float learningrate)
	{
		_x.update_parameters(learningrate);
		_rec.update_parameters(learningrate);
		_out.update_parameters(learningrate);
		_x.reset_deltas(0);
		_rec.reset_deltas(0);
		_out.reset_deltas(0);
	}
		
	#ifdef CUDA_BACKEND
	
	////////////////////////////////////////////////////////////
	void Recurrent::run(const TensorCUDA_float input, bool accumulate)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void Recurrent::backprop(TensorCUDA_float out_errors)
	{
		//TODO
	}

	////////////////////////////////////////////////////////////
	void Recurrent::accumulate_deltas(const TensorCUDA_float input)
	{
		//TODO
	}

	#else
	
	////////////////////////////////////////////////////////////
	void Recurrent::run(const Tensor_float input, bool accumulate)
	{
		//update btt pos
		int btt_oldpos = _btt_pos;
		_btt_pos = (_btt_pos + 1) % _btt_steps;
		
		//Store input
		_mem_inputs.ptr(0, _btt_pos).set(input);
		
		//Calculate state output
		_x._outputs.point(_mem_outputs.ptr(0, _btt_pos));
		_x.run(_mem_inputs.ptr(0, _btt_pos), false);
		_rec._outputs.point(_mem_outputs.ptr(0, _btt_pos));
		_rec.run(_mem_outputs.ptr(0, btt_oldpos), true);
		Tanh::foreward(_mem_outputs.ptr(0, _btt_pos), _mem_outputs.ptr(0, _btt_pos));
		_out._outputs.point(_outputs);
		_out.run(_mem_outputs.ptr(0, _btt_pos), false);
	}

	////////////////////////////////////////////////////////////
	void Recurrent::backprop(Tensor_float out_errors)
	{
		for (int k = 0; k < _btt_steps; k++)
			_mem_tmp_errors.ptr(0, k).fill(0);
		
		//Propagate errors backward to memory state
		_out._errors.point(_errors);
		_out.backprop(_mem_tmp_errors.ptr(0, 0));
		Tanh::backward(_mem_tmp_errors.ptr(0, 0), _mem_outputs.ptr(0, _btt_pos), _mem_tmp_errors.ptr(0, 0));

		//Unroll and backpropagate errors
		for (int k = 0; k < _btt_steps-1; k++) {
			int btt_id = _btt_pos - k - 1;
			if (btt_id < 0) btt_id += _btt_steps;
			_x._errors.point(_mem_tmp_errors.ptr(0, k));
			_x.backprop(out_errors);
			_rec._errors.point(_mem_tmp_errors.ptr(0, k));
			_rec.backprop(_mem_tmp_errors.ptr(0, k + 1));
			Tanh::backward(_mem_tmp_errors.ptr(0, k + 1), _mem_outputs.ptr(0, btt_id), _mem_tmp_errors.ptr(0, k + 1));
		}
		_x._errors.point(_mem_tmp_errors.ptr(0, _btt_steps-1));
		_x.backprop(out_errors);
	}

	////////////////////////////////////////////////////////////
	void Recurrent::accumulate_deltas(const Tensor_float input)
	{
		//Accumulate output deltas
		_out._errors.point(_errors);
		_out.accumulate_deltas(_mem_outputs.ptr(0, _btt_pos));
		
		//Accumulate memory deltas
		for (int i = 0; i < _btt_steps-1; i++) {
			int btt_id = (_btt_pos - i) % _btt_steps;
			if (btt_id < 0) btt_id += _btt_steps;
			int btt_id2 = (_btt_pos - i - 1) % _btt_steps;
			if (btt_id2 < 0) btt_id2 += _btt_steps;
			_x._errors.point(_mem_tmp_errors.ptr(0, i));
			_x.accumulate_deltas(_mem_inputs.ptr(0, btt_id));
			_rec._errors.point(_mem_tmp_errors.ptr(0, i));
			_rec.accumulate_deltas(_mem_outputs.ptr(0, btt_id2));
		}
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::gradient_check()
	{
		//params
		const int size = 20;
		const int input_size = 10;
		const float epsilon = 10e-4;
		
		//test node
		Recurrent node(size);
		node.initialize(input_size);
		
		//random input
        Tensor_float input(input_size);
        input.fill(0.5, 0.5);
		
		//out errors
        Tensor_float out_errors(input_size);
		out_errors.fill(0);

		//random target
		Tensor_float target(size);
        target.fill(0.5, 0.5);

		//Cost function
		Cost costfun(Cost::SquaredError);

		//computed numerical gradients
		Tensor_float& node_weights = node._x._weights;
		Tensor_float numgrad(node_weights.width(), node_weights.height(), node_weights.depth());

		//For each parameter
        for (int i = 0; i < node_weights.size(); i++) {
            float init_param = node_weights[i];
            node_weights[i] = init_param + epsilon;
			node.reset_hidden_state();
			node.run(input, false);
            float lossPlus = costfun.getError(node._outputs, target);

            node_weights[i] = init_param - epsilon;
			node.reset_hidden_state();
			node.run(input, false);
            float lossMinus = costfun.getError(node._outputs, target);
			
			numgrad[i] = (lossPlus - lossMinus) / (2.f * epsilon);

			node_weights[i] = init_param;
       }
		
		//compute gradients with backprop code
		node._x.reset_deltas(0);
		node.reset_hidden_state();
		node.run(input, false);
		costfun.getDelta(node._outputs, target, node._errors);
		node.backprop(out_errors);
		node.accumulate_deltas(input);
		
		int d = 0;
		float max = 0;
		Tensor_float distances(numgrad.width(), numgrad.height(), numgrad.depth());
		for (int i = 0; i < node_weights.width(); i++) {
			for (int k = 0; k <	node_weights.height(); k++) {
				distances.at(k, i) = fabs(numgrad.at(k, i) + node._x._deltas[d]);
				if (distances.at(k, i) > max)
					max = distances.at(k, i); 
				d++;
			}
			d++; //bias
		}

		const float tollerance = 3 * 1e-3;
		if (max > tollerance) printf("Gradient differs by %f\n", max);
		else printf("Gradient looks good\n");
			
		printf("%s\n", distances.tostring().c_str());
		
	}
	
	#endif

	////////////////////////////////////////////////////////////
	const Operation::Type Recurrent::get_type() const
	{
		return Operation::Recurrent;
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::reset_hidden_state()
	{
		_mem_inputs.fill(0);
		_mem_outputs.fill(0);
	}
	
	////////////////////////////////////////////////////////////
	void Recurrent::print()
	{
		printf("Type: Recurrent, Size: %d, Input_Size: %d, Weights: %d", _size, _input_size, _size * (_input_size + 1));
	}

} /* namespace ai */
