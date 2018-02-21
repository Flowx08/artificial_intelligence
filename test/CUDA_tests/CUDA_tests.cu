#include <stdio.h>
#include <stdlib.h>
#include "../src/AI/deeplearning/TensorCUDA.hpp"
#include "../src/AI/deeplearning/CUDA_backend.hpp"

int main()
{	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int inputs = 1000;
	int size = 1000;
	ai::TensorCUDA_float weights(size * inputs);
	ai::TensorCUDA_float deltas(size * inputs);
	weights.fill(1);
	ai::TensorCUDA_float bias(size);	
	ai::TensorCUDA_float errors(size);
	//errors.fill(1);
	errors.fill(0, 1);
	ai::TensorCUDA_float outerrors(inputs);	
	ai::TensorCUDA_float in(inputs);
	ai::Tensor in_host(inputs);
	for (int i = 0; i < inputs; i++)
		if (i % 2 == 0) in_host[i] = 0;
		else in_host[i] = 1;
	in.copyToDevice(in_host.pointer(), inputs);
	ai::TensorCUDA_float outputs(size);
	ai::TensorCUDA_float deviation(size);
	deviation.fill(0);
	ai::TensorCUDA_float normalized(size);
	normalized.fill(0);
	ai::TensorCUDA_float params(5);
	params.fill(0);
	ai::Tensor params_host(5);
	params_host[0] = 0;
	params_host[1] = 1;
	params_host[2] = 0;
	params_host[3] = 0;
	params_host[4] = 0;
	params.copyToDevice(&params_host[0], 5);

	int _filter_count = 32;
	int _filter_size = 4;
	int _input_count = 3;
	int _input_width = 28;
	int _input_height = 28;
	int _stride = 2;
	int _output_width = (_input_width - _filter_size + 1.0) / _stride;
	int _output_height = (_input_height - _filter_size + 1.0) / _stride;
	int _output_size = _output_width * _output_height;
	int _size = _output_size * _filter_count;
	
	ai::TensorCUDA_float _outputs;
	_outputs.setshape(_output_width, _output_height, _filter_count);
	_outputs.fill(0);
	ai::TensorCUDA_float _inputs;
	_inputs.setshape(_input_width, _input_height, _input_count);
	_inputs.fill(1);
	ai::TensorCUDA_float _out_errors;
	_out_errors.setshape(_input_width, _input_height, _input_count);
	_out_errors.fill(0);
	ai::TensorCUDA_float _errors;
	_errors.setshape(_size);
	_errors.fill(1);
	ai::TensorCUDA_float _deltas;
	_deltas.setshape((_filter_size * _filter_size  * _input_count + 1) * _filter_count); //+1 for bias
	_deltas.fill(1);

	//Initialize weights
	ai::TensorCUDA_float _weights;
	_weights.setshape(_filter_size * _filter_size, _input_count, _filter_count);
	//_weights.fill(1);
	_weights.fill(0.0, sqrt(6.0 / ((_filter_size * _filter_size) * _input_count + 1)));
	ai::TensorCUDA_float _bias;
	_bias.setshape(_filter_count);
	//_bias.fill(2);
	_bias.fill(0.0, sqrt(6.0 / ((_filter_size * _filter_size) * _input_count + 1)));

	printf("outputs: %d\n", _output_size * _filter_count);

	int _convmap[_output_width * _output_height * _filter_size * _filter_size];
	for (int x = 0; x < _output_width; x++) {
		for (int y = 0; y < _output_height; y++) {
			const float input_x = x * _stride;
			const float input_y = y * _stride;

			for (int kx = 0; kx < _filter_size; kx++) {
				for (int ky = 0; ky < _filter_size; ky++) {
					_convmap[(y * _output_width + x) * _filter_size * _filter_size + ky * _filter_size + kx] 
						= (input_y + ky) * _input_width + input_x + kx; 
				}
			}
		}
	}
	
	int _in_out_map[_input_width * _input_height * _filter_size * _filter_size];
	for (int i = 0; i < _input_width * _input_height * _filter_size * _filter_size; i++) _in_out_map[i] = -1;
	int _in_weight_map[_input_width * _input_height * _filter_size * _filter_size];
	for (int i = 0; i < _input_width * _input_height * _filter_size * _filter_size; i++) _in_weight_map[i] = -1;
	
	for (int x = 0; x < _output_width; x++) {
		for (int y = 0; y < _output_height; y++) {
			for (int w = 0; w < _filter_size * _filter_size; w++) {
				_in_out_map[_convmap[(y * _output_width + x) * _filter_size * _filter_size + w] * _filter_size * _filter_size + w] = y * _output_width + x; 
				_in_weight_map[_convmap[(y * _output_width + x) * _filter_size * _filter_size + w] * _filter_size * _filter_size + w] = w; 
			}
		}
	}

	ai::TensorCUDA_int _convmap_gpu(_filter_size * _filter_size, _output_width * _output_height);
	_convmap_gpu.copyToDevice(&_convmap[0], _convmap_gpu.size());
	ai::TensorCUDA_int _in_out_map_gpu(_input_width * _input_height * _filter_size * _filter_size);
	_in_out_map_gpu.copyToDevice(&_in_out_map[0], _in_out_map_gpu.size());
	ai::TensorCUDA_int _in_weight_map_gpu(_input_width * _input_height * _filter_size * _filter_size);
	_in_weight_map_gpu.copyToDevice(&_in_weight_map[0], _in_weight_map_gpu.size());
	
	cudaEventRecord(start); //start
		
	for (int i = 0; i < 100; i++) { //amplify
		//ai::cuda::conv_foreward(_weights.pointer(), _bias.pointer(), _inputs.pointer(), _outputs.pointer(),
		//	_convmap_gpu.pointer(), _input_width, _input_height, _input_count, _stride, _output_width, _output_height,
		//	_filter_count, _filter_size);
		//ai::cuda::conv_accumulate_deltas(_deltas.pointer(), _errors.pointer(), _inputs.pointer(),
		//	outputs.pointer(), _convmap_gpu.pointer(), _input_count, _input_width, _input_height,
		//	_output_size, _filter_size, _filter_count);
		//ai::cuda::conv_backward(_weights.pointer(), _out_errors.pointer(),
		//	_errors.pointer(), _in_weight_map_gpu.pointer(), _in_out_map_gpu.pointer(), _input_count,
		//	_output_size, _input_width, _input_height, _filter_size, _filter_count);
		//ai::cuda::batchnorm_foreward(in.pointer(), deviation.pointer(), normalized.pointer(), outputs.pointer(),
		//	&params.pointer()[0], &params.pointer()[1], &params.pointer()[2], 0.0001, size);
		//ai::cuda::batchnorm_backward(errors.pointer(), outerrors.pointer(), deviation.pointer(),
		//	&params.pointer()[0], &params.pointer()[1], &params.pointer()[2], 0.0001, size); 
		//ai::cuda::conv_update_parameters(_weights.pointer(), _bias.pointer(), _deltas.pointer(), _filter_size, _input_count, _filter_count, 0.1);
		
		//ai::cuda::linear_foreward(weights.pointer(), bias.pointer(), in.pointer(), outputs.pointer(), inputs, size);
		//ai::cuda::linear_accumulate_deltas(deltas.pointer(), in.pointer(), errors.pointer(), inputs, size);
		//ai::cuda::linear_backward(weights.pointer(), outerrors.pointer(), errors.pointer(), inputs, size);
		//ai::cuda::cost_crossentropy(outputs.pointer(), errors.pointer(), outerrors.pointer(), _size);
		//ai::cuda::sigmoid_foreward(in.pointer(), outputs.pointer(), size);
	}

	cudaEventRecord(stop); //stop
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f ms\n", milliseconds);

	/*
	ai::Tensor o(_out_errors.size());
	_out_errors.copyToHost(o.pointer(), o.size());
	for (int k = 0; k < 3; k++) {
		for	(int i = 0; i < _input_width; i++)
			printf("%f\n", o[k * _input_width + i]);
		printf("\n");
	}
	*/

	ai::Tensor out(size);
	/*
	outputs.copyToHost(out.pointer(), size);
	for (int i = 0; i < 40; i++) {
		printf("%f\n", out[i]);
	}
	*/
	printf("\n");
	outerrors.copyToHost(out.pointer(), size);
	for	(int i = 0; i < 40; i++) {
		printf("%f\n", out[i]);
	}
	
	return 0;
}

//////////////////////////////////////////////////
/// TEST
/// 2017-01-06
/// questo codice dimostra che knl_conv_update_weights
/// funziona correttamente
/////////////////////////////////////////////////
/*
int main()
{	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int inputs = 1000;
	int size = 100;
	ai::TensorCUDA_float weights(size * inputs);
	ai::TensorCUDA_float deltas(size * inputs);
	weights.fill(1);
	ai::TensorCUDA_float bias(size);	
	ai::TensorCUDA_float errors(size);
	errors.fill(1);
	ai::TensorCUDA_float outerrors(inputs);	
	ai::TensorCUDA_float in(inputs);
	in.fill(1);
	ai::TensorCUDA_float outputs(size);

	int _filter_count = 2;
	int _filter_size = 3;
	int _input_count = 2;
	int _input_width = 32;
	int _input_height = 32;
	int _stride = 1;
	int _output_width = (_input_width - _filter_size + 1.0) / _stride;
	int _output_height = (_input_height - _filter_size + 1.0) / _stride;
	int _output_size = _output_width * _output_height;
	int _size = _output_size * _filter_count;
	
	ai::TensorCUDA_float _outputs;
	_outputs.setshape(_output_width, _output_height, _filter_count);
	_outputs.fill(0);
	ai::TensorCUDA_float _inputs;
	_inputs.setshape(_input_width, _input_height, _input_count);
	_inputs.fill(1);
	ai::TensorCUDA_float _errors;
	_errors.setshape(_size);
	_errors.fill(1);
	ai::TensorCUDA_float _deltas;
	_deltas.setshape((_filter_size * _filter_size  * _input_count + 1) * _filter_count); //+1 for bias
	_deltas.fill(1);

	//Initialize weights
	ai::TensorCUDA_float _weights;
	_weights.setshape(_filter_size * _filter_size, _input_count, _filter_count);
	_weights.fill(1);
	//_weights.fill(0.0, sqrt(6.0 / ((_filter_size * _filter_size) * _input_count + 1)));
	ai::TensorCUDA_float _bias;
	_bias.setshape(_filter_count);
	_bias.fill(3);
	//_bias.fill(0.0, sqrt(6.0 / ((_filter_size * _filter_size) * _input_count + 1)));

	printf("outputs: %d\n", _output_size * _filter_count);

	int _convmap[_output_width * _output_height * _filter_size * _filter_size];
	for (int x = 0; x < _output_width; x++) {
		for (int y = 0; y < _output_height; y++) {
			const float input_x = x * _stride;
			const float input_y = y * _stride;

			for (int kx = 0; kx < _filter_size; kx++) {
				for (int ky = 0; ky < _filter_size; ky++) {
					_convmap[(y * _output_width + x) * _filter_size * _filter_size + ky * _filter_size + kx] 
						= (input_y + ky) * _input_width + input_x + kx; 
				}
			}
		}
	}
	ai::TensorCUDA_int _convmap_gpu(_filter_size * _filter_size, _output_width * _output_height);
	_convmap_gpu.copyToDevice(&_convmap[0], _convmap_gpu.size());
	
	cudaEventRecord(start); //start
		
	for (int i = 0; i < 100; i++) { //amplify
		//ai::cuda::conv_foreward(_weights.pointer(), _bias.pointer(), _inputs.pointer(), _outputs.pointer(),
		//_convmap_gpu.pointer(), _input_width, _input_height, _input_count, _stride, _output_width, _output_height,
		//_filter_count, _filter_size);
		//ai::cuda::conv_accumulate_deltas(_deltas.pointer(), _errors.pointer(), _inputs.pointer(),
		//	outputs.pointer(), _convmap_gpu.pointer(), _input_count, _input_width, _input_height,
		//	_output_size, _filter_size, _filter_count);
		ai::cuda::conv_update_weights(_weights.pointer(), _bias.pointer(), _deltas.pointer(), _filter_size, _input_count, _filter_count, 0.1);
		//ai::cuda::linear_foreward(weights.pointer(), bias.pointer(), in.pointer(), outputs.pointer(), inputs, size);
		//ai::cuda::linear_accumulate_deltas(deltas.pointer(), in.pointer(), errors.pointer(), inputs, size);
		//ai::cuda::linear_backward(weights.pointer(), outerrors.pointer(), errors.pointer(), inputs, size);
		//ai::cuda::cost_crossentropy(outputs.pointer(), errors.pointer(), outerrors.pointer(), _size);
		//ai::cuda::sigmoid_foreward(in.pointer(), outputs.pointer(), size);
	}

	cudaEventRecord(stop); //stop
	
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f ms\n", milliseconds);

	ai::Tensor o(_weights.size());
	ai::Tensor o2(_bias.size());
	_weights.copyToHost(o.pointer(), o.size());
	_bias.copyToHost(o2.pointer(), o2.size());
	for	(int i = 0; i < 10; i++)
		printf("%f\n", o[i]);
	for	(int i = 0; i < o2.size(); i++)
		printf("%f\n", o2[i]);
	
	return 0;
}
*/

//////////////////////////////////////////////////
/// TEST
/// 2017-01-06
/// questo codice dimostra che knl_conv_accumulate_deltas
/// funziona correttamente
/////////////////////////////////////////////////
/*
void cpu_conv_accumulate_deltas(float* data, float* _errors, float* _deltas, int* _convmap,
	int _output_size, int _input_count, int _input_width, int _input_height, int _filter_size,
	int _filter_count)
{
	for (int f = 0; f < _filter_count; f++) { //Each filter

		//Shortcut for this filter output
		const float *upcomming_errors = &_errors[f * _output_size];
		float *t_filter_deltas = &_deltas[f * _input_count * _filter_size * _filter_size];

		//For each output
		for (int o = 0; o < _output_size; o++) {

			//Jump computation
			if (upcomming_errors[o] == 0) continue;

			//Compute each input group
			for (int k = 0; k < _input_count; k++) {

				float *filter_deltas = &t_filter_deltas[k * _filter_size * _filter_size];

				//Shortcut for this input group
				const float *in = &data[_input_width * _input_height * k];

				//For each weight
				for (int w = 0; w < _filter_size * _filter_size; w++)
					filter_deltas[w] += in[_convmap[o * (_filter_size * _filter_size) + w]] * upcomming_errors[o];

			} // for each output

			//Bias
			_deltas[_input_count * _filter_size * _filter_size * _filter_count + f] += upcomming_errors[o];

		} // for each input group
	} // for each filter
}

int main()
{	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int inputs = 1000;
	int size = 100;
	ai::TensorCUDA_float weights(size * inputs);
	ai::TensorCUDA_float deltas(size * inputs);
	weights.fill(1);
	ai::TensorCUDA_float bias(size);	
	ai::TensorCUDA_float errors(size);
	errors.fill(1);
	ai::TensorCUDA_float outerrors(inputs);	
	ai::TensorCUDA_float in(inputs);
	in.fill(1);
	ai::TensorCUDA_float outputs(size);

	int _filter_count = 2;
	int _filter_size = 3;
	int _input_count = 2;
	int _input_width = 32;
	int _input_height = 32;
	int _stride = 1;
	int _output_width = (_input_width - _filter_size + 1.0) / _stride;
	int _output_height = (_input_height - _filter_size + 1.0) / _stride;
	int _output_size = _output_width * _output_height;
	int _size = _output_size * _filter_count;
	
	ai::TensorCUDA_float _outputs;
	_outputs.setshape(_output_width, _output_height, _filter_count);
	_outputs.fill(0);
	ai::TensorCUDA_float _inputs;
	_inputs.setshape(_input_width, _input_height, _input_count);
	float* _cpu_input = (float*)malloc(sizeof(float) * _input_width * _input_height * _input_count);
	for (int i = 0; i < _input_width * _input_height * _input_count; i++) _cpu_input[i] = rand() % 3;
	_inputs.copyToDevice(_cpu_input, _inputs.size());
	ai::TensorCUDA_float _errors;
	_errors.setshape(_size);
	_errors.fill(1);
	ai::TensorCUDA_float _deltas;
	_deltas.setshape((_filter_size * _filter_size  * _input_count + 1) * _filter_count); //+1 for bias
	_deltas.fill(0);

	//Initialize weights
	ai::TensorCUDA_float _weights;
	_weights.setshape(_filter_size * _filter_size, _input_count, _filter_count);
	_weights.fill(1);
	//_weights.fill(0.0, sqrt(6.0 / ((_filter_size * _filter_size) * _input_count + 1)));
	ai::TensorCUDA_float _bias;
	_bias.setshape(_filter_count);
	_bias.fill(3);
	//_bias.fill(0.0, sqrt(6.0 / ((_filter_size * _filter_size) * _input_count + 1)));

	printf("outputs: %d\n", _output_size * _filter_count);

	int _convmap[_output_width * _output_height * _filter_size * _filter_size];
	for (int x = 0; x < _output_width; x++) {
		for (int y = 0; y < _output_height; y++) {
			const float input_x = x * _stride;
			const float input_y = y * _stride;

			for (int kx = 0; kx < _filter_size; kx++) {
				for (int ky = 0; ky < _filter_size; ky++) {
					_convmap[(y * _output_width + x) * _filter_size * _filter_size + ky * _filter_size + kx] 
						= (input_y + ky) * _input_width + input_x + kx; 
				}
			}
		}
	}
	ai::TensorCUDA_int _convmap_gpu(_filter_size * _filter_size, _output_width * _output_height);
	_convmap_gpu.copyToDevice(&_convmap[0], _convmap_gpu.size());
	
	cudaEventRecord(start); //start
		
	for (int i = 0; i < 100; i++) { //amplify
		//ai::cuda::conv_foreward(_weights.pointer(), _bias.pointer(), _inputs.pointer(), _outputs.pointer(),
		//_convmap_gpu.pointer(), _input_width, _input_height, _input_count, _stride, _output_width, _output_height,
		//_filter_count, _filter_size);
		ai::cuda::conv_accumulate_deltas(_deltas.pointer(), _errors.pointer(), _inputs.pointer(),
			outputs.pointer(), _convmap_gpu.pointer(), _input_count, _input_width, _input_height,
			_output_size, _filter_size, _filter_count);
		//ai::cuda::conv_update_weights(_weights.pointer(), _bias.pointer(), _deltas.pointer(), _filter_size, _input_count, _filter_count, 0.1);
		//ai::cuda::linear_foreward(weights.pointer(), bias.pointer(), in.pointer(), outputs.pointer(), inputs, size);
		//ai::cuda::linear_accumulate_deltas(deltas.pointer(), in.pointer(), errors.pointer(), inputs, size);
		//ai::cuda::linear_backward(weights.pointer(), outerrors.pointer(), errors.pointer(), inputs, size);
		//ai::cuda::cost_crossentropy(outputs.pointer(), errors.pointer(), outerrors.pointer(), _size);
		//ai::cuda::sigmoid_foreward(in.pointer(), outputs.pointer(), size);
	}

	cudaEventRecord(stop); //stop
	
	//cpu convolution accumualte deltas test
	float* _cpu_errors = (float*)malloc(sizeof(float) * _output_width * _output_height * _filter_count);
	for (int i = 0; i < _output_width * _output_height * _filter_count; i++) _cpu_errors[i] = 1;
	float* _cpu_deltas = (float*)malloc(sizeof(float) * (_filter_size * _filter_size * _input_count + 1) * _filter_count);
	for (int i = 0; i < (_filter_size * _filter_size * _input_count + 1) * _filter_count; i++) _cpu_deltas[i] = 0;
	for (int i = 0; i < 100; i++) cpu_conv_accumulate_deltas(_cpu_input, _cpu_errors, _cpu_deltas, _convmap, _output_size, _input_count, _input_width, _input_height, _filter_size, _filter_count);
	for (int i = 0; i < 20; i++) printf("d %f\n", _cpu_deltas[i]);
	printf("\n");
	for (int i = _deltas.size()-1; i > _deltas.size()-10; i--) printf("d %f\n", _cpu_deltas[i]);
	free(_cpu_input);
	free(_cpu_errors);
	free(_cpu_deltas);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f ms\n", milliseconds);

	ai::Tensor d(_deltas.size());
	_deltas.copyToHost(d.pointer(), d.size());
	for (int i = 0; i < 20; i++)
		printf("%f\n", d[i]);
	printf("\n");
	for (int i = _deltas.size()-1; i > _deltas.size()-10; i--)
		printf("%f\n", d[i]);

	return 0;
}
*/
