////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "CUDA_backend.hpp"
#include "../util/Macros.hpp"
#include "../util/ensure.hpp"
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <sstream>
#include <iostream>

////////////////////////////////////////////////////////////
///	NAMESPACE UTIL MACROS
////////////////////////////////////////////////////////////
#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	///	NAMESPACE CUDNN
	////////////////////////////////////////////////////////////
	namespace cudnn
	{
		////////////////////////////////////////////////////////////
		///	FRAMEWORK INITIALIZATION
		////////////////////////////////////////////////////////////
		static cudnnHandle_t cudnnHandle = NULL;
		static cublasHandle_t cublasHandle = NULL;
		static int init_id = init();
			
		////////////////////////////////////////////////////////////
		int init()
		{
			checkCUDNN(cudnnCreate(&cudnnHandle));
			cublasCreate(&cublasHandle);
			printf("CUDA backend initialized!\n");
			return 0;
		}
		
		////////////////////////////////////////////////////////////
		void destroy()
		{
			checkCUDNN(cudnnDestroy(cudnnHandle));
			cublasDestroy(cublasHandle);
		}
			
		////////////////////////////////////////////////////////////
		///	TENSOR DESCRIPTION
		////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////
		TensorDescription::TensorDescription()
		{
			_tensor_description = NULL;
		}
		
		////////////////////////////////////////////////////////////
		TensorDescription::TensorDescription(const int width, const int height, const int depth, const int batch_size, const DataType type)
		{
			create(width, height, depth, batch_size, type);
		}
		
		////////////////////////////////////////////////////////////
		void TensorDescription::create(const int width, const int height, const int depth,
				const int batch_size, const DataType type)
		{
			clear();

			_tensor_description = operator new(sizeof(cudnnTensorDescriptor_t));
			checkCUDNN(cudnnCreateTensorDescriptor((cudnnTensorDescriptor_t*)_tensor_description));
			
			cudnnDataType_t datatype;
			switch (type)
			{
				case DATA_FLOAT:
					datatype = CUDNN_DATA_FLOAT;
					break;
				
				case DATA_DOUBLE:
					datatype = CUDNN_DATA_DOUBLE;
					break;

				case DATA_HALF:
					datatype = CUDNN_DATA_HALF;
					break;
				
				case DATA_INT8:
					datatype = CUDNN_DATA_INT8;
					break;
				
				case DATA_INT32:
					datatype = CUDNN_DATA_INT32;
					break;
			};

			checkCUDNN(cudnnSetTensor4dDescriptor(*(cudnnTensorDescriptor_t*)_tensor_description, CUDNN_TENSOR_NCHW,
				datatype, batch_size, depth, height, width));

		}
		
		////////////////////////////////////////////////////////////
		TensorDescription::~TensorDescription()
		{
			clear();
		}
		
		////////////////////////////////////////////////////////////
		void TensorDescription::clear()
		{
			if (_tensor_description != NULL) {
				checkCUDNN(cudnnDestroyTensorDescriptor(*(cudnnTensorDescriptor_t*)_tensor_description));
				operator delete(_tensor_description);
			}
		}
		
		////////////////////////////////////////////////////////////
		void* TensorDescription::get()
		{
			return _tensor_description;
		}
			
		////////////////////////////////////////////////////////////
		///	ACTIVATION FUNCTION
		////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////
		Activation::Activation()
		{
			_activation_description = NULL;
		}

		////////////////////////////////////////////////////////////
		Activation::Activation(const int size, const int batch_size, const ActivationType type)
		{
			create(size, batch_size, type);
		}
		
		////////////////////////////////////////////////////////////
		void Activation::create(const int size, const int batch_size, const ActivationType type)
		{
			clear();

			_size_description.create(size, 1, 1, batch_size, DATA_FLOAT);
			_activation_description = operator new(sizeof(cudnnActivationDescriptor_t));
				
			cudnnActivationMode_t activationtype;
			switch (type)
			{
				case ACTIVATION_SIGMOID:
					activationtype = CUDNN_ACTIVATION_SIGMOID;
					break;
				
				case ACTIVATION_RELU:
					activationtype = CUDNN_ACTIVATION_RELU;
					break;

				case ACTIVATION_TANH:
					activationtype = CUDNN_ACTIVATION_TANH;
					break;
				
				case ACTIVATION_CLIPPED_RELU:
					activationtype = CUDNN_ACTIVATION_CLIPPED_RELU;
					break;
				
				case ACTIVATION_ELU:
					activationtype = CUDNN_ACTIVATION_ELU;
					break;
			};

			checkCUDNN(cudnnCreateActivationDescriptor((cudnnActivationDescriptor_t*)_activation_description));
			checkCUDNN(cudnnSetActivationDescriptor(*(cudnnActivationDescriptor_t*)_activation_description, activationtype, CUDNN_PROPAGATE_NAN, 0.0));
		}

		////////////////////////////////////////////////////////////
		Activation::~Activation()
		{
			clear();
		}

		////////////////////////////////////////////////////////////
		void Activation::foreward(void* input, void* output)
		{
			const float alpha = 1.0;
			const float beta = 0.0;
			checkCUDNN(cudnnActivationForward(cudnnHandle, *(cudnnActivationDescriptor_t*)_activation_description,
				&alpha, *(cudnnTensorDescriptor_t*)_size_description.get(), input, &beta,
				*(cudnnTensorDescriptor_t*)_size_description.get(), output));
		}

		////////////////////////////////////////////////////////////
		void Activation::backward(void* input, void* output, void* errors, void* output_errors)
		{
			const float alpha = 1.0;
			const float beta = 0.0;
			checkCUDNN(cudnnActivationBackward(cudnnHandle, *(cudnnActivationDescriptor_t*)_activation_description,
						&alpha, *(cudnnTensorDescriptor_t*)_size_description.get(), output,
						*(cudnnTensorDescriptor_t*)_size_description.get(), errors, *(cudnnTensorDescriptor_t*)_size_description.get(),
						input, &beta, *(cudnnTensorDescriptor_t*)_size_description.get(), output_errors));
		}
		
		////////////////////////////////////////////////////////////
		void Activation::clear()
		{
			if (_activation_description != NULL) {
				checkCUDNN(cudnnDestroyActivationDescriptor(*(cudnnActivationDescriptor_t*)_activation_description));
				operator delete(_activation_description);
			}
		}
		
		////////////////////////////////////////////////////////////
		///	CONVOLUTION
		////////////////////////////////////////////////////////////
			
		////////////////////////////////////////////////////////////
		Convolution::Convolution()
		{
			_filter_description = NULL;
			_convolution_description = NULL;
			_fwd_algorithm_description = NULL;
			_bwd_filter_algorithm_description = NULL;
			_bwd_data_algorithm_description = NULL;
			_workspace_size = 0;
		}

		////////////////////////////////////////////////////////////
		Convolution::Convolution(const int input_width, const int input_height, const int input_depth,
				const int batch_size, const int filter_width, const int filter_height, const int filter_count,
				const int padding_w, const int padding_h, const int stride_u,
				const int stride_v, const bool backward_errors)
		{
			create(input_width, input_height, input_depth, batch_size, filter_width, filter_height,
				filter_count, padding_w, padding_h, stride_u, stride_v, backward_errors);
		}

		////////////////////////////////////////////////////////////
		Convolution::~Convolution()
		{
			clear();
		}

		////////////////////////////////////////////////////////////
		void Convolution::create(const int input_width, const int input_height, const int input_depth,
				const int batch_size, const int filter_width, const int filter_height, const int filter_count,
				const int padding_w, const int padding_h, const int stride_u,
				const int stride_v, const bool backward_errors)
		{
			clear();
			
			_input_description.create(input_width, input_height, input_depth, batch_size, DATA_FLOAT);
			_bias_description.create(1, 1, filter_count, 1, DATA_FLOAT);

			_filter_description = operator new(sizeof(cudnnFilterDescriptor_t));
			_convolution_description = operator new(sizeof(cudnnConvolutionDescriptor_t));
			_fwd_algorithm_description = operator new(sizeof(cudnnConvolutionFwdAlgo_t));
			_bwd_filter_algorithm_description = operator new(sizeof(cudnnConvolutionBwdFilterAlgo_t));
			if (backward_errors) _bwd_data_algorithm_description = operator new(sizeof(cudnnConvolutionBwdDataAlgo_t));
			else _bwd_data_algorithm_description = NULL;
			checkCUDNN(cudnnCreateFilterDescriptor((cudnnFilterDescriptor_t*)_filter_description));
			checkCUDNN(cudnnCreateConvolutionDescriptor((cudnnConvolutionDescriptor_t*)_convolution_description));
			
			checkCUDNN(cudnnSetFilter4dDescriptor(*(cudnnFilterDescriptor_t*)_filter_description,
				CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filter_count, input_depth, filter_height, filter_width));
			checkCUDNN(cudnnSetConvolution2dDescriptor(*(cudnnConvolutionDescriptor_t*)_convolution_description,
				padding_h, padding_w, stride_v, stride_u, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

			int output_w, output_h, output_c, output_n;
			 cudnnGetConvolution2dForwardOutputDim( *(cudnnConvolutionDescriptor_t*)_convolution_description,
				*(cudnnTensorDescriptor_t*)_input_description.get(), *(cudnnFilterDescriptor_t*)_filter_description,
				&output_n, &output_c, &output_h, &output_w);
			_output_description.create(output_w, output_h, output_c, output_n, DATA_FLOAT);
			_output_width = output_w;
			_output_height = output_h;
			_output_depth = output_c;

			_weights_size = filter_width * filter_height * filter_count * input_depth;
			_bias_size = filter_count;
			
			//Foreward algoritm
			checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					*(cudnnFilterDescriptor_t*)_filter_description,
					*(cudnnConvolutionDescriptor_t*)_convolution_description,
					*(cudnnTensorDescriptor_t*)_output_description.get(),
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
					0,
					(cudnnConvolutionFwdAlgo_t*)_fwd_algorithm_description));
			
			//Update workspace size
			size_t tmp_workspace_size;
			checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					*(cudnnFilterDescriptor_t*)_filter_description,
					*(cudnnConvolutionDescriptor_t*)_convolution_description,
					*(cudnnTensorDescriptor_t*)_output_description.get(),
					*(cudnnConvolutionFwdAlgo_t*)_fwd_algorithm_description,
					&tmp_workspace_size));
			_workspace_size = max((int)tmp_workspace_size, _workspace_size);

			//Filter weights gradient calculation algorithm
			checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
						cudnnHandle,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						*(cudnnConvolutionDescriptor_t*)_convolution_description,
						*(cudnnFilterDescriptor_t*)_filter_description,
						CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
						0, 
						(cudnnConvolutionBwdFilterAlgo_t*)_bwd_filter_algorithm_description));
			
			//Update workspace size
			tmp_workspace_size = 0;
			checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
						cudnnHandle,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						*(cudnnConvolutionDescriptor_t*)_convolution_description,
						*(cudnnFilterDescriptor_t*)_filter_description,
						*(cudnnConvolutionBwdFilterAlgo_t*)_bwd_filter_algorithm_description,
						&tmp_workspace_size));
			_workspace_size = max((int)tmp_workspace_size, _workspace_size);
			
			//Backpropagate gradients algorithm
			if (_bwd_data_algorithm_description != NULL) {
				
				checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
							cudnnHandle,
							*(cudnnFilterDescriptor_t*)_filter_description,
							*(cudnnTensorDescriptor_t*)_output_description.get(),
							*(cudnnConvolutionDescriptor_t*)_convolution_description,
							*(cudnnTensorDescriptor_t*)_input_description.get(),
							CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
							0,
							(cudnnConvolutionBwdDataAlgo_t*)_bwd_data_algorithm_description));

				tmp_workspace_size = 0;
				checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
							cudnnHandle,
							*(cudnnFilterDescriptor_t*)_filter_description,
							*(cudnnTensorDescriptor_t*)_output_description.get(),
							*(cudnnConvolutionDescriptor_t*)_convolution_description,
							*(cudnnTensorDescriptor_t*)_input_description.get(),
							*(cudnnConvolutionBwdDataAlgo_t*)_bwd_data_algorithm_description,
							&tmp_workspace_size));
				_workspace_size = max((int)tmp_workspace_size, _workspace_size);
			}
		}
		
		////////////////////////////////////////////////////////////
		void Convolution::foreward(void* input, void* output, void* weights, void* bias, void* workspace) 
		{
			const float alpha = 1.0;
			const float beta = 0.0;
			
			//Convolution foreward
			checkCUDNN(cudnnConvolutionForward(cudnnHandle,
						&alpha, 
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						input, 
						*(cudnnFilterDescriptor_t*)_filter_description, 
						weights,
						*(cudnnConvolutionDescriptor_t*)_convolution_description, 
						*(cudnnConvolutionFwdAlgo_t*)_fwd_algorithm_description, 
						workspace, 
						(size_t)_workspace_size, 
						&beta,
						*(cudnnTensorDescriptor_t*)_output_description.get(), 
						output));
			
			//Add bias
			checkCUDNN(cudnnAddTensor(cudnnHandle,
						&alpha,
						*(cudnnTensorDescriptor_t*)_bias_description.get(),
						bias,
						&alpha,
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						output));
		}
		
		////////////////////////////////////////////////////////////
		void Convolution::accumulate_deltas(void* input, void* output, void* errors, void* filter_deltas,
			void* bias_deltas, void* workspace)
		{
			const float alpha = 1.0;
			const float beta = 1.0;
			
			checkCUDNN(cudnnConvolutionBackwardBias(
						cudnnHandle,
						&alpha,
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						errors,
						&beta,
						*(cudnnTensorDescriptor_t*)_bias_description.get(),
						bias_deltas));

			checkCUDNN(cudnnConvolutionBackwardFilter(
						cudnnHandle,
						&alpha,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						input,
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						errors,
						*(cudnnConvolutionDescriptor_t*)_convolution_description, 
						*(cudnnConvolutionBwdFilterAlgo_t*)_bwd_filter_algorithm_description,
						workspace,
						(size_t)_workspace_size,
						&beta,
						*(cudnnFilterDescriptor_t*)_filter_description,
						filter_deltas));
			
		}

		////////////////////////////////////////////////////////////
		void Convolution::backward(void* errors, void* output_errors, void* weights, void* workspace)
		{
			const float alpha = 1.0;
			const float beta = 1.0;

			checkCUDNN(cudnnConvolutionBackwardData(
						cudnnHandle,
						&alpha,
						*(cudnnFilterDescriptor_t*)_filter_description,
						weights,
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						errors,
						*(cudnnConvolutionDescriptor_t*)_convolution_description, 
						*(cudnnConvolutionBwdDataAlgo_t*)_bwd_data_algorithm_description,
						workspace,
						_workspace_size,
						&beta,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						output_errors));
		}
		
		////////////////////////////////////////////////////////////
		void Convolution::update_weights(void* weights, void* filter_deltas, void* bias, void* bias_deltas, const float learningrate)
		{
			const float alpha = learningrate;
			cublasSaxpy(cublasHandle, _weights_size, &alpha, (float*)filter_deltas, 1, (float*)weights, 1);
			cublasSaxpy(cublasHandle, _bias_size, &alpha, (float*)bias_deltas, 1, (float*)bias, 1);
		}

		////////////////////////////////////////////////////////////
		void Convolution::clear()
		{
			if (_filter_description != NULL && _convolution_description != NULL && _fwd_algorithm_description != NULL) {
				cudnnDestroyFilterDescriptor(*(cudnnFilterDescriptor_t*)_filter_description);
				cudnnDestroyConvolutionDescriptor(*(cudnnConvolutionDescriptor_t*)_convolution_description);
				operator delete(_filter_description);
				operator delete(_convolution_description);
				operator delete(_fwd_algorithm_description);
				operator delete(_bwd_filter_algorithm_description);
				if (_bwd_data_algorithm_description != NULL)
					operator delete(_bwd_data_algorithm_description);
				_workspace_size = 0;
			}
			_filter_description = NULL;
			_convolution_description = NULL;
			_fwd_algorithm_description = NULL;
			_bwd_filter_algorithm_description = NULL;
			_bwd_data_algorithm_description = NULL;
			_workspace_size = 0;
		}
		
		////////////////////////////////////////////////////////////
		void Convolution::getOutputSize(int* output_width, int* output_height, int* output_depth)
		{
			*output_width = _output_width;
			*output_height = _output_height;
			*output_depth = _output_depth;
		}
		
		////////////////////////////////////////////////////////////
		int Convolution::getWorkspaceSize()
		{
			return _workspace_size;
		}
		
		////////////////////////////////////////////////////////////
		///	POOLING
		////////////////////////////////////////////////////////////
		
		////////////////////////////////////////////////////////////
		Pooling::Pooling()
		{
			_pooling_description = NULL;
		}
			
		////////////////////////////////////////////////////////////
		Pooling::Pooling(const int input_width, const int input_height, const int input_count,
				const int batch_size, const int pooling_width, const int pooling_height, const PoolingType type)
		{
			create(input_width, input_height, input_count, batch_size, pooling_width, pooling_height, type);
		}

		////////////////////////////////////////////////////////////
		Pooling::~Pooling()
		{
			clear();
		}
		
		////////////////////////////////////////////////////////////
		void Pooling::clear()
		{
			if (_pooling_description != NULL) {
				checkCUDNN(cudnnDestroyPoolingDescriptor(*(cudnnPoolingDescriptor_t*)_pooling_description));
				operator delete(_pooling_description);
			}
		}

		////////////////////////////////////////////////////////////
		void Pooling::create(const int input_width, const int input_height, const int input_count,
				const int batch_size, const int pooling_width, const int pooling_height, const PoolingType type)
		{
			clear();

			_input_description.create(input_width, input_height, input_count, batch_size, DATA_FLOAT);
			
			_pooling_description = operator new(sizeof(cudnnPoolingDescriptor_t));
			
			cudnnPoolingMode_t poolingmode;
			switch (type)
			{
				case POOLING_MAX:
					poolingmode = CUDNN_POOLING_MAX;
					break;
				
				case POOLING_AVERAGE:
					poolingmode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
					break;
			}

			checkCUDNN(cudnnCreatePoolingDescriptor((cudnnPoolingDescriptor_t*)_pooling_description));	
			checkCUDNN(cudnnSetPooling2dDescriptor(*(cudnnPoolingDescriptor_t*)_pooling_description,
						poolingmode,
						CUDNN_PROPAGATE_NAN,
						pooling_width, pooling_height,
						0, 0,
						pooling_width, pooling_height));

			int n, c, h, w;
			checkCUDNN(cudnnGetPooling2dForwardOutputDim(
						*(cudnnPoolingDescriptor_t*)_pooling_description,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						&n,
						&c,
						&h,
						&w
						));
			_output_description.create(w, h, c, n, DATA_FLOAT);
		}

		////////////////////////////////////////////////////////////
		void Pooling::foreward(void* input, void* output)
		{
			const float alpha = 1.0;
			const float beta = 0.0;
			
			checkCUDNN(cudnnPoolingForward(
					cudnnHandle,
					*(cudnnPoolingDescriptor_t*)_pooling_description,
					&alpha,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					input,
					&beta,
					*(cudnnTensorDescriptor_t*)_output_description.get(),
					output));
		}

		////////////////////////////////////////////////////////////
		void Pooling::backward(void* input, void* outputs, void* errors, void* out_errors)
		{
			const float alpha = 1.0;
			const float beta = 1.0;
			
			checkCUDNN(cudnnPoolingBackward(cudnnHandle,
						*(cudnnPoolingDescriptor_t*)_pooling_description,
						&alpha, 
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						outputs,
						*(cudnnTensorDescriptor_t*)_output_description.get(),
						errors,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						input,
						&beta,
						*(cudnnTensorDescriptor_t*)_input_description.get(),
						out_errors));
		}
		
		////////////////////////////////////////////////////////////
		///	DROPOUT
		////////////////////////////////////////////////////////////
			
		////////////////////////////////////////////////////////////
		Dropout::Dropout()
		{
			_dropout_description = NULL;
			_states_size = 0;
			_reserve_space_size = 0;
		}
		
		////////////////////////////////////////////////////////////
		Dropout::Dropout(const int input_size, const float dropout_probability,  void* state_buffer)
		{
			create(input_size, dropout_probability, state_buffer);			
		}
		
		////////////////////////////////////////////////////////////
		Dropout::~Dropout()
		{
			clear();
		}
		
		////////////////////////////////////////////////////////////
		void Dropout::clear()
		{
			if (_dropout_description != NULL) {
				checkCUDNN(cudnnDestroyDropoutDescriptor(*(cudnnDropoutDescriptor_t*)_dropout_description));
				operator delete(_dropout_description);
				_states_size = 0;
				_reserve_space_size = 0;
				_dropout_description = NULL;
			}
		}

		////////////////////////////////////////////////////////////
		void Dropout::create(const int input_size, const float dropout_probability,  void* state_buffer)
		{
			clear();

			_input_description.create(input_size, 1, 1, 1, DATA_FLOAT);
			_states_size = getStatesSize();
			_reserve_space_size = getReserveSpaceSize(input_size);

			_dropout_description = operator new(sizeof(cudnnDropoutDescriptor_t));
			checkCUDNN(cudnnCreateDropoutDescriptor((cudnnDropoutDescriptor_t*)_dropout_description));
			
			checkCUDNN(cudnnSetDropoutDescriptor(
				*(cudnnDropoutDescriptor_t*)_dropout_description,
				cudnnHandle,
				dropout_probability,
				state_buffer,
				(size_t)_states_size,
				1));
		}
		
		////////////////////////////////////////////////////////////
		void Dropout::foreward(void* input, void* output, void* reserve_space_buffer)
		{
			checkCUDNN(cudnnDropoutForward(
					cudnnHandle,
					*(cudnnDropoutDescriptor_t*)_dropout_description,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					input,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					output,
					reserve_space_buffer,
					(size_t)_reserve_space_size));
			}
		
		////////////////////////////////////////////////////////////
		void Dropout::backward(void* errors, void* out_errors, void* reserve_space_buffer)
		{
			checkCUDNN(cudnnDropoutBackward(
					cudnnHandle,
					*(cudnnDropoutDescriptor_t*)_dropout_description,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					errors,
					*(cudnnTensorDescriptor_t*)_input_description.get(),
					out_errors,
					reserve_space_buffer,
					(size_t)_reserve_space_size));
		}
		
		////////////////////////////////////////////////////////////
		size_t Dropout::getStatesSize()
		{
			size_t bytes;
			checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle, &bytes));
			return bytes;
		}
		
		////////////////////////////////////////////////////////////
		size_t Dropout::getReserveSpaceSize(const int input_size)
		{
			cudnn::TensorDescription input;
			input.create(input_size, 1, 1, 1, DATA_FLOAT);
			size_t bytes;
			checkCUDNN(cudnnDropoutGetReserveSpaceSize(*(cudnnTensorDescriptor_t*)input.get(), &bytes));
			return bytes;
		}

	} /* namespace cudnn */
	





	////////////////////////////////////////////////////////////
	///	NAMESPACE CUDA
	////////////////////////////////////////////////////////////
	namespace cuda
	{

		////////////////////////////////////////////////////////////
		///	CUDA KERNELS
		////////////////////////////////////////////////////////////
		
		__device__ float knl_tmp_float_buf[1024];
		__device__ int knl_tmp_int_buf[1024];
		__constant__ float _selu_alpha = 1.6732632423543772;
		__constant__ float _selu_scale = 1.0507009873554804;
		
		////////////////////////////////////////////////////////////
		__global__ void knl_conv_foreward(float* weights, float* bias, float* inputs, float* outputs,
			int* out_in_map, int input_count, int output_size, int input_size, int filter_area, int filters_count)
		{
			extern __shared__ float cache[]; //size of blockDim.x
			cache[threadIdx.x] = 0; //clear my cache position
			
			//REM
			//blockIdx.x -> output pos
			//blockIdx.y -> filter id
			
			//Shortcuts
			#define WEIGHT_OFFSET (blockIdx.y * input_count * filter_area)
			#define INPUT_OFFSET ((tid / filter_area) * input_size)
			int tid = threadIdx.x;
			int i;
			
			//Compute partial neuron output and put it inside the cache
			while (tid < filter_area * input_count) {
				const int filter_id = tid % filter_area;
				if (out_in_map[blockIdx.x * filter_area + filter_id] != -1)
					cache[threadIdx.x] += inputs[INPUT_OFFSET + out_in_map[blockIdx.x * filter_area + filter_id]]
										* weights[WEIGHT_OFFSET + tid];
				tid += blockDim.x;
			}	
			__syncthreads();
			
			//Reduce all the partial neuron outputs to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i) cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}

			//Finally, store the single value to the outputs buffer
			if (threadIdx.x == 0) outputs[blockIdx.y * output_size + blockIdx.x] = cache[0] + bias[blockIdx.y];
		}
		
		/*
		////////////////////////////////////////////////////////////
		__global__ void knl_conv_foreward(float* weights, float* bias, float* inputs, float* outputs,
			int* out_in_map, int input_count, int output_size, int input_size, int filter_area, int filters_count)
		{
			extern __shared__ float cache[]; //size of blockDim.x
			
			//REM
			//blockIdx.x -> output pos
			//blockIdx.y -> filter id
			
			//Shortcuts
			#define OUTPUT_FILTER (tid / output_size)
			#define OUTPUT_X (tid % output_width)
			#define OUTPUT_Y ((tid / output_width) % output_height)
			#define WEIGHT_OFFSET (OUTPUT_FILTER * input_count * filter_area)
			#define INPUT_OFFSET ((i / filter_area) * input_size)
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int i = 0;
			
			while (tid < output_size * filters_count) {
				
				//Clear output
				cache[threadIdx.x] = 0;

				//Compute partial neuron output and put it inside the cache
				i = 0;
				while (i < filter_area * input_count) {
					const int filter_id = i % filter_area;
					if (out_in_map[(tid % output_size) * filter_area + filter_id] != -1)
						cache[threadIdx.x] += inputs[INPUT_OFFSET + out_in_map[(tid % output_size) * filter_area + filter_id]]
							* weights[WEIGHT_OFFSET + i];
					i++;
				}
				
				//Finally, store the single value to the outputs buffer
				outputs[tid] = cache[threadIdx.x] + bias[OUTPUT_FILTER];

				//Update output pos
				tid += blockDim.x * gridDim.x;
			}
		}
		*/
		
		////////////////////////////////////////////////////////////
		__global__ void knl_conv_backward(float* weights, float* out_errors, float* errors,
			int* in_weight_map, int* in_out_map, int input_count, int output_size, int input_width,
			int input_height, int filter_area, int filters_count)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0; //clear my cache position
			
			//Shortcuts
			const int x = blockIdx.x; //inputx
			const int y = blockIdx.y; //inputy
			const int z = blockIdx.z; //input_id
			int tid = threadIdx.x;
			int i;
			
			const int input_local_id = (y * input_width + x);
			#define __INPUT_GLOBAL_ID (z * input_width * input_height + input_local_id)

			while (tid < filter_area * filters_count) {
				
				//Get weight and output index from 
				const int w_id = in_weight_map[input_local_id * filter_area + (tid % filter_area)];
				const int o_id = in_out_map[input_local_id * filter_area + (tid % filter_area)];
				
				//Check if this weight and output exists
				if (w_id != -1 && o_id != -1) {
					//Update cache
					cache[threadIdx.x] += weights[(tid / filter_area) * filter_area * input_count
						+ z * filter_area + w_id] * errors[(tid / filter_area) * output_size + o_id];
				}
				tid += blockDim.x;
			}
			__syncthreads();
			
			//Reduce all the partial errors to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			
			//Finally, store the single error value in th out_errors buffer
			if (threadIdx.x == 0) out_errors[__INPUT_GLOBAL_ID] += cache[0];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_conv_accumulate_deltas(float* weights_deltas, float* bias_deltas,
			float* errors, float* inputs, float* outputs, int* out_in_map, int input_count,
			int input_size, int output_size, int filter_area, int filters_count)
		{
			extern __shared__ float cache[]; //size of blockDim.x
			cache[threadIdx.x] = 0; //clear my cache position

			//Shortcuts
			const int x = blockIdx.x; //filter id
			const int y = blockIdx.y; //weight id
			int tid = threadIdx.x;
			int i;
				
			if (y == filter_area * input_count) //bias
			{
				//Shortcuts
				#define __DELTAS_BIAS_OFFSET x
				#define __OUTPUT_ID x * output_size + tid //the global output position

				//Compute partial neuron output and put it inside the cache
				while (tid < output_size) {
					cache[threadIdx.x] += errors[__OUTPUT_ID];
					tid += blockDim.x;
				}
				__syncthreads();

				//Reduce all the partial deltas to a single deltas value
				for (i = blockDim.x/2; i > 0; i /= 2) {
					if (threadIdx.x < i)
						cache[threadIdx.x] += cache[threadIdx.x + i];
					__syncthreads();
				}

				//Finally, store the single delta value in th deltas buffer
				if (threadIdx.x == 0) bias_deltas[__DELTAS_BIAS_OFFSET] += cache[0];
			}
			else //normal weight
			{
				//Shortcuts
				#define __INPUT_OFFSET (y / filter_area) * input_size //Where my input begins
				#define __OUTPUT_ID x * output_size + tid //the global output position
				#define __WEIGHT_ID x * filter_area * input_count + y //the global weight position
				#define __WEIGHT_FILTER_ID y % filter_area //the local weight position inside the filter in one input

				//Compute partial neuron output and put it inside the cache
				while (tid < output_size) {
					if (out_in_map[tid * filter_area + __WEIGHT_FILTER_ID] != -1)
						cache[threadIdx.x] += inputs[__INPUT_OFFSET + out_in_map[tid * filter_area + __WEIGHT_FILTER_ID]] * errors[__OUTPUT_ID];
					tid += blockDim.x;
				}
				__syncthreads();

				//Reduce all the partial deltas to a single deltas value
				for (i = blockDim.x/2; i > 0; i /= 2) {
					if (threadIdx.x < i)
						cache[threadIdx.x] += cache[threadIdx.x + i];
					__syncthreads();
				}

				//Finally, store the single delta value in th deltas buffer
				if (threadIdx.x == 0) weights_deltas[__WEIGHT_ID] += cache[0];
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_conv_update_parameters(float* weights, float* bias, float* weights_deltas,
				float* bias_deltas, int filter_area, int input_count, int filter_count, float learningrate)
		{
			//Update weights
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < filter_area * input_count * filter_count) {
				weights[tid] += weights_deltas[tid] * learningrate;
				tid += blockDim.x * gridDim.x;
			}

			//Update bias
			int offset = filter_area * input_count * filter_count;
			tid -= offset;
			while (tid < filter_count) {
				bias[tid] += bias_deltas[tid] * learningrate;
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_maxpooling_foreward(float* inputs, float* outputs, int* maxbuffer,
			int input_width, int input_height, int input_count, int stride, int filter_size,
			int output_width, int output_height)
		{
			extern __shared__ float cache[]; //size of blockDim.x
			
			//output id
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= output_width * output_height * input_count) return;
			
			cache[threadIdx.x] = -0x7FFE; //clear my cache position

			const int z = tid / (output_width * output_height);
			const int x = tid % output_width;
			const int y = (tid / output_width) % output_height;

			const int stopX = (x * stride + filter_size > input_width) ? (x * stride + filter_size) - x : filter_size;
			const int stopY = (y * stride + filter_size > input_height) ? (y * stride + filter_size) - y : filter_size;

			int index, sx, sy;
			for (sx = 0; sx < stopX; sx++) {
				for (sy = 0; sy < stopY; sy++) {
					index = z * input_width * input_height + input_width * (y * stride + sy) + x * stride + sx;
					if (inputs[index] > cache[threadIdx.x]) {
						cache[threadIdx.x] = inputs[index];
						maxbuffer[tid] = index; 
					}
				}
			}
			outputs[tid] = cache[threadIdx.x];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_maxpooling_backward(float* out_errors, float* errors, int* maxbuffer,
			int input_width, int input_height, int input_count, int stride, int filter_size,
			int output_width, int output_height)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= output_width * output_height * input_count) return;
			out_errors[maxbuffer[tid]] += errors[tid];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_averagepooling_foreward(float* inputs, float* outputs,
			int input_width, int input_height, int input_count, int stride, int filter_size,
			int output_width, int output_height)
		{
			extern __shared__ float cache[]; //size of blockDim.x
			
			//output id
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= output_width * output_height * input_count) return;
			
			cache[threadIdx.x] = 0; //clear my cache position

			const int z = tid / (output_width * output_height);
			const int x = tid % output_width;
			const int y = (tid / output_width) % output_height;

			const int stopX = (x * stride + filter_size > input_width) ? (x * stride + filter_size) - x : filter_size;
			const int stopY = (y * stride + filter_size > input_height) ? (y * stride + filter_size) - y : filter_size;

			int sx, sy;
			for (sx = 0; sx < stopX; sx++) {
				for (sy = 0; sy < stopY; sy++) {
					#define INPUT_INDEX (z * input_width * input_height + input_width * (y * stride + sy) + x * stride + sx)
					cache[threadIdx.x] += inputs[INPUT_INDEX];
				}
			}
			outputs[tid] = cache[threadIdx.x] / (float)(stopX * stopY);
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_averagepooling_backward(float* out_errors, float* errors, int input_width,
			int input_height, int input_count, int stride, int filter_size, int output_width,
			int output_height)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= output_width * output_height * input_count) return;
			
			const int z = tid / (output_width * output_height);
			const int x = tid % output_width;
			const int y = (tid / output_width) % output_height;

			const int stopX = (x * stride + filter_size > input_width) ? (x * stride + filter_size) - x : filter_size;
			const int stopY = (y * stride + filter_size > input_height) ? (y * stride + filter_size) - y : filter_size;

			int sx, sy;
			for (sx = 0; sx < stopX; sx++) {
				for (sy = 0; sy < stopY; sy++) {
					#define INPUT_INDEX (z * input_width * input_height + input_width * (y * stride + sy) + x * stride + sx)
					out_errors[INPUT_INDEX] += errors[tid];
				}
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_linear_foreward(float* weights, float* bias, float* inputs, float* outputs,
			int input_size, int output_size, bool accumulate, bool use_bias)
		{
			extern __shared__ float cache[];
			if (accumulate) cache[threadIdx.x] = outputs[blockIdx.x];
			else cache[threadIdx.x] = 0; 
			
			int tid = threadIdx.x;
			int i;

			//Compute partial neuron output and put it inside the cache
			while (tid < input_size) {
				cache[threadIdx.x] += inputs[tid] * weights[tid * output_size + blockIdx.x];
				tid += blockDim.x;
			}
			__syncthreads();

			//Reduce all the partial neuron outputs to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}

			//Finally, store the single value to the outputs buffer
			if (threadIdx.x == 0) {
				outputs[blockIdx.x] = cache[0];
				if (use_bias == true) outputs[blockIdx.x] += bias[blockIdx.x];
			}
		}
	
		////////////////////////////////////////////////////////////
		__global__ void knl_linear_backward(float* weights, float* out_errors, float* errors, int input_size, int output_size)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0; //Clear my cache position
			
			int tid = threadIdx.x;
			int i;
			
			//Compute partial leaving errors and put it inside the cache
			while (tid < output_size) {
				cache[threadIdx.x] += errors[tid] * weights[blockIdx.x * output_size + tid];
				tid += blockDim.x;
			}
			__syncthreads();

			//Reduce all the partial leaving errors to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}

			//Finally, store the single value to the out_errors buffer
			if (threadIdx.x == 0) out_errors[blockIdx.x] += cache[0];
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_linear_accumulate_deltas(float* deltas, float* inputs, float* errors, int input_size, int output_size, bool use_bias)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < input_size * output_size) {
				deltas[tid] += inputs[tid / output_size] * errors[tid % output_size];
				tid += blockDim.x * gridDim.x;
			}

			//Update bias
			if (use_bias == false) return;
			tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < output_size) {
				deltas[input_size * output_size + tid] += errors[tid];
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_linear_update_parameters(float* weights, float* bias, float* deltas, float learningrate, int input_size, int output_size)
		{
			//Update weights
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < input_size * output_size) {
				weights[tid] += deltas[tid] * learningrate;
				tid += blockDim.x * gridDim.x;
			}

			//Update bias
			tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < output_size) {
				bias[tid] += deltas[input_size * output_size + tid] * learningrate;
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_sigmoid_foreward(float* inputs, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				outputs[tid] = 1.0 / (1.0 + expf(-inputs[tid]));
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_sigmoid_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				out_errors[tid] = outputs[tid] * (1.f - outputs[tid]) * errors[tid]; //derivative * error
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_relu_foreward(float* inputs, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				outputs[tid] = inputs[tid] * (inputs[tid] > 0);
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_relu_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				out_errors[tid] = (outputs[tid] > 0) * errors[tid]; //derivative * error
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_selu_foreward(float* inputs, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				if (inputs[tid] >= 0.0) outputs[tid] = _selu_scale * inputs[tid];
				else outputs[tid] = _selu_scale * (_selu_alpha * expf(inputs[tid]) - _selu_alpha);
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_selu_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				if (outputs[tid] >= 0.0) out_errors[tid] = _selu_scale * errors[tid];
				else out_errors[tid] = errors[tid] * (outputs[tid] + _selu_scale * _selu_alpha);
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_tanh_foreward(float* inputs, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				outputs[tid] = tanh(inputs[tid]);
				tid += blockDim.x * gridDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_tanh_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				out_errors[tid] = (1.f - outputs[tid] * outputs[tid]) * errors[tid]; //derivative * error
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_dropout_foreward(float* inputs, float* outputs, unsigned int seed, float drop_probability, bool training, int size)
		{
			extern __shared__ unsigned int tseeds[];
			
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= size) return;
			
			tseeds[threadIdx.x] += seed * (tid + 1);


			if (drop_probability != 0 && training == true) 
			{
				//pseudo random number generator, or magic for short
				tseeds[threadIdx.x] ^= tseeds[threadIdx.x] << 13;
				tseeds[threadIdx.x] ^= tseeds[threadIdx.x] << 17;
				tseeds[threadIdx.x] ^= tseeds[threadIdx.x] << 5;
				
				//drop?
				if ((float)tseeds[threadIdx.x] / ((unsigned int) UINT_MAX) < drop_probability) outputs[tid] = 0.f;
				else outputs[tid] = inputs[tid];
			}
			else
			{
				outputs[tid] = inputs[tid];
			}
			
			tid += blockDim.x * gridDim.x;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_dropout_backward(float* errors, float* out_errors, float* outputs, float drop_probability, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				if (outputs[tid] == 0) out_errors[tid] = 0;
				else out_errors[tid] = errors[tid] * (1 - drop_probability);
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_softmax_foreward(float* inputs, float* outputs, float scale, int size, float epsilon)
		{
			extern __shared__ float cache[];
			int tid = threadIdx.x;
			int i;
			cache[threadIdx.x] = 0;

			while (tid < size) {
				outputs[tid] = exp(inputs[tid] * scale);
				cache[threadIdx.x] += outputs[tid];
				tid += blockDim.x;
			}
			__syncthreads();
			
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}

			tid = threadIdx.x;
			while (tid < size) {
				outputs[tid] /= (cache[0] + epsilon);
				tid += blockDim.x;
			}
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_softmax_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;

			while (tid < size) {
				out_errors[tid] = outputs[tid] * (1.f - outputs[tid]) * errors[tid];
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_cost_crossentropy(float* prediction, float* target, float* errors, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			extern __shared__ float predictions[];
			float denominator;

			while (tid < size) {
				predictions[threadIdx.x] = prediction[tid];
				denominator = predictions[threadIdx.x] - predictions[threadIdx.x] * predictions[threadIdx.x];
				if (denominator < 1e-6) denominator = 1e-6;
				errors[tid] = (target[tid] - predictions[threadIdx.x]) / denominator;
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_foreward_1(float* inputs, int size)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;

			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int i;

			//calculate mean
			while (tid < size) {
				cache[threadIdx.x] += inputs[tid];
				tid += blockDim.x * gridDim.x;
			}
			__syncthreads();
			
			//Reduce all the partial mean sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) knl_tmp_float_buf[blockIdx.x] = cache[0] / (float)size;
			if (blockIdx.x == 0 && threadIdx.x == 0) knl_tmp_int_buf[0] = gridDim.x;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_foreward_2(float* inputs, float* deviation, int size)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;

			int tid = threadIdx.x;
			int i;

			//calculate mean
			while (tid < knl_tmp_int_buf[0]) {
				cache[threadIdx.x] += knl_tmp_float_buf[tid];
				tid += blockDim.x;
			}
			__syncthreads();
			
			//Reduce all the partial mean sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) knl_tmp_float_buf[0] = cache[0];
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_foreward_3(float* inputs, float* deviation, int size)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;

			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int i;

			while (tid < size) {
				deviation[tid] = inputs[tid] - knl_tmp_float_buf[0];
				cache[threadIdx.x] += deviation[tid] * deviation[tid];
				tid += blockDim.x * gridDim.x;
			}
			
			//Reduce all the partial mean sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) knl_tmp_float_buf[blockIdx.x] = cache[0] / (float)size;
			if (threadIdx.x == 0 && blockIdx.x == 0) knl_tmp_int_buf[0] = gridDim.x;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_foreward_4(float* variance, int size)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;

			int tid = threadIdx.x;
			int i;

			//calculate mean
			while (tid < knl_tmp_int_buf[0]) {
				cache[threadIdx.x] += knl_tmp_float_buf[tid];
				tid += blockDim.x;
			}
			__syncthreads();
			
			//Reduce all the partial mean sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) *variance = cache[0];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_foreward_5(float* inputs, float* deviation, float* normalized,
			float* outputs, float* variance, float* gamma, float* beta, float epsilon, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			
			while (tid < size) {
				normalized[tid] = deviation[tid] / sqrt(*variance + epsilon);
				outputs[tid] = normalized[tid] * (*gamma) + *beta;
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_backward_1(float* errors, float* deviation, int size)
		{
			//Allocate and clear cache
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;
			
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int i;

			//Sum all errors
			while (tid < size) {
				cache[threadIdx.x] += errors[tid];
				tid += blockDim.x * gridDim.x;
			}
			__syncthreads();

			//Reduce all the partial errors sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}

			//Store results
			if (threadIdx.x == 0) knl_tmp_float_buf[blockIdx.x] = cache[0];
			
			//reset cache
			cache[threadIdx.x] = 0;

			//Sum all errors deviations
			tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < size) {
				cache[threadIdx.x] += errors[tid] * deviation[tid];
				tid += blockDim.x * gridDim.x;
			}
			__syncthreads();

			//Reduce all the partial errors deviations sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			
			//Store results
			if (threadIdx.x == 0) knl_tmp_float_buf[gridDim.x + blockIdx.x] = cache[0];
			
			//Store grid dimension for later
			if (blockIdx.x == 0 && threadIdx.x == 0) knl_tmp_int_buf[0] = gridDim.x;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_backward_2()
		{
			//Allocate and clear cache
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;
			
			int tid = threadIdx.x;
			int i;

			//Sum all errors
			while (tid < knl_tmp_int_buf[0]) {
				cache[threadIdx.x] += knl_tmp_float_buf[tid];
				tid += blockDim.x;
			}
			__syncthreads();

			//Reduce all the partial errors sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) { knl_tmp_float_buf[0] = cache[0]; }
			
			cache[threadIdx.x] = 0; //clear cache
			
			//Sum all errors
			tid = threadIdx.x;
			while (tid < knl_tmp_int_buf[0]) {
				cache[threadIdx.x] += knl_tmp_float_buf[knl_tmp_int_buf[0] + tid];
				tid += blockDim.x;
			}
			__syncthreads();

			//Reduce all the partial errors sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) { knl_tmp_float_buf[1] = cache[0]; }
		}

		__global__ void knl_normalization_backward_3(float* errors, float* out_errors, float* deviation,
			float* gamma, float* beta, float* variance, float epsilon, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			
			while (tid < size) {
				out_errors[tid] = 1.0 / (float)size * (*gamma) / sqrt(*variance + epsilon) * 
					((float)size * errors[tid] - knl_tmp_float_buf[0] - deviation[tid] / ((*variance) + epsilon)
					* knl_tmp_float_buf[1]);
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_accumulate_deltas(float* errors, float* deviation, float* variance,
			float epsilon, float* d_beta, float* d_gamma, int size)
		{
			//Allocate and clear cache
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0;
			
			int tid = threadIdx.x;
			int i;

			//Calculate d_beta delta
			while (tid < size) {
				cache[threadIdx.x] += errors[tid];
				tid += blockDim.x;
			}
			__syncthreads();

			//Reduce all the partial deltas sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) *d_beta += cache[0];
			__syncthreads();

			//reset cache
			cache[threadIdx.x] = 0;
			
			//Calculate d_gamma delta
			tid = threadIdx.x;
			while (tid < size) {
				cache[threadIdx.x] += deviation[tid] * sqrt(*variance + epsilon) * errors[tid];
				tid += blockDim.x;
			}
			__syncthreads();
			
			//Reduce all the partial deltas sum to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}
			if (threadIdx.x == 0) *d_gamma += cache[0];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_normalization_update_parameters(float* beta, float* gamma, float* d_beta,
			float* d_gamma, float momentum, int _size, float learningrate)
		{
			*beta += ((double)*d_beta / (double)_size) * learningrate;
			*gamma += ((double)*d_gamma / (double)_size) * learningrate;
			*d_beta *= momentum;
			*d_gamma *= momentum;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_sparse_indices(float* inputs, int input_size, int* indices, int* tmp_indices, int* indices_count, int chunck_size)
		{
			extern __shared__ unsigned int temp[];

			//clear counters
			temp[threadIdx.x * 2] = 0;
			temp[threadIdx.x * 2 + 1] = 0;

			#define CHUNCK_OFFSET threadIdx.x * chunck_size

			//Each thread should process a chunck
			int tid = CHUNCK_OFFSET;
			while (tid < CHUNCK_OFFSET + chunck_size && tid < input_size) {

				//Store index of non-zero inputs
				if (inputs[tid] != 0) {
					tmp_indices[CHUNCK_OFFSET + temp[threadIdx.x * 2]] = tid;
					temp[threadIdx.x * 2]++;
				}

				//Check the next element
				tid++;
			}
			__syncthreads();

			//calculate offset
			for (tid = 0; tid < threadIdx.x; tid++)
				temp[threadIdx.x * 2 + 1] += temp[tid * 2];

			//store indicies
			for (tid = 0; tid < temp[threadIdx.x * 2]; tid++)
				indices[temp[threadIdx.x * 2 + 1] + tid] = tmp_indices[CHUNCK_OFFSET + tid];

			//store indices count
			if (threadIdx.x == blockDim.x -1) 
				*indices_count = temp[threadIdx.x * 2 + 1] + temp[threadIdx.x * 2];	
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_linear_sparse_foreward(float* weights, float* bias, float* inputs, float* outputs, int* indices, int* indices_count, int input_size, int output_size)
		{
			extern __shared__ float cache[];
			cache[threadIdx.x] = 0; //Clear my cache position
			
			int tid = threadIdx.x;
			int i;

			//Compute partial neuron output and put it inside the cache
			while (tid < *indices_count) {
				cache[threadIdx.x] += inputs[indices[tid]] * weights[indices[tid] * output_size + blockIdx.x];
				tid += blockDim.x;
			}
			__syncthreads();

			//Reduce all the partial neuron outputs to a single value
			for (i = blockDim.x/2; i > 0; i /= 2) {
				if (threadIdx.x < i)
					cache[threadIdx.x] += cache[threadIdx.x + i];
				__syncthreads();
			}

			//Finally, store the single value to the outputs buffer
			if (threadIdx.x == 0) outputs[blockIdx.x] = cache[0] + bias[blockIdx.x];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_linear_sparse_accumulate_deltas(float* deltas, float* inputs, float* errors, int* indices, int* indices_count, int input_size, int output_size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < *indices_count * output_size) {
				const int input = indices[tid % *indices_count];
				const int output = tid / *indices_count;
				deltas[output_size * input + output] += inputs[input] * errors[output];
				tid += blockDim.x * gridDim.x;
			}

			//Update bias
			tid = blockIdx.x * blockDim.x + threadIdx.x;
			while (tid < output_size) {
				deltas[input_size * output_size + tid] += errors[tid];
				tid += blockDim.x * gridDim.x;
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_concatenate_foreward(float** inputs, float* outputs, int* sizes, int input_count)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int offset = tid;
			int input_id = 0;
			
			//Update offset and input index
			while (input_id < input_count && offset > sizes[input_id]) {
				offset -= sizes[input_id];
				input_id++;
			}

			while (input_id < input_count) {
				
				//Update offset and input index
				while (offset > sizes[input_id]) {
					offset -= sizes[input_id];
					input_id++;
					if (input_id >= input_count) return;
				}
				
				outputs[tid] = inputs[input_id][offset];

				tid += gridDim.x * blockDim.x; 
				offset += gridDim.x * blockDim.x; 
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_concatenate_backward(float* errors, float** out_errors, int* sizes, int input_count)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int offset = tid;
			int input_id = 0;
			
			//Update offset and input index
			while (offset > sizes[input_id]) {
				offset -= sizes[input_id];
				input_id++;
				if (input_id >= input_count) return;
			}

			while (input_id < input_count) {
				
				//Update offset and input index
				while (offset > sizes[input_id]) {
					offset -= sizes[input_id];
					input_id++;
					if (input_id >= input_count) return;
				}
				
				out_errors[input_id][offset] += errors[tid];

				tid += gridDim.x * blockDim.x; 
				offset += gridDim.x * blockDim.x; 
			}
		}
		
		////////////////////////////////////////////////////////////
		__global__ void gradient_clipping(float* deltas, int size, const float clipping_deviation)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid < size) return;
			if (deltas[tid] > clipping_deviation) deltas[tid] = clipping_deviation;
			else if (deltas[tid] < -clipping_deviation) deltas[tid] = -clipping_deviation;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_l1_regularization(float* weights, const float l1_factor, const float learningrate, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid < size) return;
			weights[tid] += (weights[tid] > 0 ? -1.f : 1.f) * l1_factor * learningrate;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_l2_regularization(float* weights, const float l2_factor, const float learningrate, int size)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid < size) return;
			weights[tid] += (0 - weights[tid]) * l2_factor * learningrate;
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_image_translate(float* image, float* result_buffer, const int width, const int height, const int channels, const int by_x, const int by_y)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= width * height * channels) return;
			const int c = tid / (width * height);
			const int x = (tid % (width * height)) % width;
			const int y = (tid % (width * height)) / width;
			if (y - by_y < 0 || y - by_y >= height || x - by_x < 0 || x - by_x >= width)
				result_buffer[y * width + x + c * width * height] = 0;
			else
				result_buffer[y * width + x + c * width * height] = image[(y - by_y) * width + x - by_x + c * width * height];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_image_horizontal_flip(float* image, const int width, const int height, const int channels)
		{
			extern __shared__ float cache[];
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int halfw = width / 2;
			if (tid >= halfw * height * channels) return;
			const int c = tid / (halfw * height);
			const int x = (tid % (halfw * height)) % halfw;
			const int y = (tid % (halfw * height)) / halfw;
			cache[threadIdx.x] = image[c * width * height + y * width + x];
			image[c * width * height + y * width + x] = image[c * width * height + y * width + (width - 1 - x)];
			image[c * width * height + y * width + (width - 1 -x)] = cache[threadIdx.x];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_image_vertical_flip(float* image, const int width, const int height, const int channels)
		{
			extern __shared__ float cache[];
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			int halfh = height / 2;
			if (tid >= width * halfh * channels) return;
			const int c = tid / (width * halfh);
			const int x = (tid % (width * halfh)) % width;
			const int y = (tid % (width * halfh)) / width;
			cache[threadIdx.x] = image[c * width * height + y * width + x];
			image[c * width * height + y * width + x] = image[c * width * height + (height -1 -y) * width + x];
			image[c * width * height + (height -1 -y) * width + x] = cache[threadIdx.x];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_image_rotate(float* image, float* result_buffer, const int width, const int height,
			const int channels, const float a, const float b, const int xoffset, const int yoffset)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= width * height * channels) return;
			const int c = tid / (width * height);
			const int x = (tid % (width * height)) % width;
			const int y = (tid % (width * height)) / width;
			const int nx = x * a - y * b + xoffset;
			const int ny = x * b + y * a + yoffset;
			if (nx < 0 || nx >= width || ny < 0 || ny > height) result_buffer[y * width + x + c * width * height] = 0;
			else result_buffer[y * width + x + c * width * height] = image[ny * width + nx + c * width * height];
		}

		////////////////////////////////////////////////////////////
		__global__ void knl_image_scale(float* image, float* result_buffer, const int width,
			const int height, const int channels, const float scale, const int center_x, const int center_y)
		{
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= width * height * channels) return;
			const int c = tid / (width * height);
			const int x = (tid % (width * height)) % width;
			const int y = (tid % (width * height)) / width;
			const int nx = center_x + (x - center_x) * scale;
			const int ny = center_y + (y - center_y) * scale;
			if (nx < 0 || nx >= width || ny < 0 || ny > height) result_buffer[y * width + x + c * width * height] = 0;
			else result_buffer[y * width + x + c * width * height] = image[ny * width + nx + c * width * height];
		}
		
		////////////////////////////////////////////////////////////
		__global__ void knl_image_add_noise(float* image, const int width, const int height, const int channels, const unsigned int seed, const float noise_probability)
		{
			extern __shared__ unsigned int tseeds[];
			
			int tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid >= width * height * channels) return;
			
			//randomize sistem with seed
			tseeds[threadIdx.x] += seed * (tid + 1);
			
			//pseudo random number generator, or magic for short
			tseeds[threadIdx.x] ^= tseeds[threadIdx.x] << 13;
			tseeds[threadIdx.x] ^= tseeds[threadIdx.x] << 17;
			tseeds[threadIdx.x] ^= tseeds[threadIdx.x] << 5;
			
			const int c = tid / (width * height);
			const int x = (tid % (width * height)) % width;
			const int y = (tid % (width * height)) / width;
			
			//Apply random data flipping
			if ((float)tseeds[threadIdx.x] / ((unsigned int) UINT_MAX) < noise_probability)
				image[c * width * height + y * width + x] = 1.f - image[c * width * height + y * width + x];
		}
		
		////////////////////////////////////////////////////////////
		///	INTERFACE
		////////////////////////////////////////////////////////////
		
		//Shortcut
		static unsigned int blocks;
		static unsigned int threads;
	
		//Get nearest lower power of two
		unsigned int low_pow2 (unsigned int x)
		{
			x = x | (x >> 1);
			x = x | (x >> 2);
			x = x | (x >> 4);
			x = x | (x >> 8);
			x = x | (x >> 16);
			return x - (x >> 1);
		}

		//Get nearest higher power of two
		unsigned long high_pow2(unsigned long v)
		{
			v--;
			v |= v >> 1;
			v |= v >> 2;
			v |= v >> 4;
			v |= v >> 8;
			v |= v >> 16;
			v++;
			return v;
		}
		
		unsigned long bestmatch_pow2(unsigned long x)
		{
			unsigned long lp2 = low_pow2(x);
			int mismatch_lp2 = x - lp2;
			int mismatch_hp2 = lp2*2 - x;
			if (mismatch_lp2 < mismatch_hp2) return lp2;
			else return lp2 * 2;
		}
		
		////////////////////////////////////////////////////////////
		void conv_foreward(float* weights, float* bias, float* inputs, float* outputs,
			int* out_in_map, int input_width, int input_height, int input_count, int stride,
			int output_width, int output_height, int filters_count, int filter_area)
		{
			dim3 numBlocks(output_width * output_height, filters_count);	
			threads = min((int)bestmatch_pow2(filter_area * input_count), CUDA_MAX_THREADS) / 4;
			if (threads < 2) threads = 2;
			knl_conv_foreward<<<numBlocks, threads, threads * sizeof(float)>>>(weights, bias,
				inputs, outputs, out_in_map, input_count, output_width * output_height,
				input_width * input_height, filter_area, filters_count);
			
			/*
			threads = min((int)bestmatch_pow2(output_width * output_height * filters_count), CUDA_MAX_THREADS) / 4;
			if (threads < 2) threads = 2;
			if (output_width * output_height * filters_count % threads == 0)
				blocks = (output_width * output_height * filters_count) / threads;
			else
				blocks = (output_width * output_height * filters_count) / threads + 1;
			knl_conv_foreward<<<blocks, threads, threads * sizeof(float)>>>(weights, bias,
				inputs, outputs, out_in_map, input_count, output_width * output_height,
				input_width * input_height, filter_area, filters_count);
			*/
		}
		
		////////////////////////////////////////////////////////////
		void conv_backward(float* weights, float* out_errors, float* errors,
			int* in_weight_map, int* in_out_map, int input_count, int output_size, int input_width,
			int input_height, int filter_area, int filters_count)
		{
			if (input_width == 0 || input_height == 0) return;
			dim3 numBlocks(input_width, input_height, input_count);	
			threads = min((int)bestmatch_pow2(filter_area * filters_count), CUDA_MAX_THREADS) / 4;
			if (threads < 2) threads = 2;
			knl_conv_backward<<<numBlocks, threads, threads * sizeof(float)>>>(weights, out_errors,
				errors, in_weight_map, in_out_map, input_count, output_size, input_width, input_height,
				filter_area, filters_count);
		}
		
		////////////////////////////////////////////////////////////
		void conv_accumulate_deltas(float* weights_deltas, float* bias_deltas, float* errors,
			float* inputs, float* outputs, int* out_in_map, int input_count, int input_width,
			int input_height, int output_size, int filter_area, int filters_count)
		{
			dim3 numBlocks(filters_count, filter_area * input_count + 1);	
			threads = min((int)bestmatch_pow2(output_size), CUDA_MAX_THREADS) / 4;
			if (threads < 2) threads = 2;
			knl_conv_accumulate_deltas<<<numBlocks, threads, threads * sizeof(float)>>>(weights_deltas,
				bias_deltas, errors, inputs, outputs, out_in_map, input_count, input_width * input_height,
				output_size, filter_area, filters_count);
		}
		
		////////////////////////////////////////////////////////////
		void conv_update_parameters(float* weights, float* bias, float* weights_deltas, float* bias_deltas,
			int filter_area, int input_count, int filter_count, float learningrate)
		{
			threads = min((int)low_pow2((filter_area * input_count + 1) * filter_count), CUDA_MAX_THREADS);
			blocks = min((filter_area * input_count + 1) * filter_count / threads + 1, CUDA_MAX_CORES);
			knl_conv_update_parameters<<<blocks, threads>>>(weights, bias, weights_deltas, 
				bias_deltas, filter_area, input_count, filter_count, learningrate);
		}
		
		////////////////////////////////////////////////////////////
		void maxpooling_foreward(float* inputs, float* outputs, int* maxbuffer, int input_width,
			int input_height, int input_count, int stride, int filter_size, int output_width,
			int output_height)
		{
			threads = min((int)low_pow2(output_width * output_height * input_count), CUDA_MAX_THREADS);
			blocks = min((output_width * output_height * input_count) / threads + 1, CUDA_MAX_CORES);
			knl_maxpooling_foreward<<<blocks, threads, threads * sizeof(float)>>>(inputs, outputs, maxbuffer, input_width,
				input_height, input_count, stride, filter_size, output_width, output_height);
		}
	
		////////////////////////////////////////////////////////////
		void maxpooling_backward(float* out_errors, float* errors, int* maxbuffer, int input_width,
			int input_height, int input_count, int stride, int filter_size, int output_width,
			int output_height)
		{
			threads = min((int)low_pow2(output_width * output_height * input_count), CUDA_MAX_THREADS);
			blocks = min((output_width * output_height * input_count) / threads + 1, CUDA_MAX_CORES);
			knl_maxpooling_backward<<<blocks, threads>>>(out_errors, errors, maxbuffer, input_width,
				input_height, input_count, stride, filter_size, output_width, output_height);
		}
		
		////////////////////////////////////////////////////////////
		void averagepooling_foreward(float* inputs, float* outputs, int input_width, int input_height,
			int input_count, int stride, int filter_size, int output_width, int output_height)
		{
			threads = min((int)low_pow2(output_width * output_height * input_count), CUDA_MAX_THREADS);
			blocks = min((output_width * output_height * input_count) / threads + 1, CUDA_MAX_CORES);
			knl_averagepooling_foreward<<<blocks, threads, threads * sizeof(float)>>>(inputs, outputs, input_width,
				input_height, input_count, stride, filter_size, output_width, output_height);
		}
		
		////////////////////////////////////////////////////////////
		void averagepooling_backward(float* out_errors, float* errors, int input_width, int input_height,
			int input_count, int stride, int filter_size, int output_width, int output_height)
		{
			threads = min((int)low_pow2(output_width * output_height * input_count), CUDA_MAX_THREADS);
			blocks = min((output_width * output_height * input_count) / threads + 1, CUDA_MAX_CORES);
			knl_averagepooling_backward<<<blocks, threads>>>(out_errors, errors, input_width,
				input_height, input_count, stride, filter_size, output_width, output_height);
		}
		
		////////////////////////////////////////////////////////////
		void linear_foreward(float* weights, float* bias, float* inputs, float* outputs, int input_size,
			int output_size, bool accumulate, bool use_bias)
		{
			blocks = min(output_size, CUDA_MAX_CORES);
			threads = min((int)low_pow2(input_size), CUDA_MAX_THREADS);
			knl_linear_foreward<<<blocks, threads, threads * sizeof(float)>>>(weights, bias, inputs,
				outputs, input_size, output_size, accumulate, use_bias);
		}

		////////////////////////////////////////////////////////////
		void linear_backward(float* weights, float* out_errors, float* errors, int input_size, int output_size)
		{
			if (input_size == 0) return;
			blocks = min(input_size, CUDA_MAX_CORES);
			threads = min(low_pow2(output_size), CUDA_MAX_THREADS);
			knl_linear_backward<<<blocks, threads, threads * sizeof(float)>>>(weights, out_errors, errors, input_size, output_size);
		}

		////////////////////////////////////////////////////////////
		void linear_accumulate_deltas(float* deltas, float* inputs, float* errors, int input_size, int output_size, bool use_bias)
		{
			blocks = min(output_size, CUDA_MAX_CORES);
			threads = min((int)low_pow2(input_size), CUDA_MAX_THREADS);
			knl_linear_accumulate_deltas<<<blocks, threads>>>(deltas, inputs, errors, input_size,
				output_size, use_bias);
		}

		////////////////////////////////////////////////////////////
		void linear_update_parameters(float* weights, float* bias, float* deltas, float learningrate, int input_size, int output_size)
		{
			threads = min((int)low_pow2(input_size * output_size), CUDA_MAX_THREADS);
			blocks = (input_size * output_size) / threads + 1;
			knl_linear_update_parameters<<<blocks, threads>>>(weights, bias, deltas, learningrate, input_size, output_size);
		}
		
		////////////////////////////////////////////////////////////
		void sigmoid_foreward(float* inputs, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_sigmoid_foreward<<<blocks, threads>>>(inputs, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void sigmoid_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_sigmoid_backward<<<blocks, threads>>>(errors, out_errors, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void relu_foreward(float* inputs, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_relu_foreward<<<blocks, threads>>>(inputs, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void relu_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_relu_backward<<<blocks, threads>>>(errors, out_errors, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void selu_foreward(float* inputs, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_selu_foreward<<<blocks, threads>>>(inputs, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void selu_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_selu_backward<<<blocks, threads>>>(errors, out_errors, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void tanh_foreward(float* inputs, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_tanh_foreward<<<blocks, threads>>>(inputs, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void tanh_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_tanh_backward<<<blocks, threads>>>(errors, out_errors, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void dropout_foreward(float* inputs, float* outputs, unsigned int seed, float dropout_probability, bool training, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_dropout_foreward<<<blocks, threads, threads * sizeof(unsigned int)>>>(inputs, outputs, seed, dropout_probability, training, size);
		}
		
		////////////////////////////////////////////////////////////
		void dropout_backward(float* errors, float* out_errors, float* outputs, float dropout_probability, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_dropout_backward<<<blocks, threads>>>(errors, out_errors, outputs, dropout_probability, size);
		}
			
		////////////////////////////////////////////////////////////
		void softmax_foreward(float* inputs, float* outputs, float scale, int size, float epsilon)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = 1;
			knl_softmax_foreward<<<blocks, threads, threads * sizeof(float)>>>(inputs, outputs, scale, size, epsilon);
		}
		
		////////////////////////////////////////////////////////////
		void softmax_backward(float* errors, float* out_errors, float* outputs, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_softmax_backward<<<blocks, threads>>>(errors, out_errors, outputs, size);
		}
		
		////////////////////////////////////////////////////////////
		void cost_crossentropy(float* prediction, float* target, float* errors, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_cost_crossentropy<<<blocks, threads, threads * sizeof(float)>>>(prediction, target, errors, size);
		}
		
		////////////////////////////////////////////////////////////
		void normalization_foreward(float* inputs, float* deviation, float* normalized,
			float* outputs, float* variance, float* gamma, float* beta, float epsilon, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_normalization_foreward_1<<<blocks, threads, threads * sizeof(float)>>>(inputs, size);
			knl_normalization_foreward_2<<<1, threads, threads * sizeof(float)>>>(inputs, deviation, size);
			knl_normalization_foreward_3<<<blocks, threads, threads * sizeof(float)>>>(inputs, deviation, size);
			knl_normalization_foreward_4<<<1, threads, threads * sizeof(float)>>>(variance, size);
			knl_normalization_foreward_5<<<blocks, threads>>>(inputs, deviation, normalized, outputs, variance, gamma, beta, epsilon, size);
		}
		
		////////////////////////////////////////////////////////////
		void normalization_backward(float* errors, float* out_errors, float* deviation,
			float* variance, float* gamma, float* beta, float epsilon, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_normalization_backward_1<<<blocks, threads, threads * sizeof(float)>>>(errors, deviation, size);
			knl_normalization_backward_2<<<1, threads, threads * sizeof(float)>>>();
			knl_normalization_backward_3<<<blocks, threads>>>(errors, out_errors, deviation, gamma, beta, variance, epsilon, size);
		}
		
		////////////////////////////////////////////////////////////
		void normalization_accumulate_deltas(float* errors, float* deviation, float* variance, float* d_gamma, float* d_beta, float epsilon, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = 1;
			knl_normalization_accumulate_deltas<<<blocks, threads, threads * sizeof(float)>>>(errors, deviation, variance, epsilon, d_beta, d_gamma, size);
		}
		
		////////////////////////////////////////////////////////////
		void normalization_update_parameters(float* gamma, float* beta, float* d_gamma, float* d_beta, float momentum, int size, float learningrate)
		{
			threads = 1;
			blocks = 1;
			knl_normalization_update_parameters<<<blocks, threads>>>(beta, gamma, d_beta, d_gamma, momentum, size, learningrate);
		}
		
		////////////////////////////////////////////////////////////
		void sparse_indices(float* inputs, int input_size, int* indices, int* tmp_indices, int* indices_count)
		{
			threads = 16;
			blocks = 1;
			int chunck_size = input_size / threads + 1;
			knl_sparse_indices<<<blocks, threads, (threads * 2) * sizeof(unsigned int)>>>(inputs, input_size, indices, tmp_indices, indices_count, chunck_size);
		}
		
		////////////////////////////////////////////////////////////
		void linear_sparse_foreward(float* weights, float* bias, float* inputs, float* outputs, int* indices, int* indices_count, int input_size, int output_size)
		{
			int host_indices_count;
			cudaMemcpy(indices_count, &host_indices_count, sizeof(int), cudaMemcpyDeviceToHost);
			blocks = min(output_size, CUDA_MAX_CORES);
			threads = min((int)low_pow2(host_indices_count), CUDA_MAX_THREADS);
			knl_linear_sparse_foreward<<<blocks, threads, threads * sizeof(float)>>>(weights, bias, inputs, outputs, indices, indices_count, input_size, output_size);
		}

		////////////////////////////////////////////////////////////
		void linear_sparse_accumulate_deltas(float* deltas, float* inputs, float* errors, int* indices, int* indices_count, int input_size, int output_size)
		{
			int host_indices_count;
			cudaMemcpy(indices_count, &host_indices_count, sizeof(int), cudaMemcpyDeviceToHost);
			blocks = min(output_size, CUDA_MAX_CORES);
			threads = min((int)low_pow2(host_indices_count), CUDA_MAX_THREADS);
			knl_linear_sparse_accumulate_deltas<<<blocks, threads>>>(deltas, inputs, errors, indices, indices_count, input_size, output_size);
		}	
		
		////////////////////////////////////////////////////////////
		void concatenate_foreward(float** inputs, float* outputs, int* sizes, int input_count, int total_size)
		{
			threads = min((int)low_pow2(total_size), CUDA_MAX_THREADS);
			blocks = min(total_size / threads + 1, CUDA_MAX_CORES);
			knl_concatenate_foreward<<<blocks, threads>>>(inputs, outputs, sizes, input_count);
		}
		
		////////////////////////////////////////////////////////////
		void concatenate_backward(float* errors, float** out_errors, int* sizes, int input_count, int total_size)
		{
			threads = min((int)low_pow2(total_size), CUDA_MAX_THREADS);
			blocks = min(total_size / threads + 1, CUDA_MAX_CORES);
			knl_concatenate_backward<<<blocks, threads>>>(errors, out_errors, sizes, input_count);
		}
		
		////////////////////////////////////////////////////////////
		void gradient_clipping(float* deltas, int size, const float clipping_deviation)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_gradient_clipping<<<blocks, threads>>>(weights, size, clipping_deviation);
		}
		
		////////////////////////////////////////////////////////////
		void l1_regularization(float* weights, const float l1_factor, const float learningrate, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_l1_regularization<<<blocks, threads>>>(weights, l1_factor, learningrate, size);
		}
		
		////////////////////////////////////////////////////////////
		void l2_regularization(float* weights, const float l2_factor, const float learningrate, int size)
		{
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_l2_regularization<<<blocks, threads>>>(weights, l2_factor, learningrate, size);
		}
		
		////////////////////////////////////////////////////////////
		void image_translate(float* image, float* result_buffer, const int width, const int height, const int channels, const int by_x, const int by_y)
		{
			int size = height * width * channels;
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_image_translate<<<blocks, threads>>>(image, result_buffer, width, height, channels, by_x, by_y);
		}
		
		////////////////////////////////////////////////////////////
		void image_vertical_flip(float* image, const int width, const int height, const int channels)
		{
			int size = (height / 2) * width * channels;
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_image_vertical_flip<<<blocks, threads, threads * sizeof(float)>>>(image, width, height, channels);
		}
		
		////////////////////////////////////////////////////////////
		void image_horizontal_flip(float* image, const int width, const int height, const int channels)
		{
			int size = (width / 2) * height * channels;
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_image_horizontal_flip<<<blocks, threads, threads * sizeof(float)>>>(image, width, height, channels);
		}
		
		////////////////////////////////////////////////////////////
		void image_rotate(float* image, float* result_buffer, const int width, const int height, const int channels, const float degrees)
		{
			int size = height * width * channels;
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			const float angle_rad = degrees * (3.14159/180.f);
			const float a = cos(angle_rad);
			const float b = sin(angle_rad);
			const int wh = width/2.f;
			const int hh = height/2.f;
			const int xoffset = wh - (wh * a - hh * b);
			const int yoffset = hh - (wh * b + hh * a);
			knl_image_rotate<<<blocks, threads>>>(image, result_buffer, width, height, channels, a, b, xoffset, yoffset);
		}
		
		////////////////////////////////////////////////////////////
		void image_scale(float* image, float* result_buffer, const int width, const int height, const int channels, const float scale_factor)
		{
			const int size = height * width * channels;
			const float scale = 1.f / scale_factor;
			const int center_x = width / 2;
			const int center_y = height / 2;
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			knl_image_scale<<<blocks, threads>>>(image, result_buffer, width, height, channels, scale, center_x, center_y);
		}
		
		////////////////////////////////////////////////////////////
		void image_add_noise(float* image, const int width, const int height, const int channels, const float noise_probability)
		{
			int size = width * height * channels;
			threads = min((int)low_pow2(size), CUDA_MAX_THREADS);
			blocks = min(size / threads + 1, CUDA_MAX_CORES);
			const unsigned int seed = rand();
			knl_image_add_noise<<<blocks, threads, threads * sizeof(unsigned int)>>>(image, width, height, channels, seed, noise_probability);
		}	
		
	} //namespace cuda

} //namespace ai
