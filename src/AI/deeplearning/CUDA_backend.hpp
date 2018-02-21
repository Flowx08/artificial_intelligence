#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

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
		int init();
		void destroy();

		enum DataType
		{
			DATA_FLOAT,
			DATA_DOUBLE,
			DATA_HALF,
			DATA_INT8,
			DATA_INT32,
		};

		enum ActivationType
		{
			ACTIVATION_SIGMOID,
			ACTIVATION_RELU,
			ACTIVATION_TANH,
			ACTIVATION_CLIPPED_RELU,
			ACTIVATION_ELU,
		};

		enum PoolingType
		{
			POOLING_MAX,
			POOLING_AVERAGE,
		};

		class TensorDescription
		{
			public:
				TensorDescription();
				TensorDescription(const int width, const int height, const int depth,
						const int batch_size, const DataType type);
				~TensorDescription();
				void create(const int width, const int height, const int depth,
						const int batch_size, const DataType type);
				void* get();

			private:
				void clear();
				void* _tensor_description;
		};

		class Activation
		{
			public:
				Activation();
				Activation(const int size, const int batch_size, const ActivationType type);
				~Activation();
				void create(const int size, const int batch_size, const ActivationType type);
				void foreward(void* input, void* output); 
				void backward(void* input, void* output, void* errors, void* output_errors); 

			private:
				void clear();
				TensorDescription _size_description;
				void* _activation_description;
		};

		class Convolution
		{
			public:
				Convolution();
				Convolution(const int input_width, const int input_height, const int input_depth,
						const int batch_size, const int filter_width, const int filter_height, 
						const int filter_count, const int padding_w, const int padding_h,
						const int stride_u, const int stride_v, const bool backward_errors);
				~Convolution();
				void create(const int input_width, const int input_height, const int input_depth,
						const int batch_size, const int filter_width, const int filter_height, 
						const int filter_count, const int padding_w, const int padding_h,
						const int stride_u, const int stride_v, const bool backward_errors);
				void foreward(void* input, void* output, void* weights, void* bias, void* workspace); 
				void backward(void* errors, void* output_errors, void* weights, void* workspace);
				void accumulate_deltas(void* input, void* output, void* errors,
						void* filter_deltas, void* bias_deltas, void* workspace);
				void update_weights(void* weights, void* filter_deltas, void* bias, void* bias_deltas, const float learningrate);
				void getOutputSize(int* output_width, int* output_height, int* output_depth);
				int getWorkspaceSize();

			private:
				void clear();
				TensorDescription _input_description;
				TensorDescription _output_description;
				TensorDescription _bias_description;
				int _workspace_size;
				int _weights_size;
				int _bias_size;
				int _output_width, _output_height, _output_depth;
				void* _filter_description;
				void* _convolution_description;
				void* _fwd_algorithm_description;
				void* _bwd_filter_algorithm_description;
				void* _bwd_data_algorithm_description;
		};

		class Pooling
		{
			public:
				Pooling();
				Pooling(const int input_width, const int input_height, const int input_count, const int batch_size,
						const int pooling_width, const int pooling_height, const PoolingType type);
				~Pooling();
				void clear();
				void create(const int input_width, const int input_height, const int input_count, const int batch_size,
						const int pooling_width, const int pooling_height, const PoolingType type);
				void foreward(void* input, void* output);
				void backward(void* input, void* outputs, void* errors, void* out_errors);

			private:
				void* _pooling_description;
				TensorDescription _input_description;
				TensorDescription _output_description;
		};

		class Dropout
		{
			public:
				Dropout();
				Dropout(const int input_size, const float dropout_probability, void* state_buffer);
				~Dropout();
				void clear();
				void create(const int input_size, const float dropout_probability, void* state_buffer);
				void foreward(void* input, void* output, void* reserve_space_buffer);
				void backward(void* errors, void* out_errors, void* reserve_space_buffer);
				size_t getStatesSize();
				size_t getReserveSpaceSize(const int input_size);

			private:
				void* _dropout_description;
				size_t _states_size;
				size_t _reserve_space_size;
				TensorDescription _input_description;
		};

	} /* namespace cudnn */

	////////////////////////////////////////////////////////////
	///	NAMESPACE CUDA
	////////////////////////////////////////////////////////////
	namespace cuda
	{
		////////////////////////////////////////////////////////////
		///	CUDA DEEPLEARNING INTERFACE
		////////////////////////////////////////////////////////////
		void conv_foreward(float* weights, float* bias, float* inputs, float* outputs,
				int* out_in_map, int input_width, int input_height, int input_count, int stride,
				int output_width, int output_height, int filters_count, int filter_area);
		void conv_backward(float* weights, float* out_errors, float* errors,
				int* in_weight_map, int* in_out_map, int input_count, int output_size, int input_width,
				int input_height, int filter_area, int filters_count);
		void conv_accumulate_deltas(float* weights_deltas, float* bias_deltas, float* errors, float* inputs, float* outputs,
				int* out_in_map, int input_count, int input_width, int input_height, int output_size,
				int filter_area, int filters_count);
		void conv_update_parameters(float* weights, float* bias, float* weights_deltas, float* bias_deltas, int filter_area, 
				int input_count, int filter_count, float learningrate);
		void maxpooling_foreward(float* inputs, float* outputs, int* maxbuffer, int input_width, int input_height,
				int input_count, int stride, int filter_size, int output_width, int output_height);
		void maxpooling_backward(float* out_errors, float* errors, int* maxbuffer, int input_width, int input_height,
				int input_count, int stride, int filter_size, int output_width, int output_height);
		void averagepooling_foreward(float* inputs, float* outputs, int input_width, int input_height,
				int input_count, int stride, int filter_size, int output_width, int output_height);
		void averagepooling_backward(float* out_errors, float* errors, int input_width, int input_height,
				int input_count, int stride, int filter_size, int output_width, int output_height);
		void linear_foreward(float* weights, float* bias, float* inputs, float* outputs, int input_size, int output_size, bool use_bias, bool accumulate);
		void linear_backward(float* weights, float* out_errors, float* errors, int input_size, int output_size);
		void linear_accumulate_deltas(float* deltas, float* inputs, float* errors, int input_size, int output_size, bool use_bias);
		void linear_update_parameters(float* weights, float* bias, float* deltas, float learningrate, int input_size, int output_size);
		void sigmoid_foreward(float* inputs, float* outputs, int size);
		void sigmoid_backward(float* errors, float* out_errors, float* outputs, int size);
		void relu_foreward(float* inputs, float* outputs, int size);
		void relu_backward(float* errors, float* out_errors, float* outputs, int size);
		void tanh_foreward(float* inputs, float* outputs, int size);
		void tanh_backward(float* errors, float* out_errors, float* outputs, int size);
		void dropout_foreward(float* inputs, float* outputs, unsigned int seed, float dropout_probability, bool training, int size);
		void dropout_backward(float* errors, float* out_errors, float* outputs, float dropout_probability, int size);
		void selu_foreward(float* inputs, float* outputs, int size);
		void selu_backward(float* errors, float* out_errors, float* outputs, int size);
		void normalization_foreward(float* inputs, float* deviation, float* normalized,
				float* outputs, float* variance, float* gamma, float* beta, float epsilon, int size);
		void normalization_backward(float* errors, float* out_errors, float* deviation,
				float* variance, float* gamma, float* beta, float epsilon, int size);
		void normalization_accumulate_deltas(float* errors, float* deviation, float* variance, float* d_gamma, float* d_beta, float epsilon, int size);
		void normalization_update_parameters(float* gamma, float* beta, float* d_gamma, float* d_beta, float momentum, int size, float learningrate);
		void sparse_indices(float* inputs, int inputs_size, int* indices, int* tmp_indices, int* indices_count);
		void linear_sparse_foreward(float* weights, float* bias, float* inputs, float* outputs, int* indices, int* indices_count, int input_size, int output_size);
		void linear_sparse_accumulate_deltas(float* deltas, float* inputs, float* errors, int* indices, int* indices_count, int input_size, int output_size);
		void concatenate_foreward(float** inputs, float* outputs, int* sizes, int input_count, int total_size);
		void concatenate_backward(float* errors, float** out_errors, int* sizes, int input_count, int total_size);
		void softmax_foreward(float* inputs, float* outputs, float scale, int size, float epsilon);
		void softmax_backward(float* errors, float* out_errors, float* outputs, int size);
		void cost_crossentropy(float* prediction, float* target, float* errors, int size);

		void gradient_clipping(float* deltas, int size, const float clipping_deviation);
		void l1_regularization(float* weights, const float l1_factor, const float learningrate, int size);
		void l2_regularization(float* weights, const float l2_factor, const float learningrate, int size);

		//Image data augumentation
		void image_translate(float* image, float* result_buffer, const int width, const int height, const int channels, const int by_x, const int by_y);
		void image_vertical_flip(float* image, const int width, const int height, const int channels);
		void image_horizontal_flip(float* image, const int width, const int height, const int channels);
		void image_rotate(float* image, float* result_buffer, const int width, const int height, const int channels, const float degrees);
		void image_scale(float* image, float* result_buffer, const int width, const int height, const int channels, const float scale_factor);
		void image_add_noise(float* image, const int width, const int height, const int channels, const float noise_probability);

	} /* namespace cuda */

} /* namespace ai */

#endif /* end of include guard: CUDA_BACKEND_H */

