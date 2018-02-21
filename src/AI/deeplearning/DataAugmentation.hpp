#ifndef DATAUGMENTATION_HPP
#define DATAUGMENTATION_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "../util/Tensor.hpp"
#include "../util/Macros.hpp"
#ifdef CUDA_BACKEND
#include "../util/TensorCUDA.hpp"
#endif
#include <vector>
#include <string>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	
	////////////////////////////////////////////////////////////
	///	NAMESPACE AUGMENTATION
	////////////////////////////////////////////////////////////
	namespace augmentation
	{
		
		#ifdef CUDA_BACKEND

		void translate(TensorCUDA_float& t, int image_width, int image_height, int image_channels, int tx, int ty);
		void rotate(TensorCUDA_float& t, int image_width, int image_height, int image_channels, float degrees);
		void noise(TensorCUDA_float& t, int image_width, int image_height, int image_channels, float noise);
		void vflip(TensorCUDA_float& t, int width, int height, int channels);
		void hflip(TensorCUDA_float& t, int width, int height, int channels);
		void scaling(TensorCUDA_float& t, int width, int height, int channel, float scale_factor);

		#else

		void translate(Tensor_float& t, int image_width, int image_height, int image_channels, int tx, int ty);
		void rotate(Tensor_float& t, int image_width, int image_height, int image_channels, float degrees);
		void noise(Tensor_float& t, int image_width, int image_height, int image_channels, float noise);
		void vflip(Tensor_float& t, int width, int height, int channels);
		void hflip(Tensor_float& t, int width, int height, int channels);
		void scaling(Tensor_float& t, int width, int height, int channel, float scale_factor);
		
		#endif

	} /* namespace augmentation */
	
} /* namespace ai */

#endif /* end of include guard: DATAUGMENTATION_HPP */

