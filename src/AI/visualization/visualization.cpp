////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "visualization.hpp"
#include "Bitmap.hpp"
#include <assert.h>
#include <algorithm>
#include <math.h>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	///	NAMESPACE VISUALIZATION
	////////////////////////////////////////////////////////////
	namespace visualization
	{
		
		////////////////////////////////////////////////////////////
		///	HIDDEN FUNCTIONS
		////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////
		/// \brief	Find the nearest perfect square of a number	
		///
		////////////////////////////////////////////////////////////
		int findminsquare(int num)
		{
			return (int)(sqrt((float)num) + 0.99);
		}

		////////////////////////////////////////////////////////////
		/// \brief	Find the minimum and maximum value in a vector
		/// of numbers
		///
		////////////////////////////////////////////////////////////
		void findvecrange(float* data, const int size, double *min, double *max)
		{
			if (min != NULL) *min = 0xFFFFFF;
			if (max != NULL) *max = -0xFFFFFF;
			for (int i = 0; i < (int)size; i++) {
				if (min != NULL && data[i] < *min) *min = data[i];
				if (max != NULL && data[i] > *max) *max = data[i];
			}
		}
		
		////////////////////////////////////////////////////////////
		///	PUBLIC FUNCTIONS
		////////////////////////////////////////////////////////////

		////////////////////////////////////////////////////////////
		void save_vec(std::string path, const Tensor_float& vector)
		{
			const int img_size = findminsquare(vector.size());
			double min, max;
			
			//find min and max for normalization
			findvecrange(vector.pointer(), vector.size(), &min, &max);
			
			//store and normalize data into a bitmap
			Bitmap bm(img_size, img_size, Bitmap::MONO, 0x000000);
			for (int i = 0; i < (int)vector.size(); i++)
				bm.m_data[i] = ((vector[i] - min) / (max - min)) * 255;

			//save bitmap
			bm.save(path);
		}
		
		////////////////////////////////////////////////////////////
		void save_multiple_vec(std::string path, const Tensor_float vector)
		{
			//Get final bitmap dimensions
			int img_width = 0;
			int img_height = 0;
			for (int i = 0; i < (int)vector.height(); i++) {
				const int block_size = findminsquare(vector.width());
				img_width += block_size + 1;
				if (block_size > img_height) img_height = block_size;
			}
				
			//Create final bitmap
			Bitmap img(img_width, img_height, Bitmap::MONO, 0x000000);
			
			int pos = 0;
			for (int j = 0; j < vector.height(); j++) {
				const int img_size = findminsquare(vector.width());
				double min, max;

				//find min and max for normalization
				const ai::Tensor_float tmp = vector.ptr(0, j);
				findvecrange(tmp.pointer(), vector.width(), &min, &max);

				//store and normalize data into a bitmap
				Bitmap bm(img_size, img_size, Bitmap::MONO, 0x000000);
				for (int i = 0; i < (int)vector.width(); i++)
					bm.m_data[i] = ((vector.at(j, i) - min) / (max - min)) * 255;
				
				//copy block to final bitmap
				bm.copyToRegion(img, 0, 0, img_size, img_size, pos, 0, img_size, img_size);
				pos += img_size + 1;
			}

			img.save(path);
		}
		
		////////////////////////////////////////////////////////////
		void save_multiple_vec(std::string path, const Tensor_float vector, int table_width, int table_height)
		{
			assert(vector.size() <= table_width * table_height);

			//Get final bitmap dimensions
			int img_width = findminsquare(vector.height());
			int img_height = img_width;
				
			//Create final bitmap
			Bitmap img(table_width * img_width, table_height * img_height, Bitmap::MONO, 0x000000);
					
			for (int j = 0; j < (int)vector.size(); j++) {
				double min, max;

				//find min and max for normalization
				const ai::Tensor_float tmp = vector.ptr(0, j);
				findvecrange(tmp.pointer(), vector.width(), &min, &max);

				//store and normalize data into a bitmap
				Bitmap bm(img_width, img_height, Bitmap::MONO, 0x000000);
				for (int i = 0; i < (int)vector.width(); i++)
					bm.m_data[i] = ((vector.at(j, i) - min) / (max - min)) * 255;
				
				//copy block to final bitmap
				bm.copyToRegion(img, 0, 0, img_width, img_height, (j % table_width) * img_width,
								(j / table_width) * img_height, img_width, img_height);
			}

			img.save(path);
		}

		////////////////////////////////////////////////////////////
		void save_vec(std::string path, const Tensor_float& vector, int width, int height)
		{
			//check if the size of the vector is correct
			assert(vector.size() <= width * height);

			//find min and max for normalization
			double min, max;
			findvecrange(vector.pointer(), vector.size(), &min, &max);
			
			//store and normalize data into a bitmap
			Bitmap bm(width, height, Bitmap::MONO, 0x000000);
			for (int i = 0; i < (int)vector.size(); i++)
				bm.m_data[i] = ((vector[i] - min) / (max - min)) * 255;

			//save bitmap
			bm.save(path);
		}

	} //namespace visualization

} //namespace AI
