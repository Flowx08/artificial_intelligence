#ifndef TENSORCUDA_HPP
#define TENSORCUDA_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include "Tensor.hpp"
#include "IOData.hpp"

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	template < typename T >
	class TensorCUDA
	{
		public:
			TensorCUDA();
			TensorCUDA(const TensorCUDA<T>& t);
			TensorCUDA(int width);
			TensorCUDA(int width, int height);
			TensorCUDA(int width, int height, int depth);
			~TensorCUDA();

			void load(ai::IOData& data, std::string dataname);
			void load(std::ifstream& file);
			void save(ai::IOData& data, std::string dataname);
			void save(std::ofstream& file);
            void setshape(const int width);
            void setshape(const int width, const int height);
            void setshape(const int width, const int height, const int depth);
            void setshape(Tensor<T>& host_tensor);
            void point(const TensorCUDA<T>& t);
            void point(const TensorCUDA<T>& t, const unsigned int offset_d);
            void point(const TensorCUDA<T>& t, const unsigned int offset_d, const unsigned int offset_y);
			void clear();
			void copy(const TensorCUDA<T>& tensor);
			void copyToHost(T *arr, int size) const;
			void copyToDevice(const T *arr, int size);
			void fill(T val);
			void fill(float mean, float dev);
			TensorCUDA<T> ptr(const int d);
			TensorCUDA<T> ptr(const int d, const int y);
			inline T* pointer() const { return _data;}
			inline const int size() const { return _size; }
			inline const int width() const { return _width; }
			inline const int height() const { return _height; }
			inline const int depth() const { return _depth; }


		private:
			T* _data = NULL;
			int _size = 0;
			int _width = 0, _height = 0, _depth = 0;
			bool _owner = false;
	};
	
	//Shortcut
	typedef TensorCUDA<float> TensorCUDA_float;
	typedef TensorCUDA<float*> TensorCUDA_float_ptr;
	typedef TensorCUDA<int> TensorCUDA_int;
	
	////////////////////////////////////////////////////////////
	///	TYPE SPECIFIC FUNCTIONS
	////////////////////////////////////////////////////////////
	void TensorCUDA_float_fill(TensorCUDA_float& t, float val);
	void TensorCUDA_float_fill(TensorCUDA_float& t, float mean, float dev);
	void TensorCUDA_float_scale(TensorCUDA_float& t, float factor);
	void TensorCUDA_float_diff(TensorCUDA_float& t1, TensorCUDA_float& t2, TensorCUDA_float& tout);
	void TensorCUDA_float_sum(TensorCUDA_float& t, TensorCUDA_float& tout);
	void TensorCUDA_float_copy(TensorCUDA_float& t, TensorCUDA_float& tout);

} /* namespace ai */

#endif /* end of include guard: TENSORCUDA_HPP */
