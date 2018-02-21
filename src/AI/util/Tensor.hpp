#ifndef TENSOR_HPP
#define TENSOR_HPP

////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <string>
#include <fstream>
#include <memory>
#include "ensure.hpp"
#include "IOData.hpp"

//Uncomment this define to debug tensor accessing
//#define TENSOR_CHECK_BADACCESS

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{
	template < typename T >
	class Tensor
	{
		public:
			Tensor();
			Tensor(const Tensor<T>& t);
			Tensor(const int width);
			Tensor(const int width, const int height);
			Tensor(const int width, const int height, const int depth);

			void load(ai::IOData& data, std::string dataname);
			void load(std::ifstream& file);
			void save(ai::IOData& data, std::string dataname);
			void save(std::ofstream& file);
			void setshape(const int width);
			void setshape(const int width, const int height);
			void setshape(const int width, const int height, const int depth);
			void fill(const T val);
			void fill(const float mean, const float dev);
			void clear();
			void point(const Tensor<T>& t);
			void point(const Tensor<T>& t, const unsigned int offset_d);
			void point(const Tensor<T>& t, const unsigned int offset_d, const unsigned int offset_y);
			void point_raw(T* raw_data, const unsigned int width, const unsigned int height, const unsigned int depth);
			void copy(const Tensor<T>& t);
			void max(T* max, int* pos) const;
			void max_at(T* max, int* pos, unsigned int offset_d, unsigned int offset_y) const;
			void min(T* min, int* pos) const;
			void min_at(T* min, int* pos, unsigned int offset_d, unsigned int offset_y) const;
			void get_mean_and_variance(float* mean, float* variance);
			bool isNaN();
			Tensor<T> ptr() const;
			Tensor<T> ptr(const int d) const;
			Tensor<T> ptr(const int d, const int y) const;
			const std::string tostring() const;

			inline const int size() const { return _size; }
			inline const int width() const { return _width; }
			inline const int height() const { return _height; }
			inline const int depth() const { return _depth; }
			inline T* pointer() const { return &_data[0]; }
			inline operator T*() const { return &_data[0]; }
			inline void operator =(const Tensor<T>& t) { copy(t); }

			////////////////////////////////////////////////////////////
			///	INDEXING
			////////////////////////////////////////////////////////////

			inline T& at(const int d, const int y, const int x) const
			{
				#ifdef TENSOR_CHECK_BADACCESS
				ensure( d < _depth && y < _height && x < _width);
				#endif
				return _data[d * _depth_size + y * _width + x];
			}

			inline T& at(const int y, const int x) const
			{
				#ifdef TENSOR_CHECK_BADACCESS
				ensure(y < _height && x < _width);
				#endif
				return _data[y * _width + x];
			}

			inline T& at(const int x) const
			{
				#ifdef TENSOR_CHECK_BADACCESS
				ensure(x < _size);
				#endif
				return _data[x];
			}

			inline T& operator [](const int x)
			{
				#ifdef TENSOR_CHECK_BADACCESS
				ensure(x < _size);
				#endif
				return _data[x];
			}

			inline const T& operator [](const int x) const
			{
				#ifdef TENSOR_CHECK_BADACCESS
				ensure(x < _size);
				#endif
				return _data[x];
			}

			////////////////////////////////////////////////////////////
			///	OPERATIONS
			////////////////////////////////////////////////////////////
			void mul(const Tensor<T>& t);
			void mul(const Tensor<T>& t1, const Tensor<T>& t2);
			void set(const Tensor<T>& t);
			void add(const Tensor<T>& t);

		private:
			T* _data = NULL;
			std::shared_ptr<T> _data_secure;
			int _size = 0;
			int _width = 0;
			int _height = 0;
			int _depth = 0;
			int _depth_size;
	};
	
	//Shortcuts
	typedef Tensor<float> Tensor_float;
	typedef Tensor<int> Tensor_int;

} /* namespace ai */

#endif //TENSOR_HPP
