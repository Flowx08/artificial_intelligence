////////////////////////////////////////////////////////////
///	INCLUDES
////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <float.h>
#include "Util.hpp"
#include "Tensor.hpp"
#include <cmath>

////////////////////////////////////////////////////////////
///	NAMESPACE AI
////////////////////////////////////////////////////////////
namespace ai
{

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T>::Tensor()
		{
			_width = 0;
			_height = 0;
			_depth = 0;
			_depth_size = 0;
			_size = 0;
			_data = NULL;
			_data_secure = std::shared_ptr<T>();
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T>::Tensor(const Tensor<T>& t)
		{
			point(t);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T>::Tensor(const int width)
		{
			setshape(width);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T>::Tensor(const int width, const int height)
		{
			setshape(width, height);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T>::Tensor(const int width, const int height, const int depth)
		{
			setshape(width, height, depth);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::setshape(const int width)
		{
			ensure(width != 0);
			clear();
			_width = width;
			_height = 1;
			_depth = 1;
			_depth_size = _width * _height;
			_size = width;
			_data_secure = std::shared_ptr<T>(new T[_size]);
			_data = _data_secure.get();
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::setshape(const int width, const int height)
		{
			ensure(width != 0 && height != 0);
			clear();
			_width = width;
			_height = height;
			_depth = 1;
			_depth_size = _width * _height;
			_size = width * height;
			_data_secure = std::shared_ptr<T>(new T[_size]);
			_data = _data_secure.get();
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::setshape(const int width, const int height, const int depth)
		{
			ensure(width != 0 && height != 0 && depth != 0);
			clear();
			_width = width;
			_height = height;
			_depth = depth;
			_depth_size = _width * _height;
			_size = width * height * depth;
			_data_secure = std::shared_ptr<T>(new T[_size]);
			_data = _data_secure.get();
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::point(const Tensor<T>& t)
		{
			clear();
			_data_secure = t._data_secure;
			_data = t._data;
			_size = t._size;
			_width = t._width;
			_height = t._height;
			_depth = t._depth;
			_depth_size = t._depth_size;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::point(const Tensor<T>& t, const unsigned int offset_d)
		{
#ifdef TENSOR_CHECK_BADACCESS
			ensure(offset_d < t._depth);
#endif
			clear();
			_data_secure = t._data_secure;
			_data = &t.at(offset_d, 0, 0);
			_size = t._width * t._height;
			_width = t._width;
			_height = t._height;
			_depth = 1;
			_depth_size = _width * _height;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::point(const Tensor<T>& t, const unsigned int offset_d, const unsigned int offset_y)
		{
#ifdef TENSOR_CHECK_BADACCESS
			ensure(offset_d < t._depth);
			ensure(offset_y < t._height);
#endif
			clear();
			_data_secure = t._data_secure;
			_data = &t.at(offset_d, offset_y, 0);
			_size = t._width;
			_width = t._width;
			_height = 1;
			_depth = 1;
			_depth_size = _width * _height;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::point_raw(T* raw_data, const unsigned int width, const unsigned int height, const unsigned int depth)
		{
			clear();
			_data_secure = std::shared_ptr<T>();
			_data = raw_data;
			_size = width * height * depth;
			_width = width;
			_height = height;
			_depth = depth;
			_depth_size = _width * _height;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::save(ai::IOData& data, std::string dataname)
		{
			data.pushNode(dataname + "_width", _width);	
			data.pushNode(dataname + "_height", _height);	
			data.pushNode(dataname + "_depth", _depth);
			data.pushNode(dataname + "_data", reinterpret_cast<char*>(&_data[0]), sizeof(T) * _size);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::load(ai::IOData& data, std::string dataname)
		{
			IOData* node_width = data.findNode(dataname + "_width");	
			IOData* node_height = data.findNode(dataname + "_height");	
			IOData* node_depth = data.findNode(dataname + "_depth");
			IOData* node_data = data.findNode(dataname + "_data");
			ensure(node_width != NULL);
			ensure(node_height != NULL);
			ensure(node_depth != NULL);
			ensure(node_data != NULL);
			node_width->get(_width);
			node_height->get(_height);
			node_depth->get(_depth);
			_size = _width * _height * _depth;
			_depth_size = _width * _height;
			_data_secure = std::shared_ptr<T>(new T[_size]);
			_data = _data_secure.get();
			node_data->get(reinterpret_cast<char*>(&_data[0]));
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::load(std::ifstream& file)
		{
			clear();
			file.read(reinterpret_cast<char*>(&_size), sizeof(_size));
			file.read(reinterpret_cast<char*>(&_width), sizeof(_width));
			file.read(reinterpret_cast<char*>(&_height), sizeof(_height));
			file.read(reinterpret_cast<char*>(&_depth), sizeof(_depth));
			_depth_size = _width * _height;
			_data_secure = std::shared_ptr<T>(new T[_size]);
			_data = _data_secure.get();
			file.read(reinterpret_cast<char*>(&_data[0]), sizeof(T) * _size);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::save(std::ofstream& file)
		{
			file.write(reinterpret_cast<char*>(&_size), sizeof(_size));
			file.write(reinterpret_cast<char*>(&_width), sizeof(_width));
			file.write(reinterpret_cast<char*>(&_height), sizeof(_height));
			file.write(reinterpret_cast<char*>(&_depth), sizeof(_depth));
			file.write(reinterpret_cast<char*>(&_data[0]), sizeof(T) * _size);
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::fill(const T val)
		{
			for (int i = 0; i < _size; i++)
				_data[i] = val;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::fill(const float mean, const float dev)
		{
			for (int i = 0; i < _size; i++)
				_data[i] = mean -dev + util::randf() * 2.f * dev;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::clear()
		{
			_data_secure = std::shared_ptr<T>();
			_data = NULL;
			_size = 0;
			_width = 0;
			_height = 0;
			_depth = 0;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::copy(const Tensor<T>& t)
		{
			clear();
			_width = t._width;
			_height = t._height;
			_depth = t._depth;
			_depth_size = _width * _height;
			_size = t._size;
			_data_secure = std::shared_ptr<T>(new T[_size]);
			_data = _data_secure.get();
			for (int i = 0; i < _size; i++)
				_data[i] = t._data[i];
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::max(T* max, int* pos) const
		{
			T _max = -3.2f * 10e4;
			int _maxid = 0;

			for (int i = 0; i < _size; i++) {
				if (_data[i] > _max) {
					_max = _data[i];
					_maxid = i;
				}
			}

			if (max != NULL) *max = _max;
			if (pos != NULL) *pos = _maxid;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::max_at(T* max, int* pos, unsigned int offset_d, unsigned int offset_y) const
		{
#ifdef TENSOR_CHECK_BADACCESS
			ensure(offset_d < _depth);
			ensure(offset_y < _height);
#endif
			float _max = -3.2f * 10e4;
			int _maxid = 0;

			const int start = offset_d * _depth_size + offset_y * _width;
			const int end = start + _width;

			for (int i = start; i < end; i++) {
				if (_data[i] > _max) {
					_max = _data[i];
					_maxid = i;
				}
			}

			if (max != NULL) *max = _max;
			if (pos != NULL) *pos = _maxid - start;	
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::min(T* min, int* pos) const
		{
			float _min = 3.2 * 10e4;
			int _minid = 0;

			for (int i = 0; i < _size; i++) {
				if (_data[i] < _min) {
					_min = _data[i];
					_minid = i;
				}
			}

			if (min != NULL) *min = _min;
			if (pos != NULL) *pos = _minid;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::min_at(T* min, int* pos, unsigned int offset_d, unsigned int offset_y) const
		{
#ifdef TENSOR_CHECK_BADACCESS
			ensure(offset_d < _depth);
			ensure(offset_y < _height);
#endif
			float _min = 3.2 * 10e4;
			int _minid = 0;

			const int start = offset_d * _depth_size + offset_y * _width;
			const int end = start + _width;

			for (int i = start; i < end; i++) {
				if (_data[i] < _min) {
					_min = _data[i];
					_minid = i;
				}
			}

			if (min != NULL) *min = _min;
			if (pos != NULL) *pos = _minid - start;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T> Tensor<T>::ptr() const
		{
			Tensor<T> t;
			t.point(*this);
			return t;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T> Tensor<T>::ptr(const int d) const
		{
			Tensor<T> t;
			t.point(*this, d);
			return t;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		Tensor<T> Tensor<T>::ptr(const int d, const int y) const
		{
			Tensor<T> t;
			t.point(*this, d, y);
			return t;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		const std::string Tensor<T>::tostring() const
		{
			if (_size == 0) return "[]";
			std::string buf;
			buf += "[ ";
			for (int i = 0; i < _size-1; i++)
				buf += std::to_string(_data[i]) + " , ";
			buf += std::to_string(_data[_size-1]) + " ]";
			return buf;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::add(const Tensor<T>& t)
		{
			ensure(_size <= t.size());
			for (int i = 0; i < _size; i++)
				_data[i] += t[i];
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::set(const Tensor<T>& t)
		{
			ensure(_size <= t.size());
			for (int i = 0; i < _size; i++)
				_data[i] = t[i];
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::mul(const Tensor<T>& t)
		{
			ensure(_size <= t.size());
			for (int i = 0; i < _size; i++)
				_data[i] *= t[i];
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::mul(const Tensor<T>& t1, const Tensor<T>& t2)
		{
			ensure(_size <= t1.size() && _size <= t2.size());
			for (int i = 0; i < _size; i++)
				_data[i] = t1[i] * t2[i];
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		void Tensor<T>::get_mean_and_variance(float* mean, float* variance)
		{
			float tmp_mean = 0, tmp_variance = 0;

			if (_size == 0) {
				if (mean != NULL) *mean = 0;
				if (variance != NULL) *variance = 0;
				return;
			}

			//Calculate mean
			for (int i = 0; i < _size; i++)
				tmp_mean += _data[i];
			tmp_mean /= (float)_size;

			//Calculate deviation
			if (variance != NULL) {
				for (int i = 0; i < _size; i++)
					tmp_variance += pow(_data[i] - tmp_mean, 2);
				tmp_variance /= (float)_size;
			}

			if (mean != NULL) *mean = tmp_mean;
			if (variance != NULL) *variance = tmp_variance;
		}

	////////////////////////////////////////////////////////////
	template <typename T>
		bool Tensor<T>::isNaN()
		{
			for (int i = 0; i < _size; i++)
				if (std::isnan((float)_data[i])) return true;
			return false;
		}

	//Explicit instantiations
	template class Tensor<float>;
	template class Tensor<int>;

} /* namespace ai */
